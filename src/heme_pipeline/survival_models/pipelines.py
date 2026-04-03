from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from heme_pipeline.logging_utils import get_logger
LOG = get_logger("survival_models.pipelines")


@dataclass
class ModelResult:
    name: str
    train_cindex: float
    val_cindex: float
    model: Any
    risk_train: np.ndarray
    risk_val: np.ndarray
    extra: dict[str, Any]


def _surv_y(duration: np.ndarray, event: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(event.astype(bool), duration.astype(float))


def train_rsf(
    X_train: pd.DataFrame,
    y_time_train: np.ndarray,
    y_event_train: np.ndarray,
    X_val: pd.DataFrame,
    y_time_val: np.ndarray,
    y_event_val: np.ndarray,
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: int | None,
    random_state: int,
) -> ModelResult:
    y = _surv_y(y_time_train, y_event_train)
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )
    rsf.fit(X_train.values, y)
    risk_tr = -rsf.predict(X_train.values)
    risk_va = -rsf.predict(X_val.values)
    ci_tr = concordance_index_censored(
        y_event_train.astype(bool),
        y_time_train.astype(float),
        risk_tr,
    )[0]
    ci_va = concordance_index_censored(
        y_event_val.astype(bool),
        y_time_val.astype(float),
        risk_va,
    )[0]
    return ModelResult(
        name="rsf",
        train_cindex=float(ci_tr),
        val_cindex=float(ci_va),
        model=rsf,
        risk_train=risk_tr,
        risk_val=risk_va,
        extra={},
    )


def train_coxnet_variant(
    X_train: pd.DataFrame,
    y_time_train: np.ndarray,
    y_event_train: np.ndarray,
    X_val: pd.DataFrame,
    y_time_val: np.ndarray,
    y_event_val: np.ndarray,
    name: str,
    l1_ratio: float,
    alphas: list[float],
    random_state: int,
) -> ModelResult:
    y = _surv_y(y_time_train, y_event_train)
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train.values)
    Xv = scaler.transform(X_val.values)
    est = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        alphas=alphas,
        fit_baseline_model=True,
    )
    est.fit(Xt, y)
    risk_tr = est.predict(Xt)
    risk_va = est.predict(Xv)
    ci_tr = concordance_index_censored(
        y_event_train.astype(bool),
        y_time_train.astype(float),
        risk_tr,
    )[0]
    ci_va = concordance_index_censored(
        y_event_val.astype(bool),
        y_time_val.astype(float),
        risk_va,
    )[0]
    return ModelResult(
        name=name,
        train_cindex=float(ci_tr),
        val_cindex=float(ci_va),
        model={"coxnet": est, "scaler": scaler, "columns": list(X_train.columns)},
        risk_train=risk_tr,
        risk_val=risk_va,
        extra={"l1_ratio": l1_ratio},
    )


def train_stepwise_cox(
    X_train: pd.DataFrame,
    y_time_train: np.ndarray,
    y_event_train: np.ndarray,
    X_val: pd.DataFrame,
    y_time_val: np.ndarray,
    y_event_val: np.ndarray,
    max_features: int,
    random_state: int,
) -> ModelResult:
    cols = list(X_train.columns)
    selected: list[str] = []
    remaining = cols.copy()
    while remaining and len(selected) < max_features:
        best_cand = None
        best_ci_step = -1.0
        for c in remaining:
            trial = selected + [c]
            df = X_train[trial].copy()
            df["T"] = y_time_train
            df["E"] = y_event_train
            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(df, duration_col="T", event_col="E")
                pred = cph.predict_partial_hazard(X_train[trial])
                ci = concordance_index(y_time_train, -pred, y_event_train)
            except Exception:
                continue
            if ci > best_ci_step:
                best_ci_step = ci
                best_cand = c
        if best_cand is None:
            break
        selected.append(best_cand)
        remaining.remove(best_cand)
    if not selected:
        raise RuntimeError("stepwise cox failed to select features")
    df = X_train[selected].copy()
    df["T"] = y_time_train
    df["E"] = y_event_train
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col="T", event_col="E")
    risk_tr = cph.predict_partial_hazard(X_train[selected]).values.flatten()
    risk_va = cph.predict_partial_hazard(X_val[selected]).values.flatten()
    ci_tr = concordance_index(y_time_train, -risk_tr, y_event_train)
    ci_va = concordance_index(y_time_val, -risk_va, y_event_val)
    return ModelResult(
        name="stepwise_cox",
        train_cindex=float(ci_tr),
        val_cindex=float(ci_va),
        model={"cph": cph, "genes": selected},
        risk_train=risk_tr,
        risk_val=risk_va,
        extra={},
    )


def train_svm_rfe_cox(
    X_train: pd.DataFrame,
    y_time_train: np.ndarray,
    y_event_train: np.ndarray,
    X_val: pd.DataFrame,
    y_time_val: np.ndarray,
    y_event_val: np.ndarray,
    n_features_to_select: int,
    random_state: int,
) -> ModelResult:
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train.values)
    y_target = y_time_train.astype(float)
    svr = SVR(kernel="linear")
    n_feat = max(1, min(n_features_to_select, X_train.shape[1]))
    rfe = RFE(estimator=svr, n_features_to_select=n_feat, step=1)
    rfe.fit(Xt, y_target)
    support = X_train.columns[rfe.support_].tolist()
    df = X_train[support].copy()
    df["T"] = y_time_train
    df["E"] = y_event_train
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col="T", event_col="E")
    risk_tr = cph.predict_partial_hazard(X_train[support]).values.flatten()
    risk_va = cph.predict_partial_hazard(X_val[support]).values.flatten()
    ci_tr = concordance_index(y_time_train, -risk_tr, y_event_train)
    ci_va = concordance_index(y_time_val, -risk_va, y_event_val)
    return ModelResult(
        name="svm_rfe_cox",
        train_cindex=float(ci_tr),
        val_cindex=float(ci_va),
        model={"cph": cph, "genes": support, "scaler": scaler, "rfe": rfe},
        risk_train=risk_tr,
        risk_val=risk_va,
        extra={},
    )


def train_xgboost_survival(
    X_train: pd.DataFrame,
    y_time_train: np.ndarray,
    y_event_train: np.ndarray,
    X_val: pd.DataFrame,
    y_time_val: np.ndarray,
    y_event_val: np.ndarray,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
) -> ModelResult:
    y = _surv_y(y_time_train, y_event_train)
    est = GradientBoostingSurvivalAnalysis(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    est.fit(X_train.values, y)
    risk_tr = est.predict(X_train.values)
    risk_va = est.predict(X_val.values)
    ci_tr = concordance_index_censored(
        y_event_train.astype(bool),
        y_time_train.astype(float),
        risk_tr,
    )[0]
    ci_va = concordance_index_censored(
        y_event_val.astype(bool),
        y_time_val.astype(float),
        risk_va,
    )[0]
    return ModelResult(
        name="xgboost_survival",
        train_cindex=float(ci_tr),
        val_cindex=float(ci_va),
        model={"gbsa": est, "columns": list(X_train.columns)},
        risk_train=risk_tr,
        risk_val=risk_va,
        extra={"implementation": "sksurv.GradientBoostingSurvivalAnalysis"},
    )
