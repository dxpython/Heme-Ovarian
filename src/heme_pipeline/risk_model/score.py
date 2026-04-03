from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def linear_risk_score(
    expr: pd.DataFrame,
    genes: list[str],
    coefficients: pd.Series,
) -> pd.Series:
    genes = [g for g in genes if g in expr.index and g in coefficients.index]
    if not genes:
        raise ValueError("no overlapping genes for risk score")
    sub = expr.loc[genes].astype(float)
    beta = coefficients.reindex(genes).fillna(0.0).values
    risk = (sub.values.T @ beta).flatten()
    return pd.Series(risk, index=sub.columns, name="risk_score")


def cox_risk_from_training(
    expr_train: pd.DataFrame,
    time_train: np.ndarray,
    event_train: np.ndarray,
    genes: list[str],
) -> tuple[pd.Series, CoxPHFitter]:
    genes = [g for g in genes if g in expr_train.index]
    if not genes:
        raise ValueError("no genes for cox fitting")
    df = expr_train.loc[genes].T.copy()
    df["T"] = time_train
    df["E"] = event_train
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col="T", event_col="E")
    coef = cph.params_
    risk_train = cph.predict_partial_hazard(df.drop(columns=["T", "E"]))
    return pd.Series(risk_train.values.flatten(), index=df.index, name="risk_score"), cph


def compute_cutoff(
    risk: pd.Series,
    method: str,
    quantile: float | None,
) -> float:
    if method == "median":
        return float(np.median(risk.values))
    if method == "mean":
        return float(np.mean(risk.values))
    if method == "quantile":
        if quantile is None:
            raise ValueError("quantile cutoff requires quantile value")
        return float(np.quantile(risk.values, quantile))
    raise ValueError(f"unknown cutoff method: {method}")


def stratify_risk(risk: pd.Series, cutoff: float) -> pd.Series:
    return pd.Series(
        np.where(risk.values > cutoff, "high", "low"),
        index=risk.index,
        name="risk_group",
    )
