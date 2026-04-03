from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.util import Surv


def build_surv_array(duration: np.ndarray, event: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(event.astype(bool), duration.astype(float))


def concordance_cindex(
    duration: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    return float(concordance_index(duration, -risk_scores, event))


def fit_cox_ph(
    X: pd.DataFrame,
    duration: np.ndarray,
    event: np.ndarray,
    penalizer: float = 0.0,
    l1_ratio: float = 0.0,
) -> CoxPHFitter:
    df = X.copy()
    df["T"] = duration
    df["E"] = event
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(df, duration_col="T", event_col="E")
    return cph


def cox_coefficients_table(cph: CoxPHFitter) -> pd.DataFrame:
    summ = cph.summary.copy()
    return summ
