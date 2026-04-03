from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from heme_pipeline.logging_utils import get_logger

LOG = get_logger("survival_models.univariate_cox")


def univariate_cox_screen(
    expr: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    genes: list[str],
    p_threshold: float,
    max_genes: int,
) -> pd.DataFrame:
    rows = []
    for g in genes:
        if g not in expr.index:
            continue
        df = pd.DataFrame({"x": expr.loc[g].astype(float).values})
        df["T"] = time
        df["E"] = event
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(df, duration_col="T", event_col="E")
            p = float(cph.summary.loc["x", "p"])
            coef = float(cph.summary.loc["x", "coef"])
            hr = float(np.exp(coef))
            rows.append({"gene": g, "coef": coef, "HR": hr, "p": p})
        except Exception as exc:
            LOG.debug("univariate cox failed for %s: %s", g, exc)
            continue
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res = res[res["p"] <= p_threshold].sort_values("p")
    return res.head(max_genes)
