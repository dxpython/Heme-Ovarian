from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from heme_pipeline.logging_utils import get_logger

LOG = get_logger("deg.analysis")


def run_deg_welch_ttest(
    expr: pd.DataFrame,
    groups: pd.Series,
    tumor_label: str,
    normal_label: str,
) -> pd.DataFrame:
    g = groups.reindex(expr.columns)
    mask_t = g == tumor_label
    mask_n = g == normal_label
    if mask_t.sum() < 2 or mask_n.sum() < 2:
        raise ValueError("insufficient tumor or normal samples for DEG")
    rows: list[dict[str, float | str]] = []
    for gene in expr.index:
        x_t = pd.to_numeric(expr.loc[gene, mask_t], errors="coerce").dropna().values
        x_n = pd.to_numeric(expr.loc[gene, mask_n], errors="coerce").dropna().values
        if x_t.size < 2 or x_n.size < 2:
            continue
        mean_t = float(np.mean(x_t))
        mean_n = float(np.mean(x_n))
        log2fc = float(np.log2((mean_t + 1e-9) / (mean_n + 1e-9)))
        stat, p = stats.ttest_ind(x_t, x_n, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "gene": gene,
                "mean_tumor": mean_t,
                "mean_normal": mean_n,
                "log2FC": log2fc,
                "t_stat": float(stat),
                "pvalue": float(p),
            }
        )
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    _, padj, _, _ = multipletests(res["pvalue"].values, method="fdr_bh")
    res["padj"] = padj
    res = res.sort_values("padj")
    return res


def filter_deg(
    deg: pd.DataFrame,
    padj_threshold: float,
    log2fc_threshold: float,
) -> pd.DataFrame:
    if deg.empty:
        return deg
    m = (deg["padj"] < padj_threshold) & (deg["log2FC"].abs() > log2fc_threshold)
    return deg.loc[m].copy()
