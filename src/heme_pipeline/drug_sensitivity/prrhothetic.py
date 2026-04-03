from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from heme_pipeline.data_ingestion.loaders import read_table
from heme_pipeline.logging_utils import get_logger

LOG = get_logger("drug_sensitivity.prrhothetic")


def load_gdsc_pair(
    expr_ref_path: Path,
    ic50_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not expr_ref_path.is_file() or not ic50_path.is_file():
        raise FileNotFoundError("GDSC reference or IC50 matrix missing")
    expr = read_table(expr_ref_path, file_format="tsv")
    ic50 = read_table(ic50_path, file_format="tsv")
    return expr, ic50


def estimate_ic50_prr(
    expr_samples: pd.DataFrame,
    expr_ref: pd.DataFrame,
    ic50_ref: pd.DataFrame,
    drugs: list[str],
    alpha: float = 1.0,
) -> pd.DataFrame:
    common_genes = expr_samples.index.intersection(expr_ref.index)
    if common_genes.size < 10:
        raise ValueError("insufficient gene overlap for drug sensitivity")
    cell_lines = expr_ref.columns.intersection(ic50_ref.index)
    if cell_lines.size < 10:
        raise ValueError("insufficient cell line overlap between expression and IC50 reference")
    Xref = expr_ref.loc[common_genes, cell_lines].astype(float).T.values
    scaler = StandardScaler()
    Xref_s = scaler.fit_transform(Xref)
    out = {}
    for drug in drugs:
        if drug not in ic50_ref.columns:
            continue
        y = ic50_ref.loc[cell_lines, drug].astype(float).values
        m = np.isfinite(y)
        if m.sum() < 10:
            continue
        model = Ridge(alpha=alpha)
        model.fit(Xref_s[m], y[m])
        Xs = scaler.transform(expr_samples.loc[common_genes].astype(float).T.values)
        pred = model.predict(Xs)
        out[drug] = pred
    return pd.DataFrame(out, index=expr_samples.columns)


def compare_groups_ic50(
    ic50_pred: pd.DataFrame,
    risk_groups: pd.Series,
) -> pd.DataFrame:
    rows = []
    for drug in ic50_pred.columns:
        x = ic50_pred[drug].astype(float)
        g = risk_groups.reindex(ic50_pred.index)
        hi = x[g == "high"]
        lo = x[g == "low"]
        if hi.size < 2 or lo.size < 2:
            continue
        from scipy.stats import mannwhitneyu

        stat, p = mannwhitneyu(hi, lo, alternative="two-sided")
        rows.append({"drug": drug, "median_high": float(np.median(hi)), "median_low": float(np.median(lo)), "p": float(p)})
    return pd.DataFrame(rows)
