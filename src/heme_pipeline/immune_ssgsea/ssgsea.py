from __future__ import annotations

from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from heme_pipeline.logging_utils import get_logger
from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual

LOG = get_logger("immune_ssgsea.ssgsea")


def run_ssgsea_immune(
    expr: pd.DataFrame,
    gmt_path: Path,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = expr.T.copy()
    res = gp.ssgsea(
        data=df,
        gene_sets=str(gmt_path),
        outdir=str(out_dir),
        sample_norm_method="rank",
        no_plot=True,
    )
    r2 = res if isinstance(res, pd.DataFrame) else getattr(res, "res2d", None)
    if r2 is None:
        return pd.DataFrame()
    out = r2.T
    return out


def compare_high_low(
    scores: pd.DataFrame,
    risk_groups: pd.Series,
) -> pd.DataFrame:
    rows = []
    for col in scores.columns:
        x = scores[col].astype(float)
        g = risk_groups.reindex(scores.index)
        hi = x[g == "high"]
        lo = x[g == "low"]
        if hi.size < 2 or lo.size < 2:
            continue
        from scipy.stats import mannwhitneyu

        stat, p = mannwhitneyu(hi, lo, alternative="two-sided")
        rows.append({"cell_type": col, "median_high": float(np.median(hi)), "median_low": float(np.median(lo)), "p": float(p)})
    return pd.DataFrame(rows)


def plot_immune_heatmap(
    scores: pd.DataFrame,
    risk_groups: pd.Series,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> None:
    apply_publication_style()
    g = risk_groups.reindex(scores.index)
    mat = scores.loc[g.dropna().index].groupby(g.dropna()).median().T
    fig, ax = plt.subplots(figsize=(6.5, 8.0))
    sns.heatmap(mat, cmap="RdBu_r", center=0, ax=ax, linewidths=0.3)
    ax.set_title("Immune ssGSEA median by risk group")
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
