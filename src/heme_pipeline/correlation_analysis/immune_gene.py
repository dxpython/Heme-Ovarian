from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual


def spearman_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = df.columns.tolist()
    n = len(cols)
    rho = np.zeros((n, n))
    pval = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                rho[i, j] = 1.0
                pval[i, j] = 0.0
                continue
            x = df.iloc[:, i].astype(float).values
            y = df.iloc[:, j].astype(float).values
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3:
                continue
            r, p = spearmanr(x[m], y[m])
            rho[i, j] = r
            pval[i, j] = p
    return pd.DataFrame(rho, index=cols, columns=cols), pd.DataFrame(pval, index=cols, columns=cols)


def plot_corr_heatmap(
    corr: pd.DataFrame,
    pval: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10.0, 8.5))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.2,
    )
    ax.set_title("Spearman correlation (* p<0.05 overlay separate)")
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)


def merge_immune_and_genes(
    immune_scores: pd.DataFrame,
    expr_genes: pd.DataFrame,
    genes: list[str],
) -> pd.DataFrame:
    ggenes = [g for g in genes if g in expr_genes.index]
    sub = expr_genes.loc[ggenes].T
    merged = immune_scores.join(sub, how="inner")
    return merged
