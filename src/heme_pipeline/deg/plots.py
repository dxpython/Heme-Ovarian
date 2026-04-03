from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual


def plot_volcano(
    deg: pd.DataFrame,
    padj_threshold: float,
    log2fc_threshold: float,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    x = deg["log2FC"].values
    y = -np.log10(np.clip(deg["padj"].values, 1e-300, None))
    sig = (deg["padj"].values < padj_threshold) & (np.abs(deg["log2FC"].values) > log2fc_threshold)
    ax.scatter(x[~sig], y[~sig], s=10, c="#9e9e9e", alpha=0.55, linewidths=0)
    ax.scatter(x[sig], y[sig], s=14, c="#c62828", alpha=0.75, linewidths=0)
    ax.axhline(-np.log10(padj_threshold), color="#616161", linewidth=0.8, linestyle="--")
    ax.axvline(log2fc_threshold, color="#616161", linewidth=0.8, linestyle="--")
    ax.axvline(-log2fc_threshold, color="#616161", linewidth=0.8, linestyle="--")
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10 adjusted P")
    sns.despine(ax=ax)
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)


def plot_deg_heatmap(
    expr: pd.DataFrame,
    genes: list[str],
    sample_groups: pd.Series,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> None:
    apply_publication_style()
    genes = [g for g in genes if g in expr.index]
    if not genes:
        raise ValueError("no genes available for heatmap")
    sub = expr.loc[genes, sample_groups.index].astype(float)
    sub = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan), axis=0)
    col_colors = sample_groups.map(lambda x: "#4a90d9" if str(x).lower() == "tumor" else "#7cb342")
    fig = plt.figure(figsize=(8.0, 5.5))
    cg = sns.clustermap(
        sub,
        col_colors=col_colors,
        cmap="vlag",
        xticklabels=False,
        yticklabels=True,
        figsize=(8.0, 5.5),
        dendrogram_ratio=(0.12, 0.12),
        cbar_kws={"label": "z-score"},
    )
    save_figure_dual(cg.figure, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(cg.figure)
