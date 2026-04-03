from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from heme_pipeline.data_ingestion.loaders import read_table
from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual


def load_qpcr(path: Path, sheet: str | int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet)
    return read_table(path, file_format="tsv")


def plot_qpcr_four_genes(
    df: pd.DataFrame,
    group_col: str,
    gene_map: dict[str, str],
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
    title: str,
) -> pd.DataFrame:
    apply_publication_style()
    rows = []
    genes = list(gene_map.keys())
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8))
    axes = axes.flatten()
    for ax, gene in zip(axes, genes, strict=False):
        col = gene_map[gene]
        sub = df[[group_col, col]].dropna()
        groups = sub[group_col].unique().tolist()
        if len(groups) < 2:
            continue
        d0 = sub.loc[sub[group_col] == groups[0], col].astype(float)
        d1 = sub.loc[sub[group_col] == groups[1], col].astype(float)
        stat, p = mannwhitneyu(d0, d1, alternative="two-sided")
        rows.append({"gene": gene, "p": float(p)})
        sns.boxplot(data=sub, x=group_col, y=col, ax=ax, palette=["#eceff1", "#ffccbc"])
        sns.stripplot(data=sub, x=group_col, y=col, ax=ax, color="#424242", alpha=0.35)
        ax.set_title(f"{gene} p={p:.3g}")
    fig.suptitle(title)
    fig.tight_layout()
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
    return pd.DataFrame(rows)
