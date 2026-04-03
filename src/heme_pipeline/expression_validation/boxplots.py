from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual


def plot_four_genes_by_binary_group(
    expr: pd.DataFrame,
    groups: pd.Series,
    genes: list[str],
    group_order: tuple[str, str],
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
    title: str,
) -> pd.DataFrame:
    apply_publication_style()
    rows = []
    g0, g1 = group_order
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8))
    axes = axes.flatten()
    for ax, gene in zip(axes, genes):
        if gene not in expr.index:
            continue
        sub = pd.DataFrame({"value": expr.loc[gene].astype(float), "group": groups.reindex(expr.columns)})
        sub = sub.dropna()
        d0 = sub.loc[sub["group"] == g0, "value"]
        d1 = sub.loc[sub["group"] == g1, "value"]
        if d0.size < 2 or d1.size < 2:
            continue
        stat, p = mannwhitneyu(d0, d1, alternative="two-sided")
        rows.append({"gene": gene, "p": float(p)})
        sns.boxplot(data=sub, x="group", y="value", order=[g0, g1], ax=ax, palette=["#eceff1", "#c8e6c9"])
        sns.stripplot(data=sub, x="group", y="value", order=[g0, g1], ax=ax, color="#424242", alpha=0.35)
        ax.set_title(f"{gene} p={p:.3g}")
        ax.set_xlabel("")
    fig.suptitle(title)
    fig.tight_layout()
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
    return pd.DataFrame(rows)
