from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual


def grouped_boxplot_with_test(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
    title: str,
) -> dict[str, Any]:
    apply_publication_style()
    sub = df[[value_col, group_col]].dropna()
    groups = sub[group_col].unique().tolist()
    data = [sub.loc[sub[group_col] == g, value_col].values for g in groups]
    if len(groups) == 2:
        stat, p = stats.mannwhitneyu(data[0], data[1], alternative="two-sided")
        test_name = "Mann-Whitney"
    else:
        stat, p = stats.kruskal(*data)
        test_name = "Kruskal-Wallis"
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.boxplot(data=sub, x=group_col, y=value_col, color="#e3f2fd", ax=ax)
    sns.stripplot(data=sub, x=group_col, y=value_col, color="#424242", alpha=0.35, ax=ax)
    ax.set_title(title)
    sns.despine(ax=ax)
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
    return {"test": test_name, "p": float(p), "statistic": float(stat)}
