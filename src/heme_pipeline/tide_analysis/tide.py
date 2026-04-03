from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from heme_pipeline.data_ingestion.loaders import read_table
from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual


def load_tide_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return read_table(path, file_format="tsv")


def align_tide_risk(
    tide: pd.DataFrame,
    sample_col: str,
    risk_groups: pd.Series,
) -> pd.DataFrame:
    df = tide.copy()
    if sample_col not in df.columns:
        raise KeyError(sample_col)
    df = df.set_index(sample_col)
    g = risk_groups.reindex(df.index)
    df = df.loc[g.dropna().index].copy()
    df["risk_group"] = g.loc[df.index].values
    return df


def plot_tide_scores(
    df: pd.DataFrame,
    score_cols: list[str],
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> pd.DataFrame:
    apply_publication_style()
    rows = []
    fig, axes = plt.subplots(1, len(score_cols), figsize=(4.2 * len(score_cols), 4.5))
    if len(score_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, score_cols, strict=False):
        sub = df[[col, "risk_group"]].dropna()
        hi = sub.loc[sub["risk_group"] == "high", col].astype(float)
        lo = sub.loc[sub["risk_group"] == "low", col].astype(float)
        stat, p = mannwhitneyu(hi, lo, alternative="two-sided")
        rows.append({"metric": col, "p": float(p)})
        sns.boxplot(data=sub, x="risk_group", y=col, ax=ax, palette=["#1565c0", "#c62828"])
        sns.stripplot(
            data=sub,
            x="risk_group",
            y=col,
            ax=ax,
            color="#424242",
            alpha=0.35,
            dodge=True,
        )
        ax.set_title(f"{col} p={p:.3g}")
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
    return pd.DataFrame(rows)
