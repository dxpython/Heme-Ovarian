from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual


def plot_risk_distribution(
    risk_train: pd.Series,
    risk_val: pd.Series,
    cutoff_tr: float,
    cutoff_va: float,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    sns.histplot(risk_train, kde=True, ax=axes[0], color="#6a1b9a")
    axes[0].axvline(cutoff_tr, color="#c62828", linewidth=1.2)
    axes[0].set_title("Training risk")
    sns.histplot(risk_val, kde=True, ax=axes[1], color="#00838f")
    axes[1].axvline(cutoff_va, color="#c62828", linewidth=1.2)
    axes[1].set_title("Validation risk")
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)


def plot_risk_survival_scatter(
    risk: pd.Series,
    time: pd.Series,
    event: pd.Series,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
    title: str,
) -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    alive = event == 0
    dead = event == 1
    ax.scatter(risk[alive], time[alive], c="#1565c0", alpha=0.55, label="censored", s=18)
    ax.scatter(risk[dead], time[dead], c="#c62828", alpha=0.75, label="event", s=22)
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Time")
    ax.set_title(title)
    ax.legend(frameon=False)
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)


def plot_signature_heatmap(
    expr: pd.DataFrame,
    genes: list[str],
    risk_order: pd.Series,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> None:
    apply_publication_style()
    genes = [g for g in genes if g in expr.index]
    sub = expr.loc[genes, risk_order.index].astype(float)
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    sns.heatmap(z, cmap="RdBu_r", center=0, ax=ax, xticklabels=False, yticklabels=True)
    ax.set_title("Signature z-score (columns ordered by risk)")
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
