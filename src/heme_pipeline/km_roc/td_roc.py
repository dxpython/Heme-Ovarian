from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual


def time_dependent_roc_binary(
    duration: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    horizon: float,
    risk_high_is_worse: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    dead_or_gt = (duration > horizon) | (event == 1)
    y = ((duration <= horizon) & (event == 1)).astype(int)
    mask = dead_or_gt
    y = y[mask]
    s = risk[mask]
    if risk_high_is_worse:
        s = -s
    if len(np.unique(y)) < 2:
        return float("nan"), np.array([]), np.array([]), np.array([])
    auc = roc_auc_score(y, s)
    fpr, tpr, thr = roc_curve(y, s)
    return float(auc), fpr, tpr, thr


def plot_td_roc(
    duration: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    horizons: list[float],
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> pd.DataFrame:
    apply_publication_style()
    rows = []
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for h in horizons:
        auc, fpr, tpr, _ = time_dependent_roc_binary(duration, event, risk, h)
        rows.append({"horizon": h, "auc": auc})
        if len(fpr):
            ax.plot(fpr, tpr, linewidth=1.4, label=f"t={h}, AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#bdbdbd", linewidth=0.8, linestyle="--")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(frameon=False, fontsize=8)
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
    return pd.DataFrame(rows)
