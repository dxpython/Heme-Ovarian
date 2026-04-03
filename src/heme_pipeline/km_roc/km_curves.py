from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual


def plot_km_two_group(
    duration: pd.Series,
    event: pd.Series,
    groups: pd.Series,
    labels_map: dict[str, str],
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
    title: str,
) -> dict[str, float | str]:
    apply_publication_style()
    d = duration.astype(float)
    e = event.astype(float)
    g = groups.astype(str)
    g1 = g.unique().tolist()
    if len(g1) < 2:
        raise ValueError("KM requires two groups")
    kmf1 = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    m1, m2 = g1[0], g1[1]
    mask1 = g == m1
    mask2 = g == m2
    kmf1.fit(d[mask1], e[mask1], label=labels_map.get(m1, m1))
    kmf2.fit(d[mask2], e[mask2], label=labels_map.get(m2, m2))
    lr = logrank_test(d[mask1], d[mask2], e[mask1], e[mask2])
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    kmf1.plot_survival_function(ax=ax, color="#c62828", linewidth=1.6)
    kmf2.plot_survival_function(ax=ax, color="#1565c0", linewidth=1.6)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    ax.text(
        0.05,
        0.05,
        f"log-rank p={lr.p_value:.4g}",
        transform=ax.transAxes,
        fontsize=9,
    )
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
    return {"logrank_p": float(lr.p_value)}
