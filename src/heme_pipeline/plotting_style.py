from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#111111",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#cccccc",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.grid": False,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.2,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure_dual(
    fig: plt.Figure,
    out_pdf: str | Path,
    out_png: str | Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> None:
    out_pdf = Path(out_pdf)
    out_png = Path(out_png)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if export_pdf:
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    if export_png:
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=dpi)


def close_figure(fig: plt.Figure | None) -> None:
    if fig is not None:
        plt.close(fig)
