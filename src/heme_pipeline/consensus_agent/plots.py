from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual


def plot_cindex_comparison(
    comp: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> None:
    if comp.empty:
        return
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = range(len(comp))
    ax.bar([i - 0.18 for i in x], comp["train_cindex"], width=0.35, label="train", color="#5c6bc0")
    ax.bar([i + 0.18 for i in x], comp["val_cindex"], width=0.35, label="validation", color="#ef5350")
    ax.set_xticks(list(x))
    ax.set_xticklabels(comp["model"].astype(str), rotation=35, ha="right")
    ax.set_ylabel("C-index")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
