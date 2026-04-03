from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn2

from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual


def intersect_deg_hmrg(
    deg_genes: Iterable[str],
    hmrg_genes: Iterable[str],
) -> pd.DataFrame:
    d = set(str(g).upper() for g in deg_genes)
    h = set(str(g).upper() for g in hmrg_genes)
    inter = sorted(d & h)
    return pd.DataFrame({"gene": inter})


def plot_venn2(
    set_a: set[str],
    set_b: set[str],
    label_a: str,
    label_b: str,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> None:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    v = venn2([set_a, set_b], (label_a, label_b), ax=ax)
    for t in v.set_labels or []:
        if t is not None:
            t.set_fontsize(10)
    for t in v.subset_labels or []:
        if t is not None:
            t.set_fontsize(10)
    ax.set_axis_off()
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
