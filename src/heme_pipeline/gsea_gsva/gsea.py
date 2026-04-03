from __future__ import annotations

from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heme_pipeline.logging_utils import get_logger
from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual

LOG = get_logger("gsea_gsva.gsea")


def run_gsea_prerank(
    ranked_genes: pd.Series,
    gmt_path: str | Path,
    out_dir: Path,
    permutations: int,
) -> pd.DataFrame:
    p = Path(gmt_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    out_dir.mkdir(parents=True, exist_ok=True)
    rnk = ranked_genes.sort_values(ascending=False)
    res = gp.prerank(
        rnk=rnk,
        gene_sets=str(p),
        outdir=str(out_dir),
        permutation_num=permutations,
        seed=42,
        no_plot=True,
    )
    if isinstance(res, pd.DataFrame):
        return res
    if hasattr(res, "res2d") and res.res2d is not None:
        return res.res2d
    return pd.DataFrame()


def plot_gsea_leading_edge(
    gsea_table: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
    top_n: int = 10,
) -> None:
    if gsea_table.empty:
        return
    apply_publication_style()
    sub = gsea_table.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    nes_col = "NES" if "NES" in sub.columns else ("nes" if "nes" in sub.columns else sub.columns[0])
    x = sub[nes_col]
    y = sub["Term"].astype(str) if "Term" in sub.columns else sub.index.astype(str)
    sns.barplot(x=x.values, y=y.values, color="#3949ab", ax=ax, orient="h")
    ax.set_xlabel("NES")
    ax.set_ylabel("")
    sns.despine(ax=ax)
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
