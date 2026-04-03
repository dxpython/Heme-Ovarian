from __future__ import annotations

from pathlib import Path
from typing import Any

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from heme_pipeline.logging_utils import get_logger
from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual

LOG = get_logger("enrichment.go_kegg")


def run_enrichr_multilib(
    gene_list: list[str],
    libraries: list[str],
    cutoff: float,
    use_api: bool,
) -> dict[str, pd.DataFrame]:
    if not gene_list:
        raise ValueError("empty gene list for enrichment")
    results: dict[str, pd.DataFrame] = {}
    if use_api:
        for lib in libraries:
            try:
                enr = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=lib,
                    organism="human",
                    outdir=None,
                    no_plot=True,
                )
                df = enr.results
                df = df[df["Adjusted P-value"] <= cutoff].copy()
                results[lib] = df
            except Exception as exc:
                LOG.warning("enrichr failed for %s: %s", lib, exc)
    else:
        LOG.warning("offline enrichr not configured; set use_enrichr_api true")
    return results


def plot_enrichment_bar(
    df: pd.DataFrame,
    title: str,
    top_n: int,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> None:
    if df.empty:
        return
    apply_publication_style()
    sub = df.sort_values("Adjusted P-value").head(top_n).copy()
    sub["Term"] = sub["Term"].astype(str).str.slice(0, 60)
    p = pd.to_numeric(sub["Adjusted P-value"], errors="coerce").clip(lower=1e-300)
    sub["neglog10p"] = -np.log10(p.values)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    sns.barplot(
        x="neglog10p",
        y="Term",
        data=sub,
        color="#2e7d32",
        ax=ax,
        orient="h",
    )
    ax.set_xlabel("-log10 adjusted P")
    ax.set_ylabel("")
    ax.set_title(title)
    sns.despine(ax=ax)
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
