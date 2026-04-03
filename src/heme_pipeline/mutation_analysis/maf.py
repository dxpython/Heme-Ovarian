from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from heme_pipeline.data_ingestion.loaders import read_table
from heme_pipeline.logging_utils import get_logger
from heme_pipeline.plotting_style import apply_publication_style, save_figure_dual

LOG = get_logger("mutation_analysis.maf")


def load_maf(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return read_table(path, file_format="tsv")


def mutation_matrix_from_maf(
    maf: pd.DataFrame,
    gene_column: str,
    sample_column: str,
    top_n: int,
) -> pd.DataFrame:
    sub = maf[[gene_column, sample_column]].dropna()
    sub[gene_column] = sub[gene_column].astype(str)
    sub[sample_column] = sub[sample_column].astype(str)
    freq = sub[gene_column].value_counts().head(top_n)
    genes = freq.index.tolist()
    mat = pd.crosstab(sub[sample_column], sub[gene_column])
    mat = mat.reindex(columns=genes, fill_value=0)
    return mat


def plot_mutation_landscape(
    mat: pd.DataFrame,
    risk_groups: pd.Series,
    out_pdf: Path,
    out_png: Path,
    dpi: int,
    export_pdf: bool,
    export_png: bool,
) -> None:
    apply_publication_style()
    g = risk_groups.reindex(mat.index)
    hi = mat.loc[g == "high"].sum(axis=0)
    lo = mat.loc[g == "low"].sum(axis=0)
    df = pd.DataFrame({"high": hi, "low": lo})
    df = df.sort_values("high", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["high"], width=w, label="high risk", color="#c62828")
    ax.bar(x + w / 2, df["low"], width=w, label="low risk", color="#1565c0")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index.astype(str), rotation=65, ha="right")
    ax.set_ylabel("Mutation count")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    save_figure_dual(fig, out_pdf, out_png, dpi, export_pdf, export_png)
    plt.close(fig)
