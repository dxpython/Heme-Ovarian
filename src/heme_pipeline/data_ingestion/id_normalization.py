from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def normalize_sample_id(value: str) -> str:
    s = str(value).strip()
    s = s.upper()
    s = s.replace(".", "-")
    s = re.sub(r"[^A-Z0-9\-]", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def normalize_gene_symbol(value: str) -> str:
    s = str(value).strip()
    s = s.upper()
    s = re.sub(r"\s+", "", s)
    return s


def normalize_column_name(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"\s+", "_", s)
    return s


def normalize_expression_matrix(
    df: pd.DataFrame,
    gene_column: str | None,
    samples_as: str,
) -> pd.DataFrame:
    if gene_column and gene_column in df.columns:
        genes = df[gene_column].map(normalize_gene_symbol)
        out = df.drop(columns=[gene_column])
        out.index = genes
        out = out[~out.index.duplicated(keep="first")]
    elif samples_as == "columns":
        out = df.copy()
        out.columns = [normalize_sample_id(str(c)) for c in out.columns]
        if isinstance(out.index, pd.Index):
            out.index = [normalize_gene_symbol(str(i)) for i in out.index]
            out = out[~out.index.duplicated(keep="first")]
    else:
        out = df.copy()
        out.index = [normalize_sample_id(str(i)) for i in out.index]
        out.columns = [normalize_gene_symbol(str(c)) for c in out.columns]
        out = out.loc[~out.index.duplicated(keep="first")]
    return out


def align_clinical_columns(
    df: pd.DataFrame,
    sample_aliases: Iterable[str],
    preferred_sample_col: str,
) -> pd.DataFrame:
    out = df.copy()
    cols = {normalize_column_name(c): c for c in out.columns}
    sample_key = None
    for alias in sample_aliases:
        nk = normalize_column_name(alias)
        if nk in cols:
            sample_key = cols[nk]
            break
    if sample_key is None and preferred_sample_col in out.columns:
        sample_key = preferred_sample_col
    if sample_key is None:
        raise KeyError("clinical table missing sample id column")
    renamed = {sample_key: "sample_id"}
    out = out.rename(columns=renamed)
    out["sample_id"] = out["sample_id"].map(lambda x: normalize_sample_id(str(x)))
    return out


def detect_survival_columns(
    df: pd.DataFrame,
    time_aliases: Iterable[str],
    event_aliases: Iterable[str],
    fallback_time: str,
    fallback_event: str,
) -> tuple[str, str]:
    cols_norm = {normalize_column_name(c): c for c in df.columns}
    tcol = None
    for a in time_aliases:
        if normalize_column_name(a) in cols_norm:
            tcol = cols_norm[normalize_column_name(a)]
            break
    ecol = None
    for a in event_aliases:
        if normalize_column_name(a) in cols_norm:
            ecol = cols_norm[normalize_column_name(a)]
            break
    if tcol is None and fallback_time in df.columns:
        tcol = fallback_time
    if ecol is None and fallback_event in df.columns:
        ecol = fallback_event
    if tcol is None or ecol is None:
        raise KeyError("survival time/event columns not found in clinical table")
    return tcol, ecol
