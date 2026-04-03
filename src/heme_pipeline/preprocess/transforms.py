from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def log2_transform(expr: pd.DataFrame, offset: float = 1.0) -> pd.DataFrame:
    x = expr.astype(float)
    return np.log2(x + offset)


def quantile_normalize(expr: pd.DataFrame) -> pd.DataFrame:
    arr = expr.values.astype(float)
    sorted_arr = np.sort(arr, axis=0)
    ref = np.mean(sorted_arr, axis=1)
    out = np.zeros_like(arr)
    n = arr.shape[0]
    idx = np.linspace(0.0, float(n - 1), n)
    for j in range(arr.shape[1]):
        ranks = rankdata(arr[:, j], method="average") - 1.0
        ranks = np.clip(ranks, 0.0, float(n - 1))
        out[:, j] = np.interp(ranks, idx, ref)
    return pd.DataFrame(out, index=expr.index, columns=expr.columns)


def zscore_genes(expr: pd.DataFrame) -> pd.DataFrame:
    return expr.sub(expr.mean(axis=1), axis=0).div(expr.std(axis=1).replace(0, np.nan), axis=0)


def subset_common_genes(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = train.index.intersection(val.index)
    return train.loc[common], val.loc[common]


def impute_median(expr: pd.DataFrame) -> pd.DataFrame:
    return expr.fillna(expr.median(axis=1), axis=0).fillna(expr.median(axis=0), axis=1)
