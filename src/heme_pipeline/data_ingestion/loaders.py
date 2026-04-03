from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from heme_pipeline.logging_utils import get_logger

LOG = get_logger("data_ingestion.loaders")


def read_table(path: str | Path, file_format: str | None = None, **kwargs: Any) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    fmt = file_format or _infer_format(p)
    fmt_l = fmt.lower().strip(".")
    if fmt_l in {"csv"}:
        return pd.read_csv(p, **kwargs)
    if fmt_l in {"tsv", "txt"}:
        sep = kwargs.pop("sep", "\t")
        return pd.read_csv(p, sep=sep, **kwargs)
    if fmt_l in {"xlsx", "xls"}:
        return pd.read_excel(p, **kwargs)
    raise ValueError(f"unsupported format: {fmt}")


def _infer_format(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in {".csv"}:
        return "csv"
    if suf in {".tsv", ".txt"}:
        return "tsv"
    if suf in {".xlsx", ".xls"}:
        return "xlsx"
    raise ValueError(f"cannot infer format from suffix: {path}")


def read_gene_list(path: str | Path) -> list[str]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    lines = p.read_text(encoding="utf-8").splitlines()
    genes: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        genes.append(line.split()[0])
    return genes
