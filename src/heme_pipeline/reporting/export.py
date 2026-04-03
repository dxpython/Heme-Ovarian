from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).lower().endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, sep="\t", index=False)
