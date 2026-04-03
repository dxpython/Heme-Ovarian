from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from heme_pipeline.logging_utils import get_logger

LOG = get_logger("gsea_gsva.gsva")


def run_gsva_rscript(
    expr_path: Path,
    gmt_path: Path,
    out_path: Path,
    r_script: Path,
    kernel: str = "gaussian",
) -> pd.DataFrame:
    if not r_script.is_file():
        raise FileNotFoundError(str(r_script))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "Rscript",
        str(r_script),
        "--expr",
        str(expr_path),
        "--gmt",
        str(gmt_path),
        "--out",
        str(out_path),
        "--kernel",
        kernel,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        LOG.warning("Rscript not found; GSVA skipped")
        return pd.DataFrame()
    except subprocess.CalledProcessError as exc:
        LOG.warning("GSVA R script failed: %s", exc.stderr)
        return pd.DataFrame()
    if out_path.is_file():
        return pd.read_csv(out_path, index_col=0)
    return pd.DataFrame()
