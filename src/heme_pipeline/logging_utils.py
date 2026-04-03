from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(log_dir: str | Path, level: str, file_name: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / file_name
    root = logging.getLogger("heme_pipeline")
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"heme_pipeline.{name}")
