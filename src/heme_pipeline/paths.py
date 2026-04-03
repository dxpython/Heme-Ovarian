from __future__ import annotations

import os
from pathlib import Path


def resolve_path(base: str | Path, *parts: str) -> Path:
    root = Path(base).expanduser().resolve()
    if not parts:
        return root
    return (root.joinpath(*parts)).resolve()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_relative(path: str | Path, base: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def as_posix(path: str | Path) -> str:
    return Path(path).as_posix()


def env_expand(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))
