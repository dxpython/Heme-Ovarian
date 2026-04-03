from __future__ import annotations

from typing import Iterable

import pandas as pd

from heme_pipeline.survival_models.pipelines import ModelResult


def select_best_model(
    results: Iterable[ModelResult],
    train_weight: float = 0.4,
    val_weight: float = 0.6,
) -> ModelResult:
    scored: list[tuple[float, ModelResult]] = []
    for r in results:
        score = train_weight * r.train_cindex + val_weight * r.val_cindex
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        raise ValueError("no models to select")
    return scored[0][1]


def results_to_comparison_table(results: Iterable[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.name,
                "train_cindex": r.train_cindex,
                "val_cindex": r.val_cindex,
                "weighted_score": 0.4 * r.train_cindex + 0.6 * r.val_cindex,
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_score", ascending=False)
