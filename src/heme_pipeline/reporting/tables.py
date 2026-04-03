from __future__ import annotations

import pandas as pd
from lifelines import CoxPHFitter


def table1_baseline(
    clinical: pd.DataFrame,
    risk_groups: pd.Series,
    age_col: str | None,
    stage_col: str | None,
) -> pd.DataFrame:
    df = clinical.copy()
    sid = "sample_id"
    if sid not in df.columns and df.index.name == "sample_id":
        df = df.reset_index()
    df["risk_group"] = risk_groups.reindex(df[sid]).values
    rows = []
    for g in ["high", "low"]:
        sub = df[df["risk_group"] == g]
        n = len(sub)
        if age_col and age_col in sub.columns:
            age = pd.to_numeric(sub[age_col], errors="coerce")
            rows.append(
                {
                    "variable": "age",
                    "group": g,
                    "n": n,
                    "summary": f"{age.mean():.1f}±{age.std():.1f}",
                }
            )
        if stage_col and stage_col in sub.columns:
            vc = sub[stage_col].astype(str).value_counts()
            rows.append({"variable": "stage", "group": g, "n": n, "summary": repr(vc.to_dict())})
    return pd.DataFrame(rows)


def table2_multivariate_signature(
    expr: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    genes: list[str],
) -> pd.DataFrame:
    genes = [g for g in genes if g in expr.index]
    df = expr.loc[genes].T.copy()
    df["T"] = time
    df["E"] = event
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df, duration_col="T", event_col="E")
    out = cph.summary.copy().reset_index().rename(columns={"index": "gene"})
    return out
