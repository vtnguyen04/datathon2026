from __future__ import annotations
import numpy as np
import pandas as pd
import zipfile
from pathlib import Path
from typing import Optional
from src.config import SUBMISSION_DIR


def ensemble_submissions(
    paths: list[str | Path],
    output: str = "ensemble.csv",
    weights: Optional[list[float]] = None,
    method: str = "mean",
    trim_pct: float = 0.1,
) -> Path:
    dfs = [pd.read_csv(p) for p in paths]
    for i, df in enumerate(dfs):
        assert set(df.columns) >= {"Date", "Revenue", "COGS"}, (
            f"Submission {i} missing required columns"
        )
        assert len(df) == len(dfs[0]), (
            f"Submission {i} has {len(df)} rows, expected {len(dfs[0])}"
        )
    result = dfs[0][["Date"]].copy()
    rev_stack = np.stack([df["Revenue"].values for df in dfs])
    cog_stack = np.stack([df["COGS"].values for df in dfs])
    if method == "mean":
        if weights:
            w = np.array(weights) / np.sum(weights)
            result["Revenue"] = np.average(rev_stack, axis=0, weights=w)
            result["COGS"] = np.average(cog_stack, axis=0, weights=w)
        else:
            result["Revenue"] = rev_stack.mean(axis=0)
            result["COGS"] = cog_stack.mean(axis=0)
    elif method == "median":
        result["Revenue"] = np.median(rev_stack, axis=0)
        result["COGS"] = np.median(cog_stack, axis=0)
    elif method == "trimmed_mean":
        from scipy.stats import trim_mean

        result["Revenue"] = np.apply_along_axis(
            lambda x: trim_mean(x, trim_pct), 0, rev_stack
        )
        result["COGS"] = np.apply_along_axis(
            lambda x: trim_mean(x, trim_pct), 0, cog_stack
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    result["Revenue"] = np.round(result["Revenue"], 2)
    result["COGS"] = np.round(result["COGS"], 2)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SUBMISSION_DIR / output
    zip_path = csv_path.with_suffix(".csv.zip")
    result.to_csv(csv_path, index=False)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, csv_path.name)
    n = len(paths)
    print(f"✅ Ensembled {n} submissions → {zip_path.name}")
    print(
        f"   Rev={result['Revenue'].mean():,.0f}  COGS={result['COGS'].mean():,.0f}  Margin={1 - result['COGS'].sum() / result['Revenue'].sum():.1%}"
    )
    return zip_path
