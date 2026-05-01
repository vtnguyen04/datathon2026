from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


class CrossSubmissionBlender:
    def __init__(self, submissions_dir: Path):
        self.submissions_dir = submissions_dir
        self._submissions: dict[str, pd.DataFrame] = {}

    def load_submissions(self, names: list[str] | None = None) -> None:
        if names is None:
            files = sorted(self.submissions_dir.glob("*.csv"))
            names = [f.stem for f in files]
        for name in names:
            path = self.submissions_dir / f"{name}.csv"
            if path.exists():
                self._submissions[name] = pd.read_csv(path, parse_dates=["Date"])

    @property
    def submission_names(self) -> list[str]:
        return list(self._submissions.keys())

    def blend(self, weights: dict[str, float], target: str = "Revenue") -> np.ndarray:
        total_w = sum(weights.values())
        result = None
        for name, w in weights.items():
            if name not in self._submissions:
                raise ValueError(f"Unknown submission: {name}")
            vals = self._submissions[name][target].values
            if result is None:
                result = np.zeros_like(vals, dtype=float)
            result += w / total_w * vals
        return result

    def diversity_matrix(self, target: str = "Revenue") -> pd.DataFrame:
        data = {}
        for name, df in self._submissions.items():
            data[name] = df[target].values
        mat = pd.DataFrame(data)
        return mat.corr()

    def find_diverse_set(
        self, target: str = "Revenue", n: int = 3, min_diversity: float = 0.001
    ) -> list[str]:
        corr = self.diversity_matrix(target)
        names = list(corr.columns)
        if len(names) <= n:
            return names
        avg_corr = corr.mean()
        selected = [avg_corr.idxmin()]
        while len(selected) < n:
            best_name = None
            best_score = float("inf")
            for name in names:
                if name in selected:
                    continue
                max_corr = max((abs(corr.loc[name, s]) for s in selected))
                if max_corr < best_score:
                    best_score = max_corr
                    best_name = name
            if best_name is None or best_score > 1 - min_diversity:
                break
            selected.append(best_name)
        return selected


class LBReverseEngineer:
    def __init__(self, n_test_days: int = 548):
        self.n_test = n_test_days
        self._probes: list[dict] = []

    def add_probe(
        self,
        name_a: str,
        score_a: float,
        name_b: str,
        score_b: float,
        description: str = "",
    ) -> None:
        self._probes.append(
            {
                "name_a": name_a,
                "score_a": score_a,
                "name_b": name_b,
                "score_b": score_b,
                "delta": score_b - score_a,
                "description": description,
            }
        )

    def analyze(self) -> pd.DataFrame:
        return pd.DataFrame(self._probes)

    @staticmethod
    def infer_direction(
        pred_a: np.ndarray,
        score_a: float,
        pred_b: np.ndarray,
        score_b: float,
        n_days: int = 548,
    ) -> dict:
        diff = pred_b - pred_a
        mask_higher = diff > 0
        mask_lower = diff < 0
        total_shift = diff.sum()
        mae_delta = (score_b - score_a) * n_days
        return {
            "total_shift": total_shift,
            "mae_delta_total": mae_delta,
            "n_days_higher": mask_higher.sum(),
            "n_days_lower": mask_lower.sum(),
            "avg_shift_higher": diff[mask_higher].mean() if mask_higher.any() else 0,
            "avg_shift_lower": diff[mask_lower].mean() if mask_lower.any() else 0,
            "direction": "predictions_too_low"
            if mae_delta < 0
            else "predictions_too_high",
        }
