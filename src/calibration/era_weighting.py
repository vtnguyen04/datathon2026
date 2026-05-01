from __future__ import annotations
import numpy as np
import pandas as pd
from src.core.experiment_config import EraWeightConfig

SCHEMES = {
    "high_era": {"base": 0.1, "ranges": {(2014, 2018): 1.0, (2019, 2022): 0.5}},
    "t1718": {"base": 0.01, "ranges": {(2017, 2018): 1.0}},
    "up_HIGH": {"base": 0.01, "ranges": {(2014, 2018): 1.0, (2020, 2022): 0.5}},
    "uniform": {"base": 1.0, "ranges": {}},
    "recent_heavy": {"base": 0.1, "ranges": {(2014, 2018): 0.5, (2019, 2022): 1.0}},
    "post_only": {"base": 0.01, "ranges": {(2019, 2022): 1.0}},
}


def build_sample_weights(
    years: np.ndarray, config: EraWeightConfig | None = None, scheme: str | None = None
) -> np.ndarray:
    scheme_name = scheme or (config.scheme if config else "high_era")
    if scheme_name in SCHEMES:
        spec = SCHEMES[scheme_name]
        base = spec["base"]
        ranges = spec["ranges"]
    elif config and scheme_name == "custom":
        base = config.base_weight
        ranges = {}
        for k, v in config.ranges.items():
            (lo, hi) = map(int, k.split("-"))
            ranges[lo, hi] = v
    else:
        raise ValueError(
            f"Unknown scheme: {scheme_name}. Available: {list(SCHEMES.keys())}"
        )
    weights = np.full(len(years), base, dtype=float)
    for (lo, hi), val in ranges.items():
        mask = (years >= lo) & (years <= hi)
        weights[mask] = val
    return weights


def build_q_boosted_weights(
    base_weights: np.ndarray, quarters: np.ndarray, target_q: int, boost: float = 2.0
) -> np.ndarray:
    w = base_weights.copy()
    w[quarters == target_q] *= boost
    return w
