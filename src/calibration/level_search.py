from __future__ import annotations
import numpy as np


def search_growth_factor(
    base_revenue: float,
    shape: np.ndarray,
    target_level_thousands: float,
    n_iter: int = 200,
) -> float:
    target = target_level_thousands * 1000
    (lo, hi) = (0.01, 20.0)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        if (base_revenue * mid * shape).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def calibrate_predictions(
    base_value: float, growth: float, shape: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    return base_value * growth * scale * shape
