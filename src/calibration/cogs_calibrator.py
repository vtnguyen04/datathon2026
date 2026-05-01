from __future__ import annotations
import numpy as np
from src.core.experiment_config import COGSConfig


def calibrate_cogs_uniform(
    base_cogs: float, growth: float, shape: np.ndarray, scale: float = 1.03
) -> np.ndarray:
    return base_cogs * growth * scale * shape


def calibrate_cogs_independent(
    base_cogs: float,
    shape: np.ndarray,
    target_level_thousands: float,
    scale: float = 1.0,
) -> np.ndarray:
    from src.calibration.level_search import search_growth_factor

    g_cogs = search_growth_factor(base_cogs, shape, target_level_thousands)
    return base_cogs * g_cogs * scale * shape


def calibrate_cogs_oddeven(
    revenue_pred: np.ndarray,
    cogs_model_pred: np.ndarray,
    quarters: np.ndarray,
    is_odd: np.ndarray,
    config: COGSConfig,
    ratio_weight: float | None = None,
) -> np.ndarray:
    rw = ratio_weight or config.oddeven_ratio_weight
    cogs_ratio = np.zeros(len(revenue_pred))
    for q in [1, 2, 3, 4]:
        for odd in [0, 1]:
            mask = (quarters == q) & (is_odd == bool(odd))
            cr = config.oddeven_ratios.get((q, odd), 0.88)
            cogs_ratio[mask] = revenue_pred[mask] * cr
    return rw * cogs_ratio + (1 - rw) * cogs_model_pred


def calibrate_cogs_permonth(
    revenue_pred: np.ndarray,
    cogs_model_pred: np.ndarray,
    months: np.ndarray,
    is_odd: np.ndarray,
    config: COGSConfig,
) -> np.ndarray:
    if config.per_month_weight <= 0 or not config.per_month_cr:
        return cogs_model_pred
    pmw = config.per_month_weight
    cogs_pm = np.zeros(len(revenue_pred))
    for m in range(1, 13):
        for odd in [0, 1]:
            mask = (months == m) & (is_odd == bool(odd))
            cr = config.per_month_cr.get((m, odd), 0.88)
            cogs_pm[mask] = revenue_pred[mask] * cr
    return pmw * cogs_pm + (1 - pmw) * cogs_model_pred
