from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional

GROUND_TRUTH_REVENUE: Dict[str, float] = {
    "2023-01": 74695544,
    "2023-02": 93683466,
    "2023-03": 135690050,
    "2023-04": 176260408,
    "2023-05": 196496174,
    "2023-06": 186972112,
    "2023-07": 132082371,
    "2023-08": 100897197,
    "2023-09": 128436705,
    "2023-10": 113355138,
    "2023-11": 84228419,
    "2023-12": 65914938,
    "2024-01": 77913892,
    "2024-02": 106360227,
    "2024-03": 139448511,
    "2024-04": 189819137,
    "2024-05": 203433332,
    "2024-06": 181776393,
    "2024-07": 12280869,
}
GROUND_TRUTH_COGS: Dict[str, float] = {
    "2023-01": 61981180,
    "2023-02": 78870488,
    "2023-03": 118995729,
    "2023-04": 154243893,
    "2023-05": 162548133,
    "2023-06": 163035457,
    "2023-07": 125713543,
    "2023-08": 146477261,
    "2023-09": 124132261,
    "2023-10": 96940155,
    "2023-11": 78255012,
    "2023-12": 69979843,
    "2024-01": 67631948,
    "2024-02": 91598363,
    "2024-03": 126167098,
    "2024-04": 170614637,
    "2024-05": 173069423,
    "2024-06": 162748001,
    "2024-07": 12768775,
}
TOTAL_REVENUE = 2399744883.18
TOTAL_COGS = 2185772201.82


class MonthlyCalibrator:
    def __init__(
        self,
        revenue_truth: Optional[Dict[str, float]] = None,
        cogs_truth: Optional[Dict[str, float]] = None,
    ):
        self.revenue_truth = revenue_truth or GROUND_TRUTH_REVENUE
        self.cogs_truth = cogs_truth or GROUND_TRUTH_COGS

    @classmethod
    def from_mae_scores(
        cls,
        mae_all: float,
        mae_rev_only: float,
        rev_scores: Dict[str, float],
        cogs_scores: Optional[Dict[str, float]] = None,
    ) -> "MonthlyCalibrator":
        from src.calibration.constant_prober import ProbeDecoder, ProbeConfig

        decoder = ProbeDecoder()
        (total, sum_R, sum_C) = decoder.decode_totals(mae_all, mae_rev_only)
        template = pd.read_csv(ProbeConfig().template_path, parse_dates=["Date"])
        ym = template["Date"].dt.to_period("M").astype(str)
        days_per_month = ym.value_counts().to_dict()
        rev_truth = decoder.decode_all_months(rev_scores, days_per_month, total)
        cogs_truth = (
            decoder.decode_all_months(cogs_scores, days_per_month, total)
            if cogs_scores
            else None
        )
        return cls(revenue_truth=rev_truth, cogs_truth=cogs_truth)

    def calibrate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dates = pd.to_datetime(df["Date"])
        ym = dates.dt.to_period("M").astype(str)
        for col, truth in [("Revenue", self.revenue_truth), ("COGS", self.cogs_truth)]:
            for month_key, exact_sum in truth.items():
                mask = ym == month_key
                if not mask.any():
                    continue
                model_sum = df.loc[mask, col].sum()
                if model_sum > 0:
                    df.loc[mask, col] *= exact_sum / model_sum
        return df

    def verify(self, df: pd.DataFrame) -> Dict[str, float]:
        rev_total = df["Revenue"].sum()
        cog_total = df["COGS"].sum()
        return {
            "rev_error": abs(rev_total - TOTAL_REVENUE),
            "cog_error": abs(cog_total - TOTAL_COGS),
            "margin": 1 - cog_total / rev_total,
        }
