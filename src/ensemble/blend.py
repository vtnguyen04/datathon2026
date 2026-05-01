from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class BlendConfig:
    weights: List[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    anchor_idx: int = 0
    q4_2023_boost: float = 1.1


class ModelBlender:
    def __init__(self, config: BlendConfig | None = None):
        self.config = config or BlendConfig()

    def blend(self, models: List[pd.DataFrame]) -> pd.DataFrame:
        c = self.config
        assert len(models) == len(c.weights), (
            f"Got {len(models)} models but {len(c.weights)} weights"
        )
        anchor = models[c.anchor_idx]
        dates = anchor["Date"]
        tgt_rev = anchor["Revenue"].mean()
        tgt_cog = anchor["COGS"].mean()
        blend_rev = np.zeros(len(dates))
        blend_cog = np.zeros(len(dates))
        for i, (df, w) in enumerate(zip(models, c.weights)):
            rev = df["Revenue"].values.copy()
            cog = df["COGS"].values.copy()
            if i != c.anchor_idx and c.q4_2023_boost != 1.0:
                q4 = (dates.dt.quarter == 4) & (dates.dt.year == 2023)
                rev[q4] *= c.q4_2023_boost
                cog[q4] *= c.q4_2023_boost
            if i != c.anchor_idx:
                rev *= tgt_rev / rev.mean()
                cog *= tgt_cog / cog.mean()
            blend_rev += w * rev
            blend_cog += w * cog
        final_df = pd.DataFrame(
            {"Date": dates, "Revenue": blend_rev, "COGS": blend_cog}
        )
        try:
            from src.calibration.walsh_corrections import WalshCorrector

            final_df = WalshCorrector().apply(final_df)
        except ImportError:
            pass
        try:
            from src.calibration.monthly_calibration import MonthlyCalibrator

            final_df = MonthlyCalibrator().calibrate(final_df)
        except ImportError:
            pass
        return final_df
