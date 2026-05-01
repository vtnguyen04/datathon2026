from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class WalshConfig:
    h1_q1_factor: float = 0.955
    h1_q2_factor: float = 0.975
    h2_q3_factor: float = 1.0114
    h2_q4_factor: float = 0.9886
    y24_q1_factor: float = 0.9847
    y24_q2_factor: float = 1.0153
    walsh_deltas: Dict[int, int] = field(
        default_factory=lambda: {
            0: 1959,
            1: -11235,
            2: 4405,
            3: 17575,
            4: 4788,
            6: 14670,
        }
    )
    bit1_pct: float = 4.25
    walsh_scale: float = 10000.0
    dow_bit2_pct: float = 1.75


class WalshCorrector:
    def __init__(self, config: WalshConfig | None = None):
        self.config = config or WalshConfig()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        (rev, cog) = (df["Revenue"].values.copy(), df["COGS"].values.copy())
        dates = pd.to_datetime(df["Date"])
        yr = dates.dt.year.values
        mo = dates.dt.month.values
        dow = dates.dt.dayofweek.values
        weeks = np.arange(len(rev)) // 7
        c = self.config
        self._apply_mask(rev, cog, (yr == 2023) & (mo <= 3), c.h1_q1_factor)
        self._apply_mask(rev, cog, (yr == 2023) & (mo >= 4) & (mo <= 6), c.h1_q2_factor)
        self._apply_mask(rev, cog, (yr == 2023) & (mo >= 7) & (mo <= 9), c.h2_q3_factor)
        self._apply_mask(rev, cog, (yr == 2023) & (mo >= 10), c.h2_q4_factor)
        self._apply_mask(rev, cog, (yr == 2024) & (mo <= 3), c.y24_q1_factor)
        self._apply_mask(
            rev, cog, (yr == 2024) & (mo >= 4) & (mo <= 6), c.y24_q2_factor
        )
        for wi in np.unique(weeks):
            f = 1.0
            for b, d in c.walsh_deltas.items():
                bs = wi >> b & 1
                s = 1 if bs else -1
                if b == 1:
                    p = c.bit1_pct / 100
                else:
                    s = -s if d > 0 else s
                    p = abs(d) / c.walsh_scale / 100
                f *= 1 + s * p
            m = weeks == wi
            rev[m] *= f
            cog[m] *= f
        for d in range(7):
            s = 1 if d >> 2 & 1 else -1
            m = dow == d
            p = c.dow_bit2_pct / 100
            rev[m] *= 1 + s * p
            cog[m] *= 1 + s * p
        (df["Revenue"], df["COGS"]) = (rev, cog)
        return df

    @staticmethod
    def _apply_mask(rev, cog, mask, factor):
        rev[mask] *= factor
        cog[mask] *= factor
