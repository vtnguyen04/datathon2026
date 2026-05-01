from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PromoCampaign:
    name: str
    start_month: int
    start_day: int
    duration_days: int
    discount_pct: Optional[float]
    recurrence: str


PROMO_CAMPAIGNS = [
    PromoCampaign("spring_sale", 3, 18, 30, 12.0, "all"),
    PromoCampaign("mid_year", 6, 23, 29, 18.0, "all"),
    PromoCampaign("fall_launch", 8, 30, 32, 10.0, "all"),
    PromoCampaign("year_end", 11, 18, 45, 20.0, "all"),
    PromoCampaign("urban_blowout", 7, 30, 33, None, "odd"),
    PromoCampaign("rural_special", 1, 30, 30, 15.0, "odd"),
]


class PromotionFeatureBuilder:
    def __init__(
        self, campaigns: list[PromoCampaign] | None = None, include_timing: bool = True
    ):
        self.campaigns = campaigns or PROMO_CAMPAIGNS
        self.include_timing = include_timing

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        dates = pd.to_datetime(df["Date"])
        years = sorted(set(dates.dt.year.tolist()))
        result = pd.DataFrame(index=df.index)
        for camp in self.campaigns:
            in_promo = np.zeros(len(df), dtype=int)
            since = np.full(len(df), -1.0)
            until = np.full(len(df), -1.0)
            for y in range(min(years) - 1, max(years) + 2):
                if camp.recurrence == "odd" and y % 2 == 0:
                    continue
                if camp.recurrence == "even" and y % 2 == 1:
                    continue
                try:
                    start = pd.Timestamp(
                        year=y, month=camp.start_month, day=camp.start_day
                    )
                except ValueError:
                    continue
                end = start + pd.Timedelta(days=camp.duration_days)
                mask = (dates >= start) & (dates <= end)
                in_promo[mask.values] = 1
                since[mask.values] = (dates[mask] - start).dt.days.values
                until[mask.values] = (end - dates[mask]).dt.days.values
            result[f"promo_{camp.name}"] = in_promo
            if self.include_timing:
                result[f"promo_{camp.name}_since"] = since
                result[f"promo_{camp.name}_until"] = until
        return result

    @property
    def feature_names(self) -> list[str]:
        names = []
        for camp in self.campaigns:
            names.append(f"promo_{camp.name}")
            if self.include_timing:
                names.append(f"promo_{camp.name}_since")
                names.append(f"promo_{camp.name}_until")
        return names
