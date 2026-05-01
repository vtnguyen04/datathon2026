from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


class SeasonalProfile:
    def __init__(self, month_day_index: pd.Series, dow_adjustment: pd.Series):
        self.month_day_index = month_day_index
        self.dow_adjustment = dow_adjustment

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        result = np.ones(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            (m, d, dw) = (int(row["month"]), int(row["day"]), int(row["dow"]))
            md_val = self.month_day_index.get((m, d), np.nan)
            if np.isnan(md_val):
                v1 = self.month_day_index.get((2, 28), 1.0)
                v2 = self.month_day_index.get((3, 1), 1.0)
                md_val = (v1 + v2) / 2
            dow_val = self.dow_adjustment.get(dw, 1.0)
            result[i] = md_val * dow_val
        return result


class SeasonalProfileBuilder:
    SUPPORTED_METHODS = ("mean", "median", "trimmed")

    def __init__(self, method: str = "median", trimmed_pct: float = 0.1):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Supported: {self.SUPPORTED_METHODS}"
            )
        self.method = method
        self.trimmed_pct = trimmed_pct

    def _aggregate(self, series: pd.Series) -> float:
        if self.method == "mean":
            return series.mean()
        elif self.method == "median":
            return series.median()
        elif self.method == "trimmed":
            return stats.trim_mean(series.values, self.trimmed_pct)
        raise ValueError(f"Unknown method: {self.method}")

    def build(
        self,
        train: pd.DataFrame,
        target_col: str,
        start_year: int = 2012,
        end_year: int = 2022,
    ) -> SeasonalProfile:
        df = train[train["Date"].dt.year.between(start_year, end_year)].copy()
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["day"] = df["Date"].dt.day
        df["dow"] = df["Date"].dt.dayofweek
        yearly_mean = df.groupby("year")[target_col].transform("mean")
        df["normalized"] = df[target_col] / yearly_mean
        month_day_index = df.groupby(["month", "day"])["normalized"].agg(
            self._aggregate
        )
        global_agg = self._aggregate(df["normalized"])
        dow_agg = df.groupby("dow")["normalized"].agg(self._aggregate)
        dow_adjustment = dow_agg / global_agg
        return SeasonalProfile(month_day_index, dow_adjustment)
