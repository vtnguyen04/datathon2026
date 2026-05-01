from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import RAW_DIR


class AuxiliaryFeatureBuilder:
    def __init__(self, raw_dir: Path | None = None):
        self.raw_dir = raw_dir or RAW_DIR
        self._wt_profile: dict | None = None
        self._ret_profile: dict | None = None

    def _build_web_traffic_profile(self) -> dict[tuple[int, int], float]:
        wt = pd.read_csv(self.raw_dir / "web_traffic.csv", parse_dates=["date"])
        daily = wt.groupby("date").agg(sessions=("sessions", "sum")).reset_index()
        daily["year"] = daily["date"].dt.year
        daily["month"] = daily["date"].dt.month
        daily["dow"] = daily["date"].dt.dayofweek
        ym = daily.groupby("year")["sessions"].transform("mean")
        daily["s_norm"] = daily["sessions"] / ym
        return daily.groupby(["month", "dow"])["s_norm"].median().to_dict()

    def _build_return_rate_profile(self) -> dict[tuple[int, int], float]:
        sales = pd.read_csv(self.raw_dir / "sales.csv", parse_dates=["Date"])
        returns = pd.read_csv(self.raw_dir / "returns.csv", parse_dates=["return_date"])
        daily_ret = (
            returns.groupby("return_date")
            .agg(refund=("refund_amount", "sum"))
            .reset_index()
        )
        daily_ret.columns = ["Date", "refund"]
        merged = sales[["Date", "Revenue"]].merge(daily_ret, on="Date", how="left")
        merged["refund"] = merged["refund"].fillna(0)
        merged["ret_rate"] = merged["refund"] / merged["Revenue"].clip(lower=1)
        merged["month"] = merged["Date"].dt.month
        merged["dow"] = merged["Date"].dt.dayofweek
        return merged.groupby(["month", "dow"])["ret_rate"].median().to_dict()

    def fit(self) -> AuxiliaryFeatureBuilder:
        self._wt_profile = self._build_web_traffic_profile()
        self._ret_profile = self._build_return_rate_profile()
        return self

    def build(
        self,
        df: pd.DataFrame,
        use_web_traffic: bool = True,
        use_return_rate: bool = True,
    ) -> pd.DataFrame:
        if self._wt_profile is None:
            self.fit()
        result = pd.DataFrame(index=df.index)
        if use_web_traffic:
            result["wt_sessions"] = [
                self._wt_profile.get((m, d), 1.0)
                for (m, d) in zip(df["month"], df["dow"])
            ]
        if use_return_rate:
            result["ret_rate"] = [
                self._ret_profile.get((m, d), 0.035)
                for (m, d) in zip(df["month"], df["dow"])
            ]
        return result

    @property
    def feature_names(self) -> list[str]:
        return ["wt_sessions", "ret_rate"]
