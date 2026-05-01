from __future__ import annotations
import numpy as np
import pandas as pd
from src.config import RAW_DIR
from src.features.temporal import add_calendar_columns


class BottomUpForecaster:
    def __init__(
        self, growth_years: list[int] | None = None, default_growth: float = 1.1
    ):
        self.growth_years = growth_years or [2020, 2021, 2022]
        self.default_growth = default_growth
        self._daily_cat: pd.DataFrame | None = None

    def _load_category_daily(self) -> pd.DataFrame:
        orders = pd.read_csv(RAW_DIR / "orders.csv", parse_dates=["order_date"])
        items = pd.read_csv(RAW_DIR / "order_items.csv", low_memory=False)
        products = pd.read_csv(RAW_DIR / "products.csv")
        items = items.merge(
            products[["product_id", "category", "cogs"]], on="product_id", how="left"
        )
        items["line_rev"] = (
            items["quantity"] * items["unit_price"] - items["discount_amount"]
        )
        items["line_cogs"] = items["quantity"] * items["cogs"]
        items = items.merge(orders[["order_id", "order_date"]], on="order_id")
        daily = (
            items.groupby([items["order_date"].dt.date, "category"])
            .agg(rev=("line_rev", "sum"), cogs=("line_cogs", "sum"))
            .reset_index()
        )
        daily.columns = ["Date", "category", "Revenue", "COGS"]
        daily["Date"] = pd.to_datetime(daily["Date"])
        return add_calendar_columns(daily)

    def fit(self) -> BottomUpForecaster:
        self._daily_cat = self._load_category_daily()
        return self

    def predict(self, sub_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._daily_cat is None:
            self.fit()
        is_2023 = sub_df["Date"].dt.year == 2023
        is_2024 = sub_df["Date"].dt.year == 2024
        total_rev = np.zeros(len(sub_df))
        total_cog = np.zeros(len(sub_df))
        for cat in sorted(self._daily_cat["category"].unique()):
            cd = self._daily_cat[self._daily_cat["category"] == cat]
            cr = np.zeros(len(sub_df))
            cc = np.zeros(len(sub_df))
            years = sub_df["Date"].dt.year.unique()
            for y in years:
                mask = sub_df["Date"].dt.year == y
                train_cutoff = y - 1
                cd_tr = cd[cd["year"].between(2012, train_cutoff)].copy()
                if len(cd_tr) == 0:
                    continue
                ym_r = cd_tr.groupby("year")["Revenue"].transform("mean")
                cd_tr["rn"] = cd_tr["Revenue"] / ym_r
                ym_c = cd_tr.groupby("year")["COGS"].transform("mean")
                cd_tr["cn"] = cd_tr["COGS"] / ym_c
                sp_r = cd_tr.groupby(["month", "dow"])["rn"].median()
                sp_c = cd_tr.groupby(["month", "dow"])["cn"].median()
                dr = np.array(
                    [
                        sp_r.get((m, d), 1.0)
                        for (m, d) in zip(sub_df["month"], sub_df["dow"])
                    ]
                )
                dc = np.array(
                    [
                        sp_c.get((m, d), 1.0)
                        for (m, d) in zip(sub_df["month"], sub_df["dow"])
                    ]
                )
                yearly = cd_tr.groupby("year")["Revenue"].mean()
                base_rev = yearly.get(y - 1, yearly.iloc[-1] if len(yearly) > 0 else 0)
                cogs_past = cd_tr[cd_tr["year"] == y - 1]["COGS"]
                base_cog = (
                    cogs_past.mean() if len(cogs_past) > 0 else cd_tr["COGS"].mean()
                )
                if y >= 2023:
                    safe_growth = self.default_growth
                else:
                    rates = []
                    for yr in self.growth_years:
                        if (
                            yr in yearly.index
                            and yr - 1 in yearly.index
                            and (yr <= train_cutoff)
                        ):
                            rates.append(yearly[yr] / yearly[yr - 1])
                    safe_growth = np.mean(rates) if rates else self.default_growth
                exponent = 1
                if y == 2024:
                    exponent = 2
                    base_rev = yearly.get(2022, base_rev)
                    cogs_past = cd_tr[cd_tr["year"] == 2022]["COGS"]
                    base_cog = cogs_past.mean() if len(cogs_past) > 0 else base_cog
                    safe_growth = self.default_growth
                cr[mask] = base_rev * safe_growth**exponent * dr[mask]
                cc[mask] = base_cog * safe_growth**exponent * dc[mask]
            total_rev += cr
            total_cog += cc
        return (total_rev, total_cog)
