from __future__ import annotations
import numpy as np
import pandas as pd
from src.core.experiment_config import ExperimentConfig, FeatureConfig
from src.features.promotions import PromotionFeatureBuilder
from src.features.auxiliary import AuxiliaryFeatureBuilder


class TemporalFeatureBuilder:
    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config
        self.fc = config.features if config else FeatureConfig()
        self._promo_builder = PromotionFeatureBuilder(
            include_timing=self.fc.use_promo_timing
        )
        self._aux_builder = AuxiliaryFeatureBuilder()
        self._feature_names: list[str] = []

    def _get_seasonal_anchor(self, year: int) -> pd.Timestamp:
        anchors = {
            2012: (1, 23),
            2013: (2, 10),
            2014: (1, 31),
            2015: (2, 19),
            2016: (2, 8),
            2017: (1, 28),
            2018: (2, 16),
            2019: (2, 5),
            2020: (1, 25),
            2021: (2, 12),
            2022: (2, 1),
            2023: (1, 22),
            2024: (2, 10),
        }
        (m, d) = anchors.get(year, (1, 1))
        return pd.Timestamp(year, m, d)

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        dates = pd.to_datetime(df["Date"])
        year = dates.dt.year.values
        month = df["month"].values
        dow = df["dow"].values
        doy = dates.dt.dayofyear.values
        dom = dates.dt.day.values
        dim = dates.dt.days_in_month.values.astype(float)
        X["month"] = month
        X["dow"] = dow
        X["doy"] = doy
        X["woy"] = df["woy"].values
        if self.fc.use_tet:
            dist = np.zeros(len(df))
            for yr in np.unique(year):
                mask = year == yr
                anchor = self._get_seasonal_anchor(yr)
                dist[mask] = (dates[mask] - anchor).dt.days
            X["days_from_tet"] = dist
            wr = self.fc.tet_window_radius
            X["tet_window"] = ((dist >= -wr) & (dist <= wr * 2)).astype(int)
            X["pre_tet"] = ((dist >= -self.fc.tet_pre_radius) & (dist < -wr)).astype(
                int
            )
            X["post_tet"] = (
                (dist > wr * 2) & (dist <= self.fc.tet_post_radius)
            ).astype(int)
            X["tet_in_7"] = (np.abs(dist) <= 7).astype(int)
            X["tet_in_14"] = (np.abs(dist) <= 14).astype(int)
        X["dom"] = dom
        days_to_eom = (dim - dom).astype(int)
        days_from_som = (dom - 1).astype(int)
        if self.fc.use_eom_edges:
            X["is_eom"] = (dom >= 29).astype(int)
            X["is_bom"] = (dom <= 3).astype(int)
            X["days_to_eom"] = days_to_eom
            X["days_from_som"] = days_from_som
            for k in range(1, self.fc.eom_last_days + 1):
                X[f"is_last{k}"] = (days_to_eom <= k - 1).astype(int)
            for k in range(1, self.fc.bom_first_days + 1):
                X[f"is_first{k}"] = (days_from_som <= k - 1).astype(int)
        if self.fc.use_dom_frac:
            X["dom_frac"] = dom / dim
        if self.fc.use_odd_year:
            X["is_odd_year"] = (year % 2).astype(int)
        if self.fc.use_quarter:
            X["quarter"] = dates.dt.quarter
        if self.fc.use_month_dow:
            X["month_dow"] = month * 10 + dow
        if self.fc.use_weekend:
            X["is_weekend"] = (dow >= 5).astype(int)
        if self.fc.use_holidays:
            md = month * 100 + dom
            X["holiday_apr30"] = ((md >= 428) & (md <= 502)).astype(int)
            X["holiday_sep2"] = ((md >= 831) & (md <= 903)).astype(int)
            X["holiday_women_mar"] = ((md >= 305) & (md <= 309)).astype(int)
            X["holiday_xmas"] = ((md >= 1223) & (md <= 1226)).astype(int)
            X["holiday_newyear"] = ((md >= 1230) | (md <= 102)).astype(int)
        if self.fc.use_ecommerce_holidays:
            md = month * 100 + dom
            X["hol_1111"] = ((md >= 1109) & (md <= 1112)).astype(int)
            X["hol_1212"] = ((md >= 1210) & (md <= 1213)).astype(int)
        TAU = 2 * np.pi
        for k in range(1, self.fc.n_fourier_yearly + 1):
            X[f"sin_y{k}"] = np.sin(TAU * k * doy / 365.25)
            X[f"cos_y{k}"] = np.cos(TAU * k * doy / 365.25)
        for k in range(1, self.fc.n_fourier_weekly + 1):
            X[f"sin_w{k}"] = np.sin(TAU * k * dow / 7.0)
            X[f"cos_w{k}"] = np.cos(TAU * k * dow / 7.0)
        for k in range(1, self.fc.n_fourier_monthly + 1):
            X[f"sin_m{k}"] = np.sin(TAU * k * (dom - 1) / dim)
            X[f"cos_m{k}"] = np.cos(TAU * k * (dom - 1) / dim)
        if self.fc.use_regime:
            X["regime_pre2019"] = (year <= 2018).astype(int)
            X["regime_post2019"] = (year >= 2020).astype(int)
            X["t_days"] = (dates - pd.Timestamp("2020-01-01")).dt.days
        if self.fc.use_promotions:
            X = pd.concat([X, self._promo_builder.build(df)], axis=1)
        if self.fc.use_web_traffic or self.fc.use_return_rate:
            X = pd.concat(
                [
                    X,
                    self._aux_builder.build(
                        df,
                        use_web_traffic=self.fc.use_web_traffic,
                        use_return_rate=self.fc.use_return_rate,
                    ),
                ],
                axis=1,
            )
        if hasattr(self.fc, "use_macro_indices") and self.fc.use_macro_indices:
            from src.features.macro_indices import MacroIndicesBuilder

            X = pd.concat([X, MacroIndicesBuilder.build(df)], axis=1)
        self._feature_names = list(X.columns)
        return X

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dow"] = df["Date"].dt.dayofweek
    df["doy"] = df["Date"].dt.dayofyear
    df["woy"] = df["Date"].dt.isocalendar().week.astype(int)
    return df
