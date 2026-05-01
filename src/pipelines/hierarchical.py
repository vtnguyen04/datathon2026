import pandas as pd
import numpy as np
import logging
import json
from src.config import RAW_DIR
from src.core.experiment_config import ExperimentConfig
from src.features.hierarchical import HierarchicalDataLoader
from src.features.seasonal import SeasonalProfileBuilder
from src.features.temporal import TemporalFeatureBuilder
from src.models.factory import create_model

logger = logging.getLogger(__name__)


class HierarchicalPipeline:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_loader = HierarchicalDataLoader()
        self.seasonal_builder = SeasonalProfileBuilder(method=config.seasonal_method)
        self.feature_builder = TemporalFeatureBuilder(config)

    def _run_single_category(
        self, df_cat: pd.DataFrame, category_name: str, cv_mode=False
    ) -> pd.DataFrame:
        if cv_mode:
            train = df_cat[
                (df_cat["Date"].dt.year >= self.config.train_start_year)
                & (df_cat["Date"] < "2021-01-01")
            ].copy()
            test = df_cat[
                (df_cat["Date"] >= "2021-01-01") & (df_cat["Date"] <= "2022-06-30")
            ].copy()
            for df_ in (train, test):
                if "year" not in df_.columns:
                    df_["year"] = df_["Date"].dt.year
                    df_["month"] = df_["Date"].dt.month
                    df_["day"] = df_["Date"].dt.day
                    df_["dow"] = df_["Date"].dt.dayofweek
                    df_["doy"] = df_["Date"].dt.dayofyear
                    df_["woy"] = df_["Date"].dt.isocalendar().week.astype(int)
            base_year = train[train["Date"].dt.year == 2020]
            prev_year = train[train["Date"].dt.year == 2019]
        else:
            train = df_cat[
                df_cat["Date"].dt.year >= self.config.train_start_year
            ].copy()
            test_dates = pd.date_range(start="2023-01-01", end="2024-12-31")
            test = pd.DataFrame({"Date": test_dates})
            for df_ in (train, test):
                if "year" not in df_.columns:
                    df_["year"] = df_["Date"].dt.year
                    df_["month"] = df_["Date"].dt.month
                    df_["day"] = df_["Date"].dt.day
                    df_["dow"] = df_["Date"].dt.dayofweek
                    df_["doy"] = df_["Date"].dt.dayofyear
                    df_["woy"] = df_["Date"].dt.isocalendar().week.astype(int)
            base_year = train[train["Date"].dt.year == 2022]
            prev_year = train[train["Date"].dt.year == 2021]
        base_rev = base_year["Revenue"].mean()
        prev_rev = prev_year["Revenue"].mean()
        if prev_rev > 0:
            growth_rate = base_rev / prev_rev
        else:
            growth_rate = 1.0
        growth_rate = np.clip(growth_rate, 0.8, 1.3)
        test["years_elapsed"] = test["Date"].dt.year - base_year["Date"].dt.year.iloc[0]
        yearly_mean = train.groupby(train["Date"].dt.year)["Revenue"].transform("mean")
        train["rn"] = np.where(yearly_mean > 0, train["Revenue"] / yearly_mean, 1.0)
        seasonal_profile = self.seasonal_builder.build(train, target_col="rn")
        det_shape = seasonal_profile.predict(test)
        train_feat = self.feature_builder.build(train)
        test_feat = self.feature_builder.build(test)
        features = [
            c
            for c in train_feat.columns
            if c not in ["Date", "Revenue", "COGS", "rn", "category", "years_elapsed"]
        ]
        preds = []
        for seed in range(self.config.n_seeds):
            model = create_model(self.config, seed=seed)
            model.fit(train_feat[features], train["rn"])
            pred = model.predict(test_feat[features])
            pred = np.maximum(pred, 0.01)
            preds.append(pred)
        gbm_shape = np.mean(preds, axis=0)
        w = self.config.det_weight
        blend_shape = w * det_shape + (1 - w) * gbm_shape
        projected_base = base_rev * growth_rate ** test["years_elapsed"]
        test["Revenue_pred"] = projected_base * blend_shape
        base_cogs = base_year["COGS"].mean()
        if base_rev > 0:
            cat_cogs_ratio = base_cogs / base_rev
        else:
            cat_cogs_ratio = 0.88
        projected_cogs_base = base_cogs * growth_rate ** test["years_elapsed"]
        test["COGS_pred"] = projected_cogs_base * blend_shape * self.config.cogs_ratio
        test["category"] = category_name
        return test

    def run(self, cv_mode=False) -> pd.DataFrame:
        logger.info(f"Loading hierarchical data...")
        df_all = self.data_loader.load_category_data()
        categories = df_all["category"].unique()
        logger.info(f"Found categories: {categories}")
        results = []
        for cat in categories:
            logger.info(f"Processing category: {cat}...")
            df_cat = df_all[df_all["category"] == cat].copy()
            if df_cat["Revenue"].sum() == 0:
                continue
            res = self._run_single_category(df_cat, cat, cv_mode=cv_mode)
            results.append(res)
        final_df = pd.concat(results)
        total_daily = (
            final_df.groupby("Date")[["Revenue_pred", "COGS_pred"]].sum().reset_index()
        )
        total_daily.rename(
            columns={"Revenue_pred": "Revenue", "COGS_pred": "COGS"}, inplace=True
        )
        if cv_mode:
            orders = pd.read_csv(RAW_DIR / "sales.csv", parse_dates=["Date"])
            cv_actual = orders[
                (orders["Date"] >= "2021-01-01") & (orders["Date"] <= "2022-06-30")
            ]
            merged = total_daily.merge(
                cv_actual, on="Date", suffixes=("_pred", "_actual")
            )
            mae_rev = np.abs(merged["Revenue_pred"] - merged["Revenue_actual"]).mean()
            mae_cogs = np.abs(merged["COGS_pred"] - merged["COGS_actual"]).mean()
            logger.info(
                f"Hierarchical CV MAE - Revenue: {mae_rev:,.0f}, COGS: {mae_cogs:,.0f}, Total: {mae_rev + mae_cogs:,.0f}"
            )
            return {
                "mae_rev": mae_rev,
                "mae_cogs": mae_cogs,
                "mae_total": mae_rev + mae_cogs,
                "df": merged,
            }
        return total_daily
