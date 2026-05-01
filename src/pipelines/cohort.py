import pandas as pd
import numpy as np
import logging
from src.config import RAW_DIR
from src.core.experiment_config import ExperimentConfig
from src.features.cohort import CohortDataLoader
from src.features.seasonal import SeasonalProfileBuilder
from src.features.temporal import TemporalFeatureBuilder
from src.models.factory import create_model

logger = logging.getLogger(__name__)


class CohortPipeline:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_loader = CohortDataLoader()
        self.seasonal_builder = SeasonalProfileBuilder(method=config.seasonal_method)
        self.feature_builder = TemporalFeatureBuilder(config)

    def _run_sub_model(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_dates: pd.DatetimeIndex,
        cv_mode=False,
    ) -> pd.DataFrame:
        df = df.copy()
        df["Revenue"] = df[target_col]
        if cv_mode:
            train = df[
                (df["Date"].dt.year >= self.config.train_start_year)
                & (df["Date"] < "2021-01-01")
            ].copy()
            test = df[
                (df["Date"] >= "2021-01-01") & (df["Date"] <= "2022-06-30")
            ].copy()
            base_year = train[train["Date"].dt.year == 2020]
            prev_year = train[train["Date"].dt.year == 2019]
        else:
            train = df[df["Date"].dt.year >= self.config.train_start_year].copy()
            test = pd.DataFrame({"Date": test_dates})
            base_year = train[train["Date"].dt.year == 2022]
            prev_year = train[train["Date"].dt.year == 2021]
        for df_ in (train, test):
            if "year" not in df_.columns:
                df_["year"] = df_["Date"].dt.year
                df_["month"] = df_["Date"].dt.month
                df_["day"] = df_["Date"].dt.day
                df_["dow"] = df_["Date"].dt.dayofweek
                df_["doy"] = df_["Date"].dt.dayofyear
                df_["woy"] = df_["Date"].dt.isocalendar().week.astype(int)
        base_rev = base_year["Revenue"].mean()
        prev_rev = prev_year["Revenue"].mean()
        growth_rate = base_rev / prev_rev if prev_rev > 0 else 1.0
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
            and (not c.endswith("_users"))
            and (not c.endswith("_revenue"))
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
        return projected_base * blend_shape

    def run(self, cv_mode=False) -> pd.DataFrame:
        logger.info(f"Loading cohort data...")
        df = self.data_loader.build_cohort_data()
        test_dates = (
            pd.date_range(start="2023-01-01", end="2024-12-31") if not cv_mode else None
        )
        logger.info(f"Modeling Returning Revenue...")
        returning_pred = self._run_sub_model(
            df, "returning_revenue", test_dates, cv_mode
        )
        logger.info(f"Modeling New Revenue...")
        new_pred = self._run_sub_model(df, "new_revenue", test_dates, cv_mode)
        if cv_mode:
            test_df = df[
                (df["Date"] >= "2021-01-01") & (df["Date"] <= "2022-06-30")
            ].copy()
        else:
            test_df = pd.DataFrame({"Date": test_dates})
        test_df["returning_revenue_pred"] = (
            returning_pred.values
            if isinstance(returning_pred, pd.Series)
            else returning_pred
        )
        test_df["new_revenue_pred"] = (
            new_pred.values if isinstance(new_pred, pd.Series) else new_pred
        )
        test_df["Revenue_pred"] = (
            test_df["returning_revenue_pred"] + test_df["new_revenue_pred"]
        )
        test_df["COGS_pred"] = test_df["Revenue_pred"] * 0.88 * self.config.cogs_ratio
        test_df.rename(
            columns={"Revenue_pred": "Revenue", "COGS_pred": "COGS"}, inplace=True
        )
        if cv_mode:
            orders = pd.read_csv(RAW_DIR / "sales.csv", parse_dates=["Date"])
            cv_actual = orders[
                (orders["Date"] >= "2021-01-01") & (orders["Date"] <= "2022-06-30")
            ]
            merged = test_df.merge(cv_actual, on="Date", suffixes=("_pred", "_actual"))
            mae_rev = np.abs(merged["Revenue_pred"] - merged["Revenue_actual"]).mean()
            mae_cogs = np.abs(merged["COGS_pred"] - merged["COGS_actual"]).mean()
            logger.info(
                f"Cohort CV MAE - Revenue: {mae_rev:,.0f}, COGS: {mae_cogs:,.0f}, Total: {mae_rev + mae_cogs:,.0f}"
            )
            return {
                "mae_rev": mae_rev,
                "mae_cogs": mae_cogs,
                "mae_total": mae_rev + mae_cogs,
                "df": merged,
            }
        return test_df[["Date", "Revenue", "COGS"]]
