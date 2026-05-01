from __future__ import annotations
import numpy as np
import pandas as pd
from src.core.experiment_config import ExperimentConfig
from src.data.loader import DataLoader
from src.features.temporal import TemporalFeatureBuilder, add_calendar_columns
from src.features.seasonal import SeasonalProfileBuilder
from src.models.lgbm import LGBMWrapper
from src.models.ridge import RidgeAnchor
from src.models.q_specialist import QSpecialist
from src.models.ensemble import TieredEnsemble
from src.calibration.era_weighting import build_sample_weights
from src.calibration.level_search import search_growth_factor


class CVResult:
    def __init__(self, fold_name: str, test_year: int):
        self.fold_name = fold_name
        self.test_year = test_year
        self.rev_mae: float = 0.0
        self.cogs_mae: float = 0.0
        self.rev_bias: float = 0.0
        self.cogs_bias: float = 0.0
        self.total_mae: float = 0.0
        self.cr_pred: float = 0.0
        self.cr_actual: float = 0.0

    def __repr__(self) -> str:
        return f"CVResult({self.fold_name}: Rev MAE={self.rev_mae:,.0f} COGS MAE={self.cogs_mae:,.0f} Total={self.total_mae:,.0f})"


def run_cv(
    config: ExperimentConfig, folds: list[int] | None = None, verbose: bool = True
) -> list[CVResult]:
    folds = folds or [config.cv.fold_a_test_year, config.cv.fold_b_test_year]
    sales = add_calendar_columns(DataLoader.sales())
    feat_builder = TemporalFeatureBuilder(config)
    results = []
    for test_yr in folds:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"CV Fold: Test Year {test_yr}")
            print(f"{'=' * 60}")
        train = sales[sales["year"] < test_yr].copy()
        test = sales[sales["year"] == test_yr].copy()
        ym_r = train.groupby("year")["Revenue"].transform("mean")
        train["Revenue_norm"] = train["Revenue"] / ym_r
        ym_c = train.groupby("year")["COGS"].transform("mean")
        train["COGS_norm"] = train["COGS"] / ym_c
        sp_r = SeasonalProfileBuilder(method=config.seasonal_method).build(
            train, "Revenue", config.train_start_year, test_yr - 1
        )
        sp_c = SeasonalProfileBuilder(method=config.seasonal_method).build(
            train, "COGS", config.train_start_year, test_yr - 1
        )
        det_r = sp_r.predict(test)
        det_c = sp_c.predict(test)
        X_tr = feat_builder.build(train).values
        X_te = feat_builder.build(test).values
        era_w = build_sample_weights(train["year"].values, config.era_weight)
        base_rev = LGBMWrapper(config.gbm_params, config.gbm_objective)
        base_rev.fit(X_tr, train["Revenue_norm"].values, sample_weight=era_w)
        gbm_r = base_rev.predict(X_te)
        base_cog = LGBMWrapper(config.gbm_params, config.gbm_objective)
        base_cog.fit(X_tr, train["COGS_norm"].values, sample_weight=era_w)
        gbm_c = base_cog.predict(X_te)
        spec_r = spec_c = None
        if config.q_specialist.enabled:
            qs_r = QSpecialist(
                config.q_specialist, config.gbm_params, config.gbm_objective
            )
            qs_r.fit(
                X_tr,
                train["Revenue_norm"].values,
                train["quarter"].values
                if "quarter" in train.columns
                else (train["month"].values - 1) // 3 + 1,
                era_w,
            )
            q_te = (
                test["quarter"].values
                if "quarter" in test.columns
                else (test["month"].values - 1) // 3 + 1
            )
            spec_r = qs_r.predict(X_te, q_te)
            qs_c = QSpecialist(
                config.q_specialist, config.gbm_params, config.gbm_objective
            )
            qs_c.fit(
                X_tr,
                train["COGS_norm"].values,
                train["quarter"].values
                if "quarter" in train.columns
                else (train["month"].values - 1) // 3 + 1,
                era_w,
            )
            spec_c = qs_c.predict(X_te, q_te)
        ridge_r = ridge_c = None
        if config.ridge.enabled:
            ra_r = RidgeAnchor(config.ridge.alpha, config.ridge.seed)
            ra_r.fit(X_tr, train["Revenue_norm"].values)
            ridge_r = ra_r.predict(X_te)
            ra_c = RidgeAnchor(config.ridge.alpha, config.ridge.seed)
            ra_c.fit(X_tr, train["COGS_norm"].values)
            ridge_c = ra_c.predict(X_te)
        ens = TieredEnsemble(
            det_weight=config.det_weight,
            ridge_weight=config.ridge.weight if config.ridge.enabled else 0.0,
            bu_weight=0.0,
        )
        final_r = ens.blend_all(
            det_r, gbm_r, spec_r, ridge_r, alpha=config.q_specialist.alpha
        )
        final_c = ens.blend_all(
            det_c, gbm_c, spec_c, ridge_c, alpha=config.q_specialist.alpha
        )
        base_rev_val = train[train["year"] == test_yr - 1]["Revenue"].mean()
        base_cog_val = train[train["year"] == test_yr - 1]["COGS"].mean()
        actual_rev = test["Revenue"].values
        actual_cog = test["COGS"].values
        g = search_growth_factor(base_rev_val, final_r, actual_rev.mean() / 1000)
        pred_rev = base_rev_val * g * final_r
        pred_cog = base_cog_val * g * config.cogs.scale * final_c
        res = CVResult(f"Fold_{test_yr}", test_yr)
        res.rev_mae = np.abs(pred_rev - actual_rev).mean()
        res.cogs_mae = np.abs(pred_cog - actual_cog).mean()
        res.rev_bias = (pred_rev - actual_rev).mean()
        res.cogs_bias = (pred_cog - actual_cog).mean()
        res.total_mae = res.rev_mae + res.cogs_mae
        res.cr_pred = pred_cog.sum() / pred_rev.sum()
        res.cr_actual = actual_cog.sum() / actual_rev.sum()
        if verbose:
            print(f"Rev MAE: {res.rev_mae:>12,.0f}  Bias: {res.rev_bias:+12,.0f}")
            print(f"COGS MAE: {res.cogs_mae:>11,.0f}  Bias: {res.cogs_bias:+12,.0f}")
            print(f"Total: {res.total_mae:>14,.0f}")
            print(f"CR pred={res.cr_pred:.4f} actual={res.cr_actual:.4f}")
        results.append(res)
    if verbose and len(results) > 1:
        avg_total = np.mean([r.total_mae for r in results])
        print(f"\n{'=' * 60}")
        print(f"AVG Total MAE: {avg_total:,.0f}")
    return results
