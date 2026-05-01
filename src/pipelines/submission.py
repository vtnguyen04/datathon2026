from __future__ import annotations
import logging
import zipfile
import numpy as np
import pandas as pd

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
from src.config import SUBMISSION_DIR
from src.core.experiment_config import ExperimentConfig
from src.data.loader import DataLoader
from src.features.temporal import TemporalFeatureBuilder, add_calendar_columns
from src.features.seasonal import SeasonalProfileBuilder
from src.models.lgbm import LGBMWrapper, MultiSeedLGBM
from src.models.ridge import RidgeAnchor
from src.models.q_specialist import QSpecialist
from src.calibration.era_weighting import build_sample_weights
from src.calibration.level_search import search_growth_factor, calibrate_predictions
from src.calibration.cogs_calibrator import (
    calibrate_cogs_uniform,
    calibrate_cogs_oddeven,
)
from src.pipelines.bottom_up import BottomUpForecaster


def _train_base_gbm(config, X_tr, y_tr, X_sub, era_w, verbose):
    if config.n_seeds > 1:
        model = MultiSeedLGBM(config.gbm_params, config.gbm_objective, config.seed_list)
        model.fit(X_tr, y_tr, sample_weight=era_w)
        if verbose:
            print(f"  Base GBM: {config.n_seeds}-seed averaged")
    else:
        model = LGBMWrapper(config.gbm_params, config.gbm_objective, config.seed_base)
        model.fit(X_tr, y_tr, sample_weight=era_w)
        if verbose:
            print(f"  Base GBM: seed={config.seed_base}")
    return model.predict(X_sub)


def _train_q_specialist(config, X_tr, y_tr, X_sub, q_tr, q_sub, era_w):
    qs = QSpecialist(
        config.q_specialist, config.gbm_params, config.gbm_objective, config.seed_base
    )
    qs.fit(X_tr, y_tr, q_tr, era_w)
    return qs.predict(X_sub, q_sub)


def _train_ridge(config, X_tr, y_tr, X_sub):
    ra = RidgeAnchor(config.ridge.alpha, config.ridge.seed)
    ra.fit(X_tr, y_tr)
    return ra.predict(X_sub)


def _train_prophet(config, dates_tr, y_tr, dates_sub):
    from src.models.prophet_model import ProphetModel

    pm = ProphetModel(
        yearly_seasonality=config.prophet.yearly_seasonality,
        weekly_seasonality=config.prophet.weekly_seasonality,
        changepoint_prior_scale=config.prophet.changepoint_prior_scale,
        seasonality_prior_scale=config.prophet.seasonality_prior_scale,
    )
    pm.fit(dates_tr, y_tr)
    return pm.predict(dates_sub)


def _blend_tier1(base, specialist, alpha):
    if specialist is not None:
        return alpha * specialist + (1 - alpha) * base
    return base


def _blend_tier2(tier1, ridge, prophet, ridge_w, prophet_w):
    tier1_w = 1.0 - ridge_w - prophet_w
    result = tier1_w * tier1
    if ridge is not None:
        result = result + ridge_w * ridge
    else:
        result = result + ridge_w * tier1
    if prophet is not None:
        result = result + prophet_w * prophet
    else:
        result = result + prophet_w * tier1
    return result


def _blend_final(det, tier2, det_weight):
    return det_weight * det + (1 - det_weight) * tier2


def generate_submission(config: ExperimentConfig, verbose: bool = True) -> str:
    sales = add_calendar_columns(DataLoader.sales())
    sub = add_calendar_columns(DataLoader.sample_submission())
    train = sales[
        sales["year"].between(config.train_start_year, config.train_end_year)
    ].copy()
    if verbose:
        print(f"=== {config.name} ===")
    ym_r = train.groupby("year")["Revenue"].transform("mean")
    train["Revenue_norm"] = train["Revenue"] / ym_r
    ym_c = train.groupby("year")["COGS"].transform("mean")
    train["COGS_norm"] = train["COGS"] / ym_c
    sp_r = SeasonalProfileBuilder(method=config.seasonal_method).build(
        train, "Revenue", config.train_start_year, config.train_end_year
    )
    sp_c = SeasonalProfileBuilder(method=config.seasonal_method).build(
        train, "COGS", config.train_start_year, config.train_end_year
    )
    det_r = sp_r.predict(sub)
    det_c = sp_c.predict(sub)
    feat_builder = TemporalFeatureBuilder(config)
    X_tr = feat_builder.build(train).values
    X_sub = feat_builder.build(sub).values
    era_w = build_sample_weights(train["year"].values, config.era_weight)
    if verbose:
        print(f"  Features: {X_tr.shape[1]}")
    gbm_r = _train_base_gbm(
        config, X_tr, train["Revenue_norm"].values, X_sub, era_w, verbose
    )
    gbm_c = _train_base_gbm(
        config, X_tr, train["COGS_norm"].values, X_sub, era_w, False
    )
    spec_r = spec_c = None
    if config.q_specialist.enabled:
        q_tr = train["Date"].dt.quarter.values
        q_sub = sub["Date"].dt.quarter.values
        spec_r = _train_q_specialist(
            config, X_tr, train["Revenue_norm"].values, X_sub, q_tr, q_sub, era_w
        )
        spec_c = _train_q_specialist(
            config, X_tr, train["COGS_norm"].values, X_sub, q_tr, q_sub, era_w
        )
        if verbose:
            print(f"  Q-Spec: α={config.q_specialist.alpha}")
    tier1_r = _blend_tier1(gbm_r, spec_r, config.q_specialist.alpha)
    tier1_c = _blend_tier1(gbm_c, spec_c, config.q_specialist.alpha)
    ridge_r = ridge_c = None
    if config.ridge.enabled:
        ridge_r = _train_ridge(config, X_tr, train["Revenue_norm"].values, X_sub)
        ridge_c = _train_ridge(config, X_tr, train["COGS_norm"].values, X_sub)
        if verbose:
            print(f"  Ridge: w={config.ridge.weight}")
    prophet_r = prophet_c = None
    if config.prophet.enabled:
        if verbose:
            print("  Prophet: training...")
        prophet_r = _train_prophet(
            config, train["Date"], train["Revenue_norm"].values, sub["Date"]
        )
        prophet_c = _train_prophet(
            config, train["Date"], train["COGS_norm"].values, sub["Date"]
        )
        if verbose:
            print(f"  Prophet: w={config.prophet.weight}")
    ridge_w = config.ridge.weight if config.ridge.enabled else 0.0
    prophet_w = config.prophet.weight if config.prophet.enabled else 0.0
    tier2_r = _blend_tier2(tier1_r, ridge_r, prophet_r, ridge_w, prophet_w)
    tier2_c = _blend_tier2(tier1_c, ridge_c, prophet_c, ridge_w, prophet_w)
    final_r = _blend_final(det_r, tier2_r, config.det_weight)
    final_c = _blend_final(det_c, tier2_c, config.det_weight)
    base_rev = train[train["year"] == config.train_end_year]["Revenue"].mean()
    base_cog = train[train["year"] == config.train_end_year]["COGS"].mean()
    g = search_growth_factor(base_rev, final_r, config.level)
    pred_rev = calibrate_predictions(base_rev, g, final_r)
    if config.cogs.mode == "uniform":
        pred_cog = calibrate_cogs_uniform(base_cog, g, final_c, config.cogs.scale)
    elif config.cogs.mode == "oddeven":
        model_cog = calibrate_cogs_uniform(base_cog, g, final_c, config.cogs.scale)
        pred_cog = calibrate_cogs_oddeven(
            pred_rev,
            model_cog,
            sub["Date"].dt.quarter.values,
            (sub["Date"].dt.year % 2 == 1).values,
            config.cogs,
        )
    else:
        pred_cog = calibrate_cogs_uniform(base_cog, g, final_c, config.cogs.scale)
    if config.bottom_up.enabled:
        bu = BottomUpForecaster(
            growth_years=config.bottom_up.growth_years,
            default_growth=config.bottom_up.default_growth,
        )
        (bu_rev, bu_cog) = bu.predict(sub)
        bu_s = pred_rev.mean() / bu_rev.mean()
        w = config.bottom_up.weight
        pred_rev = (1 - w) * pred_rev + w * bu_rev * bu_s
        pred_cog = (1 - w) * pred_cog + w * bu_cog * bu_s
        if verbose:
            print(f"  BU: w={w}")
    is23 = sub["Date"].dt.year == 2023
    is24 = sub["Date"].dt.year == 2024
    cr23 = pred_cog[is23].sum() / pred_rev[is23].sum()
    cr24 = pred_cog[is24].sum() / pred_rev[is24].sum()
    if verbose:
        print(f"  Rev={pred_rev.mean():,.0f} CR23={cr23:.4f} CR24={cr24:.4f}")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    result = sub[["Date"]].copy()
    result["Revenue"] = np.round(pred_rev, 2)
    result["COGS"] = np.round(pred_cog, 2)
    result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
    csv_path = SUBMISSION_DIR / f"{config.name}.csv"
    zip_path = SUBMISSION_DIR / f"{config.name}.csv.zip"
    result.to_csv(csv_path, index=False)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, f"{config.name}.csv")
    if verbose:
        print(f"  ✅ {zip_path.name}")
    return str(zip_path)
