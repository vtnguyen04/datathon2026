import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/quynhthu/Documents/datathon")

import numpy as np
import pandas as pd
import zipfile
from pathlib import Path

from src.core.experiment_config import (
    ExperimentConfig, GBMParams, QSpecialistConfig, 
    RidgeConfig, COGSConfig, FeatureConfig, BottomUpConfig
)
from src.pipelines.forecast import ForecastPipeline
from src.ensemble.blend import ModelBlender, BlendConfig

import os
import shutil
SUBS = Path("data/submissions")
if SUBS.exists():
    shutil.rmtree(SUBS)
SUBS.mkdir(exist_ok=True, parents=True)

def main():
    print("Starting forecasting pipeline...")

    print("Training Model A...")
    cfg_a = ExperimentConfig(
        name="no_cal_a", train_start_year=2013,
        level=4300, det_weight=0.425,
        gbm_objective="mae", n_seeds=5,
        gbm_params=GBMParams(n_estimators=300, max_depth=4, learning_rate=0.05, num_leaves=31),
        q_specialist=QSpecialistConfig(enabled=True, alpha=0.54, q_boost=3.5),
        ridge=RidgeConfig(enabled=True, weight=0.015, alpha=91.2),
        features=FeatureConfig(use_promotions=True, use_promo_timing=True, use_web_traffic=True, n_fourier_yearly=6),
        cogs=COGSConfig(scale=1.06),
        bottom_up=BottomUpConfig(enabled=False),
        q4_2023_hack=0.10
    )
    ForecastPipeline(cfg_a, verbose=False).run_submission()

    print("Training Model B...")
    cfg_b = ExperimentConfig(
        name="no_cal_b", train_start_year=2014,
        level=4400, det_weight=0.50,
        gbm_objective="huber", n_seeds=3,
        q_specialist=QSpecialistConfig(enabled=False),
        ridge=RidgeConfig(enabled=False),
        q4_2023_hack=0.10
    )
    ForecastPipeline(cfg_b, verbose=False).run_submission()

    print("Training Model C...")
    cfg_c = ExperimentConfig(
        name="no_cal_c", train_start_year=2013,
        level=4300, det_weight=0.35,
        gbm_objective="mae", n_seeds=3,
        q_specialist=QSpecialistConfig(enabled=False)
    )
    ForecastPipeline(cfg_c, verbose=False).run_submission()

    print("Blending models...")
    df_a = pd.read_csv(SUBS / "no_cal_a.csv", parse_dates=["Date"])
    df_b = pd.read_csv(SUBS / "no_cal_b.csv", parse_dates=["Date"])
    df_c = pd.read_csv(SUBS / "no_cal_c.csv", parse_dates=["Date"])
    
    blender = ModelBlender(BlendConfig(weights=[0.70, 0.15, 0.15]))
    base_blend = blender.blend([df_a, df_b, df_c])

    csv_final = SUBS / "submission.csv"
    df_save = base_blend.copy()
    df_save["Date"] = pd.to_datetime(df_save["Date"]).dt.strftime("%Y-%m-%d")
    df_save["Revenue"] = df_save["Revenue"].round(2)
    df_save["COGS"] = df_save["COGS"].round(2)
    df_save.to_csv(csv_final, index=False)

    for p in [SUBS / "no_cal_a.csv", SUBS / "no_cal_b.csv", SUBS / "no_cal_c.csv"]:
        if p.exists():
            p.unlink()
        if p.with_suffix(".csv.zip").exists():
            p.with_suffix(".csv.zip").unlink()

    print(f"Pipeline completed. Output saved to {csv_final}")

if __name__ == "__main__":
    main()
