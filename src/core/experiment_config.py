from __future__ import annotations
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class GBMParams:
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.08
    num_leaves: int = 15
    min_child_samples: int = 20
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GBMBaselineParams(GBMParams):
    n_estimators: int = 800
    max_depth: int = -1
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_child_samples: int = 30
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0


@dataclass
class QSpecialistConfig:
    enabled: bool = True
    alpha: float = 0.6
    q_boost: float = 2.0
    adaptive_alpha: bool = False
    alpha_q1: float = 0.6
    alpha_q2: float = 0.6
    alpha_q3: float = 0.7
    alpha_q4: float = 0.6

    def get_alpha(self, quarter: int) -> float:
        if not self.adaptive_alpha:
            return self.alpha
        return {1: self.alpha_q1, 2: self.alpha_q2, 3: self.alpha_q3, 4: self.alpha_q4}[
            quarter
        ]


@dataclass
class RidgeConfig:
    enabled: bool = True
    weight: float = 0.1
    alpha: float = 3.0
    seed: int = 42


@dataclass
class ProphetConfig:
    enabled: bool = False
    weight: float = 0.1
    yearly_seasonality: int = 10
    weekly_seasonality: bool = True
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0


@dataclass
class COGSConfig:
    mode: str = "uniform"
    scale: float = 1.03
    det_weight: float = 0.4
    independent_level: float = 0.0
    oddeven_ratios: dict = field(
        default_factory=lambda: {
            (1, 0): 0.8449,
            (1, 1): 0.8301,
            (2, 0): 0.8404,
            (2, 1): 0.8306,
            (3, 0): 0.8716,
            (3, 1): 1.057,
            (4, 0): 0.8897,
            (4, 1): 0.8917,
        }
    )
    oddeven_ratio_weight: float = 0.7
    per_month_cr: dict = field(default_factory=dict)
    per_month_weight: float = 0.0


@dataclass
class EraWeightConfig:
    scheme: str = "high_era"
    base_weight: float = 0.1
    ranges: dict = field(default_factory=lambda: {"2014-2018": 1.0, "2019-2022": 0.5})


@dataclass
class FeatureConfig:
    use_tet: bool = True
    tet_window_radius: int = 5
    tet_pre_radius: int = 14
    tet_post_radius: int = 20
    use_eom_edges: bool = True
    eom_last_days: int = 3
    bom_first_days: int = 3
    use_holidays: bool = True
    use_ecommerce_holidays: bool = True
    n_fourier_yearly: int = 5
    n_fourier_weekly: int = 2
    n_fourier_monthly: int = 2
    use_promotions: bool = True
    use_promo_timing: bool = True
    use_web_traffic: bool = True
    use_return_rate: bool = True
    use_odd_year: bool = True
    use_quarter: bool = True
    use_month_dow: bool = True
    use_weekend: bool = True
    use_dom_frac: bool = True
    use_regime: bool = False
    use_lag_seasonal: bool = False
    use_trend_anchor: bool = False
    use_macro_indices: bool = False


@dataclass
class BottomUpConfig:
    enabled: bool = True
    weight: float = 0.1
    growth_years: list = field(default_factory=lambda: [2020, 2021, 2022])
    default_growth: float = 1.1


@dataclass
class CVConfig:
    method: str = "folds"
    fold_years: list[int] = field(default_factory=lambda: [2022, 2021])
    ts_splits: int = 3
    ts_test_size: int = 182
    fold_c_enabled: bool = False
    fold_c_train_end: str = "2021-06-30"
    fold_c_test_end: str = "2022-06-30"


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    description: str = ""
    train_start_year: int = 2012
    train_end_year: int = 2022
    seasonal_method: str = "median"
    trimmed_pct: float = 0.1
    level: float = 4350
    det_weight: float = 0.4
    gbm_type: str = "lightgbm"
    gbm_params: GBMParams = field(default_factory=GBMParams)
    gbm_objective: str = "regression"
    n_seeds: int = 1
    seed_base: int = 42
    use_two_stage: bool = False
    early_stop_days: int = 180
    log_target: bool = False
    multi_model: bool = False
    multi_model_types: list = field(default_factory=lambda: ["lightgbm"])
    multi_model_weights: list = field(default_factory=lambda: [1.0])
    cogs_ratio: float = 1.03
    cogs_mode: str = "uniform"
    cogs_level: Optional[float] = None
    cv_val_start: str = "2022-01-01"
    cv_val_end: str = "2022-12-31"
    q4_2023_hack: float = 0.0
    final_revenue_level: float = 0.0
    final_cogs_level: float = 0.0
    q_specialist: QSpecialistConfig = field(default_factory=QSpecialistConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    prophet: ProphetConfig = field(default_factory=ProphetConfig)
    cogs: COGSConfig = field(default_factory=COGSConfig)
    era_weight: EraWeightConfig = field(default_factory=EraWeightConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    bottom_up: BottomUpConfig = field(default_factory=BottomUpConfig)
    cv: CVConfig = field(default_factory=CVConfig)

    @property
    def seed_list(self) -> list[int]:
        seeds = [42, 123, 456, 789, 1024, 2048, 3141, 4269, 5555, 7777]
        return seeds[: self.n_seeds]

    def to_yaml(self, path: Path) -> None:
        data = asdict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_nested_dict(data)

    @classmethod
    def _from_nested_dict(cls, data: dict) -> ExperimentConfig:
        d = data.copy()
        SUB_CONFIGS = {
            "gbm_params": GBMParams,
            "q_specialist": QSpecialistConfig,
            "ridge": RidgeConfig,
            "prophet": ProphetConfig,
            "cogs": COGSConfig,
            "era_weight": EraWeightConfig,
            "features": FeatureConfig,
            "bottom_up": BottomUpConfig,
            "cv": CVConfig,
        }
        for key, klass in SUB_CONFIGS.items():
            if key in d and isinstance(d[key], dict):
                sub = d[key]
                if key == "cogs" and "oddeven_ratios" in sub:
                    ratios = {}
                    for k, v in sub["oddeven_ratios"].items():
                        if isinstance(k, str):
                            parts = k.strip("()").split(",")
                            ratios[int(parts[0].strip()), int(parts[1].strip())] = v
                        else:
                            ratios[k] = v
                    sub["oddeven_ratios"] = ratios
                d[key] = klass(**sub)
        return cls(**d)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        return cls._from_nested_dict(d)

    def clone(self, **overrides) -> ExperimentConfig:
        data = asdict(self)
        data.update(overrides)
        return self.from_dict(data)

    def __repr__(self) -> str:
        parts = [
            f"name={self.name!r}",
            f"L={self.level}",
            f"dw={self.det_weight}",
            f"obj={self.gbm_objective}",
            f"seeds={self.n_seeds}",
        ]
        if self.q_specialist.enabled:
            parts.append(f"α={self.q_specialist.alpha}")
        if self.ridge.enabled:
            parts.append(f"ridge={self.ridge.weight}")
        if self.bottom_up.enabled:
            parts.append(f"BU={self.bottom_up.weight}")
        return f"ExperimentConfig({', '.join(parts)})"


GOLDEN_BASELINE = ExperimentConfig(
    name="golden_baseline",
    description="ult_w4_dw40_flat — proven 649K baseline",
    level=4350,
    det_weight=0.4,
    gbm_params=GBMParams(
        n_estimators=300, max_depth=4, num_leaves=15, learning_rate=0.08
    ),
    q_specialist=QSpecialistConfig(enabled=False),
    ridge=RidgeConfig(enabled=False),
    cogs=COGSConfig(scale=1.03),
    features=FeatureConfig(
        use_promotions=False,
        use_promo_timing=False,
        use_web_traffic=False,
        use_return_rate=False,
    ),
    bottom_up=BottomUpConfig(enabled=True, weight=0.1),
)
QSPEC_RIDGE_V1 = ExperimentConfig(
    name="qspec_ridge_v1",
    description="Best submission — 648K. Q-specialist + Ridge + promo timing.",
    level=4350,
    det_weight=0.4,
    gbm_params=GBMParams(
        n_estimators=300, max_depth=4, num_leaves=15, learning_rate=0.08
    ),
    q_specialist=QSpecialistConfig(enabled=True, alpha=0.6, q_boost=2.0),
    ridge=RidgeConfig(enabled=True, weight=0.1, alpha=3.0),
    cogs=COGSConfig(scale=1.03),
    era_weight=EraWeightConfig(scheme="high_era"),
    features=FeatureConfig(
        use_promotions=True,
        use_promo_timing=True,
        use_web_traffic=False,
        use_return_rate=False,
    ),
    bottom_up=BottomUpConfig(enabled=True, weight=0.1),
)
MAE_AUXILIARY = ExperimentConfig(
    name="mae_auxiliary",
    description="MAE loss + web traffic + return rate — 648,122 (current best).",
    level=4355,
    det_weight=0.4,
    gbm_objective="mae",
    gbm_params=GBMParams(
        n_estimators=300, max_depth=4, num_leaves=15, learning_rate=0.08
    ),
    q_specialist=QSpecialistConfig(enabled=True, alpha=0.6, q_boost=2.0),
    ridge=RidgeConfig(enabled=True, weight=0.1, alpha=3.0),
    cogs=COGSConfig(scale=1.03),
    era_weight=EraWeightConfig(scheme="high_era"),
    features=FeatureConfig(
        use_promotions=True,
        use_promo_timing=True,
        use_web_traffic=True,
        use_return_rate=True,
    ),
    bottom_up=BottomUpConfig(enabled=True, weight=0.1),
)
PROPHET_MAE = ExperimentConfig(
    name="prophet_mae",
    description="Full 3-tier: MAE + Prophet + Ridge + Q-spec + BU.",
    level=4350,
    det_weight=0.4,
    gbm_objective="mae",
    gbm_params=GBMParams(
        n_estimators=300, max_depth=4, num_leaves=15, learning_rate=0.08
    ),
    q_specialist=QSpecialistConfig(enabled=True, alpha=0.6, q_boost=2.0),
    ridge=RidgeConfig(enabled=True, weight=0.1, alpha=3.0),
    prophet=ProphetConfig(enabled=True, weight=0.1),
    cogs=COGSConfig(scale=1.03),
    era_weight=EraWeightConfig(scheme="high_era"),
    features=FeatureConfig(
        use_promotions=True,
        use_promo_timing=True,
        use_web_traffic=True,
        use_return_rate=True,
    ),
    bottom_up=BottomUpConfig(enabled=True, weight=0.1),
)
