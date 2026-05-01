from __future__ import annotations
from src.core.experiment_config import ExperimentConfig, GBMParams


def create_model(config: ExperimentConfig, target: str = "revenue", seed: int = 42):
    p = config.gbm_params.to_dict()
    p["random_state"] = seed
    if config.gbm_type == "lightgbm":
        import lightgbm as lgb

        objective = config.gbm_objective
        if objective == "mse":
            objective = "regression"
        return lgb.LGBMRegressor(objective=objective, verbose=-1, **p)
    elif config.gbm_type == "xgboost":
        import xgboost as xgb

        obj_map = {
            "regression": "reg:squarederror",
            "mae": "reg:absoluteerror",
            "huber": "reg:pseudohubererror",
        }
        return xgb.XGBRegressor(
            objective=obj_map.get(config.gbm_objective, "reg:squarederror"),
            verbosity=0,
            random_state=seed,
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            reg_alpha=p["reg_alpha"],
            reg_lambda=p["reg_lambda"],
        )
    elif config.gbm_type == "catboost":
        from catboost import CatBoostRegressor

        loss_map = {"regression": "RMSE", "mae": "MAE", "huber": "Huber"}
        return CatBoostRegressor(
            iterations=p["n_estimators"],
            depth=min(p["max_depth"], 10),
            learning_rate=p["learning_rate"],
            l2_leaf_reg=p.get("reg_lambda", 0.1),
            random_seed=seed,
            verbose=0,
            loss_function=loss_map.get(config.gbm_objective, "RMSE"),
        )
    else:
        raise ValueError(
            f"Unknown gbm_type '{config.gbm_type}'. Supported: lightgbm, xgboost, catboost"
        )
