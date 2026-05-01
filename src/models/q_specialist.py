from __future__ import annotations
import numpy as np
import lightgbm as lgb
from src.core.experiment_config import ExperimentConfig, QSpecialistConfig, GBMParams
from src.calibration.era_weighting import build_q_boosted_weights


class QSpecialist:
    def __init__(
        self,
        config: QSpecialistConfig,
        gbm_params: GBMParams,
        objective: str = "regression",
        seed: int = 42,
    ):
        self.config = config
        self.gbm_params = gbm_params
        self.objective = objective
        self.seed = seed
        self._models: dict[int, lgb.LGBMRegressor] = {}

    def _create_lgb(self) -> lgb.LGBMRegressor:
        params = self.gbm_params.to_dict()
        params["random_state"] = self.seed
        params["verbosity"] = -1
        if self.objective != "regression":
            params["objective"] = self.objective
        return lgb.LGBMRegressor(**params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        quarters: np.ndarray,
        base_weights: np.ndarray,
    ) -> QSpecialist:
        for q in [1, 2, 3, 4]:
            qw = build_q_boosted_weights(
                base_weights, quarters, target_q=q, boost=self.config.q_boost
            )
            model = self._create_lgb()
            model.fit(X, y, sample_weight=qw)
            self._models[q] = model
        return self

    def predict(self, X: np.ndarray, quarters: np.ndarray) -> np.ndarray:
        preds = np.zeros(len(X))
        for q in [1, 2, 3, 4]:
            mask = quarters == q
            if mask.any():
                preds[mask] = self._models[q].predict(X[mask])
        return preds

    def blend_with_base(
        self,
        specialist_pred: np.ndarray,
        base_pred: np.ndarray,
        quarters: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.config.adaptive_alpha and quarters is not None:
            result = np.zeros_like(specialist_pred)
            for q in [1, 2, 3, 4]:
                mask = quarters == q
                alpha = self.config.get_alpha(q)
                result[mask] = (
                    alpha * specialist_pred[mask] + (1 - alpha) * base_pred[mask]
                )
            return result
        else:
            alpha = self.config.alpha
            return alpha * specialist_pred + (1 - alpha) * base_pred
