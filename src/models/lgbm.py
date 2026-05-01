from __future__ import annotations
import numpy as np
import lightgbm as lgb
from src.core.experiment_config import GBMParams
from src.config import SEED


class LGBMWrapper:
    def __init__(
        self,
        params: GBMParams | None = None,
        objective: str = "regression",
        seed: int = SEED,
    ):
        self.params = params or GBMParams()
        self.objective = objective
        self.seed = seed
        self.model: lgb.LGBMRegressor | None = None

    def _create_model(self, seed: int) -> lgb.LGBMRegressor:
        p = self.params.to_dict()
        p["random_state"] = seed
        p["verbosity"] = -1
        if self.objective != "regression":
            p["objective"] = self.objective
        return lgb.LGBMRegressor(**p)

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> LGBMWrapper:
        self.model = self._create_model(self.seed)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


class MultiSeedLGBM:
    def __init__(
        self,
        params: GBMParams | None = None,
        objective: str = "regression",
        seeds: list[int] | None = None,
    ):
        self.params = params or GBMParams()
        self.objective = objective
        self.seeds = seeds or [42, 123, 456]
        self._models: list[lgb.LGBMRegressor] = []

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> MultiSeedLGBM:
        self._models = []
        for seed in self.seeds:
            p = self.params.to_dict()
            p["random_state"] = seed
            p["verbosity"] = -1
            if self.objective != "regression":
                p["objective"] = self.objective
            m = lgb.LGBMRegressor(**p)
            m.fit(X, y, sample_weight=sample_weight)
            self._models.append(m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([m.predict(X) for m in self._models])
        return preds.mean(axis=0)
