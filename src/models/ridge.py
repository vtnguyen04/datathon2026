from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge as SkRidge


class RidgeAnchor:
    def __init__(self, alpha: float = 3.0, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self.model = SkRidge(alpha=alpha, random_state=seed)
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> RidgeAnchor:
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1
        X_scaled = (X - self._mu) / self._sigma
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = (X - self._mu) / self._sigma
        return self.model.predict(X_scaled)
