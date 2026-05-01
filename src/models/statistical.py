from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Protocol, Optional


class ShapePredictor(Protocol):
    def fit(self, dates: pd.Series, values: np.ndarray) -> None: ...

    def predict(self, dates: pd.Series) -> np.ndarray: ...


class HoltWintersForecaster:
    def __init__(
        self,
        seasonal_periods: int = 365,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool = True,
    ):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self._model = None

    def fit(self, dates: pd.Series, values: np.ndarray) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self._model = ExponentialSmoothing(
            values,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend,
            initialization_method="estimated",
        ).fit(optimized=True)

    def predict(self, dates: pd.Series) -> np.ndarray:
        return self._model.forecast(len(dates))


class FourierRegressionForecaster:
    def __init__(self, n_harmonics: int = 6, poly_degree: int = 2):
        self.n_harmonics = n_harmonics
        self.poly_degree = poly_degree
        self._coefs: Optional[np.ndarray] = None
        self._t0: Optional[float] = None

    def _build_features(self, dates: pd.Series) -> np.ndarray:
        doy = dates.dt.dayofyear.values.astype(float)
        t = (dates - dates.min()).dt.days.values.astype(float)
        if self._t0 is None:
            self._t0 = 0
        t = t + self._t0
        features = []
        for d in range(self.poly_degree + 1):
            features.append((t / 365.25) ** d)
        for k in range(1, self.n_harmonics + 1):
            features.append(np.sin(2 * np.pi * k * doy / 365.25))
            features.append(np.cos(2 * np.pi * k * doy / 365.25))
        return np.column_stack(features)

    def fit(self, dates: pd.Series, values: np.ndarray) -> None:
        self._t0 = 0
        X = self._build_features(dates)
        from numpy.linalg import lstsq

        (self._coefs, _, _, _) = lstsq(X, values, rcond=None)
        self._train_end = dates.max()
        self._train_start = dates.min()

    def predict(self, dates: pd.Series) -> np.ndarray:
        days_offset = (dates.min() - self._train_start).days
        self._t0 = days_offset
        X = self._build_features(dates)
        self._t0 = 0
        return X @ self._coefs


class SeasonalNaiveForecaster:
    def __init__(self, lookback_years: int = 3):
        self.lookback_years = lookback_years
        self._seasonal_index: Optional[dict] = None

    def fit(self, dates: pd.Series, values: np.ndarray) -> None:
        df = pd.DataFrame({"date": dates, "value": values})
        df["doy"] = df["date"].dt.dayofyear
        df["year"] = df["date"].dt.year
        max_year = df["year"].max()
        recent = df[df["year"] >= max_year - self.lookback_years + 1]
        self._seasonal_index = recent.groupby("doy")["value"].median().to_dict()
        self._fallback = values.mean()

    def predict(self, dates: pd.Series) -> np.ndarray:
        doy = dates.dt.dayofyear.values
        return np.array([self._seasonal_index.get(d, self._fallback) for d in doy])


class StatisticalEnsemble:
    def __init__(self, models: list[tuple[str, ShapePredictor, float]] | None = None):
        if models is None:
            self.models = [
                (
                    "fourier_reg",
                    FourierRegressionForecaster(n_harmonics=6, poly_degree=2),
                    0.4,
                ),
                ("seasonal_naive", SeasonalNaiveForecaster(lookback_years=3), 0.3),
                ("seasonal_naive_5y", SeasonalNaiveForecaster(lookback_years=5), 0.3),
            ]
        else:
            self.models = models

    def fit(self, dates: pd.Series, values: np.ndarray) -> None:
        for name, model, _ in self.models:
            try:
                model.fit(dates, values)
            except Exception as e:
                print(f"  Warning: {name} failed to fit: {e}")

    def predict(self, dates: pd.Series) -> np.ndarray:
        preds = []
        weights = []
        for name, model, w in self.models:
            try:
                p = model.predict(dates)
                if np.isfinite(p).all():
                    preds.append(p)
                    weights.append(w)
            except Exception as e:
                print(f"  Warning: {name} failed to predict: {e}")
        if not preds:
            raise ValueError("All statistical models failed")
        total_w = sum(weights)
        result = sum((w / total_w * p for (w, p) in zip(weights, preds)))
        return result

    def predict_individual(self, dates: pd.Series) -> dict[str, np.ndarray]:
        results = {}
        for name, model, _ in self.models:
            try:
                p = model.predict(dates)
                if np.isfinite(p).all():
                    results[name] = p
            except Exception:
                pass
        return results
