from __future__ import annotations
import numpy as np
import pandas as pd
import logging

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
from prophet import Prophet


class ProphetModel:
    def __init__(
        self,
        yearly_seasonality: int = 10,
        weekly_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self._model: Prophet | None = None

    def fit(self, dates: pd.Series, values: np.ndarray) -> ProphetModel:
        df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})
        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
        )
        self._model.fit(df)
        return self

    def predict(self, dates: pd.Series) -> np.ndarray:
        df = pd.DataFrame({"ds": pd.to_datetime(dates)})
        forecast = self._model.predict(df)
        return forecast["yhat"].values
