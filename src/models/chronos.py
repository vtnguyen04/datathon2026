import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class ChronosForecaster:
    def __init__(self, model_id: str = "autogluon/chronos-2-small", device: str = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None

    def _load_model(self):
        if self.pipeline is None:
            from chronos import BaseChronosPipeline

            print(
                f"[ChronosForecaster] Loading model {self.model_id} on {self.device}..."
            )
            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id, device_map=self.device, torch_dtype=torch.float32
            )

    def predict_with_covariates(
        self, context_df: pd.DataFrame, future_df: pd.DataFrame, horizon: int
    ) -> pd.DataFrame:
        self._load_model()
        print(
            f"[ChronosForecaster] Forecasting {horizon} steps using Covariates (Cross-Learning: ON)"
        )
        forecast_df = self.pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="item_id",
            timestamp_column="Date",
            target="target",
            cross_learning=True,
            batch_size=2,
        )
        return forecast_df
