import xgboost as xgb
import pandas as pd
from src.core.base import BaseForecaster
from src.config import SEED


class XGBWrapper(BaseForecaster):
    def __init__(self, **kwargs):
        params = {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "random_state": SEED,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        params.update(kwargs)
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        if eval_set:
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X))

    @property
    def feature_importances_(self):
        return self.model.feature_importances_
