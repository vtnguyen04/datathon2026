from abc import ABC, abstractmethod
import pandas as pd


class FeatureTransformer(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame, target_col: str = "Revenue"):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class BaseForecaster(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

    @property
    @abstractmethod
    def feature_importances_(self):
        pass
