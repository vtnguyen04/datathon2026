from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features.temporal import MultiHorizonFeatureEngineer
from src.core.base import FeatureTransformer


class DataPrepPipeline:
    @staticmethod
    def build_pipeline():
        return Pipeline([("scaler", StandardScaler())])
