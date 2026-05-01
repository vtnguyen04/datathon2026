import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.config import FIGURES_DIR
import os


class PipelineEvaluator:
    @staticmethod
    def evaluate(model, X: pd.DataFrame, y: pd.Series):
        preds = model.predict(X)
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        print(f"Metrics — R²: {r2:.4f} | MAE: {mae:,.2f} | RMSE: {rmse:,.2f}")
        return (r2, mae, rmse)

    @staticmethod
    def generate_shap(model, X: pd.DataFrame, feature_names: list, model_name="lgbm"):
        print(f"Calculating SHAP values for {model_name}...")
        model_core = getattr(model, "lgbm", model)
        if hasattr(model_core, "model"):
            model_core = model_core.model
        explainer = shap.TreeExplainer(model_core)
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(FIGURES_DIR / "shap_summary.png", bbox_inches="tight", dpi=250)
        plt.close()
        print(f"SHAP summary saved to {FIGURES_DIR}/shap_summary.png")
