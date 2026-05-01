import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "ai-data-science-team")
)
from langchain_openai import ChatOpenAI

OPENROUTER_API_KEY = (
    "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"
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "z-ai/glm-4.5-air:free"
RAW_DIR = "data/raw"
FIGURES_DIR = "reports/figures"
SUBMISSION_DIR = "reports/submissions"
LOGS_DIR = "logs"
SEED = 42


def get_llm():
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0,
        request_timeout=120,
    )


def load_and_merge_data():
    print("📥 Bước 1: Load & Merge dữ liệu...")
    sales = pd.read_csv(os.path.join(RAW_DIR, "sales.csv"), parse_dates=["Date"])
    traffic = pd.read_csv(
        os.path.join(RAW_DIR, "web_traffic.csv"), parse_dates=["date"]
    )
    orders = pd.read_csv(
        os.path.join(RAW_DIR, "orders.csv"), parse_dates=["order_date"]
    )
    traffic_daily = (
        traffic.groupby("date")
        .agg(
            {
                "sessions": "sum",
                "unique_visitors": "sum",
                "page_views": "sum",
                "bounce_rate": "mean",
                "avg_session_duration_sec": "mean",
            }
        )
        .reset_index()
        .rename(columns={"date": "Date"})
    )
    orders_daily = (
        orders.groupby("order_date")
        .agg(
            daily_order_count=("order_id", "count"),
            daily_cancel_count=("order_status", lambda x: (x == "cancelled").sum()),
        )
        .reset_index()
        .rename(columns={"order_date": "Date"})
    )
    orders_daily["daily_cancel_rate"] = (
        orders_daily["daily_cancel_count"] / orders_daily["daily_order_count"]
    )
    df = sales.merge(traffic_daily, on="Date", how="left")
    df = df.merge(
        orders_daily[["Date", "daily_order_count", "daily_cancel_rate"]],
        on="Date",
        how="left",
    )
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"   ✅ Merged dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def run_feature_engineering_agent(df):
    print("\n🤖 Bước 2: FeatureEngineeringAgent đang tư duy...")
    try:
        from ai_data_science_team.agents import FeatureEngineeringAgent

        llm = get_llm()
        os.makedirs(os.path.join(LOGS_DIR, "fe_agent"), exist_ok=True)
        fe_agent = FeatureEngineeringAgent(
            model=llm,
            log=True,
            log_path=os.path.join(LOGS_DIR, "fe_agent"),
            overwrite=True,
        )
        instructions = "\n        This is a TIME-SERIES forecasting task. Target variable is 'Revenue'.\n        There is a 'Date' column (string, format YYYY-MM-DD). Generate these features:\n\n        1. First, convert 'Date' to datetime using pd.to_datetime().\n        2. Lag features for 'Revenue': lag_1, lag_7, lag_14, lag_30\n        3. Lag features for 'COGS': lag_1, lag_7\n        4. Rolling statistics for 'Revenue': rolling_7d_mean, rolling_7d_std, rolling_30d_mean\n        5. Rolling statistics for 'sessions': rolling_3d_sum, rolling_7d_mean\n        6. Gross margin ratio: (Revenue_lag_1 - COGS_lag_1) / Revenue_lag_1\n        7. Calendar features from 'Date': year, month, day, dayofweek, is_weekend, quarter\n        8. Drop the original 'Date' column after extracting calendar features.\n        9. Drop ALL rows with NaN values caused by lagging/rolling (dropna).\n        10. Return a single clean DataFrame with all features + target 'Revenue'.\n\n        IMPORTANT: Do NOT scale or normalize numeric features. Do NOT one-hot encode anything.\n        This is for tree-based models (LightGBM) which handle raw numerics natively.\n        "
        df_for_agent = df.copy()
        if "Date" in df_for_agent.columns and hasattr(df_for_agent["Date"].dtype, "tz"):
            df_for_agent["Date"] = df_for_agent["Date"].astype(str)
        elif "Date" in df_for_agent.columns:
            df_for_agent["Date"] = df_for_agent["Date"].dt.strftime("%Y-%m-%d")
        fe_agent.invoke_agent(
            data_raw=df_for_agent,
            user_instructions=instructions,
            target_variable="Revenue",
        )
        df_engineered = fe_agent.get_data_engineered()
        if df_engineered is not None and len(df_engineered) > 0:
            print(
                f"   ✅ Agent tạo thành công {len(df_engineered.columns)} features, {len(df_engineered)} rows"
            )
            fe_code = fe_agent.get_feature_engineer_function()
            if fe_code:
                code_path = os.path.join(LOGS_DIR, "fe_agent", "generated_fe_code.py")
                with open(code_path, "w") as f:
                    f.write(fe_code)
                print(f"   📝 Code Agent sinh ra đã lưu tại: {code_path}")
            return df_engineered
        else:
            print("   ⚠️ Agent trả về dữ liệu rỗng. Chuyển sang FE thủ công...")
    except Exception as e:
        print(f"   ⚠️ Agent gặp lỗi: {e}")
        print("   🔧 Chuyển sang Feature Engineering thủ công (fallback)...")
    return manual_feature_engineering(df)


def manual_feature_engineering(df):
    print("   🔧 Đang chạy Feature Engineering thủ công (fallback)...")
    df = df.copy()
    for lag in [1, 7, 14, 30]:
        df[f"Revenue_lag_{lag}"] = df["Revenue"].shift(lag)
        if lag <= 7:
            df[f"COGS_lag_{lag}"] = df["COGS"].shift(lag)
    df["Revenue_rolling_7d_mean"] = df["Revenue"].rolling(7).mean()
    df["Revenue_rolling_7d_std"] = df["Revenue"].rolling(7).std()
    df["Revenue_rolling_30d_mean"] = df["Revenue"].rolling(30).mean()
    if "sessions" in df.columns:
        df["sessions_rolling_3d_sum"] = df["sessions"].rolling(3).sum()
        df["sessions_rolling_7d_mean"] = df["sessions"].rolling(7).mean()
    df["Gross_Margin_lag1"] = (df["Revenue_lag_1"] - df["COGS_lag_1"]) / df[
        "Revenue_lag_1"
    ]
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["quarter"] = df["Date"].dt.quarter
    df = df.drop(columns=["Date"])
    df = df.dropna().reset_index(drop=True)
    print(f"   ✅ Manual FE: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def train_lightgbm(df_engineered, target_col="Revenue"):
    print("\n🚂 Bước 3: Huấn luyện LightGBM...")
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X = df_engineered.drop(columns=[target_col])
    y = df_engineered[target_col]
    tscv = TimeSeriesSplit(n_splits=5)
    metrics_list = []
    best_model = None
    best_rmse = float("inf")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        (X_train, X_val) = (X.iloc[train_idx], X.iloc[val_idx])
        (y_train, y_val) = (y.iloc[train_idx], y.iloc[val_idx])
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=SEED,
            verbosity=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        metrics_list.append({"fold": fold + 1, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"   Fold {fold + 1}: MAE={mae:,.0f} | RMSE={rmse:,.0f} | R²={r2:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
    print("\n   🔄 Retrain trên TOÀN BỘ dữ liệu...")
    final_model = lgb.LGBMRegressor(
        n_estimators=best_model.n_estimators_,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        verbosity=-1,
    )
    final_model.fit(X, y)
    print("   ✅ LightGBM đào tạo xong!")
    return (final_model, X.columns.tolist(), metrics_list)


def generate_shap_plots(model, X, feature_names):
    print("\n📊 Bước 4: Sinh SHAP & Feature Importance plots...")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)
    importances = model.feature_importances_
    feat_imp = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    (fig, ax) = plt.subplots(figsize=(10, 8))
    ax.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1], color="#4ECDC4")
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("Top 20 Feature Importances — LightGBM Revenue Forecaster")
    plt.tight_layout()
    fi_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"   ✅ Feature Importance saved: {fi_path}")
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        X_sample = pd.DataFrame(X if len(X) <= 500 else X[:500], columns=feature_names)
        shap_values = explainer.shap_values(X_sample)
        (fig, ax) = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
        shap_path = os.path.join(FIGURES_DIR, "shap_summary.png")
        plt.savefig(shap_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ SHAP Summary saved: {shap_path}")
    except ImportError:
        print("   ⚠️ SHAP chưa cài. Bỏ qua SHAP plot (Feature Importance đã đủ).")
    except Exception as e:
        print(f"   ⚠️ SHAP lỗi: {e}. Bỏ qua.")


def generate_submission(model, df_train_raw, feature_names):
    print("\n📤 Bước 5: Sinh submission.csv...")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    sample_sub = pd.read_csv(
        os.path.join(RAW_DIR, "sample_submission.csv"), parse_dates=["Date"]
    )
    test_dates = sample_sub["Date"].tolist()
    df_full = df_train_raw.copy().sort_values("Date").reset_index(drop=True)
    predictions = []
    for test_date in test_dates:
        row = _build_features_for_date(df_full, test_date, feature_names)
        if row is not None:
            pred_revenue = model.predict(row)[0]
        else:
            pred_revenue = df_full["Revenue"].tail(7).mean()
        hist_ratio = df_full["COGS"].mean() / df_full["Revenue"].mean()
        pred_cogs = pred_revenue * hist_ratio
        predictions.append(
            {
                "Date": test_date,
                "Revenue": round(pred_revenue, 2),
                "COGS": round(pred_cogs, 2),
            }
        )
        new_row = pd.DataFrame(
            [{"Date": test_date, "Revenue": pred_revenue, "COGS": pred_cogs}]
        )
        for col in df_full.columns:
            if col not in new_row.columns:
                new_row[col] = df_full[col].tail(7).mean()
        df_full = pd.concat([df_full, new_row[df_full.columns]], ignore_index=True)
    submission = pd.DataFrame(predictions)
    submission = submission.set_index("Date").reindex(sample_sub["Date"]).reset_index()
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"   ✅ Submission saved: {sub_path} ({len(submission)} rows)")
    return submission


def _build_features_for_date(df_history, target_date, feature_names):
    df = df_history.copy()
    last_rev = df["Revenue"].iloc[-1] if len(df) > 0 else 0
    features = {}
    for lag in [1, 7, 14, 30]:
        col = f"Revenue_lag_{lag}"
        if col in feature_names:
            idx = max(0, len(df) - lag)
            features[col] = df["Revenue"].iloc[idx] if idx < len(df) else last_rev
    for lag in [1, 7]:
        col = f"COGS_lag_{lag}"
        if col in feature_names:
            idx = max(0, len(df) - lag)
            features[col] = df["COGS"].iloc[idx] if idx < len(df) else 0
    for window, stat, col_name in [
        (7, "mean", "Revenue_rolling_7d_mean"),
        (7, "std", "Revenue_rolling_7d_std"),
        (30, "mean", "Revenue_rolling_30d_mean"),
    ]:
        if col_name in feature_names:
            vals = df["Revenue"].tail(window)
            features[col_name] = vals.mean() if stat == "mean" else vals.std()
    if "sessions_rolling_3d_sum" in feature_names and "sessions" in df.columns:
        features["sessions_rolling_3d_sum"] = df["sessions"].tail(3).sum()
    if "sessions_rolling_7d_mean" in feature_names and "sessions" in df.columns:
        features["sessions_rolling_7d_mean"] = df["sessions"].tail(7).mean()
    if "Gross_Margin_lag1" in feature_names:
        rev_l1 = features.get("Revenue_lag_1", last_rev)
        cogs_l1 = features.get("COGS_lag_1", 0)
        features["Gross_Margin_lag1"] = (
            (rev_l1 - cogs_l1) / rev_l1 if rev_l1 != 0 else 0
        )
    if isinstance(target_date, str):
        target_date = pd.Timestamp(target_date)
    cal_map = {
        "year": target_date.year,
        "month": target_date.month,
        "day": target_date.day,
        "dayofweek": target_date.dayofweek,
        "is_weekend": int(target_date.dayofweek >= 5),
        "quarter": target_date.quarter,
    }
    for k, v in cal_map.items():
        if k in feature_names:
            features[k] = v
    for col in feature_names:
        if col not in features:
            if col in df.columns:
                features[col] = df[col].tail(7).mean()
            else:
                features[col] = 0
    row = pd.DataFrame([{fn: features.get(fn, 0) for fn in feature_names}])
    return row


def main():
    print("=" * 60)
    print("🏆 DATATHON 2026 — FORECASTING PIPELINE")
    print("   Powered by ai-data-science-team + LightGBM")
    print("=" * 60)
    df_raw = load_and_merge_data()
    df_engineered = run_feature_engineering_agent(df_raw)
    target_col = "Revenue"
    if target_col not in df_engineered.columns:
        for c in df_engineered.columns:
            if "revenue" in c.lower():
                target_col = c
                break
    (model, feature_names, metrics) = train_lightgbm(df_engineered, target_col)
    X_for_shap = df_engineered.drop(columns=[target_col])
    generate_shap_plots(model, X_for_shap, feature_names)
    df_train_raw = pd.read_csv(os.path.join(RAW_DIR, "sales.csv"), parse_dates=["Date"])
    df_for_sub = load_and_merge_data()
    generate_submission(model, df_for_sub, feature_names)
    print("\n" + "=" * 60)
    print("📋 KẾT QUẢ CROSS-VALIDATION:")
    for m in metrics:
        print(
            f"   Fold {m['fold']}: MAE={m['MAE']:,.0f} | RMSE={m['RMSE']:,.0f} | R²={m['R2']:.4f}"
        )
    avg_mae = np.mean([m["MAE"] for m in metrics])
    avg_rmse = np.mean([m["RMSE"] for m in metrics])
    avg_r2 = np.mean([m["R2"] for m in metrics])
    print(
        f"\n   📊 TRUNG BÌNH: MAE={avg_mae:,.0f} | RMSE={avg_rmse:,.0f} | R²={avg_r2:.4f}"
    )
    print("=" * 60)
    print("✅ Pipeline hoàn tất! Kiểm tra:")
    print(f"   - reports/submissions/submission.csv")
    print(f"   - reports/figures/feature_importance.png")
    print(f"   - reports/figures/shap_summary.png")


if __name__ == "__main__":
    main()
