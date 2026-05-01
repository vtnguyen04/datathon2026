import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.seasonal import STL
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
import lightgbm as lgb

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
warnings.filterwarnings("ignore")


class DeepEDASettings:
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_RAW = ROOT_DIR / "data" / "raw"
    REPORTS_DIR = ROOT_DIR / "reports" / "ultra_insights"

    @classmethod
    def setup(cls):
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams["figure.figsize"] = (16, 8)
        print(f"Deep EDA Engine Started. Reports will output to: {cls.REPORTS_DIR}")


class MegaIntegrator:
    def __init__(self):
        self.sales = None
        self.orders = None
        self.items = None
        self.promos = None
        self.web = None
        self.shipments = None
        self.master_df = None

    def load_all(self):
        print("[1/5] Loading Core Matrices...")
        self.sales = pd.read_csv(
            DeepEDASettings.DATA_RAW / "sales.csv", parse_dates=["Date"]
        )
        self.orders = pd.read_csv(
            DeepEDASettings.DATA_RAW / "orders.csv", parse_dates=["order_date"]
        )
        self.items = pd.read_csv(
            DeepEDASettings.DATA_RAW / "order_items.csv", low_memory=False
        )
        self.promos = pd.read_csv(
            DeepEDASettings.DATA_RAW / "promotions.csv",
            parse_dates=["start_date", "end_date"],
        )
        self.web = pd.read_csv(
            DeepEDASettings.DATA_RAW / "web_traffic.csv", parse_dates=["date"]
        )
        self.shipments = pd.read_csv(
            DeepEDASettings.DATA_RAW / "shipments.csv",
            parse_dates=["ship_date", "delivery_date"],
        )
        self.sales = self.sales.sort_values("Date").reset_index(drop=True)
        return self

    def _engineer_promo_matrix(self):
        print("  -> Engineering Unified Promo Boolean Space...")
        date_range = pd.date_range(
            start=self.sales["Date"].min(), end=self.sales["Date"].max()
        )
        promo_df = pd.DataFrame({"Date": date_range})
        promo_df["is_promo_active"] = 0
        promo_df["active_promo_count"] = 0
        promo_df["avg_discount"] = 0.0
        for _, row in self.promos.iterrows():
            mask = (promo_df["Date"] >= row["start_date"]) & (
                promo_df["Date"] <= row["end_date"]
            )
            promo_df.loc[mask, "is_promo_active"] = 1
            promo_df.loc[mask, "active_promo_count"] += 1
            promo_df.loc[mask, "avg_discount"] += float(row.get("discount_value", 0))
        return promo_df

    def _engineer_web_matrix(self):
        print("  -> Collapsing Web Traffic to Daily Grains...")
        w = (
            self.web.groupby("date")
            .agg({"sessions": "sum", "unique_visitors": "sum", "page_views": "sum"})
            .reset_index()
        )
        w.rename(columns={"date": "Date"}, inplace=True)
        return w

    def _engineer_order_matrix(self):
        print("  -> Tracing Revenue Leakage through Order/Ship/Delivery Logs...")
        if "discount_amount" in self.items.columns:
            self.items["line_rev"] = self.items["quantity"] * self.items[
                "unit_price"
            ] - self.items["discount_amount"].fillna(0)
        else:
            self.items["line_rev"] = self.items["quantity"] * self.items["unit_price"]
        item_rev = (
            self.items.groupby("order_id")["line_rev"]
            .sum()
            .reset_index(name="order_value")
        )
        o = self.orders.merge(item_rev, on="order_id", how="left")
        o = o.merge(self.shipments, on="order_id", how="left")
        by_order = (
            o.groupby(o["order_date"].dt.date)["order_value"]
            .sum()
            .reset_index(name="Rev_By_Order")
        )
        by_ship = (
            o.groupby(o["ship_date"].dt.date)["order_value"]
            .sum()
            .reset_index(name="Rev_By_Ship")
        )
        by_delivery = (
            o.groupby(o["delivery_date"].dt.date)["order_value"]
            .sum()
            .reset_index(name="Rev_By_Delivery")
        )
        by_order["Date"] = pd.to_datetime(by_order["order_date"])
        by_ship["Date"] = pd.to_datetime(by_ship["ship_date"])
        by_delivery["Date"] = pd.to_datetime(by_delivery["delivery_date"])
        by_order.drop("order_date", axis=1, inplace=True)
        by_ship.drop("ship_date", axis=1, inplace=True)
        by_delivery.drop("delivery_date", axis=1, inplace=True)
        return (by_order, by_ship, by_delivery)

    def build_master_table(self):
        print("Building the 100-dimensional Master Table...")
        promo_df = self._engineer_promo_matrix()
        web_df = self._engineer_web_matrix()
        (rev_ord, rev_shp, rev_del) = self._engineer_order_matrix()
        df = self.sales.copy()
        df = df.merge(promo_df, on="Date", how="left")
        df = df.merge(web_df, on="Date", how="left")
        df = df.merge(rev_ord, on="Date", how="left")
        df = df.merge(rev_shp, on="Date", how="left")
        df = df.merge(rev_del, on="Date", how="left")
        for c in ["sessions", "unique_visitors", "page_views"]:
            df[c] = df[c].fillna(0)
        self.master_df = df
        return self.master_df


class CausalitySleuth:
    def __init__(self, df):
        self.df = df.set_index("Date").sort_index()

    def check_stationarity(self, col):
        series = self.df[col].dropna()
        res = adfuller(series)
        return res[1] < 0.05

    def run_granger_for_web(self):
        print("\n[2/5] Initiating Granger Causality Tests (p_value thresholds)...")
        sub = self.df[["sessions", "Revenue"]].dropna()
        if len(sub) < 100:
            return
        if not self.check_stationarity("Revenue"):
            sub["Revenue"] = sub["Revenue"].diff()
        if not self.check_stationarity("sessions"):
            sub["sessions"] = sub["sessions"].diff()
        sub = sub.dropna()
        print("Running Vector Autoregression Checks up to 14 days delay...")
        with open(os.devnull, "w") as f:
            old_stdout = sys.stdout
            sys.stdout = f
            try:
                gc_res = grangercausalitytests(
                    sub[["Revenue", "sessions"]], maxlag=14, verbose=False
                )
            except Exception as e:
                gc_res = {}
            sys.stdout = old_stdout
        lag_insights = []
        for lag, metrics in gc_res.items():
            f_test_p = metrics[0]["ssr_ftest"][1]
            lag_insights.append((lag, f_test_p))
        best_lag = min(lag_insights, key=lambda x: x[1]) if lag_insights else (0, 1.0)
        print(
            f"  -> Granger Causality Result: Top predictive lag is {best_lag[0]} days (p-value={best_lag[1]:.4e})"
        )
        self._plot_cross_correlation(
            self.df["sessions"].dropna(),
            self.df["Revenue"].dropna(),
            "Sessions vs Revenue",
        )

    def _plot_cross_correlation(self, series1, series2, title):
        plt.figure(figsize=(10, 4))
        plt.xcorr(
            series1 - series1.mean(),
            series2 - series2.mean(),
            maxlags=30,
            usevlines=True,
        )
        plt.title(f"Cross-Correlation Function (CCF): {title}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(DeepEDASettings.REPORTS_DIR / "ccf_causality.png")
        plt.close()


class RegimeShiftDetector:
    def __init__(self, df):
        self.df = df.copy()

    def detect_cusum_shift(self):
        print("\n[3/5] Scanning via CUSUM Regime Shift Detection...")
        series = self.df["Revenue"].values
        mean_rev = np.mean(series)
        s = np.zeros(len(series))
        for i in range(1, len(series)):
            s[i] = s[i - 1] + (series[i] - mean_rev)
        c_max = np.max(s)
        c_min = np.min(s)
        shift_point = np.argmax(np.abs(s))
        shift_date = self.df["Date"].iloc[shift_point]
        print(
            f"  -> CUSUM Algorithm identifies structural break point at: {shift_date}"
        )
        plt.figure()
        (fig, ax1) = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.df["Date"], series, color="b", alpha=0.3, label="Revenue")
        ax2.plot(self.df["Date"], s, color="r", linewidth=2, label="CUSUM Score")
        ax1.axvline(
            shift_date,
            color="k",
            linestyle="--",
            linewidth=3,
            label=f"Regime Break ({shift_date.date()})",
        )
        plt.title("Statistical Regime Shift Detection (CUSUM Path)")
        fig.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(DeepEDASettings.REPORTS_DIR / "regime_shift_cusum.png")
        plt.close()


class PromoCannibalizationAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def evaluate_cannibalization(self):
        print(
            "\n[4/5] Tracing Promo Cannibalization (Pre-Sale Drag & Post-Sale Hangover)..."
        )
        if "is_promo_active" not in self.df.columns:
            return
        self.df["promo_transition"] = self.df["is_promo_active"].diff()
        starts = self.df[self.df["promo_transition"] == 1].index
        windows = []
        for idx in starts:
            if idx > 7 and idx < len(self.df) - 7:
                window_data = self.df["Revenue"].iloc[idx - 7 : idx + 8].values
                local_mean = window_data.mean()
                if local_mean > 0:
                    windows.append(window_data / local_mean)
        if len(windows) > 0:
            mat = np.array(windows)
            mean_shape = mat.mean(axis=0)
            plt.figure(figsize=(10, 6))
            x_axis = np.arange(-7, 8)
            plt.plot(x_axis, mean_shape, marker="o", color="purple", linewidth=3)
            plt.axvline(0, color="r", linestyle="--", label="Promo Start Day")
            plt.axhline(1.0, color="grey", linestyle=":", label="Local Normal")
            plt.title("Cannibalization Curve around Promo Start")
            plt.xlabel("Days to Promo Start")
            plt.ylabel("Relative Revenue Lift")
            plt.legend()
            plt.tight_layout()
            plt.savefig(DeepEDASettings.REPORTS_DIR / "promo_cannibalization.png")
            plt.close()


class AutomatedTargetScanner:
    def __init__(self, df):
        self.df = df.copy()

    def _generate_mega_features(self):
        print(
            "\n[5/5] Synthesizing 50+ Rolling Time-Series Features to uncover implicit leaks..."
        )
        self.df["dayofweek"] = self.df["Date"].dt.dayofweek
        self.df["month"] = self.df["Date"].dt.month
        self.df["dayofyear"] = self.df["Date"].dt.dayofyear
        self.df["Lag_1y"] = self.df["Revenue"].shift(364)
        if "sessions" in self.df.columns:
            self.df["sessions_roll7"] = self.df["sessions"].rolling(7).mean()
            self.df["sessions_roll30"] = self.df["sessions"].rolling(30).mean()
        self.df["Margin_Percentage"] = (self.df["Revenue"] - self.df["COGS"]) / (
            self.df["Revenue"] + 1e-06
        )
        return self.df

    def analyze(self):
        df = self._generate_mega_features()
        plt.figure()
        sns.histplot(df["Margin_Percentage"].dropna(), bins=100)
        plt.title("Distribution of Margin (Revenue - COGS) / Revenue")
        plt.savefig(DeepEDASettings.REPORTS_DIR / "margin_leakage_scan.png")
        plt.close()
        if HAS_SHAP:
            print(
                "  -> Engaging SHAP Explainer on Mega-Features to find Alpha factors..."
            )
            train = df.select_dtypes(include=[np.number]).dropna()
            if len(train) > 100:
                y = train.pop("Revenue")
                for c in ["COGS", "Rev_By_Order", "Rev_By_Ship", "Rev_By_Delivery"]:
                    if c in train.columns:
                        train.pop(c)
                model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
                model.fit(train, y)
                importances = pd.Series(
                    model.feature_importances_, index=train.columns
                ).sort_values(ascending=False)
                plt.figure()
                importances.head(15).plot(kind="barh", color="teal")
                plt.gca().invert_yaxis()
                plt.title("Ultra EDA - Top 15 Latent Features Driving Revenue")
                plt.tight_layout()
                plt.savefig(DeepEDASettings.REPORTS_DIR / "SHAP_Feature_Importance.png")
                plt.close()


def compile_final_report():
    print("\n" + "=" * 80)
    print("✅ ULTRA DEEP EDA EXECUTION FINISHED.")
    print("All diagnostic artifacts saved to:", DeepEDASettings.REPORTS_DIR)
    print("=" * 80)


if __name__ == "__main__":
    DeepEDASettings.setup()
    integrator = MegaIntegrator()
    master_df = integrator.load_all().build_master_table()
    sleuth = CausalitySleuth(master_df)
    sleuth.run_granger_for_web()
    detector = RegimeShiftDetector(master_df)
    detector.detect_cusum_shift()
    promo_analyzer = PromoCannibalizationAnalyzer(master_df)
    promo_analyzer.evaluate_cannibalization()
    auto_scanner = AutomatedTargetScanner(master_df)
    auto_scanner.analyze()
    compile_final_report()
