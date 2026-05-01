import pandas as pd
from src.config import RAW_DIR


class DataLoader:
    @staticmethod
    def sales(usecols=None) -> pd.DataFrame:
        return pd.read_csv(RAW_DIR / "sales.csv", parse_dates=["Date"], usecols=usecols)

    @staticmethod
    def customers(usecols=None) -> pd.DataFrame:
        df = pd.read_csv(RAW_DIR / "customers.csv", usecols=usecols)
        if usecols is None or "signup_date" in usecols:
            df["signup_date"] = pd.to_datetime(df["signup_date"])
        return df

    @staticmethod
    def orders(usecols=None) -> pd.DataFrame:
        df = pd.read_csv(RAW_DIR / "orders.csv", usecols=usecols)
        if usecols is None or "order_date" in usecols:
            df["order_date"] = pd.to_datetime(df["order_date"])
        return df

    @staticmethod
    def order_items(usecols=None) -> pd.DataFrame:
        return pd.read_csv(RAW_DIR / "order_items.csv", usecols=usecols)

    @staticmethod
    def products(usecols=None) -> pd.DataFrame:
        return pd.read_csv(RAW_DIR / "products.csv", usecols=usecols)

    @staticmethod
    def promotions(usecols=None) -> pd.DataFrame:
        return pd.read_csv(RAW_DIR / "promotions.csv", usecols=usecols)

    @staticmethod
    def geography(usecols=None) -> pd.DataFrame:
        return pd.read_csv(RAW_DIR / "geography.csv", usecols=usecols)

    @staticmethod
    def web_traffic(usecols=None) -> pd.DataFrame:
        df = pd.read_csv(RAW_DIR / "web_traffic.csv", usecols=usecols)
        if usecols is None or "date" in usecols:
            df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def shipments(usecols=None) -> pd.DataFrame:
        df = pd.read_csv(RAW_DIR / "shipments.csv", usecols=usecols)
        if usecols is None or "ship_date" in usecols:
            df["ship_date"] = pd.to_datetime(df["ship_date"])
        if usecols is None or "delivery_date" in usecols:
            df["delivery_date"] = pd.to_datetime(df["delivery_date"])
        return df

    @staticmethod
    def returns(usecols=None) -> pd.DataFrame:
        df = pd.read_csv(RAW_DIR / "returns.csv", usecols=usecols)
        if usecols is None or "return_date" in usecols:
            df["return_date"] = pd.to_datetime(df["return_date"])
        return df

    @staticmethod
    def reviews(usecols=None) -> pd.DataFrame:
        return pd.read_csv(RAW_DIR / "reviews.csv", usecols=usecols)

    @staticmethod
    def payments(usecols=None) -> pd.DataFrame:
        return pd.read_csv(RAW_DIR / "payments.csv", usecols=usecols)

    @staticmethod
    def inventory() -> pd.DataFrame:
        return pd.read_csv(RAW_DIR / "inventory.csv")

    @staticmethod
    def sample_submission() -> pd.DataFrame:
        return pd.read_csv(RAW_DIR / "sample_submission.csv", parse_dates=["Date"])

    @staticmethod
    def sales_with_covariates() -> pd.DataFrame:
        sales = DataLoader.sales()
        traffic = DataLoader.web_traffic()
        orders = DataLoader.orders()
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
        return df.sort_values("Date").reset_index(drop=True)
