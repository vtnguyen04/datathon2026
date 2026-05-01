import pandas as pd
import numpy as np
from src.config import RAW_DIR


class CohortDataLoader:
    def __init__(self):
        pass

    def build_cohort_data(self):
        orders = pd.read_csv(
            RAW_DIR / "orders.csv",
            usecols=["order_id", "order_date", "customer_id", "order_status"],
        )
        order_items = pd.read_csv(
            RAW_DIR / "order_items.csv",
            usecols=["order_id", "quantity", "unit_price", "discount_amount"],
        )
        order_items["revenue"] = (
            order_items["quantity"] * order_items["unit_price"]
            - order_items["discount_amount"]
        )
        order_value = order_items.groupby("order_id")["revenue"].sum().reset_index()
        orders = orders.merge(order_value, on="order_id", how="left")
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        first_purchases = (
            orders.groupby("customer_id")["order_date"].min().reset_index()
        )
        first_purchases.rename(
            columns={"order_date": "first_purchase_date"}, inplace=True
        )
        orders = orders.merge(first_purchases, on="customer_id")
        orders["is_new"] = orders["order_date"] == orders["first_purchase_date"]
        daily = (
            orders.groupby(["order_date", "is_new"])
            .agg(users=("customer_id", "nunique"), revenue=("revenue", "sum"))
            .reset_index()
        )
        daily_pivot = daily.pivot(
            index="order_date", columns="is_new", values=["users", "revenue"]
        ).fillna(0)
        daily_pivot.columns = [
            "returning_users",
            "new_users",
            "returning_revenue",
            "new_revenue",
        ]
        daily_pivot = daily_pivot.reset_index().rename(columns={"order_date": "Date"})
        all_dates = pd.date_range(
            start=daily_pivot["Date"].min(), end=daily_pivot["Date"].max(), freq="D"
        )
        daily_pivot = (
            daily_pivot.set_index("Date").reindex(all_dates).fillna(0).reset_index()
        )
        daily_pivot.rename(columns={"index": "Date"}, inplace=True)
        return daily_pivot
