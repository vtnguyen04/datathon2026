import pandas as pd
import numpy as np
from src.config import RAW_DIR


class HierarchicalDataLoader:
    def __init__(self):
        pass

    def load_category_data(self) -> pd.DataFrame:
        orders = pd.read_csv(RAW_DIR / "orders.csv", usecols=["order_id", "order_date"])
        order_items = pd.read_csv(
            RAW_DIR / "order_items.csv",
            usecols=[
                "order_id",
                "product_id",
                "quantity",
                "unit_price",
                "discount_amount",
            ],
        )
        products = pd.read_csv(
            RAW_DIR / "products.csv", usecols=["product_id", "category", "cogs"]
        )
        order_items["revenue"] = (
            order_items["quantity"] * order_items["unit_price"]
            - order_items["discount_amount"]
        )
        items = order_items.merge(products, on="product_id", how="left")
        items["cogs_total"] = items["quantity"] * items["cogs"]
        items = items.merge(orders, on="order_id", how="left")
        items["order_date"] = pd.to_datetime(items["order_date"])
        daily_category = (
            items.groupby(["order_date", "category"])[["revenue", "cogs_total"]]
            .sum()
            .reset_index()
        )
        daily_category.rename(
            columns={"order_date": "Date", "revenue": "Revenue", "cogs_total": "COGS"},
            inplace=True,
        )
        all_dates = pd.date_range(
            start=daily_category["Date"].min(),
            end=daily_category["Date"].max(),
            freq="D",
        )
        categories = daily_category["category"].unique()
        idx = pd.MultiIndex.from_product(
            [all_dates, categories], names=["Date", "category"]
        )
        daily_category = (
            daily_category.set_index(["Date", "category"])
            .reindex(idx)
            .fillna(0)
            .reset_index()
        )
        daily_category["year"] = daily_category["Date"].dt.year
        daily_category["month"] = daily_category["Date"].dt.month
        daily_category["day"] = daily_category["Date"].dt.day
        daily_category["dow"] = daily_category["Date"].dt.dayofweek
        daily_category["doy"] = daily_category["Date"].dt.dayofyear
        daily_category["woy"] = daily_category["Date"].dt.isocalendar().week.astype(int)
        return daily_category
