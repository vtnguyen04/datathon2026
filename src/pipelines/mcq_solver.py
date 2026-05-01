import polars as pl
import os

raw = "data/raw"


def load(filename):
    return pl.read_csv(os.path.join(raw, f"{filename}.csv"), try_parse_dates=True)


print("=== BẮT ĐẦU GIẢI MCQ - PHẦN 1 ===")
orders = load("orders")
orders_sorted = orders.sort(["customer_id", "order_date"])
order_counts = orders.group_by("customer_id").agg(pl.count("order_id").alias("count"))
multi_order_customers = order_counts.filter(pl.col("count") > 1)[
    "customer_id"
].to_list()
orders_multi = orders_sorted.filter(pl.col("customer_id").is_in(multi_order_customers))
orders_multi = orders_multi.with_columns(
    pl.col("order_date").diff().over("customer_id").alias("days_gap")
)
if "days_gap" in orders_multi.columns and orders_multi["days_gap"].dtype == pl.Duration:
    median_gap = orders_multi["days_gap"].drop_nulls().median()
    print(f"Q1 - Median gap: {median_gap}")
products = load("products")
q2_df = products.with_columns(
    ((pl.col("price") - pl.col("cogs")) / pl.col("price")).alias("gross_margin")
)
ans2 = (
    q2_df.group_by("segment")
    .agg(pl.col("gross_margin").mean())
    .sort("gross_margin", descending=True)
)
print(f"Q2 - Avg Gross Margin by Segment:\n{ans2}")
returns = load("returns")
q3_df = returns.join(products, on="product_id", how="inner").filter(
    pl.col("category") == "Streetwear"
)
ans3 = q3_df.group_by("return_reason").agg(pl.count()).sort("count", descending=True)
print(f"Q3 - Return reasons for Streetwear:\n{ans3}")
traffic = load("web_traffic")
ans4 = (
    traffic.group_by("traffic_source")
    .agg(pl.col("bounce_rate").mean())
    .sort("bounce_rate")
)
print(f"Q4 - Lowest Avg Bounce Rate:\n{ans4}")
order_items = load("order_items")
items_with_promo = order_items.filter(pl.col("promo_id").is_not_null())
promo_percentage = len(items_with_promo) / len(order_items) * 100
print(f"Q5 - Promo Percentage: {promo_percentage:.2f}%")
customers = load("customers")
cust_valid_age = customers.filter(pl.col("age_group").is_not_null())
q6_stats = cust_valid_age.join(orders, on="customer_id", how="left")
ans6 = (
    q6_stats.group_by("age_group")
    .agg(
        pl.count("order_id").alias("total_orders"),
        pl.col("customer_id").n_unique().alias("unique_customers"),
    )
    .with_columns(
        (pl.col("total_orders") / pl.col("unique_customers")).alias(
            "avg_orders_per_cust"
        )
    )
    .sort("avg_orders_per_cust", descending=True)
)
print(f"Q6 - Average Orders per Customer by Age Group:\n{ans6}")
geo = load("geography")
q7_orders = orders.filter(
    (pl.col("order_date") >= pl.date(2012, 7, 4))
    & (pl.col("order_date") <= pl.date(2022, 12, 31))
)
geo_orders = q7_orders.join(geo, on="zip", how="inner").join(
    order_items, on="order_id", how="inner"
)
geo_orders = geo_orders.with_columns(
    (pl.col("quantity") * pl.col("unit_price")).alias("item_revenue")
)
ans7 = (
    geo_orders.group_by("region")
    .agg(pl.sum("item_revenue"))
    .sort("item_revenue", descending=True)
)
print(f"Q7 - Total Revenue by Region:\n{ans7}")
cancelled = orders.filter(pl.col("order_status") == "cancelled")
ans8 = (
    cancelled.group_by("payment_method").agg(pl.count()).sort("count", descending=True)
)
print(f"Q8 - Payment methods for cancelled orders:\n{ans8}")
items_with_products = order_items.join(products, on="product_id", how="inner")
items_by_size = items_with_products.group_by("size").agg(
    pl.count("order_id").alias("total_sold")
)
returns_with_products = returns.join(products, on="product_id", how="inner")
returns_by_size = returns_with_products.group_by("size").agg(
    pl.count("return_id").alias("total_returns")
)
ans9 = items_by_size.join(returns_by_size, on="size", how="left").fill_null(0)
ans9 = ans9.with_columns(
    (pl.col("total_returns") / pl.col("total_sold")).alias("return_rate")
).sort("return_rate", descending=True)
print(f"Q9 - Return rate by Size:\n{ans9}")
payments = load("payments")
ans10 = (
    payments.group_by("installments")
    .agg(pl.col("payment_value").mean().alias("avg_payment"))
    .sort("avg_payment", descending=True)
)
print(f"Q10 - Average Payment Value by Installments:\n{ans10}")
print("=== HOÀN TẤT ===")
