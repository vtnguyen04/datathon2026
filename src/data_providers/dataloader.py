import polars as pl
from pathlib import Path
from pydantic import BaseModel, ValidationError


class DatathonLoader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load_table(self, table_name: str) -> pl.DataFrame:
        file_path = self.data_dir / f"{table_name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(
                f"[DataLoader] Không tìm thấy file gốc: {file_path}"
            )
        df = pl.read_csv(file_path, try_parse_dates=True)
        return df

    def get_sales_train(self) -> pl.DataFrame:
        return self.load_table("sales")

    def get_transactions_bundle(self) -> dict:
        return {
            "orders": self.load_table("orders"),
            "order_items": self.load_table("order_items"),
            "returns": self.load_table("returns"),
            "promotions": self.load_table("promotions"),
        }
