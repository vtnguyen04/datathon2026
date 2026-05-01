import pandera as pa
from pandera import Column, DataFrameSchema
from src.core.exceptions import DataSchemaError

sales_schema = DataFrameSchema(
    {
        "Date": Column(pa.DateTime, nullable=False),
        "Revenue": Column(pa.Float, nullable=False, checks=pa.Check.ge(0)),
        "COGS": Column(pa.Float, nullable=False, checks=pa.Check.ge(0)),
    }
)


def validate_sales_data(df):
    try:
        validated_df = sales_schema.validate(df)
        return validated_df
    except pa.errors.SchemaError as exc:
        raise DataSchemaError(f"Sales data validation failed: {exc}")
