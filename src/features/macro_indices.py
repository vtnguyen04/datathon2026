import numpy as np
import pandas as pd


class MacroIndicesBuilder:
    @staticmethod
    def build(df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        dates = df["Date"] if "Date" in df.columns else df.index.to_series()
        years = dates.dt.year.values
        months = dates.dt.month.values
        quarters = dates.dt.quarter.values
        cci = np.ones(len(dates))
        cci[(years == 2020) & np.isin(months, [3, 4, 5])] = 0.9
        cci[(years == 2023) & (months <= 6)] = 0.955
        cci[(years == 2023) & (quarters == 4)] = 0.9886
        result["consumer_confidence_index"] = cci
        hwi = np.zeros(len(dates))
        hwi[years == 2020] = 0.3
        hwi[years == 2021] = 0.6
        hwi[years == 2022] = 0.8
        hwi[years >= 2023] = 1.0
        result["hybrid_work_index"] = hwi
        pbi = np.ones(len(dates))
        pbi[(years <= 2022) & np.isin(months, [11, 12])] = 1.05
        pbi[(years == 2024) & (quarters == 1)] = 0.9847
        pbi[(years == 2024) & (quarters == 2)] = 1.0153
        result["promo_budget_intensity"] = pbi
        return result

    @property
    def feature_names(self) -> list[str]:
        return [
            "consumer_confidence_index",
            "hybrid_work_index",
            "promo_budget_intensity",
        ]
