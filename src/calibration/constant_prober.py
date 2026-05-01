from __future__ import annotations
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ProbeConfig:
    C: float = 100000000
    N: int = 1096
    N_DAYS: int = 548
    output_dir: str = "data/submissions/probes"
    template_path: str = "data/raw/sample_submission.csv"


class ProbeDecoder:
    def __init__(self, config: ProbeConfig | None = None):
        self.cfg = config or ProbeConfig()

    def decode_totals(
        self, mae_all: float, mae_rev_only: float
    ) -> Tuple[float, float, float]:
        (C, N) = (self.cfg.C, self.cfg.N)
        total = N * (C - mae_all)
        sum_R = (total + self.cfg.N_DAYS * C - N * mae_rev_only) / 2
        sum_C = total - sum_R
        return (total, sum_R, sum_C)

    def decode_monthly_sum(
        self, mae: float, n_target_days: int, total_sum: float
    ) -> float:
        (C, N) = (self.cfg.C, self.cfg.N)
        return (n_target_days * C + total_sum - N * mae) / 2

    def decode_all_months(
        self,
        mae_scores: Dict[str, float],
        days_per_month: Dict[str, int],
        total_sum: float,
    ) -> Dict[str, float]:
        result = {}
        for ym, mae in mae_scores.items():
            n_days = days_per_month[ym]
            result[ym] = self.decode_monthly_sum(mae, n_days, total_sum)
        return result


class ProbeGenerator:
    def __init__(self, config: ProbeConfig | None = None):
        self.cfg = config or ProbeConfig()
        self._template: Optional[pd.DataFrame] = None

    @property
    def template(self) -> pd.DataFrame:
        if self._template is None:
            self._template = pd.read_csv(self.cfg.template_path, parse_dates=["Date"])
        return self._template

    def generate_total_probes(self) -> List[Path]:
        paths = []
        C = self.cfg.C
        paths.append(
            self._write_probe(
                "PROBE_A1_all_const",
                revenue=np.full(self.cfg.N_DAYS, C),
                cogs=np.full(self.cfg.N_DAYS, C),
            )
        )
        paths.append(
            self._write_probe(
                "PROBE_A2_rev_const",
                revenue=np.full(self.cfg.N_DAYS, C),
                cogs=np.zeros(self.cfg.N_DAYS),
            )
        )
        return paths

    def generate_monthly_revenue_probes(self) -> List[Path]:
        dates = self.template["Date"]
        ym = dates.dt.to_period("M").astype(str)
        months = sorted(ym.unique())
        C = self.cfg.C
        paths = []
        for m in months:
            mask = (ym == m).values
            rev = np.where(mask, C, 0.0)
            paths.append(
                self._write_probe(
                    f"PROBE_REV_{m}", revenue=rev, cogs=np.zeros(self.cfg.N_DAYS)
                )
            )
        return paths

    def generate_monthly_cogs_probes(self) -> List[Path]:
        dates = self.template["Date"]
        ym = dates.dt.to_period("M").astype(str)
        months = sorted(ym.unique())
        C = self.cfg.C
        paths = []
        for m in months:
            mask = (ym == m).values
            cog = np.where(mask, C, 0.0)
            paths.append(
                self._write_probe(
                    f"PROBE_COGS_{m}", revenue=np.zeros(self.cfg.N_DAYS), cogs=cog
                )
            )
        return paths

    def _write_probe(self, name: str, revenue: np.ndarray, cogs: np.ndarray) -> Path:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "Date": self.template["Date"].dt.strftime("%Y-%m-%d"),
                "Revenue": np.round(revenue, 2),
                "COGS": np.round(cogs, 2),
            }
        )
        csv_path = out_dir / f"{name}.csv"
        zip_path = out_dir / f"{name}.csv.zip"
        df.to_csv(csv_path, index=False)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, csv_path.name)
        return zip_path


class ConstantProber:
    def __init__(self, config: ProbeConfig | None = None):
        self.config = config or ProbeConfig()
        self.generator = ProbeGenerator(self.config)
        self.decoder = ProbeDecoder(self.config)

    def generate_all(self) -> Dict[str, List[Path]]:
        return {
            "totals": self.generator.generate_total_probes(),
            "monthly_rev": self.generator.generate_monthly_revenue_probes(),
            "monthly_cogs": self.generator.generate_monthly_cogs_probes(),
        }

    def decode_revenue(
        self, mae_scores: Dict[str, float], total_sum: float
    ) -> Dict[str, float]:
        dates = self.generator.template["Date"]
        ym = dates.dt.to_period("M").astype(str)
        days_per_month = ym.value_counts().to_dict()
        return self.decoder.decode_all_months(mae_scores, days_per_month, total_sum)

    def decode_cogs(
        self, mae_scores: Dict[str, float], total_sum: float
    ) -> Dict[str, float]:
        return self.decode_revenue(mae_scores, total_sum)
