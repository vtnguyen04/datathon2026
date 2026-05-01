from __future__ import annotations
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from src.config import PALETTE, FIGURES_DIR

plt.rcParams.update(
    {
        "figure.facecolor": "#0D1117",
        "axes.facecolor": "#161B22",
        "axes.edgecolor": "#30363D",
        "axes.labelcolor": "#C9D1D9",
        "text.color": "#C9D1D9",
        "xtick.color": "#8B949E",
        "ytick.color": "#8B949E",
        "grid.color": "#21262D",
        "grid.alpha": 0.6,
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 150,
    }
)


class ExperimentVisualizer:
    def __init__(self, results: list, output_dir: Optional[Path] = None):
        self.results = sorted(results, key=lambda r: r.cv_mae_total or float("inf"))
        self.output_dir = output_dir or FIGURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_cv_comparison(self, top_n: int = 30) -> Path:
        data = [r for r in self.results if r.cv_mae_total is not None][:top_n]
        if not data:
            return Path()
        names = [r.name for r in data]
        rev_mae = [r.cv_mae_revenue for r in data]
        cogs_mae = [r.cv_mae_cogs for r in data]
        (fig, ax) = plt.subplots(figsize=(14, max(6, len(data) * 0.35)))
        y = np.arange(len(names))
        ax.barh(
            y,
            rev_mae,
            height=0.4,
            label="Revenue MAE",
            color=PALETTE["primary"],
            alpha=0.9,
            align="center",
        )
        ax.barh(
            y + 0.4,
            cogs_mae,
            height=0.4,
            label="COGS MAE",
            color=PALETTE["accent"],
            alpha=0.9,
            align="center",
        )
        ax.set_yticks(y + 0.2)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("MAE")
        ax.set_title(
            "CV MAE Comparison — All Experiments",
            fontsize=14,
            fontweight="bold",
            color=PALETTE["light"],
        )
        ax.legend(loc="lower right", framealpha=0.3)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}K")
        )
        ax.grid(axis="x", alpha=0.3)
        for i, r in enumerate(data):
            ax.text(
                r.cv_mae_revenue + 5000,
                i,
                f"{r.cv_mae_total:,.0f}",
                va="center",
                fontsize=7,
                color=PALETTE["light"],
            )
        plt.tight_layout()
        path = self.output_dir / "cv_comparison.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_evolution(self) -> Path:
        data = []
        for r in self.results:
            if r.cv_mae_total is None:
                continue
            data.append(
                {
                    "name": r.name,
                    "total": r.cv_mae_total,
                    "revenue": r.cv_mae_revenue,
                    "cogs": r.cv_mae_cogs,
                }
            )
        if not data:
            return Path()
        df = pd.DataFrame(data)
        (fig, ax) = plt.subplots(figsize=(14, 6))
        x = range(len(df))
        ax.plot(
            x,
            df["total"],
            marker="o",
            color=PALETTE["primary"],
            linewidth=2,
            markersize=4,
            label="Total MAE",
            zorder=3,
        )
        ax.fill_between(x, df["total"], alpha=0.15, color=PALETTE["primary"])
        best_idx = df["total"].idxmin()
        ax.scatter(
            [best_idx],
            [df.loc[best_idx, "total"]],
            s=120,
            color=PALETTE["secondary"],
            zorder=5,
            edgecolors="white",
            linewidths=2,
        )
        ax.annotate(
            f"BEST: {df.loc[best_idx, 'name']}\n{df.loc[best_idx, 'total']:,.0f}",
            xy=(best_idx, df.loc[best_idx, "total"]),
            xytext=(best_idx + 1, df.loc[best_idx, "total"] * 1.02),
            fontsize=9,
            fontweight="bold",
            color=PALETTE["secondary"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["secondary"]),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(df["name"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("CV MAE (Total)")
        ax.set_title(
            "Experiment Evolution — MAE Over Time",
            fontsize=14,
            fontweight="bold",
            color=PALETTE["light"],
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{y / 1000:.0f}K")
        )
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.3)
        plt.tight_layout()
        path = self.output_dir / "experiment_evolution.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_parameter_sensitivity(self, param_name: str) -> Path:
        data = []
        for r in self.results:
            if r.cv_mae_total is None:
                continue
            val = getattr(r.config, param_name, None)
            if val is not None:
                data.append({"param": val, "total": r.cv_mae_total, "name": r.name})
        if not data:
            return Path()
        df = pd.DataFrame(data).sort_values("param")
        (fig, ax) = plt.subplots(figsize=(10, 6))
        ax.scatter(
            df["param"],
            df["total"],
            c=PALETTE["primary"],
            s=60,
            alpha=0.8,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.plot(
            df["param"], df["total"], color=PALETTE["primary"], alpha=0.4, linewidth=1
        )
        best_idx = df["total"].idxmin()
        ax.scatter(
            [df.loc[best_idx, "param"]],
            [df.loc[best_idx, "total"]],
            s=150,
            color=PALETTE["secondary"],
            zorder=5,
            edgecolors="white",
            linewidths=2,
        )
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel("CV MAE (Total)", fontsize=12)
        ax.set_title(
            f"Parameter Sensitivity: {param_name}",
            fontsize=14,
            fontweight="bold",
            color=PALETTE["light"],
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{y / 1000:.0f}K")
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = self.output_dir / f"sensitivity_{param_name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_summary_dashboard(self) -> Path:
        (fig, axes) = plt.subplots(2, 2, figsize=(18, 12))
        ax = axes[0, 0]
        data = [r for r in self.results if r.cv_mae_total][:15]
        if data:
            names = [r.name for r in data]
            totals = [r.cv_mae_total for r in data]
            colors = [
                PALETTE["secondary"] if i == 0 else PALETTE["primary"]
                for i in range(len(data))
            ]
            y = range(len(data))
            ax.barh(y, totals, color=colors, alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=7)
            ax.invert_yaxis()
            ax.set_title("Top 15 Experiments (CV MAE)", fontweight="bold")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}K")
            )
            for i, v in enumerate(totals):
                ax.text(v + 2000, i, f"{v:,.0f}", va="center", fontsize=7)
        ax = axes[0, 1]
        valid = [r for r in self.results if r.cv_mae_total]
        if valid:
            rev = [r.cv_mae_revenue for r in valid]
            cogs = [r.cv_mae_cogs for r in valid]
            ax.scatter(
                rev,
                cogs,
                c=PALETTE["cyan"],
                s=30,
                alpha=0.6,
                edgecolors="white",
                linewidths=0.3,
            )
            ax.set_xlabel("Revenue MAE")
            ax.set_ylabel("COGS MAE")
            ax.set_title("Revenue vs COGS MAE Trade-off", fontweight="bold")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}K")
            )
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: f"{y / 1000:.0f}K")
            )
        ax = axes[1, 0]
        levels = [r.config.level for r in self.results]
        totals = [r.cv_mae_total or 0 for r in self.results]
        if levels:
            ax.scatter(levels, totals, c=PALETTE["accent"], s=30, alpha=0.6)
            ax.set_xlabel("Level (L)")
            ax.set_ylabel("CV MAE")
            ax.set_title("Level Sensitivity", fontweight="bold")
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: f"{y / 1000:.0f}K")
            )
        ax = axes[1, 1]
        model_scores = {}
        for r in self.results:
            if r.cv_mae_total:
                mt = r.config.gbm_type
                if mt not in model_scores:
                    model_scores[mt] = []
                model_scores[mt].append(r.cv_mae_total)
        if model_scores:
            model_names = list(model_scores.keys())
            means = [np.mean(v) for v in model_scores.values()]
            mins = [np.min(v) for v in model_scores.values()]
            x = range(len(model_names))
            ax.bar(x, means, color=PALETTE["primary"], alpha=0.6, label="Mean")
            ax.bar(x, mins, color=PALETTE["secondary"], alpha=0.8, label="Best")
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            ax.set_title("Model Type Comparison", fontweight="bold")
            ax.legend(framealpha=0.3)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: f"{y / 1000:.0f}K")
            )
        fig.suptitle(
            "Experiment Dashboard",
            fontsize=16,
            fontweight="bold",
            color=PALETTE["light"],
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = self.output_dir / "experiment_dashboard.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def generate_all(self) -> list[Path]:
        paths = []
        paths.append(self.plot_cv_comparison())
        paths.append(self.plot_evolution())
        paths.append(self.plot_summary_dashboard())
        return [p for p in paths if p and p.exists()]
