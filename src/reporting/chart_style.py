import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PALETTE, PALETTE_LIST


def apply_theme():
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.facecolor": "#FAFAFA",
            "axes.facecolor": "#FAFAFA",
            "axes.edgecolor": "#CCCCCC",
            "grid.color": "#E6E6E6",
            "grid.alpha": 0.6,
            "font.family": "sans-serif",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 250,
            "savefig.bbox": "tight",
        }
    )


def get_colors(n: int = 8) -> list:
    return [PALETTE_LIST[i % len(PALETTE_LIST)] for i in range(n)]


def annotate_prescriptive(ax, text: str, xy, xytext=None, color=None):
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext or (xy[0], xy[1] * 1.15),
        fontsize=9,
        fontweight="bold",
        color=color or PALETTE["secondary"],
        arrowprops=dict(arrowstyle="->", color=color or PALETTE["secondary"], lw=1.5),
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=color or PALETTE["secondary"],
            alpha=0.9,
        ),
    )
