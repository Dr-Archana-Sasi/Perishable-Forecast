"""Shared paths, constants and plotting style for the whole project."""
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

for _d in (DATA_RAW, DATA_PROCESSED, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
HF_DATASET = "Dingdong-Inc/FreshRetailNet-50K"

# Hours the store is open and stock status is meaningful (06:00-22:00)
OPEN_HOURS = list(range(6, 22))

# Colour-blind-safe categorical palette (fixed order, never cycled)
C_INSTOCK = "#2a78d6"   # blue
C_STOCKOUT = "#eb6834"  # orange
C_THIRD = "#1baf7a"     # aqua
INK = "#0b0b0b"
INK_2 = "#52514e"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def set_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "lines.linewidth": 2,
        "legend.frameon": False,
    })
