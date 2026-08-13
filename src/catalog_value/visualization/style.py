"""Shared matplotlib style for research figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

from catalog_value.paths import project_root

INK = "#1c1917"
PAPER = "#f7f4ee"
MUTED = "#78716c"
GRID = "#e7e5e4"
BLUE = "#1d4e89"
RED = "#b42318"
TEAL = "#0f766e"
GOLD = "#b45309"
SLATE = "#57534e"

QUADRANT_COLORS = {
    "high popularity / high MCV": BLUE,
    "high popularity / low MCV": RED,
    "low popularity / high MCV": TEAL,
    "low popularity / low MCV": SLATE,
}

SERVICE_COLORS = {
    "Netflix": "#e50914",
    "Disney+": "#113ccf",
    "Prime Video": "#00a8e1",
    "Max": "#7b2cbf",
    "Hulu": "#1ce783",
}


def apply_style() -> None:
    rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.titlecolor": INK,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def figures_dir(phase: str) -> Path:
    path = project_root() / "docs" / "figures" / phase
    path.mkdir(parents=True, exist_ok=True)
    return path
