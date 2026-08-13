from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_popularity_vs_mcv(table: pl.DataFrame, path: Path, n_annotate: int) -> None:
    """Figure 1 prototype: popularity vs marginal catalog value."""
    x = table["n_ratings"].to_numpy()
    y = table["mcv"].to_numpy()
    titles = table["title"].to_list()
    quadrants = table["quadrant"].to_list()

    colors = {
        "high popularity / high MCV": "#2a6f97",
        "high popularity / low MCV": "#c44536",
        "low popularity / high MCV": "#2d6a4f",
        "low popularity / low MCV": "#6c757d",
    }
    point_colors = [colors.get(q, "#333333") for q in quadrants]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(x, y, c=point_colors, s=18, alpha=0.75, linewidths=0)
    ax.set_xscale("log")
    ax.set_xlabel("MovieLens rating count (popularity)")
    ax.set_ylabel("Marginal catalog value  MCVᵢ(S)")
    ax.set_title("Popularity is not portfolio value")

    pop_med = float(np.median(x))
    mcv_med = float(np.median(y))
    ax.axvline(pop_med, color="#bbbbbb", linewidth=1, linestyle="--")
    ax.axhline(mcv_med, color="#bbbbbb", linewidth=1, linestyle="--")

    labels = _annotation_indices(x, y, n_annotate)
    for i in labels:
        ax.annotate(
            titles[i],
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            color="#222222",
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=c,
            markersize=7,
            label=label,
        )
        for label, c in colors.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _annotation_indices(x: np.ndarray, y: np.ndarray, n: int) -> list[int]:
    if n <= 0 or len(x) == 0:
        return []
    logx = np.log10(np.clip(x, 1, None))
    x_z = (logx - logx.mean()) / (logx.std() + 1e-8)
    y_z = (y - y.mean()) / (y.std() + 1e-8)
    # Prefer quadrant extremes: high-pop/low-MCV and low-pop/high-MCV.
    score_redundant = x_z - y_z
    score_niche = y_z - x_z
    pick = []
    for scores in (score_redundant, score_niche):
        for idx in np.argsort(scores)[::-1]:
            i = int(idx)
            if i not in pick:
                pick.append(i)
            if len(pick) >= n:
                return pick
    return pick[:n]
