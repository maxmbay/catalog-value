"""Phase D figures: ablations, PACV, greedy portfolios."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from catalog_value.visualization.atlas import short_title
from catalog_value.visualization.style import BLUE, GOLD, INK, MUTED, TEAL, apply_style, savefig


def plot_growth(popular: list[float], greedy: list[float], path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    x = list(range(1, len(popular) + 1))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, popular, marker="o", color=MUTED, lw=2, label="Add by popularity")
    ax.plot(x, greedy, marker="o", color=TEAL, lw=2, label="Add by MCV (greedy)")
    ax.fill_between(x, popular, greedy, color=TEAL, alpha=0.12)
    ax.set_xlabel("Catalog size")
    ax.set_ylabel("Audience coverage  V(S)")
    ax.set_title("Greedy coverage pulls away from a popularity stack")
    ax.legend(loc="lower right")
    return savefig(fig, path)


def plot_pacv(table: pl.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.scatter(table["n_ratings"].to_numpy(), table["pacv"].to_numpy(), s=28, c=BLUE, alpha=0.85, linewidths=0)
    ax.set_xscale("log")
    ax.set_xlabel("MovieLens rating count")
    ax.set_ylabel("PACV   φᵢ = Eₛ[MCVᵢ(S)]")
    ax.set_title("Portfolio-adjusted value is not a popularity list")
    for row in table.sort("pacv", descending=True).head(8).iter_rows(named=True):
        ax.annotate(
            short_title(row["title"]),
            (row["n_ratings"], row["pacv"]),
            fontsize=7.5,
            color=INK,
            xytext=(5, 5),
            textcoords="offset points",
        )
    return savefig(fig, path)


def plot_ablation(neural: np.ndarray, svd: np.ndarray, content: np.ndarray, path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6))
    pairs = (
        (axes[0], svd, "SVD titles + SVD audience"),
        (axes[1], content, "Content z (cold-start encoder)"),
    )
    for ax, other, name in pairs:
        ax.scatter(neural, other, s=18, c=GOLD, alpha=0.8, linewidths=0)
        lo = min(float(neural.min()), float(other.min()))
        hi = max(float(neural.max()), float(other.max()))
        ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1)
        rho = float(pl.DataFrame({"a": neural, "b": other}).select(pl.corr("a", "b", method="spearman")).item())
        ax.set_xlabel("MCV  (taste-token neural)")
        ax.set_ylabel(f"MCV  ({name})")
        ax.set_title(f"Spearman {rho:.2f}")
    fig.suptitle("Does the backbone change which titles look valuable?", fontsize=13)
    return savefig(fig, path)


def plot_genre_mix(popular_share: dict[str, float], greedy_share: dict[str, float], path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    genres = sorted(set(popular_share) | set(greedy_share), key=lambda g: greedy_share.get(g, 0), reverse=True)[:8]
    y = np.arange(len(genres))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.barh(y + 0.18, [popular_share.get(g, 0) for g in genres], height=0.35, color=MUTED, label="popularity catalog")
    ax.barh(y - 0.18, [greedy_share.get(g, 0) for g in genres], height=0.35, color=TEAL, label="greedy MCV catalog")
    ax.set_yticks(y, genres)
    ax.set_xlabel("Share of titles")
    ax.set_title("Greedy MCV shifts the genre mix")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    return savefig(fig, path)
