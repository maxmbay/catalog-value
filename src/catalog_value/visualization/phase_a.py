"""Phase A figures: popularity is not portfolio value, and why."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from catalog_value.visualization.atlas import short_title
from catalog_value.visualization.style import (
    BLUE,
    GOLD,
    INK,
    MUTED,
    QUADRANT_COLORS,
    RED,
    TEAL,
    apply_style,
    savefig,
)

LANDMARKS = [
    "Toy Story (1995)",
    "Pulp Fiction (1994)",
    "Dark Knight, The (2008)",
    "Halloween (1978)",
    "Notebook, The (2004)",
    "Paths of Glory (1957)",
    "Super Mario Bros. (1993)",
    "Planet Earth (2006)",
    "Shawshank Redemption, The (1994)",
    "Godfather, The (1972)",
    "Silence of the Lambs, The (1991)",
    "Die Hard (1988)",
]


def plot_popularity_vs_mcv(table: pl.DataFrame, path: Path, n_annotate: int) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    x = table["n_ratings"].to_numpy()
    y = table["mcv"].to_numpy()
    titles = table["title"].to_list()
    quadrants = table["quadrant"].to_list()
    colors = [QUADRANT_COLORS.get(q, MUTED) for q in quadrants]

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    ax.scatter(x, y, c=colors, s=22, alpha=0.82, linewidths=0, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("MovieLens rating count")
    ax.set_ylabel("Marginal catalog value   MCVᵢ(S)")
    ax.set_title("Popularity is not portfolio value")

    pop_med = float(np.median(x))
    mcv_med = float(np.median(y))
    ax.axvline(pop_med, color=MUTED, linewidth=1, linestyle="--", alpha=0.7)
    ax.axhline(mcv_med, color=MUTED, linewidth=1, linestyle="--", alpha=0.7)
    ax.text(pop_med, ax.get_ylim()[1], "  median popularity", color=MUTED, va="top", fontsize=8)
    ax.text(ax.get_xlim()[0], mcv_med, "median MCV  ", color=MUTED, ha="left", va="bottom", fontsize=8)

    for i in _annotation_indices(x, y, n_annotate):
        ax.annotate(
            short_title(titles[i]),
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(7, 7),
            fontsize=8,
            color=INK,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markersize=8,
            label=label.replace(" / ", "\n"),
        )
        for label, color in QUADRANT_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, borderaxespad=0.8)
    return savefig(fig, path)


def plot_title_atlas(
    movies: pl.DataFrame,
    xy: np.ndarray,
    explained: np.ndarray,
    catalog_rows: np.ndarray,
    path: Path,
) -> Path:
    """Map of learned title embeddings, colored by overlap with a popular catalog."""
    import matplotlib.pyplot as plt

    apply_style()
    z_keep = movies["n_ratings"].to_numpy() >= 1500
    catalog = np.zeros(len(xy), dtype=bool)
    catalog[np.asarray(catalog_rows, dtype=np.int64)] = True

    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    sample = xy[z_keep]
    if sample.shape[0] > 80:
        kde = gaussian_kde(sample[:: max(1, len(sample) // 2500)].T)
        xmin, xmax = sample[:, 0].min(), sample[:, 0].max()
        ymin, ymax = sample[:, 1].min(), sample[:, 1].max()
        pad_x, pad_y = 0.08 * (xmax - xmin), 0.08 * (ymax - ymin)
        xx, yy = np.meshgrid(
            np.linspace(xmin - pad_x, xmax + pad_x, 120),
            np.linspace(ymin - pad_y, ymax + pad_y, 120),
        )
        density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contourf(xx, yy, density, levels=14, cmap="Greys", alpha=0.35, zorder=0)

    ax.scatter(xy[z_keep, 0], xy[z_keep, 1], c="#d6d3d1", s=10, alpha=0.55, linewidths=0, zorder=1)
    ax.scatter(
        xy[catalog, 0],
        xy[catalog, 1],
        facecolors="none",
        edgecolors=GOLD,
        s=28,
        linewidths=0.8,
        alpha=0.9,
        zorder=2,
        label="top-500 catalog",
    )
    _label_landmarks(ax, movies, xy)
    ax.set_xlabel(f"PC1  ({100 * explained[0]:.0f}% of embedding variance)")
    ax.set_ylabel(f"PC2  ({100 * explained[1]:.0f}%)")
    ax.set_title("A map of the learned title space")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    return savefig(fig, path)


def plot_mcv_on_atlas(
    movies: pl.DataFrame,
    xy: np.ndarray,
    mcv_table: pl.DataFrame,
    catalog_rows: np.ndarray,
    path: Path,
) -> Path:
    """The same map, now colored by marginal value given the popular catalog."""
    import matplotlib.pyplot as plt

    apply_style()
    joined = movies.join(mcv_table.select(["movie_row", "mcv", "quadrant"]), on="movie_row", how="left")
    has_mcv = joined["mcv"].is_not_null().to_numpy()
    mcv = joined["mcv"].fill_null(0.0).to_numpy()
    catalog = np.zeros(len(xy), dtype=bool)
    catalog[np.asarray(catalog_rows, dtype=np.int64)] = True

    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    ax.scatter(
        xy[~has_mcv, 0],
        xy[~has_mcv, 1],
        c="#e7e5e4",
        s=8,
        alpha=0.45,
        linewidths=0,
        zorder=1,
    )
    sc = ax.scatter(
        xy[has_mcv, 0],
        xy[has_mcv, 1],
        c=mcv[has_mcv],
        cmap="YlGnBu",
        s=18,
        alpha=0.9,
        linewidths=0,
        zorder=3,
    )
    ax.scatter(
        xy[catalog, 0],
        xy[catalog, 1],
        facecolors="none",
        edgecolors=GOLD,
        s=22,
        linewidths=0.7,
        alpha=0.75,
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("MCV given the top-500 catalog")
    _label_landmarks(ax, movies, xy, extra=["Showgirls (1995)", "Barry Lyndon (1975)"])
    ax.set_title("High-MCV titles occupy a different region than popular ones")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Same embedding map as the atlas  →")
    return savefig(fig, path)


def plot_diminishing_returns(
    sizes: list[int],
    action: list[float],
    mixed: list[float],
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(sizes, action, marker="o", color=RED, lw=2.2, label="Near-substitutes  (action stack)")
    ax.plot(sizes, mixed, marker="o", color=TEAL, lw=2.2, label="Distinct tastes")
    ax.fill_between(sizes, action, mixed, color=TEAL, alpha=0.12)
    ax.set_xlabel("Titles in the catalog")
    ax.set_ylabel("Audience coverage  V(S)")
    ax.set_title("A diverse set keeps covering; substitutes flatten")
    ax.set_xticks(sizes)
    ax.legend(loc="lower right")
    return savefig(fig, path)


def plot_neighbor_strips(
    panels: list[tuple[str, pl.DataFrame]],
    path: Path,
) -> Path:
    """Small-multiples of cosine neighbors for probe titles."""
    import matplotlib.pyplot as plt

    apply_style()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 5.4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (name, table) in zip(axes, panels, strict=True):
        titles = [short_title(t) for t in table["title"].to_list()]
        scores = table["cosine"].to_numpy()
        y = np.arange(len(titles))[::-1]
        ax.barh(y, scores, color=BLUE, height=0.62, alpha=0.9)
        ax.set_yticks(y, labels=titles, fontsize=8)
        ax.set_title(short_title(name), fontsize=11)
        ax.set_xlim(0, max(0.35, float(scores.max()) * 1.15))
        ax.tick_params(axis="x", labelsize=8)
        if ax is axes[0]:
            ax.set_xlabel("cosine in z-space")
        else:
            ax.set_xlabel("")
    fig.suptitle("Nearest titles in the learned embedding", fontsize=13, y=0.98)
    fig.subplots_adjust(wspace=0.95, left=0.06, right=0.99, top=0.86, bottom=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _label_landmarks(
    ax,
    movies: pl.DataFrame,
    xy: np.ndarray,
    extra: list[str] | None = None,
) -> None:
    names = list(LANDMARKS)
    if extra:
        names.extend(extra)
    title_to_row = {t: int(r) for t, r in zip(movies["title"].to_list(), movies["movie_row"].to_list())}
    offsets = [
        (8, 10),
        (-70, 12),
        (8, -14),
        (8, 8),
        (-80, -12),
        (8, 10),
        (8, -16),
        (-70, 8),
        (8, 12),
        (-90, 8),
        (8, -10),
        (8, 8),
        (-70, 14),
        (10, -12),
    ]
    for i, name in enumerate(names):
        row = title_to_row.get(name)
        if row is None:
            continue
        x, y = xy[row]
        dx, dy = offsets[i % len(offsets)]
        ax.scatter([x], [y], c=INK, s=18, zorder=5)
        ax.annotate(
            short_title(name),
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.5,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6},
            zorder=6,
        )


def _annotation_indices(x: np.ndarray, y: np.ndarray, n: int) -> list[int]:
    if n <= 0 or len(x) == 0:
        return []
    logx = np.log10(np.clip(x, 1, None))
    x_z = (logx - logx.mean()) / (logx.std() + 1e-8)
    y_z = (y - y.mean()) / (y.std() + 1e-8)
    pick: list[int] = []
    for scores in (x_z - y_z, y_z - x_z):
        for idx in np.argsort(scores)[::-1]:
            i = int(idx)
            if i not in pick:
                pick.append(i)
            if len(pick) >= n:
                return pick
    return pick[:n]
