"""Phase C figures: do streaming catalogs occupy different regions of title space?"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from catalog_value.visualization.style import INK, MUTED, SERVICE_COLORS, apply_style, savefig


def plot_occupancy(
    xy: np.ndarray,
    membership: pl.DataFrame,
    services: list[str],
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    n = len(services)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharex=True, sharey=True)
    keep = membership["n_ratings"].to_numpy() >= 800
    background = xy[keep]
    for ax, name in zip(axes, services, strict=True):
        ax.scatter(background[:, 0], background[:, 1], c="#e7e5e4", s=4, linewidths=0, zorder=0)
        rows = membership.filter(pl.col(name))["movie_row"].to_numpy().astype(np.int64)
        pts = xy[rows]
        if len(pts) > 40:
            kde = gaussian_kde(pts[:: max(1, len(pts) // 800)].T)
            xmin, xmax = background[:, 0].min(), background[:, 0].max()
            ymin, ymax = background[:, 1].min(), background[:, 1].max()
            xx, yy = np.meshgrid(
                np.linspace(xmin, xmax, 80),
                np.linspace(ymin, ymax, 80),
            )
            dens = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contourf(xx, yy, dens, levels=8, cmap="YlOrRd", alpha=0.55, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], c=SERVICE_COLORS.get(name, INK), s=6, alpha=0.55, linewidths=0, zorder=2)
        ax.set_title(f"{name}\nn={len(rows):,}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    fig.suptitle("Where each catalog sits in title space", fontsize=13, y=1.02)
    fig.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.78, bottom=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_overlap_pair(title_jaccard: np.ndarray, coverage_cos: np.ndarray, services: list[str], path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for ax, matrix, title, cmap in (
        (axes[0], title_jaccard, "Title overlap (Jaccard)", "Oranges"),
        (axes[1], coverage_cos, "Where they sit on the map (cosine)", "YlGnBu"),
    ):
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap)
        ax.set_xticks(range(len(services)), labels=services, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(services)), labels=services, fontsize=8)
        ax.set_title(title)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(len(services)):
            for j in range(len(services)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    fig.suptitle("Almost no shared titles; occupancy on the map still overlaps", fontsize=13)
    return savefig(fig, path)


def plot_value_bars(summary: pl.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    services = summary["service"].to_list()
    values = summary["V_S"].to_numpy()
    hybrid = summary["V_S_hybrid"].to_numpy()
    counts = summary["n_titles_in_movielens_core"].to_numpy()
    x = np.arange(len(services))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = [SERVICE_COLORS.get(s, MUTED) for s in services]
    ax.bar(x - width / 2, values, width, color=colors, alpha=0.95, label="collaborative z")
    ax.bar(x + width / 2, hybrid, width, color=colors, alpha=0.45, label="hybrid posterior mean")
    ax.set_xticks(x, services)
    ax.set_ylabel("Audience coverage  V(S)")
    ax.set_title("US catalogs, MovieLens-core intersection")
    for i, n in enumerate(counts):
        ax.text(i, max(values[i], hybrid[i]), f"n={n:,}", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.legend(loc="upper right")
    return savefig(fig, path)
