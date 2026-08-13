"""Phase B figures: content embeddings and cold-start MCV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from catalog_value.visualization.atlas import short_title
from catalog_value.visualization.style import BLUE, INK, MUTED, TEAL, apply_style, savefig


def plot_reconstruction(n_ratings: np.ndarray, cosine: np.ndarray, holdout: np.ndarray, path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.scatter(
        n_ratings[~holdout],
        cosine[~holdout],
        s=10,
        c=MUTED,
        alpha=0.35,
        linewidths=0,
        label="encoder train set",
    )
    ax.scatter(
        n_ratings[holdout],
        cosine[holdout],
        s=16,
        c=TEAL,
        alpha=0.75,
        linewidths=0,
        label="held-out cold start",
    )
    ax.set_xscale("log")
    ax.set_xlabel("MovieLens rating count")
    ax.set_ylabel("cosine(content z, collaborative z)")
    ax.set_title("Content encoder recovers well-observed titles; cold-start is the gap")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    return savefig(fig, path)


def plot_neighbor_swap(
    panels: list[tuple[str, pl.DataFrame, pl.DataFrame]],
    path: Path,
) -> Path:
    """Collaborative vs content neighbors for the same probes."""
    import matplotlib.pyplot as plt

    apply_style()
    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(10.4, 2.6 * n))
    if n == 1:
        axes = np.array([axes])
    for row, (name, collab, content) in enumerate(panels):
        for col, (table, color, subtitle) in enumerate(
            (
                (collab, BLUE, "collaborative"),
                (content, TEAL, "content (cold start)"),
            )
        ):
            ax = axes[row, col]
            labels = [short_title(t) for t in table["title"].to_list()]
            scores = table["cosine"].to_numpy()
            y = np.arange(len(labels))[::-1]
            ax.barh(y, scores, color=color, height=0.62)
            ax.set_yticks(y, labels=labels, fontsize=8)
            ax.set_xlim(0, max(0.4, float(scores.max()) * 1.2 if len(scores) else 1.0))
            if row == 0:
                ax.set_title(subtitle, fontsize=11)
            if col == 0:
                ax.set_ylabel(short_title(name).replace(" ", "\n"), fontsize=9)
    fig.suptitle("What “nearby” means before and after content", fontsize=13, y=0.995)
    fig.subplots_adjust(wspace=0.55, hspace=0.38, left=0.16, right=0.98, top=0.90, bottom=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_mcv_transfer(collab: np.ndarray, content: np.ndarray, titles: list[str], path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 6.6))
    ax.scatter(collab, content, s=22, c=BLUE, alpha=0.75, linewidths=0)
    lo = min(float(collab.min()), float(content.min()))
    hi = max(float(collab.max()), float(content.max()))
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("MCV from collaborative z")
    ax.set_ylabel("MCV from content z (held-out titles)")
    ax.set_title("Does content ranking preserve portfolio value?")
    # annotate extremes of disagreement
    delta = np.abs(collab - content)
    for i in np.argsort(delta)[::-1][:6]:
        ax.annotate(short_title(titles[i]), (collab[i], content[i]), fontsize=7, color=INK, xytext=(5, 5),
                    textcoords="offset points")
    return savefig(fig, path)


def plot_mcv_uncertainty(
    mean: np.ndarray,
    std: np.ndarray,
    n_ratings: np.ndarray,
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    sc = ax.scatter(mean, std, c=np.log10(np.clip(n_ratings, 1, None)), cmap="YlOrRd", s=22, alpha=0.85, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("log10 rating count")
    ax.set_xlabel("E[MCVᵢ(S)]  under p(zᵢ | D)")
    ax.set_ylabel("Std[MCVᵢ(S)]")
    ax.set_title("Cold titles are the uncertain ones")
    return savefig(fig, path)
