from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_platform_value(summary: pl.DataFrame, path: Path) -> None:
    services = summary["service"].to_list()
    values = summary["V_S"].to_numpy()
    counts = summary["n_titles_in_movielens_core"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = ["#e50914", "#113ccf", "#00a8e1", "#b535f6", "#1ce783"]
    ax.bar(services, values, color=colors[: len(services)], width=0.65)
    ax.set_ylabel("Audience coverage  V(S)")
    ax.set_title("US catalogs, MovieLens-core intersection")
    for i, (v, n) in enumerate(zip(values, counts, strict=True)):
        ax.text(i, v, f"  n={n:,}", ha="center", va="bottom", fontsize=8, color="#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_platform_fingerprint(fingerprint: pl.DataFrame, n_interests: int, path: Path) -> None:
    services = fingerprint["service"].unique(maintain_order=True).to_list()
    matrix = np.zeros((len(services), n_interests), dtype=np.float64)
    for i, service in enumerate(services):
        rows = fingerprint.filter(pl.col("service") == service).sort("taste")
        matrix[i] = rows["coverage"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(range(len(services)), labels=services)
    ax.set_xticks(range(n_interests), labels=[f"z{k}" for k in range(n_interests)])
    ax.set_xlabel("Latent taste")
    ax.set_title("Platform fingerprints (π-weighted coverage by taste)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="coverage")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
