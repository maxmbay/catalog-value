"""Phase C: occupancy of real US catalogs in learned audience space."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl

from catalog_value.config import Config
from catalog_value.data.movielens import require_core
from catalog_value.data.tmdb import US_FLATRATE_PROVIDERS, membership_table, snapshot_path
from catalog_value.models.audience.mixture import load_audience, subsample_audience
from catalog_value.models.catalog_value.analytical import coverage_g
from catalog_value.models.catalog_value.mcv import affinity, value_of_catalog
from catalog_value.paths import output_dir
from catalog_value.phase_a import _load_titles, artifact_dir
from catalog_value.phase_b import load_posterior
from catalog_value.visualization.atlas import load_atlas
from catalog_value.visualization.phase_c import plot_occupancy, plot_overlap_pair, plot_value_bars
from catalog_value.visualization.style import figures_dir


def phase_c_dir() -> Path:
    path = output_dir() / "phase_c"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _taste_coverage(audience, titles, catalog, tau: float) -> np.ndarray:
    if catalog.size == 0:
        return np.zeros(audience.pi.shape[1], dtype=np.float64)
    aff = affinity(audience, titles, catalog)
    per_taste = coverage_g(aff, tau)
    return (audience.pi * per_taste).mean(axis=0)


def run_phase_c(config: Config) -> Path:
    snap = snapshot_path()
    if not snap.exists():
        raise FileNotFoundError("TMDB snapshot missing. Run: python -m catalog_value snapshot-catalogs")
    movies = pl.read_parquet(require_core() / "movies.parquet")
    membership = membership_table(movies, pl.read_parquet(snap))
    audience_full, _ = load_audience(artifact_dir() / "audience.npz")
    audience = subsample_audience(audience_full, n=config.catalog_value.n_eval_users, seed=config.seed)
    titles = _load_titles(artifact_dir(), config)
    posterior, _, _ = load_posterior()
    hybrid = posterior.mean_reps()
    xy, _ = load_atlas(artifact_dir() / "title_atlas.npz")
    tau = config.catalog_value.tau
    services = list(US_FLATRATE_PROVIDERS)
    out = phase_c_dir()

    catalogs: dict[str, np.ndarray] = {}
    coverage = np.zeros((len(services), audience.pi.shape[1]), dtype=np.float64)
    rows = []
    print("Scoring US catalogs in the MovieLens-core intersection (coverage, not a buy list).")
    for i, name in enumerate(services):
        catalog = membership.filter(pl.col(name))["movie_row"].to_numpy().astype(np.int64)
        catalogs[name] = catalog
        v = value_of_catalog(audience, titles, catalog, tau)
        v_h = value_of_catalog(audience, hybrid, catalog, tau)
        coverage[i] = _taste_coverage(audience, titles, catalog, tau)
        rows.append(
            {
                "service": name,
                "n_titles_in_movielens_core": int(catalog.size),
                "V_S": v.mean,
                "V_S_hybrid": v_h.mean,
                "V_per_title": v.mean / max(catalog.size, 1),
            }
        )
        print(f"{name}: n={catalog.size:,}  V={v.mean:.3f}  hybrid V={v_h.mean:.3f}")

    summary = pl.DataFrame(rows).sort("V_S", descending=True)
    summary.write_csv(out / "summary.csv")

    n = len(services)
    title_j = np.eye(n)
    occ_cos = np.eye(n)
    # Shared binning so occupancy cosine is comparable across catalogs.
    keep = membership["n_ratings"].to_numpy() >= 800
    lo = xy[keep].min(axis=0)
    hi = xy[keep].max(axis=0)
    hists = []
    for name in services:
        rows = catalogs[name]
        hist, _, _ = np.histogram2d(
            xy[rows, 0], xy[rows, 1], bins=18, range=[[lo[0], hi[0]], [lo[1], hi[1]]]
        )
        vec = hist.ravel().astype(np.float64)
        vec = vec / np.linalg.norm(vec).clip(min=1e-8)
        hists.append(vec)
    for i, a in enumerate(services):
        for j, b in enumerate(services):
            title_j[i, j] = _jaccard(set(map(int, catalogs[a])), set(map(int, catalogs[b])))
            occ_cos[i, j] = float(hists[i] @ hists[j])
    print("title Jaccard\n", np.round(title_j, 2))
    print("occupancy cosine\n", np.round(occ_cos, 2))
    unit = coverage / np.linalg.norm(coverage, axis=1, keepdims=True).clip(min=1e-8)
    print("taste-coverage cosine (collapsed if π is uniform)\n", np.round(unit @ unit.T, 2))

    published = figures_dir("phase_c")
    occ = plot_occupancy(xy, membership, services, out / "occupancy.png")
    overlap = plot_overlap_pair(title_j, occ_cos, services, out / "overlap.png")
    bars = plot_value_bars(summary, out / "value_comparison.png")
    for src in (occ, overlap, bars):
        shutil.copy2(src, published / src.name)
        print(f"Wrote {published / src.name}")
    return published
