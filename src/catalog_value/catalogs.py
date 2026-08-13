from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from catalog_value.config import Config
from catalog_value.data.movielens import require_core
from catalog_value.data.tmdb import (
    US_FLATRATE_PROVIDERS,
    membership_table,
    snapshot_path,
    snapshot_watch_providers,
)
from catalog_value.models.audience.mixture import load_audience, subsample_audience
from catalog_value.models.catalog_value.analytical import coverage_g
from catalog_value.models.catalog_value.mcv import (
    affinity,
    marginal_content_value,
    value_of_catalog,
)
from catalog_value.models.types import AudienceStates, TitleReps
from catalog_value.paths import output_dir
from catalog_value.phase_a import _load_titles, artifact_dir
from catalog_value.visualization.catalogs import plot_platform_fingerprint, plot_platform_value


def catalogs_output_dir() -> Path:
    path = output_dir() / "catalogs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_eval_representations(config: Config) -> tuple[AudienceStates, TitleReps]:
    out = artifact_dir()
    audience_full, _ = load_audience(out / "audience.npz")
    audience = subsample_audience(
        audience_full,
        n=config.catalog_value.n_eval_users,
        seed=config.seed,
    )
    titles = _load_titles(out, config)
    return audience, titles


def _mcv_batched(
    audience: AudienceStates,
    titles: TitleReps,
    catalog: np.ndarray,
    candidates: np.ndarray,
    tau: float,
    batch_size: int = 400,
) -> np.ndarray:
    scores = np.zeros(len(candidates), dtype=np.float64)
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        scores[start : start + batch_size] = marginal_content_value(
            audience, titles, catalog, chunk, tau
        )
    return scores


def _taste_coverage(
    audience: AudienceStates,
    titles: TitleReps,
    catalog: np.ndarray,
    tau: float,
) -> np.ndarray:
    if catalog.size == 0:
        return np.zeros(audience.pi.shape[1], dtype=np.float64)
    aff = affinity(audience, titles, catalog)
    per_taste = coverage_g(aff, tau)
    return (audience.pi * per_taste).mean(axis=0)


def run_snapshot_catalogs(*, force: bool = False) -> Path:
    return snapshot_watch_providers(force=force)


def run_compare_catalogs(config: Config) -> Path:
    snap = snapshot_path()
    if not snap.exists():
        snap = snapshot_watch_providers()
    movies = pl.read_parquet(require_core() / "movies.parquet")
    snapshot = pl.read_parquet(snap)
    membership = membership_table(movies, snapshot)
    retrieved = snapshot["retrieved_at"].max()
    audience, titles = _load_eval_representations(config)
    tau = config.catalog_value.tau
    out = catalogs_output_dir()

    summary_rows = []
    fingerprint_rows = []
    n_interests = audience.pi.shape[1]

    print(
        "Scoring US flatrate catalogs restricted to the MovieLens core intersection. "
        "This is audience-preference coverage, not a licensing recommendation."
    )
    print(f"TMDB snapshot retrieved_at={retrieved}")

    for name in US_FLATRATE_PROVIDERS:
        catalog = (
            membership.filter(pl.col(name))["movie_row"].to_numpy().astype(np.int64)
        )
        in_set = set(int(i) for i in catalog)
        well_observed = membership.filter(pl.col("n_ratings") >= 1000)
        candidates = np.array(
            [
                int(row)
                for row in well_observed["movie_row"].to_list()
                if int(row) not in in_set
            ],
            dtype=np.int64,
        )
        value = value_of_catalog(audience, titles, catalog, tau)
        coverage = _taste_coverage(audience, titles, catalog, tau)
        mcv = _mcv_batched(audience, titles, catalog, candidates, tau)
        add_table = (
            membership.filter(~pl.col(name))
            .join(pl.DataFrame({"movie_row": candidates, "mcv": mcv}), on="movie_row")
            .sort("mcv", descending=True)
        )
        add_path = out / f"additions_{_slug(name)}.csv"
        add_table.select(["movieId", "title", "genres", "n_ratings", "mcv"]).head(25).write_csv(
            add_path
        )
        summary_rows.append(
            {
                "service": name,
                "n_titles_in_movielens_core": int(catalog.size),
                "V_S": value.mean,
                "top_addition": add_table["title"][0] if add_table.height else None,
                "top_addition_mcv": float(add_table["mcv"][0]) if add_table.height else None,
            }
        )
        for k, cov in enumerate(coverage):
            fingerprint_rows.append({"service": name, "taste": k, "coverage": float(cov)})
        print(
            f"{name}: |S∩MovieLens|={catalog.size:,}  V(S)={value.mean:.4f}  "
            f"top add={add_table['title'][0] if add_table.height else '—'}"
        )
        print(f"  wrote {add_path}")

    summary = pl.DataFrame(summary_rows).sort("V_S", descending=True)
    fingerprint = pl.DataFrame(fingerprint_rows)
    summary_path = out / "summary.csv"
    fingerprint_path = out / "fingerprint.csv"
    summary.write_csv(summary_path)
    fingerprint.write_csv(fingerprint_path)
    plot_platform_value(summary, out / "value_comparison.png")
    plot_platform_fingerprint(fingerprint, n_interests, out / "fingerprint.png")
    membership.write_parquet(out / "membership.parquet")
    print(f"Wrote {summary_path}")
    print(summary)
    return summary_path


def _slug(name: str) -> str:
    return name.lower().replace("+", "plus").replace(" ", "_")
