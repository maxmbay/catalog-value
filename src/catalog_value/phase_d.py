"""Phase D: ablations, PACV, greedy portfolios vs popularity."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl

from catalog_value.config import Config
from catalog_value.data.movielens import require_core
from catalog_value.models.audience.mixture import (
    build_audience_states,
    fit_item_prototypes,
    load_audience,
    subsample_audience,
)
from catalog_value.models.catalog_value.mcv import marginal_content_value, value_of_catalog
from catalog_value.models.catalog_value.pacv import portfolio_adjusted_value
from catalog_value.models.content.svd import load_fit, ratings_to_sparse, title_reps_from_fit
from catalog_value.models.types import TitleReps
from catalog_value.optimization.greedy import greedy_mcv
from catalog_value.paths import output_dir
from catalog_value.phase_a import _load_titles, _popular_rows, artifact_dir
from catalog_value.phase_b import load_posterior
from catalog_value.visualization.phase_d import plot_ablation, plot_genre_mix, plot_growth, plot_pacv
from catalog_value.visualization.style import figures_dir


def phase_d_dir() -> Path:
    path = output_dir() / "phase_d"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _primary_genre(genres: str) -> str:
    return genres.split("|")[0] if genres else "Unknown"


def _genre_share(movies: pl.DataFrame, rows: np.ndarray) -> dict[str, float]:
    picked = movies.filter(pl.col("movie_row").is_in(rows.tolist()))
    n = max(picked.height, 1)
    counts: dict[str, int] = {}
    for g in picked["genres"].to_list():
        key = _primary_genre(str(g))
        counts[key] = counts.get(key, 0) + 1
    return {k: v / n for k, v in counts.items()}


def _growth_curve(audience, titles, order: np.ndarray, tau: float) -> list[float]:
    return [value_of_catalog(audience, titles, order[:n], tau).mean for n in range(1, len(order) + 1)]


def run_phase_d(config: Config) -> Path:
    movies = pl.read_parquet(require_core() / "movies.parquet")
    ratings_df = pl.read_parquet(require_core() / "ratings.parquet")
    n_users = pl.read_parquet(require_core() / "users.parquet").height
    n_movies = movies.height
    audience_full, _ = load_audience(artifact_dir() / "audience.npz")
    audience = subsample_audience(audience_full, n=config.catalog_value.n_eval_users, seed=config.seed)
    titles = _load_titles(artifact_dir(), config)
    posterior, z_content, _ = load_posterior()
    content_titles = TitleReps(z=z_content, bias=posterior.bias, movie_row=titles.movie_row)
    tau = config.catalog_value.tau
    cfg = config.phase_d
    out = phase_d_dir()

    pool = _popular_rows(movies, 900)
    popular_order = pool[: cfg.greedy_size]
    greedy_pool = pool[cfg.greedy_size : cfg.greedy_size + 280]
    print(f"Greedy catalog from {len(greedy_pool)} candidates, size={cfg.greedy_size}")
    greedy_order = greedy_mcv(audience, titles, greedy_pool, cfg.greedy_size, tau)
    popular_v = _growth_curve(audience, titles, popular_order, tau)
    greedy_v = _growth_curve(audience, titles, greedy_order, tau)
    print(f"V(popular {cfg.greedy_size})={popular_v[-1]:.3f}  V(greedy {cfg.greedy_size})={greedy_v[-1]:.3f}")

    universe = pool[:400]
    # Mix high/low MCV candidates from the Phase A table so PACV is readable.
    mcv_table = pl.read_csv(artifact_dir() / "popularity_vs_mcv.csv")
    high = mcv_table.sort("mcv", descending=True).head(15)["movieId"].to_list()
    low = mcv_table.sort("mcv").head(15)["movieId"].to_list()
    id_to_row = dict(zip(movies["movieId"].to_list(), movies["movie_row"].to_list()))
    pacv_ids = [id_to_row[i] for i in high + low if i in id_to_row]
    pacv_candidates = np.unique(np.array(pacv_ids, dtype=np.int64))
    print(f"PACV over {cfg.n_shapley_catalogs} catalogs, {len(pacv_candidates)} candidates")
    pacv = portfolio_adjusted_value(
        audience,
        titles,
        universe,
        pacv_candidates,
        n_catalogs=cfg.n_shapley_catalogs,
        catalog_size=cfg.shapley_catalog_size,
        tau=tau,
        seed=config.seed,
    )
    title_of = dict(zip(movies["movie_row"].to_list(), movies["title"].to_list()))
    n_of = dict(zip(movies["movie_row"].to_list(), movies["n_ratings"].to_list()))
    pacv_table = pl.DataFrame(
        {
            "movie_row": pacv_candidates,
            "title": [title_of[int(i)] for i in pacv_candidates],
            "n_ratings": [n_of[int(i)] for i in pacv_candidates],
            "pacv": pacv,
        }
    ).sort("pacv", descending=True)
    pacv_table.write_csv(out / "pacv.csv")
    print(pacv_table.head(8))

    print("Rebuilding SVD audience for the backbone ablation")
    fit = load_fit(artifact_dir() / "svd.npz")
    ratings = ratings_to_sparse(
        ratings_df["user_row"].to_numpy(),
        ratings_df["movie_row"].to_numpy(),
        ratings_df["rating"].to_numpy(),
        n_users,
        n_movies,
    )
    prototypes = fit_item_prototypes(fit.item_factors, config.representation.n_interests, config.seed)
    svd_audience = subsample_audience(
        build_audience_states(ratings, fit, prototypes),
        n=config.catalog_value.n_eval_users,
        seed=config.seed,
    )
    svd_titles = title_reps_from_fit(fit)
    catalog = _popular_rows(movies, config.phase_a.catalog_size)
    abl_candidates = _popular_rows(movies, config.phase_a.catalog_size + 400)[config.phase_a.catalog_size :]
    mcv_nn = marginal_content_value(audience, titles, catalog, abl_candidates, tau)
    mcv_svd = marginal_content_value(svd_audience, svd_titles, catalog, abl_candidates, tau)
    mcv_ct = marginal_content_value(audience, content_titles, catalog, abl_candidates, tau)
    rho_svd = float(pl.DataFrame({"a": mcv_nn, "b": mcv_svd}).select(pl.corr("a", "b", method="spearman")).item())
    rho_ct = float(pl.DataFrame({"a": mcv_nn, "b": mcv_ct}).select(pl.corr("a", "b", method="spearman")).item())
    print(f"Spearman MCV neural vs SVD={rho_svd:.3f}  neural vs content={rho_ct:.3f}")

    published = figures_dir("phase_d")
    growth = plot_growth(popular_v, greedy_v, out / "greedy_vs_popular.png")
    pacv_fig = plot_pacv(pacv_table, out / "pacv.png")
    abl = plot_ablation(mcv_nn, mcv_svd, mcv_ct, out / "ablation.png")
    genres = plot_genre_mix(
        _genre_share(movies, popular_order),
        _genre_share(movies, greedy_order),
        out / "genre_mix.png",
    )
    for src in (growth, pacv_fig, abl, genres):
        shutil.copy2(src, published / src.name)
        print(f"Wrote {published / src.name}")
    return published
