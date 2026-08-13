from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl

from catalog_value.config import Config
from catalog_value.data.movielens import core_dir, ingest_movielens, require_core
from catalog_value.models.audience.mixture import (
    build_audience_states,
    fit_item_prototypes,
    load_audience,
    save_audience,
    subsample_audience,
)
from catalog_value.models.audience.train import export_neural_fit, train_taste_tokens
from catalog_value.models.catalog_value.mcv import marginal_content_value, value_of_catalog
from catalog_value.models.content.svd import (
    fit_svd,
    load_fit,
    ratings_to_sparse,
    save_fit,
    title_reps_from_fit,
)
from catalog_value.models.types import AudienceStates, TitleReps
from catalog_value.paths import output_dir
from catalog_value.visualization.atlas import fit_atlas, save_atlas
from catalog_value.visualization.phase_a import (
    plot_diminishing_returns,
    plot_mcv_on_atlas,
    plot_neighbor_strips,
    plot_popularity_vs_mcv,
    plot_title_atlas,
)
from catalog_value.visualization.style import figures_dir

ARTIFACTS = "phase_a"


def artifact_dir() -> Path:
    path = output_dir() / ARTIFACTS
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_ingest(config: Config) -> Path:
    return ingest_movielens(config)


def run_fit(config: Config) -> None:
    if config.train.backbone == "taste_tokens":
        run_fit_neural(config)
    elif config.train.backbone == "svd":
        run_fit_svd(config)
    else:
        raise ValueError(f"Unknown train.backbone: {config.train.backbone}")


def run_fit_neural(config: Config) -> None:
    core = require_core()
    ratings_df = pl.read_parquet(core / "ratings.parquet")
    n_users = pl.read_parquet(core / "users.parquet").height
    n_movies = pl.read_parquet(core / "movies.parquet").height
    out = artifact_dir()
    model = train_taste_tokens(ratings_df, n_users, n_movies, config, out)
    export_neural_fit(model, ratings_df, n_users, config, out)


def run_fit_svd(config: Config) -> None:
    core = require_core()
    ratings_df = pl.read_parquet(core / "ratings.parquet")
    movies = pl.read_parquet(core / "movies.parquet")
    n_users = pl.read_parquet(core / "users.parquet").height
    n_movies = movies.height

    ratings = ratings_to_sparse(
        ratings_df["user_row"].to_numpy(),
        ratings_df["movie_row"].to_numpy(),
        ratings_df["rating"].to_numpy(),
        n_users,
        n_movies,
    )
    print(f"Fitting SVD (dim={config.representation.embedding_dim}) on {ratings.nnz:,} ratings")
    fit = fit_svd(ratings, dim=config.representation.embedding_dim)
    prototypes = fit_item_prototypes(
        fit.item_factors,
        n_interests=config.representation.n_interests,
        seed=config.seed,
    )
    print(f"Building {config.representation.n_interests}-interest audience states")
    audience = build_audience_states(ratings, fit, prototypes)

    out = artifact_dir()
    save_fit(fit, out / "svd.npz")
    save_audience(audience, prototypes, out / "audience.npz")
    print(f"Wrote {out / 'svd.npz'} and {out / 'audience.npz'}")


def _popular_rows(movies: pl.DataFrame, n: int) -> np.ndarray:
    return (
        movies.sort("n_ratings", descending=True)
        .head(n)["movie_row"]
        .to_numpy()
        .astype(np.int64)
    )


def _load_titles(out: Path, config: Config) -> TitleReps:
    if config.train.backbone == "taste_tokens":
        payload = np.load(out / "titles.npz")
        return TitleReps(
            z=payload["z"],
            bias=payload["bias"],
            movie_row=payload["movie_row"],
        )
    fit = load_fit(out / "svd.npz")
    return title_reps_from_fit(fit)


def run_figure1(config: Config) -> Path:
    core = require_core()
    movies = pl.read_parquet(core / "movies.parquet")
    out = artifact_dir()
    audience_full, _ = load_audience(out / "audience.npz")
    audience = subsample_audience(
        audience_full,
        n=config.catalog_value.n_eval_users,
        seed=config.seed,
    )
    titles = _load_titles(out, config)

    catalog_n = config.phase_a.catalog_size
    candidate_n = config.phase_a.n_candidates
    ranked = _popular_rows(movies, catalog_n + candidate_n)
    catalog = ranked[:catalog_n]
    candidates = ranked[catalog_n:]

    v_s = value_of_catalog(audience, titles, catalog, tau=config.catalog_value.tau)
    print(
        f"V(S) for top-{catalog_n} catalog: {v_s.mean:.4f} "
        f"(eval users={audience.pi.shape[0]:,})"
    )
    mcv = marginal_content_value(
        audience,
        titles,
        catalog,
        candidates,
        tau=config.catalog_value.tau,
    )

    candidate_movies = movies.filter(pl.col("movie_row").is_in(candidates.tolist()))
    mcv_table = pl.DataFrame({"movie_row": candidates, "mcv": mcv})
    table = (
        candidate_movies.join(mcv_table, on="movie_row", how="left")
        .with_columns(pl.col("n_ratings").log(10).alias("log10_n_ratings"))
        .sort("mcv", descending=True)
    )

    pop_med = float(table["n_ratings"].median())
    mcv_med = float(table["mcv"].median())
    table = table.with_columns(
        pl.when((pl.col("n_ratings") >= pop_med) & (pl.col("mcv") >= mcv_med))
        .then(pl.lit("high popularity / high MCV"))
        .when((pl.col("n_ratings") >= pop_med) & (pl.col("mcv") < mcv_med))
        .then(pl.lit("high popularity / low MCV"))
        .when((pl.col("n_ratings") < pop_med) & (pl.col("mcv") >= mcv_med))
        .then(pl.lit("low popularity / high MCV"))
        .otherwise(pl.lit("low popularity / low MCV"))
        .alias("quadrant")
    )

    csv_path = out / "popularity_vs_mcv.csv"
    fig_path = out / "figure1_popularity_vs_mcv.png"
    table.select(
        [
            "movieId",
            "title",
            "genres",
            "n_ratings",
            "mcv",
            "quadrant",
        ]
    ).write_csv(csv_path)
    plot_popularity_vs_mcv(table, fig_path, n_annotate=config.phase_a.n_annotate)
    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")
    _print_quadrant_examples(table)
    _write_phase_a_maps(movies, titles, audience, table, catalog, out)
    return fig_path


def _print_quadrant_examples(table: pl.DataFrame) -> None:
    print("\nQuadrant examples")
    for name in (
        "high popularity / low MCV",
        "low popularity / high MCV",
        "high popularity / high MCV",
        "low popularity / low MCV",
    ):
        subset = table.filter(pl.col("quadrant") == name)
        print(f"\n{name} (n={subset.height})")
        if subset.is_empty():
            continue
        if "high MCV" in name:
            rows = subset.sort("mcv", descending=True).head(3)
        else:
            rows = subset.sort("mcv").head(3)
        for row in rows.iter_rows(named=True):
            print(f"  {row['title']}: n={row['n_ratings']:,}  MCV={row['mcv']:.4f}")


def _write_phase_a_maps(
    movies: pl.DataFrame,
    titles: TitleReps,
    audience: AudienceStates,
    mcv_table: pl.DataFrame,
    catalog: np.ndarray,
    out: Path,
) -> None:
    xy, pca = fit_atlas(titles.z)
    save_atlas(out / "title_atlas.npz", xy, pca.explained_variance_ratio_)
    published = figures_dir("phase_a")
    atlas = plot_title_atlas(
        movies, xy, pca.explained_variance_ratio_, catalog, out / "atlas.png"
    )
    landscape = plot_mcv_on_atlas(
        movies, xy, mcv_table, catalog, out / "mcv_landscape.png"
    )
    probes = [
        "Super Mario Bros. (1993)",
        "Paths of Glory (1957)",
        "Halloween (1978)",
        "Toy Story (1995)",
    ]
    panels = []
    for name in probes:
        table = _neighbor_table(movies, titles.z, name)
        if table is not None:
            panels.append((name, table))
    neighbors = plot_neighbor_strips(panels, out / "neighbors.png")
    action = _rows_for(
        movies,
        "Die Hard (1988)",
        "Die Hard 2 (1990)",
        "Die Hard: With a Vengeance (1995)",
        "Lethal Weapon (1987)",
        "Terminator 2: Judgment Day (1991)",
    )
    mixed = _rows_for(
        movies,
        "Die Hard (1988)",
        "Toy Story (1995)",
        "Silence of the Lambs, The (1991)",
        "When Harry Met Sally... (1989)",
        "Planet Earth (2006)",
    )
    action_v = _coverage_curve(audience, titles, action)
    mixed_v = _coverage_curve(audience, titles, mixed)
    diminishing = plot_diminishing_returns(
        list(range(1, len(action_v) + 1)),
        action_v,
        mixed_v,
        out / "diminishing_returns.png",
    )
    for src in (atlas, landscape, neighbors, diminishing, out / "figure1_popularity_vs_mcv.png"):
        dest = published / src.name
        shutil.copy2(src, dest)
        print(f"Wrote {dest}")


def _rows_for(movies: pl.DataFrame, *names: str) -> np.ndarray:
    title_to_row = {
        t: int(r) for t, r in zip(movies["title"].to_list(), movies["movie_row"].to_list())
    }
    return np.array([title_to_row[n] for n in names if n in title_to_row], dtype=np.int64)


def _coverage_curve(
    audience: AudienceStates, titles: TitleReps, order: np.ndarray, tau: float = 0.5
) -> list[float]:
    return [value_of_catalog(audience, titles, order[:n], tau).mean for n in range(1, len(order) + 1)]


def _neighbor_table(
    movies: pl.DataFrame, z: np.ndarray, name: str, k: int = 6, min_ratings: int = 2000
) -> pl.DataFrame | None:
    hit = movies.filter(pl.col("title") == name)
    if hit.is_empty():
        return None
    idx = int(hit["movie_row"][0])
    unit = z / np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)
    scores = unit @ unit[idx]
    scores[idx] = -np.inf
    popular = set(
        int(r) for r in movies.filter(pl.col("n_ratings") >= min_ratings)["movie_row"].to_list()
    )
    ranked = [int(i) for i in np.argsort(scores)[::-1] if int(i) in popular][:k]
    return pl.DataFrame(
        {
            "title": [
                movies.filter(pl.col("movie_row") == i)["title"][0] for i in ranked
            ],
            "cosine": [float(scores[i]) for i in ranked],
        }
    )


def _fit_artifact(config: Config) -> Path:
    out = artifact_dir()
    if config.train.backbone == "taste_tokens":
        return out / "taste_tokens.pt"
    return out / "svd.npz"


def run_phase_a(config: Config) -> None:
    if not (core_dir() / "ratings.parquet").exists():
        run_ingest(config)
    if not _fit_artifact(config).exists():
        run_fit(config)
    run_figure1(config)
