from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import polars as pl
from tqdm import tqdm

from catalog_value.config import Config
from catalog_value.paths import data_dir

MOVIELENS_MEMBERS = (
    "ratings.csv",
    "movies.csv",
    "links.csv",
    "tags.csv",
    "genome-scores.csv",
    "genome-tags.csv",
)


class _DownloadProgress(tqdm):
    def hook(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            self.total = total_size
        self.update(block_num * block_size - self.n)


def raw_movielens_dir() -> Path:
    return data_dir() / "raw" / "ml-25m"


def processed_dir() -> Path:
    return data_dir() / "processed"


def core_dir() -> Path:
    return processed_dir() / "core"


def download_movielens(url: str, *, force: bool = False) -> Path:
    """Download and extract MovieLens 25M into data/raw/ml-25m."""
    dest = raw_movielens_dir()
    ratings = dest / "ratings.csv"
    if ratings.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / "ml-25m.zip"
    if force or not zip_path.exists():
        print(f"Downloading MovieLens 25M from {url}")
        with _DownloadProgress(unit="B", unit_scale=True, desc="ml-25m.zip") as bar:
            urlretrieve(url, zip_path, reporthook=bar.hook)

    if dest.exists():
        shutil.rmtree(dest)

    print(f"Extracting {zip_path} → {dest.parent}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest.parent)

    missing = [name for name in MOVIELENS_MEMBERS if not (dest / name).exists()]
    if missing:
        raise FileNotFoundError(f"MovieLens extract missing {missing} under {dest}")
    return dest


def _write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def ingest_movielens(config: Config, *, force: bool = False) -> Path:
    """Parse CSVs to parquet and build the filtered core rating subset."""
    raw = download_movielens(config.data.movielens_url, force=force)
    out = processed_dir()
    core = core_dir()
    core.mkdir(parents=True, exist_ok=True)

    movies = pl.read_csv(raw / "movies.csv")
    links = (
        pl.read_csv(raw / "links.csv")
        .with_columns(
            pl.col("imdbId").cast(pl.Utf8),
            pl.col("tmdbId").cast(pl.Int64, strict=False),
        )
    )
    ratings = pl.read_csv(raw / "ratings.csv")
    genome_tags = pl.read_csv(raw / "genome-tags.csv")
    genome_scores = pl.read_csv(raw / "genome-scores.csv")

    _write_parquet(movies, out / "movies.parquet")
    _write_parquet(links, out / "links.parquet")
    _write_parquet(ratings, out / "ratings.parquet")
    _write_parquet(genome_tags, out / "genome_tags.parquet")
    _write_parquet(genome_scores, out / "genome_scores.parquet")

    movie_counts = ratings.group_by("movieId").len().rename({"len": "n_ratings"})
    user_counts = ratings.group_by("userId").len().rename({"len": "n_ratings"})

    keep_movies = movie_counts.filter(pl.col("n_ratings") >= config.data.min_movie_ratings)
    keep_users = user_counts.filter(pl.col("n_ratings") >= config.data.min_user_ratings)

    core_ratings = (
        ratings.join(keep_movies.select("movieId"), on="movieId", how="inner")
        .join(keep_users.select("userId"), on="userId", how="inner")
    )
    # Re-apply thresholds after the joint filter so the core matrix is dense enough.
    keep_movies = (
        core_ratings.group_by("movieId")
        .len()
        .rename({"len": "n_ratings"})
        .filter(pl.col("n_ratings") >= config.data.min_movie_ratings)
    )
    keep_users = (
        core_ratings.group_by("userId")
        .len()
        .rename({"len": "n_ratings"})
        .filter(pl.col("n_ratings") >= config.data.min_user_ratings)
    )
    core_ratings = (
        core_ratings.join(keep_movies.select("movieId"), on="movieId", how="inner")
        .join(keep_users.select("userId"), on="userId", how="inner")
    )

    movie_index = (
        keep_movies.sort("movieId")
        .rename({"n_ratings": "n_ratings_core"})
        .with_row_index("movie_row")
        .join(movies, on="movieId", how="left")
        .join(links, on="movieId", how="left")
        .join(movie_counts.rename({"n_ratings": "n_ratings"}), on="movieId", how="left")
    )
    user_index = keep_users.sort("userId").with_row_index("user_row")

    core_ratings = core_ratings.join(
        movie_index.select(["movieId", "movie_row"]), on="movieId"
    ).join(user_index.select(["userId", "user_row"]), on="userId")

    _write_parquet(movie_index, core / "movies.parquet")
    _write_parquet(user_index, core / "users.parquet")
    _write_parquet(
        core_ratings.select(["user_row", "movie_row", "userId", "movieId", "rating", "timestamp"]),
        core / "ratings.parquet",
    )

    stats = pl.DataFrame(
        {
            "n_users": [user_index.height],
            "n_movies": [movie_index.height],
            "n_ratings": [core_ratings.height],
            "min_user_ratings": [config.data.min_user_ratings],
            "min_movie_ratings": [config.data.min_movie_ratings],
        }
    )
    _write_parquet(stats, core / "stats.parquet")
    print(
        "Core subset: "
        f"{stats['n_users'][0]:,} users × {stats['n_movies'][0]:,} movies, "
        f"{stats['n_ratings'][0]:,} ratings"
    )
    return core


def require_core() -> Path:
    path = core_dir()
    if not (path / "ratings.parquet").exists():
        raise FileNotFoundError(
            "Core MovieLens subset missing. Run: python -m catalog_value ingest"
        )
    return path
