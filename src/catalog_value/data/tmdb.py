from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from tqdm import tqdm

from catalog_value.data.env import require_tmdb_api_key
from catalog_value.data.movielens import require_core
from catalog_value.paths import data_dir

TMDB_BASE = "https://api.themoviedb.org/3"
USER_AGENT = "catalog-value/0.1 (research; tmdb watch-provider snapshot)"

# US subscription catalog IDs, including ad-tier / kids SKUs of the same service.
US_FLATRATE_PROVIDERS: dict[str, tuple[int, ...]] = {
    "Netflix": (8, 175, 1796),
    "Disney+": (337,),
    "Prime Video": (9, 613, 2100),
    "Max": (1899, 1825),
    "Hulu": (15,),
}


def snapshot_path() -> Path:
    return data_dir() / "raw" / "tmdb" / "watch_providers_us.parquet"


def _request_json(path: str, params: dict[str, str], *, retries: int = 4) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{TMDB_BASE}{path}?{query}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"_http_status": 404}
            if exc.code in {429, 500, 502, 503} and attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                last_error = exc
                continue
            raise
        except urllib.error.URLError as exc:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                last_error = exc
                continue
            raise
    assert last_error is not None
    raise last_error


def fetch_us_flatrate_providers(tmdb_id: int, api_key: str) -> dict:
    retrieved = datetime.now(UTC).isoformat()
    try:
        payload = _request_json(
            f"/movie/{tmdb_id}/watch/providers",
            {"api_key": api_key},
        )
    except urllib.error.HTTPError as exc:
        return {
            "tmdb_id": tmdb_id,
            "http_status": int(exc.code),
            "retrieved_at": retrieved,
            "provider_ids": [],
        }
    if payload.get("_http_status") == 404:
        return {
            "tmdb_id": tmdb_id,
            "http_status": 404,
            "retrieved_at": retrieved,
            "provider_ids": [],
        }
    us = (payload.get("results") or {}).get("US") or {}
    flatrate = us.get("flatrate") or []
    return {
        "tmdb_id": tmdb_id,
        "http_status": 200,
        "retrieved_at": retrieved,
        "provider_ids": sorted(
            {int(row["provider_id"]) for row in flatrate if "provider_id" in row}
        ),
    }


def snapshot_watch_providers(*, force: bool = False, workers: int = 12) -> Path:
    """Fetch US flatrate providers for every core MovieLens title with a TMDB id."""
    api_key = require_tmdb_api_key()
    movies = pl.read_parquet(require_core() / "movies.parquet")
    ids = (
        movies.filter(pl.col("tmdbId").is_not_null())
        .select(pl.col("tmdbId").cast(pl.Int64).alias("tmdb_id"))
        .unique()
        .sort("tmdb_id")["tmdb_id"]
        .to_list()
    )
    dest = snapshot_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    existing: list[dict] = []
    if dest.exists() and not force:
        cached = pl.read_parquet(dest)
        existing = cached.to_dicts()
        done = set(cached["tmdb_id"].to_list())
        print(f"Resuming TMDB snapshot ({len(done):,} already cached)")

    remaining = [tid for tid in ids if tid not in done]
    print(f"Fetching US watch providers for {len(remaining):,} titles")
    rows = list(existing)
    if remaining:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch_us_flatrate_providers, tid, api_key): tid for tid in remaining
            }
            completed = 0
            for fut in tqdm(as_completed(futures), total=len(futures), desc="tmdb providers"):
                rows.append(fut.result())
                completed += 1
                if completed % 500 == 0:
                    pl.DataFrame(rows).write_parquet(dest)
        pl.DataFrame(rows).write_parquet(dest)
    print(f"Wrote {dest} ({len(rows):,} titles)")
    return dest


def membership_table(movies: pl.DataFrame, snapshot: pl.DataFrame) -> pl.DataFrame:
    """Boolean catalog membership for each MovieLens core title × service."""
    exploded = snapshot.explode("provider_ids", empty_as_null=True).rename(
        {"provider_ids": "provider_id"}
    )
    parts = [movies.select(["movie_row", "movieId", "tmdbId", "title", "genres", "n_ratings"])]
    for name, provider_ids in US_FLATRATE_PROVIDERS.items():
        present = (
            exploded.filter(pl.col("provider_id").is_in(list(provider_ids)))
            .select(pl.col("tmdb_id").alias("tmdbId"))
            .unique()
            .with_columns(pl.lit(True).alias(name))
        )
        parts[0] = parts[0].join(present, on="tmdbId", how="left")
    table = parts[0]
    for name in US_FLATRATE_PROVIDERS:
        table = table.with_columns(pl.col(name).fill_null(False))
    return table
