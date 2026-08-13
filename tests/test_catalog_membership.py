from __future__ import annotations

import polars as pl

from catalog_value.data.tmdb import membership_table


def test_membership_maps_provider_ids_to_services() -> None:
    movies = pl.DataFrame(
        {
            "movie_row": [0, 1, 2],
            "movieId": [10, 11, 12],
            "tmdbId": [100, 200, 300],
            "title": ["A", "B", "C"],
            "genres": ["Drama", "Comedy", "Horror"],
            "n_ratings": [5, 6, 7],
        }
    )
    snapshot = pl.DataFrame(
        {
            "tmdb_id": [100, 200, 300],
            "provider_ids": [[8, 9], [337], []],
        }
    )
    table = membership_table(movies, snapshot)
    assert table.filter(pl.col("title") == "A")["Netflix"][0] is True
    assert table.filter(pl.col("title") == "A")["Prime Video"][0] is True
    assert table.filter(pl.col("title") == "B")["Disney+"][0] is True
    assert table.filter(pl.col("title") == "C")["Netflix"][0] is False
    assert table.filter(pl.col("title") == "C")["Hulu"][0] is False
