from __future__ import annotations

from catalog_value.phase_c import _jaccard


def test_jaccard_extremes() -> None:
    assert _jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
    assert _jaccard({1}, {2}) == 0.0
    assert abs(_jaccard({1, 2}, {2, 3}) - 1 / 3) < 1e-12
