from __future__ import annotations

import pytest

from catalog_value.cli import main


def test_help_lists_phase_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in (
        "ingest",
        "fit",
        "figure1",
        "phase-a",
        "phase-b",
        "snapshot-catalogs",
        "compare-catalogs",
        "phase-c",
        "phase-d",
    ):
        assert cmd in out
