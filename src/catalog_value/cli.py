from __future__ import annotations

import argparse
from pathlib import Path

from catalog_value.catalogs import run_compare_catalogs, run_snapshot_catalogs
from catalog_value.config import load_config
from catalog_value.data.env import load_dotenv
from catalog_value.phase_a import run_figure1, run_fit, run_ingest, run_phase_a
from catalog_value.phase_b import run_phase_b
from catalog_value.phase_c import run_phase_c


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="catalog_value",
        description="Portfolio-aware content valuation",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: configs/phase_a.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Download MovieLens 25M and build the core subset")
    sub.add_parser("fit", help="Train the configured backbone (taste-token encoder or SVD)")
    sub.add_parser("figure1", help="Write popularity vs MCV scatter and table")
    sub.add_parser("phase-a", help="Run ingest, fit, and figure1")
    snap = sub.add_parser(
        "snapshot-catalogs",
        help="Fetch TMDB US watch-provider availability for MovieLens titles",
    )
    snap.add_argument("--force", action="store_true", help="Re-download even if a snapshot exists")
    sub.add_parser(
        "compare-catalogs",
        help="Score Netflix/Disney+/Prime/Max/Hulu catalogs under V(S) and MCV",
    )
    sub.add_parser(
        "phase-c",
        help="Map US streaming catalogs onto the learned title space and compare coverage",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "ingest":
        run_ingest(config)
    elif args.command == "fit":
        run_fit(config)
    elif args.command == "figure1":
        run_figure1(config)
    elif args.command == "phase-a":
        run_phase_a(config)
    elif args.command == "snapshot-catalogs":
        run_snapshot_catalogs(force=args.force)
    elif args.command == "compare-catalogs":
        run_compare_catalogs(config)
    elif args.command == "phase-b":
        run_phase_b(config)
    elif args.command == "phase-c":
        run_phase_c(config)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
