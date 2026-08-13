from __future__ import annotations

import argparse
from pathlib import Path

from catalog_value.config import load_config
from catalog_value.phase_a import run_figure1, run_fit, run_ingest, run_phase_a


def main(argv: list[str] | None = None) -> None:
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
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
