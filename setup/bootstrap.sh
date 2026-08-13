#!/usr/bin/env bash
set -euo pipefail

echo "Bootstrapping development environment..."

# Make sure uv exists
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed."
    echo "Install it from: https://docs.astral.sh/uv/"
    exit 1
fi

# Create/update .venv and install dependencies
uv sync

echo "Done."
echo "Run commands with:"
echo "  uv run python ..."
echo "  uv run pytest"