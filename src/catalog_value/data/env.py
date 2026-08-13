from __future__ import annotations

import os
from pathlib import Path

from catalog_value.paths import project_root


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env without overriding a real environment."""
    env_path = path if path is not None else project_root() / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_tmdb_api_key() -> str:
    load_dotenv()
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "TMDB_API_KEY is missing. Add it to .env (see .env.example)."
        )
    return key
