"""Loader konfigurasi proyek dari config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Root proyek = satu level di atas folder src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Baca config.yaml dan kembalikan sebagai dict.

    Path relatif di dalam config (paths.*) tidak diubah; gunakan
    ``resolve_path`` untuk mengubahnya menjadi absolut terhadap root proyek.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str | Path) -> Path:
    """Ubah path relatif (dari config) menjadi absolut terhadap root proyek."""
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p
