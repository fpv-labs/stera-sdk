"""Common I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
