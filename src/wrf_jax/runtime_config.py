"""Shared accessors for repository run-time configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_CONFIG = REPO_ROOT / "config" / "default_run.json"


def load_default_run_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the default ``python main.py`` run configuration."""

    config_path = Path(path) if path is not None else DEFAULT_RUN_CONFIG
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="ascii") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return data


def configured_path(name: str, *, env: str | None = None, fallback: str | Path | None = None) -> Path:
    """Resolve a path from an environment override, default config, or fallback."""

    if env is not None:
        value = os.environ.get(env)
        if value not in (None, ""):
            return Path(value).expanduser()
    value = load_default_run_config().get(name)
    if value not in (None, ""):
        return Path(str(value)).expanduser()
    if fallback is None:
        raise KeyError(f"missing configured path {name!r}")
    return Path(fallback).expanduser()
