"""Namelist-derived configuration for the configured WRF-JAX case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseConfig:
    """Runtime configuration parsed from ``namelist.input``."""

    namelist: dict[str, dict[str, Any]]
    run_seconds: int
    time_step: float
    boundary_interval_seconds: int
    history_interval_seconds: int
    max_dom: int
    e_we: int
    e_sn: int
    e_vert: int
    dx: float
    dy: float
    start_time: str
    end_time: str


def _strip_comment(line: str) -> str:
    in_quote = False
    quote = ""
    for index, char in enumerate(line):
        if char in ("'", '"'):
            if not in_quote:
                in_quote = True
                quote = char
            elif quote == char:
                in_quote = False
        elif char == "!" and not in_quote:
            return line[:index]
    return line


def _split_values(raw: str) -> list[str]:
    values: list[str] = []
    current: list[str] = []
    in_quote = False
    quote = ""
    for char in raw:
        if char in ("'", '"'):
            if not in_quote:
                in_quote = True
                quote = char
            elif quote == char:
                in_quote = False
            current.append(char)
        elif char == "," and not in_quote:
            value = "".join(current).strip()
            if value:
                values.append(value)
            current = []
        else:
            current.append(char)
    value = "".join(current).strip()
    if value:
        values.append(value)
    return values


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    lower = text.lower()
    if lower in (".true.", "true"):
        return True
    if lower in (".false.", "false"):
        return False
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]
    try:
        if any(marker in lower for marker in (".", "e", "d")):
            return float(lower.replace("d", "e"))
        return int(text)
    except ValueError:
        return text


def parse_namelist(path: str | Path) -> dict[str, dict[str, Any]]:
    """Parse the subset of Fortran namelist syntax used by WRF ``namelist.input``."""

    groups: dict[str, dict[str, Any]] = {}
    current_group: str | None = None
    pending_key: str | None = None
    pending_value: list[str] = []

    def flush_pending() -> None:
        nonlocal pending_key, pending_value
        if current_group is None or pending_key is None:
            return
        values = [_parse_scalar(item) for item in _split_values(" ".join(pending_value))]
        groups[current_group][pending_key] = values[0] if len(values) == 1 else values
        pending_key = None
        pending_value = []

    for raw_line in Path(path).read_text(encoding="ascii").splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("&"):
            flush_pending()
            current_group = line[1:].strip().lower()
            groups[current_group] = {}
            continue
        if line.startswith("/"):
            flush_pending()
            current_group = None
            continue
        if current_group is None:
            continue
        if "=" in line:
            flush_pending()
            key, value = line.split("=", 1)
            pending_key = key.strip().lower()
            pending_value = [value.strip()]
        elif pending_key is not None:
            pending_value.append(line)
    flush_pending()
    return groups


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _group_value(groups: dict[str, dict[str, Any]], group: str, key: str, default: Any = None) -> Any:
    return groups.get(group, {}).get(key, default)


def _indexed(groups: dict[str, dict[str, Any]], group: str, key: str, default: Any = None, index: int = 0) -> Any:
    values = _as_list(_group_value(groups, group, key, default))
    if index < len(values):
        return values[index]
    return values[-1]


def _timestamp(groups: dict[str, dict[str, Any]], prefix: str, index: int = 0) -> str:
    year = int(_indexed(groups, "time_control", f"{prefix}_year", index=index))
    month = int(_indexed(groups, "time_control", f"{prefix}_month", index=index))
    day = int(_indexed(groups, "time_control", f"{prefix}_day", index=index))
    hour = int(_indexed(groups, "time_control", f"{prefix}_hour", 0, index=index))
    minute = int(_indexed(groups, "time_control", f"{prefix}_minute", 0, index=index))
    second = int(_indexed(groups, "time_control", f"{prefix}_second", 0, index=index))
    return f"{year:04d}-{month:02d}-{day:02d}_{hour:02d}:{minute:02d}:{second:02d}"


def _history_interval_seconds(groups: dict[str, dict[str, Any]], index: int = 0) -> int:
    if _group_value(groups, "time_control", "history_interval_s") is not None:
        return int(_indexed(groups, "time_control", "history_interval_s", index=index))
    if _group_value(groups, "time_control", "history_interval_m") is not None:
        return int(_indexed(groups, "time_control", "history_interval_m", index=index)) * 60
    if _group_value(groups, "time_control", "history_interval_h") is not None:
        return int(_indexed(groups, "time_control", "history_interval_h", index=index)) * 3600
    return int(_indexed(groups, "time_control", "history_interval", 60, index=index)) * 60


def load_case_config(path: str | Path) -> CaseConfig:
    groups = parse_namelist(path)
    run_seconds = (
        int(_group_value(groups, "time_control", "run_days", 0)) * 86400
        + int(_group_value(groups, "time_control", "run_hours", 0)) * 3600
        + int(_group_value(groups, "time_control", "run_minutes", 0)) * 60
        + int(_group_value(groups, "time_control", "run_seconds", 0))
    )
    return CaseConfig(
        namelist=groups,
        run_seconds=run_seconds,
        time_step=float(_group_value(groups, "domains", "time_step")),
        boundary_interval_seconds=int(_group_value(groups, "time_control", "interval_seconds", run_seconds)),
        history_interval_seconds=_history_interval_seconds(groups),
        max_dom=int(_group_value(groups, "domains", "max_dom", 1)),
        e_we=int(_indexed(groups, "domains", "e_we")),
        e_sn=int(_indexed(groups, "domains", "e_sn")),
        e_vert=int(_indexed(groups, "domains", "e_vert")),
        dx=float(_indexed(groups, "domains", "dx")),
        dy=float(_indexed(groups, "domains", "dy")),
        start_time=_timestamp(groups, "start"),
        end_time=_timestamp(groups, "end"),
    )


__all__ = ["CaseConfig", "load_case_config", "parse_namelist"]
