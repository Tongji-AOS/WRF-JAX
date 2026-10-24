"""Explicit-state translation of WRF ``module_tiles`` tiling helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from wrf_jax.compat.get_ijk_from_grid2 import (
    DATA_ORDER_XYZ,
    DATA_ORDER_XZY,
    DATA_ORDER_YXZ,
    DATA_ORDER_YZX,
    DATA_ORDER_ZXY,
    DATA_ORDER_ZYX,
    MODEL_DATA_ORDER,
)
from wrf_jax.compat.rlocproc import least_aspect, region_bounds
from wrf_jax.compat.set_tiles_masked import TilesMaskedResult, set_tiles_masked

TILE_NONE = 0
TILE_X = 1
TILE_Y = 2
TILE_XY = 3
MAX_TILING_ZONES = 20

_ORDER_PLANE_DIMS = {
    DATA_ORDER_XYZ: ("31", "32"),
    DATA_ORDER_YXZ: ("32", "31"),
    DATA_ORDER_ZXY: ("32", "33"),
    DATA_ORDER_ZYX: ("33", "32"),
    DATA_ORDER_XZY: ("31", "33"),
    DATA_ORDER_YZX: ("33", "31"),
}


@dataclass
class TileZone:
    """Cached WRF tile-zone bounds."""

    i_start: list[int] = field(default_factory=list)
    i_end: list[int] = field(default_factory=list)
    j_start: list[int] = field(default_factory=list)
    j_end: list[int] = field(default_factory=list)
    num_tiles: int = 0
    num_tiles_x: int = 0
    num_tiles_y: int = 0


@dataclass
class TileGrid:
    """Minimal mutable domain fields consumed and produced by ``set_tiles2``."""

    sp31: int
    ep31: int
    sp32: int
    ep32: int
    sp33: int
    ep33: int
    sm31: int
    em31: int
    sm32: int
    em32: int
    sm33: int
    em33: int
    num_tiles_spec: int = 0
    num_tiles_x: int = 0
    num_tiles_y: int = 0
    max_tiles: int = 0
    num_tiles: int = 0
    i_start: list[int] = field(default_factory=list)
    i_end: list[int] = field(default_factory=list)
    j_start: list[int] = field(default_factory=list)
    j_end: list[int] = field(default_factory=list)
    tiling_latch: list[bool] = field(default_factory=lambda: [False] * MAX_TILING_ZONES)
    tile_zones: list[TileZone] = field(default_factory=lambda: [TileZone() for _ in range(MAX_TILING_ZONES)])


@dataclass(frozen=True)
class TileConfig:
    """Explicit namelist subset used by WRF ``set_tiles2``."""

    numtiles: int = 1
    tile_sz_x: int = 0
    tile_sz_y: int = 0
    tile_strategy: int = TILE_NONE
    numtiles_inc: int = 1
    numtiles_x: int = 0
    numtiles_y: int = 0


@dataclass(frozen=True)
class SetTiles2Result:
    """Tile metadata produced by WRF ``module_tiles::set_tiles2``."""

    num_tiles: int
    num_tiles_spec: int
    num_tiles_x: int
    num_tiles_y: int
    max_tiles: int
    i_start: tuple[int, ...]
    i_end: tuple[int, ...]
    j_start: tuple[int, ...]
    j_end: tuple[int, ...]


@dataclass(frozen=True)
class SetTilesOnceResult:
    """Tile metadata restored or created by ``set_tiles_once``."""

    zone: int
    cached: bool
    tiles: SetTiles2Result


@dataclass(frozen=True)
class SetTiles3Result:
    """Tile metadata produced by WRF ``module_tiles::set_tiles3``."""

    num_tiles: int
    max_tiles: int
    i_start: tuple[int, ...]
    i_end: tuple[int, ...]
    j_start: tuple[int, ...]
    j_end: tuple[int, ...]


def _read(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        if name not in source:
            raise ValueError(f"missing required field {name!r}")
        return source[name]
    if not hasattr(source, name):
        raise ValueError(f"missing required field {name!r}")
    return getattr(source, name)


def _read_int(source: Any, name: str) -> int:
    return int(_read(source, name))


def _read_int_default(source: Any | None, name: str, default: int) -> int:
    if source is None:
        return int(default)
    if isinstance(source, Mapping):
        return int(source.get(name, default))
    return int(getattr(source, name, default))


def _write(grid: Any, name: str, value: Any) -> None:
    if isinstance(grid, MutableMapping):
        grid[name] = value
        return
    setattr(grid, name, value)


def _ensure_latch(grid: Any) -> list[bool]:
    try:
        latch = list(_read(grid, "tiling_latch"))
    except ValueError:
        latch = []
    if len(latch) < MAX_TILING_ZONES:
        latch.extend([False] * (MAX_TILING_ZONES - len(latch)))
    _write(grid, "tiling_latch", latch)
    return latch


def _ensure_tile_zones(grid: Any) -> list[TileZone]:
    try:
        zones = list(_read(grid, "tile_zones"))
    except ValueError:
        zones = []
    while len(zones) < MAX_TILING_ZONES:
        zones.append(TileZone())
    _write(grid, "tile_zones", zones)
    return zones


def _result_from_grid(grid: Any) -> SetTiles2Result:
    num_tiles = _read_int_default(grid, "num_tiles", 0)
    return SetTiles2Result(
        num_tiles=num_tiles,
        num_tiles_spec=_read_int_default(grid, "num_tiles_spec", num_tiles),
        num_tiles_x=_read_int_default(grid, "num_tiles_x", 0),
        num_tiles_y=_read_int_default(grid, "num_tiles_y", 0),
        max_tiles=_read_int_default(grid, "max_tiles", num_tiles),
        i_start=tuple(int(value) for value in list(_read(grid, "i_start"))[:num_tiles]),
        i_end=tuple(int(value) for value in list(_read(grid, "i_end"))[:num_tiles]),
        j_start=tuple(int(value) for value in list(_read(grid, "j_start"))[:num_tiles]),
        j_end=tuple(int(value) for value in list(_read(grid, "j_end"))[:num_tiles]),
    )


def _plane_bounds(grid: Any, model_data_order: int) -> tuple[int, int, int, int, int, int, int, int]:
    try:
        xdim, ydim = _ORDER_PLANE_DIMS[int(model_data_order)]
    except KeyError as exc:
        raise ValueError(f"unsupported model_data_order {model_data_order!r}") from exc
    spx = _read_int(grid, f"sp{xdim}")
    epx = _read_int(grid, f"ep{xdim}")
    spy = _read_int(grid, f"sp{ydim}")
    epy = _read_int(grid, f"ep{ydim}")
    smx = _read_int(grid, f"sm{xdim}")
    emx = _read_int(grid, f"em{xdim}")
    smy = _read_int(grid, f"sm{ydim}")
    emy = _read_int(grid, f"em{ydim}")
    return spx, epx, spy, epy, smx, emx, smy, emy


def _positive_env_int(env: Mapping[str, str], name: str) -> int | None:
    value = str(env.get(name, ""))
    if value == "":
        return None
    parsed = int(value.strip().split()[0])
    return parsed if parsed > 0 else None


def _allocate_grid_arrays(grid: Any, num_tiles: int) -> int:
    max_tiles = _read_int_default(grid, "max_tiles", 0)
    if int(num_tiles) > max_tiles:
        _write(grid, "i_start", [0] * int(num_tiles))
        _write(grid, "i_end", [0] * int(num_tiles))
        _write(grid, "j_start", [0] * int(num_tiles))
        _write(grid, "j_end", [0] * int(num_tiles))
        max_tiles = int(num_tiles)
        _write(grid, "max_tiles", max_tiles)
    return max_tiles


def _ensure_array(grid: Any, name: str, length: int) -> list[int]:
    try:
        value = list(_read(grid, name))
    except ValueError:
        value = []
    if len(value) < length:
        value.extend([0] * (length - len(value)))
    return value


def set_tiles2(
    grid: Any,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    ips: int,
    ipe: int,
    jps: int,
    jpe: int,
    *,
    config: Any | None = None,
    env: Mapping[str, str] | None = None,
    model_data_order: int = MODEL_DATA_ORDER,
) -> SetTiles2Result:
    """Compute WRF ``set_tiles2`` tile bounds with explicit grid/config state."""

    ids_i = int(ids)
    ide_i = int(ide)
    jds_i = int(jds)
    jde_i = int(jde)
    ips_i = int(ips)
    ipe_i = int(ipe)
    jps_i = int(jps)
    jpe_i = int(jpe)

    spx, epx, spy, epy, smx, emx, smy, emy = _plane_bounds(grid, int(model_data_order))
    if ips_i < smx:
        raise ValueError("ips<smx")
    if ipe_i > emx:
        raise ValueError("ipe>emx")
    if jps_i < smy:
        raise ValueError("jps<smy")
    if jpe_i > emy:
        raise ValueError("jpe>emy")

    env_map = os.environ if env is None else env
    num_tiles_spec = _read_int_default(grid, "num_tiles_spec", 0)
    num_tiles_x = _read_int_default(grid, "num_tiles_x", 0)
    num_tiles_y = _read_int_default(grid, "num_tiles_y", 0)

    if num_tiles_spec == 0:
        num_tiles = _read_int_default(config, "numtiles", 1)
        if num_tiles == 1:
            num_tiles = 1
            env_num_tiles = _positive_env_int(env_map, "WRF_NUM_TILES")
            if env_num_tiles is not None:
                num_tiles = env_num_tiles

        tile_sz_x = _read_int_default(config, "tile_sz_x", 0)
        tile_sz_y = _read_int_default(config, "tile_sz_y", 0)
        tile_strategy_spec = _read_int_default(config, "tile_strategy", TILE_NONE)
        num_tiles_inc = _read_int_default(config, "numtiles_inc", 0)
        num_tiles_x = _read_int_default(config, "numtiles_x", 0)
        num_tiles_y = _read_int_default(config, "numtiles_y", 0)

        if num_tiles_x == 0:
            env_num_tiles_x = _positive_env_int(env_map, "WRF_NUM_TILES_X")
            if env_num_tiles_x is not None:
                num_tiles_x = env_num_tiles_x
        if num_tiles_y == 0:
            env_num_tiles_y = _positive_env_int(env_map, "WRF_NUM_TILES_Y")
            if env_num_tiles_y is not None:
                num_tiles_y = env_num_tiles_y
        if num_tiles_inc == 0:
            env_num_tiles_inc = _positive_env_int(env_map, "WRF_NUM_TILES_INC")
            if env_num_tiles_inc is not None:
                num_tiles_inc = env_num_tiles_inc

        num_tiles_inc = max(num_tiles_inc, 1)
        if tile_strategy_spec == TILE_NONE:
            tile_strategy = TILE_Y
            if num_tiles > (epy - spy + 1) and num_tiles_x == 0 and num_tiles_y == 0:
                num_tiles_x = 1
                num_tiles_y = epy - spy + 1
                while num_tiles_x * num_tiles_inc * num_tiles_y < num_tiles:
                    num_tiles_x += 1
                num_tiles_x *= num_tiles_inc
                tile_strategy = TILE_XY
        else:
            tile_strategy = tile_strategy_spec

        if tile_sz_x >= 1 and tile_sz_y >= 1:
            num_tiles_x = (epx - spx + 1) // tile_sz_x
            if tile_sz_x * num_tiles_x < epx - spx + 1:
                num_tiles_x += 1
            num_tiles_y = (epy - spy + 1) // tile_sz_y
            if tile_sz_y * num_tiles_y < epy - spy + 1:
                num_tiles_y += 1
            num_tiles = num_tiles_x * num_tiles_y
        elif num_tiles_x >= 1 or num_tiles_y >= 1:
            if num_tiles_x >= 1 and num_tiles_y >= 1:
                num_tiles = num_tiles_x * num_tiles_y
            elif num_tiles_x >= 1:
                num_tiles_y = num_tiles // num_tiles_x
            else:
                num_tiles_x = num_tiles // num_tiles_y
        elif tile_strategy == TILE_X:
            num_tiles_x = num_tiles
            num_tiles_y = 1
        elif tile_strategy == TILE_Y:
            num_tiles_x = 1
            num_tiles_y = num_tiles
        else:
            num_tiles_y, num_tiles_x = least_aspect(num_tiles, 1, 1)

        num_tiles = max(num_tiles, 1)
        num_tiles_x = max(num_tiles_x, 1)
        num_tiles_y = max(num_tiles_y, 1)
        num_tiles_spec = num_tiles
        _write(grid, "num_tiles_spec", num_tiles_spec)
        _write(grid, "num_tiles_x", num_tiles_x)
        _write(grid, "num_tiles_y", num_tiles_y)

    num_tiles = _read_int_default(grid, "num_tiles_spec", num_tiles_spec)
    num_tiles_x = _read_int_default(grid, "num_tiles_x", num_tiles_x)
    num_tiles_y = _read_int_default(grid, "num_tiles_y", num_tiles_y)
    max_tiles = _allocate_grid_arrays(grid, num_tiles)
    i_start = _ensure_array(grid, "i_start", num_tiles)
    i_end = _ensure_array(grid, "i_end", num_tiles)
    j_start = _ensure_array(grid, "j_start", num_tiles)
    j_end = _ensure_array(grid, "j_end", num_tiles)

    nt = 1
    for t in range(0, num_tiles):
        ntiles = t // num_tiles_x
        ts, te = region_bounds(spy, epy, num_tiles_y, ntiles)
        if ts <= te:
            if jps_i < spy and ts == spy:
                ts = jps_i
            if jpe_i > epy and te == epy:
                te = jpe_i
            j_start[nt - 1] = max(ts, jds_i)
            j_end[nt - 1] = min(te, jde_i)

            ntiles = t % num_tiles_x
            ts, te = region_bounds(spx, epx, num_tiles_x, ntiles)
            if ts <= te:
                if ips_i < spx and ts == spx:
                    ts = ips_i
                if ipe_i > epx and te == epx:
                    te = ipe_i
                i_start[nt - 1] = max(ts, ids_i)
                i_end[nt - 1] = min(te, ide_i)
                nt += 1

    actual_num_tiles = nt - 1
    _write(grid, "num_tiles", actual_num_tiles)
    _write(grid, "i_start", i_start)
    _write(grid, "i_end", i_end)
    _write(grid, "j_start", j_start)
    _write(grid, "j_end", j_end)

    return SetTiles2Result(
        num_tiles=actual_num_tiles,
        num_tiles_spec=int(num_tiles),
        num_tiles_x=int(num_tiles_x),
        num_tiles_y=int(num_tiles_y),
        max_tiles=int(max_tiles),
        i_start=tuple(int(value) for value in i_start[:actual_num_tiles]),
        i_end=tuple(int(value) for value in i_end[:actual_num_tiles]),
        j_start=tuple(int(value) for value in j_start[:actual_num_tiles]),
        j_end=tuple(int(value) for value in j_end[:actual_num_tiles]),
    )


def set_tiles_once(
    zone: int,
    grid: Any,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    ips: int,
    ipe: int,
    jps: int,
    jpe: int,
    *,
    config: Any | None = None,
    env: Mapping[str, str] | None = None,
    model_data_order: int = MODEL_DATA_ORDER,
    set_tiles2_fn: Callable[..., SetTiles2Result] = set_tiles2,
) -> SetTilesOnceResult:
    """Cache and restore WRF tile definitions for one tiling zone."""

    zone_i = int(zone)
    if zone_i < 1 or zone_i > MAX_TILING_ZONES:
        raise ValueError("set_tiles_once: zone out of range")

    latch = _ensure_latch(grid)
    zones = _ensure_tile_zones(grid)
    zone_index = zone_i - 1
    if not latch[zone_index]:
        latch[zone_index] = True
        _write(grid, "tiling_latch", latch)
        tiles = set_tiles2_fn(
            grid,
            ids,
            ide,
            jds,
            jde,
            ips,
            ipe,
            jps,
            jpe,
            config=config,
            env=env,
            model_data_order=model_data_order,
        )
        zone_record = TileZone(
            i_start=list(tiles.i_start),
            i_end=list(tiles.i_end),
            j_start=list(tiles.j_start),
            j_end=list(tiles.j_end),
            num_tiles=int(tiles.num_tiles),
            num_tiles_x=int(tiles.num_tiles_x),
            num_tiles_y=int(tiles.num_tiles_y),
        )
        zones[zone_index] = zone_record
        _write(grid, "tile_zones", zones)
        return SetTilesOnceResult(zone=zone_i, cached=False, tiles=tiles)

    zone_record = zones[zone_index]
    _write(grid, "i_start", list(zone_record.i_start))
    _write(grid, "i_end", list(zone_record.i_end))
    _write(grid, "j_start", list(zone_record.j_start))
    _write(grid, "j_end", list(zone_record.j_end))
    _write(grid, "num_tiles", int(zone_record.num_tiles))
    _write(grid, "num_tiles_x", int(zone_record.num_tiles_x))
    _write(grid, "num_tiles_y", int(zone_record.num_tiles_y))
    return SetTilesOnceResult(zone=zone_i, cached=True, tiles=_result_from_grid(grid))


def set_tiles3(
    grid: Any,
    imask: Any,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    ips: int,
    ipe: int,
    jps: int,
    jpe: int,
    *,
    set_tiles_masked_fn: Callable[..., TilesMaskedResult] = set_tiles_masked,
    wrf_debug: Callable[[int, str], None] | None = None,
) -> SetTiles3Result:
    """Compute masked WRF tiles and update ``grid`` like ``set_tiles3``."""

    masked = set_tiles_masked_fn(imask, ims, ime, jms, jme, ips, ipe, jps, jpe)
    num_tiles = int(masked.num_tiles)
    max_tiles = _allocate_grid_arrays(grid, num_tiles)
    i_start = _ensure_array(grid, "i_start", num_tiles)
    i_end = _ensure_array(grid, "i_end", num_tiles)
    j_start = _ensure_array(grid, "j_start", num_tiles)
    j_end = _ensure_array(grid, "j_end", num_tiles)

    for index in range(num_tiles):
        i_start[index] = int(masked.istarts[index])
        i_end[index] = int(masked.iends[index])
        j_start[index] = int(masked.jstarts[index])
        j_end[index] = int(masked.jends[index])

    _write(grid, "num_tiles", num_tiles)
    _write(grid, "i_start", i_start)
    _write(grid, "i_end", i_end)
    _write(grid, "j_start", j_start)
    _write(grid, "j_end", j_end)

    if wrf_debug is not None:
        for index in range(num_tiles):
            wrf_debug(
                1,
                (
                    f"WRF TILE {index + 1:3d} IS {i_start[index]:6d} "
                    f"IE {i_end[index]:6d} JS {j_start[index]:6d} JE {j_end[index]:6d}"
                ),
            )
        wrf_debug(1, f"set_tiles3: NUMBER OF TILES = {num_tiles:3d}")

    return SetTiles3Result(
        num_tiles=num_tiles,
        max_tiles=int(max_tiles),
        i_start=tuple(int(value) for value in i_start[:num_tiles]),
        i_end=tuple(int(value) for value in i_end[:num_tiles]),
        j_start=tuple(int(value) for value in j_start[:num_tiles]),
        j_end=tuple(int(value) for value in j_end[:num_tiles]),
    )


__all__ = [
    "SetTiles2Result",
    "SetTiles3Result",
    "SetTilesOnceResult",
    "TILE_NONE",
    "TILE_X",
    "TILE_XY",
    "TILE_Y",
    "TileConfig",
    "TileGrid",
    "TileZone",
    "MAX_TILING_ZONES",
    "set_tiles2",
    "set_tiles3",
    "set_tiles_once",
]
