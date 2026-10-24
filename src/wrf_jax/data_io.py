"""Outer-boundary NetCDF I/O for WRF-JAX case data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import xarray as xr

from wrf_jax.case_config import CaseConfig, load_case_config
from wrf_jax.forcing import BoundaryForcing
from wrf_jax.grid import WRFGrid
from wrf_jax.state import WRFState


STATE_VARIABLES = ("U", "V", "T", "QVAPOR")


@dataclass(frozen=True)
class CaseData:
    """Loaded runtime inputs for a single-domain WRF-JAX case."""

    config: CaseConfig
    grid: WRFGrid
    state: WRFState
    forcing: BoundaryForcing
    fields: dict[str, Any]
    attrs: dict[str, Any]
    input_path: Path
    boundary_path: Path
    namelist_path: Path


def _decode_times(values: Any) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim == 0:
        item = array.item()
        return (item.decode("ascii") if isinstance(item, bytes) else str(item),)
    if array.dtype.kind == "S" and array.ndim == 2:
        return tuple(b"".join(row).decode("ascii") for row in array)
    if array.dtype.kind == "S":
        return tuple(item.decode("ascii") for item in array.tolist())
    if array.dtype.kind == "O":
        decoded: list[str] = []
        for item in array.tolist():
            decoded.append(item.decode("ascii") if isinstance(item, bytes) else str(item))
        return tuple(decoded)
    return tuple(str(item) for item in array)


def _array(dataset: xr.Dataset, name: str) -> Any:
    if name not in dataset:
        raise KeyError(f"required WRF variable {name!r} is missing from {dataset.encoding.get('source', '<dataset>')}")
    values = dataset[name].values
    if values.ndim > 0 and dataset[name].dims[0] == "Time":
        values = values[0]
    return jnp.asarray(values)


def _to_wrf_memory_order(values: Any, dims: tuple[str, ...]) -> Any:
    """Convert xarray's WRF netCDF dimension order to translated kernel order."""

    arr = jnp.asarray(values)
    if dims in (
        ("soil_layers_stag", "south_north", "west_east"),
        ("soil_layers", "south_north", "west_east"),
    ):
        return jnp.transpose(arr, (2, 0, 1))
    if dims == ("bottom_top", "south_north", "west_east"):
        return jnp.transpose(arr, (2, 0, 1))
    if dims == ("bottom_top", "south_north", "west_east_stag"):
        return jnp.transpose(arr, (2, 0, 1))
    if dims == ("bottom_top", "south_north_stag", "west_east"):
        return jnp.transpose(arr, (2, 0, 1))
    if dims == ("bottom_top_stag", "south_north", "west_east"):
        return jnp.transpose(arr, (2, 0, 1))
    if dims == ("south_north", "west_east"):
        return jnp.transpose(arr, (1, 0))
    if dims == ("south_north", "west_east_stag"):
        return jnp.transpose(arr, (1, 0))
    if dims == ("south_north_stag", "west_east"):
        return jnp.transpose(arr, (1, 0))
    return arr


def _to_wrf_boundary_order(values: Any, dims: tuple[str, ...]) -> Any:
    """Convert ``wrfbdy`` boundary arrays to translated kernel order."""

    arr = jnp.asarray(values)
    if values.ndim > 0 and dims and dims[0] == "Time":
        arr = arr[0]
        dims = dims[1:]
    if len(dims) == 3 and dims[0] == "bdy_width" and dims[1] in {"bottom_top", "bottom_top_stag"}:
        return jnp.transpose(arr, (2, 1, 0))
    if len(dims) == 2 and dims[0] == "bdy_width":
        return jnp.expand_dims(jnp.transpose(arr, (1, 0)), axis=1)
    return arr


def _grid_from_dataset(dataset: xr.Dataset, config: CaseConfig) -> WRFGrid:
    sizes = dataset.sizes
    return WRFGrid(
        west_east=int(sizes["west_east"]),
        south_north=int(sizes["south_north"]),
        bottom_top=int(sizes["bottom_top"]),
        west_east_stag=int(sizes["west_east_stag"]),
        south_north_stag=int(sizes["south_north_stag"]),
        bottom_top_stag=int(sizes["bottom_top_stag"]),
        dx=float(dataset.attrs.get("DX", config.dx)),
        dy=float(dataset.attrs.get("DY", config.dy)),
        map_proj=int(dataset.attrs["MAP_PROJ"]) if "MAP_PROJ" in dataset.attrs else None,
    )


def load_wrfinput(path: str | Path, config: CaseConfig) -> tuple[WRFGrid, WRFState]:
    """Load the differentiable initial state from ``wrfinput_d01``."""

    with xr.open_dataset(path, engine="netcdf4") as dataset:
        grid = _grid_from_dataset(dataset, config)
        state = WRFState(
            u=_array(dataset, "U"),
            v=_array(dataset, "V"),
            t=_array(dataset, "T"),
            qvapor=_array(dataset, "QVAPOR"),
            time_index=0,
            time_seconds=0.0,
        )
    return grid, state


def load_wrfinput_fields(path: str | Path) -> dict[str, Any]:
    """Load numeric ``wrfinput`` variables into a lower-case field dictionary."""

    fields: dict[str, Any] = {}
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        variables = dict(dataset.data_vars)
        variables.update({name: variable for name, variable in dataset.coords.items() if name not in variables})
        for name, variable in variables.items():
            if name == "Times" or variable.dtype.kind not in ("f", "i", "u"):
                continue
            values = variable.values
            dims = tuple(str(dim) for dim in variable.dims)
            if values.ndim > 0 and variable.dims[0] == "Time":
                values = values[0]
                dims = dims[1:]
            fields[name.lower()] = _to_wrf_memory_order(values, dims)
    return fields


def load_wrfinput_attrs(path: str | Path) -> dict[str, Any]:
    """Load scalar global attributes from ``wrfinput``."""

    with xr.open_dataset(path, engine="netcdf4") as dataset:
        return {str(name): value.item() if hasattr(value, "item") else value for name, value in dataset.attrs.items()}


def load_wrfbdy(path: str | Path) -> BoundaryForcing:
    """Load lateral boundary forcing arrays from ``wrfbdy_d01``."""

    with xr.open_dataset(path, engine="netcdf4") as dataset:
        times = _decode_times(dataset["Times"].values) if "Times" in dataset else ()
        names: list[str] = []
        arrays: list[Any] = []
        for name, variable in dataset.data_vars.items():
            if name == "Times":
                continue
            if variable.dtype.kind in ("f", "i", "u"):
                names.append(name)
                arrays.append(jnp.asarray(variable.values))
    return BoundaryForcing(times=times, names=tuple(names), arrays=tuple(arrays))


def load_wrfbdy_fields(path: str | Path, *, time_index: int = 0) -> dict[str, Any]:
    """Load one ``wrfbdy`` time record into lower-case WRF-JAX names."""

    fields: dict[str, Any] = {}
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        for name, variable in dataset.data_vars.items():
            if name == "Times" or variable.dtype.kind not in ("f", "i", "u"):
                continue
            dims = tuple(str(dim) for dim in variable.dims)
            if dims and dims[0] == "Time":
                index = int(time_index)
                time_count = int(variable.sizes["Time"])
                if index < 0 or index >= time_count:
                    raise IndexError(f"wrfbdy time_index {index} out of range for {name}: 0..{time_count - 1}")
                variable = variable.isel(Time=index)
                dims = dims[1:]
            fields[name.lower()] = _to_wrf_boundary_order(variable.values, dims)
    return fields


def load_case(case_dir: str | Path) -> CaseData:
    """Load ``namelist.input``, ``wrfinput_d01``, and ``wrfbdy_d01`` from a WRF run directory."""

    root = Path(case_dir)
    namelist_path = root / "namelist.input"
    input_path = root / "wrfinput_d01"
    boundary_path = root / "wrfbdy_d01"
    config = load_case_config(namelist_path)
    grid, state = load_wrfinput(input_path, config)
    fields = load_wrfinput_fields(input_path)
    attrs = load_wrfinput_attrs(input_path)
    fields.update(load_wrfbdy_fields(boundary_path))
    forcing = load_wrfbdy(boundary_path)
    return CaseData(
        config=config,
        grid=grid,
        state=state,
        forcing=forcing,
        fields=fields,
        attrs=attrs,
        input_path=input_path,
        boundary_path=boundary_path,
        namelist_path=namelist_path,
    )


def assert_state_finite(state: WRFState) -> None:
    """Raise if any required prognostic field contains NaN or infinity."""

    for name, values in (("U", state.u), ("V", state.v), ("T", state.t), ("QVAPOR", state.qvapor)):
        finite = bool(jnp.all(jnp.isfinite(values)))
        if not finite:
            raise FloatingPointError(f"{name} contains NaN or infinity")


__all__ = [
    "CaseData",
    "STATE_VARIABLES",
    "assert_state_finite",
    "load_case",
    "load_wrfbdy",
    "load_wrfbdy_fields",
    "load_wrfinput_attrs",
    "load_wrfinput_fields",
    "load_wrfinput",
]
