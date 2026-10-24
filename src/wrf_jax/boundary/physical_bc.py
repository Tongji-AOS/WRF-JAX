"""Physical boundary-condition helpers from ``share/module_bc``."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


BDYZONE = 4


class _PhysicalBcFlags(NamedTuple):
    periodic_x: bool = False
    periodic_y: bool = False
    symmetric_xs: bool = False
    symmetric_xe: bool = False
    symmetric_ys: bool = False
    symmetric_ye: bool = False
    specified: bool = False
    nested: bool = False
    open_xs: bool = False
    open_xe: bool = False
    open_ys: bool = False
    open_ye: bool = False
    polar: bool = False


def _read(config: Any, name: str, default: Any = False) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _physical_bc_flags(config: Any) -> _PhysicalBcFlags:
    return _PhysicalBcFlags(
        periodic_x=bool(_read(config, "periodic_x", False)),
        periodic_y=bool(_read(config, "periodic_y", False)),
        symmetric_xs=bool(_read(config, "symmetric_xs", False)),
        symmetric_xe=bool(_read(config, "symmetric_xe", False)),
        symmetric_ys=bool(_read(config, "symmetric_ys", False)),
        symmetric_ye=bool(_read(config, "symmetric_ye", False)),
        specified=bool(_read(config, "specified", False)),
        nested=bool(_read(config, "nested", False)),
        open_xs=bool(_read(config, "open_xs", False)),
        open_xe=bool(_read(config, "open_xe", False)),
        open_ys=bool(_read(config, "open_ys", False)),
        open_ye=bool(_read(config, "open_ye", False)),
        polar=bool(_read(config, "polar", False)),
    )


def _specified_open_no_halo_fast_path(flags: _PhysicalBcFlags) -> bool:
    """Return True for the no-halo specified/open physical-BC shape."""

    return (
        (flags.specified or flags.nested or flags.open_xs or flags.open_xe or flags.open_ys or flags.open_ye)
        and not flags.periodic_x
        and not flags.periodic_y
        and not flags.symmetric_xs
        and not flags.symmetric_xe
        and not flags.symmetric_ys
        and not flags.symmetric_ye
        and not flags.open_xs
        and not flags.open_xe
        and not flags.open_ys
        and not flags.open_ye
        and not flags.polar
    )


def _has_no_horizontal_halo(ids: int, ide: int, jds: int, jde: int, ims: int, ime: int, jms: int, jme: int) -> bool:
    return int(ids) == int(ims) and int(ide) == int(ime) and int(jds) == int(jms) and int(jde) == int(jme)


def _set_physical_bc3d_no_halo_specified(
    dat: Any,
    variable_in: str,
    flags: _PhysicalBcFlags,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    _kds: int,
    kde: int,
    ims: int,
    _ime: int,
    jms: int,
    _jme: int,
    kms: int,
    _kme: int,
    _ips: int,
    _ipe: int,
    _jps: int,
    _jpe: int,
    _kps: int,
    _kpe: int,
    _its: int,
    ite: int,
    _jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> Any:
    out = jnp.asarray(dat)
    variable, _istag, _jstag, full_k_end, open_bc_copy, x_stagger = _variable_flags(variable_in)
    if not open_bc_copy:
        return out
    k_end = max(1, min(int(kde) if full_k_end else int(kde) - 1, int(kte)))
    z_k = _inclusive_slice(kts, k_end, kms)
    if z_k is None:
        return out
    open_xe = flags.specified or flags.nested or flags.open_xe
    if open_xe and int(ite) == int(ide) and not x_stagger:
        dst_i = _idx(int(ide), ims)
        src_i = _idx(int(ide) - 1, ims)
        if 0 <= dst_i < out.shape[0] and 0 <= src_i < out.shape[0]:
            out = _update_x_plane3d(out, dst_i, src_i, z_k)
    open_ye = flags.specified or flags.nested or flags.open_ye
    if open_ye and int(jte) == int(jde) and variable not in {"v", "y"}:
        dst_j = _idx(int(jde), jms)
        src_j = _idx(int(jde) - 1, jms)
        if 0 <= dst_j < out.shape[2] and 0 <= src_j < out.shape[2]:
            out = _update_y_plane3d(out, dst_j, src_j, z_k)
    return out


def _set_physical_bc2d_no_halo_specified(
    dat: Any,
    variable_in: str,
    flags: _PhysicalBcFlags,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    ims: int,
    _ime: int,
    jms: int,
    _jme: int,
    _ips: int,
    _ipe: int,
    _jps: int,
    _jpe: int,
    _its: int,
    ite: int,
    _jts: int,
    jte: int,
) -> Any:
    out = jnp.asarray(dat)
    variable, _istag, _jstag, open_bc_copy, x_stagger = _variable_flags_2d(variable_in)
    if not open_bc_copy:
        return out
    open_xe = flags.specified or flags.nested or flags.open_xe
    if open_xe and int(ite) == int(ide) and not x_stagger:
        dst_i = _idx(int(ide), ims)
        src_i = _idx(int(ide) - 1, ims)
        if 0 <= dst_i < out.shape[0] and 0 <= src_i < out.shape[0]:
            out = _update_x_plane2d(out, dst_i, src_i)
    open_ye = flags.specified or flags.nested or flags.open_ye
    if open_ye and int(jte) == int(jde) and variable not in {"v", "y"}:
        dst_j = _idx(int(jde), jms)
        src_j = _idx(int(jde) - 1, jms)
        if 0 <= dst_j < out.shape[1] and 0 <= src_j < out.shape[1]:
            out = _update_y_plane2d(out, dst_j, src_j)
    return out


def _variable_flags(variable_in: str) -> tuple[str, int, int, bool, bool, bool]:
    variable = str(variable_in)[0].lower()
    open_bc_copy = variable in {"u", "v", "w", "t", "d", "e", "x", "y", "f", "r", "p"}
    istag = -1
    jstag = -1
    full_k_end = False
    if variable in {"u", "x"}:
        istag = 0
    if variable in {"v", "y"}:
        jstag = 0
    if variable == "d":
        istag = 0
        jstag = 0
    if variable == "e":
        istag = 0
        full_k_end = True
    if variable == "f":
        jstag = 0
        full_k_end = True
    if variable == "w":
        full_k_end = True
    return variable, istag, jstag, full_k_end, open_bc_copy, variable in {"u", "x"}


def _variable_flags_2d(variable_in: str) -> tuple[str, int, int, bool, bool]:
    variable = str(variable_in)[0].lower()
    open_bc_copy = variable in {"u", "v", "w", "t", "x", "y", "r", "p"}
    istag = -1
    jstag = -1
    if variable in {"u", "x"}:
        istag = 0
    if variable in {"v", "y"}:
        jstag = 0
    if variable == "d":
        istag = 0
        jstag = 0
    if variable == "e":
        istag = 0
    if variable == "f":
        jstag = 0
    return variable, istag, jstag, open_bc_copy, variable in {"u", "x"}


def _idx(index: int, lower_bound: int) -> int:
    return int(index) - int(lower_bound)


def _inclusive_slice(start: int, end: int, lower_bound: int) -> slice | None:
    start_i = int(start)
    end_i = int(end)
    if end_i < start_i:
        return None
    return slice(_idx(start_i, lower_bound), _idx(end_i, lower_bound) + 1)


def _update_x_plane3d(out: Any, dst_i: int, src_i: int, z_slice: slice) -> Any:
    z_start = int(z_slice.start or 0)
    z_stop = out.shape[1] if z_slice.stop is None else int(z_slice.stop)
    if z_start == 0 and z_stop == out.shape[1]:
        return jnp.concatenate(
            [
                out[: int(dst_i), :, :],
                out[int(src_i) : int(src_i) + 1, :, :],
                out[int(dst_i) + 1 :, :, :],
            ],
            axis=0,
        )
    return jax.lax.dynamic_update_slice(
        out,
        out[int(src_i) : int(src_i) + 1, z_slice, :],
        (int(dst_i), z_start, 0),
    )


def _update_y_plane3d(out: Any, dst_j: int, src_j: int, z_slice: slice) -> Any:
    z_start = int(z_slice.start or 0)
    z_stop = out.shape[1] if z_slice.stop is None else int(z_slice.stop)
    if z_start == 0 and z_stop == out.shape[1]:
        return jnp.concatenate(
            [
                out[:, :, : int(dst_j)],
                out[:, :, int(src_j) : int(src_j) + 1],
                out[:, :, int(dst_j) + 1 :],
            ],
            axis=2,
        )
    return jax.lax.dynamic_update_slice(
        out,
        out[:, z_slice, int(src_j) : int(src_j) + 1],
        (0, z_start, int(dst_j)),
    )


def _update_x_plane2d(out: Any, dst_i: int, src_i: int) -> Any:
    return jnp.concatenate([out[: int(dst_i), :], out[int(src_i) : int(src_i) + 1, :], out[int(dst_i) + 1 :, :]], axis=0)


def _update_y_plane2d(out: Any, dst_j: int, src_j: int) -> Any:
    return jnp.concatenate([out[:, : int(dst_j)], out[:, int(src_j) : int(src_j) + 1], out[:, int(dst_j) + 1 :]], axis=1)


def _set_plane(dat: Any, index: tuple[Any, Any, Any], value: Any) -> Any:
    return dat.at[index].set(value)


def _pad_xy3d(dat: Any, ims: int, ime: int, jms: int, jme: int) -> tuple[Any, int, int, int, int]:
    padded = jnp.pad(jnp.asarray(dat), ((BDYZONE, BDYZONE), (0, 0), (BDYZONE, BDYZONE)), mode="edge")
    return padded, int(ims) - BDYZONE, int(ime) + BDYZONE, int(jms) - BDYZONE, int(jme) + BDYZONE


def _unpad_xy3d(dat: Any) -> Any:
    return dat[BDYZONE:-BDYZONE, :, BDYZONE:-BDYZONE]


def _pad_xy2d(dat: Any, ims: int, ime: int, jms: int, jme: int) -> tuple[Any, int, int, int, int]:
    padded = jnp.pad(jnp.asarray(dat), ((BDYZONE, BDYZONE), (BDYZONE, BDYZONE)), mode="edge")
    return padded, int(ims) - BDYZONE, int(ime) + BDYZONE, int(jms) - BDYZONE, int(jme) + BDYZONE


def _unpad_xy2d(dat: Any) -> Any:
    return dat[BDYZONE:-BDYZONE, BDYZONE:-BDYZONE]


@partial(jax.jit, static_argnums=tuple(range(1, 27)))
def _set_physical_bc3d_impl(
    dat: Any,
    variable_in: str,
    config_flags: _PhysicalBcFlags,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    kms: int,
    kme: int,
    ips: int,
    ipe: int,
    jps: int,
    jpe: int,
    kps: int,
    kpe: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> Any:
    """Apply the configured 3-D physical boundary copy operations."""

    del kds, kps, kpe, kme
    out, ims, ime, jms, jme = _pad_xy3d(dat, ims, ime, jms, jme)
    variable, istag, jstag, full_k_end, open_bc_copy, x_stagger = _variable_flags(variable_in)
    k_end = max(1, min(int(kde) if full_k_end else int(kde) - 1, int(kte)))

    i = lambda value: _idx(value, ims)
    k = lambda value: _idx(value, kms)
    j = lambda value: _idx(value, jms)

    x_j = _inclusive_slice(max(jds, int(jts) - 1), min(int(jte) + 1, int(jde) + jstag), jms)
    z_k = _inclusive_slice(kts, k_end, kms)

    if bool(_read(config_flags, "periodic_x", False)):
        if int(ids) == int(ips) and int(ide) == int(ipe):
            if int(its) == int(ids) and x_j is not None and z_k is not None:
                for offset in range(0, -BDYZONE, -1):
                    out = _set_plane(out, (i(int(ids) + offset - 1), z_k, x_j), out[i(int(ide) + offset - 1), z_k, x_j])

            if int(ite) == int(ide) and x_j is not None and z_k is not None:
                for offset in range(-istag, BDYZONE + 1):
                    out = _set_plane(
                        out,
                        (i(int(ide) + offset + istag), z_k, x_j),
                        out[i(int(ids) + offset + istag), z_k, x_j],
                    )
    else:
        if bool(_read(config_flags, "symmetric_xs", False)) and int(its) == int(ids) and x_j is not None and z_k is not None:
            if istag == -1:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (i(int(ids) - offset), z_k, x_j), out[i(int(ids) + offset - 1), z_k, x_j])
            elif variable == "u":
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (i(int(ids) - offset), z_k, x_j), -out[i(int(ids) + offset), z_k, x_j])
            else:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (i(int(ids) - offset), z_k, x_j), out[i(int(ids) + offset), z_k, x_j])

        if bool(_read(config_flags, "symmetric_xe", False)) and int(ite) == int(ide) and x_j is not None and z_k is not None:
            if istag == -1:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (i(int(ide) + offset - 1), z_k, x_j), out[i(int(ide) - offset), z_k, x_j])
            elif variable == "u":
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (i(int(ide) + offset), z_k, x_j), -out[i(int(ide) - offset), z_k, x_j])
            else:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (i(int(ide) + offset), z_k, x_j), out[i(int(ide) - offset), z_k, x_j])

        x_open = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))
        open_xs = x_open or bool(_read(config_flags, "open_xs", False))
        open_xe = x_open or bool(_read(config_flags, "open_xe", False))
        if open_xs and int(its) == int(ids) and open_bc_copy and z_k is not None:
            open_j = _inclusive_slice(int(jts) - BDYZONE, min(int(jte), int(jde) + jstag) + BDYZONE, jms)
            if open_j is not None:
                for offset in range(1, 4):
                    out = _set_plane(out, (i(int(ids) - offset), z_k, open_j), out[i(ids), z_k, open_j])

        if open_xe and int(ite) == int(ide) and open_bc_copy and z_k is not None:
            if not x_stagger:
                open_j = _inclusive_slice(int(jts) - BDYZONE, min(int(jte), int(jde) + jstag) + BDYZONE, jms)
                if open_j is not None:
                    for offset in range(0, 3):
                        out = _set_plane(out, (i(int(ide) + offset), z_k, open_j), out[i(int(ide) - 1), z_k, open_j])
            else:
                open_j = _inclusive_slice(
                    max(int(jds), int(jts) - 1) - BDYZONE,
                    min(int(jte) + 1, int(jde) + jstag) + BDYZONE,
                    jms,
                )
                if open_j is not None:
                    for offset in range(1, 4):
                        out = _set_plane(out, (i(int(ide) + offset), z_k, open_j), out[i(ide), z_k, open_j])

    i_start = max(int(ids), int(its) - 1)
    i_end = min(int(ite) + 1, int(ide) + istag)
    if int(its) == int(ids):
        i_start = int(ims)
    if int(ite) == int(ide):
        i_end = int(ime)
    y_i = _inclusive_slice(i_start, i_end, ims)

    if bool(_read(config_flags, "periodic_y", False)):
        if int(jds) == int(jps) and int(jde) == int(jpe) and y_i is not None and z_k is not None:
            if int(jts) == int(jds):
                for offset in range(0, -BDYZONE, -1):
                    out = _set_plane(out, (y_i, z_k, j(int(jds) + offset - 1)), out[y_i, z_k, j(int(jde) + offset - 1)])

            if int(jte) == int(jde):
                for offset in range(-jstag, BDYZONE + 1):
                    out = _set_plane(
                        out,
                        (y_i, z_k, j(int(jde) + offset + jstag)),
                        out[y_i, z_k, j(int(jds) + offset + jstag)],
                    )
    else:
        if bool(_read(config_flags, "symmetric_ys", False)) and int(jts) == int(jds) and y_i is not None and z_k is not None:
            if jstag == -1:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (y_i, z_k, j(int(jds) - offset)), out[y_i, z_k, j(int(jds) + offset - 1)])
            elif variable == "v":
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (y_i, z_k, j(int(jds) - offset)), -out[y_i, z_k, j(int(jds) + offset)])
            else:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (y_i, z_k, j(int(jds) - offset)), out[y_i, z_k, j(int(jds) + offset)])

        if bool(_read(config_flags, "symmetric_ye", False)) and int(jte) == int(jde) and y_i is not None and z_k is not None:
            if jstag == -1:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (y_i, z_k, j(int(jde) + offset - 1)), out[y_i, z_k, j(int(jde) - offset)])
            elif variable == "v":
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (y_i, z_k, j(int(jde) + offset)), -out[y_i, z_k, j(int(jde) - offset)])
            else:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (y_i, z_k, j(int(jde) + offset)), out[y_i, z_k, j(int(jde) - offset)])

        y_open = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))
        open_ys = y_open or bool(_read(config_flags, "polar", False)) or bool(_read(config_flags, "open_ys", False))
        open_ye = y_open or bool(_read(config_flags, "polar", False)) or bool(_read(config_flags, "open_ye", False))
        if open_ys and int(jts) == int(jds) and open_bc_copy and y_i is not None and z_k is not None:
            for offset in range(1, 4):
                out = _set_plane(out, (y_i, z_k, j(int(jds) - offset)), out[y_i, z_k, j(jds)])

        if open_ye and int(jte) == int(jde) and open_bc_copy and y_i is not None and z_k is not None:
            if variable not in {"v", "y"}:
                for offset in range(0, 3):
                    out = _set_plane(out, (y_i, z_k, j(int(jde) + offset)), out[y_i, z_k, j(int(jde) - 1)])
            else:
                for offset in range(1, 4):
                    out = _set_plane(out, (y_i, z_k, j(int(jde) + offset)), out[y_i, z_k, j(jde)])

    return _unpad_xy3d(out)


def set_physical_bc3d(
    dat: Any,
    variable_in: str,
    config_flags: Any,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    kms: int,
    kme: int,
    ips: int,
    ipe: int,
    jps: int,
    jpe: int,
    kps: int,
    kpe: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> Any:
    """Apply the configured 3-D physical boundary copy operations."""

    flags = _physical_bc_flags(config_flags)
    if _specified_open_no_halo_fast_path(flags) and _has_no_horizontal_halo(ids, ide, jds, jde, ims, ime, jms, jme):
        return _set_physical_bc3d_no_halo_specified(
            dat,
            variable_in,
            flags,
            ids,
            ide,
            jds,
            jde,
            kds,
            kde,
            ims,
            ime,
            jms,
            jme,
            kms,
            kme,
            ips,
            ipe,
            jps,
            jpe,
            kps,
            kpe,
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
        )
    return _set_physical_bc3d_impl(
        dat,
        variable_in,
        flags,
        ids,
        ide,
        jds,
        jde,
        kds,
        kde,
        ims,
        ime,
        jms,
        jme,
        kms,
        kme,
        ips,
        ipe,
        jps,
        jpe,
        kps,
        kpe,
        its,
        ite,
        jts,
        jte,
        kts,
        kte,
    )


@partial(jax.jit, static_argnums=tuple(range(1, 19)))
def _set_physical_bc2d_impl(
    dat: Any,
    variable_in: str,
    config_flags: _PhysicalBcFlags,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    ips: int,
    ipe: int,
    jps: int,
    jpe: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
) -> Any:
    """Apply the configured 2-D physical boundary copy operations."""

    out, ims, ime, jms, jme = _pad_xy2d(dat, ims, ime, jms, jme)
    variable, istag, jstag, open_bc_copy, x_stagger = _variable_flags_2d(variable_in)

    i = lambda value: _idx(value, ims)
    j = lambda value: _idx(value, jms)

    x_j = _inclusive_slice(max(jds, int(jts) - 1), min(int(jte) + 1, int(jde) + jstag), jms)
    if bool(_read(config_flags, "periodic_x", False)):
        if int(ids) == int(ips) and int(ide) == int(ipe) and x_j is not None:
            if int(its) == int(ids):
                for offset in range(0, -BDYZONE, -1):
                    out = _set_plane(out, (i(int(ids) + offset - 1), x_j), out[i(int(ide) + offset - 1), x_j])

            if int(ite) == int(ide):
                for offset in range(-istag, BDYZONE + 1):
                    out = _set_plane(
                        out,
                        (i(int(ide) + offset + istag), x_j),
                        out[i(int(ids) + offset + istag), x_j],
                    )
    else:
        if bool(_read(config_flags, "symmetric_xs", False)) and int(its) == int(ids) and x_j is not None:
            if not x_stagger:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (i(int(ids) - offset), x_j), out[i(int(ids) + offset - 1), x_j])
            elif variable == "u":
                for offset in range(0, BDYZONE):
                    out = _set_plane(out, (i(int(ids) - offset), x_j), -out[i(int(ids) + offset), x_j])
            else:
                for offset in range(0, BDYZONE):
                    out = _set_plane(out, (i(int(ids) - offset), x_j), out[i(int(ids) + offset), x_j])

        if bool(_read(config_flags, "symmetric_xe", False)) and int(ite) == int(ide) and x_j is not None:
            if not x_stagger:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (i(int(ide) + offset - 1), x_j), out[i(int(ide) - offset), x_j])
            elif variable == "u":
                for offset in range(0, BDYZONE):
                    out = _set_plane(out, (i(int(ide) + offset), x_j), -out[i(int(ide) - offset), x_j])
            else:
                for offset in range(0, BDYZONE):
                    out = _set_plane(out, (i(int(ide) + offset), x_j), out[i(int(ide) - offset), x_j])

        x_open = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))
        open_xs = x_open or bool(_read(config_flags, "open_xs", False))
        open_xe = x_open or bool(_read(config_flags, "open_xe", False))
        if open_xs and int(its) == int(ids) and open_bc_copy and x_j is not None:
            for offset in range(1, 4):
                out = _set_plane(out, (i(int(ids) - offset), x_j), out[i(ids), x_j])

        if open_xe and int(ite) == int(ide) and open_bc_copy and x_j is not None:
            if not x_stagger:
                for offset in range(0, 3):
                    out = _set_plane(out, (i(int(ide) + offset), x_j), out[i(int(ide) - 1), x_j])
            else:
                for offset in range(1, 4):
                    out = _set_plane(out, (i(int(ide) + offset), x_j), out[i(ide), x_j])

    istart = max(int(ids), int(its) - 1)
    iend = min(int(ite) + 1, int(ide) + istag)
    if int(its) == int(ids):
        istart = int(ims)
    if int(ite) == int(ide):
        iend = int(ime)
    y_i = _inclusive_slice(istart, iend, ims)

    if bool(_read(config_flags, "periodic_y", False)):
        if int(jds) == int(jps) and int(jde) == int(jpe) and y_i is not None:
            if int(jts) == int(jds):
                for offset in range(0, -BDYZONE, -1):
                    out = _set_plane(out, (y_i, j(int(jds) + offset - 1)), out[y_i, j(int(jde) + offset - 1)])

            if int(jte) == int(jde):
                for offset in range(-jstag, BDYZONE + 1):
                    out = _set_plane(
                        out,
                        (y_i, j(int(jde) + offset + jstag)),
                        out[y_i, j(int(jds) + offset + jstag)],
                    )
    else:
        if bool(_read(config_flags, "symmetric_ys", False)) and int(jts) == int(jds) and y_i is not None:
            if variable not in {"v", "y"}:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (y_i, j(int(jds) - offset)), out[y_i, j(int(jds) + offset - 1)])
            elif variable == "v":
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (y_i, j(int(jds) - offset)), -out[y_i, j(int(jds) + offset)])
            else:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (y_i, j(int(jds) - offset)), out[y_i, j(int(jds) + offset)])

        if bool(_read(config_flags, "symmetric_ye", False)) and int(jte) == int(jde) and y_i is not None:
            if variable not in {"v", "y"}:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (y_i, j(int(jde) + offset - 1)), out[y_i, j(int(jde) - offset)])
            elif variable == "v":
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (y_i, j(int(jde) + offset)), -out[y_i, j(int(jde) - offset)])
            else:
                for offset in range(1, BDYZONE + 1):
                    out = _set_plane(out, (y_i, j(int(jde) + offset)), out[y_i, j(int(jde) - offset)])

        y_open = bool(_read(config_flags, "specified", False)) or bool(_read(config_flags, "nested", False))
        open_ys = y_open or bool(_read(config_flags, "polar", False)) or bool(_read(config_flags, "open_ys", False))
        open_ye = y_open or bool(_read(config_flags, "polar", False)) or bool(_read(config_flags, "open_ye", False))
        if open_ys and int(jts) == int(jds) and open_bc_copy and y_i is not None:
            for offset in range(1, 4):
                out = _set_plane(out, (y_i, j(int(jds) - offset)), out[y_i, j(jds)])

        if open_ye and int(jte) == int(jde) and open_bc_copy and y_i is not None:
            if variable not in {"v", "y"}:
                for offset in range(0, 3):
                    out = _set_plane(out, (y_i, j(int(jde) + offset)), out[y_i, j(int(jde) - 1)])
            else:
                for offset in range(1, 4):
                    out = _set_plane(out, (y_i, j(int(jde) + offset)), out[y_i, j(jde)])

    if (
        bool(_read(config_flags, "periodic_x", False))
        and bool(_read(config_flags, "periodic_y", False))
        and int(ids) == int(ips)
        and int(ide) == int(ipe)
        and int(jds) == int(jps)
        and int(jde) == int(jpe)
    ):
        if int(its) == int(ids) and int(jts) == int(jds):
            for j_offset in range(0, -BDYZONE, -1):
                for i_offset in range(0, -BDYZONE, -1):
                    out = _set_plane(
                        out,
                        (i(int(ids) + i_offset - 1), j(int(jds) + j_offset - 1)),
                        out[i(int(ide) + i_offset - 1), j(int(jde) + j_offset - 1)],
                    )

        if int(ite) == int(ide) and int(jts) == int(jds):
            for j_offset in range(0, -BDYZONE, -1):
                for i_offset in range(1, BDYZONE + 1):
                    out = _set_plane(
                        out,
                        (i(int(ide) + i_offset + istag), j(int(jds) + j_offset - 1)),
                        out[i(int(ids) + i_offset + istag), j(int(jde) + j_offset - 1)],
                    )

        if int(ite) == int(ide) and int(jte) == int(jde):
            for j_offset in range(1, BDYZONE + 1):
                for i_offset in range(1, BDYZONE + 1):
                    out = _set_plane(
                        out,
                        (i(int(ide) + i_offset + istag), j(int(jde) + j_offset + jstag)),
                        out[i(int(ids) + i_offset + istag), j(int(jds) + j_offset + jstag)],
                    )

        if int(its) == int(ids) and int(jte) == int(jde):
            for j_offset in range(1, BDYZONE + 1):
                for i_offset in range(0, -BDYZONE, -1):
                    out = _set_plane(
                        out,
                        (i(int(ids) + i_offset - 1), j(int(jde) + j_offset + jstag)),
                        out[i(int(ide) + i_offset - 1), j(int(jds) + j_offset + jstag)],
                    )

    return _unpad_xy2d(out)


def set_physical_bc2d(
    dat: Any,
    variable_in: str,
    config_flags: Any,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    ims: int,
    ime: int,
    jms: int,
    jme: int,
    ips: int,
    ipe: int,
    jps: int,
    jpe: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
) -> Any:
    """Apply the configured 2-D physical boundary copy operations."""

    flags = _physical_bc_flags(config_flags)
    if _specified_open_no_halo_fast_path(flags) and _has_no_horizontal_halo(ids, ide, jds, jde, ims, ime, jms, jme):
        return _set_physical_bc2d_no_halo_specified(
            dat,
            variable_in,
            flags,
            ids,
            ide,
            jds,
            jde,
            ims,
            ime,
            jms,
            jme,
            ips,
            ipe,
            jps,
            jpe,
            its,
            ite,
            jts,
            jte,
        )
    return _set_physical_bc2d_impl(
        dat,
        variable_in,
        flags,
        ids,
        ide,
        jds,
        jde,
        ims,
        ime,
        jms,
        jme,
        ips,
        ipe,
        jps,
        jpe,
        its,
        ite,
        jts,
        jte,
    )


__all__ = ["BDYZONE", "set_physical_bc2d", "set_physical_bc3d"]
