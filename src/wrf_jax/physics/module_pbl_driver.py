"""Selected translations from WRF ``phys/module_pbl_driver``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp


DEFAULT_DIFF4D_EXCLUDED_SCALARS = (1, 1, 1, 1, 1, 1, 1)
PBL_PHYSICS_NONE = 0
MYJPBLSCHEME = 2
FITCHSCHEME = 1

_EXPLICIT_RESULT_FIELDS = frozenset(
    {
        "u_phytmp",
        "v_phytmp",
        "tskold",
        "ustold",
        "zntold",
        "hpbl_hold",
        "psfc",
        "rublten",
        "rvblten",
        "rthblten",
        "rqvblten",
        "rqcblten",
        "rqiblten",
        "pblh",
        "kpbl",
        "bldtacttime",
    }
)


@dataclass(frozen=True)
class Diff4dResult:
    scalar_tend: Any


@dataclass(frozen=True)
class PblDriverResult:
    u_phytmp: Any | None = None
    v_phytmp: Any | None = None
    tskold: Any | None = None
    ustold: Any | None = None
    zntold: Any | None = None
    hpbl_hold: Any | None = None
    psfc: Any | None = None
    rublten: Any | None = None
    rvblten: Any | None = None
    rthblten: Any | None = None
    rqvblten: Any | None = None
    rqcblten: Any | None = None
    rqiblten: Any | None = None
    pblh: Any | None = None
    kpbl: Any | None = None
    dtmin: float = 0.0
    dtbl: float = 0.0
    bldtacttime: float | None = None
    carried_fields: Mapping[str, Any] = field(default_factory=dict)
    ran_parameterization: bool = False
    external_calls: tuple[str, ...] = ()


def _idx(wrf_index: int, kms: int) -> int:
    return int(wrf_index) - int(kms)


def _tile_slices(ims: int, jms: int, its: int, ite: int, jts: int, jte: int) -> tuple[slice, slice]:
    return (
        slice(_idx(its, ims), _idx(ite, ims) + 1),
        slice(_idx(jts, jms), _idx(jte, jms) + 1),
    )


def _tile_3d_slices(
    ims: int,
    kms: int,
    jms: int,
    its: int,
    ite: int,
    kts: int,
    kte: int,
    jts: int,
    jte: int,
) -> tuple[slice, slice, slice]:
    return (
        slice(_idx(its, ims), _idx(ite, ims) + 1),
        slice(_idx(kts, kms), _idx(kte, kms) + 1),
        slice(_idx(jts, jms), _idx(jte, jms) + 1),
    )


def _as_array_or_none(value: Any) -> Any | None:
    return None if value is None else jnp.asarray(value)


def _mapping_from_result(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "__dataclass_fields__"):
        return {name: getattr(result, name) for name in result.__dataclass_fields__}
    raise TypeError("pbl_driver callback must return a mapping, dataclass result, or None")


def _carried_without_explicit_results(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in fields.items() if name not in _EXPLICIT_RESULT_FIELDS}


def _pbl_should_run(
    *,
    itimestep: int,
    stepbl: int,
    bldt: float | None,
    curr_secs: float | None,
    adapt_step_flag: bool | None,
    bldtacttime: float | None,
) -> tuple[bool, float | None]:
    doing_adapt_dt = bool(adapt_step_flag)
    bldtacttime_out = bldtacttime
    if doing_adapt_dt and bldt is not None and curr_secs is not None and bldtacttime_out == 0.0:
        bldtacttime_out = float(curr_secs) + float(bldt) * 60.0
    if int(itimestep) == 1:
        return True, bldtacttime_out
    if bldt is not None:
        if float(bldt) == 0.0 or int(stepbl) == 1:
            return True, bldtacttime_out
    elif int(stepbl) == 1:
        return True, bldtacttime_out
    if (not doing_adapt_dt) and int(itimestep) % int(stepbl) == 0:
        return True, bldtacttime_out
    if doing_adapt_dt and curr_secs is not None and bldtacttime_out is not None and float(curr_secs) >= float(bldtacttime_out):
        return True, float(curr_secs) + (0.0 if bldt is None else float(bldt) * 60.0)
    return False, bldtacttime_out


def pbl_driver(
    *,
    bl_pbl_physics: int,
    itimestep: int = 1,
    dt: float = 0.0,
    u_frame: float = 0.0,
    v_frame: float = 0.0,
    bldt: float | None = None,
    curr_secs: float | None = None,
    adapt_step_flag: bool | None = None,
    bldtacttime: float | None = None,
    stepbl: int = 1,
    ra_lw_physics: int = 0,
    sf_sfclay_physics: int = 0,
    sf_urban_physics: int = 0,
    windfarm_opt: int = 0,
    gwd_opt: int | None = None,
    tsk: Any | None = None,
    znt: Any | None = None,
    ust: Any | None = None,
    pblh: Any | None = None,
    hpbl_hold: Any | None = None,
    u_phy: Any | None = None,
    v_phy: Any | None = None,
    u_phytmp: Any | None = None,
    v_phytmp: Any | None = None,
    p8w: Any | None = None,
    psfc: Any | None = None,
    rublten: Any | None = None,
    rvblten: Any | None = None,
    rthblten: Any | None = None,
    rqvblten: Any | None = None,
    rqcblten: Any | None = None,
    rqiblten: Any | None = None,
    qv_curr: Any | None = None,
    qc_curr: Any | None = None,
    qi_curr: Any | None = None,
    f_qi: bool | None = None,
    kpbl: Any | None = None,
    ids: int = 1,
    ide: int = 1,
    jds: int = 1,
    jde: int = 1,
    kds: int = 1,
    kde: int = 1,
    ims: int = 1,
    ime: int = 1,
    jms: int = 1,
    jme: int = 1,
    kms: int = 1,
    kme: int = 1,
    i_start: Any = (1,),
    i_end: Any = (1,),
    j_start: Any = (1,),
    j_end: Any = (1,),
    kts: int = 1,
    kte: int = 1,
    myjpbl_fn: Any | None = None,
    gwd_fn: Any | None = None,
    **fields: Any,
) -> PblDriverResult:
    """Run the supported WRF ``pbl_driver`` control path.

    The configured case uses MYJ PBL. This translation keeps the driver state
    updates explicit and delegates scheme kernels to explicit callbacks or lower
    translated functions.
    """

    del ra_lw_physics, sf_sfclay_physics, sf_urban_physics, ids, ide, jds, jde, kds
    carried = dict(fields)

    if int(bl_pbl_physics) == PBL_PHYSICS_NONE:
        return PblDriverResult(
            u_phytmp=u_phytmp,
            v_phytmp=v_phytmp,
            hpbl_hold=hpbl_hold,
            psfc=psfc,
            rublten=rublten,
            rvblten=rvblten,
            rthblten=rthblten,
            rqvblten=rqvblten,
            rqcblten=rqcblten,
            rqiblten=rqiblten,
            pblh=pblh,
            kpbl=kpbl,
            bldtacttime=bldtacttime,
            carried_fields=_carried_without_explicit_results(carried),
            ran_parameterization=False,
        )

    run_param, bldtacttime_out = _pbl_should_run(
        itimestep=itimestep,
        stepbl=stepbl,
        bldt=bldt,
        curr_secs=curr_secs,
        adapt_step_flag=adapt_step_flag,
        bldtacttime=bldtacttime,
    )
    if not run_param:
        return PblDriverResult(
            u_phytmp=u_phytmp,
            v_phytmp=v_phytmp,
            hpbl_hold=hpbl_hold,
            psfc=psfc,
            rublten=rublten,
            rvblten=rvblten,
            rthblten=rthblten,
            rqvblten=rqvblten,
            rqcblten=rqcblten,
            rqiblten=rqiblten,
            pblh=pblh,
            kpbl=kpbl,
            bldtacttime=bldtacttime_out,
            carried_fields=_carried_without_explicit_results(carried),
            ran_parameterization=False,
        )

    dtmin = float(dt) / 60.0
    if bldt is not None and float(bldt) == 0.0:
        dtbl = float(dt)
    elif bldt is not None and bool(adapt_step_flag):
        dtbl = float(bldt) * 60.0
    else:
        dtbl = float(dt) * float(stepbl)

    i_start_tuple = tuple(int(value) for value in i_start)
    i_end_tuple = tuple(int(value) for value in i_end)
    j_start_tuple = tuple(int(value) for value in j_start)
    j_end_tuple = tuple(int(value) for value in j_end)

    u_phy_arr = _as_array_or_none(u_phy)
    v_phy_arr = _as_array_or_none(v_phy)
    p8w_arr = _as_array_or_none(p8w)
    tsk_arr = _as_array_or_none(tsk)
    ust_arr = _as_array_or_none(ust)
    znt_arr = _as_array_or_none(znt)
    pblh_arr = _as_array_or_none(pblh)

    u_phytmp_out = jnp.asarray(u_phytmp) if u_phytmp is not None else (jnp.zeros_like(u_phy_arr) if u_phy_arr is not None else None)
    v_phytmp_out = jnp.asarray(v_phytmp) if v_phytmp is not None else (jnp.zeros_like(v_phy_arr) if v_phy_arr is not None else None)
    psfc_out = _as_array_or_none(psfc)
    tskold_out = jnp.zeros_like(tsk_arr) if tsk_arr is not None else None
    ustold_out = jnp.zeros_like(ust_arr) if ust_arr is not None else None
    zntold_out = jnp.zeros_like(znt_arr) if znt_arr is not None else None
    hpbl_hold_out = jnp.asarray(hpbl_hold) if hpbl_hold is not None else (jnp.zeros_like(pblh_arr) if pblh_arr is not None else None)

    rublten_out = jnp.asarray(rublten) if rublten is not None else None
    rvblten_out = jnp.asarray(rvblten) if rvblten is not None else None
    rthblten_out = jnp.asarray(rthblten) if rthblten is not None else None
    rqvblten_out = jnp.asarray(rqvblten) if rqvblten is not None else None
    rqcblten_out = jnp.asarray(rqcblten) if rqcblten is not None else None
    rqiblten_out = jnp.asarray(rqiblten) if rqiblten is not None else None

    k_zero_end = min(int(kte) + 1, int(kde))
    full_single_tile = (
        len(i_start_tuple) == 1
        and int(i_start_tuple[0]) == int(ims)
        and int(i_end_tuple[0]) == int(ime)
        and int(j_start_tuple[0]) == int(jms)
        and int(j_end_tuple[0]) == int(jme)
        and int(kts) == int(kms)
        and int(kte) == int(kme)
    )
    if full_single_tile:
        if tsk_arr is not None and tskold_out is not None:
            tskold_out = tsk_arr
        if ust_arr is not None and ustold_out is not None:
            ustold_out = ust_arr
        if znt_arr is not None and zntold_out is not None:
            zntold_out = znt_arr
        if pblh_arr is not None and hpbl_hold_out is not None:
            hpbl_hold_out = pblh_arr
        if v_phy_arr is not None and v_phytmp_out is not None:
            v_phytmp_out = v_phy_arr + float(v_frame)
        if u_phy_arr is not None and u_phytmp_out is not None:
            u_phytmp_out = u_phy_arr + float(u_frame)
        if p8w_arr is not None:
            psfc_out = p8w_arr[:, _idx(kms, kms), :]
        if int(k_zero_end) == int(kme):
            if rthblten_out is not None:
                rthblten_out = jnp.zeros_like(rthblten_out)
            if rublten_out is not None:
                rublten_out = jnp.zeros_like(rublten_out)
            if rvblten_out is not None:
                rvblten_out = jnp.zeros_like(rvblten_out)
            if rqvblten_out is not None:
                rqvblten_out = jnp.zeros_like(rqvblten_out)
            if rqcblten_out is not None:
                rqcblten_out = jnp.zeros_like(rqcblten_out)
            if bool(f_qi) and rqiblten_out is not None:
                rqiblten_out = jnp.zeros_like(rqiblten_out)
        else:
            k_zero_slice = slice(_idx(kts, kms), _idx(k_zero_end, kms) + 1)
            slices = (slice(None), k_zero_slice, slice(None))
            if rthblten_out is not None:
                rthblten_out = rthblten_out.at[slices].set(0.0)
            if rublten_out is not None:
                rublten_out = rublten_out.at[slices].set(0.0)
            if rvblten_out is not None:
                rvblten_out = rvblten_out.at[slices].set(0.0)
            if rqvblten_out is not None:
                rqvblten_out = rqvblten_out.at[slices].set(0.0)
            if rqcblten_out is not None:
                rqcblten_out = rqcblten_out.at[slices].set(0.0)
            if bool(f_qi) and rqiblten_out is not None:
                rqiblten_out = rqiblten_out.at[slices].set(0.0)
    else:
        for its, ite, jts, jte in zip(i_start_tuple, i_end_tuple, j_start_tuple, j_end_tuple, strict=True):
            i_slice, j_slice = _tile_slices(ims, jms, its, ite, jts, jte)
            k_slice = slice(_idx(kts, kms), _idx(kte, kms) + 1)
            k_zero_slice = slice(_idx(kts, kms), _idx(k_zero_end, kms) + 1)
            if tsk_arr is not None and tskold_out is not None:
                tskold_out = tskold_out.at[i_slice, j_slice].set(tsk_arr[i_slice, j_slice])
            if ust_arr is not None and ustold_out is not None:
                ustold_out = ustold_out.at[i_slice, j_slice].set(ust_arr[i_slice, j_slice])
            if znt_arr is not None and zntold_out is not None:
                zntold_out = zntold_out.at[i_slice, j_slice].set(znt_arr[i_slice, j_slice])
            if pblh_arr is not None and hpbl_hold_out is not None:
                hpbl_hold_out = hpbl_hold_out.at[i_slice, j_slice].set(pblh_arr[i_slice, j_slice])
            if v_phy_arr is not None and v_phytmp_out is not None:
                v_phytmp_out = v_phytmp_out.at[i_slice, k_slice, j_slice].set(v_phy_arr[i_slice, k_slice, j_slice] + float(v_frame))
            if u_phy_arr is not None and u_phytmp_out is not None:
                u_phytmp_out = u_phytmp_out.at[i_slice, k_slice, j_slice].set(u_phy_arr[i_slice, k_slice, j_slice] + float(u_frame))
            if p8w_arr is not None:
                if psfc_out is None:
                    psfc_out = jnp.zeros_like(p8w_arr[:, _idx(kms, kms), :])
                psfc_out = psfc_out.at[i_slice, j_slice].set(p8w_arr[i_slice, _idx(kms, kms), j_slice])
            slices = (i_slice, k_zero_slice, j_slice)
            if rthblten_out is not None:
                rthblten_out = rthblten_out.at[slices].set(0.0)
            if rublten_out is not None:
                rublten_out = rublten_out.at[slices].set(0.0)
            if rvblten_out is not None:
                rvblten_out = rvblten_out.at[slices].set(0.0)
            if rqvblten_out is not None:
                rqvblten_out = rqvblten_out.at[slices].set(0.0)
            if rqcblten_out is not None:
                rqcblten_out = rqcblten_out.at[slices].set(0.0)
            if bool(f_qi) and rqiblten_out is not None:
                rqiblten_out = rqiblten_out.at[slices].set(0.0)

    pblh_out = pblh_arr
    kpbl_out = _as_array_or_none(kpbl)
    external_calls: list[str] = []
    if int(bl_pbl_physics) == MYJPBLSCHEME:
        if myjpbl_fn is None:
            raise NotImplementedError("pbl_driver MYJ branch requires an explicit myjpbl_fn callback")
        updates = _mapping_from_result(
            myjpbl_fn(
                dt=dt,
                dtbl=dtbl,
                stepbl=stepbl,
                u_phytmp=u_phytmp_out,
                v_phytmp=v_phytmp_out,
                psfc=psfc_out,
                rublten=rublten_out,
                rvblten=rvblten_out,
                rthblten=rthblten_out,
                rqvblten=rqvblten_out,
                rqcblten=rqcblten_out,
                rqiblten=rqiblten_out,
                pblh=pblh_out,
                kpbl=kpbl_out,
                qv_curr=qv_curr,
                qc_curr=qc_curr,
                qi_curr=qi_curr,
                fields=carried,
            )
        )
        rublten_out = updates.get("rublten", rublten_out)
        rvblten_out = updates.get("rvblten", rvblten_out)
        rthblten_out = updates.get("rthblten", rthblten_out)
        rqvblten_out = updates.get("rqvblten", rqvblten_out)
        rqcblten_out = updates.get("rqcblten", rqcblten_out)
        rqiblten_out = updates.get("rqiblten", rqiblten_out)
        pblh_out = updates.get("pblh", pblh_out)
        kpbl_out = updates.get("kpbl", kpbl_out)
        carried.update(updates)
        external_calls.append("MYJPBL")
    else:
        raise NotImplementedError(
            "pbl_driver currently supports only bl_pbl_physics=0 or MYJPBLSCHEME=2; "
            f"got {int(bl_pbl_physics)}"
        )

    if int(windfarm_opt) == FITCHSCHEME:
        raise NotImplementedError("pbl_driver windfarm_opt=FITCHSCHEME requires translated dragforce integration")

    if gwd_opt is not None and int(gwd_opt) in (1, 3):
        if gwd_fn is None:
            raise NotImplementedError("pbl_driver gwd_opt requires an explicit gwd_fn callback")
        updates = _mapping_from_result(
            gwd_fn(
                gwd_opt=int(gwd_opt),
                dtbl=dtbl,
                rublten=rublten_out,
                rvblten=rvblten_out,
                rthblten=rthblten_out,
                pblh=pblh_out,
                kpbl=kpbl_out,
                fields=carried,
            )
        )
        rublten_out = updates.get("rublten", rublten_out)
        rvblten_out = updates.get("rvblten", rvblten_out)
        rthblten_out = updates.get("rthblten", rthblten_out)
        carried.update({k: v for k, v in updates.items() if k not in {"rublten", "rvblten", "rthblten"}})
        external_calls.append("GWDO")

    return PblDriverResult(
        u_phytmp=u_phytmp_out,
        v_phytmp=v_phytmp_out,
        tskold=tskold_out,
        ustold=ustold_out,
        zntold=zntold_out,
        hpbl_hold=hpbl_hold_out,
        psfc=psfc_out,
        rublten=rublten_out,
        rvblten=rvblten_out,
        rthblten=rthblten_out,
        rqvblten=rqvblten_out,
        rqcblten=rqcblten_out,
        rqiblten=rqiblten_out,
        pblh=pblh_out,
        kpbl=kpbl_out,
        dtmin=dtmin,
        dtbl=dtbl,
        bldtacttime=bldtacttime_out,
        carried_fields=_carried_without_explicit_results(carried),
        ran_parameterization=True,
        external_calls=tuple(external_calls),
    )


def _invert(kms: int, kts: int, kte: int, a: Any, c: Any) -> Any:
    a_out = a
    c_out = c
    for in_wrf in range(int(kte) - 1, int(kts) - 1, -1):
        idx = _idx(in_wrf, kms)
        idxp1 = _idx(in_wrf + 1, kms)
        c_out = c_out.at[idx].set(c_out[idx] - a_out[idx, 2] * c_out[idxp1] / a_out[idxp1, 1])
        a_out = a_out.at[idx, 1].set(a_out[idx, 1] - a_out[idx, 2] * a_out[idxp1, 0] / a_out[idxp1, 1])
    for in_wrf in range(int(kts) + 1, int(kte) + 1):
        idx = _idx(in_wrf, kms)
        idxm1 = _idx(in_wrf - 1, kms)
        c_out = c_out.at[idx].set(c_out[idx] - a_out[idx, 0] * c_out[idxm1] / a_out[idxm1, 1])
    x_out = jnp.zeros_like(c_out)
    for in_wrf in range(int(kts), int(kte) + 1):
        idx = _idx(in_wrf, kms)
        x_out = x_out.at[idx].set(c_out[idx] / a_out[idx, 1])
    return x_out


def _diff_column(kms: int, kme: int, kts: int, kte: int, dt: Any, co: Any, da: Any, daz: Any, cd: Any, sf: Any, vl: Any, dz: Any) -> Any:
    dtype = jnp.result_type(dt, co, da, daz, cd, sf, vl, dz, jnp.float32)
    dt = jnp.asarray(dt, dtype=dtype)
    co_in = jnp.asarray(co, dtype=dtype)
    da = jnp.asarray(da, dtype=dtype)
    daz = jnp.asarray(daz, dtype=dtype)
    cd = jnp.asarray(cd, dtype=dtype)
    sf = jnp.asarray(sf, dtype=dtype)
    vl = jnp.asarray(vl, dtype=dtype)
    dz = jnp.asarray(dz, dtype=dtype)

    nlev = int(kme) - int(kms) + 1
    cddz = jnp.zeros((nlev,), dtype=dtype)
    a = jnp.zeros((nlev, 3), dtype=dtype)
    c = jnp.zeros((nlev,), dtype=dtype)

    kts_i = _idx(kts, kms)
    cddz = cddz.at[kts_i].set(sf[kts_i] * daz[kts_i] * cd[kts_i] / dz[kts_i])
    for iz in range(int(kts) + 1, int(kte) + 1):
        idx = _idx(iz, kms)
        idxm1 = _idx(iz - 1, kms)
        cddz = cddz.at[idx].set(2.0 * sf[idx] * daz[idx] * cd[idx] / (dz[idx] + dz[idxm1]))
    ktep1 = _idx(int(kte) + 1, kms)
    kte_i = _idx(kte, kms)
    cddz = cddz.at[ktep1].set(sf[ktep1] * daz[ktep1] * cd[ktep1] / dz[kte_i])

    iz1 = 1
    izf = 1
    for iz in range(iz1, int(kte)):
        idx = _idx(iz, kms)
        idxp1 = _idx(iz + 1, kms)
        dzv = vl[idx] * dz[idx]
        a = a.at[idx, 0].set(-cddz[idx] * dt / dzv / da[idx])
        a = a.at[idx, 1].set(1.0 + dt * (cddz[idx] + cddz[idxp1]) / dzv / da[idx])
        a = a.at[idx, 2].set(-cddz[idxp1] * dt / dzv / da[idx])
        c = c.at[idx].set(co_in[idx])

    for iz in range(int(kte) - (izf - 1), int(kte) + 1):
        idx = _idx(iz, kms)
        a = a.at[idx, 0].set(0.0)
        a = a.at[idx, 1].set(1.0)
        a = a.at[idx, 2].set(0.0)
        c = c.at[idx].set(co_in[idx])

    return _invert(kms, kts, kte, a, c)


def diff4d(
    dt: Any,
    dz: Any,
    scalar: Any,
    is_scalar: bool,
    rho: Any,
    exch_h: Any,
    exch_m: Any,
    scalar_tend: Any,
    num_scalar: int,
    param_first_scalar: int,
    *,
    ids: int = 1,
    ide: int | None = None,
    jds: int = 1,
    jde: int | None = None,
    kds: int = 1,
    kde: int | None = None,
    ims: int = 1,
    ime: int | None = None,
    jms: int = 1,
    jme: int | None = None,
    kms: int = 1,
    kme: int | None = None,
    its: int = 1,
    ite: int | None = None,
    jts: int = 1,
    jte: int | None = None,
    kts: int = 1,
    kte: int | None = None,
    excluded_scalar_indices: tuple[int, ...] = DEFAULT_DIFF4D_EXCLUDED_SCALARS,
) -> Diff4dResult:
    """Update scalar tendencies with WRF ``DIFF4D`` vertical diffusion."""

    del ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, exch_m
    dtype = jnp.result_type(dt, dz, scalar, rho, exch_h, scalar_tend, jnp.float32)
    dt = jnp.asarray(dt, dtype=dtype)
    dz_arr = jnp.asarray(dz, dtype=dtype)
    scalar_arr = jnp.asarray(scalar, dtype=dtype)
    rho_arr = jnp.asarray(rho, dtype=dtype)
    exch_h_arr = jnp.asarray(exch_h, dtype=dtype)
    scalar_tend_out = jnp.asarray(scalar_tend, dtype=dtype)

    kme_i = int(kme) if kme is not None else scalar_arr.shape[1]
    kte_i = int(kte) if kte is not None else scalar_arr.shape[1]
    i_start = int(its) - 1
    i_stop = int(ite) if ite is not None else scalar_arr.shape[0]
    j_start = int(jts) - 1
    j_stop = int(jte) if jte is not None else scalar_arr.shape[2]
    im_start = int(param_first_scalar) - 1
    im_stop = int(num_scalar)
    excluded = {int(index) for index in excluded_scalar_indices}
    nlev = kme_i - int(kms) + 1

    for im in range(im_start, im_stop):
        wrf_im = im + 1
        if bool(is_scalar) and wrf_im in excluded:
            continue
        for j in range(j_start, j_stop):
            for i in range(i_start, i_stop):
                s1d = jnp.zeros((nlev,), dtype=dtype)
                dz1d = jnp.zeros((nlev,), dtype=dtype)
                rho1d = jnp.zeros((nlev,), dtype=dtype)
                rhoz1d = jnp.zeros((nlev,), dtype=dtype)
                sf1d = jnp.zeros((nlev,), dtype=dtype)
                vl1d = jnp.zeros((nlev,), dtype=dtype)
                exch_h1d = jnp.zeros((nlev,), dtype=dtype)

                for k in range(int(kts), kte_i + 1):
                    idx = _idx(k, kms)
                    k0 = k - 1
                    s1d = s1d.at[idx].set(scalar_arr[i, k0, j, im])
                    dz1d = dz1d.at[idx].set(dz_arr[i, k0, j])
                    rho1d = rho1d.at[idx].set(rho_arr[i, k0, j])
                    vl1d = vl1d.at[idx].set(1.0)
                    sf1d = sf1d.at[idx].set(1.0)
                    exch_h1d = exch_h1d.at[idx].set(exch_h_arr[i, k0, j])

                sf1d = sf1d.at[_idx(kte_i + 1, kms)].set(1.0)
                exch_h1d = exch_h1d.at[_idx(kts, kms)].set(0.0)
                rhoz1d = rhoz1d.at[_idx(kts, kms)].set(rho1d[_idx(kts, kms)])
                for k in range(int(kts) + 1, kte_i + 1):
                    idx = _idx(k, kms)
                    idxm1 = _idx(k - 1, kms)
                    rhoz1d = rhoz1d.at[idx].set((rho1d[idx] * dz1d[idxm1] + rho1d[idxm1] * dz1d[idx]) / (dz1d[idxm1] + dz1d[idx]))
                rhoz1d = rhoz1d.at[_idx(kte_i + 1, kms)].set(rho1d[_idx(kte_i, kms)])

                s1d_new = _diff_column(kms, kme_i, kts, kte_i, dt, s1d, rho1d, rhoz1d, exch_h1d, sf1d, vl1d, dz1d)
                for k in range(int(kts), kte_i + 1):
                    idx = _idx(k, kms)
                    k0 = k - 1
                    scalar_tend_out = scalar_tend_out.at[i, k0, j, im].set((s1d_new[idx] - scalar_arr[i, k0, j, im]) / dt)

    return Diff4dResult(scalar_tend=scalar_tend_out)


__all__ = [
    "DEFAULT_DIFF4D_EXCLUDED_SCALARS",
    "FITCHSCHEME",
    "MYJPBLSCHEME",
    "PBL_PHYSICS_NONE",
    "Diff4dResult",
    "PblDriverResult",
    "diff4d",
    "pbl_driver",
]
