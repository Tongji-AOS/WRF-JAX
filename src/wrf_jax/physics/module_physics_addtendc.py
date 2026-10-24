"""Translations from WRF ``phys/module_physics_addtendc``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from wrf_jax.physics.moist_coupling import advance_precipitation


@dataclass(frozen=True)
class AddTendConfig:
    specified: bool = False
    nested: bool = False
    periodic_x: bool = False
    ra_lw_physics: int = 0
    ra_sw_physics: int = 0
    cu_physics: int = 0
    bl_pbl_physics: int = 0
    shcu_physics: int = 0
    grid_fdda: int = 0
    ifire: int = 0


def _as_config(config_flags: AddTendConfig | Any) -> AddTendConfig:
    if hasattr(config_flags, "get"):
        return AddTendConfig(
            specified=bool(config_flags.get("specified", False)),
            nested=bool(config_flags.get("nested", False)),
            periodic_x=bool(config_flags.get("periodic_x", False)),
            ra_lw_physics=int(config_flags.get("ra_lw_physics", 0)),
            ra_sw_physics=int(config_flags.get("ra_sw_physics", 0)),
            cu_physics=int(config_flags.get("cu_physics", 0)),
            bl_pbl_physics=int(config_flags.get("bl_pbl_physics", 0)),
            shcu_physics=int(config_flags.get("shcu_physics", 0)),
            grid_fdda=int(config_flags.get("grid_fdda", 0)),
            ifire=int(config_flags.get("ifire", 0)),
        )
    return AddTendConfig(
        specified=bool(getattr(config_flags, "specified", False)),
        nested=bool(getattr(config_flags, "nested", False)),
        periodic_x=bool(getattr(config_flags, "periodic_x", False)),
        ra_lw_physics=int(getattr(config_flags, "ra_lw_physics", 0)),
        ra_sw_physics=int(getattr(config_flags, "ra_sw_physics", 0)),
        cu_physics=int(getattr(config_flags, "cu_physics", 0)),
        bl_pbl_physics=int(getattr(config_flags, "bl_pbl_physics", 0)),
        shcu_physics=int(getattr(config_flags, "shcu_physics", 0)),
        grid_fdda=int(getattr(config_flags, "grid_fdda", 0)),
        ifire=int(getattr(config_flags, "ifire", 0)),
    )


YSUSCHEME = 1
MYJPBLSCHEME = 2
GFSSCHEME = 3
QNSEPBLSCHEME = 4
MYNNPBLSCHEME = 5
ACMPBLSCHEME = 7
BOULACSCHEME = 8
CAMUWPBLSCHEME = 9
TEMFPBLSCHEME = 10
SHINHONGSCHEME = 11
GBMPBLSCHEME = 12
EEPSSCHEME = 16
KEPSSCHEME = 17
MRFSCHEME = 99
KFETASCHEME = 1
BMJSCHEME = 2
CAMZMSCHEME = 7
KFCUPSCHEME = 10
MSKFSCHEME = 11
TIEDTKESCHEME = 6
NTIEDTKESCHEME = 16
KFSCHEME = 99
PARAM_FIRST_SCALAR = 2
P_QV = 2
P_QC = 3
P_QR = 4
P_QI = 5
P_QS = 6
P_QG = 7
P_QT = 1
P_QNI = 2
P_QNR = 3


@dataclass(frozen=True)
class PhyBlTenResult:
    rt_tendf: Any
    ru_tendf: Any
    rv_tendf: Any
    moist_tendf: Any
    scalar_tendf: Any


@dataclass(frozen=True)
class UpdatePhyTenResult:
    rph_tendf: Any
    rt_tendf: Any
    ru_tendf: Any
    rv_tendf: Any
    moist_tendf: Any
    scalar_tendf: Any
    mu_tendf: Any


@dataclass(frozen=True)
class AdvancePPTResult:
    rthcuten: Any
    rqvcuten: Any
    rqccuten: Any
    rqrcuten: Any
    cldfra_cup: Any
    rqicuten: Any
    rqscuten: Any
    rainc: Any
    raincv: Any
    rainsh: Any
    pratec: Any
    pratesh: Any
    nca: Any
    htop: Any
    hbot: Any
    cutop: Any
    cubot: Any
    cuppt: Any


def _a2a_bounds(
    config_flags: AddTendConfig | Any,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    jms: int,
    kms: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> tuple[slice, slice, slice]:
    del kds
    config = _as_config(config_flags)
    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)

    if config.specified or config.nested:
        i_start = max(int(ids) + 1, int(its))
        i_end = min(int(ide) - 2, int(ite))
        j_start = max(int(jds) + 1, int(jts))
        j_end = min(int(jde) - 2, int(jte))
    if config.periodic_x:
        i_start = int(its)
        i_end = min(int(ite), int(ide) - 1)

    return (
        slice(i_start - int(ims), i_end - int(ims) + 1),
        slice(int(kts) - int(kms), ktf - int(kms) + 1),
        slice(j_start - int(jms), j_end - int(jms) + 1),
    )


def add_a2a(
    lvar: Any,
    rvar: Any,
    config_flags: AddTendConfig | Any,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int = 1,
    ime: int | None = None,
    jms: int = 1,
    jme: int | None = None,
    kms: int = 1,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> Any:
    """Return ``lvar`` after WRF ``ADD_A2A`` tendency accumulation."""

    del ime, jme, kme
    dtype = jnp.result_type(lvar, rvar, jnp.float32)
    lvar_out = jnp.asarray(lvar, dtype=dtype)
    rvar_arr = jnp.asarray(rvar, dtype=dtype)
    region = _a2a_bounds(config_flags, ids, ide, jds, jde, kds, kde, ims, jms, kms, its, ite, jts, jte, kts, kte)
    return lvar_out.at[region].add(rvar_arr[region])


def _a2c_u_bounds(
    config_flags: AddTendConfig | Any,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    jms: int,
    kms: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    del kds
    config = _as_config(config_flags)
    i_start = int(its)
    i_end = int(ite)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    ktf = min(int(kte), int(kde) - 1)

    if config.specified or config.nested:
        i_start = max(int(ids) + 1, int(its))
        i_end = min(int(ide) - 1, int(ite))
        j_start = max(int(jds) + 1, int(jts))
        j_end = min(int(jde) - 2, int(jte))
    if config.periodic_x:
        i_start = int(its)
        i_end = int(ite)

    k_slice = slice(int(kts) - int(kms), ktf - int(kms) + 1)
    j_slice = slice(j_start - int(jms), j_end - int(jms) + 1)
    l_slice = (slice(i_start - int(ims), i_end - int(ims) + 1), k_slice, j_slice)
    r_center = (slice(i_start - int(ims), i_end - int(ims) + 1), k_slice, j_slice)
    r_left = (slice(i_start - 1 - int(ims), i_end - 1 - int(ims) + 1), k_slice, j_slice)
    return l_slice, r_center, r_left


def add_a2c_u(
    lvar: Any,
    rvar: Any,
    config_flags: AddTendConfig | Any,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int = 1,
    jme: int | None = None,
    kms: int = 1,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> Any:
    """Return ``lvar`` after WRF ``ADD_A2C_U`` staggered tendency accumulation."""

    del ime, jme, kme
    dtype = jnp.result_type(lvar, rvar, jnp.float32)
    lvar_out = jnp.asarray(lvar, dtype=dtype)
    rvar_arr = jnp.asarray(rvar, dtype=dtype)
    l_region, r_center, r_left = _a2c_u_bounds(config_flags, ids, ide, jds, jde, kds, kde, ims, jms, kms, its, ite, jts, jte, kts, kte)
    increment = 0.5 * (rvar_arr[r_center] + rvar_arr[r_left])
    return lvar_out.at[l_region].add(increment)


def _a2c_v_bounds(
    config_flags: AddTendConfig | Any,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    jms: int,
    kms: int,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    del kds
    config = _as_config(config_flags)
    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = int(jte)
    ktf = min(int(kte), int(kde) - 1)

    if config.specified or config.nested:
        i_start = max(int(ids) + 1, int(its))
        i_end = min(int(ide) - 2, int(ite))
        j_start = max(int(jds) + 1, int(jts))
        j_end = min(int(jde) - 1, int(jte))
    if config.periodic_x:
        i_start = int(its)
        i_end = min(int(ite), int(ide) - 1)

    i_slice = slice(i_start - int(ims), i_end - int(ims) + 1)
    k_slice = slice(int(kts) - int(kms), int(kte) - int(kms) + 1)
    l_slice = (i_slice, k_slice, slice(j_start - int(jms), j_end - int(jms) + 1))
    r_center = (i_slice, k_slice, slice(j_start - int(jms), j_end - int(jms) + 1))
    r_down = (i_slice, k_slice, slice(j_start - 1 - int(jms), j_end - 1 - int(jms) + 1))
    return l_slice, r_center, r_down


def add_a2c_v(
    lvar: Any,
    rvar: Any,
    config_flags: AddTendConfig | Any,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int = 1,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int = 1,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> Any:
    """Return ``lvar`` after WRF ``ADD_A2C_V`` staggered tendency accumulation."""

    del ime, jme, kme
    dtype = jnp.result_type(lvar, rvar, jnp.float32)
    lvar_out = jnp.asarray(lvar, dtype=dtype)
    rvar_arr = jnp.asarray(rvar, dtype=dtype)
    l_region, r_center, r_down = _a2c_v_bounds(config_flags, ids, ide, jds, jde, kds, kde, ims, jms, kms, its, ite, jts, jte, kts, kte)
    increment = 0.5 * (rvar_arr[r_center] + rvar_arr[r_down])
    target_shape = lvar_out[l_region].shape
    if increment.shape != target_shape:
        increment = increment[tuple(slice(0, size) for size in target_shape)]
    return lvar_out.at[l_region].add(increment)


def _add_species_tendency(
    field: Any,
    species_index: int,
    tendency: Any,
    config_flags: AddTendConfig | Any,
    *,
    param_first_scalar: int,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None,
    jms: int,
    jme: int | None,
    kms: int,
    kme: int | None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> Any:
    if int(species_index) < int(param_first_scalar):
        return field
    species_axis_index = int(species_index) - 1
    if species_axis_index < 0 or species_axis_index >= jnp.asarray(field).shape[-1]:
        return field
    component = add_a2a(
        jnp.asarray(field)[..., species_axis_index],
        tendency,
        config_flags,
        ids=ids,
        ide=ide,
        jds=jds,
        jde=jde,
        kds=kds,
        kde=kde,
        ims=ims,
        ime=ime,
        jms=jms,
        jme=jme,
        kms=kms,
        kme=kme,
        its=its,
        ite=ite,
        jts=jts,
        jte=jte,
        kts=kts,
        kte=kte,
    )
    return jnp.asarray(field).at[..., species_axis_index].set(component)


def phy_cu_ten(
    config_flags: AddTendConfig | Any,
    rk_step: int,
    n_moist: int,
    n_scalar: int,
    rt_tendf: Any,
    ru_tendf: Any,
    rv_tendf: Any,
    rucuten: Any,
    rvcuten: Any,
    rthcuten: Any,
    rqvcuten: Any,
    rqccuten: Any,
    rqrcuten: Any,
    rqicuten: Any,
    rqscuten: Any,
    rqcncuten: Any,
    rqincuten: Any,
    moist_tendf: Any,
    scalar_tendf: Any,
    adv_moist_cond: bool,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> PhyBlTenResult:
    """Return WRF ``phy_cu_ten`` tendency accumulation for translated schemes."""

    del rk_step, n_moist, n_scalar, rqrcuten, rqscuten, rqcncuten, rqincuten
    config = _as_config(config_flags)
    if config.cu_physics not in (TIEDTKESCHEME, NTIEDTKESCHEME):
        raise NotImplementedError(f"phy_cu_ten branch not translated for cu_physics={config.cu_physics}")

    bounds = dict(
        ids=ids,
        ide=ide,
        jds=jds,
        jde=jde,
        kds=kds,
        kde=kde,
        ims=ims,
        ime=ime,
        jms=jms,
        jme=jme,
        kms=kms,
        kme=kme,
        its=its,
        ite=ite,
        jts=jts,
        jte=jte,
        kts=kts,
        kte=kte,
    )
    rt_out = add_a2a(rt_tendf, rthcuten, config, **bounds)
    ru_out = add_a2c_u(ru_tendf, rucuten, config, **bounds)
    rv_out = add_a2c_v(rv_tendf, rvcuten, config, **bounds)

    moist_out = _add_species_tendency(
        moist_tendf,
        P_QV,
        rqvcuten,
        config,
        param_first_scalar=PARAM_FIRST_SCALAR,
        **bounds,
    )
    moist_out = _add_species_tendency(
        moist_out,
        P_QC,
        rqccuten,
        config,
        param_first_scalar=PARAM_FIRST_SCALAR,
        **bounds,
    )
    moist_out = _add_species_tendency(
        moist_out,
        P_QI,
        rqicuten,
        config,
        param_first_scalar=PARAM_FIRST_SCALAR,
        **bounds,
    )

    scalar_out = jnp.asarray(scalar_tendf)
    if not adv_moist_cond:
        scalar_out = _add_species_tendency(
            scalar_out,
            P_QT,
            rqccuten,
            config,
            param_first_scalar=PARAM_FIRST_SCALAR,
            **bounds,
        )
        scalar_out = _add_species_tendency(
            scalar_out,
            P_QT,
            rqicuten,
            config,
            param_first_scalar=PARAM_FIRST_SCALAR,
            **bounds,
        )

    return PhyBlTenResult(rt_tendf=rt_out, ru_tendf=ru_out, rv_tendf=rv_out, moist_tendf=moist_out, scalar_tendf=scalar_out)


def phy_ra_ten(
    config_flags: AddTendConfig | Any,
    rt_tendf: Any,
    rthraten: Any,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> Any:
    """Return WRF ``phy_ra_ten`` radiation tendency accumulation."""

    return add_a2a(
        rt_tendf,
        rthraten,
        config_flags,
        ids=ids,
        ide=ide,
        jds=jds,
        jde=jde,
        kds=kds,
        kde=kde,
        ims=ims,
        ime=ime,
        jms=jms,
        jme=jme,
        kms=kms,
        kme=kme,
        its=its,
        ite=ite,
        jts=jts,
        jte=jte,
        kts=kts,
        kte=kte,
    )


def update_phy_ten(
    rph_tendf: Any,
    rt_tendf: Any,
    ru_tendf: Any,
    rv_tendf: Any,
    moist_tendf: Any,
    scalar_tendf: Any,
    mu_tendf: Any,
    rthraten: Any,
    rthblten: Any,
    rthcuten: Any,
    rthshten: Any,
    rublten: Any,
    rucuten: Any,
    rushten: Any,
    rvblten: Any,
    rvcuten: Any,
    rvshten: Any,
    rqvblten: Any,
    rqcblten: Any,
    rqiblten: Any,
    rqniblten: Any,
    rqvcuten: Any,
    rqccuten: Any,
    rqrcuten: Any,
    rqicuten: Any,
    rqscuten: Any,
    rqcncuten: Any,
    rqincuten: Any,
    rqvshten: Any,
    rqcshten: Any,
    rqrshten: Any,
    rqishten: Any,
    rqsshten: Any,
    rqgshten: Any,
    rqcnshten: Any,
    rqinshten: Any,
    rundgdten: Any,
    rvndgdten: Any,
    rthndgdten: Any,
    rphndgdten: Any,
    rqvndgdten: Any,
    rmundgdten: Any,
    rthfrten: Any,
    rqvfrten: Any,
    n_moist: int,
    n_scalar: int,
    config_flags: AddTendConfig | Any,
    rk_step: int,
    adv_moist_cond: bool,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> UpdatePhyTenResult:
    """Return WRF ``update_phy_ten`` tendency state for translated branches."""

    del (
        rthshten,
        rushten,
        rvshten,
        rqvshten,
        rqcshten,
        rqrshten,
        rqishten,
        rqsshten,
        rqgshten,
        rqcnshten,
        rqinshten,
        rundgdten,
        rvndgdten,
        rthndgdten,
        rphndgdten,
        rqvndgdten,
        rmundgdten,
        rthfrten,
        rqvfrten,
    )
    config = _as_config(config_flags)
    rph_out = jnp.asarray(rph_tendf)
    rt_out = jnp.asarray(rt_tendf)
    ru_out = jnp.asarray(ru_tendf)
    rv_out = jnp.asarray(rv_tendf)
    moist_out = jnp.asarray(moist_tendf)
    scalar_out = jnp.asarray(scalar_tendf)
    mu_out = jnp.asarray(mu_tendf)

    bounds = dict(
        ids=ids,
        ide=ide,
        jds=jds,
        jde=jde,
        kds=kds,
        kde=kde,
        ims=ims,
        ime=ime,
        jms=jms,
        jme=jme,
        kms=kms,
        kme=kme,
        its=its,
        ite=ite,
        jts=jts,
        jte=jte,
        kts=kts,
        kte=kte,
    )

    if config.ra_lw_physics > 0 or config.ra_sw_physics > 0:
        rt_out = phy_ra_ten(config, rt_out, rthraten, **bounds)

    if config.bl_pbl_physics > 0:
        bl = phy_bl_ten(
            config,
            rk_step,
            n_moist,
            n_scalar,
            rt_out,
            ru_out,
            rv_out,
            moist_out,
            scalar_out,
            adv_moist_cond,
            rthblten,
            rublten,
            rvblten,
            rqvblten,
            rqcblten,
            rqiblten,
            rqniblten,
            **bounds,
        )
        rt_out = bl.rt_tendf
        ru_out = bl.ru_tendf
        rv_out = bl.rv_tendf
        moist_out = bl.moist_tendf
        scalar_out = bl.scalar_tendf

    if config.cu_physics > 0:
        cu = phy_cu_ten(
            config,
            rk_step,
            n_moist,
            n_scalar,
            rt_out,
            ru_out,
            rv_out,
            rucuten,
            rvcuten,
            rthcuten,
            rqvcuten,
            rqccuten,
            rqrcuten,
            rqicuten,
            rqscuten,
            rqcncuten,
            rqincuten,
            moist_out,
            scalar_out,
            adv_moist_cond,
            **bounds,
        )
        rt_out = cu.rt_tendf
        ru_out = cu.ru_tendf
        rv_out = cu.rv_tendf
        moist_out = cu.moist_tendf
        scalar_out = cu.scalar_tendf

    unsupported = []
    if config.shcu_physics > 0:
        unsupported.append("shcu_physics")
    if config.grid_fdda > 0:
        unsupported.append("grid_fdda")
    if config.ifire > 0:
        unsupported.append("ifire")
    if unsupported:
        raise NotImplementedError("update_phy_ten branches not translated: " + ", ".join(unsupported))

    return UpdatePhyTenResult(
        rph_tendf=rph_out,
        rt_tendf=rt_out,
        ru_tendf=ru_out,
        rv_tendf=rv_out,
        moist_tendf=moist_out,
        scalar_tendf=scalar_out,
        mu_tendf=mu_out,
    )

def phy_bl_ten(
    config_flags: AddTendConfig | Any,
    rk_step: int,
    n_moist: int,
    n_scalar: int,
    rt_tendf: Any,
    ru_tendf: Any,
    rv_tendf: Any,
    moist_tendf: Any,
    scalar_tendf: Any,
    adv_moist_cond: bool,
    rthblten: Any,
    rublten: Any,
    rvblten: Any,
    rqvblten: Any,
    rqcblten: Any,
    rqiblten: Any,
    rqniblten: Any,
    *,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    param_first_scalar: int = PARAM_FIRST_SCALAR,
    p_qv: int = P_QV,
    p_qc: int = P_QC,
    p_qi: int = P_QI,
    p_qt: int = P_QT,
    p_qni: int = P_QNI,
) -> PhyBlTenResult:
    """Return WRF ``phy_bl_ten`` PBL tendency accumulation state."""

    del rk_step, n_moist, n_scalar
    config = _as_config(config_flags)
    scheme = int(config.bl_pbl_physics)
    dtype = jnp.result_type(rt_tendf, ru_tendf, rv_tendf, moist_tendf, scalar_tendf, rthblten, rublten, rvblten, rqvblten, rqcblten, rqiblten, rqniblten, jnp.float32)
    rt_out = jnp.asarray(rt_tendf, dtype=dtype)
    ru_out = jnp.asarray(ru_tendf, dtype=dtype)
    rv_out = jnp.asarray(rv_tendf, dtype=dtype)
    moist_out = jnp.asarray(moist_tendf, dtype=dtype)
    scalar_out = jnp.asarray(scalar_tendf, dtype=dtype)

    supported_schemes = (
        YSUSCHEME,
        SHINHONGSCHEME,
        MRFSCHEME,
        ACMPBLSCHEME,
        MYJPBLSCHEME,
        QNSEPBLSCHEME,
        GFSSCHEME,
        MYNNPBLSCHEME,
        EEPSSCHEME,
        KEPSSCHEME,
        BOULACSCHEME,
        CAMUWPBLSCHEME,
        TEMFPBLSCHEME,
        GBMPBLSCHEME,
    )
    if scheme not in supported_schemes:
        return PhyBlTenResult(rt_tendf=rt_out, ru_tendf=ru_out, rv_tendf=rv_out, moist_tendf=moist_out, scalar_tendf=scalar_out)

    bounds = dict(
        ids=ids,
        ide=ide,
        jds=jds,
        jde=jde,
        kds=kds,
        kde=kde,
        ims=ims,
        ime=ime,
        jms=jms,
        jme=jme,
        kms=kms,
        kme=kme,
        its=its,
        ite=ite,
        jts=jts,
        jte=jte,
        kts=kts,
        kte=kte,
    )
    rt_out = add_a2a(rt_out, rthblten, config, **bounds)
    ru_out = add_a2c_u(ru_out, rublten, config, **bounds)
    rv_out = add_a2c_v(rv_out, rvblten, config, **bounds)

    species_bounds = dict(bounds, param_first_scalar=param_first_scalar)

    def add_moist(species_index: int, tendency: Any) -> None:
        nonlocal moist_out
        moist_out = _add_species_tendency(moist_out, species_index, tendency, config, **species_bounds)

    def add_scalar(species_index: int, tendency: Any) -> None:
        nonlocal scalar_out
        scalar_out = _add_species_tendency(scalar_out, species_index, tendency, config, **species_bounds)

    add_moist(p_qv, rqvblten)

    if scheme in (YSUSCHEME, SHINHONGSCHEME, MRFSCHEME, ACMPBLSCHEME, GFSSCHEME, EEPSSCHEME):
        add_moist(p_qc, rqcblten)
        add_moist(p_qi, rqiblten)
        if not adv_moist_cond:
            add_scalar(p_qt, rqcblten)
            add_scalar(p_qt, rqiblten)
    elif scheme in (MYJPBLSCHEME, MYNNPBLSCHEME):
        if not adv_moist_cond:
            add_scalar(p_qt, rqcblten)
            add_scalar(p_qt, rqiblten)
        else:
            add_moist(p_qc, rqcblten)
            add_moist(p_qi, rqiblten)
    elif scheme in (QNSEPBLSCHEME, KEPSSCHEME, BOULACSCHEME):
        if not adv_moist_cond:
            add_scalar(p_qt, rqcblten)
        else:
            add_moist(p_qc, rqcblten)
    elif scheme == CAMUWPBLSCHEME:
        if not adv_moist_cond:
            add_scalar(p_qt, rqcblten)
            add_scalar(p_qt, rqiblten)
            add_scalar(p_qt, rqniblten)
        else:
            add_moist(p_qc, rqcblten)
            add_moist(p_qi, rqiblten)
            add_scalar(p_qni, rqniblten)
    elif scheme in (TEMFPBLSCHEME, GBMPBLSCHEME):
        add_moist(p_qc, rqcblten)
        add_moist(p_qi, rqiblten)

    return PhyBlTenResult(rt_tendf=rt_out, ru_tendf=ru_out, rv_tendf=rv_out, moist_tendf=moist_out, scalar_tendf=scalar_out)


def _wrf_nint(value: Any) -> int:
    return int(jnp.rint(value).item())


def advance_ppt(
    rthcuten: Any,
    rqvcuten: Any,
    rqccuten: Any,
    rqrcuten: Any,
    cldfra_cup: Any,
    rqicuten: Any,
    rqscuten: Any,
    rainc: Any,
    raincv: Any,
    rainsh: Any,
    pratec: Any,
    pratesh: Any,
    nca: Any,
    htop: Any,
    hbot: Any,
    cutop: Any,
    cubot: Any,
    cuppt: Any,
    dt: Any,
    config_flags: AddTendConfig | Any,
    *,
    preserve_eager_rounding: bool = False,
    ids: int,
    ide: int,
    jds: int,
    jde: int,
    kds: int,
    kde: int,
    ims: int,
    ime: int | None = None,
    jms: int,
    jme: int | None = None,
    kms: int,
    kme: int | None = None,
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
) -> AdvancePPTResult:
    """Return WRF ``advance_ppt`` precipitation and cumulus tendency state."""

    del ids, jds, kds, ime, jme, kme
    config = _as_config(config_flags)
    dtype = jnp.result_type(
        rthcuten,
        rqvcuten,
        rqccuten,
        rqrcuten,
        cldfra_cup,
        rqicuten,
        rqscuten,
        rainc,
        raincv,
        rainsh,
        pratec,
        pratesh,
        nca,
        htop,
        hbot,
        cutop,
        cubot,
        cuppt,
        dt,
        jnp.float32,
    )
    rth_out = jnp.asarray(rthcuten, dtype=dtype)
    rqv_out = jnp.asarray(rqvcuten, dtype=dtype)
    rqc_out = jnp.asarray(rqccuten, dtype=dtype)
    rqr_out = jnp.asarray(rqrcuten, dtype=dtype)
    cldfra_out = jnp.asarray(cldfra_cup, dtype=dtype)
    rqi_out = jnp.asarray(rqicuten, dtype=dtype)
    rqs_out = jnp.asarray(rqscuten, dtype=dtype)
    rainc_out = jnp.asarray(rainc, dtype=dtype)
    raincv_out = jnp.asarray(raincv, dtype=dtype)
    rainsh_out = jnp.asarray(rainsh, dtype=dtype)
    pratec_out = jnp.asarray(pratec, dtype=dtype)
    pratesh_out = jnp.asarray(pratesh, dtype=dtype)
    nca_out = jnp.asarray(nca, dtype=dtype)
    htop_out = jnp.asarray(htop, dtype=dtype)
    hbot_out = jnp.asarray(hbot, dtype=dtype)
    cutop_out = jnp.asarray(cutop, dtype=dtype)
    cubot_out = jnp.asarray(cubot, dtype=dtype)
    cuppt_out = jnp.asarray(cuppt, dtype=dtype)
    dt_val = jnp.asarray(dt, dtype=dtype)

    if config.cu_physics == 0:
        return AdvancePPTResult(
            rthcuten=rth_out,
            rqvcuten=rqv_out,
            rqccuten=rqc_out,
            rqrcuten=rqr_out,
            cldfra_cup=cldfra_out,
            rqicuten=rqi_out,
            rqscuten=rqs_out,
            rainc=rainc_out,
            raincv=raincv_out,
            rainsh=rainsh_out,
            pratec=pratec_out,
            pratesh=pratesh_out,
            nca=nca_out,
            htop=htop_out,
            hbot=hbot_out,
            cutop=cutop_out,
            cubot=cubot_out,
            cuppt=cuppt_out,
        )

    i_start = int(its)
    i_end = min(int(ite), int(ide) - 1)
    j_start = int(jts)
    j_end = min(int(jte), int(jde) - 1)
    k_start = int(kts)
    k_end = min(int(kte), int(kde) - 1)

    i_slice = slice(i_start - int(ims), i_end - int(ims) + 1)
    j_slice = slice(j_start - int(jms), j_end - int(jms) + 1)
    active_2d = (i_slice, j_slice)
    pratec_active = pratec_out[active_2d]
    pratesh_active = pratesh_out[active_2d]
    rainc_active, rainsh_active, cuppt_active = advance_precipitation(
        (rainc_out[active_2d], rainsh_out[active_2d], cuppt_out[active_2d],
         pratec_active, pratesh_active),
        dt_val, preserve_eager_rounding=preserve_eager_rounding,
    )
    rainc_out = rainc_out.at[active_2d].set(rainc_active)
    rainsh_out = rainsh_out.at[active_2d].set(rainsh_active)
    cuppt_out = cuppt_out.at[active_2d].set(cuppt_active)

    def zero_tendencies(ii: int, jj: int) -> None:
        nonlocal rth_out, rqv_out, rqc_out, rqr_out, rqi_out, rqs_out
        for k in range(k_start, k_end + 1):
            kk = k - int(kms)
            rth_out = rth_out.at[ii, kk, jj].set(0.0)
            rqv_out = rqv_out.at[ii, kk, jj].set(0.0)
            rqc_out = rqc_out.at[ii, kk, jj].set(0.0)
            rqr_out = rqr_out.at[ii, kk, jj].set(0.0)
            if P_QI >= PARAM_FIRST_SCALAR:
                rqi_out = rqi_out.at[ii, kk, jj].set(0.0)
            if P_QS >= PARAM_FIRST_SCALAR:
                rqs_out = rqs_out.at[ii, kk, jj].set(0.0)

    if config.cu_physics == KFSCHEME:
        for j in range(j_start, j_end + 1):
            jj = j - int(jms)
            for i in range(i_start, i_end + 1):
                ii = i - int(ims)
                if float(nca_out[ii, jj]) > 0.0:
                    if _wrf_nint(nca_out[ii, jj] / dt_val) <= 0:
                        zero_tendencies(ii, jj)
                    nca_out = nca_out.at[ii, jj].set(nca_out[ii, jj] - dt_val)
    elif config.cu_physics in (BMJSCHEME, CAMZMSCHEME):
        for j in range(j_start, j_end + 1):
            jj = j - int(jms)
            for i in range(i_start, i_end + 1):
                ii = i - int(ims)
                ncutop = _wrf_nint(cutop_out[ii, jj])
                ncubot = _wrf_nint(cubot_out[ii, jj])
                if ncutop > 1 and ncutop < int(kde):
                    htop_out = htop_out.at[ii, jj].set(jnp.maximum(cutop_out[ii, jj], htop_out[ii, jj]))
                if ncubot > 0 and ncubot < int(kde):
                    hbot_out = hbot_out.at[ii, jj].set(jnp.minimum(cubot_out[ii, jj], hbot_out[ii, jj]))
    elif config.cu_physics in (KFETASCHEME, MSKFSCHEME, KFCUPSCHEME):
        for j in range(j_start, j_end + 1):
            jj = j - int(jms)
            for i in range(i_start, i_end + 1):
                ii = i - int(ims)
                ncutop = _wrf_nint(cutop_out[ii, jj])
                ncubot = _wrf_nint(cubot_out[ii, jj])
                if ncutop > 1 and ncutop < int(kde):
                    htop_out = htop_out.at[ii, jj].set(jnp.maximum(cutop_out[ii, jj], htop_out[ii, jj]))
                if ncubot > 0 and ncubot < int(kde):
                    hbot_out = hbot_out.at[ii, jj].set(jnp.minimum(cubot_out[ii, jj], hbot_out[ii, jj]))
                if float(nca_out[ii, jj]) > 0.0:
                    if _wrf_nint(nca_out[ii, jj] / dt_val) <= 1:
                        zero_tendencies(ii, jj)
                    nca_out = nca_out.at[ii, jj].set(nca_out[ii, jj] - dt_val)

        if config.cu_physics == KFCUPSCHEME:
            for j in range(j_start, j_end + 1):
                jj = j - int(jms)
                for i in range(i_start, i_end + 1):
                    ii = i - int(ims)
                    if float(nca_out[ii, jj]) > 0.0 and _wrf_nint(nca_out[ii, jj] / dt_val) <= 1:
                        for k in range(k_start, k_end + 1):
                            kk = k - int(kms)
                            cldfra_out = cldfra_out.at[ii, kk, jj].set(0.0)

    return AdvancePPTResult(
        rthcuten=rth_out,
        rqvcuten=rqv_out,
        rqccuten=rqc_out,
        rqrcuten=rqr_out,
        cldfra_cup=cldfra_out,
        rqicuten=rqi_out,
        rqscuten=rqs_out,
        rainc=rainc_out,
        raincv=raincv_out,
        rainsh=rainsh_out,
        pratec=pratec_out,
        pratesh=pratesh_out,
        nca=nca_out,
        htop=htop_out,
        hbot=hbot_out,
        cutop=cutop_out,
        cubot=cubot_out,
        cuppt=cuppt_out,
    )


__all__ = [
    "ACMPBLSCHEME",
    "AddTendConfig",
    "AdvancePPTResult",
    "BOULACSCHEME",
    "CAMUWPBLSCHEME",
    "EEPSSCHEME",
    "GBMPBLSCHEME",
    "GFSSCHEME",
    "KEPSSCHEME",
    "MRFSCHEME",
    "MYJPBLSCHEME",
    "MYNNPBLSCHEME",
    "P_QC",
    "P_QG",
    "P_QI",
    "P_QNI",
    "P_QNR",
    "P_QR",
    "P_QS",
    "P_QT",
    "P_QV",
    "PARAM_FIRST_SCALAR",
    "PhyBlTenResult",
    "QNSEPBLSCHEME",
    "SHINHONGSCHEME",
    "TEMFPBLSCHEME",
    "UpdatePhyTenResult",
    "YSUSCHEME",
    "add_a2a",
    "add_a2c_u",
    "add_a2c_v",
    "advance_ppt",
    "phy_bl_ten",
    "phy_ra_ten",
    "update_phy_ten",
]
