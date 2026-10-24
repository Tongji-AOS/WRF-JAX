"""Outer-shell translations from WRF ``phys/module_sf_noahdrv``."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from wrf_jax.core.optimization import semantic_call, optimization_barrier
from wrf_jax.core.real_arithmetic import real_assignment
from wrf_jax.physics.module_sf_noahlsm import NoahLsmParameters, SflxResult, frh2o, sflx
from wrf_jax.physics.module_sf_noahlsm_glacial import glacial_columns

NLUS = 50
NSLTYPE = 30
NSLOPE = 30

_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?")


@dataclass(frozen=True)
class SoilVegGenParmResult:
    """Explicit parameter state filled by WRF ``SOIL_VEG_GEN_PARM``."""

    lutype: str
    lucats: int
    veg_iindex: int
    shdtbl: Any
    nrotbl: Any
    rstbl: Any
    rgltbl: Any
    hstbl: Any
    snuptbl: Any
    maxalb: Any
    laimintbl: Any
    laimaxtbl: Any
    emissmintbl: Any
    emissmaxtbl: Any
    albedomintbl: Any
    albedomaxtbl: Any
    z0mintbl: Any
    z0maxtbl: Any
    ztopvtbl: Any
    zbotvtbl: Any
    topt_data: np.float32
    cmcmax_data: np.float32
    cfactr_data: np.float32
    rsmax_data: np.float32
    bare: int
    natural: int
    lcz_1: int
    lcz_2: int
    lcz_3: int
    lcz_4: int
    lcz_5: int
    lcz_6: int
    lcz_7: int
    lcz_8: int
    lcz_9: int
    lcz_10: int
    lcz_11: int
    sltype: str
    slcats: int
    soil_iindex: int
    bb: Any
    drysmc: Any
    f11: Any
    maxsmc: Any
    refsmc: Any
    satpsi: Any
    satdk: Any
    satdw: Any
    wltsmc: Any
    qtz: Any
    num_slope: int
    slpcats: int
    slope_data: Any
    sbeta_data: np.float32
    fxexp_data: np.float32
    csoil_data: np.float32
    salp_data: np.float32
    refdk_data: np.float32
    refkdt_data: np.float32
    frzk_data: np.float32
    zbot_data: np.float32
    czil_data: np.float32
    smlow_data: np.float32
    smhigh_data: np.float32
    lvcoef_data: np.float32

    def to_noah_lsm_parameters(self) -> NoahLsmParameters:
        """Return the subset consumed by translated Noah LSM kernels."""

        return NoahLsmParameters(
            lucats=self.lucats,
            slcats=self.slcats,
            slpcats=self.slpcats,
            bare=self.bare,
            nrotbl=self.nrotbl,
            snuptbl=self.snuptbl,
            rstbl=self.rstbl,
            rgltbl=self.rgltbl,
            hstbl=self.hstbl,
            emissmintbl=self.emissmintbl,
            emissmaxtbl=self.emissmaxtbl,
            laimintbl=self.laimintbl,
            laimaxtbl=self.laimaxtbl,
            z0mintbl=self.z0mintbl,
            z0maxtbl=self.z0maxtbl,
            albedomintbl=self.albedomintbl,
            albedomaxtbl=self.albedomaxtbl,
            ztopvtbl=self.ztopvtbl,
            zbotvtbl=self.zbotvtbl,
            topt_data=self.topt_data,
            cmcmax_data=self.cmcmax_data,
            cfactr_data=self.cfactr_data,
            rsmax_data=self.rsmax_data,
            slop_data=self.slope_data,
            sbeta_data=self.sbeta_data,
            fxexp_data=self.fxexp_data,
            csoil_data=self.csoil_data,
            salp_data=self.salp_data,
            refdk_data=self.refdk_data,
            refkdt_data=self.refkdt_data,
            frzk_data=self.frzk_data,
            zbot_data=self.zbot_data,
            czil_data=self.czil_data,
            lvcoef_data=self.lvcoef_data,
            bb=self.bb,
            drysmc=self.drysmc,
            f11=self.f11,
            maxsmc=self.maxsmc,
            refsmc=self.refsmc,
            satpsi=self.satpsi,
            satdk=self.satdk,
            satdw=self.satdw,
            wltsmc=self.wltsmc,
            qtz=self.qtz,
        )


@dataclass(frozen=True)
class LsminitResult:
    vegfra: Any
    snow: Any
    snowc: Any
    snowh: Any
    canwat: Any
    smstav: Any
    smstot: Any
    sfcrunoff: Any
    udrunoff: Any
    acsnow: Any
    acsnom: Any
    snoalb: Any
    tslb: Any
    smois: Any
    sh2o: Any
    zs: Any
    dzs: Any
    irr_rand_field: Any
    parameters: Any
    messages: tuple[str, ...]


@dataclass(frozen=True)
class LsmResult:
    tsk: Any
    hfx: Any
    qfx: Any
    lh: Any
    grdflx: Any
    qsfc: Any
    cqs2: Any
    chs: Any
    chs2: Any
    snow: Any
    snowc: Any
    snowh: Any
    snotime: Any
    canwat: Any
    smstav: Any
    smstot: Any
    sfcrunoff: Any
    udrunoff: Any
    acsnom: Any
    acsnow: Any
    snopcx: Any
    emiss: Any
    rib: Any
    potevp: Any
    albedo: Any
    albbck: Any
    z0: Any
    znt: Any
    smois: Any
    sh2o: Any
    tslb: Any
    smcrel: Any
    flx4_2d: Any
    fvb_2d: Any
    fbur_2d: Any
    fgsn_2d: Any
    noahres: Any
    chklowq: Any
    lai: Any
    rc2: Any
    xlai2: Any


def _param_array(parameters: Any, name: str) -> np.ndarray:
    if isinstance(parameters, dict):
        value = parameters[name]
    else:
        value = getattr(parameters, name)
    return np.asarray(value, dtype=np.float32)


def _copy_array(value: Any, dtype: Any = np.float32) -> np.ndarray:
    return np.array(value, dtype=dtype, copy=True)


def _float_value(value: Any) -> np.float32:
    return np.asarray(value, dtype=np.float32).reshape(-1)[0].astype(np.float32)


def _array_value(value: Any, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


_SFLX_BATCH_FIELDS = (
    "xlai",
    "rc",
    "cmc",
    "sneqv",
    "snowh",
    "albedo",
    "alb",
    "z0brd",
    "emissi",
    "z0",
    "t1",
    "sheat",
    "etp",
    "eta_kinematic",
    "eta",
    "ssoil",
    "sncovr",
    "snotime1",
    "q1",
    "smc",
    "stc",
    "sh2o",
    "flx4",
    "fvb",
    "fbur",
    "fgsn",
    "flx1",
    "flx2",
    "flx3",
    "soilw",
    "soilm",
    "smav",
    "runoff1",
    "runoff2",
    "snomlt",
)
_SFLX_BATCH_CACHE: dict[tuple[Any, ...], Any] = {}
_NOAH_PARAMETER_SIGNATURE_CACHE: dict[int, tuple[Any, ...]] = {}
_JAX_NOAH_BATCH = os.environ.get("WRF_PY_JAX_NOAH_BATCH", "1").strip().lower() in {"1", "true", "yes", "on"}
_JIT_NOAH_SFLX_BATCH = os.environ.get("WRF_PY_JIT_NOAH_SFLX_BATCH", "1").strip().lower() in {"1", "true", "yes", "on"}
_INLINE_NOAH_SFLX_BATCH = os.environ.get("WRF_PY_INLINE_NOAH_SFLX_BATCH", "0").strip().lower() in {"1", "true", "yes", "on"}


def _contains_jax_tracer(value: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(value)
    return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def _noah_parameter_signature(parameters: NoahLsmParameters) -> tuple[Any, ...]:
    cached = _NOAH_PARAMETER_SIGNATURE_CACHE.get(id(parameters))
    if cached is not None:
        return cached
    values: list[Any] = []
    for name in parameters.__dataclass_fields__:
        value = getattr(parameters, name)
        if isinstance(value, (int, np.integer)):
            values.append((name, int(value)))
            continue
        arr = np.asarray(value)
        if arr.ndim == 0:
            if arr.dtype.kind in {"i", "u"}:
                values.append((name, int(arr)))
            else:
                values.append((name, float(np.asarray(arr, dtype=np.float32))))
            continue
        arr32 = np.asarray(arr, dtype=np.float32)
        values.append((name, arr32.shape, tuple(float(item) for item in arr32.reshape(-1))))
    signature = tuple(values)
    _NOAH_PARAMETER_SIGNATURE_CACHE[id(parameters)] = signature
    return signature


def _get_sflx_batch_fn(
    *,
    parameters: NoahLsmParameters,
    isurban: int,
    dt_value: np.float32,
    nsoil: int,
    sldpth: np.ndarray,
    mminlu: str,
    rdlai2d: bool,
    usemonalb: bool,
    opt_thcnd: int,
    jit_compile: bool,
) -> Any:
    sldpth_host = np.asarray(sldpth, dtype=np.float32)[: int(nsoil)]
    dt_host = np.asarray(dt_value, dtype=np.float32)
    key = (
        _noah_parameter_signature(parameters),
        int(isurban),
        float(dt_host),
        int(nsoil),
        tuple(float(x) for x in sldpth_host.reshape(-1)),
        str(mminlu),
        bool(rdlai2d),
        bool(usemonalb),
        int(opt_thcnd),
        bool(jit_compile),
    )

    sldpth_static = jnp.asarray(sldpth_host, dtype=jnp.float32)
    dt_static = jnp.asarray(dt_host, dtype=jnp.float32)
    cacheable = not _contains_jax_tracer((dt_static, sldpth_static))
    if cacheable:
        cached = _SFLX_BATCH_CACHE.get(key)
        if cached is not None:
            return cached

    def _one(
        f_frozp: Any,
        lwdn: Any,
        soldn: Any,
        solnet: Any,
        sfcprs: Any,
        prcp: Any,
        sfctmp: Any,
        q2: Any,
        th2: Any,
        q2sat: Any,
        dqsdt2: Any,
        shdfac: Any,
        shdmin: Any,
        shdmax: Any,
        alb: Any,
        snoalb: Any,
        tbot: Any,
        z0brd: Any,
        emissi: Any,
        cmc: Any,
        t1: Any,
        stc: Any,
        smc: Any,
        sh2o: Any,
        snowh: Any,
        sneqv: Any,
        ch: Any,
        sncovr: Any,
        xlai: Any,
        snotime1: Any,
        ribb: Any,
        irrigation_channel: Any,
        vegtyp: Any,
        soiltyp: Any,
        rounding_token: Any,
    ) -> tuple[Any, ...]:
        result = sflx(
            f_frozp=f_frozp,
            isurban=int(isurban),
            dt=dt_static,
            zlvl=jnp.asarray(0.0, dtype=jnp.float32),
            nsoil=int(nsoil),
            sldpth=sldpth_static,
            local=False,
            llanduse=str(mminlu),
            lsoil="STAS",
            lwdn=lwdn,
            soldn=soldn,
            solnet=solnet,
            sfcprs=sfcprs,
            prcp=prcp,
            sfctmp=sfctmp,
            q2=q2,
            sfcspd=jnp.asarray(0.0, dtype=jnp.float32),
            cosz=jnp.asarray(0.0, dtype=jnp.float32),
            prcprain=jnp.asarray(0.0, dtype=jnp.float32),
            solardirect=jnp.asarray(0.0, dtype=jnp.float32),
            th2=th2,
            q2sat=q2sat,
            dqsdt2=dqsdt2,
            vegtyp=vegtyp,
            soiltyp=soiltyp,
            slopetyp=1,
            shdfac=shdfac,
            shdmin=shdmin,
            shdmax=shdmax,
            alb=alb,
            snoalb=snoalb,
            tbot=tbot,
            z0brd=z0brd,
            z0=jnp.asarray(0.0, dtype=jnp.float32),
            emissi=emissi,
            cmc=cmc,
            t1=t1,
            stc=stc,
            smc=smc,
            sh2o=sh2o,
            snowh=snowh,
            sneqv=sneqv,
            albedo=jnp.asarray(0.0, dtype=jnp.float32),
            ch=ch,
            cm=jnp.asarray(0.0, dtype=jnp.float32),
            eta=jnp.asarray(0.0, dtype=jnp.float32),
            sheat=jnp.asarray(0.0, dtype=jnp.float32),
            eta_kinematic=jnp.asarray(0.0, dtype=jnp.float32),
            fdown=jnp.asarray(0.0, dtype=jnp.float32),
            ec=jnp.asarray(0.0, dtype=jnp.float32),
            edir=jnp.asarray(0.0, dtype=jnp.float32),
            et=jnp.zeros((int(nsoil),), dtype=jnp.float32),
            ett=jnp.asarray(0.0, dtype=jnp.float32),
            esnow=jnp.asarray(0.0, dtype=jnp.float32),
            drip=jnp.asarray(0.0, dtype=jnp.float32),
            dew=jnp.asarray(0.0, dtype=jnp.float32),
            beta=jnp.asarray(0.0, dtype=jnp.float32),
            etp=jnp.asarray(0.0, dtype=jnp.float32),
            ssoil=jnp.asarray(0.0, dtype=jnp.float32),
            flx1=jnp.asarray(0.0, dtype=jnp.float32),
            flx2=jnp.asarray(0.0, dtype=jnp.float32),
            flx3=jnp.asarray(0.0, dtype=jnp.float32),
            flx4=jnp.asarray(0.0, dtype=jnp.float32),
            fvb=jnp.asarray(0.0, dtype=jnp.float32),
            fbur=jnp.asarray(0.0, dtype=jnp.float32),
            fgsn=jnp.asarray(0.0, dtype=jnp.float32),
            ua_phys=False,
            snomlt=jnp.asarray(0.0, dtype=jnp.float32),
            sncovr=sncovr,
            runoff1=jnp.asarray(0.0, dtype=jnp.float32),
            runoff2=jnp.asarray(0.0, dtype=jnp.float32),
            runoff3=jnp.asarray(0.0, dtype=jnp.float32),
            rc=jnp.asarray(0.0, dtype=jnp.float32),
            pc=jnp.asarray(0.0, dtype=jnp.float32),
            rsmin=jnp.asarray(0.0, dtype=jnp.float32),
            xlai=xlai,
            rcs=jnp.asarray(0.0, dtype=jnp.float32),
            rct=jnp.asarray(0.0, dtype=jnp.float32),
            rcq=jnp.asarray(0.0, dtype=jnp.float32),
            rcsoil=jnp.asarray(0.0, dtype=jnp.float32),
            soilw=jnp.asarray(0.0, dtype=jnp.float32),
            soilm=jnp.asarray(0.0, dtype=jnp.float32),
            q1=jnp.asarray(0.0, dtype=jnp.float32),
            smav=jnp.zeros((int(nsoil),), dtype=jnp.float32),
            rdlai2d=bool(rdlai2d),
            usemonalb=bool(usemonalb),
            snotime1=snotime1,
            ribb=ribb,
            smcwlt=jnp.asarray(0.0, dtype=jnp.float32),
            smcdry=jnp.asarray(0.0, dtype=jnp.float32),
            smcref=jnp.asarray(0.0, dtype=jnp.float32),
            smcmax=jnp.asarray(0.0, dtype=jnp.float32),
            nroot=1,
            sfhead1rt=jnp.asarray(0.0, dtype=jnp.float32),
            infxs1rt=jnp.asarray(0.0, dtype=jnp.float32),
            etpnd1=jnp.asarray(0.0, dtype=jnp.float32),
            opt_thcnd=int(opt_thcnd),
            aoasis=jnp.asarray(1.0, dtype=jnp.float32),
            xsda_qfx=jnp.asarray(0.0, dtype=jnp.float32),
            hfx_phy=jnp.asarray(0.0, dtype=jnp.float32),
            qfx_phy=jnp.asarray(0.0, dtype=jnp.float32),
            xqnorm=jnp.asarray(0.0, dtype=jnp.float32),
            fasdas=0,
            hcpct_fasdas=jnp.asarray(0.0, dtype=jnp.float32),
            params=parameters,
            irrigation_channel=irrigation_channel,
            rounding_token=rounding_token,
        )
        return tuple(getattr(result, name) for name in _SFLX_BATCH_FIELDS)

    batched = jax.vmap(_one)
    cached = jax.jit(batched, inline=_INLINE_NOAH_SFLX_BATCH) if bool(jit_compile) else batched
    if cacheable:
        _SFLX_BATCH_CACHE[key] = cached
    return cached


def _all_nonpositive_host(value: Any) -> bool:
    return bool(np.all(np.asarray(value, dtype=np.float32) <= np.float32(0.0)))


def _all_zero_host(value: Any) -> bool:
    return bool(np.all(np.asarray(value, dtype=np.float32) == np.float32(0.0)))


def _store_wrf_real(value: Any, rounding_token: Any | None) -> Any:
    array = jnp.asarray(value, dtype=jnp.float32)
    if rounding_token is None:
        return array
    return optimization_barrier(real_assignment(array, rounding_token))


def _call_sflx_batch(batch_fn: Any, *args: Any) -> tuple[Any, ...]:
    """Keep the compiled SFLX column kernel distinct inside larger graphs."""

    if _contains_jax_tracer(args):
        return semantic_call(batch_fn, *args)
    return batch_fn(*args)


def _lsm_batch_jax(args: Mapping[str, Any]) -> LsmResult:
    """JAX-resident Noah land update for snow-free and snowy columns."""

    dtype = jnp.float32
    rounding_token = args.get("rounding_token")
    dz8w_arr = jnp.asarray(args["dz8w"], dtype=dtype)
    qv3d_arr = jnp.asarray(args["qv3d"], dtype=dtype)
    p8w3d_arr = jnp.asarray(args["p8w3d"], dtype=dtype)
    t3d_arr = jnp.asarray(args["t3d"], dtype=dtype)
    tsk_out = jnp.asarray(args["tsk"], dtype=dtype)
    hfx_out = jnp.asarray(args["hfx"], dtype=dtype)
    qfx_out = jnp.asarray(args["qfx"], dtype=dtype)
    lh_out = jnp.asarray(args["lh"], dtype=dtype)
    grdflx_out = jnp.asarray(args["grdflx"], dtype=dtype)
    qgh_arr = jnp.asarray(args["qgh"], dtype=dtype)
    swdown_arr = jnp.asarray(args["swdown"], dtype=dtype)
    glw_arr = jnp.asarray(args["glw"], dtype=dtype)
    smstav_out = jnp.asarray(args["smstav"], dtype=dtype)
    smstot_out = jnp.asarray(args["smstot"], dtype=dtype)
    sfcrunoff_out = jnp.asarray(args["sfcrunoff"], dtype=dtype)
    udrunoff_out = jnp.asarray(args["udrunoff"], dtype=dtype)
    ivgtyp_arr = jnp.asarray(args["ivgtyp"], dtype=jnp.int32)
    isltyp_arr = jnp.asarray(args["isltyp"], dtype=jnp.int32)
    vegfra_arr = jnp.asarray(args["vegfra"], dtype=dtype)
    albedo_out = jnp.asarray(args["albedo"], dtype=dtype)
    albbck_out = jnp.asarray(args["albbck"], dtype=dtype)
    znt_out = jnp.asarray(args["znt"], dtype=dtype)
    z0_out = jnp.asarray(args["z0"], dtype=dtype)
    tmn_arr = jnp.asarray(args["tmn"], dtype=dtype)
    xland_arr = jnp.asarray(args["xland"], dtype=dtype)
    xice_arr = jnp.asarray(args["xice"], dtype=dtype)
    emiss_out = jnp.asarray(args["emiss"], dtype=dtype)
    snowc_out = jnp.asarray(args["snowc"], dtype=dtype)
    qsfc_out = jnp.asarray(args["qsfc"], dtype=dtype)
    rainbl_arr = jnp.asarray(args["rainbl"], dtype=dtype)
    smois_out = jnp.asarray(args["smois"], dtype=dtype)
    tslb_out = jnp.asarray(args["tslb"], dtype=dtype)
    snow_out = jnp.asarray(args["snow"], dtype=dtype)
    canwat_out = jnp.asarray(args["canwat"], dtype=dtype)
    chs_out = jnp.asarray(args["chs"], dtype=dtype)
    chs2_out = jnp.asarray(args["chs2"], dtype=dtype)
    cqs2_out = jnp.asarray(args["cqs2"], dtype=dtype)
    qz0_arr = jnp.asarray(args["qz0"], dtype=dtype)
    sh2o_out = jnp.asarray(args["sh2o"], dtype=dtype)
    snowh_out = jnp.asarray(args["snowh"], dtype=dtype)
    snoalb_arr = jnp.asarray(args["snoalb"], dtype=dtype)
    shdmin_arr = jnp.asarray(args["shdmin"], dtype=dtype)
    shdmax_arr = jnp.asarray(args["shdmax"], dtype=dtype)
    snotime_out = jnp.asarray(args["snotime"], dtype=dtype)
    acsnom_out = jnp.asarray(args["acsnom"], dtype=dtype)
    acsnow_out = jnp.asarray(args["acsnow"], dtype=dtype)
    snopcx_out = jnp.asarray(args["snopcx"], dtype=dtype)
    potevp_out = jnp.asarray(args["potevp"], dtype=dtype)
    smcrel_out = jnp.asarray(args["smcrel"], dtype=dtype)
    rib_out = jnp.asarray(args["rib"], dtype=dtype)
    noahres_out = jnp.asarray(args["noahres"], dtype=dtype)
    flx4_out = jnp.asarray(args["flx4_2d"], dtype=dtype)
    fvb_out = jnp.asarray(args["fvb_2d"], dtype=dtype)
    fbur_out = jnp.asarray(args["fbur_2d"], dtype=dtype)
    fgsn_out = jnp.asarray(args["fgsn_2d"], dtype=dtype)
    lai_out = jnp.asarray(args["lai"], dtype=dtype)
    rc2_out = None if args.get("rc2") is None else jnp.asarray(args["rc2"], dtype=dtype)
    xlai2_out = None if args.get("xlai2") is None else jnp.asarray(args["xlai2"], dtype=dtype)

    nsoil = int(args["num_soil_layers"])
    dzs_arr = jnp.asarray(args["dzs"], dtype=dtype)
    sldpth = np.asarray(args["dzs"], dtype=np.float32)[:nsoil].copy()
    dt_value_np = np.float32(args["dt"])
    dt_value = jnp.asarray(dt_value_np, dtype=dtype)
    xice_limit = jnp.asarray(args["xice_threshold"], dtype=dtype)
    fdtliw = dt_value / jnp.asarray(1000.0 * 3.335e5, dtype=dtype)
    fdtw = dt_value / jnp.asarray(2.50e6 * 1000.0, dtype=dtype)
    capa = jnp.asarray(287.0 / 1004.5, dtype=dtype)
    tresh = jnp.asarray(0.95, dtype=dtype)
    a2 = jnp.asarray(17.67, dtype=dtype)
    a3 = jnp.asarray(273.15, dtype=dtype)
    a4 = jnp.asarray(29.65, dtype=dtype)
    a23m4 = a2 * (a3 - a4)
    stb = jnp.asarray(5.67e-8, dtype=dtype)

    i0 = int(args["its"]) - int(args["ims"])
    i1 = int(args["ite"]) - int(args["ims"]) + 1
    j0 = int(args["jts"]) - int(args["jms"])
    j1 = int(args["jte"]) - int(args["jms"]) + 1
    kk0 = int(args["kts"]) - int(args["kms"])
    kk1 = kk0 + 1
    sl2 = np.s_[i0:i1, j0:j1]
    sl3 = np.s_[i0:i1, :nsoil, j0:j1]
    ni = i1 - i0
    nj = j1 - j0
    ncol = ni * nj

    xland_tile = xland_arr[sl2]
    xice_tile = xice_arr[sl2]
    if int(args["itimestep"]) == 1:
        water_mask = xland_tile >= jnp.asarray(1.5, dtype=dtype)
        ice_init_mask = xice_tile >= xice_limit
        init_mask = water_mask | ice_init_mask
        smstav_out = smstav_out.at[sl2].set(jnp.where(init_mask, jnp.asarray(1.0, dtype=dtype), smstav_out[sl2]))
        smstot_out = smstot_out.at[sl2].set(jnp.where(init_mask, jnp.asarray(1.0, dtype=dtype), smstot_out[sl2]))
        smois_out = smois_out.at[sl3].set(jnp.where(init_mask[:, None, :], jnp.asarray(1.0, dtype=dtype), smois_out[sl3]))
        smcrel_out = smcrel_out.at[sl3].set(jnp.where(init_mask[:, None, :], jnp.asarray(1.0, dtype=dtype), smcrel_out[sl3]))
        tslb_out = tslb_out.at[sl3].set(jnp.where(water_mask[:, None, :], jnp.asarray(273.16, dtype=dtype), tslb_out[sl3]))

    land_mask = xland_tile < jnp.asarray(1.5, dtype=dtype)
    ice_mask = land_mask & (xice_tile >= xice_limit)
    glacial_mask = land_mask & ~ice_mask & (ivgtyp_arr[sl2] == int(args["isice"]))
    active_mask = land_mask & ~ice_mask & ~glacial_mask

    sh2o_out = sh2o_out.at[sl3].set(jnp.where(ice_mask[:, None, :], jnp.asarray(1.0, dtype=dtype), sh2o_out[sl3]))
    lai_out = lai_out.at[sl2].set(jnp.where(ice_mask, jnp.asarray(0.01, dtype=dtype), lai_out[sl2]))

    psfc = p8w3d_arr[i0:i1, kk0, j0:j1]
    sfcprs = (p8w3d_arr[i0:i1, kk1, j0:j1] + p8w3d_arr[i0:i1, kk0, j0:j1]) * jnp.asarray(0.5, dtype=dtype)
    q2k = qv3d_arr[i0:i1, kk0, j0:j1] / (jnp.asarray(1.0, dtype=dtype) + qv3d_arr[i0:i1, kk0, j0:j1])
    q2sat = qgh_arr[sl2] / (jnp.asarray(1.0, dtype=dtype) + qgh_arr[sl2])
    myj_lowq = (q2k >= q2sat * tresh) & (q2k < qz0_arr[sl2])
    chklowq_tile = jnp.where(bool(args["myj"]), jnp.where(myj_lowq, jnp.asarray(0.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype)), jnp.asarray(1.0, dtype=dtype))
    chklowq_out = jnp.asarray(args["chklowq"], dtype=dtype).at[sl2].set(chklowq_tile)

    sfctmp = t3d_arr[i0:i1, kk0, j0:j1]
    pressure_reference = _store_wrf_real(jnp.asarray(1.0e5, dtype=dtype), rounding_token)
    apes = _store_wrf_real(
        _store_wrf_real(pressure_reference / psfc, rounding_token) ** capa,
        rounding_token,
    )
    apelm = _store_wrf_real(
        _store_wrf_real(pressure_reference / sfcprs, rounding_token) ** capa,
        rounding_token,
    )
    th2 = _store_wrf_real(
        _store_wrf_real(sfctmp * apelm, rounding_token) / apes,
        rounding_token,
    )
    emissi = emiss_out[sl2]
    lwdn = glw_arr[sl2] * emissi
    soldn = swdown_arr[sl2]
    solnet = soldn * (jnp.asarray(1.0, dtype=dtype) - albedo_out[sl2])
    prcp = rainbl_arr[sl2] / dt_value
    if args.get("irr_chan") is None:
        irrigation_channel = jnp.zeros_like(prcp)
    else:
        irr_tile = jnp.asarray(args["irr_chan"], dtype=dtype)[sl2]
        irrigation_channel = jnp.where(irr_tile != jnp.asarray(0.0, dtype=dtype), irr_tile / dt_value, jnp.asarray(0.0, dtype=dtype))

    vegtyp = ivgtyp_arr[sl2]
    soiltyp = jnp.where((isltyp_arr[sl2] == 14) & (xice_tile == jnp.asarray(0.0, dtype=dtype)), 7, isltyp_arr[sl2])
    percent = _store_wrf_real(jnp.asarray(100.0, dtype=dtype), rounding_token)
    shdfac = _store_wrf_real(vegfra_arr[sl2] / percent, rounding_token)
    if int(args["isurban"]) == 1:
        urban_lu = (vegtyp == 25) | (vegtyp == 26) | (vegtyp == 27)
        shdfac = jnp.where(urban_lu, jnp.asarray(0.0, dtype=dtype), shdfac)
    shmin = _store_wrf_real(shdmin_arr[sl2] / percent, rounding_token)
    shmax = _store_wrf_real(shdmax_arr[sl2] / percent, rounding_token)
    sneqv = snow_out[sl2] * jnp.asarray(0.001, dtype=dtype)
    snowhk = jnp.where(((sneqv != 0.0) & (snowh_out[sl2] == 0.0)) | (snowh_out[sl2] <= sneqv), jnp.asarray(5.0, dtype=dtype) * sneqv, snowh_out[sl2])
    if bool(args["rdlai2d"]):
        lai_tile = jnp.where(land_mask & (shdfac > 0.0) & (lai_out[sl2] <= 0.0), jnp.asarray(0.01, dtype=dtype), lai_out[sl2])
        lai_out = lai_out.at[sl2].set(lai_tile)
    else:
        lai_tile = lai_out[sl2]
    ffrozp = jnp.where(sfctmp <= jnp.asarray(273.15, dtype=dtype), jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype))
    if bool(args["frpcpn"]):
        ffrozp = jnp.asarray(args["sr"], dtype=dtype)[sl2]
    dqsdt2_base = _store_wrf_real(sfctmp - a4, rounding_token)
    dqsdt2_exponent = _store_wrf_real(jnp.asarray(2.0, dtype=dtype), rounding_token)
    dqsdt2_denominator = _store_wrf_real(dqsdt2_base**dqsdt2_exponent, rounding_token)
    dqsdt2 = _store_wrf_real(
        _store_wrf_real(q2sat * a23m4, rounding_token) / dqsdt2_denominator,
        rounding_token,
    )

    # WRF LSM uses saturation over ice on cold snow, and snow-fraction
    # weighting near melting. Use incoming SNOWC, before this step's SFLX.
    snow_present = snow_out[sl2] > 0.0
    e2sat = _store_wrf_real(611.2 * jnp.exp(6174.0 * (1.0 / jnp.asarray(273.15, dtype=dtype) - 1.0 / sfctmp)), rounding_token)
    q2sati = _store_wrf_real(0.622 * e2sat / (sfcprs - e2sat), rounding_token)
    q2sati = _store_wrf_real(q2sati / (1.0 + q2sati), rounding_token)
    ice_derivative = _store_wrf_real(q2sati * 6174.0 / sfctmp**2.0, rounding_token)
    melting_surface = tsk_out[sl2] > 273.14
    cover = snowc_out[sl2]
    snow_qsat = jnp.where(melting_surface, q2sat * (1.0 - cover) + q2sati * cover, q2sati)
    snow_derivative = jnp.where(melting_surface, dqsdt2 * (1.0 - cover) + ice_derivative * cover, ice_derivative)
    melt_limit = (tsk_out[sl2] > 273.0) & (cover > 0.0) & (soldn > 10.0)
    snow_derivative = jnp.where(melt_limit, snow_derivative * (1.0 - cover), snow_derivative)
    q2sat = jnp.where(snow_present, snow_qsat, q2sat)
    dqsdt2 = jnp.where(snow_present, snow_derivative, dqsdt2)

    safe_active = active_mask.reshape((ncol,))
    update_diagnostics = bool(args.get("update_diagnostics", True))

    glacier_i, glacier_j = args.get("noah_glacial_i"), args.get("noah_glacial_j")
    if glacier_i is None or glacier_j is None:
        if isinstance(glacial_mask, jax.core.Tracer):
            local_i, local_j = jnp.indices((ni, nj))
            glacier_i, glacier_j = local_i.reshape(-1) + i0, local_j.reshape(-1) + j0
        else:
            local_i, local_j = np.nonzero(np.asarray(glacial_mask))
            glacier_i, glacier_j = local_i + i0, local_j + j0
    glacier_i, glacier_j = jnp.asarray(glacier_i), jnp.asarray(glacier_j)
    if glacier_i.size:
        gi, gj = glacier_i - i0, glacier_j - j0
        valid = glacial_mask[gi, gj]
        pick = lambda value: value[glacier_i, glacier_j]
        tile = lambda value: value[gi, gj]
        inputs = dict(
            dt=jnp.full(valid.shape, dt_value),
            sldpth=jnp.broadcast_to(dzs_arr[:nsoil], (valid.size, nsoil)),
            ffrozp=tile(ffrozp), lwdn=tile(lwdn), solnet=tile(solnet),
            sfcprs=tile(sfcprs), prcp=tile(prcp), sfctmp=tile(sfctmp),
            q2=tile(q2k), th2=tile(th2), q2sat=tile(q2sat), dqsdt2=tile(dqsdt2),
            alb=pick(albbck_out), snoalb=pick(snoalb_arr), tbot=pick(tmn_arr),
            z0brd=pick(z0_out), t1=pick(tsk_out),
            stc=tslb_out[glacier_i, :nsoil, glacier_j],
            snowh=tile(snowhk), sneqv=tile(sneqv),
            ch=jnp.where(valid, pick(chs_out), 0.01), sncovr=pick(snowc_out),
            snotime1=pick(snotime_out), ribb=pick(rib_out),
            lvcoef=jnp.full(valid.shape, args["parameters"].lvcoef_data),
        )
        glacier = semantic_call(glacial_columns, inputs) if _contains_jax_tracer(inputs) else glacial_columns(inputs)

        def put(value, updated):
            return value.at[glacier_i, glacier_j].set(jnp.where(valid, updated, pick(value)))

        tsk_out = put(tsk_out, glacier["t1"])
        hfx_out = put(hfx_out, glacier["sheat"])
        qfx_out = put(qfx_out, glacier["eta_kinematic"])
        lh_out = put(lh_out, glacier["eta"])
        grdflx_out = put(grdflx_out, glacier["ssoil"])
        qsfc_out = put(qsfc_out, glacier["q1"] / (1.0 - glacier["q1"]))
        chs2_out = put(chs2_out, pick(cqs2_out))
        snow_out = put(snow_out, glacier["sneqv"] * 1000.0)
        snowh_out = put(snowh_out, glacier["snowh"])
        snowc_out = put(snowc_out, glacier["sncovr"])
        snotime_out = put(snotime_out, glacier["snotime1"])
        albedo_out = put(albedo_out, glacier["albedo"])
        emiss_out = put(emiss_out, glacier["emissi"])
        znt_out = put(znt_out, glacier["z0"])
        lai_out = put(lai_out, 0.01)
        index = (glacier_i, slice(0, nsoil), glacier_j)
        tslb_out = tslb_out.at[index].set(jnp.where(valid[:, None], glacier["stc"], tslb_out[index]))
        smois_out = smois_out.at[index].set(jnp.where(valid[:, None], 1.0, smois_out[index]))
        sh2o_out = sh2o_out.at[index].set(jnp.where(valid[:, None], 1.0, sh2o_out[index]))
        if update_diagnostics:
            smstot_out = put(smstot_out, 0.0)
            smcrel_out = smcrel_out.at[index].set(jnp.where(valid[:, None], 1.0, smcrel_out[index]))
            if xlai2_out is not None:
                xlai2_out = put(xlai2_out, 0.01)
            potevp_out = put(potevp_out, pick(potevp_out) + glacier["etp"] * fdtw)
            sfcrunoff_out = put(sfcrunoff_out, pick(sfcrunoff_out) + glacier["runoff1"] * dt_value * 1000.0)
            acsnow_out = put(acsnow_out, pick(acsnow_out) + jnp.where(tile(ffrozp) > .5, tile(prcp) * dt_value, 0.0))
            acsnom_out = put(acsnom_out, pick(acsnom_out) + glacier["snomlt"] * 1000.0)
            snopcx_out = put(snopcx_out, pick(snopcx_out) - glacier["snomlt"] / fdtliw)
            noahres_out = put(noahres_out, tile(solnet) + tile(lwdn) - glacier["sheat"]
                              + glacier["ssoil"] - glacier["eta"]
                              - glacier["emissi"] * stb * glacier["t1"]**4
                              - glacier["flx1"] - glacier["flx2"] - glacier["flx3"])

    def flat2(value: Any) -> Any:
        return jnp.reshape(value, (ncol,))

    def flat_soil(value: Any) -> Any:
        return jnp.reshape(jnp.transpose(value, (0, 2, 1)), (ncol, nsoil))

    def tile2(value: Any) -> Any:
        return jnp.reshape(value, (ni, nj))

    def tile_soil(value: Any) -> Any:
        return jnp.transpose(jnp.reshape(value[:, :nsoil], (ni, nj, nsoil)), (0, 2, 1))

    vegtyp_safe = jnp.where(active_mask, vegtyp, jnp.ones_like(vegtyp))
    soiltyp_safe = jnp.where(active_mask, soiltyp, jnp.ones_like(soiltyp))
    canopy_water_scale = _store_wrf_real(
        jnp.asarray(1000.0, dtype=dtype),
        rounding_token,
    )
    batch_fn = _get_sflx_batch_fn(
        parameters=args["parameters"],
        isurban=int(args["isurban"]),
        dt_value=dt_value_np,
        nsoil=nsoil,
        sldpth=sldpth,
        mminlu=str(args["mminlu"]),
        rdlai2d=bool(args["rdlai2d"]),
        usemonalb=bool(args["usemonalb"]),
        opt_thcnd=int(args["opt_thcnd"]),
        jit_compile=_JIT_NOAH_SFLX_BATCH,
    )
    sflx_rounding_token = jnp.asarray(
        0 if rounding_token is None else rounding_token,
        dtype=jnp.uint32,
    )
    active_row_i_static = args.get("noah_active_row_i")
    active_row_j_static = args.get("noah_active_row_j")
    active_row_mask_static = args.get("noah_active_row_mask")
    if active_row_i_static is not None and active_row_j_static is not None and active_row_mask_static is not None:
        active_row_i = jnp.asarray(active_row_i_static, dtype=jnp.int32)
        active_row_j = jnp.asarray(active_row_j_static, dtype=jnp.int32)
        active_row_mask = jnp.asarray(active_row_mask_static, dtype=bool)
        active_local_i = active_row_i - int(i0)
        active_local_j = active_row_j - int(j0)
        pack_shape = active_row_mask.shape
        pack_count = int(pack_shape[0]) * int(pack_shape[1])

        def pack_tile2(value: Any) -> Any:
            return jnp.asarray(value)[active_local_i, active_local_j].reshape((pack_count,))

        def pack_full2(value: Any) -> Any:
            return jnp.asarray(value)[active_row_i, active_row_j].reshape((pack_count,))

        def pack_soil(value: Any) -> Any:
            return jnp.asarray(value)[active_row_i, :nsoil, active_row_j].reshape((pack_count, nsoil))

        valid_flat = active_row_mask.reshape((pack_count,))
        packed_vegtyp = jnp.where(valid_flat, pack_tile2(vegtyp), jnp.ones((pack_count,), dtype=jnp.int32)).astype(jnp.int32)
        packed_soiltyp = jnp.where(valid_flat, pack_tile2(soiltyp), jnp.ones((pack_count,), dtype=jnp.int32)).astype(jnp.int32)
        output_tuple = _call_sflx_batch(
            batch_fn,
            pack_tile2(ffrozp),
            pack_tile2(lwdn),
            pack_tile2(soldn),
            pack_tile2(solnet),
            pack_tile2(sfcprs),
            pack_tile2(prcp),
            pack_tile2(sfctmp),
            pack_tile2(q2k),
            pack_tile2(th2),
            pack_tile2(q2sat),
            pack_tile2(dqsdt2),
            pack_tile2(shdfac),
            pack_tile2(shmin),
            pack_tile2(shmax),
            pack_full2(albbck_out),
            pack_full2(snoalb_arr),
            pack_full2(tmn_arr),
            pack_full2(z0_out),
            pack_tile2(emissi),
            _store_wrf_real(
                pack_full2(canwat_out) / canopy_water_scale,
                rounding_token,
            ),
            pack_full2(tsk_out),
            pack_soil(tslb_out),
            pack_soil(smois_out),
            pack_soil(sh2o_out),
            pack_tile2(snowhk),
            pack_tile2(sneqv),
            pack_full2(chs_out),
            pack_full2(snowc_out),
            pack_tile2(lai_tile),
            pack_full2(snotime_out),
            pack_full2(rib_out),
            pack_tile2(irrigation_channel),
            packed_vegtyp,
            packed_soiltyp,
            jnp.broadcast_to(sflx_rounding_token, (pack_count,)),
        )
        outputs = {name: value for name, value in zip(_SFLX_BATCH_FIELDS, output_tuple)}

        def reshape2(value: Any) -> Any:
            return jnp.asarray(value).reshape(pack_shape)

        def reshape_soil(value: Any) -> Any:
            return jnp.asarray(value).reshape((*pack_shape, nsoil))

        def scatter2(base: Any, value: Any) -> Any:
            base_arr = jnp.asarray(base)
            old = base_arr[active_row_i, active_row_j]
            merged = jnp.where(active_row_mask, reshape2(value), old)
            return base_arr.at[active_row_i, active_row_j].set(merged)

        def add2(base: Any, value: Any) -> Any:
            return jnp.asarray(base).at[active_row_i, active_row_j].add(jnp.where(active_row_mask, reshape2(value), jnp.asarray(0.0, dtype=dtype)))

        def scatter_soil(base: Any, value: Any) -> Any:
            base_arr = jnp.asarray(base)
            old = base_arr[active_row_i, :nsoil, active_row_j]
            merged = jnp.where(active_row_mask[..., None], reshape_soil(value[:, :nsoil]), old)
            return base_arr.at[active_row_i, :nsoil, active_row_j].set(merged)

        if update_diagnostics:
            if rc2_out is not None and xlai2_out is not None:
                rc2_out = scatter2(rc2_out, outputs["rc"])
                xlai2_out = scatter2(xlai2_out, outputs["xlai"])
        lai_out = scatter2(lai_out, outputs["xlai"])
        canwat_out = scatter2(
            canwat_out,
            _store_wrf_real(outputs["cmc"] * canopy_water_scale, rounding_token),
        )
        snow_updated = outputs["sneqv"] * jnp.asarray(1000.0, dtype=dtype)
        snow_out = scatter2(snow_out, snow_updated)
        snowh_out = scatter2(snowh_out, outputs["snowh"])
        albedo_out = scatter2(albedo_out, outputs["albedo"])
        albbck_out = scatter2(albbck_out, outputs["alb"])
        z0_out = scatter2(z0_out, outputs["z0brd"])
        emiss_out = scatter2(emiss_out, outputs["emissi"])
        znt_out = scatter2(znt_out, outputs["z0"])
        tsk_out = scatter2(tsk_out, outputs["t1"])
        hfx_out = scatter2(hfx_out, outputs["sheat"])
        qfx_out = scatter2(qfx_out, outputs["eta_kinematic"])
        lh_out = scatter2(lh_out, outputs["eta"])
        grdflx_out = scatter2(grdflx_out, outputs["ssoil"])
        snowc_out = scatter2(snowc_out, outputs["sncovr"])
        chs2_out = scatter2(chs2_out, pack_full2(cqs2_out))
        snotime_out = scatter2(snotime_out, outputs["snotime1"])
        q1 = outputs["q1"]
        qsfc_out = scatter2(qsfc_out, q1 / (jnp.asarray(1.0, dtype=dtype) - q1))
        smois_out = scatter_soil(smois_out, outputs["smc"])
        tslb_out = scatter_soil(tslb_out, outputs["stc"])
        sh2o_out = scatter_soil(sh2o_out, outputs["sh2o"])
        if update_diagnostics:
            potevp_out = add2(potevp_out, outputs["etp"] * fdtw)
            flx4_out = scatter2(flx4_out, outputs["flx4"])
            fvb_out = scatter2(fvb_out, outputs["fvb"])
            fbur_out = scatter2(fbur_out, outputs["fbur"])
            fgsn_out = scatter2(fgsn_out, outputs["fgsn"])
            noahres_active = (
                (pack_tile2(solnet) + pack_tile2(lwdn))
                - outputs["sheat"]
                + outputs["ssoil"]
                - outputs["eta"]
                - (outputs["emissi"] * stb * (outputs["t1"] ** jnp.asarray(4.0, dtype=dtype)))
                - outputs["flx1"]
                - outputs["flx2"]
                - outputs["flx3"]
            )
            noahres_out = scatter2(noahres_out, noahres_active)
            smstav_out = scatter2(smstav_out, outputs["soilw"])
            smstot_out = scatter2(smstot_out, outputs["soilm"] * jnp.asarray(1000.0, dtype=dtype))
            smcrel_out = scatter_soil(smcrel_out, outputs["smav"])
            sfcrunoff_out = add2(sfcrunoff_out, outputs["runoff1"] * dt_value * jnp.asarray(1000.0, dtype=dtype))
            udrunoff_out = add2(udrunoff_out, outputs["runoff2"] * dt_value * jnp.asarray(1000.0, dtype=dtype))
            acsnow_out = add2(acsnow_out, jnp.where(pack_tile2(ffrozp) > 0.5, pack_tile2(prcp) * dt_value, jnp.asarray(0.0, dtype=dtype)))
            snow_active = snow_updated > 0.0
            acsnom_out = add2(acsnom_out, jnp.where(snow_active, outputs["snomlt"] * jnp.asarray(1000.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)))
            snopcx_out = add2(snopcx_out, jnp.where(snow_active, -outputs["snomlt"] / fdtliw, jnp.asarray(0.0, dtype=dtype)))
        del dz8w_arr, safe_active, dzs_arr

        return LsmResult(
            snotime=snotime_out,
            tsk=tsk_out,
            hfx=hfx_out,
            qfx=qfx_out,
            lh=lh_out,
            grdflx=grdflx_out,
            qsfc=qsfc_out,
            cqs2=cqs2_out,
            chs=chs_out,
            chs2=chs2_out,
            snow=snow_out,
            snowc=snowc_out,
            snowh=snowh_out,
            canwat=canwat_out,
            smstav=smstav_out,
            smstot=smstot_out,
            sfcrunoff=sfcrunoff_out,
            udrunoff=udrunoff_out,
            acsnom=acsnom_out,
            acsnow=acsnow_out,
            snopcx=snopcx_out,
            emiss=emiss_out,
            rib=rib_out,
            potevp=potevp_out,
            albedo=albedo_out,
            albbck=albbck_out,
            z0=z0_out,
            znt=znt_out,
            smois=smois_out,
            sh2o=sh2o_out,
            tslb=tslb_out,
            smcrel=smcrel_out,
            flx4_2d=flx4_out,
            fvb_2d=fvb_out,
            fbur_2d=fbur_out,
            fgsn_2d=fgsn_out,
            noahres=noahres_out,
            chklowq=chklowq_out,
            lai=lai_out,
            rc2=rc2_out,
            xlai2=xlai2_out,
        )

    active_i_static = args.get("noah_active_i")
    active_j_static = args.get("noah_active_j")
    if active_i_static is not None and active_j_static is not None:
        active_i = jnp.asarray(active_i_static, dtype=jnp.int32)
        active_j = jnp.asarray(active_j_static, dtype=jnp.int32)
        active_local_i = active_i - int(i0)
        active_local_j = active_j - int(j0)

        def pick2(value: Any) -> Any:
            return jnp.asarray(value)[active_local_i, active_local_j]

        def pick2_full(value: Any) -> Any:
            return jnp.asarray(value)[active_i, active_j]

        def pick_soil(value: Any) -> Any:
            return jnp.asarray(value)[active_i, :nsoil, active_j]

        output_tuple = _call_sflx_batch(
            batch_fn,
            pick2(ffrozp),
            pick2(lwdn),
            pick2(soldn),
            pick2(solnet),
            pick2(sfcprs),
            pick2(prcp),
            pick2(sfctmp),
            pick2(q2k),
            pick2(th2),
            pick2(q2sat),
            pick2(dqsdt2),
            pick2(shdfac),
            pick2(shmin),
            pick2(shmax),
            pick2_full(albbck_out),
            pick2_full(snoalb_arr),
            pick2_full(tmn_arr),
            pick2_full(z0_out),
            pick2(emissi),
            _store_wrf_real(
                pick2_full(canwat_out) / canopy_water_scale,
                rounding_token,
            ),
            pick2_full(tsk_out),
            pick_soil(tslb_out),
            pick_soil(smois_out),
            pick_soil(sh2o_out),
            pick2(snowhk),
            pick2(sneqv),
            pick2_full(chs_out),
            pick2_full(snowc_out),
            pick2(lai_tile),
            pick2_full(snotime_out),
            pick2_full(rib_out),
            pick2(irrigation_channel),
            pick2(vegtyp).astype(jnp.int32),
            pick2(soiltyp).astype(jnp.int32),
            jnp.broadcast_to(sflx_rounding_token, pick2(ffrozp).shape),
        )
        outputs = {name: value for name, value in zip(_SFLX_BATCH_FIELDS, output_tuple)}

        def scatter2(base: Any, value: Any) -> Any:
            return jnp.asarray(base).at[active_i, active_j].set(value)

        def add2(base: Any, value: Any) -> Any:
            return jnp.asarray(base).at[active_i, active_j].add(value)

        def scatter_soil(base: Any, value: Any) -> Any:
            return jnp.asarray(base).at[active_i, :nsoil, active_j].set(value[:, :nsoil])

        if update_diagnostics:
            if rc2_out is not None and xlai2_out is not None:
                rc2_out = scatter2(rc2_out, outputs["rc"])
                xlai2_out = scatter2(xlai2_out, outputs["xlai"])
        lai_out = scatter2(lai_out, outputs["xlai"])
        canwat_out = scatter2(
            canwat_out,
            _store_wrf_real(outputs["cmc"] * canopy_water_scale, rounding_token),
        )
        snow_updated = outputs["sneqv"] * jnp.asarray(1000.0, dtype=dtype)
        snow_out = scatter2(snow_out, snow_updated)
        snowh_out = scatter2(snowh_out, outputs["snowh"])
        albedo_out = scatter2(albedo_out, outputs["albedo"])
        albbck_out = scatter2(albbck_out, outputs["alb"])
        z0_out = scatter2(z0_out, outputs["z0brd"])
        emiss_out = scatter2(emiss_out, outputs["emissi"])
        znt_out = scatter2(znt_out, outputs["z0"])
        tsk_out = scatter2(tsk_out, outputs["t1"])
        hfx_out = scatter2(hfx_out, outputs["sheat"])
        qfx_out = scatter2(qfx_out, outputs["eta_kinematic"])
        lh_out = scatter2(lh_out, outputs["eta"])
        grdflx_out = scatter2(grdflx_out, outputs["ssoil"])
        snowc_out = scatter2(snowc_out, outputs["sncovr"])
        chs2_out = scatter2(chs2_out, pick2_full(cqs2_out))
        snotime_out = scatter2(snotime_out, outputs["snotime1"])
        q1 = outputs["q1"]
        qsfc_out = scatter2(qsfc_out, q1 / (jnp.asarray(1.0, dtype=dtype) - q1))
        smois_out = scatter_soil(smois_out, outputs["smc"])
        tslb_out = scatter_soil(tslb_out, outputs["stc"])
        sh2o_out = scatter_soil(sh2o_out, outputs["sh2o"])
        if update_diagnostics:
            potevp_out = add2(potevp_out, outputs["etp"] * fdtw)
            flx4_out = scatter2(flx4_out, outputs["flx4"])
            fvb_out = scatter2(fvb_out, outputs["fvb"])
            fbur_out = scatter2(fbur_out, outputs["fbur"])
            fgsn_out = scatter2(fgsn_out, outputs["fgsn"])
            noahres_active = (
                (pick2(solnet) + pick2(lwdn))
                - outputs["sheat"]
                + outputs["ssoil"]
                - outputs["eta"]
                - (outputs["emissi"] * stb * (outputs["t1"] ** jnp.asarray(4.0, dtype=dtype)))
                - outputs["flx1"]
                - outputs["flx2"]
                - outputs["flx3"]
            )
            noahres_out = scatter2(noahres_out, noahres_active)
            smstav_out = scatter2(smstav_out, outputs["soilw"])
            smstot_out = scatter2(smstot_out, outputs["soilm"] * jnp.asarray(1000.0, dtype=dtype))
            smcrel_out = scatter_soil(smcrel_out, outputs["smav"])
            sfcrunoff_out = add2(sfcrunoff_out, outputs["runoff1"] * dt_value * jnp.asarray(1000.0, dtype=dtype))
            udrunoff_out = add2(udrunoff_out, outputs["runoff2"] * dt_value * jnp.asarray(1000.0, dtype=dtype))
            acsnow_out = add2(acsnow_out, jnp.where(pick2(ffrozp) > 0.5, pick2(prcp) * dt_value, jnp.asarray(0.0, dtype=dtype)))
            snow_active = snow_updated > 0.0
            acsnom_out = add2(acsnom_out, jnp.where(snow_active, outputs["snomlt"] * jnp.asarray(1000.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)))
            snopcx_out = add2(snopcx_out, jnp.where(snow_active, -outputs["snomlt"] / fdtliw, jnp.asarray(0.0, dtype=dtype)))
        del dz8w_arr, safe_active, dzs_arr

        return LsmResult(
            snotime=snotime_out,
            tsk=tsk_out,
            hfx=hfx_out,
            qfx=qfx_out,
            lh=lh_out,
            grdflx=grdflx_out,
            qsfc=qsfc_out,
            cqs2=cqs2_out,
            chs=chs_out,
            chs2=chs2_out,
            snow=snow_out,
            snowc=snowc_out,
            snowh=snowh_out,
            canwat=canwat_out,
            smstav=smstav_out,
            smstot=smstot_out,
            sfcrunoff=sfcrunoff_out,
            udrunoff=udrunoff_out,
            acsnom=acsnom_out,
            acsnow=acsnow_out,
            snopcx=snopcx_out,
            emiss=emiss_out,
            rib=rib_out,
            potevp=potevp_out,
            albedo=albedo_out,
            albbck=albbck_out,
            z0=z0_out,
            znt=znt_out,
            smois=smois_out,
            sh2o=sh2o_out,
            tslb=tslb_out,
            smcrel=smcrel_out,
            flx4_2d=flx4_out,
            fvb_2d=fvb_out,
            fbur_2d=fbur_out,
            fgsn_2d=fgsn_out,
            noahres=noahres_out,
            chklowq=chklowq_out,
            lai=lai_out,
            rc2=rc2_out,
            xlai2=xlai2_out,
        )

    output_tuple = _call_sflx_batch(
        batch_fn,
        flat2(ffrozp),
        flat2(lwdn),
        flat2(soldn),
        flat2(solnet),
        flat2(sfcprs),
        flat2(prcp),
        flat2(sfctmp),
        flat2(q2k),
        flat2(th2),
        flat2(q2sat),
        flat2(dqsdt2),
        flat2(shdfac),
        flat2(shmin),
        flat2(shmax),
        flat2(albbck_out[sl2]),
        flat2(snoalb_arr[sl2]),
        flat2(tmn_arr[sl2]),
        flat2(z0_out[sl2]),
        flat2(emissi),
        flat2(
            _store_wrf_real(
                canwat_out[sl2] / canopy_water_scale,
                rounding_token,
            )
        ),
        flat2(tsk_out[sl2]),
        flat_soil(tslb_out[sl3]),
        flat_soil(smois_out[sl3]),
        flat_soil(sh2o_out[sl3]),
        flat2(snowhk),
        flat2(sneqv),
        flat2(chs_out[sl2]),
        flat2(snowc_out[sl2]),
        flat2(lai_tile),
        flat2(snotime_out[sl2]),
        flat2(rib_out[sl2]),
        flat2(irrigation_channel),
        flat2(vegtyp_safe).astype(jnp.int32),
        flat2(soiltyp_safe).astype(jnp.int32),
        jnp.broadcast_to(sflx_rounding_token, flat2(ffrozp).shape),
    )
    outputs = {name: value for name, value in zip(_SFLX_BATCH_FIELDS, output_tuple)}

    def merge2(base: Any, value: Any) -> Any:
        tile = jnp.asarray(base)[sl2]
        return jnp.asarray(base).at[sl2].set(jnp.where(active_mask, tile2(value), tile))

    def merge_soil(base: Any, value: Any) -> Any:
        tile = jnp.asarray(base)[sl3]
        return jnp.asarray(base).at[sl3].set(jnp.where(active_mask[:, None, :], tile_soil(value), tile))

    if update_diagnostics:
        if rc2_out is not None and xlai2_out is not None:
            rc2_out = merge2(rc2_out, outputs["rc"])
            xlai2_out = merge2(xlai2_out, outputs["xlai"])
    lai_out = merge2(lai_out, outputs["xlai"])
    canwat_out = merge2(
        canwat_out,
        _store_wrf_real(outputs["cmc"] * canopy_water_scale, rounding_token),
    )
    snow_updated_tile = tile2(outputs["sneqv"] * jnp.asarray(1000.0, dtype=dtype))
    snow_out = snow_out.at[sl2].set(jnp.where(active_mask, snow_updated_tile, snow_out[sl2]))
    snowh_out = merge2(snowh_out, outputs["snowh"])
    albedo_out = merge2(albedo_out, outputs["albedo"])
    albbck_out = merge2(albbck_out, outputs["alb"])
    z0_out = merge2(z0_out, outputs["z0brd"])
    emiss_out = merge2(emiss_out, outputs["emissi"])
    znt_out = merge2(znt_out, outputs["z0"])
    tsk_out = merge2(tsk_out, outputs["t1"])
    hfx_out = merge2(hfx_out, outputs["sheat"])
    qfx_out = merge2(qfx_out, outputs["eta_kinematic"])
    lh_out = merge2(lh_out, outputs["eta"])
    grdflx_out = merge2(grdflx_out, outputs["ssoil"])
    snowc_out = merge2(snowc_out, outputs["sncovr"])
    chs2_out = chs2_out.at[sl2].set(jnp.where(active_mask, cqs2_out[sl2], chs2_out[sl2]))
    snotime_out = merge2(snotime_out, outputs["snotime1"])
    q1_tile = tile2(outputs["q1"])
    qsfc_out = qsfc_out.at[sl2].set(jnp.where(active_mask, q1_tile / (jnp.asarray(1.0, dtype=dtype) - q1_tile), qsfc_out[sl2]))
    smois_out = merge_soil(smois_out, outputs["smc"])
    tslb_out = merge_soil(tslb_out, outputs["stc"])
    sh2o_out = merge_soil(sh2o_out, outputs["sh2o"])
    if update_diagnostics:
        potevp_out = potevp_out.at[sl2].set(jnp.where(active_mask, potevp_out[sl2] + tile2(outputs["etp"]) * fdtw, potevp_out[sl2]))
        flx4_out = merge2(flx4_out, outputs["flx4"])
        fvb_out = merge2(fvb_out, outputs["fvb"])
        fbur_out = merge2(fbur_out, outputs["fbur"])
        fgsn_out = merge2(fgsn_out, outputs["fgsn"])
        noahres_tile = (
            (solnet + lwdn)
            - tile2(outputs["sheat"])
            + tile2(outputs["ssoil"])
            - tile2(outputs["eta"])
            - (tile2(outputs["emissi"]) * stb * (tile2(outputs["t1"]) ** jnp.asarray(4.0, dtype=dtype)))
            - tile2(outputs["flx1"])
            - tile2(outputs["flx2"])
            - tile2(outputs["flx3"])
        )
        noahres_out = noahres_out.at[sl2].set(jnp.where(active_mask, noahres_tile, noahres_out[sl2]))
        smstav_out = merge2(smstav_out, outputs["soilw"])
        smstot_out = merge2(smstot_out, outputs["soilm"] * jnp.asarray(1000.0, dtype=dtype))
        smcrel_out = merge_soil(smcrel_out, outputs["smav"])
        sfcrunoff_out = sfcrunoff_out.at[sl2].set(jnp.where(active_mask, sfcrunoff_out[sl2] + tile2(outputs["runoff1"]) * dt_value * jnp.asarray(1000.0, dtype=dtype), sfcrunoff_out[sl2]))
        udrunoff_out = udrunoff_out.at[sl2].set(jnp.where(active_mask, udrunoff_out[sl2] + tile2(outputs["runoff2"]) * dt_value * jnp.asarray(1000.0, dtype=dtype), udrunoff_out[sl2]))
        acsnow_out = acsnow_out.at[sl2].set(
            jnp.where(active_mask & (ffrozp > 0.5), acsnow_out[sl2] + prcp * dt_value, acsnow_out[sl2])
        )
        snow_mask = active_mask & (snow_updated_tile > 0.0)
        acsnom_out = acsnom_out.at[sl2].set(jnp.where(snow_mask, acsnom_out[sl2] + tile2(outputs["snomlt"]) * jnp.asarray(1000.0, dtype=dtype), acsnom_out[sl2]))
        snopcx_out = snopcx_out.at[sl2].set(jnp.where(snow_mask, snopcx_out[sl2] - tile2(outputs["snomlt"]) / fdtliw, snopcx_out[sl2]))
    del dz8w_arr, safe_active, dzs_arr

    return LsmResult(
        snotime=snotime_out,
        tsk=tsk_out,
        hfx=hfx_out,
        qfx=qfx_out,
        lh=lh_out,
        grdflx=grdflx_out,
        qsfc=qsfc_out,
        cqs2=cqs2_out,
        chs=chs_out,
        chs2=chs2_out,
        snow=snow_out,
        snowc=snowc_out,
        snowh=snowh_out,
        canwat=canwat_out,
        smstav=smstav_out,
        smstot=smstot_out,
        sfcrunoff=sfcrunoff_out,
        udrunoff=udrunoff_out,
        acsnom=acsnom_out,
        acsnow=acsnow_out,
        snopcx=snopcx_out,
        emiss=emiss_out,
        rib=rib_out,
        potevp=potevp_out,
        albedo=albedo_out,
        albbck=albbck_out,
        z0=z0_out,
        znt=znt_out,
        smois=smois_out,
        sh2o=sh2o_out,
        tslb=tslb_out,
        smcrel=smcrel_out,
        flx4_2d=flx4_out,
        fvb_2d=fvb_out,
        fbur_2d=fbur_out,
        fgsn_2d=fgsn_out,
        noahres=noahres_out,
        chklowq=chklowq_out,
        lai=lai_out,
        rc2=rc2_out,
        xlai2=xlai2_out,
    )


def lsm(
    *,
    dz8w: Any,
    qv3d: Any,
    p8w3d: Any,
    t3d: Any,
    tsk: Any,
    hfx: Any,
    qfx: Any,
    lh: Any,
    grdflx: Any,
    qgh: Any,
    gsw: Any,
    swdown: Any,
    swddir: Any,
    swddif: Any,
    glw: Any,
    smstav: Any,
    smstot: Any,
    sfcrunoff: Any,
    udrunoff: Any,
    ivgtyp: Any,
    isltyp: Any,
    isurban: int,
    isice: int,
    vegfra: Any,
    albedo: Any,
    albbck: Any,
    znt: Any,
    z0: Any,
    tmn: Any,
    xland: Any,
    xice: Any,
    emiss: Any,
    embck: Any,
    snowc: Any,
    qsfc: Any,
    rainbl: Any,
    mminlu: str,
    num_soil_layers: int,
    dt: Any,
    dzs: Any,
    itimestep: int,
    smois: Any,
    tslb: Any,
    snow: Any,
    canwat: Any,
    chs: Any,
    chs2: Any,
    cqs2: Any,
    cpm: Any,
    rovcp: Any,
    sr: Any,
    chklowq: Any,
    lai: Any,
    qz0: Any,
    myj: bool,
    frpcpn: bool,
    sh2o: Any,
    snowh: Any,
    snoalb: Any,
    shdmin: Any,
    shdmax: Any,
    snotime: Any,
    acsnom: Any,
    acsnow: Any,
    snopcx: Any,
    potevp: Any,
    smcrel: Any,
    xice_threshold: Any,
    rdlai2d: bool,
    usemonalb: bool,
    rib: Any,
    noahres: Any,
    opt_thcnd: int,
    ua_phys: bool,
    flx4_2d: Any,
    fvb_2d: Any,
    fbur_2d: Any,
    fgsn_2d: Any,
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
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    sf_urban_physics: int = 0,
    rc2: Any | None = None,
    xlai2: Any | None = None,
    fasdas: int = 0,
    sda_hfx: Any | None = None,
    sda_qfx: Any | None = None,
    qnorm: Any | None = None,
    rho: Any | None = None,
    irr_chan: Any | None = None,
    noah_active_row_i: Any | None = None,
    noah_active_row_j: Any | None = None,
    noah_active_row_mask: Any | None = None,
    noah_active_i: Any | None = None,
    noah_active_j: Any | None = None,
    noah_glacial_i: Any | None = None,
    noah_glacial_j: Any | None = None,
    parameters: NoahLsmParameters | None = None,
    sflx_fn: Any | None = None,
    update_diagnostics: bool = True,
    rounding_token: Any | None = None,
) -> LsmResult:
    """Run the translated WRF Noah LSM driver for the non-urban Noah path.

    The default JAX batch handles ordinary land and permanent land ice.
    Urban canopy and FASDAS are not implemented in this driver.
    """

    del gsw, swddir, swddif, ids, jds, kds, kde, kte
    if int(sf_urban_physics) != 0:
        raise NotImplementedError("lsm urban canopy branches are not translated yet")
    if int(fasdas) != 0:
        raise NotImplementedError("lsm FASDAS branch is not translated yet")
    if parameters is None:
        raise ValueError("lsm requires Noah table parameters for SFLX")

    sflx_call = sflx if sflx_fn is None else sflx_fn
    if (
        _JAX_NOAH_BATCH
        and sflx_fn is None
        and not bool(ua_phys)
        and int(fasdas) == 0
    ):
        return _lsm_batch_jax(locals())

    dtype = np.float32
    dz8w_arr = _copy_array(dz8w)
    qv3d_arr = _copy_array(qv3d)
    p8w3d_arr = _copy_array(p8w3d)
    t3d_arr = _copy_array(t3d)
    tsk_out = _copy_array(tsk)
    hfx_out = _copy_array(hfx)
    qfx_out = _copy_array(qfx)
    lh_out = _copy_array(lh)
    grdflx_out = _copy_array(grdflx)
    qgh_arr = _copy_array(qgh)
    swdown_arr = _copy_array(swdown)
    glw_arr = _copy_array(glw)
    smstav_out = _copy_array(smstav)
    smstot_out = _copy_array(smstot)
    sfcrunoff_out = _copy_array(sfcrunoff)
    udrunoff_out = _copy_array(udrunoff)
    ivgtyp_arr = np.asarray(ivgtyp, dtype=np.int32)
    isltyp_arr = np.asarray(isltyp, dtype=np.int32)
    vegfra_arr = _copy_array(vegfra)
    albedo_out = _copy_array(albedo)
    albbck_out = _copy_array(albbck)
    znt_out = _copy_array(znt)
    z0_out = _copy_array(z0)
    tmn_arr = _copy_array(tmn)
    xland_arr = _copy_array(xland)
    xice_arr = _copy_array(xice)
    emiss_out = _copy_array(emiss)
    embck_arr = _copy_array(embck)
    snowc_out = _copy_array(snowc)
    qsfc_out = _copy_array(qsfc)
    rainbl_arr = _copy_array(rainbl)
    smois_out = _copy_array(smois)
    tslb_out = _copy_array(tslb)
    snow_out = _copy_array(snow)
    canwat_out = _copy_array(canwat)
    chs_out = _copy_array(chs)
    chs2_out = _copy_array(chs2)
    cqs2_out = _copy_array(cqs2)
    cpm_arr = _copy_array(cpm)
    sr_arr = _copy_array(sr)
    chklowq_out = _copy_array(chklowq)
    lai_out = _copy_array(lai)
    qz0_arr = _copy_array(qz0)
    sh2o_out = _copy_array(sh2o)
    snowh_out = _copy_array(snowh)
    snoalb_arr = _copy_array(snoalb)
    shdmin_arr = _copy_array(shdmin)
    shdmax_arr = _copy_array(shdmax)
    snotime_out = _copy_array(snotime)
    acsnom_out = _copy_array(acsnom)
    acsnow_out = _copy_array(acsnow)
    snopcx_out = _copy_array(snopcx)
    potevp_out = _copy_array(potevp)
    smcrel_out = _copy_array(smcrel)
    rib_out = _copy_array(rib)
    noahres_out = _copy_array(noahres)
    flx4_out = _copy_array(flx4_2d)
    fvb_out = _copy_array(fvb_2d)
    fbur_out = _copy_array(fbur_2d)
    fgsn_out = _copy_array(fgsn_2d)
    rc2_out = None if rc2 is None else _copy_array(rc2)
    xlai2_out = None if xlai2 is None else _copy_array(xlai2)

    dzs_arr = np.asarray(dzs, dtype=dtype)
    nsoil = int(num_soil_layers)
    sldpth = dzs_arr[:nsoil].copy()
    dt_value = np.float32(dt)
    xice_limit = np.float32(xice_threshold)
    rowliw = np.float32(1000.0 * 3.335e5)
    fdtliw = dt_value / rowliw
    fdtw = dt_value / np.float32(2.50e6 * 1000.0)
    capa = np.float32(287.0 / 1004.5)
    tresh = np.float32(0.95)
    a2 = np.float32(17.67)
    a3 = np.float32(273.15)
    a4 = np.float32(29.65)
    a23m4 = a2 * (a3 - a4)
    stb = np.float32(5.67e-8)
    batch_no_snow = (
        sflx_fn is None
        and not bool(ua_phys)
        and int(fasdas) == 0
        and not bool(frpcpn)
        and np.all(snow_out <= np.float32(0.0))
        and np.all(snowh_out <= np.float32(0.0))
        and np.all(snowc_out <= np.float32(0.0))
        and np.all(rainbl_arr == np.float32(0.0))
    )
    batch_records: list[dict[str, Any]] = []

    def _finish_result() -> LsmResult:
        return LsmResult(
            snotime=snotime_out,
            tsk=tsk_out,
            hfx=hfx_out,
            qfx=qfx_out,
            lh=lh_out,
            grdflx=grdflx_out,
            qsfc=qsfc_out,
            cqs2=cqs2_out,
            chs=chs_out,
            chs2=chs2_out,
            snow=snow_out,
            snowc=snowc_out,
            snowh=snowh_out,
            canwat=canwat_out,
            smstav=smstav_out,
            smstot=smstot_out,
            sfcrunoff=sfcrunoff_out,
            udrunoff=udrunoff_out,
            acsnom=acsnom_out,
            acsnow=acsnow_out,
            snopcx=snopcx_out,
            emiss=emiss_out,
            rib=rib_out,
            potevp=potevp_out,
            albedo=albedo_out,
            albbck=albbck_out,
            z0=z0_out,
            znt=znt_out,
            smois=smois_out,
            sh2o=sh2o_out,
            tslb=tslb_out,
            smcrel=smcrel_out,
            flx4_2d=flx4_out,
            fvb_2d=fvb_out,
            fbur_2d=fbur_out,
            fgsn_2d=fgsn_out,
            noahres=noahres_out,
            chklowq=chklowq_out,
            lai=lai_out,
            rc2=rc2_out,
            xlai2=xlai2_out,
        )

    if batch_no_snow:
        i0 = int(its) - int(ims)
        i1 = int(ite) - int(ims) + 1
        j0 = int(jts) - int(jms)
        j1 = int(jte) - int(jms) + 1
        kk0 = int(kts) - int(kms)
        kk1 = kk0 + 1
        sl2 = np.s_[i0:i1, j0:j1]
        sl3 = np.s_[i0:i1, :nsoil, j0:j1]

        xland_tile = xland_arr[sl2]
        xice_tile = xice_arr[sl2]
        if int(itimestep) == 1:
            water_mask = xland_tile - np.float32(1.5) >= np.float32(0.0)
            ice_init_mask = xice_tile >= xice_limit
            init_mask = water_mask | ice_init_mask
            smstav_out[sl2] = np.where(init_mask, np.float32(1.0), smstav_out[sl2])
            smstot_out[sl2] = np.where(init_mask, np.float32(1.0), smstot_out[sl2])
            smcrel_out[sl3] = np.where(init_mask[:, None, :], np.float32(1.0), smcrel_out[sl3])
            smois_out[sl3] = np.where(init_mask[:, None, :], np.float32(1.0), smois_out[sl3])
            tslb_out[sl3] = np.where(water_mask[:, None, :], np.float32(273.16), tslb_out[sl3])

        land_mask = xland_tile - np.float32(1.5) < np.float32(0.0)
        ice_mask = land_mask & (xice_tile >= xice_limit)
        active_mask = land_mask & ~ice_mask
        glacial_mask = active_mask & (ivgtyp_arr[sl2] == int(isice))
        if np.any(glacial_mask):
            raise NotImplementedError("lsm glacial Noah path is not translated yet")

        if np.any(ice_mask):
            ice_i, ice_j = np.nonzero(ice_mask)
            ice_ii = ice_i + i0
            ice_jj = ice_j + j0
            sh2o_out[ice_ii, :nsoil, ice_jj] = np.float32(1.0)
            lai_out[ice_ii, ice_jj] = np.float32(0.01)

        if np.any(active_mask):
            loc_i, loc_j = np.nonzero(active_mask)
            ii_idx = loc_i + i0
            jj_idx = loc_j + j0

            psfc = p8w3d_arr[i0:i1, kk0, j0:j1]
            sfcprs = (p8w3d_arr[i0:i1, kk1, j0:j1] + p8w3d_arr[i0:i1, kk0, j0:j1]) * np.float32(0.5)
            q2k = qv3d_arr[i0:i1, kk0, j0:j1] / (np.float32(1.0) + qv3d_arr[i0:i1, kk0, j0:j1])
            q2sat = qgh_arr[sl2] / (np.float32(1.0) + qgh_arr[sl2])
            myj_lowq = (q2k >= q2sat * tresh) & (q2k < qz0_arr[sl2])
            chklowq_tile = np.where(bool(myj), np.where(myj_lowq, np.float32(0.0), np.float32(1.0)), np.float32(1.0))
            chklowq_out[sl2] = chklowq_tile

            sfctmp = t3d_arr[i0:i1, kk0, j0:j1]
            apes = (np.float32(1.0e5) / psfc) ** capa
            apelm = (np.float32(1.0e5) / sfcprs) ** capa
            th2 = (sfctmp * apelm) / apes
            emissi = emiss_out[sl2]
            lwdn = glw_arr[sl2] * emissi
            soldn = swdown_arr[sl2]
            solnet = soldn * (np.float32(1.0) - albedo_out[sl2])
            prcp = rainbl_arr[sl2] / dt_value
            if irr_chan is None:
                irrigation_channel = np.zeros_like(prcp, dtype=dtype)
            else:
                irr_tile = np.asarray(irr_chan, dtype=dtype)[sl2]
                irrigation_channel = np.where(irr_tile != np.float32(0.0), irr_tile / dt_value, np.float32(0.0))

            vegtyp = ivgtyp_arr[sl2]
            soiltyp = np.where((isltyp_arr[sl2] == 14) & (xice_tile == np.float32(0.0)), 7, isltyp_arr[sl2])
            shdfac = vegfra_arr[sl2] / np.float32(100.0)
            if int(isurban) == 1:
                shdfac = np.where(np.isin(vegtyp, np.asarray([25, 26, 27], dtype=np.int32)), np.float32(0.0), shdfac)
            shmin = shdmin_arr[sl2] / np.float32(100.0)
            shmax = shdmax_arr[sl2] / np.float32(100.0)
            sneqv = snow_out[sl2] * np.float32(0.001)
            snowhk = np.where(((sneqv != np.float32(0.0)) & (snowh_out[sl2] == np.float32(0.0))) | (snowh_out[sl2] <= sneqv), np.float32(5.0) * sneqv, snowh_out[sl2])
            if bool(rdlai2d):
                lai_out[sl2] = np.where(land_mask & (shdfac > np.float32(0.0)) & (lai_out[sl2] <= np.float32(0.0)), np.float32(0.01), lai_out[sl2])
            ffrozp = np.where(sfctmp <= np.float32(273.15), np.float32(1.0), np.float32(0.0))
            dqsdt2 = q2sat * a23m4 / (sfctmp - a4) ** np.float32(2.0)

            input_arrays = {
                "ffrozp": ffrozp,
                "lwdn": lwdn,
                "soldn": soldn,
                "solnet": solnet,
                "sfcprs": sfcprs,
                "prcp": prcp,
                "sfctmp": sfctmp,
                "q2": q2k,
                "th2": th2,
                "q2sat": q2sat,
                "dqsdt2": dqsdt2,
                "shdfac": shdfac,
                "shdmin": shmin,
                "shdmax": shmax,
                "alb": albbck_out[sl2],
                "snoalb": snoalb_arr[sl2],
                "tbot": tmn_arr[sl2],
                "z0brd": z0_out[sl2],
                "emissi": emissi,
                "cmc": canwat_out[sl2] / np.float32(1000.0),
                "t1": tsk_out[sl2],
                "snowh": snowhk,
                "sneqv": sneqv,
                "ch": chs_out[sl2],
                "sncovr": snowc_out[sl2],
                "xlai": lai_out[sl2],
                "snotime1": snotime_out[sl2],
                "ribb": rib_out[sl2],
                "irrigation_channel": irrigation_channel,
                "vegtyp": vegtyp,
                "soiltyp": soiltyp,
            }
            args = [
                jnp.asarray(input_arrays[name][active_mask], dtype=jnp.int32 if name in {"vegtyp", "soiltyp"} else jnp.float32)
                for name in (
                    "ffrozp",
                    "lwdn",
                    "soldn",
                    "solnet",
                    "sfcprs",
                    "prcp",
                    "sfctmp",
                    "q2",
                    "th2",
                    "q2sat",
                    "dqsdt2",
                    "shdfac",
                    "shdmin",
                    "shdmax",
                    "alb",
                    "snoalb",
                    "tbot",
                    "z0brd",
                    "emissi",
                    "cmc",
                    "t1",
                )
            ]
            args.extend(
                [
                    jnp.asarray(tslb_out[ii_idx, :nsoil, jj_idx], dtype=jnp.float32),
                    jnp.asarray(smois_out[ii_idx, :nsoil, jj_idx], dtype=jnp.float32),
                    jnp.asarray(sh2o_out[ii_idx, :nsoil, jj_idx], dtype=jnp.float32),
                ]
            )
            args.extend(
                [
                    jnp.asarray(input_arrays[name][active_mask], dtype=jnp.int32 if name in {"vegtyp", "soiltyp"} else jnp.float32)
                    for name in (
                        "snowh",
                        "sneqv",
                        "ch",
                        "sncovr",
                        "xlai",
                        "snotime1",
                        "ribb",
                        "irrigation_channel",
                        "vegtyp",
                        "soiltyp",
                    )
                ]
            )
            batch_fn = _get_sflx_batch_fn(
                parameters=parameters,
                isurban=int(isurban),
                dt_value=dt_value,
                nsoil=nsoil,
                sldpth=sldpth,
                mminlu=str(mminlu),
                rdlai2d=bool(rdlai2d),
                usemonalb=bool(usemonalb),
                opt_thcnd=int(opt_thcnd),
                jit_compile=True,
            )
            args.append(jnp.zeros_like(args[0], dtype=jnp.uint32))
            output_tuple = jax.device_get(batch_fn(*args))
            outputs = {name: np.asarray(value, dtype=dtype) for name, value in zip(_SFLX_BATCH_FIELDS, output_tuple)}
            solnet_arr = input_arrays["solnet"][active_mask].astype(dtype, copy=False)
            lwdn_arr = input_arrays["lwdn"][active_mask].astype(dtype, copy=False)
            ffrozp_arr = input_arrays["ffrozp"][active_mask].astype(dtype, copy=False)
            prcp_arr = input_arrays["prcp"][active_mask].astype(dtype, copy=False)

            lai_out[ii_idx, jj_idx] = outputs["xlai"]
            if rc2_out is not None and xlai2_out is not None:
                rc2_out[ii_idx, jj_idx] = outputs["rc"]
                xlai2_out[ii_idx, jj_idx] = outputs["xlai"]
            canwat_out[ii_idx, jj_idx] = outputs["cmc"] * np.float32(1000.0)
            snow_updated = outputs["sneqv"] * np.float32(1000.0)
            snow_out[ii_idx, jj_idx] = snow_updated
            snowh_out[ii_idx, jj_idx] = outputs["snowh"]
            albedo_out[ii_idx, jj_idx] = outputs["albedo"]
            albbck_out[ii_idx, jj_idx] = outputs["alb"]
            z0_out[ii_idx, jj_idx] = outputs["z0brd"]
            emiss_out[ii_idx, jj_idx] = outputs["emissi"]
            znt_out[ii_idx, jj_idx] = outputs["z0"]
            tsk_out[ii_idx, jj_idx] = outputs["t1"]
            hfx_out[ii_idx, jj_idx] = outputs["sheat"]
            potevp_out[ii_idx, jj_idx] = potevp_out[ii_idx, jj_idx] + outputs["etp"] * fdtw
            qfx_out[ii_idx, jj_idx] = outputs["eta_kinematic"]
            lh_out[ii_idx, jj_idx] = outputs["eta"]
            grdflx_out[ii_idx, jj_idx] = outputs["ssoil"]
            snowc_out[ii_idx, jj_idx] = outputs["sncovr"]
            chs2_out[ii_idx, jj_idx] = cqs2_out[ii_idx, jj_idx]
            snotime_out[ii_idx, jj_idx] = outputs["snotime1"]
            q1 = outputs["q1"]
            qsfc_out[ii_idx, jj_idx] = q1 / (np.float32(1.0) - q1)
            smois_out[ii_idx, :nsoil, jj_idx] = outputs["smc"][:, :nsoil]
            tslb_out[ii_idx, :nsoil, jj_idx] = outputs["stc"][:, :nsoil]
            sh2o_out[ii_idx, :nsoil, jj_idx] = outputs["sh2o"][:, :nsoil]
            flx4_out[ii_idx, jj_idx] = outputs["flx4"]
            fvb_out[ii_idx, jj_idx] = outputs["fvb"]
            fbur_out[ii_idx, jj_idx] = outputs["fbur"]
            fgsn_out[ii_idx, jj_idx] = outputs["fgsn"]
            noahres_out[ii_idx, jj_idx] = (
                (solnet_arr + lwdn_arr)
                - outputs["sheat"]
                + outputs["ssoil"]
                - outputs["eta"]
                - (outputs["emissi"] * stb * (outputs["t1"] ** np.float32(4.0)))
                - outputs["flx1"]
                - outputs["flx2"]
                - outputs["flx3"]
            )
            smstav_out[ii_idx, jj_idx] = outputs["soilw"]
            smstot_out[ii_idx, jj_idx] = outputs["soilm"] * np.float32(1000.0)
            smcrel_out[ii_idx, :nsoil, jj_idx] = outputs["smav"][:, :nsoil]
            sfcrunoff_out[ii_idx, jj_idx] = sfcrunoff_out[ii_idx, jj_idx] + outputs["runoff1"] * dt_value * np.float32(1000.0)
            udrunoff_out[ii_idx, jj_idx] = udrunoff_out[ii_idx, jj_idx] + outputs["runoff2"] * dt_value * np.float32(1000.0)

            frozen_mask = ffrozp_arr > np.float32(0.5)
            if np.any(frozen_mask):
                acsnow_out[ii_idx[frozen_mask], jj_idx[frozen_mask]] = acsnow_out[ii_idx[frozen_mask], jj_idx[frozen_mask]] + prcp_arr[frozen_mask] * dt_value
            snow_mask = snow_updated > np.float32(0.0)
            if np.any(snow_mask):
                acsnom_out[ii_idx[snow_mask], jj_idx[snow_mask]] = acsnom_out[ii_idx[snow_mask], jj_idx[snow_mask]] + outputs["snomlt"][snow_mask] * np.float32(1000.0)
                snopcx_out[ii_idx[snow_mask], jj_idx[snow_mask]] = snopcx_out[ii_idx[snow_mask], jj_idx[snow_mask]] - outputs["snomlt"][snow_mask] / fdtliw

        return _finish_result()

    itf = int(ite)
    jtf = int(jte)
    for j in range(int(jts), jtf + 1):
        jj = j - int(jms)
        if int(itimestep) == 1:
            for i in range(int(its), itf + 1):
                ii = i - int(ims)
                if xland_arr[ii, jj] - np.float32(1.5) >= np.float32(0.0):
                    smstav_out[ii, jj] = np.float32(1.0)
                    smstot_out[ii, jj] = np.float32(1.0)
                    for ns in range(nsoil):
                        smois_out[ii, ns, jj] = np.float32(1.0)
                        tslb_out[ii, ns, jj] = np.float32(273.16)
                        smcrel_out[ii, ns, jj] = np.float32(1.0)
                elif xice_arr[ii, jj] >= xice_limit:
                    smstav_out[ii, jj] = np.float32(1.0)
                    smstot_out[ii, jj] = np.float32(1.0)
                    for ns in range(nsoil):
                        smois_out[ii, ns, jj] = np.float32(1.0)
                        smcrel_out[ii, ns, jj] = np.float32(1.0)

        for i in range(int(its), itf + 1):
            ii = i - int(ims)
            kk0 = int(kts) - int(kms)
            kk1 = kk0 + 1
            psfc = p8w3d_arr[ii, kk0, jj]
            sfcprs = (p8w3d_arr[ii, kk1, jj] + p8w3d_arr[ii, kk0, jj]) * np.float32(0.5)
            q2k = qv3d_arr[ii, kk0, jj] / (np.float32(1.0) + qv3d_arr[ii, kk0, jj])
            q2sat = qgh_arr[ii, jj] / (np.float32(1.0) + qgh_arr[ii, jj])
            if bool(myj) and q2k >= q2sat * tresh and q2k < qz0_arr[ii, jj]:
                satflg = np.float32(0.0)
                chklowq_out[ii, jj] = np.float32(0.0)
            else:
                satflg = np.float32(1.0)
                chklowq_out[ii, jj] = np.float32(1.0)
            del satflg

            sfctmp = t3d_arr[ii, kk0, jj]
            zlvl = np.float32(0.5) * dz8w_arr[ii, kk0, jj]
            apes = (np.float32(1.0e5) / psfc) ** capa
            apelm = (np.float32(1.0e5) / sfcprs) ** capa
            sfcth2 = sfctmp * apelm
            th2 = sfcth2 / apes
            emissi = emiss_out[ii, jj]
            lwdn = glw_arr[ii, jj] * emissi
            soldn = swdown_arr[ii, jj]
            solnet = soldn * (np.float32(1.0) - albedo_out[ii, jj])
            prcp = rainbl_arr[ii, jj] / dt_value
            irrigation_channel = np.float32(0.0)
            if irr_chan is not None:
                irr_arr = np.asarray(irr_chan, dtype=dtype)
                if irr_arr[ii, jj] != np.float32(0.0):
                    irrigation_channel = irr_arr[ii, jj] / dt_value

            vegtyp = int(ivgtyp_arr[ii, jj])
            soiltyp = int(isltyp_arr[ii, jj])
            shdfac = vegfra_arr[ii, jj] / np.float32(100.0)
            t1 = tsk_out[ii, jj]
            chk = chs_out[ii, jj]
            shmin = shdmin_arr[ii, jj] / np.float32(100.0)
            shmax = shdmax_arr[ii, jj] / np.float32(100.0)
            sneqv = snow_out[ii, jj] * np.float32(0.001)
            snowhk = snowh_out[ii, jj]
            sncovr = snowc_out[ii, jj]

            if bool(frpcpn):
                ffrozp = sr_arr[ii, jj]
            elif sfctmp <= np.float32(273.15):
                ffrozp = np.float32(1.0)
            else:
                ffrozp = np.float32(0.0)

            if xland_arr[ii, jj] - np.float32(1.5) >= np.float32(0.0):
                continue

            if xice_arr[ii, jj] >= xice_limit:
                ice = 1
            elif vegtyp == int(isice):
                ice = -1
            else:
                ice = 0

            dqsdt2 = q2sat * a23m4 / (sfctmp - a4) ** np.float32(2.0)
            if snow_out[ii, jj] > np.float32(0.0):
                sfctsno = sfctmp
                e2sat = np.float32(611.2) * np.exp(np.float32(6174.0) * (np.float32(1.0) / np.float32(273.15) - np.float32(1.0) / sfctsno))
                q2sati = np.float32(0.622) * e2sat / (sfcprs - e2sat)
                q2sati = q2sati / (np.float32(1.0) + q2sati)
                if t1 > np.float32(273.14):
                    q2sat = q2sat * (np.float32(1.0) - snowc_out[ii, jj]) + q2sati * snowc_out[ii, jj]
                    dqsdt2 = dqsdt2 * (np.float32(1.0) - snowc_out[ii, jj]) + q2sati * np.float32(6174.0) / (sfctsno**np.float32(2.0)) * snowc_out[ii, jj]
                else:
                    q2sat = q2sati
                    dqsdt2 = q2sati * np.float32(6174.0) / (sfctsno**np.float32(2.0))
                if t1 > np.float32(273.0) and snowc_out[ii, jj] > np.float32(0.0) and swdown_arr[ii, jj] > np.float32(10.0):
                    dqsdt2 = dqsdt2 * (np.float32(1.0) - snowc_out[ii, jj])

            tbot = tmn_arr[ii, jj]
            if int(isurban) == 1 and vegtyp in {25, 26, 27}:
                shdfac = np.float32(0.0)
            if soiltyp == 14 and xice_arr[ii, jj] == np.float32(0.0):
                soiltyp = 7
            snoalb1 = snoalb_arr[ii, jj]
            cmc = canwat_out[ii, jj] / np.float32(1000.0)
            albbrd = albbck_out[ii, jj]
            z0brd = z0_out[ii, jj]
            embrd = embck_arr[ii, jj]
            snotime1 = snotime_out[ii, jj]
            ribb = rib_out[ii, jj]
            smc = smois_out[ii, :nsoil, jj].copy()
            stc = tslb_out[ii, :nsoil, jj].copy()
            swc = sh2o_out[ii, :nsoil, jj].copy()
            if (sneqv != np.float32(0.0) and snowhk == np.float32(0.0)) or snowhk <= sneqv:
                snowhk = np.float32(5.0) * sneqv

            if bool(rdlai2d):
                if shdfac > np.float32(0.0) and lai_out[ii, jj] <= np.float32(0.0):
                    lai_out[ii, jj] = np.float32(0.01)
                xlai = lai_out[ii, jj]
            else:
                xlai = lai_out[ii, jj]

            if ice == 1:
                sh2o_out[ii, :nsoil, jj] = np.float32(1.0)
                lai_out[ii, jj] = np.float32(0.01)
                continue
            if ice == -1:
                raise NotImplementedError("lsm glacial Noah path is not translated yet")

            if batch_no_snow:
                batch_records.append(
                    {
                        "ii": ii,
                        "jj": jj,
                        "vegtyp": vegtyp,
                        "soiltyp": soiltyp,
                        "ffrozp": ffrozp,
                        "lwdn": lwdn,
                        "soldn": soldn,
                        "solnet": solnet,
                        "sfcprs": sfcprs,
                        "prcp": prcp,
                        "sfctmp": sfctmp,
                        "q2": q2k,
                        "th2": th2,
                        "q2sat": q2sat,
                        "dqsdt2": dqsdt2,
                        "shdfac": shdfac,
                        "shdmin": shmin,
                        "shdmax": shmax,
                        "alb": albbrd,
                        "snoalb": snoalb1,
                        "tbot": tbot,
                        "z0brd": z0brd,
                        "emissi": emissi,
                        "cmc": cmc,
                        "t1": t1,
                        "stc": stc,
                        "smc": smc,
                        "sh2o": swc,
                        "snowh": snowhk,
                        "sneqv": sneqv,
                        "ch": chk,
                        "sncovr": sncovr,
                        "xlai": xlai,
                        "snotime1": snotime1,
                        "ribb": ribb,
                        "irrigation_channel": irrigation_channel,
                    }
                )
                continue

            result: SflxResult = sflx_call(
                f_frozp=ffrozp,
                isurban=isurban,
                dt=dt_value,
                zlvl=zlvl,
                nsoil=nsoil,
                sldpth=sldpth,
                local=False,
                llanduse=str(mminlu),
                lsoil="STAS",
                lwdn=lwdn,
                soldn=soldn,
                solnet=solnet,
                sfcprs=sfcprs,
                prcp=prcp,
                sfctmp=sfctmp,
                q2=q2k,
                sfcspd=np.float32(0.0),
                cosz=np.float32(0.0),
                prcprain=np.float32(0.0),
                solardirect=np.float32(0.0),
                th2=th2,
                q2sat=q2sat,
                dqsdt2=dqsdt2,
                vegtyp=vegtyp,
                soiltyp=soiltyp,
                slopetyp=1,
                shdfac=shdfac,
                shdmin=shmin,
                shdmax=shmax,
                alb=albbrd,
                snoalb=snoalb1,
                tbot=tbot,
                z0brd=z0brd,
                z0=np.float32(0.0),
                emissi=emissi,
                cmc=cmc,
                t1=t1,
                stc=stc,
                smc=smc,
                sh2o=swc,
                snowh=snowhk,
                sneqv=sneqv,
                albedo=np.float32(0.0),
                ch=chk,
                cm=np.float32(0.0),
                eta=np.float32(0.0),
                sheat=np.float32(0.0),
                eta_kinematic=np.float32(0.0),
                fdown=np.float32(0.0),
                ec=np.float32(0.0),
                edir=np.float32(0.0),
                et=np.zeros((nsoil,), dtype=dtype),
                ett=np.float32(0.0),
                esnow=np.float32(0.0),
                drip=np.float32(0.0),
                dew=np.float32(0.0),
                beta=np.float32(0.0),
                etp=np.float32(0.0),
                ssoil=np.float32(0.0),
                flx1=np.float32(0.0),
                flx2=np.float32(0.0),
                flx3=np.float32(0.0),
                flx4=np.float32(0.0),
                fvb=np.float32(0.0),
                fbur=np.float32(0.0),
                fgsn=np.float32(0.0),
                ua_phys=ua_phys,
                snomlt=np.float32(0.0),
                sncovr=sncovr,
                runoff1=np.float32(0.0),
                runoff2=np.float32(0.0),
                runoff3=np.float32(0.0),
                rc=np.float32(0.0),
                pc=np.float32(0.0),
                rsmin=np.float32(0.0),
                xlai=xlai,
                rcs=np.float32(0.0),
                rct=np.float32(0.0),
                rcq=np.float32(0.0),
                rcsoil=np.float32(0.0),
                soilw=np.float32(0.0),
                soilm=np.float32(0.0),
                q1=np.float32(0.0),
                smav=np.zeros((nsoil,), dtype=dtype),
                rdlai2d=rdlai2d,
                usemonalb=usemonalb,
                snotime1=snotime1,
                ribb=ribb,
                smcwlt=np.float32(0.0),
                smcdry=np.float32(0.0),
                smcref=np.float32(0.0),
                smcmax=np.float32(0.0),
                nroot=1,
                sfhead1rt=np.float32(0.0),
                infxs1rt=np.float32(0.0),
                etpnd1=np.float32(0.0),
                opt_thcnd=opt_thcnd,
                aoasis=np.float32(1.0),
                xsda_qfx=np.float32(0.0),
                hfx_phy=np.float32(0.0),
                qfx_phy=np.float32(0.0),
                xqnorm=np.float32(0.0),
                fasdas=fasdas,
                hcpct_fasdas=np.float32(0.0),
                params=parameters,
                irrigation_channel=irrigation_channel,
            )

            lai_out[ii, jj] = _float_value(result.xlai)
            if rc2_out is not None and xlai2_out is not None:
                rc2_out[ii, jj] = _float_value(result.rc)
                xlai2_out[ii, jj] = _float_value(result.xlai)

            canwat_out[ii, jj] = _float_value(result.cmc) * np.float32(1000.0)
            snow_out[ii, jj] = _float_value(result.sneqv) * np.float32(1000.0)
            snowh_out[ii, jj] = _float_value(result.snowh)
            albedo_out[ii, jj] = _float_value(result.albedo)
            albbck_out[ii, jj] = _float_value(result.alb)
            z0_out[ii, jj] = _float_value(result.z0brd)
            emiss_out[ii, jj] = _float_value(result.emissi)
            znt_out[ii, jj] = _float_value(result.z0)
            tsk_out[ii, jj] = _float_value(result.t1)
            hfx_out[ii, jj] = _float_value(result.sheat)
            potevp_out[ii, jj] = potevp_out[ii, jj] + _float_value(result.etp) * fdtw
            qfx_out[ii, jj] = _float_value(result.eta_kinematic)
            lh_out[ii, jj] = _float_value(result.eta)
            grdflx_out[ii, jj] = _float_value(result.ssoil)
            snowc_out[ii, jj] = _float_value(result.sncovr)
            chs2_out[ii, jj] = cqs2_out[ii, jj]
            snotime_out[ii, jj] = _float_value(result.snotime1)
            q1 = _float_value(result.q1)
            qsfc_out[ii, jj] = q1 / (np.float32(1.0) - q1)
            smois_out[ii, :nsoil, jj] = _array_value(result.smc)[:nsoil]
            tslb_out[ii, :nsoil, jj] = _array_value(result.stc)[:nsoil]
            sh2o_out[ii, :nsoil, jj] = _array_value(result.sh2o)[:nsoil]
            flx4_out[ii, jj] = _float_value(result.flx4)
            fvb_out[ii, jj] = _float_value(result.fvb)
            fbur_out[ii, jj] = _float_value(result.fbur)
            fgsn_out[ii, jj] = _float_value(result.fgsn)

            noahres_out[ii, jj] = (
                (solnet + lwdn)
                - _float_value(result.sheat)
                + _float_value(result.ssoil)
                - _float_value(result.eta)
                - (_float_value(result.emissi) * stb * (_float_value(result.t1) ** np.float32(4.0)))
                - _float_value(result.flx1)
                - _float_value(result.flx2)
                - _float_value(result.flx3)
            )
            if bool(ua_phys):
                noahres_out[ii, jj] = noahres_out[ii, jj] - _float_value(result.flx4)

            smstav_out[ii, jj] = _float_value(result.soilw)
            smstot_out[ii, jj] = _float_value(result.soilm) * np.float32(1000.0)
            smcrel_out[ii, :nsoil, jj] = _array_value(result.smav)[:nsoil]
            sfcrunoff_out[ii, jj] = sfcrunoff_out[ii, jj] + _float_value(result.runoff1) * dt_value * np.float32(1000.0)
            udrunoff_out[ii, jj] = udrunoff_out[ii, jj] + _float_value(result.runoff2) * dt_value * np.float32(1000.0)
            if ffrozp > np.float32(0.5):
                acsnow_out[ii, jj] = acsnow_out[ii, jj] + prcp * dt_value
            if snow_out[ii, jj] > np.float32(0.0):
                acsnom_out[ii, jj] = acsnom_out[ii, jj] + _float_value(result.snomlt) * np.float32(1000.0)
                snopcx_out[ii, jj] = snopcx_out[ii, jj] - _float_value(result.snomlt) / fdtliw

    if batch_records:
        input_fields = (
            "ffrozp",
            "lwdn",
            "soldn",
            "solnet",
            "sfcprs",
            "prcp",
            "sfctmp",
            "q2",
            "th2",
            "q2sat",
            "dqsdt2",
            "shdfac",
            "shdmin",
            "shdmax",
            "alb",
            "snoalb",
            "tbot",
            "z0brd",
            "emissi",
            "cmc",
            "t1",
            "stc",
            "smc",
            "sh2o",
            "snowh",
            "sneqv",
            "ch",
            "sncovr",
            "xlai",
            "snotime1",
            "ribb",
            "irrigation_channel",
            "vegtyp",
            "soiltyp",
        )
        records = batch_records
        batch_fn = _get_sflx_batch_fn(
            parameters=parameters,
            isurban=int(isurban),
            dt_value=dt_value,
            nsoil=nsoil,
            sldpth=sldpth,
            mminlu=str(mminlu),
            rdlai2d=bool(rdlai2d),
            usemonalb=bool(usemonalb),
            opt_thcnd=int(opt_thcnd),
            jit_compile=True,
        )
        args = []
        for field in input_fields:
            if field in {"stc", "smc", "sh2o"}:
                args.append(jnp.asarray(np.stack([np.asarray(record[field], dtype=dtype) for record in records]), dtype=jnp.float32))
            elif field in {"vegtyp", "soiltyp"}:
                args.append(jnp.asarray(np.asarray([record[field] for record in records], dtype=np.int32), dtype=jnp.int32))
            else:
                args.append(jnp.asarray(np.asarray([record[field] for record in records], dtype=dtype), dtype=jnp.float32))
        args.append(jnp.zeros_like(args[0], dtype=jnp.uint32))
        output_tuple = jax.device_get(batch_fn(*args))
        outputs = {name: np.asarray(value, dtype=dtype) for name, value in zip(_SFLX_BATCH_FIELDS, output_tuple)}

        ii_idx = np.asarray([int(record["ii"]) for record in records], dtype=np.intp)
        jj_idx = np.asarray([int(record["jj"]) for record in records], dtype=np.intp)
        solnet_arr = np.asarray([record["solnet"] for record in records], dtype=dtype)
        lwdn_arr = np.asarray([record["lwdn"] for record in records], dtype=dtype)
        ffrozp_arr = np.asarray([record["ffrozp"] for record in records], dtype=dtype)
        prcp_arr = np.asarray([record["prcp"] for record in records], dtype=dtype)

        lai_out[ii_idx, jj_idx] = outputs["xlai"]
        if rc2_out is not None and xlai2_out is not None:
            rc2_out[ii_idx, jj_idx] = outputs["rc"]
            xlai2_out[ii_idx, jj_idx] = outputs["xlai"]

        canwat_out[ii_idx, jj_idx] = outputs["cmc"] * np.float32(1000.0)
        snow_updated = outputs["sneqv"] * np.float32(1000.0)
        snow_out[ii_idx, jj_idx] = snow_updated
        snowh_out[ii_idx, jj_idx] = outputs["snowh"]
        albedo_out[ii_idx, jj_idx] = outputs["albedo"]
        albbck_out[ii_idx, jj_idx] = outputs["alb"]
        z0_out[ii_idx, jj_idx] = outputs["z0brd"]
        emiss_out[ii_idx, jj_idx] = outputs["emissi"]
        znt_out[ii_idx, jj_idx] = outputs["z0"]
        tsk_out[ii_idx, jj_idx] = outputs["t1"]
        hfx_out[ii_idx, jj_idx] = outputs["sheat"]
        potevp_out[ii_idx, jj_idx] = potevp_out[ii_idx, jj_idx] + outputs["etp"] * fdtw
        qfx_out[ii_idx, jj_idx] = outputs["eta_kinematic"]
        lh_out[ii_idx, jj_idx] = outputs["eta"]
        grdflx_out[ii_idx, jj_idx] = outputs["ssoil"]
        snowc_out[ii_idx, jj_idx] = outputs["sncovr"]
        chs2_out[ii_idx, jj_idx] = cqs2_out[ii_idx, jj_idx]
        snotime_out[ii_idx, jj_idx] = outputs["snotime1"]
        q1 = outputs["q1"]
        qsfc_out[ii_idx, jj_idx] = q1 / (np.float32(1.0) - q1)
        smois_out[ii_idx, :nsoil, jj_idx] = outputs["smc"][:, :nsoil]
        tslb_out[ii_idx, :nsoil, jj_idx] = outputs["stc"][:, :nsoil]
        sh2o_out[ii_idx, :nsoil, jj_idx] = outputs["sh2o"][:, :nsoil]
        flx4_out[ii_idx, jj_idx] = outputs["flx4"]
        fvb_out[ii_idx, jj_idx] = outputs["fvb"]
        fbur_out[ii_idx, jj_idx] = outputs["fbur"]
        fgsn_out[ii_idx, jj_idx] = outputs["fgsn"]

        noahres_out[ii_idx, jj_idx] = (
            (solnet_arr + lwdn_arr)
            - outputs["sheat"]
            + outputs["ssoil"]
            - outputs["eta"]
            - (outputs["emissi"] * stb * (outputs["t1"] ** np.float32(4.0)))
            - outputs["flx1"]
            - outputs["flx2"]
            - outputs["flx3"]
        )

        smstav_out[ii_idx, jj_idx] = outputs["soilw"]
        smstot_out[ii_idx, jj_idx] = outputs["soilm"] * np.float32(1000.0)
        smcrel_out[ii_idx, :nsoil, jj_idx] = outputs["smav"][:, :nsoil]
        sfcrunoff_out[ii_idx, jj_idx] = sfcrunoff_out[ii_idx, jj_idx] + outputs["runoff1"] * dt_value * np.float32(1000.0)
        udrunoff_out[ii_idx, jj_idx] = udrunoff_out[ii_idx, jj_idx] + outputs["runoff2"] * dt_value * np.float32(1000.0)

        frozen_mask = ffrozp_arr > np.float32(0.5)
        if np.any(frozen_mask):
            frozen_i = ii_idx[frozen_mask]
            frozen_j = jj_idx[frozen_mask]
            acsnow_out[frozen_i, frozen_j] = acsnow_out[frozen_i, frozen_j] + prcp_arr[frozen_mask] * dt_value

        snow_mask = snow_updated > np.float32(0.0)
        if np.any(snow_mask):
            snow_i = ii_idx[snow_mask]
            snow_j = jj_idx[snow_mask]
            acsnom_out[snow_i, snow_j] = acsnom_out[snow_i, snow_j] + outputs["snomlt"][snow_mask] * np.float32(1000.0)
            snopcx_out[snow_i, snow_j] = snopcx_out[snow_i, snow_j] - outputs["snomlt"][snow_mask] / fdtliw

    return _finish_result()


def lsminit(
    vegfra: Any,
    snow: Any,
    snowc: Any,
    snowh: Any,
    canwat: Any,
    smstav: Any,
    smstot: Any,
    sfcrunoff: Any,
    udrunoff: Any,
    acsnow: Any,
    acsnom: Any,
    ivgtyp: Any,
    isltyp: Any,
    tslb: Any,
    smois: Any,
    sh2o: Any,
    zs: Any,
    dzs: Any,
    mminlu: str,
    snoalb: Any,
    fndsoilw: bool,
    fndsnowh: bool,
    rdmaxalb: bool,
    num_soil_layers: int,
    restart: bool,
    allowed_to_read: bool,
    irr_rand_field: Any,
    irr_ph: int,
    irr_freq: int,
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
    its: int,
    ite: int,
    jts: int,
    jte: int,
    kts: int,
    kte: int,
    *,
    parameters: Any | None = None,
    table_dir: str | Path = ".",
    soil_veg_gen_parm_fn: Any | None = None,
    random_values: Any | None = None,
) -> LsminitResult:
    """Run WRF ``module_sf_noahdrv::LSMINIT`` with explicit parameter state."""

    del fndsoilw, kds, kde, kms, kme, kts, kte
    messages: list[str] = []
    active_parameters = parameters
    if bool(allowed_to_read):
        messages.append("INITIALIZE THREE Noah LSM RELATED TABLES")
        if soil_veg_gen_parm_fn is None:
            soil_veg_gen_parm_fn = soil_veg_gen_parm
        active_parameters = soil_veg_gen_parm_fn(str(mminlu), "STAS", table_dir=table_dir)
    if active_parameters is None:
        raise ValueError("lsminit requires Noah table parameters")

    vegfra_out = _copy_array(vegfra)
    snow_out = _copy_array(snow)
    snowc_out = _copy_array(snowc)
    snowh_out = _copy_array(snowh)
    canwat_out = _copy_array(canwat)
    smstav_out = _copy_array(smstav)
    smstot_out = _copy_array(smstot)
    sfcrunoff_out = _copy_array(sfcrunoff)
    udrunoff_out = _copy_array(udrunoff)
    acsnow_out = _copy_array(acsnow)
    acsnom_out = _copy_array(acsnom)
    snoalb_out = _copy_array(snoalb)
    tslb_out = _copy_array(tslb)
    smois_out = _copy_array(smois)
    sh2o_out = _copy_array(sh2o)
    zs_out = _copy_array(zs)
    dzs_out = _copy_array(dzs)
    irr_rand_out = _copy_array(irr_rand_field, dtype=np.int32)
    ivgtyp_arr = np.asarray(ivgtyp, dtype=np.int32)
    isltyp_arr = np.asarray(isltyp, dtype=np.int32)

    if not bool(restart):
        if int(irr_ph) != 0:
            if random_values is None:
                rng = np.random.default_rng(0)
                random_values_arr = rng.random(size=irr_rand_out.shape, dtype=np.float32)
            else:
                random_values_arr = np.asarray(random_values, dtype=np.float32)
            irr_rand_out[...] = np.mod(random_values_arr * 100.0, float(irr_freq)).astype(np.int32)

        itf = min(int(ite), int(ide) - 1)
        jtf = min(int(jte), int(jde) - 1)
        maxalb = _param_array(active_parameters, "maxalb")
        bb = _param_array(active_parameters, "bb")
        maxsmc = _param_array(active_parameters, "maxsmc")
        satpsi = _param_array(active_parameters, "satpsi")

        for j in range(int(jts), jtf + 1):
            jj = j - int(jms)
            for i in range(int(its), itf + 1):
                ii = i - int(ims)
                soil_type = int(isltyp_arr[ii, jj])
                if soil_type < 1:
                    raise ValueError(f"lsminit: out of range ISLTYP {i} {j} {soil_type}")
                if not bool(rdmaxalb):
                    veg_type = int(ivgtyp_arr[ii, jj])
                    snoalb_out[ii, jj] = maxalb[veg_type - 1] * np.float32(0.01)

        blim = np.float32(5.5)
        hlice = np.float32(3.335e5)
        grav = np.float32(9.81)
        t0 = np.float32(273.15)
        frozen = np.zeros(sh2o_out.shape, dtype=bool)
        for j in range(int(jts), jtf + 1):
            jj = j - int(jms)
            for i in range(int(its), itf + 1):
                ii = i - int(ims)
                soil_index = int(isltyp_arr[ii, jj]) - 1
                bx_base = np.float32(bb[soil_index])
                smcmax_base = np.float32(maxsmc[soil_index])
                psisat_base = np.float32(satpsi[soil_index])
                valid_soil = bx_base > 0.0 and smcmax_base > 0.0 and psisat_base > 0.0
                for ns in range(int(num_soil_layers)):
                    if not valid_soil:
                        sh2o_out[ii, ns, jj] = smois_out[ii, ns, jj]
                    elif tslb_out[ii, ns, jj] < np.float32(273.149):
                        bx = min(bx_base, blim)
                        fk = (
                            (
                                (hlice / (grav * (-psisat_base)))
                                * ((tslb_out[ii, ns, jj] - t0) / tslb_out[ii, ns, jj])
                            )
                            ** (-1.0 / bx)
                        ) * smcmax_base
                        if fk < np.float32(0.02):
                            fk = np.float32(0.02)
                        sh2o_out[ii, ns, jj] = min(fk, smois_out[ii, ns, jj])
                        frozen[ii, ns, jj] = True
                    else:
                        sh2o_out[ii, ns, jj] = smois_out[ii, ns, jj]

        if np.any(frozen):
            # Columns are independent. Keep eager FRH2O operation boundaries,
            # but transfer and evaluate all cold layers in one array call.
            cold_i, cold_k, cold_j = np.nonzero(frozen)
            cold = (cold_i, cold_k, cold_j)
            soil_indices = isltyp_arr[cold_i, cold_j] - 1
            sh2o_out[cold] = np.asarray(
                frh2o(
                    tslb_out[cold], smois_out[cold], sh2o_out[cold],
                    maxsmc[soil_indices], np.minimum(bb[soil_indices], blim),
                    satpsi[soil_indices],
                ),
                dtype=np.float32,
            )

        if not bool(fndsnowh):
            messages.append("SNOW HEIGHT NOT FOUND - VALUE DEFINED IN LSMINIT")
            for j in range(int(jts), jtf + 1):
                jj = j - int(jms)
                for i in range(int(its), itf + 1):
                    ii = i - int(ims)
                    snowh_out[ii, jj] = snow_out[ii, jj] * np.float32(0.005)

        for j in range(int(jts), jtf + 1):
            jj = j - int(jms)
            for i in range(int(its), itf + 1):
                ii = i - int(ims)
                canwat_out[ii, jj] = np.float32(0.0)

    return LsminitResult(
        vegfra=vegfra_out,
        snow=snow_out,
        snowc=snowc_out,
        snowh=snowh_out,
        canwat=canwat_out,
        smstav=smstav_out,
        smstot=smstot_out,
        sfcrunoff=sfcrunoff_out,
        udrunoff=udrunoff_out,
        acsnow=acsnow_out,
        acsnom=acsnom_out,
        snoalb=snoalb_out,
        tslb=tslb_out,
        smois=smois_out,
        sh2o=sh2o_out,
        zs=zs_out,
        dzs=dzs_out,
        irr_rand_field=irr_rand_out,
        parameters=active_parameters,
        messages=tuple(messages),
    )


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="ascii").splitlines()


def _numbers(line: str) -> list[float]:
    before_quote = line.split("'", 1)[0]
    return [float(match.replace("D", "E").replace("d", "e")) for match in _NUMBER_RE.findall(before_quote)]


def _first_int(line: str) -> int:
    values = _numbers(line)
    if not values:
        raise ValueError(f"expected integer value in line {line!r}")
    return int(values[0])


def _first_float32(line: str) -> np.float32:
    values = _numbers(line)
    if not values:
        raise ValueError(f"expected real value in line {line!r}")
    return np.float32(values[0])


def _next_nonempty(lines: list[str], start: int) -> int:
    index = start
    while index < len(lines) and lines[index].strip() == "":
        index += 1
    if index >= len(lines):
        raise ValueError("unexpected end of table")
    return index


def _as_float32(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _as_int32(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def _parse_vegetation(path: Path, mminlu: str) -> dict[str, Any]:
    lines = _lines(path)
    target = str(mminlu).strip()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "Vegetation Parameters":
            index += 1
            continue
        lutype_index = _next_nonempty(lines, index + 1)
        lutype = lines[lutype_index].strip()
        cats_index = _next_nonempty(lines, lutype_index + 1)
        cats_values = _numbers(lines[cats_index])
        if len(cats_values) < 2:
            raise ValueError(f"malformed vegetation category line in {path}")
        lucats = int(cats_values[0])
        veg_iindex = int(cats_values[1])
        data_start = cats_index + 1
        if lutype != target:
            index = data_start + lucats
            continue

        shdtbl: list[float] = []
        nrotbl: list[int] = []
        rstbl: list[float] = []
        rgltbl: list[float] = []
        hstbl: list[float] = []
        snuptbl: list[float] = []
        maxalb: list[float] = []
        laimintbl: list[float] = []
        laimaxtbl: list[float] = []
        emissmintbl: list[float] = []
        emissmaxtbl: list[float] = []
        albedomintbl: list[float] = []
        albedomaxtbl: list[float] = []
        z0mintbl: list[float] = []
        z0maxtbl: list[float] = []
        ztopvtbl: list[float] = []
        zbotvtbl: list[float] = []

        for row_index in range(data_start, data_start + lucats):
            values = _numbers(lines[row_index])
            if len(values) < 18:
                raise ValueError(f"vegetation row has too few values: {lines[row_index]!r}")
            shdtbl.append(values[1])
            nrotbl.append(int(values[2]))
            rstbl.append(values[3])
            rgltbl.append(values[4])
            hstbl.append(values[5])
            snuptbl.append(values[6])
            maxalb.append(values[7])
            laimintbl.append(values[8])
            laimaxtbl.append(values[9])
            emissmintbl.append(values[10])
            emissmaxtbl.append(values[11])
            albedomintbl.append(values[12])
            albedomaxtbl.append(values[13])
            z0mintbl.append(values[14])
            z0maxtbl.append(values[15])
            ztopvtbl.append(values[16])
            zbotvtbl.append(values[17])

        scalar_index = data_start + lucats

        def read_pair_float() -> np.float32:
            nonlocal scalar_index
            scalar_index = _next_nonempty(lines, scalar_index)
            scalar_index = _next_nonempty(lines, scalar_index + 1)
            value = _first_float32(lines[scalar_index])
            scalar_index += 1
            return value

        def read_pair_int() -> int:
            nonlocal scalar_index
            scalar_index = _next_nonempty(lines, scalar_index)
            scalar_index = _next_nonempty(lines, scalar_index + 1)
            value = _first_int(lines[scalar_index])
            scalar_index += 1
            return value

        topt_data = read_pair_float()
        cmcmax_data = read_pair_float()
        cfactr_data = read_pair_float()
        rsmax_data = read_pair_float()
        bare = read_pair_int()
        natural = read_pair_int()
        scalar_index = _next_nonempty(lines, scalar_index)
        if lines[scalar_index].strip().startswith("CROP"):
            scalar_index = _next_nonempty(lines, scalar_index + 1)
            scalar_index += 1

        lcz_values = []
        for _ in range(11):
            lcz_values.append(read_pair_int())

        return {
            "lutype": lutype,
            "lucats": lucats,
            "veg_iindex": veg_iindex,
            "shdtbl": _as_float32(shdtbl),
            "nrotbl": _as_int32(nrotbl),
            "rstbl": _as_float32(rstbl),
            "rgltbl": _as_float32(rgltbl),
            "hstbl": _as_float32(hstbl),
            "snuptbl": _as_float32(snuptbl),
            "maxalb": _as_float32(maxalb),
            "laimintbl": _as_float32(laimintbl),
            "laimaxtbl": _as_float32(laimaxtbl),
            "emissmintbl": _as_float32(emissmintbl),
            "emissmaxtbl": _as_float32(emissmaxtbl),
            "albedomintbl": _as_float32(albedomintbl),
            "albedomaxtbl": _as_float32(albedomaxtbl),
            "z0mintbl": _as_float32(z0mintbl),
            "z0maxtbl": _as_float32(z0maxtbl),
            "ztopvtbl": _as_float32(ztopvtbl),
            "zbotvtbl": _as_float32(zbotvtbl),
            "topt_data": topt_data,
            "cmcmax_data": cmcmax_data,
            "cfactr_data": cfactr_data,
            "rsmax_data": rsmax_data,
            "bare": bare,
            "natural": natural,
            "lcz_1": lcz_values[0],
            "lcz_2": lcz_values[1],
            "lcz_3": lcz_values[2],
            "lcz_4": lcz_values[3],
            "lcz_5": lcz_values[4],
            "lcz_6": lcz_values[5],
            "lcz_7": lcz_values[6],
            "lcz_8": lcz_values[7],
            "lcz_9": lcz_values[8],
            "lcz_10": lcz_values[9],
            "lcz_11": lcz_values[10],
        }
    raise ValueError(f"Land Use Dataset {target!r} not found in VEGPARM.TBL")


def _parse_soil(path: Path, mminsl: str) -> dict[str, Any]:
    lines = _lines(path)
    index = _next_nonempty(lines, 0)
    if lines[index].strip() != "Soil Parameters":
        raise ValueError("SOILPARM.TBL does not start with 'Soil Parameters'")
    sltype_index = _next_nonempty(lines, index + 1)
    sltype = lines[sltype_index].strip()[:4]
    cats_index = _next_nonempty(lines, sltype_index + 1)
    cats_values = _numbers(lines[cats_index])
    if len(cats_values) < 2:
        raise ValueError(f"malformed soil category line in {path}")
    slcats = int(cats_values[0])
    soil_iindex = int(cats_values[1])
    if sltype != str(mminsl).strip()[:4]:
        raise ValueError("INCONSISTENT OR MISSING SOILPARM FILE")

    bb: list[float] = []
    drysmc: list[float] = []
    f11: list[float] = []
    maxsmc: list[float] = []
    refsmc: list[float] = []
    satpsi: list[float] = []
    satdk: list[float] = []
    satdw: list[float] = []
    wltsmc: list[float] = []
    qtz: list[float] = []
    data_start = cats_index + 1
    for row_index in range(data_start, data_start + slcats):
        values = _numbers(lines[row_index])
        if len(values) < 11:
            raise ValueError(f"soil row has too few values: {lines[row_index]!r}")
        bb.append(values[1])
        drysmc.append(values[2])
        f11.append(values[3])
        maxsmc.append(values[4])
        refsmc.append(values[5])
        satpsi.append(values[6])
        satdk.append(values[7])
        satdw.append(values[8])
        wltsmc.append(values[9])
        qtz.append(values[10])

    return {
        "sltype": sltype,
        "slcats": slcats,
        "soil_iindex": soil_iindex,
        "bb": _as_float32(bb),
        "drysmc": _as_float32(drysmc),
        "f11": _as_float32(f11),
        "maxsmc": _as_float32(maxsmc),
        "refsmc": _as_float32(refsmc),
        "satpsi": _as_float32(satpsi),
        "satdk": _as_float32(satdk),
        "satdw": _as_float32(satdw),
        "wltsmc": _as_float32(wltsmc),
        "qtz": _as_float32(qtz),
    }


def _parse_gen(path: Path) -> dict[str, Any]:
    lines = _lines(path)
    index = _next_nonempty(lines, 0)
    if lines[index].strip() != "General Parameters":
        raise ValueError("GENPARM.TBL does not start with 'General Parameters'")
    index = _next_nonempty(lines, index + 1)
    index = _next_nonempty(lines, index + 1)
    num_slope = _first_int(lines[index])
    slope_data = []
    index += 1
    for _ in range(num_slope):
        index = _next_nonempty(lines, index)
        slope_data.append(_first_float32(lines[index]))
        index += 1

    def read_pair_float() -> np.float32:
        nonlocal index
        index = _next_nonempty(lines, index)
        index = _next_nonempty(lines, index + 1)
        value = _first_float32(lines[index])
        index += 1
        return value

    return {
        "num_slope": num_slope,
        "slpcats": num_slope,
        "slope_data": np.asarray(slope_data, dtype=np.float32),
        "sbeta_data": read_pair_float(),
        "fxexp_data": read_pair_float(),
        "csoil_data": read_pair_float(),
        "salp_data": read_pair_float(),
        "refdk_data": read_pair_float(),
        "refkdt_data": read_pair_float(),
        "frzk_data": read_pair_float(),
        "zbot_data": read_pair_float(),
        "czil_data": read_pair_float(),
        "smlow_data": read_pair_float(),
        "smhigh_data": read_pair_float(),
        "lvcoef_data": read_pair_float(),
    }


def soil_veg_gen_parm(
    mminlu: str,
    mminsl: str,
    *,
    table_dir: str | Path = ".",
    vegparm_path: str | Path | None = None,
    soilparm_path: str | Path | None = None,
    genparm_path: str | Path | None = None,
) -> SoilVegGenParmResult:
    """Read WRF Noah vegetation, soil, and general parameter tables."""

    base = Path(table_dir)
    veg_path = Path(vegparm_path) if vegparm_path is not None else base / "VEGPARM.TBL"
    soil_path = Path(soilparm_path) if soilparm_path is not None else base / "SOILPARM.TBL"
    gen_path = Path(genparm_path) if genparm_path is not None else base / "GENPARM.TBL"

    veg = _parse_vegetation(veg_path, mminlu)
    soil = _parse_soil(soil_path, mminsl)
    gen = _parse_gen(gen_path)
    return SoilVegGenParmResult(**veg, **soil, **gen)


__all__ = [
    "LsmResult",
    "LsminitResult",
    "NLUS",
    "NSLOPE",
    "NSLTYPE",
    "SoilVegGenParmResult",
    "lsm",
    "lsminit",
    "soil_veg_gen_parm",
]
