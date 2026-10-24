"""Strict WRF-JAX trajectory runner and WRF-output comparison utilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import partial
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from wrf_jax.core.wrf_math import sqrt_for_wrf
from wrf_jax.core.executable_cache import compile_model_program, reusable_terminal_jit
from wrf_jax.core.devices import configured_devices
from wrf_jax.core.tiled_stage import TiledFieldSequence
from wrf_jax.core.rk_stage import POST_RK_TILE_GROUPS, run_post_rk_field_stages, run_rk_field_stages
from wrf_jax.core.direct_step import (
    RK_COMPILED_SUBSTAGES,
    _initialize_direct_step_stage_context,
    _run_final_boundary_stage_family,
    _run_first_rk_step_part1_stage_family,
    _run_first_rk_step_part2_stage,
    _run_post_rk_physics_stage_group,
    _run_rk_step_stage_family,
    _seed_rk_step_size_fields,
    _seed_first_rk_physics_surface_context,
    build_direct_step_plan,
    run_direct_step,
)
from wrf_jax.core.field_sets import CONFIGURED_DYNAMIC_ARRAY_NAMES, CONFIGURED_RECURRENT_ARRAY_NAMES
from wrf_jax.core.partition import (
    deduplicate_input_buffers as _deduplicate_input_buffers,
    is_dynamic_array_value as _is_dynamic_array_value,
    model_state_to_dynamic_fields,
    partition_fields_for_compiled_step as _partition_fields_for_compiled_step,
    split_static_dynamic_fields as _split_static_dynamic_fields,
)
from wrf_jax.core.state import ModelState
from wrf_jax.core.optimization import (
    SEMANTIC_CALL_COMPILER_OPTIONS,
    optimization_barrier,
    semantic_call,
)
from wrf_jax.data_io import load_wrfbdy_fields
from wrf_jax.progress_logging import write_progress
from wrf_jax.runtime import RuntimeAssembly, assemble_runtime, surface_row_tile_metadata
from wrf_jax.dyn_em.solve_em import SolveEmResult, solve_em

PROGNOSTIC_OUTPUTS = ("U", "V", "T", "QVAPOR")
MICROPHYSICS_KEY_OUTPUTS = ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
CUMULUS_KEY_OUTPUTS = ("RAINC", "RAINSH", "NCA", "HTOP", "HBOT")
RADIATION_KEY_OUTPUTS = ("SWDNB", "SWUPB", "SWDNT", "SWUPT", "LWDNB", "LWUPB", "LWDNT", "LWUPT")
KEY_OUTPUTS = PROGNOSTIC_OUTPUTS + MICROPHYSICS_KEY_OUTPUTS + CUMULUS_KEY_OUTPUTS + RADIATION_KEY_OUTPUTS
METEOROLOGICAL_BASE_OUTPUTS = (
    "P", "PB", "PH", "PHB", "W",
    "T2", "Q2", "U10", "V10", "PSFC", "RAINNC",
    "HGT", "LANDMASK", "COSALPHA", "SINALPHA",
    "PBLH", "TSK", "HFX", "QFX", "LH", "GRDFLX",
    "SNOW", "SNOWH", "SNOWC", "ALBEDO", "EMISS",
)
_STAGGERED_EXTRA_FIELD_SOURCES: dict[str, tuple[str, ...]] = {
    "PH": ("ph_2", "ph"),
    "PHB": ("phb",),
    "W": ("w_2", "w"),
}
WRFBDY_FIELD_SUFFIXES = ("_bxs", "_bxe", "_bys", "_bye", "_btxs", "_btxe", "_btys", "_btye")
WRF_REPRO_6H_NRMSE_PERCENT = {
    "U": 0.631976778,
    "V": 1.46761068,
    "T": 0.0643649622,
    "QVAPOR": 0.615319563,
    "QCLOUD": 30.5646087,
    "QRAIN": 10.8921602,
    "QICE": 27.381309,
    "QSNOW": 18.4366491,
    "QGRAUP": 10.1166138,
    "RAINC": 18.2250618,
    "RAINSH": 0.0,
    "SWDNB": 2.33254078,
    "SWUPB": 2.50602772,
    "SWDNT": 4.9026279e-06,
    "SWUPT": 10.2122619,
    "LWDNB": 0.467987839,
    "LWUPB": 0.16197289,
    "LWDNT": 0.0,
    "LWUPT": 1.72010268,
    "CLDFRA": 21.3179815,
}
SECONDS_PER_DAY = np.float32(86400.0)
SECONDS_PER_HOUR = np.float32(3600.0)
GRAMS_PER_KILOGRAM = np.float32(1000.0)
COMPACT_EXTRA_OUTPUTS = frozenset(
    (
        *MICROPHYSICS_KEY_OUTPUTS,
        *CUMULUS_KEY_OUTPUTS,
        *RADIATION_KEY_OUTPUTS,
        "CLDFRA",
        "PI_PHY",
        "RTHRATENLW",
        "RTHRATENSW",
        "RTHRATEN",
        "RTHCUTEN",
        "RTHMPTEN",
        "RTHPHTEN",
        "RQVCUTEN",
        "RQVMPTEN",
        "LW_HEATING_RATE_KDAY",
        "SW_HEATING_RATE_KDAY",
        "RADIATION_HEATING_RATE_KDAY",
        "CUMULUS_HEATING_RATE_KDAY",
        "MICROPHYSICS_HEATING_RATE_KDAY",
        "PHYSICS_HEATING_RATE_KDAY",
    )
)
_THREED_EXTRA_FIELD_SOURCES: dict[str, tuple[str, ...]] = {
    "P": ("p",),
    "PB": ("pb",),
    "PI_PHY": ("pi_phy", "pii"),
    "QCLOUD": ("qcloud",),
    "QRAIN": ("qrain",),
    "QICE": ("qice",),
    "QSNOW": ("qsnow",),
    "QGRAUP": ("qgraup",),
    "QNICE": ("qnice", "qni"),
    "QNRAIN": ("qnrain", "qnr"),
    "CLDFRA": ("cldfra",),
    "RE_CLOUD": ("re_cloud",),
    "RE_ICE": ("re_ice",),
    "RE_SNOW": ("re_snow",),
    "RTHCUTEN": ("rthcuten",),
    "RQVCUTEN": ("rqvcuten",),
    "RQCCUTEN": ("rqccuten",),
    "RQRCUTEN": ("rqrcuten",),
    "RQICUTEN": ("rqicuten",),
    "RQSCUTEN": ("rqscuten",),
    "RUCUTEN": ("rucuten",),
    "RVCUTEN": ("rvcuten",),
    "RTHRATENLW": ("rthratenlw",),
    "RTHRATENLWC": ("rthratenlwc",),
    "RTHRATENSW": ("rthratensw",),
    "RTHRATENSWC": ("rthratenswc",),
    "RTHRATEN": ("rthraten",),
    "RTHBLTEN": ("rthblten",),
    "RQVBLTEN": ("rqvblten",),
    "RQCBLTEN": ("rqcblten",),
    "RQIBLTEN": ("rqiblten",),
    "RQNIBLTEN": ("rqniblten",),
    "RUBLTEN": ("rublten",),
    "RVBLTEN": ("rvblten",),
    "RUSHTEN": ("rushten",),
    "RVSHTEN": ("rvshten",),
    "RTHSHTEN": ("rthshten",),
    "RQVSHTEN": ("rqvshten",),
    "RQCSHTEN": ("rqcshten",),
    "RQRSHTEN": ("rqrshten",),
    "RQISHTEN": ("rqishten",),
    "RQSSHTEN": ("rqsshten",),
    "RQGSHTEN": ("rqgshten",),
    "RTHFTEN": ("rthften",),
    "RQVFTEN": ("rqvften",),
    "RUNDGDTEN": ("rundgdten",),
    "RVNDGDTEN": ("rvndgdten",),
    "RTHNDGDTEN": ("rthndgdten",),
    "RPHNDGDTEN": ("rphndgdten",),
    "RQVNDGDTEN": ("rqvndgdten",),
    "RTHMPTEN": ("rthmpten",),
    "RQVMPTEN": ("rqvmpten",),
    "RQCMPTEN": ("rqcmpten",),
    "RQRMPTEN": ("rqrmpten",),
    "RQIMPTEN": ("rqimpten",),
    "RQSMPTEN": ("rqsmpten",),
    "RQGMPTEN": ("rqgmpten",),
    "H_DIABATIC": ("h_diabatic",),
    "QV_DIABATIC": ("qv_diabatic",),
    "QC_DIABATIC": ("qc_diabatic",),
}
_MOIST_OUTPUT_COMPONENTS: dict[str, int] = {
    "QCLOUD": 2,
    "QRAIN": 3,
    "QICE": 4,
    "QSNOW": 5,
    "QGRAUP": 6,
}
_TOTAL_PHYSICS_TENDENCY_COMPONENTS: dict[str, tuple[str, ...]] = {
    "RUPHTEN": ("RUBLTEN", "RUCUTEN", "RUSHTEN"),
    "RVPHTEN": ("RVBLTEN", "RVCUTEN", "RVSHTEN"),
    "RTHPHTEN": ("RTHBLTEN", "RTHCUTEN", "RTHRATEN", "RTHMPTEN", "RTHSHTEN"),
    "RQVPHTEN": ("RQVBLTEN", "RQVCUTEN", "RQVMPTEN", "RQVSHTEN"),
    "RQCPHTEN": ("RQCBLTEN", "RQCCUTEN", "RQCMPTEN", "RQCSHTEN"),
    "RQRPHTEN": ("RQRCUTEN", "RQRMPTEN", "RQRSHTEN"),
    "RQIPHTEN": ("RQIBLTEN", "RQICUTEN", "RQIMPTEN", "RQISHTEN"),
    "RQSPHTEN": ("RQSCUTEN", "RQSMPTEN", "RQSSHTEN"),
    "RQGPHTEN": ("RQGMPTEN", "RQGSHTEN"),
}
_TENDENCY_ALIAS_SOURCES: dict[str, str] = {
    "LW_HEATING_RATE": "RTHRATENLW",
    "LW_CLEAR_HEATING_RATE": "RTHRATENLWC",
    "SW_HEATING_RATE": "RTHRATENSW",
    "SW_CLEAR_HEATING_RATE": "RTHRATENSWC",
    "LONGWAVE_HEATING_RATE": "RTHRATENLW",
    "LONGWAVE_CLEAR_SKY_HEATING_RATE": "RTHRATENLWC",
    "SHORTWAVE_HEATING_RATE": "RTHRATENSW",
    "SHORTWAVE_CLEAR_SKY_HEATING_RATE": "RTHRATENSWC",
    "RADIATION_HEATING_RATE": "RTHRATEN",
    "TOTAL_RADIATION_HEATING_RATE": "RTHRATEN",
    "THETA_LW_TENDENCY": "RTHRATENLW",
    "THETA_LW_CLEAR_TENDENCY": "RTHRATENLWC",
    "THETA_SW_TENDENCY": "RTHRATENSW",
    "THETA_SW_CLEAR_TENDENCY": "RTHRATENSWC",
    "THETA_RADIATION_TENDENCY": "RTHRATEN",
    "CUMULUS_HEATING_RATE": "RTHCUTEN",
    "THETA_CUMULUS_TENDENCY": "RTHCUTEN",
    "PBL_HEATING_RATE": "RTHBLTEN",
    "THETA_PBL_TENDENCY": "RTHBLTEN",
    "MICROPHYSICS_HEATING_RATE": "RTHMPTEN",
    "THETA_MICROPHYSICS_TENDENCY": "RTHMPTEN",
    "SHALLOW_CUMULUS_HEATING_RATE": "RTHSHTEN",
    "THETA_SHALLOW_CUMULUS_TENDENCY": "RTHSHTEN",
    "PHYSICS_HEATING_RATE": "RTHPHTEN",
    "TOTAL_PHYSICS_HEATING_RATE": "RTHPHTEN",
    "THETA_PHYSICS_TENDENCY": "RTHPHTEN",
    "ADVECTIVE_THETA_TENDENCY": "RTHFTEN",
    "TOTAL_DIABATIC_THETA_TENDENCY": "H_DIABATIC",
    "THETA_DIABATIC_TENDENCY": "H_DIABATIC",
    "U_CUMULUS_TENDENCY": "RUCUTEN",
    "V_CUMULUS_TENDENCY": "RVCUTEN",
    "U_PBL_TENDENCY": "RUBLTEN",
    "V_PBL_TENDENCY": "RVBLTEN",
    "U_SHALLOW_CUMULUS_TENDENCY": "RUSHTEN",
    "V_SHALLOW_CUMULUS_TENDENCY": "RVSHTEN",
    "U_PHYSICS_TENDENCY": "RUPHTEN",
    "V_PHYSICS_TENDENCY": "RVPHTEN",
    "QV_CUMULUS_TENDENCY": "RQVCUTEN",
    "QC_CUMULUS_TENDENCY": "RQCCUTEN",
    "QR_CUMULUS_TENDENCY": "RQRCUTEN",
    "QI_CUMULUS_TENDENCY": "RQICUTEN",
    "QS_CUMULUS_TENDENCY": "RQSCUTEN",
    "QV_PBL_TENDENCY": "RQVBLTEN",
    "QC_PBL_TENDENCY": "RQCBLTEN",
    "QI_PBL_TENDENCY": "RQIBLTEN",
    "QV_MICROPHYSICS_TENDENCY": "RQVMPTEN",
    "QC_MICROPHYSICS_TENDENCY": "RQCMPTEN",
    "QR_MICROPHYSICS_TENDENCY": "RQRMPTEN",
    "QI_MICROPHYSICS_TENDENCY": "RQIMPTEN",
    "QS_MICROPHYSICS_TENDENCY": "RQSMPTEN",
    "QG_MICROPHYSICS_TENDENCY": "RQGMPTEN",
    "QV_PHYSICS_TENDENCY": "RQVPHTEN",
    "QC_PHYSICS_TENDENCY": "RQCPHTEN",
    "QR_PHYSICS_TENDENCY": "RQRPHTEN",
    "QI_PHYSICS_TENDENCY": "RQIPHTEN",
    "QS_PHYSICS_TENDENCY": "RQSPHTEN",
    "QG_PHYSICS_TENDENCY": "RQGPHTEN",
    "TOTAL_DIABATIC_WATER_VAPOR_TENDENCY": "QV_DIABATIC",
    "TOTAL_DIABATIC_CLOUD_WATER_TENDENCY": "QC_DIABATIC",
}
_TEMPERATURE_HEATING_SOURCES: dict[str, str] = {
    "LW_TEMPERATURE_HEATING_RATE": "RTHRATENLW",
    "LW_CLEAR_TEMPERATURE_HEATING_RATE": "RTHRATENLWC",
    "SW_TEMPERATURE_HEATING_RATE": "RTHRATENSW",
    "SW_CLEAR_TEMPERATURE_HEATING_RATE": "RTHRATENSWC",
    "LONGWAVE_TEMPERATURE_HEATING_RATE": "RTHRATENLW",
    "LONGWAVE_CLEAR_SKY_TEMPERATURE_HEATING_RATE": "RTHRATENLWC",
    "SHORTWAVE_TEMPERATURE_HEATING_RATE": "RTHRATENSW",
    "SHORTWAVE_CLEAR_SKY_TEMPERATURE_HEATING_RATE": "RTHRATENSWC",
    "RADIATION_TEMPERATURE_HEATING_RATE": "RTHRATEN",
    "TOTAL_RADIATION_TEMPERATURE_HEATING_RATE": "RTHRATEN",
    "CUMULUS_TEMPERATURE_HEATING_RATE": "RTHCUTEN",
    "PBL_TEMPERATURE_HEATING_RATE": "RTHBLTEN",
    "MICROPHYSICS_TEMPERATURE_HEATING_RATE": "RTHMPTEN",
    "SHALLOW_CUMULUS_TEMPERATURE_HEATING_RATE": "RTHSHTEN",
    "PHYSICS_TEMPERATURE_HEATING_RATE": "RTHPHTEN",
    "TOTAL_PHYSICS_TEMPERATURE_HEATING_RATE": "RTHPHTEN",
    "ADVECTIVE_TEMPERATURE_TENDENCY": "RTHFTEN",
    "DIABATIC_TEMPERATURE_TENDENCY": "H_DIABATIC",
    "TOTAL_DIABATIC_TEMPERATURE_TENDENCY": "H_DIABATIC",
}
_TEMPERATURE_HEATING_KHOUR_SOURCES: dict[str, str] = {
    f"{output_name}_KHOUR": output_name for output_name in _TEMPERATURE_HEATING_SOURCES
}
_TEMPERATURE_HEATING_KDAY_SOURCES: dict[str, str] = {
    f"{output_name}_KDAY": output_name for output_name in _TEMPERATURE_HEATING_SOURCES
}
_KHOUR_TENDENCY_SOURCES: dict[str, str] = {
    "RTHRATENLW_KHOUR": "RTHRATENLW",
    "RTHRATENLWC_KHOUR": "RTHRATENLWC",
    "RTHRATENSW_KHOUR": "RTHRATENSW",
    "RTHRATENSWC_KHOUR": "RTHRATENSWC",
    "RTHRATEN_KHOUR": "RTHRATEN",
    "RTHCUTEN_KHOUR": "RTHCUTEN",
    "RTHBLTEN_KHOUR": "RTHBLTEN",
    "RTHMPTEN_KHOUR": "RTHMPTEN",
    "RTHSHTEN_KHOUR": "RTHSHTEN",
    "RTHFTEN_KHOUR": "RTHFTEN",
    "H_DIABATIC_KHOUR": "H_DIABATIC",
    "RTHPHTEN_KHOUR": "RTHPHTEN",
    "LW_HEATING_RATE_KHOUR": "RTHRATENLW",
    "LW_CLEAR_HEATING_RATE_KHOUR": "RTHRATENLWC",
    "SW_HEATING_RATE_KHOUR": "RTHRATENSW",
    "SW_CLEAR_HEATING_RATE_KHOUR": "RTHRATENSWC",
    "LONGWAVE_HEATING_RATE_KHOUR": "RTHRATENLW",
    "LONGWAVE_CLEAR_SKY_HEATING_RATE_KHOUR": "RTHRATENLWC",
    "SHORTWAVE_HEATING_RATE_KHOUR": "RTHRATENSW",
    "SHORTWAVE_CLEAR_SKY_HEATING_RATE_KHOUR": "RTHRATENSWC",
    "RADIATION_HEATING_RATE_KHOUR": "RTHRATEN",
    "TOTAL_RADIATION_HEATING_RATE_KHOUR": "RTHRATEN",
    "THETA_LW_TENDENCY_KHOUR": "RTHRATENLW",
    "THETA_LW_CLEAR_TENDENCY_KHOUR": "RTHRATENLWC",
    "THETA_SW_TENDENCY_KHOUR": "RTHRATENSW",
    "THETA_SW_CLEAR_TENDENCY_KHOUR": "RTHRATENSWC",
    "THETA_RADIATION_TENDENCY_KHOUR": "RTHRATEN",
    "THETA_CUMULUS_TENDENCY_KHOUR": "RTHCUTEN",
    "THETA_PBL_TENDENCY_KHOUR": "RTHBLTEN",
    "THETA_MICROPHYSICS_TENDENCY_KHOUR": "RTHMPTEN",
    "THETA_SHALLOW_CUMULUS_TENDENCY_KHOUR": "RTHSHTEN",
    "THETA_PHYSICS_TENDENCY_KHOUR": "RTHPHTEN",
    "ADVECTIVE_THETA_TENDENCY_KHOUR": "RTHFTEN",
    "TOTAL_DIABATIC_THETA_TENDENCY_KHOUR": "H_DIABATIC",
    "THETA_DIABATIC_TENDENCY_KHOUR": "H_DIABATIC",
}
_KDAY_TENDENCY_SOURCES: dict[str, str] = {
    "RTHRATENLW_KDAY": "RTHRATENLW",
    "RTHRATENLWC_KDAY": "RTHRATENLWC",
    "RTHRATENSW_KDAY": "RTHRATENSW",
    "RTHRATENSWC_KDAY": "RTHRATENSWC",
    "RTHRATEN_KDAY": "RTHRATEN",
    "RTHCUTEN_KDAY": "RTHCUTEN",
    "RTHBLTEN_KDAY": "RTHBLTEN",
    "RTHMPTEN_KDAY": "RTHMPTEN",
    "RTHSHTEN_KDAY": "RTHSHTEN",
    "RTHFTEN_KDAY": "RTHFTEN",
    "H_DIABATIC_KDAY": "H_DIABATIC",
    "RTHPHTEN_KDAY": "RTHPHTEN",
    "LW_HEATING_RATE_KDAY": "RTHRATENLW",
    "LW_CLEAR_HEATING_RATE_KDAY": "RTHRATENLWC",
    "SW_HEATING_RATE_KDAY": "RTHRATENSW",
    "SW_CLEAR_HEATING_RATE_KDAY": "RTHRATENSWC",
    "THETA_LW_TENDENCY_KDAY": "RTHRATENLW",
    "THETA_LW_CLEAR_TENDENCY_KDAY": "RTHRATENLWC",
    "THETA_SW_TENDENCY_KDAY": "RTHRATENSW",
    "THETA_SW_CLEAR_TENDENCY_KDAY": "RTHRATENSWC",
    "THETA_RADIATION_TENDENCY_KDAY": "RTHRATEN",
    "THETA_CUMULUS_TENDENCY_KDAY": "RTHCUTEN",
    "THETA_PBL_TENDENCY_KDAY": "RTHBLTEN",
    "THETA_MICROPHYSICS_TENDENCY_KDAY": "RTHMPTEN",
    "THETA_SHALLOW_CUMULUS_TENDENCY_KDAY": "RTHSHTEN",
    "THETA_PHYSICS_TENDENCY_KDAY": "RTHPHTEN",
    "LONGWAVE_HEATING_RATE_KDAY": "RTHRATENLW",
    "LONGWAVE_CLEAR_SKY_HEATING_RATE_KDAY": "RTHRATENLWC",
    "SHORTWAVE_HEATING_RATE_KDAY": "RTHRATENSW",
    "SHORTWAVE_CLEAR_SKY_HEATING_RATE_KDAY": "RTHRATENSWC",
    "LONGWAVE_RADIATION_HEATING_RATE_KDAY": "RTHRATENLW",
    "LONGWAVE_CLEAR_SKY_RADIATION_HEATING_RATE_KDAY": "RTHRATENLWC",
    "SHORTWAVE_RADIATION_HEATING_RATE_KDAY": "RTHRATENSW",
    "SHORTWAVE_CLEAR_SKY_RADIATION_HEATING_RATE_KDAY": "RTHRATENSWC",
    "TOTAL_RADIATION_HEATING_RATE_KDAY": "RTHRATEN",
    "TOTAL_RADIATION_THETA_TENDENCY_KDAY": "RTHRATEN",
    "RADIATION_HEATING_RATE_KDAY": "RTHRATEN",
    "CUMULUS_HEATING_RATE_KDAY": "RTHCUTEN",
    "PBL_HEATING_RATE_KDAY": "RTHBLTEN",
    "MICROPHYSICS_HEATING_RATE_KDAY": "RTHMPTEN",
    "SHALLOW_CUMULUS_HEATING_RATE_KDAY": "RTHSHTEN",
    "PHYSICS_HEATING_RATE_KDAY": "RTHPHTEN",
    "TOTAL_PHYSICS_HEATING_RATE_KDAY": "RTHPHTEN",
    "TOTAL_PHYSICS_THETA_TENDENCY_KDAY": "RTHPHTEN",
    "TOTAL_DIABATIC_THETA_TENDENCY_KDAY": "H_DIABATIC",
    "ADVECTIVE_THETA_TENDENCY_KDAY": "RTHFTEN",
}
_MS_DAY_TENDENCY_SOURCES: dict[str, str] = {
    "RUCUTEN_MS_DAY": "RUCUTEN",
    "RVCUTEN_MS_DAY": "RVCUTEN",
    "RUBLTEN_MS_DAY": "RUBLTEN",
    "RVBLTEN_MS_DAY": "RVBLTEN",
    "RUSHTEN_MS_DAY": "RUSHTEN",
    "RVSHTEN_MS_DAY": "RVSHTEN",
    "RUPHTEN_MS_DAY": "RUPHTEN",
    "RVPHTEN_MS_DAY": "RVPHTEN",
    "U_CUMULUS_TENDENCY_MS_DAY": "RUCUTEN",
    "V_CUMULUS_TENDENCY_MS_DAY": "RVCUTEN",
    "U_PBL_TENDENCY_MS_DAY": "RUBLTEN",
    "V_PBL_TENDENCY_MS_DAY": "RVBLTEN",
    "U_SHALLOW_CUMULUS_TENDENCY_MS_DAY": "RUSHTEN",
    "V_SHALLOW_CUMULUS_TENDENCY_MS_DAY": "RVSHTEN",
    "U_PHYSICS_TENDENCY_MS_DAY": "RUPHTEN",
    "V_PHYSICS_TENDENCY_MS_DAY": "RVPHTEN",
    "TOTAL_PHYSICS_U_TENDENCY_MS_DAY": "RUPHTEN",
    "TOTAL_PHYSICS_V_TENDENCY_MS_DAY": "RVPHTEN",
}
_MS_HOUR_TENDENCY_SOURCES: dict[str, str] = {
    output_name.removesuffix("_MS_DAY") + "_MS_HOUR": source_name
    for output_name, source_name in _MS_DAY_TENDENCY_SOURCES.items()
}
_GKG_DAY_TENDENCY_SOURCES: dict[str, str] = {
    "RQVCUTEN_GKG_DAY": "RQVCUTEN",
    "RQVBLTEN_GKG_DAY": "RQVBLTEN",
    "RQCBLTEN_GKG_DAY": "RQCBLTEN",
    "RQIBLTEN_GKG_DAY": "RQIBLTEN",
    "RQVMPTEN_GKG_DAY": "RQVMPTEN",
    "RQVSHTEN_GKG_DAY": "RQVSHTEN",
    "RQCSHTEN_GKG_DAY": "RQCSHTEN",
    "RQRSHTEN_GKG_DAY": "RQRSHTEN",
    "RQISHTEN_GKG_DAY": "RQISHTEN",
    "RQSSHTEN_GKG_DAY": "RQSSHTEN",
    "RQGSHTEN_GKG_DAY": "RQGSHTEN",
    "RQVPHTEN_GKG_DAY": "RQVPHTEN",
    "RQVFTEN_GKG_DAY": "RQVFTEN",
    "QV_DIABATIC_GKG_DAY": "QV_DIABATIC",
    "RQCCUTEN_GKG_DAY": "RQCCUTEN",
    "RQCMPTEN_GKG_DAY": "RQCMPTEN",
    "RQCPHTEN_GKG_DAY": "RQCPHTEN",
    "QC_DIABATIC_GKG_DAY": "QC_DIABATIC",
    "RQRCUTEN_GKG_DAY": "RQRCUTEN",
    "RQRMPTEN_GKG_DAY": "RQRMPTEN",
    "RQRPHTEN_GKG_DAY": "RQRPHTEN",
    "RQICUTEN_GKG_DAY": "RQICUTEN",
    "RQIMPTEN_GKG_DAY": "RQIMPTEN",
    "RQIPHTEN_GKG_DAY": "RQIPHTEN",
    "RQSCUTEN_GKG_DAY": "RQSCUTEN",
    "RQSMPTEN_GKG_DAY": "RQSMPTEN",
    "RQSPHTEN_GKG_DAY": "RQSPHTEN",
    "RQGMPTEN_GKG_DAY": "RQGMPTEN",
    "RQGPHTEN_GKG_DAY": "RQGPHTEN",
    "QV_PHYSICS_TENDENCY_GKG_DAY": "RQVPHTEN",
    "QC_PHYSICS_TENDENCY_GKG_DAY": "RQCPHTEN",
    "QR_PHYSICS_TENDENCY_GKG_DAY": "RQRPHTEN",
    "QI_PHYSICS_TENDENCY_GKG_DAY": "RQIPHTEN",
    "QS_PHYSICS_TENDENCY_GKG_DAY": "RQSPHTEN",
    "QG_PHYSICS_TENDENCY_GKG_DAY": "RQGPHTEN",
    "TOTAL_PHYSICS_WATER_VAPOR_TENDENCY_GKG_DAY": "RQVPHTEN",
    "TOTAL_PHYSICS_CLOUD_WATER_TENDENCY_GKG_DAY": "RQCPHTEN",
    "TOTAL_PHYSICS_RAIN_WATER_TENDENCY_GKG_DAY": "RQRPHTEN",
    "TOTAL_PHYSICS_ICE_TENDENCY_GKG_DAY": "RQIPHTEN",
    "TOTAL_PHYSICS_SNOW_TENDENCY_GKG_DAY": "RQSPHTEN",
    "TOTAL_PHYSICS_GRAUPEL_TENDENCY_GKG_DAY": "RQGPHTEN",
    "QV_MICROPHYSICS_TENDENCY_GKG_DAY": "RQVMPTEN",
    "QC_MICROPHYSICS_TENDENCY_GKG_DAY": "RQCMPTEN",
    "QR_MICROPHYSICS_TENDENCY_GKG_DAY": "RQRMPTEN",
    "QI_MICROPHYSICS_TENDENCY_GKG_DAY": "RQIMPTEN",
    "QS_MICROPHYSICS_TENDENCY_GKG_DAY": "RQSMPTEN",
    "QG_MICROPHYSICS_TENDENCY_GKG_DAY": "RQGMPTEN",
    "QV_CUMULUS_TENDENCY_GKG_DAY": "RQVCUTEN",
    "QC_CUMULUS_TENDENCY_GKG_DAY": "RQCCUTEN",
    "QR_CUMULUS_TENDENCY_GKG_DAY": "RQRCUTEN",
    "QI_CUMULUS_TENDENCY_GKG_DAY": "RQICUTEN",
    "QS_CUMULUS_TENDENCY_GKG_DAY": "RQSCUTEN",
    "QV_PBL_TENDENCY_GKG_DAY": "RQVBLTEN",
    "QC_PBL_TENDENCY_GKG_DAY": "RQCBLTEN",
    "QI_PBL_TENDENCY_GKG_DAY": "RQIBLTEN",
    "TOTAL_DIABATIC_WATER_VAPOR_TENDENCY_GKG_DAY": "QV_DIABATIC",
    "TOTAL_DIABATIC_CLOUD_WATER_TENDENCY_GKG_DAY": "QC_DIABATIC",
}
_GKG_HOUR_TENDENCY_SOURCES: dict[str, str] = {
    output_name.removesuffix("_GKG_DAY") + "_GKG_HOUR": source_name
    for output_name, source_name in _GKG_DAY_TENDENCY_SOURCES.items()
}
_TWOD_EXTRA_FIELD_SOURCES: dict[str, tuple[str, ...]] = {
    "T2": ("t2",),
    "Q2": ("q2",),
    "U10": ("u10",),
    "V10": ("v10",),
    "PSFC": ("psfc",),
    "HGT": ("hgt", "ht"),
    "LANDMASK": ("landmask",),
    "COSALPHA": ("cosalpha", "cosa"),
    "SINALPHA": ("sinalpha", "sina"),
    "PBLH": ("pblh",),
    "TSK": ("tsk",),
    "HFX": ("hfx",),
    "QFX": ("qfx",),
    "LH": ("lh", "flx_lh"),
    "GRDFLX": ("grdflx",),
    "SNOW": ("snow",),
    "SNOWH": ("snowh",),
    "SNOWC": ("snowc",),
    "ALBEDO": ("albedo",),
    "EMISS": ("emiss",),
    "RAINC": ("rainc",),
    "RAINCV": ("raincv",),
    "PRATEC": ("pratec",),
    "RAINSH": ("rainsh",),
    "RAINNC": ("rainnc",),
    "RAINNCV": ("rainncv",),
    "SNOWNC": ("snownc",),
    "SNOWNCV": ("snowncv",),
    "GRAUPELNC": ("graupelnc",),
    "GRAUPELNCV": ("graupelncv",),
    "SR": ("sr",),
    "NCA": ("nca",),
    "HTOP": ("htop",),
    "HBOT": ("hbot",),
    "SWDNB": ("swdnb", "swdown"),
    "SWUPB": ("swupb",),
    "SWDNT": ("swdnt",),
    "SWUPT": ("swupt",),
    "LWDNB": ("lwdnb", "glw"),
    "LWUPB": ("lwupb",),
    "LWDNT": ("lwdnt",),
    "LWUPT": ("lwupt",),
}

EXTRA_FIELD_METADATA: dict[str, dict[str, Any]] = {
    **{
        name: {"description": description, "units": "kg kg-1"}
        for name, description in (
            ("QCLOUD", "Cloud water mixing ratio"),
            ("QRAIN", "Rain water mixing ratio"),
            ("QICE", "Ice mixing ratio"),
            ("QSNOW", "Snow mixing ratio"),
            ("QGRAUP", "Graupel mixing ratio"),
        )
    },
    **{
        name: {"description": description, "units": "W m-2"}
        for name, description in (
            ("SWDNB", "DOWNWARD SHORT WAVE FLUX AT BOTTOM"),
            ("SWUPB", "UPWARD SHORT WAVE FLUX AT BOTTOM"),
            ("SWDNT", "DOWNWARD SHORT WAVE FLUX AT TOP"),
            ("SWUPT", "UPWARD SHORT WAVE FLUX AT TOP"),
            ("LWDNB", "DOWNWARD LONG WAVE FLUX AT BOTTOM"),
            ("LWUPB", "UPWARD LONG WAVE FLUX AT BOTTOM"),
            ("LWDNT", "DOWNWARD LONG WAVE FLUX AT TOP"),
            ("LWUPT", "UPWARD LONG WAVE FLUX AT TOP"),
        )
    },
    "P": {"description": "perturbation pressure", "units": "Pa"},
    "PB": {"description": "BASE STATE PRESSURE", "units": "Pa"},
    "PH": {"description": "perturbation geopotential", "units": "m2 s-2", "stagger": "Z"},
    "PHB": {"description": "base-state geopotential", "units": "m2 s-2", "stagger": "Z"},
    "W": {"description": "z-wind component", "units": "m s-1", "stagger": "Z"},
    "T2": {"description": "TEMP at 2 M", "units": "K"},
    "Q2": {"description": "Water vapor mixing ratio at 2 M", "units": "kg kg-1"},
    "U10": {"description": "U at 10 M (grid-relative)", "units": "m s-1"},
    "V10": {"description": "V at 10 M (grid-relative)", "units": "m s-1"},
    "PSFC": {"description": "SFC PRESSURE", "units": "Pa"},
    "RAINNC": {"description": "ACCUMULATED TOTAL GRID SCALE PRECIPITATION", "units": "mm"},
    "RAINC": {"description": "ACCUMULATED TOTAL CUMULUS PRECIPITATION", "units": "mm"},
    "RAINSH": {"description": "ACCUMULATED SHALLOW CUMULUS PRECIPITATION", "units": "mm"},
    "HGT": {"description": "Terrain Height", "units": "m"},
    "LANDMASK": {"description": "LAND MASK (1 FOR LAND, 0 FOR WATER)", "units": "1"},
    "COSALPHA": {"description": "Local cosine of map rotation", "units": "1"},
    "SINALPHA": {"description": "Local sine of map rotation", "units": "1"},
    "PBLH": {"description": "PBL HEIGHT", "units": "m"},
    "TSK": {"description": "SURFACE SKIN TEMPERATURE", "units": "K"},
    "HFX": {"description": "UPWARD HEAT FLUX AT THE SURFACE", "units": "W m-2"},
    "QFX": {"description": "UPWARD MOISTURE FLUX AT THE SURFACE", "units": "kg m-2 s-1"},
    "LH": {"description": "LATENT HEAT FLUX AT THE SURFACE", "units": "W m-2"},
    "GRDFLX": {"description": "GROUND HEAT FLUX", "units": "W m-2"},
    "SNOW": {"description": "SNOW WATER EQUIVALENT", "units": "kg m-2"},
    "SNOWH": {"description": "PHYSICAL SNOW DEPTH", "units": "m"},
    "SNOWC": {"description": "SNOW COVERAGE", "units": "1"},
    "ALBEDO": {"description": "SURFACE ALBEDO", "units": "1"},
    "EMISS": {"description": "SURFACE EMISSIVITY", "units": "1"},
    "PI_PHY": {"description": "Exner function used to convert theta tendency to temperature tendency", "units": "1"},
    "RUCUTEN": {"description": "X wind tendency due to cumulus parameterization", "units": "m s-2"},
    "RVCUTEN": {"description": "Y wind tendency due to cumulus parameterization", "units": "m s-2"},
    "RTHCUTEN": {"description": "Theta tendency due to cumulus scheme", "units": "K s-1"},
    "RQVCUTEN": {"description": "Q_V tendency due to cumulus scheme", "units": "kg kg-1 s-1"},
    "RQCCUTEN": {"description": "Q_C tendency due to cumulus scheme", "units": "kg kg-1 s-1"},
    "RQRCUTEN": {"description": "Q_R tendency due to cumulus scheme", "units": "kg kg-1 s-1"},
    "RQICUTEN": {"description": "Q_I tendency due to cumulus scheme", "units": "kg kg-1 s-1"},
    "RQSCUTEN": {"description": "Q_S tendency due to cumulus scheme", "units": "kg kg-1 s-1"},
    "RTHRATEN": {"description": "Theta tendency due to radiation", "units": "K s-1"},
    "RTHRATENLW": {"description": "Uncoupled theta tendency due to longwave radiation", "units": "K s-1"},
    "RTHRATENLWC": {"description": "Uncoupled theta tendency due to clear-sky longwave radiation", "units": "K s-1"},
    "RTHRATENSW": {"description": "Uncoupled theta tendency due to shortwave radiation", "units": "K s-1"},
    "RTHRATENSWC": {"description": "Uncoupled theta tendency due to clear-sky shortwave radiation", "units": "K s-1"},
    "RUBLTEN": {"description": "X wind tendency due to PBL parameterization", "units": "m s-2"},
    "RVBLTEN": {"description": "Y wind tendency due to PBL parameterization", "units": "m s-2"},
    "RTHBLTEN": {"description": "Theta tendency due to PBL parameterization", "units": "K s-1"},
    "RQVBLTEN": {"description": "Q_V tendency due to PBL parameterization", "units": "kg kg-1 s-1"},
    "RQCBLTEN": {"description": "Q_C tendency due to PBL parameterization", "units": "kg kg-1 s-1"},
    "RQIBLTEN": {"description": "Q_I tendency due to PBL parameterization", "units": "kg kg-1 s-1"},
    "RQNIBLTEN": {"description": "Q_NI tendency due to PBL parameterization", "units": "# kg-1 s-1"},
    "RUSHTEN": {"description": "X wind tendency due to shallow cumulus scheme", "units": "m s-2"},
    "RVSHTEN": {"description": "Y wind tendency due to shallow cumulus scheme", "units": "m s-2"},
    "RTHSHTEN": {"description": "Theta tendency due to shallow cumulus scheme", "units": "K s-1"},
    "RQVSHTEN": {"description": "Q_V tendency due to shallow cumulus scheme", "units": "kg kg-1 s-1"},
    "RQCSHTEN": {"description": "Q_C tendency due to shallow cumulus scheme", "units": "kg kg-1 s-1"},
    "RQRSHTEN": {"description": "Q_R tendency due to shallow cumulus scheme", "units": "kg kg-1 s-1"},
    "RQISHTEN": {"description": "Q_I tendency due to shallow cumulus scheme", "units": "kg kg-1 s-1"},
    "RQSSHTEN": {"description": "Q_S tendency due to shallow cumulus scheme", "units": "kg kg-1 s-1"},
    "RQGSHTEN": {"description": "Q_G tendency due to shallow cumulus scheme", "units": "kg kg-1 s-1"},
    "RTHFTEN": {"description": "Total advective potential temperature tendency", "units": "K s-1"},
    "RQVFTEN": {"description": "Total advective moisture tendency", "units": "kg kg-1 s-1"},
    "RUNDGDTEN": {"description": "X wind tendency due to FDDA grid nudging", "units": "m s-2"},
    "RVNDGDTEN": {"description": "Y wind tendency due to FDDA grid nudging", "units": "m s-2"},
    "RTHNDGDTEN": {"description": "Theta tendency due to FDDA grid nudging", "units": "K s-1"},
    "RPHNDGDTEN": {"description": "Geopotential tendency due to FDDA grid nudging", "units": "m2 s-3"},
    "RQVNDGDTEN": {"description": "Q_V tendency due to FDDA grid nudging", "units": "kg kg-1 s-1"},
    "RTHMPTEN": {"description": "Theta tendency due to microphysics", "units": "K s-1"},
    "RQVMPTEN": {"description": "Q_V tendency due to microphysics", "units": "kg kg-1 s-1"},
    "RQCMPTEN": {"description": "Q_C tendency due to microphysics", "units": "kg kg-1 s-1"},
    "RQRMPTEN": {"description": "Q_R tendency due to microphysics", "units": "kg kg-1 s-1"},
    "RQIMPTEN": {"description": "Q_I tendency due to microphysics", "units": "kg kg-1 s-1"},
    "RQSMPTEN": {"description": "Q_S tendency due to microphysics", "units": "kg kg-1 s-1"},
    "RQGMPTEN": {"description": "Q_G tendency due to microphysics", "units": "kg kg-1 s-1"},
    "RUPHTEN": {"description": "Total physical X wind tendency", "units": "m s-2"},
    "RVPHTEN": {"description": "Total physical Y wind tendency", "units": "m s-2"},
    "RTHPHTEN": {"description": "Total physical theta tendency", "units": "K s-1"},
    "RQVPHTEN": {"description": "Total physical Q_V tendency", "units": "kg kg-1 s-1"},
    "RQCPHTEN": {"description": "Total physical Q_C tendency", "units": "kg kg-1 s-1"},
    "RQRPHTEN": {"description": "Total physical Q_R tendency", "units": "kg kg-1 s-1"},
    "RQIPHTEN": {"description": "Total physical Q_I tendency", "units": "kg kg-1 s-1"},
    "RQSPHTEN": {"description": "Total physical Q_S tendency", "units": "kg kg-1 s-1"},
    "RQGPHTEN": {"description": "Total physical Q_G tendency", "units": "kg kg-1 s-1"},
    "LW_HEATING_RATE": {"description": "Longwave radiation theta heating rate", "units": "K s-1"},
    "LW_CLEAR_HEATING_RATE": {"description": "Clear-sky longwave radiation theta heating rate", "units": "K s-1"},
    "SW_HEATING_RATE": {"description": "Shortwave radiation theta heating rate", "units": "K s-1"},
    "SW_CLEAR_HEATING_RATE": {"description": "Clear-sky shortwave radiation theta heating rate", "units": "K s-1"},
    "LONGWAVE_HEATING_RATE": {"description": "Longwave radiation theta heating rate", "units": "K s-1"},
    "LONGWAVE_CLEAR_SKY_HEATING_RATE": {"description": "Clear-sky longwave radiation theta heating rate", "units": "K s-1"},
    "SHORTWAVE_HEATING_RATE": {"description": "Shortwave radiation theta heating rate", "units": "K s-1"},
    "SHORTWAVE_CLEAR_SKY_HEATING_RATE": {"description": "Clear-sky shortwave radiation theta heating rate", "units": "K s-1"},
    "RADIATION_HEATING_RATE": {"description": "Total radiation theta heating rate", "units": "K s-1"},
    "TOTAL_RADIATION_HEATING_RATE": {"description": "Total radiation theta heating rate", "units": "K s-1"},
    "THETA_LW_TENDENCY": {"description": "Theta tendency from longwave radiation", "units": "K s-1"},
    "THETA_LW_CLEAR_TENDENCY": {"description": "Theta tendency from clear-sky longwave radiation", "units": "K s-1"},
    "THETA_SW_TENDENCY": {"description": "Theta tendency from shortwave radiation", "units": "K s-1"},
    "THETA_SW_CLEAR_TENDENCY": {"description": "Theta tendency from clear-sky shortwave radiation", "units": "K s-1"},
    "THETA_RADIATION_TENDENCY": {"description": "Theta tendency from total radiation", "units": "K s-1"},
    "CUMULUS_HEATING_RATE": {"description": "Cumulus theta heating rate", "units": "K s-1"},
    "THETA_CUMULUS_TENDENCY": {"description": "Theta tendency from cumulus parameterization", "units": "K s-1"},
    "PBL_HEATING_RATE": {"description": "PBL theta heating rate", "units": "K s-1"},
    "THETA_PBL_TENDENCY": {"description": "Theta tendency from PBL parameterization", "units": "K s-1"},
    "MICROPHYSICS_HEATING_RATE": {"description": "Microphysics theta heating rate", "units": "K s-1"},
    "THETA_MICROPHYSICS_TENDENCY": {"description": "Theta tendency from microphysics", "units": "K s-1"},
    "SHALLOW_CUMULUS_HEATING_RATE": {"description": "Shallow-cumulus theta heating rate", "units": "K s-1"},
    "THETA_SHALLOW_CUMULUS_TENDENCY": {"description": "Theta tendency from shallow cumulus", "units": "K s-1"},
    "PHYSICS_HEATING_RATE": {"description": "Total physical theta heating rate", "units": "K s-1"},
    "TOTAL_PHYSICS_HEATING_RATE": {"description": "Total physical theta heating rate", "units": "K s-1"},
    "THETA_PHYSICS_TENDENCY": {"description": "Theta tendency from all physical parameterizations", "units": "K s-1"},
    "ADVECTIVE_THETA_TENDENCY": {"description": "Advective potential temperature tendency", "units": "K s-1"},
    "TOTAL_DIABATIC_THETA_TENDENCY": {"description": "Total diabatic theta tendency", "units": "K s-1"},
    "THETA_DIABATIC_TENDENCY": {"description": "Total diabatic theta tendency", "units": "K s-1"},
    "U_CUMULUS_TENDENCY": {"description": "X wind cumulus tendency", "units": "m s-2"},
    "V_CUMULUS_TENDENCY": {"description": "Y wind cumulus tendency", "units": "m s-2"},
    "U_PBL_TENDENCY": {"description": "X wind PBL tendency", "units": "m s-2"},
    "V_PBL_TENDENCY": {"description": "Y wind PBL tendency", "units": "m s-2"},
    "U_SHALLOW_CUMULUS_TENDENCY": {"description": "X wind shallow-cumulus tendency", "units": "m s-2"},
    "V_SHALLOW_CUMULUS_TENDENCY": {"description": "Y wind shallow-cumulus tendency", "units": "m s-2"},
    "U_PHYSICS_TENDENCY": {"description": "Total physical X wind tendency", "units": "m s-2"},
    "V_PHYSICS_TENDENCY": {"description": "Total physical Y wind tendency", "units": "m s-2"},
    "QV_CUMULUS_TENDENCY": {"description": "Q_V cumulus tendency", "units": "kg kg-1 s-1"},
    "QC_CUMULUS_TENDENCY": {"description": "Q_C cumulus tendency", "units": "kg kg-1 s-1"},
    "QR_CUMULUS_TENDENCY": {"description": "Q_R cumulus tendency", "units": "kg kg-1 s-1"},
    "QI_CUMULUS_TENDENCY": {"description": "Q_I cumulus tendency", "units": "kg kg-1 s-1"},
    "QS_CUMULUS_TENDENCY": {"description": "Q_S cumulus tendency", "units": "kg kg-1 s-1"},
    "QV_PBL_TENDENCY": {"description": "Q_V PBL tendency", "units": "kg kg-1 s-1"},
    "QC_PBL_TENDENCY": {"description": "Q_C PBL tendency", "units": "kg kg-1 s-1"},
    "QI_PBL_TENDENCY": {"description": "Q_I PBL tendency", "units": "kg kg-1 s-1"},
    "QV_MICROPHYSICS_TENDENCY": {"description": "Q_V microphysics tendency", "units": "kg kg-1 s-1"},
    "QC_MICROPHYSICS_TENDENCY": {"description": "Q_C microphysics tendency", "units": "kg kg-1 s-1"},
    "QR_MICROPHYSICS_TENDENCY": {"description": "Q_R microphysics tendency", "units": "kg kg-1 s-1"},
    "QI_MICROPHYSICS_TENDENCY": {"description": "Q_I microphysics tendency", "units": "kg kg-1 s-1"},
    "QS_MICROPHYSICS_TENDENCY": {"description": "Q_S microphysics tendency", "units": "kg kg-1 s-1"},
    "QG_MICROPHYSICS_TENDENCY": {"description": "Q_G microphysics tendency", "units": "kg kg-1 s-1"},
    "QV_PHYSICS_TENDENCY": {"description": "Total physical Q_V tendency", "units": "kg kg-1 s-1"},
    "QC_PHYSICS_TENDENCY": {"description": "Total physical Q_C tendency", "units": "kg kg-1 s-1"},
    "QR_PHYSICS_TENDENCY": {"description": "Total physical Q_R tendency", "units": "kg kg-1 s-1"},
    "QI_PHYSICS_TENDENCY": {"description": "Total physical Q_I tendency", "units": "kg kg-1 s-1"},
    "QS_PHYSICS_TENDENCY": {"description": "Total physical Q_S tendency", "units": "kg kg-1 s-1"},
    "QG_PHYSICS_TENDENCY": {"description": "Total physical Q_G tendency", "units": "kg kg-1 s-1"},
    "TOTAL_DIABATIC_WATER_VAPOR_TENDENCY": {"description": "Total diabatic water vapor tendency", "units": "kg kg-1 s-1"},
    "TOTAL_DIABATIC_CLOUD_WATER_TENDENCY": {"description": "Total diabatic cloud water tendency", "units": "kg kg-1 s-1"},
    "LW_TEMPERATURE_HEATING_RATE": {"description": "Longwave radiation temperature heating rate", "units": "K s-1"},
    "LW_CLEAR_TEMPERATURE_HEATING_RATE": {"description": "Clear-sky longwave radiation temperature heating rate", "units": "K s-1"},
    "SW_TEMPERATURE_HEATING_RATE": {"description": "Shortwave radiation temperature heating rate", "units": "K s-1"},
    "SW_CLEAR_TEMPERATURE_HEATING_RATE": {"description": "Clear-sky shortwave radiation temperature heating rate", "units": "K s-1"},
    "LONGWAVE_TEMPERATURE_HEATING_RATE": {"description": "Longwave radiation temperature heating rate", "units": "K s-1"},
    "LONGWAVE_CLEAR_SKY_TEMPERATURE_HEATING_RATE": {"description": "Clear-sky longwave radiation temperature heating rate", "units": "K s-1"},
    "SHORTWAVE_TEMPERATURE_HEATING_RATE": {"description": "Shortwave radiation temperature heating rate", "units": "K s-1"},
    "SHORTWAVE_CLEAR_SKY_TEMPERATURE_HEATING_RATE": {"description": "Clear-sky shortwave radiation temperature heating rate", "units": "K s-1"},
    "RADIATION_TEMPERATURE_HEATING_RATE": {"description": "Total radiation temperature heating rate", "units": "K s-1"},
    "TOTAL_RADIATION_TEMPERATURE_HEATING_RATE": {"description": "Total radiation temperature heating rate", "units": "K s-1"},
    "CUMULUS_TEMPERATURE_HEATING_RATE": {"description": "Cumulus temperature heating rate", "units": "K s-1"},
    "PBL_TEMPERATURE_HEATING_RATE": {"description": "PBL temperature heating rate", "units": "K s-1"},
    "MICROPHYSICS_TEMPERATURE_HEATING_RATE": {"description": "Microphysics temperature heating rate", "units": "K s-1"},
    "SHALLOW_CUMULUS_TEMPERATURE_HEATING_RATE": {"description": "Shallow-cumulus temperature heating rate", "units": "K s-1"},
    "PHYSICS_TEMPERATURE_HEATING_RATE": {"description": "Total physical temperature heating rate", "units": "K s-1"},
    "TOTAL_PHYSICS_TEMPERATURE_HEATING_RATE": {"description": "Total physical temperature heating rate", "units": "K s-1"},
    "ADVECTIVE_TEMPERATURE_TENDENCY": {"description": "Advective temperature tendency", "units": "K s-1"},
    "DIABATIC_TEMPERATURE_TENDENCY": {"description": "Total diabatic temperature tendency", "units": "K s-1"},
    "TOTAL_DIABATIC_TEMPERATURE_TENDENCY": {"description": "Total diabatic temperature tendency", "units": "K s-1"},
    "RTHRATENLW_KHOUR": {"description": "Longwave radiation theta heating rate", "units": "K hour-1"},
    "RTHRATENLWC_KHOUR": {"description": "Clear-sky longwave radiation theta heating rate", "units": "K hour-1"},
    "RTHRATENSW_KHOUR": {"description": "Shortwave radiation theta heating rate", "units": "K hour-1"},
    "RTHRATENSWC_KHOUR": {"description": "Clear-sky shortwave radiation theta heating rate", "units": "K hour-1"},
    "RTHRATEN_KHOUR": {"description": "Total radiation theta heating rate", "units": "K hour-1"},
    "RTHCUTEN_KHOUR": {"description": "Cumulus theta heating rate", "units": "K hour-1"},
    "RTHBLTEN_KHOUR": {"description": "PBL theta heating rate", "units": "K hour-1"},
    "RTHMPTEN_KHOUR": {"description": "Microphysics theta heating rate", "units": "K hour-1"},
    "RTHSHTEN_KHOUR": {"description": "Shallow-cumulus theta heating rate", "units": "K hour-1"},
    "RTHFTEN_KHOUR": {"description": "Advective potential temperature tendency", "units": "K hour-1"},
    "H_DIABATIC_KHOUR": {"description": "Total diabatic theta tendency", "units": "K hour-1"},
    "RTHPHTEN_KHOUR": {"description": "Total physical theta heating rate", "units": "K hour-1"},
    "LW_HEATING_RATE_KHOUR": {"description": "Longwave radiation theta heating rate", "units": "K hour-1"},
    "LW_CLEAR_HEATING_RATE_KHOUR": {"description": "Clear-sky longwave radiation theta heating rate", "units": "K hour-1"},
    "SW_HEATING_RATE_KHOUR": {"description": "Shortwave radiation theta heating rate", "units": "K hour-1"},
    "SW_CLEAR_HEATING_RATE_KHOUR": {"description": "Clear-sky shortwave radiation theta heating rate", "units": "K hour-1"},
    "LONGWAVE_HEATING_RATE_KHOUR": {"description": "Longwave radiation theta heating rate", "units": "K hour-1"},
    "LONGWAVE_CLEAR_SKY_HEATING_RATE_KHOUR": {"description": "Clear-sky longwave radiation theta heating rate", "units": "K hour-1"},
    "SHORTWAVE_HEATING_RATE_KHOUR": {"description": "Shortwave radiation theta heating rate", "units": "K hour-1"},
    "SHORTWAVE_CLEAR_SKY_HEATING_RATE_KHOUR": {"description": "Clear-sky shortwave radiation theta heating rate", "units": "K hour-1"},
    "RADIATION_HEATING_RATE_KHOUR": {"description": "Total radiation theta heating rate", "units": "K hour-1"},
    "TOTAL_RADIATION_HEATING_RATE_KHOUR": {"description": "Total radiation theta heating rate", "units": "K hour-1"},
    "THETA_LW_TENDENCY_KHOUR": {"description": "Theta tendency from longwave radiation", "units": "K hour-1"},
    "THETA_LW_CLEAR_TENDENCY_KHOUR": {"description": "Theta tendency from clear-sky longwave radiation", "units": "K hour-1"},
    "THETA_SW_TENDENCY_KHOUR": {"description": "Theta tendency from shortwave radiation", "units": "K hour-1"},
    "THETA_SW_CLEAR_TENDENCY_KHOUR": {"description": "Theta tendency from clear-sky shortwave radiation", "units": "K hour-1"},
    "THETA_RADIATION_TENDENCY_KHOUR": {"description": "Theta tendency from total radiation", "units": "K hour-1"},
    "THETA_CUMULUS_TENDENCY_KHOUR": {"description": "Theta tendency from cumulus parameterization", "units": "K hour-1"},
    "THETA_PBL_TENDENCY_KHOUR": {"description": "Theta tendency from PBL parameterization", "units": "K hour-1"},
    "THETA_MICROPHYSICS_TENDENCY_KHOUR": {"description": "Theta tendency from microphysics", "units": "K hour-1"},
    "THETA_SHALLOW_CUMULUS_TENDENCY_KHOUR": {"description": "Theta tendency from shallow cumulus", "units": "K hour-1"},
    "THETA_PHYSICS_TENDENCY_KHOUR": {"description": "Theta tendency from all physical parameterizations", "units": "K hour-1"},
    "ADVECTIVE_THETA_TENDENCY_KHOUR": {"description": "Advective potential temperature tendency", "units": "K hour-1"},
    "TOTAL_DIABATIC_THETA_TENDENCY_KHOUR": {"description": "Total diabatic theta tendency", "units": "K hour-1"},
    "THETA_DIABATIC_TENDENCY_KHOUR": {"description": "Total diabatic theta tendency", "units": "K hour-1"},
    "RTHRATENLW_KDAY": {"description": "Longwave radiation theta heating rate", "units": "K day-1"},
    "RTHRATENLWC_KDAY": {"description": "Clear-sky longwave radiation theta heating rate", "units": "K day-1"},
    "RTHRATENSW_KDAY": {"description": "Shortwave radiation theta heating rate", "units": "K day-1"},
    "RTHRATENSWC_KDAY": {"description": "Clear-sky shortwave radiation theta heating rate", "units": "K day-1"},
    "RTHRATEN_KDAY": {"description": "Total radiation theta heating rate", "units": "K day-1"},
    "RTHCUTEN_KDAY": {"description": "Cumulus theta heating rate", "units": "K day-1"},
    "RTHBLTEN_KDAY": {"description": "PBL theta heating rate", "units": "K day-1"},
    "RTHMPTEN_KDAY": {"description": "Microphysics theta heating rate", "units": "K day-1"},
    "RTHSHTEN_KDAY": {"description": "Shallow-cumulus theta heating rate", "units": "K day-1"},
    "RTHFTEN_KDAY": {"description": "Advective potential temperature tendency", "units": "K day-1"},
    "H_DIABATIC_KDAY": {"description": "Total diabatic theta tendency", "units": "K day-1"},
    "RTHPHTEN_KDAY": {"description": "Total physical theta heating rate", "units": "K day-1"},
    "LW_HEATING_RATE_KDAY": {"description": "Longwave radiation theta heating rate", "units": "K day-1"},
    "LW_CLEAR_HEATING_RATE_KDAY": {"description": "Clear-sky longwave radiation theta heating rate", "units": "K day-1"},
    "SW_HEATING_RATE_KDAY": {"description": "Shortwave radiation theta heating rate", "units": "K day-1"},
    "SW_CLEAR_HEATING_RATE_KDAY": {"description": "Clear-sky shortwave radiation theta heating rate", "units": "K day-1"},
    "THETA_LW_TENDENCY_KDAY": {"description": "Theta tendency from longwave radiation", "units": "K day-1"},
    "THETA_LW_CLEAR_TENDENCY_KDAY": {"description": "Theta tendency from clear-sky longwave radiation", "units": "K day-1"},
    "THETA_SW_TENDENCY_KDAY": {"description": "Theta tendency from shortwave radiation", "units": "K day-1"},
    "THETA_SW_CLEAR_TENDENCY_KDAY": {"description": "Theta tendency from clear-sky shortwave radiation", "units": "K day-1"},
    "THETA_RADIATION_TENDENCY_KDAY": {"description": "Theta tendency from total radiation", "units": "K day-1"},
    "THETA_CUMULUS_TENDENCY_KDAY": {"description": "Theta tendency from cumulus parameterization", "units": "K day-1"},
    "THETA_PBL_TENDENCY_KDAY": {"description": "Theta tendency from PBL parameterization", "units": "K day-1"},
    "THETA_MICROPHYSICS_TENDENCY_KDAY": {"description": "Theta tendency from microphysics", "units": "K day-1"},
    "THETA_SHALLOW_CUMULUS_TENDENCY_KDAY": {"description": "Theta tendency from shallow cumulus", "units": "K day-1"},
    "THETA_PHYSICS_TENDENCY_KDAY": {"description": "Theta tendency from all physical parameterizations", "units": "K day-1"},
    "LONGWAVE_HEATING_RATE_KDAY": {"description": "Longwave radiation theta heating rate", "units": "K day-1"},
    "LONGWAVE_CLEAR_SKY_HEATING_RATE_KDAY": {"description": "Clear-sky longwave radiation theta heating rate", "units": "K day-1"},
    "SHORTWAVE_HEATING_RATE_KDAY": {"description": "Shortwave radiation theta heating rate", "units": "K day-1"},
    "SHORTWAVE_CLEAR_SKY_HEATING_RATE_KDAY": {"description": "Clear-sky shortwave radiation theta heating rate", "units": "K day-1"},
    "LONGWAVE_RADIATION_HEATING_RATE_KDAY": {"description": "Longwave radiation theta heating rate", "units": "K day-1"},
    "LONGWAVE_CLEAR_SKY_RADIATION_HEATING_RATE_KDAY": {"description": "Clear-sky longwave radiation theta heating rate", "units": "K day-1"},
    "SHORTWAVE_RADIATION_HEATING_RATE_KDAY": {"description": "Shortwave radiation theta heating rate", "units": "K day-1"},
    "SHORTWAVE_CLEAR_SKY_RADIATION_HEATING_RATE_KDAY": {"description": "Clear-sky shortwave radiation theta heating rate", "units": "K day-1"},
    "TOTAL_RADIATION_HEATING_RATE_KDAY": {"description": "Total radiation theta heating rate", "units": "K day-1"},
    "TOTAL_RADIATION_THETA_TENDENCY_KDAY": {"description": "Theta tendency from total radiation", "units": "K day-1"},
    "RADIATION_HEATING_RATE_KDAY": {"description": "Total radiation theta heating rate", "units": "K day-1"},
    "CUMULUS_HEATING_RATE_KDAY": {"description": "Cumulus theta heating rate", "units": "K day-1"},
    "PBL_HEATING_RATE_KDAY": {"description": "PBL theta heating rate", "units": "K day-1"},
    "MICROPHYSICS_HEATING_RATE_KDAY": {"description": "Microphysics theta heating rate", "units": "K day-1"},
    "SHALLOW_CUMULUS_HEATING_RATE_KDAY": {"description": "Shallow-cumulus theta heating rate", "units": "K day-1"},
    "PHYSICS_HEATING_RATE_KDAY": {"description": "Total physical theta heating rate", "units": "K day-1"},
    "TOTAL_PHYSICS_HEATING_RATE_KDAY": {"description": "Total physical theta heating rate", "units": "K day-1"},
    "TOTAL_PHYSICS_THETA_TENDENCY_KDAY": {"description": "Theta tendency from all physical parameterizations", "units": "K day-1"},
    "TOTAL_DIABATIC_THETA_TENDENCY_KDAY": {"description": "Total diabatic theta tendency", "units": "K day-1"},
    "ADVECTIVE_THETA_TENDENCY_KDAY": {"description": "Advective potential temperature tendency", "units": "K day-1"},
    "RUCUTEN_MS_DAY": {"description": "X wind cumulus tendency", "units": "m s-1 day-1"},
    "RVCUTEN_MS_DAY": {"description": "Y wind cumulus tendency", "units": "m s-1 day-1"},
    "RUBLTEN_MS_DAY": {"description": "X wind PBL tendency", "units": "m s-1 day-1"},
    "RVBLTEN_MS_DAY": {"description": "Y wind PBL tendency", "units": "m s-1 day-1"},
    "RUSHTEN_MS_DAY": {"description": "X wind shallow-cumulus tendency", "units": "m s-1 day-1"},
    "RVSHTEN_MS_DAY": {"description": "Y wind shallow-cumulus tendency", "units": "m s-1 day-1"},
    "RUPHTEN_MS_DAY": {"description": "Total physical X wind tendency", "units": "m s-1 day-1"},
    "RVPHTEN_MS_DAY": {"description": "Total physical Y wind tendency", "units": "m s-1 day-1"},
    "U_CUMULUS_TENDENCY_MS_DAY": {"description": "X wind cumulus tendency", "units": "m s-1 day-1"},
    "V_CUMULUS_TENDENCY_MS_DAY": {"description": "Y wind cumulus tendency", "units": "m s-1 day-1"},
    "U_PBL_TENDENCY_MS_DAY": {"description": "X wind PBL tendency", "units": "m s-1 day-1"},
    "V_PBL_TENDENCY_MS_DAY": {"description": "Y wind PBL tendency", "units": "m s-1 day-1"},
    "U_SHALLOW_CUMULUS_TENDENCY_MS_DAY": {"description": "X wind shallow-cumulus tendency", "units": "m s-1 day-1"},
    "V_SHALLOW_CUMULUS_TENDENCY_MS_DAY": {"description": "Y wind shallow-cumulus tendency", "units": "m s-1 day-1"},
    "U_PHYSICS_TENDENCY_MS_DAY": {"description": "Total physical X wind tendency", "units": "m s-1 day-1"},
    "V_PHYSICS_TENDENCY_MS_DAY": {"description": "Total physical Y wind tendency", "units": "m s-1 day-1"},
    "TOTAL_PHYSICS_U_TENDENCY_MS_DAY": {"description": "Total physical X wind tendency", "units": "m s-1 day-1"},
    "TOTAL_PHYSICS_V_TENDENCY_MS_DAY": {"description": "Total physical Y wind tendency", "units": "m s-1 day-1"},
    "RQVCUTEN_GKG_DAY": {"description": "Q_V cumulus tendency", "units": "g kg-1 day-1"},
    "RQVBLTEN_GKG_DAY": {"description": "Q_V PBL tendency", "units": "g kg-1 day-1"},
    "RQCBLTEN_GKG_DAY": {"description": "Q_C PBL tendency", "units": "g kg-1 day-1"},
    "RQIBLTEN_GKG_DAY": {"description": "Q_I PBL tendency", "units": "g kg-1 day-1"},
    "RQVMPTEN_GKG_DAY": {"description": "Q_V microphysics tendency", "units": "g kg-1 day-1"},
    "RQVSHTEN_GKG_DAY": {"description": "Q_V shallow-cumulus tendency", "units": "g kg-1 day-1"},
    "RQCSHTEN_GKG_DAY": {"description": "Q_C shallow-cumulus tendency", "units": "g kg-1 day-1"},
    "RQRSHTEN_GKG_DAY": {"description": "Q_R shallow-cumulus tendency", "units": "g kg-1 day-1"},
    "RQISHTEN_GKG_DAY": {"description": "Q_I shallow-cumulus tendency", "units": "g kg-1 day-1"},
    "RQSSHTEN_GKG_DAY": {"description": "Q_S shallow-cumulus tendency", "units": "g kg-1 day-1"},
    "RQGSHTEN_GKG_DAY": {"description": "Q_G shallow-cumulus tendency", "units": "g kg-1 day-1"},
    "RQVPHTEN_GKG_DAY": {"description": "Total physical Q_V tendency", "units": "g kg-1 day-1"},
    "RQVFTEN_GKG_DAY": {"description": "Advective Q_V tendency", "units": "g kg-1 day-1"},
    "QV_DIABATIC_GKG_DAY": {"description": "Total diabatic Q_V tendency", "units": "g kg-1 day-1"},
    "RQCCUTEN_GKG_DAY": {"description": "Q_C cumulus tendency", "units": "g kg-1 day-1"},
    "RQCMPTEN_GKG_DAY": {"description": "Q_C microphysics tendency", "units": "g kg-1 day-1"},
    "RQCPHTEN_GKG_DAY": {"description": "Total physical Q_C tendency", "units": "g kg-1 day-1"},
    "QC_DIABATIC_GKG_DAY": {"description": "Total diabatic Q_C tendency", "units": "g kg-1 day-1"},
    "RQRCUTEN_GKG_DAY": {"description": "Q_R cumulus tendency", "units": "g kg-1 day-1"},
    "RQRMPTEN_GKG_DAY": {"description": "Q_R microphysics tendency", "units": "g kg-1 day-1"},
    "RQRPHTEN_GKG_DAY": {"description": "Total physical Q_R tendency", "units": "g kg-1 day-1"},
    "RQICUTEN_GKG_DAY": {"description": "Q_I cumulus tendency", "units": "g kg-1 day-1"},
    "RQIMPTEN_GKG_DAY": {"description": "Q_I microphysics tendency", "units": "g kg-1 day-1"},
    "RQIPHTEN_GKG_DAY": {"description": "Total physical Q_I tendency", "units": "g kg-1 day-1"},
    "RQSCUTEN_GKG_DAY": {"description": "Q_S cumulus tendency", "units": "g kg-1 day-1"},
    "RQSMPTEN_GKG_DAY": {"description": "Q_S microphysics tendency", "units": "g kg-1 day-1"},
    "RQSPHTEN_GKG_DAY": {"description": "Total physical Q_S tendency", "units": "g kg-1 day-1"},
    "RQGMPTEN_GKG_DAY": {"description": "Q_G microphysics tendency", "units": "g kg-1 day-1"},
    "RQGPHTEN_GKG_DAY": {"description": "Total physical Q_G tendency", "units": "g kg-1 day-1"},
    "QV_PHYSICS_TENDENCY_GKG_DAY": {"description": "Total physical Q_V tendency", "units": "g kg-1 day-1"},
    "QC_PHYSICS_TENDENCY_GKG_DAY": {"description": "Total physical Q_C tendency", "units": "g kg-1 day-1"},
    "QR_PHYSICS_TENDENCY_GKG_DAY": {"description": "Total physical Q_R tendency", "units": "g kg-1 day-1"},
    "QI_PHYSICS_TENDENCY_GKG_DAY": {"description": "Total physical Q_I tendency", "units": "g kg-1 day-1"},
    "QS_PHYSICS_TENDENCY_GKG_DAY": {"description": "Total physical Q_S tendency", "units": "g kg-1 day-1"},
    "QG_PHYSICS_TENDENCY_GKG_DAY": {"description": "Total physical Q_G tendency", "units": "g kg-1 day-1"},
    "TOTAL_PHYSICS_WATER_VAPOR_TENDENCY_GKG_DAY": {"description": "Total physical water vapor tendency", "units": "g kg-1 day-1"},
    "TOTAL_PHYSICS_CLOUD_WATER_TENDENCY_GKG_DAY": {"description": "Total physical cloud water tendency", "units": "g kg-1 day-1"},
    "TOTAL_PHYSICS_RAIN_WATER_TENDENCY_GKG_DAY": {"description": "Total physical rain water tendency", "units": "g kg-1 day-1"},
    "TOTAL_PHYSICS_ICE_TENDENCY_GKG_DAY": {"description": "Total physical ice tendency", "units": "g kg-1 day-1"},
    "TOTAL_PHYSICS_SNOW_TENDENCY_GKG_DAY": {"description": "Total physical snow tendency", "units": "g kg-1 day-1"},
    "TOTAL_PHYSICS_GRAUPEL_TENDENCY_GKG_DAY": {"description": "Total physical graupel tendency", "units": "g kg-1 day-1"},
    "QV_MICROPHYSICS_TENDENCY_GKG_DAY": {"description": "Q_V microphysics tendency", "units": "g kg-1 day-1"},
    "QC_MICROPHYSICS_TENDENCY_GKG_DAY": {"description": "Q_C microphysics tendency", "units": "g kg-1 day-1"},
    "QR_MICROPHYSICS_TENDENCY_GKG_DAY": {"description": "Q_R microphysics tendency", "units": "g kg-1 day-1"},
    "QI_MICROPHYSICS_TENDENCY_GKG_DAY": {"description": "Q_I microphysics tendency", "units": "g kg-1 day-1"},
    "QS_MICROPHYSICS_TENDENCY_GKG_DAY": {"description": "Q_S microphysics tendency", "units": "g kg-1 day-1"},
    "QG_MICROPHYSICS_TENDENCY_GKG_DAY": {"description": "Q_G microphysics tendency", "units": "g kg-1 day-1"},
    "QV_CUMULUS_TENDENCY_GKG_DAY": {"description": "Q_V cumulus tendency", "units": "g kg-1 day-1"},
    "QC_CUMULUS_TENDENCY_GKG_DAY": {"description": "Q_C cumulus tendency", "units": "g kg-1 day-1"},
    "QR_CUMULUS_TENDENCY_GKG_DAY": {"description": "Q_R cumulus tendency", "units": "g kg-1 day-1"},
    "QI_CUMULUS_TENDENCY_GKG_DAY": {"description": "Q_I cumulus tendency", "units": "g kg-1 day-1"},
    "QS_CUMULUS_TENDENCY_GKG_DAY": {"description": "Q_S cumulus tendency", "units": "g kg-1 day-1"},
    "QV_PBL_TENDENCY_GKG_DAY": {"description": "Q_V PBL tendency", "units": "g kg-1 day-1"},
    "QC_PBL_TENDENCY_GKG_DAY": {"description": "Q_C PBL tendency", "units": "g kg-1 day-1"},
    "QI_PBL_TENDENCY_GKG_DAY": {"description": "Q_I PBL tendency", "units": "g kg-1 day-1"},
    "TOTAL_DIABATIC_WATER_VAPOR_TENDENCY_GKG_DAY": {"description": "Total diabatic water vapor tendency", "units": "g kg-1 day-1"},
    "TOTAL_DIABATIC_CLOUD_WATER_TENDENCY_GKG_DAY": {"description": "Total diabatic cloud water tendency", "units": "g kg-1 day-1"},
}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_GOAL_REFERENCE = REPO_ROOT / "goal.md"
R_D = 287.0
R_V = 461.6
T0 = 300.0
WRF_PROJECTION_ATTRS = (
    "TITLE",
    "START_DATE",
    "SIMULATION_START_DATE",
    "WEST-EAST_GRID_DIMENSION",
    "SOUTH-NORTH_GRID_DIMENSION",
    "BOTTOM-TOP_GRID_DIMENSION",
    "DX",
    "DY",
    "CEN_LAT",
    "CEN_LON",
    "TRUELAT1",
    "TRUELAT2",
    "MOAD_CEN_LAT",
    "STAND_LON",
    "POLE_LAT",
    "POLE_LON",
    "MAP_PROJ",
    "MAP_PROJ_CHAR",
    "MMINLU",
    "NUM_LAND_CAT",
    "ISWATER",
    "ISLAKE",
    "ISICE",
    "ISURBAN",
    "ISOILWATER",
    "GRIDTYPE",
)
FIELD_METADATA = {
    "U": {
        "FieldType": np.int32(104),
        "MemoryOrder": "XYZ",
        "description": "x-wind component",
        "units": "m s-1",
        "stagger": "X",
        "coordinates": "XLONG_U XLAT_U XTIME",
    },
    "V": {
        "FieldType": np.int32(104),
        "MemoryOrder": "XYZ",
        "description": "y-wind component",
        "units": "m s-1",
        "stagger": "Y",
        "coordinates": "XLONG_V XLAT_V XTIME",
    },
    "T": {
        "FieldType": np.int32(104),
        "MemoryOrder": "XYZ",
        "description": "perturbation potential temperature theta-t0",
        "units": "K",
        "stagger": "",
        "coordinates": "XLONG XLAT XTIME",
    },
    "QVAPOR": {
        "FieldType": np.int32(104),
        "MemoryOrder": "XYZ",
        "description": "Water vapor mixing ratio",
        "units": "kg kg-1",
        "stagger": "",
        "coordinates": "XLONG XLAT XTIME",
    },
}


@dataclass(frozen=True)
class PrognosticSnapshot:
    """U/V/T/QVAPOR snapshot in WRF NetCDF dimension order."""

    elapsed_seconds: int
    timestamp: str
    u: Any
    v: Any
    t: Any
    qvapor: Any
    extra: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class FieldComparison:
    """Single field error against a WRF history output frame."""

    hour: int
    elapsed_seconds: int
    timestamp: str
    field: str
    rmse: float
    nrmse_percent: float
    max_abs_error: float
    units: str
    finite: bool


@dataclass(frozen=True)
class TrajectoryResult:
    """Result of a strict multi-step WRF-JAX trajectory."""

    assembly: RuntimeAssembly
    final_step: SolveEmResult
    snapshots: tuple[PrognosticSnapshot, ...]
    runtime_metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class MilestoneOutput:
    """Files written for a milestone trajectory validation run."""

    output_dir: Path
    netcdf_path: Path
    comparison_path: Path | None
    manifest_path: Path
    nrmse_summary_path: Path | None = None
    diagnostic_fields_path: Path | None = None
    goal_check_path: Path | None = None
    nonfinite_summary_path: Path | None = None


def _decode_wrf_times(values: Any) -> tuple[str, ...]:
    times: list[str] = []
    for row in np.asarray(values):
        row_array = np.asarray(row)
        if row_array.dtype.kind == "S":
            times.append(row_array.tobytes().decode("ascii").rstrip("\x00"))
        else:
            times.append("".join(chr(int(item)) for item in row_array).rstrip("\x00"))
    return tuple(times)


def _case_start(timestamp: str) -> datetime:
    return datetime.strptime(timestamp, "%Y-%m-%d_%H:%M:%S")


def _timestamp(start_timestamp: str, elapsed_seconds: int) -> str:
    return (_case_start(start_timestamp) + timedelta(seconds=int(elapsed_seconds))).strftime("%Y-%m-%d_%H:%M:%S")


def _trajectory_start_timestamp(snapshot: PrognosticSnapshot) -> str:
    return _timestamp(snapshot.timestamp, -int(snapshot.elapsed_seconds))


def _times_string_array(timestamps: list[str]) -> np.ndarray:
    values: list[bytes] = []
    for timestamp in timestamps:
        encoded = timestamp.encode("ascii")
        if len(encoded) != 19:
            raise ValueError(f"WRF timestamp must be 19 ASCII characters, got {timestamp!r}")
        values.append(encoded)
    return np.asarray(values, dtype="S19")


def _xkj_to_wrf(values: Any) -> Any:
    arr = jnp.asarray(values)
    if arr.ndim != 3:
        raise ValueError(f"expected 3-D WRF-JAX field, got shape {arr.shape}")
    return jnp.transpose(arr, (1, 2, 0))


def _crop_to_shape(values: Any, shape: tuple[int, int, int], i0: int = 0, j0: int = 0) -> Any:
    arr = jnp.asarray(values)
    if arr.ndim != 3:
        raise ValueError(f"expected 3-D WRF-JAX field, got shape {arr.shape}")
    ni, nk, nj = (int(dim) for dim in shape)
    return arr[int(i0) : int(i0) + ni, :nk, int(j0) : int(j0) + nj]


def _crop2_to_shape(values: Any, shape: tuple[int, int], i0: int = 0, j0: int = 0) -> Any:
    arr = jnp.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D WRF-JAX field, got shape {arr.shape}")
    ni, nj = (int(dim) for dim in shape)
    return arr[int(i0) : int(i0) + ni, int(j0) : int(j0) + nj]


def _xy_to_wrf(values: Any) -> Any:
    arr = jnp.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D WRF-JAX field, got shape {arr.shape}")
    return jnp.transpose(arr, (1, 0))


def _default_extra_attrs(name: str, memory_order: str) -> dict[str, Any]:
    attrs = {
        "FieldType": np.int32(104),
        "MemoryOrder": memory_order,
        "description": name,
        "units": "",
        "stagger": "",
        "coordinates": "XLONG XLAT XTIME",
    }
    attrs.update(_extra_field_metadata(name))
    return attrs


def _extra_field_metadata(name: str) -> dict[str, Any]:
    direct = EXTRA_FIELD_METADATA.get(name)
    if direct is not None:
        return dict(direct)
    suffix_units = (
        ("_KHOUR", "K hour-1"),
        ("_KDAY", "K day-1"),
        ("_MS_HOUR", "m s-1 hour-1"),
        ("_MS_DAY", "m s-1 day-1"),
        ("_GKG_HOUR", "g kg-1 hour-1"),
        ("_GKG_DAY", "g kg-1 day-1"),
    )
    for suffix, units in suffix_units:
        if not name.endswith(suffix):
            continue
        base = name[: -len(suffix)]
        base_metadata = EXTRA_FIELD_METADATA.get(base)
        if base_metadata is None and suffix == "_MS_HOUR":
            base_metadata = EXTRA_FIELD_METADATA.get(base + "_MS_DAY")
        if base_metadata is None and suffix == "_GKG_HOUR":
            base_metadata = EXTRA_FIELD_METADATA.get(base + "_GKG_DAY")
        if base_metadata is None:
            base_metadata = {}
        description = str(base_metadata.get("description", base if base else name))
        return {"description": description, "units": units}
    return {}


def _replace_last_axis_component(array: Any, index: int, value: Any) -> Any:
    arr = jnp.asarray(array)
    component = jnp.expand_dims(jnp.asarray(value, dtype=arr.dtype), axis=-1)
    return jnp.concatenate([arr[..., : int(index)], component, arr[..., int(index) + 1 :]], axis=-1)


def _moist_theta_to_dry_theta_perturb(theta_m_perturb: Any, qvapor: Any) -> Any:
    theta_m = jnp.asarray(theta_m_perturb) + jnp.asarray(T0, dtype=jnp.asarray(theta_m_perturb).dtype)
    qv = jnp.asarray(qvapor)
    ratio = jnp.asarray(R_V, dtype=theta_m.dtype) / jnp.asarray(R_D, dtype=theta_m.dtype)
    return theta_m / (jnp.asarray(1.0, dtype=theta_m.dtype) + ratio * qv) - jnp.asarray(T0, dtype=theta_m.dtype)


def _snapshot_arrays_from_fields(
    fields: dict[str, Any],
    config_flags: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    if _qvapor_only_carry_enabled() and "qvapor" in fields:
        qvapor = fields["qvapor"]
    else:
        qvapor = jnp.asarray(fields["moist"])[..., 1] if "moist" in fields else fields["qvapor"]
    if config_flags is not None and int(getattr(config_flags, "use_theta_m", 0)) == 1:
        t_history = _moist_theta_to_dry_theta_perturb(fields["t_2"], qvapor)
    else:
        t_history = fields["t_2"]
    qv_arr = jnp.asarray(qvapor)
    u_arr = jnp.asarray(fields["u_2"])
    v_arr = jnp.asarray(fields["v_2"])
    i0 = max(1 - int(fields.get("ims", 1)), 0)
    j0 = max(1 - int(fields.get("jms", 1)), 0)
    if "ide" in fields and "jde" in fields:
        mass_shape = (int(fields["ide"]) - 1, int(qv_arr.shape[1]), int(fields["jde"]) - 1)
        u_shape = (int(fields["ide"]), int(u_arr.shape[1]), int(fields["jde"]) - 1)
        v_shape = (int(fields["ide"]) - 1, int(v_arr.shape[1]), int(fields["jde"]))
    else:
        mass_shape = (max(int(qv_arr.shape[0]) - 1, 1), int(qv_arr.shape[1]), max(int(qv_arr.shape[2]) - 1, 1))
        u_shape = (int(u_arr.shape[0]), int(u_arr.shape[1]), mass_shape[2])
        v_shape = (mass_shape[0], int(v_arr.shape[1]), int(v_arr.shape[2]))
    t_history = _crop_to_shape(t_history, mass_shape, i0, j0)
    qvapor = _crop_to_shape(qvapor, mass_shape, i0, j0)
    return (
        _xkj_to_wrf(_crop_to_shape(fields["u_2"], u_shape, i0, j0)),
        _xkj_to_wrf(_crop_to_shape(fields["v_2"], v_shape, i0, j0)),
        _xkj_to_wrf(t_history),
        _xkj_to_wrf(qvapor),
    )


def _first_available_field(fields: Mapping[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in fields and fields[name] is not None:
            return fields[name]
    return None


def _sum_available_extra_fields(extras: Mapping[str, Any], names: tuple[str, ...]) -> Any | None:
    total: Any | None = None
    for name in names:
        if name not in extras:
            continue
        value = jnp.asarray(extras[name])
        total = value if total is None else total + value
    return total


def _add_derived_tendency_extras(extras: dict[str, Any]) -> None:
    for output_name, components in _TOTAL_PHYSICS_TENDENCY_COMPONENTS.items():
        value = _sum_available_extra_fields(extras, components)
        if value is not None:
            extras[output_name] = value
    for output_name, source_name in _TENDENCY_ALIAS_SOURCES.items():
        if source_name in extras:
            extras[output_name] = jnp.asarray(extras[source_name])
    if "PI_PHY" in extras:
        pi = jnp.asarray(extras["PI_PHY"])
        for output_name, source_name in _TEMPERATURE_HEATING_SOURCES.items():
            if source_name in extras:
                source = jnp.asarray(extras[source_name])
                extras[output_name] = source * jnp.asarray(pi, dtype=source.dtype)
    for output_name, source_name in _KHOUR_TENDENCY_SOURCES.items():
        if source_name in extras:
            source = jnp.asarray(extras[source_name])
            extras[output_name] = source * jnp.asarray(SECONDS_PER_HOUR, dtype=source.dtype)
    for output_name, source_name in _TEMPERATURE_HEATING_KHOUR_SOURCES.items():
        if source_name in extras:
            source = jnp.asarray(extras[source_name])
            extras[output_name] = source * jnp.asarray(SECONDS_PER_HOUR, dtype=source.dtype)
    for output_name, source_name in _KDAY_TENDENCY_SOURCES.items():
        if source_name in extras:
            extras[output_name] = jnp.asarray(extras[source_name]) * jnp.asarray(SECONDS_PER_DAY, dtype=jnp.asarray(extras[source_name]).dtype)
    for output_name, source_name in _TEMPERATURE_HEATING_KDAY_SOURCES.items():
        if source_name in extras:
            source = jnp.asarray(extras[source_name])
            extras[output_name] = source * jnp.asarray(SECONDS_PER_DAY, dtype=source.dtype)
    for output_name, source_name in _MS_DAY_TENDENCY_SOURCES.items():
        if source_name in extras:
            source = jnp.asarray(extras[source_name])
            extras[output_name] = source * jnp.asarray(SECONDS_PER_DAY, dtype=source.dtype)
    for output_name, source_name in _MS_HOUR_TENDENCY_SOURCES.items():
        if source_name in extras:
            source = jnp.asarray(extras[source_name])
            extras[output_name] = source * jnp.asarray(SECONDS_PER_HOUR, dtype=source.dtype)
    gkg_day = jnp.asarray(SECONDS_PER_DAY * GRAMS_PER_KILOGRAM)
    for output_name, source_name in _GKG_DAY_TENDENCY_SOURCES.items():
        if source_name in extras:
            source = jnp.asarray(extras[source_name])
            extras[output_name] = source * jnp.asarray(gkg_day, dtype=source.dtype)
    gkg_hour = jnp.asarray(SECONDS_PER_HOUR * GRAMS_PER_KILOGRAM)
    for output_name, source_name in _GKG_HOUR_TENDENCY_SOURCES.items():
        if source_name in extras:
            source = jnp.asarray(extras[source_name])
            extras[output_name] = source * jnp.asarray(gkg_hour, dtype=source.dtype)


def _extra_output_mode(default: str = "compact") -> str:
    value = os.environ.get("WRF_PY_EXTRA_OUTPUT_MODE", default).strip().lower()
    if value in {"", "compact"}:
        return "compact"
    if value in {"key", "keys", "minimal"}:
        return "key"
    if value in {"full", "all"}:
        return "full"
    return default


def _extra_output_field_allowlist() -> frozenset[str] | None:
    mode = _extra_output_mode()
    if mode == "full":
        return None
    if mode == "key":
        extras = set(MICROPHYSICS_KEY_OUTPUTS + CUMULUS_KEY_OUTPUTS + RADIATION_KEY_OUTPUTS)
    else:
        extras = set(COMPACT_EXTRA_OUTPUTS)
    extras.update(METEOROLOGICAL_BASE_OUTPUTS)
    raw = os.environ.get("WRF_PY_EXTRA_OUTPUT_FIELDS", "")
    for item in raw.split(","):
        name = item.strip().upper()
        if name:
            extras.add(name)
    return frozenset(extras)


def _filter_output_extras(extras: dict[str, Any]) -> dict[str, Any]:
    allowlist = _extra_output_field_allowlist()
    if allowlist is None:
        return extras
    return {name: value for name, value in extras.items() if name in allowlist}


def _extras_with_derived_tendencies(snapshot: PrognosticSnapshot) -> dict[str, Any]:
    extras = dict(snapshot.extra or {})
    _add_derived_tendency_extras(extras)
    return _filter_output_extras(extras)


def _snapshot_extra_fields_from_fields(fields: dict[str, Any]) -> dict[str, Any]:
    qv_arr = jnp.asarray(fields["qvapor"] if "qvapor" in fields else jnp.asarray(fields["moist"])[..., 1])
    i0 = max(1 - int(fields.get("ims", 1)), 0)
    j0 = max(1 - int(fields.get("jms", 1)), 0)
    if "ide" in fields and "jde" in fields:
        ni = int(fields["ide"]) - 1
        nj = int(fields["jde"]) - 1
    else:
        ni = max(int(qv_arr.shape[0]) - 1, 1)
        nj = max(int(qv_arr.shape[2]) - 1, 1)
    nk = int(qv_arr.shape[1])

    extras: dict[str, Any] = {}
    mass3_shape = (ni, nk, nj)
    mass2_shape = (ni, nj)
    moist = jnp.asarray(fields["moist"]) if "moist" in fields else None
    for output_name, sources in _THREED_EXTRA_FIELD_SOURCES.items():
        moist_index = _MOIST_OUTPUT_COMPONENTS.get(output_name)
        if moist_index is not None and moist is not None and moist.ndim == 4 and moist.shape[-1] > moist_index:
            value = moist[..., moist_index]
        else:
            value = _first_available_field(fields, sources)
        if value is not None:
            extras[output_name] = _xkj_to_wrf(_crop_to_shape(value, mass3_shape, i0, j0))
    for output_name, sources in _STAGGERED_EXTRA_FIELD_SOURCES.items():
        value = _first_available_field(fields, sources)
        if value is not None:
            extras[output_name] = _xkj_to_wrf(_crop_to_shape(value, (ni, nk + 1, nj), i0, j0))
    for output_name, sources in _TWOD_EXTRA_FIELD_SOURCES.items():
        value = _first_available_field(fields, sources)
        if value is not None:
            extras[output_name] = _xy_to_wrf(_crop2_to_shape(value, mass2_shape, i0, j0))
    _add_derived_tendency_extras(extras)
    return _filter_output_extras(extras)


def _snapshot_payload_from_fields(
    fields: dict[str, Any],
    config_flags: Any | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    u, v, t, qvapor = _snapshot_arrays_from_fields(fields, config_flags)
    return u, v, t, qvapor, _snapshot_extra_fields_from_fields(fields)


def _snapshot_from_fields(
    fields: dict[str, Any],
    start_timestamp: str,
    elapsed_seconds: int,
    config_flags: Any | None = None,
    *,
    check_finite: bool = True,
) -> PrognosticSnapshot:
    u, v, t, qvapor = _snapshot_arrays_from_fields(fields, config_flags)
    snapshot = PrognosticSnapshot(
        elapsed_seconds=int(elapsed_seconds),
        timestamp=_timestamp(start_timestamp, int(elapsed_seconds)),
        u=u,
        v=v,
        t=t,
        qvapor=qvapor,
        extra=_snapshot_extra_fields_from_fields(fields),
    )
    if check_finite:
        assert_snapshot_finite(snapshot)
    return snapshot


def assert_snapshot_finite(snapshot: PrognosticSnapshot) -> None:
    """Raise if any required prognostic output contains NaN or infinity."""

    for name, values in (
        ("U", snapshot.u),
        ("V", snapshot.v),
        ("T", snapshot.t),
        ("QVAPOR", snapshot.qvapor),
    ):
        if not bool(jnp.all(jnp.isfinite(values))):
            raise FloatingPointError(f"{name} contains NaN or infinity at {snapshot.timestamp}")
    for name, values in _extras_with_derived_tendencies(snapshot).items():
        if not bool(jnp.all(jnp.isfinite(values))):
            raise FloatingPointError(f"{name} contains NaN or infinity at {snapshot.timestamp}")


def assert_snapshots_finite(snapshots: tuple[PrognosticSnapshot, ...]) -> None:
    """Raise if any required prognostic output contains NaN or infinity."""

    for snapshot in snapshots:
        for name, values in (
            ("U", snapshot.u),
            ("V", snapshot.v),
            ("T", snapshot.t),
            ("QVAPOR", snapshot.qvapor),
        ):
            if not bool(np.all(np.isfinite(np.asarray(values)))):
                raise FloatingPointError(f"{name} contains NaN or infinity at {snapshot.timestamp}")
        for name, values in _extras_with_derived_tendencies(snapshot).items():
            if not bool(np.all(np.isfinite(np.asarray(values)))):
                raise FloatingPointError(f"{name} contains NaN or infinity at {snapshot.timestamp}")


def _snapshot_output_fields(snapshot: PrognosticSnapshot) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "U": snapshot.u,
        "V": snapshot.v,
        "T": snapshot.t,
        "QVAPOR": snapshot.qvapor,
    }
    fields.update(_extras_with_derived_tendencies(snapshot))
    return fields


def _nonfinite_detail(name: str, values: Any) -> str | None:
    array = np.asarray(values)
    finite = np.isfinite(array)
    if bool(np.all(finite)):
        return None
    bad = np.argwhere(~finite)
    first_index = tuple(int(item) for item in bad[0]) if bad.size else ()
    first_value = array[first_index] if first_index else np.asarray(array).reshape(-1)[0]
    return (
        f"{name}: count={int(bad.shape[0])}, shape={tuple(int(dim) for dim in array.shape)}, "
        f"first_index={first_index}, first_value={first_value}"
    )


def snapshot_nonfinite_summaries(snapshot: PrognosticSnapshot) -> tuple[str, ...]:
    """Return compact diagnostics for non-finite values in a snapshot."""

    details = [
        detail
        for name, values in sorted(_snapshot_output_fields(snapshot).items())
        if (detail := _nonfinite_detail(name, values)) is not None
    ]
    return tuple(details)


def materialize_snapshots(snapshots: tuple[PrognosticSnapshot, ...]) -> tuple[PrognosticSnapshot, ...]:
    """Transfer snapshot arrays to host once for comparison and NetCDF writing."""

    return tuple(
        PrognosticSnapshot(
            elapsed_seconds=snapshot.elapsed_seconds,
            timestamp=snapshot.timestamp,
            u=np.asarray(snapshot.u),
            v=np.asarray(snapshot.v),
            t=np.asarray(snapshot.t),
            qvapor=np.asarray(snapshot.qvapor),
            extra={name: np.asarray(value) for name, value in _extras_with_derived_tendencies(snapshot).items()},
        )
        for snapshot in snapshots
    )


def _materialize_snapshot_during_trajectory(default: bool = True) -> bool:
    value = os.environ.get("WRF_PY_MATERIALIZE_SNAPSHOTS_DURING_TRAJECTORY")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def nrmse_percent(actual: Any, expected: Any) -> float:
    """Return normalized root mean square error in percent."""

    actual_np = np.asarray(actual, dtype=np.float64)
    expected_np = np.asarray(expected, dtype=np.float64)
    return _nrmse_percent_arrays(actual_np, expected_np)


def rmse(actual: Any, expected: Any) -> float:
    """Return root mean square error in the arrays' native units."""

    actual_np = np.asarray(actual, dtype=np.float64)
    expected_np = np.asarray(expected, dtype=np.float64)
    return _rmse_arrays(actual_np, expected_np)


def _rmse_arrays(actual_np: np.ndarray, expected_np: np.ndarray) -> float:
    """Return root mean square error for NumPy arrays."""

    diff = actual_np - expected_np
    return float(np.sqrt(np.mean(diff * diff)))


def _nrmse_percent_arrays(actual_np: np.ndarray, expected_np: np.ndarray) -> float:
    """Return normalized root mean square error in percent for NumPy arrays."""

    diff = actual_np - expected_np
    rmse_value = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.sqrt(np.mean(expected_np * expected_np)))
    if denom == 0.0:
        return 0.0 if rmse_value == 0.0 else float("inf")
    return 100.0 * rmse_value / denom


def _safe_netcdf_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (str, bytes, int, float, np.integer, np.floating)):
            safe[key] = value
        elif isinstance(value, np.ndarray) and value.ndim == 0:
            safe[key] = value.item()
    return safe


def _wrf_global_attrs(reference: xr.Dataset | None) -> dict[str, Any]:
    attrs = {
        "title": "WRF-JAX prognostic trajectory",
        "history": "Generated by wrf_jax.trajectory.write_snapshots_netcdf",
        "Conventions": "WRF-IO plus CF auxiliary latitude/longitude coordinates",
    }
    if reference is not None:
        for name in WRF_PROJECTION_ATTRS:
            if name in reference.attrs:
                attrs[name] = reference.attrs[name]
    return _safe_netcdf_attrs(attrs)


def _copy_reference_var_attrs(reference: xr.Dataset | None, name: str) -> dict[str, Any]:
    if reference is not None and name in reference:
        return _safe_netcdf_attrs(dict(reference[name].attrs))
    return {}


def _coord_attrs(reference: xr.Dataset | None, name: str) -> dict[str, Any]:
    attrs = _copy_reference_var_attrs(reference, name)
    if name.startswith("XLAT"):
        attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    elif name.startswith("XLONG"):
        attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    return attrs


def _static_reference_coord(reference: xr.Dataset, name: str, time_count: int) -> np.ndarray:
    values = reference[name]
    if "Time" in values.dims:
        base = values.isel(Time=0).values
    else:
        base = values.values
    return np.repeat(np.asarray(base, dtype=np.float32)[None, ...], int(time_count), axis=0)


def _default_xtime_attrs(start_timestamp: str) -> dict[str, Any]:
    start = start_timestamp.replace("_", " ")
    return {
        "FieldType": np.int32(104),
        "MemoryOrder": "0  ",
        "description": f"minutes since {start}",
        "units": f"minutes since {start}",
        "stagger": "",
    }


def _set_grid_elapsed_seconds(grid: Any, elapsed_seconds: int) -> None:
    setattr(grid, "current_time", float(elapsed_seconds))
    clock = getattr(grid, "domain_clock", None)
    clockint = getattr(clock, "clockint", None)
    curr_time = getattr(clockint, "curr_time", None)
    if curr_time is not None and hasattr(curr_time, "base"):
        # Keep the outer clock coherent for code paths that inspect it. The
        # strict solve path currently reads ``grid.current_time`` for seconds.
        curr_time.base.s = int(elapsed_seconds)


def _reset_dtbc_on_boundary_time(grid: Any, elapsed_seconds: int, boundary_interval_seconds: int) -> bool:
    """Mirror WRF's outer lateral-boundary clock reset for strict trajectories."""

    interval = int(boundary_interval_seconds)
    if interval > 0 and int(elapsed_seconds) > 0 and int(elapsed_seconds) % interval == 0:
        setattr(grid, "dtbc", 0.0)
        return True
    return False


def _refresh_wrfbdy_fields_on_boundary_time(
    assembly: RuntimeAssembly,
    grid: Any,
    fields: dict[str, Any],
    elapsed_seconds: int,
) -> bool:
    """Refresh lateral-boundary fields when WRF would read the next wrfbdy record."""

    if not _reset_dtbc_on_boundary_time(grid, elapsed_seconds, assembly.case.config.boundary_interval_seconds):
        return False
    fields["dtbc"] = jnp.asarray(0.0, dtype=jnp.float32)
    interval = int(assembly.case.config.boundary_interval_seconds)
    record_index = int(elapsed_seconds) // interval if interval > 0 else 0
    if 0 < record_index < len(assembly.case.forcing.times):
        fields.update(load_wrfbdy_fields(assembly.case.boundary_path, time_index=record_index))
        _trajectory_progress(
            "boundary refresh "
            f"elapsed_seconds={int(elapsed_seconds)} "
            f"record_index={record_index} "
            f"record_time={assembly.case.forcing.times[record_index]}"
        )
    return True


def run_strict_trajectory(
    case_dir: str | Path,
    *,
    hours: int = 6,
    duration_seconds: int | None = None,
    snapshot_interval_seconds: int = 3600,
) -> TrajectoryResult:
    """Run the strict WRF-JAX model using the case namelist."""

    assembly = assemble_runtime(str(case_dir))
    dt = float(assembly.case.config.time_step)
    total_seconds = int(duration_seconds) if duration_seconds is not None else int(hours) * 3600
    if total_seconds <= 0:
        raise ValueError("trajectory duration must be positive")
    if total_seconds % int(dt) != 0:
        raise ValueError(f"requested duration {total_seconds}s is not divisible by time_step {dt}s")
    step_count = total_seconds // int(dt)
    snapshot_seconds = set(range(int(snapshot_interval_seconds), total_seconds + 1, int(snapshot_interval_seconds)))

    fields = dict(assembly.fields)
    grid = assembly.grid
    config_flags = assembly.config_flags
    final_step: SolveEmResult | None = None
    snapshots: list[PrognosticSnapshot] = []

    _set_grid_elapsed_seconds(grid, 0)
    for step in range(1, step_count + 1):
        final_step = solve_em(
            grid=grid,
            config_flags=config_flags,
            fields=fields,
            moist=fields["moist"],
            scalar=fields.get("scalar"),
            num_moist=_num_moist_from_fields(fields),
            num_scalar=_num_scalar_from_fields(fields),
            callbacks=assembly.callbacks,
            record_events=False,
        )
        fields = final_step.fields
        elapsed = int(round(step * dt))
        _set_grid_elapsed_seconds(grid, elapsed)
        _refresh_wrfbdy_fields_on_boundary_time(assembly, grid, fields, elapsed)
        if elapsed in snapshot_seconds:
            snapshots.append(_snapshot_from_fields(fields, assembly.case.config.start_time, elapsed, config_flags, check_finite=False))

    if final_step is None:
        raise ValueError("trajectory requested zero steps")
    return TrajectoryResult(assembly=assembly, final_step=final_step, snapshots=tuple(snapshots))


def run_direct_trajectory(
    case_dir: str | Path,
    *,
    hours: int = 6,
    duration_seconds: int | None = None,
    snapshot_interval_seconds: int = 3600,
    snapshot_callback: Callable[[PrognosticSnapshot], None] | None = None,
    keep_snapshots: bool = True,
) -> TrajectoryResult:
    """Run the prepared direct WRF-JAX model core using the case namelist."""

    _trajectory_progress(f"direct assemble_runtime start case_dir={case_dir}")
    assembly = assemble_runtime(str(case_dir))
    _trajectory_progress("direct assemble_runtime done")
    dt = float(assembly.case.config.time_step)
    total_seconds = int(duration_seconds) if duration_seconds is not None else int(hours) * 3600
    if total_seconds <= 0:
        raise ValueError("trajectory duration must be positive")
    if total_seconds % int(dt) != 0:
        raise ValueError(f"requested duration {total_seconds}s is not divisible by time_step {dt}s")
    step_count = total_seconds // int(dt)
    snapshot_seconds = set(range(int(snapshot_interval_seconds), total_seconds + 1, int(snapshot_interval_seconds)))
    progress_interval_seconds = _trajectory_progress_interval_seconds(snapshot_interval_seconds)

    fields = dict(assembly.fields)
    grid = assembly.grid
    config_flags = assembly.config_flags
    plan = build_direct_step_plan(
        grid=grid,
        config_flags=config_flags,
        callbacks=assembly.callbacks,
        num_moist=_num_moist_from_fields(fields),
        num_scalar=_num_scalar_from_fields(fields),
    )
    final_step: SolveEmResult | None = None
    snapshots: list[PrognosticSnapshot] = []
    materialize_snapshot = _materialize_snapshot_during_trajectory()

    _set_grid_elapsed_seconds(grid, 0)
    loop_start = perf_counter()
    progress_start = loop_start
    next_progress_elapsed = progress_interval_seconds
    _trajectory_progress(
        "direct integration start "
        f"steps={step_count} dt_seconds={int(dt)} total_seconds={total_seconds} "
        f"snapshot_interval_seconds={int(snapshot_interval_seconds)} "
        f"progress_interval_seconds={progress_interval_seconds}"
    )
    for step in range(1, step_count + 1):
        final_step = run_direct_step(
            plan=plan,
            grid=grid,
            config_flags=config_flags,
            fields=fields,
            moist=fields["moist"],
            scalar=fields.get("scalar"),
        )
        fields = final_step.fields
        elapsed = int(round(step * dt))
        _set_grid_elapsed_seconds(grid, elapsed)
        _refresh_wrfbdy_fields_on_boundary_time(assembly, grid, fields, elapsed)
        if elapsed in snapshot_seconds:
            snapshot = _snapshot_from_fields(
                fields,
                assembly.case.config.start_time,
                elapsed,
                config_flags,
                check_finite=False,
            )
            if materialize_snapshot:
                snapshot = materialize_snapshots((snapshot,))[0]
                nonfinite = snapshot_nonfinite_summaries(snapshot)
                if nonfinite:
                    _trajectory_progress(
                        "[trajectory] direct nonfinite "
                        f"elapsed_seconds={elapsed} model_time={snapshot.timestamp} "
                        + "; ".join(nonfinite[:4])
                    )
            if snapshot_callback is not None:
                snapshot_callback(snapshot)
            if keep_snapshots:
                snapshots.append(snapshot)
        if elapsed >= next_progress_elapsed or step == step_count:
            now = perf_counter()
            _trajectory_progress(
                "direct progress "
                f"step={step}/{step_count} elapsed_seconds={elapsed}/{total_seconds} "
                f"model_time={_timestamp(assembly.case.config.start_time, elapsed)} "
                f"since_last_progress_seconds={now - progress_start:.3f} "
                f"total_integration_wall_seconds={now - loop_start:.3f} "
                f"snapshots={len(snapshots)} materialized_snapshots={int(materialize_snapshot)}"
            )
            progress_start = now
            while next_progress_elapsed <= elapsed:
                next_progress_elapsed += progress_interval_seconds

    if final_step is None:
        raise ValueError("trajectory requested zero steps")
    _trajectory_progress(
        "direct integration done "
        f"steps={step_count} total_seconds={total_seconds} "
        f"total_integration_wall_seconds={perf_counter() - loop_start:.3f}"
    )
    return TrajectoryResult(assembly=assembly, final_step=final_step, snapshots=tuple(snapshots))


def _is_wrfbdy_field_name(name: str) -> bool:
    return str(name).endswith(WRFBDY_FIELD_SUFFIXES)


def _arrays_exactly_equal(left: Any, right: Any) -> bool:
    left_np = np.asarray(jax.device_get(jnp.asarray(left)))
    right_np = np.asarray(jax.device_get(jnp.asarray(right)))
    return left_np.shape == right_np.shape and left_np.dtype == right_np.dtype and bool(np.array_equal(left_np, right_np))


def _block_until_ready_tree(value: Any) -> Any:
    """Wait for all computed fields, including independent submissions.

    The first leaf may simply be an already-ready passthrough input.
    """
    jax.block_until_ready(value)
    return value


def _changed_dynamic_field_names(before: dict[str, Any], after: dict[str, Any]) -> tuple[str, ...]:
    """Return array-valued fields whose values must be carried between steady steps."""

    names: list[str] = []
    for name, after_value in after.items():
        if not _is_dynamic_array_value(after_value):
            continue
        before_value = before.get(name)
        if before_value is None or not _is_dynamic_array_value(before_value) or not _arrays_exactly_equal(before_value, after_value):
            names.append(name)
    if "dtbc" not in names:
        names.append("dtbc")
    return tuple(names)


_EXPLICIT_STEADY_CARRY_NAMES = (
    "u_2",
    "v_2",
    "w_2",
    "t_2",
    "ph_2",
    "mu_2",
    "moist",
    "moist_old",
    "scalar",
    "scalar_old",
    "qvapor",
    "qni",
    "qnice",
    "qnr",
    "qnrain",
    "dtbc",
    "u",
    "v",
    "w",
    "t",
    "ph",
    "mu",
    "muts",
    "ru",
    "rv",
    "rw",
    "mut",
    "muu",
    "muv",
    "cqu",
    "cqv",
    "cqw",
    "muus",
    "muvs",
    "mudf",
    "muave",
    "ru_m",
    "rv_m",
    "ww_m",
    "ww",
    "ww_1",
    "al",
    "p",
    "pm1",
    "tsk",
    "tskold",
    "ust",
    "ustar",
    "ustm",
    "ustold",
    "znt",
    "zntold",
    "qsfc",
    "qgh",
    "chs",
    "chs2",
    "cpm",
    "wspd",
    "ch",
    "akhs",
    "akms",
    "pblh",
    "kpbl",
    "hpbl_hold",
    "mixht",
    "ct",
    "exch_h",
    "tke_pbl",
    "el_pbl",
    "thz0",
    "qz0",
    "uz0",
    "vz0",
    "lowlyr",
    "pi_phy",
    "th_phy",
    "th_phy_m_t0",
    "p_phy",
    "u_phy",
    "v_phy",
    "p8w",
    "t_phy",
    "t8w",
    "z",
    "z_at_w",
    "dz8w",
    "p_hyd",
    "p_hyd_w",
    "rthraten",
    "rthratenlw",
    "rthratenlwc",
    "rthratensw",
    "rthratenswc",
    "rthcuten",
    "rqvcuten",
    "rqccuten",
    "rqrcuten",
    "rqicuten",
    "rqscuten",
    "rucuten",
    "rvcuten",
    "rublten",
    "rvblten",
    "rthblten",
    "rqvblten",
    "rqcblten",
    "rqiblten",
    "rthmpten",
    "rqvmpten",
    "rqcmpten",
    "rqrmpten",
    "rqimpten",
    "rqsmpten",
    "rqgmpten",
    "rushten",
    "rvshten",
    "rthshten",
    "rqvshten",
    "rqcshten",
    "rqrshten",
    "rqishten",
    "rqsshten",
    "rqgshten",
    "rthften",
    "rqvften",
    "h_diabatic",
    "qv_diabatic",
    "qc_diabatic",
    "dtaux3d",
    "dtauy3d",
    "dusfcg",
    "dvsfcg",
    "tslb",
    "smois",
    "sh2o",
    "canwat",
    "snow",
    "snowh",
    "snowc",
    "snotime",
    "albedo",
    "albbck",
    "emiss",
    "grdflx",
    "hfx",
    "qfx",
    "lh",
    "ru_tendf",
    "rv_tendf",
    "rw_tendf",
    "ph_tendf",
    "t_tendf",
    "tke_tend",
    "mu_tendf",
    "moist_tend",
    "chem_tend",
    "scalar_tend",
    "tracer_tend",
)
_DIAGNOSTIC_TENDENCY_CARRY_NAMES = frozenset(
    (
        "pi_phy",
        "rthraten",
        "rthratenlw",
        "rthratenlwc",
        "rthratensw",
        "rthratenswc",
        "rthcuten",
        "rqvcuten",
        "rqccuten",
        "rqrcuten",
        "rqicuten",
        "rqscuten",
        "rucuten",
        "rvcuten",
        "rublten",
        "rvblten",
        "rthblten",
        "rqvblten",
        "rqcblten",
        "rqiblten",
        "rqniblten",
        "rthmpten",
        "rqvmpten",
        "rqcmpten",
        "rqrmpten",
        "rqimpten",
        "rqsmpten",
        "rqgmpten",
        "rushten",
        "rvshten",
        "rthshten",
        "rqvshten",
        "rqcshten",
        "rqrshten",
        "rqishten",
        "rqsshten",
        "rqgshten",
        "rthften",
        "rqvften",
        "h_diabatic",
        "qv_diabatic",
        "qc_diabatic",
    )
)

_DERIVED_DRY_ALIAS_NAMES = frozenset(("u", "v", "w", "t", "ph", "mu"))
_DERIVED_STEADY_FIELD_NAMES = frozenset(
    (
        "qvapor",
        "qcloud",
        "qrain",
        "qice",
        "qsnow",
        "qgraup",
        "qni",
        "qnice",
        "qnr",
        "qnrain",
        "tskold",
        "ustold",
        "zntold",
        "ustar",
    )
)
_REFRESHED_STEADY_STATIC_FIELD_NAMES = _DERIVED_DRY_ALIAS_NAMES | _DERIVED_STEADY_FIELD_NAMES
_RK_PREP_SCRATCH_NAMES = frozenset(("mut", "muu", "muv", "ww", "php", "alt"))
_SMALL_STEP_ACCUM_SCRATCH_NAMES = frozenset(("muave", "ru_m", "rv_m", "ww_m", "ww_1"))
_PHYSICS_TENDENCY_SCRATCH_NAMES = frozenset(
    (
        "rublten",
        "rvblten",
        "rthblten",
        "rqvblten",
        "rqcblten",
        "rqiblten",
        "dtaux3d",
        "dtauy3d",
        "dusfcg",
        "dvsfcg",
    )
)
_RECOMPUTED_STEADY_SCRATCH_NAMES = frozenset(("mudf", "rho"))
_PRESTEP_CARRY_NAMES = frozenset(
    (
        "dtaux3d",
        "dtauy3d",
        "dusfcg",
        "dvsfcg",
        "muave",
        "mudf",
        "rho",
        "ru_m",
        "rv_m",
        "ww_1",
        "ww_m",
    )
)
_REBUILT_STEP_WORKSPACE_NAMES = frozenset(
    (
        "alb",
        "cldfra",
        "cldfra_old",
        "cqu",
        "cqv",
        "cqw",
        "dz8w",
        "f_ice_phy",
        "f_rain_phy",
        "moist_old",
        "p8w",
        "p_hyd",
        "p_hyd_w",
        "p_phy",
        "pi_phy",
        "qc_diabatic",
        "qv_diabatic",
        "rqcmpten",
        "rqgmpten",
        "rqimpten",
        "rqrmpten",
        "rqsmpten",
        "rqvmpten",
        "rthmpten",
        "rthratenlw",
        "rthratenlwc",
        "rthratensw",
        "rthratenswc",
        "ru",
        "rv",
        "rw",
        "t8w",
        "t_ave",
        "t_phy",
        "th_phy",
        "th_phy_m_t0",
        "thm",
        "u_phy",
        "uam",
        "v_phy",
        "vam",
        "wwam",
        "z",
        "z_at_w",
    )
)
_RESEEDED_STEADY_TENDENCY_NAMES = frozenset(
    (
        "ru_tendf",
        "rv_tendf",
        "rw_tendf",
        "ph_tendf",
        "t_tendf",
        "tke_tend",
        "mu_tendf",
        "moist_tend",
        "scalar_tend",
        "chem_tend",
        "tracer_tend",
    )
)
_TRANSIENT_STEADY_SCRATCH_NAMES = (
    _RK_PREP_SCRATCH_NAMES
    | _SMALL_STEP_ACCUM_SCRATCH_NAMES
    | _PHYSICS_TENDENCY_SCRATCH_NAMES
    | _RECOMPUTED_STEADY_SCRATCH_NAMES
)


def _drop_derived_dry_aliases_from_carry(default: bool = True) -> bool:
    value = os.environ.get("WRF_PY_DROP_DERIVED_DRY_ALIASES")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _drop_rk_prep_scratch_from_carry(default: bool = True) -> bool:
    value = os.environ.get("WRF_PY_DROP_RK_PREP_SCRATCH")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _drop_transient_scratch_from_carry(default: bool = True) -> bool:
    value = os.environ.get("WRF_PY_DROP_TRANSIENT_SCRATCH")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _step_local_workspace_enabled(default: bool = False) -> bool:
    """Return whether one compiled step owns its transient stage workspace."""

    value = os.environ.get("WRF_PY_STEP_LOCAL_WORKSPACE")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _carry_diagnostic_tendencies_enabled(default: bool = True) -> bool:
    value = os.environ.get("WRF_PY_CARRY_DIAGNOSTIC_TENDENCIES")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _extra_drop_carry_names() -> frozenset[str]:
    raw = os.environ.get("WRF_PY_EXTRA_DROP_CARRY", "")
    names = [item.strip() for item in raw.replace(";", ",").split(",")]
    return frozenset(name for name in names if name)


def _qvapor_only_carry_enabled(default: bool = False) -> bool:
    value = os.environ.get("WRF_PY_QVAPOR_ONLY_CARRY")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _include_configured_dynamic_carry_enabled(default: bool = True) -> bool:
    value = os.environ.get("WRF_PY_INCLUDE_CONFIGURED_DYNAMIC_CARRY")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _num_moist_from_fields(fields: Mapping[str, Any]) -> int:
    moist = fields.get("moist")
    if moist is None:
        return 2
    return int(jnp.asarray(moist).shape[-1])


def _num_scalar_from_fields(fields: Mapping[str, Any]) -> int:
    scalar = fields.get("scalar")
    if scalar is None:
        return 0
    return int(jnp.asarray(scalar).shape[-1])


def _seed_startup_steady_fields_enabled(default: bool = False) -> bool:
    value = os.environ.get("WRF_PY_SEED_STARTUP_STEADY_FIELDS")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


_STARTUP_GENERATED_MASS3_NAMES = (
    "advect_tend",
    "cldfra_cup",
    "h_tendency",
    "rqvndgdten",
    "rthndgdten",
    "rundgdten",
    "rvndgdten",
    "z_tendency",
)
_STARTUP_GENERATED_W3_NAMES = (
    "bn2",
    "defor11",
    "defor12",
    "defor13",
    "defor22",
    "defor23",
    "defor33",
    "div",
    "rdz",
    "rdzw",
    "rphndgdten",
    "u_phytmp",
    "v_phytmp",
    "xkhh",
    "xkhv",
    "xkmh",
    "xkmv",
    "xkmv_meso",
    "zx",
    "zy",
)
_STARTUP_GENERATED_2D_NAMES = (
    "chklowq",
    "elflx",
    "fbur_2d",
    "fgsn_2d",
    "flx4_2d",
    "fvb_2d",
    "gamu",
    "gamv",
    "pshltr",
    "q02",
    "q10",
    "qshltr",
    "rc2",
    "rmundgdten",
    "swddni",
    "t02",
    "th02",
    "th10",
    "tshltr",
    "u10e",
    "v10e",
)
_STARTUP_ONLY_DYNAMIC_NAMES = (
    *_STARTUP_GENERATED_MASS3_NAMES,
    *_STARTUP_GENERATED_W3_NAMES,
    *_STARTUP_GENERATED_2D_NAMES,
    "nba_mij",
    "chem",
    "tracer",
    "gsw",
)


def _seed_missing_startup_steady_fields(fields: dict[str, Any]) -> None:
    """Seed steady carry fields normally created by the first physics step."""

    shape2_source = fields.get("tsk", fields.get("mu_2"))
    if shape2_source is None:
        return
    zeros2 = jnp.zeros_like(jnp.asarray(shape2_source))
    if "w_2" in fields:
        zeros3w = jnp.zeros_like(jnp.asarray(fields["w_2"]))
    elif "t_2" in fields:
        zeros3w = jnp.zeros_like(jnp.asarray(fields["t_2"]))
    else:
        zeros3w = None

    if "t_2" in fields:
        zeros3 = jnp.zeros_like(jnp.asarray(fields["t_2"]))
        for name in _STARTUP_GENERATED_MASS3_NAMES:
            fields.setdefault(name, zeros3)
    if zeros3w is not None:
        for name in _STARTUP_GENERATED_W3_NAMES:
            fields.setdefault(name, zeros3w)
        fields.setdefault("nba_mij", jnp.zeros(zeros3w.shape + (6,), dtype=zeros3w.dtype))
    for name in _STARTUP_GENERATED_2D_NAMES:
        fields.setdefault(name, zeros2)

    for name in ("grdflx", "noahres", "potevp", "snopcx", "snotime", "xlai2", "hpbl_hold", "mixht"):
        fields.setdefault(name, zeros2)
    if "t_2" in fields:
        zeros3 = jnp.zeros_like(jnp.asarray(fields["t_2"]))
        for name in (
            "rthmpten",
            "rqvmpten",
            "rqcmpten",
            "rqrmpten",
            "rqimpten",
            "rqsmpten",
            "rqgmpten",
            "rushten",
            "rvshten",
            "rthshten",
            "rqvshten",
            "rqcshten",
            "rqrshten",
            "rqishten",
            "rqsshten",
            "rqgshten",
            "rqvften",
            "dtaux3d",
            "dtauy3d",
        ):
            fields.setdefault(name, zeros3)
    for name in ("dusfcg", "dvsfcg"):
        fields.setdefault(name, zeros2)
    fields.setdefault("kpbl", jnp.zeros_like(zeros2, dtype=jnp.int32))
    fields.setdefault("ch", fields.get("chs", zeros2))
    fields.setdefault("ustm", fields.get("ust", zeros2))
    if "u_2" in fields and "v_2" in fields:
        fields.setdefault(
            "wspd",
            jnp.maximum(
                sqrt_for_wrf(jnp.asarray(fields["u_2"])[:, 0, :] ** 2 + jnp.asarray(fields["v_2"])[:, 0, :] ** 2),
                jnp.asarray(0.001, dtype=jnp.asarray(fields["u_2"]).dtype),
            ),
        )
    else:
        fields.setdefault("wspd", zeros2)
    if zeros3w is not None:
        fields.setdefault("el_pbl", zeros3w)


def _compiled_startup_dynamic_names(fields: dict[str, Any]) -> tuple[str, ...]:
    """Return first-step state, including arrays created only during startup."""

    ordinary = _explicit_steady_dynamic_names(fields)
    startup_only = tuple(name for name in _STARTUP_ONLY_DYNAMIC_NAMES if name in fields)
    return tuple(dict.fromkeys((*ordinary, *startup_only)))


def _refresh_current_dry_aliases(fields: dict[str, Any]) -> None:
    for alias, current in (
        ("u", "u_2"),
        ("v", "v_2"),
        ("w", "w_2"),
        ("t", "t_2"),
        ("ph", "ph_2"),
        ("mu", "mu_2"),
    ):
        if current in fields:
            fields[alias] = fields[current]


@partial(jax.jit, static_argnames=("indices",))
def _last_axis_components(array: Any, indices: tuple[int, ...]) -> tuple[Any, ...]:
    """Submit independent species views together, including outside a JIT."""

    return tuple(array[..., index] for index in indices)


def _refresh_derived_steady_fields(fields: dict[str, Any]) -> None:
    if _qvapor_only_carry_enabled() and "moist" in fields and "qvapor" in fields:
        fields["moist"] = _replace_last_axis_component(fields["moist"], 1, fields["qvapor"])
    elif "moist" in fields:
        moist = jnp.asarray(fields["moist"])
        names = ("qvapor", "qcloud", "qrain", "qice", "qsnow", "qgraup")
        indices = (1, *range(2, min(7, moist.shape[-1])))
        values = _last_axis_components(moist, indices)
        fields.update(zip(names[:len(indices)], values, strict=True))
    if "scalar" in fields:
        scalar = jnp.asarray(fields["scalar"])
        if scalar.ndim == 4:
            indices = tuple(range(1, min(3, scalar.shape[-1])))
            values = _last_axis_components(scalar, indices)
            names = (("qni", "qnice"), ("qnr", "qnrain"))
            for (canonical, alias), value in zip(names[:len(indices)], values, strict=True):
                fields[canonical] = fields[alias] = value
    for alias, source in (
        ("tskold", "tsk"),
        ("ustold", "ust"),
        ("zntold", "znt"),
        ("ustar", "ust"),
    ):
        if source in fields:
            fields[alias] = fields[source]


def _zero_width_species_like(fields: Mapping[str, Any]) -> Any:
    source = fields.get("moist", fields.get("scalar"))
    if source is None:
        raise KeyError("cannot seed inactive species scratch without moist or scalar")
    arr = jnp.asarray(source)
    return jnp.zeros(arr.shape[:-1] + (0,), dtype=arr.dtype)


def _seed_steady_scratch_fields(fields: dict[str, Any]) -> None:
    """Create per-step scratch arrays that should not live in the compiled closure."""

    shape_sources = (
        ("ru_tendf", "u_2"),
        ("rv_tendf", "v_2"),
        ("rw_tendf", "w_2"),
        ("ph_tendf", "ph_2"),
        ("t_tendf", "t_2"),
        ("tke_tend", "t_2"),
        ("mu_tendf", "mu_2"),
        ("moist_tend", "moist"),
        ("scalar_tend", "scalar"),
    )
    for name, source_name in shape_sources:
        if name not in fields and source_name in fields:
            fields[name] = jnp.zeros_like(jnp.asarray(fields[source_name]))
    for name, source_name in (("chem_tend", "chem"), ("tracer_tend", "tracer")):
        if name not in fields:
            fields[name] = jnp.zeros_like(jnp.asarray(fields[source_name])) if source_name in fields else _zero_width_species_like(fields)
    for name in ("chem", "tracer"):
        if name not in fields:
            fields[name] = _zero_width_species_like(fields)
    if "moist_old" not in fields and "moist" in fields:
        fields["moist_old"] = jnp.asarray(fields["moist"])


def _seed_interstage_rk_scratch_fields(fields: dict[str, Any]) -> None:
    """Seed RK scratch arrays that cross staged subgraph boundaries."""

    shape_sources = (
        ("ru", "u_2"),
        ("rv", "v_2"),
        ("rw", "w_2"),
        ("ww", "w_2"),
        ("php", "ph_2"),
        ("alt", "t_2"),
        ("muu", "mu_2"),
        ("muv", "mu_2"),
        ("mut", "mu_2"),
        ("cqu", "u_2"),
        ("cqv", "v_2"),
        ("cqw", "w_2"),
        ("th_phy", "t_2"),
        ("th_phy_m_t0", "t_2"),
        ("p_phy", "t_2"),
        ("pi_phy", "t_2"),
        ("u_phy", "u_2"),
        ("v_phy", "v_2"),
        ("p8w", "w_2"),
        ("t_phy", "t_2"),
        ("t8w", "w_2"),
        ("z", "t_2"),
        ("z_at_w", "w_2"),
        ("dz8w", "t_2"),
        ("p_hyd", "t_2"),
        ("p_hyd_w", "w_2"),
    )
    for name, source_name in shape_sources:
        if name not in fields and source_name in fields:
            fields[name] = jnp.zeros_like(jnp.asarray(fields[source_name]))


def _refresh_completed_block_fields(
    fields: dict[str, Any],
    *,
    elapsed_seconds: int,
    total_seconds: int,
    snapshot_seconds: set[int],
    boundary_interval_seconds: int,
) -> None:
    """Expand device state only when an outer consumer needs the full fields."""

    at_boundary = (
        boundary_interval_seconds > 0
        and elapsed_seconds > 0
        and elapsed_seconds % boundary_interval_seconds == 0
    )
    if elapsed_seconds == total_seconds or elapsed_seconds in snapshot_seconds or at_boundary:
        _refresh_current_dry_aliases(fields)
        _refresh_derived_steady_fields(fields)
        _seed_steady_scratch_fields(fields)


def _prune_refreshed_steady_static_fields(static_fields: dict[str, Any]) -> dict[str, Any]:
    """Drop array aliases that the compiled steady step rebuilds from carry fields."""

    num_chem = int(static_fields.get("num_chem", 0) or 0)
    num_tracer = int(static_fields.get("num_tracer", 0) or 0)

    def should_drop(name: str, value: Any) -> bool:
        if not _is_dynamic_array_value(value):
            return False
        if (
            name in _REFRESHED_STEADY_STATIC_FIELD_NAMES
            or name in _RESEEDED_STEADY_TENDENCY_NAMES
            or (_compiled_direct_stage_families_enabled() and name in _TRANSIENT_STEADY_SCRATCH_NAMES)
        ):
            return True
        if name in ("chem", "chem_tend") and num_chem == 0:
            return True
        if name in ("tracer", "tracer_tend") and num_tracer == 0:
            return True
        return False

    return {
        name: value
        for name, value in static_fields.items()
        if not should_drop(name, value)
    }


def _explicit_steady_dynamic_names(fields: dict[str, Any]) -> tuple[str, ...]:
    """Return the curated cross-step carry for the configured em_real case."""

    names: list[str] = []
    drop_aliases = _drop_derived_dry_aliases_from_carry()
    drop_rk_prep_scratch = _drop_rk_prep_scratch_from_carry()
    drop_transient_scratch = _drop_transient_scratch_from_carry()
    step_local_workspace = (
        _compiled_direct_stage_families_enabled()
        and _step_local_workspace_enabled()
    )
    if _compiled_direct_stage_families_enabled() and not step_local_workspace:
        # Stage-family mode splits one WRF time step across several compiled
        # calls.  These fields are transient across model steps, but they are
        # live values between the split RK/physics subgraphs inside a step.
        drop_rk_prep_scratch = False
        drop_transient_scratch = False
    carry_diagnostics = (
        _carry_diagnostic_tendencies_enabled()
        and not step_local_workspace
    )
    qvapor_only = _qvapor_only_carry_enabled()
    extra_drop = _extra_drop_carry_names()
    if step_local_workspace:
        persistent_candidates = dict.fromkeys(
            (
                *CONFIGURED_RECURRENT_ARRAY_NAMES,
                *CONFIGURED_DYNAMIC_ARRAY_NAMES,
                *_EXPLICIT_STEADY_CARRY_NAMES,
            )
        )
        for name in persistent_candidates:
            if qvapor_only and name == "moist":
                continue
            if name in extra_drop or name in _REBUILT_STEP_WORKSPACE_NAMES:
                continue
            if drop_aliases and (
                name in _DERIVED_DRY_ALIAS_NAMES
                or (
                    name in _DERIVED_STEADY_FIELD_NAMES
                    and not (qvapor_only and name == "qvapor")
                )
            ):
                continue
            if name in _RESEEDED_STEADY_TENDENCY_NAMES:
                continue
            if (
                name in _TRANSIENT_STEADY_SCRATCH_NAMES
                and name not in _PRESTEP_CARRY_NAMES
            ):
                continue
            if name in fields and _is_dynamic_array_value(fields[name]):
                names.append(name)
        if "dtbc" in fields:
            names.append("dtbc")
        return tuple(dict.fromkeys(names))
    if not _compiled_direct_stage_families_enabled():
        for name in CONFIGURED_RECURRENT_ARRAY_NAMES:
            if name in extra_drop:
                continue
            if drop_aliases and (name in _DERIVED_DRY_ALIAS_NAMES or name in _DERIVED_STEADY_FIELD_NAMES):
                continue
            if name in fields and _is_dynamic_array_value(fields[name]):
                names.append(name)
        if "dtbc" in fields:
            names.append("dtbc")
        return tuple(dict.fromkeys(names))

    for name in _EXPLICIT_STEADY_CARRY_NAMES:
        is_diagnostic_tendency = carry_diagnostics and name in _DIAGNOSTIC_TENDENCY_CARRY_NAMES
        if qvapor_only and name == "moist":
            continue
        if name in extra_drop:
            continue
        if drop_aliases and (name in _DERIVED_DRY_ALIAS_NAMES or (name in _DERIVED_STEADY_FIELD_NAMES and not (qvapor_only and name == "qvapor"))):
            continue
        if drop_transient_scratch and name in _TRANSIENT_STEADY_SCRATCH_NAMES and not is_diagnostic_tendency:
            continue
        if drop_rk_prep_scratch and name in _RK_PREP_SCRATCH_NAMES and not is_diagnostic_tendency:
            continue
        if name in fields and (name == "dtbc" or _is_dynamic_array_value(fields[name])):
            names.append(name)
    if _include_configured_dynamic_carry_enabled():
        for name in CONFIGURED_DYNAMIC_ARRAY_NAMES:
            if name in names:
                continue
            if qvapor_only and name == "moist":
                continue
            if name in extra_drop:
                continue
            if drop_aliases and (name in _DERIVED_DRY_ALIAS_NAMES or name in _DERIVED_STEADY_FIELD_NAMES):
                continue
            if name in fields and _is_dynamic_array_value(fields[name]):
                names.append(name)
    for name in ("php", "alt"):
        if drop_rk_prep_scratch:
            continue
        if name in extra_drop:
            continue
        if name in fields and _is_dynamic_array_value(fields[name]):
            names.append(name)
    for name, value in fields.items():
        if _is_wrfbdy_field_name(name) and _is_dynamic_array_value(value):
            names.append(name)
    if (
        "rho" not in extra_drop
        and not (drop_transient_scratch and "rho" in _TRANSIENT_STEADY_SCRATCH_NAMES)
        and "rho" in fields
        and _is_dynamic_array_value(fields["rho"])
    ):
        # ``rho`` is read by the first physical-boundary substage before the
        # next stage recomputes it. Its halo layout may differ from ``w_2``;
        # shape matching here incorrectly removed a required pre-step value.
        names.append("rho")
    return tuple(dict.fromkeys(names))


def _group_radiation_event_units(
    units: tuple[tuple[str, Any, Any, bool], ...],
    merge_updates: Any,
    *,
    static_physics_clock: bool,
    host_microphysics_diagnostics: bool = False,
    compile_water_preparation: bool = False,
    distributed_stages: frozenset[str] = frozenset(),
    final_output_names: frozenset[str] | None = None,
) -> tuple[tuple[str, Any, Any, bool], ...] | None:
    """Batch event computation while retaining required host boundaries.

    Keep calendar dispatch and the remaining REAL-rounding boundaries as their
    original calls. CUDA pointwise kernels preserve water preparation rounding
    inside the group; CPU retains its independently submitted calls. Thompson's
    shared kernel retains its own compiler boundary; its coupling is compiled
    with the surrounding stages. Optional host-only radar diagnostics remain
    outside the group. Custom partitions retain their existing execution units.
    """
    host_labels = {
        "rk1_radiation", "post_advance_ppt", "post_moist_prep",
    }
    host_labels.update(distributed_stages)
    required_labels = set(host_labels)
    if compile_water_preparation:
        host_labels.difference_update(("post_advance_ppt", "post_moist_prep"))
    if host_microphysics_diagnostics:
        host_labels.add("post_microphysics")
    if not static_physics_clock:
        host_labels.update(("rk1_surface_pbl", "rk1_cumulus"))
    if not (required_labels | host_labels).issubset({unit[0] for unit in units}):
        return None

    def compiled_group(selected):
        def apply_event_group(state, params):
            updated_names = set()
            for _label, _compiled, original, full_state in selected:
                result = semantic_call(original, state, params)
                updated_names.update(result.array_names)
                state = result if full_state else merge_updates(state, result)
            if final_output_names is not None and selected[-1] is units[-1]:
                updated_names.intersection_update(final_output_names)
            return state.select_arrays(updated_names)

        program = jax.jit(apply_event_group, compiler_options=SEMANTIC_CALL_COMPILER_OPTIONS)
        label = f"event[{selected[0][0]}..{selected[-1][0]}]"
        return label, program, program, False

    grouped = []
    pending = []
    for unit in units:
        if unit[0] in host_labels:
            if pending:
                grouped.append(compiled_group(tuple(pending)))
                pending.clear()
            grouped.append(unit)
        else:
            pending.append(unit)
    if pending:
        grouped.append(compiled_group(tuple(pending)))
    return tuple(grouped)


def _execute_stage_function(
    state: ModelState,
    static_arrays: dict[str, Any],
    *,
    host_fn: Any,
    reference_fn: Any,
    scan_fn: Any,
    semantic_calls: bool,
    compiled_scan_calls: bool,
) -> ModelState:
    """Use prepared stage programs on the host, preserving scan boundaries."""
    inside_trace = any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(state)
    )
    if not inside_trace:
        return host_fn(state, static_arrays)
    if compiled_scan_calls:
        return semantic_call(scan_fn, state, static_arrays)
    if semantic_calls:
        return semantic_call(reference_fn, state, static_arrays)
    return reference_fn(state, static_arrays)


def _compiled_steady_direct_functions(
    *,
    static_scalar_fields: dict[str, Any],
    static_array_fields: dict[str, Any],
    dynamic_state: ModelState,
    dynamic_names: tuple[str, ...],
    grid: Any,
    config_flags: Any,
    callbacks: Any,
    snapshot_block_steps: int,
    eager_stage_names: frozenset[str] | None = None,
    stage_scan_semantics: bool | None = None,
    scan_compiled_calls: bool | None = None,
    full_state_barriers: bool | None = None,
) -> tuple[Any, Any, Any, Any, ModelState, dict[str, Any]]:
    """Build compiled direct steppers for the post-startup steady trajectory."""

    scan_unroll = _compiled_direct_scan_unroll()
    compiled_grid = copy.copy(grid)
    try:
        current_itimestep = int(np.asarray(jax.device_get(static_scalar_fields.get("itimestep", getattr(grid, "itimestep", 0)))).reshape(-1)[0])
    except (TypeError, ValueError):
        current_itimestep = int(getattr(grid, "itimestep", 0))
    setattr(compiled_grid, "itimestep", max(1, current_itimestep + 1))
    plan = build_direct_step_plan(
        grid=compiled_grid,
        config_flags=config_flags,
        callbacks=callbacks,
        num_moist=_num_moist_from_fields(dynamic_state.as_dict()),
        num_scalar=_num_scalar_from_fields(dynamic_state.as_dict()),
    )
    dt_seconds = float(getattr(grid, "dt", 0.0))
    boundary_interval_seconds = int(getattr(config_flags, "boundary_interval_seconds", 0))
    step_local_workspace = (
        _compiled_direct_stage_families_enabled()
        and _step_local_workspace_enabled()
    )
    persistent_array_names = dynamic_state.array_names
    spatial_devices = configured_devices()
    distributed_stage_names = frozenset({"rk_physics_dynamics"}) if len(spatial_devices) > 1 else frozenset()
    tiled_physics = None
    if distributed_stage_names:
        def refresh_fields(fields):
            _refresh_current_dry_aliases(fields)
            _refresh_derived_steady_fields(fields)

        stencil = 1 + (max(int(getattr(config_flags, "h_mom_adv_order", 5)),
                           int(getattr(config_flags, "h_sca_adv_order", 5))) + 1) // 2
        dynamics_halo = 4 * stencil + 2 * plan.num_sound_steps + 4

        def dynamics_segment(rk_step):
            names = tuple(name for name in RK_COMPILED_SUBSTAGES
                          if rk_step != 1 or name not in {"rk_step_prep", "rk_phys_bc_dry_1"})
            def execute(fields):
                return run_rk_field_stages(
                    fields, plan=plan, grid=compiled_grid, config_flags=config_flags,
                    callbacks=callbacks, rk_step=rk_step, stage_names=names,
                    refresh_fields=refresh_fields, overlap_margin=stencil,
                )
            return execute

        def physics_segment(names):
            def execute(fields):
                local = dict(fields)
                _refresh_current_dry_aliases(local)
                _refresh_derived_steady_fields(local)
                before = dict(local)
                if "surface_pbl_driver" in names:
                    _seed_first_rk_physics_surface_context(local)
                for name in names:
                    callback = callbacks.get(name)
                    if callback is not None:
                        updates = callback(local)
                        if updates:
                            local.update(updates)
                return {name: value for name, value in local.items()
                        if _is_dynamic_array_value(value) and value is not before.get(name)}
            return execute

        def post_segment(names):
            def execute(fields):
                return run_post_rk_field_stages(
                    fields, plan=plan, grid=compiled_grid, config_flags=config_flags,
                    callbacks=callbacks, stage_names=names, refresh_fields=refresh_fields,
                )
            return execute

        def part2_segment(start, stop):
            def execute(fields):
                fields = dict(fields)
                if start == "calculate_phy_tend":
                    _refresh_current_dry_aliases(fields)
                    _refresh_derived_steady_fields(fields)
                before = dict(fields)
                fields["first_rk_step_part2_start_at"] = start
                fields["first_rk_step_part2_stop_after"] = stop
                updates = callbacks["first_rk_step_part2"](fields)
                if updates:
                    fields.update(updates)
                return {name: value for name, value in fields.items()
                        if _is_dynamic_array_value(value) and value is not before.get(name)}
            return execute

        tiled_physics = TiledFieldSequence(
            (physics_segment(("surface_pbl_driver",)),
             physics_segment(("cumulus_driver",)),
             physics_segment(("shallowcu_driver", "fddagd_driver")))
            + tuple(part2_segment(start, stop) for start, stop in (
                ("calculate_phy_tend", "diffusion_bc"),
                ("cal_deform_and_div", "cal_deform_and_div"),
                ("calculate_km_kh", "calculate_km_kh"),
                ("phy_bc", "horizontal_diffusion_2"),
            )) + tuple(dynamics_segment(rk_step) for rk_step in range(1, plan.rk_order + 1))
            + tuple(post_segment(names) for names in POST_RK_TILE_GROUPS),
            global_x=int(compiled_grid.ed31), global_y=int(compiled_grid.ed33),
            row_shards=len(spatial_devices),
            halo_width=(1, 0, 0, 1, 1, 1, 1) + (dynamics_halo,) * plan.rk_order
            + tuple(0 if names == ("post_microphysics",) else 1 for names in POST_RK_TILE_GROUPS),
            tile_metadata=surface_row_tile_metadata,
            output_names=persistent_array_names,
        )

    def dynamic_fields_from_state(state: ModelState) -> dict[str, Any]:
        values = state.as_dict()
        if "dtbc" in dynamic_names:
            values["dtbc"] = state.time.dtbc
        if step_local_workspace:
            return values
        return {name: values[name] for name in dynamic_names}

    def state_from_dynamic_fields(previous: ModelState, dynamic: dict[str, Any], *, advance_time: bool = True) -> ModelState:
        next_time = previous.time.advance(dt_seconds, boundary_interval_seconds) if advance_time else previous.time
        if "dtbc" in dynamic:
            next_time = replace(next_time, dtbc=dynamic["dtbc"])
        return replace(
            previous,
            arrays=tuple(dynamic[name] for name in previous.array_names),
            time=next_time,
        )

    def fields_from_state(
        state: ModelState,
        static_arrays: dict[str, Any],
        *,
        step_clock_advanced: bool,
    ) -> dict[str, Any]:
        dynamic = dynamic_fields_from_state(state)
        fields = dict(static_scalar_fields)
        fields.update(static_arrays)
        fields.update(dynamic)
        fields["itimestep"] = state.time.itimestep
        fields["elapsed_seconds"] = state.time.elapsed_seconds
        active_step_seconds = state.time.elapsed_seconds
        if step_clock_advanced:
            active_step_seconds = active_step_seconds - jnp.asarray(
                dt_seconds,
                dtype=jnp.asarray(active_step_seconds).dtype,
            )
        fields["curr_secs"] = active_step_seconds
        fields.update(curr_secs2=active_step_seconds, _preserve_water_statement_rounding=use_stage_families and not bool(stage_scan_semantics))
        _refresh_current_dry_aliases(fields)
        _refresh_derived_steady_fields(fields)
        _seed_steady_scratch_fields(fields)
        _seed_interstage_rk_scratch_fields(fields)
        return fields

    def dynamic_fields_from_fields(previous: ModelState, fields: dict[str, Any]) -> dict[str, Any]:
        previous_dynamic = dynamic_fields_from_state(previous)
        return {name: fields.get(name, previous_dynamic[name]) for name in dynamic_names}

    def stage_updates_from_fields(
        previous: ModelState,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        *,
        advance_time: bool,
    ) -> ModelState:
        """Return only array leaves replaced by one compiled stage family."""

        previous_dynamic = dynamic_fields_from_state(previous)
        candidate_names = set(previous.array_names)
        if step_local_workspace:
            candidate_names.update(
                name
                for name, value in after.items()
                if _is_dynamic_array_value(value)
                and value is not before.get(name)
            )
        update_names = tuple(
            name
            for name in sorted(candidate_names)
            if after.get(name, previous_dynamic.get(name))
            is not before.get(name, previous_dynamic.get(name))
        )
        next_time = previous.time.advance(dt_seconds, boundary_interval_seconds) if advance_time else previous.time
        if "dtbc" in after and after["dtbc"] is not before.get("dtbc", previous.time.dtbc):
            next_time = replace(next_time, dtbc=after["dtbc"])
        return ModelState(
            array_names=update_names,
            arrays=tuple(after[name] for name in update_names),
            time=next_time,
        )

    def merge_stage_updates(previous: ModelState, updates: ModelState) -> ModelState:
        if not updates.array_names:
            return replace(previous, time=updates.time)
        changed = updates.as_dict()
        if step_local_workspace:
            values = previous.as_dict()
            values.update(changed)
            names = tuple(sorted(values))
            return ModelState(
                array_names=names,
                arrays=tuple(values[name] for name in names),
                time=updates.time,
            )
        return replace(
            previous,
            arrays=tuple(changed.get(name, value) for name, value in zip(previous.array_names, previous.arrays, strict=True)),
            time=updates.time,
        )

    def barrier_state(state: ModelState) -> ModelState:
        """Route every recurrent leaf through a WRF semantic boundary."""

        return jax.tree_util.tree_map(optimization_barrier, state)

    def rk_stage_for(
        rk_step: int,
        *,
        advance_clock: bool,
        initialize: bool,
        stop_before_first_rk_part1: bool = False,
        stop_after_first_rk_part1: bool = False,
        start_after_first_rk_part1: bool = False,
    ) -> Callable[[ModelState, dict[str, Any]], ModelState]:
        def rk_stage(state: ModelState, static_arrays: dict[str, Any]) -> ModelState:
            fields = fields_from_state(
                state,
                static_arrays,
                step_clock_advanced=not advance_clock,
            )
            before = dict(fields)
            stage_fields, context = _initialize_direct_step_stage_context(
                plan=plan,
                grid=compiled_grid,
                config_flags=config_flags,
                fields=fields,
                moist=fields["moist"],
                scalar=fields.get("scalar"),
                advance_clock=advance_clock,
            )
            _run_rk_step_stage_family(
                state=stage_fields,
                callbacks=callbacks,
                plan=plan,
                grid=compiled_grid,
                config_flags=config_flags,
                boundary_config=context.boundary_config,
                boundary_kwargs_fn=context.boundary_kwargs,
                physical_bc_kwargs_fn=context.physical_bc_kwargs,
                num_3d_m=context.num_3d_m,
                num_3d_s=context.num_3d_s,
                specified=context.specified,
                constant_bc=context.constant_bc,
                specified_or_nested=context.specified_or_nested,
                non_hydrostatic=context.non_hydrostatic,
                fused_boundary_groups=context.fused_boundary_groups,
                fused_small_boundaries=context.fused_small_boundaries,
                spec_zone=context.spec_zone,
                f_flux=context.f_flux,
                restart_flag=context.restart_flag,
                curr_secs=context.curr_secs,
                adapt_step_flag=context.adapt_step_flag,
                max_vert_cfl_tmp=context.max_vert_cfl_tmp,
                max_horiz_cfl_tmp=context.max_horiz_cfl_tmp,
                rk_steps=(rk_step,),
                initialize=initialize,
                stop_before_first_rk_part1=stop_before_first_rk_part1,
                stop_after_first_rk_part1=stop_after_first_rk_part1,
                start_after_first_rk_part1=start_after_first_rk_part1,
            )
            return stage_updates_from_fields(
                state,
                before,
                stage_fields,
                advance_time=advance_clock,
            )

        return rk_stage

    def rk_substage_for(rk_step: int, stage_name: str) -> Callable[[ModelState, dict[str, Any]], ModelState]:
        def rk_substage(state: ModelState, static_arrays: dict[str, Any]) -> ModelState:
            fields = fields_from_state(state, static_arrays, step_clock_advanced=True)
            before = dict(fields)
            stage_fields, context = _initialize_direct_step_stage_context(
                plan=plan,
                grid=compiled_grid,
                config_flags=config_flags,
                fields=fields,
                moist=fields["moist"],
                scalar=fields.get("scalar"),
                advance_clock=False,
            )
            _run_rk_step_stage_family(
                state=stage_fields,
                callbacks=callbacks,
                plan=plan,
                grid=compiled_grid,
                config_flags=config_flags,
                boundary_config=context.boundary_config,
                boundary_kwargs_fn=context.boundary_kwargs,
                physical_bc_kwargs_fn=context.physical_bc_kwargs,
                num_3d_m=context.num_3d_m,
                num_3d_s=context.num_3d_s,
                specified=context.specified,
                constant_bc=context.constant_bc,
                specified_or_nested=context.specified_or_nested,
                non_hydrostatic=context.non_hydrostatic,
                fused_boundary_groups=context.fused_boundary_groups,
                fused_small_boundaries=context.fused_small_boundaries,
                spec_zone=context.spec_zone,
                f_flux=context.f_flux,
                restart_flag=context.restart_flag,
                curr_secs=context.curr_secs,
                adapt_step_flag=context.adapt_step_flag,
                max_vert_cfl_tmp=context.max_vert_cfl_tmp,
                max_horiz_cfl_tmp=context.max_horiz_cfl_tmp,
                rk_steps=(rk_step,),
                initialize=False,
                stage_names=(stage_name,),
            )
            return stage_updates_from_fields(state, before, stage_fields, advance_time=False)

        return rk_substage

    def first_rk_part1_stage_for(stage_names: tuple[str, ...]) -> Callable[[ModelState, dict[str, Any]], ModelState]:
        def first_rk_part1_stage(state: ModelState, static_arrays: dict[str, Any]) -> ModelState:
            fields = fields_from_state(state, static_arrays, step_clock_advanced=True)
            if "radiation_driver" in stage_names:
                # Ordinary compiled physics uses a post-startup static step
                # marker to eliminate first-step-only Python branches.  The
                # radiation event is an eager outer-schedule boundary and
                # must instead read the explicit TimeState step so its WRF
                # cadence continues after startup.
                fields.pop("_physics_itimestep", None)
            before = dict(fields)
            stage_fields, context = _initialize_direct_step_stage_context(
                plan=plan,
                grid=compiled_grid,
                config_flags=config_flags,
                fields=fields,
                moist=fields["moist"],
                scalar=fields.get("scalar"),
                advance_clock=False,
            )
            _seed_rk_step_size_fields(state=stage_fields, grid=compiled_grid, plan=plan, rk_step=1)
            stage_fields["itimestep"] = state.time.itimestep
            _run_first_rk_step_part1_stage_family(
                state=stage_fields,
                callbacks=callbacks,
                plan=plan,
                f_flux=context.f_flux,
                stage_names=stage_names,
            )
            return stage_updates_from_fields(state, before, stage_fields, advance_time=False)

        return first_rk_part1_stage

    def first_rk_part2_stage(state: ModelState, static_arrays: dict[str, Any]) -> ModelState:
        fields = fields_from_state(state, static_arrays, step_clock_advanced=True)
        before = dict(fields)
        stage_fields, context = _initialize_direct_step_stage_context(
            plan=plan,
            grid=compiled_grid,
            config_flags=config_flags,
            fields=fields,
            moist=fields["moist"],
            scalar=fields.get("scalar"),
            advance_clock=False,
        )
        _seed_rk_step_size_fields(state=stage_fields, grid=compiled_grid, plan=plan, rk_step=1)
        stage_fields["itimestep"] = state.time.itimestep
        stage_fields.update(adapt_step_flag=context.adapt_step_flag, curr_secs=context.curr_secs)
        _run_first_rk_step_part2_stage(
            state=stage_fields,
            callbacks=callbacks,
            adapt_step_flag=context.adapt_step_flag,
            curr_secs=context.curr_secs,
        )
        return stage_updates_from_fields(state, before, stage_fields, advance_time=False)

    def first_rk_physics_stage(state: ModelState, static_arrays: dict[str, Any]) -> ModelState:
        fields = fields_from_state(state, static_arrays, step_clock_advanced=True)
        before = dict(fields)
        stage_fields, context = _initialize_direct_step_stage_context(
            plan=plan, grid=compiled_grid, config_flags=config_flags,
            fields=fields, moist=fields["moist"], scalar=fields.get("scalar"),
            advance_clock=False,
        )
        _seed_rk_step_size_fields(state=stage_fields, grid=compiled_grid, plan=plan, rk_step=1)
        stage_fields.update(itimestep=state.time.itimestep, l_flux=bool(context.f_flux),
                            adapt_step_flag=context.adapt_step_flag, curr_secs=context.curr_secs)
        stage_fields.update(tiled_physics(stage_fields))
        return stage_updates_from_fields(state, before, stage_fields, advance_time=False)

    def post_rk_stage_for(stage_names: tuple[str, ...]) -> Callable[[ModelState, dict[str, Any]], ModelState]:
        def post_rk_stage(state: ModelState, static_arrays: dict[str, Any]) -> ModelState:
            fields = fields_from_state(state, static_arrays, step_clock_advanced=True)
            before = dict(fields)
            stage_fields, context = _initialize_direct_step_stage_context(
                plan=plan,
                grid=compiled_grid,
                config_flags=config_flags,
                fields=fields,
                moist=fields["moist"],
                scalar=fields.get("scalar"),
                advance_clock=False,
            )
            _run_post_rk_physics_stage_group(
                state=stage_fields,
                callbacks=callbacks,
                plan=plan,
                grid=compiled_grid,
                config_flags=config_flags,
                num_3d_m=context.num_3d_m,
                num_3d_s=context.num_3d_s,
                ids=context.ids,
                ide=context.ide,
                jds=context.jds,
                jde=context.jde,
                stage_names=stage_names,
            )
            return stage_updates_from_fields(state, before, stage_fields, advance_time=False)

        return post_rk_stage

    def final_stage(state: ModelState, static_arrays: dict[str, Any]) -> ModelState:
        fields = fields_from_state(state, static_arrays, step_clock_advanced=True)
        before = dict(fields)
        stage_fields, context = _initialize_direct_step_stage_context(
            plan=plan,
            grid=compiled_grid,
            config_flags=config_flags,
            fields=fields,
            moist=fields["moist"],
            scalar=fields.get("scalar"),
            advance_clock=False,
        )
        _run_final_boundary_stage_family(
            state=stage_fields,
            callbacks=callbacks,
            plan=plan,
            grid=compiled_grid,
            config_flags=config_flags,
            boundary_config=context.boundary_config,
            specified_or_nested=context.specified_or_nested,
            fused_boundary_groups=context.fused_boundary_groups,
            spec_bdy_width=context.spec_bdy_width,
            spec_zone=context.spec_zone,
            num_3d_m=context.num_3d_m,
            curr_secs=context.curr_secs,
            curr_secs2=context.curr_secs2,
            diag_flag=context.diag_flag,
        )
        return stage_updates_from_fields(state, before, stage_fields, advance_time=False)

    use_stage_families = _compiled_direct_stage_families_enabled()
    if stage_scan_semantics is None:
        stage_scan_semantics = _compiled_direct_stage_scan_enabled()
    use_semantic_calls = use_stage_families and bool(stage_scan_semantics)
    if full_state_barriers is None:
        full_state_barriers = _compiled_direct_full_state_barriers_enabled()
    use_full_state_barriers = use_stage_families and bool(full_state_barriers)
    rk_stage_specs: list[tuple[str, Callable[[ModelState, dict[str, Any]], ModelState]]] = [
        (
            "rk1_prepare",
            rk_stage_for(
                1,
                advance_clock=True,
                initialize=True,
                stop_before_first_rk_part1=True,
            ),
        ),
        ("rk1_init_zero", first_rk_part1_stage_for(("init_zero_tendency",))),
        ("rk1_phy_prep", first_rk_part1_stage_for(("phy_prep",))),
        ("rk1_radiation", first_rk_part1_stage_for(("radiation_driver",))),
        ("rk1_surface_pbl", first_rk_part1_stage_for(("surface_pbl_driver",))),
        ("rk1_cumulus", first_rk_part1_stage_for(("cumulus_driver",))),
        ("rk1_misc", first_rk_part1_stage_for(("shallowcu_driver", "fddagd_driver"))),
        ("rk1_part2", first_rk_part2_stage),
    ]
    rk_stage_specs.extend(
        (f"rk1_{stage_name}", rk_substage_for(1, stage_name))
        for stage_name in RK_COMPILED_SUBSTAGES
        if stage_name not in {"rk_step_prep", "rk_phys_bc_dry_1"}
    )
    rk_stage_specs.extend(
        (f"rk{rk_step}_{stage_name}", rk_substage_for(rk_step, stage_name))
        for rk_step in range(2, plan.rk_order + 1)
        for stage_name in RK_COMPILED_SUBSTAGES
    )
    rk1_dycore_diagnostic_stages = tuple(
        (f"rk1_{stage_name}", rk_substage_for(1, stage_name))
        for stage_name in RK_COMPILED_SUBSTAGES
        if stage_name not in {"rk_step_prep", "rk_phys_bc_dry_1"}
    )
    post_rk_stage_specs: tuple[tuple[str, Callable[[ModelState, dict[str, Any]], ModelState]], ...] = (
        ("post_advance_ppt", post_rk_stage_for(("post_advance_ppt",))),
        ("post_phy_prep_part2", post_rk_stage_for(("post_phy_prep_part2",))),
        ("post_moist_prep", post_rk_stage_for(("post_moist_prep",))),
        ("post_microphysics", post_rk_stage_for(("post_microphysics",))),
        ("post_moist_finish", post_rk_stage_for(("post_moist_finish",))),
        ("post_calc_p_rho_phi_final", post_rk_stage_for(("post_calc_p_rho_phi_final",))),
        ("post_physical_bc", post_rk_stage_for(("post_physical_bc",))),
    )
    stage_functions = tuple(rk_stage_specs) + tuple(post_rk_stage_specs) + (("final_boundary", final_stage),)
    if eager_stage_names is None:
        eager_stage_names = _compiled_direct_eager_stage_names()
    eager_stage_names = eager_stage_names | distributed_stage_names

    def stage_group_for(
        specs: tuple[tuple[str, Callable[[ModelState, dict[str, Any]], ModelState]], ...],
    ) -> Callable[[ModelState, dict[str, Any]], ModelState]:
        """Compose adjacent safe stages over one recurrent-state boundary."""

        def grouped_stage(state: ModelState, static_arrays: dict[str, Any]) -> ModelState:
            updated_names = set()
            for index, (_label, stage_fn) in enumerate(specs):
                updates = stage_fn(state, static_arrays)
                updated_names.update(updates.array_names)
                if index + 1 < len(specs) and not use_full_state_barriers:
                    updates = optimization_barrier(updates)
                state = merge_stage_updates(state, updates)
                if use_full_state_barriers and index + 1 < len(specs):
                    state = barrier_state(state)
            if step_local_workspace and specs[-1][0] == "final_boundary":
                # These temporaries die at the step boundary; do not materialize
                # them as executable outputs only to discard them on the host.
                updated_names.intersection_update(persistent_array_names)
            return state.select_arrays(updated_names)

        return grouped_stage

    execution_units: list[
        tuple[
            str,
            Callable[[ModelState, dict[str, Any]], ModelState],
            Callable[[ModelState, dict[str, Any]], ModelState],
            bool,
        ]
    ] = []
    pending_compiled: list[tuple[str, Callable[[ModelState, dict[str, Any]], ModelState]]] = []

    def reusable_stage_group(function: Any, labels: tuple[str, ...]) -> Any:
        return reusable_terminal_jit(function, preserve_call=True, context={
            "program": "compiled_stage_group", "stages": labels,
            "static_fields": static_scalar_fields, "grid": compiled_grid,
            "config_flags": config_flags, "plan": plan,
            "dynamic_names": dynamic_names, "persistent_names": persistent_array_names,
            "step_local_workspace": step_local_workspace,
            "stage_families": use_stage_families,
            "stage_scan_semantics": bool(stage_scan_semantics),
            "full_state_barriers": use_full_state_barriers,
            "dt_seconds": dt_seconds, "boundary_interval_seconds": boundary_interval_seconds,
        })

    def flush_compiled_group() -> None:
        if not pending_compiled:
            return
        specs = tuple(pending_compiled)
        label = f"compiled[{specs[0][0]}..{specs[-1][0]}]"
        eager_group = stage_group_for(specs)
        program = reusable_stage_group(eager_group, tuple(name for name, _fn in specs))
        execution_units.append((label, program, eager_group, False))
        pending_compiled.clear()

    resident_labels = {"rk1_surface_pbl", "rk1_cumulus", "rk1_misc", "rk1_part2"}
    resident_labels.update(f"rk{step}_{name}" for step in range(1, plan.rk_order + 1)
                           for name in RK_COMPILED_SUBSTAGES
                           if step != 1 or name not in {"rk_step_prep", "rk_phys_bc_dry_1"})
    resident_labels.update(name for group in POST_RK_TILE_GROUPS for name in group)
    for label, stage_fn in stage_functions:
        if tiled_physics is not None and label in resident_labels:
            if label == "rk1_surface_pbl":
                flush_compiled_group()
                specs = tuple((name, fn) for name, fn in stage_functions
                              if name in resident_labels)
                reference = stage_group_for(specs)
                execution_units.append(("rk_physics_dynamics", first_rk_physics_stage, reference, False))
            continue
        # These complete numerical stages match their individual calls on the
        # same first-step inputs. Host startup/radiation events can submit one
        # program per stage; enclosing scans still trace the original function.
        terminal_stage = label in {"rk1_part2", "rk1_rk_tendency", "rk3_moist_scalar_advance", "rk3_scalar_advance"}
        # The first step is specialized to 1; ordinary every-step physics
        # already has its post-initialization marker. Other schedules retain
        # their dynamic clock and the original host dispatch.
        clocked_physics = "_physics_itimestep" in static_scalar_fields and label in {"rk1_surface_pbl", "rk1_cumulus"}
        if (terminal_stage or clocked_physics) and label not in distributed_stage_names:
            stage_fn = reusable_terminal_jit(stage_fn, context={
                "program": "terminal_stage", "stage": label,
                "static_fields": static_scalar_fields, "grid": compiled_grid,
                "config_flags": config_flags, "callbacks": callbacks,
                "dynamic_names": dynamic_names,
            })
        if label in eager_stage_names:
            flush_compiled_group()
            execution_units.append((label, stage_fn, stage_fn, False))
        else:
            pending_compiled.append((label, stage_fn))
    flush_compiled_group()
    base_execution_units = tuple(execution_units)

    def coarse_execution_unit(
        units: tuple[
            tuple[
                str,
                Callable[[ModelState, dict[str, Any]], ModelState],
                Callable[[ModelState, dict[str, Any]], ModelState],
                bool,
            ],
            ...,
        ],
        *,
        preserve_units: bool = False,
        final_output_names: tuple[str, ...] | None = None,
    ) -> tuple[
        str,
        Callable[[ModelState, dict[str, Any]], ModelState],
        Callable[[ModelState, dict[str, Any]], ModelState],
        bool,
    ]:
        """Compile adjacent proven-safe execution units as one coarse call."""

        def compose(
            state: ModelState,
            static_arrays: dict[str, Any],
            *,
            reference: bool,
        ) -> ModelState:
            updated_names = set()
            for index, (_label, actual_fn, reference_fn, returns_full_state) in enumerate(units):
                function = reference_fn if reference else actual_fn
                result = (
                    semantic_call(function, state, static_arrays)
                    if preserve_units else function(state, static_arrays)
                )
                updated_names.update(result.array_names)
                state = result if returns_full_state else merge_stage_updates(state, result)
                if use_full_state_barriers and index + 1 < len(units):
                    state = barrier_state(state)
            if final_output_names is not None:
                updated_names.intersection_update(final_output_names)
            return state.select_arrays(updated_names)

        def actual_group(state: ModelState, static_arrays: dict[str, Any]) -> ModelState:
            return compose(state, static_arrays, reference=False)

        def reference_group(state: ModelState, static_arrays: dict[str, Any]) -> ModelState:
            return compose(state, static_arrays, reference=True)

        label = f"coarse[{units[0][0]}..{units[-1][0]}]"
        program = reusable_stage_group(actual_group, tuple(unit[0] for unit in units))
        return label, program, reference_group, False

    # Same-input range diagnostics establish these boundaries for the current
    # em_real production core.  Each RK preparation may share one graph with
    # its own small-step loop, but crossing an RK-stage boundary changes the
    # pressure solve materially.  The post-RK preparation tail is also safe to
    # coarsen only up to (and not through) Thompson microphysics.
    coarse_unit_ends = {
        "compiled[rk1_rk_tendency..rk1_small_step_prep]": "rk1_small_step_loop",
        "compiled[rk3_small_step_finish..rk3_moist_scalar_advance]": "post_moist_prep",
    }
    coarsened_units: list[
        tuple[
            str,
            Callable[[ModelState, dict[str, Any]], ModelState],
            Callable[[ModelState, dict[str, Any]], ModelState],
            bool,
        ]
    ] = []
    unit_index = 0
    while unit_index < len(base_execution_units):
        start_label = base_execution_units[unit_index][0]
        end_label = coarse_unit_ends.get(start_label)
        if end_label is None:
            coarsened_units.append(base_execution_units[unit_index])
            unit_index += 1
            continue
        end_index = unit_index
        while end_index < len(base_execution_units) and base_execution_units[end_index][0] != end_label:
            end_index += 1
        if end_index >= len(base_execution_units):
            coarsened_units.append(base_execution_units[unit_index])
            unit_index += 1
            continue
        coarsened_units.append(coarse_execution_unit(base_execution_units[unit_index : end_index + 1]))
        unit_index = end_index + 1
    compiled_execution_units = tuple(coarsened_units)

    # Split LLVM code generation makes these larger modules practical.  Both
    # ranges are bitwise exact against the ordered production units on the
    # configured em_real steady state; boundaries around Tiedtke, misc
    # physics, part2, and the post-microphysics tail remain explicit.
    verified_coarse_unit_ends = {
        "coarse[compiled[rk1_rk_tendency..rk1_small_step_prep]..rk1_small_step_loop]": "rk3_small_step_loop",
    }
    # Radiation calendar and scheduling remain host operations. Only ordinary
    # steps with radiation explicitly skipped can compile across this boundary.
    if static_scalar_fields.get("_skip_radiation", False):
        verified_coarse_unit_ends["compiled[rk1_prepare..rk1_init_zero]"] = "rk1_surface_pbl"
    verified_coarsened_units: list[
        tuple[
            str,
            Callable[[ModelState, dict[str, Any]], ModelState],
            Callable[[ModelState, dict[str, Any]], ModelState],
            bool,
        ]
    ] = []
    unit_index = 0
    while unit_index < len(compiled_execution_units):
        start_label = compiled_execution_units[unit_index][0]
        end_label = verified_coarse_unit_ends.get(start_label)
        if end_label is None:
            verified_coarsened_units.append(compiled_execution_units[unit_index])
            unit_index += 1
            continue
        end_index = unit_index
        while end_index < len(compiled_execution_units) and compiled_execution_units[end_index][0] != end_label:
            end_index += 1
        if end_index >= len(compiled_execution_units):
            verified_coarsened_units.append(compiled_execution_units[unit_index])
            unit_index += 1
            continue
        verified_coarsened_units.append(coarse_execution_unit(compiled_execution_units[unit_index : end_index + 1]))
        unit_index = end_index + 1
    compiled_execution_units = tuple(verified_coarsened_units)
    dynamics_label = "coarse[coarse[compiled[rk1_rk_tendency..rk1_small_step_prep]..rk1_small_step_loop]..rk3_small_step_loop]"
    post_preparation_label = "coarse[compiled[rk3_small_step_finish..rk3_moist_scalar_advance]..post_moist_prep]"
    post_finish_label = "compiled[post_microphysics..final_boundary]"
    if (step_local_workspace and not use_full_state_barriers
            and tuple(unit[0] for unit in compiled_execution_units[-3:])
            == (dynamics_label, post_preparation_label, post_finish_label)):
        # Keep the scientific kernels distinct while giving XLA ownership of
        # their shared dynamics/physics workspace until the end of the step.
        compiled_execution_units = (*compiled_execution_units[:-3], coarse_execution_unit(
            compiled_execution_units[-3:], preserve_units=True,
            final_output_names=persistent_array_names,
        ))
    if static_scalar_fields.get("_force_radiation", False) and not use_full_state_barriers:
        event_units = _group_radiation_event_units(
            base_execution_units,
            merge_stage_updates,
            static_physics_clock="_physics_itimestep" in static_scalar_fields,
            host_microphysics_diagnostics=bool(getattr(config_flags, "do_radar_ref", 0)),
            compile_water_preparation=jax.default_backend() == "gpu",
            distributed_stages=distributed_stage_names,
            final_output_names=frozenset(persistent_array_names) if step_local_workspace else None,
        )
        if event_units is not None:
            compiled_execution_units = event_units
    sync_stage_outputs = _compiled_direct_sync_stages_enabled()
    report_stage_progress = _compiled_direct_stage_progress_enabled()
    if scan_compiled_calls is None:
        scan_compiled_calls = os.environ.get(
            "WRF_PY_COMPILED_DIRECT_SCAN_COMPILED_CALLS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
    use_compiled_scan_calls = bool(scan_compiled_calls)
    scan_call_units = tuple(
        (
            actual_fn
            if returns_full_state
            else jax.jit(actual_fn, inline=False)
        )
        for _label, actual_fn, _reference_fn, returns_full_state in compiled_execution_units
    )

    def step(state: ModelState, static_arrays: dict[str, Any]) -> ModelState:
        if use_stage_families:
            persistent_input = state
            if step_local_workspace:
                # Radiation can replace an array supplied as a read-only input
                # on ordinary steps. Keep one field schema on both schedules,
                # with live state taking precedence and no array copies.
                values = dict(static_arrays)
                values.update(state.as_dict())
                names = tuple(sorted(values))
                state = ModelState(names, tuple(values[name] for name in names), state.time)
                static_arrays = {}

            def run_stage(
                label: str,
                fn: Any,
                reference_fn: Any,
                scan_fn: Any,
                returns_full_state: bool,
                current_state: ModelState,
            ) -> ModelState:
                if report_stage_progress:
                    _trajectory_progress(f"compiled staged {label} start")
                    stage_start = perf_counter()
                result = _execute_stage_function(
                    current_state, static_arrays,
                    host_fn=fn, reference_fn=reference_fn, scan_fn=scan_fn,
                    semantic_calls=use_semantic_calls,
                    compiled_scan_calls=use_compiled_scan_calls,
                )
                if sync_stage_outputs:
                    _block_until_ready_tree(result)
                    if report_stage_progress:
                        _trajectory_progress(f"compiled staged {label} done seconds={perf_counter() - stage_start:.3f}")
                elif report_stage_progress:
                    _trajectory_progress(f"compiled staged {label} enqueued seconds={perf_counter() - stage_start:.3f}")
                if returns_full_state:
                    return result
                return merge_stage_updates(current_state, result)

            for index, (label, stage_unit, reference_unit, returns_full_state) in enumerate(compiled_execution_units):
                state = run_stage(
                    label, stage_unit, reference_unit, scan_call_units[index],
                    returns_full_state, state,
                )
                if use_full_state_barriers and index + 1 < len(compiled_execution_units):
                    state = barrier_state(state)
            if step_local_workspace:
                final_values = state.as_dict()
                input_values = persistent_input.as_dict()
                state = ModelState(
                    array_names=persistent_array_names,
                    arrays=tuple(
                        final_values.get(name, input_values[name])
                        for name in persistent_array_names
                    ),
                    time=state.time,
                )
            return state
        fields = fields_from_state(state, static_arrays, step_clock_advanced=False)
        result = run_direct_step(
            plan=plan,
            grid=compiled_grid,
            config_flags=config_flags,
            fields=fields,
            moist=fields["moist"],
            scalar=fields.get("scalar"),
        )
        next_dynamic = {name: result.fields.get(name, fields[name]) for name in dynamic_names}
        return state_from_dynamic_fields(state, next_dynamic)

    def block(state: ModelState, static_arrays: dict[str, Any], valid_steps: Any) -> ModelState:
        def body(carry: ModelState, index: Any) -> tuple[ModelState, tuple[()]]:
            active = index < valid_steps
            next_dynamic = jax.lax.cond(active, lambda current: step(current, static_arrays), lambda unchanged: unchanged, carry)
            return next_dynamic, ()

        final_dynamic, _ = jax.lax.scan(body, state, jnp.arange(int(snapshot_block_steps)), unroll=scan_unroll)
        return final_dynamic

    def snapshot_block(
        state: ModelState,
        static_arrays: dict[str, Any],
        valid_steps: Any,
    ) -> tuple[ModelState, tuple[Any, Any, Any, Any, dict[str, Any]]]:
        def snapshot_arrays(current_state: ModelState) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
            current_dynamic = dynamic_fields_from_state(current_state)
            fields = dict(static_scalar_fields)
            fields.update(static_arrays)
            fields.update(current_dynamic)
            _refresh_current_dry_aliases(fields)
            _refresh_derived_steady_fields(fields)
            _seed_steady_scratch_fields(fields)
            return _snapshot_payload_from_fields(fields, config_flags)

        def body(carry: ModelState, index: Any) -> tuple[ModelState, tuple[Any, Any, Any, Any, dict[str, Any]]]:
            current = carry
            active = index < valid_steps
            next_dynamic = jax.lax.cond(active, lambda current_dynamic: step(current_dynamic, static_arrays), lambda unchanged: unchanged, current)
            return next_dynamic, snapshot_arrays(next_dynamic)

        return jax.lax.scan(body, state, jnp.arange(int(snapshot_block_steps)), unroll=scan_unroll)

    def sampled_block(
        state: ModelState,
        static_arrays: dict[str, Any],
        valid_steps: Any,
    ) -> tuple[ModelState, tuple[Any, Any, Any, Any, dict[str, Any]]]:
        final_dynamic = block(state, static_arrays, valid_steps)
        dynamic = dynamic_fields_from_state(final_dynamic)
        fields = dict(static_scalar_fields)
        fields.update(static_arrays)
        fields.update(dynamic)
        _refresh_current_dry_aliases(fields)
        _refresh_derived_steady_fields(fields)
        _seed_steady_scratch_fields(fields)
        return final_dynamic, _snapshot_payload_from_fields(fields, config_flags)

    compiled_step = step if use_stage_families else jax.jit(step, donate_argnums=(0,))
    if use_stage_families:
        cumulus_indices = tuple(
            index
            for index, (label, _actual, _reference, _returns_full_state) in enumerate(compiled_execution_units)
            if label == "rk1_cumulus"
        )
        microphysics_indices = tuple(
            index
            for index, (label, _actual, _reference, _returns_full_state) in enumerate(compiled_execution_units)
            if label == "post_microphysics"
        )
        if len(cumulus_indices) == 1 and len(microphysics_indices) == 1:
            cumulus_index = cumulus_indices[0]
            microphysics_index = microphysics_indices[0]
            if cumulus_index >= microphysics_index:
                raise ValueError("rk1_cumulus must precede post_microphysics")

            def segmented_group(
                units: tuple[
                    tuple[
                        str,
                        Callable[[ModelState, dict[str, Any]], ModelState],
                        Callable[[ModelState, dict[str, Any]], ModelState],
                        bool,
                    ],
                    ...,
                ],
            ) -> Callable[[ModelState, dict[str, Any]], ModelState]:
                def run_group(state: ModelState, static_arrays: dict[str, Any]) -> ModelState:
                    for _label, _actual, reference_fn, returns_full_state in units:
                        result = semantic_call(reference_fn, state, static_arrays)
                        state = result if returns_full_state else merge_stage_updates(state, result)
                    return state

                # The GPU memcpy rewrite loses loop state inside a semantic
                # call in jaxlib 0.9.2. Keep ordinary slice/copy operations and
                # preserve the scientific loop boundaries on both devices.
                # These GPU-only passes are absent from CPU compilation.
                return jax.jit(
                    run_group,
                    inline=False,
                    compiler_options=SEMANTIC_CALL_COMPILER_OPTIONS,
                )

            pre_cumulus = segmented_group(compiled_execution_units[:cumulus_index])
            between_column_physics = segmented_group(
                compiled_execution_units[cumulus_index + 1 : microphysics_index]
            )
            post_microphysics = segmented_group(compiled_execution_units[microphysics_index + 1 :])
            _cumulus_label, _cumulus_actual, cumulus_reference, cumulus_returns_full = (
                compiled_execution_units[cumulus_index]
            )
            _microphysics_label, _microphysics_actual, microphysics_reference, microphysics_returns_full = (
                compiled_execution_units[microphysics_index]
            )
            def segmented_column_physics_step(
                state: ModelState,
                static_arrays: dict[str, Any],
            ) -> ModelState:
                persistent_input = state
                state = pre_cumulus(state, static_arrays)
                result = cumulus_reference(state, static_arrays)
                state = result if cumulus_returns_full else merge_stage_updates(state, result)
                state = between_column_physics(state, static_arrays)
                result = microphysics_reference(state, static_arrays)
                state = result if microphysics_returns_full else merge_stage_updates(state, result)
                state = post_microphysics(state, static_arrays)
                if step_local_workspace:
                    final_values = state.as_dict()
                    input_values = persistent_input.as_dict()
                    state = ModelState(
                        array_names=persistent_array_names,
                        arrays=tuple(
                            final_values.get(name, input_values[name])
                            for name in persistent_array_names
                        ),
                        time=state.time,
                    )
                return state

            setattr(
                compiled_step,
                "_wrf_segmented_column_physics_step",
                segmented_column_physics_step,
            )
        setattr(compiled_step, "_wrf_stage_functions", stage_functions)
        setattr(compiled_step, "_wrf_merge_stage_updates", merge_stage_updates)
        setattr(compiled_step, "_wrf_rk1_dycore_diagnostic_stages", rk1_dycore_diagnostic_stages)
        setattr(compiled_step, "_wrf_base_execution_units", base_execution_units)
        setattr(compiled_step, "_wrf_execution_units", compiled_execution_units)
    return (
        compiled_step,
        jax.jit(block, donate_argnums=(0,)),
        jax.jit(snapshot_block, donate_argnums=(0,)),
        jax.jit(sampled_block, donate_argnums=(0,)),
        dynamic_state,
        static_array_fields,
    )


def _compiled_startup_direct_function(
    *,
    static_fields: dict[str, Any],
    grid: Any,
    config_flags: Any,
    callbacks: Any,
) -> Any:
    """Build a compiled direct step for the first model step.

    The first step produces many scratch and diagnostic arrays that the
    post-startup steady carry does not know how to recreate. Return every array
    field from the traced direct step so the following steady path starts from
    the same field surface as the eager startup path.
    """

    compiled_grid = copy.copy(grid)
    plan = build_direct_step_plan(
        grid=compiled_grid,
        config_flags=config_flags,
        callbacks=callbacks,
        num_moist=int(getattr(config_flags, "num_moist", 2)),
        num_scalar=int(getattr(config_flags, "num_scalar", 0)),
    )

    def step(dynamic: dict[str, Any]) -> dict[str, Any]:
        fields = dict(static_fields)
        fields.update(dynamic)
        result = run_direct_step(
            plan=plan,
            grid=compiled_grid,
            config_flags=config_flags,
            fields=fields,
            moist=fields["moist"],
            scalar=fields.get("scalar"),
        )
        if _compiled_startup_steady_only_enabled():
            names = _explicit_steady_dynamic_names(result.fields)
            return {name: result.fields[name] for name in names if name in result.fields}
        return {name: value for name, value in result.fields.items() if _is_dynamic_array_value(value)}

    donate_argnums = (0,) if _compiled_startup_donate_enabled() else ()
    return jax.jit(step, donate_argnums=donate_argnums)


def _compiled_direct_compile_startup_enabled(default: bool = True) -> bool:
    value = os.environ.get("WRF_PY_COMPILED_DIRECT_COMPILE_STARTUP")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compiled_direct_stage_families_enabled(default: bool = True) -> bool:
    value = os.environ.get("WRF_PY_COMPILED_DIRECT_STAGE_FAMILIES")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compiled_direct_stage_scan_enabled(default: bool = False) -> bool:
    """Return whether stage-family steps execute inside a multi-step scan."""

    value = os.environ.get("WRF_PY_COMPILED_DIRECT_STAGE_SCAN")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compiled_direct_segmented_column_physics_enabled(default: bool = False) -> bool:
    """Return whether Tiedtke and Thompson execute between persistent graphs."""

    value = os.environ.get("WRF_PY_COMPILED_DIRECT_SEGMENT_COLUMN_PHYSICS")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compiled_direct_full_state_barriers_enabled(default: bool = False) -> bool:
    """Return whether experimental scans barrier every recurrent state leaf."""

    value = os.environ.get("WRF_PY_COMPILED_DIRECT_FULL_STATE_BARRIERS")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compiled_direct_eager_rk1_physics_enabled(default: bool = True) -> bool:
    value = os.environ.get("WRF_PY_COMPILED_DIRECT_EAGER_RK1_PHYSICS")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compiled_direct_eager_stage_names() -> frozenset[str]:
    raw = os.environ.get("WRF_PY_COMPILED_DIRECT_EAGER_STAGES")
    if raw is None:
        if _compiled_direct_eager_rk1_physics_enabled():
            raw = "rk1_phy_prep,rk1_radiation,rk1_surface_pbl,rk1_cumulus,rk1_misc,post_microphysics"
        else:
            raw = ""
    return frozenset(name.strip() for name in raw.split(",") if name.strip())


_STARTUP_SEMANTIC_EAGER_STAGE_NAMES = frozenset(
    (
        "rk1_dry_boundary_tendency",
        "rk1_moist_scalar_advance",
    )
)


def _compiled_direct_startup_eager_stage_names() -> frozenset[str]:
    """Return the steady boundaries plus the two first-step-only boundaries.

    Same-input startup diagnostics show that these two additional boundaries
    preserve the strict WRF first-step result while the ordinary recurrent
    trajectory does not need to pay for them on every model step.
    """

    return _compiled_direct_eager_stage_names() | _STARTUP_SEMANTIC_EAGER_STAGE_NAMES


def _compiled_direct_eager_radiation_fallback_enabled(default: bool | None = None) -> bool:
    value = os.environ.get("WRF_PY_COMPILED_DIRECT_EAGER_RADIATION_FALLBACK")
    if default is None:
        default = not _compiled_direct_stage_families_enabled()
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compiled_startup_steady_only_enabled(default: bool = False) -> bool:
    value = os.environ.get("WRF_PY_COMPILED_STARTUP_STEADY_ONLY")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compiled_startup_donate_enabled(default: bool = True) -> bool:
    value = os.environ.get("WRF_PY_DONATE_COMPILED_STARTUP")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compiled_direct_block_steps(default: int = 256) -> int:
    raw = os.environ.get("WRF_PY_COMPILED_DIRECT_BLOCK_STEPS", str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _compiled_direct_scan_unroll(default: int = 1) -> int:
    raw = os.environ.get("WRF_PY_COMPILED_DIRECT_SCAN_UNROLL", str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _compiled_direct_startup_steps(default: int = 1) -> int:
    raw = os.environ.get("WRF_PY_COMPILED_DIRECT_STARTUP_STEPS", str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, value)


def _compiled_direct_probe_enabled(default: bool = False) -> bool:
    value = os.environ.get("WRF_PY_COMPILED_DIRECT_PROBE_STEP")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compiled_direct_snapshot_scan_enabled(default: bool = True) -> bool:
    value = os.environ.get("WRF_PY_COMPILED_DIRECT_SNAPSHOT_SCAN")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compiled_direct_sync_stages_enabled(default: bool = False) -> bool:
    value = os.environ.get("WRF_PY_COMPILED_DIRECT_SYNC_STAGES")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _compiled_direct_stage_progress_enabled(default: bool = False) -> bool:
    value = os.environ.get("WRF_PY_COMPILED_DIRECT_STAGE_PROGRESS")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _trajectory_progress_enabled(default: bool = True) -> bool:
    value = os.environ.get("WRF_PY_TRAJECTORY_PROGRESS")
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _trajectory_progress_interval_seconds(default: int) -> int:
    value = os.environ.get("WRF_PY_TRAJECTORY_PROGRESS_INTERVAL_SECONDS")
    if value is None:
        return max(1, int(default))
    try:
        return max(1, int(value))
    except ValueError:
        return max(1, int(default))


def _trajectory_progress(message: str) -> None:
    if _trajectory_progress_enabled():
        write_progress(
            message,
            os.environ.get("WRF_PY_TRAJECTORY_PROGRESS_FILE"),
            os.environ.get("WRF_PY_TRAJECTORY_TIMING_FILE"),
        )


def _next_snapshot_step(step: int, snapshot_steps: set[int], step_count: int) -> int | None:
    candidates = [snapshot_step for snapshot_step in snapshot_steps if snapshot_step >= step]
    if not candidates:
        return None
    return min(min(candidates), step_count)


def _next_boundary_reset_step(step: int, step_count: int, dt_seconds: int, boundary_interval_seconds: int) -> int | None:
    if boundary_interval_seconds <= 0:
        return None
    interval_steps = int(boundary_interval_seconds) // int(dt_seconds)
    if interval_steps <= 0:
        return None
    next_step = ((int(step) + interval_steps - 1) // interval_steps) * interval_steps
    if next_step < int(step):
        next_step += interval_steps
    if next_step > int(step_count):
        return None
    return next_step


def _next_radiation_step(step: int, step_count: int, config_flags: Any) -> int | None:
    lw = int(getattr(config_flags, "ra_lw_physics", 0))
    sw = int(getattr(config_flags, "ra_sw_physics", 0))
    if lw == 0 and sw == 0:
        return None
    stepra = max(1, int(getattr(config_flags, "stepra", 1)))
    radt = float(getattr(config_flags, "radt", 0.0))
    if step <= 1 or stepra == 1 or radt == 0.0:
        return int(step)
    target_mod = (1 + int(getattr(config_flags, "ra_call_offset", 0))) % stepra
    delta = (target_mod - (int(step) % stepra)) % stepra
    next_step = int(step) + delta
    if next_step > int(step_count):
        return None
    return next_step


def _model_state_with_dynamic_fields(state: ModelState, dynamic_fields: Mapping[str, Any]) -> ModelState:
    values = state.as_dict()
    values.update({name: value for name, value in dynamic_fields.items() if name in values})
    next_time = state.time
    if "dtbc" in dynamic_fields:
        next_time = replace(next_time, dtbc=dynamic_fields["dtbc"])
    return replace(state, arrays=tuple(values[name] for name in state.array_names), time=next_time)


def _snapshot_from_arrays(
    *,
    arrays: tuple[Any, Any, Any, Any],
    start_timestamp: str,
    elapsed_seconds: int,
) -> PrognosticSnapshot:
    if len(arrays) == 4:
        u, v, t, qvapor = arrays
        extra = None
    else:
        u, v, t, qvapor, extra = arrays
    return PrognosticSnapshot(
        elapsed_seconds=int(elapsed_seconds),
        timestamp=_timestamp(start_timestamp, int(elapsed_seconds)),
        u=u,
        v=v,
        t=t,
        qvapor=qvapor,
        extra=extra,
    )


def run_compiled_direct_trajectory(
    case_dir: str | Path,
    *,
    hours: int = 6,
    duration_seconds: int | None = None,
    snapshot_interval_seconds: int = 3600,
    snapshot_callback: Callable[[PrognosticSnapshot], None] | None = None,
    keep_snapshots: bool = True,
) -> TrajectoryResult:
    """Run direct WRF-JAX with a compiled steady-state step after startup."""

    _trajectory_progress(f"assemble_runtime start case_dir={case_dir}")
    assembly = assemble_runtime(str(case_dir))
    _trajectory_progress("assemble_runtime done")
    dt = float(assembly.case.config.time_step)
    total_seconds = int(duration_seconds) if duration_seconds is not None else int(hours) * 3600
    if total_seconds <= 0:
        raise ValueError("trajectory duration must be positive")
    if total_seconds % int(dt) != 0:
        raise ValueError(f"requested duration {total_seconds}s is not divisible by time_step {dt}s")
    step_count = total_seconds // int(dt)
    snapshot_seconds = set(range(int(snapshot_interval_seconds), total_seconds + 1, int(snapshot_interval_seconds)))
    snapshot_seconds_steps = {int(seconds // int(dt)) for seconds in snapshot_seconds}
    snapshot_interval_steps = max(1, int(snapshot_interval_seconds) // int(dt))

    fields = dict(assembly.fields)
    fields.setdefault("dtbc", jnp.asarray(0.0, dtype=jnp.float32))
    grid = assembly.grid
    config_flags = assembly.config_flags
    use_stage_families = _compiled_direct_stage_families_enabled()
    use_segmented_column_physics = (
        use_stage_families and _compiled_direct_segmented_column_physics_enabled()
    )
    use_stage_scan = (
        use_stage_families
        and _compiled_direct_stage_scan_enabled()
        and not use_segmented_column_physics
    )
    use_compiled_radiation_events = use_stage_scan or use_segmented_column_physics
    use_eager_radiation_fallback = _compiled_direct_eager_radiation_fallback_enabled()
    use_scheduled_radiation_events = use_compiled_radiation_events or use_eager_radiation_fallback
    plan = build_direct_step_plan(
        grid=grid,
        config_flags=config_flags,
        callbacks=assembly.callbacks,
        num_moist=_num_moist_from_fields(fields),
        num_scalar=_num_scalar_from_fields(fields),
    )
    final_step: SolveEmResult | None = None
    snapshots: list[PrognosticSnapshot] = []

    def record_snapshot(snapshot: PrognosticSnapshot) -> None:
        # Archived outputs must not pin device buffers from earlier steps.
        # Reuse the host snapshot for both streaming and the returned history.
        if _materialize_snapshot_during_trajectory():
            snapshot = materialize_snapshots((snapshot,))[0]
        if snapshot_callback is not None:
            snapshot_callback(snapshot)
        if keep_snapshots:
            snapshots.append(snapshot)

    runtime_metrics: dict[str, float] = {
        "startup_execute_seconds": 0.0,
        "probe_execute_seconds": 0.0,
        "compiled_step_execute_seconds": 0.0,
        "compiled_block_compile_seconds": 0.0,
        "compiled_block_execute_seconds": 0.0,
        "compiled_snapshot_block_compile_seconds": 0.0,
        "compiled_snapshot_block_execute_seconds": 0.0,
        "compiled_sampled_block_compile_seconds": 0.0,
        "compiled_sampled_block_execute_seconds": 0.0,
        "compiled_startup_compile_seconds": 0.0,
        "compiled_startup_execute_seconds": 0.0,
        "compiled_step_chunks": 0.0,
        "compiled_block_chunks": 0.0,
        "compiled_snapshot_block_chunks": 0.0,
        "compiled_sampled_block_chunks": 0.0,
        "compiled_startup_chunks": 0.0,
        "compiled_block_model_steps": 0.0,
        "compiled_snapshot_block_model_steps": 0.0,
        "compiled_sampled_block_model_steps": 0.0,
        "compiled_startup_model_steps": 0.0,
        "compiled_radiation_fallback_execute_seconds": 0.0,
        "compiled_radiation_fallback_chunks": 0.0,
        "compiled_radiation_fallback_model_steps": 0.0,
        "compiled_radiation_event_execute_seconds": 0.0,
        "compiled_radiation_event_chunks": 0.0,
        "compiled_radiation_event_model_steps": 0.0,
        "startup_steps": 0.0,
        "probe_steps": 0.0,
        "steady_dynamic_carry_fields": 0.0,
        "steady_input_buffer_copies": 0.0,
        "startup_input_buffer_copies": 0.0,
        "compiled_snapshot_scan_enabled": 0.0,
        "compiled_direct_block_steps": 0.0,
        "snapshot_interval_steps": float(snapshot_interval_steps),
        "steady_static_array_fields": 0.0,
        "steady_static_array_bytes": 0.0,
        "steady_static_scalar_fields": 0.0,
        "compiled_direct_stage_families": 1.0 if use_stage_families else 0.0,
        "compiled_direct_stage_scan": 1.0 if use_stage_scan else 0.0,
        "compiled_direct_segmented_column_physics": 1.0 if use_segmented_column_physics else 0.0,
        "compiled_direct_radiation_events": 1.0 if use_compiled_radiation_events else 0.0,
        "compiled_direct_eager_rk1_physics": 1.0 if _compiled_direct_eager_rk1_physics_enabled() else 0.0,
        "compiled_direct_eager_stage_count": float(len(_compiled_direct_eager_stage_names())),
        "compiled_startup_eager_stage_count": float(len(_compiled_direct_startup_eager_stage_names())),
        "compiled_startup_stage_family": 0.0,
        "compiled_direct_eager_radiation_fallback": 1.0 if use_eager_radiation_fallback else 0.0,
    }

    startup_steps = min(_compiled_direct_startup_steps(), step_count)
    runtime_metrics["startup_steps"] = float(startup_steps)
    _set_grid_elapsed_seconds(grid, 0)

    compiled_startup_done = 0
    if startup_steps > 0 and _compiled_direct_compile_startup_enabled():
        if use_stage_families:
            _trajectory_progress(f"compiled staged startup execute start steps={startup_steps}")
            _seed_missing_startup_steady_fields(fields)
            _seed_steady_scratch_fields(fields)
            _seed_interstage_rk_scratch_fields(fields)
            startup_dynamic_names = _compiled_startup_dynamic_names(fields)
            startup_partition = _partition_fields_for_compiled_step(
                fields,
                dynamic_names=startup_dynamic_names,
                prune_static_fields=_prune_refreshed_steady_static_fields,
                deduplicate_buffers=False,
            )
            startup_scalar_fields = dict(startup_partition.static_scalar_fields)
            # This executable runs only the first step. Keep Noah's initial
            # water/ice soil updates and all other step-1 physics branches
            # static while TimeState continues to carry the model clock.
            startup_scalar_fields["_physics_itimestep"] = 1
            (
                compiled_startup,
                _startup_block,
                _startup_snapshot_block,
                _startup_sampled_block,
                startup_state,
                startup_static_arrays,
            ) = _compiled_steady_direct_functions(
                static_scalar_fields=startup_scalar_fields,
                static_array_fields=startup_partition.static_array_fields,
                dynamic_state=startup_partition.model_state(grid),
                dynamic_names=startup_dynamic_names,
                grid=grid,
                config_flags=config_flags,
                callbacks=assembly.callbacks,
                snapshot_block_steps=1,
                eager_stage_names=_compiled_direct_startup_eager_stage_names(),
            )
            metric_start = perf_counter()
            startup_state = compiled_startup(startup_state, startup_static_arrays)
            _block_until_ready_tree(startup_state)
            runtime_metrics["compiled_startup_execute_seconds"] += perf_counter() - metric_start
            runtime_metrics["compiled_startup_stage_family"] = 1.0
            fields = dict(startup_partition.static_scalar_fields)
            fields.update(startup_static_arrays)
            fields.update(startup_partition.dynamic_fields_from_state(startup_state))
            _refresh_current_dry_aliases(fields)
            _refresh_derived_steady_fields(fields)
            _seed_steady_scratch_fields(fields)
            _seed_interstage_rk_scratch_fields(fields)
            # These startup services return scalar completion flags alongside
            # array updates.  The staged PyTree carries arrays and explicit
            # model time, so restore the two deterministic flags after all of
            # their corresponding arrays have crossed the startup boundary.
            fields["_boundary_weights_ready"] = True
            fields["_surface_defaults_complete"] = True
            # Startup-only containers otherwise retain an entire obsolete
            # forecast, scratch arrays and executable closures for the run.
            del startup_partition, startup_state, startup_static_arrays, startup_scalar_fields
            del compiled_startup, _startup_block, _startup_snapshot_block, _startup_sampled_block
            _trajectory_progress(
                "compiled staged startup execute done "
                f"seconds={runtime_metrics['compiled_startup_execute_seconds']:.3f}"
            )
        else:
            _trajectory_progress(f"compiled startup lower/compile start steps={startup_steps}")
            static_startup_fields, dynamic_startup_fields = _split_static_dynamic_fields(fields)
            dynamic_startup_fields, copy_count = _deduplicate_input_buffers(dynamic_startup_fields)
            runtime_metrics["startup_input_buffer_copies"] += float(copy_count)
            compiled_startup = _compiled_startup_direct_function(
                static_fields=static_startup_fields,
                grid=grid,
                config_flags=config_flags,
                callbacks=assembly.callbacks,
            )
            metric_start = perf_counter()
            compiled_startup_executable = compiled_startup.lower(dynamic_startup_fields).compile()
            runtime_metrics["compiled_startup_compile_seconds"] += perf_counter() - metric_start
            _trajectory_progress(
                f"compiled startup compile done seconds={runtime_metrics['compiled_startup_compile_seconds']:.3f}"
            )
            metric_start = perf_counter()
            startup_arrays = compiled_startup_executable(dynamic_startup_fields)
            _block_until_ready_tree(startup_arrays)
            runtime_metrics["compiled_startup_execute_seconds"] += perf_counter() - metric_start
            fields = dict(static_startup_fields)
            if _compiled_startup_steady_only_enabled() and not _compiled_startup_donate_enabled():
                fields.update(dynamic_startup_fields)
            fields.update(startup_arrays)
            _trajectory_progress(
                f"compiled startup execute done seconds={runtime_metrics['compiled_startup_execute_seconds']:.3f}"
            )
        runtime_metrics["compiled_startup_chunks"] += 1.0
        runtime_metrics["compiled_startup_model_steps"] += 1.0
        fields["itimestep"] = 1
        fields["elapsed_seconds"] = float(dt)
        setattr(grid, "itimestep", 1)
        if "dtbc" in fields:
            try:
                setattr(grid, "dtbc", float(np.asarray(jax.device_get(fields["dtbc"])).reshape(-1)[0]))
            except (TypeError, ValueError):
                pass
        elapsed = int(round(dt))
        _set_grid_elapsed_seconds(grid, elapsed)
        _refresh_wrfbdy_fields_on_boundary_time(assembly, grid, fields, elapsed)
        if elapsed in snapshot_seconds:
            record_snapshot(
                _snapshot_from_fields(
                    fields,
                    assembly.case.config.start_time,
                    elapsed,
                    config_flags,
                    check_finite=False,
                )
            )
        final_step = SolveEmResult(
            grid=grid,
            config_flags=config_flags,
            fields=fields,
            ijk=plan.ijk,
            subgrid_ijk=plan.subgrid_ijk,
            rk_order=plan.rk_order,
            num_sound_steps=plan.num_sound_steps,
            dts=plan.dts,
            adapt_step_flag=False,
            restart_flag=False,
            diag_flag=False,
            curr_secs=float(elapsed),
            curr_secs2=float(elapsed),
            max_vert_cfl_tmp=(0.0,),
            max_horiz_cfl_tmp=(0.0,),
            events=(),
        )
        compiled_startup_done = 1

    for step in range(compiled_startup_done + 1, startup_steps + 1):
        _trajectory_progress(f"eager startup step start step={step}")
        metric_start = perf_counter()
        final_step = run_direct_step(
            plan=plan,
            grid=grid,
            config_flags=config_flags,
            fields=fields,
            moist=fields["moist"],
            scalar=fields.get("scalar"),
        )
        runtime_metrics["startup_execute_seconds"] += perf_counter() - metric_start
        _trajectory_progress(
            f"eager startup step done step={step} cumulative_seconds={runtime_metrics['startup_execute_seconds']:.3f}"
        )
        fields = final_step.fields
        elapsed = int(round(step * dt))
        _set_grid_elapsed_seconds(grid, elapsed)
        _refresh_wrfbdy_fields_on_boundary_time(assembly, grid, fields, elapsed)
        if elapsed in snapshot_seconds:
            record_snapshot(
                _snapshot_from_fields(
                    fields,
                    assembly.case.config.start_time,
                    elapsed,
                    config_flags,
                    check_finite=False,
                )
            )

    if startup_steps < step_count:
        probe_step = startup_steps
        auto_steady_carry = os.environ.get("WRF_PY_AUTO_STEADY_CARRY", "0").strip().lower() in {"1", "true", "yes", "on"}
        before_probe = dict(fields) if auto_steady_carry else None
        if _compiled_direct_probe_enabled():
            probe_step = startup_steps + 1
            runtime_metrics["probe_steps"] = 1.0
            metric_start = perf_counter()
            final_step = run_direct_step(
                plan=plan,
                grid=grid,
                config_flags=config_flags,
                fields=fields,
                moist=fields["moist"],
                scalar=fields.get("scalar"),
            )
            runtime_metrics["probe_execute_seconds"] += perf_counter() - metric_start
            fields = final_step.fields
            elapsed = int(round(probe_step * dt))
            _set_grid_elapsed_seconds(grid, elapsed)
            _refresh_wrfbdy_fields_on_boundary_time(assembly, grid, fields, elapsed)
            if elapsed in snapshot_seconds:
                record_snapshot(
                    _snapshot_from_fields(
                        fields,
                        assembly.case.config.start_time,
                        elapsed,
                        config_flags,
                        check_finite=False,
                    )
                )

        if use_scheduled_radiation_events:
            # Startup must retain WRF's normal radiation schedule, including
            # the first-step call.  The steady executable omits radiation and
            # the outer event loop executes only the scheduled radiation
            # steps eagerly.
            fields["_skip_radiation"] = True

        if auto_steady_carry:
            dynamic_names = _changed_dynamic_field_names(before_probe, fields)
        else:
            if (
                float(getattr(config_flags, "bldt", 0.0)) == 0.0
                and float(getattr(config_flags, "cudt", 0.0)) == 0.0
            ):
                # Startup-only column-physics predicates are invariant after
                # the eager first step for the every-step bldt/cudt setup.
                # Keep that marker static so the steady graph never applies
                # Python int() to TimeState tracers.  The radiation stage
                # explicitly removes it and therefore still follows the
                # dynamic WRF event clock.
                fields["_physics_itimestep"] = max(4, int(probe_step) + 1)
            if startup_steps == 0 and (_seed_startup_steady_fields_enabled() or use_stage_families):
                _seed_missing_startup_steady_fields(fields)
            if use_stage_families:
                _seed_steady_scratch_fields(fields)
                _seed_interstage_rk_scratch_fields(fields)
            dynamic_names = _explicit_steady_dynamic_names(fields)
        del before_probe
        final_step = None
        runtime_metrics["steady_dynamic_carry_fields"] = float(len(dynamic_names))
        partition = _partition_fields_for_compiled_step(
            fields,
            dynamic_names=dynamic_names,
            prune_static_fields=_prune_refreshed_steady_static_fields,
            deduplicate_buffers=not use_stage_families or use_stage_scan,
        )
        dynamic_state = partition.model_state(grid)
        static_array_fields = partition.static_array_fields
        static_fields = partition.static_scalar_fields
        runtime_metrics["steady_input_buffer_copies"] += float(partition.input_buffer_copies)
        runtime_metrics["steady_static_array_fields"] = float(len(static_array_fields))
        runtime_metrics["steady_static_array_bytes"] = float(partition.static_array_bytes)
        runtime_metrics["steady_static_scalar_fields"] = float(len(static_fields))
        del partition
        use_snapshot_scan = (
            _compiled_direct_snapshot_scan_enabled()
            and int(snapshot_interval_seconds) == int(dt)
            and int(snapshot_interval_steps) == 1
        )
        runtime_metrics["compiled_snapshot_scan_enabled"] = 1.0 if use_snapshot_scan else 0.0
        remaining_steady_steps = max(1, int(step_count) - int(probe_step))
        block_steps = min(_compiled_direct_block_steps(), remaining_steady_steps)
        if use_stage_families and not use_stage_scan:
            block_steps = 1
        if use_scheduled_radiation_events:
            radiation_interval_steps = max(1, int(getattr(config_flags, "stepra", 1)))
            if float(getattr(config_flags, "radt", 0.0)) == 0.0:
                radiation_interval_steps = 1
            block_steps = min(block_steps, max(1, radiation_interval_steps - 1))
        if snapshot_seconds_steps and not use_snapshot_scan:
            # The compiled block scans a fixed length and masks inactive steps.
            # Compile only the largest chunk that the snapshot schedule can
            # actually execute; otherwise multi-hour runs repeatedly scan all
            # remaining steps while only the first output interval is active.
            block_steps = min(block_steps, max(1, int(snapshot_interval_steps)))
        runtime_metrics["compiled_direct_block_steps"] = float(block_steps)
        _trajectory_progress(
            "steady setup "
            f"probe_step={probe_step} step_count={step_count} dynamic_fields={len(dynamic_names)} "
            f"block_steps={block_steps} snapshot_scan={int(use_snapshot_scan)}"
        )
        if probe_step < step_count:
            (
                compiled_step,
                compiled_block,
                compiled_snapshot_block,
                compiled_sampled_block,
                dynamic_state,
                static_array_fields,
            ) = _compiled_steady_direct_functions(
                static_scalar_fields=static_fields,
                static_array_fields=static_array_fields,
                dynamic_state=dynamic_state,
                dynamic_names=dynamic_names,
                grid=grid,
                config_flags=config_flags,
                callbacks=assembly.callbacks,
                snapshot_block_steps=block_steps,
            )
            if use_segmented_column_physics:
                segmented_step = getattr(
                    compiled_step,
                    "_wrf_segmented_column_physics_step",
                    None,
                )
                if segmented_step is None:
                    raise RuntimeError(
                        "segmented column physics requires rk1_cumulus and post_microphysics execution units"
                    )
                compiled_step = segmented_step
        compiled_radiation_step = None
        if probe_step < step_count and use_compiled_radiation_events:
            radiation_static_fields = dict(static_fields)
            radiation_static_fields["_skip_radiation"] = False
            radiation_static_fields["_force_radiation"] = True
            (
                compiled_radiation_step,
                _radiation_block,
                _radiation_snapshot_block,
                _radiation_sampled_block,
                radiation_dynamic_state,
                radiation_static_arrays,
            ) = _compiled_steady_direct_functions(
                static_scalar_fields=radiation_static_fields,
                static_array_fields=static_array_fields,
                dynamic_state=dynamic_state,
                dynamic_names=dynamic_names,
                grid=grid,
                config_flags=config_flags,
                callbacks=assembly.callbacks,
                snapshot_block_steps=1,
                eager_stage_names=_compiled_direct_eager_stage_names(),
                stage_scan_semantics=False,
                scan_compiled_calls=False,
            )
            if radiation_dynamic_state.array_names != dynamic_state.array_names:
                raise ValueError("radiation event state does not match the ordinary recurrent state")
            if tuple(radiation_static_arrays) != tuple(static_array_fields):
                raise ValueError("radiation event static arrays do not match the ordinary step")
            del radiation_dynamic_state, radiation_static_arrays
        compiled_block_executable = None
        compiled_snapshot_block_executable = None
        compiled_sampled_block_executable = None
        if probe_step < step_count and block_steps > 1 and use_snapshot_scan:
            _trajectory_progress(f"compiled snapshot block compile start block_steps={block_steps}")
            metric_start = perf_counter()
            compiled_snapshot_block_executable = compiled_snapshot_block.lower(
                dynamic_state,
                static_array_fields,
                jnp.asarray(block_steps, dtype=jnp.int32),
            ).compile()
            runtime_metrics["compiled_snapshot_block_compile_seconds"] += perf_counter() - metric_start
            _trajectory_progress(
                "compiled snapshot block compile done "
                f"seconds={runtime_metrics['compiled_snapshot_block_compile_seconds']:.3f}"
            )
        elif probe_step < step_count and block_steps > 1:
            _trajectory_progress(f"compiled sampled block compile start block_steps={block_steps}")
            metric_start = perf_counter()
            compiled_sampled_block_executable = compile_model_program(
                compiled_sampled_block,
                dynamic_state,
                static_array_fields,
                jnp.asarray(block_steps, dtype=jnp.int32),
                context={
                    "program": "sampled_block",
                    "block_steps": block_steps,
                    "dynamic_names": dynamic_names,
                    "static_fields": static_fields,
                    "grid": grid,
                    "config_flags": config_flags,
                    "callbacks": assembly.callbacks,
                },
                report=_trajectory_progress,
            )
            runtime_metrics["compiled_sampled_block_compile_seconds"] += perf_counter() - metric_start
            _trajectory_progress(
                "compiled sampled block compile done "
                f"seconds={runtime_metrics['compiled_sampled_block_compile_seconds']:.3f}"
            )
        step = probe_step + 1
        # Bound asynchronous submission to eight unchanged step calls. This lets
        # host dispatch overlap device work without retaining a long trajectory.
        host_batch_steps = 8 if (use_stage_families and not use_stage_scan
                                 and not use_segmented_column_physics and block_steps == 1) else 1
        runtime_metrics["host_step_batch_size"] = float(host_batch_steps)
        while step <= step_count:
            # Only the PyTree owns the live forecast during advancement.
            # Output/diagnostic dictionaries from the previous iteration can
            # retain derived arrays that are not part of recurrent state.
            fields = {}
            dynamic_fields = {}
            next_radiation = _next_radiation_step(step, step_count, config_flags)
            if use_compiled_radiation_events and next_radiation == step and step > startup_steps:
                if compiled_radiation_step is None:
                    raise RuntimeError("compiled radiation event step was not initialized")
                _trajectory_progress(f"compiled radiation event start step={step}")
                metric_start = perf_counter()
                dynamic_state = compiled_radiation_step(dynamic_state, static_array_fields)
                _block_until_ready_tree(dynamic_state)
                event_seconds = perf_counter() - metric_start
                runtime_metrics["compiled_radiation_event_execute_seconds"] += event_seconds
                runtime_metrics["compiled_radiation_event_chunks"] += 1.0
                runtime_metrics["compiled_radiation_event_model_steps"] += 1.0
                _trajectory_progress(
                    f"compiled radiation event done step={step} seconds={event_seconds:.3f} "
                    f"cumulative_seconds={runtime_metrics['compiled_radiation_event_execute_seconds']:.3f}"
                )
                dynamic_fields = model_state_to_dynamic_fields(dynamic_state, dynamic_names)
                fields = dict(static_fields)
                fields.update(static_array_fields)
                fields.update(dynamic_fields)
                elapsed = int(round(step * dt))
                _refresh_completed_block_fields(
                    fields,
                    elapsed_seconds=elapsed,
                    total_seconds=total_seconds,
                    snapshot_seconds=snapshot_seconds,
                    boundary_interval_seconds=assembly.case.config.boundary_interval_seconds,
                )
                _set_grid_elapsed_seconds(grid, elapsed)
                if _refresh_wrfbdy_fields_on_boundary_time(assembly, grid, fields, elapsed):
                    for name, value in fields.items():
                        if name in dynamic_fields:
                            dynamic_fields[name] = value
                        elif _is_wrfbdy_field_name(name):
                            if _is_dynamic_array_value(value):
                                static_array_fields[name] = value
                            else:
                                static_fields[name] = value
                    dynamic_state = _model_state_with_dynamic_fields(dynamic_state, dynamic_fields)
                if elapsed in snapshot_seconds:
                    record_snapshot(
                        _snapshot_from_fields(
                            fields,
                            assembly.case.config.start_time,
                            elapsed,
                            config_flags,
                            check_finite=False,
                        )
                    )
                step += 1
                continue
            if use_eager_radiation_fallback and next_radiation == step and step > startup_steps:
                dynamic_fields = model_state_to_dynamic_fields(dynamic_state, dynamic_names)
                fields = dict(static_fields)
                fields.update(static_array_fields)
                fields.update(dynamic_fields)
                fields["itimestep"] = int(step) - 1
                fields["_skip_radiation"] = False
                fields.pop("_physics_itimestep", None)
                _refresh_current_dry_aliases(fields)
                _refresh_derived_steady_fields(fields)
                _seed_steady_scratch_fields(fields)
                _seed_interstage_rk_scratch_fields(fields)
                _set_grid_elapsed_seconds(grid, int(round((int(step) - 1) * dt)))
                setattr(grid, "itimestep", int(step) - 1)
                if "dtbc" in fields:
                    try:
                        setattr(grid, "dtbc", float(np.asarray(jax.device_get(fields["dtbc"])).reshape(-1)[0]))
                    except (TypeError, ValueError):
                        pass
                _trajectory_progress(f"eager radiation fallback step start step={step}")
                metric_start = perf_counter()
                final_step = run_direct_step(
                    plan=plan,
                    grid=grid,
                    config_flags=config_flags,
                    fields=fields,
                    moist=fields["moist"],
                    scalar=fields.get("scalar"),
                )
                _block_until_ready_tree(final_step.fields)
                runtime_metrics["compiled_radiation_fallback_execute_seconds"] += perf_counter() - metric_start
                runtime_metrics["compiled_radiation_fallback_chunks"] += 1.0
                runtime_metrics["compiled_radiation_fallback_model_steps"] += 1.0
                _trajectory_progress(
                    "eager radiation fallback step done "
                    f"step={step} cumulative_seconds="
                    f"{runtime_metrics['compiled_radiation_fallback_execute_seconds']:.3f}"
                )
                fields = final_step.fields
                fields["_skip_radiation"] = True
                fields["_physics_itimestep"] = max(4, int(step) + 1)
                elapsed = int(round(step * dt))
                _set_grid_elapsed_seconds(grid, elapsed)
                _refresh_wrfbdy_fields_on_boundary_time(assembly, grid, fields, elapsed)
                fallback_partition = _partition_fields_for_compiled_step(
                    fields,
                    dynamic_names=dynamic_names,
                    prune_static_fields=_prune_refreshed_steady_static_fields,
                    deduplicate_buffers=not use_stage_families or use_stage_scan,
                )
                dynamic_state = fallback_partition.model_state(grid)
                static_array_fields = fallback_partition.static_array_fields
                static_fields = fallback_partition.static_scalar_fields
                runtime_metrics["steady_input_buffer_copies"] += float(fallback_partition.input_buffer_copies)
                runtime_metrics["steady_static_array_fields"] = float(len(static_array_fields))
                runtime_metrics["steady_static_array_bytes"] = float(fallback_partition.static_array_bytes)
                runtime_metrics["steady_static_scalar_fields"] = float(len(static_fields))
                del fallback_partition
                final_step = None
                if elapsed in snapshot_seconds:
                    record_snapshot(
                        _snapshot_from_fields(
                            fields,
                            assembly.case.config.start_time,
                            elapsed,
                            config_flags,
                            check_finite=False,
                        )
                    )
                step += 1
                continue

            next_snapshot = _next_snapshot_step(step, snapshot_seconds_steps, step_count)
            next_boundary = _next_boundary_reset_step(
                step,
                step_count,
                int(dt),
                int(assembly.case.config.boundary_interval_seconds),
            )
            next_radiation = _next_radiation_step(step, step_count, config_flags)
            if use_snapshot_scan:
                max_until_snapshot = step_count - step + 1
            else:
                max_until_snapshot = (next_snapshot - step + 1) if next_snapshot is not None else (step_count - step + 1)
            max_until_boundary = (next_boundary - step + 1) if next_boundary is not None else (step_count - step + 1)
            max_until_radiation = (
                1 if next_radiation == step else
                (next_radiation - step) if next_radiation is not None else (step_count - step + 1)
            )
            chunk_steps = min(max(block_steps, host_batch_steps), step_count - step + 1,
                              max_until_snapshot, max_until_boundary, max_until_radiation)
            # A singleton at an output/boundary must use the same compiled
            # arithmetic as a full block, with the remaining scan steps masked.
            used_compiled_block = (chunk_steps > 1 and host_batch_steps == 1) or any(
                executable is not None for executable in (
                    compiled_snapshot_block_executable,
                    compiled_sampled_block_executable,
                    compiled_block_executable,
                )
            )
            if not used_compiled_block:
                if not use_stage_families:
                    dynamic_fields = model_state_to_dynamic_fields(dynamic_state, dynamic_names)
                    dynamic_fields, copy_count = _deduplicate_input_buffers(dynamic_fields)
                    dynamic_state = _model_state_with_dynamic_fields(dynamic_state, dynamic_fields)
                    runtime_metrics["steady_input_buffer_copies"] += float(copy_count)
                metric_start = perf_counter()
                for _ in range(chunk_steps):
                    dynamic_state = compiled_step(dynamic_state, static_array_fields)
                _block_until_ready_tree(dynamic_state)
                runtime_metrics["compiled_step_execute_seconds"] += perf_counter() - metric_start
                runtime_metrics["compiled_step_chunks"] += float(chunk_steps)
            else:
                dynamic_fields = model_state_to_dynamic_fields(dynamic_state, dynamic_names)
                dynamic_fields, copy_count = _deduplicate_input_buffers(dynamic_fields)
                dynamic_state = _model_state_with_dynamic_fields(dynamic_state, dynamic_fields)
                runtime_metrics["steady_input_buffer_copies"] += float(copy_count)
                metric_start = perf_counter()
                sampled_snapshot_arrays = None
                if compiled_snapshot_block_executable is not None:
                    _trajectory_progress(f"compiled snapshot block execute start step={step} chunk_steps={chunk_steps}")
                    dynamic_state, snapshot_arrays = compiled_snapshot_block_executable(
                        dynamic_state,
                        static_array_fields,
                        jnp.asarray(chunk_steps, dtype=jnp.int32),
                    )
                    _block_until_ready_tree((dynamic_state, snapshot_arrays))
                    runtime_metrics["compiled_snapshot_block_execute_seconds"] += perf_counter() - metric_start
                    runtime_metrics["compiled_snapshot_block_chunks"] += 1.0
                    runtime_metrics["compiled_snapshot_block_model_steps"] += float(chunk_steps)
                    _trajectory_progress(
                        "compiled snapshot block execute done "
                        f"step={step} chunk_steps={chunk_steps} "
                        f"cumulative_seconds={runtime_metrics['compiled_snapshot_block_execute_seconds']:.3f}"
                    )
                    for offset in range(chunk_steps):
                        elapsed_for_snapshot = int(round((step + offset) * dt))
                        record_snapshot(
                            _snapshot_from_arrays(
                                arrays=(
                                    snapshot_arrays[0][offset],
                                    snapshot_arrays[1][offset],
                                    snapshot_arrays[2][offset],
                                    snapshot_arrays[3][offset],
                                    {name: values[offset] for name, values in snapshot_arrays[4].items()},
                                ),
                                start_timestamp=assembly.case.config.start_time,
                                elapsed_seconds=elapsed_for_snapshot,
                            )
                        )
                elif compiled_sampled_block_executable is not None:
                    _trajectory_progress(
                        f"compiled sampled block execute start step={step} chunk_steps={chunk_steps}"
                    )
                    dynamic_state, sampled_snapshot_arrays = compiled_sampled_block_executable(
                        dynamic_state,
                        static_array_fields,
                        jnp.asarray(chunk_steps, dtype=jnp.int32),
                    )
                    _block_until_ready_tree((dynamic_state, sampled_snapshot_arrays))
                    runtime_metrics["compiled_sampled_block_execute_seconds"] += perf_counter() - metric_start
                    runtime_metrics["compiled_sampled_block_chunks"] += 1.0
                    runtime_metrics["compiled_sampled_block_model_steps"] += float(chunk_steps)
                    _trajectory_progress(
                        f"compiled sampled block execute done step={step} chunk_steps={chunk_steps} "
                        f"seconds={perf_counter() - metric_start:.3f} "
                        f"elapsed_seconds={int(round((step + chunk_steps - 1) * dt))}"
                    )
                elif compiled_block_executable is not None:
                    dynamic_state = compiled_block_executable(dynamic_state, static_array_fields, jnp.asarray(chunk_steps, dtype=jnp.int32))
                    _block_until_ready_tree(dynamic_state)
                    runtime_metrics["compiled_block_execute_seconds"] += perf_counter() - metric_start
                    runtime_metrics["compiled_block_chunks"] += 1.0
                    runtime_metrics["compiled_block_model_steps"] += float(chunk_steps)
                else:
                    dynamic_state = compiled_block(dynamic_state, static_array_fields, jnp.asarray(chunk_steps, dtype=jnp.int32))
                    _block_until_ready_tree(dynamic_state)
                    runtime_metrics["compiled_block_execute_seconds"] += perf_counter() - metric_start
                    runtime_metrics["compiled_block_chunks"] += 1.0
                    runtime_metrics["compiled_block_model_steps"] += float(chunk_steps)
            step += chunk_steps - 1
            dynamic_fields = model_state_to_dynamic_fields(dynamic_state, dynamic_names)
            fields = dict(static_fields)
            fields.update(static_array_fields)
            fields.update(dynamic_fields)
            elapsed = int(round(step * dt))
            _refresh_completed_block_fields(
                fields,
                elapsed_seconds=elapsed,
                total_seconds=total_seconds,
                snapshot_seconds=snapshot_seconds,
                boundary_interval_seconds=assembly.case.config.boundary_interval_seconds,
            )
            _set_grid_elapsed_seconds(grid, elapsed)
            if _refresh_wrfbdy_fields_on_boundary_time(assembly, grid, fields, elapsed):
                for name, value in fields.items():
                    if name in dynamic_fields:
                        dynamic_fields[name] = value
                    elif _is_wrfbdy_field_name(name):
                        if _is_dynamic_array_value(value):
                            static_array_fields[name] = value
                        else:
                            static_fields[name] = value
                dynamic_state = _model_state_with_dynamic_fields(dynamic_state, dynamic_fields)
            if (
                elapsed in snapshot_seconds
                and used_compiled_block
                and compiled_sampled_block_executable is not None
                and sampled_snapshot_arrays is not None
            ):
                record_snapshot(
                    _snapshot_from_arrays(
                        arrays=sampled_snapshot_arrays,
                        start_timestamp=assembly.case.config.start_time,
                        elapsed_seconds=elapsed,
                    )
                )
            elif elapsed in snapshot_seconds and not (use_snapshot_scan and used_compiled_block):
                record_snapshot(
                    _snapshot_from_fields(
                        fields,
                        assembly.case.config.start_time,
                        elapsed,
                        config_flags,
                        check_finite=False,
                    )
                )
            step += 1

        final_step = SolveEmResult(
            grid=grid,
            config_flags=config_flags,
            fields=fields,
            ijk=plan.ijk,
            subgrid_ijk=plan.subgrid_ijk,
            rk_order=plan.rk_order,
            num_sound_steps=plan.num_sound_steps,
            dts=plan.dts,
            adapt_step_flag=False,
            restart_flag=False,
            diag_flag=False,
            curr_secs=float(total_seconds),
            curr_secs2=float(total_seconds),
            max_vert_cfl_tmp=(0.0,),
            max_horiz_cfl_tmp=(0.0,),
            events=(),
        )

    if final_step is None:
        raise ValueError("trajectory requested zero steps")
    return TrajectoryResult(assembly=assembly, final_step=final_step, snapshots=tuple(snapshots), runtime_metrics=runtime_metrics)


def compare_snapshots_to_wrfout(
    snapshots: tuple[PrognosticSnapshot, ...],
    wrfout_path: str | Path,
) -> tuple[FieldComparison, ...]:
    """Compare trajectory snapshots against a WRF history output file."""

    by_time = {snapshot.timestamp: snapshot for snapshot in snapshots}
    comparisons: list[FieldComparison] = []
    with xr.open_dataset(wrfout_path, engine="netcdf4", decode_times=False) as dataset:
        times = _decode_wrf_times(dataset["Times"].values)
        time_index = {timestamp: index for index, timestamp in enumerate(times)}
        for timestamp, snapshot in by_time.items():
            if timestamp not in time_index:
                raise KeyError(f"{timestamp} not found in {wrfout_path}")
            index = time_index[timestamp]
            hour = int(snapshot.elapsed_seconds // 3600)
            actuals = {
                "U": snapshot.u,
                "V": snapshot.v,
                "T": snapshot.t,
                "QVAPOR": snapshot.qvapor,
            }
            actuals.update(_extras_with_derived_tendencies(snapshot))
            for field, actual in actuals.items():
                if field not in dataset:
                    continue
                actual_np = np.asarray(actual, dtype=np.float64)
                expected = np.asarray(dataset[field].isel(Time=index).values, dtype=np.float64)
                diff = actual_np - expected
                rmse_value = _rmse_arrays(actual_np, expected)
                comparisons.append(
                    FieldComparison(
                        hour=hour,
                        elapsed_seconds=int(snapshot.elapsed_seconds),
                        timestamp=timestamp,
                        field=field,
                        rmse=rmse_value,
                        nrmse_percent=_nrmse_percent_arrays(actual_np, expected),
                        max_abs_error=float(np.max(np.abs(diff))),
                        units=_comparison_units(field, dataset[field]),
                        finite=bool(np.all(np.isfinite(actual_np))),
                    )
                )
    return tuple(comparisons)


def _comparison_units(field: str, reference_var: xr.DataArray) -> str:
    units = reference_var.attrs.get("units")
    if units is None:
        metadata = _extra_field_metadata(field) or FIELD_METADATA.get(field, {})
        units = metadata.get("units", "")
    return str(units)


def write_snapshots_netcdf(
    snapshots: tuple[PrognosticSnapshot, ...],
    path: str | Path,
    *,
    reference_path: str | Path | None = None,
) -> None:
    """Write WRF-JAX prognostic snapshots to a Panoply-readable WRF-style NetCDF file."""

    if not snapshots:
        raise ValueError("no snapshots to write")
    timestamps = [snapshot.timestamp for snapshot in snapshots]
    elapsed_values = np.asarray([snapshot.elapsed_seconds for snapshot in snapshots], dtype=np.int32)
    reference: xr.Dataset | None = None
    if reference_path is not None:
        reference = xr.open_dataset(reference_path, engine="netcdf4", decode_times=False)
    try:
        start_timestamp = _trajectory_start_timestamp(snapshots[0])
        data_vars: dict[str, tuple[tuple[str, ...], Any, dict[str, Any]]] = {
            "Times": (("Time",), _times_string_array(timestamps), {}),
            "XTIME": (("Time",), elapsed_values.astype(np.float32) / 60.0, _default_xtime_attrs(start_timestamp)),
            "elapsed_seconds": (("Time",), elapsed_values, {"description": "elapsed model seconds since trajectory start", "units": "s"}),
            "U": (
                ("Time", "bottom_top", "south_north", "west_east_stag"),
                np.stack([np.asarray(s.u, dtype=np.float32) for s in snapshots]),
                FIELD_METADATA["U"],
            ),
            "V": (
                ("Time", "bottom_top", "south_north_stag", "west_east"),
                np.stack([np.asarray(s.v, dtype=np.float32) for s in snapshots]),
                FIELD_METADATA["V"],
            ),
            "T": (
                ("Time", "bottom_top", "south_north", "west_east"),
                np.stack([np.asarray(s.t, dtype=np.float32) for s in snapshots]),
                FIELD_METADATA["T"],
            ),
            "QVAPOR": (
                ("Time", "bottom_top", "south_north", "west_east"),
                np.stack([np.asarray(s.qvapor, dtype=np.float32) for s in snapshots]),
                FIELD_METADATA["QVAPOR"],
            ),
        }
        snapshot_extras = [_extras_with_derived_tendencies(snapshot) for snapshot in snapshots]
        extra_names = sorted(set().union(*(set(extra.keys()) for extra in snapshot_extras)))
        for name in extra_names:
            if not all(name in extra for extra in snapshot_extras):
                continue
            stacked = np.stack([np.asarray(extra[name], dtype=np.float32) for extra in snapshot_extras])
            if reference is not None and name in reference:
                dims = tuple(str(dim) for dim in reference[name].dims)
                attrs = _extra_field_metadata(name)
                attrs.update(_copy_reference_var_attrs(reference, name))
                if "coordinates" not in attrs:
                    if dims == ("Time", "bottom_top", "south_north", "west_east"):
                        attrs["coordinates"] = "XLONG XLAT XTIME"
                    elif dims == ("Time", "bottom_top_stag", "south_north", "west_east"):
                        attrs["coordinates"] = "XLONG XLAT XTIME"
                    elif dims == ("Time", "bottom_top", "south_north", "west_east_stag"):
                        attrs["coordinates"] = "XLONG_U XLAT_U XTIME"
                    elif dims == ("Time", "bottom_top", "south_north_stag", "west_east"):
                        attrs["coordinates"] = "XLONG_V XLAT_V XTIME"
                    elif dims == ("Time", "south_north", "west_east"):
                        attrs["coordinates"] = "XLONG XLAT XTIME"
                    elif dims == ("Time", "south_north", "west_east_stag"):
                        attrs["coordinates"] = "XLONG_U XLAT_U XTIME"
                    elif dims == ("Time", "south_north_stag", "west_east"):
                        attrs["coordinates"] = "XLONG_V XLAT_V XTIME"
            elif stacked.ndim == 4:
                vertical_dim = "bottom_top_stag" if name in _STAGGERED_EXTRA_FIELD_SOURCES else "bottom_top"
                dims = ("Time", vertical_dim, "south_north", "west_east")
                attrs = _default_extra_attrs(name, "XYZ")
            elif stacked.ndim == 3:
                dims = ("Time", "south_north", "west_east")
                attrs = _default_extra_attrs(name, "XY ")
            else:
                raise ValueError(f"unsupported extra snapshot field {name} with stacked shape {stacked.shape}")
            data_vars[name] = (dims, stacked, attrs)
        if reference is not None:
            for name, dims in (
                ("XLAT", ("Time", "south_north", "west_east")),
                ("XLONG", ("Time", "south_north", "west_east")),
                ("XLAT_U", ("Time", "south_north", "west_east_stag")),
                ("XLONG_U", ("Time", "south_north", "west_east_stag")),
                ("XLAT_V", ("Time", "south_north_stag", "west_east")),
                ("XLONG_V", ("Time", "south_north_stag", "west_east")),
                ("ZNU", ("Time", "bottom_top")),
                ("ZNW", ("Time", "bottom_top_stag")),
            ):
                if name in reference:
                    data_vars[name] = (dims, _static_reference_coord(reference, name, len(snapshots)), _coord_attrs(reference, name))
        dataset = xr.Dataset(data_vars=data_vars, attrs=_wrf_global_attrs(reference))
        for name in PROGNOSTIC_OUTPUTS:
            dataset[name].attrs.update(FIELD_METADATA[name])
        if "XTIME" in dataset and reference is not None and "XTIME" in reference:
            dataset["XTIME"].attrs.update(_copy_reference_var_attrs(reference, "XTIME"))
    finally:
        if reference is not None:
            reference.close()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    encoding: dict[str, dict[str, Any]] = {
        "Times": {"dtype": "S1", "char_dim_name": "DateStrLen", "_FillValue": None},
    }
    for name in dataset.data_vars:
        if name != "Times":
            encoding[name] = {"_FillValue": None}
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding, unlimited_dims=["Time"])


def write_comparison_markdown(comparisons: tuple[FieldComparison, ...], path: str | Path) -> None:
    """Write a compact Markdown comparison table."""

    lines = [
        "| hour | elapsed_seconds | timestamp | field | nRMSE% | RMSE | units | max abs error | finite |",
        "|---:|---:|---|---|---:|---:|---|---:|---|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item.hour} | {item.elapsed_seconds} | {item.timestamp} | {item.field} | "
            f"{item.nrmse_percent:.8g} | {item.rmse:.8g} | {item.units} | {item.max_abs_error:.8g} | {item.finite} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def _summary_units(units: str) -> str:
    return str(units).replace(",", ";") if units else ""


def write_nrmse_summary(comparisons: tuple[FieldComparison, ...], path: str | Path) -> None:
    """Write final-time nRMSE% summary for the required milestone fields."""

    if not comparisons:
        raise ValueError("no comparisons to summarize")
    final_elapsed = max(item.elapsed_seconds for item in comparisons)
    final_items = {item.field: item for item in comparisons if item.elapsed_seconds == final_elapsed}
    final_timestamp = next(item.timestamp for item in comparisons if item.elapsed_seconds == final_elapsed)
    groups = (
        ("prognostic", PROGNOSTIC_OUTPUTS),
        ("microphysics", MICROPHYSICS_KEY_OUTPUTS),
        ("cumulus", CUMULUS_KEY_OUTPUTS),
        ("radiation", RADIATION_KEY_OUTPUTS),
    )
    present_key_items = [final_items[name] for name in KEY_OUTPUTS if name in final_items]
    not_compared = [name for name in KEY_OUTPUTS if name not in final_items]
    lines = [
        f"elapsed_seconds={final_elapsed}",
        f"timestamp={final_timestamp}",
    ]
    if present_key_items:
        max_item = max(present_key_items, key=lambda item: item.nrmse_percent)
        lines.extend(
            [
                f"max_key_nRMSE_percent={max_item.nrmse_percent:.8g}",
                f"max_key_nRMSE_field={max_item.field}",
            ]
        )
    if not_compared:
        lines.append(f"not_compared_key_fields={','.join(not_compared)}")
    lines.append("")
    for title, fields in groups:
        lines.append(f"[{title}]")
        for field in fields:
            item = final_items.get(field)
            if item is None:
                lines.append(f"{field}=not_compared")
            else:
                lines.append(
                    f"{field}=nRMSE%:{item.nrmse_percent:.8g},RMSE:{item.rmse:.8g},"
                    f"units:{_summary_units(item.units)},max_abs:{item.max_abs_error:.8g},finite:{item.finite}"
                )
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="ascii")


def write_goal_check(comparisons: tuple[FieldComparison, ...], path: str | Path, reference_path: str | Path | None) -> None:
    """Write final-time pass/fail status against ``goal.md`` thresholds."""

    if not comparisons:
        raise ValueError("no comparisons to check")
    final_elapsed = max(item.elapsed_seconds for item in comparisons)
    final_items = {item.field: item for item in comparisons if item.elapsed_seconds == final_elapsed}
    checked_fields = tuple(name for name in WRF_REPRO_6H_NRMSE_PERCENT if name in final_items)
    pass_fields: list[str] = []
    fail_fields: list[str] = []
    lines = [
        f"goal_reference={DEFAULT_GOAL_REFERENCE}",
        f"wrf_reference={reference_path if reference_path is not None else ''}",
        f"elapsed_seconds={final_elapsed}",
        "threshold_rule=one sixth of the 6h WRF 48c-vs-64c nRMSE% table",
        "zero_threshold_rule=PASS only when nRMSE% is exactly zero",
        "",
        "field,nRMSE_percent,threshold_percent,status",
    ]
    for name in checked_fields:
        item = final_items[name]
        base_threshold = WRF_REPRO_6H_NRMSE_PERCENT[name]
        threshold = 0.0 if base_threshold == 0.0 else base_threshold / 6.0
        passed = item.nrmse_percent == 0.0 if threshold == 0.0 else item.nrmse_percent <= threshold
        status = "PASS" if passed else "FAIL"
        if passed:
            pass_fields.append(name)
        else:
            fail_fields.append(name)
        lines.append(f"{name},{item.nrmse_percent:.8g},{threshold:.8g},{status}")

    not_compared = [name for name in KEY_OUTPUTS if name not in final_items]
    if not_compared:
        lines.extend(
            [
                "",
                f"not_compared_key_fields={','.join(not_compared)}",
                "not_compared_reason=The em_real_5 WRF reference file does not contain matching NCA/HTOP/HBOT fields.",
            ]
        )
    if checked_fields:
        dominant = max((final_items[name] for name in checked_fields), key=lambda item: item.nrmse_percent)
        lines.extend(
            [
                "",
                "summary:",
                f"  pass_fields={','.join(pass_fields) if pass_fields else 'none'}",
                f"  fail_fields={','.join(fail_fields) if fail_fields else 'none'}",
                f"  dominant_remaining_key_error={dominant.field} {dominant.nrmse_percent:.8g}%",
            ]
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def write_diagnostic_fields_summary(
    snapshots: tuple[PrognosticSnapshot, ...],
    comparisons: tuple[FieldComparison, ...] | None,
    path: str | Path,
) -> None:
    """Write a machine-readable inventory of fields written to the milestone NetCDF."""

    if not snapshots:
        raise ValueError("no snapshots to summarize")
    final = snapshots[-1]
    fields = _snapshot_output_fields(final)
    final_items: dict[str, FieldComparison] = {}
    if comparisons:
        final_elapsed = max(item.elapsed_seconds for item in comparisons)
        final_items = {item.field: item for item in comparisons if item.elapsed_seconds == final_elapsed}

    lines = [
        f"elapsed_seconds={final.elapsed_seconds}",
        f"timestamp={final.timestamp}",
        "field,status,shape,finite,min,max,nRMSE_percent,RMSE_original_units,max_abs_error,units,description",
    ]
    for name in sorted(fields):
        values = np.asarray(fields[name], dtype=np.float32)
        finite = bool(np.all(np.isfinite(values)))
        if values.size:
            min_value = float(np.nanmin(values))
            max_value = float(np.nanmax(values))
        else:
            min_value = float("nan")
            max_value = float("nan")
        item = final_items.get(name)
        metadata = _extra_field_metadata(name) or FIELD_METADATA.get(name, {})
        units = str(metadata.get("units", "")).replace(",", ";")
        description = str(metadata.get("description", name)).replace(",", ";")
        shape = "x".join(str(dim) for dim in values.shape)
        if item is None:
            status = "not_compared"
            nrmse = ""
            rmse_value = ""
            max_abs = ""
        else:
            status = "compared"
            nrmse = f"{item.nrmse_percent:.8g}"
            rmse_value = f"{item.rmse:.8g}"
            max_abs = f"{item.max_abs_error:.8g}"
        lines.append(
            f"{name},{status},{shape},{finite},{min_value:.8g},{max_value:.8g},{nrmse},{rmse_value},{max_abs},{units},{description}"
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def write_nonfinite_summary(
    snapshots: tuple[PrognosticSnapshot, ...],
    comparisons: tuple[FieldComparison, ...] | None,
    path: str | Path,
) -> None:
    """Write non-finite value diagnostics without blocking milestone output."""

    lines = [
        "status=none",
        "source=snapshots",
        "elapsed_seconds,timestamp,field,count,shape,first_index,first_value",
    ]
    found = False
    for snapshot in snapshots:
        for name, values in sorted(_snapshot_output_fields(snapshot).items()):
            array = np.asarray(values)
            finite = np.isfinite(array)
            if bool(np.all(finite)):
                continue
            found = True
            bad = np.argwhere(~finite)
            first_index = tuple(int(item) for item in bad[0]) if bad.size else ()
            first_value = array[first_index] if first_index else np.asarray(array).reshape(-1)[0]
            shape = "x".join(str(int(dim)) for dim in array.shape)
            index_text = "x".join(str(item) for item in first_index)
            lines.append(
                f"{snapshot.elapsed_seconds},{snapshot.timestamp},{name},{int(bad.shape[0])},"
                f"{shape},{index_text},{first_value}"
            )
    if comparisons is not None:
        lines.extend(["", "source=comparisons", "elapsed_seconds,timestamp,field,finite"])
        for item in comparisons:
            if item.finite:
                continue
            found = True
            lines.append(f"{item.elapsed_seconds},{item.timestamp},{item.field},{item.finite}")
    if found:
        lines[0] = "status=nonfinite_found"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def _wrfjax_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _current_git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "detached"
    branch = result.stdout.strip()
    return branch or "detached"


def _safe_branch_path(branch: str) -> Path:
    parts: list[str] = []
    for raw_part in branch.split("/"):
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in raw_part.strip())
        if safe:
            parts.append(safe)
    return Path(*parts) if parts else Path("detached")


def branch_output_root(root: str | Path = DEFAULT_OUTPUT_ROOT, branch: str | None = None) -> Path:
    """Return the branch-scoped output root for milestone trajectory files."""

    return Path(root) / _safe_branch_path(_current_git_branch() if branch is None else branch)


def new_wrfjax_output_dir(root: str | Path | None = None, timestamp: str | None = None) -> Path:
    """Create a unique ``wrfjax-<timestamp>`` milestone output directory."""

    stamp = _wrfjax_timestamp() if timestamp is None else str(timestamp)
    output_root = branch_output_root() if root is None else Path(root)
    base = output_root / f"wrfjax-{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}-{suffix:02d}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def write_wrfjax_milestone_outputs(
    snapshots: tuple[PrognosticSnapshot, ...],
    output_dir: str | Path | None = None,
    comparisons: tuple[FieldComparison, ...] | None = None,
    reference_path: str | Path | None = None,
) -> MilestoneOutput:
    """Write the required milestone NetCDF trajectory and optional error report."""

    if not snapshots:
        raise ValueError("no snapshots to write")
    last_elapsed = int(snapshots[-1].elapsed_seconds)
    requested_dir = Path(output_dir) if output_dir is not None else None
    timestamped_output = requested_dir is None or requested_dir.name.startswith("wrfjax-20")
    if timestamped_output and last_elapsed < 3600:
        raise ValueError(
            f"timestamped WRF-JAX outputs must include at least 3600s of model time; got {last_elapsed}s"
        )
    out_dir = new_wrfjax_output_dir() if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    netcdf_path = out_dir / "wrfjax_uvtq_steps.nc"
    comparison_path = out_dir / "comparison.md" if comparisons is not None else None
    nrmse_summary_path = out_dir / "nrmse_summary.txt" if comparisons is not None else None
    diagnostic_fields_path = out_dir / "diagnostic_fields.txt"
    goal_check_path = out_dir / "goal_check.txt" if comparisons is not None else None
    nonfinite_summary_path = out_dir / "nonfinite_summary.txt"
    manifest_path = out_dir / "manifest.txt"

    write_snapshots_netcdf(snapshots, netcdf_path, reference_path=reference_path)
    if comparisons is not None and comparison_path is not None:
        write_comparison_markdown(comparisons, comparison_path)
    if comparisons is not None and nrmse_summary_path is not None:
        write_nrmse_summary(comparisons, nrmse_summary_path)
    if comparisons is not None and goal_check_path is not None:
        write_goal_check(comparisons, goal_check_path, reference_path)
    write_diagnostic_fields_summary(snapshots, comparisons, diagnostic_fields_path)
    write_nonfinite_summary(snapshots, comparisons, nonfinite_summary_path)

    first_timestamp = snapshots[0].timestamp if snapshots else ""
    last_timestamp = snapshots[-1].timestamp if snapshots else ""
    lines = [
        f"created_at={datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}",
        f"snapshot_count={len(snapshots)}",
        f"first_timestamp={first_timestamp}",
        f"last_timestamp={last_timestamp}",
        f"netcdf={netcdf_path.name}",
        f"minimum_elapsed_seconds={snapshots[-1].elapsed_seconds if snapshots else 0}",
        f"panoply_geo2d_metadata={'true' if reference_path is not None else 'false'}",
    ]
    if reference_path is not None:
        lines.append(f"reference_path={reference_path}")
    if comparison_path is not None:
        lines.append(f"comparison={comparison_path.name}")
    if nrmse_summary_path is not None:
        lines.append(f"nrmse_summary={nrmse_summary_path.name}")
    if goal_check_path is not None:
        lines.append(f"goal_check={goal_check_path.name}")
    lines.append(f"nonfinite_summary={nonfinite_summary_path.name}")
    lines.append(f"diagnostic_fields={diagnostic_fields_path.name}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return MilestoneOutput(
        output_dir=out_dir,
        netcdf_path=netcdf_path,
        comparison_path=comparison_path,
        manifest_path=manifest_path,
        nrmse_summary_path=nrmse_summary_path,
        diagnostic_fields_path=diagnostic_fields_path,
        goal_check_path=goal_check_path,
        nonfinite_summary_path=nonfinite_summary_path,
    )


__all__ = [
    "assert_snapshots_finite",
    "FieldComparison",
    "MilestoneOutput",
    "PROGNOSTIC_OUTPUTS",
    "MICROPHYSICS_KEY_OUTPUTS",
    "CUMULUS_KEY_OUTPUTS",
    "RADIATION_KEY_OUTPUTS",
    "KEY_OUTPUTS",
    "EXTRA_FIELD_METADATA",
    "PrognosticSnapshot",
    "TrajectoryResult",
    "assert_snapshot_finite",
    "branch_output_root",
    "compare_snapshots_to_wrfout",
    "materialize_snapshots",
    "nrmse_percent",
    "new_wrfjax_output_dir",
    "run_strict_trajectory",
    "write_comparison_markdown",
    "write_diagnostic_fields_summary",
    "write_snapshots_netcdf",
    "write_wrfjax_milestone_outputs",
]
