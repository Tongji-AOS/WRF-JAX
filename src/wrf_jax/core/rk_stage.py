"""Run existing RK stages with either global or tile-local storage bounds."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from wrf_jax.core.direct_step import (
    _initialize_direct_step_stage_context,
    _run_final_boundary_stage_family,
    _run_post_rk_physics_stage_group,
    _run_rk_step_stage_family,
)


POST_RK_TILE_GROUPS = (
    ("post_advance_ppt", "post_phy_prep_part2", "post_moist_prep"),
    ("post_microphysics",),
    ("post_moist_finish", "post_calc_p_rho_phi_final", "post_physical_bc"),
    ("final_boundary",),
)


def _initialize_field_context(fields, *, plan, grid, config_flags, overlap_margin=None):
    state = dict(fields)
    local_plan = plan
    if (int(state["jms"]), int(state["jme"])) != (plan.ijk[8], plan.ijk[9]):
        if overlap_margin is not None:
            margin = int(overlap_margin)
            state["jts"] = state["jps"] = max(plan.ijk[2], int(state["jms"]) + margin)
            state["jte"] = state["jpe"] = min(plan.ijk[3], int(state["jme"]) - margin)
        bounds = list(plan.ijk)
        for base in (6, 18, 30):
            bounds[base + 2:base + 4] = (state["jms"], state["jme"])
            bounds[base + 8:base + 10] = (state["jps"], state["jpe"])
        local_plan = replace(plan, ijk=tuple(bounds), tile_bounds=((
            state["its"], state["ite"], state["jts"], state["jte"],
        ),))
        # X-face forcing uses row as its first dimension, unlike model fields.
        # The unstaggered forcing can have one fewer row than model storage.
        for name, value in tuple(state.items()):
            if (name.endswith(("_bxs", "_bxe", "_btxs", "_btxe"))
                    and hasattr(value, "shape") and value.ndim > 0):
                rows = jnp.clip(jnp.arange(state["jms"] - plan.ijk[8],
                                          state["jme"] - plan.ijk[8] + 1),
                                0, value.shape[0] - 1)
                state[name] = jnp.take(value, rows, axis=0)
    state, context = _initialize_direct_step_stage_context(
        plan=local_plan, grid=grid, config_flags=config_flags,
        fields=state, advance_clock=False,
    )
    return state, context, local_plan


def run_rk_field_stages(fields, *, plan, grid, config_flags, callbacks,
                        rk_step, stage_names, refresh_fields, overlap_margin=None):
    state, context, local_plan = _initialize_field_context(
        fields, plan=plan, grid=grid, config_flags=config_flags,
        overlap_margin=overlap_margin,
    )
    produced = {}
    for name in stage_names:
        refresh_fields(state)
        before = dict(state)
        _run_rk_step_stage_family(
            state=state, callbacks=callbacks, plan=local_plan,
            grid=grid, config_flags=config_flags,
            boundary_config=context.boundary_config,
            boundary_kwargs_fn=context.boundary_kwargs,
            physical_bc_kwargs_fn=context.physical_bc_kwargs,
            num_3d_m=context.num_3d_m, num_3d_s=context.num_3d_s,
            specified=context.specified, constant_bc=context.constant_bc,
            specified_or_nested=context.specified_or_nested,
            non_hydrostatic=context.non_hydrostatic,
            fused_boundary_groups=context.fused_boundary_groups,
            fused_small_boundaries=context.fused_small_boundaries,
            spec_zone=context.spec_zone, f_flux=context.f_flux,
            restart_flag=context.restart_flag, curr_secs=context.curr_secs,
            adapt_step_flag=context.adapt_step_flag,
            max_vert_cfl_tmp=context.max_vert_cfl_tmp,
            max_horiz_cfl_tmp=context.max_horiz_cfl_tmp,
            rk_steps=(rk_step,), initialize=False, stage_names=(name,),
        )
        updates = {key: value for key, value in state.items()
                   if hasattr(value, "shape") and value is not before.get(key)}
        if len(stage_names) > 1:
            updates = jax.lax.optimization_barrier(updates)
            state.update(updates)
        produced.update(updates)
    return produced


def run_post_rk_field_stages(fields, *, plan, grid, config_flags, callbacks,
                            stage_names, refresh_fields):
    state, context, local_plan = _initialize_field_context(
        fields, plan=plan, grid=grid, config_flags=config_flags,
        overlap_margin=0 if "final_boundary" in stage_names else None,
    )
    produced = {}
    for name in stage_names:
        refresh_fields(state)
        before = dict(state)
        if name == "final_boundary":
            _run_final_boundary_stage_family(
                state=state, callbacks=callbacks, plan=local_plan, grid=grid,
                config_flags=config_flags, boundary_config=context.boundary_config,
                specified_or_nested=context.specified_or_nested,
                fused_boundary_groups=context.fused_boundary_groups,
                spec_bdy_width=context.spec_bdy_width, spec_zone=context.spec_zone,
                num_3d_m=context.num_3d_m, curr_secs=context.curr_secs,
                curr_secs2=context.curr_secs2, diag_flag=context.diag_flag,
            )
        else:
            _run_post_rk_physics_stage_group(
                state=state, callbacks=callbacks, plan=local_plan, grid=grid,
                config_flags=config_flags, num_3d_m=context.num_3d_m,
                num_3d_s=context.num_3d_s, ids=context.ids, ide=context.ide,
                jds=context.jds, jde=context.jde, stage_names=(name,),
            )
        updates = {key: value for key, value in state.items()
                   if hasattr(value, "shape") and value is not before.get(key)}
        if len(stage_names) > 1:
            updates = jax.lax.optimization_barrier(updates)
            state.update(updates)
        produced.update(updates)
    return produced
