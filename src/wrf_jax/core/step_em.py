"""Typed one-step bridge for the production WRF-JAX core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from wrf_jax.core.conversion import apply_time_state_to_grid, fields_to_model_state, model_state_to_fields
from wrf_jax.core.direct_step import DirectStepPlan, build_direct_step_plan, run_direct_step
from wrf_jax.core.state import ModelState
from wrf_jax.dyn_em.solve_em import SolveEmResult


@dataclass(frozen=True)
class StepEmCoreContext:
    """Static and outer-shell data needed by the typed one-step core.

    The context still holds the strict runtime template while the production
    core is being split out.  The hot state is explicit in ``ModelState`` so the
    next milestone can remove dict mutation one stage family at a time.
    """

    plan: DirectStepPlan
    grid: Any
    config_flags: Any
    fields_template: Mapping[str, Any]
    dynamic_only: bool = True


@dataclass(frozen=True)
class StepEmCoreResult:
    """Result of one typed WRF-JAX em step."""

    state: ModelState
    fields: Mapping[str, Any]
    solve_result: SolveEmResult


def build_step_em_core_context(
    *,
    grid: Any,
    config_flags: Any,
    fields_template: Mapping[str, Any],
    callbacks: Mapping[str, Any],
    num_moist: int = 2,
    num_scalar: int = 0,
    dynamic_only: bool = True,
) -> StepEmCoreContext:
    """Build the static context for ``step_em_core_once``."""

    plan = build_direct_step_plan(
        grid=grid,
        config_flags=config_flags,
        callbacks=callbacks,
        num_moist=num_moist,
        num_scalar=num_scalar,
    )
    return StepEmCoreContext(
        plan=plan,
        grid=grid,
        config_flags=config_flags,
        fields_template=dict(fields_template),
        dynamic_only=bool(dynamic_only),
    )


def initial_step_em_state(context: StepEmCoreContext) -> ModelState:
    """Create the initial typed state from the context field template."""

    return fields_to_model_state(context.fields_template, context.grid, dynamic_only=context.dynamic_only)


def step_em_core_once(state: ModelState, context: StepEmCoreContext) -> StepEmCoreResult:
    """Run one em_real step with explicit ``ModelState`` input and output.

    This function preserves the current direct numerical path.  Its purpose is
    to make the production state boundary explicit before the stage families
    are lifted out of the strict mutable dictionary.
    """

    apply_time_state_to_grid(state.time, context.grid)
    fields = model_state_to_fields(state, context.fields_template)
    result = run_direct_step(
        plan=context.plan,
        grid=context.grid,
        config_flags=context.config_flags,
        fields=fields,
        moist=fields["moist"],
        scalar=fields.get("scalar"),
    )
    next_state = fields_to_model_state(result.fields, result.grid, dynamic_only=context.dynamic_only)
    return StepEmCoreResult(state=next_state, fields=result.fields, solve_result=result)


__all__ = [
    "StepEmCoreContext",
    "StepEmCoreResult",
    "build_step_em_core_context",
    "initial_step_em_state",
    "step_em_core_once",
]
