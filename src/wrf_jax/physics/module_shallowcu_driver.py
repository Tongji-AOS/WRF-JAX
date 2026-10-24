"""Translation support for WRF ``module_shallowcu_driver``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


SHCU_PHYSICS_NONE = 0


@dataclass(frozen=True)
class ShallowCuDriverResult:
    """Outputs carried by the translated shallow-cumulus driver."""

    pratesh: Any | None = None
    rainsh: Any | None = None
    nca: Any | None = None
    htop: Any | None = None
    hbot: Any | None = None
    carried_fields: Mapping[str, Any] = field(default_factory=dict)
    ran_parameterization: bool = False
    external_calls: tuple[str, ...] = ()


def shallowcu_driver(
    *,
    shcu_physics: int,
    pratesh: Any | None = None,
    rainsh: Any | None = None,
    nca: Any | None = None,
    htop: Any | None = None,
    hbot: Any | None = None,
    **fields: Any,
) -> ShallowCuDriverResult:
    """Run the translated WRF shallow-cumulus driver for supported branches.

    The configured WRF case has ``shcu_physics = 0``. WRF returns before any
    scheme-specific updates in that branch, so all carried fields are preserved.
    """

    if int(shcu_physics) == SHCU_PHYSICS_NONE:
        return ShallowCuDriverResult(
            pratesh=pratesh,
            rainsh=rainsh,
            nca=nca,
            htop=htop,
            hbot=hbot,
            carried_fields=dict(fields),
            ran_parameterization=False,
        )

    raise NotImplementedError(
        "shallowcu_driver currently implements only the WRF shcu_physics=0 branch; "
        f"active shcu_physics={int(shcu_physics)} requires a scheme-specific translation"
    )


__all__ = ["SHCU_PHYSICS_NONE", "ShallowCuDriverResult", "shallowcu_driver"]
