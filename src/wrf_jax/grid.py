"""Static grid metadata for the configured single-domain WRF case."""

from __future__ import annotations

from dataclasses import dataclass

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class WRFGrid:
    """Static grid geometry and dimensions used by JAX kernels."""

    west_east: int
    south_north: int
    bottom_top: int
    west_east_stag: int
    south_north_stag: int
    bottom_top_stag: int
    dx: float
    dy: float
    map_proj: int | None = None

    def tree_flatten(self) -> tuple[tuple[()], tuple[int, int, int, int, int, int, float, float, int | None]]:
        aux = (
            int(self.west_east),
            int(self.south_north),
            int(self.bottom_top),
            int(self.west_east_stag),
            int(self.south_north_stag),
            int(self.bottom_top_stag),
            float(self.dx),
            float(self.dy),
            None if self.map_proj is None else int(self.map_proj),
        )
        return (), aux

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[int, int, int, int, int, int, float, float, int | None], children: tuple[()]) -> "WRFGrid":
        del children
        return cls(*aux_data)


__all__ = ["WRFGrid"]
