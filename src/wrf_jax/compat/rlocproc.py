from __future__ import annotations

from collections.abc import MutableSequence


def rlocproc(p: int, maxi: int, nproc: int, ml: int, mr: int, ret: MutableSequence[int] | None = None) -> int:
    """Return WRF ``module_machine::rlocproc`` recursive location mapping."""

    p_i = int(p)
    maxi_i = int(maxi)
    nproc_i = int(nproc)
    ml_i = int(ml)
    mr_i = int(mr)

    adjust = 0
    rem = maxi_i % nproc_i
    width = maxi_i // nproc_i
    mid = maxi_i // 2
    if rem > 0 and (((rem % 2 == 0) or (rem > 2)) or (p_i <= mid)):
        width += 1
    if p_i <= mid and rem % 2 != 0:
        adjust += 1

    bl = max(width, ml_i)
    br = max(width, mr_i)
    if p_i < bl:
        result = 0
    elif p_i > maxi_i - br - 1:
        result = nproc_i - 1
    else:
        result = rlocproc(p_i - bl, maxi_i - bl - br + adjust, max(nproc_i - 2, 1), 0, 0) + 1

    if ret is not None:
        if len(ret) < 1:
            raise ValueError("ret output holder must contain at least one element")
        ret[0] = result
    return result


def locproc(i: int, m: int, numpart: int) -> int:
    """Return WRF ``module_machine::locproc`` process index for a point."""

    return rlocproc(int(i), int(m), int(numpart), 0, 0)


def region_bounds(
    region_start: int,
    region_end: int,
    num_p: int,
    p: int,
    patch_start: MutableSequence[int] | None = None,
    patch_end: MutableSequence[int] | None = None,
) -> tuple[int, int]:
    """Return WRF ``module_machine::region_bounds`` patch start and end."""

    start = int(region_start)
    end = int(region_end)
    num_parts = int(num_p)
    part = int(p)
    local_patch_end = -999999999
    local_patch_start = 999999999
    offset = start
    for i in range(0, end - offset + 1):
        if locproc(i, end - start + 1, num_parts) == part:
            local_patch_end = max(local_patch_end, i)
            local_patch_start = min(local_patch_start, i)
    local_patch_start += offset
    local_patch_end += offset

    if patch_start is not None:
        if len(patch_start) < 1:
            raise ValueError("patch_start output holder must contain at least one element")
        patch_start[0] = local_patch_start
    if patch_end is not None:
        if len(patch_end) < 1:
            raise ValueError("patch_end output holder must contain at least one element")
        patch_end[0] = local_patch_end
    return local_patch_start, local_patch_end


def least_aspect(nparts: int, minparts_y: int, minparts_x: int) -> tuple[int, int]:
    """Return WRF ``module_machine::least_aspect`` ``(nparts_y, nparts_x)``."""

    parts = int(nparts)
    min_y = int(minparts_y)
    min_x = int(minparts_x)
    mini = 2 * parts
    parts_y = 1
    parts_x = parts
    for y in range(1, parts + 1):
        if parts % y != 0:
            continue
        x = parts // y
        if abs(y - x) < mini and y >= min_y and x >= min_x:
            mini = abs(y - x)
            parts_y = y
            parts_x = x
    return parts_y, parts_x


__all__ = ["least_aspect", "locproc", "region_bounds", "rlocproc"]
