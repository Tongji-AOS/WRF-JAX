"""Configured local devices and bounded independent-batch submission."""

import atexit
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os

import jax


def configured_devices():
    """Select N visible GPUs, or the existing CPU device for CPU execution."""
    raw = os.environ.get("WRF_PY_GPU_COUNT", "1")
    try:
        count = int(raw)
    except ValueError as error:
        raise ValueError("gpu_count must be a positive integer") from error
    if count < 1:
        raise ValueError("gpu_count must be a positive integer")
    devices = tuple(jax.local_devices())
    if devices[0].platform == "cpu":
        return devices[:1]
    if count > len(devices):
        raise ValueError(f"gpu_count={count}, but only {len(devices)} GPUs are visible")
    return devices[:count]


def batch_partitions(length, batch_size, device_count):
    """Distribute whole existing batches without changing column grouping."""
    if min(length, batch_size, device_count) < 1:
        raise ValueError("positive length, batch size and device count required")
    batches = (length + batch_size - 1) // batch_size
    used = min(device_count, batches)
    quotient, remainder = divmod(batches, used)
    ranges, offset = [], 0
    for index in range(used):
        start = offset
        offset += (quotient + (index < remainder)) * batch_size
        ranges.append((start, min(offset, length)))
    return tuple(ranges)


@lru_cache(maxsize=None)
def _executor(count):
    pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="wrf-gpu")
    atexit.register(pool.shutdown)
    return pool


def submit_device_batches(function, batches, devices):
    """Run independent batches on their devices and return in domain order.

    The caller controls batch geometry and scientific equations. Device copies
    remain asynchronous; no model array is staged through NumPy or the host.
    Single-device calls retain the original allocations and call boundary.
    """
    if len(batches) != len(devices):
        raise ValueError("one batch per selected device required")
    if len(devices) == 1:
        return [function(batches[0], devices[0])]

    def run(batch, device):
        local = jax.device_put(batch, device)
        return function(local, device)

    futures = [_executor(len(devices)).submit(run, batch, device)
               for batch, device in zip(batches, devices, strict=True)]
    return [future.result() for future in futures]
