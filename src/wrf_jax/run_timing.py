"""Wall and sustained-integration reporting outside the numerical kernels."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def write_timing_summary(
    output_dir: Path, wall_seconds: float, *, warmup_seconds: int = 3600,
) -> dict[str, float]:
    """Report total wall time and complete intervals after configured warmup.

    The interval clock stops before snapshot processing and resumes after it.
    Do not subtract only the final NetCDF write from the trajectory timer:
    streamed writes and their comparisons are inside that timer as well.
    """
    if type(warmup_seconds) is not int or warmup_seconds < 0:
        raise ValueError("sustained warmup must be a nonnegative integer number of seconds")
    index_path = output_dir / "snapshot_index.txt"
    measured = []
    previous = 0
    if index_path.exists():
        with index_path.open(newline="", encoding="ascii") as stream:
            for row in csv.DictReader(stream):
                elapsed = int(row["elapsed_seconds"])
                seconds = float(row["integrate_wall_since_previous_snapshot_seconds"])
                if elapsed < previous or (elapsed == previous and elapsed != 0):
                    raise ValueError("snapshot times must increase")
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("snapshot integration duration must be finite and nonnegative")
                if previous >= warmup_seconds and elapsed > previous:
                    measured.append((previous, elapsed, seconds))
                previous = elapsed

    metrics = {}
    lines = [
        "# Run Timing",
        "",
        "| Time category | Seconds | Scope |",
        "|---|---:|---|",
        f"| Total wall time | {wall_seconds:.3f} | Full recorded run, including preparation and output |",
    ]
    if measured:
        start, end = measured[0][0], measured[-1][1]
        seconds = math.fsum(interval[2] for interval in measured)
        per_hour = seconds * 3600 / (end - start)
        metrics = {
            "sustained_integration_seconds": seconds,
            "sustained_model_start_seconds": float(start),
            "sustained_model_end_seconds": float(end),
            "sustained_seconds_per_model_hour": per_hour,
        }
        lines.extend((
            f"| Sustained integration, excluding output | {seconds:.3f} | Model {start / 3600:g}-{end / 3600:g} h; {per_hour:.3f} s per model hour |",
            "",
            "All dynamics and scheduled physics in this interval are included.",
        ))
    else:
        lines.extend((
            f"| Sustained integration, excluding output | N/A | No complete measured output interval after model second {warmup_seconds} |",
            "",
            "Do not infer a sustained time from an unmeasured interval.",
        ))
    lines.extend((
        "",
        f"The first {warmup_seconds} model seconds are excluded from the sustained window to separate",
        "startup/initial compilation. Any later preparation remains included.",
        "The snapshot callback (transfer, comparison, writing and bookkeeping) is",
        "excluded. Snapshot work done earlier by the trajectory remains included.",
        "Total wall time retains the existing main() timer",
        "scope; interpreter startup and earlier module imports are not in that counter.",
        "Legacy integrate_seconds includes streamed output and preparation; it is",
        "not the sustained-integration figure. Compare identical model-time windows.",
    ))
    (output_dir / "timing_summary.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    return metrics
