"""Shared measurement helpers for the intake harnesses.

Noise is the full spread of the measured runs: (max - min) / median * 100.
One untimed warmup run is discarded. The reported samples are the five runs after that.
"""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc
from typing import Callable

REPEATS = 5
WARMUP = 1


def spread_pct(samples : list[float]) -> float:
    """Full spread of the samples as a percent of their median."""
    med = statistics.median(samples)
    if med == 0:
        return 0.0
    return (max(samples) - min(samples)) / med * 100.0


def measure_ms(fn : Callable[[], object]) -> tuple[list[float], float, float]:
    """Wall time of fn, in milliseconds, after one warmup."""
    for _warmup in range(WARMUP):
        fn()

    samples : list[float] = []
    for _repeat in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)

    return samples, statistics.median(samples), spread_pct(samples)


def measure_peak_bytes(fn : Callable[[], object]) -> tuple[list[int], int, float]:
    """tracemalloc peak during fn. The input should already be allocated."""
    samples : list[int] = []
    for _index in range(WARMUP + REPEATS):
        gc.collect()
        tracemalloc.start()
        try:
            fn()
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        if _index >= WARMUP:
            samples.append(peak)

    median = statistics.median(samples)
    # Five samples: the middle value is one of the observed peaks.
    median_bytes = int(median)
    return samples, median_bytes, spread_pct([float(sample) for sample in samples])


def fmt_ms(samples : list[float], median : float, noise : float) -> str:
    rounded = ", ".join(f"{sample:.3f}" for sample in samples)
    return f"samples_ms=[{rounded}]\nmedian_ms={median:.3f}\nnoise_pct={noise:.4f}"


def fmt_bytes(samples : list[int], median : int, noise : float) -> str:
    rounded = ", ".join(str(sample) for sample in samples)
    return f"samples_bytes=[{rounded}]\nbaseline_value={median}\nnoise_pct={noise:.4f}"
