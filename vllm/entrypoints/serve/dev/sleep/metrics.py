# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metrics for dispatched sleep-mode frontend operations."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from time import perf_counter
from typing import Literal

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from vllm.v1.metrics.prometheus import get_prometheus_registry

Operation = Literal["sleep", "release_kv_cache_memory", "wake"]
BUCKETS = (0.01, 0.1, 1, 10, 30, 60, 120, 300, 600)


class SleepModeOperationMetrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        registry = get_prometheus_registry() if registry is None else registry
        self.operations = Counter(
            "vllm:rl_sleep_mode_operations_total",
            "Dispatched sleep-mode operations by outcome.",
            ["operation", "status"],
            registry=registry,
        )
        self.duration = Histogram(
            "vllm:rl_sleep_mode_operation_duration_seconds",
            "Duration of one sleep-mode operation.",
            ["operation"],
            buckets=BUCKETS,
            registry=registry,
        )
        self.in_flight = Gauge(
            "vllm:rl_sleep_mode_operations_in_flight",
            "Sleep-mode operations currently awaited.",
            ["operation"],
            multiprocess_mode="livesum",
            registry=registry,
        )

    @contextmanager
    def record(self, operation: Operation) -> Iterator[None]:
        active = self.in_flight.labels(operation)
        started = perf_counter()
        active.inc()
        status = "error"
        try:
            yield
            status = "success"
        finally:
            active.dec()
            self.duration.labels(operation).observe(perf_counter() - started)
            self.operations.labels(operation, status).inc()


_metrics: SleepModeOperationMetrics | None = None
_metrics_lock = Lock()


def sleep_mode_operation_metrics() -> SleepModeOperationMetrics:
    """Return the process-wide sleep-mode collectors.

    The registry is resolved on the first sleep-mode request, i.e. after the
    server has decided whether metrics are aggregated across processes.
    """
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = SleepModeOperationMetrics()
    return _metrics


def reset_sleep_mode_operation_metrics() -> None:
    """Drop the cached collectors so tests can rebind them to a registry."""
    global _metrics
    with _metrics_lock:
        _metrics = None
