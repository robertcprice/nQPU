"""Cross-backend benchmarking framework for nQPU.

Compare quantum circuits across trapped-ion and superconducting backends,
measuring fidelity, gate counts, depth, noise impact, and resource usage.

Example:
    from nqpu.benchmarks import CrossBackendBenchmark
    bench = CrossBackendBenchmark()
    results = bench.run_all()
    bench.print_report(results)
"""

from .cross_backend import (
    CrossBackendBenchmark,
    BackendComparison,
    CircuitBenchmark,
)

__all__ = [
    "CrossBackendBenchmark",
    "BackendComparison",
    "CircuitBenchmark",
]
