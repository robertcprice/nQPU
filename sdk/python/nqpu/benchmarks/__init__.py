"""Cross-backend benchmarking framework for nQPU.

Compare quantum circuits across trapped-ion, superconducting, and neutral-atom
backends, measuring fidelity, gate counts, depth, noise impact, and resource
usage.  Includes a Toffoli benchmark that highlights the native CCZ gate
advantage of neutral-atom hardware.

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
    BenchmarkCircuit,
    BackendAdapter,
    IonTrapAdapter,
    NeutralAtomAdapter,
    SuperconductingAdapter,
    build_digital_twin,
    fidelity_comparison_chart,
    gate_overhead_analysis,
    scaling_summary,
    toffoli_advantage_report,
)

__all__ = [
    "CrossBackendBenchmark",
    "BackendComparison",
    "CircuitBenchmark",
    "BenchmarkCircuit",
    "BackendAdapter",
    "IonTrapAdapter",
    "NeutralAtomAdapter",
    "SuperconductingAdapter",
    "build_digital_twin",
    "fidelity_comparison_chart",
    "gate_overhead_analysis",
    "scaling_summary",
    "toffoli_advantage_report",
]
