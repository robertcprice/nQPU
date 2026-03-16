"""nQPU integration dashboard: benchmarks, pipelines, and advisory tools.

Provides unified access to cross-package capabilities for practical
quantum computing workflow management.

Modules
-------
benchmark_dashboard
    Multi-backend benchmark suite with ASCII reports.
pipeline
    End-to-end transpile -> optimise -> simulate pipeline.
hardware_advisor
    Circuit analysis -> hardware recommendation engine.
cost_estimator
    Physical qubit count, cloud QPU pricing, budget optimiser.
"""

from __future__ import annotations

from .benchmark_dashboard import (
    DashboardBenchmark,
    BenchmarkReport,
    MultiBackendBenchmark,
    quick_benchmark,
    full_benchmark,
    compare_backends,
)
from .pipeline import (
    QuantumPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    StageResult,
    quick_run,
    optimized_run,
)
from .hardware_advisor import (
    HardwareAdvisor,
    HardwareRecommendation,
    CircuitProfile,
    HardwareProfile,
    DeviceDatabase,
    recommend_hardware,
)
from .cost_estimator import (
    CostEstimator,
    CostEstimate,
    CostBreakdown,
    CloudProvider,
    QECOverhead,
    estimate_cost,
    compare_providers,
)

__all__ = [
    # benchmark_dashboard
    "DashboardBenchmark",
    "BenchmarkReport",
    "MultiBackendBenchmark",
    "quick_benchmark",
    "full_benchmark",
    "compare_backends",
    # pipeline
    "QuantumPipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "StageResult",
    "quick_run",
    "optimized_run",
    # hardware_advisor
    "HardwareAdvisor",
    "HardwareRecommendation",
    "CircuitProfile",
    "HardwareProfile",
    "DeviceDatabase",
    "recommend_hardware",
    # cost_estimator
    "CostEstimator",
    "CostEstimate",
    "CostBreakdown",
    "CloudProvider",
    "QECOverhead",
    "estimate_cost",
    "compare_providers",
]
