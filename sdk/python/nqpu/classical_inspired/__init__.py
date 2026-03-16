"""Quantum-inspired classical algorithms.

Classical algorithms that borrow structural ideas from quantum computing
to achieve speedups or better solutions on classical hardware.  Every
routine here runs on standard CPUs with numpy -- no quantum simulator
or hardware required.

Modules
-------
optimization
    Simulated quantum annealing, QAOA-inspired, quantum walk optimizer.
sampling
    Tang's dequantized sampling, tensor-network sampling, QI-MCMC.
linear_algebra
    QI-SVD, QI-regression, QI-PCA with exponential dimensionality tricks.
benchmarks
    Classical vs quantum-inspired comparison harness.
"""

from __future__ import annotations

from .optimization import (
    IsingProblem,
    SimulatedQuantumAnnealing,
    SQAResult,
    QAOAInspiredOptimizer,
    QAOAInspiredResult,
    QuantumWalkOptimizer,
    QuantumWalkOptimizerResult,
)
from .sampling import (
    DequantizedSampler,
    DequantizedSample,
    TNSampler,
    TNSampleResult,
    QIMonteCarlo,
    QIMCResult,
)
from .linear_algebra import (
    QISVD,
    QISVDResult,
    QIRegression,
    QIRegressionResult,
    QIPCA,
    QIPCAResult,
)
from .benchmarks import (
    BenchmarkSuite,
    BenchmarkResult,
    BenchmarkComparison,
    run_optimization_benchmark,
    run_sampling_benchmark,
    run_linear_algebra_benchmark,
)

__all__ = [
    # optimization
    "IsingProblem",
    "SimulatedQuantumAnnealing",
    "SQAResult",
    "QAOAInspiredOptimizer",
    "QAOAInspiredResult",
    "QuantumWalkOptimizer",
    "QuantumWalkOptimizerResult",
    # sampling
    "DequantizedSampler",
    "DequantizedSample",
    "TNSampler",
    "TNSampleResult",
    "QIMonteCarlo",
    "QIMCResult",
    # linear_algebra
    "QISVD",
    "QISVDResult",
    "QIRegression",
    "QIRegressionResult",
    "QIPCA",
    "QIPCAResult",
    # benchmarks
    "BenchmarkSuite",
    "BenchmarkResult",
    "BenchmarkComparison",
    "run_optimization_benchmark",
    "run_sampling_benchmark",
    "run_linear_algebra_benchmark",
]
