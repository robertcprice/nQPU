"""nQPU Quantum Circuit Transpiler -- routing, optimization, and basis translation.

Pure-Python transpiler pipeline for mapping abstract quantum circuits to
hardware-constrained physical circuits.  Mirrors the Rust transpiler in
``sdk/rust/src/circuits/synthesis/`` while providing an accessible Python
interface for rapid prototyping and research.

Pipeline stages
---------------
1. **Circuit construction** (``circuits``): Build a logical quantum circuit
   from the standard gate library.
2. **Routing** (``routing``): Map logical qubits to physical qubits and
   insert SWAP gates to satisfy coupling-map constraints.  Includes the
   SABRE bidirectional heuristic, greedy nearest-neighbour, and trivial
   identity routing.
3. **Optimization** (``optimization``): Reduce gate count via cancellation,
   rotation merging, single-qubit fusion, and commutation analysis.
4. **Decomposition** (``decomposition``): Rewrite circuits into a target
   hardware basis gate set (IBM, Google, Rigetti) via ZYZ and KAK
   decompositions.

All modules are pure numpy with no external dependencies.

Example::

    from nqpu.transpiler import (
        QuantumCircuit, CouplingMap, SABRERouter,
        optimize, decompose, BasisSet,
    )

    # Build a 3-qubit circuit
    qc = QuantumCircuit(3)
    qc.h(0).cx(0, 1).cx(1, 2)

    # Route for a linear topology
    cm = CouplingMap.from_line(3)
    result = SABRERouter().route(qc, cm)

    # Optimize and decompose to IBM basis
    opt = optimize(result.circuit, level=2)
    final = decompose(opt, basis=BasisSet.IBM)
"""

from __future__ import annotations

# ----- Circuits -----
from .circuits import (
    Gate,
    QuantumCircuit,
    CircuitStats,
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
    T,
    Tdg,
    SX,
    Id,
    Rx,
    Ry,
    Rz,
    U3,
    CX,
    CNOT,
    CZ,
    SWAP,
    CCX,
    Toffoli,
)

# ----- Coupling Map -----
from .coupling import CouplingMap

# ----- Routing -----
from .routing import (
    Layout,
    InitialLayout,
    RoutingResult,
    TrivialRouter,
    GreedyRouter,
    SABRERouter,
    SabreConfig,
    SabreHeuristic,
    route,
)

# ----- Optimization -----
from .optimization import (
    GateCancellation,
    RotationMerging,
    SingleQubitFusion,
    CommutationAnalysis,
    TwoQubitDecomposition,
    OptimizationLevel,
    optimize,
)

# ----- Decomposition -----
from .decomposition import (
    BasisSet,
    BasisTranslator,
    ZYZDecomposition,
    KAKDecomposition,
    KAKResult,
    ToffoliDecomposition,
    decompose,
)

# ----- Shannon Decomposition -----
from .shannon import (
    DecomposedGate,
    ShannonDecomposition,
    CSDResult,
    reconstruct_unitary,
)

# ----- Solovay-Kitaev -----
from .solovay_kitaev import (
    GateSequence,
    BasicApproximations,
    SolovayKitaev,
    SKResult,
    operator_distance,
    approximate_rotation,
    approximate_u3,
    H_GATE,
    T_GATE,
    S_GATE,
)

# ----- Template Matching -----
from .template_matching import (
    CircuitDAG,
    DAGNode,
    Template,
    TemplateMatcher,
    TemplateMatchResult,
    default_templates,
)

# ----- Noise Adaptive -----
from .noise_adaptive import (
    CalibrationData,
    NoiseAdaptiveRouter,
    NoiseAdaptiveResult,
    NoiseAdaptiveDecomposer,
    CircuitFidelityEstimator,
)

__all__ = [
    # Circuits
    "Gate",
    "QuantumCircuit",
    "CircuitStats",
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "Sdg",
    "T",
    "Tdg",
    "SX",
    "Id",
    "Rx",
    "Ry",
    "Rz",
    "U3",
    "CX",
    "CNOT",
    "CZ",
    "SWAP",
    "CCX",
    "Toffoli",
    # Coupling map
    "CouplingMap",
    # Routing
    "Layout",
    "InitialLayout",
    "RoutingResult",
    "TrivialRouter",
    "GreedyRouter",
    "SABRERouter",
    "SabreConfig",
    "SabreHeuristic",
    "route",
    # Optimization
    "GateCancellation",
    "RotationMerging",
    "SingleQubitFusion",
    "CommutationAnalysis",
    "TwoQubitDecomposition",
    "OptimizationLevel",
    "optimize",
    # Decomposition
    "BasisSet",
    "BasisTranslator",
    "ZYZDecomposition",
    "KAKDecomposition",
    "KAKResult",
    "ToffoliDecomposition",
    "decompose",
    # Shannon Decomposition
    "DecomposedGate",
    "ShannonDecomposition",
    "CSDResult",
    "reconstruct_unitary",
    # Solovay-Kitaev
    "GateSequence",
    "BasicApproximations",
    "SolovayKitaev",
    "SKResult",
    "operator_distance",
    "approximate_rotation",
    "approximate_u3",
    "H_GATE",
    "T_GATE",
    "S_GATE",
    # Template Matching
    "CircuitDAG",
    "DAGNode",
    "Template",
    "TemplateMatcher",
    "TemplateMatchResult",
    "default_templates",
    # Noise Adaptive
    "CalibrationData",
    "NoiseAdaptiveRouter",
    "NoiseAdaptiveResult",
    "NoiseAdaptiveDecomposer",
    "CircuitFidelityEstimator",
]
