"""Stable core API for circuits, simulation, and PennyLane-backed workflows."""

from .simulator import (
    Backend,
    NQPUBackend,
    QuantumBackend,
    QuantumCircuit,
    Result,
    SimulationResult,
)

# Optional PennyLane-based quantum backend symbols.
# These lived in the legacy sdk/python/core/ directory which has been removed.
# They are only available when PennyLane is installed and the backend module
# is accessible.  We gracefully degrade so the rest of the SDK works without it.
try:
    from core.quantum_backend import (  # type: ignore[import-not-found]
        BackendType,
        ClassicalFallback,
        HAS_PENNYLANE,
        MolecularGeometry,
        PENNYLANE_VERSION,
        QuantumBackendConfig,
        QuantumFingerprint,
        QuantumKernel,
        REFERENCE_ENERGIES,
        VQEMolecule,
        check_quantum_backend,
        quick_h2_energy,
    )
    _HAS_QUANTUM_BACKEND = True
except (ImportError, ModuleNotFoundError):
    _HAS_QUANTUM_BACKEND = False
    HAS_PENNYLANE = False
    PENNYLANE_VERSION = None

__all__ = [
    "Backend",
    "NQPUBackend",
    "QuantumBackend",
    "QuantumCircuit",
    "Result",
    "SimulationResult",
]

if _HAS_QUANTUM_BACKEND:
    __all__ += [
        "BackendType",
        "ClassicalFallback",
        "HAS_PENNYLANE",
        "MolecularGeometry",
        "PENNYLANE_VERSION",
        "QuantumBackendConfig",
        "QuantumFingerprint",
        "QuantumKernel",
        "REFERENCE_ENERGIES",
        "VQEMolecule",
        "check_quantum_backend",
        "quick_h2_energy",
    ]
