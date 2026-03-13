"""Stable core API for circuits, simulation, and PennyLane-backed workflows."""

from .simulator import (
    Backend,
    NQPUBackend,
    QuantumBackend,
    QuantumCircuit,
    Result,
    SimulationResult,
)

from core.quantum_backend import (
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

__all__ = [
    "Backend",
    "BackendType",
    "ClassicalFallback",
    "HAS_PENNYLANE",
    "MolecularGeometry",
    "NQPUBackend",
    "PENNYLANE_VERSION",
    "QuantumBackend",
    "QuantumBackendConfig",
    "QuantumCircuit",
    "QuantumFingerprint",
    "QuantumKernel",
    "REFERENCE_ENERGIES",
    "Result",
    "SimulationResult",
    "VQEMolecule",
    "check_quantum_backend",
    "quick_h2_energy",
]
