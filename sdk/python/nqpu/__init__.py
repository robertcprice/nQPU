"""nQPU public Python API.

The package now exposes two stable entrypoints:

- `nqpu.core` for gate-based circuits and simulator access
- `nqpu.physics` for model Hamiltonians, solvers, and sweep workflows
"""

from .core import (
    Backend,
    HAS_PENNYLANE,
    NQPUBackend,
    PENNYLANE_VERSION,
    QuantumBackend,
    QuantumCircuit,
    Result,
    SimulationResult,
)

# Optional symbols that require the PennyLane quantum backend
try:
    from .core import (
        BackendType,
        QuantumBackendConfig,
        QuantumFingerprint,
        QuantumKernel,
        VQEMolecule,
    )
except ImportError:
    pass
from .physics import (
    AdaptiveDQPTDiagnosticsResult,
    AutoSolver,
    CorrelationMatrixResult,
    DQPTCandidate,
    DQPTDiagnosticsResult,
    DQPTScanPoint,
    DQPTScanResult,
    DynamicStructureFactorResult,
    EntanglementSpectrumResult,
    ExactDiagonalizationSolver,
    FrequencyStructureFactorResult,
    GroundStateResult,
    HeisenbergXXZ1D,
    HeisenbergXYZ1D,
    LoschmidtEchoResult,
    ModelQPU,
    ResponseSpectrumResult,
    RustTensorNetworkSolver,
    StructureFactorResult,
    SweepPoint,
    SweepResult,
    TimeEvolutionResult,
    TransverseFieldIsing1D,
    TwoTimeCorrelatorResult,
    analyze_dqpt_from_loschmidt,
    dump_tensor_network_state,
    load_dqpt_diagnostics_result,
    load_dqpt_scan_result,
    load_dynamic_structure_factor_result,
    load_entanglement_spectrum_result,
    load_frequency_structure_factor_result,
    load_ground_state_result,
    load_loschmidt_echo_result,
    load_response_spectrum_result,
    load_sweep_result,
    load_tensor_network_state,
    load_two_time_correlator_result,
    load_time_evolution_result,
    fourier_transform_structure_factor,
    response_spectrum_from_correlator,
    restore_tensor_network_state,
    save_dqpt_diagnostics_result,
    save_dqpt_scan_result,
    save_dynamic_structure_factor_result,
    save_entanglement_spectrum_result,
    save_frequency_structure_factor_result,
    save_ground_state_result,
    save_loschmidt_echo_result,
    save_response_spectrum_result,
    save_sweep_result,
    save_tensor_network_state,
    save_two_time_correlator_result,
    save_time_evolution_result,
)

__version__ = "0.1.0"

__all__ = [
    # core (always available)
    "Backend",
    "HAS_PENNYLANE",
    "NQPUBackend",
    "PENNYLANE_VERSION",
    "QuantumBackend",
    "QuantumCircuit",
    "Result",
    "SimulationResult",
    # physics (always available)
    "AdaptiveDQPTDiagnosticsResult",
    "AutoSolver",
    "CorrelationMatrixResult",
    "DQPTCandidate",
    "DQPTDiagnosticsResult",
    "DQPTScanPoint",
    "DQPTScanResult",
    "DynamicStructureFactorResult",
    "EntanglementSpectrumResult",
    "ExactDiagonalizationSolver",
    "FrequencyStructureFactorResult",
    "GroundStateResult",
    "HeisenbergXXZ1D",
    "HeisenbergXYZ1D",
    "LoschmidtEchoResult",
    "ModelQPU",
    "ResponseSpectrumResult",
    "RustTensorNetworkSolver",
    "StructureFactorResult",
    "SweepPoint",
    "SweepResult",
    "TimeEvolutionResult",
    "TransverseFieldIsing1D",
    "TwoTimeCorrelatorResult",
    "analyze_dqpt_from_loschmidt",
    "dump_tensor_network_state",
    "fourier_transform_structure_factor",
    "load_dqpt_diagnostics_result",
    "load_dqpt_scan_result",
    "load_dynamic_structure_factor_result",
    "load_entanglement_spectrum_result",
    "load_frequency_structure_factor_result",
    "load_ground_state_result",
    "load_loschmidt_echo_result",
    "load_response_spectrum_result",
    "load_sweep_result",
    "load_tensor_network_state",
    "load_two_time_correlator_result",
    "load_time_evolution_result",
    "response_spectrum_from_correlator",
    "restore_tensor_network_state",
    "save_dqpt_diagnostics_result",
    "save_dqpt_scan_result",
    "save_dynamic_structure_factor_result",
    "save_entanglement_spectrum_result",
    "save_frequency_structure_factor_result",
    "save_ground_state_result",
    "save_loschmidt_echo_result",
    "save_response_spectrum_result",
    "save_sweep_result",
    "save_tensor_network_state",
    "save_two_time_correlator_result",
    "save_time_evolution_result",
]

# Append optional PennyLane backend symbols if available
from .core import _HAS_QUANTUM_BACKEND as _HQB  # noqa: E402
if _HQB:
    __all__ += [
        "BackendType",
        "QuantumBackendConfig",
        "QuantumFingerprint",
        "QuantumKernel",
        "VQEMolecule",
    ]
del _HQB
