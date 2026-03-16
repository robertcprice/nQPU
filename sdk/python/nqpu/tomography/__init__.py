"""nQPU Quantum State Tomography and Verification.

Full-stack quantum state reconstruction and characterisation toolkit
for the nQPU quantum computing SDK.

Four capabilities:
  - State tomography: Reconstruct density matrices from Pauli measurements
  - Process tomography: Characterise quantum channels via chi/Choi matrices
  - Shadow tomography: Efficient observable estimation from random measurements
  - Verification: Fidelity, purity, entropy, entanglement, and benchmarks

Example:
    from nqpu.tomography import (
        StateTomographer,
        ClassicalShadow,
        create_shadow_from_state,
        state_fidelity,
        concurrence,
    )

    # State tomography
    tomo = StateTomographer(num_qubits=2)
    circuits = tomo.measurement_circuits()
    # ... run on hardware, collect measurements ...
    result = tomo.reconstruct(measurements, method='mle')

    # Shadow tomography
    shadow = create_shadow_from_state(state_vector, num_snapshots=5000)
    fid = estimate_fidelity(shadow, target_state)
"""

from .state_tomography import (
    MeasurementCircuit,
    StateTomographer,
    StateTomographyResult,
    TomographyMeasurementResult,
    generate_measurement_circuits,
    generate_tetrahedral_circuits,
    simulate_tomography_measurements,
    state_fidelity,
)
from .process_tomography import (
    ProcessCircuit,
    ProcessTomographer,
    ProcessTomographyResult,
    chi_to_choi,
    choi_to_chi,
    simulate_process_tomography,
)
from .shadow_tomography import (
    ClassicalShadow,
    PauliObservable,
    ShadowSnapshot,
    create_shadow_from_state,
    estimate_expectation,
    estimate_fidelity,
    shadow_size_bound,
)
from .verification import (
    EntanglementWitnessResult,
    QuantumVolumeResult,
    XEBResult,
    average_gate_fidelity,
    concurrence,
    cross_entropy_benchmark,
    entanglement_of_formation,
    entanglement_witness_bell,
    entanglement_witness_ghz,
    estimate_quantum_volume,
    linear_entropy,
    negativity,
    partial_trace,
    purity,
    relative_entropy,
    schmidt_decomposition,
    schmidt_number,
    trace_distance,
    von_neumann_entropy,
)

__all__ = [
    # State tomography
    "MeasurementCircuit",
    "StateTomographer",
    "StateTomographyResult",
    "TomographyMeasurementResult",
    "generate_measurement_circuits",
    "generate_tetrahedral_circuits",
    "simulate_tomography_measurements",
    "state_fidelity",
    # Process tomography
    "ProcessCircuit",
    "ProcessTomographer",
    "ProcessTomographyResult",
    "chi_to_choi",
    "choi_to_chi",
    "simulate_process_tomography",
    # Shadow tomography
    "ClassicalShadow",
    "PauliObservable",
    "ShadowSnapshot",
    "create_shadow_from_state",
    "estimate_expectation",
    "estimate_fidelity",
    "shadow_size_bound",
    # Verification
    "EntanglementWitnessResult",
    "QuantumVolumeResult",
    "XEBResult",
    "average_gate_fidelity",
    "concurrence",
    "cross_entropy_benchmark",
    "entanglement_of_formation",
    "entanglement_witness_bell",
    "entanglement_witness_ghz",
    "estimate_quantum_volume",
    "linear_entropy",
    "negativity",
    "partial_trace",
    "purity",
    "relative_entropy",
    "schmidt_decomposition",
    "schmidt_number",
    "trace_distance",
    "von_neumann_entropy",
]
