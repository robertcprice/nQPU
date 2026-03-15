"""
nQPU-Metal Python Package

High-performance quantum simulator with Metal GPU acceleration,
MPS support for 100+ qubits, and comprehensive noise modeling.
"""

# Version
__version__ = '1.0.0'

# Import when built with maturin
try:
    from .nqpu_metal import *
    __all__ = [
        'QuantumSimulator',
        'MPSSimulator',
        'TensorNetworkState1D',
        'statevector_to_mps_1d',
        'QuantumState',
        'QuantumCircuit',
        'EnhancedSimulator',
        'SimulationResult',
        'Backend',
        'NoiseModel',
        'create_bell_state',
        'create_ghz_state',
        'run_grover',
        'benchmark_gates',
        'apply_qft',
        'run_phase_estimation',
        'run_vqe',
        'simulate_noisy_circuit',
        'compare_ideal_vs_noisy',
        'dmrg_ground_state_1d',
        'tdvp_time_evolution_1d',
        'tdvp_loschmidt_echo_1d',
        'entanglement_spectrum_1d',
        'apply_local_pauli_1d',
        'tdvp_transition_observables_1d',
        # Quantum entropy extraction for LLM seeding
        'QuantumEntropyExtractor',
        'py_quick_seed',
        'py_batch_seeds',
    ]
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'target', 'debug'))
    import nqpu_metal as _nqpu_metal
    for attr in dir(_nqpu_metal):
        if not attr.startswith('_'):
            globals()[attr] = getattr(_nqpu_metal, attr)
    __all__ = [
        'QuantumSimulator',
        'MPSSimulator',
        'TensorNetworkState1D',
        'statevector_to_mps_1d',
        'QuantumState',
        'QuantumCircuit',
        'EnhancedSimulator',
        'SimulationResult',
        'Backend',
        'NoiseModel',
        'create_bell_state',
        'create_ghz_state',
        'run_grover',
        'benchmark_gates',
        'apply_qft',
        'run_phase_estimation',
        'run_vqe',
        'simulate_noisy_circuit',
        'compare_ideal_vs_noisy',
        'dmrg_ground_state_1d',
        'tdvp_time_evolution_1d',
        'tdvp_loschmidt_echo_1d',
        'entanglement_spectrum_1d',
        'apply_local_pauli_1d',
        'tdvp_transition_observables_1d',
        # Quantum entropy extraction for LLM seeding
        'QuantumEntropyExtractor',
        'py_quick_seed',
        'py_batch_seeds',
    ]

# Python utilities have moved to sdk/python/nqpu/metal/
# Import them from the new location for backward compatibility
try:
    from nqpu.metal import qiskit_compat
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

__all__.extend(['__version__', 'QISKIT_AVAILABLE'])
