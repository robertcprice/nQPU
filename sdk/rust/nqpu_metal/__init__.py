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
        # Quantum entropy extraction for LLM seeding
        'QuantumEntropyExtractor',
        'py_quick_seed',
        'py_batch_seeds',
    ]

# Try to import Qiskit compatibility
try:
    from . import qiskit_compat
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

__all__.extend(['__version__', 'QISKIT_AVAILABLE'])
