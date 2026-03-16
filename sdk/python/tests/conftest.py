"""Test-level conftest for the nQPU test suite.

Provides quantum-specific fixtures and pytest mark registration
used across all test modules.
"""

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Pytest marks registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers so they are visible in --markers output."""
    config.addinivalue_line("markers", "slow: longer-running backend and VQE validation tests")
    config.addinivalue_line("markers", "integration: cross-package integration tests")


# ---------------------------------------------------------------------------
# Quantum state fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bell_pair():
    """Bell state |Phi+> = (|00> + |11>) / sqrt(2) as a 4-element statevector."""
    state = np.zeros(4, dtype=np.complex128)
    state[0] = 1.0 / math.sqrt(2)
    state[3] = 1.0 / math.sqrt(2)
    return state


@pytest.fixture
def ghz_state():
    """3-qubit GHZ state (|000> + |111>) / sqrt(2) as an 8-element statevector."""
    state = np.zeros(8, dtype=np.complex128)
    state[0] = 1.0 / math.sqrt(2)
    state[7] = 1.0 / math.sqrt(2)
    return state


@pytest.fixture
def random_circuit(rng):
    """Generate a simple random circuit description for testing.

    Returns a list of (gate_name, qubit_indices, params) tuples representing
    a 4-qubit circuit with 10 random single- and two-qubit gates.
    """
    n_qubits = 4
    n_gates = 10
    single_gates = ["H", "X", "Y", "Z", "T", "S"]
    two_gates = ["CNOT", "CZ"]
    ops = []

    for _ in range(n_gates):
        if rng.random() < 0.6:
            gate = rng.choice(single_gates)
            qubit = int(rng.integers(0, n_qubits))
            ops.append((gate, (qubit,), ()))
        else:
            gate = rng.choice(two_gates)
            q0 = int(rng.integers(0, n_qubits))
            q1 = int(rng.integers(0, n_qubits - 1))
            if q1 >= q0:
                q1 += 1
            ops.append((gate, (q0, q1), ()))

    return ops
