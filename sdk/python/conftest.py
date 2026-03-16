"""Root conftest for the nQPU Python SDK.

Ensures sdk/python is on sys.path so that ``import nqpu`` resolves
correctly regardless of working directory, and provides lightweight
shared fixtures used across the entire test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup -- guarantee sdk/python is importable
# ---------------------------------------------------------------------------
_SDK_PYTHON = str(Path(__file__).resolve().parent)
if _SDK_PYTHON not in sys.path:
    sys.path.insert(0, _SDK_PYTHON)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded NumPy random generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def small_state():
    """2-qubit zero state |00> as a complex128 statevector."""
    state = np.zeros(4, dtype=np.complex128)
    state[0] = 1.0
    return state
