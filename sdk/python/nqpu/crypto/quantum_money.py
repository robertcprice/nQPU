"""Quantum money: unforgeable quantum banknotes.

Implements two quantum money schemes:

1. **WiesnerMoney**: Wiesner's original 1983 private-key scheme where the
   bank creates banknotes with qubits in random bases (computational or
   Hadamard).  The no-cloning theorem prevents counterfeiting -- any
   forgery attempt corrupts ~25% of qubits on average.

2. **PublicKeyMoney**: Simplified Aaronson-Christiano (2012) public-key
   quantum money using hidden subspaces.  Anyone can verify a banknote
   but only the bank (holding the secret key) can mint new ones.

All implementations use pure numpy -- no external dependencies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helper states and measurements
# ---------------------------------------------------------------------------

_KET_0 = np.array([1, 0], dtype=complex)
_KET_1 = np.array([0, 1], dtype=complex)
_KET_PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
_KET_MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)

_H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _prepare_qubit(basis: int, value: int) -> np.ndarray:
    """Prepare a qubit in given basis (0=Z, 1=X) with given value (0 or 1)."""
    if basis == 0:
        return _KET_0.copy() if value == 0 else _KET_1.copy()
    else:
        return _KET_PLUS.copy() if value == 0 else _KET_MINUS.copy()


def _measure_qubit(
    state: np.ndarray, basis: int, rng: np.random.Generator
) -> int:
    """Measure a single-qubit state in given basis. Returns 0 or 1."""
    if basis == 1:
        # Rotate to computational basis
        state = _H_GATE @ state
    prob_0 = float(np.abs(state[0]) ** 2)
    return 0 if rng.random() < prob_0 else 1


# ---------------------------------------------------------------------------
# WiesnerMoney
# ---------------------------------------------------------------------------
@dataclass
class WiesnerMoney:
    """Wiesner's quantum money scheme (1983).

    The bank creates banknotes by preparing n qubits, each in a random
    basis (Z or X) with a random bit value.  The serial number maps to
    the basis choices and bit values, which constitute the bank's secret.

    Verification: the bank measures each qubit in the recorded basis and
    checks that the result matches the recorded bit value.

    Security: the no-cloning theorem guarantees that a counterfeiter who
    does not know the bases cannot clone the banknote.  Any forgery
    attempt (measuring in a random basis) corrupts ~25% of qubits.
    """

    n_qubits: int = 16

    def __post_init__(self):
        self._bank_database: dict = {}

    def mint(
        self, serial: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[int, np.ndarray]:
        """Mint a new banknote.

        Parameters
        ----------
        serial : int
            Unique serial number for the banknote.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        (serial, quantum_state) where quantum_state is a list of
        single-qubit state vectors (each of shape (2,)).
        """
        if rng is None:
            rng = np.random.default_rng()

        bases = rng.integers(0, 2, size=self.n_qubits)
        values = rng.integers(0, 2, size=self.n_qubits)

        # Store bank's secret
        self._bank_database[serial] = {
            "bases": bases.copy(),
            "values": values.copy(),
        }

        # Prepare qubits
        qubits = np.zeros((self.n_qubits, 2), dtype=complex)
        for i in range(self.n_qubits):
            qubits[i] = _prepare_qubit(int(bases[i]), int(values[i]))

        return serial, qubits

    def verify(
        self,
        serial: int,
        state: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> bool:
        """Bank verifies a banknote by measuring in correct bases.

        Parameters
        ----------
        serial : int
            Serial number to look up in bank database.
        state : np.ndarray
            Array of shape (n_qubits, 2) -- single-qubit states.
        rng : np.random.Generator, optional
            Random number generator for measurement.

        Returns
        -------
        bool
            True if all qubits measure correctly.
        """
        if rng is None:
            rng = np.random.default_rng()

        if serial not in self._bank_database:
            return False

        record = self._bank_database[serial]
        bases = record["bases"]
        values = record["values"]

        for i in range(self.n_qubits):
            result = _measure_qubit(state[i].copy(), int(bases[i]), rng)
            if result != int(values[i]):
                return False
        return True

    def forge_attempt(
        self, state: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Simulate a counterfeiting attempt.

        The forger does not know the bases, so they measure each qubit
        in a random basis, then re-prepare based on their measurement
        result.  This corrupts ~25% of qubits on average.

        Parameters
        ----------
        state : np.ndarray
            Original banknote qubits, shape (n_qubits, 2).
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        np.ndarray
            Forged banknote qubits (degraded copy).
        """
        if rng is None:
            rng = np.random.default_rng()

        forged = np.zeros_like(state)
        for i in range(self.n_qubits):
            # Forger picks random basis
            forge_basis = int(rng.integers(0, 2))
            result = _measure_qubit(state[i].copy(), forge_basis, rng)
            forged[i] = _prepare_qubit(forge_basis, result)

        return forged

    def security_analysis(
        self, n_trials: int = 1000, rng: Optional[np.random.Generator] = None
    ) -> "MoneySecurityResult":
        """Analyze security by simulating forgery attempts.

        Mints banknotes, simulates forgery, and measures verification
        success rate.  Expected forgery success rate is (3/4)^n_qubits.

        Parameters
        ----------
        n_trials : int
            Number of forgery trials to simulate.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        MoneySecurityResult
            Statistical analysis of forgery attempts.
        """
        if rng is None:
            rng = np.random.default_rng()

        successes = 0
        for trial in range(n_trials):
            serial, qubits = self.mint(serial=1_000_000 + trial, rng=rng)
            forged = self.forge_attempt(qubits, rng=rng)
            if self.verify(serial, forged, rng=rng):
                successes += 1

        forgery_rate = successes / n_trials
        expected_rate = (3 / 4) ** self.n_qubits

        return MoneySecurityResult(
            n_trials=n_trials,
            forgery_success_rate=forgery_rate,
            expected_success_rate=expected_rate,
            secure=forgery_rate < 0.01,
        )


# ---------------------------------------------------------------------------
# PublicKeyMoney
# ---------------------------------------------------------------------------
@dataclass
class PublicKeyMoney:
    """Aaronson-Christiano public-key quantum money (simplified).

    Anyone can verify a banknote using the public key, but only the bank
    (holding the secret key) can mint new ones.

    Simplified scheme using a hidden subspace:
    - Secret key: a random subspace A of dimension subspace_dim in
      an n_qubits-dimensional space.
    - Public key: projector onto A (presented as a verification oracle).
    - Banknote: uniform superposition over vectors in A.
    - Verification: project onto A and check overlap is high.
    """

    n_qubits: int = 8
    subspace_dim: int = 4

    def __post_init__(self):
        if self.subspace_dim > 2 ** self.n_qubits:
            raise ValueError("subspace_dim must be <= 2^n_qubits")
        if self.subspace_dim < 1:
            raise ValueError("subspace_dim must be >= 1")

    def generate_keys(
        self, rng: Optional[np.random.Generator] = None
    ) -> Tuple[dict, dict]:
        """Generate (public_key, secret_key) pair.

        Returns
        -------
        (public_key, secret_key) where:
        - public_key has 'projector' (the verification projector)
        - secret_key has 'basis' (orthonormal basis of the hidden subspace)
        """
        if rng is None:
            rng = np.random.default_rng()

        dim = 2 ** self.n_qubits

        # Generate random subspace via QR of random matrix
        random_matrix = rng.standard_normal((dim, self.subspace_dim)) + \
                       1j * rng.standard_normal((dim, self.subspace_dim))
        basis, _ = np.linalg.qr(random_matrix)
        basis = basis[:, : self.subspace_dim]

        # Projector onto subspace
        projector = basis @ basis.conj().T

        public_key = {"projector": projector}
        secret_key = {"basis": basis}

        return public_key, secret_key

    def mint(
        self,
        secret_key: dict,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Mint banknote using secret key.

        Creates a uniform superposition over the hidden subspace.

        Parameters
        ----------
        secret_key : dict
            Secret key containing 'basis'.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        np.ndarray
            Banknote state vector.
        """
        if rng is None:
            rng = np.random.default_rng()

        basis = secret_key["basis"]

        # Random superposition over subspace
        coeffs = rng.standard_normal(self.subspace_dim) + \
                1j * rng.standard_normal(self.subspace_dim)
        coeffs /= np.linalg.norm(coeffs)

        banknote = basis @ coeffs
        return banknote

    def verify(self, state: np.ndarray, public_key: dict) -> bool:
        """Publicly verify banknote.

        Projects the state onto the hidden subspace and checks that the
        overlap is close to 1.

        Parameters
        ----------
        state : np.ndarray
            Candidate banknote state.
        public_key : dict
            Public key containing 'projector'.

        Returns
        -------
        bool
            True if the state lies (approximately) in the subspace.
        """
        projector = public_key["projector"]
        projected = projector @ state
        overlap = float(np.abs(np.vdot(state, projected)) ** 2)
        # Allow small numerical error
        return overlap > 0.99


# ---------------------------------------------------------------------------
# MoneySecurityResult
# ---------------------------------------------------------------------------
@dataclass
class MoneySecurityResult:
    """Result of quantum money security analysis."""

    n_trials: int
    forgery_success_rate: float
    expected_success_rate: float
    secure: bool

    def __repr__(self) -> str:
        return (
            f"MoneySecurityResult(\n"
            f"  n_trials={self.n_trials},\n"
            f"  forgery_success_rate={self.forgery_success_rate:.6f},\n"
            f"  expected_success_rate={self.expected_success_rate:.6e},\n"
            f"  secure={self.secure}\n"
            f")"
        )
