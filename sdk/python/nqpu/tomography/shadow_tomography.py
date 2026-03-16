"""Classical shadow tomography for efficient observable estimation.

Implements the randomised measurement protocol of Huang, Kueng & Preskill
(Nature Physics, 2020) for estimating expectation values of many
observables from far fewer measurements than full state tomography
requires.

The protocol:
1. Apply a random single-qubit Clifford rotation to each qubit.
2. Measure in the computational basis.
3. Construct "classical snapshots" by inverting the measurement channel.
4. Estimate observables by averaging over snapshots.

Shadow size formula: O(log(M) * max_weight^3 / epsilon^2) measurements
suffice to estimate M observables to additive accuracy epsilon, where
max_weight is the maximum locality (number of non-identity Paulis).

References:
    - Huang et al., Nature Physics 16, 1050 (2020) [Classical shadows]
    - Huang et al., Science 376, 1182 (2022) [Applications]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .state_tomography import (
    _I,
    _X,
    _Y,
    _Z,
    _PAULI_MAP,
    _ensure_density_matrix,
    _pauli_tensor,
    _embed_single_qubit,
)

# ---------------------------------------------------------------------------
# Random Clifford bases
# ---------------------------------------------------------------------------

# For single-qubit random Clifford shadows, we sample uniformly from {X, Y, Z}
# basis measurements.  The measurement channel inverse for the random Pauli
# protocol on a single qubit is:
#
#   M^{-1}(|b><b|) = 3 |b><b| - I
#
# where |b> is the Pauli eigenstate corresponding to outcome b in basis B.

_SQRT2_INV = 1.0 / math.sqrt(2.0)

# Pauli eigenstates: basis -> outcome -> state vector
_PAULI_EIGENSTATES = {
    "X": {
        0: np.array([_SQRT2_INV, _SQRT2_INV], dtype=np.complex128),
        1: np.array([_SQRT2_INV, -_SQRT2_INV], dtype=np.complex128),
    },
    "Y": {
        0: np.array([_SQRT2_INV, 1j * _SQRT2_INV], dtype=np.complex128),
        1: np.array([_SQRT2_INV, -1j * _SQRT2_INV], dtype=np.complex128),
    },
    "Z": {
        0: np.array([1.0, 0.0], dtype=np.complex128),
        1: np.array([0.0, 1.0], dtype=np.complex128),
    },
}


# ---------------------------------------------------------------------------
# Shadow snapshot
# ---------------------------------------------------------------------------


@dataclass
class ShadowSnapshot:
    """A single classical shadow snapshot.

    Attributes
    ----------
    bases : tuple[str, ...]
        Random Pauli basis per qubit.
    outcomes : tuple[int, ...]
        Measurement outcomes per qubit (0 or 1).
    """

    bases: tuple[str, ...]
    outcomes: tuple[int, ...]


# ---------------------------------------------------------------------------
# Classical shadow dataset
# ---------------------------------------------------------------------------


@dataclass
class ClassicalShadow:
    """A collection of classical shadow snapshots.

    Attributes
    ----------
    num_qubits : int
        Number of qubits.
    snapshots : list[ShadowSnapshot]
        Individual measurement snapshots.
    """

    num_qubits: int
    snapshots: list[ShadowSnapshot] = field(default_factory=list)

    @property
    def num_snapshots(self) -> int:
        """Number of snapshots in the dataset."""
        return len(self.snapshots)

    def snapshot_state(self, index: int) -> np.ndarray:
        """Reconstruct the inverted single-snapshot density matrix.

        For the random Pauli protocol, each snapshot yields:
            rho_hat = tensor_product_q (3 |b_q><b_q| - I)

        This is generally NOT a valid density matrix, but its average
        over many snapshots converges to the true state.

        Parameters
        ----------
        index : int
            Snapshot index.

        Returns
        -------
        np.ndarray
            Snapshot density matrix (2^n x 2^n).
        """
        snap = self.snapshots[index]
        rho_hat = np.array([[1.0]], dtype=np.complex128)

        for q in range(self.num_qubits):
            eigenstate = _PAULI_EIGENSTATES[snap.bases[q]][snap.outcomes[q]]
            local_rho = 3.0 * np.outer(eigenstate, eigenstate.conj()) - _I
            rho_hat = np.kron(rho_hat, local_rho)

        return rho_hat

    def estimate_density_matrix(self) -> np.ndarray:
        """Estimate the full density matrix by averaging snapshots.

        This is less efficient than targeted observable estimation but
        useful for small systems or when many properties are needed.

        Returns
        -------
        np.ndarray
            Estimated density matrix (2^n x 2^n).
        """
        dim = 2**self.num_qubits
        rho_est = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(self.num_snapshots):
            rho_est += self.snapshot_state(i)
        rho_est /= self.num_snapshots
        return rho_est


# ---------------------------------------------------------------------------
# Pauli observable
# ---------------------------------------------------------------------------


@dataclass
class PauliObservable:
    """A Pauli string observable, e.g. 'XIZZ'.

    Each character is one of 'I', 'X', 'Y', 'Z'.  The weight is the
    number of non-identity terms.

    Attributes
    ----------
    paulis : str
        Pauli string, e.g. 'XIZZ'.
    """

    paulis: str

    def __post_init__(self) -> None:
        for c in self.paulis:
            if c not in "IXYZ":
                raise ValueError(
                    f"Invalid Pauli character {c!r}. Must be I, X, Y, or Z."
                )

    @property
    def num_qubits(self) -> int:
        """Number of qubits the observable acts on."""
        return len(self.paulis)

    @property
    def weight(self) -> int:
        """Number of non-identity Paulis (locality)."""
        return sum(1 for c in self.paulis if c != "I")

    def matrix(self) -> np.ndarray:
        """Full matrix representation of the Pauli string."""
        return _pauli_tensor(list(self.paulis))


# ---------------------------------------------------------------------------
# Observable estimation
# ---------------------------------------------------------------------------


def estimate_expectation(
    shadow: ClassicalShadow,
    observable: PauliObservable,
) -> tuple[float, float]:
    """Estimate the expectation value of a Pauli observable.

    Uses the median-of-means estimator for robust estimation.  For
    each snapshot, the single-shot estimator is:

        hat{o}_i = prod_{q in support(O)} tr(O_q . hat{rho}^(q)_i)

    where O_q is the local Pauli on qubit q and hat{rho}^(q)_i is
    the local inverted state.

    Parameters
    ----------
    shadow : ClassicalShadow
        Classical shadow dataset.
    observable : PauliObservable
        Pauli string to estimate.

    Returns
    -------
    tuple[float, float]
        (estimate, standard_error) where estimate is the mean and
        standard_error is the standard error of the mean.
    """
    if observable.num_qubits != shadow.num_qubits:
        raise ValueError(
            f"Observable has {observable.num_qubits} qubits but shadow "
            f"has {shadow.num_qubits} qubits"
        )

    estimates: list[float] = []

    for snap in shadow.snapshots:
        val = _single_shot_estimator(snap, observable)
        estimates.append(val)

    if len(estimates) == 0:
        return 0.0, float("inf")

    mean = float(np.mean(estimates))
    std_err = float(np.std(estimates, ddof=1) / math.sqrt(len(estimates)))
    return mean, std_err


def _single_shot_estimator(
    snap: ShadowSnapshot, observable: PauliObservable
) -> float:
    """Compute the single-shot estimator for one snapshot.

    For the random Pauli protocol, the estimator is the product over
    qubits in the support of the observable:

    For each qubit q with Pauli O_q != I:
      - If the measurement basis matches O_q: the estimator
        contributes (-1)^{outcome_q} * 3
      - If the measurement basis does not match O_q: contributes 0

    The factor of 3 comes from the channel inversion: for a single
    qubit, tr(O_q (3|b><b| - I)) = 3*<b|O_q|b> - tr(O_q) = 3*<b|O_q|b>
    since tr(Pauli) = 0 for X, Y, Z.
    """
    result = 1.0

    for q in range(observable.num_qubits):
        pauli_char = observable.paulis[q]
        if pauli_char == "I":
            continue

        basis = snap.bases[q]
        outcome = snap.outcomes[q]

        if basis != pauli_char:
            # Measurement in a different basis: estimator is zero
            return 0.0

        # Same basis: eigenvalue is (-1)^outcome, scaled by 3
        result *= 3.0 * ((-1) ** outcome)

    return result


def estimate_fidelity(
    shadow: ClassicalShadow,
    target_state: np.ndarray,
) -> float:
    """Estimate the fidelity with a target pure state.

    For a pure target |psi>, the fidelity is:
        F = <psi|rho|psi> = sum_P  <psi|P|psi> * tr(P rho) / d

    where the sum is over all Pauli strings P.  We decompose the target
    state into its Pauli expansion and estimate each term.

    For efficiency, only Pauli terms with non-negligible coefficients
    are estimated.

    Parameters
    ----------
    shadow : ClassicalShadow
        Classical shadow dataset.
    target_state : np.ndarray
        Target state vector or density matrix.

    Returns
    -------
    float
        Estimated fidelity in [0, 1].
    """
    target_dm = _ensure_density_matrix(target_state)
    n = shadow.num_qubits
    d = 2**n

    # Decompose target into Pauli basis
    # <psi|P|psi> = tr(P . target_dm)
    import itertools

    fidelity_estimate = 0.0

    for labels in itertools.product("IXYZ", repeat=n):
        pauli_op = _pauli_tensor(labels)
        coeff = float(np.real(np.trace(pauli_op @ target_dm)))

        if abs(coeff) < 1e-12:
            continue

        obs = PauliObservable("".join(labels))
        if obs.weight == 0:
            # Identity: tr(I * rho) = 1 always
            fidelity_estimate += coeff / d
        else:
            est, _ = estimate_expectation(shadow, obs)
            fidelity_estimate += coeff * est / d

    return float(np.clip(fidelity_estimate, 0.0, 1.0))


def shadow_size_bound(
    num_observables: int,
    max_weight: int,
    epsilon: float,
    delta: float = 0.01,
) -> int:
    """Compute the required number of shadow snapshots.

    For the random Pauli measurement protocol, the number of samples
    needed to estimate M observables of weight at most k to additive
    error epsilon with failure probability delta is:

        N = O(log(M/delta) * 3^k / epsilon^2)

    Parameters
    ----------
    num_observables : int
        Number of observables to estimate (M).
    max_weight : int
        Maximum locality / weight of any observable (k).
    epsilon : float
        Target additive accuracy.
    delta : float
        Failure probability.

    Returns
    -------
    int
        Required number of shadow snapshots (upper bound).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")

    # Constant factor from the theoretical bound
    log_factor = math.log(2.0 * num_observables / delta)
    shadow_norm = 3**max_weight  # shadow norm for random Pauli protocol
    return int(math.ceil(2.0 * log_factor * shadow_norm / (epsilon**2)))


# ---------------------------------------------------------------------------
# Shadow generation from known state (for testing)
# ---------------------------------------------------------------------------


def create_shadow_from_state(
    state: np.ndarray,
    num_snapshots: int,
    rng: np.random.Generator | None = None,
) -> ClassicalShadow:
    """Generate a classical shadow from a known quantum state.

    Simulates the random Pauli measurement protocol by computing
    Born-rule probabilities for each randomly chosen basis.

    Parameters
    ----------
    state : np.ndarray
        State vector or density matrix.
    num_snapshots : int
        Number of shadow snapshots to generate.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    ClassicalShadow
        Shadow dataset with the requested number of snapshots.
    """
    if rng is None:
        rng = np.random.default_rng()

    rho = _ensure_density_matrix(state)
    num_qubits = int(np.log2(rho.shape[0]))
    dim = 2**num_qubits

    basis_choices = ["X", "Y", "Z"]
    snapshots: list[ShadowSnapshot] = []

    for _ in range(num_snapshots):
        # Random basis per qubit
        bases = tuple(
            basis_choices[rng.integers(3)] for _ in range(num_qubits)
        )

        # Build the rotation for this basis choice
        rotation = np.eye(dim, dtype=np.complex128)
        for q in range(num_qubits):
            if bases[q] == "X":
                # Hadamard
                h = np.array(
                    [[_SQRT2_INV, _SQRT2_INV], [_SQRT2_INV, -_SQRT2_INV]],
                    dtype=np.complex128,
                )
                rotation = _embed_single_qubit(h, q, num_qubits) @ rotation
            elif bases[q] == "Y":
                # S^dag then Hadamard
                h = np.array(
                    [[_SQRT2_INV, _SQRT2_INV], [_SQRT2_INV, -_SQRT2_INV]],
                    dtype=np.complex128,
                )
                s_dag = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
                gate = h @ s_dag
                rotation = _embed_single_qubit(gate, q, num_qubits) @ rotation
            # Z basis: no rotation needed

        # Compute probabilities in computational basis after rotation
        rho_rotated = rotation @ rho @ rotation.conj().T
        probs = np.real(np.diag(rho_rotated))
        probs = np.maximum(probs, 0.0)
        total = np.sum(probs)
        if total > 0:
            probs /= total

        # Sample an outcome
        outcome_idx = rng.choice(dim, p=probs)
        outcomes = tuple(
            int((outcome_idx >> (num_qubits - 1 - q)) & 1)
            for q in range(num_qubits)
        )

        snapshots.append(ShadowSnapshot(bases=bases, outcomes=outcomes))

    return ClassicalShadow(num_qubits=num_qubits, snapshots=snapshots)
