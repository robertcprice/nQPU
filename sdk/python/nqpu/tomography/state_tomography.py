"""Quantum state tomography for full density-matrix reconstruction.

Implements three reconstruction methods for recovering an unknown quantum
state from measurement statistics collected in multiple Pauli bases:

1. **Linear inversion**: Direct formula using Pauli expectation values.
   Fast but can produce non-physical density matrices (negative eigenvalues).

2. **Maximum Likelihood Estimation (MLE)**: Hradil's iterative R*rho*R
   algorithm that converges to the physical state most consistent with
   the observed data.  Always produces a valid density matrix.

3. **Least-squares with physicality constraints**: Minimises the
   Frobenius-norm residual subject to trace-1 and positive-semidefinite
   constraints via projected gradient descent.

References:
    - Hradil, Phys. Rev. A 55, R1561 (1997) [MLE algorithm]
    - James et al., Phys. Rev. A 64, 052312 (2001) [Linear inversion]
    - Smolin et al., Phys. Rev. Lett. 108, 070502 (2012) [Least squares]
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

_PAULI_MAP = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}

# Single-qubit Pauli eigenstates (columns of the diagonalising unitary)
# |+x>, |-x>, |+y>, |-y>, |0>, |1>
_SQRT2_INV = 1.0 / math.sqrt(2.0)


def _pauli_tensor(labels: Sequence[str]) -> np.ndarray:
    """Compute the tensor product of single-qubit Pauli matrices.

    Parameters
    ----------
    labels : sequence of str
        Each element is one of 'I', 'X', 'Y', 'Z'.

    Returns
    -------
    np.ndarray
        The 2^n x 2^n tensor-product matrix.
    """
    result = np.array([[1.0]], dtype=np.complex128)
    for label in labels:
        result = np.kron(result, _PAULI_MAP[label])
    return result


def _basis_rotation_gate(basis: str) -> np.ndarray:
    """Return the single-qubit rotation that maps a Pauli eigenbasis to Z.

    After applying this gate, measuring in the Z basis is equivalent
    to measuring in the given Pauli basis.

    Parameters
    ----------
    basis : str
        One of 'X', 'Y', 'Z'.

    Returns
    -------
    np.ndarray
        2x2 unitary rotation matrix.
    """
    if basis == "Z":
        return _I.copy()
    elif basis == "X":
        # Hadamard: maps X eigenstates to Z eigenstates
        return np.array(
            [[_SQRT2_INV, _SQRT2_INV], [_SQRT2_INV, -_SQRT2_INV]],
            dtype=np.complex128,
        )
    elif basis == "Y":
        # S^dag then Hadamard: maps Y eigenstates to Z eigenstates
        h = np.array(
            [[_SQRT2_INV, _SQRT2_INV], [_SQRT2_INV, -_SQRT2_INV]],
            dtype=np.complex128,
        )
        s_dag = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
        return h @ s_dag
    else:
        raise ValueError(f"Unknown basis: {basis!r}")


# ---------------------------------------------------------------------------
# Measurement circuit generation
# ---------------------------------------------------------------------------


@dataclass
class MeasurementCircuit:
    """Description of a tomographic measurement setting.

    Attributes
    ----------
    bases : tuple[str, ...]
        Pauli basis label per qubit, e.g. ('X', 'Z', 'Y').
    rotations : list[tuple[int, np.ndarray]]
        List of (qubit_index, rotation_matrix) to apply before
        Z-basis measurement.
    """

    bases: tuple[str, ...]
    rotations: list[tuple[int, np.ndarray]] = field(default_factory=list)


def generate_measurement_circuits(
    num_qubits: int,
) -> list[MeasurementCircuit]:
    """Generate all 3^n Pauli measurement circuits for full state tomography.

    For *n* qubits the full set of tomographic measurement settings
    consists of every combination of {X, Y, Z} on each qubit, giving
    3^n distinct settings.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    Returns
    -------
    list[MeasurementCircuit]
        One entry per measurement setting, containing the basis
        labels and the single-qubit rotations required.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")

    circuits: list[MeasurementCircuit] = []
    for combo in itertools.product("XYZ", repeat=num_qubits):
        rotations: list[tuple[int, np.ndarray]] = []
        for qubit_idx, basis in enumerate(combo):
            gate = _basis_rotation_gate(basis)
            if basis != "Z":
                rotations.append((qubit_idx, gate))
        circuits.append(MeasurementCircuit(bases=combo, rotations=rotations))
    return circuits


def generate_tetrahedral_circuits(
    num_qubits: int,
) -> list[MeasurementCircuit]:
    """Generate measurement circuits using tetrahedral (SIC-POVM) directions.

    Uses four measurement directions per qubit that form a regular
    tetrahedron on the Bloch sphere.  This gives 4^n settings instead
    of 3^n, but provides a more symmetric informationally-complete
    measurement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    Returns
    -------
    list[MeasurementCircuit]
        One entry per measurement setting.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")

    # Tetrahedral directions on the Bloch sphere
    tet_directions = [
        (0.0, 0.0, 1.0),                                          # +Z
        (2.0 * math.sqrt(2.0) / 3.0, 0.0, -1.0 / 3.0),          # equatorial 1
        (-math.sqrt(2.0) / 3.0, math.sqrt(6.0) / 3.0, -1.0 / 3.0),   # equatorial 2
        (-math.sqrt(2.0) / 3.0, -math.sqrt(6.0) / 3.0, -1.0 / 3.0),  # equatorial 3
    ]

    def _rotation_for_direction(nx: float, ny: float, nz: float) -> np.ndarray:
        """Rotation that maps n-hat eigenstate to Z eigenstate."""
        # Construct the unitary that diagonalises n . sigma
        # n . sigma = nx X + ny Y + nz Z
        n_dot_sigma = nx * _X + ny * _Y + nz * _Z
        eigvals, eigvecs = np.linalg.eigh(n_dot_sigma)
        # eigvecs columns are eigenstates; we want U such that
        # U (n.sigma) U^dag = Z  =>  U = eigvecs^dag (with column reorder)
        # eigh returns ascending eigenvalues, so col0 -> -1, col1 -> +1
        # We want |+n> -> |0> and |-n> -> |1>, so reverse columns
        return eigvecs[:, ::-1].conj().T

    direction_labels = ["T0", "T1", "T2", "T3"]
    circuits: list[MeasurementCircuit] = []

    for combo_idx in itertools.product(range(4), repeat=num_qubits):
        bases = tuple(direction_labels[i] for i in combo_idx)
        rotations: list[tuple[int, np.ndarray]] = []
        for qubit_idx, dir_idx in enumerate(combo_idx):
            nx, ny, nz = tet_directions[dir_idx]
            gate = _rotation_for_direction(nx, ny, nz)
            if not np.allclose(gate, _I):
                rotations.append((qubit_idx, gate))
        circuits.append(MeasurementCircuit(bases=bases, rotations=rotations))

    return circuits


# ---------------------------------------------------------------------------
# Measurement result container
# ---------------------------------------------------------------------------


@dataclass
class TomographyMeasurementResult:
    """Raw measurement data from one tomographic setting.

    Attributes
    ----------
    bases : tuple[str, ...]
        Pauli basis per qubit.
    counts : dict[str, int]
        Bitstring histogram, e.g. ``{'00': 512, '11': 488}``.
    shots : int
        Total number of measurement shots.
    """

    bases: tuple[str, ...]
    counts: dict[str, int]
    shots: int

    @property
    def probabilities(self) -> dict[str, float]:
        """Normalised probability of each outcome."""
        return {k: v / self.shots for k, v in self.counts.items()}


# ---------------------------------------------------------------------------
# Tomography result
# ---------------------------------------------------------------------------


@dataclass
class StateTomographyResult:
    """Result of quantum state tomography.

    Attributes
    ----------
    density_matrix : np.ndarray
        Reconstructed density matrix (2^n x 2^n complex).
    method : str
        Reconstruction method used ('linear', 'mle', 'lstsq').
    num_qubits : int
        Number of qubits.
    iterations : int
        Number of iterations for iterative methods (0 for linear).
    """

    density_matrix: np.ndarray
    method: str
    num_qubits: int
    iterations: int = 0

    @property
    def purity(self) -> float:
        """Purity tr(rho^2), 1 for pure states, 1/d for maximally mixed."""
        rho = self.density_matrix
        return float(np.real(np.trace(rho @ rho)))

    @property
    def von_neumann_entropy(self) -> float:
        """Von Neumann entropy -tr(rho log rho) in nats."""
        eigvals = np.linalg.eigvalsh(self.density_matrix)
        eigvals = eigvals[eigvals > 1e-15]
        return float(-np.sum(eigvals * np.log(eigvals)))

    def fidelity_with(self, target: np.ndarray) -> float:
        """State fidelity F(rho, sigma) with a target state.

        For a pure target |psi>, this simplifies to <psi|rho|psi>.
        For mixed states uses the Uhlmann fidelity:
        F = (tr(sqrt(sqrt(rho) sigma sqrt(rho))))^2.

        Parameters
        ----------
        target : np.ndarray
            Target density matrix or state vector.

        Returns
        -------
        float
            Fidelity in [0, 1].
        """
        sigma = _ensure_density_matrix(target)
        return state_fidelity(self.density_matrix, sigma)

    @property
    def is_physical(self) -> bool:
        """Check if the density matrix is a valid quantum state.

        A valid density matrix is Hermitian, positive semidefinite,
        and has unit trace.
        """
        rho = self.density_matrix
        # Hermitian
        if not np.allclose(rho, rho.conj().T, atol=1e-10):
            return False
        # Trace 1
        if abs(np.trace(rho).real - 1.0) > 1e-8:
            return False
        # Positive semidefinite
        eigvals = np.linalg.eigvalsh(rho)
        if np.any(eigvals < -1e-10):
            return False
        return True


# ---------------------------------------------------------------------------
# Helper: ensure density matrix
# ---------------------------------------------------------------------------


def _ensure_density_matrix(state: np.ndarray) -> np.ndarray:
    """Convert a state vector to a density matrix if needed."""
    if state.ndim == 1:
        return np.outer(state, state.conj())
    return state


# ---------------------------------------------------------------------------
# State fidelity
# ---------------------------------------------------------------------------


def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the Uhlmann fidelity between two density matrices.

    F(rho, sigma) = (tr(sqrt(sqrt(rho) . sigma . sqrt(rho))))^2

    Parameters
    ----------
    rho, sigma : np.ndarray
        Density matrices of equal dimension.

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    rho = _ensure_density_matrix(rho)
    sigma = _ensure_density_matrix(sigma)

    # Compute sqrt(rho) via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_rho = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T

    # Product: sqrt(rho) . sigma . sqrt(rho)
    product = sqrt_rho @ sigma @ sqrt_rho

    # Eigenvalues of the product
    prod_eigvals = np.linalg.eigvalsh(product)
    prod_eigvals = np.maximum(prod_eigvals, 0.0)

    fid = float(np.sum(np.sqrt(prod_eigvals)) ** 2)
    return float(np.clip(fid, 0.0, 1.0))


# ---------------------------------------------------------------------------
# StateTomographer
# ---------------------------------------------------------------------------


class StateTomographer:
    """Full quantum state tomography from Pauli measurements.

    Generates measurement circuits, accepts raw counts, and
    reconstructs the density matrix via one of three methods.

    Parameters
    ----------
    num_qubits : int
        Number of qubits to characterise.

    Examples
    --------
    >>> tomo = StateTomographer(1)
    >>> circuits = tomo.measurement_circuits()
    >>> # ... run circuits on hardware / simulator ...
    >>> result = tomo.reconstruct(measurements, method='mle')
    >>> print(result.purity)
    """

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits
        self._circuits: list[MeasurementCircuit] | None = None

    def measurement_circuits(self) -> list[MeasurementCircuit]:
        """Generate the 3^n Pauli measurement circuits.

        Returns
        -------
        list[MeasurementCircuit]
            All measurement settings for full state tomography.
        """
        self._circuits = generate_measurement_circuits(self.num_qubits)
        return self._circuits

    def reconstruct(
        self,
        measurements: list[TomographyMeasurementResult],
        method: str = "mle",
        max_iterations: int = 1000,
        tol: float = 1e-8,
    ) -> StateTomographyResult:
        """Reconstruct the quantum state from measurement data.

        Parameters
        ----------
        measurements : list[TomographyMeasurementResult]
            One result per measurement circuit.
        method : str
            Reconstruction method: 'linear', 'mle', or 'lstsq'.
        max_iterations : int
            Maximum iterations for iterative methods.
        tol : float
            Convergence tolerance for iterative methods.

        Returns
        -------
        StateTomographyResult
            Reconstructed state with diagnostics.
        """
        if method == "linear":
            rho = self._linear_inversion(measurements)
            return StateTomographyResult(
                density_matrix=rho,
                method="linear",
                num_qubits=self.num_qubits,
                iterations=0,
            )
        elif method == "mle":
            rho, iters = self._mle(measurements, max_iterations, tol)
            return StateTomographyResult(
                density_matrix=rho,
                method="mle",
                num_qubits=self.num_qubits,
                iterations=iters,
            )
        elif method == "lstsq":
            rho, iters = self._least_squares(measurements, max_iterations, tol)
            return StateTomographyResult(
                density_matrix=rho,
                method="lstsq",
                num_qubits=self.num_qubits,
                iterations=iters,
            )
        else:
            raise ValueError(
                f"Unknown method {method!r}. Use 'linear', 'mle', or 'lstsq'."
            )

    # ------------------------------------------------------------------
    # Linear inversion
    # ------------------------------------------------------------------

    def _linear_inversion(
        self, measurements: list[TomographyMeasurementResult]
    ) -> np.ndarray:
        """Reconstruct density matrix via linear inversion.

        rho = (1/d) * sum_P  <P> * P

        where the sum runs over all n-qubit Pauli operators (including I)
        and <P> is the measured expectation value.
        """
        d = self.dim
        rho = np.zeros((d, d), dtype=np.complex128)

        # Build Pauli expectation values from measurement data
        pauli_expectations = self._extract_pauli_expectations(measurements)

        # Reconstruct: rho = (1/d) sum_{P} <P> P
        for labels, expectation in pauli_expectations.items():
            pauli_op = _pauli_tensor(labels)
            rho += expectation * pauli_op

        rho /= d
        return rho

    def _extract_pauli_expectations(
        self, measurements: list[TomographyMeasurementResult]
    ) -> dict[tuple[str, ...], float]:
        """Extract expectation values for all Pauli operators from data.

        Each measurement in a Pauli basis (e.g. X on qubit 0, Z on qubit 1)
        gives us expectation values for product operators built from
        {I, <basis>} on each qubit.

        Returns
        -------
        dict[tuple[str, ...], float]
            Mapping from Pauli label tuple to expectation value.
        """
        expectations: dict[tuple[str, ...], float] = {}
        # Identity always has expectation 1
        expectations[("I",) * self.num_qubits] = 1.0

        for meas in measurements:
            probs = meas.probabilities
            bases = meas.bases

            # For each subset of qubits, compute the expectation value
            # of the corresponding product Pauli operator
            for mask in range(1, 2**self.num_qubits):
                labels = []
                for q in range(self.num_qubits):
                    if mask & (1 << q):
                        labels.append(bases[q])
                    else:
                        labels.append("I")
                label_tuple = tuple(labels)

                # Expectation = sum_b (-1)^(parity of measured bits at masked positions) * prob(b)
                expectation = 0.0
                for bitstring, prob in probs.items():
                    parity = 0
                    for q in range(self.num_qubits):
                        if mask & (1 << q):
                            parity ^= int(bitstring[q])
                    expectation += ((-1) ** parity) * prob

                # Average if we see duplicate Pauli labels from different settings
                if label_tuple in expectations:
                    expectations[label_tuple] = (
                        expectations[label_tuple] + expectation
                    ) / 2.0
                else:
                    expectations[label_tuple] = expectation

        return expectations

    # ------------------------------------------------------------------
    # Maximum Likelihood Estimation (Hradil's algorithm)
    # ------------------------------------------------------------------

    def _mle(
        self,
        measurements: list[TomographyMeasurementResult],
        max_iterations: int,
        tol: float,
    ) -> tuple[np.ndarray, int]:
        """Reconstruct via MLE using Hradil's iterative algorithm.

        The iteration is:
            R = sum_i  (f_i / tr(Pi rho)) Pi
            rho_{k+1} = R rho_k R / tr(R rho_k R)

        where f_i are observed frequencies and Pi are measurement
        projectors.
        """
        d = self.dim

        # Build list of (projector, observed_frequency) pairs
        projectors, frequencies = self._build_projectors_and_frequencies(
            measurements
        )

        # Start from maximally mixed state
        rho = np.eye(d, dtype=np.complex128) / d

        for iteration in range(1, max_iterations + 1):
            # Build the R operator
            r_op = np.zeros((d, d), dtype=np.complex128)
            for proj, freq in zip(projectors, frequencies):
                tr_val = np.real(np.trace(proj @ rho))
                if tr_val > 1e-15:
                    r_op += (freq / tr_val) * proj

            # Iterate: rho_new = R rho R
            rho_new = r_op @ rho @ r_op

            # Normalise
            trace = np.real(np.trace(rho_new))
            if trace > 1e-15:
                rho_new /= trace
            else:
                break

            # Check convergence
            diff = np.linalg.norm(rho_new - rho, "fro")
            rho = rho_new
            if diff < tol:
                return rho, iteration

        return rho, max_iterations

    # ------------------------------------------------------------------
    # Least squares with physicality constraints
    # ------------------------------------------------------------------

    def _least_squares(
        self,
        measurements: list[TomographyMeasurementResult],
        max_iterations: int,
        tol: float,
    ) -> tuple[np.ndarray, int]:
        """Projected gradient descent with physicality constraints.

        Minimises ||A vec(rho) - b||^2 subject to rho >= 0, tr(rho) = 1.

        The projection step forces positive-semidefiniteness by clipping
        negative eigenvalues and re-normalising.
        """
        d = self.dim

        # Build the linear system from Pauli expectations
        pauli_expectations = self._extract_pauli_expectations(measurements)

        # Start from maximally mixed
        rho = np.eye(d, dtype=np.complex128) / d

        # Step size for gradient descent
        step_size = 0.1 / d

        for iteration in range(1, max_iterations + 1):
            # Compute gradient
            grad = np.zeros((d, d), dtype=np.complex128)
            for labels, target_val in pauli_expectations.items():
                pauli_op = _pauli_tensor(labels)
                current_val = np.real(np.trace(pauli_op @ rho))
                grad += (current_val - target_val) * pauli_op

            grad /= len(pauli_expectations)

            # Gradient step
            rho_new = rho - step_size * grad

            # Project onto physical states
            rho_new = self._project_to_physical(rho_new)

            # Check convergence
            diff = np.linalg.norm(rho_new - rho, "fro")
            rho = rho_new
            if diff < tol:
                return rho, iteration

        return rho, max_iterations

    @staticmethod
    def _project_to_physical(rho: np.ndarray) -> np.ndarray:
        """Project a matrix onto the set of valid density matrices.

        Clips negative eigenvalues to zero and renormalises trace to 1.
        """
        # Force Hermiticity
        rho = (rho + rho.conj().T) / 2.0

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(rho)

        # Clip negative eigenvalues
        eigvals = np.maximum(eigvals, 0.0)

        # Renormalise
        total = np.sum(eigvals)
        if total > 1e-15:
            eigvals /= total
        else:
            # Fallback to maximally mixed
            eigvals = np.ones(len(eigvals)) / len(eigvals)

        return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

    # ------------------------------------------------------------------
    # Projector construction
    # ------------------------------------------------------------------

    def _build_projectors_and_frequencies(
        self,
        measurements: list[TomographyMeasurementResult],
    ) -> tuple[list[np.ndarray], list[float]]:
        """Build measurement projectors and observed frequencies.

        Each bitstring outcome under a given basis setting defines a
        rank-1 projector |psi><psi| where |psi> is the tensor product
        of the corresponding Pauli eigenstates.
        """
        projectors: list[np.ndarray] = []
        frequencies: list[float] = []

        for meas in measurements:
            total_shots = meas.shots
            bases = meas.bases

            for bitstring, count in meas.counts.items():
                # Build the projector for this outcome
                proj = self._outcome_projector(bases, bitstring)
                projectors.append(proj)
                frequencies.append(count / total_shots)

        return projectors, frequencies

    def _outcome_projector(
        self, bases: tuple[str, ...], bitstring: str
    ) -> np.ndarray:
        """Construct the projector for a specific measurement outcome.

        Parameters
        ----------
        bases : tuple of str
            Pauli basis per qubit ('X', 'Y', or 'Z').
        bitstring : str
            Measured bitstring, e.g. '01'.

        Returns
        -------
        np.ndarray
            Rank-1 projector |psi><psi|.
        """
        state = np.array([1.0], dtype=np.complex128)

        for q in range(self.num_qubits):
            bit = int(bitstring[q])
            eigenstate = self._pauli_eigenstate(bases[q], bit)
            state = np.kron(state, eigenstate)

        return np.outer(state, state.conj())

    @staticmethod
    def _pauli_eigenstate(basis: str, outcome: int) -> np.ndarray:
        """Return the eigenstate of a Pauli operator for a given outcome.

        Parameters
        ----------
        basis : str
            'X', 'Y', or 'Z'.
        outcome : int
            0 for +1 eigenvalue, 1 for -1 eigenvalue.

        Returns
        -------
        np.ndarray
            Column vector (length 2).
        """
        if basis == "Z":
            if outcome == 0:
                return np.array([1.0, 0.0], dtype=np.complex128)
            else:
                return np.array([0.0, 1.0], dtype=np.complex128)
        elif basis == "X":
            if outcome == 0:
                return np.array([_SQRT2_INV, _SQRT2_INV], dtype=np.complex128)
            else:
                return np.array([_SQRT2_INV, -_SQRT2_INV], dtype=np.complex128)
        elif basis == "Y":
            if outcome == 0:
                return np.array([_SQRT2_INV, 1j * _SQRT2_INV], dtype=np.complex128)
            else:
                return np.array([_SQRT2_INV, -1j * _SQRT2_INV], dtype=np.complex128)
        else:
            raise ValueError(f"Unknown basis: {basis!r}")


# ---------------------------------------------------------------------------
# Synthetic measurement data generator (for testing)
# ---------------------------------------------------------------------------


def simulate_tomography_measurements(
    state: np.ndarray,
    circuits: list[MeasurementCircuit],
    shots: int = 10000,
    rng: np.random.Generator | None = None,
) -> list[TomographyMeasurementResult]:
    """Generate synthetic tomography measurement data from a known state.

    Simulates ideal (noiseless) measurements by computing Born-rule
    probabilities and sampling from them.

    Parameters
    ----------
    state : np.ndarray
        State vector or density matrix of the state to measure.
    circuits : list[MeasurementCircuit]
        Measurement settings from ``generate_measurement_circuits``.
    shots : int
        Number of measurement shots per setting.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    list[TomographyMeasurementResult]
        Simulated measurement results.
    """
    if rng is None:
        rng = np.random.default_rng()

    rho = _ensure_density_matrix(state)
    num_qubits = int(np.log2(rho.shape[0]))
    dim = 2**num_qubits

    results: list[TomographyMeasurementResult] = []

    for circuit in circuits:
        # Build the full rotation for this measurement setting
        rotation = np.eye(dim, dtype=np.complex128)
        for qubit_idx, gate in circuit.rotations:
            full_gate = _embed_single_qubit(gate, qubit_idx, num_qubits)
            rotation = full_gate @ rotation

        # Rotate the state
        rho_rotated = rotation @ rho @ rotation.conj().T

        # Born-rule probabilities in the computational basis
        probs = np.real(np.diag(rho_rotated))
        probs = np.maximum(probs, 0.0)
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs /= prob_sum

        # Sample
        outcomes = rng.choice(dim, size=shots, p=probs)
        counts: dict[str, int] = {}
        for outcome in outcomes:
            bs = format(outcome, f"0{num_qubits}b")
            counts[bs] = counts.get(bs, 0) + 1

        # Use Pauli-only basis labels for standard circuits
        pauli_bases = tuple(
            b if b in ("X", "Y", "Z") else "Z" for b in circuit.bases
        )

        results.append(
            TomographyMeasurementResult(
                bases=pauli_bases,
                counts=counts,
                shots=shots,
            )
        )

    return results


def _embed_single_qubit(
    gate: np.ndarray, qubit: int, num_qubits: int
) -> np.ndarray:
    """Embed a 2x2 gate into the full 2^n Hilbert space."""
    result = np.array([[1.0]], dtype=np.complex128)
    i2 = np.eye(2, dtype=np.complex128)
    for q in range(num_qubits):
        result = np.kron(result, gate if q == qubit else i2)
    return result
