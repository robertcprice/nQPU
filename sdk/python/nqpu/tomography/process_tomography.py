"""Quantum process tomography for channel characterisation.

Reconstructs the chi-matrix (process matrix) and Choi matrix representations
of an unknown quantum channel from input-state preparation and output-state
measurement data.

The protocol works by:
1. Preparing a tomographically complete set of input states (6 per qubit:
   |0>, |1>, |+>, |->, |+i>, |-i>).
2. For each input state, performing full state tomography on the output.
3. Inverting the resulting linear system to obtain the chi-matrix in the
   Pauli basis.

References:
    - Chuang & Nielsen, J. Mod. Opt. 44, 2455 (1997) [Process tomography]
    - Choi, Lin. Alg. Appl. 10, 285 (1975) [Choi-Jamiolkowski isomorphism]
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
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
    state_fidelity,
    StateTomographer,
    TomographyMeasurementResult,
)

# ---------------------------------------------------------------------------
# Standard input states for process tomography
# ---------------------------------------------------------------------------

_SQRT2_INV = 1.0 / math.sqrt(2.0)

# 6 informationally-complete single-qubit input states
_INPUT_STATES_1Q = {
    "Z+": np.array([1.0, 0.0], dtype=np.complex128),           # |0>
    "Z-": np.array([0.0, 1.0], dtype=np.complex128),           # |1>
    "X+": np.array([_SQRT2_INV, _SQRT2_INV], dtype=np.complex128),    # |+>
    "X-": np.array([_SQRT2_INV, -_SQRT2_INV], dtype=np.complex128),   # |->
    "Y+": np.array([_SQRT2_INV, 1j * _SQRT2_INV], dtype=np.complex128),  # |+i>
    "Y-": np.array([_SQRT2_INV, -1j * _SQRT2_INV], dtype=np.complex128), # |-i>
}


def _make_input_state(labels: tuple[str, ...]) -> np.ndarray:
    """Construct a multi-qubit input state from per-qubit labels.

    Parameters
    ----------
    labels : tuple of str
        Per-qubit state label, e.g. ('Z+', 'X-').

    Returns
    -------
    np.ndarray
        Tensor-product state vector.
    """
    state = np.array([1.0], dtype=np.complex128)
    for label in labels:
        state = np.kron(state, _INPUT_STATES_1Q[label])
    return state


# ---------------------------------------------------------------------------
# Process tomography circuit description
# ---------------------------------------------------------------------------


@dataclass
class ProcessCircuit:
    """Description of a process tomography experiment.

    Attributes
    ----------
    input_labels : tuple[str, ...]
        Per-qubit input state label.
    input_state : np.ndarray
        Input state vector.
    measurement_bases : tuple[str, ...]
        Output Pauli measurement bases.
    """

    input_labels: tuple[str, ...]
    input_state: np.ndarray
    measurement_bases: tuple[str, ...]


# ---------------------------------------------------------------------------
# Process tomography result
# ---------------------------------------------------------------------------


@dataclass
class ProcessTomographyResult:
    """Result of quantum process tomography.

    Attributes
    ----------
    chi_matrix : np.ndarray
        Process matrix in the Pauli basis (d^2 x d^2).
    choi_matrix : np.ndarray
        Choi representation of the channel (d^2 x d^2).
    num_qubits : int
        Number of qubits.
    """

    chi_matrix: np.ndarray
    choi_matrix: np.ndarray
    num_qubits: int

    @property
    def average_gate_fidelity(self) -> float:
        """Average gate fidelity of the process with the identity channel.

        F_avg = (d * F_ent + 1) / (d + 1)

        where F_ent is the entanglement fidelity (overlap of the Choi
        matrix with the maximally entangled state).
        """
        d = 2**self.num_qubits
        # Entanglement fidelity = tr(choi_ideal^T . choi) / d
        # For identity channel, choi_ideal = |Phi+><Phi+| where |Phi+> = sum|ii>/sqrt(d)
        phi_plus = np.zeros(d * d, dtype=np.complex128)
        for i in range(d):
            phi_plus[i * d + i] = 1.0 / math.sqrt(d)
        choi_ideal = np.outer(phi_plus, phi_plus.conj())

        f_ent = float(np.real(np.trace(choi_ideal @ self.choi_matrix)))
        f_avg = (d * f_ent + 1.0) / (d + 1.0)
        return float(np.clip(f_avg, 0.0, 1.0))

    def gate_fidelity_with(self, target_unitary: np.ndarray) -> float:
        """Average gate fidelity with a target unitary.

        Parameters
        ----------
        target_unitary : np.ndarray
            The ideal unitary gate matrix (d x d).

        Returns
        -------
        float
            Average gate fidelity in [0, 1].
        """
        d = 2**self.num_qubits
        # Build Choi matrix of target unitary channel
        target_choi = _unitary_to_choi(target_unitary)

        f_ent = float(np.real(np.trace(target_choi @ self.choi_matrix)))
        f_avg = (d * f_ent + 1.0) / (d + 1.0)
        return float(np.clip(f_avg, 0.0, 1.0))

    @property
    def diamond_norm_estimate(self) -> float:
        """Rough upper bound on the diamond norm distance to identity.

        Uses the relation ||E - I||_diamond <= d * ||choi_E - choi_I||_1
        where ||.||_1 is the trace norm.  This is an upper bound, not exact.

        Returns
        -------
        float
            Estimated diamond norm distance.
        """
        d = 2**self.num_qubits
        phi_plus = np.zeros(d * d, dtype=np.complex128)
        for i in range(d):
            phi_plus[i * d + i] = 1.0 / math.sqrt(d)
        choi_ideal = np.outer(phi_plus, phi_plus.conj())

        diff = self.choi_matrix - choi_ideal
        singular_values = np.linalg.svd(diff, compute_uv=False)
        trace_norm = float(np.sum(singular_values))
        return d * trace_norm

    @property
    def is_physical(self) -> bool:
        """Check if the process is completely positive and trace-preserving.

        A physical channel has:
        - Choi matrix is positive semidefinite (complete positivity)
        - Partial trace over output = I/d (trace preserving)
        """
        d = 2**self.num_qubits

        # Positive semidefinite check
        eigvals = np.linalg.eigvalsh(self.choi_matrix)
        if np.any(eigvals < -1e-8):
            return False

        # Trace preserving: partial trace over output should be I
        partial_trace = _partial_trace_output(self.choi_matrix, d)
        if not np.allclose(partial_trace, np.eye(d) / d, atol=1e-6):
            # Note: the normalisation convention may vary
            # Check if proportional to identity
            diag = np.diag(partial_trace)
            if not np.allclose(diag, diag[0], atol=1e-6):
                return False

        return True


# ---------------------------------------------------------------------------
# Choi matrix utilities
# ---------------------------------------------------------------------------


def _unitary_to_choi(unitary: np.ndarray) -> np.ndarray:
    """Compute the Choi matrix of a unitary channel.

    For a unitary U, the Choi matrix is:
        choi = (I tensor U) |Phi+><Phi+| (I tensor U^dag)

    Parameters
    ----------
    unitary : np.ndarray
        Unitary gate matrix (d x d).

    Returns
    -------
    np.ndarray
        Choi matrix (d^2 x d^2).
    """
    d = unitary.shape[0]
    # Maximally entangled state |Phi+> = (1/sqrt(d)) sum_i |i>|i>
    phi_plus = np.zeros(d * d, dtype=np.complex128)
    for i in range(d):
        phi_plus[i * d + i] = 1.0 / math.sqrt(d)

    # Apply I tensor U
    iu = np.kron(np.eye(d, dtype=np.complex128), unitary)
    state = iu @ phi_plus
    return np.outer(state, state.conj())


def _partial_trace_output(
    choi: np.ndarray, d: int
) -> np.ndarray:
    """Partial trace over the output system of a Choi matrix.

    Treats the d^2 x d^2 Choi matrix as living on H_in tensor H_out,
    and traces out H_out.

    Parameters
    ----------
    choi : np.ndarray
        Choi matrix (d^2 x d^2).
    d : int
        Dimension of the single system.

    Returns
    -------
    np.ndarray
        Reduced matrix on H_in (d x d).
    """
    # Reshape to (d, d, d, d) with indices (in1, out1, in2, out2)
    reshaped = choi.reshape(d, d, d, d)
    # Trace over output indices (1 and 3)
    return np.trace(reshaped, axis1=1, axis2=3)


def chi_to_choi(
    chi: np.ndarray, num_qubits: int
) -> np.ndarray:
    """Convert a chi-matrix (Pauli basis) to a Choi matrix.

    chi_ij encodes the channel as E(rho) = sum_{ij} chi_ij P_i rho P_j
    where P_i are the n-qubit Pauli operators.

    The Choi matrix is:
        choi = sum_{ij} chi_ij  (P_i tensor I) |Phi+><Phi+| (P_j tensor I)^dag

    Parameters
    ----------
    chi : np.ndarray
        Process matrix in the Pauli basis (d^2 x d^2).
    num_qubits : int
        Number of qubits.

    Returns
    -------
    np.ndarray
        Choi matrix (d^2 x d^2).
    """
    d = 2**num_qubits
    paulis = _all_pauli_operators(num_qubits)

    # Maximally entangled state
    phi_plus = np.zeros(d * d, dtype=np.complex128)
    for i in range(d):
        phi_plus[i * d + i] = 1.0 / math.sqrt(d)
    phi_proj = np.outer(phi_plus, phi_plus.conj())

    choi = np.zeros((d * d, d * d), dtype=np.complex128)
    for i, pi in enumerate(paulis):
        for j, pj in enumerate(paulis):
            if abs(chi[i, j]) < 1e-15:
                continue
            pi_kron_i = np.kron(pi, np.eye(d, dtype=np.complex128))
            pj_kron_i = np.kron(pj, np.eye(d, dtype=np.complex128))
            choi += chi[i, j] * (pi_kron_i @ phi_proj @ pj_kron_i.conj().T)

    return choi


def choi_to_chi(
    choi: np.ndarray, num_qubits: int
) -> np.ndarray:
    """Convert a Choi matrix to a chi-matrix (Pauli basis).

    Inverts the ``chi_to_choi`` transformation.  Uses the orthogonality
    of the vectorised Pauli basis:

        chi_ij = d * <phi_i | choi | phi_j>

    where |phi_i> = (P_i tensor I)|Phi+> and the factor d comes from
    the normalisation <phi_i|phi_j> = delta_ij / d.

    Parameters
    ----------
    choi : np.ndarray
        Choi matrix (d^2 x d^2).
    num_qubits : int
        Number of qubits.

    Returns
    -------
    np.ndarray
        Chi matrix in the Pauli basis (d^2 x d^2).
    """
    d = 2**num_qubits
    paulis = _all_pauli_operators(num_qubits)
    n_paulis = len(paulis)

    # Maximally entangled state
    phi_plus = np.zeros(d * d, dtype=np.complex128)
    for i in range(d):
        phi_plus[i * d + i] = 1.0 / math.sqrt(d)

    # Precompute |phi_i> = (P_i tensor I)|Phi+>
    phi_vecs = []
    for pi in paulis:
        pi_kron_i = np.kron(pi, np.eye(d, dtype=np.complex128))
        phi_vecs.append(pi_kron_i @ phi_plus)

    chi = np.zeros((n_paulis, n_paulis), dtype=np.complex128)
    for i in range(n_paulis):
        for j in range(n_paulis):
            chi[i, j] = phi_vecs[i].conj() @ choi @ phi_vecs[j]

    return chi


def _all_pauli_operators(num_qubits: int) -> list[np.ndarray]:
    """Generate all d^2 n-qubit Pauli operators in canonical order.

    Order: I...I, I...X, I...Y, I...Z, I..XI, ..., Z...Z

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    Returns
    -------
    list[np.ndarray]
        List of d^2 Pauli matrices, each d x d.
    """
    labels = list(itertools.product("IXYZ", repeat=num_qubits))
    return [_pauli_tensor(label) for label in labels]


# ---------------------------------------------------------------------------
# ProcessTomographer
# ---------------------------------------------------------------------------


class ProcessTomographer:
    """Quantum process tomography from input-state preparation and measurement.

    Characterises an unknown quantum channel by preparing a complete
    set of input states, passing them through the channel, and
    performing state tomography on each output.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    Examples
    --------
    >>> pt = ProcessTomographer(1)
    >>> input_states = pt.input_state_labels()
    >>> # For each input state, run the process and do state tomography
    >>> result = pt.reconstruct(output_density_matrices)
    """

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits
        self._input_labels_list: list[tuple[str, ...]] = list(
            itertools.product(
                ["Z+", "Z-", "X+", "X-", "Y+", "Y-"],
                repeat=num_qubits,
            )
        )

    def input_state_labels(self) -> list[tuple[str, ...]]:
        """Return the labels for all input states.

        Returns
        -------
        list[tuple[str, ...]]
            List of per-qubit state labels.
        """
        return self._input_labels_list

    def input_states(self) -> list[np.ndarray]:
        """Return density matrices for all input states.

        Returns
        -------
        list[np.ndarray]
            List of input density matrices.
        """
        states = []
        for labels in self._input_labels_list:
            sv = _make_input_state(labels)
            states.append(np.outer(sv, sv.conj()))
        return states

    def generate_process_circuits(
        self,
    ) -> list[ProcessCircuit]:
        """Generate all input-state / measurement-basis combinations.

        Returns
        -------
        list[ProcessCircuit]
            One entry per (input_state, measurement_basis) pair.
        """
        circuits: list[ProcessCircuit] = []
        measurement_bases = list(
            itertools.product("XYZ", repeat=self.num_qubits)
        )

        for input_labels in self._input_labels_list:
            input_sv = _make_input_state(input_labels)
            for meas_bases in measurement_bases:
                circuits.append(
                    ProcessCircuit(
                        input_labels=input_labels,
                        input_state=input_sv,
                        measurement_bases=meas_bases,
                    )
                )

        return circuits

    def reconstruct(
        self,
        output_states: dict[tuple[str, ...], np.ndarray],
    ) -> ProcessTomographyResult:
        """Reconstruct the process from output density matrices.

        Given the output density matrix for each input state, solves
        for the chi-matrix in the Pauli basis.

        Parameters
        ----------
        output_states : dict[tuple[str, ...], np.ndarray]
            Mapping from input state labels to output density matrices.

        Returns
        -------
        ProcessTomographyResult
            Reconstructed process with chi and Choi matrices.
        """
        d = self.dim
        d2 = d * d
        paulis = _all_pauli_operators(self.num_qubits)

        # Build the linear system: for each input rho_in,
        # E(rho_in) = sum_{ij} chi_ij P_i rho_in P_j
        # Taking tr(P_k E(rho_in)) for each Pauli P_k and input rho_in,
        # we get a system of equations for chi.

        n_inputs = len(self._input_labels_list)
        n_equations = n_inputs * d2

        a_matrix = np.zeros((n_equations, d2 * d2), dtype=np.complex128)
        b_vector = np.zeros(n_equations, dtype=np.complex128)

        eq_idx = 0
        for input_labels in self._input_labels_list:
            sv_in = _make_input_state(input_labels)
            rho_in = np.outer(sv_in, sv_in.conj())

            if input_labels not in output_states:
                raise ValueError(
                    f"Missing output state for input {input_labels}"
                )
            rho_out = _ensure_density_matrix(output_states[input_labels])

            for k, pk in enumerate(paulis):
                b_vector[eq_idx] = np.trace(pk @ rho_out)

                for i, pi in enumerate(paulis):
                    for j, pj in enumerate(paulis):
                        a_matrix[eq_idx, i * d2 + j] = np.trace(
                            pk @ pi @ rho_in @ pj
                        )
                eq_idx += 1

        # Solve via least squares
        chi_flat, _, _, _ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
        chi = chi_flat.reshape(d2, d2)

        # Enforce Hermiticity
        chi = (chi + chi.conj().T) / 2.0

        # Convert to Choi
        choi = chi_to_choi(chi, self.num_qubits)

        return ProcessTomographyResult(
            chi_matrix=chi,
            choi_matrix=choi,
            num_qubits=self.num_qubits,
        )

    def reconstruct_from_measurements(
        self,
        measurements: dict[
            tuple[str, ...], list[TomographyMeasurementResult]
        ],
        method: str = "mle",
    ) -> ProcessTomographyResult:
        """Reconstruct process from raw measurement data.

        First performs state tomography on each output, then reconstructs
        the process matrix.

        Parameters
        ----------
        measurements : dict
            Mapping from input state labels to list of tomography
            measurement results for the output state.
        method : str
            State tomography reconstruction method ('linear', 'mle', 'lstsq').

        Returns
        -------
        ProcessTomographyResult
            Reconstructed process.
        """
        tomo = StateTomographer(self.num_qubits)

        output_states: dict[tuple[str, ...], np.ndarray] = {}
        for input_labels, meas_data in measurements.items():
            result = tomo.reconstruct(meas_data, method=method)
            output_states[input_labels] = result.density_matrix

        return self.reconstruct(output_states)


# ---------------------------------------------------------------------------
# Simulate process tomography (for testing)
# ---------------------------------------------------------------------------


def simulate_process_tomography(
    channel: np.ndarray,
    num_qubits: int,
) -> dict[tuple[str, ...], np.ndarray]:
    """Apply a unitary channel to all input states for testing.

    Parameters
    ----------
    channel : np.ndarray
        Unitary matrix representing the quantum channel.
    num_qubits : int
        Number of qubits.

    Returns
    -------
    dict[tuple[str, ...], np.ndarray]
        Mapping from input state labels to output density matrices.
    """
    pt = ProcessTomographer(num_qubits)
    output_states: dict[tuple[str, ...], np.ndarray] = {}

    for labels in pt.input_state_labels():
        sv_in = _make_input_state(labels)
        sv_out = channel @ sv_in
        output_states[labels] = np.outer(sv_out, sv_out.conj())

    return output_states
