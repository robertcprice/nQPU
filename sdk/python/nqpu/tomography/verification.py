"""Quantum state and process verification metrics.

Provides a comprehensive set of functions for verifying and
characterising quantum states, processes, and correlations.

Includes:
- State fidelity (Uhlmann)
- Process fidelity (average gate fidelity)
- Purity
- Von Neumann entropy
- Concurrence (2-qubit entanglement)
- Entanglement witnesses
- Quantum volume estimation
- Cross-entropy benchmarking (XEB)

All functions operate on numpy arrays (density matrices, unitaries)
and require no external dependencies beyond numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .state_tomography import (
    _I,
    _X,
    _Y,
    _Z,
    _ensure_density_matrix,
    state_fidelity,
)

# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def purity(rho: np.ndarray) -> float:
    """Compute the purity tr(rho^2) of a quantum state.

    Purity ranges from 1/d (maximally mixed) to 1 (pure state),
    where d is the Hilbert space dimension.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix or state vector.

    Returns
    -------
    float
        Purity in [1/d, 1].
    """
    rho = _ensure_density_matrix(rho)
    return float(np.real(np.trace(rho @ rho)))


# ---------------------------------------------------------------------------
# Von Neumann entropy
# ---------------------------------------------------------------------------


def von_neumann_entropy(rho: np.ndarray, base: str = "e") -> float:
    """Compute the Von Neumann entropy S(rho) = -tr(rho log rho).

    Parameters
    ----------
    rho : np.ndarray
        Density matrix or state vector.
    base : str
        Logarithm base: 'e' (nats), '2' (bits), or '10'.

    Returns
    -------
    float
        Entropy (non-negative).  Zero for pure states, log(d) for
        maximally mixed states.
    """
    rho = _ensure_density_matrix(rho)
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]

    if base == "e":
        return float(-np.sum(eigvals * np.log(eigvals)))
    elif base == "2":
        return float(-np.sum(eigvals * np.log2(eigvals)))
    elif base == "10":
        return float(-np.sum(eigvals * np.log10(eigvals)))
    else:
        raise ValueError(f"Unknown base {base!r}. Use 'e', '2', or '10'.")


# ---------------------------------------------------------------------------
# Concurrence
# ---------------------------------------------------------------------------


def concurrence(rho: np.ndarray) -> float:
    """Compute the concurrence of a 2-qubit state.

    The concurrence is an entanglement measure defined as:
        C(rho) = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)

    where lambda_i are the square roots of the eigenvalues (in
    decreasing order) of  rho . tilde{rho}  where
    tilde{rho} = (Y tensor Y) rho* (Y tensor Y).

    Parameters
    ----------
    rho : np.ndarray
        Density matrix of a 2-qubit system (4x4).

    Returns
    -------
    float
        Concurrence in [0, 1].  0 for separable, 1 for maximally entangled.

    Raises
    ------
    ValueError
        If rho is not 4x4 (not a 2-qubit state).
    """
    rho = _ensure_density_matrix(rho)

    if rho.shape != (4, 4):
        raise ValueError(
            f"Concurrence is defined for 2-qubit states (4x4 matrices). "
            f"Got shape {rho.shape}."
        )

    # sigma_y tensor sigma_y
    yy = np.kron(_Y, _Y)

    # rho_tilde = (Y tensor Y) rho* (Y tensor Y)
    rho_tilde = yy @ rho.conj() @ yy

    # Product rho . rho_tilde
    product = rho @ rho_tilde

    # Eigenvalues (may be complex due to numerical errors)
    eigvals = np.linalg.eigvals(product)
    eigvals = np.real(eigvals)
    eigvals = np.maximum(eigvals, 0.0)

    # Square roots in decreasing order
    sqrt_eigvals = np.sqrt(eigvals)
    sqrt_eigvals = np.sort(sqrt_eigvals)[::-1]

    c = sqrt_eigvals[0] - np.sum(sqrt_eigvals[1:])
    return float(max(0.0, c))


# ---------------------------------------------------------------------------
# Entanglement of formation (from concurrence)
# ---------------------------------------------------------------------------


def entanglement_of_formation(rho: np.ndarray) -> float:
    """Compute the entanglement of formation for a 2-qubit state.

    EoF = h((1 + sqrt(1 - C^2)) / 2)

    where h(x) = -x log(x) - (1-x) log(1-x) is the binary entropy
    and C is the concurrence.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix of a 2-qubit system (4x4).

    Returns
    -------
    float
        Entanglement of formation in [0, 1] (in bits with log base 2).
    """
    c = concurrence(rho)
    if c < 1e-15:
        return 0.0

    x = (1.0 + math.sqrt(1.0 - c * c)) / 2.0

    # Binary entropy in bits
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return float(-x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x))


# ---------------------------------------------------------------------------
# Entanglement witness
# ---------------------------------------------------------------------------


@dataclass
class EntanglementWitnessResult:
    """Result of an entanglement witness test.

    Attributes
    ----------
    witness_value : float
        tr(W . rho).  Negative value certifies entanglement.
    is_entangled : bool
        Whether the witness detected entanglement.
    witness_operator : np.ndarray
        The witness operator used.
    """

    witness_value: float
    is_entangled: bool
    witness_operator: np.ndarray


def entanglement_witness_ghz(
    rho: np.ndarray, num_qubits: int
) -> EntanglementWitnessResult:
    """Test entanglement using a GHZ-state witness.

    The witness is W = (1/2)I - |GHZ><GHZ| where
    |GHZ> = (|00...0> + |11...1>) / sqrt(2).

    tr(W . rho) < 0 certifies genuine multipartite entanglement.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (2^n x 2^n).
    num_qubits : int
        Number of qubits.

    Returns
    -------
    EntanglementWitnessResult
        Witness test result.
    """
    rho = _ensure_density_matrix(rho)
    d = 2**num_qubits

    if rho.shape != (d, d):
        raise ValueError(
            f"Expected {d}x{d} density matrix for {num_qubits} qubits, "
            f"got {rho.shape}"
        )

    # |GHZ> = (|00...0> + |11...1>) / sqrt(2)
    ghz = np.zeros(d, dtype=np.complex128)
    ghz[0] = 1.0 / math.sqrt(2.0)
    ghz[d - 1] = 1.0 / math.sqrt(2.0)
    ghz_proj = np.outer(ghz, ghz.conj())

    witness = 0.5 * np.eye(d, dtype=np.complex128) - ghz_proj
    witness_val = float(np.real(np.trace(witness @ rho)))

    return EntanglementWitnessResult(
        witness_value=witness_val,
        is_entangled=witness_val < -1e-10,
        witness_operator=witness,
    )


def entanglement_witness_bell(rho: np.ndarray) -> EntanglementWitnessResult:
    """Test entanglement using a Bell-state witness.

    The witness is W = (1/2)I - |Phi+><Phi+| where
    |Phi+> = (|00> + |11>) / sqrt(2).

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (4x4, 2-qubit state).

    Returns
    -------
    EntanglementWitnessResult
        Witness test result.
    """
    rho = _ensure_density_matrix(rho)
    if rho.shape != (4, 4):
        raise ValueError(f"Expected 4x4 density matrix, got {rho.shape}")

    bell = np.array(
        [1.0 / math.sqrt(2.0), 0.0, 0.0, 1.0 / math.sqrt(2.0)],
        dtype=np.complex128,
    )
    bell_proj = np.outer(bell, bell.conj())
    witness = 0.5 * np.eye(4, dtype=np.complex128) - bell_proj
    witness_val = float(np.real(np.trace(witness @ rho)))

    return EntanglementWitnessResult(
        witness_value=witness_val,
        is_entangled=witness_val < -1e-10,
        witness_operator=witness,
    )


# ---------------------------------------------------------------------------
# Trace distance
# ---------------------------------------------------------------------------


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the trace distance between two density matrices.

    T(rho, sigma) = (1/2) ||rho - sigma||_1

    where ||.||_1 is the trace norm (sum of singular values).

    Parameters
    ----------
    rho, sigma : np.ndarray
        Density matrices or state vectors.

    Returns
    -------
    float
        Trace distance in [0, 1].
    """
    rho = _ensure_density_matrix(rho)
    sigma = _ensure_density_matrix(sigma)
    diff = rho - sigma
    singular_values = np.linalg.svd(diff, compute_uv=False)
    return float(0.5 * np.sum(singular_values))


# ---------------------------------------------------------------------------
# Relative entropy
# ---------------------------------------------------------------------------


def relative_entropy(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the quantum relative entropy S(rho || sigma).

    S(rho || sigma) = tr(rho (log rho - log sigma))

    Returns infinity if the support of rho is not contained in
    the support of sigma.

    Parameters
    ----------
    rho, sigma : np.ndarray
        Density matrices.

    Returns
    -------
    float
        Relative entropy (non-negative, possibly infinite).
    """
    rho = _ensure_density_matrix(rho)
    sigma = _ensure_density_matrix(sigma)

    # Eigendecompositions
    eigvals_rho, eigvecs_rho = np.linalg.eigh(rho)
    eigvals_sigma, eigvecs_sigma = np.linalg.eigh(sigma)

    # Check support condition
    for i, ev_rho in enumerate(eigvals_rho):
        if ev_rho > 1e-12:
            # This eigenvector of rho must be in the support of sigma
            vec = eigvecs_rho[:, i]
            overlap = vec.conj() @ sigma @ vec
            if np.real(overlap) < 1e-12:
                return float("inf")

    # Compute log matrices
    log_rho = _matrix_log(rho)
    log_sigma = _matrix_log(sigma)

    result = np.trace(rho @ (log_rho - log_sigma))
    return float(max(0.0, np.real(result)))


def _matrix_log(matrix: np.ndarray) -> np.ndarray:
    """Compute the matrix logarithm via eigendecomposition.

    Clips zero/negative eigenvalues for numerical stability.
    """
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 1e-30)
    return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.conj().T


# ---------------------------------------------------------------------------
# Process fidelity
# ---------------------------------------------------------------------------


def average_gate_fidelity(
    channel_choi: np.ndarray,
    target_unitary: np.ndarray,
) -> float:
    """Compute the average gate fidelity between a channel and a target gate.

    F_avg = (d * F_ent + 1) / (d + 1)

    where F_ent = <Phi+| (I tensor U^dag) . choi . (I tensor U) |Phi+>
    is the entanglement fidelity.

    Parameters
    ----------
    channel_choi : np.ndarray
        Choi matrix of the actual channel (d^2 x d^2).
    target_unitary : np.ndarray
        Target unitary gate matrix (d x d).

    Returns
    -------
    float
        Average gate fidelity in [0, 1].
    """
    d = target_unitary.shape[0]

    # Maximally entangled state |Phi+>
    phi_plus = np.zeros(d * d, dtype=np.complex128)
    for i in range(d):
        phi_plus[i * d + i] = 1.0 / math.sqrt(d)

    # Apply (I tensor U^dag) on the Choi state
    iu_dag = np.kron(np.eye(d, dtype=np.complex128), target_unitary.conj().T)
    iu = np.kron(np.eye(d, dtype=np.complex128), target_unitary)

    rotated_choi = iu_dag @ channel_choi @ iu

    # Entanglement fidelity
    f_ent = float(np.real(phi_plus.conj() @ rotated_choi @ phi_plus))

    f_avg = (d * f_ent + 1.0) / (d + 1.0)
    return float(np.clip(f_avg, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Quantum Volume
# ---------------------------------------------------------------------------


@dataclass
class QuantumVolumeResult:
    """Result of quantum volume estimation.

    Attributes
    ----------
    log2_quantum_volume : int
        Estimated log2(QV).
    heavy_output_probability : float
        Fraction of heavy outputs observed.
    threshold : float
        Threshold for heavy-output probability (2/3).
    is_achieved : bool
        Whether QV is achieved (heavy_output_prob > 2/3).
    num_circuits : int
        Number of random circuits used.
    """

    log2_quantum_volume: int
    heavy_output_probability: float
    threshold: float = 2.0 / 3.0
    is_achieved: bool = False
    num_circuits: int = 0


def estimate_quantum_volume(
    num_qubits: int,
    circuit_outputs: list[dict[str, int]],
    ideal_probabilities: list[np.ndarray],
) -> QuantumVolumeResult:
    """Estimate quantum volume from random circuit measurements.

    Quantum volume is determined by whether the heavy output
    probability exceeds 2/3 for random SU(4) circuits of depth
    num_qubits.

    A "heavy output" for a circuit is a bitstring whose ideal
    probability exceeds the median ideal probability.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (circuit width = depth).
    circuit_outputs : list[dict[str, int]]
        Measurement histograms from each random circuit.
    ideal_probabilities : list[np.ndarray]
        Ideal output probability distributions for each circuit.

    Returns
    -------
    QuantumVolumeResult
        QV estimation result.
    """
    heavy_probs: list[float] = []

    for counts, ideal_probs in zip(circuit_outputs, ideal_probabilities):
        median_prob = float(np.median(ideal_probs))
        total_shots = sum(counts.values())

        # Identify heavy outputs (ideal probability > median)
        heavy_count = 0
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            if ideal_probs[idx] > median_prob:
                heavy_count += count

        heavy_probs.append(heavy_count / total_shots if total_shots > 0 else 0.0)

    mean_heavy = float(np.mean(heavy_probs)) if heavy_probs else 0.0
    is_achieved = mean_heavy > 2.0 / 3.0

    return QuantumVolumeResult(
        log2_quantum_volume=num_qubits if is_achieved else 0,
        heavy_output_probability=mean_heavy,
        is_achieved=is_achieved,
        num_circuits=len(circuit_outputs),
    )


# ---------------------------------------------------------------------------
# Cross-Entropy Benchmarking (XEB)
# ---------------------------------------------------------------------------


@dataclass
class XEBResult:
    """Result of cross-entropy benchmarking.

    Attributes
    ----------
    xeb_fidelity : float
        Linear cross-entropy fidelity.
    num_circuits : int
        Number of random circuits used.
    mean_log_xeb : float
        Mean logarithmic cross-entropy (log-XEB).
    """

    xeb_fidelity: float
    num_circuits: int
    mean_log_xeb: float = 0.0


def cross_entropy_benchmark(
    circuit_outputs: list[dict[str, int]],
    ideal_probabilities: list[np.ndarray],
    num_qubits: int,
) -> XEBResult:
    """Compute the linear cross-entropy benchmarking fidelity.

    The linear XEB fidelity is:
        F_XEB = d * <p_ideal(x)>_observed - 1

    where <p_ideal(x)>_observed is the average ideal probability of
    the bitstrings actually measured, and d = 2^n is the Hilbert
    space dimension.

    Parameters
    ----------
    circuit_outputs : list[dict[str, int]]
        Measurement histograms from random circuits.
    ideal_probabilities : list[np.ndarray]
        Ideal output probability distributions.
    num_qubits : int
        Number of qubits.

    Returns
    -------
    XEBResult
        XEB fidelity result.
    """
    d = 2**num_qubits
    xeb_values: list[float] = []
    log_xeb_values: list[float] = []

    for counts, ideal_probs in zip(circuit_outputs, ideal_probabilities):
        total_shots = sum(counts.values())
        if total_shots == 0:
            continue

        # Average ideal probability of observed outcomes
        avg_ideal_prob = 0.0
        avg_log_prob = 0.0
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            p = ideal_probs[idx]
            avg_ideal_prob += p * count
            if p > 1e-30:
                avg_log_prob += math.log(p) * count

        avg_ideal_prob /= total_shots
        avg_log_prob /= total_shots

        # Linear XEB for this circuit
        xeb_values.append(d * avg_ideal_prob - 1.0)
        log_xeb_values.append(avg_log_prob + math.log(d) + np.euler_gamma)

    if not xeb_values:
        return XEBResult(xeb_fidelity=0.0, num_circuits=0)

    return XEBResult(
        xeb_fidelity=float(np.mean(xeb_values)),
        num_circuits=len(xeb_values),
        mean_log_xeb=float(np.mean(log_xeb_values)),
    )


# ---------------------------------------------------------------------------
# Partial trace
# ---------------------------------------------------------------------------


def partial_trace(
    rho: np.ndarray,
    keep_qubits: list[int],
    num_qubits: int,
) -> np.ndarray:
    """Compute the partial trace, keeping specified qubits.

    Parameters
    ----------
    rho : np.ndarray
        Full density matrix (2^n x 2^n).
    keep_qubits : list[int]
        Qubit indices to keep (0-indexed).
    num_qubits : int
        Total number of qubits.

    Returns
    -------
    np.ndarray
        Reduced density matrix (2^k x 2^k where k = len(keep_qubits)).
    """
    rho = _ensure_density_matrix(rho)
    d = 2**num_qubits
    if rho.shape != (d, d):
        raise ValueError(
            f"Expected {d}x{d} density matrix for {num_qubits} qubits"
        )

    # Reshape into tensor with 2n indices
    rho_tensor = rho.reshape([2] * (2 * num_qubits))

    # Determine which axes to trace out
    trace_qubits = sorted(set(range(num_qubits)) - set(keep_qubits))

    # Trace out qubits one at a time, from highest index to lowest
    # Each qubit q has axes q (row) and q + num_qubits (col)
    for q in reversed(trace_qubits):
        rho_tensor = np.trace(rho_tensor, axis1=q, axis2=q + num_qubits)
        # After tracing, the number of remaining axes decreases by 2
        num_qubits -= 1

    # Reshape back to matrix
    d_keep = 2 ** len(keep_qubits)
    return rho_tensor.reshape(d_keep, d_keep)


# ---------------------------------------------------------------------------
# Schmidt decomposition
# ---------------------------------------------------------------------------


def schmidt_decomposition(
    state: np.ndarray,
    partition_a: list[int],
    num_qubits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Schmidt decomposition of a pure state.

    Splits the Hilbert space into subsystem A (partition_a qubits)
    and subsystem B (remaining qubits), then computes the SVD.

    Parameters
    ----------
    state : np.ndarray
        State vector (length 2^n).
    partition_a : list[int]
        Qubit indices belonging to subsystem A.
    num_qubits : int
        Total number of qubits.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (schmidt_coefficients, basis_a, basis_b)
        - schmidt_coefficients: 1D array of non-negative reals
        - basis_a: rows are basis states of subsystem A
        - basis_b: rows are basis states of subsystem B
    """
    if state.ndim != 1:
        raise ValueError("Schmidt decomposition requires a pure state vector")

    partition_b = sorted(set(range(num_qubits)) - set(partition_a))
    n_a = len(partition_a)
    n_b = len(partition_b)
    d_a = 2**n_a
    d_b = 2**n_b

    # Reshape state into a matrix (d_a x d_b)
    # Need to reorder qubits so that A qubits come first
    sv_tensor = state.reshape([2] * num_qubits)
    perm = list(partition_a) + list(partition_b)
    sv_tensor = sv_tensor.transpose(perm)
    matrix = sv_tensor.reshape(d_a, d_b)

    # SVD
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)

    return s, u.T, vh


# ---------------------------------------------------------------------------
# Schmidt number (entanglement rank)
# ---------------------------------------------------------------------------


def schmidt_number(
    state: np.ndarray,
    partition_a: list[int],
    num_qubits: int,
    tol: float = 1e-10,
) -> int:
    """Compute the Schmidt number (entanglement rank) of a pure state.

    The Schmidt number is the number of non-zero Schmidt coefficients.
    It equals 1 for product states and > 1 for entangled states.

    Parameters
    ----------
    state : np.ndarray
        State vector.
    partition_a : list[int]
        Qubit indices for subsystem A.
    num_qubits : int
        Total number of qubits.
    tol : float
        Threshold below which a coefficient is considered zero.

    Returns
    -------
    int
        Schmidt number.
    """
    coeffs, _, _ = schmidt_decomposition(state, partition_a, num_qubits)
    return int(np.sum(coeffs > tol))


# ---------------------------------------------------------------------------
# Linear entropy
# ---------------------------------------------------------------------------


def linear_entropy(rho: np.ndarray) -> float:
    """Compute the linear entropy S_L(rho) = 1 - tr(rho^2).

    A simpler alternative to Von Neumann entropy that avoids matrix
    logarithms.  Ranges from 0 (pure) to (d-1)/d (maximally mixed).

    Parameters
    ----------
    rho : np.ndarray
        Density matrix or state vector.

    Returns
    -------
    float
        Linear entropy in [0, (d-1)/d].
    """
    return 1.0 - purity(rho)


# ---------------------------------------------------------------------------
# Negativity (entanglement measure)
# ---------------------------------------------------------------------------


def negativity(rho: np.ndarray, num_qubits_a: int) -> float:
    """Compute the negativity of a bipartite state.

    N(rho) = (||rho^{T_A}||_1 - 1) / 2

    where rho^{T_A} is the partial transpose over subsystem A.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix of the bipartite system.
    num_qubits_a : int
        Number of qubits in subsystem A.

    Returns
    -------
    float
        Negativity (non-negative; > 0 implies entanglement by the
        Peres-Horodecki criterion).
    """
    rho = _ensure_density_matrix(rho)
    d = rho.shape[0]
    total_qubits = int(np.log2(d))
    d_a = 2**num_qubits_a
    d_b = d // d_a

    # Partial transpose over subsystem A
    rho_pt = _partial_transpose(rho, d_a, d_b)

    # Trace norm
    singular_values = np.linalg.svd(rho_pt, compute_uv=False)
    trace_norm = float(np.sum(singular_values))

    return (trace_norm - 1.0) / 2.0


def _partial_transpose(
    rho: np.ndarray, d_a: int, d_b: int
) -> np.ndarray:
    """Compute the partial transpose over subsystem A.

    Reshapes rho as (d_a, d_b, d_a, d_b) and transposes the A indices.
    """
    reshaped = rho.reshape(d_a, d_b, d_a, d_b)
    # Transpose A indices: (0,1,2,3) -> (2,1,0,3)
    pt = reshaped.transpose(2, 1, 0, 3)
    return pt.reshape(d_a * d_b, d_a * d_b)
