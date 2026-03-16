"""Measurement Error Mitigation for quantum error mitigation.

Corrects systematic readout (measurement) errors by characterizing the
confusion matrix of the measurement apparatus and applying its inverse
to raw measurement distributions.

Two inversion strategies are supported:

1. **Matrix inversion**: Direct or least-squares constrained inversion
   of the full 2^N confusion matrix.  Exact but exponential in qubit
   count.  Best for <= 10 qubits.

2. **Tensor product approximation**: Assumes per-qubit readout errors are
   independent, builds the full calibration as a tensor product of
   per-qubit 2x2 matrices.  Efficient but ignores correlated errors.

3. **Iterative Bayesian unfolding (IBU)**: Iteratively corrects the
   distribution using Bayes' theorem.  Does not require explicit matrix
   inversion and naturally produces non-negative probabilities.

Key classes:
  - :class:`ReadoutCalibration` -- Characterize readout errors.
  - :class:`ReadoutCorrector` -- Apply corrections to raw measurements.

References:
    - Bravyi et al., "Mitigating measurement errors", arXiv:2006.14044
    - D'Agostini, NIM A 362, 487 (1995) [Bayesian unfolding]
    - Nation et al., PRX 11, 041058 (2021) [M3 method]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# =====================================================================
# Readout calibration
# =====================================================================


class CorrectionMethod(Enum):
    """Readout correction method."""

    MATRIX_INVERSION = auto()
    LEAST_SQUARES = auto()
    BAYESIAN_UNFOLDING = auto()
    TENSOR_PRODUCT = auto()


@dataclass
class ReadoutCalibration:
    """Characterization of readout errors.

    The calibration matrix M has entries M[i,j] = P(measure i | prepared j),
    meaning the probability of observing bitstring i when the true state
    is bitstring j.

    For the full 2^N calibration, each column j is obtained by preparing
    the computational basis state |j>, measuring many times, and recording
    the empirical distribution over outcomes.

    For the tensor-product approximation, only per-qubit error rates
    are needed: p0_given_1 (measuring 0 when prepared 1) and p1_given_0
    (measuring 1 when prepared 0) for each qubit.

    Attributes
    ----------
    num_qubits : int
        Number of qubits.
    calibration_matrix : np.ndarray or None
        Full 2^N x 2^N calibration matrix (if available).
    qubit_error_rates : list of (p0_given_1, p1_given_0) or None
        Per-qubit error rates for tensor product approximation.
    """

    num_qubits: int
    calibration_matrix: Optional[np.ndarray] = None
    qubit_error_rates: Optional[List[Tuple[float, float]]] = None

    @classmethod
    def from_confusion_matrix(
        cls, matrix: np.ndarray, num_qubits: int
    ) -> "ReadoutCalibration":
        """Create calibration from a pre-computed confusion matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Shape (2^N, 2^N) confusion matrix where M[i,j] = P(meas i | prep j).
        num_qubits : int
            Number of qubits.
        """
        dim = 2 ** num_qubits
        if matrix.shape != (dim, dim):
            raise ValueError(
                f"Expected ({dim}, {dim}) matrix, got {matrix.shape}"
            )
        # Validate columns sum to ~1
        col_sums = matrix.sum(axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-6):
            raise ValueError(
                f"Columns must sum to 1, got sums: {col_sums}"
            )
        return cls(
            num_qubits=num_qubits,
            calibration_matrix=matrix.copy(),
        )

    @classmethod
    def from_qubit_error_rates(
        cls, error_rates: List[Tuple[float, float]]
    ) -> "ReadoutCalibration":
        """Create calibration from per-qubit error rates.

        Parameters
        ----------
        error_rates : list of (p0_given_1, p1_given_0)
            Per-qubit readout error rates.
            p0_given_1: probability of reading 0 when qubit is in |1>.
            p1_given_0: probability of reading 1 when qubit is in |0>.
        """
        for i, (p01, p10) in enumerate(error_rates):
            if not (0.0 <= p01 <= 1.0) or not (0.0 <= p10 <= 1.0):
                raise ValueError(
                    f"Qubit {i} error rates must be in [0, 1], "
                    f"got p0_given_1={p01}, p1_given_0={p10}"
                )
        return cls(
            num_qubits=len(error_rates),
            qubit_error_rates=list(error_rates),
        )

    @classmethod
    def from_symmetric_error(
        cls, num_qubits: int, error_rate: float
    ) -> "ReadoutCalibration":
        """Create calibration where all qubits have the same symmetric error.

        Parameters
        ----------
        num_qubits : int
            Number of qubits.
        error_rate : float
            Symmetric readout error: P(0|1) = P(1|0) = error_rate.
        """
        return cls.from_qubit_error_rates(
            [(error_rate, error_rate)] * num_qubits
        )

    @classmethod
    def from_calibration_circuits(
        cls,
        num_qubits: int,
        executor: Callable[[List[int]], Dict[int, int]],
        num_shots: int = 8192,
    ) -> "ReadoutCalibration":
        """Build calibration by preparing and measuring basis states.

        For N <= 10 qubits, prepares all 2^N basis states.
        For N > 10, uses the tensor product approximation (prepares
        only |0...0> and |1...1> or per-qubit calibration states).

        Parameters
        ----------
        num_qubits : int
            Number of qubits.
        executor : callable
            Function that takes a list of qubit states to prepare (0 or 1)
            and returns a counts dictionary {bitstring_int: count}.
        num_shots : int
            Shots per calibration circuit.
        """
        dim = 2 ** num_qubits

        if num_qubits <= 10:
            # Full calibration
            matrix = np.zeros((dim, dim))
            for j in range(dim):
                # Prepare basis state |j>
                prep_state = [(j >> q) & 1 for q in range(num_qubits)]
                counts = executor(prep_state)
                total = sum(counts.values())
                for i, count in counts.items():
                    matrix[i, j] = count / max(total, 1)
            return cls.from_confusion_matrix(matrix, num_qubits)
        else:
            # Tensor product approximation: per-qubit calibration
            error_rates: List[Tuple[float, float]] = []
            for q in range(num_qubits):
                # Prepare |0> on qubit q
                prep_0 = [0] * num_qubits
                counts_0 = executor(prep_0)
                total_0 = sum(counts_0.values())
                # Count how often qubit q reads as 1 when prepared in 0
                p1_given_0 = 0.0
                for bitstring, count in counts_0.items():
                    if (bitstring >> q) & 1:
                        p1_given_0 += count / max(total_0, 1)

                # Prepare |1> on qubit q
                prep_1 = [0] * num_qubits
                prep_1[q] = 1
                counts_1 = executor(prep_1)
                total_1 = sum(counts_1.values())
                # Count how often qubit q reads as 0 when prepared in 1
                p0_given_1 = 0.0
                for bitstring, count in counts_1.items():
                    if not ((bitstring >> q) & 1):
                        p0_given_1 += count / max(total_1, 1)

                error_rates.append((p0_given_1, p1_given_0))

            return cls.from_qubit_error_rates(error_rates)

    def get_full_matrix(self) -> np.ndarray:
        """Return the full 2^N x 2^N calibration matrix.

        If only per-qubit error rates are available, constructs the
        tensor product of per-qubit confusion matrices.
        """
        if self.calibration_matrix is not None:
            return self.calibration_matrix

        if self.qubit_error_rates is None:
            # Identity calibration (no errors)
            dim = 2 ** self.num_qubits
            return np.eye(dim)

        # Build tensor product
        result = np.array([[1.0]])
        for p01, p10 in self.qubit_error_rates:
            # Per-qubit confusion: M[meas][prep]
            #   M[0][0] = 1 - p10    M[0][1] = p01
            #   M[1][0] = p10        M[1][1] = 1 - p01
            qubit_matrix = np.array([
                [1.0 - p10, p01],
                [p10, 1.0 - p01],
            ])
            result = np.kron(result, qubit_matrix)

        return result


# =====================================================================
# Readout corrector
# =====================================================================


class ReadoutCorrector:
    """Correct measurement results using readout calibration data.

    Parameters
    ----------
    calibration : ReadoutCalibration
        Readout error characterization.
    method : CorrectionMethod
        Correction strategy.
    max_ibu_iterations : int
        Maximum iterations for Bayesian unfolding.
    ibu_tolerance : float
        Convergence tolerance for Bayesian unfolding.
    """

    def __init__(
        self,
        calibration: ReadoutCalibration,
        method: CorrectionMethod = CorrectionMethod.LEAST_SQUARES,
        max_ibu_iterations: int = 100,
        ibu_tolerance: float = 1e-6,
    ) -> None:
        self.calibration = calibration
        self.method = method
        self.max_ibu_iterations = max_ibu_iterations
        self.ibu_tolerance = ibu_tolerance

    def correct_counts(
        self, raw_counts: Dict[int, int]
    ) -> Dict[int, float]:
        """Correct raw measurement counts.

        Parameters
        ----------
        raw_counts : dict
            Mapping from bitstring (as int) to count.

        Returns
        -------
        dict
            Mapping from bitstring to corrected probability.
        """
        dim = 2 ** self.calibration.num_qubits
        total = sum(raw_counts.values())
        if total == 0:
            return {}

        # Convert counts to probability vector
        raw_probs = np.zeros(dim)
        for bitstring, count in raw_counts.items():
            if 0 <= bitstring < dim:
                raw_probs[bitstring] = count / total

        corrected = self.correct_probabilities(raw_probs)

        # Convert back to dictionary (filter near-zero entries)
        result: Dict[int, float] = {}
        for i, p in enumerate(corrected):
            if abs(p) > 1e-10:
                result[i] = float(p)
        return result

    def correct_probabilities(
        self, raw_probs: np.ndarray
    ) -> np.ndarray:
        """Correct a raw probability distribution.

        Parameters
        ----------
        raw_probs : np.ndarray
            Raw measurement probability distribution of length 2^N.

        Returns
        -------
        np.ndarray
            Corrected probability distribution.
        """
        if self.method == CorrectionMethod.TENSOR_PRODUCT:
            return self._correct_tensor_product(raw_probs)
        elif self.method == CorrectionMethod.BAYESIAN_UNFOLDING:
            return self._correct_bayesian(raw_probs)
        elif self.method == CorrectionMethod.MATRIX_INVERSION:
            return self._correct_matrix_inversion(raw_probs)
        elif self.method == CorrectionMethod.LEAST_SQUARES:
            return self._correct_least_squares(raw_probs)
        else:
            raise ValueError(f"Unknown correction method: {self.method}")

    def _correct_matrix_inversion(
        self, raw_probs: np.ndarray
    ) -> np.ndarray:
        """Direct matrix inversion: p_true = M^{-1} . p_raw."""
        M = self.calibration.get_full_matrix()
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse
            M_inv = np.linalg.pinv(M)
        corrected = M_inv @ raw_probs
        return corrected

    def _correct_least_squares(
        self, raw_probs: np.ndarray
    ) -> np.ndarray:
        """Constrained least-squares: minimize ||M.p - raw||^2 s.t. p >= 0, sum(p) = 1.

        Uses iterative projection (simplified NNLS with normalization).
        """
        M = self.calibration.get_full_matrix()
        dim = len(raw_probs)

        # Start with matrix inversion solution
        try:
            M_inv = np.linalg.inv(M)
            x = M_inv @ raw_probs
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(M, raw_probs, rcond=None)[0]

        # Project to probability simplex: clip negatives, renormalize
        for _ in range(50):
            x = np.maximum(x, 0.0)
            total = np.sum(x)
            if total > 1e-15:
                x = x / total
            else:
                x = np.ones(dim) / dim
                break

            # Check convergence: is M.x close to raw_probs?
            residual = np.linalg.norm(M @ x - raw_probs)
            if residual < 1e-10:
                break

        return x

    def _correct_tensor_product(
        self, raw_probs: np.ndarray
    ) -> np.ndarray:
        """Tensor product inversion using per-qubit inverse matrices.

        Efficient O(N * 2^N) method that avoids forming the full matrix.
        """
        if self.calibration.qubit_error_rates is None:
            return self._correct_least_squares(raw_probs)

        n = self.calibration.num_qubits
        corrected = raw_probs.copy()

        # Apply per-qubit inverse confusion in-place via butterfly structure
        for q, (p01, p10) in enumerate(self.calibration.qubit_error_rates):
            # Per-qubit confusion:
            #   M_q = [[1-p10, p01], [p10, 1-p01]]
            # Inverse:
            det = (1.0 - p10) * (1.0 - p01) - p01 * p10
            if abs(det) < 1e-15:
                continue  # Skip degenerate qubits
            inv = np.array([
                [1.0 - p01, -p01],
                [-p10, 1.0 - p10],
            ]) / det

            # Apply to the dimension corresponding to qubit q
            dim = 2 ** n
            stride = 2 ** q
            for block_start in range(0, dim, 2 * stride):
                for j in range(stride):
                    idx0 = block_start + j
                    idx1 = block_start + j + stride
                    v0 = corrected[idx0]
                    v1 = corrected[idx1]
                    corrected[idx0] = inv[0, 0] * v0 + inv[0, 1] * v1
                    corrected[idx1] = inv[1, 0] * v0 + inv[1, 1] * v1

        # Project to valid distribution
        corrected = np.maximum(corrected, 0.0)
        total = np.sum(corrected)
        if total > 1e-15:
            corrected /= total

        return corrected

    def _correct_bayesian(
        self, raw_probs: np.ndarray
    ) -> np.ndarray:
        """Iterative Bayesian Unfolding (IBU).

        Uses Bayes' theorem to iteratively refine an estimate of the
        true distribution from the observed (noisy) distribution.

        The update rule is:
            p_true^{k+1}[j] = p_true^k[j] * sum_i (M[i,j] * p_raw[i] / sum_l (M[i,l] * p_true^k[l]))
        """
        M = self.calibration.get_full_matrix()
        dim = len(raw_probs)

        # Initial guess: uniform
        p_true = np.ones(dim) / dim

        for iteration in range(self.max_ibu_iterations):
            # E-step: compute expected observed distribution
            p_expected = M @ p_true
            p_expected = np.maximum(p_expected, 1e-15)

            # M-step: Bayesian update
            ratio = raw_probs / p_expected
            p_new = p_true * (M.T @ ratio)

            # Normalize
            total = np.sum(p_new)
            if total > 1e-15:
                p_new /= total
            else:
                break

            # Check convergence
            delta = np.max(np.abs(p_new - p_true))
            p_true = p_new
            if delta < self.ibu_tolerance:
                break

        return p_true


# =====================================================================
# Convenience function
# =====================================================================


def correct_counts(
    raw_counts: Dict[int, int],
    calibration: ReadoutCalibration,
    method: CorrectionMethod = CorrectionMethod.LEAST_SQUARES,
) -> Dict[int, float]:
    """One-shot readout error correction.

    Parameters
    ----------
    raw_counts : dict
        Raw measurement counts {bitstring_int: count}.
    calibration : ReadoutCalibration
        Readout error characterization.
    method : CorrectionMethod
        Correction approach.

    Returns
    -------
    dict
        Corrected probability distribution.
    """
    corrector = ReadoutCorrector(calibration, method=method)
    return corrector.correct_counts(raw_counts)
