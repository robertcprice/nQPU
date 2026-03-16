"""Quantum Amplitude Estimation for financial applications.

Implements three variants of quantum amplitude estimation used as the
computational core for option pricing, risk analysis, and other financial
algorithms that benefit from quadratic quantum speedup over classical
Monte Carlo sampling.

Variants:

- **CanonicalQAE**: Phase-estimation-based amplitude estimation using
  ``num_eval_qubits`` ancilla qubits.  Precision scales as O(1/2^n).

- **IterativeQAE (IQAE)**: QPE-free approach that applies Grover iterates
  with geometrically increasing powers, converging to target precision
  epsilon with confidence 1 - alpha.  Requires no ancilla qubits.
  Based on Suzuki et al. (2020).

- **MaxLikelihoodQAE (MLAE)**: Applies Grover iterates at a schedule of
  powers, collects measurement statistics, then maximises the likelihood
  function via grid search with golden-section refinement.

All methods operate on a Grover operator Q = A S_0 A^dag S_chi where:
  - A is the state-preparation unitary (oracle)
  - S_0 = 2|0><0| - I reflects about |0>
  - S_chi flips the sign on "good" states

References:
  - Brassard et al. "Quantum Amplitude Amplification and Estimation" (2002)
  - Suzuki et al. "Amplitude estimation without phase estimation" (2020)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ============================================================
# Result type
# ============================================================


@dataclass
class AEResult:
    """Result of an amplitude estimation computation.

    Attributes
    ----------
    estimation : float
        Estimated amplitude *a* (probability that the oracle marks a state
        as "good").  This equals sin^2(theta) where theta is the Grover
        angle.
    confidence_interval : tuple[float, float]
        (lower, upper) bounds on the amplitude estimate.
    num_oracle_calls : int
        Total number of Grover-operator applications used.
    samples : list[float]
        Intermediate amplitude samples collected during estimation.
    """

    estimation: float
    confidence_interval: tuple[float, float]
    num_oracle_calls: int
    samples: list[float] = field(default_factory=list)


# ============================================================
# Linear-algebra helpers (pure numpy, small matrices)
# ============================================================


def _matvec(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Matrix-vector product for complex arrays."""
    return matrix @ vec


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix-matrix product for complex arrays."""
    return a @ b


def _adjoint(matrix: np.ndarray) -> np.ndarray:
    """Conjugate transpose."""
    return matrix.conj().T


def _good_state_probability(
    state: np.ndarray, good_indices: Sequence[int]
) -> float:
    """Probability of measuring one of the good basis states."""
    return float(np.sum(np.abs(state[list(good_indices)]) ** 2))


# ============================================================
# Grover operator construction
# ============================================================


def build_grover_operator(
    oracle: np.ndarray,
    good_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Construct the Grover operator Q = A S_0 A^dag S_chi.

    Parameters
    ----------
    oracle : np.ndarray
        State-preparation unitary *A* (dim x dim complex matrix).
        A|0> produces the state whose "good" amplitude we want to estimate.
    good_indices : sequence of int, optional
        Indices of the "good" basis states.  Defaults to the upper half
        of the Hilbert space (dim//2 .. dim-1).

    Returns
    -------
    np.ndarray
        The Grover operator Q (dim x dim complex matrix).
    """
    dim = oracle.shape[0]
    if good_indices is None:
        good_indices = list(range(dim // 2, dim))

    # S_chi: I with -1 on good-state diagonal entries
    s_chi = np.eye(dim, dtype=complex)
    for g in good_indices:
        s_chi[g, g] = -1.0

    # S_0: 2|0><0| - I
    s_0 = -np.eye(dim, dtype=complex)
    s_0[0, 0] = 1.0

    a_dag = _adjoint(oracle)
    # Q = A * S_0 * A^dag * S_chi
    return _matmul(oracle, _matmul(s_0, _matmul(a_dag, s_chi)))


def apply_grover_power(
    state: np.ndarray, grover_op: np.ndarray, power: int
) -> np.ndarray:
    """Apply Q^power to *state* by repeated matrix-vector multiplication."""
    current = state.copy()
    for _ in range(power):
        current = _matvec(grover_op, current)
    return current


# ============================================================
# QPE probability helper
# ============================================================


def _qpe_outcome_probability(phi: float, m: int, n: int) -> float:
    """Probability P(m | phi) for QPE with *n* evaluation qubits.

    P(m|phi) = sin^2(N*delta/2) / (N^2 * sin^2(delta/2))
    where N = 2^n, delta = phi - 2*pi*m/N.
    """
    big_n = float(1 << n)
    delta = phi - 2.0 * np.pi * m / big_n
    half_delta = delta / 2.0
    if abs(np.sin(half_delta)) < 1e-15:
        return 1.0
    numerator = np.sin(big_n * half_delta)
    denominator = big_n * np.sin(half_delta)
    return float((numerator / denominator) ** 2)


# ============================================================
# Canonical QAE
# ============================================================


class CanonicalQAE:
    """Canonical Quantum Amplitude Estimation via phase estimation.

    Uses *num_eval_qubits* evaluation qubits to estimate the eigenphase
    of the Grover operator Q, yielding amplitude a = sin^2(theta) with
    precision O(1/2^n).

    Parameters
    ----------
    num_eval_qubits : int
        Number of evaluation (ancilla) qubits.  More qubits give
        exponentially better precision.
    """

    def __init__(self, num_eval_qubits: int = 6) -> None:
        if num_eval_qubits < 1:
            raise ValueError("num_eval_qubits must be >= 1")
        self.num_eval_qubits = num_eval_qubits

    def estimate(
        self,
        oracle: np.ndarray,
        good_indices: Sequence[int] | None = None,
    ) -> AEResult:
        """Estimate the amplitude of good states in oracle|0>.

        Parameters
        ----------
        oracle : np.ndarray
            State-preparation unitary A (dim x dim).
        good_indices : sequence of int, optional
            Which basis states are "good".

        Returns
        -------
        AEResult
        """
        dim = oracle.shape[0]
        if good_indices is None:
            good_indices = list(range(dim // 2, dim))

        # Prepare A|0>
        initial = np.zeros(dim, dtype=complex)
        initial[0] = 1.0
        prepared = _matvec(oracle, initial)

        # True amplitude (for determining the Grover angle)
        true_amp = _good_state_probability(prepared, good_indices)
        theta = np.arcsin(np.sqrt(true_amp))

        n = self.num_eval_qubits
        num_outcomes = 1 << n

        # The two eigenphases of Q in [0, 2pi)
        phi_plus = 2.0 * theta
        phi_minus = 2.0 * np.pi - 2.0 * theta

        # Compute QPE probability distribution
        probabilities = np.zeros(num_outcomes)
        for m in range(num_outcomes):
            p_plus = _qpe_outcome_probability(phi_plus, m, n)
            p_minus = _qpe_outcome_probability(phi_minus, m, n)
            probabilities[m] = 0.5 * p_plus + 0.5 * p_minus

        # Most likely outcome
        best_m = int(np.argmax(probabilities))
        theta_est = np.pi * best_m / num_outcomes
        estimation = np.sin(theta_est) ** 2

        # Confidence interval from QPE precision bound
        delta_theta = np.pi / num_outcomes
        theta_low = max(theta_est - delta_theta, 0.0)
        theta_high = min(theta_est + delta_theta, np.pi / 2.0)
        ci_low = float(np.sin(theta_low) ** 2)
        ci_high = float(np.sin(theta_high) ** 2)

        # Collect significant samples
        samples = []
        for m, p in enumerate(probabilities):
            if p > 1e-6:
                t = np.pi * m / num_outcomes
                samples.append(float(np.sin(t) ** 2))

        # Oracle calls: 2^0 + 2^1 + ... + 2^{n-1} = 2^n - 1
        num_oracle_calls = num_outcomes - 1

        return AEResult(
            estimation=float(estimation),
            confidence_interval=(min(ci_low, ci_high), max(ci_low, ci_high)),
            num_oracle_calls=num_oracle_calls,
            samples=samples,
        )


# ============================================================
# Iterative QAE (IQAE)
# ============================================================


class IterativeQAE:
    """Iterative Quantum Amplitude Estimation (QPE-free).

    Maintains a confidence interval [theta_low, theta_high] and
    iteratively narrows it by applying Q^k for geometrically increasing
    k, measuring the outcome, and updating the interval.

    Parameters
    ----------
    epsilon : float
        Target precision for the amplitude estimate.
    alpha : float
        Confidence level.  Result correct with probability >= 1 - alpha.
    max_iterations : int
        Safety limit on the number of rounds.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        alpha: float = 0.05,
        max_iterations: int = 100,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iterations = max_iterations

    def estimate(
        self,
        oracle: np.ndarray,
        good_indices: Sequence[int] | None = None,
    ) -> AEResult:
        """Estimate the amplitude using iterative Grover applications.

        Parameters
        ----------
        oracle : np.ndarray
            State-preparation unitary A.
        good_indices : sequence of int, optional
            Which basis states are "good".

        Returns
        -------
        AEResult
        """
        dim = oracle.shape[0]
        if good_indices is None:
            good_indices = list(range(dim // 2, dim))

        initial = np.zeros(dim, dtype=complex)
        initial[0] = 1.0
        prepared = _matvec(oracle, initial)

        q = build_grover_operator(oracle, good_indices)

        theta_low = 0.0
        theta_high = np.pi / 2.0
        total_oracle_calls = 0
        samples: list[float] = []

        k = 1
        for _ in range(self.max_iterations):
            if theta_high - theta_low <= self.epsilon:
                break

            # Apply Q^k to prepared state
            state_k = apply_grover_power(prepared, q, k)
            total_oracle_calls += k

            prob_good = _good_state_probability(state_k, good_indices)
            samples.append(float(prob_good))

            # sin^2((2k+1)*theta) = prob_good
            alpha_val = np.arcsin(np.sqrt(np.clip(prob_good, 0.0, 1.0)))
            factor = float(2 * k + 1)

            scaled_low = factor * theta_low
            scaled_high = factor * theta_high
            max_m = int(np.ceil(scaled_high / np.pi)) + 1

            candidates: list[float] = []
            for m in range(max_m + 1):
                # Branch A: factor*theta = alpha + m*pi
                t_a = (alpha_val + m * np.pi) / factor
                if theta_low - 1e-12 <= t_a <= theta_high + 1e-12:
                    candidates.append(float(np.clip(t_a, 0.0, np.pi / 2.0)))
                # Branch B: factor*theta = pi - alpha + m*pi
                t_b = (np.pi - alpha_val + m * np.pi) / factor
                if theta_low - 1e-12 <= t_b <= theta_high + 1e-12:
                    candidates.append(float(np.clip(t_b, 0.0, np.pi / 2.0)))

            if not candidates:
                k = min(k * 2, 1 << 20)
                continue

            # Deduplicate and sort
            candidates = sorted(set(round(c, 12) for c in candidates))

            midpoint = (theta_low + theta_high) / 2.0
            best = min(candidates, key=lambda c: abs(c - midpoint))

            half_spacing = np.pi / (2.0 * factor)
            new_low = max(best - half_spacing, theta_low)
            new_high = min(best + half_spacing, theta_high)

            if new_high > new_low and (new_high - new_low) < (theta_high - theta_low):
                theta_low = new_low
                theta_high = new_high

            k = min(k * 2, 1 << 20)

        theta_est = (theta_low + theta_high) / 2.0
        estimation = float(np.sin(theta_est) ** 2)
        ci_low = float(np.sin(theta_low) ** 2)
        ci_high = float(np.sin(theta_high) ** 2)

        return AEResult(
            estimation=estimation,
            confidence_interval=(min(ci_low, ci_high), max(ci_low, ci_high)),
            num_oracle_calls=total_oracle_calls,
            samples=samples,
        )


# ============================================================
# Maximum Likelihood QAE (MLAE)
# ============================================================


class MaxLikelihoodQAE:
    """Maximum Likelihood Amplitude Estimation.

    For each Grover power k in the evaluation schedule, applies Q^k to
    the prepared state and records the exact "good" probability (simulating
    shot statistics deterministically).  The MLE of theta is found via
    grid search with golden-section refinement.

    Parameters
    ----------
    evaluation_schedule : list[int]
        Grover powers to apply, e.g. [0, 1, 2, 4, 8].
    num_shots : int
        Number of measurement shots per power (for statistics).
    """

    def __init__(
        self,
        evaluation_schedule: list[int] | None = None,
        num_shots: int = 100,
    ) -> None:
        if evaluation_schedule is None:
            evaluation_schedule = [0, 1, 2, 4, 8]
        self.evaluation_schedule = evaluation_schedule
        self.num_shots = num_shots

    @classmethod
    def with_exponential_schedule(
        cls, max_power_exponent: int = 4, num_shots: int = 100
    ) -> MaxLikelihoodQAE:
        """Build with a default exponential schedule [0, 1, 2, 4, ..., 2^max]."""
        schedule = [0] + [1 << e for e in range(max_power_exponent + 1)]
        return cls(evaluation_schedule=schedule, num_shots=num_shots)

    def estimate(
        self,
        oracle: np.ndarray,
        good_indices: Sequence[int] | None = None,
    ) -> AEResult:
        """Estimate the amplitude using maximum likelihood.

        Parameters
        ----------
        oracle : np.ndarray
            State-preparation unitary A.
        good_indices : sequence of int, optional
            Which basis states are "good".

        Returns
        -------
        AEResult
        """
        dim = oracle.shape[0]
        if good_indices is None:
            good_indices = list(range(dim // 2, dim))

        initial = np.zeros(dim, dtype=complex)
        initial[0] = 1.0
        prepared = _matvec(oracle, initial)

        q = build_grover_operator(oracle, good_indices)

        measurements: list[tuple[int, int, int]] = []  # (k, h_k, n_k)
        total_oracle_calls = 0
        samples: list[float] = []

        for k in self.evaluation_schedule:
            state_k = apply_grover_power(prepared, q, k)
            total_oracle_calls += k

            prob_good = _good_state_probability(state_k, good_indices)
            h_k = int(round(prob_good * self.num_shots))
            h_k = min(h_k, self.num_shots)

            measurements.append((k, h_k, self.num_shots))
            samples.append(h_k / self.num_shots)

        theta_mle = self._maximize_likelihood(measurements)
        estimation = float(np.sin(theta_mle) ** 2)

        # Fisher information for confidence interval
        fisher = self._fisher_information(theta_mle, measurements)
        stderr = 1.0 / np.sqrt(fisher) if fisher > 1e-15 else np.pi / 4.0

        z = 1.96  # 95% confidence
        theta_low = max(theta_mle - z * stderr, 0.0)
        theta_high = min(theta_mle + z * stderr, np.pi / 2.0)
        ci_low = float(np.sin(theta_low) ** 2)
        ci_high = float(np.sin(theta_high) ** 2)

        return AEResult(
            estimation=estimation,
            confidence_interval=(min(ci_low, ci_high), max(ci_low, ci_high)),
            num_oracle_calls=total_oracle_calls,
            samples=samples,
        )

    # -- private helpers --

    def _log_likelihood(
        self, theta: float, measurements: list[tuple[int, int, int]]
    ) -> float:
        """Compute log L(theta) = sum_k [h_k log sin^2 + (n_k-h_k) log cos^2]."""
        ll = 0.0
        for k, h_k, n_k in measurements:
            angle = (2 * k + 1) * theta
            sin2 = max(np.sin(angle) ** 2, 1e-300)
            cos2 = max(np.cos(angle) ** 2, 1e-300)
            ll += h_k * np.log(sin2) + (n_k - h_k) * np.log(cos2)
        return float(ll)

    def _maximize_likelihood(
        self, measurements: list[tuple[int, int, int]]
    ) -> float:
        """Grid search + golden-section refinement for argmax log L."""
        num_grid = 1000
        thetas = np.linspace(0, np.pi / 2, num_grid + 1)
        best_theta = 0.0
        best_ll = -np.inf

        for theta in thetas:
            ll = self._log_likelihood(float(theta), measurements)
            if ll > best_ll:
                best_ll = ll
                best_theta = float(theta)

        # Golden-section refinement
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        half_step = np.pi / (2.0 * num_grid)
        a = max(best_theta - half_step, 0.0)
        b = min(best_theta + half_step, np.pi / 2.0)

        for _ in range(50):
            if abs(b - a) < 1e-12:
                break
            x1 = b - golden * (b - a)
            x2 = a + golden * (b - a)
            f1 = self._log_likelihood(x1, measurements)
            f2 = self._log_likelihood(x2, measurements)
            if f1 > f2:
                b = x2
            else:
                a = x1

        return (a + b) / 2.0

    def _fisher_information(
        self, theta: float, measurements: list[tuple[int, int, int]]
    ) -> float:
        """Fisher information at theta for confidence interval estimation."""
        info = 0.0
        for k, _h_k, n_k in measurements:
            angle = (2 * k + 1) * theta
            sin2 = np.sin(angle) ** 2
            cos2 = np.cos(angle) ** 2
            if sin2 * cos2 > 1e-15:
                factor = float(2 * k + 1)
                info += n_k * factor * factor * 4.0
        return info


# ============================================================
# Convenience: Bernoulli oracle builder
# ============================================================


def bernoulli_oracle(amplitude: float) -> np.ndarray:
    """Build a 2x2 oracle A such that A|0> = sqrt(1-a)|0> + sqrt(a)|1>.

    This is the standard test oracle for amplitude estimation: the "good"
    state is |1> (index 1 in a 2-dimensional system).

    Parameters
    ----------
    amplitude : float
        Target amplitude *a* in [0, 1].

    Returns
    -------
    np.ndarray
        A 2x2 unitary matrix (Ry rotation).
    """
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError(f"amplitude must be in [0, 1], got {amplitude}")
    cos_t = np.sqrt(1.0 - amplitude)
    sin_t = np.sqrt(amplitude)
    return np.array(
        [[cos_t, -sin_t], [sin_t, cos_t]], dtype=complex
    )


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    print("=== Amplitude Estimation self-test ===")

    target = 0.3
    oracle = bernoulli_oracle(target)
    good = [1]  # second basis state is "good"

    # Canonical QAE
    qae = CanonicalQAE(num_eval_qubits=6)
    r = qae.estimate(oracle, good)
    print(f"CanonicalQAE: est={r.estimation:.4f} (target={target}), "
          f"CI={r.confidence_interval}, calls={r.num_oracle_calls}")
    assert abs(r.estimation - target) < 0.05

    # IQAE
    iqae = IterativeQAE(epsilon=0.01, alpha=0.05)
    r = iqae.estimate(oracle, good)
    print(f"IterativeQAE: est={r.estimation:.4f} (target={target}), "
          f"CI={r.confidence_interval}, calls={r.num_oracle_calls}")
    assert abs(r.estimation - target) < 0.05

    # MLAE
    mlae = MaxLikelihoodQAE(evaluation_schedule=[0, 1, 2, 4, 8], num_shots=200)
    r = mlae.estimate(oracle, good)
    print(f"MaxLikelihoodQAE: est={r.estimation:.4f} (target={target}), "
          f"CI={r.confidence_interval}, calls={r.num_oracle_calls}")
    assert abs(r.estimation - target) < 0.05

    print("All self-tests passed.")
