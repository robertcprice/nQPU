"""Quantum portfolio optimization via QAOA and classical solvers.

Implements Markowitz mean-variance portfolio optimization where the
combinatorial selection problem is encoded as a QUBO (Quadratic
Unconstrained Binary Optimization) and solved using QAOA (Quantum
Approximate Optimization Algorithm) simulation.

Architecture:

  1. Encode the portfolio problem as a QUBO:
     min x^T Sigma x - lambda * r^T x + penalty * (sum(x) - budget)^2
  2. Convert QUBO to Ising Hamiltonian (H = J Z_i Z_j + h_i Z_i + C).
  3. Run QAOA: alternating problem-unitary and mixer-unitary layers,
     with angles (gamma, beta) tuned by a classical optimizer.
  4. Extract the most probable bitstring and decode into portfolio weights.

Also provides a classical brute-force solver for validation on small
instances and efficient-frontier computation.

References:
  - Barkoutsos et al. (2020), "Improving Variational Quantum Optimization
    using CVaR"
  - Egger et al. (2020), "Quantum Computing for Finance"
  - Markowitz (1952), "Portfolio Selection"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ============================================================
# Result types
# ============================================================


@dataclass
class PortfolioResult:
    """Result of portfolio optimization.

    Attributes
    ----------
    weights : np.ndarray
        Optimal asset weights (sum to 1).
    objective : float
        Objective value (risk - lambda * return).
    expected_return : float
        Expected portfolio return r^T w.
    variance : float
        Portfolio variance w^T Sigma w.
    best_bitstring : np.ndarray
        Optimal binary assignment from QAOA.
    optimal_angles : np.ndarray
        Optimized QAOA variational parameters.
    iterations : int
        Number of classical optimizer iterations.
    """

    weights: np.ndarray
    objective: float
    expected_return: float
    variance: float
    best_bitstring: np.ndarray
    optimal_angles: np.ndarray
    iterations: int


@dataclass
class EfficientFrontierPoint:
    """A single point on the efficient frontier.

    Attributes
    ----------
    target_return : float
        Targeted return level.
    weights : np.ndarray
        Optimal weights for that return level.
    variance : float
        Minimum variance at the target return.
    """

    target_return: float
    weights: np.ndarray
    variance: float


# ============================================================
# QUBO / Ising encoding
# ============================================================


@dataclass
class QuboMatrix:
    """QUBO matrix representation: minimize x^T Q x + offset."""

    matrix: np.ndarray
    offset: float
    num_variables: int


@dataclass
class IsingHamiltonian:
    """Ising Hamiltonian: H = sum J_ij Z_i Z_j + sum h_i Z_i + offset."""

    j_couplings: list[tuple[int, int, float]]
    h_fields: np.ndarray
    offset: float
    num_qubits: int

    def energy(self, spins: np.ndarray) -> float:
        """Evaluate Ising energy for spin configuration (spins in {-1, +1})."""
        e = self.offset
        for i, j, jij in self.j_couplings:
            e += jij * spins[i] * spins[j]
        e += np.dot(self.h_fields, spins)
        return float(e)

    def energy_bitstring(self, bits: np.ndarray) -> float:
        """Evaluate energy for bitstring (bits in {0, 1})."""
        spins = 1.0 - 2.0 * bits.astype(float)
        return self.energy(spins)


def portfolio_to_qubo(
    returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float = 0.5,
    penalty_strength: float = 10.0,
    budget: int | None = None,
    num_bits_per_asset: int = 1,
) -> QuboMatrix:
    """Encode portfolio optimization as QUBO.

    Parameters
    ----------
    returns : np.ndarray
        Expected return for each asset (length n).
    covariance : np.ndarray
        Covariance matrix (n x n).
    risk_aversion : float
        Risk-aversion parameter lambda.
    penalty_strength : float
        Penalty for budget constraint violation.
    budget : int or None
        Budget constraint: sum of selected assets.  If None, defaults to
        half the number of assets.
    num_bits_per_asset : int
        Binary precision per asset (1 = select/don't-select).
    """
    n = len(returns)
    nb = num_bits_per_asset
    total = n * nb

    if budget is None:
        budget = max(n // 2, 1)

    q = np.zeros((total, total))
    offset = 0.0

    # For 1-bit-per-asset: w_i = x_i (binary)
    # Objective: x^T Sigma x - lambda * r^T x
    if nb == 1:
        # Quadratic risk
        for i in range(n):
            for j in range(n):
                q[i, j] += covariance[i, j]
        # Linear return (on diagonal)
        for i in range(n):
            q[i, i] -= risk_aversion * returns[i]
        # Budget constraint: penalty * (sum x_i - budget)^2
        # = penalty * (sum x_i)^2 - 2*budget*sum(x_i) + budget^2
        offset += penalty_strength * budget ** 2
        for i in range(n):
            q[i, i] += penalty_strength * (1 - 2 * budget)
            for j in range(n):
                q[i, j] += penalty_strength
    else:
        # Multi-bit encoding
        wmin, wmax = 0.0, 1.0
        scale = (wmax - wmin) / ((1 << nb) - 1)

        for i in range(n):
            for j in range(n):
                for ki in range(nb):
                    for kj in range(nb):
                        idx_i = i * nb + ki
                        idx_j = j * nb + kj
                        pw = (1 << ki) * (1 << kj)
                        q[idx_i, idx_j] += covariance[i, j] * scale ** 2 * pw

        for i in range(n):
            for ki in range(nb):
                idx = i * nb + ki
                q[idx, idx] -= risk_aversion * returns[i] * scale * (1 << ki)

        # Budget constraint for multi-bit is on sum of decoded weights
        # sum w_i = sum_i (wmin + scale * sum_k x_{i,k} * 2^k)
        sum_wmin = n * wmin
        offset += penalty_strength * (sum_wmin - 1.0) ** 2
        for i in range(n):
            for ki in range(nb):
                idx = i * nb + ki
                pk = (1 << ki)
                q[idx, idx] += penalty_strength * 2.0 * (sum_wmin - 1.0) * scale * pk
        for i1 in range(n):
            for k1 in range(nb):
                idx1 = i1 * nb + k1
                for i2 in range(n):
                    for k2 in range(nb):
                        idx2 = i2 * nb + k2
                        q[idx1, idx2] += (
                            penalty_strength * scale ** 2 * (1 << k1) * (1 << k2)
                        )

    return QuboMatrix(matrix=q, offset=offset, num_variables=total)


def qubo_to_ising(qubo: QuboMatrix) -> IsingHamiltonian:
    """Convert QUBO to Ising Hamiltonian via x_i = (1 - s_i)/2."""
    n = qubo.num_variables
    h = np.zeros(n)
    offset = qubo.offset
    j_couplings: list[tuple[int, int, float]] = []

    # Diagonal
    for i in range(n):
        offset += qubo.matrix[i, i] / 2.0
        h[i] -= qubo.matrix[i, i] / 2.0

    # Off-diagonal
    for i in range(n):
        for j in range(i + 1, n):
            qij = qubo.matrix[i, j] + qubo.matrix[j, i]
            if abs(qij) > 1e-15:
                j_couplings.append((i, j, qij / 4.0))
                h[i] -= qij / 4.0
                h[j] -= qij / 4.0
                offset += qij / 4.0

    return IsingHamiltonian(
        j_couplings=j_couplings,
        h_fields=h,
        offset=offset,
        num_qubits=n,
    )


# ============================================================
# Statevector simulator (minimal for QAOA)
# ============================================================


def _qaoa_statevector(
    ising: IsingHamiltonian, gammas: np.ndarray, betas: np.ndarray
) -> np.ndarray:
    """Build the QAOA state for given Ising Hamiltonian and variational angles.

    Returns the full probability vector.
    """
    n = ising.num_qubits
    dim = 1 << n

    # |+>^n
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)

    p = len(gammas)
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        # Problem unitary: exp(-i * gamma * H_C)
        # For diagonal Ising, apply phase to each basis state
        for idx in range(dim):
            spins = np.array(
                [1.0 - 2.0 * ((idx >> q) & 1) for q in range(n)]
            )
            phase = ising.energy(spins) * gamma
            state[idx] *= np.exp(-1j * phase)

        # Mixer: exp(-i * beta * sum X_j)
        # Each X_j can be applied independently: Rx(2*beta) on each qubit
        for q in range(n):
            step = 1 << q
            cos_b = np.cos(beta)
            sin_b = -1j * np.sin(beta)
            i = 0
            while i < dim:
                for k in range(step):
                    a = state[i + k]
                    b = state[i + k + step]
                    state[i + k] = cos_b * a + sin_b * b
                    state[i + k + step] = sin_b * a + cos_b * b
                i += step << 1

    return np.abs(state) ** 2


# ============================================================
# QAOA portfolio optimizer
# ============================================================


class PortfolioOptimizer:
    """QAOA-based portfolio optimizer.

    Parameters
    ----------
    num_layers : int
        Number of QAOA layers (p).
    risk_aversion : float
        Risk-aversion parameter lambda.
    penalty_strength : float
        Constraint-violation penalty.
    budget : int or None
        Number of assets to select (for 1-bit-per-asset).
    num_bits_per_asset : int
        Binary precision per asset.
    max_optimizer_iterations : int
        Iteration cap for the classical optimizer.
    """

    def __init__(
        self,
        num_layers: int = 2,
        risk_aversion: float = 0.5,
        penalty_strength: float = 10.0,
        budget: int | None = None,
        num_bits_per_asset: int = 1,
        max_optimizer_iterations: int = 200,
    ) -> None:
        self.num_layers = num_layers
        self.risk_aversion = risk_aversion
        self.penalty_strength = penalty_strength
        self.budget = budget
        self.num_bits_per_asset = num_bits_per_asset
        self.max_optimizer_iterations = max_optimizer_iterations

    def optimize(
        self, returns: np.ndarray, covariance: np.ndarray
    ) -> PortfolioResult:
        """Run QAOA portfolio optimization.

        Parameters
        ----------
        returns : np.ndarray
            Expected returns per asset (length n).
        covariance : np.ndarray
            Covariance matrix (n x n).

        Returns
        -------
        PortfolioResult
        """
        n = len(returns)
        qubo = portfolio_to_qubo(
            returns,
            covariance,
            self.risk_aversion,
            self.penalty_strength,
            self.budget,
            self.num_bits_per_asset,
        )
        ising = qubo_to_ising(qubo)
        p = self.num_layers

        # Cost function
        def cost_fn(params: np.ndarray) -> float:
            gammas = params[:p]
            betas = params[p:]
            probs = _qaoa_statevector(ising, gammas, betas)
            nq = ising.num_qubits
            expected = 0.0
            for idx in range(len(probs)):
                if probs[idx] < 1e-15:
                    continue
                spins = np.array(
                    [1.0 - 2.0 * ((idx >> q) & 1) for q in range(nq)]
                )
                expected += probs[idx] * ising.energy(spins)
            return expected

        # Nelder-Mead optimization
        initial = np.full(2 * p, 0.1)
        result = _nelder_mead(cost_fn, initial, self.max_optimizer_iterations)

        # Decode result
        gammas_opt = result.params[:p]
        betas_opt = result.params[p:]
        probs = _qaoa_statevector(ising, gammas_opt, betas_opt)
        best_idx = int(np.argmax(probs))

        nq = ising.num_qubits
        best_bits = np.array(
            [(best_idx >> q) & 1 for q in range(nq)], dtype=np.int64
        )

        weights = _decode_weights(
            best_bits, n, self.num_bits_per_asset
        )

        variance = float(weights @ covariance @ weights)
        exp_return = float(returns @ weights)
        objective = variance - self.risk_aversion * exp_return

        return PortfolioResult(
            weights=weights,
            objective=objective,
            expected_return=exp_return,
            variance=variance,
            best_bitstring=best_bits,
            optimal_angles=result.params,
            iterations=result.iterations,
        )


# ============================================================
# Classical brute-force solver (for validation)
# ============================================================


def classical_portfolio_optimize(
    returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float = 0.5,
    budget: int | None = None,
) -> PortfolioResult:
    """Brute-force classical solver for small portfolio problems.

    Enumerates all 2^n bitstrings (1 bit per asset) and returns the
    one with the best risk-return objective.

    Parameters
    ----------
    returns : np.ndarray
        Expected returns (length n).
    covariance : np.ndarray
        Covariance matrix (n x n).
    risk_aversion : float
        Lambda parameter.
    budget : int or None
        Exact number of assets to hold.  None means no constraint.
    """
    n = len(returns)
    if n > 20:
        raise ValueError("Brute-force solver limited to n <= 20")

    best_obj = np.inf
    best_bits = np.zeros(n, dtype=np.int64)

    for idx in range(1 << n):
        bits = np.array([(idx >> q) & 1 for q in range(n)], dtype=float)
        if budget is not None and int(bits.sum()) != budget:
            continue
        w = bits / max(bits.sum(), 1e-10)
        var = float(w @ covariance @ w)
        ret = float(returns @ w)
        obj = var - risk_aversion * ret
        if obj < best_obj:
            best_obj = obj
            best_bits = bits.astype(np.int64)

    weights = best_bits.astype(float)
    s = weights.sum()
    if s > 0:
        weights /= s
    variance = float(weights @ covariance @ weights)
    exp_return = float(returns @ weights)

    return PortfolioResult(
        weights=weights,
        objective=best_obj,
        expected_return=exp_return,
        variance=variance,
        best_bitstring=best_bits,
        optimal_angles=np.array([]),
        iterations=0,
    )


# ============================================================
# Efficient frontier
# ============================================================


def compute_efficient_frontier(
    returns: np.ndarray,
    covariance: np.ndarray,
    num_points: int = 20,
) -> list[EfficientFrontierPoint]:
    """Compute the mean-variance efficient frontier via analytical solution.

    Uses the two-fund theorem to trace out the frontier for the
    unconstrained (long/short) case.

    Parameters
    ----------
    returns : np.ndarray
        Expected returns (length n).
    covariance : np.ndarray
        Covariance matrix (n x n).
    num_points : int
        Number of frontier points.

    Returns
    -------
    list[EfficientFrontierPoint]
    """
    n = len(returns)
    ones = np.ones(n)

    try:
        cov_inv = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(covariance)

    a_val = float(ones @ cov_inv @ returns)
    b_val = float(returns @ cov_inv @ returns)
    c_val = float(ones @ cov_inv @ ones)
    det = b_val * c_val - a_val ** 2

    if abs(det) < 1e-15:
        # Degenerate case: all returns identical
        w = cov_inv @ ones / c_val
        var = float(w @ covariance @ w)
        return [
            EfficientFrontierPoint(
                target_return=float(returns @ w),
                weights=w,
                variance=var,
            )
        ]

    r_min = a_val / c_val
    r_max = max(returns)
    targets = np.linspace(r_min, r_max, num_points)

    frontier: list[EfficientFrontierPoint] = []
    for r_target in targets:
        # w* = (1/det) * [(b - a*r_target) Sigma^{-1} 1 + (c*r_target - a) Sigma^{-1} r]
        lam = (c_val * r_target - a_val) / det
        mu = (b_val - a_val * r_target) / det
        w = cov_inv @ (lam * returns + mu * ones)
        var = float(w @ covariance @ w)
        frontier.append(
            EfficientFrontierPoint(
                target_return=float(r_target),
                weights=w,
                variance=var,
            )
        )

    return frontier


# ============================================================
# Nelder-Mead optimizer (minimal, standalone)
# ============================================================


@dataclass
class _NMResult:
    params: np.ndarray
    cost: float
    iterations: int


def _nelder_mead(
    fn: Callable[[np.ndarray], float],
    initial: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> _NMResult:
    """Simplified Nelder-Mead (downhill simplex)."""
    n = len(initial)
    simplex = [initial.copy()]
    for i in range(n):
        pt = initial.copy()
        pt[i] += 0.05 * max(abs(pt[i]), 0.005)
        simplex.append(pt)

    values = [fn(p) for p in simplex]

    for iteration in range(max_iter):
        order = np.argsort(values)
        best_i = order[0]
        worst_i = order[-1]
        second_worst_i = order[-2]

        if values[worst_i] - values[best_i] < tol:
            break

        centroid = np.mean([simplex[i] for i in order[:-1]], axis=0)

        # Reflection
        reflected = 2.0 * centroid - simplex[worst_i]
        fr = fn(reflected)

        if fr < values[second_worst_i] and fr >= values[best_i]:
            simplex[worst_i] = reflected
            values[worst_i] = fr
            continue

        if fr < values[best_i]:
            expanded = 3.0 * centroid - 2.0 * simplex[worst_i]
            fe = fn(expanded)
            if fe < fr:
                simplex[worst_i] = expanded
                values[worst_i] = fe
            else:
                simplex[worst_i] = reflected
                values[worst_i] = fr
            continue

        contracted = 0.5 * (centroid + simplex[worst_i])
        fc = fn(contracted)
        if fc < values[worst_i]:
            simplex[worst_i] = contracted
            values[worst_i] = fc
            continue

        best = simplex[best_i].copy()
        for idx in range(n + 1):
            if idx != best_i:
                simplex[idx] = 0.5 * (simplex[idx] + best)
                values[idx] = fn(simplex[idx])

    best_i = int(np.argmin(values))
    return _NMResult(
        params=simplex[best_i], cost=values[best_i], iterations=iteration + 1
    )


# ============================================================
# Weight decoding
# ============================================================


def _decode_weights(
    bits: np.ndarray, num_assets: int, bits_per_asset: int
) -> np.ndarray:
    """Decode a binary bitstring into portfolio weights."""
    if bits_per_asset == 1:
        weights = bits[:num_assets].astype(float)
    else:
        weights = np.zeros(num_assets)
        for i in range(num_assets):
            val = 0
            for k in range(bits_per_asset):
                val += int(bits[i * bits_per_asset + k]) << k
            weights[i] = val / max((1 << bits_per_asset) - 1, 1)

    s = weights.sum()
    if s > 1e-10:
        weights /= s
    else:
        weights = np.ones(num_assets) / num_assets
    return weights


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    print("=== Portfolio Optimization self-test ===")

    # 3-asset problem
    returns_arr = np.array([0.10, 0.12, 0.08])
    cov_mat = np.array([
        [0.04, 0.006, 0.002],
        [0.006, 0.09, 0.009],
        [0.002, 0.009, 0.01],
    ])

    # Classical brute-force
    classical = classical_portfolio_optimize(
        returns_arr, cov_mat, risk_aversion=0.5, budget=2
    )
    print(f"Classical: weights={classical.weights}, obj={classical.objective:.6f}")

    # QAOA
    qaoa = PortfolioOptimizer(num_layers=2, risk_aversion=0.5, budget=2)
    quantum = qaoa.optimize(returns_arr, cov_mat)
    print(f"QAOA:      weights={quantum.weights}, obj={quantum.objective:.6f}")

    # Efficient frontier
    frontier = compute_efficient_frontier(returns_arr, cov_mat, num_points=5)
    for pt in frontier:
        print(f"  return={pt.target_return:.4f}  var={pt.variance:.6f}")

    print("Self-test complete.")
