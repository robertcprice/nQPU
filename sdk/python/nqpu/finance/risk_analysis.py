"""Quantum-enhanced risk analysis: VaR, CVaR, and risk metrics.

Implements Value at Risk (VaR) and Conditional VaR (CVaR, also known as
Expected Shortfall) estimation using quantum amplitude estimation for a
quadratic speedup over classical Monte Carlo.

The quantum approach encodes the portfolio loss distribution into a quantum
state and uses QAE to estimate the probability of losses exceeding a
threshold (for VaR) and the conditional mean of losses beyond VaR (for CVaR).

Classical Monte Carlo baselines are provided for comparison.

References:
  - Woerner & Egger (2019), "Quantum Risk Analysis"
  - Egger et al. (2020), "Quantum Computing for Finance"
  - Barkoutsos et al. (2020), "Improving Variational Quantum Optimization
    using CVaR"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

import numpy as np

from .amplitude_estimation import (
    AEResult,
    CanonicalQAE,
    IterativeQAE,
    MaxLikelihoodQAE,
)


# ============================================================
# Configuration
# ============================================================


class DistributionModel(Enum):
    """Return distribution model."""

    NORMAL = auto()
    STUDENT_T = auto()
    HISTORICAL = auto()


@dataclass
class RiskConfig:
    """Configuration for risk analysis.

    Attributes
    ----------
    confidence_level : float
        Confidence level for VaR, e.g. 0.95 for 95% VaR.
    time_horizon : int
        Time horizon in days.
    num_scenarios : int
        Number of MC scenarios to generate.
    distribution : DistributionModel
        Which return distribution to use.
    df : float
        Degrees of freedom (only for Student-t).
    num_qubits : int
        Number of qubits for the QAE distribution encoding.
    seed : int
        Random seed for reproducibility.
    """

    confidence_level: float = 0.95
    time_horizon: int = 1
    num_scenarios: int = 10000
    distribution: DistributionModel = DistributionModel.NORMAL
    df: float = 5.0
    num_qubits: int = 4
    seed: int = 42


# ============================================================
# Result types
# ============================================================


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics.

    Attributes
    ----------
    var : float
        Value at Risk (positive number = loss amount).
    cvar : float
        Conditional VaR / Expected Shortfall (positive = loss amount).
    max_drawdown : float
        Maximum drawdown from the scenario series.
    sharpe_ratio : float
        Risk-adjusted return (mean / std).
    sortino_ratio : float
        Downside-risk-adjusted return (mean / downside_std).
    var_confidence_interval : tuple[float, float]
        95% CI on the VaR estimate.
    method : str
        Description of the computation method.
    """

    var: float
    cvar: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    var_confidence_interval: tuple[float, float] = (0.0, 0.0)
    method: str = "classical"


# ============================================================
# Scenario generation
# ============================================================


def generate_scenarios(
    mean: float,
    vol: float,
    config: RiskConfig,
) -> np.ndarray:
    """Generate portfolio return scenarios.

    Parameters
    ----------
    mean : float
        Annualized expected return of the portfolio.
    vol : float
        Annualized volatility of the portfolio.
    config : RiskConfig
        Configuration parameters.

    Returns
    -------
    np.ndarray
        Array of simulated returns, shape (num_scenarios,).
    """
    rng = np.random.RandomState(config.seed)
    n = config.num_scenarios
    horizon = float(config.time_horizon)

    if config.distribution == DistributionModel.NORMAL:
        z = rng.randn(n)
        return mean * horizon / 252.0 + vol * np.sqrt(horizon / 252.0) * z
    elif config.distribution == DistributionModel.STUDENT_T:
        t_samples = rng.standard_t(config.df, size=n)
        # Scale to match vol
        t_std = np.sqrt(config.df / (config.df - 2)) if config.df > 2 else 3.0
        scaled = t_samples / t_std
        return mean * horizon / 252.0 + vol * np.sqrt(horizon / 252.0) * scaled
    else:
        raise ValueError(
            "HISTORICAL distribution requires pre-generated scenarios"
        )


# ============================================================
# Classical risk computation
# ============================================================


def compute_var(scenarios: np.ndarray, confidence: float = 0.95) -> float:
    """Compute classical Value at Risk.

    VaR at confidence level alpha is the loss value such that
    P(loss > VaR) = 1 - alpha.

    Parameters
    ----------
    scenarios : np.ndarray
        Simulated returns.
    confidence : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    float
        VaR as a positive loss amount.
    """
    sorted_returns = np.sort(scenarios)
    idx = int((1.0 - confidence) * len(sorted_returns))
    idx = min(idx, len(sorted_returns) - 1)
    return float(-sorted_returns[idx])


def compute_cvar(
    scenarios: np.ndarray, var_value: float
) -> float:
    """Compute classical Conditional VaR (Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR.

    Parameters
    ----------
    scenarios : np.ndarray
        Simulated returns.
    var_value : float
        VaR threshold (positive number).

    Returns
    -------
    float
        CVaR as a positive loss amount.
    """
    threshold = -var_value
    tail = scenarios[scenarios <= threshold]
    if len(tail) == 0:
        return var_value
    return float(-tail.mean())


def compute_max_drawdown(scenarios: np.ndarray) -> float:
    """Compute maximum drawdown from a return series.

    Parameters
    ----------
    scenarios : np.ndarray
        Sequential returns.

    Returns
    -------
    float
        Maximum drawdown (as a positive fraction).
    """
    cumulative = np.cumprod(1.0 + scenarios)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / np.maximum(peak, 1e-15)
    return float(drawdown.max())


def compute_sharpe_ratio(scenarios: np.ndarray) -> float:
    """Annualized Sharpe ratio from return scenarios."""
    mean_r = scenarios.mean()
    std_r = scenarios.std()
    if std_r < 1e-15:
        return 0.0
    return float(mean_r / std_r)


def compute_sortino_ratio(scenarios: np.ndarray) -> float:
    """Annualized Sortino ratio from return scenarios."""
    mean_r = scenarios.mean()
    downside = scenarios[scenarios < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std < 1e-15:
        return 0.0
    return float(mean_r / downside_std)


# ============================================================
# Quantum-enhanced risk estimation
# ============================================================


def _build_loss_oracle(
    scenarios: np.ndarray, threshold: float, num_qubits: int
) -> tuple[np.ndarray, list[int]]:
    """Build QAE oracle for estimating P(loss > threshold).

    Discretizes the loss distribution into 2^num_qubits bins, then builds
    a state-preparation unitary where the "good" states correspond to bins
    with losses exceeding the threshold.

    Parameters
    ----------
    scenarios : np.ndarray
        Simulated returns.
    threshold : float
        Loss threshold (as negative return, e.g. -VaR).
    num_qubits : int
        Qubits for discretization.

    Returns
    -------
    (oracle, good_indices)
    """
    n_bins = 1 << num_qubits

    # Histogram of returns
    hist, bin_edges = np.histogram(scenarios, bins=n_bins, density=False)
    probs = hist.astype(float)
    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        probs = np.ones(n_bins) / n_bins

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Oracle: load probability distribution
    oracle = np.zeros((n_bins, n_bins), dtype=complex)
    col0 = np.sqrt(np.maximum(probs, 0.0)).astype(complex)
    norm = np.linalg.norm(col0)
    if norm > 1e-15:
        col0 /= norm

    oracle[:, 0] = col0
    for j in range(1, n_bins):
        ej = np.zeros(n_bins, dtype=complex)
        ej[j] = 1.0
        v = ej.copy()
        for prev in range(j):
            proj = np.dot(oracle[:, prev].conj(), v)
            v -= proj * oracle[:, prev]
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            oracle[:, j] = v / nv
        else:
            oracle[:, j] = ej

    # Good indices: bins where return <= threshold (loss > threshold)
    good_indices = [i for i, center in enumerate(bin_centers) if center <= threshold]

    return oracle, good_indices


def quantum_var(
    scenarios: np.ndarray,
    confidence: float = 0.95,
    num_qubits: int = 4,
) -> tuple[float, AEResult]:
    """Estimate VaR using quantum amplitude estimation.

    Uses bisection on the threshold combined with QAE to estimate
    P(loss > threshold) = 1 - confidence.

    Parameters
    ----------
    scenarios : np.ndarray
        Simulated returns.
    confidence : float
        Confidence level.
    num_qubits : int
        Qubits for distribution encoding.

    Returns
    -------
    (var_estimate, ae_result) : tuple[float, AEResult]
    """
    # Classical VaR as initial estimate
    classical_var = compute_var(scenarios, confidence)

    # Bisection around the classical VaR
    target_prob = 1.0 - confidence
    low = 0.0
    high = max(classical_var * 3.0, abs(scenarios.min()))

    best_ae: AEResult | None = None
    best_var = classical_var

    for _ in range(20):
        mid = (low + high) / 2.0
        threshold = -mid  # negative return
        oracle, good_indices = _build_loss_oracle(scenarios, threshold, num_qubits)

        if not good_indices:
            # All bins above threshold => VaR is lower
            high = mid
            continue

        qae = CanonicalQAE(num_eval_qubits=max(num_qubits, 4))
        ae_result = qae.estimate(oracle, good_indices)
        best_ae = ae_result

        if ae_result.estimation > target_prob:
            low = mid
        else:
            high = mid
            best_var = mid

        if high - low < classical_var * 0.01:
            break

    if best_ae is None:
        best_ae = AEResult(
            estimation=target_prob,
            confidence_interval=(target_prob, target_prob),
            num_oracle_calls=0,
        )

    return best_var, best_ae


def quantum_cvar(
    scenarios: np.ndarray,
    var_value: float,
    num_qubits: int = 4,
) -> tuple[float, AEResult]:
    """Estimate CVaR using quantum amplitude estimation.

    Encodes the tail of the loss distribution (losses > VaR) and uses
    QAE to estimate the expected loss in the tail.

    Parameters
    ----------
    scenarios : np.ndarray
        Simulated returns.
    var_value : float
        VaR threshold (positive).
    num_qubits : int
        Qubits for encoding.

    Returns
    -------
    (cvar_estimate, ae_result) : tuple[float, AEResult]
    """
    threshold = -var_value
    tail = scenarios[scenarios <= threshold]

    if len(tail) == 0:
        return var_value, AEResult(
            estimation=0.0,
            confidence_interval=(0.0, 0.0),
            num_oracle_calls=0,
        )

    # Scale tail losses to [0, 1] for amplitude encoding
    losses = -tail  # positive losses
    max_loss = max(losses.max(), 1e-10)
    normalised = np.clip(losses / max_loss, 0.0, 1.0)

    n_bins = 1 << num_qubits
    hist, bin_edges = np.histogram(normalised, bins=n_bins, range=(0, 1), density=False)
    probs = hist.astype(float)
    total = probs.sum()
    if total > 0:
        probs /= total

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Oracle with payoff flag: good states encode weighted loss magnitude
    dim = 2 * n_bins
    oracle = np.zeros((dim, dim), dtype=complex)
    col0 = np.zeros(dim, dtype=complex)
    for i in range(n_bins):
        sqrt_p = np.sqrt(max(probs[i], 0.0))
        f = bin_centers[i]  # already in [0, 1]
        col0[2 * i] = sqrt_p * np.sqrt(max(1.0 - f, 0.0))
        col0[2 * i + 1] = sqrt_p * np.sqrt(max(f, 0.0))

    norm = np.linalg.norm(col0)
    if norm > 1e-15:
        col0 /= norm

    oracle[:, 0] = col0
    for j in range(1, dim):
        ej = np.zeros(dim, dtype=complex)
        ej[j] = 1.0
        v = ej.copy()
        for prev in range(j):
            proj = np.dot(oracle[:, prev].conj(), v)
            v -= proj * oracle[:, prev]
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            oracle[:, j] = v / nv
        else:
            oracle[:, j] = ej

    good_indices = list(range(1, dim, 2))

    qae = CanonicalQAE(num_eval_qubits=max(num_qubits, 4))
    ae_result = qae.estimate(oracle, good_indices)

    # Decode: estimated amplitude ~= E[normalised_loss] = E[loss]/max_loss
    cvar_estimate = ae_result.estimation * max_loss

    return cvar_estimate, ae_result


# ============================================================
# Main risk analyzer
# ============================================================


class RiskAnalyzer:
    """Quantum-enhanced risk analyzer.

    Computes VaR, CVaR, and other risk metrics using both classical MC
    and quantum amplitude estimation.

    Parameters
    ----------
    config : RiskConfig
        Risk analysis configuration.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def analyze(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        weights: np.ndarray,
    ) -> RiskMetrics:
        """Compute risk metrics for a portfolio.

        Parameters
        ----------
        returns : np.ndarray
            Expected asset returns (length n).
        covariance : np.ndarray
            Covariance matrix (n x n).
        weights : np.ndarray
            Portfolio weights (length n, sum to 1).

        Returns
        -------
        RiskMetrics
        """
        # Portfolio mean and volatility
        port_return = float(weights @ returns)
        port_var = float(weights @ covariance @ weights)
        port_vol = np.sqrt(port_var)

        # Generate scenarios
        scenarios = generate_scenarios(port_return, port_vol, self.config)

        return self.analyze_scenarios(scenarios)

    def analyze_scenarios(self, scenarios: np.ndarray) -> RiskMetrics:
        """Compute risk metrics from pre-generated scenarios.

        Parameters
        ----------
        scenarios : np.ndarray
            Array of return scenarios.

        Returns
        -------
        RiskMetrics
        """
        confidence = self.config.confidence_level

        # Classical metrics
        var_cl = compute_var(scenarios, confidence)
        cvar_cl = compute_cvar(scenarios, var_cl)
        max_dd = compute_max_drawdown(scenarios)
        sharpe = compute_sharpe_ratio(scenarios)
        sortino = compute_sortino_ratio(scenarios)

        # Quantum-enhanced VaR
        var_q, ae_var = quantum_var(
            scenarios, confidence, self.config.num_qubits
        )
        cvar_q, ae_cvar = quantum_cvar(
            scenarios, var_q, self.config.num_qubits
        )

        # Use quantum estimates (they refine the classical ones)
        ci_width = abs(
            ae_var.confidence_interval[1] - ae_var.confidence_interval[0]
        )
        var_ci = (max(var_q - ci_width * var_q, 0.0), var_q + ci_width * var_q)

        return RiskMetrics(
            var=var_q,
            cvar=cvar_q,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            var_confidence_interval=var_ci,
            method="QAE-enhanced",
        )

    def classical_analyze(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        weights: np.ndarray,
    ) -> RiskMetrics:
        """Pure classical risk analysis (no QAE).

        Parameters
        ----------
        returns, covariance, weights : np.ndarray
            Same as `analyze`.

        Returns
        -------
        RiskMetrics
        """
        port_return = float(weights @ returns)
        port_var = float(weights @ covariance @ weights)
        port_vol = np.sqrt(port_var)

        scenarios = generate_scenarios(port_return, port_vol, self.config)

        var_val = compute_var(scenarios, self.config.confidence_level)
        cvar_val = compute_cvar(scenarios, var_val)

        # Bootstrap CI for VaR
        rng = np.random.RandomState(self.config.seed + 1)
        var_bootstrap = []
        for _ in range(200):
            boot = rng.choice(scenarios, size=len(scenarios), replace=True)
            var_bootstrap.append(compute_var(boot, self.config.confidence_level))
        var_bootstrap.sort()
        ci_low = var_bootstrap[int(0.025 * len(var_bootstrap))]
        ci_high = var_bootstrap[int(0.975 * len(var_bootstrap))]

        return RiskMetrics(
            var=var_val,
            cvar=cvar_val,
            max_drawdown=compute_max_drawdown(scenarios),
            sharpe_ratio=compute_sharpe_ratio(scenarios),
            sortino_ratio=compute_sortino_ratio(scenarios),
            var_confidence_interval=(ci_low, ci_high),
            method="classical",
        )


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    print("=== Risk Analysis self-test ===")

    # 3-asset portfolio
    returns_arr = np.array([0.10, 0.15, 0.08])
    cov_mat = np.array([
        [0.04, 0.006, 0.002],
        [0.006, 0.09, 0.009],
        [0.002, 0.009, 0.01],
    ])
    weights_arr = np.array([0.4, 0.3, 0.3])

    config = RiskConfig(
        confidence_level=0.95,
        num_scenarios=5000,
        num_qubits=4,
    )

    analyzer = RiskAnalyzer(config)

    # Classical
    classical = analyzer.classical_analyze(returns_arr, cov_mat, weights_arr)
    print(f"Classical: VaR={classical.var:.4f}, CVaR={classical.cvar:.4f}")
    print(f"  Sharpe={classical.sharpe_ratio:.4f}, Sortino={classical.sortino_ratio:.4f}")

    # Quantum-enhanced
    quantum = analyzer.analyze(returns_arr, cov_mat, weights_arr)
    print(f"Quantum:   VaR={quantum.var:.4f}, CVaR={quantum.cvar:.4f}")
    print(f"  Sharpe={quantum.sharpe_ratio:.4f}, Sortino={quantum.sortino_ratio:.4f}")

    # Consistency checks
    assert classical.var > 0, "VaR should be positive"
    assert classical.cvar >= classical.var, "CVaR >= VaR"
    assert quantum.var > 0, "Quantum VaR should be positive"

    print("Self-test complete.")
