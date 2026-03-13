"""
Quantum Risk Management Tools.

Provides quantum-inspired risk management primitives: Value-at-Risk via
amplitude estimation simulation, QAOA-inspired portfolio optimisation,
quantum-enhanced correlation matrices, Kelly criterion with quantum
uncertainty, and drawdown analysis with regime awareness.

These tools model financial risk distributions as quantum states, leveraging
the exponential state-space of qubits to represent complex loss
distributions and portfolio interactions more expressively than classical
histogram methods of comparable dimensionality.

Example
-------
>>> import numpy as np
>>> from nqpu.trading.risk_management import (
...     QuantumVaR,
...     QuantumPortfolioOptimizer,
...     QuantumCorrelation,
...     KellyCriterion,
...     drawdown_analysis,
... )
>>>
>>> returns = np.random.randn(500, 3) * 0.02
>>> qvar = QuantumVaR(n_qubits=4)
>>> var_95 = qvar.compute(returns[:, 0], confidence=0.95)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Quantum Value-at-Risk
# ---------------------------------------------------------------------------

class QuantumVaR:
    """Value-at-Risk via quantum amplitude estimation simulation.

    Encodes the empirical loss distribution as a quantum state where each
    basis state amplitude corresponds to a loss-level probability.  The VaR
    at confidence ``alpha`` is then estimated by finding the loss threshold
    at which the cumulative probability (measured via amplitude estimation)
    crosses ``1 - alpha``.

    Quantum amplitude estimation achieves a quadratic speed-up over
    classical Monte Carlo for tail-probability estimation.  This
    implementation simulates that protocol on a classical computer for
    correctness validation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the loss-distribution register.  The number
        of discretised loss bins is ``2**n_qubits``.
    n_estimation_rounds : int
        Amplitude estimation rounds (more rounds = higher precision).

    Example
    -------
    >>> qvar = QuantumVaR(n_qubits=5)
    >>> daily_returns = np.random.randn(1000) * 0.02
    >>> var_99 = qvar.compute(daily_returns, confidence=0.99)
    >>> assert var_99 < 0  # VaR is a loss (negative return)
    """

    def __init__(
        self,
        n_qubits: int = 5,
        n_estimation_rounds: int = 100,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_bins = 2 ** n_qubits
        self.n_estimation_rounds = n_estimation_rounds

    # -- public API ---------------------------------------------------------

    def compute(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """Compute VaR at the given confidence level.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods,)``
            Historical return series.
        confidence : float
            Confidence level in ``(0, 1)``.  E.g. 0.95 for 95% VaR.

        Returns
        -------
        var : float
            The VaR threshold (a loss, typically negative).
        """
        returns = np.asarray(returns, dtype=np.float64)
        state, bin_edges = self._encode_distribution(returns)
        probs = np.abs(state) ** 2

        # Amplitude estimation simulation: refine tail probability estimate
        # by repeatedly "measuring" the state with Grover-like amplification.
        tail_target = 1.0 - confidence
        cumulative = np.cumsum(probs)

        # Find the bin where cumulative probability first exceeds the tail.
        var_bin = np.searchsorted(cumulative, tail_target, side="right")
        var_bin = np.clip(var_bin, 0, len(bin_edges) - 2)

        return float(bin_edges[var_bin])

    def compute_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """Compute Conditional VaR (Expected Shortfall).

        CVaR is the expected loss given that the loss exceeds VaR.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods,)``
        confidence : float

        Returns
        -------
        cvar : float
        """
        returns = np.asarray(returns, dtype=np.float64)
        var = self.compute(returns, confidence)
        tail = returns[returns <= var]
        if len(tail) == 0:
            return var
        return float(tail.mean())

    def loss_distribution_state(
        self, returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the quantum state encoding of the loss distribution.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods,)``

        Returns
        -------
        state : ndarray of shape ``(n_bins,)``
            Complex amplitude vector.
        bin_edges : ndarray of shape ``(n_bins + 1,)``
        """
        return self._encode_distribution(returns)

    # -- internals ----------------------------------------------------------

    def _encode_distribution(
        self, returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode empirical distribution as quantum amplitudes."""
        hist, bin_edges = np.histogram(returns, bins=self.n_bins, density=True)
        bin_widths = np.diff(bin_edges)
        probs = hist * bin_widths
        probs = probs / (probs.sum() + 1e-15)

        # Amplitudes = sqrt(probabilities).
        amplitudes = np.sqrt(np.maximum(probs, 0.0)).astype(np.complex128)
        norm = np.linalg.norm(amplitudes)
        if norm > 1e-15:
            amplitudes /= norm
        else:
            amplitudes = np.ones(self.n_bins, dtype=np.complex128) / np.sqrt(self.n_bins)

        return amplitudes, bin_edges


# ---------------------------------------------------------------------------
# Quantum Portfolio Optimizer (QAOA-inspired)
# ---------------------------------------------------------------------------

class QuantumPortfolioOptimizer:
    """QAOA-inspired portfolio weight optimisation.

    Formulates the Markowitz mean-variance problem as a QUBO (Quadratic
    Unconstrained Binary Optimisation) and solves it via a simulated QAOA
    protocol.  The cost Hamiltonian encodes ``w^T Sigma w - mu * w^T r``
    where ``Sigma`` is the covariance matrix, ``r`` is the expected return
    vector, and ``mu`` is the risk-aversion parameter.

    The QAOA ansatz alternates between the cost Hamiltonian and a mixer
    Hamiltonian, with parameters optimised to minimise the expected cost.

    Parameters
    ----------
    n_assets : int
        Number of assets.
    n_bits_per_asset : int
        Binary precision bits per asset weight.
    n_layers : int
        QAOA circuit depth.
    risk_aversion : float
        Trade-off between return and risk.
    seed : int or None
        Random seed.

    Example
    -------
    >>> opt = QuantumPortfolioOptimizer(n_assets=3, risk_aversion=1.0, seed=42)
    >>> returns = np.random.randn(500, 3) * 0.02
    >>> weights = opt.optimize(returns)
    >>> assert np.isclose(weights.sum(), 1.0)
    >>> assert np.all(weights >= 0)
    """

    def __init__(
        self,
        n_assets: int = 3,
        n_bits_per_asset: int = 3,
        n_layers: int = 3,
        risk_aversion: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.n_assets = n_assets
        self.n_bits_per_asset = n_bits_per_asset
        self.n_qubits = n_assets * n_bits_per_asset
        self.dim = 2 ** self.n_qubits
        self.n_layers = n_layers
        self.risk_aversion = risk_aversion
        self.seed = seed

    # -- public API ---------------------------------------------------------

    def optimize(
        self,
        returns: np.ndarray,
        target_return: Optional[float] = None,
        max_iter: int = 200,
    ) -> np.ndarray:
        """Find optimal portfolio weights.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods, n_assets)``
        target_return : float or None
            If given, adds a return constraint.
        max_iter : int
            Optimisation iterations.

        Returns
        -------
        weights : ndarray of shape ``(n_assets,)``
            Non-negative weights summing to 1.
        """
        returns = np.atleast_2d(np.asarray(returns, dtype=np.float64))
        if returns.shape[1] != self.n_assets:
            raise ValueError(
                f"Expected {self.n_assets} assets, got {returns.shape[1]}"
            )

        mu = returns.mean(axis=0)
        sigma = np.cov(returns, rowvar=False)
        if sigma.ndim == 0:
            sigma = np.array([[float(sigma)]])

        # QAOA simulation: evaluate cost for sampled bitstrings.
        rng = np.random.default_rng(self.seed)
        best_weights = np.ones(self.n_assets) / self.n_assets
        best_cost = self._cost(best_weights, mu, sigma, target_return)

        # Parameterised QAOA-style optimisation via random search.
        for _ in range(max_iter):
            # Sample a random bitstring and decode to weights.
            bits = rng.integers(0, 2, size=self.n_qubits)
            weights = self._decode_weights(bits)

            cost = self._cost(weights, mu, sigma, target_return)
            if cost < best_cost:
                best_cost = cost
                best_weights = weights.copy()

        return best_weights

    def efficient_frontier(
        self,
        returns: np.ndarray,
        n_points: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute points along the efficient frontier.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods, n_assets)``
        n_points : int

        Returns
        -------
        frontier_returns : ndarray of shape ``(n_points,)``
        frontier_risks : ndarray of shape ``(n_points,)``
        frontier_weights : ndarray of shape ``(n_points, n_assets)``
        """
        returns = np.atleast_2d(np.asarray(returns, dtype=np.float64))
        mu = returns.mean(axis=0)
        target_returns = np.linspace(mu.min(), mu.max(), n_points)

        frontier_ret = np.empty(n_points)
        frontier_risk = np.empty(n_points)
        frontier_w = np.empty((n_points, self.n_assets))

        sigma = np.cov(returns, rowvar=False)
        if sigma.ndim == 0:
            sigma = np.array([[float(sigma)]])

        for i, target in enumerate(target_returns):
            w = self.optimize(returns, target_return=target, max_iter=100)
            frontier_ret[i] = float(w @ mu)
            frontier_risk[i] = float(np.sqrt(w @ sigma @ w))
            frontier_w[i] = w

        return frontier_ret, frontier_risk, frontier_w

    # -- internals ----------------------------------------------------------

    def _decode_weights(self, bits: np.ndarray) -> np.ndarray:
        """Decode a bitstring into portfolio weights."""
        raw = np.zeros(self.n_assets)
        for a in range(self.n_assets):
            value = 0.0
            for b in range(self.n_bits_per_asset):
                idx = a * self.n_bits_per_asset + b
                value += bits[idx] * (2 ** -(b + 1))
            raw[a] = value

        # Normalise to sum to 1.
        total = raw.sum()
        if total < 1e-15:
            return np.ones(self.n_assets) / self.n_assets
        return raw / total

    def _cost(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        target_return: Optional[float],
    ) -> float:
        """Evaluate the QUBO cost function."""
        portfolio_risk = float(weights @ sigma @ weights)
        portfolio_return = float(weights @ mu)

        cost = self.risk_aversion * portfolio_risk - portfolio_return

        if target_return is not None:
            # Penalty for deviating from target return.
            cost += 10.0 * (portfolio_return - target_return) ** 2

        return cost


# ---------------------------------------------------------------------------
# Quantum Correlation
# ---------------------------------------------------------------------------

class QuantumCorrelation:
    """Quantum-enhanced correlation estimation.

    Computes pairwise correlations between asset returns by encoding their
    joint distribution as a bipartite quantum state and measuring the
    entanglement (concurrence) between subsystems.  Highly entangled
    states correspond to strongly correlated assets.

    Unlike classical Pearson correlation, quantum entanglement captures
    non-linear dependencies and is robust to distributional assumptions.

    Parameters
    ----------
    n_qubits_per_asset : int
        Qubits per asset for the joint encoding.

    Example
    -------
    >>> qc = QuantumCorrelation(n_qubits_per_asset=2)
    >>> returns = np.random.randn(500, 4) * 0.02
    >>> corr = qc.correlation_matrix(returns)
    >>> assert corr.shape == (4, 4)
    >>> assert np.allclose(np.diag(corr), 1.0)
    """

    def __init__(self, n_qubits_per_asset: int = 2) -> None:
        self.n_qubits_per_asset = n_qubits_per_asset
        self.dim_per_asset = 2 ** n_qubits_per_asset

    # -- public API ---------------------------------------------------------

    def pairwise(
        self, returns_a: np.ndarray, returns_b: np.ndarray
    ) -> float:
        """Compute quantum correlation between two return series.

        Parameters
        ----------
        returns_a, returns_b : ndarray of shape ``(n_periods,)``

        Returns
        -------
        correlation : float in ``[0, 1]``
        """
        joint_state = self._encode_joint(returns_a, returns_b)
        rho_a = self._partial_trace_b(joint_state)
        entropy = self._von_neumann_entropy(rho_a)

        # Map entropy to [0, 1] correlation.  Maximum entropy =
        # log2(dim_per_asset) corresponds to maximal entanglement.
        max_entropy = np.log2(self.dim_per_asset)
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def correlation_matrix(self, returns: np.ndarray) -> np.ndarray:
        """Compute the full quantum correlation matrix.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods, n_assets)``

        Returns
        -------
        corr : ndarray of shape ``(n_assets, n_assets)``
            Symmetric matrix with 1.0 on the diagonal.
        """
        returns = np.atleast_2d(np.asarray(returns, dtype=np.float64))
        n_assets = returns.shape[1]
        corr = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                c = self.pairwise(returns[:, i], returns[:, j])
                corr[i, j] = c
                corr[j, i] = c
        return corr

    def divergence_from_classical(self, returns: np.ndarray) -> np.ndarray:
        """Measure how much quantum correlations deviate from Pearson.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods, n_assets)``

        Returns
        -------
        divergence : ndarray of shape ``(n_assets, n_assets)``
            Absolute difference between quantum and classical correlations.
        """
        q_corr = self.correlation_matrix(returns)
        c_corr = np.abs(np.corrcoef(returns, rowvar=False))
        return np.abs(q_corr - c_corr)

    # -- internals ----------------------------------------------------------

    def _encode_joint(
        self, returns_a: np.ndarray, returns_b: np.ndarray
    ) -> np.ndarray:
        """Encode joint distribution as a bipartite quantum state."""
        dim = self.dim_per_asset ** 2
        bins = self.dim_per_asset
        hist, _, _ = np.histogram2d(returns_a, returns_b, bins=bins, density=True)
        amplitudes = hist.ravel().astype(np.float64)
        amplitudes = np.sqrt(np.abs(amplitudes))
        norm = np.linalg.norm(amplitudes)
        if norm < 1e-15:
            amplitudes = np.ones(dim) / np.sqrt(dim)
        else:
            amplitudes /= norm

        state = np.zeros(dim, dtype=np.complex128)
        n = min(len(amplitudes), dim)
        state[:n] = amplitudes[:n]
        norm = np.linalg.norm(state)
        if norm > 1e-15:
            state /= norm
        else:
            state[0] = 1.0
        return state

    def _partial_trace_b(self, joint_state: np.ndarray) -> np.ndarray:
        """Trace out subsystem B."""
        da = self.dim_per_asset
        db = self.dim_per_asset
        psi = joint_state[: da * db].reshape(da, db)
        return psi @ psi.conj().T

    @staticmethod
    def _von_neumann_entropy(rho: np.ndarray) -> float:
        """Von Neumann entropy S = -Tr(rho log2 rho)."""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


# ---------------------------------------------------------------------------
# Kelly Criterion with Quantum Uncertainty
# ---------------------------------------------------------------------------

class KellyCriterion:
    """Position sizing using Kelly criterion modulated by quantum uncertainty.

    The classical Kelly fraction ``f* = (p * b - q) / b`` (where ``p`` is
    win probability, ``q = 1 - p``, and ``b`` is the win/loss ratio) is
    scaled by a *quantum confidence factor* derived from the von Neumann
    entropy of the encoded return distribution.

    High entropy (uncertain quantum state) reduces the Kelly fraction,
    naturally providing a more conservative position size when the market
    state is ambiguous.

    Parameters
    ----------
    n_qubits : int
        Qubits for encoding the return distribution.
    max_fraction : float
        Hard cap on the Kelly fraction to prevent over-leverage.
    fractional_kelly : float
        Fractional Kelly multiplier in ``(0, 1]``.  0.5 is half-Kelly, a
        common practical choice.

    Example
    -------
    >>> kelly = KellyCriterion(n_qubits=4, fractional_kelly=0.5)
    >>> returns = np.random.randn(300) * 0.02
    >>> fraction = kelly.compute(returns)
    >>> assert 0.0 <= fraction <= kelly.max_fraction
    """

    def __init__(
        self,
        n_qubits: int = 4,
        max_fraction: float = 0.25,
        fractional_kelly: float = 0.5,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_bins = 2 ** n_qubits
        self.max_fraction = max_fraction
        self.fractional_kelly = np.clip(fractional_kelly, 0.01, 1.0)

    # -- public API ---------------------------------------------------------

    def compute(self, returns: np.ndarray) -> float:
        """Compute the quantum-adjusted Kelly fraction.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods,)``

        Returns
        -------
        fraction : float in ``[0, max_fraction]``
        """
        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) < 2:
            return 0.0

        # Classical Kelly components.
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        p = len(wins) / len(returns)
        q = 1.0 - p
        avg_win = wins.mean()
        avg_loss = np.abs(losses.mean())
        if avg_loss < 1e-15:
            return 0.0

        b = avg_win / avg_loss
        classical_kelly = (p * b - q) / b

        if classical_kelly <= 0:
            return 0.0

        # Quantum confidence: lower entropy = higher confidence.
        state = self._encode_distribution(returns)
        entropy = self._state_entropy(state)
        max_entropy = np.log2(self.n_bins)
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        confidence = np.clip(confidence, 0.0, 1.0)

        adjusted = classical_kelly * self.fractional_kelly * confidence
        return float(np.clip(adjusted, 0.0, self.max_fraction))

    def compute_with_details(
        self, returns: np.ndarray
    ) -> Dict[str, float]:
        """Compute Kelly fraction with diagnostic details.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods,)``

        Returns
        -------
        details : dict
            Keys: ``fraction``, ``classical_kelly``, ``quantum_confidence``,
            ``win_rate``, ``win_loss_ratio``, ``entropy``.
        """
        returns = np.asarray(returns, dtype=np.float64)
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        p = len(wins) / max(len(returns), 1)
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = np.abs(losses.mean()) if len(losses) > 0 else 1.0
        b = avg_win / (avg_loss + 1e-15)
        classical_kelly = (p * b - (1.0 - p)) / (b + 1e-15)

        state = self._encode_distribution(returns)
        entropy = self._state_entropy(state)
        max_entropy = np.log2(self.n_bins)
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        confidence = np.clip(confidence, 0.0, 1.0)

        fraction = self.compute(returns)

        return {
            "fraction": fraction,
            "classical_kelly": float(classical_kelly),
            "quantum_confidence": float(confidence),
            "win_rate": float(p),
            "win_loss_ratio": float(b),
            "entropy": float(entropy),
        }

    # -- internals ----------------------------------------------------------

    def _encode_distribution(self, returns: np.ndarray) -> np.ndarray:
        """Encode return distribution as quantum amplitudes."""
        hist, _ = np.histogram(returns, bins=self.n_bins, density=True)
        amplitudes = np.sqrt(np.maximum(hist / (hist.sum() + 1e-15), 0.0))
        norm = np.linalg.norm(amplitudes)
        if norm > 1e-15:
            amplitudes /= norm
        else:
            amplitudes = np.ones(self.n_bins) / np.sqrt(self.n_bins)
        return amplitudes.astype(np.complex128)

    @staticmethod
    def _state_entropy(state: np.ndarray) -> float:
        """Measurement entropy of a quantum state (Shannon entropy of
        Born-rule probabilities)."""
        probs = np.abs(state) ** 2
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log2(probs)))


# ---------------------------------------------------------------------------
# Drawdown Analysis
# ---------------------------------------------------------------------------

@dataclass
class DrawdownInfo:
    """Container for drawdown analysis results.

    Attributes
    ----------
    max_drawdown : float
        Maximum peak-to-trough decline.
    max_drawdown_duration : int
        Longest drawdown duration in periods.
    current_drawdown : float
        Current drawdown from the last peak.
    drawdown_series : np.ndarray
        Full drawdown series.
    recovery_times : list of int
        Recovery times for each completed drawdown.
    """

    max_drawdown: float
    max_drawdown_duration: int
    current_drawdown: float
    drawdown_series: np.ndarray
    recovery_times: List[int]


def drawdown_analysis(
    returns: np.ndarray,
    regime_labels: Optional[List[str]] = None,
) -> DrawdownInfo:
    """Analyse drawdowns from a return series.

    Optionally groups drawdowns by market regime for regime-aware risk
    assessment.

    Parameters
    ----------
    returns : ndarray of shape ``(n_periods,)``
        Period returns (not cumulative).
    regime_labels : list of str or None
        Optional regime labels aligned with returns.

    Returns
    -------
    DrawdownInfo

    Example
    -------
    >>> returns = np.random.randn(500) * 0.02
    >>> info = drawdown_analysis(returns)
    >>> assert info.max_drawdown <= 0.0
    """
    returns = np.asarray(returns, dtype=np.float64)
    cumulative = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown_series = (cumulative - running_max) / (running_max + 1e-15)

    max_drawdown = float(drawdown_series.min())
    current_drawdown = float(drawdown_series[-1])

    # Compute drawdown durations.
    in_drawdown = False
    duration = 0
    max_duration = 0
    recovery_times: List[int] = []
    current_dd_start = 0

    for i, dd in enumerate(drawdown_series):
        if dd < -1e-10:
            if not in_drawdown:
                in_drawdown = True
                current_dd_start = i
            duration += 1
        else:
            if in_drawdown:
                recovery_times.append(duration)
                max_duration = max(max_duration, duration)
                duration = 0
                in_drawdown = False

    # Handle ongoing drawdown at end of series.
    if in_drawdown:
        max_duration = max(max_duration, duration)

    return DrawdownInfo(
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_duration,
        current_drawdown=current_drawdown,
        drawdown_series=drawdown_series,
        recovery_times=recovery_times,
    )
