"""Quantum-enhanced option pricing via amplitude estimation.

Implements European, Asian, and Barrier option pricing using quantum
amplitude estimation (QAE) to achieve a quadratic speedup over classical
Monte Carlo.  Each pricer encodes the option payoff into a quantum oracle
whose "good" amplitude is proportional to the discounted expected payoff,
then applies QAE to estimate that amplitude.

Black-Scholes analytical formulas are provided for validation.

Architecture:

  1. Discretize the asset-price distribution (log-normal GBM) into
     2^n_price_qubits bins.
  2. Build a state-preparation unitary that loads the price distribution
     into a quantum register.
  3. Build a payoff comparator that marks states where the payoff is
     positive (these are the "good" states).
  4. Apply amplitude estimation to extract the expected payoff.

References:
  - Stamatopoulos et al. (2020), "Option Pricing using Quantum Computers"
  - Woerner & Egger (2019), "Quantum Risk Analysis"
  - Egger et al. (2020), "Quantum Computing for Finance"
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
# Option types
# ============================================================


class OptionType(Enum):
    """Supported option contract types."""

    EUROPEAN_CALL = auto()
    EUROPEAN_PUT = auto()
    ASIAN_CALL = auto()
    ASIAN_PUT = auto()
    BARRIER_UP_AND_OUT = auto()
    BARRIER_DOWN_AND_OUT = auto()


class QAEMethod(Enum):
    """Which amplitude estimation backend to use."""

    CANONICAL = auto()
    ITERATIVE = auto()
    MAX_LIKELIHOOD = auto()


# ============================================================
# Result container
# ============================================================


@dataclass
class OptionPricingResult:
    """Result of an option pricing computation.

    Attributes
    ----------
    price : float
        Estimated option price (discounted expected payoff).
    delta : float
        Price sensitivity to spot (first-order Greek).
    confidence_interval : tuple[float, float]
        95 % confidence interval around the price.
    num_oracle_calls : int
        Total Grover-operator applications.
    method : str
        Human-readable description of the pricing method.
    analytical_price : float or None
        Black-Scholes price for comparison (European only).
    """

    price: float
    delta: float
    confidence_interval: tuple[float, float]
    num_oracle_calls: int
    method: str
    analytical_price: float | None = None


# ============================================================
# Black-Scholes analytical formulas
# ============================================================


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun approximation)."""
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911
    sign = -1.0 if x < 0.0 else 1.0
    x_abs = abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(
        -x_abs * x_abs
    )
    return float(0.5 * (1.0 + sign * y))


def black_scholes_call(
    s: float, k: float, r: float, sigma: float, t: float
) -> float:
    """Black-Scholes European call price.

    Parameters
    ----------
    s : float
        Current spot price.
    k : float
        Strike price.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility (annualized).
    t : float
        Time to maturity in years.
    """
    if t <= 0.0:
        return max(s - k, 0.0)
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return float(s * _norm_cdf(d1) - k * np.exp(-r * t) * _norm_cdf(d2))


def black_scholes_put(
    s: float, k: float, r: float, sigma: float, t: float
) -> float:
    """Black-Scholes European put price."""
    if t <= 0.0:
        return max(k - s, 0.0)
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return float(
        k * np.exp(-r * t) * _norm_cdf(-d2) - s * _norm_cdf(-d1)
    )


def black_scholes_delta(
    s: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    is_call: bool = True,
) -> float:
    """Black-Scholes delta."""
    if t <= 0.0:
        if is_call:
            return 1.0 if s > k else 0.0
        return -1.0 if s < k else 0.0
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    delta_call = _norm_cdf(d1)
    return float(delta_call if is_call else delta_call - 1.0)


# ============================================================
# Price distribution loader
# ============================================================


def _build_price_distribution(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    num_bins: int,
    num_std: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize GBM terminal price into *num_bins* bins.

    Returns (bin_centers, probabilities) where probabilities are the
    log-normal PDF integrated over each bin.
    """
    # Log-normal parameters for terminal price
    mu_log = np.log(s0) + (r - 0.5 * sigma ** 2) * t
    sigma_log = sigma * np.sqrt(t)

    # Price range: [exp(mu - num_std*sigma), exp(mu + num_std*sigma)]
    log_low = mu_log - num_std * sigma_log
    log_high = mu_log + num_std * sigma_log
    s_low = max(np.exp(log_low), 1e-6)
    s_high = np.exp(log_high)

    bin_edges = np.linspace(s_low, s_high, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Log-normal CDF at bin edges
    cdf_vals = np.zeros(num_bins + 1)
    for i, edge in enumerate(bin_edges):
        if edge <= 0:
            cdf_vals[i] = 0.0
        else:
            z = (np.log(edge) - mu_log) / sigma_log
            cdf_vals[i] = _norm_cdf(z)

    probs = np.diff(cdf_vals)
    total = probs.sum()
    if total > 1e-15:
        probs /= total  # normalise to valid probability distribution
    else:
        probs = np.ones(num_bins) / num_bins

    return bin_centers, probs


def _build_oracle_and_good(
    bin_centers: np.ndarray,
    probs: np.ndarray,
    strike: float,
    is_call: bool,
    max_payoff: float,
) -> tuple[np.ndarray, list[int]]:
    """Build the state-preparation unitary and identify good states.

    The oracle A acts on n_price + 1 qubits.  The first n_price qubits
    encode the price bin; the last qubit is the "payoff" flag set to |1>
    with amplitude proportional to the normalised payoff.

    The "good" states are those with the payoff flag qubit in |1>.

    Returns (oracle_matrix, good_indices).
    """
    n_bins = len(bin_centers)

    # Payoffs (non-negative)
    if is_call:
        payoffs = np.maximum(bin_centers - strike, 0.0)
    else:
        payoffs = np.maximum(strike - bin_centers, 0.0)

    if max_payoff <= 0:
        max_payoff = max(payoffs.max(), 1e-10)
    normalised_payoffs = np.clip(payoffs / max_payoff, 0.0, 1.0)

    # Full Hilbert space: n_bins price states x 2 payoff-flag states = 2*n_bins
    dim = 2 * n_bins
    oracle = np.zeros((dim, dim), dtype=complex)

    # Column 0 (|0...0>) is mapped to the loaded state
    # |psi> = sum_i sqrt(p_i) [ sqrt(1-f_i)|i,0> + sqrt(f_i)|i,1> ]
    col0 = np.zeros(dim, dtype=complex)
    for i in range(n_bins):
        sqrt_p = np.sqrt(max(probs[i], 0.0))
        f = normalised_payoffs[i]
        sqrt_f = np.sqrt(max(f, 0.0))
        sqrt_1mf = np.sqrt(max(1.0 - f, 0.0))
        col0[2 * i] = sqrt_p * sqrt_1mf  # |i, 0>
        col0[2 * i + 1] = sqrt_p * sqrt_f  # |i, 1>

    # Normalise (should already be ~1 but enforce)
    norm = np.linalg.norm(col0)
    if norm > 1e-15:
        col0 /= norm

    # Extend to a unitary via Gram-Schmidt
    oracle[:, 0] = col0
    for j in range(1, dim):
        ej = np.zeros(dim, dtype=complex)
        ej[j] = 1.0
        v = ej.copy()
        for prev in range(j):
            proj = np.dot(oracle[:, prev].conj(), v)
            v -= proj * oracle[:, prev]
        n_v = np.linalg.norm(v)
        if n_v > 1e-12:
            oracle[:, j] = v / n_v
        else:
            oracle[:, j] = ej

    # Good states: those with the payoff flag = |1>, i.e. indices 1, 3, 5, ...
    good_indices = list(range(1, dim, 2))

    return oracle, good_indices


# ============================================================
# Main pricer class
# ============================================================


class QuantumOptionPricer:
    """Option pricer using quantum amplitude estimation.

    Parameters
    ----------
    spot : float
        Current asset price S_0.
    strike : float
        Option strike price K.
    rate : float
        Risk-free interest rate r (annualized).
    volatility : float
        Asset volatility sigma (annualized).
    maturity : float
        Time to maturity T in years.
    option_type : OptionType
        Contract type (European call/put, Asian, Barrier).
    barrier : float or None
        Barrier level (only for barrier options).
    num_price_qubits : int
        Number of qubits encoding the price distribution (default 4,
        giving 16 price bins).  More qubits improve payoff resolution.
    qae_method : QAEMethod
        Which QAE variant to use.
    num_time_steps : int
        Time steps for path-dependent options (Asian, Barrier).
    num_paths : int
        MC paths for path-dependent payoff estimation.
    """

    def __init__(
        self,
        spot: float = 100.0,
        strike: float = 100.0,
        rate: float = 0.05,
        volatility: float = 0.2,
        maturity: float = 1.0,
        option_type: OptionType = OptionType.EUROPEAN_CALL,
        barrier: float | None = None,
        num_price_qubits: int = 4,
        qae_method: QAEMethod = QAEMethod.CANONICAL,
        num_time_steps: int = 12,
        num_paths: int = 10000,
        _compute_delta: bool = True,
    ) -> None:
        if spot <= 0:
            raise ValueError("spot must be positive")
        if strike <= 0:
            raise ValueError("strike must be positive")
        if volatility <= 0:
            raise ValueError("volatility must be positive")
        if maturity <= 0:
            raise ValueError("maturity must be positive")

        self.spot = spot
        self.strike = strike
        self.rate = rate
        self.volatility = volatility
        self.maturity = maturity
        self.option_type = option_type
        self.barrier = barrier
        self.num_price_qubits = num_price_qubits
        self.qae_method = qae_method
        self.num_time_steps = num_time_steps
        self.num_paths = num_paths
        self._compute_delta = _compute_delta

    def price(self) -> OptionPricingResult:
        """Price the option.

        Returns
        -------
        OptionPricingResult
        """
        if self.option_type in (OptionType.EUROPEAN_CALL, OptionType.EUROPEAN_PUT):
            return self._price_european()
        elif self.option_type in (OptionType.ASIAN_CALL, OptionType.ASIAN_PUT):
            return self._price_asian()
        elif self.option_type in (
            OptionType.BARRIER_UP_AND_OUT,
            OptionType.BARRIER_DOWN_AND_OUT,
        ):
            return self._price_barrier()
        else:
            raise ValueError(f"Unsupported option type: {self.option_type}")

    # ----------------------------------------------------------
    # European pricing (exact QAE on discretized distribution)
    # ----------------------------------------------------------

    def _price_european(self) -> OptionPricingResult:
        is_call = self.option_type == OptionType.EUROPEAN_CALL

        n_bins = 1 << self.num_price_qubits
        bin_centers, probs = _build_price_distribution(
            self.spot, self.rate, self.volatility, self.maturity, n_bins
        )

        # Compute payoffs for max-payoff scaling
        if is_call:
            payoffs = np.maximum(bin_centers - self.strike, 0.0)
        else:
            payoffs = np.maximum(self.strike - bin_centers, 0.0)
        max_payoff = max(payoffs.max(), 1e-10)

        oracle, good_indices = _build_oracle_and_good(
            bin_centers, probs, self.strike, is_call, max_payoff
        )

        # Run QAE
        ae_result = self._run_qae(oracle, good_indices)

        # The estimated amplitude is proportional to E[payoff/max_payoff]
        # Actual expected payoff = amplitude * max_payoff
        discount = np.exp(-self.rate * self.maturity)
        raw_price = ae_result.estimation * max_payoff * discount

        # Confidence interval
        ci_low = ae_result.confidence_interval[0] * max_payoff * discount
        ci_high = ae_result.confidence_interval[1] * max_payoff * discount

        # Analytical for comparison
        if is_call:
            analytical = black_scholes_call(
                self.spot, self.strike, self.rate, self.volatility, self.maturity
            )
        else:
            analytical = black_scholes_put(
                self.spot, self.strike, self.rate, self.volatility, self.maturity
            )

        # Delta via finite difference on the QAE (skip if called from delta bump)
        delta = 0.0
        if self._compute_delta:
            bump = self.spot * 0.01
            pricer_up = QuantumOptionPricer(
                spot=self.spot + bump,
                strike=self.strike,
                rate=self.rate,
                volatility=self.volatility,
                maturity=self.maturity,
                option_type=self.option_type,
                num_price_qubits=self.num_price_qubits,
                qae_method=self.qae_method,
                _compute_delta=False,
            )
            price_up = pricer_up.price().price
            delta = (price_up - raw_price) / bump

        return OptionPricingResult(
            price=raw_price,
            delta=delta,
            confidence_interval=(ci_low, ci_high),
            num_oracle_calls=ae_result.num_oracle_calls,
            method=f"QAE ({self.qae_method.name}) European "
            f"{'Call' if is_call else 'Put'}",
            analytical_price=analytical,
        )

    # ----------------------------------------------------------
    # Asian pricing (MC payoff + QAE enhancement)
    # ----------------------------------------------------------

    def _price_asian(self) -> OptionPricingResult:
        is_call = self.option_type == OptionType.ASIAN_CALL
        payoffs = self._mc_asian_payoffs(is_call)
        return self._mc_to_qae_result(payoffs, is_call, "Asian")

    def _mc_asian_payoffs(self, is_call: bool) -> np.ndarray:
        """Generate Asian option payoffs via MC paths."""
        rng = np.random.RandomState(42)
        dt = self.maturity / self.num_time_steps
        drift = (self.rate - 0.5 * self.volatility ** 2) * dt
        vol = self.volatility * np.sqrt(dt)

        payoffs = np.zeros(self.num_paths)
        for p in range(self.num_paths):
            s = self.spot
            total = 0.0
            for _ in range(self.num_time_steps):
                z = rng.randn()
                s *= np.exp(drift + vol * z)
                total += s
            avg = total / self.num_time_steps
            if is_call:
                payoffs[p] = max(avg - self.strike, 0.0)
            else:
                payoffs[p] = max(self.strike - avg, 0.0)
        return payoffs

    # ----------------------------------------------------------
    # Barrier pricing (MC payoff + QAE enhancement)
    # ----------------------------------------------------------

    def _price_barrier(self) -> OptionPricingResult:
        if self.barrier is None:
            raise ValueError("barrier must be set for barrier options")
        is_up = self.option_type == OptionType.BARRIER_UP_AND_OUT
        payoffs = self._mc_barrier_payoffs(is_up)
        return self._mc_to_qae_result(payoffs, True, "Barrier")

    def _mc_barrier_payoffs(self, is_up: bool) -> np.ndarray:
        """Generate barrier option payoffs via MC paths."""
        rng = np.random.RandomState(99)
        dt = self.maturity / self.num_time_steps
        drift = (self.rate - 0.5 * self.volatility ** 2) * dt
        vol = self.volatility * np.sqrt(dt)
        barrier = self.barrier

        payoffs = np.zeros(self.num_paths)
        for p in range(self.num_paths):
            s = self.spot
            knocked_out = False
            for _ in range(self.num_time_steps):
                z = rng.randn()
                s *= np.exp(drift + vol * z)
                if (is_up and s >= barrier) or (not is_up and s <= barrier):
                    knocked_out = True
                    break
            if not knocked_out:
                payoffs[p] = max(s - self.strike, 0.0)
        return payoffs

    # ----------------------------------------------------------
    # MC payoff -> QAE result
    # ----------------------------------------------------------

    def _mc_to_qae_result(
        self, payoffs: np.ndarray, is_call: bool, label: str
    ) -> OptionPricingResult:
        """Convert MC payoffs into a QAE-enhanced price estimate.

        We discretize the empirical payoff distribution into bins, build an
        oracle, and run QAE to estimate the mean payoff.
        """
        discount = np.exp(-self.rate * self.maturity)
        max_payoff = max(payoffs.max(), 1e-10)

        n_bins = 1 << self.num_price_qubits
        # Build histogram of payoffs
        hist, bin_edges = np.histogram(payoffs, bins=n_bins, density=False)
        probs = hist.astype(float)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(n_bins) / n_bins
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # For oracle construction, treat bin_centers as "prices" and
        # payoff(center) as the payoff.  For Asian/Barrier the payoffs are
        # already the payoff values, so we treat them as the "price - strike" proxy.
        # We directly use the bin centers as payoff magnitudes.
        normalised = np.clip(bin_centers / max_payoff, 0.0, 1.0)

        dim = 2 * n_bins
        oracle = np.zeros((dim, dim), dtype=complex)
        col0 = np.zeros(dim, dtype=complex)
        for i in range(n_bins):
            sqrt_p = np.sqrt(max(probs[i], 0.0))
            f = normalised[i]
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
            n_v = np.linalg.norm(v)
            if n_v > 1e-12:
                oracle[:, j] = v / n_v
            else:
                oracle[:, j] = ej

        good_indices = list(range(1, dim, 2))
        ae_result = self._run_qae(oracle, good_indices)

        price_qae = ae_result.estimation * max_payoff * discount
        ci_low = ae_result.confidence_interval[0] * max_payoff * discount
        ci_high = ae_result.confidence_interval[1] * max_payoff * discount

        # Classical MC price for reference
        mc_price = discount * payoffs.mean()
        mc_se = discount * payoffs.std() / np.sqrt(len(payoffs))

        return OptionPricingResult(
            price=price_qae,
            delta=0.0,
            confidence_interval=(ci_low, ci_high),
            num_oracle_calls=ae_result.num_oracle_calls,
            method=f"QAE ({self.qae_method.name}) {label}",
            analytical_price=mc_price,
        )

    # ----------------------------------------------------------
    # QAE dispatch
    # ----------------------------------------------------------

    def _run_qae(
        self, oracle: np.ndarray, good_indices: list[int]
    ) -> AEResult:
        """Run the selected QAE method."""
        if self.qae_method == QAEMethod.CANONICAL:
            qae = CanonicalQAE(num_eval_qubits=6)
            return qae.estimate(oracle, good_indices)
        elif self.qae_method == QAEMethod.ITERATIVE:
            qae = IterativeQAE(epsilon=0.01, alpha=0.05)
            return qae.estimate(oracle, good_indices)
        else:
            qae = MaxLikelihoodQAE(
                evaluation_schedule=[0, 1, 2, 4, 8], num_shots=200
            )
            return qae.estimate(oracle, good_indices)


# ============================================================
# Convenience functions
# ============================================================


def price_european_call(
    spot: float = 100.0,
    strike: float = 100.0,
    rate: float = 0.05,
    volatility: float = 0.2,
    maturity: float = 1.0,
    num_price_qubits: int = 4,
    qae_method: QAEMethod = QAEMethod.CANONICAL,
) -> OptionPricingResult:
    """Quick European call pricing."""
    pricer = QuantumOptionPricer(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        option_type=OptionType.EUROPEAN_CALL,
        num_price_qubits=num_price_qubits,
        qae_method=qae_method,
    )
    return pricer.price()


def price_european_put(
    spot: float = 100.0,
    strike: float = 100.0,
    rate: float = 0.05,
    volatility: float = 0.2,
    maturity: float = 1.0,
    num_price_qubits: int = 4,
    qae_method: QAEMethod = QAEMethod.CANONICAL,
) -> OptionPricingResult:
    """Quick European put pricing."""
    pricer = QuantumOptionPricer(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        option_type=OptionType.EUROPEAN_PUT,
        num_price_qubits=num_price_qubits,
        qae_method=qae_method,
    )
    return pricer.price()


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    print("=== Option Pricing self-test ===")

    # European call
    result = price_european_call(spot=100, strike=100, rate=0.05,
                                 volatility=0.2, maturity=1.0)
    bs = black_scholes_call(100, 100, 0.05, 0.2, 1.0)
    print(f"European Call QAE: {result.price:.4f}  BS: {bs:.4f}")

    # European put
    result_put = price_european_put(spot=100, strike=100, rate=0.05,
                                    volatility=0.2, maturity=1.0)
    bs_put = black_scholes_put(100, 100, 0.05, 0.2, 1.0)
    print(f"European Put  QAE: {result_put.price:.4f}  BS: {bs_put:.4f}")

    # Put-call parity check:  C - P = S - K*exp(-rT)
    parity_lhs = result.price - result_put.price
    parity_rhs = 100 - 100 * np.exp(-0.05 * 1.0)
    print(f"Put-Call parity:  LHS={parity_lhs:.4f}  RHS={parity_rhs:.4f}")

    print("Self-test complete.")
