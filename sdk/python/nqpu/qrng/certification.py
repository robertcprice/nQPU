"""Device-independent randomness certification via Bell inequality violations.

Provides tools to certify that random bits are genuinely quantum-origin
by measuring CHSH Bell inequality violations and computing lower bounds
on the extractable min-entropy.

Key insight: if the CHSH S-value exceeds the classical bound of 2, no
local hidden variable model can explain the correlations. This certifies
that the randomness is fundamentally unpredictable, even to an adversary
with full knowledge of the device internals.

Classes provided:
  - CHSHCertifier: Single-round CHSH certification with confidence intervals
  - EntropyAccumulation: Multi-round entropy accumulation theorem
  - RandomnessExpansion: Certify more random bits than seed bits consumed

References:
  - Clauser, Horne, Shimony & Holt, Phys. Rev. Lett. 23, 880 (1969)
  - Pironio et al., Nature 464, 1021 (2010) [DI randomness expansion]
  - Dupuis, Fawzi & Renner, Commun. Math. Phys. 379, 867 (2020) [EAT]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# CHSH Certifier
# ---------------------------------------------------------------------------


@dataclass
class CHSHCertifier:
    """Certify randomness via CHSH Bell inequality violation.

    The CHSH inequality states that for any local hidden variable model:
        S = |E(a0,b0) - E(a0,b1) + E(a1,b0) + E(a1,b1)| <= 2

    Quantum mechanics predicts a maximum violation of S = 2*sqrt(2) ~ 2.828
    (Tsirelson bound). For S > 2, we can certify min-entropy per round:

        H_min >= 1 - log2(1 + sqrt(2 - S^2/4))

    Parameters
    ----------
    n_rounds : int
        Number of measurement rounds per CHSH test.
    confidence_level : float
        Desired confidence level for the entropy bound (e.g. 0.99).
    angles : tuple of 4 floats, optional
        Custom measurement angles (a0, a1, b0, b1) in radians.
        Default: optimal CHSH angles (0, pi/2, pi/4, -pi/4).
    seed : int or None
        RNG seed for reproducible Bell tests.
    """

    n_rounds: int = 10000
    confidence_level: float = 0.99
    angles: Optional[tuple[float, float, float, float]] = None
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_rounds < 10:
            raise ValueError("n_rounds must be >= 10")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.angles is None:
            # Optimal CHSH angles for E(a,b) = cos(a - b)
            # a0=0, a1=pi/2, b0=pi/4, b1=3pi/4 gives S = 2*sqrt(2)
            self.angles = (0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4)
        self._rng = np.random.default_rng(self.seed)

    def run_bell_test(self) -> CHSHResult:
        """Execute a CHSH Bell test and return certification result.

        Simulates a Bell-state source and measures CHSH correlations
        using the configured angles. Returns a certificate containing
        the estimated S-value, min-entropy bound, and confidence interval.

        Returns
        -------
        CHSHResult
            Full certification result with entropy bounds.
        """
        a0, a1, b0, b1 = self.angles

        # Simulate Bell state |Phi+> = (|00> + |11>) / sqrt(2)
        # For each setting pair, measure n_rounds times
        correlations = {}
        for label, theta_a, theta_b in [
            ("e_00", a0, b0),
            ("e_01", a0, b1),
            ("e_10", a1, b0),
            ("e_11", a1, b1),
        ]:
            correlations[label] = self._measure_correlation(
                theta_a, theta_b, self.n_rounds
            )

        # CHSH S-value
        s_value = abs(
            correlations["e_00"]
            - correlations["e_01"]
            + correlations["e_10"]
            + correlations["e_11"]
        )

        # Standard error on S (propagated from individual correlations)
        # Each E(a,b) has stderr ~ 1/sqrt(n) for +/-1 variables
        se_individual = 1.0 / math.sqrt(self.n_rounds)
        se_s = 2.0 * se_individual  # sum of 4 terms, each with same SE

        # Confidence interval
        z = _norm_ppf((1 + self.confidence_level) / 2)
        s_lower = s_value - z * se_s
        s_upper = s_value + z * se_s

        # Min-entropy bound from the lower confidence bound
        h_min = self._min_entropy_from_s(s_lower)
        h_min_point = self._min_entropy_from_s(s_value)

        return CHSHResult(
            s_value=s_value,
            s_lower=s_lower,
            s_upper=s_upper,
            correlations=correlations,
            min_entropy_per_round=h_min,
            min_entropy_point_estimate=h_min_point,
            is_quantum=s_value > 2.0,
            is_certified=s_lower > 2.0,
            n_rounds=self.n_rounds,
            confidence_level=self.confidence_level,
            tsirelson_fraction=s_value / (2.0 * math.sqrt(2)),
        )

    @staticmethod
    def _min_entropy_from_s(s: float) -> float:
        """Compute min-entropy bound from CHSH S-value.

        H_min >= 1 - log2(1 + sqrt(2 - S^2/4))

        For S <= 2 (classical), returns 0.
        For S = 2*sqrt(2) (Tsirelson), returns 1.
        """
        if s <= 2.0:
            return 0.0
        s = min(s, 2.0 * math.sqrt(2))  # Cap at Tsirelson bound

        inner = 2.0 - s * s / 4.0
        if inner < 0:
            inner = 0.0
        return 1.0 - math.log2(1.0 + math.sqrt(inner))

    def _measure_correlation(
        self,
        theta_a: float,
        theta_b: float,
        n: int,
    ) -> float:
        """Simulate measuring a Bell pair with given angles and return E(a,b).

        For |Phi+>, the quantum prediction is:
            E(theta_a, theta_b) = cos(theta_a - theta_b)

        We simulate this with shot noise from finite statistics.
        """
        # Quantum prediction for correlation
        delta = theta_a - theta_b
        p_same = math.cos(delta / 2) ** 2
        # p_same = P(both 0) + P(both 1) = cos^2(delta/2)
        # E = P(same) - P(different) = 2*cos^2(delta/2) - 1 = cos(delta)

        # Simulate measurement outcomes
        same = self._rng.binomial(n, p_same)
        different = n - same
        e_val = (same - different) / n
        return float(e_val)


@dataclass
class CHSHResult:
    """Result of a CHSH Bell test certification.

    Attributes
    ----------
    s_value : float
        Estimated CHSH S-value.
    s_lower : float
        Lower bound of confidence interval.
    s_upper : float
        Upper bound of confidence interval.
    correlations : dict
        Individual E(a,b) correlation estimates.
    min_entropy_per_round : float
        Certified min-entropy per round (from lower CI bound).
    min_entropy_point_estimate : float
        Min-entropy from point estimate of S.
    is_quantum : bool
        Whether S > 2 (point estimate violates Bell inequality).
    is_certified : bool
        Whether the lower CI bound exceeds 2 (certified violation).
    n_rounds : int
        Number of measurement rounds.
    confidence_level : float
        Statistical confidence level.
    tsirelson_fraction : float
        S / (2*sqrt(2)), indicating how close to maximum violation.
    """

    s_value: float
    s_lower: float
    s_upper: float
    correlations: dict
    min_entropy_per_round: float
    min_entropy_point_estimate: float
    is_quantum: bool
    is_certified: bool
    n_rounds: int
    confidence_level: float
    tsirelson_fraction: float

    def summary(self) -> str:
        """Human-readable summary of the certification result."""
        lines = [
            f"CHSH Bell Test Certification (n={self.n_rounds})",
            "=" * 50,
            f"  S-value:    {self.s_value:.4f} "
            f"[{self.s_lower:.4f}, {self.s_upper:.4f}] "
            f"({self.confidence_level:.0%} CI)",
            f"  Classical:  2.0000",
            f"  Tsirelson:  {2*math.sqrt(2):.4f}",
            f"  Fraction:   {self.tsirelson_fraction:.2%} of Tsirelson",
            "",
            f"  Quantum:    {'YES' if self.is_quantum else 'NO'} "
            f"(S {'>' if self.is_quantum else '<='} 2)",
            f"  Certified:  {'YES' if self.is_certified else 'NO'} "
            f"(CI lower {'>' if self.is_certified else '<='} 2)",
            "",
            f"  H_min/round: {self.min_entropy_per_round:.4f} bits "
            f"(certified)",
            f"  H_min/round: {self.min_entropy_point_estimate:.4f} bits "
            f"(point est.)",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entropy Accumulation
# ---------------------------------------------------------------------------


@dataclass
class EntropyAccumulation:
    """Entropy accumulation theorem (EAT) for sequential CHSH rounds.

    The EAT provides a tight lower bound on the smooth min-entropy
    accumulated over n sequential rounds of a CHSH game, accounting
    for memory effects and finite-size corrections.

    For n rounds with average CHSH winning probability omega:
        H_min^eps >= n * h(omega) - sqrt(n) * O(log(1/eps))

    where h(omega) is the single-round entropy rate function.

    Parameters
    ----------
    n_rounds : int
        Total number of sequential rounds.
    smoothness : float
        Smoothness parameter epsilon for smooth min-entropy.
    test_fraction : float
        Fraction of rounds used for Bell testing (rest used for generation).
    seed : int or None
        RNG seed.
    """

    n_rounds: int = 100000
    smoothness: float = 1e-6
    test_fraction: float = 0.1
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_rounds < 100:
            raise ValueError("n_rounds must be >= 100")
        if not 0 < self.smoothness < 1:
            raise ValueError("smoothness must be in (0, 1)")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in (0, 1)")
        self._rng = np.random.default_rng(self.seed)

    def accumulate(self) -> EntropyAccumulationResult:
        """Run the entropy accumulation protocol.

        Simulates n rounds of Bell-state measurements, designating a
        fraction as test rounds (CHSH check) and the rest as generation
        rounds (randomness output).

        Returns
        -------
        EntropyAccumulationResult
        """
        n_test = int(self.n_rounds * self.test_fraction)
        n_gen = self.n_rounds - n_test

        # Simulate test rounds: measure CHSH winning probability
        # For optimal quantum strategy, winning probability = cos^2(pi/8)
        p_win_quantum = math.cos(math.pi / 8) ** 2  # ~0.8536

        # Simulate with finite statistics
        n_wins = self._rng.binomial(n_test, p_win_quantum)
        omega_hat = n_wins / n_test

        # Single-round von Neumann entropy rate
        # h(omega) = 1 - H_bin(1/2 + sqrt(2*omega - 1)/2)  for omega > 3/4
        h_rate = self._entropy_rate(omega_hat)

        # Finite-size correction (simplified EAT bound)
        # Second-order correction: sqrt(n) * V * z_eps
        # where V is the entropy variance and z_eps depends on smoothness
        z_eps = _norm_ppf(1 - self.smoothness)
        v_bound = 2.0  # Conservative bound on single-round entropy variance
        correction = math.sqrt(n_gen) * v_bound * z_eps

        # Total smooth min-entropy
        total_entropy = max(0.0, n_gen * h_rate - correction)
        # Additional log correction
        log_correction = 2.0 * math.log2(1.0 / self.smoothness)
        total_entropy = max(0.0, total_entropy - log_correction)

        return EntropyAccumulationResult(
            n_rounds=self.n_rounds,
            n_test_rounds=n_test,
            n_generation_rounds=n_gen,
            chsh_winning_prob=omega_hat,
            entropy_rate_per_round=h_rate,
            total_smooth_min_entropy=total_entropy,
            effective_entropy_per_round=total_entropy / n_gen if n_gen > 0 else 0.0,
            smoothness=self.smoothness,
            finite_size_correction=correction + log_correction,
            is_positive=total_entropy > 0,
        )

    @staticmethod
    def _entropy_rate(omega: float) -> float:
        """Compute the single-round entropy rate from CHSH winning probability.

        For the optimal quantum strategy:
            h(omega) = 1 - H_bin(1/2 + sqrt(2*omega - 1)/2)

        where H_bin is the binary entropy function. This applies for
        omega > 3/4 (the classical threshold for CHSH winning probability).
        """
        if omega <= 0.75:
            return 0.0
        omega = min(omega, math.cos(math.pi / 8) ** 2)

        inner = 2.0 * omega - 1.0
        if inner <= 0:
            return 0.0
        p = 0.5 + math.sqrt(inner) / 2.0
        p = max(1e-15, min(1 - 1e-15, p))
        h_bin = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        return max(0.0, 1.0 - h_bin)


@dataclass
class EntropyAccumulationResult:
    """Result of entropy accumulation protocol.

    Attributes
    ----------
    n_rounds : int
        Total rounds executed.
    n_test_rounds : int
        Rounds used for CHSH testing.
    n_generation_rounds : int
        Rounds used for randomness generation.
    chsh_winning_prob : float
        Estimated CHSH winning probability.
    entropy_rate_per_round : float
        Single-round entropy rate (asymptotic).
    total_smooth_min_entropy : float
        Total certified smooth min-entropy (bits).
    effective_entropy_per_round : float
        Total entropy / generation rounds.
    smoothness : float
        Smoothness parameter used.
    finite_size_correction : float
        Total finite-size penalty (bits).
    is_positive : bool
        Whether any certified entropy was accumulated.
    """

    n_rounds: int
    n_test_rounds: int
    n_generation_rounds: int
    chsh_winning_prob: float
    entropy_rate_per_round: float
    total_smooth_min_entropy: float
    effective_entropy_per_round: float
    smoothness: float
    finite_size_correction: float
    is_positive: bool

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Entropy Accumulation Result (n={self.n_rounds})",
            "=" * 50,
            f"  Test rounds:     {self.n_test_rounds}",
            f"  Generation:      {self.n_generation_rounds}",
            f"  CHSH win prob:   {self.chsh_winning_prob:.4f} "
            f"(classical: 0.75)",
            f"  Rate/round:      {self.entropy_rate_per_round:.4f} bits",
            f"  Total H_min:     {self.total_smooth_min_entropy:.1f} bits",
            f"  Effective/round: {self.effective_entropy_per_round:.4f} bits",
            f"  Correction:      {self.finite_size_correction:.1f} bits",
            f"  Positive:        {'YES' if self.is_positive else 'NO'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Randomness Expansion
# ---------------------------------------------------------------------------


@dataclass
class RandomnessExpansion:
    """Certify that more random bits are produced than seed bits consumed.

    The protocol uses a short random seed to choose measurement bases in
    a CHSH game, then certifies the output randomness via Bell violation.
    Positive expansion occurs when the certified output entropy exceeds
    the seed length.

    Protocol:
      1. Use seed bits to randomly select measurement bases
      2. Perform Bell-state measurements with chosen bases
      3. Estimate CHSH S-value from test rounds
      4. Certify output entropy via entropy accumulation
      5. Expansion rate = (output entropy - seed length) / n_rounds

    Parameters
    ----------
    n_rounds : int
        Number of measurement rounds.
    seed_length : int
        Number of seed bits consumed.
    confidence_level : float
        Confidence for the entropy bound.
    seed : int or None
        RNG seed for simulation.
    """

    n_rounds: int = 100000
    seed_length: int = 1000
    confidence_level: float = 0.99
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_rounds < 100:
            raise ValueError("n_rounds must be >= 100")
        if self.seed_length < 1:
            raise ValueError("seed_length must be >= 1")
        self._rng = np.random.default_rng(self.seed)

    def run(self) -> ExpansionResult:
        """Execute the randomness expansion protocol.

        Returns
        -------
        ExpansionResult
        """
        # Use seed bits to select test/generation partition
        # Approximately 2 bits per round for basis choice
        test_fraction = min(0.5, self.seed_length / (2 * self.n_rounds))
        test_fraction = max(0.01, test_fraction)

        n_test = max(10, int(self.n_rounds * test_fraction))
        n_gen = self.n_rounds - n_test

        # Simulate Bell test with quantum strategy
        p_win = math.cos(math.pi / 8) ** 2
        n_wins = self._rng.binomial(n_test, p_win)
        omega = n_wins / n_test

        # Compute CHSH S-value from winning probability
        # S = 4*omega - 2 (approximate)
        # More precisely: omega = 1/2 + S/(4*sqrt(2))
        # So S = 4*sqrt(2) * (omega - 1/2)
        s_est = 4 * math.sqrt(2) * (omega - 0.5)
        s_est = max(0.0, min(2 * math.sqrt(2), s_est))

        # Entropy rate (use static method directly)
        h_rate = EntropyAccumulation._entropy_rate(omega)

        # Certified output entropy (with finite-size correction)
        smoothness = 1 - self.confidence_level
        z = _norm_ppf(1 - smoothness)
        correction = math.sqrt(n_gen) * 2.0 * z
        log_correction = 2.0 * math.log2(1.0 / max(1e-15, smoothness))
        output_entropy = max(0.0, n_gen * h_rate - correction - log_correction)

        # Expansion
        net_expansion = output_entropy - self.seed_length
        expansion_rate = net_expansion / self.n_rounds if self.n_rounds > 0 else 0.0

        return ExpansionResult(
            n_rounds=self.n_rounds,
            seed_length=self.seed_length,
            n_test_rounds=n_test,
            n_generation_rounds=n_gen,
            chsh_s_value=s_est,
            chsh_winning_prob=omega,
            entropy_rate=h_rate,
            output_entropy=output_entropy,
            net_expansion=net_expansion,
            expansion_rate=expansion_rate,
            has_expansion=net_expansion > 0,
            expansion_factor=output_entropy / self.seed_length
            if self.seed_length > 0
            else 0.0,
        )


@dataclass
class ExpansionResult:
    """Result of randomness expansion protocol.

    Attributes
    ----------
    n_rounds : int
        Total measurement rounds.
    seed_length : int
        Seed bits consumed.
    n_test_rounds : int
        Rounds used for CHSH testing.
    n_generation_rounds : int
        Rounds used for generation.
    chsh_s_value : float
        Estimated CHSH S-value.
    chsh_winning_prob : float
        Estimated CHSH winning probability.
    entropy_rate : float
        Single-round entropy rate.
    output_entropy : float
        Total certified output entropy (bits).
    net_expansion : float
        Output entropy minus seed length (bits).
    expansion_rate : float
        Net expansion per round (bits/round).
    has_expansion : bool
        Whether net expansion is positive.
    expansion_factor : float
        Output entropy / seed length.
    """

    n_rounds: int
    seed_length: int
    n_test_rounds: int
    n_generation_rounds: int
    chsh_s_value: float
    chsh_winning_prob: float
    entropy_rate: float
    output_entropy: float
    net_expansion: float
    expansion_rate: float
    has_expansion: bool
    expansion_factor: float

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Randomness Expansion Result (n={self.n_rounds})",
            "=" * 50,
            f"  Seed consumed:   {self.seed_length} bits",
            f"  Output certified:{self.output_entropy:.1f} bits",
            f"  Net expansion:   {self.net_expansion:.1f} bits",
            f"  Factor:          {self.expansion_factor:.2f}x",
            f"  Rate:            {self.expansion_rate:.4f} bits/round",
            f"  CHSH S-value:    {self.chsh_s_value:.4f}",
            f"  Win probability: {self.chsh_winning_prob:.4f}",
            f"  Expansion:       {'YES' if self.has_expansion else 'NO'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (percent point function).

    Uses the rational approximation from Abramowitz and Stegun.
    Accurate to about 4.5e-4 for p in (0, 1).
    """
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        return -_rational_approx(math.sqrt(-2.0 * math.log(p)))
    else:
        return _rational_approx(math.sqrt(-2.0 * math.log(1.0 - p)))


def _rational_approx(t: float) -> float:
    """Rational approximation for the inverse normal."""
    # Coefficients for the Abramowitz & Stegun approximation
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (
        1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    )
