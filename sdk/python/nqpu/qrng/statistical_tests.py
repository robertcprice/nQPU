"""NIST SP 800-22 statistical test suite for randomness validation.

Implements a subset of the NIST Special Publication 800-22 Rev. 1a tests
for evaluating the quality of random and pseudorandom number generators.
Each test returns a p-value; the null hypothesis is that the sequence is
random. If p-value < alpha (default 0.01), the sequence is considered
non-random for that test.

Tests implemented:
  1. Frequency (monobit) test
  2. Block frequency test
  3. Runs test
  4. Longest run of ones test
  5. Serial test (2-bit and 3-bit overlapping patterns)
  6. Approximate entropy test
  7. Cumulative sums test
  8. DFT spectral test

References:
  - NIST SP 800-22 Rev. 1a, "A Statistical Test Suite for Random and
    Pseudorandom Number Generators for Cryptographic Applications" (2010)
  - Rukhin et al., NIST Special Publication 800-22
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# We need erfc and gammaincc. scipy is available in the nQPU environment.
# Use a fallback pure-numpy approach if scipy is not present.
try:
    from scipy.special import erfc, gammaincc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

    def erfc(x):
        """Complementary error function (pure numpy fallback)."""
        # Use numpy's built-in
        return _erfc_approx(np.asarray(x, dtype=np.float64))

    def _erfc_approx(x):
        """Abramowitz and Stegun approximation 7.1.26 for erfc."""
        a = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * a)
        poly = t * (0.254829592 + t * (-0.284496736 + t * (
            1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        result = poly * np.exp(-a * a)
        # erfc(-x) = 2 - erfc(x)
        return np.where(x >= 0, result, 2.0 - result)

    def gammaincc(a, x):
        """Upper incomplete gamma function ratio (series expansion fallback).

        This is a simplified version sufficient for the NIST tests.
        For large a or x, the series may converge slowly.
        """
        a = float(a)
        x = float(x)
        if x < 0 or a <= 0:
            return 1.0
        if x == 0:
            return 1.0

        # Use the regularised lower incomplete gamma via series expansion
        # and compute the complement: Q(a,x) = 1 - P(a,x)
        if x < a + 1:
            # Series for P(a,x) = gamma(a,x) / Gamma(a)
            return 1.0 - _gamma_series(a, x)
        else:
            # Continued fraction for Q(a,x)
            return _gamma_cf(a, x)

    def _gamma_series(a, x, max_iter=200, eps=1e-12):
        """Lower regularised incomplete gamma by series."""
        if x == 0:
            return 0.0
        ap = a
        s = 1.0 / a
        delta = s
        for _ in range(max_iter):
            ap += 1.0
            delta *= x / ap
            s += delta
            if abs(delta) < abs(s) * eps:
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))

    def _gamma_cf(a, x, max_iter=200, eps=1e-12):
        """Upper regularised incomplete gamma by continued fraction."""
        b = x + 1.0 - a
        c = 1e30
        d = 1.0 / b if b != 0 else 1e30
        h = d
        for i in range(1, max_iter + 1):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


# ---------------------------------------------------------------------------
# Test result container
# ---------------------------------------------------------------------------


@dataclass
class StatisticalTestResult:
    """Result of a single NIST statistical test.

    Attributes
    ----------
    name : str
        Human-readable test name.
    p_value : float
        Computed p-value (0 to 1).
    passed : bool
        Whether p_value >= alpha.
    statistic : float
        Test statistic value.
    alpha : float
        Significance level used.
    details : dict
        Additional test-specific information.
    """

    name: str
    p_value: float
    passed: bool
    statistic: float
    alpha: float = 0.01
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Test 1: Frequency (Monobit) Test
# ---------------------------------------------------------------------------


def frequency_test(bits: np.ndarray, alpha: float = 0.01) -> StatisticalTestResult:
    """NIST frequency (monobit) test.

    Tests whether the proportion of 0s and 1s in the sequence is
    approximately equal, as expected for a random sequence.

    The test statistic S_obs = |sum(X_i)| / sqrt(n) where X_i = 2*bit_i - 1.
    Under H0, S_obs follows a half-normal distribution and
    p-value = erfc(S_obs / sqrt(2)).

    Parameters
    ----------
    bits : numpy.ndarray
        Binary sequence (uint8, each element 0 or 1).
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits)
    if n < 100:
        raise ValueError("Frequency test requires n >= 100")

    # Convert 0/1 to -1/+1
    x = 2.0 * bits.astype(np.float64) - 1.0
    s_n = np.abs(np.sum(x))
    s_obs = s_n / math.sqrt(n)
    p_value = float(erfc(s_obs / math.sqrt(2)))

    return StatisticalTestResult(
        name="Frequency (Monobit)",
        p_value=p_value,
        passed=p_value >= alpha,
        statistic=s_obs,
        alpha=alpha,
        details={"n": n, "sum": float(np.sum(x))},
    )


# ---------------------------------------------------------------------------
# Test 2: Block Frequency Test
# ---------------------------------------------------------------------------


def block_frequency_test(
    bits: np.ndarray,
    block_size: int = 128,
    alpha: float = 0.01,
) -> StatisticalTestResult:
    """NIST block frequency test.

    Divides the sequence into M-bit blocks and tests whether the proportion
    of ones in each block is approximately M/2.

    The chi-squared statistic is:
        chi2 = 4*M * sum_i (pi_i - 0.5)^2

    where pi_i is the proportion of ones in block i. Under H0,
    chi2 ~ chi-squared(N) where N is the number of blocks.

    Parameters
    ----------
    bits : numpy.ndarray
        Binary sequence.
    block_size : int
        Block size M (recommended >= 20, must satisfy n/M >= 100).
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits)
    if block_size < 20:
        raise ValueError("block_size should be >= 20")

    n_blocks = n // block_size
    if n_blocks < 1:
        raise ValueError(
            f"Sequence too short ({n}) for block_size={block_size}"
        )

    # Compute proportion of ones in each block
    blocks = bits[: n_blocks * block_size].reshape(n_blocks, block_size)
    proportions = blocks.astype(np.float64).mean(axis=1)

    # Chi-squared statistic
    chi2 = 4.0 * block_size * np.sum((proportions - 0.5) ** 2)
    p_value = float(gammaincc(n_blocks / 2.0, chi2 / 2.0))

    return StatisticalTestResult(
        name="Block Frequency",
        p_value=p_value,
        passed=p_value >= alpha,
        statistic=chi2,
        alpha=alpha,
        details={
            "block_size": block_size,
            "n_blocks": n_blocks,
            "mean_proportion": float(np.mean(proportions)),
        },
    )


# ---------------------------------------------------------------------------
# Test 3: Runs Test
# ---------------------------------------------------------------------------


def runs_test(bits: np.ndarray, alpha: float = 0.01) -> StatisticalTestResult:
    """NIST runs test.

    A run is an uninterrupted sequence of identical bits. The total number
    of runs is tested against the expected number for a random sequence.

    Pre-requisite: the sequence must pass the frequency test (|pi - 0.5|
    should be < 2/sqrt(n)). If it doesn't, the test returns p=0.

    Parameters
    ----------
    bits : numpy.ndarray
        Binary sequence.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits)
    if n < 100:
        raise ValueError("Runs test requires n >= 100")

    # Pre-test: proportion of ones
    pi = np.mean(bits.astype(np.float64))
    tau = 2.0 / math.sqrt(n)

    if abs(pi - 0.5) >= tau:
        # Frequency test prerequisite failed
        return StatisticalTestResult(
            name="Runs",
            p_value=0.0,
            passed=False,
            statistic=0.0,
            alpha=alpha,
            details={
                "pi": float(pi),
                "tau": tau,
                "prerequisite_failed": True,
            },
        )

    # Count runs: transitions between different bits
    v_obs = 1 + np.sum(bits[:-1] != bits[1:])
    v_obs = float(v_obs)

    # Expected runs and p-value
    p_value = float(
        erfc(
            abs(v_obs - 2.0 * n * pi * (1.0 - pi))
            / (2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi))
        )
    )

    return StatisticalTestResult(
        name="Runs",
        p_value=p_value,
        passed=p_value >= alpha,
        statistic=v_obs,
        alpha=alpha,
        details={
            "n": n,
            "pi": float(pi),
            "expected_runs": 2.0 * n * pi * (1.0 - pi),
        },
    )


# ---------------------------------------------------------------------------
# Test 4: Longest Run of Ones Test
# ---------------------------------------------------------------------------


def longest_run_test(bits: np.ndarray, alpha: float = 0.01) -> StatisticalTestResult:
    """NIST longest run of ones in a block test.

    Determines whether the longest run of ones within M-bit blocks is
    consistent with a random sequence. Block size M and frequency classes
    depend on the sequence length.

    Parameters
    ----------
    bits : numpy.ndarray
        Binary sequence.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits)

    # Select parameters based on sequence length
    if n < 128:
        raise ValueError("Longest run test requires n >= 128")
    elif n < 6272:
        m = 8
        k = 3
        # Classes: <=1, 2, 3, >=4
        boundaries = [1, 2, 3]
        pi_values = np.array([0.2148, 0.3672, 0.2305, 0.1875])
    elif n < 750000:
        m = 128
        k = 5
        # Classes: <=4, 5, 6, 7, 8, >=9
        boundaries = [4, 5, 6, 7, 8]
        pi_values = np.array([0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124])
    else:
        m = 10000
        k = 6
        # Classes: <=10, 11, 12, 13, 14, 15, >=16
        boundaries = [10, 11, 12, 13, 14, 15]
        pi_values = np.array([
            0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727
        ])

    n_blocks = n // m
    if n_blocks < 1:
        raise ValueError(f"Sequence too short for block size {m}")

    # Find longest run of ones in each block
    longest_runs = np.zeros(n_blocks, dtype=np.int64)
    for i in range(n_blocks):
        block = bits[i * m : (i + 1) * m]
        longest_runs[i] = _longest_run_in_block(block)

    # Classify into frequency buckets
    n_classes = len(pi_values)
    observed = np.zeros(n_classes, dtype=np.float64)

    for run_len in longest_runs:
        # Determine which class this falls into
        cls = n_classes - 1  # default: last class
        for j, bound in enumerate(boundaries):
            if run_len <= bound:
                cls = j
                break
        observed[cls] += 1

    # Chi-squared statistic
    expected = n_blocks * pi_values
    # Avoid division by zero
    mask = expected > 0
    chi2 = np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
    p_value = float(gammaincc(k / 2.0, chi2 / 2.0))

    return StatisticalTestResult(
        name="Longest Run of Ones",
        p_value=p_value,
        passed=p_value >= alpha,
        statistic=float(chi2),
        alpha=alpha,
        details={
            "block_size": m,
            "n_blocks": n_blocks,
            "n_classes": n_classes,
            "observed": observed.tolist(),
            "expected": expected.tolist(),
        },
    )


def _longest_run_in_block(block: np.ndarray) -> int:
    """Find the length of the longest run of ones in a block."""
    max_run = 0
    current_run = 0
    for bit in block:
        if bit == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


# ---------------------------------------------------------------------------
# Test 5: Serial Test
# ---------------------------------------------------------------------------


def serial_test(
    bits: np.ndarray,
    block_size: int = 3,
    alpha: float = 0.01,
) -> StatisticalTestResult:
    """NIST serial test (overlapping m-bit patterns).

    Tests whether all 2^m overlapping m-bit patterns occur with
    approximately equal frequency.

    Computes two p-values from the statistics:
        delta_psi2_m   = psi2_m   - psi2_{m-1}
        delta2_psi2_m  = psi2_m - 2*psi2_{m-1} + psi2_{m-2}

    Parameters
    ----------
    bits : numpy.ndarray
        Binary sequence.
    block_size : int
        Pattern size m (2 to min(16, floor(log2(n))-2)).
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
        The p_value field contains the minimum of the two serial p-values.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits)
    m = block_size

    if m < 2:
        raise ValueError("block_size must be >= 2")
    max_m = max(2, int(math.log2(n)) - 2) if n > 0 else 2
    if m > max_m:
        raise ValueError(f"block_size too large for sequence length; max={max_m}")

    # Compute psi-squared for m, m-1, m-2
    psi2_m = _psi_squared(bits, n, m)
    psi2_m1 = _psi_squared(bits, n, m - 1)
    psi2_m2 = _psi_squared(bits, n, m - 2) if m >= 3 else 0.0

    delta1 = psi2_m - psi2_m1
    delta2 = psi2_m - 2.0 * psi2_m1 + psi2_m2

    p1 = float(gammaincc(2.0 ** (m - 2), delta1 / 2.0))
    p2 = float(gammaincc(2.0 ** (m - 3), delta2 / 2.0)) if m >= 3 else p1

    p_value = min(p1, p2)

    return StatisticalTestResult(
        name="Serial",
        p_value=p_value,
        passed=p_value >= alpha,
        statistic=float(delta1),
        alpha=alpha,
        details={
            "block_size": m,
            "psi2_m": psi2_m,
            "psi2_m1": psi2_m1,
            "psi2_m2": psi2_m2,
            "delta1": delta1,
            "delta2": delta2,
            "p1": p1,
            "p2": p2,
        },
    )


def _psi_squared(bits: np.ndarray, n: int, m: int) -> float:
    """Compute the psi-squared statistic for m-bit overlapping patterns.

    psi2_m = (2^m / n) * sum_i (count_i)^2 - n

    where count_i is the number of occurrences of pattern i.
    """
    if m == 0:
        return 0.0

    n_patterns = 2**m
    counts = np.zeros(n_patterns, dtype=np.int64)

    # Circular extension: append first (m-1) bits
    extended = np.concatenate([bits, bits[: m - 1]])

    # Count overlapping m-bit patterns
    for i in range(n):
        pattern = 0
        for j in range(m):
            pattern = (pattern << 1) | int(extended[i + j])
        counts[pattern] += 1

    return float(
        (n_patterns / n) * np.sum(counts.astype(np.float64) ** 2) - n
    )


# ---------------------------------------------------------------------------
# Test 6: Approximate Entropy Test
# ---------------------------------------------------------------------------


def approximate_entropy_test(
    bits: np.ndarray,
    block_size: int = 4,
    alpha: float = 0.01,
) -> StatisticalTestResult:
    """NIST approximate entropy (ApEn) test.

    Compares the frequency of overlapping blocks of lengths m and m+1.
    Random sequences have ApEn close to the maximum (log 2).

    Parameters
    ----------
    bits : numpy.ndarray
        Binary sequence.
    block_size : int
        Pattern size m.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits)
    m = block_size

    if m < 1:
        raise ValueError("block_size must be >= 1")

    phi_m = _phi(bits, n, m)
    phi_m1 = _phi(bits, n, m + 1)

    apen = phi_m - phi_m1
    chi2 = 2.0 * n * (math.log(2) - apen)
    p_value = float(gammaincc(2.0 ** (m - 1), chi2 / 2.0))

    return StatisticalTestResult(
        name="Approximate Entropy",
        p_value=p_value,
        passed=p_value >= alpha,
        statistic=float(chi2),
        alpha=alpha,
        details={
            "block_size": m,
            "phi_m": phi_m,
            "phi_m1": phi_m1,
            "apen": apen,
        },
    )


def _phi(bits: np.ndarray, n: int, m: int) -> float:
    """Compute phi(m) for the approximate entropy test.

    phi(m) = sum_j (C_j * log(C_j)) where C_j = count_j / n.
    """
    if m == 0:
        return 0.0

    n_patterns = 2**m
    counts = np.zeros(n_patterns, dtype=np.int64)

    extended = np.concatenate([bits, bits[: m - 1]])
    for i in range(n):
        pattern = 0
        for j in range(m):
            pattern = (pattern << 1) | int(extended[i + j])
        counts[pattern] += 1

    # Compute phi
    c = counts.astype(np.float64) / n
    # Avoid log(0)
    mask = c > 0
    return float(np.sum(c[mask] * np.log(c[mask])))


# ---------------------------------------------------------------------------
# Test 7: Cumulative Sums Test
# ---------------------------------------------------------------------------


def cumulative_sums_test(
    bits: np.ndarray,
    mode: str = "forward",
    alpha: float = 0.01,
) -> StatisticalTestResult:
    """NIST cumulative sums (CUSUM) test.

    Determines whether the cumulative sum of the adjusted (-1, +1) sequence
    is too large or too small relative to the expected behaviour of a
    random walk.

    Parameters
    ----------
    bits : numpy.ndarray
        Binary sequence.
    mode : str
        'forward' or 'backward'.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits)
    if n < 100:
        raise ValueError("Cumulative sums test requires n >= 100")

    # Convert to +1/-1
    x = 2.0 * bits.astype(np.float64) - 1.0
    if mode == "backward":
        x = x[::-1]
    elif mode != "forward":
        raise ValueError("mode must be 'forward' or 'backward'")

    # Cumulative sum
    s = np.cumsum(x)
    z = float(np.max(np.abs(s)))

    # Compute p-value using the distribution of max|S_k|
    sqrt_n = math.sqrt(n)
    p_value = 0.0

    # Sum terms for the p-value
    # p = 1 - sum_{k=-floor((-n/z+1)/4)}^{floor((n/z-1)/4)}
    #     [Phi((4k+1)z/sqrt(n)) - Phi((4k-1)z/sqrt(n))]
    # where Phi is the standard normal CDF
    if z > 0:
        start = int(math.floor((-n / z + 1) / 4))
        end = int(math.floor((n / z - 1) / 4))

        sum1 = 0.0
        for k in range(start, end + 1):
            a = (4 * k + 1) * z / sqrt_n
            b = (4 * k - 1) * z / sqrt_n
            sum1 += _norm_cdf(a) - _norm_cdf(b)

        start2 = int(math.floor((-n / z - 3) / 4))
        end2 = int(math.floor((n / z - 1) / 4))

        sum2 = 0.0
        for k in range(start2, end2 + 1):
            a = (4 * k + 3) * z / sqrt_n
            b = (4 * k + 1) * z / sqrt_n
            sum2 += _norm_cdf(a) - _norm_cdf(b)

        p_value = 1.0 - sum1 + sum2
    else:
        p_value = 1.0

    p_value = max(0.0, min(1.0, p_value))

    return StatisticalTestResult(
        name=f"Cumulative Sums ({mode})",
        p_value=p_value,
        passed=p_value >= alpha,
        statistic=z,
        alpha=alpha,
        details={"n": n, "z": z, "mode": mode},
    )


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using erfc."""
    return 0.5 * float(erfc(-x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Test 8: DFT Spectral Test
# ---------------------------------------------------------------------------


def dft_spectral_test(bits: np.ndarray, alpha: float = 0.01) -> StatisticalTestResult:
    """NIST discrete Fourier transform (spectral) test.

    Detects periodic features in the sequence that would indicate
    deviation from randomness. The test computes the DFT of the +/-1
    sequence and checks whether the peak heights are consistent with
    a random sequence.

    Parameters
    ----------
    bits : numpy.ndarray
        Binary sequence.
    alpha : float
        Significance level.

    Returns
    -------
    TestResult
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits)
    if n < 100:
        raise ValueError("DFT test requires n >= 100")

    # Convert to +1/-1
    x = 2.0 * bits.astype(np.float64) - 1.0

    # Compute DFT (only first n/2 components are meaningful)
    dft = np.fft.fft(x)
    half = n // 2
    magnitudes = np.abs(dft[:half])

    # Threshold: T = sqrt(log(1/0.05) * n) = sqrt(2.995732 * n)
    threshold = math.sqrt(math.log(1.0 / 0.05) * n)

    # Expected number of peaks below threshold (under H0): 0.95 * n/2
    n_below = np.sum(magnitudes < threshold)
    expected_below = 0.95 * half

    # Test statistic
    d = (n_below - expected_below) / math.sqrt(half * 0.95 * 0.05)
    p_value = float(erfc(abs(d) / math.sqrt(2)))

    return StatisticalTestResult(
        name="DFT Spectral",
        p_value=p_value,
        passed=p_value >= alpha,
        statistic=float(d),
        alpha=alpha,
        details={
            "n": n,
            "threshold": threshold,
            "n_below_threshold": int(n_below),
            "expected_below": expected_below,
            "fraction_below": float(n_below) / half,
        },
    )


# ---------------------------------------------------------------------------
# Randomness Report (run all tests)
# ---------------------------------------------------------------------------


@dataclass
class RandomnessReport:
    """Run all NIST tests on a bit sequence and produce a summary report.

    Parameters
    ----------
    bits : numpy.ndarray
        Binary sequence to test (uint8, each element 0 or 1).
    alpha : float
        Significance level for all tests (default 0.01).
    """

    bits: np.ndarray
    alpha: float = 0.01
    _results: list = field(init=False, default_factory=list, repr=False)
    _run_complete: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.bits = np.asarray(self.bits, dtype=np.uint8)
        if len(self.bits) < 100:
            raise ValueError("Sequence must be >= 100 bits for NIST tests")

    def run_all(self) -> list[TestResult]:
        """Execute all available NIST tests.

        Returns
        -------
        list[TestResult]
            Results for each test.
        """
        self._results = []
        n = len(self.bits)

        # 1. Frequency test
        self._results.append(frequency_test(self.bits, self.alpha))

        # 2. Block frequency test
        block_sz = min(128, n // 10)
        if block_sz >= 20:
            self._results.append(
                block_frequency_test(self.bits, block_sz, self.alpha)
            )

        # 3. Runs test
        self._results.append(runs_test(self.bits, self.alpha))

        # 4. Longest run test
        if n >= 128:
            self._results.append(longest_run_test(self.bits, self.alpha))

        # 5. Serial test
        max_m = max(2, int(math.log2(n)) - 2) if n > 16 else 2
        serial_m = min(3, max_m)
        if serial_m >= 2:
            self._results.append(
                serial_test(self.bits, serial_m, self.alpha)
            )

        # 6. Approximate entropy test
        if n >= 100:
            apen_m = min(4, max(1, int(math.log2(n)) - 5))
            self._results.append(
                approximate_entropy_test(self.bits, apen_m, self.alpha)
            )

        # 7. Cumulative sums test (both directions)
        self._results.append(
            cumulative_sums_test(self.bits, "forward", self.alpha)
        )
        self._results.append(
            cumulative_sums_test(self.bits, "backward", self.alpha)
        )

        # 8. DFT spectral test
        self._results.append(dft_spectral_test(self.bits, self.alpha))

        self._run_complete = True
        return self._results

    @property
    def results(self) -> list[TestResult]:
        """List of test results (run_all() must be called first)."""
        if not self._run_complete:
            raise RuntimeError("Call run_all() before accessing results")
        return self._results

    @property
    def n_passed(self) -> int:
        """Number of tests passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        """Number of tests failed."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def n_tests(self) -> int:
        """Total number of tests run."""
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        """Whether all tests passed."""
        return all(r.passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        """Fraction of tests passed."""
        if self.n_tests == 0:
            return 0.0
        return self.n_passed / self.n_tests

    def summary(self) -> str:
        """Human-readable summary of test results.

        Returns
        -------
        str
            Formatted summary string.
        """
        if not self._run_complete:
            return "No tests run yet. Call run_all() first."

        lines = [
            f"NIST SP 800-22 Randomness Report (n={len(self.bits)}, "
            f"alpha={self.alpha})",
            "=" * 70,
        ]

        for r in self._results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"  [{status}] {r.name:<35s} p={r.p_value:.6f}  "
                f"stat={r.statistic:.4f}"
            )

        lines.append("=" * 70)
        lines.append(
            f"  Result: {self.n_passed}/{self.n_tests} passed "
            f"({self.pass_rate:.1%})"
        )

        if self.all_passed:
            lines.append("  Conclusion: Sequence is consistent with randomness")
        else:
            failed = [r.name for r in self._results if not r.passed]
            lines.append(f"  Failed tests: {', '.join(failed)}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export results as a dictionary for serialisation."""
        if not self._run_complete:
            raise RuntimeError("Call run_all() before exporting")
        return {
            "n_bits": len(self.bits),
            "alpha": self.alpha,
            "n_tests": self.n_tests,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "all_passed": self.all_passed,
            "pass_rate": self.pass_rate,
            "tests": [
                {
                    "name": r.name,
                    "p_value": r.p_value,
                    "passed": r.passed,
                    "statistic": r.statistic,
                    "details": r.details,
                }
                for r in self._results
            ],
        }
