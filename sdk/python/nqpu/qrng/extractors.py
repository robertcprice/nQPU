"""Randomness extraction and post-processing for quantum random bit streams.

Raw quantum measurements often contain bias (e.g. detector asymmetry, state
preparation imperfections). Randomness extractors transform a weakly random
source with min-entropy k into a near-uniform output of length m < k.

Extractors provided:
  - VonNeumannExtractor: Classic pair-based debiasing (no seed required)
  - ToeplitzExtractor: Universal hash family with Toeplitz matrix (seeded)
  - TrevisanExtractor: Near-optimal seed length extractor (simplified)
  - XORExtractor: Multi-source XOR combination (no seed required)
  - MinEntropyEstimator: Estimate min-entropy from empirical data

References:
  - Von Neumann, "Various Techniques Used in Connection with Random Digits",
    NBS Applied Mathematics Series 12, 36-38 (1951)
  - Ma et al., Phys. Rev. A 87, 062327 (2013) [Toeplitz extraction for QRNG]
  - Trevisan, J. ACM 48(4), 860-879 (2001)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Von Neumann Extractor
# ---------------------------------------------------------------------------


@dataclass
class VonNeumannExtractor:
    """Classic debiasing extractor for i.i.d. biased bits.

    Takes pairs of bits and applies the rule:
        01 -> 0
        10 -> 1
        00, 11 -> discard

    For a source with bias p, the output is perfectly uniform but the
    extraction rate is 2p(1-p), which is at most 0.5 for p=0.5.

    This extractor requires no seed but assumes the input bits are
    independent (not just identically distributed).

    Parameters
    ----------
    recursive : bool
        If True, apply Von Neumann extraction recursively to the discarded
        same-value pairs to improve efficiency. Typically gains 10-20%
        more output bits.
    """

    recursive: bool = False
    _total_input: int = field(init=False, default=0, repr=False)
    _total_output: int = field(init=False, default=0, repr=False)

    def extract(self, bits: np.ndarray) -> np.ndarray:
        """Extract unbiased bits from a biased input stream.

        Parameters
        ----------
        bits : numpy.ndarray
            Input bit array (uint8, each element 0 or 1).

        Returns
        -------
        numpy.ndarray
            Debiased output bits (uint8).
        """
        bits = np.asarray(bits, dtype=np.uint8)
        if len(bits) < 2:
            return np.array([], dtype=np.uint8)

        self._total_input += len(bits)
        output = self._extract_single_pass(bits)

        if self.recursive:
            # Apply recursively to discarded pairs
            same_pairs = self._get_same_pairs(bits)
            while len(same_pairs) >= 2:
                extra = self._extract_single_pass(same_pairs)
                output = np.concatenate([output, extra])
                same_pairs = self._get_same_pairs(same_pairs)

        self._total_output += len(output)
        return output

    @staticmethod
    def _extract_single_pass(bits: np.ndarray) -> np.ndarray:
        """Single pass of Von Neumann extraction."""
        n_pairs = len(bits) // 2
        first = bits[0 : 2 * n_pairs : 2]
        second = bits[1 : 2 * n_pairs : 2]

        # Keep only pairs where bits differ
        mask = first != second
        # Output bit is the first bit of each differing pair
        return first[mask]

    @staticmethod
    def _get_same_pairs(bits: np.ndarray) -> np.ndarray:
        """Extract same-value pairs for recursive processing.

        For pairs (0,0) -> 0, (1,1) -> 1, others discarded.
        """
        n_pairs = len(bits) // 2
        first = bits[0 : 2 * n_pairs : 2]
        second = bits[1 : 2 * n_pairs : 2]
        mask = first == second
        return first[mask]

    @property
    def extraction_rate(self) -> float:
        """Empirical extraction rate (output/input bits)."""
        if self._total_input == 0:
            return 0.0
        return self._total_output / self._total_input

    def theoretical_rate(self, bias: float) -> float:
        """Theoretical extraction rate for a given bias p = P(bit=1).

        Parameters
        ----------
        bias : float
            Probability of outputting 1 (in [0, 1]).

        Returns
        -------
        float
            Expected output bits per input pair: 2*p*(1-p) / 2 = p*(1-p).
            (Factor of 2 because we consume 2 bits per pair.)
        """
        return bias * (1 - bias)


# ---------------------------------------------------------------------------
# Toeplitz Extractor
# ---------------------------------------------------------------------------


@dataclass
class ToeplitzExtractor:
    """Universal hash extractor using a random Toeplitz matrix.

    A Toeplitz matrix is defined by its first row and first column, requiring
    only (n + m - 1) random seed bits for an n-input, m-output extractor.
    The extraction is provably correct: if the input has min-entropy k,
    the output is epsilon-close to uniform for m <= k - 2*log(1/epsilon).

    The matrix-vector multiplication is performed efficiently using the
    structure of Toeplitz matrices (equivalent to polynomial multiplication).

    Parameters
    ----------
    input_length : int
        Number of input bits n.
    output_length : int
        Number of output bits m (must be < input_length).
    seed : numpy.ndarray or None
        Random seed bits of length (input_length + output_length - 1).
        If None, generated randomly.
    rng_seed : int or None
        Seed for generating the Toeplitz matrix seed.
    """

    input_length: int = 1024
    output_length: int = 256
    seed: Optional[np.ndarray] = None
    rng_seed: Optional[int] = None
    _matrix_seed: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.output_length >= self.input_length:
            raise ValueError(
                f"output_length ({self.output_length}) must be < "
                f"input_length ({self.input_length})"
            )
        if self.output_length < 1:
            raise ValueError("output_length must be >= 1")

        seed_length = self.input_length + self.output_length - 1
        if self.seed is not None:
            self.seed = np.asarray(self.seed, dtype=np.uint8)
            if len(self.seed) != seed_length:
                raise ValueError(
                    f"seed length must be {seed_length}, got {len(self.seed)}"
                )
            self._matrix_seed = self.seed
        else:
            rng = np.random.default_rng(self.rng_seed)
            self._matrix_seed = rng.integers(
                0, 2, size=seed_length, dtype=np.uint8
            )

    def extract(self, bits: np.ndarray) -> np.ndarray:
        """Extract output_length near-uniform bits from input bits.

        Parameters
        ----------
        bits : numpy.ndarray
            Input bit array of length input_length.

        Returns
        -------
        numpy.ndarray
            Output bit array of length output_length (uint8, each 0 or 1).
        """
        bits = np.asarray(bits, dtype=np.uint8)
        if len(bits) != self.input_length:
            raise ValueError(
                f"Input must have {self.input_length} bits, got {len(bits)}"
            )

        # Toeplitz matrix-vector multiply via convolution-style computation
        output = np.zeros(self.output_length, dtype=np.uint8)
        for i in range(self.output_length):
            # Row i of the Toeplitz matrix is seed[m-1-i : m-1-i+n]
            row_start = self.output_length - 1 - i
            row = self._matrix_seed[row_start : row_start + self.input_length]
            output[i] = np.sum(row.astype(np.int32) * bits.astype(np.int32)) % 2

        return output

    def extract_stream(self, bits: np.ndarray) -> np.ndarray:
        """Extract from a long stream by processing input_length-sized blocks.

        Parameters
        ----------
        bits : numpy.ndarray
            Input bit array (length must be a multiple of input_length).

        Returns
        -------
        numpy.ndarray
            Concatenated output blocks.
        """
        bits = np.asarray(bits, dtype=np.uint8)
        if len(bits) % self.input_length != 0:
            raise ValueError(
                f"Input length {len(bits)} is not a multiple of "
                f"input_length {self.input_length}"
            )

        n_blocks = len(bits) // self.input_length
        output_blocks = []
        for i in range(n_blocks):
            block = bits[i * self.input_length : (i + 1) * self.input_length]
            output_blocks.append(self.extract(block))

        return np.concatenate(output_blocks)

    @property
    def matrix_seed(self) -> np.ndarray:
        """The Toeplitz matrix seed (read-only copy)."""
        return self._matrix_seed.copy()

    @property
    def compression_ratio(self) -> float:
        """Ratio of output bits to input bits."""
        return self.output_length / self.input_length

    @staticmethod
    def required_min_entropy(
        output_length: int,
        epsilon: float = 1e-6,
    ) -> float:
        """Minimum min-entropy required for epsilon-secure extraction.

        Parameters
        ----------
        output_length : int
            Desired number of output bits.
        epsilon : float
            Security parameter (distance from uniform).

        Returns
        -------
        float
            Required min-entropy k >= m + 2*log2(1/epsilon).
        """
        return output_length + 2 * math.log2(1.0 / epsilon)


# ---------------------------------------------------------------------------
# Trevisan Extractor (simplified)
# ---------------------------------------------------------------------------


@dataclass
class TrevisanExtractor:
    """Simplified Trevisan extractor with near-optimal seed length.

    Trevisan's construction uses a weak design (combinatorial structure) to
    select seed bits for each output bit, then applies a one-bit extractor
    (inner code) to the selected positions. This achieves seed length
    O(log(n) * log(n/epsilon)) which is exponentially better than Toeplitz.

    This simplified implementation uses a linear-feedback approach where
    each output bit is computed from a different subset of seed and input
    bits, determined by a pseudorandom selection pattern.

    Parameters
    ----------
    input_length : int
        Number of input bits.
    output_length : int
        Number of output bits (must be < input_length).
    seed : numpy.ndarray or None
        Explicit seed bits. If None, generated from rng_seed.
    rng_seed : int or None
        Seed for generating the extractor seed.
    """

    input_length: int = 1024
    output_length: int = 256
    seed: Optional[np.ndarray] = None
    rng_seed: Optional[int] = None
    _seed_bits: np.ndarray = field(init=False, repr=False)
    _design: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.output_length >= self.input_length:
            raise ValueError(
                f"output_length ({self.output_length}) must be < "
                f"input_length ({self.input_length})"
            )

        # Seed length for the simplified construction
        self._seed_length = self._compute_seed_length()

        if self.seed is not None:
            self._seed_bits = np.asarray(self.seed, dtype=np.uint8)
            if len(self._seed_bits) != self._seed_length:
                raise ValueError(
                    f"seed length must be {self._seed_length}, "
                    f"got {len(self._seed_bits)}"
                )
        else:
            rng = np.random.default_rng(self.rng_seed)
            self._seed_bits = rng.integers(
                0, 2, size=self._seed_length, dtype=np.uint8
            )

        # Build the weak design: for each output bit, select a subset of
        # input positions to XOR together
        self._build_design()

    def _compute_seed_length(self) -> int:
        """Compute required seed length.

        For the simplified construction, we use
        O(log^2(n)) seed bits where n = input_length.
        """
        log_n = max(1, math.ceil(math.log2(self.input_length)))
        # Each output bit uses log_n input positions selected by log_n seed bits
        return max(64, log_n * log_n)

    def _build_design(self) -> None:
        """Build the weak design (subset selection pattern).

        For each of the m output bits, compute a subset of input positions
        of size t = ceil(log2(n)). The subsets are generated using the
        seed bits and a simple combinatorial construction.
        """
        n = self.input_length
        m = self.output_length
        t = max(1, math.ceil(math.log2(n)))  # subset size

        # Use seed to deterministically generate subsets
        rng = np.random.default_rng(
            int.from_bytes(
                bytes(self._seed_bits[:min(8, len(self._seed_bits))]),
                "little",
            )
        )
        self._design = np.zeros((m, t), dtype=np.int64)
        for i in range(m):
            self._design[i] = rng.choice(n, size=t, replace=False)

    def extract(self, bits: np.ndarray) -> np.ndarray:
        """Extract output_length bits from input.

        Parameters
        ----------
        bits : numpy.ndarray
            Input bit array of length input_length.

        Returns
        -------
        numpy.ndarray
            Output bit array (uint8).
        """
        bits = np.asarray(bits, dtype=np.uint8)
        if len(bits) != self.input_length:
            raise ValueError(
                f"Input must have {self.input_length} bits, got {len(bits)}"
            )

        output = np.zeros(self.output_length, dtype=np.uint8)
        for i in range(self.output_length):
            # XOR the bits at the selected positions
            selected = bits[self._design[i]]
            output[i] = np.bitwise_xor.reduce(selected)

        return output

    @property
    def seed_length(self) -> int:
        """Length of the extractor seed in bits."""
        return self._seed_length

    @property
    def seed_bits(self) -> np.ndarray:
        """The extractor seed (read-only copy)."""
        return self._seed_bits.copy()


# ---------------------------------------------------------------------------
# XOR Extractor
# ---------------------------------------------------------------------------


@dataclass
class XORExtractor:
    """Multi-source XOR combination extractor.

    Given k independent sources (each with some min-entropy), XOR-combining
    them produces output with min-entropy at least as high as any single
    source. This is one of the simplest multi-source extractors.

    When combining two independent sources where at least one has min-entropy
    rate > 0.5, the XOR is provably close to uniform.

    This extractor requires NO seed -- it exploits independence between sources.
    """

    _extractions: int = field(init=False, default=0, repr=False)

    def extract(self, *sources: np.ndarray) -> np.ndarray:
        """XOR-combine multiple independent bit sources.

        Parameters
        ----------
        *sources : numpy.ndarray
            Two or more bit arrays of equal length. Each should come from
            an independent random source.

        Returns
        -------
        numpy.ndarray
            XOR combination of all sources (uint8, each 0 or 1).

        Raises
        ------
        ValueError
            If fewer than 2 sources or sources have different lengths.
        """
        if len(sources) < 2:
            raise ValueError("At least 2 independent sources are required")

        arrays = [np.asarray(s, dtype=np.uint8) for s in sources]
        lengths = {len(a) for a in arrays}
        if len(lengths) > 1:
            raise ValueError(
                f"All sources must have the same length; got {lengths}"
            )

        result = arrays[0].copy()
        for arr in arrays[1:]:
            result = np.bitwise_xor(result, arr)

        self._extractions += 1
        return result

    def extract_with_shift(
        self,
        source: np.ndarray,
        n_shifts: int = 3,
    ) -> np.ndarray:
        """Extract from a single source by XOR-combining shifted copies.

        This is a heuristic (not provably secure) but effective for removing
        short-range correlations. Each shifted copy is treated as a
        quasi-independent source.

        Parameters
        ----------
        source : numpy.ndarray
            Input bit array.
        n_shifts : int
            Number of shifted copies to XOR (>= 2).

        Returns
        -------
        numpy.ndarray
            XOR of original and shifted copies (shorter by n_shifts-1).
        """
        source = np.asarray(source, dtype=np.uint8)
        if n_shifts < 2:
            raise ValueError("n_shifts must be >= 2")
        if len(source) < n_shifts:
            raise ValueError("Source too short for the requested shifts")

        out_len = len(source) - (n_shifts - 1)
        result = source[:out_len].copy()
        for shift in range(1, n_shifts):
            result = np.bitwise_xor(result, source[shift : shift + out_len])

        self._extractions += 1
        return result

    @property
    def total_extractions(self) -> int:
        """Number of extraction calls performed."""
        return self._extractions


# ---------------------------------------------------------------------------
# Min-Entropy Estimator
# ---------------------------------------------------------------------------


@dataclass
class MinEntropyEstimator:
    """Estimate the min-entropy of a binary sequence.

    Min-entropy H_inf = -log2(max_x P(X=x)) is the most conservative
    entropy measure. For a uniform binary source, H_inf = 1 per bit.

    Three estimation methods are provided:
      1. Collision test: Estimate from collision probability
      2. Compression test: Estimate from Maurer's universal statistic
      3. Frequency test: Direct estimation from bit frequencies

    Parameters
    ----------
    block_size : int
        Block size for pattern-based estimates (collision, compression).
        Typical values: 4-8 for short sequences, 8-16 for long ones.
    """

    block_size: int = 8

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")

    def estimate_all(self, bits: np.ndarray) -> dict:
        """Run all estimation methods and return a comprehensive report.

        Parameters
        ----------
        bits : numpy.ndarray
            Binary sequence (uint8 array, each element 0 or 1).

        Returns
        -------
        dict
            Keys: 'frequency', 'collision', 'compression', 'combined'.
            'combined' is the minimum (most conservative) estimate.
        """
        bits = np.asarray(bits, dtype=np.uint8)
        freq = self.frequency_estimate(bits)
        coll = self.collision_estimate(bits)
        comp = self.compression_estimate(bits)

        return {
            "frequency": freq,
            "collision": coll,
            "compression": comp,
            "combined": min(freq, coll, comp),
            "n_bits": len(bits),
            "block_size": self.block_size,
        }

    def frequency_estimate(self, bits: np.ndarray) -> float:
        """Estimate min-entropy from single-bit frequencies.

        H_inf = -log2(max(p0, p1)) where p0, p1 are empirical bit probs.

        Parameters
        ----------
        bits : numpy.ndarray
            Input bit sequence.

        Returns
        -------
        float
            Estimated min-entropy per bit (0 to 1).
        """
        bits = np.asarray(bits, dtype=np.uint8)
        if len(bits) == 0:
            return 0.0

        p1 = np.mean(bits)
        p0 = 1.0 - p1
        p_max = max(p0, p1)
        if p_max <= 0 or p_max >= 1:
            return 0.0
        return -math.log2(p_max)

    def collision_estimate(self, bits: np.ndarray) -> float:
        """Estimate min-entropy via the collision probability.

        The collision entropy H_2 = -log2(sum_x p(x)^2) is an upper bound
        on min-entropy. We estimate it from the empirical collision rate
        of b-bit blocks.

        Parameters
        ----------
        bits : numpy.ndarray
            Input bit sequence.

        Returns
        -------
        float
            Estimated min-entropy per bit.
        """
        bits = np.asarray(bits, dtype=np.uint8)
        b = self.block_size
        n_blocks = len(bits) // b
        if n_blocks < 2:
            return self.frequency_estimate(bits)

        # Convert blocks to integer labels
        blocks = self._bits_to_blocks(bits, b, n_blocks)

        # Count collisions: pairs of identical blocks
        unique, counts = np.unique(blocks, return_counts=True)
        # Collision probability estimate = sum(c*(c-1)) / (n*(n-1))
        collision_pairs = np.sum(counts.astype(np.int64) * (counts.astype(np.int64) - 1))
        total_pairs = n_blocks * (n_blocks - 1)

        if total_pairs == 0 or collision_pairs == 0:
            return float(b)  # Maximum entropy

        collision_prob = collision_pairs / total_pairs
        # H_2 = -log2(collision_prob) for b-bit blocks
        # Per-bit min-entropy estimate
        h2_per_block = -math.log2(collision_prob)
        return min(1.0, h2_per_block / b)

    def compression_estimate(self, bits: np.ndarray) -> float:
        """Estimate min-entropy via Maurer's universal statistical test.

        The test statistic f_n approximates the per-bit entropy. Lower
        values indicate less randomness.

        Parameters
        ----------
        bits : numpy.ndarray
            Input bit sequence.

        Returns
        -------
        float
            Estimated min-entropy per bit (capped at 1.0).
        """
        bits = np.asarray(bits, dtype=np.uint8)
        b = self.block_size
        n_blocks = len(bits) // b
        if n_blocks < 2 * (2**b):
            # Not enough data for reliable estimate
            return self.frequency_estimate(bits)

        blocks = self._bits_to_blocks(bits, b, n_blocks)

        # Initialisation: first 2^b blocks build the lookup table
        q = 2**b  # initialisation segment length
        if n_blocks <= q:
            return self.frequency_estimate(bits)

        # Table of last-seen positions for each pattern
        last_seen = np.full(2**b, -1, dtype=np.int64)
        for i in range(q):
            last_seen[blocks[i]] = i

        # Test: compute average of log2(distance to last occurrence)
        k = n_blocks - q  # test segment length
        total = 0.0
        for i in range(q, n_blocks):
            pattern = blocks[i]
            if last_seen[pattern] >= 0:
                distance = i - last_seen[pattern]
                total += math.log2(distance)
            last_seen[pattern] = i

        f_n = total / k  # Maurer's statistic
        # Expected value for random data: approximately b - some small correction
        # Normalise to per-bit entropy (cap at 1.0)
        return min(1.0, f_n / b)

    @staticmethod
    def _bits_to_blocks(
        bits: np.ndarray,
        block_size: int,
        n_blocks: int,
    ) -> np.ndarray:
        """Convert a bit array to an array of integer block labels."""
        blocks = np.zeros(n_blocks, dtype=np.int64)
        for i in range(n_blocks):
            val = 0
            for j in range(block_size):
                val |= int(bits[i * block_size + j]) << j
            blocks[i] = val
        return blocks
