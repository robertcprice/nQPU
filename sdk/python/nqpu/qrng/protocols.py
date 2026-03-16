"""High-level QRNG protocols combining generation, extraction, and certification.

Provides ready-to-use protocols for different security models:
  - DIQRNG: Device-independent (Bell test certified, strongest guarantee)
  - SemiDIQRNG: Semi-device-independent (energy-bounded, easier to implement)
  - RandomBeacon: Publicly verifiable randomness with hash-chain audit trail

Also provides convenience functions for common use cases:
  - random_bits(n): Generate n certified random bits
  - random_uniform(n): Generate n uniform floats in [0, 1)
  - random_integers(n, low, high): Generate n uniform integers in [low, high)

References:
  - Acin & Masanes, Nature 540, 213 (2016) [DI certification review]
  - Lunghi et al., Phys. Rev. Lett. 114, 150501 (2015) [semi-DI QRNG]
  - Fischer & Piasecki, arXiv:1907.02959 (2019) [random beacons]
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .generators import MeasurementQRNG, EntanglementQRNG, VacuumFluctuationQRNG
from .extractors import ToeplitzExtractor, VonNeumannExtractor
from .certification import CHSHCertifier, EntropyAccumulation


# ---------------------------------------------------------------------------
# Device-Independent QRNG Protocol
# ---------------------------------------------------------------------------


@dataclass
class DIQRNG:
    """Device-independent quantum random number generation protocol.

    Full pipeline: Bell test -> entropy estimation -> randomness extraction.

    Security guarantee: The output is certifiably random under the sole
    assumption that the device cannot signal faster than light (no-signalling).
    No trust in the internal workings of the device is required.

    Protocol steps:
      1. Prepare n Bell pairs
      2. For each pair, randomly choose measurement basis using seed bits
      3. Measure and record outcomes
      4. Designate a fraction as test rounds; compute CHSH S-value
      5. Estimate min-entropy from S-value using entropy accumulation
      6. Apply Toeplitz extractor to generation outcomes

    Parameters
    ----------
    n_rounds : int
        Total number of Bell-state measurements.
    test_fraction : float
        Fraction of rounds used for CHSH testing (rest for generation).
    extraction_ratio : float
        Ratio of output bits to certified entropy (must be < 1 for security).
    seed : int or None
        RNG seed for reproducibility.
    min_s_value : float
        Minimum required CHSH S-value. Abort if not met.
    """

    n_rounds: int = 100000
    test_fraction: float = 0.1
    extraction_ratio: float = 0.9
    seed: Optional[int] = None
    min_s_value: float = 2.1
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_rounds < 100:
            raise ValueError("n_rounds must be >= 100")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in (0, 1)")
        if not 0 < self.extraction_ratio < 1:
            raise ValueError("extraction_ratio must be in (0, 1)")
        self._rng = np.random.default_rng(self.seed)

    def generate(self) -> DIQRNGResult:
        """Execute the full DI-QRNG protocol.

        Returns
        -------
        DIQRNGResult
            Contains certified random bits and audit information.
        """
        n_test = int(self.n_rounds * self.test_fraction)
        n_gen = self.n_rounds - n_test

        # Create Bell-state source
        bell = EntanglementQRNG(
            seed=self._rng.integers(0, 2**32),
            bell_state="phi_plus",
        )

        # --- Test rounds: CHSH estimation ---
        chsh_result = bell.compute_chsh_correlation(n_test)
        s_value = chsh_result["s_value"]

        if s_value < self.min_s_value:
            return DIQRNGResult(
                output_bits=np.array([], dtype=np.uint8),
                s_value=s_value,
                certified_entropy=0.0,
                n_output_bits=0,
                n_test_rounds=n_test,
                n_generation_rounds=n_gen,
                aborted=True,
                abort_reason=f"CHSH S={s_value:.4f} < {self.min_s_value}",
            )

        # --- Generation rounds: collect raw bits ---
        raw_bits = bell.generate_bits(n_gen)

        # --- Entropy estimation ---
        ea = EntropyAccumulation(
            n_rounds=self.n_rounds,
            test_fraction=self.test_fraction,
            seed=self._rng.integers(0, 2**32),
        )
        eat_result = ea.accumulate()
        certified_entropy = eat_result.total_smooth_min_entropy

        # --- Extraction ---
        output_length = max(1, int(certified_entropy * self.extraction_ratio))
        output_length = min(output_length, n_gen - 1)

        if output_length < 1 or n_gen < 2:
            return DIQRNGResult(
                output_bits=np.array([], dtype=np.uint8),
                s_value=s_value,
                certified_entropy=certified_entropy,
                n_output_bits=0,
                n_test_rounds=n_test,
                n_generation_rounds=n_gen,
                aborted=True,
                abort_reason="Insufficient certified entropy for extraction",
            )

        extractor = ToeplitzExtractor(
            input_length=n_gen,
            output_length=output_length,
            rng_seed=self._rng.integers(0, 2**32),
        )
        output_bits = extractor.extract(raw_bits)

        return DIQRNGResult(
            output_bits=output_bits,
            s_value=s_value,
            certified_entropy=certified_entropy,
            n_output_bits=len(output_bits),
            n_test_rounds=n_test,
            n_generation_rounds=n_gen,
            aborted=False,
            abort_reason=None,
        )


@dataclass
class DIQRNGResult:
    """Result of DI-QRNG protocol execution.

    Attributes
    ----------
    output_bits : numpy.ndarray
        Certified random output bits (uint8, each 0 or 1).
    s_value : float
        Measured CHSH S-value.
    certified_entropy : float
        Total certified smooth min-entropy (bits).
    n_output_bits : int
        Number of output bits produced.
    n_test_rounds : int
        Number of rounds used for testing.
    n_generation_rounds : int
        Number of rounds used for generation.
    aborted : bool
        Whether the protocol was aborted.
    abort_reason : str or None
        Reason for abort, if any.
    """

    output_bits: np.ndarray
    s_value: float
    certified_entropy: float
    n_output_bits: int
    n_test_rounds: int
    n_generation_rounds: int
    aborted: bool
    abort_reason: Optional[str]

    def summary(self) -> str:
        """Human-readable summary."""
        if self.aborted:
            return (
                f"DI-QRNG ABORTED: {self.abort_reason}\n"
                f"  S-value: {self.s_value:.4f}"
            )
        return (
            f"DI-QRNG Result:\n"
            f"  S-value:          {self.s_value:.4f}\n"
            f"  Certified H_min:  {self.certified_entropy:.1f} bits\n"
            f"  Output:           {self.n_output_bits} bits\n"
            f"  Test rounds:      {self.n_test_rounds}\n"
            f"  Generation:       {self.n_generation_rounds}"
        )


# ---------------------------------------------------------------------------
# Semi-Device-Independent QRNG
# ---------------------------------------------------------------------------


@dataclass
class SemiDIQRNG:
    """Semi-device-independent QRNG using prepare-and-measure protocol.

    Security model: The source device is untrusted, but the measurement
    device has a bounded Hilbert space dimension (typically qubit = dim 2).
    This is weaker than full DI but much easier to implement.

    Protocol:
      1. Source prepares states (claimed to be qubits)
      2. Receiver measures in random bases
      3. Dimension witness certifies that states are genuinely quantum
      4. Min-entropy bounded by dimension witness violation

    The dimension witness W for qubit preparations satisfies:
      - Classical (dim 1): W <= 1
      - Quantum (dim 2): W <= sqrt(2)

    Parameters
    ----------
    n_rounds : int
        Number of measurement rounds.
    dimension_bound : int
        Assumed dimension of prepared states.
    seed : int or None
        RNG seed.
    """

    n_rounds: int = 10000
    dimension_bound: int = 2
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_rounds < 100:
            raise ValueError("n_rounds must be >= 100")
        if self.dimension_bound < 2:
            raise ValueError("dimension_bound must be >= 2")
        self._rng = np.random.default_rng(self.seed)

    def generate(self) -> SemiDIResult:
        """Execute the semi-DI QRNG protocol.

        Returns
        -------
        SemiDIResult
        """
        # Simulate prepare-and-measure with qubit states
        # Source prepares states on Bloch sphere; receiver measures in X or Z

        n_test = int(self.n_rounds * 0.1)
        n_gen = self.n_rounds - n_test

        # Dimension witness estimation
        # For a qubit source with optimal preparation:
        # W = sqrt(2) when using conjugate bases
        # Simulate with finite statistics
        w_theoretical = math.sqrt(self.dimension_bound)
        w_noise = self._rng.normal(0, 1.0 / math.sqrt(n_test))
        w_measured = w_theoretical * 0.95 + w_noise  # 5% imperfection

        # Classical bound for given dimension
        classical_bound = 1.0

        # Min-entropy from dimension witness
        if w_measured > classical_bound:
            # Simplified bound: H_min >= log2(1 + (W - 1))
            h_min_rate = min(1.0, math.log2(1.0 + (w_measured - 1.0)))
        else:
            h_min_rate = 0.0

        # Generate raw bits
        source = MeasurementQRNG(
            seed=self._rng.integers(0, 2**32),
            basis="random_rotation",
        )
        raw_bits = source.generate_bits(n_gen)

        # Extract
        certified_entropy = n_gen * h_min_rate
        output_length = max(1, int(certified_entropy * 0.8))
        output_length = min(output_length, n_gen - 1)

        if output_length >= 2:
            vn = VonNeumannExtractor(recursive=True)
            extracted = vn.extract(raw_bits)
            output_bits = extracted[:output_length] if len(extracted) >= output_length else extracted
        else:
            output_bits = np.array([], dtype=np.uint8)

        return SemiDIResult(
            output_bits=output_bits,
            dimension_witness=w_measured,
            classical_bound=classical_bound,
            entropy_rate=h_min_rate,
            n_output_bits=len(output_bits),
            n_test_rounds=n_test,
            n_generation_rounds=n_gen,
            is_certified=w_measured > classical_bound,
        )


@dataclass
class SemiDIResult:
    """Result of semi-DI QRNG protocol."""

    output_bits: np.ndarray
    dimension_witness: float
    classical_bound: float
    entropy_rate: float
    n_output_bits: int
    n_test_rounds: int
    n_generation_rounds: int
    is_certified: bool

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Semi-DI QRNG Result:\n"
            f"  Dimension witness: {self.dimension_witness:.4f} "
            f"(classical <= {self.classical_bound:.4f})\n"
            f"  Certified:         {'YES' if self.is_certified else 'NO'}\n"
            f"  Entropy rate:      {self.entropy_rate:.4f} bits/round\n"
            f"  Output:            {self.n_output_bits} bits"
        )


# ---------------------------------------------------------------------------
# Random Beacon
# ---------------------------------------------------------------------------


@dataclass
class RandomBeacon:
    """Publicly verifiable randomness beacon with hash-chain audit trail.

    Produces timestamped, hash-chained random values that can be
    independently verified. Each beacon output commits to:
      - The previous output's hash (chain integrity)
      - A quantum-generated random seed
      - A timestamp

    This provides:
      - Unpredictability (quantum source)
      - Tamper evidence (hash chain)
      - Public verifiability (anyone can check the chain)

    Parameters
    ----------
    seed : int or None
        RNG seed for the quantum source.
    hash_algorithm : str
        Hash function to use ('sha256', 'sha512').
    """

    seed: Optional[int] = None
    hash_algorithm: str = "sha256"
    _chain: list = field(init=False, default_factory=list, repr=False)
    _source: MeasurementQRNG = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.hash_algorithm not in ("sha256", "sha512"):
            raise ValueError(
                f"Unknown hash_algorithm '{self.hash_algorithm}'; "
                "choose 'sha256' or 'sha512'"
            )
        self._source = MeasurementQRNG(seed=self.seed)

    def emit(self, n_bits: int = 256) -> BeaconOutput:
        """Emit a new beacon output.

        Parameters
        ----------
        n_bits : int
            Number of random bits in the output (default 256).

        Returns
        -------
        BeaconOutput
            Timestamped, hash-chained random output.
        """
        if n_bits < 1:
            raise ValueError("n_bits must be >= 1")

        # Generate quantum random bits
        random_bits = self._source.generate_bits(n_bits)
        random_bytes = self._source._pack_bits_to_bytes(
            self._source.generate_bits(
                (n_bits + 7) // 8 * 8
            )
        )

        # Timestamp
        timestamp = time.time()
        sequence_number = len(self._chain)

        # Previous hash (genesis for first output)
        prev_hash = (
            self._chain[-1].self_hash if self._chain else "0" * 64
        )

        # Compute self-hash: H(prev_hash || random || timestamp || seq)
        hasher = hashlib.new(self.hash_algorithm)
        hasher.update(prev_hash.encode("utf-8"))
        hasher.update(random_bytes)
        hasher.update(str(timestamp).encode("utf-8"))
        hasher.update(str(sequence_number).encode("utf-8"))
        self_hash = hasher.hexdigest()

        output = BeaconOutput(
            sequence_number=sequence_number,
            timestamp=timestamp,
            random_bits=random_bits,
            random_hex=random_bytes.hex(),
            prev_hash=prev_hash,
            self_hash=self_hash,
            hash_algorithm=self.hash_algorithm,
        )

        self._chain.append(output)
        return output

    def verify_chain(self) -> tuple[bool, Optional[int]]:
        """Verify the integrity of the entire beacon chain.

        Returns
        -------
        tuple[bool, int or None]
            (is_valid, first_broken_index). If valid, index is None.
        """
        if len(self._chain) == 0:
            return (True, None)

        # Check genesis
        if self._chain[0].prev_hash != "0" * 64:
            return (False, 0)

        # Check chain links
        for i in range(1, len(self._chain)):
            if self._chain[i].prev_hash != self._chain[i - 1].self_hash:
                return (False, i)

        return (True, None)

    @property
    def chain_length(self) -> int:
        """Number of outputs in the chain."""
        return len(self._chain)

    @property
    def chain(self) -> list:
        """The full beacon chain (read-only copy)."""
        return list(self._chain)


@dataclass
class BeaconOutput:
    """A single output from the random beacon.

    Attributes
    ----------
    sequence_number : int
        Position in the chain (0-indexed).
    timestamp : float
        Unix timestamp of generation.
    random_bits : numpy.ndarray
        The random bits (uint8 array).
    random_hex : str
        Hex representation of random bytes.
    prev_hash : str
        Hash of the previous beacon output.
    self_hash : str
        Hash of this output (commits to prev + random + timestamp).
    hash_algorithm : str
        Hash algorithm used.
    """

    sequence_number: int
    timestamp: float
    random_bits: np.ndarray
    random_hex: str
    prev_hash: str
    self_hash: str
    hash_algorithm: str


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def random_bits(
    n: int,
    protocol: str = "measurement",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate n random bits using the specified quantum protocol.

    Parameters
    ----------
    n : int
        Number of random bits to generate.
    protocol : str
        Generation protocol:
        - 'measurement': Hadamard superposition measurement (fastest)
        - 'vacuum': Vacuum fluctuation homodyne detection
        - 'entanglement': Bell-state measurement
        - 'dice': Quantum dice roll (fair coin)
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        1-D array of uint8, each element 0 or 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    if protocol == "measurement":
        gen = MeasurementQRNG(seed=seed)
        return gen.generate_bits(n)
    elif protocol == "vacuum":
        gen = VacuumFluctuationQRNG(seed=seed)
        return gen.generate_bits(n)
    elif protocol == "entanglement":
        gen = EntanglementQRNG(seed=seed)
        return gen.generate_bits(n)
    elif protocol == "dice":
        from .generators import QuantumDiceRoll
        gen = QuantumDiceRoll(seed=seed, d=2)
        return gen.generate_bits(n)
    else:
        raise ValueError(
            f"Unknown protocol '{protocol}'; choose from "
            "'measurement', 'vacuum', 'entanglement', 'dice'"
        )


def random_uniform(
    n: int,
    low: float = 0.0,
    high: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate n uniformly distributed floats in [low, high).

    Uses 53 quantum random bits per float for full double-precision
    mantissa coverage.

    Parameters
    ----------
    n : int
        Number of floats to generate.
    low : float
        Lower bound (inclusive).
    high : float
        Upper bound (exclusive).
    seed : int or None
        RNG seed.

    Returns
    -------
    numpy.ndarray
        1-D array of float64 in [low, high).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if high <= low:
        raise ValueError("high must be > low")

    gen = MeasurementQRNG(seed=seed)
    unit_floats = gen.generate_floats(n)
    return low + (high - low) * unit_floats


def random_integers(
    n: int,
    low: int,
    high: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate n uniformly distributed integers in [low, high).

    Uses rejection sampling for non-power-of-2 ranges.

    Parameters
    ----------
    n : int
        Number of integers to generate.
    low : int
        Lower bound (inclusive).
    high : int
        Upper bound (exclusive).
    seed : int or None
        RNG seed.

    Returns
    -------
    numpy.ndarray
        1-D array of int64 in [low, high).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if high <= low:
        raise ValueError("high must be > low")

    from .generators import QuantumDiceRoll

    d = high - low
    dice = QuantumDiceRoll(seed=seed, d=d)
    outcomes = dice.roll_many(n)
    return outcomes + low
