"""Quantum random number generators based on different quantum phenomena.

Each generator simulates a specific physical quantum process and extracts
randomness from measurement outcomes. The generators do NOT require real
quantum hardware -- they faithfully simulate the underlying quantum mechanics
using state-vector evolution and Born-rule sampling.

Generators provided:
  - MeasurementQRNG: Superposition state measurement (Hadamard + Z-basis)
  - VacuumFluctuationQRNG: Homodyne measurement of vacuum quadratures
  - EntanglementQRNG: Bell-state measurement for correlated random bits
  - QuantumDiceRoll: Uniform integer generation via amplitude encoding

References:
  - Herrero-Collantes & Garcia-Escartin, Rev. Mod. Phys. 89, 015004 (2017)
  - Ma et al., npj Quantum Information 2, 16021 (2016)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Measurement-based QRNG
# ---------------------------------------------------------------------------


@dataclass
class MeasurementQRNG:
    """Random number generator based on measuring quantum superposition states.

    Prepare |+> = H|0>, measure in the computational basis. Each measurement
    yields 0 or 1 with equal probability 1/2, producing one truly random bit.

    For bias removal, an optional random rotation angle can be applied before
    measurement. This defeats any systematic bias in the preparation or
    measurement apparatus.

    Parameters
    ----------
    seed : int or None
        Seed for the internal numpy RNG used to simulate quantum measurements.
        Set for reproducibility in testing; leave None for maximum randomness.
    basis : str
        Measurement preparation strategy:
        - 'hadamard': Apply H gate to |0> (standard, fastest)
        - 'random_rotation': Apply R_y(theta) with random theta each round
          (slower but removes preparation bias)
    batch_size : int
        Number of qubits prepared and measured per batch call. Larger batches
        are more efficient for bulk generation.
    """

    seed: Optional[int] = None
    basis: str = "hadamard"
    batch_size: int = 1024
    _rng: np.random.Generator = field(init=False, repr=False)
    _bits_generated: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if self.basis not in ("hadamard", "random_rotation"):
            raise ValueError(
                f"Unknown basis '{self.basis}'; "
                "choose 'hadamard' or 'random_rotation'"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._rng = np.random.default_rng(self.seed)

    def generate_bits(self, n: int) -> np.ndarray:
        """Generate *n* random bits from quantum measurement simulation.

        Parameters
        ----------
        n : int
            Number of random bits to produce (must be >= 1).

        Returns
        -------
        numpy.ndarray
            1-D array of dtype uint8, each element 0 or 1.
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        bits = np.empty(n, dtype=np.uint8)
        generated = 0

        while generated < n:
            count = min(self.batch_size, n - generated)
            batch = self._generate_batch(count)
            bits[generated : generated + count] = batch
            generated += count

        self._bits_generated += n
        return bits

    def generate_bytes(self, n: int) -> bytes:
        """Generate *n* random bytes (8*n bits).

        Parameters
        ----------
        n : int
            Number of bytes to produce.

        Returns
        -------
        bytes
            Random byte string of length n.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        raw_bits = self.generate_bits(n * 8)
        return self._pack_bits_to_bytes(raw_bits)

    def generate_float(self) -> float:
        """Generate a single uniformly distributed float in [0, 1).

        Uses 53 random bits for full double-precision mantissa coverage.
        """
        bits = self.generate_bits(53)
        # Convert 53 bits to a float in [0, 1)
        value = 0
        for i, b in enumerate(bits):
            value += int(b) << i
        return value / (2**53)

    def generate_floats(self, n: int) -> np.ndarray:
        """Generate *n* uniformly distributed floats in [0, 1).

        Parameters
        ----------
        n : int
            Number of floats to generate.

        Returns
        -------
        numpy.ndarray
            1-D array of float64 in [0, 1).
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        bits = self.generate_bits(n * 53)
        bits_2d = bits.reshape(n, 53)
        powers = 2.0 ** np.arange(53)
        return bits_2d.astype(np.float64) @ powers / (2**53)

    @property
    def total_bits_generated(self) -> int:
        """Total number of bits generated since construction."""
        return self._bits_generated

    # -- internal -----------------------------------------------------------

    def _generate_batch(self, count: int) -> np.ndarray:
        """Simulate preparing and measuring *count* qubits.

        For each qubit:
          1. Start in |0>
          2. Apply gate (H or R_y) to create superposition
          3. Compute Born-rule probabilities
          4. Sample measurement outcome
        """
        if self.basis == "hadamard":
            # |+> = H|0> = (|0> + |1>) / sqrt(2)
            # P(0) = P(1) = 0.5 exactly
            return self._rng.integers(0, 2, size=count, dtype=np.uint8)

        # random_rotation: R_y(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
        thetas = self._rng.uniform(0, 2 * np.pi, size=count)
        prob_one = np.sin(thetas / 2) ** 2
        outcomes = (self._rng.random(count) < prob_one).astype(np.uint8)
        return outcomes

    @staticmethod
    def _pack_bits_to_bytes(bits: np.ndarray) -> bytes:
        """Pack a bit array (0/1 uint8) into bytes, MSB first per byte."""
        n_bytes = len(bits) // 8
        result = bytearray(n_bytes)
        for i in range(n_bytes):
            byte_val = 0
            for j in range(8):
                byte_val |= int(bits[i * 8 + j]) << (7 - j)
            result[i] = byte_val
        return bytes(result)


# ---------------------------------------------------------------------------
# Vacuum Fluctuation QRNG
# ---------------------------------------------------------------------------


@dataclass
class VacuumFluctuationQRNG:
    """Random number generator based on simulated vacuum-state homodyne detection.

    In quantum optics, the vacuum state |0> has zero mean amplitude but non-zero
    quadrature variance due to the Heisenberg uncertainty principle. Homodyne
    detection of the vacuum state yields Gaussian-distributed samples with:
        mean = 0
        variance = 1/(4 * eta)  (shot-noise limited, eta = detector efficiency)

    This generator simulates the homodyne measurement process and extracts
    random bits from the least significant bits of the digitised quadrature
    values.

    Parameters
    ----------
    seed : int or None
        RNG seed for reproducibility.
    detector_efficiency : float
        Simulated homodyne detector efficiency eta in (0, 1].
        eta=1 is perfect detection (shot-noise limited).
        Lower values increase variance (add classical noise on top of vacuum).
    adc_bits : int
        Simulated analog-to-digital converter resolution. Raw quadrature
        values are discretised to 2^adc_bits levels before bit extraction.
    lsb_extract : int
        Number of least-significant bits extracted per sample. Typically 1-4
        for best randomness quality.
    """

    seed: Optional[int] = None
    detector_efficiency: float = 0.95
    adc_bits: int = 16
    lsb_extract: int = 2
    _rng: np.random.Generator = field(init=False, repr=False)
    _samples_taken: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.detector_efficiency <= 1.0:
            raise ValueError("detector_efficiency must be in (0, 1]")
        if self.adc_bits < 4:
            raise ValueError("adc_bits must be >= 4")
        if not 1 <= self.lsb_extract <= self.adc_bits:
            raise ValueError(
                f"lsb_extract must be in [1, {self.adc_bits}]"
            )
        self._rng = np.random.default_rng(self.seed)

    @property
    def vacuum_variance(self) -> float:
        """Quadrature variance of the vacuum state at the detector.

        Shot-noise limited variance = 1/4 for eta=1 (vacuum state in
        natural units where hbar=1).
        """
        return 0.25 / self.detector_efficiency

    def sample_quadratures(self, n: int) -> np.ndarray:
        """Draw *n* homodyne quadrature samples from the simulated vacuum.

        Parameters
        ----------
        n : int
            Number of quadrature samples.

        Returns
        -------
        numpy.ndarray
            Gaussian-distributed float64 samples.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        sigma = math.sqrt(self.vacuum_variance)
        samples = self._rng.normal(0.0, sigma, size=n)
        self._samples_taken += n
        return samples

    def generate_bits(self, n: int) -> np.ndarray:
        """Generate *n* random bits from vacuum quadrature measurements.

        Each quadrature sample is digitised and the *lsb_extract* least
        significant bits are retained.

        Parameters
        ----------
        n : int
            Number of bits to produce.

        Returns
        -------
        numpy.ndarray
            1-D uint8 array, each element 0 or 1.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        # Number of quadrature samples needed
        samples_needed = math.ceil(n / self.lsb_extract)
        raw = self.sample_quadratures(samples_needed)

        # Digitise: map Gaussian to ADC range
        adc_range = 2**self.adc_bits
        # Clip to +/- 4 sigma for the ADC
        sigma = math.sqrt(self.vacuum_variance)
        clip_range = 4 * sigma
        clipped = np.clip(raw, -clip_range, clip_range)
        # Normalise to [0, adc_range)
        normalised = (clipped + clip_range) / (2 * clip_range)
        digitised = (normalised * (adc_range - 1)).astype(np.int64)

        # Extract LSBs
        bits = np.empty(samples_needed * self.lsb_extract, dtype=np.uint8)
        for bit_idx in range(self.lsb_extract):
            bits[bit_idx::self.lsb_extract] = (
                (digitised >> bit_idx) & 1
            ).astype(np.uint8)

        return bits[:n]

    @property
    def total_samples(self) -> int:
        """Total homodyne samples taken since construction."""
        return self._samples_taken


# ---------------------------------------------------------------------------
# Entanglement-based QRNG
# ---------------------------------------------------------------------------


@dataclass
class EntanglementQRNG:
    r"""Random number generator based on Bell-state measurements.

    Prepares the Bell state |Phi+> = (|00> + |11>) / sqrt(2), then measures
    both qubits in the computational basis. Each measurement yields a
    correlated pair of random bits (always 00 or 11 for |Phi+>).

    For randomness, we take the outcome of qubit A as the random bit. The
    correlation with qubit B enables device-independent certification via
    the CHSH inequality.

    For basis rotations (used in CHSH testing):
        E(theta_A, theta_B) = cos(theta_A - theta_B)

    where theta_A, theta_B are the measurement angles.

    Parameters
    ----------
    seed : int or None
        RNG seed for reproducibility.
    bell_state : str
        Which Bell state to prepare: 'phi_plus', 'phi_minus',
        'psi_plus', or 'psi_minus'.
    """

    seed: Optional[int] = None
    bell_state: str = "phi_plus"
    _rng: np.random.Generator = field(init=False, repr=False)
    _pairs_generated: int = field(init=False, default=0, repr=False)

    # Bell state definitions: amplitude vectors in {|00>, |01>, |10>, |11>}
    _BELL_STATES = {
        "phi_plus": np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2),
        "phi_minus": np.array([1, 0, 0, -1], dtype=np.complex128) / np.sqrt(2),
        "psi_plus": np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2),
        "psi_minus": np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2),
    }

    def __post_init__(self) -> None:
        if self.bell_state not in self._BELL_STATES:
            raise ValueError(
                f"Unknown bell_state '{self.bell_state}'; "
                f"choose from {list(self._BELL_STATES.keys())}"
            )
        self._rng = np.random.default_rng(self.seed)

    @property
    def state_vector(self) -> np.ndarray:
        """The 4-element Bell state amplitude vector."""
        return self._BELL_STATES[self.bell_state].copy()

    def measure_pair(
        self,
        theta_a: float = 0.0,
        theta_b: float = 0.0,
    ) -> tuple[int, int]:
        """Measure the Bell state with optional basis rotations.

        Parameters
        ----------
        theta_a : float
            Measurement angle for qubit A (radians from Z axis).
        theta_b : float
            Measurement angle for qubit B (radians from Z axis).

        Returns
        -------
        tuple[int, int]
            Measurement outcomes (a, b), each 0 or 1.
        """
        state = self.state_vector

        # Apply local rotations R_y(theta) = [[cos, -sin], [sin, cos]]
        if theta_a != 0.0 or theta_b != 0.0:
            state = self._apply_local_rotations(state, theta_a, theta_b)

        # Born rule sampling
        probs = np.abs(state) ** 2
        probs /= probs.sum()  # normalise for numerical safety
        outcome = self._rng.choice(4, p=probs)
        self._pairs_generated += 1
        a = (outcome >> 1) & 1
        b = outcome & 1
        return (a, b)

    def measure_pairs(
        self,
        n: int,
        theta_a: float = 0.0,
        theta_b: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Measure *n* Bell pairs and return arrays of outcomes.

        Parameters
        ----------
        n : int
            Number of Bell pairs to measure.
        theta_a : float
            Measurement angle for qubit A.
        theta_b : float
            Measurement angle for qubit B.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (outcomes_a, outcomes_b), each 1-D uint8 array of length n.
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        state = self.state_vector
        if theta_a != 0.0 or theta_b != 0.0:
            state = self._apply_local_rotations(state, theta_a, theta_b)

        probs = np.abs(state) ** 2
        probs /= probs.sum()
        outcomes = self._rng.choice(4, size=n, p=probs)
        self._pairs_generated += n

        outcomes_a = ((outcomes >> 1) & 1).astype(np.uint8)
        outcomes_b = (outcomes & 1).astype(np.uint8)
        return (outcomes_a, outcomes_b)

    def generate_bits(self, n: int) -> np.ndarray:
        """Generate *n* random bits from Bell-state measurements.

        Takes the outcome of qubit A from each Bell pair as the random bit.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        a, _ = self.measure_pairs(n)
        return a

    def compute_chsh_correlation(
        self,
        n_samples: int,
        angles: Optional[tuple[float, float, float, float]] = None,
    ) -> dict:
        """Estimate the CHSH S-value from measurement correlations.

        Uses the standard CHSH angles that maximise quantum violation:
            a0=0, a1=pi/2, b0=pi/4, b1=-pi/4

        Parameters
        ----------
        n_samples : int
            Number of measurement pairs per setting (total = 4 * n_samples).
        angles : tuple of 4 floats, optional
            Custom angles (a0, a1, b0, b1) in radians.

        Returns
        -------
        dict
            Keys: 's_value', 'e_00', 'e_01', 'e_10', 'e_11', 'is_quantum'.
        """
        if n_samples < 10:
            raise ValueError("n_samples must be >= 10")

        if angles is None:
            # Optimal CHSH angles for E(a,b) = cos(a - b)
            # S = |E(a0,b0) - E(a0,b1) + E(a1,b0) + E(a1,b1)| = 2*sqrt(2)
            a0, a1 = 0.0, np.pi / 2
            b0, b1 = np.pi / 4, 3 * np.pi / 4
        else:
            a0, a1, b0, b1 = angles

        correlations = {}
        for (label, ta, tb) in [
            ("e_00", a0, b0),
            ("e_01", a0, b1),
            ("e_10", a1, b0),
            ("e_11", a1, b1),
        ]:
            out_a, out_b = self.measure_pairs(n_samples, ta, tb)
            # Map 0->+1, 1->-1 for correlation calculation
            val_a = 1 - 2 * out_a.astype(np.float64)
            val_b = 1 - 2 * out_b.astype(np.float64)
            correlations[label] = np.mean(val_a * val_b)

        s_value = abs(
            correlations["e_00"]
            - correlations["e_01"]
            + correlations["e_10"]
            + correlations["e_11"]
        )

        return {
            "s_value": float(s_value),
            "e_00": float(correlations["e_00"]),
            "e_01": float(correlations["e_01"]),
            "e_10": float(correlations["e_10"]),
            "e_11": float(correlations["e_11"]),
            "is_quantum": s_value > 2.0,
            "n_samples_per_setting": n_samples,
        }

    @property
    def total_pairs(self) -> int:
        """Total number of Bell pairs measured."""
        return self._pairs_generated

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _apply_local_rotations(
        state: np.ndarray,
        theta_a: float,
        theta_b: float,
    ) -> np.ndarray:
        """Apply R_y(theta_a) x R_y(theta_b) to a 2-qubit state vector.

        R_y(theta) = [[cos(theta/2), -sin(theta/2)],
                       [sin(theta/2),  cos(theta/2)]]
        """
        ca, sa = math.cos(theta_a / 2), math.sin(theta_a / 2)
        cb, sb = math.cos(theta_b / 2), math.sin(theta_b / 2)

        ra = np.array([[ca, -sa], [sa, ca]], dtype=np.complex128)
        rb = np.array([[cb, -sb], [sb, cb]], dtype=np.complex128)

        # Kronecker product for 2-qubit rotation
        rotation = np.kron(ra, rb)
        return rotation @ state


# ---------------------------------------------------------------------------
# Quantum Dice Roll
# ---------------------------------------------------------------------------


@dataclass
class QuantumDiceRoll:
    """Uniform random integer generation on [0, d-1] using quantum amplitudes.

    For a d-sided die where d is a power of 2, prepare a uniform superposition
    over d states using Hadamard gates on log2(d) qubits, then measure.

    For non-power-of-2 d, use rejection sampling: prepare the next larger
    power-of-2 superposition and discard outcomes >= d.

    Parameters
    ----------
    seed : int or None
        RNG seed for reproducibility.
    d : int
        Number of sides on the quantum die (must be >= 2).
    """

    seed: Optional[int] = None
    d: int = 6
    _rng: np.random.Generator = field(init=False, repr=False)
    _rolls: int = field(init=False, default=0, repr=False)
    _rejections: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if self.d < 2:
            raise ValueError("d must be >= 2")
        self._rng = np.random.default_rng(self.seed)
        # Number of qubits needed
        self._n_qubits = math.ceil(math.log2(self.d))
        self._dim = 2**self._n_qubits  # Hilbert space dimension

    @property
    def n_qubits(self) -> int:
        """Number of qubits used for the quantum die."""
        return self._n_qubits

    @property
    def rejection_probability(self) -> float:
        """Probability of rejection per trial (0 for power-of-2 d)."""
        return 1.0 - self.d / self._dim

    def roll(self) -> int:
        """Roll the quantum die once.

        Returns
        -------
        int
            Uniform random integer in [0, d-1].
        """
        while True:
            # Simulate uniform superposition measurement
            outcome = self._rng.integers(0, self._dim)
            if outcome < self.d:
                self._rolls += 1
                return int(outcome)
            self._rejections += 1

    def roll_many(self, n: int) -> np.ndarray:
        """Roll the quantum die *n* times.

        Parameters
        ----------
        n : int
            Number of rolls.

        Returns
        -------
        numpy.ndarray
            1-D int64 array of outcomes in [0, d-1].
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        if self.d == self._dim:
            # Power of 2: no rejection needed, vectorised
            outcomes = self._rng.integers(0, self.d, size=n)
            self._rolls += n
            return outcomes.astype(np.int64)

        # Non-power-of-2: rejection sampling
        results = np.empty(n, dtype=np.int64)
        filled = 0
        while filled < n:
            # Over-sample to reduce iterations
            needed = n - filled
            oversample = int(needed / (self.d / self._dim) * 1.2) + 16
            raw = self._rng.integers(0, self._dim, size=oversample)
            valid = raw[raw < self.d]
            take = min(len(valid), needed)
            results[filled : filled + take] = valid[:take]
            filled += take
            self._rejections += oversample - len(valid)

        self._rolls += n
        return results

    def generate_bits(self, n: int) -> np.ndarray:
        """Generate *n* random bits using the quantum die.

        For a d-sided die, each roll produces floor(log2(d)) usable bits.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        bits_per_roll = int(math.log2(self.d))
        if bits_per_roll < 1:
            raise ValueError("d must be >= 2 to generate bits")

        rolls_needed = math.ceil(n / bits_per_roll)
        outcomes = self.roll_many(rolls_needed)

        bits = np.empty(rolls_needed * bits_per_roll, dtype=np.uint8)
        for bit_idx in range(bits_per_roll):
            bits[bit_idx::bits_per_roll] = (
                (outcomes >> bit_idx) & 1
            ).astype(np.uint8)

        return bits[:n]

    @property
    def total_rolls(self) -> int:
        """Total successful die rolls."""
        return self._rolls

    @property
    def total_rejections(self) -> int:
        """Total rejected samples (non-power-of-2 d only)."""
        return self._rejections

    def expected_rolls_for(self, n: int) -> float:
        """Expected number of raw trials needed for *n* successful outcomes."""
        if self.d == self._dim:
            return float(n)
        return n * self._dim / self.d
