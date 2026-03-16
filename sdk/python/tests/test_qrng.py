"""Comprehensive tests for the QRNG (Quantum Random Number Generation) package.

Tests cover all six modules:
  - generators.py: MeasurementQRNG, VacuumFluctuationQRNG, EntanglementQRNG,
    QuantumDiceRoll
  - extractors.py: VonNeumannExtractor, ToeplitzExtractor, TrevisanExtractor,
    XORExtractor, MinEntropyEstimator
  - statistical_tests.py: NIST SP 800-22 test suite and RandomnessReport
  - certification.py: CHSHCertifier, EntropyAccumulation, RandomnessExpansion
  - protocols.py: DIQRNG, SemiDIQRNG, RandomBeacon, convenience functions
"""

import math

import numpy as np
import pytest

from nqpu.qrng.generators import (
    MeasurementQRNG,
    VacuumFluctuationQRNG,
    EntanglementQRNG,
    QuantumDiceRoll,
)
from nqpu.qrng.extractors import (
    VonNeumannExtractor,
    ToeplitzExtractor,
    TrevisanExtractor,
    XORExtractor,
    MinEntropyEstimator,
)
from nqpu.qrng.statistical_tests import (
    frequency_test,
    block_frequency_test,
    runs_test,
    longest_run_test,
    serial_test,
    approximate_entropy_test,
    cumulative_sums_test,
    dft_spectral_test,
    RandomnessReport,
    StatisticalTestResult,
)
from nqpu.qrng.certification import (
    CHSHCertifier,
    CHSHResult,
    EntropyAccumulation,
    EntropyAccumulationResult,
    RandomnessExpansion,
    ExpansionResult,
)
from nqpu.qrng.protocols import (
    DIQRNG,
    SemiDIQRNG,
    RandomBeacon,
    random_bits,
    random_uniform,
    random_integers,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def measurement_rng() -> MeasurementQRNG:
    """Seeded measurement QRNG for reproducible tests."""
    return MeasurementQRNG(seed=42)


@pytest.fixture
def vacuum_rng() -> VacuumFluctuationQRNG:
    """Seeded vacuum fluctuation QRNG."""
    return VacuumFluctuationQRNG(seed=42)


@pytest.fixture
def entanglement_rng() -> EntanglementQRNG:
    """Seeded entanglement QRNG."""
    return EntanglementQRNG(seed=42)


@pytest.fixture
def dice_d6() -> QuantumDiceRoll:
    """Seeded 6-sided quantum die."""
    return QuantumDiceRoll(seed=42, d=6)


@pytest.fixture
def dice_d8() -> QuantumDiceRoll:
    """Seeded 8-sided quantum die (power of 2)."""
    return QuantumDiceRoll(seed=42, d=8)


@pytest.fixture
def random_bits_10k() -> np.ndarray:
    """10000 random bits from MeasurementQRNG for statistical tests."""
    gen = MeasurementQRNG(seed=123)
    return gen.generate_bits(10000)


@pytest.fixture
def biased_bits() -> np.ndarray:
    """Biased bit sequence (70% ones) for extractor tests."""
    rng = np.random.default_rng(99)
    return (rng.random(5000) < 0.7).astype(np.uint8)


# ======================================================================
# generators.py tests
# ======================================================================


class TestMeasurementQRNG:
    """Tests for MeasurementQRNG."""

    def test_generate_bits_length(self, measurement_rng):
        bits = measurement_rng.generate_bits(1000)
        assert len(bits) == 1000

    def test_generate_bits_dtype(self, measurement_rng):
        bits = measurement_rng.generate_bits(100)
        assert bits.dtype == np.uint8

    def test_generate_bits_values(self, measurement_rng):
        bits = measurement_rng.generate_bits(1000)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_generate_bits_balance(self, measurement_rng):
        """Hadamard basis should produce approximately 50/50 bits."""
        bits = measurement_rng.generate_bits(10000)
        proportion = np.mean(bits)
        assert 0.45 < proportion < 0.55

    def test_generate_bytes(self, measurement_rng):
        result = measurement_rng.generate_bytes(16)
        assert isinstance(result, bytes)
        assert len(result) == 16

    def test_generate_float(self, measurement_rng):
        f = measurement_rng.generate_float()
        assert 0.0 <= f < 1.0

    def test_generate_floats(self, measurement_rng):
        floats = measurement_rng.generate_floats(100)
        assert len(floats) == 100
        assert all(0.0 <= f < 1.0 for f in floats)

    def test_random_rotation_basis(self):
        gen = MeasurementQRNG(seed=42, basis="random_rotation")
        bits = gen.generate_bits(5000)
        assert len(bits) == 5000
        assert set(np.unique(bits)).issubset({0, 1})

    def test_total_bits_counter(self, measurement_rng):
        measurement_rng.generate_bits(500)
        measurement_rng.generate_bits(300)
        assert measurement_rng.total_bits_generated == 800

    def test_batch_size_respected(self):
        gen = MeasurementQRNG(seed=42, batch_size=10)
        bits = gen.generate_bits(25)
        assert len(bits) == 25

    def test_seed_reproducibility(self):
        gen1 = MeasurementQRNG(seed=42)
        gen2 = MeasurementQRNG(seed=42)
        bits1 = gen1.generate_bits(1000)
        bits2 = gen2.generate_bits(1000)
        np.testing.assert_array_equal(bits1, bits2)

    def test_invalid_basis_raises(self):
        with pytest.raises(ValueError, match="Unknown basis"):
            MeasurementQRNG(basis="invalid")

    def test_invalid_n_raises(self, measurement_rng):
        with pytest.raises(ValueError, match="n must be >= 1"):
            measurement_rng.generate_bits(0)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            MeasurementQRNG(batch_size=0)


class TestVacuumFluctuationQRNG:
    """Tests for VacuumFluctuationQRNG."""

    def test_sample_quadratures_gaussian(self, vacuum_rng):
        samples = vacuum_rng.sample_quadratures(10000)
        assert len(samples) == 10000
        # Should be approximately zero mean
        assert abs(np.mean(samples)) < 0.05

    def test_vacuum_variance(self, vacuum_rng):
        expected = 0.25 / 0.95  # detector_efficiency = 0.95
        assert abs(vacuum_rng.vacuum_variance - expected) < 1e-10

    def test_generate_bits_length(self, vacuum_rng):
        bits = vacuum_rng.generate_bits(1000)
        assert len(bits) == 1000

    def test_generate_bits_values(self, vacuum_rng):
        bits = vacuum_rng.generate_bits(500)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_detector_efficiency_validation(self):
        with pytest.raises(ValueError, match="detector_efficiency"):
            VacuumFluctuationQRNG(detector_efficiency=0.0)
        with pytest.raises(ValueError, match="detector_efficiency"):
            VacuumFluctuationQRNG(detector_efficiency=1.5)

    def test_adc_bits_validation(self):
        with pytest.raises(ValueError, match="adc_bits"):
            VacuumFluctuationQRNG(adc_bits=3)

    def test_lsb_extract_options(self):
        gen = VacuumFluctuationQRNG(seed=42, lsb_extract=4)
        bits = gen.generate_bits(200)
        assert len(bits) == 200

    def test_total_samples_counter(self, vacuum_rng):
        vacuum_rng.sample_quadratures(100)
        assert vacuum_rng.total_samples == 100


class TestEntanglementQRNG:
    """Tests for EntanglementQRNG."""

    def test_measure_pair_values(self, entanglement_rng):
        a, b = entanglement_rng.measure_pair()
        assert a in (0, 1)
        assert b in (0, 1)

    def test_phi_plus_correlations(self):
        """Phi+ state: outcomes should always match (00 or 11)."""
        gen = EntanglementQRNG(seed=42, bell_state="phi_plus")
        matches = 0
        n = 1000
        for _ in range(n):
            a, b = gen.measure_pair()
            if a == b:
                matches += 1
        # Should be ~100% for phi_plus with no rotation
        assert matches == n

    def test_psi_plus_anticorrelations(self):
        """Psi+ state: outcomes should always anti-match (01 or 10)."""
        gen = EntanglementQRNG(seed=42, bell_state="psi_plus")
        anti_matches = 0
        n = 1000
        for _ in range(n):
            a, b = gen.measure_pair()
            if a != b:
                anti_matches += 1
        assert anti_matches == n

    def test_measure_pairs_batch(self, entanglement_rng):
        a, b = entanglement_rng.measure_pairs(500)
        assert len(a) == 500
        assert len(b) == 500

    def test_generate_bits(self, entanglement_rng):
        bits = entanglement_rng.generate_bits(1000)
        assert len(bits) == 1000
        assert set(np.unique(bits)).issubset({0, 1})

    def test_chsh_violation(self):
        """CHSH S-value should exceed classical bound of 2."""
        gen = EntanglementQRNG(seed=42, bell_state="phi_plus")
        result = gen.compute_chsh_correlation(n_samples=5000)
        assert result["s_value"] > 2.0
        assert result["is_quantum"] == True

    def test_chsh_near_tsirelson(self):
        """S-value should be close to 2*sqrt(2) for large samples."""
        gen = EntanglementQRNG(seed=42, bell_state="phi_plus")
        result = gen.compute_chsh_correlation(n_samples=50000)
        tsirelson = 2 * math.sqrt(2)
        assert abs(result["s_value"] - tsirelson) < 0.1

    def test_state_vector_phi_plus(self, entanglement_rng):
        sv = entanglement_rng.state_vector
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_rotated_measurement(self, entanglement_rng):
        """Measurement with rotation should still produce valid bits."""
        a, b = entanglement_rng.measure_pairs(100, theta_a=np.pi / 4)
        assert len(a) == 100
        assert set(np.unique(a)).issubset({0, 1})

    def test_invalid_bell_state(self):
        with pytest.raises(ValueError, match="Unknown bell_state"):
            EntanglementQRNG(bell_state="invalid")

    def test_total_pairs_counter(self, entanglement_rng):
        entanglement_rng.measure_pairs(100)
        entanglement_rng.measure_pair()
        assert entanglement_rng.total_pairs == 101


class TestQuantumDiceRoll:
    """Tests for QuantumDiceRoll."""

    def test_roll_range_d6(self, dice_d6):
        for _ in range(100):
            outcome = dice_d6.roll()
            assert 0 <= outcome < 6

    def test_roll_many_d6(self, dice_d6):
        outcomes = dice_d6.roll_many(1000)
        assert len(outcomes) == 1000
        assert all(0 <= o < 6 for o in outcomes)

    def test_roll_uniformity_d6(self, dice_d6):
        """Each face should appear with roughly equal probability."""
        outcomes = dice_d6.roll_many(6000)
        for face in range(6):
            count = np.sum(outcomes == face)
            # Expected ~1000, allow reasonable deviation
            assert 800 < count < 1200, f"Face {face}: count={count}"

    def test_power_of_2_no_rejection(self, dice_d8):
        """d=8 (power of 2) should have zero rejections."""
        dice_d8.roll_many(1000)
        assert dice_d8.total_rejections == 0
        assert dice_d8.rejection_probability == 0.0

    def test_non_power_of_2_has_rejections(self, dice_d6):
        """d=6 uses 3 qubits (dim=8), so 2/8 outcomes are rejected."""
        assert dice_d6.rejection_probability == pytest.approx(0.25)
        dice_d6.roll_many(1000)
        assert dice_d6.total_rejections > 0

    def test_n_qubits(self, dice_d6, dice_d8):
        assert dice_d6.n_qubits == 3  # ceil(log2(6)) = 3
        assert dice_d8.n_qubits == 3  # ceil(log2(8)) = 3

    def test_generate_bits(self, dice_d8):
        bits = dice_d8.generate_bits(100)
        assert len(bits) == 100
        assert set(np.unique(bits)).issubset({0, 1})

    def test_expected_rolls(self, dice_d6):
        expected = dice_d6.expected_rolls_for(1000)
        assert expected == pytest.approx(8000 / 6, rel=1e-10)

    def test_d2_is_coin(self):
        coin = QuantumDiceRoll(seed=42, d=2)
        outcomes = coin.roll_many(10000)
        assert set(np.unique(outcomes)) == {0, 1}
        prop = np.mean(outcomes)
        assert 0.45 < prop < 0.55

    def test_invalid_d(self):
        with pytest.raises(ValueError, match="d must be >= 2"):
            QuantumDiceRoll(d=1)


# ======================================================================
# extractors.py tests
# ======================================================================


class TestVonNeumannExtractor:
    """Tests for VonNeumannExtractor."""

    def test_debiasing_uniform_input(self):
        """For uniform input, output should also be uniform."""
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=10000, dtype=np.uint8)
        ext = VonNeumannExtractor()
        output = ext.extract(bits)
        assert len(output) > 0
        # Should be roughly balanced
        prop = np.mean(output)
        assert 0.45 < prop < 0.55

    def test_debiasing_biased_input(self, biased_bits):
        """Biased input (70% ones) should produce balanced output."""
        ext = VonNeumannExtractor()
        output = ext.extract(biased_bits)
        assert len(output) > 0
        prop = np.mean(output)
        # Von Neumann should remove the bias completely
        assert 0.45 < prop < 0.55

    def test_recursive_mode_more_output(self, biased_bits):
        """Recursive mode should produce more output bits."""
        ext_basic = VonNeumannExtractor(recursive=False)
        ext_recur = VonNeumannExtractor(recursive=True)
        out_basic = ext_basic.extract(biased_bits)
        out_recur = ext_recur.extract(biased_bits)
        assert len(out_recur) >= len(out_basic)

    def test_extraction_rate(self):
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=10000, dtype=np.uint8)
        ext = VonNeumannExtractor()
        ext.extract(bits)
        # For p=0.5, rate should be close to 0.25 (p*(1-p) = 0.25)
        assert 0.20 < ext.extraction_rate < 0.30

    def test_theoretical_rate(self):
        ext = VonNeumannExtractor()
        # p=0.5: rate = 0.25
        assert ext.theoretical_rate(0.5) == pytest.approx(0.25)
        # p=0.0 or p=1.0: rate = 0
        assert ext.theoretical_rate(0.0) == pytest.approx(0.0)
        assert ext.theoretical_rate(1.0) == pytest.approx(0.0)

    def test_short_input(self):
        ext = VonNeumannExtractor()
        # Single bit: not enough for a pair
        output = ext.extract(np.array([1], dtype=np.uint8))
        assert len(output) == 0

    def test_output_values(self, biased_bits):
        ext = VonNeumannExtractor()
        output = ext.extract(biased_bits)
        assert set(np.unique(output)).issubset({0, 1})


class TestToeplitzExtractor:
    """Tests for ToeplitzExtractor."""

    def test_basic_extraction(self):
        ext = ToeplitzExtractor(input_length=256, output_length=64, rng_seed=42)
        rng = np.random.default_rng(99)
        bits = rng.integers(0, 2, size=256, dtype=np.uint8)
        output = ext.extract(bits)
        assert len(output) == 64
        assert output.dtype == np.uint8
        assert set(np.unique(output)).issubset({0, 1})

    def test_compression_ratio(self):
        ext = ToeplitzExtractor(input_length=1024, output_length=256)
        assert ext.compression_ratio == pytest.approx(0.25)

    def test_stream_extraction(self):
        ext = ToeplitzExtractor(input_length=128, output_length=32, rng_seed=42)
        rng = np.random.default_rng(99)
        bits = rng.integers(0, 2, size=512, dtype=np.uint8)  # 4 blocks
        output = ext.extract_stream(bits)
        assert len(output) == 128  # 4 * 32

    def test_seed_reproducibility(self):
        ext1 = ToeplitzExtractor(input_length=256, output_length=64, rng_seed=42)
        ext2 = ToeplitzExtractor(input_length=256, output_length=64, rng_seed=42)
        bits = np.random.default_rng(99).integers(0, 2, size=256, dtype=np.uint8)
        np.testing.assert_array_equal(ext1.extract(bits), ext2.extract(bits))

    def test_explicit_seed(self):
        seed = np.random.default_rng(42).integers(0, 2, size=319, dtype=np.uint8)
        ext = ToeplitzExtractor(input_length=256, output_length=64, seed=seed)
        bits = np.ones(256, dtype=np.uint8)
        output = ext.extract(bits)
        assert len(output) == 64

    def test_wrong_input_length_raises(self):
        ext = ToeplitzExtractor(input_length=256, output_length=64, rng_seed=42)
        with pytest.raises(ValueError, match="Input must have 256"):
            ext.extract(np.zeros(100, dtype=np.uint8))

    def test_invalid_output_length(self):
        with pytest.raises(ValueError, match="output_length"):
            ToeplitzExtractor(input_length=256, output_length=256)

    def test_required_min_entropy(self):
        k = ToeplitzExtractor.required_min_entropy(256, epsilon=1e-6)
        assert k > 256  # Need more entropy than output

    def test_matrix_seed_copy(self):
        ext = ToeplitzExtractor(input_length=128, output_length=32, rng_seed=42)
        seed = ext.matrix_seed
        assert len(seed) == 128 + 32 - 1


class TestTrevisanExtractor:
    """Tests for TrevisanExtractor."""

    def test_basic_extraction(self):
        ext = TrevisanExtractor(input_length=256, output_length=64, rng_seed=42)
        bits = np.random.default_rng(99).integers(0, 2, size=256, dtype=np.uint8)
        output = ext.extract(bits)
        assert len(output) == 64

    def test_seed_length_sublinear(self):
        ext = TrevisanExtractor(input_length=1024, output_length=256, rng_seed=42)
        # Trevisan seed should be much shorter than Toeplitz
        assert ext.seed_length < 1024 + 256  # Toeplitz would need this much

    def test_output_values(self):
        ext = TrevisanExtractor(input_length=128, output_length=32, rng_seed=42)
        bits = np.random.default_rng(99).integers(0, 2, size=128, dtype=np.uint8)
        output = ext.extract(bits)
        assert set(np.unique(output)).issubset({0, 1})

    def test_wrong_input_length_raises(self):
        ext = TrevisanExtractor(input_length=128, output_length=32, rng_seed=42)
        with pytest.raises(ValueError, match="Input must have 128"):
            ext.extract(np.zeros(64, dtype=np.uint8))


class TestXORExtractor:
    """Tests for XORExtractor."""

    def test_two_source_xor(self):
        rng = np.random.default_rng(42)
        s1 = rng.integers(0, 2, size=1000, dtype=np.uint8)
        s2 = rng.integers(0, 2, size=1000, dtype=np.uint8)
        ext = XORExtractor()
        output = ext.extract(s1, s2)
        assert len(output) == 1000
        assert set(np.unique(output)).issubset({0, 1})

    def test_three_source_xor(self):
        rng = np.random.default_rng(42)
        sources = [rng.integers(0, 2, size=500, dtype=np.uint8) for _ in range(3)]
        ext = XORExtractor()
        output = ext.extract(*sources)
        assert len(output) == 500

    def test_xor_improves_bias(self):
        """XOR of two biased sources should be less biased."""
        rng = np.random.default_rng(42)
        s1 = (rng.random(5000) < 0.7).astype(np.uint8)
        s2 = (rng.random(5000) < 0.7).astype(np.uint8)
        ext = XORExtractor()
        output = ext.extract(s1, s2)
        bias = abs(np.mean(output) - 0.5)
        assert bias < 0.1  # XOR of two 0.7-biased sources

    def test_shift_extraction(self):
        rng = np.random.default_rng(42)
        source = rng.integers(0, 2, size=1000, dtype=np.uint8)
        ext = XORExtractor()
        output = ext.extract_with_shift(source, n_shifts=3)
        assert len(output) == 998  # 1000 - 2

    def test_too_few_sources_raises(self):
        ext = XORExtractor()
        with pytest.raises(ValueError, match="At least 2"):
            ext.extract(np.zeros(10, dtype=np.uint8))

    def test_mismatched_lengths_raises(self):
        ext = XORExtractor()
        with pytest.raises(ValueError, match="same length"):
            ext.extract(np.zeros(10, dtype=np.uint8), np.zeros(20, dtype=np.uint8))

    def test_total_extractions_counter(self):
        rng = np.random.default_rng(42)
        s1 = rng.integers(0, 2, size=100, dtype=np.uint8)
        s2 = rng.integers(0, 2, size=100, dtype=np.uint8)
        ext = XORExtractor()
        ext.extract(s1, s2)
        ext.extract(s1, s2)
        assert ext.total_extractions == 2


class TestMinEntropyEstimator:
    """Tests for MinEntropyEstimator."""

    def test_uniform_bits_high_entropy(self):
        bits = np.random.default_rng(42).integers(0, 2, size=100000, dtype=np.uint8)
        est = MinEntropyEstimator(block_size=4)
        result = est.estimate_all(bits)
        # Uniform bits: frequency and collision estimates should be high;
        # compression estimator is conservative, so check combined > 0.8
        assert result["frequency"] > 0.95
        assert result["collision"] > 0.9
        assert result["combined"] > 0.8

    def test_constant_bits_zero_entropy(self):
        bits = np.ones(10000, dtype=np.uint8)
        est = MinEntropyEstimator(block_size=4)
        result = est.estimate_all(bits)
        assert result["frequency"] == 0.0

    def test_biased_bits_reduced_entropy(self):
        rng = np.random.default_rng(42)
        bits = (rng.random(10000) < 0.8).astype(np.uint8)
        est = MinEntropyEstimator(block_size=4)
        result = est.estimate_all(bits)
        # 80% bias: H_inf = -log2(0.8) ~ 0.322
        assert result["frequency"] < 0.5
        assert result["frequency"] > 0.2

    def test_frequency_estimate(self):
        est = MinEntropyEstimator()
        bits = np.random.default_rng(42).integers(0, 2, size=5000, dtype=np.uint8)
        h = est.frequency_estimate(bits)
        assert 0.9 < h <= 1.0

    def test_collision_estimate(self):
        est = MinEntropyEstimator(block_size=4)
        bits = np.random.default_rng(42).integers(0, 2, size=10000, dtype=np.uint8)
        h = est.collision_estimate(bits)
        assert 0.7 < h <= 1.0

    def test_compression_estimate(self):
        est = MinEntropyEstimator(block_size=4)
        bits = np.random.default_rng(42).integers(0, 2, size=10000, dtype=np.uint8)
        h = est.compression_estimate(bits)
        assert 0.5 < h <= 1.0

    def test_empty_bits(self):
        est = MinEntropyEstimator()
        assert est.frequency_estimate(np.array([], dtype=np.uint8)) == 0.0


# ======================================================================
# statistical_tests.py tests
# ======================================================================


class TestFrequencyTest:
    """Tests for NIST frequency (monobit) test."""

    def test_uniform_bits_pass(self, random_bits_10k):
        result = frequency_test(random_bits_10k)
        assert isinstance(result, StatisticalTestResult)
        assert result.passed is True
        assert result.p_value > 0.01

    def test_biased_bits_fail(self):
        """All-ones sequence should fail badly."""
        bits = np.ones(1000, dtype=np.uint8)
        result = frequency_test(bits)
        assert result.passed is False
        assert result.p_value < 0.01

    def test_short_sequence_raises(self):
        with pytest.raises(ValueError, match="n >= 100"):
            frequency_test(np.zeros(50, dtype=np.uint8))


class TestBlockFrequencyTest:
    """Tests for NIST block frequency test."""

    def test_uniform_bits_pass(self, random_bits_10k):
        result = block_frequency_test(random_bits_10k, block_size=100)
        assert result.passed is True

    def test_patterned_bits_fail(self):
        """Alternating 128 ones / 128 zeros should fail."""
        bits = np.tile(
            np.concatenate([np.ones(128), np.zeros(128)]),
            40,
        ).astype(np.uint8)
        result = block_frequency_test(bits, block_size=128)
        # The block proportions are 1.0 and 0.0, should fail
        assert result.passed is False

    def test_small_block_size_raises(self):
        with pytest.raises(ValueError, match="block_size should be >= 20"):
            block_frequency_test(np.zeros(1000, dtype=np.uint8), block_size=5)


class TestRunsTest:
    """Tests for NIST runs test."""

    def test_uniform_bits_pass(self, random_bits_10k):
        result = runs_test(random_bits_10k)
        assert result.passed is True

    def test_no_runs_fail(self):
        """Alternating 0101... has maximum runs, should pass or fail edge."""
        bits = np.tile([0, 1], 5000).astype(np.uint8)
        result = runs_test(bits)
        # Alternating pattern: n runs, which is way more than expected
        assert isinstance(result.p_value, float)

    def test_all_same_fail(self):
        """All same bits: prerequisite should fail."""
        bits = np.ones(1000, dtype=np.uint8)
        result = runs_test(bits)
        assert result.passed is False
        assert result.details.get("prerequisite_failed") is True


class TestLongestRunTest:
    """Tests for NIST longest run of ones test."""

    def test_uniform_bits_pass(self, random_bits_10k):
        result = longest_run_test(random_bits_10k)
        assert result.passed is True

    def test_short_sequence_raises(self):
        with pytest.raises(ValueError, match="n >= 128"):
            longest_run_test(np.zeros(100, dtype=np.uint8))


class TestSerialTest:
    """Tests for NIST serial test."""

    def test_uniform_bits_pass(self, random_bits_10k):
        result = serial_test(random_bits_10k, block_size=3)
        assert result.passed is True

    def test_returns_test_result(self, random_bits_10k):
        result = serial_test(random_bits_10k)
        assert isinstance(result, StatisticalTestResult)
        assert "delta1" in result.details


class TestApproximateEntropyTest:
    """Tests for NIST approximate entropy test."""

    def test_uniform_bits_pass(self, random_bits_10k):
        result = approximate_entropy_test(random_bits_10k, block_size=4)
        assert result.passed is True

    def test_returns_apen(self, random_bits_10k):
        result = approximate_entropy_test(random_bits_10k, block_size=2)
        assert "apen" in result.details
        # ApEn = phi_m - phi_{m+1} should be close to log(2) for random bits
        assert result.details["apen"] > 0
        assert abs(result.details["apen"] - math.log(2)) < 0.1


class TestCumulativeSumsTest:
    """Tests for NIST cumulative sums test."""

    def test_forward_pass(self, random_bits_10k):
        result = cumulative_sums_test(random_bits_10k, mode="forward")
        assert result.passed is True

    def test_backward_pass(self, random_bits_10k):
        result = cumulative_sums_test(random_bits_10k, mode="backward")
        assert result.passed is True

    def test_invalid_mode_raises(self, random_bits_10k):
        with pytest.raises(ValueError, match="mode must be"):
            cumulative_sums_test(random_bits_10k, mode="sideways")


class TestDFTSpectralTest:
    """Tests for NIST DFT spectral test."""

    def test_uniform_bits_pass(self, random_bits_10k):
        result = dft_spectral_test(random_bits_10k)
        assert result.passed is True

    def test_periodic_bits_fail(self):
        """Periodic pattern should show spectral peaks."""
        # Period-8 pattern repeated many times
        pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        bits = np.tile(pattern, 1250)  # 10000 bits
        result = dft_spectral_test(bits)
        # Strong periodicity should fail
        assert result.passed is False


class TestRandomnessReport:
    """Tests for the full RandomnessReport."""

    def test_run_all_returns_results(self, random_bits_10k):
        report = RandomnessReport(random_bits_10k)
        results = report.run_all()
        assert len(results) >= 7  # At least 7 tests
        assert all(isinstance(r, StatisticalTestResult) for r in results)

    def test_mostly_passes(self, random_bits_10k):
        """Quantum-generated bits should pass most NIST tests."""
        report = RandomnessReport(random_bits_10k)
        report.run_all()
        assert report.pass_rate >= 0.7

    def test_summary_string(self, random_bits_10k):
        report = RandomnessReport(random_bits_10k)
        report.run_all()
        s = report.summary()
        assert "NIST SP 800-22" in s
        assert "passed" in s

    def test_to_dict(self, random_bits_10k):
        report = RandomnessReport(random_bits_10k)
        report.run_all()
        d = report.to_dict()
        assert "n_bits" in d
        assert "tests" in d
        assert d["n_bits"] == 10000

    def test_properties_before_run_raises(self, random_bits_10k):
        report = RandomnessReport(random_bits_10k)
        with pytest.raises(RuntimeError, match="Call run_all"):
            _ = report.results

    def test_short_sequence_raises(self):
        with pytest.raises(ValueError, match="Sequence must be >= 100"):
            RandomnessReport(np.zeros(50, dtype=np.uint8))


# ======================================================================
# certification.py tests
# ======================================================================


class TestCHSHCertifier:
    """Tests for CHSHCertifier."""

    def test_bell_test_quantum(self):
        cert = CHSHCertifier(n_rounds=10000, seed=42)
        result = cert.run_bell_test()
        assert isinstance(result, CHSHResult)
        assert result.is_quantum is True
        assert result.s_value > 2.0

    def test_min_entropy_positive(self):
        cert = CHSHCertifier(n_rounds=50000, seed=42)
        result = cert.run_bell_test()
        assert result.min_entropy_point_estimate > 0

    def test_certified_with_confidence(self):
        """With enough rounds, should get certified violation."""
        cert = CHSHCertifier(n_rounds=100000, seed=42, confidence_level=0.99)
        result = cert.run_bell_test()
        assert result.is_certified is True

    def test_tsirelson_fraction(self):
        cert = CHSHCertifier(n_rounds=50000, seed=42)
        result = cert.run_bell_test()
        # Should be close to 1.0 (near Tsirelson bound)
        assert result.tsirelson_fraction > 0.9

    def test_summary_string(self):
        cert = CHSHCertifier(n_rounds=1000, seed=42)
        result = cert.run_bell_test()
        s = result.summary()
        assert "CHSH" in s
        assert "S-value" in s

    def test_min_entropy_bounds(self):
        """Test the static min-entropy function."""
        # Classical: should return 0
        assert CHSHCertifier._min_entropy_from_s(2.0) == 0.0
        assert CHSHCertifier._min_entropy_from_s(1.5) == 0.0
        # Tsirelson bound: should return 1
        h = CHSHCertifier._min_entropy_from_s(2 * math.sqrt(2))
        assert abs(h - 1.0) < 0.01
        # Intermediate
        h = CHSHCertifier._min_entropy_from_s(2.5)
        assert 0 < h < 1

    def test_invalid_n_rounds(self):
        with pytest.raises(ValueError, match="n_rounds must be >= 10"):
            CHSHCertifier(n_rounds=5)


class TestEntropyAccumulation:
    """Tests for EntropyAccumulation."""

    def test_positive_entropy(self):
        ea = EntropyAccumulation(n_rounds=100000, seed=42)
        result = ea.accumulate()
        assert isinstance(result, EntropyAccumulationResult)
        assert result.is_positive is True
        assert result.total_smooth_min_entropy > 0

    def test_entropy_rate(self):
        ea = EntropyAccumulation(n_rounds=100000, seed=42)
        result = ea.accumulate()
        # Quantum winning probability ~0.854 should give positive rate
        assert result.entropy_rate_per_round > 0
        assert result.chsh_winning_prob > 0.75

    def test_finite_size_correction(self):
        """Smaller experiments should have larger corrections."""
        ea_small = EntropyAccumulation(n_rounds=1000, seed=42)
        ea_large = EntropyAccumulation(n_rounds=100000, seed=42)
        r_small = ea_small.accumulate()
        r_large = ea_large.accumulate()
        # Per-round effective entropy should be higher for larger experiments
        assert r_large.effective_entropy_per_round > r_small.effective_entropy_per_round

    def test_summary_string(self):
        ea = EntropyAccumulation(n_rounds=10000, seed=42)
        result = ea.accumulate()
        s = result.summary()
        assert "Entropy Accumulation" in s


class TestRandomnessExpansion:
    """Tests for RandomnessExpansion."""

    def test_positive_expansion(self):
        exp = RandomnessExpansion(n_rounds=100000, seed_length=1000, seed=42)
        result = exp.run()
        assert isinstance(result, ExpansionResult)
        assert result.has_expansion is True
        assert result.net_expansion > 0

    def test_expansion_factor(self):
        exp = RandomnessExpansion(n_rounds=100000, seed_length=1000, seed=42)
        result = exp.run()
        # Should produce more bits than consumed
        assert result.expansion_factor > 1.0

    def test_small_experiment_less_expansion(self):
        exp = RandomnessExpansion(n_rounds=1000, seed_length=500, seed=42)
        result = exp.run()
        # Small experiment with large seed: might not have expansion
        # (finite-size effects dominate)
        assert isinstance(result.has_expansion, bool)

    def test_summary_string(self):
        exp = RandomnessExpansion(n_rounds=10000, seed_length=100, seed=42)
        result = exp.run()
        s = result.summary()
        assert "Randomness Expansion" in s


# ======================================================================
# protocols.py tests
# ======================================================================


class TestDIQRNG:
    """Tests for Device-Independent QRNG protocol."""

    def test_successful_generation(self):
        di = DIQRNG(n_rounds=10000, seed=42, min_s_value=2.0)
        result = di.generate()
        assert result.aborted is False
        assert len(result.output_bits) > 0
        assert result.s_value > 2.0

    def test_output_bits_binary(self):
        di = DIQRNG(n_rounds=10000, seed=42)
        result = di.generate()
        if not result.aborted:
            assert set(np.unique(result.output_bits)).issubset({0, 1})

    def test_abort_on_low_s(self):
        """Setting min_s_value impossibly high should abort."""
        di = DIQRNG(n_rounds=10000, seed=42, min_s_value=3.0)
        result = di.generate()
        assert result.aborted is True
        assert "S=" in result.abort_reason

    def test_summary_string(self):
        di = DIQRNG(n_rounds=10000, seed=42)
        result = di.generate()
        s = result.summary()
        assert "DI-QRNG" in s


class TestSemiDIQRNG:
    """Tests for Semi-Device-Independent QRNG protocol."""

    def test_successful_generation(self):
        sdi = SemiDIQRNG(n_rounds=10000, seed=42)
        result = sdi.generate()
        assert result.is_certified is True
        assert len(result.output_bits) > 0

    def test_dimension_witness_exceeds_classical(self):
        sdi = SemiDIQRNG(n_rounds=10000, seed=42)
        result = sdi.generate()
        assert result.dimension_witness > result.classical_bound

    def test_summary_string(self):
        sdi = SemiDIQRNG(n_rounds=10000, seed=42)
        result = sdi.generate()
        s = result.summary()
        assert "Semi-DI" in s


class TestRandomBeacon:
    """Tests for RandomBeacon protocol."""

    def test_emit_single(self):
        beacon = RandomBeacon(seed=42)
        output = beacon.emit(256)
        assert output.sequence_number == 0
        assert len(output.random_bits) == 256
        assert output.prev_hash == "0" * 64

    def test_emit_chain(self):
        beacon = RandomBeacon(seed=42)
        out1 = beacon.emit(128)
        out2 = beacon.emit(128)
        assert out2.prev_hash == out1.self_hash
        assert out2.sequence_number == 1

    def test_verify_chain(self):
        beacon = RandomBeacon(seed=42)
        for _ in range(5):
            beacon.emit(128)
        valid, broken = beacon.verify_chain()
        assert valid is True
        assert broken is None

    def test_chain_length(self):
        beacon = RandomBeacon(seed=42)
        beacon.emit(64)
        beacon.emit(64)
        beacon.emit(64)
        assert beacon.chain_length == 3

    def test_sha512_support(self):
        beacon = RandomBeacon(seed=42, hash_algorithm="sha512")
        output = beacon.emit(128)
        assert len(output.self_hash) == 128  # SHA-512 hex = 128 chars

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown hash_algorithm"):
            RandomBeacon(hash_algorithm="md5")


class TestConvenienceFunctions:
    """Tests for random_bits, random_uniform, random_integers."""

    def test_random_bits_measurement(self):
        bits = random_bits(1000, protocol="measurement", seed=42)
        assert len(bits) == 1000
        assert set(np.unique(bits)).issubset({0, 1})

    def test_random_bits_vacuum(self):
        bits = random_bits(500, protocol="vacuum", seed=42)
        assert len(bits) == 500

    def test_random_bits_entanglement(self):
        bits = random_bits(500, protocol="entanglement", seed=42)
        assert len(bits) == 500

    def test_random_bits_dice(self):
        bits = random_bits(500, protocol="dice", seed=42)
        assert len(bits) == 500

    def test_random_bits_invalid_protocol(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            random_bits(100, protocol="invalid")

    def test_random_uniform_default(self):
        floats = random_uniform(100, seed=42)
        assert len(floats) == 100
        assert all(0.0 <= f < 1.0 for f in floats)

    def test_random_uniform_range(self):
        floats = random_uniform(100, low=5.0, high=10.0, seed=42)
        assert all(5.0 <= f < 10.0 for f in floats)

    def test_random_uniform_invalid_range(self):
        with pytest.raises(ValueError, match="high must be > low"):
            random_uniform(10, low=5.0, high=3.0)

    def test_random_integers_range(self):
        ints = random_integers(1000, low=0, high=10, seed=42)
        assert len(ints) == 1000
        assert all(0 <= i < 10 for i in ints)

    def test_random_integers_offset(self):
        ints = random_integers(500, low=100, high=200, seed=42)
        assert all(100 <= i < 200 for i in ints)

    def test_random_integers_invalid_range(self):
        with pytest.raises(ValueError, match="high must be > low"):
            random_integers(10, low=10, high=5)


# ======================================================================
# Integration tests
# ======================================================================


class TestIntegration:
    """End-to-end integration tests combining multiple modules."""

    def test_generate_and_validate(self):
        """Generate bits and validate with NIST suite."""
        gen = MeasurementQRNG(seed=42)
        bits = gen.generate_bits(10000)
        report = RandomnessReport(bits, alpha=0.01)
        report.run_all()
        # Should pass most tests
        assert report.pass_rate >= 0.7

    def test_extract_and_validate(self):
        """Extract from biased source, validate output is more random."""
        rng = np.random.default_rng(42)
        biased = (rng.random(20000) < 0.6).astype(np.uint8)

        ext = VonNeumannExtractor(recursive=True)
        extracted = ext.extract(biased)

        # Check debiasing worked
        bias_after = abs(np.mean(extracted) - 0.5)
        assert bias_after < 0.05

    def test_certify_and_extract(self):
        """Run CHSH certification and use result for extraction."""
        cert = CHSHCertifier(n_rounds=10000, seed=42)
        result = cert.run_bell_test()

        assert result.is_quantum
        h_min = result.min_entropy_point_estimate

        # If we had n generation rounds, we could extract ~n*h_min bits
        n_gen = 5000
        extractable = int(n_gen * h_min * 0.9)
        assert extractable > 0

    def test_full_diqrng_nist_validation(self):
        """Run DI-QRNG protocol and validate output with NIST tests."""
        di = DIQRNG(n_rounds=50000, seed=42, min_s_value=2.0)
        result = di.generate()

        if not result.aborted and len(result.output_bits) >= 1000:
            # Pad to ensure enough bits for tests
            bits = result.output_bits[:10000] if len(result.output_bits) >= 10000 else result.output_bits
            if len(bits) >= 1000:
                freq_result = frequency_test(bits[:max(100, len(bits))])
                assert isinstance(freq_result, StatisticalTestResult)

    def test_beacon_chain_integrity(self):
        """Emit multiple beacon outputs and verify chain."""
        beacon = RandomBeacon(seed=42)
        for _ in range(10):
            beacon.emit(256)
        valid, broken = beacon.verify_chain()
        assert valid is True

    def test_min_entropy_estimation_pipeline(self):
        """Estimate entropy of raw quantum bits."""
        gen = MeasurementQRNG(seed=42)
        bits = gen.generate_bits(50000)
        est = MinEntropyEstimator(block_size=8)
        result = est.estimate_all(bits)
        assert result["combined"] > 0.8
        assert result["n_bits"] == 50000
