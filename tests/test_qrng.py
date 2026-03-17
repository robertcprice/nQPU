"""Comprehensive tests for nqpu.qrng -- quantum random number generation,
extractors, statistical tests, and certification.
"""
from __future__ import annotations

import numpy as np
import pytest

from nqpu.qrng import (
    # Generators
    MeasurementQRNG,
    VacuumFluctuationQRNG,
    EntanglementQRNG,
    QuantumDiceRoll,
    # Extractors
    VonNeumannExtractor,
    ToeplitzExtractor,
    TrevisanExtractor,
    XORExtractor,
    MinEntropyEstimator,
    # Statistical tests
    frequency_test,
    runs_test,
    cumulative_sums_test,
    dft_spectral_test,
    RandomnessReport,
    StatisticalTestResult,
    # Certification
    CHSHCertifier,
    EntropyAccumulation,
    RandomnessExpansion,
    # Protocols
    random_bits,
    random_uniform,
    random_integers,
    RandomBeacon,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def measurement_qrng():
    """Deterministic measurement-based QRNG."""
    return MeasurementQRNG(seed=42, basis="hadamard")


@pytest.fixture
def fair_bits():
    """1024 fair random bits from a deterministic source."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=1024, dtype=np.uint8)


@pytest.fixture
def biased_bits():
    """1024 biased bits (p=0.7 for 1)."""
    rng = np.random.default_rng(42)
    return (rng.random(1024) < 0.7).astype(np.uint8)


# =====================================================================
# MeasurementQRNG tests
# =====================================================================

class TestMeasurementQRNG:

    def test_generate_bits_length(self, measurement_qrng):
        bits = measurement_qrng.generate_bits(100)
        assert len(bits) == 100
        assert bits.dtype == np.uint8

    def test_generate_bits_values(self, measurement_qrng):
        bits = measurement_qrng.generate_bits(500)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_generate_bytes(self, measurement_qrng):
        b = measurement_qrng.generate_bytes(4)
        assert isinstance(b, bytes)
        assert len(b) == 4

    def test_generate_float_range(self, measurement_qrng):
        f = measurement_qrng.generate_float()
        assert 0.0 <= f < 1.0

    def test_generate_floats(self, measurement_qrng):
        floats = measurement_qrng.generate_floats(10)
        assert len(floats) == 10
        assert all(0.0 <= f < 1.0 for f in floats)

    def test_total_bits_tracked(self):
        qrng = MeasurementQRNG(seed=42)
        qrng.generate_bits(100)
        qrng.generate_bits(50)
        assert qrng.total_bits_generated == 150

    def test_invalid_basis_raises(self):
        with pytest.raises(ValueError, match="Unknown basis"):
            MeasurementQRNG(basis="invalid_basis")

    def test_random_rotation_basis(self):
        qrng = MeasurementQRNG(seed=42, basis="random_rotation")
        bits = qrng.generate_bits(200)
        assert len(bits) == 200


# =====================================================================
# VacuumFluctuationQRNG tests
# =====================================================================

class TestVacuumFluctuationQRNG:

    def test_vacuum_variance(self):
        gen = VacuumFluctuationQRNG(seed=42, detector_efficiency=1.0)
        assert gen.vacuum_variance == pytest.approx(0.25, abs=1e-10)

    def test_generate_bits(self):
        gen = VacuumFluctuationQRNG(seed=42)
        bits = gen.generate_bits(100)
        assert len(bits) == 100
        assert set(np.unique(bits)).issubset({0, 1})

    def test_sample_quadratures_gaussian(self):
        gen = VacuumFluctuationQRNG(seed=42, detector_efficiency=1.0)
        samples = gen.sample_quadratures(10000)
        # Mean should be close to 0
        assert abs(np.mean(samples)) < 0.05
        # Variance should be close to 0.25
        assert abs(np.var(samples) - 0.25) < 0.05

    def test_invalid_efficiency_raises(self):
        with pytest.raises(ValueError, match="detector_efficiency"):
            VacuumFluctuationQRNG(detector_efficiency=0.0)


# =====================================================================
# EntanglementQRNG tests
# =====================================================================

class TestEntanglementQRNG:

    def test_phi_plus_correlations(self):
        """Phi+ Bell state should produce perfectly correlated pairs."""
        gen = EntanglementQRNG(seed=42, bell_state="phi_plus")
        a, b = gen.measure_pairs(1000)
        # For phi+, outcomes should always match (a == b)
        agreement = np.mean(a == b)
        assert agreement == pytest.approx(1.0, abs=0.01)

    def test_psi_plus_anticorrelations(self):
        """Psi+ Bell state should produce anti-correlated pairs."""
        gen = EntanglementQRNG(seed=42, bell_state="psi_plus")
        a, b = gen.measure_pairs(1000)
        # For psi+, outcomes should always differ (a != b)
        disagreement = np.mean(a != b)
        assert disagreement == pytest.approx(1.0, abs=0.01)

    def test_generate_bits(self):
        gen = EntanglementQRNG(seed=42)
        bits = gen.generate_bits(100)
        assert len(bits) == 100

    def test_chsh_violation(self):
        gen = EntanglementQRNG(seed=42, bell_state="phi_plus")
        result = gen.compute_chsh_correlation(n_samples=500)
        # S value should exceed classical bound of 2
        assert result["s_value"] > 2.0
        assert result["is_quantum"]

    def test_invalid_bell_state_raises(self):
        with pytest.raises(ValueError, match="Unknown bell_state"):
            EntanglementQRNG(bell_state="invalid")


# =====================================================================
# QuantumDiceRoll tests
# =====================================================================

class TestQuantumDiceRoll:

    def test_power_of_two_die(self):
        dice = QuantumDiceRoll(seed=42, d=8)
        outcomes = dice.roll_many(100)
        assert all(0 <= o < 8 for o in outcomes)
        assert dice.rejection_probability == 0.0

    def test_non_power_of_two_die(self):
        dice = QuantumDiceRoll(seed=42, d=6)
        outcomes = dice.roll_many(100)
        assert all(0 <= o < 6 for o in outcomes)
        assert dice.rejection_probability > 0.0

    def test_single_roll(self):
        dice = QuantumDiceRoll(seed=42, d=6)
        outcome = dice.roll()
        assert 0 <= outcome < 6

    @pytest.mark.parametrize("d", [2, 3, 6, 8, 10, 16])
    def test_uniform_range(self, d):
        dice = QuantumDiceRoll(seed=42, d=d)
        outcomes = dice.roll_many(200)
        assert outcomes.min() >= 0
        assert outcomes.max() < d

    def test_invalid_d_raises(self):
        with pytest.raises(ValueError, match="d must be >= 2"):
            QuantumDiceRoll(d=1)


# =====================================================================
# Von Neumann Extractor tests
# =====================================================================

class TestVonNeumannExtractor:

    def test_basic_extraction(self, fair_bits):
        ext = VonNeumannExtractor()
        output = ext.extract(fair_bits)
        assert len(output) > 0
        assert set(np.unique(output)).issubset({0, 1})

    def test_extraction_rate(self, fair_bits):
        ext = VonNeumannExtractor()
        output = ext.extract(fair_bits)
        # For fair bits, extraction rate should be around 0.25
        rate = len(output) / len(fair_bits)
        assert 0.1 < rate < 0.5

    def test_recursive_improves_output(self, fair_bits):
        ext_basic = VonNeumannExtractor(recursive=False)
        ext_recursive = VonNeumannExtractor(recursive=True)
        out_basic = ext_basic.extract(fair_bits)
        out_recursive = ext_recursive.extract(fair_bits)
        assert len(out_recursive) >= len(out_basic)

    def test_theoretical_rate(self):
        ext = VonNeumannExtractor()
        # At p=0.5, theoretical rate = 0.5 * 0.5 = 0.25
        assert ext.theoretical_rate(0.5) == pytest.approx(0.25, abs=1e-10)

    def test_short_input(self):
        ext = VonNeumannExtractor()
        output = ext.extract(np.array([1], dtype=np.uint8))
        assert len(output) == 0


# =====================================================================
# Toeplitz Extractor tests
# =====================================================================

class TestToeplitzExtractor:

    def test_extraction_length(self):
        ext = ToeplitzExtractor(input_length=128, output_length=32, rng_seed=42)
        bits = np.random.default_rng(42).integers(0, 2, size=128, dtype=np.uint8)
        output = ext.extract(bits)
        assert len(output) == 32

    def test_output_is_binary(self):
        ext = ToeplitzExtractor(input_length=64, output_length=16, rng_seed=42)
        bits = np.ones(64, dtype=np.uint8)
        output = ext.extract(bits)
        assert set(np.unique(output)).issubset({0, 1})

    def test_compression_ratio(self):
        ext = ToeplitzExtractor(input_length=1024, output_length=256)
        assert ext.compression_ratio == pytest.approx(0.25, abs=1e-10)

    def test_wrong_input_length_raises(self):
        ext = ToeplitzExtractor(input_length=64, output_length=16, rng_seed=42)
        with pytest.raises(ValueError, match="Input must have"):
            ext.extract(np.zeros(32, dtype=np.uint8))

    def test_stream_extraction(self):
        ext = ToeplitzExtractor(input_length=64, output_length=16, rng_seed=42)
        bits = np.random.default_rng(42).integers(0, 2, size=128, dtype=np.uint8)
        output = ext.extract_stream(bits)
        assert len(output) == 32  # 2 blocks * 16


# =====================================================================
# Trevisan Extractor tests
# =====================================================================

class TestTrevisanExtractor:

    def test_extraction(self):
        ext = TrevisanExtractor(input_length=256, output_length=64, rng_seed=42)
        bits = np.random.default_rng(42).integers(0, 2, size=256, dtype=np.uint8)
        output = ext.extract(bits)
        assert len(output) == 64
        assert set(np.unique(output)).issubset({0, 1})


# =====================================================================
# XOR Extractor tests
# =====================================================================

class TestXORExtractor:

    def test_two_sources(self):
        rng = np.random.default_rng(42)
        s1 = rng.integers(0, 2, size=100, dtype=np.uint8)
        s2 = rng.integers(0, 2, size=100, dtype=np.uint8)
        ext = XORExtractor()
        output = ext.extract(s1, s2)
        assert len(output) == 100
        np.testing.assert_array_equal(output, np.bitwise_xor(s1, s2))

    def test_single_source_raises(self):
        ext = XORExtractor()
        with pytest.raises(ValueError, match="At least 2"):
            ext.extract(np.zeros(10, dtype=np.uint8))

    def test_extract_with_shift(self):
        rng = np.random.default_rng(42)
        source = rng.integers(0, 2, size=100, dtype=np.uint8)
        ext = XORExtractor()
        output = ext.extract_with_shift(source, n_shifts=3)
        assert len(output) == 98  # 100 - (3-1)


# =====================================================================
# MinEntropyEstimator tests
# =====================================================================

class TestMinEntropyEstimator:

    def test_fair_bits_high_entropy(self, fair_bits):
        est = MinEntropyEstimator(block_size=4)
        result = est.estimate_all(fair_bits)
        assert result["frequency"] > 0.9

    def test_biased_bits_lower_entropy(self, biased_bits):
        est = MinEntropyEstimator(block_size=4)
        freq = est.frequency_estimate(biased_bits)
        # Biased bits should have lower min-entropy
        assert freq < 0.8

    def test_combined_is_minimum(self, fair_bits):
        est = MinEntropyEstimator(block_size=4)
        result = est.estimate_all(fair_bits)
        assert result["combined"] == min(
            result["frequency"], result["collision"], result["compression"]
        )


# =====================================================================
# Statistical Tests
# =====================================================================

class TestStatisticalTests:

    def test_frequency_test_fair_bits(self, fair_bits):
        result = frequency_test(fair_bits)
        assert isinstance(result, StatisticalTestResult)
        assert result.name == "Frequency (Monobit)"
        # Fair bits should pass
        assert result.passed

    def test_frequency_test_biased_fails(self):
        """Highly biased bits should fail the frequency test."""
        biased = np.ones(1000, dtype=np.uint8)
        biased[:50] = 0
        result = frequency_test(biased)
        assert not result.passed

    def test_runs_test(self, fair_bits):
        result = runs_test(fair_bits)
        assert isinstance(result, StatisticalTestResult)

    def test_cumulative_sums_test(self, fair_bits):
        result = cumulative_sums_test(fair_bits, mode="forward")
        assert isinstance(result, StatisticalTestResult)

    def test_dft_spectral_test(self, fair_bits):
        result = dft_spectral_test(fair_bits)
        assert isinstance(result, StatisticalTestResult)

    def test_short_sequence_raises(self):
        with pytest.raises(ValueError, match="n >= 100"):
            frequency_test(np.array([0, 1, 0], dtype=np.uint8))


# =====================================================================
# RandomnessReport tests
# =====================================================================

class TestRandomnessReport:

    def test_full_report(self, fair_bits):
        report = RandomnessReport(fair_bits)
        results = report.run_all()
        assert len(results) > 0
        assert report.n_tests > 0

    def test_summary_string(self, fair_bits):
        report = RandomnessReport(fair_bits)
        report.run_all()
        summary = report.summary()
        assert "NIST" in summary
        assert "Frequency" in summary

    def test_to_dict(self, fair_bits):
        report = RandomnessReport(fair_bits)
        report.run_all()
        d = report.to_dict()
        assert "n_bits" in d
        assert "tests" in d


# =====================================================================
# CHSH Certifier tests
# =====================================================================

class TestCHSHCertifier:

    def test_bell_violation(self):
        cert = CHSHCertifier(n_rounds=5000, seed=42)
        result = cert.run_bell_test()
        assert result.s_value > 2.0
        assert result.is_quantum
        assert result.tsirelson_fraction > 0.8

    def test_min_entropy_positive(self):
        cert = CHSHCertifier(n_rounds=5000, seed=42)
        result = cert.run_bell_test()
        assert result.min_entropy_point_estimate > 0.0

    def test_summary_string(self):
        cert = CHSHCertifier(n_rounds=1000, seed=42)
        result = cert.run_bell_test()
        summary = result.summary()
        assert "S-value" in summary


# =====================================================================
# EntropyAccumulation tests
# =====================================================================

class TestEntropyAccumulation:

    def test_accumulation_positive(self):
        ea = EntropyAccumulation(
            n_rounds=10000, smoothness=1e-4, test_fraction=0.1, seed=42
        )
        result = ea.accumulate()
        assert result.total_smooth_min_entropy > 0
        assert result.is_positive
        assert result.chsh_winning_prob > 0.75

    def test_entropy_rate_bounds(self):
        # Classical threshold: omega <= 0.75 gives 0 entropy
        rate = EntropyAccumulation._entropy_rate(0.75)
        assert rate == 0.0
        # Quantum strategy: omega ~ 0.854
        rate_q = EntropyAccumulation._entropy_rate(0.85)
        assert rate_q > 0.0


# =====================================================================
# RandomnessExpansion tests
# =====================================================================

class TestRandomnessExpansion:

    def test_expansion_protocol(self):
        exp = RandomnessExpansion(
            n_rounds=50000, seed_length=500, seed=42
        )
        result = exp.run()
        assert result.output_entropy > 0
        # Should have net positive expansion
        assert result.has_expansion
        assert result.expansion_factor > 1.0


# =====================================================================
# Protocol convenience functions
# =====================================================================

class TestProtocols:

    @pytest.mark.parametrize("protocol", [
        "measurement", "vacuum", "entanglement", "dice",
    ])
    def test_random_bits_protocols(self, protocol):
        bits = random_bits(200, protocol=protocol, seed=42)
        assert len(bits) == 200
        assert set(np.unique(bits)).issubset({0, 1})

    def test_random_uniform(self):
        floats = random_uniform(50, low=0.0, high=1.0, seed=42)
        assert len(floats) == 50
        assert all(0.0 <= f < 1.0 for f in floats)

    def test_random_uniform_range(self):
        floats = random_uniform(50, low=5.0, high=10.0, seed=42)
        assert all(5.0 <= f < 10.0 for f in floats)

    def test_random_integers(self):
        ints = random_integers(100, low=0, high=10, seed=42)
        assert len(ints) == 100
        assert all(0 <= i < 10 for i in ints)

    def test_invalid_protocol_raises(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            random_bits(10, protocol="nonexistent")


# =====================================================================
# RandomBeacon tests
# =====================================================================

class TestRandomBeacon:

    def test_emit_and_chain(self):
        beacon = RandomBeacon(seed=42)
        o1 = beacon.emit(n_bits=128)
        o2 = beacon.emit(n_bits=128)
        assert beacon.chain_length == 2
        assert o2.prev_hash == o1.self_hash

    def test_chain_verification(self):
        beacon = RandomBeacon(seed=42)
        beacon.emit(64)
        beacon.emit(64)
        beacon.emit(64)
        valid, broken_idx = beacon.verify_chain()
        assert valid
        assert broken_idx is None

    def test_genesis_prev_hash(self):
        beacon = RandomBeacon(seed=42)
        o = beacon.emit(64)
        assert o.prev_hash == "0" * 64
        assert o.sequence_number == 0
