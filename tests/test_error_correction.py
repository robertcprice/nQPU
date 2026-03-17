"""Comprehensive tests for nqpu.error_correction package.

Tests cover quantum error correcting codes (repetition, Steane, Shor, surface, color),
syndrome decoders (lookup table, MWPM, union-find, BP), noise models (depolarizing,
phenomenological, circuit-level, biased), and decoder benchmarking.
"""

import numpy as np
import pytest

from nqpu.error_correction import (
    # Codes
    RepetitionCode,
    SteaneCode,
    ShorCode,
    SurfaceCode,
    ColorCode,
    PauliType,
    # Decoders
    LookupTableDecoder,
    MWPMDecoder,
    UnionFindDecoder,
    BPDecoder,
    DecoderBenchmark,
    benchmark_decoder,
    # Noise models
    DepolarizingNoise,
    PhenomenologicalNoise,
    CircuitLevelNoise,
    BiasedNoise,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def repetition_code():
    """Distance-3 bit-flip repetition code."""
    return RepetitionCode(distance=3, code_type="bit_flip")


@pytest.fixture
def steane_code():
    """Steane [[7,1,3]] code."""
    return SteaneCode()


@pytest.fixture
def shor_code():
    """Shor [[9,1,3]] code."""
    return ShorCode()


@pytest.fixture
def surface_code_d3():
    """Distance-3 rotated surface code."""
    return SurfaceCode(distance=3, rotated=True)


@pytest.fixture
def color_code():
    """[[7,1,3]] color code."""
    return ColorCode()


# ------------------------------------------------------------------ #
# Code parameter tests
# ------------------------------------------------------------------ #


class TestCodeParameters:
    """Verify [[n, k, d]] parameters for each code family."""

    def test_repetition_code_params(self, repetition_code):
        assert repetition_code.code_params == (3, 1, 3)

    def test_steane_code_params(self, steane_code):
        assert steane_code.code_params == (7, 1, 3)

    def test_shor_code_params(self, shor_code):
        assert shor_code.code_params == (9, 1, 3)

    def test_surface_code_params(self, surface_code_d3):
        assert surface_code_d3.code_params == (9, 1, 3)

    def test_color_code_params(self, color_code):
        assert color_code.code_params == (7, 1, 3)

    @pytest.mark.parametrize("distance", [3, 5, 7])
    def test_repetition_code_distances(self, distance):
        code = RepetitionCode(distance=distance)
        assert code.n == distance
        assert code.k == 1
        assert code.d == distance

    def test_repetition_code_invalid_distance_even(self):
        with pytest.raises(ValueError):
            RepetitionCode(distance=4)

    def test_repetition_code_invalid_distance_small(self):
        with pytest.raises(ValueError):
            RepetitionCode(distance=1)

    def test_surface_code_invalid_distance(self):
        with pytest.raises(ValueError):
            SurfaceCode(distance=4)


# ------------------------------------------------------------------ #
# Check matrix structure tests
# ------------------------------------------------------------------ #


class TestCheckMatrices:
    """Verify check matrix shapes and properties."""

    def test_repetition_bit_flip_hz_shape(self, repetition_code):
        # Bit-flip code has Z checks (Hz), no X checks
        assert repetition_code.Hz.shape == (2, 3)
        assert repetition_code.Hx.shape[0] == 0

    def test_repetition_phase_flip_hx_shape(self):
        code = RepetitionCode(distance=3, code_type="phase_flip")
        assert code.Hx.shape == (2, 3)
        assert code.Hz.shape[0] == 0

    def test_steane_css_symmetry(self, steane_code):
        # Steane code is CSS with Hx = Hz = Hamming parity check
        assert steane_code.Hx.shape == (3, 7)
        assert steane_code.Hz.shape == (3, 7)
        np.testing.assert_array_equal(steane_code.Hx, steane_code.Hz)

    def test_shor_check_matrix_shapes(self, shor_code):
        assert shor_code.Hz.shape == (6, 9)
        assert shor_code.Hx.shape == (2, 9)

    def test_surface_code_check_matrix_shapes(self, surface_code_d3):
        # Distance-3 rotated surface code: n = 9
        assert surface_code_d3.n == 9
        # Total stabilizers should be n - k = 8
        total_checks = surface_code_d3.Hx.shape[0] + surface_code_d3.Hz.shape[0]
        assert total_checks == 8


# ------------------------------------------------------------------ #
# Syndrome computation tests
# ------------------------------------------------------------------ #


class TestSyndromeComputation:
    """Verify syndrome extraction from error vectors."""

    def test_no_error_trivial_syndrome(self, steane_code):
        error = np.zeros(2 * steane_code.n, dtype=np.int8)
        syndrome = steane_code.syndrome(error)
        np.testing.assert_array_equal(syndrome, np.zeros_like(syndrome))

    def test_single_x_error_syndrome(self, repetition_code):
        n = repetition_code.n
        # X error on qubit 0
        error = np.zeros(2 * n, dtype=np.int8)
        error[0] = 1
        syndrome = repetition_code.syndrome(error)
        # Should trigger Z checks
        assert np.any(syndrome)

    def test_single_z_error_steane(self, steane_code):
        n = steane_code.n
        # Z error on qubit 0
        error = np.zeros(2 * n, dtype=np.int8)
        error[n] = 1  # z_part[0] = 1
        syndrome = steane_code.syndrome(error)
        # Should trigger X checks (first part of syndrome)
        assert np.any(syndrome[:steane_code.Hx.shape[0]])

    def test_syndrome_x_and_z_consistency(self, steane_code):
        n = steane_code.n
        error = np.zeros(2 * n, dtype=np.int8)
        error[0] = 1  # X on qubit 0
        error[n + 1] = 1  # Z on qubit 1
        full_syn = steane_code.syndrome(error)
        sx = steane_code.syndrome_x(error)
        sz = steane_code.syndrome_z(error)
        np.testing.assert_array_equal(full_syn, np.concatenate([sx, sz]))


# ------------------------------------------------------------------ #
# Stabilizer and logical operator tests
# ------------------------------------------------------------------ #


class TestStabilizersAndLogicals:
    """Verify stabilizer generators and logical operators."""

    def test_stabilizers_count(self, steane_code):
        stabs = steane_code.stabilizers()
        # Steane: 3 X stabs + 3 Z stabs = 6 total
        assert len(stabs) == 6

    def test_shor_stabilizers_count(self, shor_code):
        stabs = shor_code.stabilizers()
        # Shor: 6 Z stabs + 2 X stabs = 8 total
        assert len(stabs) == 8

    def test_logical_operators_length(self, steane_code):
        lx = steane_code.logical_x()
        lz = steane_code.logical_z()
        assert len(lx) == 1  # k=1
        assert len(lz) == 1
        assert len(lx[0]) == 2 * steane_code.n
        assert len(lz[0]) == 2 * steane_code.n

    def test_stabilizer_trivial_syndrome(self, steane_code):
        # Each stabilizer should have trivial syndrome
        for stab in steane_code.stabilizers():
            syn = steane_code.syndrome(stab)
            np.testing.assert_array_equal(syn, np.zeros_like(syn))

    def test_distance_method(self, steane_code):
        assert steane_code.distance() == 3


# ------------------------------------------------------------------ #
# Encode/decode tests
# ------------------------------------------------------------------ #


class TestEncodeDecode:
    """Verify encoding and decoding roundtrip."""

    def test_repetition_encode_zero(self, repetition_code):
        logical = np.array([1.0, 0.0])
        encoded = repetition_code.encode(logical)
        assert len(encoded) == 2 ** repetition_code.n
        # |0_L> = |000> for bit-flip code
        assert abs(encoded[0] - 1.0) < 1e-10

    def test_repetition_encode_one(self, repetition_code):
        logical = np.array([0.0, 1.0])
        encoded = repetition_code.encode(logical)
        # |1_L> = |111> for bit-flip code
        assert abs(encoded[2 ** repetition_code.n - 1] - 1.0) < 1e-10

    def test_steane_encode_decode_roundtrip(self, steane_code):
        logical_in = np.array([1.0, 0.0], dtype=np.complex128)
        encoded = steane_code.encode(logical_in)
        decoded = steane_code.decode(encoded)
        # Decoded should match input (up to global phase)
        overlap = abs(np.dot(decoded.conj(), logical_in))
        assert overlap > 0.99

    def test_shor_encode_decode_roundtrip(self, shor_code):
        logical_in = np.array([1.0, 0.0], dtype=np.complex128)
        encoded = shor_code.encode(logical_in)
        decoded = shor_code.decode(encoded)
        overlap = abs(np.dot(decoded.conj(), logical_in))
        assert overlap > 0.99


# ------------------------------------------------------------------ #
# Decoder tests
# ------------------------------------------------------------------ #


class TestLookupTableDecoder:
    """Test lookup table decoder on small codes."""

    def test_no_error_no_correction(self, steane_code):
        decoder = LookupTableDecoder(steane_code, max_weight=1)
        error = np.zeros(2 * steane_code.n, dtype=np.int8)
        syndrome = steane_code.syndrome(error)
        correction = decoder.decode(syndrome)
        np.testing.assert_array_equal(correction, np.zeros_like(correction))

    def test_single_x_error_corrected(self, steane_code):
        decoder = LookupTableDecoder(steane_code, max_weight=1)
        n = steane_code.n
        # X error on qubit 2
        error = np.zeros(2 * n, dtype=np.int8)
        error[2] = 1
        syndrome = steane_code.syndrome(error)
        correction = decoder.decode(syndrome)
        # Check that the correction fixes the error
        assert steane_code.check_correction(error, correction)

    def test_single_z_error_corrected(self, steane_code):
        decoder = LookupTableDecoder(steane_code, max_weight=1)
        n = steane_code.n
        # Z error on qubit 4
        error = np.zeros(2 * n, dtype=np.int8)
        error[n + 4] = 1
        syndrome = steane_code.syndrome(error)
        correction = decoder.decode(syndrome)
        assert steane_code.check_correction(error, correction)


class TestMWPMDecoder:
    """Test minimum weight perfect matching decoder."""

    def test_mwpm_on_surface_code_no_error(self, surface_code_d3):
        decoder = MWPMDecoder(surface_code_d3)
        error = np.zeros(2 * surface_code_d3.n, dtype=np.int8)
        syndrome = surface_code_d3.syndrome(error)
        correction = decoder.decode(syndrome)
        np.testing.assert_array_equal(correction, np.zeros_like(correction))

    def test_mwpm_single_error(self, surface_code_d3):
        decoder = MWPMDecoder(surface_code_d3)
        n = surface_code_d3.n
        # Single X error
        error = np.zeros(2 * n, dtype=np.int8)
        error[4] = 1  # X on qubit 4 (center of 3x3 grid)
        syndrome = surface_code_d3.syndrome(error)
        correction = decoder.decode(syndrome)
        assert surface_code_d3.check_correction(error, correction)


class TestBPDecoder:
    """Test belief propagation decoder."""

    def test_bp_decoder_no_error(self, steane_code):
        decoder = BPDecoder(steane_code, max_iterations=20, channel_error_rate=0.05)
        error = np.zeros(2 * steane_code.n, dtype=np.int8)
        syndrome = steane_code.syndrome(error)
        correction = decoder.decode(syndrome)
        np.testing.assert_array_equal(correction, np.zeros_like(correction))

    def test_bp_decoder_single_error(self, steane_code):
        decoder = BPDecoder(steane_code, max_iterations=50, channel_error_rate=0.1)
        n = steane_code.n
        error = np.zeros(2 * n, dtype=np.int8)
        error[0] = 1  # X on qubit 0
        syndrome = steane_code.syndrome(error)
        correction = decoder.decode(syndrome)
        # BP may or may not correct perfectly, but it should produce a valid correction vector
        assert len(correction) == 2 * n


# ------------------------------------------------------------------ #
# Noise model tests
# ------------------------------------------------------------------ #


class TestDepolarizingNoise:
    """Test depolarizing noise model."""

    def test_zero_probability_no_errors(self):
        noise = DepolarizingNoise(p=0.0)
        rng = np.random.default_rng(42)
        error = noise.sample_error(10, rng=rng)
        np.testing.assert_array_equal(error, np.zeros(20, dtype=np.int8))

    def test_error_vector_shape(self):
        noise = DepolarizingNoise(p=0.1)
        rng = np.random.default_rng(42)
        error = noise.sample_error(5, rng=rng)
        assert error.shape == (10,)
        assert error.dtype == np.int8

    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            DepolarizingNoise(p=1.5)

    def test_expected_error_weight(self):
        noise = DepolarizingNoise(p=0.3)
        assert noise.expected_error_weight(10) == pytest.approx(3.0)

    def test_sample_syndrome(self, steane_code):
        noise = DepolarizingNoise(p=0.1)
        rng = np.random.default_rng(42)
        error, syndrome = noise.sample_syndrome(steane_code, rng=rng)
        assert error.shape == (2 * steane_code.n,)
        expected_syn = steane_code.syndrome(error)
        np.testing.assert_array_equal(syndrome, expected_syn)


class TestPhenomenologicalNoise:
    """Test phenomenological noise model."""

    def test_measurement_noise_can_flip_syndrome(self):
        noise = PhenomenologicalNoise(p=0.0, q=1.0, num_rounds=1)
        code = SteaneCode()
        rng = np.random.default_rng(42)
        error, noisy_syn = noise.sample_syndrome(code, rng=rng)
        # With q=1.0, every syndrome bit should be flipped
        perfect_syn = code.syndrome(error)
        for i in range(len(perfect_syn)):
            assert noisy_syn[i] == (perfect_syn[i] + 1) % 2

    def test_multi_round_sampling(self):
        noise = PhenomenologicalNoise(p=0.05, q=0.01, num_rounds=3)
        code = SteaneCode()
        rng = np.random.default_rng(42)
        total_error, history = noise.sample_multi_round(code, rng=rng)
        assert len(history) == 3
        assert total_error.shape == (2 * code.n,)


class TestBiasedNoise:
    """Test biased noise model."""

    def test_bias_ratio(self):
        noise = BiasedNoise(p_x=0.1, p_z=0.01)
        assert noise.bias_ratio == pytest.approx(0.1)

    def test_total_error_rate(self):
        noise = BiasedNoise(p_x=0.1, p_z=0.01)
        expected = 0.1 + 0.01 - 0.1 * 0.01
        assert noise.total_error_rate == pytest.approx(expected)

    def test_biased_noise_sample_shape(self):
        noise = BiasedNoise(p_x=0.1, p_z=0.01)
        rng = np.random.default_rng(42)
        error = noise.sample_error(5, rng=rng)
        assert error.shape == (10,)


class TestCircuitLevelNoise:
    """Test circuit-level noise model."""

    def test_effective_error_rate(self):
        noise = CircuitLevelNoise(p_data=0.001, p_gate=0.002, p_meas=0.003, p_prep=0.004)
        assert noise.effective_error_rate == pytest.approx(0.01)

    def test_sample_error_shape(self):
        noise = CircuitLevelNoise()
        rng = np.random.default_rng(42)
        error = noise.sample_error(9, rng=rng)
        assert error.shape == (18,)


# ------------------------------------------------------------------ #
# Benchmark test
# ------------------------------------------------------------------ #


class TestBenchmark:
    """Test decoder benchmarking functionality."""

    def test_benchmark_returns_result(self, steane_code):
        decoder = LookupTableDecoder(steane_code, max_weight=1)
        result = benchmark_decoder(
            steane_code, decoder,
            physical_error_rate=0.01,
            num_trials=50,
            seed=42,
        )
        assert isinstance(result, DecoderBenchmark)
        assert result.num_trials == 50
        assert result.code_params == (7, 1, 3)
        assert result.decoder_name == "LookupTableDecoder"
        assert 0.0 <= result.logical_error_rate <= 1.0

    def test_benchmark_low_error_rate_succeeds(self, repetition_code):
        decoder = LookupTableDecoder(repetition_code, max_weight=1)
        result = benchmark_decoder(
            repetition_code, decoder,
            physical_error_rate=0.001,
            num_trials=100,
            error_type="x_only",
            seed=42,
        )
        # At very low error rates, most trials should succeed
        assert result.logical_error_rate < 0.5


# ------------------------------------------------------------------ #
# PauliType enum test
# ------------------------------------------------------------------ #


class TestPauliType:
    """Test PauliType enum."""

    def test_pauli_values(self):
        assert PauliType.I.value == 0
        assert PauliType.X.value == 1
        assert PauliType.Z.value == 2
        assert PauliType.Y.value == 3
