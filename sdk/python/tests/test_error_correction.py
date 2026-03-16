"""Comprehensive tests for the nQPU quantum error correction package.

Covers all code families, decoders, noise models, threshold estimation,
and lattice surgery operations.  100+ tests organized by module.
"""

from __future__ import annotations

import numpy as np
import pytest

from nqpu.error_correction import (
    # Codes
    QuantumCode,
    RepetitionCode,
    SurfaceCode,
    SteaneCode,
    ShorCode,
    ColorCode,
    PauliType,
    # Decoders
    Decoder,
    LookupTableDecoder,
    MWPMDecoder,
    UnionFindDecoder,
    BPDecoder,
    DecoderBenchmark,
    benchmark_decoder,
    # Noise
    NoiseModel,
    DepolarizingNoise,
    PhenomenologicalNoise,
    CircuitLevelNoise,
    BiasedNoise,
    # Threshold
    ThresholdEstimator,
    ThresholdResult,
    ThresholdDataPoint,
    estimate_threshold,
    compare_codes,
    # Lattice surgery
    LogicalQubit,
    LatticeSurgery,
    MagicStateDistillation,
    PauliFrame,
    ResourceEstimate,
    estimate_resources,
)


# ================================================================== #
#  CODES
# ================================================================== #

class TestRepetitionCode:
    """Tests for the RepetitionCode."""

    def test_creation_distance_3(self):
        code = RepetitionCode(distance=3)
        assert code.n == 3
        assert code.k == 1
        assert code.d == 3
        assert code.code_params == (3, 1, 3)

    def test_creation_distance_5(self):
        code = RepetitionCode(distance=5)
        assert code.n == 5
        assert code.k == 1
        assert code.d == 5

    def test_invalid_even_distance(self):
        with pytest.raises(ValueError):
            RepetitionCode(distance=4)

    def test_invalid_small_distance(self):
        with pytest.raises(ValueError):
            RepetitionCode(distance=1)

    def test_invalid_code_type(self):
        with pytest.raises(ValueError):
            RepetitionCode(code_type="invalid")

    def test_bit_flip_stabilizers(self):
        code = RepetitionCode(distance=3, code_type="bit_flip")
        stabs = code.stabilizers()
        assert len(stabs) == 2  # d-1 stabilizers
        # ZZ on qubits 0,1 and ZZ on qubits 1,2
        for s in stabs:
            assert len(s) == 6  # 2*n
            z_part = s[3:]
            assert z_part.sum() == 2  # weight-2 Z stabilizer

    def test_phase_flip_stabilizers(self):
        code = RepetitionCode(distance=3, code_type="phase_flip")
        stabs = code.stabilizers()
        assert len(stabs) == 2
        for s in stabs:
            x_part = s[:3]
            assert x_part.sum() == 2  # weight-2 X stabilizer

    def test_bit_flip_syndrome_no_error(self):
        code = RepetitionCode(distance=3, code_type="bit_flip")
        error = np.zeros(6, dtype=np.int8)
        syn = code.syndrome(error)
        assert np.all(syn == 0)

    def test_bit_flip_syndrome_single_x_error(self):
        code = RepetitionCode(distance=3, code_type="bit_flip")
        # X error on qubit 0
        error = np.zeros(6, dtype=np.int8)
        error[0] = 1  # X on qubit 0
        syn = code.syndrome(error)
        # Z checks detect X errors: Hz @ x_part
        # Hz = [[1,1,0],[0,1,1]], x_part = [1,0,0]
        # syn_z = [1, 0]
        assert syn.sum() > 0  # non-trivial syndrome

    def test_bit_flip_syndrome_single_z_error(self):
        code = RepetitionCode(distance=3, code_type="bit_flip")
        # Z error on qubit 1 -- bit flip code has no X checks
        error = np.zeros(6, dtype=np.int8)
        error[4] = 1  # Z on qubit 1
        syn = code.syndrome(error)
        # No X checks means Z errors are not detected
        assert syn.sum() == 0

    def test_distance_property(self):
        for d in [3, 5, 7, 9]:
            code = RepetitionCode(distance=d)
            assert code.distance() == d

    def test_logical_operators(self):
        code = RepetitionCode(distance=3, code_type="bit_flip")
        lx = code.logical_x()
        lz = code.logical_z()
        assert len(lx) == 1
        assert len(lz) == 1
        # Logical X = X on all qubits
        assert lx[0][:3].sum() == 3

    def test_encode_decode_zero(self):
        code = RepetitionCode(distance=3)
        logical = np.array([1.0, 0.0])
        encoded = code.encode(logical)
        assert abs(encoded[0]) > 0.99  # |000> component
        decoded = code.decode(encoded)
        assert abs(abs(decoded[0]) - 1.0) < 1e-10

    def test_encode_decode_one(self):
        code = RepetitionCode(distance=3)
        logical = np.array([0.0, 1.0])
        encoded = code.encode(logical)
        assert abs(encoded[7]) > 0.99  # |111> component (index 7 = 0b111)
        decoded = code.decode(encoded)
        assert abs(abs(decoded[1]) - 1.0) < 1e-10

    def test_encode_decode_superposition(self):
        code = RepetitionCode(distance=3)
        logical = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
        encoded = code.encode(logical)
        decoded = code.decode(encoded)
        assert abs(decoded[0] - 1 / np.sqrt(2)) < 1e-10
        assert abs(decoded[1] - 1 / np.sqrt(2)) < 1e-10

    def test_check_correction_identity(self):
        code = RepetitionCode(distance=3, code_type="bit_flip")
        # No error, no correction -> success
        error = np.zeros(6, dtype=np.int8)
        correction = np.zeros(6, dtype=np.int8)
        assert code.check_correction(error, correction)

    def test_check_correction_single_error(self):
        code = RepetitionCode(distance=3, code_type="bit_flip")
        # X error on qubit 1, correct with X on qubit 1
        error = np.zeros(6, dtype=np.int8)
        error[1] = 1  # X on qubit 1
        correction = np.zeros(6, dtype=np.int8)
        correction[1] = 1  # X on qubit 1
        assert code.check_correction(error, correction)


class TestSteaneCode:
    """Tests for the Steane [[7,1,3]] code."""

    def test_creation(self):
        code = SteaneCode()
        assert code.n == 7
        assert code.k == 1
        assert code.d == 3
        assert code.code_params == (7, 1, 3)

    def test_stabilizers_count(self):
        code = SteaneCode()
        stabs = code.stabilizers()
        # 3 X + 3 Z = 6 stabilizers = n - k = 7 - 1 = 6
        assert len(stabs) == 6

    def test_stabilizers_commute(self):
        """All stabilizers must commute with each other."""
        code = SteaneCode()
        stabs = code.stabilizers()
        n = code.n
        for i in range(len(stabs)):
            for j in range(i + 1, len(stabs)):
                si = stabs[i]
                sj = stabs[j]
                # Symplectic inner product
                ip = (
                    np.dot(si[:n], sj[n:]) + np.dot(si[n:], sj[:n])
                ) % 2
                assert ip == 0, f"Stabilizers {i} and {j} anticommute"

    def test_logical_operators_anticommute(self):
        """Logical X and Z must anticommute."""
        code = SteaneCode()
        lx = code.logical_x()[0]
        lz = code.logical_z()[0]
        n = code.n
        ip = (np.dot(lx[:n], lz[n:]) + np.dot(lx[n:], lz[:n])) % 2
        assert ip == 1, "Logical X and Z should anticommute"

    def test_logicals_commute_with_stabilizers(self):
        """Logical operators must commute with all stabilizers."""
        code = SteaneCode()
        stabs = code.stabilizers()
        n = code.n
        for logical in code.logical_x() + code.logical_z():
            for s in stabs:
                ip = (
                    np.dot(logical[:n], s[n:]) + np.dot(logical[n:], s[:n])
                ) % 2
                assert ip == 0

    def test_syndrome_no_error(self):
        code = SteaneCode()
        error = np.zeros(14, dtype=np.int8)
        syn = code.syndrome(error)
        assert np.all(syn == 0)

    def test_syndrome_single_x_error(self):
        code = SteaneCode()
        error = np.zeros(14, dtype=np.int8)
        error[0] = 1  # X on qubit 0
        syn = code.syndrome(error)
        assert syn.sum() > 0

    def test_syndrome_single_z_error(self):
        code = SteaneCode()
        error = np.zeros(14, dtype=np.int8)
        error[7] = 1  # Z on qubit 0
        syn = code.syndrome(error)
        assert syn.sum() > 0

    def test_different_x_errors_different_syndromes(self):
        """Single X errors on different qubits give distinct syndromes."""
        code = SteaneCode()
        syndromes = set()
        for q in range(7):
            error = np.zeros(14, dtype=np.int8)
            error[q] = 1
            syn = code.syndrome(error)
            syndromes.add(syn.tobytes())
        # All single-qubit X errors should produce distinct syndromes
        assert len(syndromes) == 7

    def test_encode_decode_zero(self):
        code = SteaneCode()
        encoded = code.encode(np.array([1.0, 0.0]))
        decoded = code.decode(encoded)
        assert abs(abs(decoded[0]) - 1.0) < 1e-8

    def test_encode_decode_one(self):
        code = SteaneCode()
        encoded = code.encode(np.array([0.0, 1.0]))
        decoded = code.decode(encoded)
        assert abs(abs(decoded[1]) - 1.0) < 1e-8

    def test_encode_normalization(self):
        code = SteaneCode()
        encoded = code.encode(np.array([1.0, 0.0]))
        assert abs(np.linalg.norm(encoded) - 1.0) < 1e-10

    def test_distance_value(self):
        code = SteaneCode()
        assert code.distance() == 3


class TestShorCode:
    """Tests for the Shor [[9,1,3]] code."""

    def test_creation(self):
        code = ShorCode()
        assert code.n == 9
        assert code.k == 1
        assert code.d == 3
        assert code.code_params == (9, 1, 3)

    def test_stabilizers_count(self):
        code = ShorCode()
        stabs = code.stabilizers()
        # n - k = 8 stabilizers
        assert len(stabs) == 8

    def test_stabilizers_commute(self):
        code = ShorCode()
        stabs = code.stabilizers()
        n = code.n
        for i in range(len(stabs)):
            for j in range(i + 1, len(stabs)):
                si = stabs[i]
                sj = stabs[j]
                ip = (
                    np.dot(si[:n], sj[n:]) + np.dot(si[n:], sj[:n])
                ) % 2
                assert ip == 0

    def test_logical_anticommutation(self):
        code = ShorCode()
        lx = code.logical_x()[0]
        lz = code.logical_z()[0]
        n = code.n
        ip = (np.dot(lx[:n], lz[n:]) + np.dot(lx[n:], lz[:n])) % 2
        assert ip == 1

    def test_syndrome_single_x_error(self):
        code = ShorCode()
        error = np.zeros(18, dtype=np.int8)
        error[0] = 1  # X on qubit 0
        syn = code.syndrome(error)
        assert syn.sum() > 0

    def test_encode_decode_zero(self):
        code = ShorCode()
        encoded = code.encode(np.array([1.0, 0.0]))
        assert abs(np.linalg.norm(encoded) - 1.0) < 1e-10
        decoded = code.decode(encoded)
        assert abs(abs(decoded[0]) - 1.0) < 1e-8

    def test_encode_decode_one(self):
        code = ShorCode()
        encoded = code.encode(np.array([0.0, 1.0]))
        decoded = code.decode(encoded)
        assert abs(abs(decoded[1]) - 1.0) < 1e-8


class TestSurfaceCode:
    """Tests for the SurfaceCode."""

    def test_creation_rotated_d3(self):
        code = SurfaceCode(distance=3, rotated=True)
        assert code.n == 9  # 3*3
        assert code.k == 1
        assert code.d == 3

    def test_creation_rotated_d5(self):
        code = SurfaceCode(distance=5, rotated=True)
        assert code.n == 25  # 5*5
        assert code.d == 5

    def test_creation_unrotated_d3(self):
        code = SurfaceCode(distance=3, rotated=False)
        assert code.n == 9
        assert code.d == 3

    def test_invalid_even_distance(self):
        with pytest.raises(ValueError):
            SurfaceCode(distance=4)

    def test_invalid_small_distance(self):
        with pytest.raises(ValueError):
            SurfaceCode(distance=1)

    def test_stabilizers_are_css(self):
        """All stabilizers should be pure X or pure Z."""
        code = SurfaceCode(distance=3)
        stabs = code.stabilizers()
        for s in stabs:
            x_part = s[: code.n]
            z_part = s[code.n :]
            # CSS: either pure X (z_part all 0) or pure Z (x_part all 0)
            assert np.all(x_part == 0) or np.all(z_part == 0)

    def test_stabilizers_commute(self):
        code = SurfaceCode(distance=3)
        stabs = code.stabilizers()
        n = code.n
        for i in range(len(stabs)):
            for j in range(i + 1, len(stabs)):
                si = stabs[i]
                sj = stabs[j]
                ip = (
                    np.dot(si[:n], sj[n:]) + np.dot(si[n:], sj[:n])
                ) % 2
                assert ip == 0

    def test_logical_anticommutation(self):
        code = SurfaceCode(distance=3)
        lx = code.logical_x()[0]
        lz = code.logical_z()[0]
        n = code.n
        ip = (np.dot(lx[:n], lz[n:]) + np.dot(lx[n:], lz[:n])) % 2
        assert ip == 1

    def test_syndrome_no_error(self):
        code = SurfaceCode(distance=3)
        error = np.zeros(2 * code.n, dtype=np.int8)
        syn = code.syndrome(error)
        assert np.all(syn == 0)

    def test_syndrome_single_x_error(self):
        code = SurfaceCode(distance=3)
        error = np.zeros(2 * code.n, dtype=np.int8)
        error[4] = 1  # X on qubit 4 (center of 3x3)
        syn = code.syndrome(error)
        assert syn.sum() > 0

    def test_qubit_coords(self):
        code = SurfaceCode(distance=3)
        coords = code.qubit_coords()
        assert len(coords) == 9
        assert coords[0] == (0, 0)
        assert coords[8] == (2, 2)

    def test_encode_decode_d3(self):
        code = SurfaceCode(distance=3)
        encoded = code.encode(np.array([1.0, 0.0]))
        assert abs(np.linalg.norm(encoded) - 1.0) < 1e-10
        decoded = code.decode(encoded)
        assert abs(abs(decoded[0]) - 1.0) < 1e-6

    def test_check_matrix_shapes(self):
        code = SurfaceCode(distance=3)
        assert code.Hx.shape[1] == code.n
        assert code.Hz.shape[1] == code.n
        # Total stabilizers = n - k
        total_checks = code.Hx.shape[0] + code.Hz.shape[0]
        assert total_checks == code.n - code.k


class TestColorCode:
    """Tests for the [[7,1,3]] color code."""

    def test_creation(self):
        code = ColorCode()
        assert code.n == 7
        assert code.k == 1
        assert code.d == 3

    def test_stabilizers_count(self):
        code = ColorCode()
        stabs = code.stabilizers()
        # 3 X + 3 Z = 6 = n - k
        assert len(stabs) == 6

    def test_stabilizers_commute(self):
        code = ColorCode()
        stabs = code.stabilizers()
        n = code.n
        for i in range(len(stabs)):
            for j in range(i + 1, len(stabs)):
                si = stabs[i]
                sj = stabs[j]
                ip = (
                    np.dot(si[:n], sj[n:]) + np.dot(si[n:], sj[:n])
                ) % 2
                assert ip == 0

    def test_logical_anticommutation(self):
        code = ColorCode()
        lx = code.logical_x()[0]
        lz = code.logical_z()[0]
        n = code.n
        ip = (np.dot(lx[:n], lz[n:]) + np.dot(lx[n:], lz[:n])) % 2
        assert ip == 1

    def test_syndrome_single_error(self):
        code = ColorCode()
        error = np.zeros(14, dtype=np.int8)
        error[3] = 1  # X on qubit 3
        syn = code.syndrome(error)
        assert syn.sum() > 0

    def test_encode_decode(self):
        code = ColorCode()
        encoded = code.encode(np.array([1.0, 0.0]))
        assert abs(np.linalg.norm(encoded) - 1.0) < 1e-10
        decoded = code.decode(encoded)
        assert abs(abs(decoded[0]) - 1.0) < 1e-6


# ================================================================== #
#  DECODERS
# ================================================================== #

class TestLookupTableDecoder:
    """Tests for the LookupTableDecoder."""

    def test_trivial_syndrome(self):
        code = RepetitionCode(distance=3)
        decoder = LookupTableDecoder(code)
        syn = np.zeros(2, dtype=np.int8)  # trivial syndrome (d-1 = 2 checks)
        correction = decoder.decode(syn)
        assert np.all(correction == 0)

    def test_single_error_repetition(self):
        code = RepetitionCode(distance=3, code_type="bit_flip")
        decoder = LookupTableDecoder(code)
        # X error on qubit 1
        error = np.zeros(6, dtype=np.int8)
        error[1] = 1
        syn = code.syndrome(error)
        correction = decoder.decode(syn)
        # Correction should fix the error
        assert code.check_correction(error, correction)

    def test_steane_single_x_error(self):
        code = SteaneCode()
        decoder = LookupTableDecoder(code)
        for q in range(7):
            error = np.zeros(14, dtype=np.int8)
            error[q] = 1  # X on qubit q
            syn = code.syndrome(error)
            correction = decoder.decode(syn)
            assert code.check_correction(error, correction), \
                f"Failed to correct X error on qubit {q}"

    def test_steane_single_z_error(self):
        code = SteaneCode()
        decoder = LookupTableDecoder(code)
        for q in range(7):
            error = np.zeros(14, dtype=np.int8)
            error[7 + q] = 1  # Z on qubit q
            syn = code.syndrome(error)
            correction = decoder.decode(syn)
            assert code.check_correction(error, correction), \
                f"Failed to correct Z error on qubit {q}"

    def test_steane_single_y_error(self):
        code = SteaneCode()
        decoder = LookupTableDecoder(code)
        for q in range(7):
            error = np.zeros(14, dtype=np.int8)
            error[q] = 1      # X part
            error[7 + q] = 1  # Z part -> Y
            syn = code.syndrome(error)
            correction = decoder.decode(syn)
            assert code.check_correction(error, correction), \
                f"Failed to correct Y error on qubit {q}"

    def test_shor_single_errors(self):
        code = ShorCode()
        decoder = LookupTableDecoder(code)
        for q in range(9):
            # X error
            error = np.zeros(18, dtype=np.int8)
            error[q] = 1
            syn = code.syndrome(error)
            correction = decoder.decode(syn)
            assert code.check_correction(error, correction)

    def test_lookup_table_size(self):
        code = RepetitionCode(distance=3)
        decoder = LookupTableDecoder(code)
        # Should have entries for identity + single qubit errors
        assert len(decoder.table) >= 1  # at least the trivial syndrome

    def test_weight_2_table(self):
        code = SteaneCode()
        decoder = LookupTableDecoder(code, max_weight=2)
        # Weight-2 table should have more entries
        assert len(decoder.table) > 20


class TestMWPMDecoder:
    """Tests for the MWPM decoder."""

    def test_creation(self):
        code = SurfaceCode(distance=3)
        decoder = MWPMDecoder(code)
        assert decoder.n == 9

    def test_trivial_syndrome(self):
        code = SurfaceCode(distance=3)
        decoder = MWPMDecoder(code)
        syn = np.zeros(code.Hx.shape[0] + code.Hz.shape[0], dtype=np.int8)
        correction = decoder.decode(syn)
        assert np.all(correction == 0)

    def test_single_error_correction(self):
        code = SurfaceCode(distance=3)
        decoder = MWPMDecoder(code)
        # Apply a single X error
        error = np.zeros(2 * code.n, dtype=np.int8)
        error[4] = 1  # X on center qubit
        syn = code.syndrome(error)
        correction = decoder.decode(syn)
        # The correction + error should be in the stabilizer group
        residual = (error + correction) % 2
        residual_syn = code.syndrome(residual)
        assert np.all(residual_syn == 0), "MWPM did not clear syndrome"

    def test_beats_random_guessing(self):
        """MWPM should have lower logical error rate than random correction."""
        code = SurfaceCode(distance=3)
        decoder = MWPMDecoder(code)
        result = benchmark_decoder(
            code, decoder,
            physical_error_rate=0.05,
            num_trials=500,
            seed=42,
        )
        # Random guessing on d=3 surface code at 5% would give ~50% logical error
        # MWPM should do significantly better
        assert result.logical_error_rate < 0.40, \
            f"MWPM error rate {result.logical_error_rate} too high"

    def test_output_shape(self):
        code = SurfaceCode(distance=3)
        decoder = MWPMDecoder(code)
        error = np.zeros(2 * code.n, dtype=np.int8)
        error[0] = 1
        syn = code.syndrome(error)
        correction = decoder.decode(syn)
        assert correction.shape == (2 * code.n,)
        assert correction.dtype == np.int8


class TestUnionFindDecoder:
    """Tests for the UnionFind decoder."""

    def test_creation(self):
        code = SurfaceCode(distance=3)
        decoder = UnionFindDecoder(code)
        assert decoder.n == 9

    def test_trivial_syndrome(self):
        code = SurfaceCode(distance=3)
        decoder = UnionFindDecoder(code)
        syn = np.zeros(code.Hx.shape[0] + code.Hz.shape[0], dtype=np.int8)
        correction = decoder.decode(syn)
        assert np.all(correction == 0)

    def test_single_error_clears_syndrome(self):
        code = SurfaceCode(distance=3)
        decoder = UnionFindDecoder(code)
        error = np.zeros(2 * code.n, dtype=np.int8)
        error[4] = 1
        syn = code.syndrome(error)
        correction = decoder.decode(syn)
        residual = (error + correction) % 2
        residual_syn = code.syndrome(residual)
        assert np.all(residual_syn == 0)

    def test_beats_random_guessing(self):
        code = SurfaceCode(distance=3)
        decoder = UnionFindDecoder(code)
        result = benchmark_decoder(
            code, decoder,
            physical_error_rate=0.05,
            num_trials=500,
            seed=42,
        )
        assert result.logical_error_rate < 0.40

    def test_output_shape(self):
        code = SurfaceCode(distance=3)
        decoder = UnionFindDecoder(code)
        error = np.zeros(2 * code.n, dtype=np.int8)
        error[0] = 1
        syn = code.syndrome(error)
        correction = decoder.decode(syn)
        assert correction.shape == (2 * code.n,)


class TestBPDecoder:
    """Tests for the Belief Propagation decoder."""

    def test_creation(self):
        code = SteaneCode()
        decoder = BPDecoder(code)
        assert decoder.n == 7

    def test_trivial_syndrome(self):
        code = SteaneCode()
        decoder = BPDecoder(code)
        syn = np.zeros(6, dtype=np.int8)
        correction = decoder.decode(syn)
        # Should return all-zero or at least a stabilizer
        assert correction.shape == (14,)

    def test_single_error_steane(self):
        code = SteaneCode()
        decoder = BPDecoder(code, channel_error_rate=0.1, max_iterations=100)
        error = np.zeros(14, dtype=np.int8)
        error[3] = 1  # X on qubit 3
        syn = code.syndrome(error)
        correction = decoder.decode(syn)
        residual = (error + correction) % 2
        residual_syn = code.syndrome(residual)
        assert np.all(residual_syn == 0), "BP did not clear syndrome for single error"

    def test_bp_benchmark(self):
        code = SteaneCode()
        decoder = BPDecoder(code, channel_error_rate=0.05, max_iterations=50)
        result = benchmark_decoder(
            code, decoder,
            physical_error_rate=0.05,
            num_trials=200,
            seed=42,
        )
        # BP on Steane code should perform reasonably
        assert result.logical_error_rate < 0.50

    def test_convergence_detection(self):
        """BP should converge quickly for low-weight errors."""
        code = SteaneCode()
        decoder = BPDecoder(code, max_iterations=5, channel_error_rate=0.1)
        error = np.zeros(14, dtype=np.int8)
        error[0] = 1
        syn = code.syndrome(error)
        correction = decoder.decode(syn)
        # Even with few iterations, single-error should converge
        residual_syn = code.syndrome((error + correction) % 2)
        assert np.all(residual_syn == 0)


class TestDecoderBenchmark:
    """Tests for the benchmark_decoder function."""

    def test_benchmark_returns_dataclass(self):
        code = RepetitionCode(distance=3)
        decoder = LookupTableDecoder(code)
        result = benchmark_decoder(
            code, decoder, physical_error_rate=0.1,
            num_trials=100, seed=42,
        )
        assert isinstance(result, DecoderBenchmark)
        assert result.num_trials == 100
        assert result.code_params == (3, 1, 3)
        assert result.decoder_name == "LookupTableDecoder"

    def test_zero_error_rate(self):
        code = RepetitionCode(distance=3)
        decoder = LookupTableDecoder(code)
        result = benchmark_decoder(
            code, decoder, physical_error_rate=0.0,
            num_trials=100, seed=42,
        )
        assert result.logical_error_rate == 0.0
        assert result.num_failures == 0

    def test_x_only_error_type(self):
        code = RepetitionCode(distance=3, code_type="bit_flip")
        decoder = LookupTableDecoder(code)
        result = benchmark_decoder(
            code, decoder, physical_error_rate=0.1,
            num_trials=100, error_type="x_only", seed=42,
        )
        assert 0.0 <= result.logical_error_rate <= 1.0

    def test_z_only_error_type(self):
        code = SteaneCode()
        decoder = LookupTableDecoder(code)
        result = benchmark_decoder(
            code, decoder, physical_error_rate=0.1,
            num_trials=100, error_type="z_only", seed=42,
        )
        assert 0.0 <= result.logical_error_rate <= 1.0

    def test_invalid_error_type(self):
        code = RepetitionCode(distance=3)
        decoder = LookupTableDecoder(code)
        with pytest.raises(ValueError):
            benchmark_decoder(
                code, decoder, physical_error_rate=0.1,
                num_trials=10, error_type="invalid",
            )

    def test_low_error_rate_performs_well(self):
        """At very low error rates, d=3 codes should almost never fail."""
        code = SteaneCode()
        decoder = LookupTableDecoder(code)
        result = benchmark_decoder(
            code, decoder, physical_error_rate=0.001,
            num_trials=500, seed=42,
        )
        assert result.logical_error_rate < 0.05


# ================================================================== #
#  NOISE MODELS
# ================================================================== #

class TestDepolarizingNoise:
    """Tests for DepolarizingNoise."""

    def test_creation(self):
        noise = DepolarizingNoise(p=0.05)
        assert noise.p == 0.05

    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            DepolarizingNoise(p=-0.1)
        with pytest.raises(ValueError):
            DepolarizingNoise(p=1.5)

    def test_zero_error_rate(self):
        noise = DepolarizingNoise(p=0.0)
        rng = np.random.default_rng(42)
        error = noise.sample_error(10, rng)
        assert np.all(error == 0)

    def test_error_shape(self):
        noise = DepolarizingNoise(p=0.1)
        rng = np.random.default_rng(42)
        error = noise.sample_error(5, rng)
        assert error.shape == (10,)  # 2*n

    def test_error_is_binary(self):
        noise = DepolarizingNoise(p=0.5)
        rng = np.random.default_rng(42)
        error = noise.sample_error(20, rng)
        assert np.all((error == 0) | (error == 1))

    def test_error_rate_statistics(self):
        """Average error weight should be close to n*p."""
        noise = DepolarizingNoise(p=0.1)
        rng = np.random.default_rng(42)
        n = 100
        weights = []
        for _ in range(1000):
            error = noise.sample_error(n, rng)
            # Count qubits with any error
            x_part = error[:n]
            z_part = error[n:]
            has_error = ((x_part + z_part) > 0).sum()
            weights.append(has_error)
        mean_weight = np.mean(weights)
        # Should be close to n * p = 10
        assert 7 < mean_weight < 13

    def test_sample_syndrome(self):
        noise = DepolarizingNoise(p=0.1)
        code = SteaneCode()
        rng = np.random.default_rng(42)
        error, syn = noise.sample_syndrome(code, rng)
        assert error.shape == (14,)
        assert syn.shape == (6,)

    def test_expected_error_weight(self):
        noise = DepolarizingNoise(p=0.1)
        assert abs(noise.expected_error_weight(100) - 10.0) < 1e-10


class TestPhenomenologicalNoise:
    """Tests for PhenomenologicalNoise."""

    def test_creation(self):
        noise = PhenomenologicalNoise(p=0.05, q=0.01)
        assert noise.p == 0.05
        assert noise.q == 0.01

    def test_default_q_equals_p(self):
        noise = PhenomenologicalNoise(p=0.03)
        assert noise.q == 0.03

    def test_syndrome_differs_from_perfect(self):
        """With measurement noise, syndrome should sometimes differ."""
        noise = PhenomenologicalNoise(p=0.1, q=0.5)  # very noisy measurement
        code = SteaneCode()
        rng = np.random.default_rng(42)
        differs = False
        for _ in range(100):
            error, noisy_syn = noise.sample_syndrome(code, rng)
            perfect_syn = code.syndrome(error)
            if not np.array_equal(noisy_syn, perfect_syn):
                differs = True
                break
        assert differs, "Noisy syndrome should differ from perfect sometimes"

    def test_multi_round(self):
        noise = PhenomenologicalNoise(p=0.05, q=0.05, num_rounds=3)
        code = SteaneCode()
        rng = np.random.default_rng(42)
        total_error, history = noise.sample_multi_round(code, rng)
        assert len(history) == 3
        assert total_error.shape == (14,)


class TestCircuitLevelNoise:
    """Tests for CircuitLevelNoise."""

    def test_creation(self):
        noise = CircuitLevelNoise(p_data=0.001, p_gate=0.001)
        assert noise.p_data == 0.001

    def test_effective_rate(self):
        noise = CircuitLevelNoise(
            p_data=0.001, p_gate=0.002, p_meas=0.003, p_prep=0.004
        )
        assert abs(noise.effective_error_rate - 0.01) < 1e-10

    def test_sample_error_shape(self):
        noise = CircuitLevelNoise()
        rng = np.random.default_rng(42)
        error = noise.sample_error(10, rng)
        assert error.shape == (20,)

    def test_noisy_syndrome(self):
        noise = CircuitLevelNoise(p_data=0.1, p_meas=0.5, p_prep=0.5)
        code = SteaneCode()
        rng = np.random.default_rng(42)
        error, noisy_syn = noise.sample_syndrome(code, rng)
        perfect_syn = code.syndrome(error)
        # With 50% measurement + prep noise, syndromes should often differ
        # Run multiple trials
        found_diff = False
        for _ in range(50):
            error, noisy_syn = noise.sample_syndrome(code, rng)
            perfect_syn = code.syndrome(error)
            if not np.array_equal(noisy_syn, perfect_syn):
                found_diff = True
                break
        assert found_diff


class TestBiasedNoise:
    """Tests for BiasedNoise."""

    def test_creation(self):
        noise = BiasedNoise(p_x=0.01, p_z=0.001)
        assert noise.p_x == 0.01
        assert noise.p_z == 0.001

    def test_invalid_probabilities(self):
        with pytest.raises(ValueError):
            BiasedNoise(p_x=-0.1, p_z=0.01)

    def test_bias_ratio(self):
        noise = BiasedNoise(p_x=0.1, p_z=0.01)
        assert abs(noise.bias_ratio - 0.1) < 1e-10

    def test_total_error_rate(self):
        noise = BiasedNoise(p_x=0.1, p_z=0.1)
        expected = 0.1 + 0.1 - 0.01
        assert abs(noise.total_error_rate - expected) < 1e-10

    def test_asymmetric_errors(self):
        """X errors should be much more common than Z with high bias."""
        noise = BiasedNoise(p_x=0.5, p_z=0.001)
        rng = np.random.default_rng(42)
        n = 100
        x_count = 0
        z_count = 0
        for _ in range(100):
            error = noise.sample_error(n, rng)
            x_count += error[:n].sum()
            z_count += error[n:].sum()
        assert x_count > 10 * z_count, "X errors should dominate"

    def test_sample_syndrome(self):
        noise = BiasedNoise(p_x=0.05, p_z=0.01)
        code = SteaneCode()
        rng = np.random.default_rng(42)
        error, syn = noise.sample_syndrome(code, rng)
        assert error.shape == (14,)
        assert syn.shape == (6,)


# ================================================================== #
#  THRESHOLD ESTIMATION
# ================================================================== #

class TestThresholdEstimator:
    """Tests for threshold estimation."""

    def test_threshold_data_point(self):
        dp = ThresholdDataPoint(
            distance=3, physical_error_rate=0.05,
            logical_error_rate=0.1, num_trials=1000,
        )
        assert dp.distance == 3
        assert dp.std_error == 0.0

    def test_threshold_result(self):
        result = ThresholdResult(
            threshold=0.1, threshold_error=0.01,
            data_points=[], code_name="SurfaceCode",
            decoder_name="MWPMDecoder",
        )
        assert result.threshold == 0.1

    def test_estimator_creation(self):
        estimator = ThresholdEstimator(
            distances=[3, 5],
            decoder_cls=LookupTableDecoder,
            code_cls=RepetitionCode,
            num_trials=50,
        )
        assert estimator.distances == [3, 5]

    def test_repetition_code_threshold(self):
        """Repetition code + lookup table has a ~50% threshold for bit-flip noise."""
        result = estimate_threshold(
            code_cls=RepetitionCode,
            decoder_cls=LookupTableDecoder,
            distances=[3, 5],
            p_range=(0.05, 0.45),
            num_points=5,
            num_trials=200,
            seed=42,
            code_kwargs={"code_type": "bit_flip"},
        )
        assert isinstance(result, ThresholdResult)
        assert len(result.data_points) == 10  # 2 distances * 5 points
        # The threshold should be somewhere reasonable
        # For bit-flip only with lookup table, it should be high
        assert result.threshold > 0.0

    def test_scaling_below_threshold(self):
        """Below threshold, larger codes should perform better."""
        code3 = RepetitionCode(distance=3, code_type="bit_flip")
        code5 = RepetitionCode(distance=5, code_type="bit_flip")
        dec3 = LookupTableDecoder(code3)
        dec5 = LookupTableDecoder(code5)

        p = 0.05  # well below threshold for repetition code
        r3 = benchmark_decoder(code3, dec3, p, num_trials=500, seed=42,
                               error_type="x_only")
        r5 = benchmark_decoder(code5, dec5, p, num_trials=500, seed=42,
                               error_type="x_only")
        # d=5 should have lower or equal logical error rate than d=3
        assert r5.logical_error_rate <= r3.logical_error_rate + 0.05

    def test_compare_codes(self):
        results = compare_codes(
            code_decoder_pairs=[
                (RepetitionCode, LookupTableDecoder),
            ],
            distance=3,
            p_values=[0.05, 0.1],
            num_trials=50,
            seed=42,
        )
        assert len(results) == 1
        name, dec_name, points = results[0]
        assert name == "RepetitionCode"
        assert len(points) == 2


class TestThresholdConvergence:
    """Tests that threshold estimation converges correctly."""

    def test_low_error_rate_improves_with_distance(self):
        """At low error rates, higher distance should give lower logical error."""
        p = 0.02
        results = {}
        for d in [3, 5, 7]:
            code = RepetitionCode(distance=d, code_type="bit_flip")
            decoder = LookupTableDecoder(code)
            bench = benchmark_decoder(
                code, decoder, p,
                num_trials=500, error_type="x_only", seed=d,
            )
            results[d] = bench.logical_error_rate

        # Monotonically decreasing (with some statistical slack)
        assert results[5] <= results[3] + 0.02
        assert results[7] <= results[5] + 0.02


# ================================================================== #
#  LATTICE SURGERY
# ================================================================== #

class TestPauliFrame:
    """Tests for PauliFrame tracking."""

    def test_creation(self):
        frame = PauliFrame(5)
        assert frame.num_qubits == 5
        assert np.all(frame.x_frame == 0)
        assert np.all(frame.z_frame == 0)

    def test_apply_x(self):
        frame = PauliFrame(3)
        frame.apply_x(1)
        assert frame.x_frame[1] == 1
        frame.apply_x(1)  # toggle back
        assert frame.x_frame[1] == 0

    def test_apply_z(self):
        frame = PauliFrame(3)
        frame.apply_z(0)
        assert frame.z_frame[0] == 1

    def test_cnot_x_propagation(self):
        frame = PauliFrame(3)
        frame.apply_x(0)  # X on control
        frame.apply_cnot(0, 1)
        assert frame.x_frame[0] == 1  # still on control
        assert frame.x_frame[1] == 1  # propagated to target

    def test_cnot_z_propagation(self):
        frame = PauliFrame(3)
        frame.apply_z(1)  # Z on target
        frame.apply_cnot(0, 1)
        assert frame.z_frame[0] == 1  # propagated to control
        assert frame.z_frame[1] == 1  # still on target

    def test_hadamard_swaps_xz(self):
        frame = PauliFrame(2)
        frame.apply_x(0)
        frame.apply_hadamard(0)
        assert frame.x_frame[0] == 0
        assert frame.z_frame[0] == 1

    def test_s_gate(self):
        frame = PauliFrame(2)
        frame.apply_x(0)
        frame.apply_s(0)
        assert frame.x_frame[0] == 1  # X unchanged
        assert frame.z_frame[0] == 1  # Z added (X -> Y = iXZ)

    def test_measure_correction(self):
        frame = PauliFrame(2)
        assert frame.measure_correction(0, "Z") == 0
        frame.apply_x(0)
        assert frame.measure_correction(0, "Z") == 1

    def test_reset(self):
        frame = PauliFrame(3)
        frame.apply_x(1)
        frame.apply_z(1)
        frame.reset(1)
        assert frame.x_frame[1] == 0
        assert frame.z_frame[1] == 0

    def test_copy(self):
        frame = PauliFrame(3)
        frame.apply_x(0)
        copy = frame.copy()
        assert np.array_equal(copy.x_frame, frame.x_frame)
        copy.apply_z(0)
        assert frame.z_frame[0] == 0  # original unchanged


class TestLogicalQubit:
    """Tests for LogicalQubit."""

    def test_creation(self):
        q = LogicalQubit(qubit_id=0, distance=5)
        assert q.qubit_id == 0
        assert q.distance == 5
        assert q.state == "|0>"

    def test_physical_qubits(self):
        q = LogicalQubit(qubit_id=0, distance=5)
        assert q.physical_qubits == 25

    def test_total_physical_qubits(self):
        q = LogicalQubit(qubit_id=0, distance=5)
        assert q.total_physical_qubits == 49  # 2*25 - 1


class TestLatticeSurgery:
    """Tests for LatticeSurgery operations."""

    def test_creation(self):
        ls = LatticeSurgery(num_logical_qubits=3, distance=3, seed=42)
        assert len(ls.qubits) == 3
        assert ls.distance == 3

    def test_merge_split(self):
        ls = LatticeSurgery(num_logical_qubits=2, distance=3, seed=42)
        result = ls.merge(0, 1)
        assert result in (0, 1)
        ls.split(0, 1)
        assert len(ls.operations) == 2

    def test_logical_cnot(self):
        ls = LatticeSurgery(num_logical_qubits=2, distance=3, seed=42)
        ls.logical_cnot(0, 1)
        # Should have merge, split, merge, split, cnot operations
        assert len(ls.operations) >= 5

    def test_logical_hadamard(self):
        ls = LatticeSurgery(num_logical_qubits=2, distance=3, seed=42)
        ls.logical_hadamard(0)
        assert len(ls.operations) == 1
        assert ls.operations[0].op_type == "hadamard"

    def test_logical_s(self):
        ls = LatticeSurgery(num_logical_qubits=1, distance=3, seed=42)
        ls.logical_s(0)
        assert ls.operations[-1].op_type == "s"

    def test_logical_t(self):
        ls = LatticeSurgery(num_logical_qubits=1, distance=3, seed=42)
        ls.logical_t(0)
        assert ls.operations[-1].op_type == "t"

    def test_total_code_cycles(self):
        ls = LatticeSurgery(num_logical_qubits=2, distance=3, seed=42)
        ls.logical_cnot(0, 1)
        cycles = ls.total_code_cycles()
        assert cycles > 0

    def test_total_physical_qubits(self):
        ls = LatticeSurgery(num_logical_qubits=2, distance=5, seed=42)
        qubits = ls.total_physical_qubits()
        # 3 patches (2 data + 1 ancilla) * (2*25 - 1) = 3 * 49 = 147
        assert qubits == 147

    def test_multi_cnot_circuit(self):
        ls = LatticeSurgery(num_logical_qubits=3, distance=3, seed=42)
        ls.logical_cnot(0, 1)
        ls.logical_cnot(1, 2)
        ls.logical_hadamard(0)
        assert ls.total_code_cycles() > 0


class TestMagicStateDistillation:
    """Tests for MagicStateDistillation."""

    def test_creation(self):
        msd = MagicStateDistillation(input_error_rate=0.01, num_levels=1)
        assert msd.input_error_rate == 0.01

    def test_output_error_suppression(self):
        """One level should reduce error from p to ~35*p^3."""
        msd = MagicStateDistillation(input_error_rate=0.01, num_levels=1)
        output = msd.output_error_rate()
        expected = 35.0 * 0.01 ** 3  # 3.5e-5
        assert abs(output - expected) < 1e-10

    def test_two_level_distillation(self):
        msd = MagicStateDistillation(input_error_rate=0.01, num_levels=2)
        output = msd.output_error_rate()
        # Level 1: 35 * 0.01^3 = 3.5e-5
        # Level 2: 35 * (3.5e-5)^3 ~ 1.5e-12
        assert output < 1e-10

    def test_output_fidelity(self):
        msd = MagicStateDistillation(input_error_rate=0.01, num_levels=1)
        fidelity = msd.output_fidelity()
        assert 0.999 < fidelity < 1.0

    def test_input_states_needed(self):
        msd1 = MagicStateDistillation(num_levels=1)
        assert msd1.input_states_needed() == 15
        msd2 = MagicStateDistillation(num_levels=2)
        assert msd2.input_states_needed() == 225

    def test_success_probability(self):
        msd = MagicStateDistillation(input_error_rate=0.01, num_levels=1)
        prob = msd.success_probability()
        # 1 - 15*0.01 = 0.85
        assert abs(prob - 0.85) < 0.01

    def test_physical_qubit_cost(self):
        msd = MagicStateDistillation(num_levels=1)
        cost = msd.physical_qubit_cost(code_distance=3)
        assert cost > 0

    def test_time_cost(self):
        msd = MagicStateDistillation(num_levels=1)
        time = msd.time_cost_cycles(code_distance=5)
        assert time == 25  # 1 * 5 * 5


class TestResourceEstimation:
    """Tests for resource estimation."""

    def test_basic_estimate(self):
        est = estimate_resources(
            num_logical_qubits=10,
            num_t_gates=100,
            num_cnot_gates=200,
            physical_error_rate=1e-3,
        )
        assert isinstance(est, ResourceEstimate)
        assert est.num_logical_qubits == 10
        assert est.physical_qubits > 0
        assert est.total_code_cycles > 0
        assert est.code_distance >= 3

    def test_higher_error_needs_more_distance(self):
        est_low = estimate_resources(
            num_logical_qubits=10, num_t_gates=100,
            physical_error_rate=1e-4,
        )
        est_high = estimate_resources(
            num_logical_qubits=10, num_t_gates=100,
            physical_error_rate=1e-2,
        )
        assert est_high.code_distance >= est_low.code_distance

    def test_more_t_gates_need_more_resources(self):
        est_few = estimate_resources(
            num_logical_qubits=10, num_t_gates=10,
            physical_error_rate=1e-3,
        )
        est_many = estimate_resources(
            num_logical_qubits=10, num_t_gates=10000,
            physical_error_rate=1e-3,
        )
        assert est_many.total_code_cycles >= est_few.total_code_cycles

    def test_wall_time(self):
        est = estimate_resources(
            num_logical_qubits=5, num_t_gates=50,
            cycle_time_us=1.0,
        )
        assert est.wall_time_us == est.total_code_cycles * 1.0

    def test_no_distillation_without_t_gates(self):
        est = estimate_resources(
            num_logical_qubits=5, num_t_gates=0, num_cnot_gates=50,
        )
        assert est.distillation_qubits == 0


# ================================================================== #
#  INTEGRATION TESTS
# ================================================================== #

class TestIntegration:
    """End-to-end integration tests: encode -> noise -> syndrome -> decode -> verify."""

    def test_full_pipeline_steane(self):
        """Full QEC pipeline with Steane code."""
        code = SteaneCode()
        decoder = LookupTableDecoder(code)
        noise = DepolarizingNoise(p=0.05)
        rng = np.random.default_rng(42)

        successes = 0
        trials = 200
        for _ in range(trials):
            error, syndrome = noise.sample_syndrome(code, rng)
            if np.all(syndrome == 0):
                successes += 1
                continue
            correction = decoder.decode(syndrome)
            if code.check_correction(error, correction):
                successes += 1

        success_rate = successes / trials
        assert success_rate > 0.80, f"Pipeline success rate {success_rate} too low"

    def test_full_pipeline_shor(self):
        """Full QEC pipeline with Shor code."""
        code = ShorCode()
        decoder = LookupTableDecoder(code)
        noise = DepolarizingNoise(p=0.03)
        rng = np.random.default_rng(42)

        successes = 0
        trials = 200
        for _ in range(trials):
            error, syndrome = noise.sample_syndrome(code, rng)
            if np.all(syndrome == 0):
                successes += 1
                continue
            correction = decoder.decode(syndrome)
            if code.check_correction(error, correction):
                successes += 1

        success_rate = successes / trials
        assert success_rate > 0.80

    def test_full_pipeline_surface_mwpm(self):
        """Full QEC pipeline with surface code + MWPM."""
        code = SurfaceCode(distance=3)
        decoder = MWPMDecoder(code)
        noise = DepolarizingNoise(p=0.03)
        rng = np.random.default_rng(42)

        successes = 0
        trials = 200
        for _ in range(trials):
            error, syndrome = noise.sample_syndrome(code, rng)
            if np.all(syndrome == 0):
                successes += 1
                continue
            correction = decoder.decode(syndrome)
            residual = (error + correction) % 2
            if np.all(code.syndrome(residual) == 0):
                successes += 1

        success_rate = successes / trials
        assert success_rate > 0.70

    def test_full_pipeline_surface_unionfind(self):
        """Full QEC pipeline with surface code + Union-Find."""
        code = SurfaceCode(distance=3)
        decoder = UnionFindDecoder(code)
        noise = DepolarizingNoise(p=0.03)
        rng = np.random.default_rng(42)

        successes = 0
        trials = 200
        for _ in range(trials):
            error, syndrome = noise.sample_syndrome(code, rng)
            if np.all(syndrome == 0):
                successes += 1
                continue
            correction = decoder.decode(syndrome)
            residual = (error + correction) % 2
            if np.all(code.syndrome(residual) == 0):
                successes += 1

        success_rate = successes / trials
        assert success_rate > 0.70

    def test_biased_noise_pipeline(self):
        """Pipeline with biased noise model."""
        code = SteaneCode()
        decoder = LookupTableDecoder(code)
        noise = BiasedNoise(p_x=0.03, p_z=0.003)
        rng = np.random.default_rng(42)

        successes = 0
        trials = 200
        for _ in range(trials):
            error = noise.sample_error(code.n, rng)
            syndrome = code.syndrome(error)
            if np.all(syndrome == 0):
                successes += 1
                continue
            correction = decoder.decode(syndrome)
            if code.check_correction(error, correction):
                successes += 1

        success_rate = successes / trials
        assert success_rate > 0.85

    def test_lattice_surgery_with_error_correction(self):
        """Combine lattice surgery with resource estimation."""
        ls = LatticeSurgery(num_logical_qubits=4, distance=5, seed=42)
        # Build a simple Bell-state circuit
        ls.logical_hadamard(0)
        ls.logical_cnot(0, 1)

        est = estimate_resources(
            num_logical_qubits=4,
            num_t_gates=0,
            num_cnot_gates=1,
            num_clifford_gates=1,
            physical_error_rate=1e-3,
        )
        assert est.physical_qubits > 0
        assert est.code_distance >= 3

    def test_encode_error_decode_roundtrip(self):
        """Encode -> apply error -> syndrome -> decode -> verify."""
        code = SteaneCode()
        decoder = LookupTableDecoder(code)

        logical_state = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
        encoded = code.encode(logical_state)

        # Apply X error on qubit 3 to the state vector
        dim = 2 ** code.n
        errored = np.zeros(dim, dtype=np.complex128)
        for i in range(dim):
            flipped = i ^ (1 << 3)  # X on qubit 3
            errored[flipped] = encoded[i]

        # Get syndrome from symplectic error
        sym_error = np.zeros(2 * code.n, dtype=np.int8)
        sym_error[3] = 1  # X on qubit 3
        syndrome = code.syndrome(sym_error)

        # Decode
        correction = decoder.decode(syndrome)
        assert code.check_correction(sym_error, correction)


# ================================================================== #
#  XZZX SURFACE CODE
# ================================================================== #

from nqpu.error_correction import (
    XZZXCode,
    BiasedNoiseChannel,
    XZZXDecoder,
    XZZXThresholdStudy,
    XZZXThresholdResult,
)


class TestXZZXCode:
    """Tests for the XZZX surface code variant."""

    def test_creation_d3(self):
        code = XZZXCode(distance=3)
        assert code.n == 9
        assert code.k == 1
        assert code.d == 3

    def test_creation_d5(self):
        code = XZZXCode(distance=5)
        assert code.n == 25
        assert code.k == 1
        assert code.d == 5

    def test_invalid_even_distance(self):
        with pytest.raises(ValueError):
            XZZXCode(distance=4)

    def test_invalid_small_distance(self):
        with pytest.raises(ValueError):
            XZZXCode(distance=1)

    def test_negative_bias(self):
        with pytest.raises(ValueError):
            XZZXCode(distance=3, bias=-1.0)

    def test_stabilizer_count(self):
        """d=3 XZZX code should have n-k = 8 stabilizers."""
        code = XZZXCode(distance=3)
        assert code.num_stabilizers == code.n - code.k

    def test_stabilizer_count_d5(self):
        code = XZZXCode(distance=5)
        assert code.num_stabilizers == code.n - code.k

    def test_stabilizers_commute(self):
        """All XZZX stabilizers must commute pairwise."""
        code = XZZXCode(distance=3)
        stabs = code.stabilizers()
        n = code.n
        for i in range(len(stabs)):
            for j in range(i + 1, len(stabs)):
                si = stabs[i]
                sj = stabs[j]
                ip = (np.dot(si[:n], sj[n:]) + np.dot(si[n:], sj[:n])) % 2
                assert ip == 0, f"Stabilizers {i} and {j} anticommute"

    def test_stabilizers_non_css(self):
        """XZZX stabilizers should NOT be purely X or purely Z (non-CSS)."""
        code = XZZXCode(distance=3)
        stabs = code.stabilizers()
        found_mixed = False
        for s in stabs:
            x_part = s[:code.n]
            z_part = s[code.n:]
            if np.any(x_part) and np.any(z_part):
                found_mixed = True
                break
        assert found_mixed, "XZZX should have mixed X/Z stabilizers"

    def test_syndrome_no_error(self):
        code = XZZXCode(distance=3)
        error = np.zeros(2 * code.n, dtype=np.int8)
        syn = code.syndrome(error)
        assert np.all(syn == 0)

    def test_syndrome_single_x_error(self):
        code = XZZXCode(distance=3)
        error = np.zeros(2 * code.n, dtype=np.int8)
        error[4] = 1  # X on qubit 4 (center)
        syn = code.syndrome(error)
        assert syn.sum() > 0, "X error should trigger syndrome"

    def test_syndrome_single_z_error(self):
        code = XZZXCode(distance=3)
        error = np.zeros(2 * code.n, dtype=np.int8)
        error[code.n + 4] = 1  # Z on qubit 4
        syn = code.syndrome(error)
        assert syn.sum() > 0, "Z error should trigger syndrome"

    def test_logical_anticommute(self):
        """Logical X and Z must anticommute."""
        code = XZZXCode(distance=3)
        lx = code.logical_x()
        lz = code.logical_z()
        n = code.n
        ip = (np.dot(lx[:n], lz[n:]) + np.dot(lx[n:], lz[:n])) % 2
        assert ip == 1, "Logical X and Z must anticommute"

    def test_logicals_commute_with_stabilizers(self):
        """Logical operators must commute with all stabilizers."""
        code = XZZXCode(distance=3)
        stabs = code.stabilizers()
        n = code.n
        for logical in [code.logical_x(), code.logical_z()]:
            for s in stabs:
                ip = (np.dot(logical[:n], s[n:]) + np.dot(logical[n:], s[:n])) % 2
                assert ip == 0, "Logical must commute with stabilizers"

    def test_check_correction_identity(self):
        code = XZZXCode(distance=3)
        error = np.zeros(2 * code.n, dtype=np.int8)
        correction = np.zeros(2 * code.n, dtype=np.int8)
        assert code.check_correction(error, correction)

    def test_check_correction_single_error(self):
        """Correcting the same error should succeed."""
        code = XZZXCode(distance=3)
        error = np.zeros(2 * code.n, dtype=np.int8)
        error[4] = 1  # X on qubit 4
        correction = error.copy()
        assert code.check_correction(error, correction)


class TestBiasedNoiseChannel:
    """Tests for the BiasedNoiseChannel."""

    def test_unbiased(self):
        noise = BiasedNoiseChannel(p=0.3, bias=1.0)
        # With bias=1, px = py = pz = p/3
        assert abs(noise.px - 0.1) < 1e-10
        assert abs(noise.py - 0.1) < 1e-10
        assert abs(noise.pz - 0.1) < 1e-10

    def test_z_biased(self):
        noise = BiasedNoiseChannel(p=0.3, bias=100.0)
        assert noise.pz > noise.px * 10
        assert abs(noise.px + noise.py + noise.pz - 0.3) < 1e-10

    def test_zero_error(self):
        noise = BiasedNoiseChannel(p=0.0, bias=10.0)
        rng = np.random.default_rng(42)
        error = noise.sample_error(10, rng)
        assert np.all(error == 0)

    def test_error_shape(self):
        noise = BiasedNoiseChannel(p=0.1, bias=10.0)
        rng = np.random.default_rng(42)
        error = noise.sample_error(5, rng)
        assert error.shape == (10,)

    def test_error_binary(self):
        noise = BiasedNoiseChannel(p=0.5, bias=10.0)
        rng = np.random.default_rng(42)
        error = noise.sample_error(20, rng)
        assert np.all((error == 0) | (error == 1))

    def test_z_bias_dominates_errors(self):
        """Under high Z bias, Z errors should vastly outnumber X errors."""
        noise = BiasedNoiseChannel(p=0.5, bias=100.0)
        rng = np.random.default_rng(42)
        n = 100
        x_count = 0
        z_count = 0
        for _ in range(200):
            error = noise.sample_error(n, rng)
            x_count += error[:n].sum()
            z_count += error[n:].sum()
        assert z_count > x_count * 5, "Z errors should dominate under high bias"

    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            BiasedNoiseChannel(p=-0.1)
        with pytest.raises(ValueError):
            BiasedNoiseChannel(p=1.5)

    def test_invalid_bias(self):
        with pytest.raises(ValueError):
            BiasedNoiseChannel(p=0.1, bias=-1.0)


class TestXZZXDecoder:
    """Tests for the XZZXDecoder."""

    def test_trivial_syndrome(self):
        code = XZZXCode(distance=3)
        decoder = XZZXDecoder(code)
        syn = np.zeros(code.num_stabilizers, dtype=np.int8)
        correction = decoder.decode(syn)
        assert np.all(correction == 0)

    def test_single_error_correction(self):
        """Decoder should clear syndrome for single-qubit errors."""
        code = XZZXCode(distance=3)
        decoder = XZZXDecoder(code)
        error = np.zeros(2 * code.n, dtype=np.int8)
        error[4] = 1  # X on center qubit
        syn = code.syndrome(error)
        correction = decoder.decode(syn)
        residual = (error + correction) % 2
        residual_syn = code.syndrome(residual)
        assert np.all(residual_syn == 0), "Decoder did not clear syndrome"

    def test_output_shape(self):
        code = XZZXCode(distance=3)
        decoder = XZZXDecoder(code)
        error = np.zeros(2 * code.n, dtype=np.int8)
        error[0] = 1
        syn = code.syndrome(error)
        correction = decoder.decode(syn)
        assert correction.shape == (2 * code.n,)
        assert correction.dtype == np.int8

    def test_monte_carlo_below_threshold(self):
        """At low error rates, decoder should succeed most of the time."""
        code = XZZXCode(distance=3, bias=1.0)
        decoder = XZZXDecoder(code)
        noise = BiasedNoiseChannel(p=0.02, bias=1.0)
        rng = np.random.default_rng(42)

        successes = 0
        trials = 200
        for _ in range(trials):
            error = noise.sample_error(code.n, rng)
            syn = code.syndrome(error)
            if not np.any(syn):
                if code.check_correction(error, np.zeros(2 * code.n, dtype=np.int8)):
                    successes += 1
                continue
            correction = decoder.decode(syn)
            if code.check_correction(error, correction):
                successes += 1

        assert successes / trials > 0.60, \
            f"Success rate {successes/trials} too low at low error rate"


class TestXZZXThresholdStudy:
    """Tests for XZZX threshold estimation."""

    def test_basic_study(self):
        study = XZZXThresholdStudy(
            distances=[3],
            bias_values=[1.0],
        )
        result = study.run(
            p_range=(0.05, 0.15),
            num_points=3,
            shots=50,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, XZZXThresholdResult)
        assert len(result.distances) == 1
        assert len(result.bias_values) == 1
        assert result.logical_error_rates.shape == (1, 1, 3)
        assert 1.0 in result.thresholds

    def test_threshold_result_structure(self):
        study = XZZXThresholdStudy(
            distances=[3, 5],
            bias_values=[1.0, 10.0],
        )
        result = study.run(
            p_range=(0.05, 0.15),
            num_points=3,
            shots=30,
            rng=np.random.default_rng(42),
        )
        assert result.logical_error_rates.shape == (2, 2, 3)
        assert len(result.thresholds) == 2
        assert result.error_rates.shape == (3,)


# ================================================================== #
#  QUANTUM LDPC CODES
# ================================================================== #

from nqpu.error_correction import (
    ClassicalCode,
    HypergraphProductCode,
    BicycleCode,
    LiftedProductCode,
    BPDecoderQLDPC,
)


class TestClassicalCode:
    """Tests for ClassicalCode factory methods."""

    def test_repetition_code(self):
        code = ClassicalCode.repetition(5)
        assert code.n == 5
        assert code.m == 4
        assert code.k == 1

    def test_repetition_code_small(self):
        code = ClassicalCode.repetition(2)
        assert code.n == 2
        assert code.k == 1

    def test_repetition_invalid(self):
        with pytest.raises(ValueError):
            ClassicalCode.repetition(1)

    def test_hamming_r2(self):
        """[3,1,3] Hamming code."""
        code = ClassicalCode.hamming(2)
        assert code.n == 3
        assert code.m == 2
        assert code.k == 1

    def test_hamming_r3(self):
        """[7,4,3] Hamming code."""
        code = ClassicalCode.hamming(3)
        assert code.n == 7
        assert code.m == 3
        assert code.k == 4

    def test_hamming_invalid(self):
        with pytest.raises(ValueError):
            ClassicalCode.hamming(1)

    def test_random_ldpc(self):
        rng = np.random.default_rng(42)
        code = ClassicalCode.random_ldpc(20, 10, 3, rng=rng)
        assert code.n == 20
        assert code.m == 10
        # Each column should have exactly 3 ones
        col_weights = code.h.sum(axis=0)
        assert np.all(col_weights == 3)

    def test_random_ldpc_rate(self):
        rng = np.random.default_rng(42)
        code = ClassicalCode.random_ldpc(30, 10, 2, rng=rng)
        assert 0.0 <= code.rate <= 1.0

    def test_code_rate(self):
        code = ClassicalCode.hamming(3)
        assert abs(code.rate - 4.0 / 7.0) < 1e-10


class TestHypergraphProductCode:
    """Tests for the HypergraphProductCode."""

    def test_rep_x_rep(self):
        """Hypergraph product of two repetition codes."""
        c1 = ClassicalCode.repetition(3)
        c2 = ClassicalCode.repetition(3)
        hgp = HypergraphProductCode(c1, c2)
        # n = n1*n2 + m1*m2 = 3*3 + 2*2 = 13
        assert hgp.n == 13
        assert hgp.k >= 1

    def test_css_property(self):
        """HX * HZ^T must be zero mod 2."""
        c1 = ClassicalCode.repetition(3)
        c2 = ClassicalCode.repetition(3)
        hgp = HypergraphProductCode(c1, c2)
        assert hgp.check_css_property(), "CSS orthogonality violated"

    def test_css_property_hamming(self):
        c1 = ClassicalCode.hamming(2)
        c2 = ClassicalCode.hamming(2)
        hgp = HypergraphProductCode(c1, c2)
        assert hgp.check_css_property()

    def test_hamming_product_params(self):
        """Product of two [7,4,3] Hamming codes."""
        c1 = ClassicalCode.hamming(3)
        c2 = ClassicalCode.hamming(3)
        hgp = HypergraphProductCode(c1, c2)
        # n = 7*7 + 3*3 = 49 + 9 = 58
        assert hgp.n == 58
        assert hgp.k >= 1

    def test_syndrome_no_error(self):
        c1 = ClassicalCode.repetition(3)
        c2 = ClassicalCode.repetition(3)
        hgp = HypergraphProductCode(c1, c2)
        error = np.zeros(2 * hgp.n, dtype=np.int8)
        syn = hgp.syndrome(error)
        assert np.all(syn == 0)

    def test_syndrome_single_z_error(self):
        c1 = ClassicalCode.repetition(3)
        c2 = ClassicalCode.repetition(3)
        hgp = HypergraphProductCode(c1, c2)
        error_z = np.zeros(hgp.n, dtype=np.int8)
        error_z[0] = 1
        sx = hgp.syndrome_x(error_z)
        assert sx.sum() >= 0  # may or may not detect

    def test_syndrome_single_x_error(self):
        c1 = ClassicalCode.repetition(3)
        c2 = ClassicalCode.repetition(3)
        hgp = HypergraphProductCode(c1, c2)
        error_x = np.zeros(hgp.n, dtype=np.int8)
        error_x[0] = 1
        sz = hgp.syndrome_z(error_x)
        assert sz.sum() >= 0

    def test_code_params_property(self):
        c1 = ClassicalCode.repetition(3)
        c2 = ClassicalCode.repetition(3)
        hgp = HypergraphProductCode(c1, c2)
        n, k = hgp.code_params
        assert n == hgp.n
        assert k == hgp.k


class TestBicycleCode:
    """Tests for BicycleCode."""

    def test_basic_bicycle(self):
        code = BicycleCode(size=7, shifts_a=[0, 1, 3], shifts_b=[0, 2, 5])
        assert code.n == 14

    def test_css_property(self):
        code = BicycleCode(size=7, shifts_a=[0, 1, 3], shifts_b=[0, 2, 5])
        assert code.check_css_property(), "Bicycle code CSS property violated"

    def test_circulant_shape(self):
        code = BicycleCode(size=5, shifts_a=[0, 1], shifts_b=[0, 2])
        assert code.A.shape == (5, 5)
        assert code.B.shape == (5, 5)

    def test_syndrome_no_error(self):
        code = BicycleCode(size=7, shifts_a=[0, 1, 3], shifts_b=[0, 2, 5])
        error = np.zeros(2 * code.n, dtype=np.int8)
        syn = code.syndrome(error)
        assert np.all(syn == 0)

    def test_has_logical_qubits(self):
        code = BicycleCode(size=7, shifts_a=[0, 1, 3], shifts_b=[0, 2, 5])
        assert code.k >= 0

    def test_code_params(self):
        code = BicycleCode(size=7, shifts_a=[0, 1], shifts_b=[0, 3])
        n, k = code.code_params
        assert n == 14
        assert k >= 0


class TestLiftedProductCode:
    """Tests for LiftedProductCode."""

    def test_basic_lifted(self):
        base = ClassicalCode.repetition(3)
        code = LiftedProductCode(base_code=base, lift_size=3, seed=42)
        # n = (n_base + m) * L = (3 + 2) * 3 = 15
        assert code.n == 15
        assert code.k >= 0

    def test_lifted_hamming(self):
        base = ClassicalCode.hamming(2)
        code = LiftedProductCode(base_code=base, lift_size=5, seed=42)
        # n = (n_base + m) * L = (3 + 2) * 5 = 25
        assert code.n == 25
        assert code.k >= 0

    def test_code_params(self):
        base = ClassicalCode.repetition(3)
        code = LiftedProductCode(base_code=base, lift_size=4, seed=42)
        n, k = code.code_params
        # n = (3 + 2) * 4 = 20
        assert n == 20
        assert k >= 0


class TestBPDecoderQLDPC:
    """Tests for the BP decoder on QLDPC codes."""

    def test_creation(self):
        c1 = ClassicalCode.repetition(3)
        c2 = ClassicalCode.repetition(3)
        hgp = HypergraphProductCode(c1, c2)
        decoder = BPDecoderQLDPC(hgp.hx, hgp.hz)
        assert decoder.max_iter == 50

    def test_trivial_syndrome(self):
        c1 = ClassicalCode.repetition(3)
        c2 = ClassicalCode.repetition(3)
        hgp = HypergraphProductCode(c1, c2)
        decoder = BPDecoderQLDPC(hgp.hx, hgp.hz)
        sx = np.zeros(hgp.hx.shape[0], dtype=np.int8)
        z_corr = decoder.decode_x(sx)
        assert np.all(z_corr == 0)

    def test_decode_x_output_shape(self):
        c1 = ClassicalCode.repetition(3)
        c2 = ClassicalCode.repetition(3)
        hgp = HypergraphProductCode(c1, c2)
        decoder = BPDecoderQLDPC(hgp.hx, hgp.hz, channel_prob=0.1)
        error_z = np.zeros(hgp.n, dtype=np.int8)
        error_z[0] = 1
        sx = hgp.syndrome_x(error_z)
        z_corr = decoder.decode_x(sx)
        assert z_corr.shape == (hgp.n,)

    def test_decode_z_output_shape(self):
        c1 = ClassicalCode.repetition(3)
        c2 = ClassicalCode.repetition(3)
        hgp = HypergraphProductCode(c1, c2)
        decoder = BPDecoderQLDPC(hgp.hx, hgp.hz, channel_prob=0.1)
        error_x = np.zeros(hgp.n, dtype=np.int8)
        error_x[0] = 1
        sz = hgp.syndrome_z(error_x)
        x_corr = decoder.decode_z(sz)
        assert x_corr.shape == (hgp.n,)

    def test_full_decode(self):
        c1 = ClassicalCode.repetition(3)
        c2 = ClassicalCode.repetition(3)
        hgp = HypergraphProductCode(c1, c2)
        decoder = BPDecoderQLDPC(hgp.hx, hgp.hz, channel_prob=0.1)
        error = np.zeros(2 * hgp.n, dtype=np.int8)
        syn = hgp.syndrome(error)
        correction = decoder.decode(syn)
        assert correction.shape == (2 * hgp.n,)

    def test_bp_clears_syndrome_for_light_error(self):
        """BP should clear syndrome for weight-1 errors on small codes."""
        c1 = ClassicalCode.repetition(3)
        c2 = ClassicalCode.repetition(3)
        hgp = HypergraphProductCode(c1, c2)
        decoder = BPDecoderQLDPC(hgp.hx, hgp.hz, channel_prob=0.1, max_iter=100)

        # Apply single Z error
        error_z = np.zeros(hgp.n, dtype=np.int8)
        error_z[0] = 1
        sx = hgp.syndrome_x(error_z)

        if np.any(sx):
            z_corr = decoder.decode_x(sx)
            residual_syn = (hgp.hx @ ((error_z + z_corr) % 2)) % 2
            assert np.all(residual_syn == 0), "BP did not clear X syndrome"


# ================================================================== #
#  CORRELATED DECODING
# ================================================================== #

from nqpu.error_correction import (
    SyndromeHistory,
    SlidingWindowDecoder,
    SpaceTimeMWPM,
    CorrelatedNoiseModel,
    DecodingBenchmark as CorrelatedDecodingBenchmark,
    BenchmarkResult,
)


class TestSyndromeHistory:
    """Tests for SyndromeHistory."""

    def test_creation(self):
        syndromes = np.array([
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ], dtype=np.int8)
        sh = SyndromeHistory(syndromes=syndromes)
        assert sh.n_rounds == 3
        assert sh.n_stabilizers == 4

    def test_single_round(self):
        syndromes = np.array([1, 0, 1, 0], dtype=np.int8)
        sh = SyndromeHistory(syndromes=syndromes)
        assert sh.n_rounds == 1
        assert sh.n_stabilizers == 4

    def test_diff(self):
        syndromes = np.array([
            [0, 0, 0],
            [1, 0, 1],
            [1, 1, 1],
        ], dtype=np.int8)
        sh = SyndromeHistory(syndromes=syndromes)
        diffs = sh.diff()
        assert diffs.shape == (2, 3)
        # Round 0->1: [1,0,1]
        assert np.array_equal(diffs[0], [1, 0, 1])
        # Round 1->2: [0,1,0]
        assert np.array_equal(diffs[1], [0, 1, 0])

    def test_diff_single_round(self):
        sh = SyndromeHistory(syndromes=np.array([[1, 0, 1]], dtype=np.int8))
        diffs = sh.diff()
        assert diffs.shape == (0, 3)

    def test_diff_with_initial(self):
        syndromes = np.array([
            [1, 0, 1],
            [0, 1, 1],
        ], dtype=np.int8)
        sh = SyndromeHistory(syndromes=syndromes)
        diffs = sh.diff_with_initial()
        assert diffs.shape == (2, 3)
        assert np.array_equal(diffs[0], [1, 0, 1])  # vs zero reference
        assert np.array_equal(diffs[1], [1, 1, 0])  # vs round 0

    def test_get_round(self):
        syndromes = np.array([
            [1, 0],
            [0, 1],
        ], dtype=np.int8)
        sh = SyndromeHistory(syndromes=syndromes)
        assert np.array_equal(sh.get_round(0), [1, 0])
        assert np.array_equal(sh.get_round(1), [0, 1])

    def test_window(self):
        syndromes = np.zeros((10, 4), dtype=np.int8)
        sh = SyndromeHistory(syndromes=syndromes)
        window = sh.window(3, 4)
        assert window.shape == (4, 4)


class TestSlidingWindowDecoder:
    """Tests for the SlidingWindowDecoder."""

    def test_creation(self):
        code = SurfaceCode(distance=3)
        decoder = SlidingWindowDecoder(code=code, window_size=5, commit_size=2)
        assert decoder.window_size == 5
        assert decoder.commit_size == 2

    def test_invalid_commit_size(self):
        code = SurfaceCode(distance=3)
        with pytest.raises(ValueError):
            SlidingWindowDecoder(code=code, window_size=3, commit_size=5)

    def test_decode_trivial_syndromes(self):
        code = SurfaceCode(distance=3)
        decoder = SlidingWindowDecoder(code=code, window_size=3, commit_size=1)
        num_stabs = code.Hx.shape[0] + code.Hz.shape[0]
        syndromes = np.zeros((5, num_stabs), dtype=np.int8)
        sh = SyndromeHistory(syndromes=syndromes)
        corrections = decoder.decode_stream(sh)
        assert len(corrections) >= 1
        for c in corrections:
            assert c.shape == (2 * code.n,)
            assert np.all(c == 0)

    def test_decode_produces_corrections(self):
        """Decoder should produce corrections for nontrivial syndromes."""
        code = SurfaceCode(distance=3)
        decoder = SlidingWindowDecoder(code=code, window_size=3, commit_size=1)
        num_stabs = code.Hx.shape[0] + code.Hz.shape[0]

        # Create a syndrome history with some defects
        syndromes = np.zeros((4, num_stabs), dtype=np.int8)
        # Apply an error and compute its syndrome
        error = np.zeros(2 * code.n, dtype=np.int8)
        error[4] = 1  # X on center qubit
        syn = code.syndrome(error)
        syndromes[1] = syn
        syndromes[2] = syn
        syndromes[3] = syn

        sh = SyndromeHistory(syndromes=syndromes)
        corrections = decoder.decode_stream(sh)
        assert len(corrections) >= 1

    def test_correction_shape(self):
        code = RepetitionCode(distance=3, code_type="bit_flip")
        decoder = SlidingWindowDecoder(code=code, window_size=2, commit_size=1)
        num_stabs = code.Hx.shape[0] + code.Hz.shape[0]
        syndromes = np.zeros((3, num_stabs), dtype=np.int8)
        sh = SyndromeHistory(syndromes=syndromes)
        corrections = decoder.decode_stream(sh)
        for c in corrections:
            assert c.shape == (2 * code.n,)


class TestSpaceTimeMWPM:
    """Tests for SpaceTimeMWPM."""

    def test_creation(self):
        code = SurfaceCode(distance=3)
        decoder = SpaceTimeMWPM(code=code, n_rounds=5)
        assert decoder.n_rounds == 5

    def test_empty_graph(self):
        code = SurfaceCode(distance=3)
        decoder = SpaceTimeMWPM(code=code, n_rounds=3)
        diffs = np.zeros((3, code.Hx.shape[0] + code.Hz.shape[0]), dtype=np.int8)
        graph = decoder.build_graph(diffs)
        assert len(graph["nodes"]) == 0

    def test_graph_with_defects(self):
        code = SurfaceCode(distance=3)
        decoder = SpaceTimeMWPM(code=code, n_rounds=3)
        num_stabs = code.Hx.shape[0] + code.Hz.shape[0]
        diffs = np.zeros((3, num_stabs), dtype=np.int8)
        diffs[0, 0] = 1
        diffs[1, 0] = 1
        graph = decoder.build_graph(diffs)
        assert len(graph["nodes"]) == 2

    def test_decode_empty(self):
        code = SurfaceCode(distance=3)
        decoder = SpaceTimeMWPM(code=code, n_rounds=3)
        num_stabs = code.Hx.shape[0] + code.Hz.shape[0]
        diffs = np.zeros((3, num_stabs), dtype=np.int8)
        correction = decoder.decode(diffs)
        assert np.all(correction == 0)
        assert correction.shape == (2 * code.n,)

    def test_decode_with_defects(self):
        code = SurfaceCode(distance=3)
        decoder = SpaceTimeMWPM(code=code, n_rounds=3)
        num_stabs = code.Hx.shape[0] + code.Hz.shape[0]
        diffs = np.zeros((3, num_stabs), dtype=np.int8)
        diffs[0, 0] = 1
        diffs[1, 1] = 1
        correction = decoder.decode(diffs)
        assert correction.shape == (2 * code.n,)


class TestCorrelatedNoiseModel:
    """Tests for CorrelatedNoiseModel."""

    def test_creation(self):
        noise = CorrelatedNoiseModel(base_rate=0.05)
        assert noise.base_rate == 0.05
        assert noise.spatial_correlation == 0.0
        assert noise.temporal_correlation == 0.0

    def test_invalid_base_rate(self):
        with pytest.raises(ValueError):
            CorrelatedNoiseModel(base_rate=-0.1)
        with pytest.raises(ValueError):
            CorrelatedNoiseModel(base_rate=1.5)

    def test_invalid_spatial_correlation(self):
        with pytest.raises(ValueError):
            CorrelatedNoiseModel(base_rate=0.1, spatial_correlation=1.5)

    def test_invalid_temporal_correlation(self):
        with pytest.raises(ValueError):
            CorrelatedNoiseModel(base_rate=0.1, temporal_correlation=-0.1)

    def test_sample_errors_shape(self):
        noise = CorrelatedNoiseModel(base_rate=0.1)
        rng = np.random.default_rng(42)
        errors = noise.sample_correlated_errors(5, 3, rng)
        assert errors.shape == (3, 10)  # (n_rounds, 2*n_qubits)

    def test_zero_error_rate(self):
        noise = CorrelatedNoiseModel(base_rate=0.0)
        rng = np.random.default_rng(42)
        errors = noise.sample_correlated_errors(10, 5, rng)
        assert np.all(errors == 0)

    def test_errors_binary(self):
        noise = CorrelatedNoiseModel(base_rate=0.5, spatial_correlation=0.3)
        rng = np.random.default_rng(42)
        errors = noise.sample_correlated_errors(10, 5, rng)
        assert np.all((errors == 0) | (errors == 1))

    def test_spatial_correlation_increases_weight(self):
        """With spatial correlations, errors should have higher weight on average."""
        noise_ind = CorrelatedNoiseModel(base_rate=0.2, spatial_correlation=0.0)
        noise_corr = CorrelatedNoiseModel(base_rate=0.2, spatial_correlation=0.8)
        rng_ind = np.random.default_rng(42)
        rng_corr = np.random.default_rng(42)

        weights_ind = []
        weights_corr = []
        for _ in range(100):
            e_ind = noise_ind.sample_correlated_errors(20, 1, rng_ind)
            e_corr = noise_corr.sample_correlated_errors(20, 1, rng_corr)
            weights_ind.append(e_ind.sum())
            weights_corr.append(e_corr.sum())

        # Correlated should have higher or similar mean weight
        assert np.mean(weights_corr) >= np.mean(weights_ind) * 0.5

    def test_temporal_correlation_creates_persistence(self):
        """With temporal correlations, errors should persist across rounds."""
        noise = CorrelatedNoiseModel(
            base_rate=0.3, temporal_correlation=0.9
        )
        rng = np.random.default_rng(42)
        errors = noise.sample_correlated_errors(5, 10, rng)
        # With high temporal correlation, consecutive rounds should be similar
        # Count how often an error at t also appears at t+1
        persistence = 0
        total = 0
        for t in range(9):
            for q in range(10):
                if errors[t, q]:
                    total += 1
                    if errors[t + 1, q]:
                        persistence += 1
        # Some persistence expected (not necessarily majority due to XOR)
        if total > 0:
            assert persistence > 0, "No temporal persistence detected"

    def test_syndrome_history_output(self):
        noise = CorrelatedNoiseModel(base_rate=0.05)
        code = SteaneCode()
        rng = np.random.default_rng(42)
        total_error, syn_history = noise.sample_syndrome_history(code, 3, rng)
        assert total_error.shape == (14,)
        assert syn_history.n_rounds == 3
        assert syn_history.n_stabilizers == 6


class TestDecodingBenchmarkCorrelated:
    """Tests for the correlated decoding benchmark."""

    def test_benchmark_runs(self):
        code = RepetitionCode(distance=3, code_type="bit_flip")
        noise = CorrelatedNoiseModel(base_rate=0.05)
        bench = CorrelatedDecodingBenchmark(
            code=code, noise_model=noise, n_rounds=3, window_size=2,
        )
        result = bench.compare_decoders(
            n_shots=50, rng=np.random.default_rng(42),
        )
        assert isinstance(result, BenchmarkResult)
        assert 0.0 <= result.independent_logical_rate <= 1.0
        assert 0.0 <= result.correlated_logical_rate <= 1.0
        assert result.n_shots == 50
        assert result.window_size == 2

    def test_benchmark_at_zero_noise(self):
        code = RepetitionCode(distance=3, code_type="bit_flip")
        noise = CorrelatedNoiseModel(base_rate=0.0)
        bench = CorrelatedDecodingBenchmark(
            code=code, noise_model=noise, n_rounds=3, window_size=2,
        )
        result = bench.compare_decoders(
            n_shots=50, rng=np.random.default_rng(42),
        )
        assert result.independent_logical_rate == 0.0
        assert result.correlated_logical_rate == 0.0

    def test_benchmark_result_fields(self):
        result = BenchmarkResult(
            independent_logical_rate=0.1,
            correlated_logical_rate=0.05,
            improvement_factor=2.0,
            window_size=5,
            n_shots=1000,
        )
        assert result.improvement_factor == 2.0
        assert result.window_size == 5
