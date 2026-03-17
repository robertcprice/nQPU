"""Comprehensive tests for the nqpu.mitigation package.

Tests cover: ZNE (gate folding, extrapolation methods), PEC (noise channels,
decomposition, sampling), Pauli twirling (frames, randomized compiling),
readout error correction (calibration, matrix inversion, Bayesian), CDR
(Clifford replacement, model training, correction), and the top-level
mitigate() convenience function.
"""

from __future__ import annotations

import numpy as np
import pytest

from nqpu.mitigation import (
    mitigate,
    # ZNE
    ZNEEstimator,
    ZNEResult,
    NoiseScaler,
    FoldingStrategy,
    ExtrapolationMethod,
    run_zne,
    # PEC
    PECEstimator,
    PECResult,
    PECDecomposition,
    PECOperation,
    NoiseChannel,
    TwoQubitNoiseChannel,
    ChannelType,
    run_pec,
    # Twirling
    PauliTwirler,
    PauliFrame,
    TwirledCircuit,
    RandomizedCompiling,
    twirl_and_average,
    # Readout
    ReadoutCalibration,
    ReadoutCorrector,
    CorrectionMethod,
    correct_counts,
    # CDR
    CDREstimator,
    CDRModel,
    CDRResult,
    CDRTrainingPoint,
    cdr_correct,
    replace_non_clifford,
)


# Gate type alias
Gate = tuple


# ---- Fixtures ----


@pytest.fixture
def simple_circuit():
    """Simple 2-gate circuit on 2 qubits."""
    return [
        ("H", (0,), ()),
        ("CNOT", (0, 1), ()),
    ]


@pytest.fixture
def rotation_circuit():
    """Circuit with non-Clifford rotation gates."""
    return [
        ("H", (0,), ()),
        ("RZ", (0,), (0.7,)),
        ("CNOT", (0, 1), ()),
        ("RX", (1,), (1.2,)),
    ]


@pytest.fixture
def ideal_executor():
    """Executor that always returns 0.85 (ideal expectation)."""
    return lambda circuit: 0.85


@pytest.fixture
def noisy_executor():
    """Executor that returns a noise-decayed value based on circuit length."""
    def executor(circuit):
        noise_factor = 0.98 ** len(circuit)
        return 0.85 * noise_factor
    return executor


@pytest.fixture
def depolarizing_channel():
    """Depolarizing noise channel with 1% error."""
    return NoiseChannel.depolarizing(0.01)


# ---- ZNE Tests ----


class TestNoiseScaler:
    """Tests for gate folding noise amplification."""

    def test_fold_factor_one_no_change(self, simple_circuit):
        scaler = NoiseScaler(FoldingStrategy.LOCAL)
        folded = scaler.fold_gates(simple_circuit, 1)
        assert len(folded) == len(simple_circuit)

    def test_local_folding_triples_length(self, simple_circuit):
        scaler = NoiseScaler(FoldingStrategy.LOCAL)
        folded = scaler.fold_gates(simple_circuit, 3)
        # Each gate becomes g, g_dag, g => 3x
        assert len(folded) == 3 * len(simple_circuit)

    def test_global_folding_triples_length(self, simple_circuit):
        scaler = NoiseScaler(FoldingStrategy.GLOBAL)
        folded = scaler.fold_gates(simple_circuit, 3)
        # U, U_dag, U => 3 * len(circuit)
        assert len(folded) == 3 * len(simple_circuit)

    def test_even_factor_rounded_up(self, simple_circuit):
        scaler = NoiseScaler(FoldingStrategy.LOCAL)
        folded = scaler.fold_gates(simple_circuit, 4)
        # 4 rounds up to 5
        assert len(folded) == 5 * len(simple_circuit)

    def test_invalid_factor(self, simple_circuit):
        scaler = NoiseScaler(FoldingStrategy.LOCAL)
        with pytest.raises(ValueError, match="scale_factor must be"):
            scaler.fold_gates(simple_circuit, 0)

    def test_pulse_stretch(self, simple_circuit):
        scaler = NoiseScaler()
        stretched = scaler.pulse_stretch(simple_circuit, factor=2.0)
        assert len(stretched) == len(simple_circuit)
        for _, _, params in stretched:
            assert params[-1] == 2.0


class TestZNEEstimator:
    """Tests for the ZNE estimation pipeline."""

    def test_linear_extrapolation(self, simple_circuit, noisy_executor):
        estimator = ZNEEstimator(
            noise_factors=[1, 3, 5],
            method=ExtrapolationMethod.LINEAR,
        )
        result = estimator.estimate(simple_circuit, noisy_executor)
        assert isinstance(result, ZNEResult)
        assert np.isfinite(result.estimated_value)
        assert len(result.raw_values) == 3

    @pytest.mark.parametrize("method", [
        ExtrapolationMethod.LINEAR,
        ExtrapolationMethod.POLYNOMIAL,
        ExtrapolationMethod.EXPONENTIAL,
        ExtrapolationMethod.RICHARDSON,
    ])
    def test_all_extrapolation_methods(self, simple_circuit, noisy_executor, method):
        estimator = ZNEEstimator(
            noise_factors=[1, 3, 5],
            method=method,
            poly_degree=2,
        )
        result = estimator.estimate(simple_circuit, noisy_executor)
        assert np.isfinite(result.estimated_value)
        assert result.method == method

    def test_empty_noise_factors_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            ZNEEstimator(noise_factors=[])

    def test_negative_noise_factor_raises(self):
        with pytest.raises(ValueError, match="noise factor must be >= 1"):
            ZNEEstimator(noise_factors=[0, 3])

    def test_run_zne_convenience(self, simple_circuit, noisy_executor):
        result = run_zne(simple_circuit, noisy_executor)
        assert isinstance(result, ZNEResult)
        assert np.isfinite(result.estimated_value)


# ---- PEC Tests ----


class TestNoiseChannel:
    """Tests for noise channel representations."""

    def test_depolarizing_channel(self):
        ch = NoiseChannel.depolarizing(0.03)
        assert abs(ch.error_rate - 0.03) < 1e-10
        assert abs(sum(ch.probabilities.values()) - 1.0) < 1e-10

    def test_pauli_channel(self):
        ch = NoiseChannel.pauli_channel(px=0.01, py=0.02, pz=0.03)
        assert abs(ch.probabilities["I"] - 0.94) < 1e-10

    def test_invalid_probabilities_sum(self):
        with pytest.raises(ValueError, match="sum to 1"):
            NoiseChannel(probabilities={"I": 0.5, "X": 0.5, "Y": 0.5, "Z": 0.0})

    def test_ptm_depolarizing(self):
        ch = NoiseChannel.depolarizing(0.0)
        ptm = ch.pauli_transfer_matrix()
        assert np.allclose(ptm, np.eye(4))

    def test_depolarizing_error_rate_bounds(self):
        with pytest.raises(ValueError):
            NoiseChannel.depolarizing(-0.1)
        with pytest.raises(ValueError):
            NoiseChannel.depolarizing(0.75)


class TestTwoQubitNoiseChannel:
    """Tests for two-qubit noise channels."""

    def test_default_identity_channel(self):
        ch = TwoQubitNoiseChannel()
        total = sum(ch.probabilities.values())
        assert abs(total - 1.0) < 1e-10
        assert abs(ch.probabilities[("I", "I")] - 1.0) < 1e-10

    def test_two_qubit_depolarizing(self):
        ch = TwoQubitNoiseChannel.depolarizing(0.05)
        total = sum(ch.probabilities.values())
        assert abs(total - 1.0) < 1e-10


class TestPECDecomposition:
    """Tests for PEC quasi-probability decomposition."""

    def test_decomposition_from_depolarizing(self):
        decomp = PECDecomposition.from_depolarizing(0.01, qubit=0)
        assert decomp.gamma >= 1.0
        assert len(decomp.operations) == 4

    def test_decomposition_coefficients_sum(self, depolarizing_channel):
        decomp = PECDecomposition.from_noise_channel(depolarizing_channel, qubit=0)
        coeffs = [op.coefficient for op in decomp.operations]
        # Coefficients should reconstruct identity channel inverse
        assert decomp.gamma == pytest.approx(sum(abs(c) for c in coeffs))

    def test_sample_correction(self, depolarizing_channel):
        decomp = PECDecomposition.from_noise_channel(depolarizing_channel, qubit=0)
        rng = np.random.default_rng(42)
        pauli, sign = decomp.sample_correction(rng)
        assert pauli in ("I", "X", "Y", "Z")
        assert sign in (-1.0, 1.0)

    def test_cost_estimate(self, depolarizing_channel):
        decomp = PECDecomposition.from_noise_channel(depolarizing_channel, qubit=0)
        cost = decomp.cost_estimate(circuit_depth=10)
        assert cost >= 1.0


class TestPECEstimator:
    """Tests for PEC Monte Carlo estimation."""

    def test_pec_estimate(self, simple_circuit, noisy_executor, depolarizing_channel):
        estimator = PECEstimator(depolarizing_channel, num_samples=50, seed=42)
        result = estimator.estimate(simple_circuit, noisy_executor)
        assert isinstance(result, PECResult)
        assert np.isfinite(result.estimated_value)
        assert result.num_samples == 50
        assert result.std_error >= 0

    def test_run_pec_convenience(self, simple_circuit, noisy_executor):
        result = run_pec(simple_circuit, noisy_executor, error_rate=0.01,
                         num_samples=50, seed=42)
        assert isinstance(result, PECResult)


# ---- Twirling Tests ----


class TestPauliTwirler:
    """Tests for Pauli twirling frame generation."""

    def test_twirl_gate_produces_frame(self):
        twirler = PauliTwirler(seed=42)
        frame = twirler.twirl_gate("CNOT", ctrl=0, tgt=1)
        assert isinstance(frame, PauliFrame)
        assert 0 in frame.before
        assert 1 in frame.before
        assert frame.before[0] in ("I", "X", "Y", "Z")

    def test_twirl_circuit_generates_variants(self, simple_circuit):
        twirler = PauliTwirler(seed=42)
        twirled = twirler.twirl_circuit(simple_circuit, num_samples=10)
        assert isinstance(twirled, TwirledCircuit)
        assert len(twirled.variants) == 10

    def test_twirl_single_qubit(self):
        twirler = PauliTwirler(seed=42)
        frame = twirler.twirl_single_qubit(qubit=0)
        assert 0 in frame.before
        assert 0 in frame.after

    def test_unsupported_gate_raises(self):
        twirler = PauliTwirler(seed=42)
        with pytest.raises(ValueError, match="No twirling table"):
            twirler.twirl_gate("ISWAP", ctrl=0, tgt=1)


class TestTwirlAndAverage:
    """Tests for the twirl-and-average convenience function."""

    def test_twirl_and_average_returns_tuple(self, simple_circuit, ideal_executor):
        mean, std_err = twirl_and_average(
            simple_circuit, ideal_executor, num_samples=20, seed=42,
        )
        assert np.isfinite(mean)
        assert std_err >= 0


class TestRandomizedCompiling:
    """Tests for randomized compiling pipeline."""

    def test_compile_generates_variants(self, simple_circuit):
        rc = RandomizedCompiling(num_compilations=10, seed=42)
        twirled = rc.compile(simple_circuit)
        assert len(twirled.variants) == 10

    def test_execute_returns_mean_and_stderr(self, simple_circuit, ideal_executor):
        rc = RandomizedCompiling(num_compilations=10, seed=42)
        mean, std_err = rc.execute(simple_circuit, ideal_executor)
        assert np.isfinite(mean)
        assert std_err >= 0


# ---- Readout Correction Tests ----


class TestReadoutCalibration:
    """Tests for readout error calibration."""

    def test_from_symmetric_error(self):
        cal = ReadoutCalibration.from_symmetric_error(2, error_rate=0.05)
        assert cal.num_qubits == 2
        assert cal.qubit_error_rates is not None
        assert len(cal.qubit_error_rates) == 2

    def test_from_qubit_error_rates(self):
        rates = [(0.02, 0.03), (0.01, 0.04)]
        cal = ReadoutCalibration.from_qubit_error_rates(rates)
        assert cal.num_qubits == 2

    def test_from_confusion_matrix(self):
        matrix = np.array([[0.95, 0.05], [0.05, 0.95]])
        cal = ReadoutCalibration.from_confusion_matrix(matrix, num_qubits=1)
        assert cal.calibration_matrix is not None

    def test_get_full_matrix_identity(self):
        cal = ReadoutCalibration.from_symmetric_error(1, error_rate=0.0)
        M = cal.get_full_matrix()
        assert np.allclose(M, np.eye(2))

    def test_invalid_confusion_matrix_shape(self):
        with pytest.raises(ValueError, match="Expected"):
            ReadoutCalibration.from_confusion_matrix(np.eye(3), num_qubits=1)

    def test_invalid_error_rate(self):
        with pytest.raises(ValueError, match="must be in"):
            ReadoutCalibration.from_qubit_error_rates([(1.5, 0.0)])


class TestReadoutCorrector:
    """Tests for readout error correction."""

    def test_identity_correction(self):
        cal = ReadoutCalibration.from_symmetric_error(2, error_rate=0.0)
        corrector = ReadoutCorrector(cal, method=CorrectionMethod.MATRIX_INVERSION)
        raw_counts = {0: 700, 1: 100, 2: 100, 3: 100}
        corrected = corrector.correct_counts(raw_counts)
        assert abs(corrected.get(0, 0.0) - 0.7) < 0.01

    @pytest.mark.parametrize("method", [
        CorrectionMethod.MATRIX_INVERSION,
        CorrectionMethod.LEAST_SQUARES,
        CorrectionMethod.BAYESIAN_UNFOLDING,
        CorrectionMethod.TENSOR_PRODUCT,
    ])
    def test_all_correction_methods(self, method):
        cal = ReadoutCalibration.from_symmetric_error(2, error_rate=0.05)
        corrector = ReadoutCorrector(cal, method=method)
        raw_probs = np.array([0.7, 0.1, 0.1, 0.1])
        corrected = corrector.correct_probabilities(raw_probs)
        assert abs(corrected.sum() - 1.0) < 0.01 or method == CorrectionMethod.MATRIX_INVERSION

    def test_correct_counts_convenience(self):
        cal = ReadoutCalibration.from_symmetric_error(1, error_rate=0.05)
        raw_counts = {0: 900, 1: 100}
        corrected = correct_counts(raw_counts, cal)
        assert isinstance(corrected, dict)
        total = sum(corrected.values())
        assert abs(total - 1.0) < 0.05


# ---- CDR Tests ----


class TestCDRModel:
    """Tests for CDR regression model."""

    def test_train_linear_model(self):
        ideal = [0.1, 0.2, 0.3, 0.4, 0.5]
        noisy = [0.08, 0.16, 0.24, 0.32, 0.40]
        model = CDRModel.train(ideal, noisy, degree=1)
        assert model.degree == 1
        assert model.r_squared > 0.99

    def test_correct_applies_model(self):
        ideal = [0.0, 0.5, 1.0]
        noisy = [0.0, 0.4, 0.8]
        model = CDRModel.train(ideal, noisy, degree=1)
        corrected = model.correct(0.4)
        assert abs(corrected - 0.5) < 0.05

    def test_slope_and_intercept(self):
        model = CDRModel.train([0.0, 1.0], [0.0, 0.8], degree=1)
        assert model.slope > 1.0  # should be ~1.25
        assert abs(model.intercept) < 0.1

    def test_empty_training_data_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            CDRModel.train([], [])


class TestCDREstimator:
    """Tests for the full CDR pipeline."""

    def test_generate_training_circuits(self, rotation_circuit):
        estimator = CDREstimator(num_training_circuits=5, seed=42)
        circuits = estimator.generate_training_circuits(rotation_circuit)
        assert len(circuits) == 5

    def test_estimate_pipeline(self, rotation_circuit, noisy_executor, ideal_executor):
        estimator = CDREstimator(num_training_circuits=10, degree=1, seed=42)
        result = estimator.estimate(rotation_circuit, noisy_executor, ideal_executor)
        assert isinstance(result, CDRResult)
        assert np.isfinite(result.corrected_value)
        assert result.num_training_circuits == 10

    def test_cdr_correct_convenience(self):
        corrected = cdr_correct(
            noisy_value=0.4,
            ideal_values=[0.0, 0.5, 1.0],
            noisy_values=[0.0, 0.4, 0.8],
            degree=1,
        )
        assert abs(corrected - 0.5) < 0.05


class TestReplaceClifford:
    """Tests for non-Clifford gate replacement."""

    def test_clifford_gate_unchanged(self):
        circuit = [("H", (0,), ()), ("CNOT", (0, 1), ())]
        result = replace_non_clifford(circuit)
        assert result == circuit

    def test_rz_replaced(self):
        circuit = [("RZ", (0,), (0.1,))]
        result = replace_non_clifford(circuit)
        assert result[0][0] != "RZ"

    def test_t_gate_replaced_with_s(self):
        circuit = [("T", (0,), ())]
        result = replace_non_clifford(circuit)
        assert result[0][0] == "S"


# ---- Top-level mitigate() Tests ----


class TestMitigate:
    """Tests for the top-level mitigate convenience function."""

    def test_mitigate_zne(self, simple_circuit, noisy_executor):
        result = mitigate(simple_circuit, noisy_executor, method="zne")
        assert isinstance(result, ZNEResult)
        assert np.isfinite(result.estimated_value)

    def test_mitigate_pec(self, simple_circuit, noisy_executor):
        result = mitigate(simple_circuit, noisy_executor, method="pec",
                          num_samples=50, seed=42)
        assert isinstance(result, PECResult)

    def test_mitigate_twirling(self, simple_circuit, ideal_executor):
        mean, std_err = mitigate(simple_circuit, ideal_executor, method="twirling",
                                 num_samples=20, seed=42)
        assert np.isfinite(mean)
        assert std_err >= 0

    def test_mitigate_cdr(self, rotation_circuit, noisy_executor, ideal_executor):
        result = mitigate(
            rotation_circuit, noisy_executor, method="cdr",
            ideal_executor=ideal_executor, num_training_circuits=5, seed=42,
        )
        assert isinstance(result, CDRResult)

    def test_mitigate_cdr_requires_ideal_executor(self, simple_circuit, noisy_executor):
        with pytest.raises(ValueError, match="ideal_executor"):
            mitigate(simple_circuit, noisy_executor, method="cdr")

    def test_mitigate_unknown_method(self, simple_circuit, noisy_executor):
        with pytest.raises(ValueError, match="Unknown mitigation method"):
            mitigate(simple_circuit, noisy_executor, method="unknown")
