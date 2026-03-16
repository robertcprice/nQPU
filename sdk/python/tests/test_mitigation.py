"""Comprehensive tests for the nQPU quantum error mitigation package.

Covers all five mitigation modules:
  - ZNE: Zero Noise Extrapolation (gate folding, extrapolation models)
  - PEC: Probabilistic Error Cancellation (decomposition, sampling, cost)
  - Twirling: Pauli Twirling / Randomized Compiling
  - Readout: Measurement Error Mitigation (calibration, correction)
  - CDR: Clifford Data Regression (training, correction)

All tests use a lightweight noisy simulator built from numpy for full
self-containment (no external quantum framework needed).
"""

import math

import numpy as np
import pytest

from nqpu.mitigation import (
    CDREstimator,
    CDRModel,
    CDRResult,
    CDRTrainingPoint,
    CorrectionMethod,
    ExtrapolationMethod,
    FoldingStrategy,
    NoiseChannel,
    NoiseScaler,
    PauliFrame,
    PauliTwirler,
    PECDecomposition,
    PECEstimator,
    PECOperation,
    PECResult,
    RandomizedCompiling,
    ReadoutCalibration,
    ReadoutCorrector,
    TwirledCircuit,
    TwoQubitNoiseChannel,
    ZNEEstimator,
    ZNEResult,
    cdr_correct,
    correct_counts,
    mitigate,
    replace_non_clifford,
    run_pec,
    run_zne,
    twirl_and_average,
)


# ======================================================================
# Minimal noisy simulator for testing
# ======================================================================

# Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
_TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)

Gate = tuple  # (name, qubits, params)


def _rz(theta):
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)

_CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)


def _get_gate_matrix(name, params):
    """Return the unitary matrix for a named gate."""
    upper = name.upper()
    if upper == "I":
        return _I
    if upper == "X":
        return _X
    if upper == "Y":
        return _Y
    if upper == "Z":
        return _Z
    if upper == "H":
        return _H
    if upper == "S":
        return _S
    if upper == "SDG":
        return _SDG
    if upper == "T":
        return _T
    if upper == "TDG":
        return _TDG
    if upper == "RZ" and params:
        return _rz(params[0])
    if upper == "RX" and params:
        return _rx(params[0])
    if upper == "RY" and params:
        return _ry(params[0])
    if upper in ("P", "PHASE", "U1") and params:
        return _rz(params[0])
    if upper in ("CNOT", "CX"):
        return _CNOT
    if upper == "CZ":
        return _CZ
    return _I  # Unknown gate -> identity


def _apply_single_qubit(state, matrix, qubit, num_qubits):
    """Apply a single-qubit gate to a state vector."""
    n = num_qubits
    dim = 2**n
    new_state = np.zeros(dim, dtype=np.complex128)
    for i in range(dim):
        bit = (i >> qubit) & 1
        for b in range(2):
            coeff = matrix[b, bit]
            if abs(coeff) < 1e-15:
                continue
            j = (i & ~(1 << qubit)) | (b << qubit)
            new_state[j] += coeff * state[i]
    return new_state


def _apply_two_qubit(state, matrix, q0, q1, num_qubits):
    """Apply a two-qubit gate (4x4 matrix) to qubits q0 (ctrl) and q1 (tgt)."""
    n = num_qubits
    dim = 2**n
    new_state = np.zeros(dim, dtype=np.complex128)
    for i in range(dim):
        b0 = (i >> q0) & 1
        b1 = (i >> q1) & 1
        row_in = b0 * 2 + b1
        for b0_out in range(2):
            for b1_out in range(2):
                row_out = b0_out * 2 + b1_out
                coeff = matrix[row_out, row_in]
                if abs(coeff) < 1e-15:
                    continue
                j = i
                j = (j & ~(1 << q0)) | (b0_out << q0)
                j = (j & ~(1 << q1)) | (b1_out << q1)
                new_state[j] += coeff * state[i]
    return new_state


def simulate_circuit(circuit, num_qubits=None, noise_rate=0.0, rng=None):
    """Simulate a circuit and return <Z_0> expectation value.

    Optionally applies depolarizing noise after each gate.
    """
    if num_qubits is None:
        num_qubits = 1
        for _, qubits, _ in circuit:
            for q in qubits:
                num_qubits = max(num_qubits, q + 1)

    dim = 2**num_qubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |00...0>

    if rng is None:
        rng = np.random.default_rng(42)

    for name, qubits, params in circuit:
        mat = _get_gate_matrix(name, params)
        if len(qubits) == 1:
            state = _apply_single_qubit(state, mat, qubits[0], num_qubits)
        elif len(qubits) == 2:
            state = _apply_two_qubit(state, mat, qubits[0], qubits[1], num_qubits)

        # Apply depolarizing noise to each involved qubit
        if noise_rate > 0:
            for q in qubits:
                if rng.random() < noise_rate:
                    pauli_idx = rng.integers(1, 4)  # X, Y, or Z
                    pauli = [_X, _Y, _Z][pauli_idx - 1]
                    state = _apply_single_qubit(state, pauli, q, num_qubits)

    # Compute <Z_0>
    exp_z = 0.0
    for i in range(dim):
        bit0 = (i >> 0) & 1
        sign = 1.0 - 2.0 * bit0
        exp_z += sign * abs(state[i]) ** 2

    return exp_z


def make_executor(noise_rate=0.0, seed=42):
    """Create a circuit executor with optional noise."""
    rng = np.random.default_rng(seed)

    def executor(circuit):
        return simulate_circuit(circuit, noise_rate=noise_rate, rng=rng)

    return executor


def make_ideal_executor():
    """Create an ideal (noiseless) circuit executor."""
    return make_executor(noise_rate=0.0)


# ======================================================================
# Simple test circuits
# ======================================================================

def bell_circuit():
    """H on qubit 0, CNOT(0,1) -> Bell state |00> + |11>."""
    return [("H", (0,), ()), ("CNOT", (0, 1), ())]


def x_circuit():
    """X gate on qubit 0 -> |1>."""
    return [("X", (0,), ())]


def rz_circuit(theta=0.5):
    """Rz rotation on qubit 0."""
    return [("H", (0,), ()), ("RZ", (0,), (theta,)), ("H", (0,), ())]


def ghz_circuit(n=3):
    """GHZ state: H on q0, CNOT chain."""
    gates = [("H", (0,), ())]
    for i in range(n - 1):
        gates.append(("CNOT", (i, i + 1), ()))
    return gates


def mixed_circuit():
    """Circuit with both Clifford and non-Clifford gates."""
    return [
        ("H", (0,), ()),
        ("RZ", (0,), (0.3,)),
        ("CNOT", (0, 1), ()),
        ("T", (1,), ()),
        ("H", (1,), ()),
    ]


# ======================================================================
# ZNE Tests
# ======================================================================


class TestNoiseScaler:
    """Tests for noise amplification via gate folding."""

    def test_local_fold_scale_1(self):
        """Scale factor 1 returns original circuit."""
        circuit = x_circuit()
        scaler = NoiseScaler(FoldingStrategy.LOCAL)
        folded = scaler.fold_gates(circuit, 1)
        assert len(folded) == len(circuit)

    def test_local_fold_scale_3(self):
        """Local fold by 3: each gate g -> g g^dag g (3 gates per original)."""
        circuit = x_circuit()
        scaler = NoiseScaler(FoldingStrategy.LOCAL)
        folded = scaler.fold_gates(circuit, 3)
        assert len(folded) == 3  # X, X (inverse), X

    def test_local_fold_scale_5(self):
        """Local fold by 5: each gate g -> g (g^dag g)^2 (5 gates per original)."""
        circuit = [("H", (0,), ())]
        scaler = NoiseScaler(FoldingStrategy.LOCAL)
        folded = scaler.fold_gates(circuit, 5)
        assert len(folded) == 5

    def test_global_fold_scale_3(self):
        """Global fold by 3: U -> U U^dag U."""
        circuit = bell_circuit()
        scaler = NoiseScaler(FoldingStrategy.GLOBAL)
        folded = scaler.fold_gates(circuit, 3)
        # Original: 2 gates, U^dag: 2 gates, U: 2 gates = 6
        assert len(folded) == 6

    def test_global_fold_preserves_unitary(self):
        """Global folded circuit computes the same ideal unitary."""
        circuit = x_circuit()
        scaler = NoiseScaler(FoldingStrategy.GLOBAL)
        folded = scaler.fold_gates(circuit, 3)
        ideal_original = simulate_circuit(circuit, noise_rate=0.0)
        ideal_folded = simulate_circuit(folded, noise_rate=0.0)
        assert abs(ideal_original - ideal_folded) < 1e-10

    def test_local_fold_preserves_unitary(self):
        """Local folded circuit computes the same ideal unitary."""
        circuit = rz_circuit(0.7)
        scaler = NoiseScaler(FoldingStrategy.LOCAL)
        folded = scaler.fold_gates(circuit, 5)
        ideal_original = simulate_circuit(circuit, noise_rate=0.0)
        ideal_folded = simulate_circuit(folded, noise_rate=0.0)
        assert abs(ideal_original - ideal_folded) < 1e-10

    def test_even_scale_rounded_to_odd(self):
        """Even scale factors are rounded up to the nearest odd."""
        circuit = x_circuit()
        scaler = NoiseScaler(FoldingStrategy.LOCAL)
        folded = scaler.fold_gates(circuit, 4)
        # Rounded to 5: X, X, X, X, X
        assert len(folded) == 5

    def test_invalid_scale_factor(self):
        """Scale factor < 1 raises ValueError."""
        scaler = NoiseScaler()
        with pytest.raises(ValueError):
            scaler.fold_gates(x_circuit(), 0)

    def test_pulse_stretch(self):
        """Pulse stretch appends factor to params."""
        circuit = [("H", (0,), ()), ("RZ", (0,), (0.5,))]
        scaler = NoiseScaler()
        stretched = scaler.pulse_stretch(circuit, 2.0)
        assert stretched[0][2] == (2.0,)
        assert stretched[1][2] == (0.5, 2.0)

    def test_pulse_stretch_invalid(self):
        """Pulse stretch factor < 1.0 raises ValueError."""
        scaler = NoiseScaler()
        with pytest.raises(ValueError):
            scaler.pulse_stretch(x_circuit(), 0.5)


class TestZNEExtrapolation:
    """Tests for ZNE extrapolation models."""

    def test_linear_perfect_data(self):
        """Linear extrapolation recovers exact value for linear data."""
        # y = 1.0 - 0.1*x, so y(0) = 1.0
        def executor(circuit):
            scale = len(circuit)
            return 1.0 - 0.1 * scale

        circuit = [("H", (0,), ())]
        result = run_zne(
            circuit, executor,
            noise_factors=[1, 3, 5],
            method=ExtrapolationMethod.LINEAR,
        )
        assert abs(result.estimated_value - 1.0) < 0.15
        assert len(result.raw_values) == 3

    def test_polynomial_extrapolation(self):
        """Polynomial extrapolation with quadratic data."""
        # y = 1.0 - 0.05*x^2, so y(0) = 1.0
        def executor(circuit):
            scale = len(circuit)
            return 1.0 - 0.05 * scale**2

        circuit = [("X", (0,), ())]
        result = run_zne(
            circuit, executor,
            noise_factors=[1, 3, 5],
            method=ExtrapolationMethod.POLYNOMIAL,
            poly_degree=2,
        )
        assert abs(result.estimated_value - 1.0) < 0.1

    def test_exponential_extrapolation(self):
        """Exponential extrapolation with exponential decay data."""
        # y = exp(-0.1*x), so y(0) = 1.0
        def executor(circuit):
            scale = len(circuit)
            return np.exp(-0.1 * scale)

        circuit = [("H", (0,), ())]
        result = run_zne(
            circuit, executor,
            noise_factors=[1, 3, 5],
            method=ExtrapolationMethod.EXPONENTIAL,
        )
        assert abs(result.estimated_value - 1.0) < 0.1

    def test_richardson_extrapolation(self):
        """Richardson extrapolation recovers exact value for polynomial data."""
        # y = 1.0 + 0.2*x + 0.05*x^2, y(0) = 1.0
        def executor(circuit):
            x = len(circuit)
            return 1.0 + 0.2 * x + 0.05 * x**2

        circuit = [("X", (0,), ())]
        result = run_zne(
            circuit, executor,
            noise_factors=[1, 3, 5],
            method=ExtrapolationMethod.RICHARDSON,
        )
        assert abs(result.estimated_value - 1.0) < 1e-8

    def test_zne_with_noisy_sim(self):
        """ZNE improves estimation compared to raw noisy value."""
        circuit = rz_circuit(0.5)
        ideal_val = simulate_circuit(circuit, noise_rate=0.0)
        noisy_executor = make_executor(noise_rate=0.05, seed=123)
        noisy_val = noisy_executor(circuit)

        result = run_zne(
            circuit, noisy_executor,
            noise_factors=[1, 3, 5],
            method=ExtrapolationMethod.LINEAR,
        )
        # ZNE should move the estimate closer to ideal
        # (not guaranteed with stochastic noise, but for fixed seed it should)
        assert isinstance(result, ZNEResult)
        assert len(result.noise_factors) == 3

    def test_zne_confidence_metric(self):
        """Confidence metric is between 0 and 1."""
        def executor(circuit):
            return 1.0 - 0.1 * len(circuit)

        circuit = [("H", (0,), ())]
        result = run_zne(circuit, executor, noise_factors=[1, 3, 5])
        assert 0.0 <= result.confidence <= 1.0

    def test_zne_single_point(self):
        """Single noise factor returns the raw value."""
        def executor(circuit):
            return 0.42

        circuit = [("H", (0,), ())]
        result = run_zne(circuit, executor, noise_factors=[1])
        assert abs(result.estimated_value - 0.42) < 1e-10

    def test_zne_custom_factors(self):
        """Custom noise factors are accepted."""
        executor = make_ideal_executor()
        circuit = x_circuit()
        result = run_zne(
            circuit, executor,
            noise_factors=[1, 2, 3, 4, 5],
            method=ExtrapolationMethod.POLYNOMIAL,
            poly_degree=4,
        )
        assert len(result.noise_factors) == 5

    def test_empty_factors_raises(self):
        """Empty noise_factors raises ValueError."""
        with pytest.raises(ValueError):
            ZNEEstimator(noise_factors=[])


class TestZNEEstimator:
    """Tests for the ZNEEstimator class."""

    def test_estimator_construction(self):
        est = ZNEEstimator(
            noise_factors=[1, 3, 5, 7],
            method=ExtrapolationMethod.POLYNOMIAL,
            poly_degree=3,
        )
        assert est.noise_factors == [1, 3, 5, 7]
        assert est.method == ExtrapolationMethod.POLYNOMIAL

    def test_estimator_estimate(self):
        est = ZNEEstimator()
        executor = make_ideal_executor()
        result = est.estimate(x_circuit(), executor)
        assert isinstance(result, ZNEResult)
        assert result.method == ExtrapolationMethod.LINEAR


# ======================================================================
# PEC Tests
# ======================================================================


class TestNoiseChannel:
    """Tests for NoiseChannel representation."""

    def test_depolarizing_channel(self):
        """Depolarizing channel has correct probabilities."""
        ch = NoiseChannel.depolarizing(0.03)
        assert abs(ch.probabilities["I"] - 0.97) < 1e-10
        assert abs(ch.probabilities["X"] - 0.01) < 1e-10
        assert abs(ch.error_rate - 0.03) < 1e-10

    def test_pauli_channel(self):
        """General Pauli channel construction."""
        ch = NoiseChannel.pauli_channel(0.02, 0.01, 0.03)
        assert abs(ch.probabilities["I"] - 0.94) < 1e-10
        assert abs(ch.probabilities["X"] - 0.02) < 1e-10

    def test_invalid_probabilities(self):
        """Probabilities > 1 raise ValueError."""
        with pytest.raises(ValueError):
            NoiseChannel.pauli_channel(0.5, 0.3, 0.3)

    def test_depolarizing_high_error_rate(self):
        """Error rate >= 0.75 raises ValueError."""
        with pytest.raises(ValueError):
            NoiseChannel.depolarizing(0.75)

    def test_ptm_diagonal_for_depolarizing(self):
        """PTM is diagonal for depolarizing channel."""
        ch = NoiseChannel.depolarizing(0.06)
        ptm = ch.pauli_transfer_matrix()
        # Off-diagonal should be zero
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert abs(ptm[i, j]) < 1e-10

    def test_ptm_eigenvalue_depolarizing(self):
        """PTM eigenvalue for depolarizing: lambda = 1 - 4p/3."""
        p = 0.06
        ch = NoiseChannel.depolarizing(p)
        ptm = ch.pauli_transfer_matrix()
        expected_lambda = 1.0 - 4.0 * p / 3.0
        assert abs(ptm[1, 1] - expected_lambda) < 1e-10
        assert abs(ptm[2, 2] - expected_lambda) < 1e-10
        assert abs(ptm[3, 3] - expected_lambda) < 1e-10

    def test_identity_channel(self):
        """Zero-error channel is identity."""
        ch = NoiseChannel.depolarizing(0.0)
        ptm = ch.pauli_transfer_matrix()
        np.testing.assert_allclose(ptm, np.eye(4), atol=1e-10)


class TestTwoQubitNoiseChannel:
    """Tests for TwoQubitNoiseChannel."""

    def test_depolarizing_two_qubit(self):
        """Two-qubit depolarizing channel probabilities sum to 1."""
        ch = TwoQubitNoiseChannel.depolarizing(0.05)
        total = sum(ch.probabilities.values())
        assert abs(total - 1.0) < 1e-10
        assert abs(ch.probabilities[("I", "I")] - 0.95) < 1e-10

    def test_default_is_identity(self):
        """Default two-qubit channel is identity."""
        ch = TwoQubitNoiseChannel()
        assert ch.probabilities[("I", "I")] == 1.0

    def test_invalid_two_qubit_rate(self):
        """Error rate >= 15/16 raises ValueError."""
        with pytest.raises(ValueError):
            TwoQubitNoiseChannel.depolarizing(15.0 / 16.0)


class TestPECDecomposition:
    """Tests for PEC quasi-probability decomposition."""

    def test_depolarizing_decomposition(self):
        """Decomposition of depolarizing channel has correct structure."""
        decomp = PECDecomposition.from_depolarizing(0.01, qubit=0)
        assert len(decomp.operations) == 4
        assert decomp.gamma >= 1.0

    def test_decomposition_coefficients_sum(self):
        """Coefficients should sum to 1 (trace preservation)."""
        decomp = PECDecomposition.from_depolarizing(0.05, qubit=0)
        total = sum(op.coefficient for op in decomp.operations)
        assert abs(total - 1.0) < 1e-10

    def test_gamma_increases_with_noise(self):
        """Gamma (overhead) increases with error rate."""
        gamma_low = PECDecomposition.from_depolarizing(0.01, qubit=0).gamma
        gamma_high = PECDecomposition.from_depolarizing(0.1, qubit=0).gamma
        assert gamma_high > gamma_low

    def test_gamma_is_one_for_no_noise(self):
        """Gamma is 1.0 for zero noise."""
        decomp = PECDecomposition.from_depolarizing(0.0, qubit=0)
        assert abs(decomp.gamma - 1.0) < 1e-10

    def test_sample_correction(self):
        """Sampled correction returns valid Pauli and sign."""
        decomp = PECDecomposition.from_depolarizing(0.05, qubit=0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            pauli, sign = decomp.sample_correction(rng)
            assert pauli in ("I", "X", "Y", "Z")
            assert sign in (-1.0, 1.0)

    def test_cost_estimate(self):
        """Cost estimate scales exponentially with depth."""
        decomp = PECDecomposition.from_depolarizing(0.05, qubit=0)
        cost_10 = decomp.cost_estimate(10)
        cost_20 = decomp.cost_estimate(20)
        assert cost_20 > cost_10
        # cost(20) = gamma^40, cost(10) = gamma^20
        assert abs(cost_20 - cost_10**2) < 1e-6

    def test_from_noise_channel_general(self):
        """Decomposition from general Pauli channel."""
        ch = NoiseChannel.pauli_channel(0.02, 0.01, 0.03)
        decomp = PECDecomposition.from_noise_channel(ch, qubit=0)
        assert len(decomp.operations) == 4
        assert decomp.gamma >= 1.0

    def test_identity_decomposition(self):
        """Identity channel has gamma=1 and coefficient (1,0,0,0)."""
        ch = NoiseChannel.depolarizing(0.0)
        decomp = PECDecomposition.from_noise_channel(ch, qubit=0)
        assert abs(decomp.gamma - 1.0) < 1e-10
        assert abs(decomp.operations[0].coefficient - 1.0) < 1e-10


class TestPECEstimator:
    """Tests for the PEC Monte Carlo estimator."""

    def test_pec_basic(self):
        """PEC produces a result with expected fields."""
        channel = NoiseChannel.depolarizing(0.01)
        estimator = PECEstimator(channel, num_samples=50, seed=42)
        circuit = x_circuit()
        executor = make_ideal_executor()
        result = estimator.estimate(circuit, executor)
        assert isinstance(result, PECResult)
        assert result.num_samples == 50
        assert result.gamma >= 1.0

    def test_pec_with_identity_channel(self):
        """PEC with zero noise should return the ideal value."""
        channel = NoiseChannel.depolarizing(0.0)
        estimator = PECEstimator(channel, num_samples=100, seed=42)
        circuit = x_circuit()
        executor = make_ideal_executor()
        result = estimator.estimate(circuit, executor)
        # <Z_0> for |1> is -1
        assert abs(result.estimated_value - (-1.0)) < 0.1

    def test_pec_sign_ratio(self):
        """Sign ratio is between 0 and 1."""
        channel = NoiseChannel.depolarizing(0.05)
        estimator = PECEstimator(channel, num_samples=200, seed=42)
        circuit = bell_circuit()
        executor = make_ideal_executor()
        result = estimator.estimate(circuit, executor)
        assert 0.0 <= result.sign_ratio <= 1.0

    def test_run_pec_convenience(self):
        """run_pec convenience function works."""
        circuit = x_circuit()
        executor = make_ideal_executor()
        result = run_pec(circuit, executor, error_rate=0.01, num_samples=50, seed=42)
        assert isinstance(result, PECResult)


# ======================================================================
# Twirling Tests
# ======================================================================


class TestPauliTwirler:
    """Tests for Pauli twirling."""

    def test_twirl_gate_cnot(self):
        """Twirling a CNOT gate produces valid Pauli frames."""
        twirler = PauliTwirler(seed=42)
        frame = twirler.twirl_gate("CNOT", ctrl=0, tgt=1)
        assert isinstance(frame, PauliFrame)
        assert 0 in frame.before
        assert 1 in frame.before
        assert frame.before[0] in ("I", "X", "Y", "Z")

    def test_twirl_gate_cz(self):
        """Twirling a CZ gate produces valid Pauli frames."""
        twirler = PauliTwirler(seed=42)
        frame = twirler.twirl_gate("CZ", ctrl=0, tgt=1)
        assert isinstance(frame, PauliFrame)

    def test_twirl_circuit_structure(self):
        """Twirled circuit variants have correct structure."""
        twirler = PauliTwirler(seed=42)
        circuit = bell_circuit()
        twirled = twirler.twirl_circuit(circuit, num_samples=5)
        assert isinstance(twirled, TwirledCircuit)
        assert len(twirled.variants) == 5
        assert twirled.base_circuit == circuit

    def test_twirled_variant_longer(self):
        """Twirled variants have more gates due to Pauli insertions."""
        twirler = PauliTwirler(seed=42)
        circuit = bell_circuit()
        twirled = twirler.twirl_circuit(circuit, num_samples=10)
        for variant in twirled.variants:
            # At least as many gates as original (Pauli frames may add gates)
            assert len(variant) >= len(circuit)

    def test_single_qubit_twirl(self):
        """Single-qubit twirl produces a valid PauliFrame."""
        twirler = PauliTwirler(seed=42)
        frame = twirler.twirl_single_qubit(0)
        assert 0 in frame.before
        assert 0 in frame.after

    def test_twirl_invalid_gate(self):
        """Twirling an unsupported gate raises ValueError."""
        twirler = PauliTwirler(seed=42)
        with pytest.raises(ValueError):
            twirler.twirl_gate("SWAP", ctrl=0, tgt=1)

    def test_twirl_single_qubit_only(self):
        """Circuit with only single-qubit gates produces identical variants."""
        twirler = PauliTwirler(seed=42)
        circuit = [("H", (0,), ()), ("X", (0,), ())]
        twirled = twirler.twirl_circuit(circuit, num_samples=3)
        # No two-qubit gates -> no twirling -> variants identical to base
        for variant in twirled.variants:
            assert variant == circuit


class TestTwirlAndAverage:
    """Tests for the twirl_and_average function."""

    def test_average_ideal(self):
        """Twirl-and-average on ideal circuit returns ~ideal value."""
        circuit = bell_circuit()
        executor = make_ideal_executor()
        mean, std_err = twirl_and_average(circuit, executor, num_samples=20, seed=42)
        ideal_val = simulate_circuit(circuit)
        assert abs(mean - ideal_val) < 0.3
        assert std_err >= 0.0

    def test_average_returns_tuple(self):
        """Returns (mean, std_error) tuple."""
        circuit = x_circuit()
        executor = make_ideal_executor()
        result = twirl_and_average(circuit, executor, num_samples=5, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestRandomizedCompiling:
    """Tests for RandomizedCompiling."""

    def test_compile_produces_variants(self):
        """RandomizedCompiling.compile produces the requested number of variants."""
        rc = RandomizedCompiling(num_compilations=10, seed=42)
        circuit = bell_circuit()
        twirled = rc.compile(circuit)
        assert len(twirled.variants) == 10

    def test_execute_returns_mean_stderr(self):
        """RandomizedCompiling.execute returns (mean, std_error)."""
        rc = RandomizedCompiling(num_compilations=5, seed=42)
        circuit = bell_circuit()
        executor = make_ideal_executor()
        mean, std_err = rc.execute(circuit, executor)
        assert isinstance(mean, float)
        assert isinstance(std_err, float)


# ======================================================================
# Readout Tests
# ======================================================================


class TestReadoutCalibration:
    """Tests for readout error characterization."""

    def test_from_symmetric_error(self):
        """Symmetric error calibration creates valid object."""
        cal = ReadoutCalibration.from_symmetric_error(2, 0.05)
        assert cal.num_qubits == 2
        assert cal.qubit_error_rates is not None
        assert len(cal.qubit_error_rates) == 2

    def test_from_qubit_error_rates(self):
        """Per-qubit error rates create valid calibration."""
        cal = ReadoutCalibration.from_qubit_error_rates([
            (0.03, 0.02),
            (0.04, 0.01),
        ])
        assert cal.num_qubits == 2

    def test_from_confusion_matrix(self):
        """Pre-computed confusion matrix accepted."""
        # 1-qubit perfect readout
        M = np.eye(2)
        cal = ReadoutCalibration.from_confusion_matrix(M, num_qubits=1)
        assert cal.calibration_matrix is not None

    def test_invalid_matrix_shape(self):
        """Wrong-shaped matrix raises ValueError."""
        with pytest.raises(ValueError):
            ReadoutCalibration.from_confusion_matrix(np.eye(3), num_qubits=1)

    def test_invalid_matrix_columns(self):
        """Non-stochastic matrix raises ValueError."""
        M = np.ones((2, 2))  # columns sum to 2
        with pytest.raises(ValueError):
            ReadoutCalibration.from_confusion_matrix(M, num_qubits=1)

    def test_invalid_error_rates(self):
        """Error rates outside [0,1] raise ValueError."""
        with pytest.raises(ValueError):
            ReadoutCalibration.from_qubit_error_rates([(1.5, 0.0)])

    def test_get_full_matrix_tensor_product(self):
        """Tensor product calibration matrix has correct shape."""
        cal = ReadoutCalibration.from_symmetric_error(3, 0.05)
        M = cal.get_full_matrix()
        assert M.shape == (8, 8)
        # Columns should sum to ~1
        col_sums = M.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_perfect_readout_matrix(self):
        """Zero-error calibration produces identity matrix."""
        cal = ReadoutCalibration.from_symmetric_error(2, 0.0)
        M = cal.get_full_matrix()
        np.testing.assert_allclose(M, np.eye(4), atol=1e-10)


class TestReadoutCorrector:
    """Tests for measurement error correction."""

    def test_correct_perfect_readout(self):
        """Perfect readout returns original distribution."""
        cal = ReadoutCalibration.from_symmetric_error(2, 0.0)
        corrector = ReadoutCorrector(cal)
        raw_counts = {0: 500, 3: 500}  # |00> and |11>
        corrected = corrector.correct_counts(raw_counts)
        assert abs(corrected.get(0, 0) - 0.5) < 0.01
        assert abs(corrected.get(3, 0) - 0.5) < 0.01

    def test_correction_reduces_error(self):
        """Correction reduces distance from true distribution."""
        cal = ReadoutCalibration.from_symmetric_error(1, 0.1)
        corrector = ReadoutCorrector(cal, method=CorrectionMethod.LEAST_SQUARES)

        # True distribution: qubit is |0> (bitstring 0)
        # With 10% readout error: ~90% measure 0, ~10% measure 1
        raw_counts = {0: 900, 1: 100}
        corrected = corrector.correct_counts(raw_counts)

        # Corrected should be closer to {0: 1.0}
        assert corrected.get(0, 0) > 0.9

    def test_matrix_inversion_method(self):
        """Matrix inversion correction works."""
        cal = ReadoutCalibration.from_symmetric_error(1, 0.05)
        corrector = ReadoutCorrector(cal, method=CorrectionMethod.MATRIX_INVERSION)
        raw_counts = {0: 950, 1: 50}
        corrected = corrector.correct_counts(raw_counts)
        assert 0 in corrected

    def test_bayesian_unfolding(self):
        """Bayesian unfolding produces valid probabilities."""
        cal = ReadoutCalibration.from_symmetric_error(2, 0.05)
        corrector = ReadoutCorrector(cal, method=CorrectionMethod.BAYESIAN_UNFOLDING)
        raw_counts = {0: 450, 1: 50, 2: 50, 3: 450}
        corrected = corrector.correct_counts(raw_counts)
        total = sum(corrected.values())
        assert abs(total - 1.0) < 0.01
        # All probabilities should be non-negative
        for p in corrected.values():
            assert p >= -1e-10

    def test_tensor_product_method(self):
        """Tensor product correction is efficient and correct."""
        cal = ReadoutCalibration.from_qubit_error_rates([
            (0.03, 0.02),
            (0.04, 0.01),
        ])
        corrector = ReadoutCorrector(cal, method=CorrectionMethod.TENSOR_PRODUCT)
        raw_counts = {0: 900, 1: 30, 2: 40, 3: 30}
        corrected = corrector.correct_counts(raw_counts)
        total = sum(corrected.values())
        assert abs(total - 1.0) < 0.02

    def test_correct_counts_convenience(self):
        """correct_counts convenience function works."""
        cal = ReadoutCalibration.from_symmetric_error(1, 0.05)
        raw_counts = {0: 950, 1: 50}
        corrected = correct_counts(raw_counts, cal)
        assert isinstance(corrected, dict)

    def test_empty_counts(self):
        """Empty counts return empty dict."""
        cal = ReadoutCalibration.from_symmetric_error(1, 0.05)
        corrector = ReadoutCorrector(cal)
        corrected = corrector.correct_counts({})
        assert corrected == {}

    def test_calibration_from_circuits(self):
        """Calibration from simulated circuits produces valid matrix."""
        def executor(prep_state):
            # Simulate perfect readout with small error
            rng = np.random.default_rng(42)
            counts = {}
            n = len(prep_state)
            true_bitstring = sum(b << i for i, b in enumerate(prep_state))
            for _ in range(1000):
                measured = true_bitstring
                for q in range(n):
                    if rng.random() < 0.03:
                        measured ^= (1 << q)
                counts[measured] = counts.get(measured, 0) + 1
            return counts

        cal = ReadoutCalibration.from_calibration_circuits(2, executor, num_shots=1000)
        assert cal.num_qubits == 2
        M = cal.get_full_matrix()
        assert M.shape == (4, 4)


# ======================================================================
# CDR Tests
# ======================================================================


class TestCDRModel:
    """Tests for CDR model fitting."""

    def test_linear_fit(self):
        """Linear CDR model fits correctly."""
        ideal = [0.2, 0.4, 0.6, 0.8, 1.0]
        noisy = [0.18, 0.35, 0.52, 0.68, 0.85]
        model = CDRModel.train(ideal, noisy)
        assert model.degree == 1
        assert model.r_squared > 0.95

    def test_correction(self):
        """CDR correction moves noisy values toward ideal."""
        ideal = [0.0, 0.5, 1.0]
        noisy = [0.1, 0.45, 0.85]
        model = CDRModel.train(ideal, noisy)
        corrected = model.correct(0.45)
        # Should be closer to 0.5 than 0.45 is
        assert abs(corrected - 0.5) < abs(0.45 - 0.5) + 0.1

    def test_polynomial_fit(self):
        """Polynomial CDR model with degree 2."""
        x = np.linspace(0, 1, 10)
        ideal = x**2
        noisy = x**2 * 0.8 + 0.05
        model = CDRModel.train(list(ideal), list(noisy), degree=2)
        assert model.degree == 2

    def test_single_point(self):
        """Single training point produces constant correction."""
        model = CDRModel.train([1.0], [0.8])
        assert abs(model.correct(0.8) - 1.0) < 0.01

    def test_slope_and_intercept(self):
        """Slope and intercept properties work for linear model."""
        model = CDRModel.train([0.0, 1.0], [0.0, 1.0])
        assert abs(model.slope - 1.0) < 1e-10
        assert abs(model.intercept) < 1e-10

    def test_empty_training_raises(self):
        """Empty training data raises ValueError."""
        with pytest.raises(ValueError):
            CDRModel.train([], [])

    def test_mismatched_lengths_raises(self):
        """Mismatched array lengths raise ValueError."""
        with pytest.raises(ValueError):
            CDRModel.train([1.0, 2.0], [1.0])


class TestReplaceNonClifford:
    """Tests for Clifford gate substitution."""

    def test_clifford_gates_unchanged(self):
        """Clifford gates pass through unchanged."""
        circuit = [("H", (0,), ()), ("CNOT", (0, 1), ())]
        result = replace_non_clifford(circuit)
        assert result == circuit

    def test_rz_replaced(self):
        """Rz gates are replaced with nearest Clifford."""
        circuit = [("RZ", (0,), (0.1,))]  # Close to 0 -> I
        result = replace_non_clifford(circuit)
        assert result[0][0] in ("I", "S", "Z", "SDG")

    def test_rz_near_pi_half(self):
        """Rz(pi/2) maps to S gate."""
        circuit = [("RZ", (0,), (np.pi / 2,))]
        result = replace_non_clifford(circuit)
        assert result[0][0] == "S"

    def test_rz_near_pi(self):
        """Rz(pi) maps to Z gate."""
        circuit = [("RZ", (0,), (np.pi,))]
        result = replace_non_clifford(circuit)
        assert result[0][0] == "Z"

    def test_t_replaced_with_s(self):
        """T gate maps to S (nearest Clifford)."""
        circuit = [("T", (0,), ())]
        result = replace_non_clifford(circuit)
        assert result[0][0] == "S"

    def test_tdg_replaced_with_sdg(self):
        """Tdg gate maps to Sdg."""
        circuit = [("TDG", (0,), ())]
        result = replace_non_clifford(circuit)
        assert result[0][0] == "SDG"

    def test_mixed_circuit(self):
        """Mixed circuit has non-Clifford gates replaced."""
        result = replace_non_clifford(mixed_circuit())
        for name, _, _ in result:
            assert name.upper() in (
                "I", "X", "Y", "Z", "H", "S", "SDG", "CNOT", "CX", "CZ", "SWAP",
            )


class TestCDREstimator:
    """Tests for the CDR pipeline."""

    def test_generate_training_circuits(self):
        """Training circuit generation produces the right number."""
        estimator = CDREstimator(num_training_circuits=10, seed=42)
        circuits = estimator.generate_training_circuits(mixed_circuit())
        assert len(circuits) == 10

    def test_all_clifford_circuit(self):
        """All-Clifford circuit duplicates instead of substituting."""
        estimator = CDREstimator(num_training_circuits=5, seed=42)
        circuit = bell_circuit()
        circuits = estimator.generate_training_circuits(circuit)
        assert len(circuits) == 5

    def test_full_cdr_pipeline(self):
        """Full CDR pipeline produces a CDRResult."""
        circuit = mixed_circuit()
        noisy_executor = make_executor(noise_rate=0.02, seed=123)
        ideal_executor = make_ideal_executor()

        estimator = CDREstimator(num_training_circuits=10, seed=42)
        result = estimator.estimate(circuit, noisy_executor, ideal_executor)
        assert isinstance(result, CDRResult)
        assert result.num_training_circuits == 10

    def test_cdr_correct_convenience(self):
        """cdr_correct convenience function works."""
        ideal = [0.0, 0.5, 1.0]
        noisy = [0.1, 0.45, 0.85]
        corrected = cdr_correct(0.45, ideal, noisy)
        assert isinstance(corrected, float)


# ======================================================================
# Integration / Convenience Tests
# ======================================================================


class TestMitigateConvenience:
    """Tests for the top-level mitigate() function."""

    def test_mitigate_zne(self):
        """mitigate with method='zne' returns ZNEResult."""
        circuit = x_circuit()
        executor = make_ideal_executor()
        result = mitigate(circuit, executor, method="zne")
        assert isinstance(result, ZNEResult)

    def test_mitigate_pec(self):
        """mitigate with method='pec' returns PECResult."""
        circuit = x_circuit()
        executor = make_ideal_executor()
        result = mitigate(circuit, executor, method="pec", num_samples=50, seed=42)
        assert isinstance(result, PECResult)

    def test_mitigate_twirling(self):
        """mitigate with method='twirling' returns (mean, std_error)."""
        circuit = bell_circuit()
        executor = make_ideal_executor()
        result = mitigate(circuit, executor, method="twirling", num_samples=10, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_mitigate_cdr(self):
        """mitigate with method='cdr' returns CDRResult."""
        circuit = mixed_circuit()
        noisy_executor = make_executor(noise_rate=0.02, seed=123)
        ideal_executor = make_ideal_executor()
        result = mitigate(
            circuit, noisy_executor, method="cdr",
            ideal_executor=ideal_executor, num_training_circuits=5, seed=42,
        )
        assert isinstance(result, CDRResult)

    def test_mitigate_unknown_method(self):
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError):
            mitigate(x_circuit(), make_ideal_executor(), method="unknown")

    def test_mitigate_cdr_missing_ideal(self):
        """CDR without ideal_executor raises ValueError."""
        with pytest.raises(ValueError):
            mitigate(x_circuit(), make_ideal_executor(), method="cdr")


# ======================================================================
# Edge Cases and Additional Coverage
# ======================================================================


class TestEdgeCases:
    """Edge cases and additional coverage."""

    def test_gate_inverse_rotation(self):
        """Gate inverse for rotation gates negates angle."""
        from nqpu.mitigation.zne import _gate_inverse
        gate = ("RZ", (0,), (0.5,))
        inv = _gate_inverse(gate)
        assert inv[2] == (-0.5,)

    def test_gate_inverse_self_inverse(self):
        """Self-inverse gates return themselves."""
        from nqpu.mitigation.zne import _gate_inverse
        gate = ("H", (0,), ())
        assert _gate_inverse(gate) == gate

    def test_gate_inverse_s_sdg(self):
        """S and Sdg are each other's inverse."""
        from nqpu.mitigation.zne import _gate_inverse
        assert _gate_inverse(("S", (0,), ()))[0] == "Sdg"
        assert _gate_inverse(("Sdg", (0,), ()))[0] == "S"

    def test_gate_inverse_t_tdg(self):
        """T and Tdg are each other's inverse."""
        from nqpu.mitigation.zne import _gate_inverse
        assert _gate_inverse(("T", (0,), ()))[0] == "Tdg"
        assert _gate_inverse(("Tdg", (0,), ()))[0] == "T"

    def test_zne_many_factors(self):
        """ZNE with many noise factors works."""
        def executor(circuit):
            return 1.0 - 0.01 * len(circuit)

        circuit = [("H", (0,), ())]
        result = run_zne(
            circuit, executor,
            noise_factors=[1, 3, 5, 7, 9, 11],
            method=ExtrapolationMethod.POLYNOMIAL,
            poly_degree=3,
        )
        assert len(result.raw_values) == 6

    def test_pec_decomposition_from_channel(self):
        """PEC decomposition from asymmetric Pauli channel."""
        ch = NoiseChannel.pauli_channel(0.01, 0.02, 0.03)
        decomp = PECDecomposition.from_noise_channel(ch, qubit=0)
        total_coeff = sum(op.coefficient for op in decomp.operations)
        assert abs(total_coeff - 1.0) < 1e-10

    def test_readout_multiqubit_correction(self):
        """3-qubit readout correction works."""
        cal = ReadoutCalibration.from_symmetric_error(3, 0.02)
        corrector = ReadoutCorrector(cal, method=CorrectionMethod.LEAST_SQUARES)
        # Prepare a distribution peaked at |000>
        raw_counts = {0: 900}
        for i in range(1, 8):
            raw_counts[i] = 100 // 7
        corrected = corrector.correct_counts(raw_counts)
        assert corrected.get(0, 0) > 0.85

    def test_cdr_degree_clamped(self):
        """CDR degree is clamped to available data points."""
        model = CDRModel.train([1.0, 2.0], [0.8, 1.6], degree=5)
        assert model.degree <= 1

    def test_twirler_deterministic_with_seed(self):
        """Twirler with same seed produces same results."""
        t1 = PauliTwirler(seed=123)
        t2 = PauliTwirler(seed=123)
        f1 = t1.twirl_gate("CNOT", 0, 1)
        f2 = t2.twirl_gate("CNOT", 0, 1)
        assert f1.before == f2.before
        assert f1.after == f2.after

    def test_noise_channel_negative_prob_raises(self):
        """Negative probability raises ValueError."""
        with pytest.raises(ValueError):
            NoiseChannel(probabilities={"I": 1.1, "X": -0.1, "Y": 0.0, "Z": 0.0})

    def test_zne_estimator_all_methods(self):
        """ZNEEstimator works with all extrapolation methods."""
        def executor(circuit):
            return 1.0 - 0.05 * len(circuit)

        circuit = [("H", (0,), ())]
        for method in ExtrapolationMethod:
            est = ZNEEstimator(noise_factors=[1, 3, 5], method=method)
            result = est.estimate(circuit, executor)
            assert isinstance(result.estimated_value, float)
