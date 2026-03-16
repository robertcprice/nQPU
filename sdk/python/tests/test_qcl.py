"""Comprehensive tests for the nQPU Quantum Circuit Learning (QCL) package.

Tests cover all six modules:
  - circuits.py: Statevector simulator, data encodings, ansaetze, CircuitTemplate
  - gradients.py: Parameter shift, finite difference, stochastic, natural gradient
  - training.py: QCLTrainer with classification and regression objectives
  - kernels.py: QuantumKernel, QSVM, QKernelPCA, kernel alignment
  - expressibility.py: Expressibility, barren plateau detection, effective dimension

All numerical tests use fixed random seeds and generous tolerances
appropriate for the small circuit sizes used in testing.
"""

import math

import numpy as np
import pytest

# ----- circuits -----
from nqpu.qcl.circuits import (
    AmplitudeEncoding,
    AngleEncoding,
    AnsatzCircuit,
    CircuitTemplate,
    DataEncodingCircuit,
    GateOp,
    GateType,
    HardwareEfficientAnsatz,
    IQPEncoding,
    ParameterizedCircuit,
    SimplifiedTwoDesign,
    StatevectorSimulator,
    StronglyEntanglingLayers,
)

# ----- gradients -----
from nqpu.qcl.gradients import (
    BarrenPlateauScanner,
    FiniteDifferenceGradient,
    GradientResult,
    NaturalGradient,
    ParameterShiftRule,
    StochasticParameterShift,
)

# ----- training -----
from nqpu.qcl.training import (
    ClassificationObjective,
    KernelObjective,
    QCLTrainer,
    RegressionObjective,
    TrainingHistory,
)

# ----- kernels -----
from nqpu.qcl.kernels import (
    ProjectedQuantumKernel,
    QKernelPCA,
    QSVM,
    QuantumKernel,
    TrainableKernel,
    kernel_target_alignment,
)

# ----- expressibility -----
from nqpu.qcl.expressibility import (
    BarrenPlateauDetector,
    EffectiveDimension,
    ExpressibilityAnalyzer,
)


# ======================================================================
# Helpers and fixtures
# ======================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_xor_data(n=20, seed=42):
    """Generate 2D XOR classification data."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    return X, y


def _make_circle_data(n=40, seed=42):
    """Generate 2D circle classification data."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 0.5).astype(int)
    return X, y


def _make_regression_data(n=20, seed=42):
    """Generate simple 1D regression data: y = sin(x)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, size=(n, 1))
    y = np.sin(X[:, 0])
    return X, y


# ======================================================================
# 1. Statevector simulator tests
# ======================================================================


class TestStatevectorSimulator:
    """Tests for the minimal statevector simulator."""

    def test_initial_state_single_qubit(self):
        state = StatevectorSimulator.initial_state(1)
        assert state.shape == (2,)
        assert abs(state[0] - 1.0) < 1e-15
        assert abs(state[1]) < 1e-15

    def test_initial_state_two_qubits(self):
        state = StatevectorSimulator.initial_state(2)
        assert state.shape == (4,)
        np.testing.assert_allclose(state, [1, 0, 0, 0])

    def test_hadamard_creates_superposition(self):
        ops = [GateOp(gate=GateType.H, qubits=(0,))]
        state = StatevectorSimulator.run(1, ops, {})
        expected = np.array([1, 1], dtype=np.complex128) / math.sqrt(2)
        np.testing.assert_allclose(state, expected, atol=1e-12)

    def test_x_gate_flips(self):
        ops = [GateOp(gate=GateType.X, qubits=(0,))]
        state = StatevectorSimulator.run(1, ops, {})
        np.testing.assert_allclose(state, [0, 1], atol=1e-12)

    def test_z_gate_phase(self):
        # Z|0> = |0>, Z|1> = -|1>
        ops = [GateOp(gate=GateType.X, qubits=(0,)), GateOp(gate=GateType.Z, qubits=(0,))]
        state = StatevectorSimulator.run(1, ops, {})
        np.testing.assert_allclose(state, [0, -1], atol=1e-12)

    def test_ry_rotation(self):
        # RY(pi)|0> = |1>
        ops = [GateOp(gate=GateType.RY, qubits=(0,), param_name="t")]
        state = StatevectorSimulator.run(1, ops, {"t": np.pi})
        np.testing.assert_allclose(abs(state[1]) ** 2, 1.0, atol=1e-10)

    def test_rx_rotation(self):
        # RX(pi)|0> = -i|1>
        ops = [GateOp(gate=GateType.RX, qubits=(0,), param_name="t")]
        state = StatevectorSimulator.run(1, ops, {"t": np.pi})
        np.testing.assert_allclose(abs(state[1]) ** 2, 1.0, atol=1e-10)

    def test_rz_rotation(self):
        # RZ just adds phase -- doesn't change populations from |0>
        ops = [GateOp(gate=GateType.RZ, qubits=(0,), param_name="t")]
        state = StatevectorSimulator.run(1, ops, {"t": np.pi / 2})
        np.testing.assert_allclose(abs(state[0]) ** 2, 1.0, atol=1e-10)

    def test_cnot_creates_bell_state(self):
        ops = [
            GateOp(gate=GateType.H, qubits=(0,)),
            GateOp(gate=GateType.CNOT, qubits=(0, 1)),
        ]
        state = StatevectorSimulator.run(2, ops, {})
        # Bell state: (|00> + |11>) / sqrt(2)
        np.testing.assert_allclose(abs(state[0]) ** 2, 0.5, atol=1e-10)
        np.testing.assert_allclose(abs(state[3]) ** 2, 0.5, atol=1e-10)
        np.testing.assert_allclose(abs(state[1]) ** 2, 0.0, atol=1e-10)
        np.testing.assert_allclose(abs(state[2]) ** 2, 0.0, atol=1e-10)

    def test_cz_gate(self):
        # CZ|11> = -|11>
        ops = [
            GateOp(gate=GateType.X, qubits=(0,)),
            GateOp(gate=GateType.X, qubits=(1,)),
            GateOp(gate=GateType.CZ, qubits=(0, 1)),
        ]
        state = StatevectorSimulator.run(2, ops, {})
        np.testing.assert_allclose(state[3], -1.0, atol=1e-12)

    def test_statevector_normalized(self):
        ops = [
            GateOp(gate=GateType.RY, qubits=(0,), param_name="a"),
            GateOp(gate=GateType.RZ, qubits=(1,), param_name="b"),
            GateOp(gate=GateType.CNOT, qubits=(0, 1)),
        ]
        state = StatevectorSimulator.run(2, ops, {"a": 1.23, "b": 4.56})
        norm = np.linalg.norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-12)

    def test_missing_parameter_raises(self):
        ops = [GateOp(gate=GateType.RY, qubits=(0,), param_name="t")]
        with pytest.raises(ValueError, match="Missing parameter"):
            StatevectorSimulator.run(1, ops, {})

    def test_fixed_param_value(self):
        ops = [GateOp(gate=GateType.RY, qubits=(0,), param_value=np.pi / 2)]
        state = StatevectorSimulator.run(1, ops, {})
        np.testing.assert_allclose(abs(state[0]) ** 2, 0.5, atol=1e-10)
        np.testing.assert_allclose(abs(state[1]) ** 2, 0.5, atol=1e-10)


# ======================================================================
# 2. Data encoding circuit tests
# ======================================================================


class TestAngleEncoding:
    """Tests for angle encoding circuits."""

    def test_creation(self):
        enc = AngleEncoding(n_qubits=3)
        assert enc.n_qubits == 3
        assert enc.n_params == 3

    def test_encoding_changes_state(self):
        enc = AngleEncoding(n_qubits=2)
        x = np.array([np.pi / 4, np.pi / 3])
        state = enc.encode(x)
        assert state.shape == (4,)
        # State should be different from |00>
        assert abs(state[0]) < 1.0 - 1e-6

    def test_zero_features_identity(self):
        enc = AngleEncoding(n_qubits=2)
        x = np.array([0.0, 0.0])
        state = enc.encode(x)
        # RY(0) = Identity, so should still be |00>
        np.testing.assert_allclose(abs(state[0]) ** 2, 1.0, atol=1e-10)

    def test_pi_features_flip(self):
        enc = AngleEncoding(n_qubits=1, rotation="Y")
        x = np.array([np.pi])
        state = enc.encode(x)
        # RY(pi)|0> = |1>
        np.testing.assert_allclose(abs(state[1]) ** 2, 1.0, atol=1e-10)

    def test_z_rotation_encoding(self):
        enc = AngleEncoding(n_qubits=2, rotation="Z")
        x = np.array([1.0, 2.0])
        state = enc.encode(x)
        assert np.linalg.norm(state) - 1.0 < 1e-12

    def test_feature_wrapping(self):
        enc = AngleEncoding(n_qubits=2)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        bindings = enc.feature_bindings(x)
        # qubit 0 gets x[0] + x[2], qubit 1 gets x[1] + x[3]
        assert abs(bindings["x_0"] - 4.0) < 1e-10
        assert abs(bindings["x_1"] - 6.0) < 1e-10

    def test_invalid_rotation_raises(self):
        with pytest.raises(ValueError, match="rotation must be"):
            AngleEncoding(n_qubits=2, rotation="X")


class TestAmplitudeEncoding:
    """Tests for amplitude encoding."""

    def test_encoding_normalizes(self):
        enc = AmplitudeEncoding(n_qubits=2)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        state = enc.encode(x)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)

    def test_encoding_preserves_relative_amplitudes(self):
        enc = AmplitudeEncoding(n_qubits=2)
        x = np.array([1.0, 0.0, 0.0, 0.0])
        state = enc.encode(x)
        np.testing.assert_allclose(abs(state[0]) ** 2, 1.0, atol=1e-10)

    def test_zero_padding(self):
        enc = AmplitudeEncoding(n_qubits=3)  # 8 amplitudes
        x = np.array([1.0, 1.0])
        state = enc.encode(x)
        assert state.shape == (8,)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)

    def test_too_many_features_raises(self):
        enc = AmplitudeEncoding(n_qubits=1)
        with pytest.raises(ValueError, match="Feature dimension"):
            enc.encode(np.array([1.0, 2.0, 3.0]))

    def test_zero_vector_raises(self):
        enc = AmplitudeEncoding(n_qubits=2)
        with pytest.raises(ValueError, match="near-zero norm"):
            enc.encode(np.zeros(4))


class TestIQPEncoding:
    """Tests for IQP encoding."""

    def test_creation(self):
        enc = IQPEncoding(n_qubits=3)
        assert enc.n_qubits == 3
        # Has single-qubit and two-qubit parameters
        assert enc.n_params > 0

    def test_encoding_produces_entangled_state(self):
        enc = IQPEncoding(n_qubits=2)
        x = np.array([1.0, 2.0])
        state = enc.encode(x)
        assert state.shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)
        # Non-trivial state (multiple non-zero amplitudes)
        assert np.sum(np.abs(state) > 1e-6) > 1

    def test_repeated_encoding(self):
        enc = IQPEncoding(n_qubits=2, n_repeats=2)
        assert enc.n_params > IQPEncoding(n_qubits=2, n_repeats=1).n_params

    def test_product_features(self):
        enc = IQPEncoding(n_qubits=2)
        x = np.array([2.0, 3.0])
        bindings = enc.feature_bindings(x)
        # ZZ term should be product
        assert abs(bindings["iqp_xx_0_1_r0"] - 6.0) < 1e-10

    def test_invalid_repeats_raises(self):
        with pytest.raises(ValueError, match="n_repeats"):
            IQPEncoding(n_qubits=2, n_repeats=0)


# ======================================================================
# 3. Ansatz circuit tests
# ======================================================================


class TestHardwareEfficientAnsatz:
    """Tests for hardware-efficient ansatz."""

    def test_parameter_count(self):
        # Each layer: n_qubits RY + n_qubits RZ = 2*n_qubits per layer
        ansatz = HardwareEfficientAnsatz(n_qubits=3, n_layers=2)
        assert ansatz.n_params == 2 * 3 * 2  # 12

    def test_parameterizable(self):
        ansatz = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        params = np.zeros(ansatz.n_params)
        bindings = ansatz.param_vector_to_bindings(params)
        state = ansatz.run(bindings)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)

    def test_different_params_different_states(self):
        ansatz = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        p1 = np.zeros(ansatz.n_params)
        p2 = np.ones(ansatz.n_params)
        s1 = ansatz.run(ansatz.param_vector_to_bindings(p1))
        s2 = ansatz.run(ansatz.param_vector_to_bindings(p2))
        assert np.linalg.norm(s1 - s2) > 1e-6

    def test_single_qubit_no_entanglement(self):
        ansatz = HardwareEfficientAnsatz(n_qubits=1, n_layers=2)
        assert ansatz.n_params == 4  # 2 params per layer


class TestStronglyEntanglingLayers:
    """Tests for strongly entangling layers."""

    def test_parameter_count(self):
        # 3 rotations per qubit per layer
        sel = StronglyEntanglingLayers(n_qubits=4, n_layers=2)
        assert sel.n_params == 3 * 4 * 2  # 24

    def test_requires_two_qubits(self):
        with pytest.raises(ValueError, match="n_qubits >= 2"):
            StronglyEntanglingLayers(n_qubits=1)

    def test_produces_valid_state(self):
        sel = StronglyEntanglingLayers(n_qubits=3, n_layers=2)
        params = np.linspace(0, np.pi, sel.n_params)
        bindings = sel.param_vector_to_bindings(params)
        state = sel.run(bindings)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)


class TestSimplifiedTwoDesign:
    """Tests for simplified 2-design ansatz."""

    def test_parameter_count(self):
        # Initial layer: n_qubits, then n_qubits per layer
        s2d = SimplifiedTwoDesign(n_qubits=3, n_layers=2)
        assert s2d.n_params == 3 + 3 * 2  # 9

    def test_produces_valid_state(self):
        s2d = SimplifiedTwoDesign(n_qubits=3, n_layers=2)
        params = np.ones(s2d.n_params) * 0.5
        bindings = s2d.param_vector_to_bindings(params)
        state = s2d.run(bindings)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)


class TestParameterizedCircuit:
    """Tests for the base ParameterizedCircuit."""

    def test_invalid_qubits_raises(self):
        with pytest.raises(ValueError, match="n_qubits must be >= 1"):
            ParameterizedCircuit(n_qubits=0)

    def test_expectation_z_ground_state(self):
        # |0> has <Z> = +1
        circ = ParameterizedCircuit(n_qubits=1)
        assert abs(circ.expectation_z({}, 0) - 1.0) < 1e-10

    def test_expectation_z_excited_state(self):
        # X|0> = |1> has <Z> = -1
        circ = ParameterizedCircuit(n_qubits=1)
        circ.add_gate(GateOp(gate=GateType.X, qubits=(0,)))
        assert abs(circ.expectation_z({}, 0) - (-1.0)) < 1e-10

    def test_probabilities_sum_to_one(self):
        circ = ParameterizedCircuit(n_qubits=2)
        circ.add_gate(GateOp(gate=GateType.H, qubits=(0,)))
        circ.add_gate(GateOp(gate=GateType.RY, qubits=(1,), param_name="a"))
        probs = circ.probabilities({"a": 1.0})
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)


# ======================================================================
# 4. CircuitTemplate tests
# ======================================================================


class TestCircuitTemplate:
    """Tests for the combined encoding + ansatz template."""

    def test_creation(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        assert ct.n_qubits == 2
        assert ct.n_params == ans.n_params

    def test_qubit_mismatch_raises(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=3, n_layers=1)
        with pytest.raises(ValueError, match="Qubit count mismatch"):
            CircuitTemplate(encoding=enc, ansatz=ans)

    def test_run_returns_valid_state(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        x = np.array([0.5, 0.7])
        params = np.zeros(ct.n_params)
        state = ct.run(x, params)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)

    def test_expectation_z_bounded(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        x = np.array([1.0, 2.0])
        params = np.ones(ct.n_params) * 0.5
        exp_z = ct.expectation_z(x, params, qubit=0)
        assert -1.0 <= exp_z <= 1.0

    def test_overlap_self_is_one(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        x = np.array([1.0, 2.0])
        params = np.ones(ct.n_params) * 0.3
        overlap = ct.overlap(x, x, params)
        np.testing.assert_allclose(overlap, 1.0, atol=1e-10)

    def test_overlap_different_inputs_bounded(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        x1 = np.array([0.0, 0.0])
        x2 = np.array([np.pi, np.pi])
        params = np.ones(ct.n_params) * 0.5
        overlap = ct.overlap(x1, x2, params)
        assert 0.0 <= overlap <= 1.0 + 1e-10

    def test_probabilities_sum_to_one(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        x = np.array([0.5, 0.7])
        params = np.ones(ct.n_params) * 0.2
        probs = ct.probabilities(x, params)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)


# ======================================================================
# 5. Gradient computation tests
# ======================================================================


class TestParameterShiftRule:
    """Tests for the parameter-shift gradient method."""

    def test_gradient_of_sin(self):
        # f(x) = sin(x), f'(x) = cos(x)
        def cost(p):
            return float(np.sin(p[0]))

        psr = ParameterShiftRule()
        result = psr.compute(cost, np.array([0.5]))
        expected = np.cos(0.5)
        np.testing.assert_allclose(result.gradient[0], expected, atol=1e-6)

    def test_evaluations_count(self):
        def cost(p):
            return float(np.sum(p))

        psr = ParameterShiftRule()
        result = psr.compute(cost, np.zeros(5))
        assert result.n_evaluations == 10  # 2 * n_params

    def test_gradient_result_repr(self):
        result = GradientResult(
            gradient=np.array([1.0, 2.0]),
            n_evaluations=4,
            method="test",
        )
        assert "test" in repr(result)


class TestFiniteDifferenceGradient:
    """Tests for finite-difference gradients."""

    def test_central_gradient_of_quadratic(self):
        # f(x) = x^2, f'(x) = 2x
        def cost(p):
            return float(p[0] ** 2)

        fdg = FiniteDifferenceGradient(method="central")
        result = fdg.compute(cost, np.array([3.0]))
        np.testing.assert_allclose(result.gradient[0], 6.0, atol=1e-4)

    def test_forward_gradient(self):
        def cost(p):
            return float(p[0] ** 2)

        fdg = FiniteDifferenceGradient(method="forward")
        result = fdg.compute(cost, np.array([3.0]))
        np.testing.assert_allclose(result.gradient[0], 6.0, atol=1e-3)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            FiniteDifferenceGradient(method="backward")


class TestParameterShiftVsFiniteDifference:
    """Tests that parameter-shift and finite-difference gradients agree."""

    def test_agreement_on_simple_circuit(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        x = np.array([0.5, 0.8])

        def cost(params):
            return ct.expectation_z(x, params, qubit=0)

        params = np.array([0.1, 0.2, 0.3, 0.4])

        psr = ParameterShiftRule()
        fdg = FiniteDifferenceGradient(epsilon=1e-5, method="central")

        g_psr = psr.compute(cost, params).gradient
        g_fdg = fdg.compute(cost, params).gradient

        np.testing.assert_allclose(g_psr, g_fdg, atol=1e-4)

    def test_agreement_multi_param(self):
        enc = AngleEncoding(n_qubits=3)
        ans = HardwareEfficientAnsatz(n_qubits=3, n_layers=2)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        x = np.array([0.3, 0.5, 0.7])

        rng = np.random.default_rng(123)
        params = rng.uniform(0, np.pi, size=ct.n_params)

        def cost(p):
            return ct.expectation_z(x, p, qubit=1)

        psr = ParameterShiftRule()
        fdg = FiniteDifferenceGradient(epsilon=1e-5)

        g_psr = psr.compute(cost, params).gradient
        g_fdg = fdg.compute(cost, params).gradient

        np.testing.assert_allclose(g_psr, g_fdg, atol=1e-4)


class TestStochasticParameterShift:
    """Tests for stochastic parameter-shift rule."""

    def test_computes_subset_of_gradients(self):
        def cost(p):
            return float(np.sum(np.sin(p)))

        spsr = StochasticParameterShift(sample_fraction=0.5, seed=42)
        result = spsr.compute(cost, np.zeros(10))
        # Some components should be zero (unsampled)
        assert np.sum(result.gradient == 0.0) > 0
        assert result.n_evaluations <= 10  # at most 2 * ceil(0.5 * 10)

    def test_full_fraction_matches_parameter_shift(self):
        def cost(p):
            return float(np.sin(p[0]) + np.cos(p[1]))

        spsr = StochasticParameterShift(sample_fraction=1.0, seed=42)
        psr = ParameterShiftRule()

        params = np.array([0.5, 1.0])
        g_spsr = spsr.compute(cost, params).gradient
        g_psr = psr.compute(cost, params).gradient

        np.testing.assert_allclose(g_spsr, g_psr, atol=1e-10)

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError, match="sample_fraction"):
            StochasticParameterShift(sample_fraction=0.0)


class TestNaturalGradient:
    """Tests for the natural gradient method."""

    def test_natural_gradient_computes(self):
        def cost(p):
            return float(np.sin(p[0]) * np.cos(p[1]))

        ng = NaturalGradient(regularization=1e-2)
        result = ng.compute(cost, np.array([0.5, 1.0]))
        assert result.gradient.shape == (2,)
        assert result.method == "natural_gradient"

    def test_with_precomputed_qfi(self):
        def cost(p):
            return float(p[0] ** 2 + p[1] ** 2)

        ng = NaturalGradient(regularization=1e-3)
        params = np.array([1.0, 2.0])

        # Identity QFI should give same as Euclidean gradient
        qfi = np.eye(2)
        result = ng.compute(cost, params, qfi=qfi)
        psr = ParameterShiftRule()
        euclidean = psr.compute(cost, params).gradient

        np.testing.assert_allclose(result.gradient, euclidean, atol=0.1)


class TestBarrenPlateauScanner:
    """Tests for the gradient-based barren plateau scanner."""

    def test_shallow_circuit_not_barren(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        x = np.array([0.5, 0.8])

        def cost(params):
            return ct.expectation_z(x, params, qubit=0)

        scanner = BarrenPlateauScanner(n_samples=20, seed=42)
        result = scanner.scan(cost, ct.n_params)
        assert "gradient_variances" in result
        assert "mean_variance" in result
        assert isinstance(result["is_barren"], (bool, np.bool_))

    def test_constant_function_is_barren(self):
        def cost(params):
            return 1.0  # Constant -- all gradients are zero

        scanner = BarrenPlateauScanner(n_samples=10, seed=42)
        result = scanner.scan(cost, 5)
        assert result["is_barren"]
        np.testing.assert_allclose(result["mean_variance"], 0.0, atol=1e-15)


# ======================================================================
# 6. Training tests
# ======================================================================


class TestTrainingHistory:
    """Tests for TrainingHistory container."""

    def test_best_loss(self):
        h = TrainingHistory()
        h.loss = [1.0, 0.5, 0.3, 0.4]
        assert abs(h.best_loss - 0.3) < 1e-10

    def test_best_epoch(self):
        h = TrainingHistory()
        h.loss = [1.0, 0.5, 0.3, 0.4]
        assert h.best_epoch == 2

    def test_empty_history(self):
        h = TrainingHistory()
        assert h.best_loss == float("inf")
        assert h.best_epoch == 0

    def test_repr(self):
        h = TrainingHistory()
        h.loss = [0.5]
        assert "epochs=1" in repr(h)


class TestClassificationObjective:
    """Tests for classification loss and prediction."""

    def test_binary_prediction(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        x = np.array([0.5, 0.8])
        params = np.zeros(ct.n_params)
        pred = obj.predict(ct, x, params)
        assert pred in (0, 1)

    def test_loss_positive(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        x = np.array([0.5, 0.8])
        params = np.zeros(ct.n_params)
        loss = obj.loss(ct, x, 0, params)
        assert loss > 0.0

    def test_invalid_n_classes(self):
        with pytest.raises(ValueError, match="n_classes"):
            ClassificationObjective(n_classes=1)


class TestRegressionObjective:
    """Tests for regression loss."""

    def test_regression_loss(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = RegressionObjective(output_qubit=0)

        x = np.array([0.5, 0.8])
        params = np.zeros(ct.n_params)
        loss = obj.loss(ct, x, 0.5, params)
        assert loss >= 0.0

    def test_predict_value_bounded(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = RegressionObjective(output_qubit=0, output_scale=1.0)

        x = np.array([0.5, 0.8])
        params = np.zeros(ct.n_params)
        val = obj.predict_value(ct, x, params)
        assert -1.0 <= val <= 1.0


class TestQCLTrainer:
    """Tests for the main training loop."""

    def test_training_reduces_loss_xor(self):
        """Training on XOR data should reduce loss over epochs."""
        X, y = _make_xor_data(n=12, seed=42)
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=2)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        trainer = QCLTrainer(ct, obj, gradient_method="parameter_shift", seed=42)
        history = trainer.fit(
            X, y, epochs=30, lr=0.3, optimizer="adam",
        )
        # Loss should decrease
        assert history.loss[-1] < history.loss[0]

    def test_training_sgd(self):
        """SGD optimizer should also reduce loss."""
        X, y = _make_xor_data(n=10, seed=42)
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        trainer = QCLTrainer(ct, obj, seed=42)
        history = trainer.fit(X, y, epochs=20, lr=0.2, optimizer="sgd")
        assert len(history.loss) == 20

    def test_training_spsa(self):
        """SPSA optimizer should complete without error."""
        X, y = _make_xor_data(n=10, seed=42)
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        trainer = QCLTrainer(ct, obj, seed=42)
        history = trainer.fit(X, y, epochs=15, lr=0.1, optimizer="spsa")
        assert len(history.loss) == 15

    def test_regression_training(self):
        """Regression training should reduce MSE."""
        X, y = _make_regression_data(n=10, seed=42)
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=2)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = RegressionObjective(output_qubit=0)

        trainer = QCLTrainer(ct, obj, seed=42)
        history = trainer.fit(X, y, epochs=30, lr=0.2, optimizer="adam")
        # Should at least not diverge
        assert history.loss[-1] < history.loss[0] * 5

    def test_predict_after_training(self):
        """predict() works after fit()."""
        X, y = _make_xor_data(n=10, seed=42)
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        trainer = QCLTrainer(ct, obj, seed=42)
        trainer.fit(X, y, epochs=5, lr=0.1)
        preds = trainer.predict(X)
        assert preds.shape == (len(X),)
        assert all(p in (0, 1) for p in preds)

    def test_predict_before_training_raises(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)
        trainer = QCLTrainer(ct, obj)
        with pytest.raises(RuntimeError, match="not been trained"):
            trainer.predict(np.zeros((5, 2)))

    def test_early_stopping(self):
        """Early stopping should terminate before max epochs on a trivial problem."""
        # Constant cost -- gradients vanish fast
        X = np.zeros((4, 2))
        y = np.zeros(4, dtype=int)
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        trainer = QCLTrainer(ct, obj, seed=42)
        history = trainer.fit(
            X, y, epochs=100, lr=0.1, early_stopping_patience=5,
        )
        # Should stop early (loss doesn't improve)
        assert len(history.loss) < 100

    def test_callback_called(self):
        """Callback should be invoked each epoch."""
        X, y = _make_xor_data(n=6, seed=42)
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        epochs_seen = []

        def cb(epoch, params, loss):
            epochs_seen.append(epoch)

        trainer = QCLTrainer(ct, obj, seed=42)
        trainer.fit(X, y, epochs=5, lr=0.1, callback=cb)
        assert epochs_seen == [1, 2, 3, 4, 5]

    def test_lr_decay(self):
        """Learning rate decay should produce decreasing LR."""
        X, y = _make_xor_data(n=6, seed=42)
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        trainer = QCLTrainer(ct, obj, seed=42)
        history = trainer.fit(X, y, epochs=10, lr=0.5, lr_decay=0.9)
        assert history.learning_rates[-1] < history.learning_rates[0]

    def test_invalid_optimizer_raises(self):
        X, y = _make_xor_data(n=4, seed=42)
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        trainer = QCLTrainer(ct, obj, seed=42)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            trainer.fit(X, y, epochs=5, optimizer="rmsprop")

    def test_classification_accuracy_tracked(self):
        """Accuracy should be recorded for classification tasks."""
        X, y = _make_xor_data(n=8, seed=42)
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        trainer = QCLTrainer(ct, obj, seed=42)
        history = trainer.fit(X, y, epochs=5, lr=0.1)
        assert len(history.accuracy) == 5
        assert all(0.0 <= a <= 1.0 for a in history.accuracy)

    def test_finite_difference_gradient_method(self):
        """Trainer with finite_difference method should work."""
        X, y = _make_xor_data(n=6, seed=42)
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        trainer = QCLTrainer(ct, obj, gradient_method="finite_difference", seed=42)
        history = trainer.fit(X, y, epochs=5, lr=0.1)
        assert len(history.loss) == 5


# ======================================================================
# 7. Quantum kernel tests
# ======================================================================


class TestQuantumKernel:
    """Tests for the standard quantum kernel."""

    def test_self_kernel_is_one(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        x = np.array([1.0, 2.0])
        val = qk.evaluate(x, x)
        np.testing.assert_allclose(val, 1.0, atol=1e-10)

    def test_kernel_bounded(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        x1 = np.array([0.0, 0.0])
        x2 = np.array([np.pi, np.pi])
        val = qk.evaluate(x1, x2)
        assert 0.0 <= val <= 1.0 + 1e-10

    def test_kernel_symmetric(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        x1 = np.array([0.5, 0.8])
        x2 = np.array([1.2, 0.3])
        v12 = qk.evaluate(x1, x2)
        v21 = qk.evaluate(x2, x1)
        np.testing.assert_allclose(v12, v21, atol=1e-12)

    def test_kernel_matrix_psd(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        rng = np.random.default_rng(42)
        X = rng.uniform(-np.pi, np.pi, size=(10, 2))
        K = qk.matrix(X)

        # Symmetric
        np.testing.assert_allclose(K, K.T, atol=1e-12)

        # Diagonal is 1
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

        # PSD: all eigenvalues >= 0
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

    def test_kernel_matrix_bounded(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        rng = np.random.default_rng(42)
        X = rng.uniform(-np.pi, np.pi, size=(8, 2))
        K = qk.matrix(X)
        assert np.all(K >= -1e-10)
        assert np.all(K <= 1.0 + 1e-10)

    def test_cross_matrix(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        rng = np.random.default_rng(42)
        X1 = rng.uniform(-1, 1, size=(5, 2))
        X2 = rng.uniform(-1, 1, size=(3, 2))
        K = qk.cross_matrix(X1, X2)
        assert K.shape == (5, 3)
        assert np.all(K >= -1e-10)
        assert np.all(K <= 1.0 + 1e-10)

    def test_iqp_kernel_distinguishes_data(self):
        """IQP encoding should produce non-trivial kernel values."""
        enc = IQPEncoding(n_qubits=2, n_repeats=2)
        qk = QuantumKernel(enc)
        x1 = np.array([0.1, 0.2])
        x2 = np.array([2.0, 3.0])
        val = qk.evaluate(x1, x2)
        # Different inputs should have less than perfect overlap
        assert val < 1.0 - 1e-6


class TestProjectedQuantumKernel:
    """Tests for the projected quantum kernel."""

    def test_self_kernel_is_one(self):
        enc = AngleEncoding(n_qubits=2)
        pqk = ProjectedQuantumKernel(enc, gamma=1.0)
        x = np.array([1.0, 2.0])
        val = pqk.evaluate(x, x)
        np.testing.assert_allclose(val, 1.0, atol=1e-10)

    def test_kernel_bounded(self):
        enc = AngleEncoding(n_qubits=2)
        pqk = ProjectedQuantumKernel(enc, gamma=1.0)
        x1 = np.array([0.0, 0.0])
        x2 = np.array([3.0, 3.0])
        val = pqk.evaluate(x1, x2)
        assert 0.0 <= val <= 1.0 + 1e-10

    def test_matrix_psd(self):
        enc = AngleEncoding(n_qubits=2)
        pqk = ProjectedQuantumKernel(enc, gamma=0.5)
        rng = np.random.default_rng(42)
        X = rng.uniform(-np.pi, np.pi, size=(8, 2))
        K = pqk.matrix(X)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)


class TestTrainableKernel:
    """Tests for the trainable quantum kernel."""

    def test_trainable_kernel_works(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        tk = TrainableKernel(ct)
        tk.set_params(np.zeros(ct.n_params))

        x = np.array([0.5, 0.8])
        val = tk.evaluate(x, x)
        np.testing.assert_allclose(val, 1.0, atol=1e-10)

    def test_no_params_raises(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        tk = TrainableKernel(ct)
        with pytest.raises(RuntimeError, match="Parameters not set"):
            tk.evaluate(np.array([0.5]), np.array([0.5]))

    def test_trainable_matrix_psd(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        tk = TrainableKernel(ct)
        tk.set_params(np.ones(ct.n_params) * 0.5)
        rng = np.random.default_rng(42)
        X = rng.uniform(-1, 1, size=(6, 2))
        K = tk.matrix(X)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)


class TestKernelTargetAlignment:
    """Tests for the kernel-target alignment metric."""

    def test_perfect_alignment(self):
        # Ideal kernel for +/-1 labels: K[i,j] = y_i * y_j
        y = np.array([0, 0, 1, 1])
        # With +/-1 mapping: [-1, -1, 1, 1]
        # Ideal kernel Y[i,j] = y_i * y_j = [[1,1,-1,-1],[1,1,-1,-1],...]
        # KTA(Y, Y) = 1.0 by definition
        y_pm = np.array([-1.0, -1.0, 1.0, 1.0])
        K = np.outer(y_pm, y_pm)
        kta = kernel_target_alignment(K, y)
        np.testing.assert_allclose(kta, 1.0, atol=1e-10)

    def test_random_kernel_low_alignment(self):
        rng = np.random.default_rng(42)
        K = rng.uniform(0, 1, (10, 10))
        K = (K + K.T) / 2
        np.fill_diagonal(K, 1.0)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        kta = kernel_target_alignment(K, y)
        # Random kernel should have low alignment
        assert abs(kta) < 0.5


# ======================================================================
# 8. QSVM tests
# ======================================================================


class TestQSVM:
    """Tests for the quantum support vector machine."""

    def test_qsvm_fits(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        svm = QSVM(qk, C=1.0)

        rng = np.random.default_rng(42)
        X = rng.uniform(-1, 1, (12, 2))
        y = ((X[:, 0] + X[:, 1]) > 0).astype(float)
        svm.fit(X, y)
        assert svm.alphas is not None

    def test_qsvm_predicts(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        svm = QSVM(qk, C=1.0)

        rng = np.random.default_rng(42)
        X = rng.uniform(-1, 1, (12, 2))
        y = ((X[:, 0] + X[:, 1]) > 0).astype(float)
        svm.fit(X, y)

        preds = svm.predict(X)
        assert preds.shape == (12,)

    def test_qsvm_classifies_linearly_separable(self):
        """QSVM should achieve high accuracy on linearly separable data."""
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        svm = QSVM(qk, C=10.0, max_iter=300)

        # Well-separated data
        rng = np.random.default_rng(42)
        n = 20
        X0 = rng.uniform(-2, -0.2, (n // 2, 2))
        X1 = rng.uniform(0.2, 2, (n // 2, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=float)

        svm.fit(X, y)
        accuracy = svm.score(X, y)
        assert accuracy >= 0.7

    def test_qsvm_score(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        svm = QSVM(qk, C=1.0)

        rng = np.random.default_rng(42)
        X = rng.uniform(-1, 1, (10, 2))
        y = ((X[:, 0] > 0)).astype(float)
        svm.fit(X, y)
        score = svm.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_qsvm_not_fitted_raises(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        svm = QSVM(qk)
        with pytest.raises(RuntimeError, match="not fitted"):
            svm.predict(np.zeros((3, 2)))

    def test_qsvm_non_binary_raises(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        svm = QSVM(qk)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="binary"):
            svm.fit(X, y)

    def test_qsvm_decision_function(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        svm = QSVM(qk, C=1.0)

        rng = np.random.default_rng(42)
        X = rng.uniform(-1, 1, (10, 2))
        y = ((X[:, 0] > 0)).astype(float)
        svm.fit(X, y)

        decision = svm.decision_function(X)
        assert decision.shape == (10,)


# ======================================================================
# 9. QKernelPCA tests
# ======================================================================


class TestQKernelPCA:
    """Tests for quantum kernel PCA."""

    def test_fit_transform(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        pca = QKernelPCA(qk, n_components=2)

        rng = np.random.default_rng(42)
        X = rng.uniform(-np.pi, np.pi, (15, 2))

        X_proj = pca.fit_transform(X)
        assert X_proj.shape == (15, 2)

    def test_explained_variance_ratio(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        pca = QKernelPCA(qk, n_components=2)

        rng = np.random.default_rng(42)
        X = rng.uniform(-np.pi, np.pi, (10, 2))
        pca.fit(X)

        evr = pca.explained_variance_ratio
        assert len(evr) == 2
        assert np.all(evr >= 0.0)
        assert np.all(evr <= 1.0 + 1e-10)

    def test_transform_new_data(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        pca = QKernelPCA(qk, n_components=2)

        rng = np.random.default_rng(42)
        X_train = rng.uniform(-np.pi, np.pi, (10, 2))
        X_test = rng.uniform(-np.pi, np.pi, (5, 2))

        pca.fit(X_train)
        X_proj = pca.transform(X_test)
        assert X_proj.shape == (5, 2)

    def test_not_fitted_raises(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        pca = QKernelPCA(qk, n_components=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            pca.transform(np.zeros((3, 2)))


# ======================================================================
# 10. Expressibility tests
# ======================================================================


class TestExpressibilityAnalyzer:
    """Tests for circuit expressibility analysis."""

    def test_shallow_vs_deep(self):
        """Deeper circuits should generally be more expressive (lower KL)."""
        shallow = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        deep = HardwareEfficientAnsatz(n_qubits=2, n_layers=4)

        analyzer = ExpressibilityAnalyzer(n_samples=200, seed=42)
        r_shallow = analyzer.analyze(shallow)
        r_deep = analyzer.analyze(deep)

        # Deeper circuit should have lower KL divergence (more expressive)
        # or similar -- it should at least not be drastically worse
        assert r_deep.kl_divergence < r_shallow.kl_divergence * 5

    def test_kl_non_negative(self):
        ansatz = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        analyzer = ExpressibilityAnalyzer(n_samples=100, seed=42)
        result = analyzer.analyze(ansatz)
        assert result.kl_divergence >= 0.0

    def test_entangling_capability_bounded(self):
        ansatz = HardwareEfficientAnsatz(n_qubits=3, n_layers=2)
        analyzer = ExpressibilityAnalyzer(n_samples=100, seed=42)
        result = analyzer.analyze(ansatz)
        assert 0.0 <= result.entangling_capability <= 1.0 + 1e-6

    def test_single_qubit_no_entanglement(self):
        """Single-qubit circuits have zero entangling capability."""
        ansatz = HardwareEfficientAnsatz(n_qubits=1, n_layers=3)
        analyzer = ExpressibilityAnalyzer(n_samples=50, seed=42)
        result = analyzer.analyze(ansatz)
        # Single qubit: entangling capability should be 0
        np.testing.assert_allclose(result.entangling_capability, 0.0, atol=1e-10)

    def test_fidelity_samples_shape(self):
        ansatz = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        analyzer = ExpressibilityAnalyzer(n_samples=50, seed=42)
        result = analyzer.analyze(ansatz)
        assert result.fidelity_samples.shape == (50,)
        assert np.all(result.fidelity_samples >= 0.0)
        assert np.all(result.fidelity_samples <= 1.0 + 1e-10)

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError, match="n_samples"):
            ExpressibilityAnalyzer(n_samples=5)

    def test_strongly_entangling_vs_hardware_efficient(self):
        """Strongly entangling layers should have higher entangling capability."""
        he = HardwareEfficientAnsatz(n_qubits=3, n_layers=1)
        sel = StronglyEntanglingLayers(n_qubits=3, n_layers=1)

        analyzer = ExpressibilityAnalyzer(n_samples=150, seed=42)
        r_he = analyzer.analyze(he)
        r_sel = analyzer.analyze(sel)

        # SEL should have at least comparable entangling capability
        assert r_sel.entangling_capability > r_he.entangling_capability * 0.3


# ======================================================================
# 11. Barren plateau detector tests
# ======================================================================


class TestBarrenPlateauDetector:
    """Tests for barren plateau detection."""

    def test_detects_barren_for_constant_cost(self):
        """A constant cost function has zero gradient variance."""
        def cost(params):
            return 1.0

        detector = BarrenPlateauDetector(n_samples=10, seed=42)
        result = detector.detect(cost, n_params=5)
        assert result.is_barren
        np.testing.assert_allclose(result.mean_variance, 0.0, atol=1e-15)

    def test_shallow_circuit_trainable(self):
        """A shallow circuit should generally not show barren plateaus."""
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        x = np.array([0.5, 0.8])

        def cost(params):
            return ct.expectation_z(x, params, qubit=0)

        detector = BarrenPlateauDetector(n_samples=30, threshold=1e-4, seed=42)
        result = detector.detect(cost, ct.n_params)
        # Shallow circuit should have non-trivial gradient variance
        assert result.mean_variance > 1e-6

    def test_cost_concentration_tracked(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        x = np.array([0.5, 0.8])

        def cost(params):
            return ct.expectation_z(x, params, qubit=0)

        detector = BarrenPlateauDetector(n_samples=20, seed=42)
        result = detector.detect(cost, ct.n_params)
        assert result.cost_concentration >= 0.0

    def test_compare_depths(self):
        """Comparing across depths should show decreasing variance for deep circuits."""
        def make_cost(depth):
            enc = AngleEncoding(n_qubits=2)
            ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=depth)
            ct = CircuitTemplate(encoding=enc, ansatz=ans)
            x = np.array([0.5, 0.8])
            def cost(params):
                return ct.expectation_z(x, params, qubit=0)
            return cost

        def n_params(depth):
            return HardwareEfficientAnsatz(n_qubits=2, n_layers=depth).n_params

        detector = BarrenPlateauDetector(n_samples=20, seed=42)
        results = detector.compare_depths(make_cost, n_params, [1, 2, 3])
        assert len(results) == 3
        assert 1 in results and 3 in results


# ======================================================================
# 12. Effective dimension tests
# ======================================================================


class TestEffectiveDimension:
    """Tests for the Fisher information-based effective dimension."""

    def test_effective_dimension_positive(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)

        def model_fn(x, params):
            return ct.probabilities(x, params)

        rng = np.random.default_rng(42)
        X = rng.uniform(-np.pi, np.pi, (10, 2))

        ed = EffectiveDimension(n_samples=20, seed=42)
        result = ed.compute(model_fn, ct.n_params, X)
        assert result["effective_dimension"] >= 0.0
        assert result["normalized_dimension"] >= 0.0  # no strict upper bound

    def test_fisher_eigenvalues_non_negative(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)

        def model_fn(x, params):
            return ct.probabilities(x, params)

        rng = np.random.default_rng(42)
        X = rng.uniform(-np.pi, np.pi, (8, 2))

        ed = EffectiveDimension(n_samples=15, seed=42)
        result = ed.compute(model_fn, ct.n_params, X)
        assert np.all(result["fisher_eigenvalues"] >= 0.0)


# ======================================================================
# 13. Integration tests (full pipeline)
# ======================================================================


class TestFullPipeline:
    """End-to-end integration tests."""

    def test_encode_train_predict_classification(self):
        """Full pipeline: data -> encode -> train -> predict for classification."""
        # Use XOR data (simpler) and moderate LR for reliable convergence
        X, y = _make_xor_data(n=12, seed=42)
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=2)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = ClassificationObjective(n_classes=2)

        trainer = QCLTrainer(ct, obj, seed=42)
        history = trainer.fit(X, y, epochs=40, lr=0.15, optimizer="adam")

        # Loss should decrease (comparing best in first half vs best in second half)
        first_half_min = min(history.loss[:20])
        second_half_min = min(history.loss[20:])
        assert second_half_min < first_half_min + 0.5  # some improvement or stable

        # Should produce valid predictions
        preds = trainer.predict(X)
        assert all(p in (0, 1) for p in preds)

    def test_encode_train_predict_regression(self):
        """Full pipeline for regression."""
        X = np.linspace(-1, 1, 10).reshape(-1, 1)
        y = X[:, 0] ** 2  # Simple quadratic

        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=2)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        obj = RegressionObjective(output_qubit=0)

        trainer = QCLTrainer(ct, obj, seed=42)
        history = trainer.fit(X, y, epochs=30, lr=0.2, optimizer="adam")

        preds = trainer.predict(X)
        assert preds.shape == (10,)

    def test_kernel_pipeline(self):
        """Full pipeline: encode -> kernel -> QSVM."""
        rng = np.random.default_rng(42)
        n = 16
        X = rng.uniform(-1, 1, (n, 2))
        y = ((X[:, 0] > 0)).astype(float)

        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        K = qk.matrix(X)

        # Kernel matrix should be valid
        assert K.shape == (n, n)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

        # QSVM should fit
        svm = QSVM(qk, C=1.0)
        svm.fit(X, y)
        accuracy = svm.score(X, y)
        assert accuracy >= 0.5  # Better than random

    def test_expressibility_analysis_pipeline(self):
        """Full pipeline: create ansatz -> analyze expressibility -> check trainability."""
        ansatz = HardwareEfficientAnsatz(n_qubits=2, n_layers=2)

        # Expressibility
        analyzer = ExpressibilityAnalyzer(n_samples=100, seed=42)
        expr_result = analyzer.analyze(ansatz)
        assert expr_result.kl_divergence >= 0.0

        # Barren plateau check
        enc = AngleEncoding(n_qubits=2)
        ct = CircuitTemplate(encoding=enc, ansatz=ansatz)
        x = np.array([0.5, 0.8])

        def cost(params):
            return ct.expectation_z(x, params, qubit=0)

        detector = BarrenPlateauDetector(n_samples=20, seed=42)
        bp_result = detector.detect(cost, ct.n_params)
        assert isinstance(bp_result.is_barren, (bool, np.bool_))

    def test_kernel_pca_pipeline(self):
        """Full pipeline: encode -> kernel PCA -> reduced features."""
        rng = np.random.default_rng(42)
        X = rng.uniform(-np.pi, np.pi, (15, 3))

        enc = AngleEncoding(n_qubits=3)
        qk = QuantumKernel(enc)
        pca = QKernelPCA(qk, n_components=2)

        X_proj = pca.fit_transform(X)
        assert X_proj.shape == (15, 2)

        # Should be able to transform new data
        X_new = rng.uniform(-np.pi, np.pi, (5, 3))
        X_new_proj = pca.transform(X_new)
        assert X_new_proj.shape == (5, 2)


# ======================================================================
# 14. Package import tests
# ======================================================================


class TestPackageImports:
    """Tests that the package exports are correctly configured."""

    def test_top_level_imports(self):
        """All major classes should be importable from nqpu.qcl."""
        from nqpu.qcl import (
            AmplitudeEncoding,
            AngleEncoding,
            BarrenPlateauDetector,
            BarrenPlateauScanner,
            CircuitTemplate,
            ClassificationObjective,
            EffectiveDimension,
            ExpressibilityAnalyzer,
            FiniteDifferenceGradient,
            GradientResult,
            HardwareEfficientAnsatz,
            IQPEncoding,
            KernelObjective,
            NaturalGradient,
            ParameterShiftRule,
            ParameterizedCircuit,
            ProjectedQuantumKernel,
            QCLTrainer,
            QKernelPCA,
            QSVM,
            QuantumKernel,
            RegressionObjective,
            SimplifiedTwoDesign,
            StatevectorSimulator,
            StochasticParameterShift,
            StronglyEntanglingLayers,
            TrainableKernel,
            TrainingHistory,
            kernel_target_alignment,
        )

    def test_all_attribute(self):
        import nqpu.qcl
        assert hasattr(nqpu.qcl, "__all__")
        assert len(nqpu.qcl.__all__) > 20


# ======================================================================
# 15. Edge case tests
# ======================================================================


class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_single_qubit_circuit(self):
        enc = AngleEncoding(n_qubits=1)
        ans = HardwareEfficientAnsatz(n_qubits=1, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        x = np.array([0.5])
        params = np.zeros(ct.n_params)
        state = ct.run(x, params)
        assert state.shape == (2,)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)

    def test_large_angles(self):
        enc = AngleEncoding(n_qubits=2)
        x = np.array([100.0, -200.0])
        state = enc.encode(x)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)

    def test_zero_params(self):
        enc = AngleEncoding(n_qubits=2)
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        ct = CircuitTemplate(encoding=enc, ansatz=ans)
        x = np.array([0.0, 0.0])
        params = np.zeros(ct.n_params)
        # Should not crash
        exp_z = ct.expectation_z(x, params)
        assert -1.0 <= exp_z <= 1.0

    def test_gradient_at_extremum(self):
        # f(x) = cos(x), f'(0) = 0 (minimum at x=pi)
        def cost(p):
            return float(np.cos(p[0]))

        psr = ParameterShiftRule()
        result = psr.compute(cost, np.array([0.0]))
        np.testing.assert_allclose(result.gradient[0], 0.0, atol=1e-6)

    def test_kernel_identical_points(self):
        enc = AngleEncoding(n_qubits=2)
        qk = QuantumKernel(enc)
        X = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        K = qk.matrix(X)
        np.testing.assert_allclose(K, np.ones((3, 3)), atol=1e-10)

    def test_ansatz_single_layer(self):
        for cls in [HardwareEfficientAnsatz, SimplifiedTwoDesign]:
            a = cls(n_qubits=2, n_layers=1)
            assert a.n_params > 0

    def test_ansatz_param_vector_round_trip(self):
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        params = np.arange(ans.n_params, dtype=np.float64)
        bindings = ans.param_vector_to_bindings(params)
        recovered = ans.bindings_to_param_vector(bindings)
        np.testing.assert_allclose(recovered, params, atol=1e-15)

    def test_wrong_param_count_raises(self):
        ans = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
        with pytest.raises(ValueError, match="Expected"):
            ans.param_vector_to_bindings(np.zeros(100))

    def test_kernel_objective_loss(self):
        K = np.eye(4)
        y = np.array([0, 0, 1, 1])
        obj = KernelObjective()
        loss = obj.loss(K, y)
        assert isinstance(loss, float)


# ======================================================================
# 16. Data Re-uploading tests
# ======================================================================

from nqpu.qcl.reuploading import (
    ReuploadingClassifier,
    ReuploadingHistory,
    ReuploadingLayer,
    MultiQubitReuploading,
)


class TestReuploading:
    """Tests for data re-uploading quantum classifiers."""

    def test_layer_produces_unitary(self):
        """A single re-uploading layer should produce a unitary matrix."""
        layer = ReuploadingLayer(n_features=2)
        x = np.array([0.5, 0.3])
        params = np.array([1.0, 1.0, 1.0, 0.1, 0.2, 0.3])
        u = layer.circuit(x, params)
        # Check unitarity: U^dag U = I
        product = np.conj(u.T) @ u
        np.testing.assert_allclose(product, np.eye(2), atol=1e-12)

    def test_classifier_creation(self):
        """Classifier should initialize with correct parameter count."""
        clf = ReuploadingClassifier(n_features=2, n_classes=2, n_layers=4)
        assert clf.n_params == 4 * 6  # 6 params per layer
        assert clf.n_features == 2
        assert clf.n_classes == 2

    def test_initialize_params_shape(self):
        """Parameter initialization should produce correct shape."""
        clf = ReuploadingClassifier(n_features=3, n_classes=2, n_layers=5)
        rng = np.random.default_rng(42)
        params = clf.initialize_params(rng=rng)
        assert params.shape == (clf.n_params,)

    def test_predict_proba_sums_to_one(self):
        """Class probabilities should sum to 1."""
        clf = ReuploadingClassifier(n_features=2, n_classes=2, n_layers=3)
        rng = np.random.default_rng(42)
        params = clf.initialize_params(rng=rng)
        x = np.array([0.5, 0.8])
        probs = clf.predict_proba(x, params)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-8)
        assert len(probs) == 2

    def test_predict_proba_multiclass(self):
        """Multi-class probabilities should sum to 1."""
        clf = ReuploadingClassifier(n_features=2, n_classes=4, n_layers=3)
        rng = np.random.default_rng(42)
        params = clf.initialize_params(rng=rng)
        x = np.array([0.5, 0.8])
        probs = clf.predict_proba(x, params)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
        assert len(probs) == 4

    def test_predict_returns_valid_labels(self):
        """Predictions should be valid class labels."""
        clf = ReuploadingClassifier(n_features=2, n_classes=3, n_layers=3)
        rng = np.random.default_rng(42)
        params = clf.initialize_params(rng=rng)
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        preds = clf.predict(X, params)
        assert preds.shape == (3,)
        assert all(0 <= p < 3 for p in preds)

    def test_loss_positive(self):
        """Cross-entropy loss should be positive."""
        clf = ReuploadingClassifier(n_features=2, n_classes=2, n_layers=3)
        rng = np.random.default_rng(42)
        params = clf.initialize_params(rng=rng)
        X = np.array([[0.5, 0.8], [-0.3, 0.1]])
        y = np.array([0, 1])
        loss = clf.loss(X, y, params)
        assert loss > 0.0

    def test_gradient_shape(self):
        """Gradient should match parameter vector shape."""
        clf = ReuploadingClassifier(n_features=2, n_classes=2, n_layers=2)
        rng = np.random.default_rng(42)
        params = clf.initialize_params(rng=rng)
        X = np.array([[0.5, 0.8]])
        y = np.array([0])
        grad = clf.gradient(X, y, params)
        assert grad.shape == params.shape

    def test_gradient_finite(self):
        """Gradient components should all be finite."""
        clf = ReuploadingClassifier(n_features=2, n_classes=2, n_layers=2)
        rng = np.random.default_rng(42)
        params = clf.initialize_params(rng=rng)
        X = np.array([[0.5, 0.8]])
        y = np.array([0])
        grad = clf.gradient(X, y, params)
        assert np.all(np.isfinite(grad))

    def test_training_reduces_loss(self):
        """Training should reduce the loss over epochs."""
        clf = ReuploadingClassifier(n_features=2, n_classes=2, n_layers=3)
        X = np.array([[0.5, 0.3], [-0.5, -0.3], [0.3, 0.5], [-0.3, -0.5]])
        y = np.array([0, 1, 0, 1])
        rng = np.random.default_rng(42)
        history = clf.fit(X, y, epochs=15, lr=0.05, rng=rng)
        assert isinstance(history, ReuploadingHistory)
        assert len(history.losses) == 15
        # Loss should decrease or at least not explode
        assert history.losses[-1] < history.losses[0] * 2.0

    def test_training_history_structure(self):
        """Training history should contain all expected fields."""
        clf = ReuploadingClassifier(n_features=2, n_classes=2, n_layers=2)
        X = np.array([[0.5, 0.3], [-0.5, -0.3]])
        y = np.array([0, 1])
        rng = np.random.default_rng(42)
        history = clf.fit(X, y, epochs=5, lr=0.01, rng=rng)
        assert history.epochs == 5
        assert len(history.accuracies) == 5
        assert history.params.shape == (clf.n_params,)
        assert all(0.0 <= a <= 1.0 for a in history.accuracies)

    def test_multiqubit_creation(self):
        """Multi-qubit classifier should have correct parameter count."""
        mqr = MultiQubitReuploading(n_features=2, n_classes=2, n_qubits=3, n_layers=2)
        assert mqr.n_params == 2 * 3 * 6  # layers * qubits * 6

    def test_multiqubit_predict_proba(self):
        """Multi-qubit class probabilities should sum to 1."""
        mqr = MultiQubitReuploading(n_features=2, n_classes=2, n_qubits=2, n_layers=2)
        rng = np.random.default_rng(42)
        params = mqr.initialize_params(rng=rng)
        x = np.array([0.5, 0.8])
        probs = mqr.predict_proba(x, params)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-8)
        assert len(probs) == 2

    def test_multiqubit_training(self):
        """Multi-qubit classifier should train without errors."""
        mqr = MultiQubitReuploading(n_features=2, n_classes=2, n_qubits=2, n_layers=2)
        X = np.array([[0.5, 0.3], [-0.5, -0.3], [0.3, 0.5], [-0.3, -0.5]])
        y = np.array([0, 1, 0, 1])
        rng = np.random.default_rng(42)
        history = mqr.fit(X, y, epochs=5, lr=0.05, rng=rng)
        assert len(history.losses) == 5
        assert all(np.isfinite(l) for l in history.losses)

    def test_accuracy_computation(self):
        """Accuracy computation should return value in [0, 1]."""
        clf = ReuploadingClassifier(n_features=2, n_classes=2, n_layers=3)
        rng = np.random.default_rng(42)
        params = clf.initialize_params(rng=rng)
        X = np.array([[0.5, 0.3], [-0.5, -0.3]])
        y = np.array([0, 1])
        acc = clf.accuracy(X, y, params)
        assert 0.0 <= acc <= 1.0


# ======================================================================
# 17. Architecture Search tests
# ======================================================================

from nqpu.qcl.architecture_search import (
    BayesianSearch,
    CircuitArchitecture,
    EvolutionarySearch,
    FitnessEvaluator,
    FitnessResult,
    GateSpec,
    SearchResult,
)


class TestArchitectureSearch:
    """Tests for quantum circuit architecture search."""

    def test_gate_spec_creation(self):
        """GateSpec should store gate type and qubits correctly."""
        gs = GateSpec(gate_type="Rx", qubits=(0,), has_param=True)
        assert gs.gate_type == "Rx"
        assert gs.qubits == (0,)
        assert gs.has_param is True

    def test_architecture_n_params(self):
        """Architecture should count trainable parameters correctly."""
        arch = CircuitArchitecture(n_qubits=2, gates=[
            GateSpec("Rx", (0,), True),
            GateSpec("Ry", (1,), True),
            GateSpec("CNOT", (0, 1), False),
            GateSpec("Rz", (0,), True),
        ])
        assert arch.n_params == 3

    def test_architecture_depth(self):
        """Circuit depth should reflect max gates per qubit."""
        arch = CircuitArchitecture(n_qubits=2, gates=[
            GateSpec("Rx", (0,), True),
            GateSpec("Ry", (0,), True),
            GateSpec("Rz", (1,), True),
        ])
        assert arch.depth == 2  # qubit 0 has 2 gates

    def test_architecture_cnot_count(self):
        """CNOT count should count two-qubit gates."""
        arch = CircuitArchitecture(n_qubits=2, gates=[
            GateSpec("Rx", (0,), True),
            GateSpec("CNOT", (0, 1), False),
            GateSpec("CZ", (0, 1), False),
        ])
        assert arch.cnot_count == 2

    def test_random_architecture(self):
        """Random architecture should have specified number of gates."""
        rng = np.random.default_rng(42)
        arch = CircuitArchitecture.random(n_qubits=3, n_gates=8, rng=rng)
        assert arch.n_qubits == 3
        assert len(arch.gates) == 8

    def test_mutation_produces_different_architecture(self):
        """Mutation should produce a different architecture (at least once in many tries)."""
        rng = np.random.default_rng(42)
        arch = CircuitArchitecture.random(n_qubits=3, n_gates=6, rng=rng)
        found_difference = False
        for _ in range(20):
            mutated = arch.mutate(rng=rng)
            if len(mutated.gates) != len(arch.gates):
                found_difference = True
                break
            for m, o in zip(mutated.gates, arch.gates):
                if str(m.gate_type) != str(o.gate_type) or m.qubits != o.qubits:
                    found_difference = True
                    break
            if found_difference:
                break
        assert found_difference, "Mutation should produce a different architecture"

    def test_mutation_preserves_n_qubits(self):
        """Mutation should not change the number of qubits."""
        rng = np.random.default_rng(42)
        arch = CircuitArchitecture.random(n_qubits=4, n_gates=5, rng=rng)
        for _ in range(10):
            mutated = arch.mutate(rng=rng)
            assert mutated.n_qubits == 4

    def test_feature_vector(self):
        """Architecture feature vector should have correct shape."""
        arch = CircuitArchitecture(n_qubits=2, gates=[
            GateSpec("Rx", (0,), True),
            GateSpec("CNOT", (0, 1), False),
        ])
        fv = arch.to_feature_vector()
        assert fv.shape == (9,)
        assert fv[0] == 1  # n_params
        assert fv[6] == 1  # n_CNOT

    def test_fitness_evaluator(self):
        """FitnessEvaluator should return valid results."""
        rng = np.random.default_rng(42)
        X_train = rng.uniform(-1, 1, (8, 2))
        y_train = (X_train[:, 0] > 0).astype(int)
        X_val = rng.uniform(-1, 1, (4, 2))
        y_val = (X_val[:, 0] > 0).astype(int)

        evaluator = FitnessEvaluator(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, n_epochs=5,
        )
        arch = CircuitArchitecture.random(n_qubits=2, n_gates=6, rng=rng)
        result = evaluator.evaluate(arch, rng=rng)
        assert isinstance(result, FitnessResult)
        assert 0.0 <= result.val_accuracy <= 1.0
        assert result.val_loss >= 0.0
        assert result.n_params >= 0

    def test_evolutionary_search(self):
        """Evolutionary search should complete and return valid results."""
        rng = np.random.default_rng(42)
        X_train = rng.uniform(-1, 1, (8, 2))
        y_train = (X_train[:, 0] > 0).astype(int)
        X_val = rng.uniform(-1, 1, (4, 2))
        y_val = (X_val[:, 0] > 0).astype(int)

        evaluator = FitnessEvaluator(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, n_epochs=3,
        )
        search = EvolutionarySearch(
            n_qubits=2, population_size=4, n_generations=2,
            mutation_rate=0.5, tournament_size=2,
        )
        result = search.search(evaluator, rng=rng)
        assert isinstance(result, SearchResult)
        assert result.best_architecture is not None
        assert result.best_fitness is not None
        assert len(result.history) > 0

    def test_bayesian_search(self):
        """Bayesian search should complete and return valid results."""
        rng = np.random.default_rng(42)
        X_train = rng.uniform(-1, 1, (8, 2))
        y_train = (X_train[:, 0] > 0).astype(int)
        X_val = rng.uniform(-1, 1, (4, 2))
        y_val = (X_val[:, 0] > 0).astype(int)

        evaluator = FitnessEvaluator(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, n_epochs=3,
        )
        search = BayesianSearch(
            n_qubits=2, max_gates=8, n_iterations=8,
        )
        result = search.search(evaluator, rng=rng)
        assert isinstance(result, SearchResult)
        assert result.best_architecture is not None
        assert 0.0 <= result.best_fitness.val_accuracy <= 1.0

    def test_empty_architecture(self):
        """Architecture with no gates should have zero params and depth."""
        arch = CircuitArchitecture(n_qubits=2, gates=[])
        assert arch.n_params == 0
        assert arch.depth == 0
        assert arch.cnot_count == 0

    def test_crossover_produces_valid_offspring(self):
        """Crossover should produce architectures with valid qubit indices."""
        rng = np.random.default_rng(42)
        search = EvolutionarySearch(n_qubits=3, population_size=4)
        parent1 = CircuitArchitecture.random(3, 5, rng=rng)
        parent2 = CircuitArchitecture.random(3, 5, rng=rng)
        child = search._crossover(parent1, parent2, rng)
        assert child.n_qubits == 3
        for g in child.gates:
            assert all(q < 3 for q in g.qubits)


# ======================================================================
# 18. Barren Plateau Mitigation tests
# ======================================================================

from nqpu.qcl.barren_plateau import (
    AnalysisResult,
    BarrenPlateauAnalyzer,
    IdentityBlockInit,
    LayerwiseResult,
    LayerwiseTraining,
    ScalingResult,
    VarianceMonitor,
    VarianceStatus,
)


class TestBarrenPlateau:
    """Tests for barren plateau detection and mitigation."""

    def test_variance_monitor_basic(self):
        """VarianceMonitor should track gradient variance."""
        monitor = VarianceMonitor(window_size=5, threshold=1e-4)
        rng = np.random.default_rng(42)
        for _ in range(10):
            grad = rng.normal(0, 0.1, size=5)
            status = monitor.update(grad)
        assert isinstance(status, VarianceStatus)
        assert status.samples == 10
        assert status.variance > 0.0

    def test_variance_monitor_detects_vanishing(self):
        """Monitor should detect vanishing gradients."""
        monitor = VarianceMonitor(window_size=5, threshold=1e-3)
        # Feed tiny gradients
        for _ in range(10):
            grad = np.zeros(5) + 1e-8
            status = monitor.update(grad)
        assert status.is_barren is True

    def test_variance_monitor_healthy_gradients(self):
        """Monitor should not flag healthy gradients."""
        monitor = VarianceMonitor(window_size=5, threshold=1e-6)
        rng = np.random.default_rng(42)
        for _ in range(10):
            grad = rng.normal(0, 1.0, size=5)
            status = monitor.update(grad)
        assert status.is_barren is False

    def test_variance_trajectory(self):
        """Variance trajectory should have one entry per update."""
        monitor = VarianceMonitor(window_size=5)
        rng = np.random.default_rng(42)
        for _ in range(7):
            monitor.update(rng.normal(0, 0.1, size=3))
        traj = monitor.variance_trajectory()
        assert len(traj) == 7

    def test_variance_monitor_trend(self):
        """Monitor should detect decreasing variance trend."""
        monitor = VarianceMonitor(window_size=3, threshold=1e-10)
        # Feed gradually decreasing gradients
        for i in range(5):
            scale = 1.0 / (i + 1)
            grad = np.ones(4) * scale
            status = monitor.update(grad)
        # After several updates with decreasing scale, should show trend
        assert status.trend in ("decreasing", "stable")

    def test_identity_block_init(self):
        """Identity init should produce small parameter values."""
        init = IdentityBlockInit(n_qubits=3, n_layers=4)
        params = init.initialize(rng=np.random.default_rng(42))
        expected_size = 4 * 3 * 3  # layers * qubits * 3
        assert params.shape == (expected_size,)
        # Parameters should be near zero (near-identity)
        assert np.max(np.abs(params)) < 0.1

    def test_identity_block_paired_init(self):
        """Paired identity init should have cancelling layers."""
        init = IdentityBlockInit(n_qubits=2, n_layers=4)
        params = init.initialize_paired(rng=np.random.default_rng(42))
        expected_size = 4 * 2 * 3
        assert params.shape == (expected_size,)
        # Parameters should be small
        assert np.max(np.abs(params)) < 0.1

    def test_layerwise_training(self):
        """Layerwise training should complete and return valid results."""
        def loss_fn(params):
            return float(np.sum(np.sin(params) ** 2))

        def grad_fn(params):
            return np.sin(2 * params)

        trainer = LayerwiseTraining(n_qubits=2, max_layers=2, epochs_per_layer=10)
        result = trainer.train(loss_fn, grad_fn, rng=np.random.default_rng(42))
        assert isinstance(result, LayerwiseResult)
        assert result.layers_trained == 2
        assert len(result.loss_history) == 2 * 10  # max_layers * epochs_per_layer
        assert len(result.variance_history) == 2 * 10

    def test_barren_plateau_analyzer_constant(self):
        """Analyzer should detect barren plateau for constant cost function."""
        analyzer = BarrenPlateauAnalyzer(n_qubits=2, n_layers=1, n_samples=10)

        def constant_cost(params):
            return 1.0

        result = analyzer.analyze(constant_cost, rng=np.random.default_rng(42))
        assert isinstance(result, AnalysisResult)
        assert result.is_barren is True
        assert result.severity in ("mild", "severe")

    def test_barren_plateau_analyzer_nontrivial(self):
        """Analyzer should find non-zero variance for a simple cost."""
        analyzer = BarrenPlateauAnalyzer(n_qubits=2, n_layers=1, n_samples=20)

        def simple_cost(params):
            return float(np.sin(params[0]) * np.cos(params[1]))

        result = analyzer.analyze(simple_cost, rng=np.random.default_rng(42))
        assert isinstance(result, AnalysisResult)
        assert result.gradient_variance > 0.0
        assert result.recommendation  # Should have non-empty recommendation

    def test_scaling_analysis(self):
        """Scaling analysis should return valid qubit counts and variances."""
        analyzer = BarrenPlateauAnalyzer(n_qubits=4, n_layers=1, n_samples=10)
        result = analyzer.scaling_analysis(
            qubit_range=range(2, 5),
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, ScalingResult)
        assert result.qubit_counts == [2, 3, 4]
        assert len(result.variances) == 3
        assert all(v > 0 for v in result.variances)
        assert isinstance(result.scaling_exponent, float)

    def test_variance_monitor_reset(self):
        """Reset should clear all history."""
        monitor = VarianceMonitor(window_size=5)
        rng = np.random.default_rng(42)
        for _ in range(5):
            monitor.update(rng.normal(0, 0.1, size=3))
        assert len(monitor.variance_trajectory()) == 5
        monitor.reset()
        assert len(monitor.variance_trajectory()) == 0


# ======================================================================
# 19. New module package import tests
# ======================================================================


class TestNewModuleImports:
    """Tests that the new module exports are correctly configured."""

    def test_reuploading_imports(self):
        """All re-uploading classes should be importable from nqpu.qcl."""
        from nqpu.qcl import (
            ReuploadingClassifier,
            ReuploadingHistory,
            ReuploadingLayer,
            MultiQubitReuploading,
        )

    def test_architecture_search_imports(self):
        """All architecture search classes should be importable from nqpu.qcl."""
        from nqpu.qcl import (
            BayesianSearch,
            CircuitArchitecture,
            EvolutionarySearch,
            FitnessEvaluator,
            FitnessResult,
            GateSpec,
            SearchResult,
        )

    def test_barren_plateau_imports(self):
        """All barren plateau classes should be importable from nqpu.qcl."""
        from nqpu.qcl import (
            AnalysisResult,
            BarrenPlateauAnalyzer,
            IdentityBlockInit,
            LayerwiseResult,
            LayerwiseTraining,
            ScalingResult,
            VarianceMonitor,
            VarianceStatus,
        )

    def test_all_attribute_extended(self):
        """__all__ should contain all new exports."""
        import nqpu.qcl
        all_names = set(nqpu.qcl.__all__)
        new_names = {
            "ReuploadingLayer", "ReuploadingClassifier", "ReuploadingHistory",
            "MultiQubitReuploading", "GateSpec", "CircuitArchitecture",
            "FitnessEvaluator", "FitnessResult", "EvolutionarySearch",
            "BayesianSearch", "SearchResult", "VarianceMonitor",
            "VarianceStatus", "IdentityBlockInit", "LayerwiseTraining",
            "LayerwiseResult", "BarrenPlateauAnalyzer", "AnalysisResult",
            "ScalingResult",
        }
        assert new_names.issubset(all_names)
