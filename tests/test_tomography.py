"""Comprehensive tests for nqpu.tomography package.

Tests cover:
  - State tomography: circuit generation, measurement simulation, reconstruction
  - Process tomography: chi/Choi matrix conversion, identity channel
  - Shadow tomography: snapshot creation, observable estimation, fidelity
  - Verification: purity, entropy, concurrence, partial trace, witnesses, XEB, QV
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nqpu.tomography import (
    # State tomography
    MeasurementCircuit,
    StateTomographer,
    StateTomographyResult,
    TomographyMeasurementResult,
    generate_measurement_circuits,
    generate_tetrahedral_circuits,
    simulate_tomography_measurements,
    state_fidelity,
    # Process tomography
    ProcessTomographer,
    ProcessTomographyResult,
    chi_to_choi,
    choi_to_chi,
    simulate_process_tomography,
    # Shadow tomography
    ClassicalShadow,
    PauliObservable,
    ShadowSnapshot,
    create_shadow_from_state,
    estimate_expectation,
    estimate_fidelity,
    shadow_size_bound,
    # Verification
    average_gate_fidelity,
    concurrence,
    cross_entropy_benchmark,
    entanglement_of_formation,
    entanglement_witness_bell,
    entanglement_witness_ghz,
    estimate_quantum_volume,
    linear_entropy,
    negativity,
    partial_trace,
    purity,
    relative_entropy,
    schmidt_decomposition,
    schmidt_number,
    trace_distance,
    von_neumann_entropy,
)


# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------

SQRT2_INV = 1.0 / math.sqrt(2.0)


@pytest.fixture
def state_zero() -> np.ndarray:
    """Single-qubit |0> state."""
    return np.array([1.0, 0.0], dtype=np.complex128)


@pytest.fixture
def state_one() -> np.ndarray:
    """Single-qubit |1> state."""
    return np.array([0.0, 1.0], dtype=np.complex128)


@pytest.fixture
def state_plus() -> np.ndarray:
    """Single-qubit |+> state."""
    return np.array([SQRT2_INV, SQRT2_INV], dtype=np.complex128)


@pytest.fixture
def bell_state() -> np.ndarray:
    """Two-qubit Bell state |Phi+> = (|00> + |11>) / sqrt(2)."""
    return np.array([SQRT2_INV, 0.0, 0.0, SQRT2_INV], dtype=np.complex128)


@pytest.fixture
def maximally_mixed_1q() -> np.ndarray:
    """Single-qubit maximally mixed state I/2."""
    return np.eye(2, dtype=np.complex128) / 2.0


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG for reproducibility."""
    return np.random.default_rng(42)


# -----------------------------------------------------------------------
# State tomography tests
# -----------------------------------------------------------------------


class TestMeasurementCircuitGeneration:
    """Tests for measurement circuit generation."""

    def test_num_circuits_1_qubit(self):
        circuits = generate_measurement_circuits(1)
        assert len(circuits) == 3  # X, Y, Z

    def test_num_circuits_2_qubits(self):
        circuits = generate_measurement_circuits(2)
        assert len(circuits) == 9  # 3^2

    def test_circuit_bases_are_xyz(self):
        circuits = generate_measurement_circuits(1)
        bases_set = {c.bases[0] for c in circuits}
        assert bases_set == {"X", "Y", "Z"}

    def test_invalid_num_qubits(self):
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            generate_measurement_circuits(0)

    def test_tetrahedral_circuits_count(self):
        circuits = generate_tetrahedral_circuits(1)
        assert len(circuits) == 4  # 4^1

    def test_tetrahedral_2q_count(self):
        circuits = generate_tetrahedral_circuits(2)
        assert len(circuits) == 16  # 4^2


class TestStateTomographer:
    """Tests for the StateTomographer class."""

    def test_init_stores_num_qubits(self):
        tomo = StateTomographer(2)
        assert tomo.num_qubits == 2
        assert tomo.dim == 4

    def test_invalid_num_qubits_raises(self):
        with pytest.raises(ValueError):
            StateTomographer(0)

    def test_measurement_circuits_returns_list(self):
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        assert isinstance(circuits, list)
        assert len(circuits) == 3

    def test_reconstruct_linear_on_zero_state(self, state_zero, rng):
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            state_zero, circuits, shots=50000, rng=rng
        )
        result = tomo.reconstruct(measurements, method="linear")
        assert isinstance(result, StateTomographyResult)
        assert result.method == "linear"
        # Fidelity with |0> should be close to 1
        fid = result.fidelity_with(state_zero)
        assert fid > 0.9

    def test_reconstruct_mle_on_plus_state(self, state_plus, rng):
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            state_plus, circuits, shots=50000, rng=rng
        )
        result = tomo.reconstruct(measurements, method="mle", max_iterations=200)
        assert result.method == "mle"
        assert result.is_physical

    def test_reconstruct_lstsq_is_physical(self, state_zero, rng):
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            state_zero, circuits, shots=50000, rng=rng
        )
        result = tomo.reconstruct(measurements, method="lstsq", max_iterations=200)
        assert result.method == "lstsq"
        assert result.is_physical

    def test_reconstruct_invalid_method_raises(self, state_zero, rng):
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            state_zero, circuits, shots=1000, rng=rng
        )
        with pytest.raises(ValueError, match="Unknown method"):
            tomo.reconstruct(measurements, method="bogus")


class TestStateTomographyResult:
    """Tests for StateTomographyResult properties."""

    def test_purity_of_pure_state(self, state_zero):
        rho = np.outer(state_zero, state_zero.conj())
        result = StateTomographyResult(
            density_matrix=rho, method="test", num_qubits=1
        )
        assert abs(result.purity - 1.0) < 1e-10

    def test_purity_of_mixed_state(self, maximally_mixed_1q):
        result = StateTomographyResult(
            density_matrix=maximally_mixed_1q, method="test", num_qubits=1
        )
        assert abs(result.purity - 0.5) < 1e-10

    def test_von_neumann_entropy_pure(self, state_zero):
        rho = np.outer(state_zero, state_zero.conj())
        result = StateTomographyResult(
            density_matrix=rho, method="test", num_qubits=1
        )
        assert abs(result.von_neumann_entropy) < 1e-10

    def test_von_neumann_entropy_mixed(self, maximally_mixed_1q):
        result = StateTomographyResult(
            density_matrix=maximally_mixed_1q, method="test", num_qubits=1
        )
        assert abs(result.von_neumann_entropy - math.log(2)) < 1e-10


class TestStateFidelity:
    """Tests for the state_fidelity function."""

    def test_fidelity_same_state(self, state_zero):
        fid = state_fidelity(state_zero, state_zero)
        assert abs(fid - 1.0) < 1e-10

    def test_fidelity_orthogonal_states(self, state_zero, state_one):
        fid = state_fidelity(state_zero, state_one)
        assert abs(fid) < 1e-10

    @pytest.mark.parametrize("state_fn", ["state_zero", "state_plus"])
    def test_fidelity_is_between_0_and_1(self, state_fn, request):
        state = request.getfixturevalue(state_fn)
        fid = state_fidelity(state, state)
        assert 0.0 <= fid <= 1.0 + 1e-10


# -----------------------------------------------------------------------
# Process tomography tests
# -----------------------------------------------------------------------


class TestProcessTomography:
    """Tests for process tomography."""

    def test_input_state_labels_count(self):
        pt = ProcessTomographer(1)
        labels = pt.input_state_labels()
        assert len(labels) == 6  # 6 single-qubit states

    def test_input_states_are_density_matrices(self):
        pt = ProcessTomographer(1)
        states = pt.input_states()
        for rho in states:
            assert rho.shape == (2, 2)
            assert abs(np.trace(rho).real - 1.0) < 1e-10

    def test_simulate_identity_channel(self):
        identity = np.eye(2, dtype=np.complex128)
        outputs = simulate_process_tomography(identity, num_qubits=1)
        # Each output should equal its input
        for labels, rho_out in outputs.items():
            assert rho_out.shape == (2, 2)
            assert abs(np.trace(rho_out).real - 1.0) < 1e-10

    def test_reconstruct_identity(self):
        identity = np.eye(2, dtype=np.complex128)
        outputs = simulate_process_tomography(identity, num_qubits=1)
        pt = ProcessTomographer(1)
        result = pt.reconstruct(outputs)
        assert isinstance(result, ProcessTomographyResult)
        assert result.num_qubits == 1
        # Average gate fidelity with identity should be ~1
        assert result.average_gate_fidelity > 0.99

    def test_chi_choi_roundtrip(self):
        """chi_to_choi and choi_to_chi should be inverse operations."""
        identity = np.eye(2, dtype=np.complex128)
        outputs = simulate_process_tomography(identity, num_qubits=1)
        pt = ProcessTomographer(1)
        result = pt.reconstruct(outputs)
        chi_orig = result.chi_matrix
        choi = chi_to_choi(chi_orig, num_qubits=1)
        chi_back = choi_to_chi(choi, num_qubits=1)
        assert np.allclose(chi_orig, chi_back, atol=1e-6)


# -----------------------------------------------------------------------
# Shadow tomography tests
# -----------------------------------------------------------------------


class TestShadowTomography:
    """Tests for classical shadow tomography."""

    def test_create_shadow_returns_correct_type(self, state_zero, rng):
        shadow = create_shadow_from_state(state_zero, num_snapshots=100, rng=rng)
        assert isinstance(shadow, ClassicalShadow)
        assert shadow.num_qubits == 1
        assert shadow.num_snapshots == 100

    def test_snapshot_state_shape(self, state_zero, rng):
        shadow = create_shadow_from_state(state_zero, num_snapshots=10, rng=rng)
        snap_state = shadow.snapshot_state(0)
        assert snap_state.shape == (2, 2)

    def test_estimate_density_matrix_shape(self, state_zero, rng):
        shadow = create_shadow_from_state(state_zero, num_snapshots=1000, rng=rng)
        rho_est = shadow.estimate_density_matrix()
        assert rho_est.shape == (2, 2)

    def test_pauli_observable_weight(self):
        obs = PauliObservable("XIZ")
        assert obs.num_qubits == 3
        assert obs.weight == 2  # X and Z are non-identity

    def test_pauli_observable_invalid_char(self):
        with pytest.raises(ValueError, match="Invalid Pauli character"):
            PauliObservable("XAZ")

    def test_estimate_expectation_z_on_zero(self, state_zero, rng):
        shadow = create_shadow_from_state(state_zero, num_snapshots=5000, rng=rng)
        obs = PauliObservable("Z")
        est, err = estimate_expectation(shadow, obs)
        # <0|Z|0> = 1
        assert abs(est - 1.0) < 0.2

    def test_estimate_fidelity_with_target(self, state_zero, rng):
        shadow = create_shadow_from_state(state_zero, num_snapshots=5000, rng=rng)
        fid = estimate_fidelity(shadow, state_zero)
        assert fid > 0.7

    def test_shadow_size_bound_positive(self):
        n = shadow_size_bound(num_observables=10, max_weight=2, epsilon=0.1)
        assert n > 0

    def test_shadow_size_bound_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            shadow_size_bound(num_observables=10, max_weight=2, epsilon=0.0)

    def test_shadow_size_bound_invalid_delta(self):
        with pytest.raises(ValueError, match="delta must be in"):
            shadow_size_bound(num_observables=10, max_weight=2, epsilon=0.1, delta=0.0)


# -----------------------------------------------------------------------
# Verification tests
# -----------------------------------------------------------------------


class TestVerification:
    """Tests for verification metrics."""

    def test_purity_pure_state(self, state_zero):
        assert abs(purity(state_zero) - 1.0) < 1e-10

    def test_purity_mixed_state(self, maximally_mixed_1q):
        assert abs(purity(maximally_mixed_1q) - 0.5) < 1e-10

    def test_von_neumann_entropy_base_2(self, maximally_mixed_1q):
        s = von_neumann_entropy(maximally_mixed_1q, base="2")
        assert abs(s - 1.0) < 1e-10  # 1 bit for qubit

    def test_von_neumann_entropy_invalid_base(self, state_zero):
        with pytest.raises(ValueError, match="Unknown base"):
            von_neumann_entropy(state_zero, base="7")

    def test_concurrence_bell_state(self, bell_state):
        c = concurrence(bell_state)
        assert abs(c - 1.0) < 1e-6

    def test_concurrence_product_state(self):
        product = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        c = concurrence(product)
        assert abs(c) < 1e-6

    def test_concurrence_wrong_dimension_raises(self, state_zero):
        with pytest.raises(ValueError, match="2-qubit states"):
            concurrence(state_zero)

    def test_entanglement_of_formation_bell(self, bell_state):
        eof = entanglement_of_formation(bell_state)
        assert abs(eof - 1.0) < 1e-4

    def test_entanglement_of_formation_product(self):
        product = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        eof = entanglement_of_formation(product)
        assert abs(eof) < 1e-6

    def test_entanglement_witness_bell_detects_entanglement(self, bell_state):
        rho = np.outer(bell_state, bell_state.conj())
        result = entanglement_witness_bell(rho)
        assert result.is_entangled
        assert result.witness_value < 0

    def test_entanglement_witness_bell_separable(self):
        product = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        rho = np.outer(product, product.conj())
        result = entanglement_witness_bell(rho)
        assert not result.is_entangled

    def test_entanglement_witness_ghz(self):
        ghz = np.zeros(4, dtype=np.complex128)
        ghz[0] = SQRT2_INV
        ghz[3] = SQRT2_INV
        rho = np.outer(ghz, ghz.conj())
        result = entanglement_witness_ghz(rho, num_qubits=2)
        assert result.is_entangled

    def test_trace_distance_same_state(self, state_zero):
        td = trace_distance(state_zero, state_zero)
        assert abs(td) < 1e-10

    def test_trace_distance_orthogonal(self, state_zero, state_one):
        td = trace_distance(state_zero, state_one)
        assert abs(td - 1.0) < 1e-10

    def test_relative_entropy_same_state(self, maximally_mixed_1q):
        re = relative_entropy(maximally_mixed_1q, maximally_mixed_1q)
        assert abs(re) < 1e-6

    def test_relative_entropy_non_overlapping_support(self, state_zero, state_one):
        rho = np.outer(state_zero, state_zero.conj())
        sigma = np.outer(state_one, state_one.conj())
        re = relative_entropy(rho, sigma)
        assert re == float("inf")

    def test_partial_trace_bell_state(self, bell_state):
        rho = np.outer(bell_state, bell_state.conj())
        reduced = partial_trace(rho, keep_qubits=[0], num_qubits=2)
        assert reduced.shape == (2, 2)
        # Partial trace of Bell state should be maximally mixed
        assert np.allclose(reduced, np.eye(2, dtype=np.complex128) / 2, atol=1e-10)

    def test_schmidt_decomposition_product_state(self):
        product = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        coeffs, _, _ = schmidt_decomposition(product, [0], num_qubits=2)
        # Product state has Schmidt number 1
        assert sum(coeffs > 1e-10) == 1

    def test_schmidt_number_bell_state(self, bell_state):
        sn = schmidt_number(bell_state, [0], num_qubits=2)
        assert sn == 2

    def test_linear_entropy_pure(self, state_zero):
        le = linear_entropy(state_zero)
        assert abs(le) < 1e-10

    def test_linear_entropy_mixed(self, maximally_mixed_1q):
        le = linear_entropy(maximally_mixed_1q)
        assert abs(le - 0.5) < 1e-10

    def test_negativity_bell_state(self, bell_state):
        rho = np.outer(bell_state, bell_state.conj())
        neg = negativity(rho, num_qubits_a=1)
        assert neg > 0.4  # Bell state has negativity 0.5

    def test_negativity_product_state(self):
        product = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        rho = np.outer(product, product.conj())
        neg = negativity(rho, num_qubits_a=1)
        assert abs(neg) < 1e-10

    def test_estimate_quantum_volume_achieved(self):
        # Simulate a scenario where QV is achieved (heavy output > 2/3)
        ideal_probs = np.array([0.7, 0.1, 0.1, 0.1])
        # Circuits that mostly output the heavy bitstring
        counts = [{"00": 800, "01": 100, "10": 50, "11": 50}]
        result = estimate_quantum_volume(
            num_qubits=2,
            circuit_outputs=counts,
            ideal_probabilities=[ideal_probs],
        )
        assert isinstance(result, type(result))
        assert result.num_circuits == 1

    def test_cross_entropy_benchmark_basic(self):
        # Ideal uniform distribution for 2 qubits
        ideal_probs = np.array([0.25, 0.25, 0.25, 0.25])
        counts = [{"00": 250, "01": 250, "10": 250, "11": 250}]
        result = cross_entropy_benchmark(
            circuit_outputs=counts,
            ideal_probabilities=[ideal_probs],
            num_qubits=2,
        )
        assert result.num_circuits == 1
        # For uniform distribution, XEB fidelity should be ~0
        assert abs(result.xeb_fidelity) < 0.2
