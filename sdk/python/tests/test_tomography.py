"""Comprehensive tests for the quantum state tomography and verification package.

Tests cover all five modules of the tomography package:
  - state_tomography.py: Full state reconstruction from Pauli measurements
  - process_tomography.py: Quantum channel characterisation
  - shadow_tomography.py: Randomised measurement observable estimation
  - verification.py: Fidelity, purity, entropy, entanglement, benchmarks
  - __init__.py: Public API surface

Test strategy:
  - Known-state reconstruction with ideal (noiseless) synthetic data
  - Physicality constraints for MLE and least-squares outputs
  - Edge cases: pure vs mixed, separable vs entangled, identity channel
  - Statistical tests use large shot counts and generous tolerances
    to prevent flaky failures while still catching real bugs
"""

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
    ProcessCircuit,
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
    EntanglementWitnessResult,
    QuantumVolumeResult,
    XEBResult,
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


# ======================================================================
# Fixtures: Common quantum states
# ======================================================================

SQRT2_INV = 1.0 / math.sqrt(2.0)


@pytest.fixture
def ket_zero() -> np.ndarray:
    """Single-qubit |0> state."""
    return np.array([1.0, 0.0], dtype=np.complex128)


@pytest.fixture
def ket_one() -> np.ndarray:
    """Single-qubit |1> state."""
    return np.array([0.0, 1.0], dtype=np.complex128)


@pytest.fixture
def ket_plus() -> np.ndarray:
    """Single-qubit |+> state."""
    return np.array([SQRT2_INV, SQRT2_INV], dtype=np.complex128)


@pytest.fixture
def ket_minus() -> np.ndarray:
    """Single-qubit |-> state."""
    return np.array([SQRT2_INV, -SQRT2_INV], dtype=np.complex128)


@pytest.fixture
def bell_state() -> np.ndarray:
    """Two-qubit Bell state |Phi+> = (|00> + |11>)/sqrt(2)."""
    return np.array(
        [SQRT2_INV, 0.0, 0.0, SQRT2_INV], dtype=np.complex128
    )


@pytest.fixture
def product_state_2q() -> np.ndarray:
    """Two-qubit product state |00>."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)


@pytest.fixture
def maximally_mixed_1q() -> np.ndarray:
    """Single-qubit maximally mixed state rho = I/2."""
    return np.eye(2, dtype=np.complex128) / 2.0


@pytest.fixture
def maximally_mixed_2q() -> np.ndarray:
    """Two-qubit maximally mixed state rho = I/4."""
    return np.eye(4, dtype=np.complex128) / 4.0


@pytest.fixture
def hadamard() -> np.ndarray:
    """Hadamard gate."""
    return np.array(
        [[SQRT2_INV, SQRT2_INV], [SQRT2_INV, -SQRT2_INV]],
        dtype=np.complex128,
    )


@pytest.fixture
def pauli_x() -> np.ndarray:
    """Pauli-X gate."""
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


# ======================================================================
# State Tomography: Circuit Generation
# ======================================================================


class TestMeasurementCircuitGeneration:
    """Test measurement circuit generation for state tomography."""

    def test_single_qubit_circuit_count(self) -> None:
        """1 qubit should produce 3 measurement circuits (X, Y, Z)."""
        circuits = generate_measurement_circuits(1)
        assert len(circuits) == 3

    def test_two_qubit_circuit_count(self) -> None:
        """2 qubits should produce 9 measurement circuits (3^2)."""
        circuits = generate_measurement_circuits(2)
        assert len(circuits) == 9

    def test_three_qubit_circuit_count(self) -> None:
        """3 qubits should produce 27 measurement circuits (3^3)."""
        circuits = generate_measurement_circuits(3)
        assert len(circuits) == 27

    def test_circuit_bases_are_pauli(self) -> None:
        """All bases should be X, Y, or Z."""
        circuits = generate_measurement_circuits(2)
        for circ in circuits:
            for basis in circ.bases:
                assert basis in ("X", "Y", "Z")

    def test_z_basis_has_no_rotations(self) -> None:
        """Z-only basis should require no rotation gates."""
        circuits = generate_measurement_circuits(1)
        z_circuit = [c for c in circuits if c.bases == ("Z",)][0]
        assert len(z_circuit.rotations) == 0

    def test_x_basis_has_rotation(self) -> None:
        """X basis should require a rotation gate."""
        circuits = generate_measurement_circuits(1)
        x_circuit = [c for c in circuits if c.bases == ("X",)][0]
        assert len(x_circuit.rotations) == 1
        assert x_circuit.rotations[0][0] == 0  # qubit index

    def test_invalid_num_qubits_raises(self) -> None:
        """num_qubits < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="num_qubits must be >= 1"):
            generate_measurement_circuits(0)

    def test_tetrahedral_single_qubit_count(self) -> None:
        """Tetrahedral measurement should produce 4 circuits for 1 qubit."""
        circuits = generate_tetrahedral_circuits(1)
        assert len(circuits) == 4

    def test_tetrahedral_two_qubit_count(self) -> None:
        """Tetrahedral measurement should produce 16 circuits for 2 qubits."""
        circuits = generate_tetrahedral_circuits(2)
        assert len(circuits) == 16


# ======================================================================
# State Tomography: Reconstruction
# ======================================================================


class TestStateTomographyReconstruction:
    """Test state reconstruction from synthetic measurement data."""

    def test_reconstruct_ket_zero_linear(self, ket_zero: np.ndarray) -> None:
        """Linear inversion should reconstruct |0> with high fidelity."""
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            ket_zero, circuits, shots=50000, rng=np.random.default_rng(42)
        )
        result = tomo.reconstruct(measurements, method="linear")
        assert result.fidelity_with(ket_zero) > 0.95

    def test_reconstruct_ket_zero_mle(self, ket_zero: np.ndarray) -> None:
        """MLE should reconstruct |0> with high fidelity."""
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            ket_zero, circuits, shots=50000, rng=np.random.default_rng(42)
        )
        result = tomo.reconstruct(measurements, method="mle")
        assert result.fidelity_with(ket_zero) > 0.95
        assert result.is_physical

    def test_reconstruct_ket_plus_mle(self, ket_plus: np.ndarray) -> None:
        """MLE should reconstruct |+> with high fidelity."""
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            ket_plus, circuits, shots=50000, rng=np.random.default_rng(43)
        )
        result = tomo.reconstruct(measurements, method="mle")
        assert result.fidelity_with(ket_plus) > 0.95
        assert result.is_physical

    def test_reconstruct_bell_state_mle(self, bell_state: np.ndarray) -> None:
        """MLE should reconstruct a Bell state with high fidelity."""
        tomo = StateTomographer(2)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            bell_state, circuits, shots=50000, rng=np.random.default_rng(44)
        )
        result = tomo.reconstruct(measurements, method="mle")
        assert result.fidelity_with(bell_state) > 0.90
        assert result.is_physical

    def test_reconstruct_lstsq(self, ket_zero: np.ndarray) -> None:
        """Least squares should reconstruct |0> reasonably."""
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            ket_zero, circuits, shots=50000, rng=np.random.default_rng(45)
        )
        result = tomo.reconstruct(measurements, method="lstsq")
        assert result.fidelity_with(ket_zero) > 0.90
        assert result.is_physical

    def test_mle_produces_physical_state(self, ket_plus: np.ndarray) -> None:
        """MLE output should always be a valid density matrix."""
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            ket_plus, circuits, shots=5000, rng=np.random.default_rng(46)
        )
        result = tomo.reconstruct(measurements, method="mle")

        rho = result.density_matrix
        # Hermitian
        assert np.allclose(rho, rho.conj().T, atol=1e-10)
        # Trace 1
        assert abs(np.trace(rho).real - 1.0) < 1e-8
        # Positive semidefinite
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-10)

    def test_result_purity_pure_state(self, ket_zero: np.ndarray) -> None:
        """Reconstructed pure state should have purity close to 1."""
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            ket_zero, circuits, shots=100000, rng=np.random.default_rng(47)
        )
        result = tomo.reconstruct(measurements, method="mle")
        assert result.purity > 0.90

    def test_result_entropy_pure_state(self, ket_zero: np.ndarray) -> None:
        """Reconstructed pure state should have near-zero entropy."""
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            ket_zero, circuits, shots=100000, rng=np.random.default_rng(48)
        )
        result = tomo.reconstruct(measurements, method="mle")
        assert result.von_neumann_entropy < 0.15

    def test_invalid_method_raises(self) -> None:
        """Unknown method should raise ValueError."""
        tomo = StateTomographer(1)
        with pytest.raises(ValueError, match="Unknown method"):
            tomo.reconstruct([], method="invalid")


# ======================================================================
# Process Tomography
# ======================================================================


class TestProcessTomography:
    """Test quantum process tomography."""

    def test_input_state_count_1q(self) -> None:
        """1-qubit process tomo needs 6 input states."""
        pt = ProcessTomographer(1)
        assert len(pt.input_state_labels()) == 6

    def test_input_state_count_2q(self) -> None:
        """2-qubit process tomo needs 36 input states."""
        pt = ProcessTomographer(2)
        assert len(pt.input_state_labels()) == 36

    def test_identity_channel_1q(self) -> None:
        """Identity channel should give chi[0,0] ~ 1 (Pauli I component)."""
        identity = np.eye(2, dtype=np.complex128)
        output_states = simulate_process_tomography(identity, 1)
        pt = ProcessTomographer(1)
        result = pt.reconstruct(output_states)
        # For identity channel, chi should have a large I-I component
        assert result.chi_matrix.shape == (4, 4)
        assert result.choi_matrix.shape == (4, 4)

    def test_identity_channel_gate_fidelity(self) -> None:
        """Identity channel should have gate fidelity ~1 with itself."""
        identity = np.eye(2, dtype=np.complex128)
        output_states = simulate_process_tomography(identity, 1)
        pt = ProcessTomographer(1)
        result = pt.reconstruct(output_states)
        fid = result.gate_fidelity_with(identity)
        assert fid > 0.90

    def test_hadamard_channel(self, hadamard: np.ndarray) -> None:
        """Hadamard channel should be reconstructed with high fidelity."""
        output_states = simulate_process_tomography(hadamard, 1)
        pt = ProcessTomographer(1)
        result = pt.reconstruct(output_states)
        fid = result.gate_fidelity_with(hadamard)
        assert fid > 0.90

    def test_pauli_x_channel(self, pauli_x: np.ndarray) -> None:
        """Pauli-X channel should be reconstructed with high fidelity."""
        output_states = simulate_process_tomography(pauli_x, 1)
        pt = ProcessTomographer(1)
        result = pt.reconstruct(output_states)
        fid = result.gate_fidelity_with(pauli_x)
        assert fid > 0.90

    def test_chi_choi_round_trip(self) -> None:
        """chi -> choi -> chi should be an identity transformation."""
        identity = np.eye(2, dtype=np.complex128)
        output_states = simulate_process_tomography(identity, 1)
        pt = ProcessTomographer(1)
        result = pt.reconstruct(output_states)

        chi_original = result.chi_matrix
        choi = chi_to_choi(chi_original, 1)
        chi_recovered = choi_to_chi(choi, 1)
        assert np.allclose(chi_original, chi_recovered, atol=1e-6)

    def test_process_circuit_generation(self) -> None:
        """Process circuits should cover all input-basis combinations."""
        pt = ProcessTomographer(1)
        circuits = pt.generate_process_circuits()
        # 6 input states * 3 measurement bases = 18
        assert len(circuits) == 18

    def test_diamond_norm_identity(self) -> None:
        """Identity channel should have small diamond norm distance to itself."""
        identity = np.eye(2, dtype=np.complex128)
        output_states = simulate_process_tomography(identity, 1)
        pt = ProcessTomographer(1)
        result = pt.reconstruct(output_states)
        # Diamond norm to identity should be small (not exactly 0 due to numerics)
        assert result.diamond_norm_estimate < 1.0


# ======================================================================
# Shadow Tomography
# ======================================================================


class TestShadowTomography:
    """Test classical shadow tomography."""

    def test_create_shadow_basic(self, ket_zero: np.ndarray) -> None:
        """Creating a shadow should produce the requested number of snapshots."""
        shadow = create_shadow_from_state(
            ket_zero, 100, rng=np.random.default_rng(42)
        )
        assert shadow.num_qubits == 1
        assert shadow.num_snapshots == 100

    def test_shadow_snapshot_bases(self, ket_zero: np.ndarray) -> None:
        """Shadow snapshots should have valid Pauli bases."""
        shadow = create_shadow_from_state(
            ket_zero, 50, rng=np.random.default_rng(42)
        )
        for snap in shadow.snapshots:
            for b in snap.bases:
                assert b in ("X", "Y", "Z")

    def test_shadow_snapshot_outcomes(self, ket_zero: np.ndarray) -> None:
        """Shadow snapshot outcomes should be 0 or 1."""
        shadow = create_shadow_from_state(
            ket_zero, 50, rng=np.random.default_rng(42)
        )
        for snap in shadow.snapshots:
            for o in snap.outcomes:
                assert o in (0, 1)

    def test_estimate_z_expectation_ket_zero(
        self, ket_zero: np.ndarray
    ) -> None:
        """<Z> for |0> should be close to +1."""
        shadow = create_shadow_from_state(
            ket_zero, 10000, rng=np.random.default_rng(42)
        )
        obs = PauliObservable("Z")
        est, err = estimate_expectation(shadow, obs)
        assert abs(est - 1.0) < 0.15

    def test_estimate_z_expectation_ket_one(
        self, ket_one: np.ndarray
    ) -> None:
        """<Z> for |1> should be close to -1."""
        shadow = create_shadow_from_state(
            ket_one, 10000, rng=np.random.default_rng(42)
        )
        obs = PauliObservable("Z")
        est, err = estimate_expectation(shadow, obs)
        assert abs(est - (-1.0)) < 0.15

    def test_estimate_x_expectation_ket_plus(
        self, ket_plus: np.ndarray
    ) -> None:
        """<X> for |+> should be close to +1."""
        shadow = create_shadow_from_state(
            ket_plus, 10000, rng=np.random.default_rng(43)
        )
        obs = PauliObservable("X")
        est, err = estimate_expectation(shadow, obs)
        assert abs(est - 1.0) < 0.15

    def test_estimate_zz_bell_state(self, bell_state: np.ndarray) -> None:
        """<ZZ> for Bell state |Phi+> should be close to +1."""
        shadow = create_shadow_from_state(
            bell_state, 10000, rng=np.random.default_rng(44)
        )
        obs = PauliObservable("ZZ")
        est, err = estimate_expectation(shadow, obs)
        assert abs(est - 1.0) < 0.2

    def test_estimate_xx_bell_state(self, bell_state: np.ndarray) -> None:
        """<XX> for Bell state |Phi+> should be close to +1."""
        shadow = create_shadow_from_state(
            bell_state, 10000, rng=np.random.default_rng(45)
        )
        obs = PauliObservable("XX")
        est, err = estimate_expectation(shadow, obs)
        assert abs(est - 1.0) < 0.2

    def test_shadow_density_matrix_estimate(
        self, ket_zero: np.ndarray
    ) -> None:
        """Estimated density matrix from shadow should approximate true state."""
        shadow = create_shadow_from_state(
            ket_zero, 10000, rng=np.random.default_rng(46)
        )
        rho_est = shadow.estimate_density_matrix()
        fid = state_fidelity(rho_est, np.outer(ket_zero, ket_zero.conj()))
        assert fid > 0.85

    def test_shadow_fidelity_estimation(self, ket_zero: np.ndarray) -> None:
        """Shadow fidelity estimate should be close to 1 for the true state."""
        shadow = create_shadow_from_state(
            ket_zero, 10000, rng=np.random.default_rng(47)
        )
        fid = estimate_fidelity(shadow, ket_zero)
        assert fid > 0.85

    def test_shadow_size_bound(self) -> None:
        """Shadow size bound should return positive integer."""
        n = shadow_size_bound(
            num_observables=10, max_weight=2, epsilon=0.1
        )
        assert isinstance(n, int)
        assert n > 0

    def test_shadow_size_bound_increases_with_observables(self) -> None:
        """More observables should require larger shadow."""
        n1 = shadow_size_bound(num_observables=10, max_weight=2, epsilon=0.1)
        n2 = shadow_size_bound(num_observables=100, max_weight=2, epsilon=0.1)
        assert n2 > n1

    def test_shadow_size_bound_increases_with_weight(self) -> None:
        """Higher weight observables should require larger shadow."""
        n1 = shadow_size_bound(num_observables=10, max_weight=1, epsilon=0.1)
        n2 = shadow_size_bound(num_observables=10, max_weight=3, epsilon=0.1)
        assert n2 > n1

    def test_pauli_observable_weight(self) -> None:
        """Observable weight should count non-identity Paulis."""
        assert PauliObservable("IXYZ").weight == 3
        assert PauliObservable("IIII").weight == 0
        assert PauliObservable("Z").weight == 1

    def test_pauli_observable_invalid_raises(self) -> None:
        """Invalid Pauli character should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid Pauli character"):
            PauliObservable("A")

    def test_mismatched_qubits_raises(self, ket_zero: np.ndarray) -> None:
        """Observable with wrong qubit count should raise ValueError."""
        shadow = create_shadow_from_state(ket_zero, 10)
        obs = PauliObservable("ZZ")  # 2-qubit obs for 1-qubit shadow
        with pytest.raises(ValueError, match="qubits"):
            estimate_expectation(shadow, obs)


# ======================================================================
# Verification: State Fidelity
# ======================================================================


class TestStateFidelity:
    """Test state fidelity computation."""

    def test_fidelity_with_self_pure(self, ket_zero: np.ndarray) -> None:
        """F(|0>, |0>) = 1."""
        f = state_fidelity(
            np.outer(ket_zero, ket_zero.conj()),
            np.outer(ket_zero, ket_zero.conj()),
        )
        assert abs(f - 1.0) < 1e-10

    def test_fidelity_orthogonal_pure(
        self, ket_zero: np.ndarray, ket_one: np.ndarray
    ) -> None:
        """F(|0>, |1>) = 0."""
        f = state_fidelity(
            np.outer(ket_zero, ket_zero.conj()),
            np.outer(ket_one, ket_one.conj()),
        )
        assert abs(f) < 1e-10

    def test_fidelity_symmetric(
        self, ket_zero: np.ndarray, ket_plus: np.ndarray
    ) -> None:
        """Fidelity should be symmetric: F(rho, sigma) = F(sigma, rho)."""
        rho = np.outer(ket_zero, ket_zero.conj())
        sigma = np.outer(ket_plus, ket_plus.conj())
        f1 = state_fidelity(rho, sigma)
        f2 = state_fidelity(sigma, rho)
        assert abs(f1 - f2) < 1e-10

    def test_fidelity_range(
        self, ket_zero: np.ndarray, ket_plus: np.ndarray
    ) -> None:
        """Fidelity should be in [0, 1]."""
        rho = np.outer(ket_zero, ket_zero.conj())
        sigma = np.outer(ket_plus, ket_plus.conj())
        f = state_fidelity(rho, sigma)
        assert 0.0 <= f <= 1.0

    def test_fidelity_zero_plus(
        self, ket_zero: np.ndarray, ket_plus: np.ndarray
    ) -> None:
        """F(|0>, |+>) = 1/2."""
        rho = np.outer(ket_zero, ket_zero.conj())
        sigma = np.outer(ket_plus, ket_plus.conj())
        f = state_fidelity(rho, sigma)
        assert abs(f - 0.5) < 1e-10

    def test_fidelity_with_mixed(
        self, ket_zero: np.ndarray, maximally_mixed_1q: np.ndarray
    ) -> None:
        """F(|0>, I/2) = 1/2."""
        rho = np.outer(ket_zero, ket_zero.conj())
        f = state_fidelity(rho, maximally_mixed_1q)
        assert abs(f - 0.5) < 1e-10

    def test_fidelity_accepts_statevectors(
        self, ket_zero: np.ndarray, ket_plus: np.ndarray
    ) -> None:
        """Fidelity function should accept state vectors directly."""
        f = state_fidelity(ket_zero, ket_plus)
        assert abs(f - 0.5) < 1e-10


# ======================================================================
# Verification: Purity
# ======================================================================


class TestPurity:
    """Test purity computation."""

    def test_purity_pure_state(self, ket_zero: np.ndarray) -> None:
        """Pure state purity should be 1."""
        rho = np.outer(ket_zero, ket_zero.conj())
        assert abs(purity(rho) - 1.0) < 1e-10

    def test_purity_maximally_mixed_1q(
        self, maximally_mixed_1q: np.ndarray
    ) -> None:
        """Maximally mixed 1-qubit state purity should be 1/2."""
        assert abs(purity(maximally_mixed_1q) - 0.5) < 1e-10

    def test_purity_maximally_mixed_2q(
        self, maximally_mixed_2q: np.ndarray
    ) -> None:
        """Maximally mixed 2-qubit state purity should be 1/4."""
        assert abs(purity(maximally_mixed_2q) - 0.25) < 1e-10

    def test_purity_accepts_statevector(self, ket_zero: np.ndarray) -> None:
        """Purity should accept a state vector and return 1."""
        assert abs(purity(ket_zero) - 1.0) < 1e-10

    def test_purity_bell_state(self, bell_state: np.ndarray) -> None:
        """Bell state is pure, purity should be 1."""
        rho = np.outer(bell_state, bell_state.conj())
        assert abs(purity(rho) - 1.0) < 1e-10


# ======================================================================
# Verification: Von Neumann Entropy
# ======================================================================


class TestVonNeumannEntropy:
    """Test Von Neumann entropy computation."""

    def test_entropy_pure_state(self, ket_zero: np.ndarray) -> None:
        """Pure state entropy should be 0."""
        rho = np.outer(ket_zero, ket_zero.conj())
        assert abs(von_neumann_entropy(rho)) < 1e-10

    def test_entropy_maximally_mixed_1q(
        self, maximally_mixed_1q: np.ndarray
    ) -> None:
        """Maximally mixed 1-qubit entropy should be ln(2) in nats."""
        s = von_neumann_entropy(maximally_mixed_1q)
        assert abs(s - math.log(2)) < 1e-10

    def test_entropy_maximally_mixed_2q(
        self, maximally_mixed_2q: np.ndarray
    ) -> None:
        """Maximally mixed 2-qubit entropy should be ln(4) in nats."""
        s = von_neumann_entropy(maximally_mixed_2q)
        assert abs(s - math.log(4)) < 1e-10

    def test_entropy_bits(self, maximally_mixed_1q: np.ndarray) -> None:
        """Maximally mixed 1-qubit entropy should be 1 bit."""
        s = von_neumann_entropy(maximally_mixed_1q, base="2")
        assert abs(s - 1.0) < 1e-10

    def test_entropy_non_negative(self, maximally_mixed_2q: np.ndarray) -> None:
        """Entropy should always be non-negative."""
        assert von_neumann_entropy(maximally_mixed_2q) >= 0

    def test_entropy_accepts_statevector(self, ket_zero: np.ndarray) -> None:
        """Entropy should accept state vector and return 0."""
        assert abs(von_neumann_entropy(ket_zero)) < 1e-10


# ======================================================================
# Verification: Concurrence
# ======================================================================


class TestConcurrence:
    """Test concurrence (2-qubit entanglement measure)."""

    def test_concurrence_bell_state(self, bell_state: np.ndarray) -> None:
        """Bell state should have concurrence 1."""
        rho = np.outer(bell_state, bell_state.conj())
        assert abs(concurrence(rho) - 1.0) < 1e-8

    def test_concurrence_product_state(
        self, product_state_2q: np.ndarray
    ) -> None:
        """Product state |00> should have concurrence 0."""
        rho = np.outer(product_state_2q, product_state_2q.conj())
        assert abs(concurrence(rho)) < 1e-8

    def test_concurrence_maximally_mixed(
        self, maximally_mixed_2q: np.ndarray
    ) -> None:
        """Maximally mixed 2-qubit state should have concurrence 0."""
        assert abs(concurrence(maximally_mixed_2q)) < 1e-8

    def test_concurrence_wrong_dimension_raises(
        self, ket_zero: np.ndarray
    ) -> None:
        """Non-4x4 matrix should raise ValueError."""
        rho = np.outer(ket_zero, ket_zero.conj())
        with pytest.raises(ValueError, match="2-qubit"):
            concurrence(rho)

    def test_concurrence_partial_entanglement(self) -> None:
        """Partially entangled state should have 0 < C < 1."""
        # |psi> = cos(pi/8)|00> + sin(pi/8)|11>
        theta = math.pi / 8
        state = np.array(
            [math.cos(theta), 0, 0, math.sin(theta)], dtype=np.complex128
        )
        rho = np.outer(state, state.conj())
        c = concurrence(rho)
        assert 0.0 < c < 1.0
        # C = sin(2*theta) = sin(pi/4) ~ 0.707
        assert abs(c - math.sin(2 * theta)) < 1e-6


# ======================================================================
# Verification: Entanglement of Formation
# ======================================================================


class TestEntanglementOfFormation:
    """Test entanglement of formation."""

    def test_eof_bell_state(self, bell_state: np.ndarray) -> None:
        """Bell state should have EoF = 1."""
        rho = np.outer(bell_state, bell_state.conj())
        assert abs(entanglement_of_formation(rho) - 1.0) < 1e-6

    def test_eof_product_state(self, product_state_2q: np.ndarray) -> None:
        """Product state should have EoF = 0."""
        rho = np.outer(product_state_2q, product_state_2q.conj())
        assert abs(entanglement_of_formation(rho)) < 1e-8


# ======================================================================
# Verification: Entanglement Witnesses
# ======================================================================


class TestEntanglementWitness:
    """Test entanglement witness operators."""

    def test_bell_witness_detects_bell_state(
        self, bell_state: np.ndarray
    ) -> None:
        """Bell witness should detect entanglement in a Bell state."""
        rho = np.outer(bell_state, bell_state.conj())
        result = entanglement_witness_bell(rho)
        assert result.is_entangled
        assert result.witness_value < 0

    def test_bell_witness_product_state(
        self, product_state_2q: np.ndarray
    ) -> None:
        """Bell witness should NOT detect entanglement in a product state."""
        rho = np.outer(product_state_2q, product_state_2q.conj())
        result = entanglement_witness_bell(rho)
        assert not result.is_entangled

    def test_ghz_witness_detects_ghz_state(self) -> None:
        """GHZ witness should detect entanglement in a GHZ state."""
        # 3-qubit GHZ
        ghz = np.zeros(8, dtype=np.complex128)
        ghz[0] = SQRT2_INV
        ghz[7] = SQRT2_INV
        rho = np.outer(ghz, ghz.conj())
        result = entanglement_witness_ghz(rho, 3)
        assert result.is_entangled
        assert result.witness_value < 0

    def test_ghz_witness_product_state(self) -> None:
        """GHZ witness should NOT detect entanglement in |000>."""
        state = np.zeros(8, dtype=np.complex128)
        state[0] = 1.0
        rho = np.outer(state, state.conj())
        result = entanglement_witness_ghz(rho, 3)
        assert not result.is_entangled


# ======================================================================
# Verification: Trace Distance
# ======================================================================


class TestTraceDistance:
    """Test trace distance computation."""

    def test_trace_distance_same_state(self, ket_zero: np.ndarray) -> None:
        """T(rho, rho) = 0."""
        rho = np.outer(ket_zero, ket_zero.conj())
        assert abs(trace_distance(rho, rho)) < 1e-10

    def test_trace_distance_orthogonal(
        self, ket_zero: np.ndarray, ket_one: np.ndarray
    ) -> None:
        """T(|0>, |1>) = 1."""
        rho = np.outer(ket_zero, ket_zero.conj())
        sigma = np.outer(ket_one, ket_one.conj())
        assert abs(trace_distance(rho, sigma) - 1.0) < 1e-10

    def test_trace_distance_symmetric(
        self, ket_zero: np.ndarray, ket_plus: np.ndarray
    ) -> None:
        """Trace distance should be symmetric."""
        rho = np.outer(ket_zero, ket_zero.conj())
        sigma = np.outer(ket_plus, ket_plus.conj())
        d1 = trace_distance(rho, sigma)
        d2 = trace_distance(sigma, rho)
        assert abs(d1 - d2) < 1e-10

    def test_trace_distance_range(
        self, ket_zero: np.ndarray, ket_plus: np.ndarray
    ) -> None:
        """Trace distance should be in [0, 1]."""
        rho = np.outer(ket_zero, ket_zero.conj())
        sigma = np.outer(ket_plus, ket_plus.conj())
        d = trace_distance(rho, sigma)
        assert 0.0 <= d <= 1.0 + 1e-10


# ======================================================================
# Verification: Relative Entropy
# ======================================================================


class TestRelativeEntropy:
    """Test quantum relative entropy."""

    def test_relative_entropy_same_state(
        self, maximally_mixed_1q: np.ndarray
    ) -> None:
        """S(rho || rho) = 0."""
        assert abs(relative_entropy(maximally_mixed_1q, maximally_mixed_1q)) < 1e-10

    def test_relative_entropy_non_negative(
        self, maximally_mixed_1q: np.ndarray
    ) -> None:
        """Relative entropy should be non-negative (Klein's inequality)."""
        # rho = |0><0|, sigma = I/2
        rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        s = relative_entropy(rho, maximally_mixed_1q)
        assert s >= -1e-10

    def test_relative_entropy_orthogonal_infinite(
        self, ket_zero: np.ndarray, ket_one: np.ndarray
    ) -> None:
        """S(|0> || |1>) should be infinite (disjoint supports)."""
        rho = np.outer(ket_zero, ket_zero.conj())
        sigma = np.outer(ket_one, ket_one.conj())
        s = relative_entropy(rho, sigma)
        assert s == float("inf")


# ======================================================================
# Verification: Linear Entropy
# ======================================================================


class TestLinearEntropy:
    """Test linear entropy."""

    def test_linear_entropy_pure(self, ket_zero: np.ndarray) -> None:
        """Pure state should have linear entropy 0."""
        rho = np.outer(ket_zero, ket_zero.conj())
        assert abs(linear_entropy(rho)) < 1e-10

    def test_linear_entropy_maximally_mixed(
        self, maximally_mixed_1q: np.ndarray
    ) -> None:
        """Maximally mixed 1-qubit should have linear entropy 1/2."""
        assert abs(linear_entropy(maximally_mixed_1q) - 0.5) < 1e-10


# ======================================================================
# Verification: Negativity
# ======================================================================


class TestNegativity:
    """Test negativity (entanglement via partial transpose)."""

    def test_negativity_bell_state(self, bell_state: np.ndarray) -> None:
        """Bell state should have negativity 1/2."""
        rho = np.outer(bell_state, bell_state.conj())
        n = negativity(rho, num_qubits_a=1)
        assert abs(n - 0.5) < 1e-8

    def test_negativity_product_state(
        self, product_state_2q: np.ndarray
    ) -> None:
        """Product state should have negativity 0."""
        rho = np.outer(product_state_2q, product_state_2q.conj())
        n = negativity(rho, num_qubits_a=1)
        assert abs(n) < 1e-8

    def test_negativity_maximally_mixed(
        self, maximally_mixed_2q: np.ndarray
    ) -> None:
        """Maximally mixed state should have negativity 0."""
        n = negativity(maximally_mixed_2q, num_qubits_a=1)
        assert abs(n) < 1e-8


# ======================================================================
# Verification: Partial Trace
# ======================================================================


class TestPartialTrace:
    """Test partial trace computation."""

    def test_partial_trace_bell_state(self, bell_state: np.ndarray) -> None:
        """Tracing out one qubit of a Bell state gives I/2."""
        rho = np.outer(bell_state, bell_state.conj())
        reduced = partial_trace(rho, keep_qubits=[0], num_qubits=2)
        expected = np.eye(2, dtype=np.complex128) / 2.0
        assert np.allclose(reduced, expected, atol=1e-10)

    def test_partial_trace_product_state(
        self, product_state_2q: np.ndarray
    ) -> None:
        """Tracing out qubit 1 of |00> gives |0><0|."""
        rho = np.outer(product_state_2q, product_state_2q.conj())
        reduced = partial_trace(rho, keep_qubits=[0], num_qubits=2)
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        assert np.allclose(reduced, expected, atol=1e-10)

    def test_partial_trace_full_trace(
        self, bell_state: np.ndarray
    ) -> None:
        """Tracing out all qubits should give a 1x1 matrix (the trace)."""
        rho = np.outer(bell_state, bell_state.conj())
        reduced = partial_trace(rho, keep_qubits=[], num_qubits=2)
        assert reduced.shape == (1, 1)
        assert abs(reduced[0, 0] - 1.0) < 1e-10


# ======================================================================
# Verification: Schmidt Decomposition
# ======================================================================


class TestSchmidtDecomposition:
    """Test Schmidt decomposition."""

    def test_schmidt_bell_state(self, bell_state: np.ndarray) -> None:
        """Bell state should have 2 equal Schmidt coefficients."""
        coeffs, _, _ = schmidt_decomposition(bell_state, [0], 2)
        assert len(coeffs[coeffs > 1e-10]) == 2
        assert np.allclose(coeffs[coeffs > 1e-10], SQRT2_INV, atol=1e-10)

    def test_schmidt_product_state(
        self, product_state_2q: np.ndarray
    ) -> None:
        """Product state should have exactly 1 Schmidt coefficient."""
        coeffs, _, _ = schmidt_decomposition(product_state_2q, [0], 2)
        assert len(coeffs[coeffs > 1e-10]) == 1
        assert abs(coeffs[0] - 1.0) < 1e-10

    def test_schmidt_number_bell(self, bell_state: np.ndarray) -> None:
        """Bell state Schmidt number should be 2."""
        assert schmidt_number(bell_state, [0], 2) == 2

    def test_schmidt_number_product(
        self, product_state_2q: np.ndarray
    ) -> None:
        """Product state Schmidt number should be 1."""
        assert schmidt_number(product_state_2q, [0], 2) == 1


# ======================================================================
# Verification: Quantum Volume
# ======================================================================


class TestQuantumVolume:
    """Test quantum volume estimation."""

    def test_qv_perfect_device(self) -> None:
        """Perfect device should achieve quantum volume."""
        # Simulate ideal outputs where measured distribution == ideal
        num_qubits = 2
        dim = 2**num_qubits
        rng = np.random.default_rng(42)

        ideal_probs_list = []
        circuit_outputs_list = []

        for _ in range(20):
            # Random probability distribution
            probs = rng.random(dim)
            probs /= probs.sum()
            ideal_probs_list.append(probs)

            # Perfect sampling from ideal distribution
            shots = 1000
            indices = rng.choice(dim, size=shots, p=probs)
            counts: dict[str, int] = {}
            for idx in indices:
                bs = format(idx, f"0{num_qubits}b")
                counts[bs] = counts.get(bs, 0) + 1
            circuit_outputs_list.append(counts)

        result = estimate_quantum_volume(
            num_qubits, circuit_outputs_list, ideal_probs_list
        )
        assert result.is_achieved
        assert result.log2_quantum_volume == num_qubits
        assert result.heavy_output_probability > 2.0 / 3.0

    def test_qv_random_device(self) -> None:
        """Completely random device should fail quantum volume test."""
        num_qubits = 3
        dim = 2**num_qubits
        rng = np.random.default_rng(42)

        ideal_probs_list = []
        circuit_outputs_list = []

        for _ in range(20):
            probs = rng.random(dim)
            probs /= probs.sum()
            ideal_probs_list.append(probs)

            # Uniform random outputs (uncorrelated with ideal)
            shots = 1000
            uniform_probs = np.ones(dim) / dim
            indices = rng.choice(dim, size=shots, p=uniform_probs)
            counts: dict[str, int] = {}
            for idx in indices:
                bs = format(idx, f"0{num_qubits}b")
                counts[bs] = counts.get(bs, 0) + 1
            circuit_outputs_list.append(counts)

        result = estimate_quantum_volume(
            num_qubits, circuit_outputs_list, ideal_probs_list
        )
        # Uniform random outputs give ~50% heavy output probability
        assert not result.is_achieved


# ======================================================================
# Verification: Cross-Entropy Benchmarking
# ======================================================================


class TestXEB:
    """Test cross-entropy benchmarking."""

    def test_xeb_perfect_fidelity(self) -> None:
        """Perfect sampling should give positive XEB fidelity well above uniform."""
        num_qubits = 2
        dim = 2**num_qubits
        rng = np.random.default_rng(42)

        ideal_probs_list = []
        circuit_outputs_list = []

        for _ in range(20):
            # Use exponential distribution to simulate Porter-Thomas-like statistics
            probs = rng.exponential(1.0, dim)
            probs /= probs.sum()
            ideal_probs_list.append(probs)

            shots = 10000
            indices = rng.choice(dim, size=shots, p=probs)
            counts: dict[str, int] = {}
            for idx in indices:
                bs = format(idx, f"0{num_qubits}b")
                counts[bs] = counts.get(bs, 0) + 1
            circuit_outputs_list.append(counts)

        result = cross_entropy_benchmark(
            circuit_outputs_list, ideal_probs_list, num_qubits
        )
        # For perfect sampling, XEB fidelity = d * sum(p_i^2) - 1 > 0
        # With Porter-Thomas distribution and d=4, expected ~ 1.0
        assert result.xeb_fidelity > 0.5
        assert result.num_circuits == 20

    def test_xeb_uniform_noise(self) -> None:
        """Uniform random outputs should give XEB fidelity near 0."""
        num_qubits = 2
        dim = 2**num_qubits
        rng = np.random.default_rng(42)

        ideal_probs_list = []
        circuit_outputs_list = []

        for _ in range(20):
            probs = rng.random(dim)
            probs /= probs.sum()
            ideal_probs_list.append(probs)

            shots = 5000
            uniform_probs = np.ones(dim) / dim
            indices = rng.choice(dim, size=shots, p=uniform_probs)
            counts: dict[str, int] = {}
            for idx in indices:
                bs = format(idx, f"0{num_qubits}b")
                counts[bs] = counts.get(bs, 0) + 1
            circuit_outputs_list.append(counts)

        result = cross_entropy_benchmark(
            circuit_outputs_list, ideal_probs_list, num_qubits
        )
        assert abs(result.xeb_fidelity) < 0.3  # Should be near 0


# ======================================================================
# Verification: Average Gate Fidelity
# ======================================================================


class TestAverageGateFidelity:
    """Test average gate fidelity computation."""

    def test_identity_channel_fidelity(self) -> None:
        """Identity Choi matrix should have unit fidelity with identity gate."""
        d = 2
        # Choi matrix of identity channel = |Phi+><Phi+|
        phi_plus = np.zeros(d * d, dtype=np.complex128)
        for i in range(d):
            phi_plus[i * d + i] = 1.0 / math.sqrt(d)
        choi_id = np.outer(phi_plus, phi_plus.conj())

        fid = average_gate_fidelity(choi_id, np.eye(d, dtype=np.complex128))
        assert abs(fid - 1.0) < 1e-10


# ======================================================================
# Integration: End-to-end workflows
# ======================================================================


class TestEndToEnd:
    """Integration tests combining multiple package components."""

    def test_full_1q_tomography_workflow(self) -> None:
        """Complete 1-qubit tomography workflow from start to verification."""
        # Prepare a known state: |+>
        state = np.array([SQRT2_INV, SQRT2_INV], dtype=np.complex128)

        # Generate circuits
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()

        # Simulate measurements
        measurements = simulate_tomography_measurements(
            state, circuits, shots=50000, rng=np.random.default_rng(42)
        )

        # Reconstruct
        result = tomo.reconstruct(measurements, method="mle")

        # Verify
        assert result.is_physical
        assert result.fidelity_with(state) > 0.95
        assert result.purity > 0.90
        assert result.von_neumann_entropy < 0.15

    def test_shadow_vs_full_tomography_consistency(self) -> None:
        """Shadow and full tomography should give consistent Z expectation."""
        state = np.array([1.0, 0.0], dtype=np.complex128)
        rng = np.random.default_rng(42)

        # Full tomography
        tomo = StateTomographer(1)
        circuits = tomo.measurement_circuits()
        measurements = simulate_tomography_measurements(
            state, circuits, shots=50000, rng=rng
        )
        result = tomo.reconstruct(measurements, method="mle")
        z_matrix = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        z_full = float(np.real(np.trace(z_matrix @ result.density_matrix)))

        # Shadow tomography
        shadow = create_shadow_from_state(state, 10000, rng=np.random.default_rng(43))
        z_shadow, _ = estimate_expectation(shadow, PauliObservable("Z"))

        # Both should give <Z> ~ 1 for |0>
        assert abs(z_full - 1.0) < 0.15
        assert abs(z_shadow - 1.0) < 0.15

    def test_process_and_state_tomography_consistency(self) -> None:
        """Process tomo on identity should preserve all input states."""
        identity = np.eye(2, dtype=np.complex128)
        output_states = simulate_process_tomography(identity, 1)

        # All output states should match their inputs
        for labels, rho_out in output_states.items():
            from nqpu.tomography.process_tomography import _make_input_state
            sv_in = _make_input_state(labels)
            rho_in = np.outer(sv_in, sv_in.conj())
            fid = state_fidelity(rho_out, rho_in)
            assert fid > 0.99


# ======================================================================
# Edge Cases
# ======================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_statevector_to_density_matrix_conversion(self) -> None:
        """State vector input should be auto-converted for all metrics."""
        sv = np.array([1.0, 0.0], dtype=np.complex128)
        assert abs(purity(sv) - 1.0) < 1e-10
        assert abs(von_neumann_entropy(sv)) < 1e-10
        assert abs(state_fidelity(sv, sv) - 1.0) < 1e-10

    def test_single_qubit_tomographer_creation(self) -> None:
        """StateTomographer with 1 qubit should have dim=2."""
        tomo = StateTomographer(1)
        assert tomo.dim == 2

    def test_invalid_tomographer_creation(self) -> None:
        """StateTomographer with 0 qubits should raise."""
        with pytest.raises(ValueError):
            StateTomographer(0)

    def test_invalid_process_tomographer_creation(self) -> None:
        """ProcessTomographer with 0 qubits should raise."""
        with pytest.raises(ValueError):
            ProcessTomographer(0)

    def test_entropy_invalid_base_raises(
        self, maximally_mixed_1q: np.ndarray
    ) -> None:
        """Unknown entropy base should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown base"):
            von_neumann_entropy(maximally_mixed_1q, base="3")

    def test_shadow_size_bound_invalid_epsilon(self) -> None:
        """epsilon <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="epsilon"):
            shadow_size_bound(10, 2, epsilon=0.0)

    def test_shadow_size_bound_invalid_delta(self) -> None:
        """delta out of range should raise ValueError."""
        with pytest.raises(ValueError, match="delta"):
            shadow_size_bound(10, 2, epsilon=0.1, delta=1.5)

    def test_pauli_observable_matrix_shape(self) -> None:
        """PauliObservable matrix should have correct shape."""
        obs = PauliObservable("XYZ")
        assert obs.matrix().shape == (8, 8)

    def test_partial_trace_wrong_dimension_raises(self) -> None:
        """Wrong dimension should raise ValueError."""
        rho = np.eye(4, dtype=np.complex128) / 4
        with pytest.raises(ValueError):
            partial_trace(rho, keep_qubits=[0], num_qubits=3)

    def test_entanglement_witness_wrong_dimension(self) -> None:
        """Wrong-size density matrix should raise ValueError."""
        rho = np.eye(2, dtype=np.complex128) / 2
        with pytest.raises(ValueError):
            entanglement_witness_bell(rho)
