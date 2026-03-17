"""Comprehensive tests for the nqpu.simulation package.

Tests cover: Hamiltonians (Pauli operators, Ising/Heisenberg/Hubbard models),
time evolution (exact, Trotter, QDrift, adiabatic), Lindblad master equation
(noise channels, solvers, steady state), observables (magnetization, fidelity,
entanglement entropy), and integrators.
"""

from __future__ import annotations

import numpy as np
import pytest

from nqpu.simulation import (
    PauliOperator,
    SparsePauliHamiltonian,
    ising_model,
    heisenberg_model,
    hubbard_model,
    random_hamiltonian,
    EvolutionResult,
    ExactEvolution,
    TrotterEvolution,
    QDrift,
    AdiabaticEvolution,
    SCHEDULE_FUNCTIONS,
    LindbladOperator,
    LindbladMasterEquation,
    LindbladResult,
    LindbladSolver,
    amplitude_damping_operators,
    dephasing_operators,
    depolarizing_operators,
    thermal_operators,
    Observable,
    EntanglementEntropy,
    Magnetization,
    Fidelity,
)


# ---- Fixtures ----


@pytest.fixture
def ising_2q():
    """2-qubit transverse-field Ising model."""
    return ising_model(2, J=1.0, h=0.5)


@pytest.fixture
def initial_state_2q():
    """2-qubit |00> state."""
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1.0
    return psi


@pytest.fixture
def initial_state_4q():
    """4-qubit |0000> state."""
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1.0
    return psi


@pytest.fixture
def qubit_density_matrix():
    """Single qubit |0><0| density matrix."""
    return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)


# ---- Pauli Operator Tests ----


class TestPauliOperator:
    """Tests for PauliOperator construction and algebra."""

    def test_single_qubit_pauli_matrices(self):
        x = PauliOperator("X")
        assert x.matrix().shape == (2, 2)
        assert x.n_qubits == 1
        assert x.weight == 1

    def test_identity_detection(self):
        assert PauliOperator("II").is_identity
        assert not PauliOperator("XZ").is_identity

    def test_pauli_product_xx_equals_i(self):
        x = PauliOperator("X")
        result = x * x
        assert result.label == "I"
        assert abs(result.coeff - 1.0) < 1e-12

    def test_pauli_trace_identity(self):
        op = PauliOperator("II", coeff=1.0)
        assert abs(op.trace() - 4.0) < 1e-12

    def test_pauli_trace_non_identity_zero(self):
        op = PauliOperator("XZ", coeff=1.0)
        assert abs(op.trace()) < 1e-12

    def test_adjoint_hermitian(self):
        op = PauliOperator("XY", coeff=1.0 + 2j)
        adj = op.adjoint()
        assert abs(adj.coeff - (1.0 - 2j)) < 1e-12

    def test_invalid_pauli_label(self):
        with pytest.raises(ValueError, match="Invalid Pauli character"):
            PauliOperator("ABC")

    def test_empty_label(self):
        with pytest.raises(ValueError, match="non-empty"):
            PauliOperator("")


class TestSparsePauliHamiltonian:
    """Tests for Hamiltonian construction and manipulation."""

    def test_matrix_hermitian(self, ising_2q):
        assert ising_2q.is_hermitian()

    def test_eigenvalue_ordering(self, ising_2q):
        evals = ising_2q.eigenvalues()
        assert np.all(np.diff(evals) >= -1e-10)

    def test_ground_state_energy(self, ising_2q):
        energy, state = ising_2q.ground_state()
        evals = ising_2q.eigenvalues()
        assert abs(energy - evals[0]) < 1e-10

    def test_spectral_gap_positive(self, ising_2q):
        gap = ising_2q.spectral_gap()
        assert gap > 0

    def test_expectation_ground_state(self, ising_2q):
        energy, gs = ising_2q.ground_state()
        exp_val = ising_2q.expectation(gs)
        assert abs(exp_val - energy) < 1e-10

    def test_simplify_combines_terms(self):
        t1 = PauliOperator("XZ", coeff=0.5)
        t2 = PauliOperator("XZ", coeff=0.3)
        h = SparsePauliHamiltonian([t1, t2], n_qubits=2)
        simplified = h.simplify()
        assert simplified.num_terms == 1
        assert abs(simplified.terms[0].coeff - 0.8) < 1e-12

    def test_add_hamiltonians(self, ising_2q):
        doubled = ising_2q + ising_2q
        assert doubled.num_terms == 2 * ising_2q.num_terms


# ---- Model Hamiltonian Tests ----


class TestModelHamiltonians:
    """Tests for standard model Hamiltonian constructors."""

    def test_ising_hermitian(self):
        h = ising_model(3, J=1.0, h=0.5)
        assert h.is_hermitian()
        assert h.n_qubits == 3

    def test_heisenberg_hermitian(self):
        h = heisenberg_model(3, Jx=1.0, Jy=1.0, Jz=1.0)
        assert h.is_hermitian()
        assert h.n_qubits == 3

    def test_hubbard_hermitian(self):
        h = hubbard_model(2, t=1.0, U=2.0)
        assert h.is_hermitian()
        assert h.n_qubits == 4  # 2 sites * 2 spins

    def test_random_hamiltonian_hermitian(self):
        h = random_hamiltonian(2, n_terms=5, seed=42)
        assert h.is_hermitian()

    def test_ising_periodic(self):
        h_open = ising_model(3, J=1.0, h=0.5, periodic=False)
        h_periodic = ising_model(3, J=1.0, h=0.5, periodic=True)
        assert h_periodic.num_terms > h_open.num_terms

    def test_heisenberg_requires_two_spins(self):
        with pytest.raises(ValueError, match="at least 2"):
            heisenberg_model(1)


# ---- Time Evolution Tests ----


class TestExactEvolution:
    """Tests for exact matrix-exponential time evolution."""

    def test_unitarity(self, ising_2q, initial_state_2q):
        evolver = ExactEvolution(ising_2q)
        psi_t = evolver.evolve_state(initial_state_2q, t=1.0)
        assert abs(np.linalg.norm(psi_t) - 1.0) < 1e-10

    def test_zero_time_identity(self, ising_2q, initial_state_2q):
        evolver = ExactEvolution(ising_2q)
        psi_t = evolver.evolve_state(initial_state_2q, t=0.0)
        assert np.allclose(psi_t, initial_state_2q, atol=1e-12)

    def test_evolve_records_trajectory(self, ising_2q, initial_state_2q):
        evolver = ExactEvolution(ising_2q)
        result = evolver.evolve(initial_state_2q, t_final=1.0, n_steps=10)
        assert isinstance(result, EvolutionResult)
        assert len(result.states) == 11  # 10 steps + initial
        assert len(result.times) == 11

    def test_propagator_is_unitary(self, ising_2q):
        evolver = ExactEvolution(ising_2q)
        U = evolver.propagator(1.0)
        identity = U @ U.conj().T
        assert np.allclose(identity, np.eye(4), atol=1e-10)


class TestTrotterEvolution:
    """Tests for product formula time evolution."""

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_trotter_preserves_norm(self, ising_2q, initial_state_2q, order):
        trotter = TrotterEvolution(ising_2q, order=order)
        psi_t = trotter.evolve_state(initial_state_2q, t=0.5, n_steps=20)
        assert abs(np.linalg.norm(psi_t) - 1.0) < 1e-6

    def test_trotter_converges_to_exact(self, ising_2q, initial_state_2q):
        exact = ExactEvolution(ising_2q)
        psi_exact = exact.evolve_state(initial_state_2q, t=0.5)

        trotter = TrotterEvolution(ising_2q, order=2)
        psi_trotter = trotter.evolve_state(initial_state_2q, t=0.5, n_steps=50)

        fidelity = abs(np.dot(psi_exact.conj(), psi_trotter)) ** 2
        assert fidelity > 0.95

    def test_invalid_order(self, ising_2q):
        with pytest.raises(ValueError, match="Trotter order must be"):
            TrotterEvolution(ising_2q, order=3)

    def test_estimate_steps(self, ising_2q):
        steps = TrotterEvolution.estimate_steps(ising_2q, t=1.0, target_error=0.01)
        assert steps >= 1


class TestQDrift:
    """Tests for randomized product formula evolution."""

    def test_qdrift_preserves_norm(self, ising_2q, initial_state_2q):
        qdrift = QDrift(ising_2q, seed=42)
        psi_t = qdrift.evolve_state(initial_state_2q, t=0.5, n_samples=100)
        assert abs(np.linalg.norm(psi_t) - 1.0) < 1e-6

    def test_qdrift_evolve_trajectory(self, ising_2q, initial_state_2q):
        qdrift = QDrift(ising_2q, seed=42)
        result = qdrift.evolve(initial_state_2q, t_final=1.0, n_samples=50, n_records=5)
        assert len(result.states) == 6  # 5 records + initial
        assert result.method == "qdrift"


class TestAdiabaticEvolution:
    """Tests for adiabatic quantum evolution."""

    def test_adiabatic_preserves_norm(self):
        h0 = ising_model(2, J=0.0, h=1.0)
        h1 = ising_model(2, J=1.0, h=0.0)
        adiabatic = AdiabaticEvolution(h0, h1, schedule="linear")
        _, gs0 = h0.ground_state()
        result = adiabatic.evolve(gs0, T=5.0, n_steps=50)
        assert abs(np.linalg.norm(result.final_state) - 1.0) < 1e-6

    def test_minimum_gap(self):
        h0 = ising_model(2, J=0.0, h=1.0)
        h1 = ising_model(2, J=1.0, h=0.0)
        adiabatic = AdiabaticEvolution(h0, h1)
        s_min, gap_min = adiabatic.minimum_gap(n_points=50)
        assert 0.0 <= s_min <= 1.0
        assert gap_min >= 0.0

    def test_schedule_functions(self):
        assert "linear" in SCHEDULE_FUNCTIONS
        assert "polynomial" in SCHEDULE_FUNCTIONS
        assert "exponential" in SCHEDULE_FUNCTIONS
        assert SCHEDULE_FUNCTIONS["linear"](0.0) == pytest.approx(0.0)
        assert SCHEDULE_FUNCTIONS["linear"](1.0) == pytest.approx(1.0)


# ---- Lindblad Master Equation Tests ----


class TestLindbladOperator:
    """Tests for Lindblad jump operator construction."""

    def test_amplitude_damping_operator_count(self):
        ops = amplitude_damping_operators(2, gamma=0.1)
        assert len(ops) == 2  # one per qubit

    def test_dephasing_operator_count(self):
        ops = dephasing_operators(2, gamma=0.1)
        assert len(ops) == 2

    def test_depolarizing_operator_count(self):
        ops = depolarizing_operators(2, gamma=0.1)
        assert len(ops) == 6  # 3 per qubit (X, Y, Z)

    def test_thermal_operators_count(self):
        ops = thermal_operators(1, gamma=0.1, n_thermal=0.5)
        assert len(ops) == 2  # emission + absorption

    def test_thermal_zero_occupation(self):
        ops = thermal_operators(1, gamma=0.1, n_thermal=0.0)
        assert len(ops) == 1  # emission only

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            LindbladOperator(operator=np.eye(2), rate=-1.0)


class TestLindbladMasterEquation:
    """Tests for the Lindblad master equation representation."""

    def test_trace_preserving(self, qubit_density_matrix):
        H = np.array([[0, 1], [1, 0]], dtype=complex)  # X Hamiltonian
        ops = dephasing_operators(1, gamma=0.1)
        lme = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        assert lme.is_trace_preserving()

    def test_hermiticity_preserving(self):
        H = np.array([[1, 0], [0, -1]], dtype=complex)  # Z Hamiltonian
        ops = amplitude_damping_operators(1, gamma=0.1)
        lme = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        assert lme.is_hermiticity_preserving()

    def test_liouvillian_shape(self):
        H = np.eye(2, dtype=complex)
        lme = LindbladMasterEquation(hamiltonian=H, jump_operators=[])
        L = lme.liouvillian()
        assert L.shape == (4, 4)  # dim^2 x dim^2


class TestLindbladSolver:
    """Tests for Lindblad equation solvers."""

    def test_rk4_trace_preservation(self, qubit_density_matrix):
        H = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
        ops = dephasing_operators(1, gamma=0.1)
        lme = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=lme, method="rk4")
        result = solver.evolve(qubit_density_matrix, t_final=1.0, n_steps=100)
        traces = result.trace()
        assert np.allclose(traces, 1.0, atol=1e-4)

    def test_purity_decays(self, qubit_density_matrix):
        H = np.zeros((2, 2), dtype=complex)
        ops = depolarizing_operators(1, gamma=0.5)
        lme = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=lme, method="rk4")
        result = solver.evolve(qubit_density_matrix, t_final=5.0, n_steps=200)
        purities = result.purity()
        assert purities[0] > purities[-1]  # purity decreases

    def test_exact_solver(self, qubit_density_matrix):
        H = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
        ops = dephasing_operators(1, gamma=0.1)
        lme = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=lme, method="exact")
        result = solver.evolve(qubit_density_matrix, t_final=1.0, n_steps=20)
        assert len(result.states) == 21

    def test_steady_state(self):
        H = np.zeros((2, 2), dtype=complex)
        ops = amplitude_damping_operators(1, gamma=1.0)
        lme = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=lme, method="rk4")
        rho_ss = solver.steady_state()
        assert abs(np.trace(rho_ss) - 1.0) < 1e-6
        # For pure amplitude damping, steady state is |0><0|
        assert rho_ss[0, 0] > 0.9

    def test_invalid_method(self):
        H = np.eye(2, dtype=complex)
        lme = LindbladMasterEquation(hamiltonian=H)
        with pytest.raises(ValueError, match="Unknown method"):
            LindbladSolver(equation=lme, method="invalid")


class TestLindbladResult:
    """Tests for Lindblad result analysis."""

    def test_von_neumann_entropy_pure_state(self, qubit_density_matrix):
        result = LindbladResult(
            times=np.array([0.0]),
            states=[qubit_density_matrix],
        )
        entropy = result.von_neumann_entropy()
        assert entropy[0] < 0.01  # ~0 for pure state

    def test_populations(self, qubit_density_matrix):
        result = LindbladResult(
            times=np.array([0.0]),
            states=[qubit_density_matrix],
        )
        pops = result.populations()
        assert pops.shape == (1, 2)
        assert abs(pops[0, 0] - 1.0) < 1e-12


# ---- Observables Tests ----


class TestObservables:
    """Tests for observable measurement tools."""

    def test_observable_expectation_eigenstate(self):
        z_obs = Observable(np.array([[1, 0], [0, -1]], dtype=complex), name="Z")
        state_0 = np.array([1.0, 0.0], dtype=complex)
        assert z_obs.expectation(state_0) == pytest.approx(1.0)

    def test_observable_variance_eigenstate_zero(self):
        z_obs = Observable(np.array([[1, 0], [0, -1]], dtype=complex), name="Z")
        state_0 = np.array([1.0, 0.0], dtype=complex)
        assert z_obs.variance(state_0) == pytest.approx(0.0, abs=1e-10)

    def test_magnetization_all_up(self):
        mag = Magnetization(2)
        state_00 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        local = mag.local(state_00)
        assert np.allclose(local, [1.0, 1.0], atol=1e-10)
        assert mag.total(state_00) == pytest.approx(1.0)

    def test_fidelity_same_state(self):
        state = np.array([1.0, 0.0], dtype=complex)
        fid = Fidelity(state)
        assert fid.compute(state) == pytest.approx(1.0)

    def test_fidelity_orthogonal_states(self):
        state0 = np.array([1.0, 0.0], dtype=complex)
        state1 = np.array([0.0, 1.0], dtype=complex)
        fid = Fidelity(state0)
        assert fid.compute(state1) == pytest.approx(0.0)

    def test_entanglement_entropy_product_state(self):
        ee = EntanglementEntropy(2, n_a=1)
        product_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        entropy = ee.von_neumann(product_state)
        assert entropy == pytest.approx(0.0, abs=1e-8)

    def test_entanglement_entropy_bell_state(self):
        ee = EntanglementEntropy(2, n_a=1)
        bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)
        entropy = ee.von_neumann(bell)
        assert entropy == pytest.approx(np.log(2), abs=1e-8)

    def test_renyi_entropy(self):
        ee = EntanglementEntropy(2, n_a=1)
        bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)
        s2 = ee.renyi(bell, alpha=2.0)
        assert s2 == pytest.approx(np.log(2), abs=1e-8)

    def test_fidelity_trajectory(self, ising_2q, initial_state_2q):
        evolver = ExactEvolution(ising_2q)
        result = evolver.evolve(initial_state_2q, t_final=1.0, n_steps=5)
        fid = Fidelity(initial_state_2q)
        traj = fid.trajectory(result.times, result.states)
        assert traj[0] == pytest.approx(1.0)
        assert len(traj) == len(result.times)
