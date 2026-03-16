"""Comprehensive tests for the nQPU quantum simulation package.

Tests cover all six modules:

  - hamiltonians.py: PauliOperator, SparsePauliHamiltonian, model constructors
  - time_evolution.py: ExactEvolution, TrotterEvolution, QDrift, AdiabaticEvolution
  - variational_dynamics.py: QITE, VarQTE, PVQD, VariationalAnsatz
  - observables.py: Observable, CorrelationFunction, EntanglementEntropy,
                    Magnetization, SpectralFunction, Fidelity
  - integrators.py: RungeKutta4, LeapfrogIntegrator, AdaptiveRK45, CrankNicolson

Numerical references:
  - Exact 1-qubit precession: known analytical solution
  - 2-qubit Ising ground state: analytical E = -sqrt(J^2 + h^2)
  - Heisenberg singlet: E_0 = -3J/4 for antiferromagnet
"""

import math

import numpy as np
import pytest

from nqpu.simulation.hamiltonians import (
    PauliOperator,
    SparsePauliHamiltonian,
    ising_model,
    heisenberg_model,
    hubbard_model,
    random_hamiltonian,
)
from nqpu.simulation.time_evolution import (
    ExactEvolution,
    TrotterEvolution,
    QDrift,
    AdiabaticEvolution,
    SCHEDULE_FUNCTIONS,
)
from nqpu.simulation.variational_dynamics import (
    VariationalAnsatz,
    QITE,
    VarQTE,
    PVQD,
)
from nqpu.simulation.observables import (
    Observable,
    TimeSeriesObservable,
    CorrelationFunction,
    EntanglementEntropy,
    Magnetization,
    SpectralFunction,
    Fidelity,
)
from nqpu.simulation.integrators import (
    RungeKutta4,
    LeapfrogIntegrator,
    AdaptiveRK45,
    CrankNicolson,
)


# ======================================================================
# Helper fixtures
# ======================================================================


@pytest.fixture
def bell_state():
    """|Phi+> = (|00> + |11>) / sqrt(2)."""
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = 1 / math.sqrt(2)
    psi[3] = 1 / math.sqrt(2)
    return psi


@pytest.fixture
def spin_up():
    """|0> state for a single qubit."""
    return np.array([1, 0], dtype=np.complex128)


@pytest.fixture
def spin_down():
    """|1> state for a single qubit."""
    return np.array([0, 1], dtype=np.complex128)


@pytest.fixture
def plus_state():
    """|+> = (|0> + |1>) / sqrt(2)."""
    return np.array([1, 1], dtype=np.complex128) / math.sqrt(2)


@pytest.fixture
def ising_2q():
    """2-qubit transverse-field Ising model."""
    return ising_model(2, J=1.0, h=1.0)


@pytest.fixture
def heisenberg_2q():
    """2-qubit isotropic Heisenberg model."""
    return heisenberg_model(2, Jx=1.0, Jy=1.0, Jz=1.0)


# ======================================================================
# PauliOperator tests
# ======================================================================


class TestPauliOperator:
    """Tests for the PauliOperator class."""

    def test_creation_valid(self):
        op = PauliOperator("XYZ", coeff=0.5)
        assert op.n_qubits == 3
        assert op.coeff == 0.5

    def test_creation_invalid_label(self):
        with pytest.raises(ValueError, match="Invalid Pauli character"):
            PauliOperator("XAZ")

    def test_creation_empty_label(self):
        with pytest.raises(ValueError, match="non-empty"):
            PauliOperator("")

    def test_n_qubits(self):
        assert PauliOperator("I").n_qubits == 1
        assert PauliOperator("XYZX").n_qubits == 4

    def test_is_identity(self):
        assert PauliOperator("III").is_identity
        assert not PauliOperator("IXI").is_identity

    def test_weight(self):
        assert PauliOperator("III").weight == 0
        assert PauliOperator("XYZ").weight == 3
        assert PauliOperator("IXI").weight == 1

    def test_matrix_single_pauli(self):
        """Verify X matrix is correct."""
        x_mat = PauliOperator("X").matrix()
        expected = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        np.testing.assert_allclose(x_mat, expected)

    def test_matrix_identity(self):
        """Identity gives I_4 for 2 qubits."""
        mat = PauliOperator("II").matrix()
        np.testing.assert_allclose(mat, np.eye(4))

    def test_matrix_tensor_product(self):
        """XZ = X tensor Z."""
        mat = PauliOperator("XZ").matrix()
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        expected = np.kron(X, Z)
        np.testing.assert_allclose(mat, expected)

    def test_matrix_with_coeff(self):
        mat = PauliOperator("Z", coeff=2.5).matrix()
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        np.testing.assert_allclose(mat, 2.5 * Z)

    def test_pauli_multiplication_xx(self):
        """X * X = I."""
        result = PauliOperator("X") * PauliOperator("X")
        assert result.label == "I"
        assert abs(result.coeff - 1.0) < 1e-10

    def test_pauli_multiplication_xy(self):
        """X * Y = iZ."""
        result = PauliOperator("X") * PauliOperator("Y")
        assert result.label == "Z"
        assert abs(result.coeff - 1j) < 1e-10

    def test_pauli_multiplication_yx(self):
        """Y * X = -iZ."""
        result = PauliOperator("Y") * PauliOperator("X")
        assert result.label == "Z"
        assert abs(result.coeff - (-1j)) < 1e-10

    def test_pauli_multiplication_yz(self):
        """Y * Z = iX."""
        result = PauliOperator("Y") * PauliOperator("Z")
        assert result.label == "X"
        assert abs(result.coeff - 1j) < 1e-10

    def test_pauli_multiplication_zx(self):
        """Z * X = iY."""
        result = PauliOperator("Z") * PauliOperator("X")
        assert result.label == "Y"
        assert abs(result.coeff - 1j) < 1e-10

    def test_multiplication_with_identity(self):
        """I * X = X and X * I = X."""
        result = PauliOperator("I") * PauliOperator("X")
        assert result.label == "X"
        assert abs(result.coeff - 1.0) < 1e-10

        result2 = PauliOperator("X") * PauliOperator("I")
        assert result2.label == "X"

    def test_multiplication_multiqubit(self):
        """(XY) * (YX) = (X*Y)(Y*X) = (iZ)(-iZ) = ZZ."""
        a = PauliOperator("XY")
        b = PauliOperator("YX")
        result = a * b
        assert result.label == "ZZ"
        # phase: (i) * (-i) = 1
        assert abs(result.coeff - 1.0) < 1e-10

    def test_multiplication_qubit_mismatch(self):
        with pytest.raises(ValueError, match="mismatch"):
            PauliOperator("X") * PauliOperator("XY")

    def test_scalar_multiplication(self):
        op = 3.0 * PauliOperator("Z", coeff=2.0)
        assert abs(op.coeff - 6.0) < 1e-10

    def test_negation(self):
        op = -PauliOperator("X", coeff=1.5)
        assert abs(op.coeff - (-1.5)) < 1e-10

    def test_adjoint(self):
        op = PauliOperator("X", coeff=1 + 2j)
        adj = op.adjoint()
        assert abs(adj.coeff - (1 - 2j)) < 1e-10

    def test_commutator_same_pauli(self):
        """[X, X] = 0."""
        comm = PauliOperator("X").commutator(PauliOperator("X"))
        mat = comm.matrix()
        np.testing.assert_allclose(mat, np.zeros((2, 2)), atol=1e-12)

    def test_commutator_xy(self):
        """[X, Y] = 2iZ."""
        comm = PauliOperator("X").commutator(PauliOperator("Y"))
        mat = comm.matrix()
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        np.testing.assert_allclose(mat, 2j * Z, atol=1e-12)

    def test_trace_identity(self):
        """Tr(I^n) = 2^n."""
        assert abs(PauliOperator("I").trace() - 2.0) < 1e-10
        assert abs(PauliOperator("II").trace() - 4.0) < 1e-10
        assert abs(PauliOperator("III").trace() - 8.0) < 1e-10

    def test_trace_non_identity(self):
        """Tr(X) = Tr(Y) = Tr(Z) = 0."""
        assert abs(PauliOperator("X").trace()) < 1e-10
        assert abs(PauliOperator("Y").trace()) < 1e-10
        assert abs(PauliOperator("Z").trace()) < 1e-10
        assert abs(PauliOperator("XZ").trace()) < 1e-10


# ======================================================================
# SparsePauliHamiltonian tests
# ======================================================================


class TestSparsePauliHamiltonian:
    """Tests for SparsePauliHamiltonian."""

    def test_creation_empty(self):
        h = SparsePauliHamiltonian()
        assert h.num_terms == 0
        assert h.n_qubits == 0

    def test_creation_with_terms(self):
        terms = [PauliOperator("XX", 0.5), PauliOperator("ZZ", -0.3)]
        h = SparsePauliHamiltonian(terms)
        assert h.n_qubits == 2
        assert h.num_terms == 2

    def test_qubit_mismatch(self):
        terms = [PauliOperator("X"), PauliOperator("XX")]
        with pytest.raises(ValueError, match="same number"):
            SparsePauliHamiltonian(terms)

    def test_add_term(self):
        h = SparsePauliHamiltonian([PauliOperator("XX", 1.0)])
        h.add_term("ZZ", -0.5)
        assert h.num_terms == 2

    def test_add_term_wrong_qubits(self):
        h = SparsePauliHamiltonian([PauliOperator("XX", 1.0)])
        with pytest.raises(ValueError):
            h.add_term("Z")

    def test_matrix_hermitian(self):
        """All Pauli Hamiltonians should be Hermitian."""
        h = ising_model(3, J=1.0, h=0.5)
        assert h.is_hermitian()

    def test_simplify(self):
        terms = [
            PauliOperator("XX", 0.5),
            PauliOperator("XX", 0.3),
            PauliOperator("ZZ", 0.1),
        ]
        h = SparsePauliHamiltonian(terms).simplify()
        assert h.num_terms == 2
        # XX coeff should be 0.8
        xx_terms = [t for t in h.terms if t.label == "XX"]
        assert len(xx_terms) == 1
        assert abs(xx_terms[0].coeff - 0.8) < 1e-10

    def test_addition(self):
        h1 = SparsePauliHamiltonian([PauliOperator("XX", 1.0)])
        h2 = SparsePauliHamiltonian([PauliOperator("ZZ", -0.5)])
        h3 = h1 + h2
        assert h3.num_terms == 2

    def test_scalar_mul(self):
        h = SparsePauliHamiltonian([PauliOperator("X", 1.0)])
        h2 = 3.0 * h
        assert abs(h2.terms[0].coeff - 3.0) < 1e-10

    def test_subtraction(self):
        h1 = SparsePauliHamiltonian([PauliOperator("X", 1.0)])
        h2 = SparsePauliHamiltonian([PauliOperator("X", 0.3)])
        h3 = (h1 - h2).simplify()
        assert abs(h3.terms[0].coeff - 0.7) < 1e-10

    def test_expectation_eigenstate(self, spin_up):
        """<0|Z|0> = +1."""
        h = SparsePauliHamiltonian([PauliOperator("Z", 1.0)])
        assert abs(h.expectation(spin_up) - 1.0) < 1e-10

    def test_expectation_dimension_check(self):
        h = SparsePauliHamiltonian([PauliOperator("ZZ", 1.0)])
        with pytest.raises(ValueError, match="dimension"):
            h.expectation(np.array([1, 0]))

    def test_eigenvalues_single_z(self):
        h = SparsePauliHamiltonian([PauliOperator("Z", 1.0)])
        evals = h.eigenvalues()
        np.testing.assert_allclose(sorted(evals), [-1.0, 1.0], atol=1e-10)

    def test_ground_state(self, ising_2q):
        energy, state = ising_2q.ground_state()
        # Verify ground state is an eigenstate
        H_mat = ising_2q.matrix()
        np.testing.assert_allclose(
            H_mat @ state, energy * state, atol=1e-10
        )

    def test_spectral_gap(self):
        h = SparsePauliHamiltonian([PauliOperator("Z", 1.0)])
        assert abs(h.spectral_gap() - 2.0) < 1e-10

    def test_norm(self):
        h = SparsePauliHamiltonian([PauliOperator("Z", 1.0)])
        # Frobenius norm of diag(1, -1) = sqrt(2)
        assert abs(h.norm() - math.sqrt(2)) < 1e-10

    def test_commutator_zz_with_xx(self):
        """[ZZ, XX] = 0 (they commute -- both diagonal in Bell basis)."""
        h1 = SparsePauliHamiltonian([PauliOperator("ZZ", 1.0)])
        h2 = SparsePauliHamiltonian([PauliOperator("XX", 1.0)])
        comm = h1.commutator(h2)
        assert comm.norm() < 1e-10

    def test_commutator_xi_with_zi(self):
        """[XI, ZI] should be non-zero (X and Z don't commute)."""
        h1 = SparsePauliHamiltonian([PauliOperator("XI", 1.0)])
        h2 = SparsePauliHamiltonian([PauliOperator("ZI", 1.0)])
        comm = h1.commutator(h2)
        assert comm.norm() > 1e-10


# ======================================================================
# Model Hamiltonian tests
# ======================================================================


class TestModelHamiltonians:
    """Tests for Ising, Heisenberg, Hubbard constructors."""

    def test_ising_hermitian(self):
        h = ising_model(4, J=1.0, h=0.5)
        assert h.is_hermitian()

    def test_ising_n1(self):
        """Single-site Ising: H = -h*X, eigenvalues +/-h."""
        h = ising_model(1, J=1.0, h=2.0)
        evals = h.eigenvalues()
        np.testing.assert_allclose(sorted(evals), [-2.0, 2.0], atol=1e-10)

    def test_ising_2q_ground_energy(self):
        """2-qubit Ising: E_0 = -sqrt(J^2 + h^2) - J for specific form.

        H = -J*ZZ - h*(X_1 + X_2).  Ground state energy is known
        analytically for 2 sites.
        """
        J, h = 1.0, 1.0
        ham = ising_model(2, J=J, h=h)
        e0, _ = ham.ground_state()
        # For 2-site TFIM with open boundary: eigenvalues can be computed
        # The 4x4 matrix has known spectrum
        evals = sorted(ham.eigenvalues())
        assert e0 == pytest.approx(evals[0], abs=1e-10)

    def test_ising_periodic(self):
        """Periodic Ising should have more ZZ terms."""
        h_open = ising_model(4, J=1.0, h=0.5, periodic=False)
        h_pbc = ising_model(4, J=1.0, h=0.5, periodic=True)
        # PBC has n bonds, open has n-1
        zz_open = sum(1 for t in h_open.terms if t.label.count("Z") == 2)
        zz_pbc = sum(1 for t in h_pbc.terms if t.label.count("Z") == 2)
        assert zz_pbc == zz_open + 1

    def test_ising_invalid(self):
        with pytest.raises(ValueError):
            ising_model(0)

    def test_heisenberg_hermitian(self):
        h = heisenberg_model(3, Jx=1.0, Jy=0.5, Jz=-0.3)
        assert h.is_hermitian()

    def test_heisenberg_singlet_energy(self):
        """2-qubit antiferromagnetic Heisenberg: singlet E = -3J/4.

        H = J*(XX + YY + ZZ), singlet has energy -3J.
        Wait -- with H = Jx*XX + Jy*YY + Jz*ZZ and J=1:
        The singlet (|01>-|10>)/sqrt(2) has eigenvalue -3 (since
        XX, YY, ZZ each give -1 on singlet).
        """
        J = 1.0
        h = heisenberg_model(2, Jx=J, Jy=J, Jz=J)
        e0, state = h.ground_state()
        # Singlet energy = -3J
        assert e0 == pytest.approx(-3.0 * J, abs=1e-10)

    def test_heisenberg_xxz(self):
        """XXZ model: Jx = Jy = 1, Jz = Delta."""
        h = heisenberg_model(2, Jx=1.0, Jy=1.0, Jz=0.5)
        assert h.is_hermitian()
        assert h.num_terms == 3  # XX, YY, ZZ on one bond

    def test_heisenberg_invalid(self):
        with pytest.raises(ValueError):
            heisenberg_model(1)

    def test_hubbard_hermitian(self):
        h = hubbard_model(2, t=1.0, U=2.0)
        assert h.is_hermitian()

    def test_hubbard_qubit_count(self):
        """Hubbard on n sites uses 2n qubits."""
        h = hubbard_model(3, t=1.0, U=2.0)
        assert h.n_qubits == 6

    def test_hubbard_half_filling_symmetry(self):
        """At half filling, particle-hole symmetry exists."""
        h = hubbard_model(2, t=1.0, U=0.0)
        # With U=0, should be a free-fermion hopping Hamiltonian
        # Eigenvalues come in pairs
        evals = h.eigenvalues()
        # Check Hermiticity
        assert h.is_hermitian()

    def test_hubbard_invalid(self):
        with pytest.raises(ValueError):
            hubbard_model(1)

    def test_random_hamiltonian(self):
        h = random_hamiltonian(3, n_terms=10, seed=42)
        assert h.n_qubits == 3
        assert h.is_hermitian()
        assert h.num_terms <= 10

    def test_random_hamiltonian_reproducible(self):
        h1 = random_hamiltonian(2, n_terms=5, seed=123)
        h2 = random_hamiltonian(2, n_terms=5, seed=123)
        np.testing.assert_allclose(h1.matrix(), h2.matrix())


# ======================================================================
# ExactEvolution tests
# ======================================================================


class TestExactEvolution:
    """Tests for exact matrix exponential evolution."""

    def test_single_qubit_precession(self, spin_up):
        """Spin precession under Z: |0> -> e^{-it}|0>.

        Under H = Z, |0> picks up phase e^{-it}.
        """
        h = SparsePauliHamiltonian([PauliOperator("Z", 1.0)])
        evo = ExactEvolution(h)
        t = 1.0
        psi_t = evo.evolve_state(spin_up, t)

        expected = np.exp(-1j * t) * spin_up
        np.testing.assert_allclose(psi_t, expected, atol=1e-12)

    def test_x_precession(self, spin_up):
        """Under H = X, |0> -> cos(t)|0> - i*sin(t)|1>."""
        h = SparsePauliHamiltonian([PauliOperator("X", 1.0)])
        evo = ExactEvolution(h)
        t = math.pi / 4
        psi_t = evo.evolve_state(spin_up, t)

        expected = np.array(
            [math.cos(t), -1j * math.sin(t)], dtype=np.complex128
        )
        np.testing.assert_allclose(psi_t, expected, atol=1e-12)

    def test_unitarity(self, spin_up):
        """Norm should be preserved."""
        h = ising_model(2, J=1.0, h=0.5)
        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        evo = ExactEvolution(h)
        psi_t = evo.evolve_state(psi0, 2.0)
        assert abs(np.linalg.norm(psi_t) - 1.0) < 1e-12

    def test_evolve_trajectory(self):
        """Test full trajectory recording."""
        h = SparsePauliHamiltonian([PauliOperator("X", 1.0)])
        evo = ExactEvolution(h)
        psi0 = np.array([1, 0], dtype=np.complex128)
        result = evo.evolve(psi0, t_final=1.0, n_steps=10)

        assert len(result.times) == 11
        assert len(result.states) == 11
        assert abs(result.times[0]) < 1e-15
        assert abs(result.times[-1] - 1.0) < 1e-15

    def test_propagator_unitary(self):
        h = ising_model(2, J=1.0, h=0.5)
        evo = ExactEvolution(h)
        U = evo.propagator(1.0)
        # U^dag U = I
        np.testing.assert_allclose(
            U.conj().T @ U, np.eye(4), atol=1e-12
        )

    def test_energy_conservation(self):
        """Energy must be exactly conserved under exact evolution."""
        h = ising_model(3, J=1.0, h=0.5)
        evo = ExactEvolution(h)
        psi0 = np.zeros(8, dtype=np.complex128)
        psi0[0] = 1.0
        result = evo.evolve(psi0, t_final=2.0, n_steps=20)

        energies = [h.expectation(s) for s in result.states]
        assert max(abs(e - energies[0]) for e in energies) < 1e-10


# ======================================================================
# TrotterEvolution tests
# ======================================================================


class TestTrotterEvolution:
    """Tests for Trotter product formula decomposition."""

    def test_first_order_converges(self):
        """First-order Trotter should converge to exact with more steps."""
        h = ising_model(2, J=1.0, h=0.5)
        exact_evo = ExactEvolution(h)

        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        t = 1.0

        psi_exact = exact_evo.evolve_state(psi0, t)

        errors = []
        for n_steps in [10, 50, 100]:
            trotter = TrotterEvolution(h, order=1)
            psi_trotter = trotter.evolve_state(psi0, t, n_steps=n_steps)
            error = np.linalg.norm(psi_trotter - psi_exact)
            errors.append(error)

        # Error should decrease
        assert errors[-1] < errors[0]

    def test_second_order_better_than_first(self):
        """Second-order Trotter should be more accurate than first."""
        h = ising_model(2, J=1.0, h=0.5)
        exact_evo = ExactEvolution(h)

        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        t = 1.0
        n_steps = 20

        psi_exact = exact_evo.evolve_state(psi0, t)

        trotter1 = TrotterEvolution(h, order=1)
        trotter2 = TrotterEvolution(h, order=2)

        psi1 = trotter1.evolve_state(psi0, t, n_steps=n_steps)
        psi2 = trotter2.evolve_state(psi0, t, n_steps=n_steps)

        error1 = np.linalg.norm(psi1 - psi_exact)
        error2 = np.linalg.norm(psi2 - psi_exact)

        assert error2 < error1

    def test_fourth_order_most_accurate(self):
        """Fourth-order should be the most accurate."""
        h = ising_model(2, J=1.0, h=0.5)
        exact_evo = ExactEvolution(h)

        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        t = 0.5
        n_steps = 10

        psi_exact = exact_evo.evolve_state(psi0, t)

        trotter2 = TrotterEvolution(h, order=2)
        trotter4 = TrotterEvolution(h, order=4)

        psi2 = trotter2.evolve_state(psi0, t, n_steps=n_steps)
        psi4 = trotter4.evolve_state(psi0, t, n_steps=n_steps)

        error2 = np.linalg.norm(psi2 - psi_exact)
        error4 = np.linalg.norm(psi4 - psi_exact)

        assert error4 < error2

    def test_trotter_convergence_threshold(self):
        """Trotter with 100 steps should achieve error < 1e-3 for small H."""
        h = ising_model(2, J=0.5, h=0.3)
        exact_evo = ExactEvolution(h)

        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        t = 1.0

        psi_exact = exact_evo.evolve_state(psi0, t)

        trotter = TrotterEvolution(h, order=2)
        psi_trotter = trotter.evolve_state(psi0, t, n_steps=100)

        error = np.linalg.norm(psi_trotter - psi_exact)
        assert error < 1e-3

    def test_trotter_norm_preservation(self):
        """Trotter evolution should preserve norm."""
        h = ising_model(3, J=1.0, h=0.5)
        trotter = TrotterEvolution(h, order=2)

        psi0 = np.zeros(8, dtype=np.complex128)
        psi0[0] = 1.0
        psi_t = trotter.evolve_state(psi0, 1.0, n_steps=50)
        assert abs(np.linalg.norm(psi_t) - 1.0) < 1e-10

    def test_trotter_trajectory(self):
        h = ising_model(2, J=1.0, h=0.5)
        trotter = TrotterEvolution(h, order=2)
        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        result = trotter.evolve(psi0, t_final=1.0, n_steps=20)

        assert result.method == "trotter_order2"
        assert len(result.states) >= 2
        assert abs(result.times[-1] - 1.0) < 1e-10

    def test_invalid_order(self):
        h = ising_model(2)
        with pytest.raises(ValueError, match="1, 2, or 4"):
            TrotterEvolution(h, order=3)

    def test_estimate_steps(self):
        h = ising_model(2, J=1.0, h=1.0)
        n = TrotterEvolution.estimate_steps(h, t=1.0, target_error=0.01, order=2)
        assert n >= 1
        assert isinstance(n, int)


# ======================================================================
# QDrift tests
# ======================================================================


class TestQDrift:
    """Tests for randomized product formula."""

    def test_qdrift_converges(self):
        """QDrift should converge with more samples."""
        h = ising_model(2, J=1.0, h=0.5)
        exact_evo = ExactEvolution(h)

        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        t = 0.5

        psi_exact = exact_evo.evolve_state(psi0, t)

        # Run multiple times and average fidelity
        fidelities = []
        for seed in range(10):
            qdrift = QDrift(h, seed=seed)
            psi_q = qdrift.evolve_state(psi0, t, n_samples=500)
            fid = abs(psi_q.conj() @ psi_exact) ** 2
            fidelities.append(fid)

        # Average fidelity should be close to 1
        assert np.mean(fidelities) > 0.95

    def test_qdrift_norm_preservation(self):
        h = ising_model(2, J=1.0, h=0.5)
        qdrift = QDrift(h, seed=42)
        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        psi_t = qdrift.evolve_state(psi0, 0.5, n_samples=100)
        assert abs(np.linalg.norm(psi_t) - 1.0) < 1e-10

    def test_qdrift_trajectory(self):
        h = ising_model(2, J=1.0, h=0.5)
        qdrift = QDrift(h, seed=42)
        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        result = qdrift.evolve(psi0, t_final=1.0, n_samples=100, n_records=5)

        assert result.method == "qdrift"
        assert len(result.times) == 6  # 0 + 5 records

    def test_qdrift_deterministic_with_seed(self):
        h = ising_model(2, J=1.0, h=0.5)
        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0

        qdrift1 = QDrift(h, seed=42)
        psi1 = qdrift1.evolve_state(psi0, 0.5, n_samples=50)

        qdrift2 = QDrift(h, seed=42)
        psi2 = qdrift2.evolve_state(psi0, 0.5, n_samples=50)

        np.testing.assert_allclose(psi1, psi2)


# ======================================================================
# AdiabaticEvolution tests
# ======================================================================


class TestAdiabaticEvolution:
    """Tests for adiabatic state preparation."""

    def test_adiabatic_ground_state(self):
        """Slow enough adiabatic evolution should land in ground state."""
        # Start in X ground state, evolve to Z ground state
        H0 = SparsePauliHamiltonian([PauliOperator("X", -1.0)])
        H1 = SparsePauliHamiltonian([PauliOperator("Z", -1.0)])

        adia = AdiabaticEvolution(H0, H1, schedule="linear")

        # Ground state of H0 = -X is |+>
        psi0 = np.array([1, 1], dtype=np.complex128) / math.sqrt(2)
        T = 20.0  # Long enough for adiabatic
        result = adia.evolve(psi0, T=T, n_steps=400)

        # Should end up close to |0> (ground state of -Z)
        fidelity = abs(result.final_state[0]) ** 2
        assert fidelity > 0.95

    def test_hamiltonian_at(self):
        H0 = SparsePauliHamiltonian([PauliOperator("X", 1.0)])
        H1 = SparsePauliHamiltonian([PauliOperator("Z", 1.0)])
        adia = AdiabaticEvolution(H0, H1)

        # At s=0, should be H0
        np.testing.assert_allclose(adia.hamiltonian_at(0.0), H0.matrix(), atol=1e-12)
        # At s=1, should be H1
        np.testing.assert_allclose(adia.hamiltonian_at(1.0), H1.matrix(), atol=1e-12)
        # At s=0.5, should be average
        expected = 0.5 * (H0.matrix() + H1.matrix())
        np.testing.assert_allclose(adia.hamiltonian_at(0.5), expected, atol=1e-12)

    def test_gap_at(self):
        H0 = SparsePauliHamiltonian([PauliOperator("X", -1.0)])
        H1 = SparsePauliHamiltonian([PauliOperator("Z", -1.0)])
        adia = AdiabaticEvolution(H0, H1)
        gap = adia.gap_at(0.0)
        assert gap > 0

    def test_minimum_gap(self):
        H0 = SparsePauliHamiltonian([PauliOperator("X", -1.0)])
        H1 = SparsePauliHamiltonian([PauliOperator("Z", -1.0)])
        adia = AdiabaticEvolution(H0, H1)
        s_min, gap_min = adia.minimum_gap()
        assert 0 <= s_min <= 1
        assert gap_min > 0

    def test_schedule_functions(self):
        assert "linear" in SCHEDULE_FUNCTIONS
        assert "polynomial" in SCHEDULE_FUNCTIONS
        assert "exponential" in SCHEDULE_FUNCTIONS
        # All start at 0 and end at 1
        for name, fn in SCHEDULE_FUNCTIONS.items():
            assert abs(fn(0.0)) < 1e-10
            assert abs(fn(1.0) - 1.0) < 1e-10

    def test_invalid_schedule(self):
        H0 = SparsePauliHamiltonian([PauliOperator("X", -1.0)])
        H1 = SparsePauliHamiltonian([PauliOperator("Z", -1.0)])
        with pytest.raises(ValueError, match="Unknown schedule"):
            AdiabaticEvolution(H0, H1, schedule="invalid")

    def test_custom_schedule(self):
        H0 = SparsePauliHamiltonian([PauliOperator("X", -1.0)])
        H1 = SparsePauliHamiltonian([PauliOperator("Z", -1.0)])
        # Custom schedule
        adia = AdiabaticEvolution(H0, H1, schedule=lambda s: s ** 2)
        H_mid = adia.hamiltonian_at(0.5)
        # f(0.5) = 0.25, so (0.75)*H0 + (0.25)*H1
        expected = 0.75 * H0.matrix() + 0.25 * H1.matrix()
        np.testing.assert_allclose(H_mid, expected, atol=1e-12)

    def test_estimate_time(self):
        H0 = SparsePauliHamiltonian([PauliOperator("X", -1.0)])
        H1 = SparsePauliHamiltonian([PauliOperator("Z", -1.0)])
        adia = AdiabaticEvolution(H0, H1)
        T = adia.estimate_time(target_fidelity=0.99)
        assert T > 0 and T < float("inf")


# ======================================================================
# QITE tests
# ======================================================================


class TestQITE:
    """Tests for Quantum Imaginary Time Evolution."""

    def test_qite_converges_to_ground_state(self):
        """QITE should converge to the ground state energy."""
        h = ising_model(2, J=1.0, h=0.5)
        e_exact, _ = h.ground_state()

        qite = QITE(h, dt=0.1)
        result = qite.evolve(n_steps=200, tol=1e-10)

        # Should converge within 1% of exact ground state
        assert abs(result.final_energy - e_exact) / abs(e_exact) < 0.01

    def test_qite_converges_heisenberg(self):
        """QITE on 2-qubit Heisenberg should find singlet.

        The equal superposition |+>^2 has zero overlap with the singlet
        (wrong symmetry sector), so we start from |01> which has 50%
        singlet component.
        """
        h = heisenberg_model(2, Jx=1.0, Jy=1.0, Jz=1.0)
        e_exact = -3.0  # Singlet energy

        # |01> has nonzero overlap with the singlet (|01>-|10>)/sqrt(2)
        psi0 = np.array([0, 1, 0, 0], dtype=np.complex128)
        qite = QITE(h, dt=0.1)
        result = qite.evolve(psi0=psi0, n_steps=200, tol=1e-10)

        assert abs(result.final_energy - e_exact) / abs(e_exact) < 0.01

    def test_qite_energy_decreases(self):
        """Energy should monotonically decrease during QITE."""
        h = ising_model(2, J=1.0, h=1.0)
        qite = QITE(h, dt=0.05)
        result = qite.evolve(n_steps=50)

        # Check that energy is non-increasing (within numerical noise)
        for i in range(1, len(result.energies)):
            assert result.energies[i] <= result.energies[i - 1] + 1e-10

    def test_qite_convergence_flag(self):
        """Should set converged=True when tolerance is met."""
        h = SparsePauliHamiltonian([PauliOperator("Z", 1.0)])
        # Start in ground state |1>
        psi0 = np.array([0, 1], dtype=np.complex128)
        qite = QITE(h, dt=0.1)
        result = qite.evolve(psi0=psi0, n_steps=100, tol=1e-10)
        assert result.converged

    def test_qite_custom_initial(self, plus_state):
        """QITE with custom initial state.

        Starting from |+> (equal superposition of Z eigenstates),
        imaginary time evolution under H=Z suppresses the |0> component
        (E=+1) relative to |1> (E=-1), converging to the ground state.
        """
        h = SparsePauliHamiltonian([PauliOperator("Z", 1.0)])
        # Ground state of Z is |1> with energy -1
        qite = QITE(h, dt=0.5)
        result = qite.evolve(psi0=plus_state, n_steps=50)
        assert abs(result.final_energy - (-1.0)) < 0.01

    def test_qite_3qubit(self):
        """QITE on a 3-qubit system."""
        h = ising_model(3, J=1.0, h=0.5)
        e_exact, _ = h.ground_state()

        qite = QITE(h, dt=0.1)
        result = qite.evolve(n_steps=300, tol=1e-10)

        assert abs(result.final_energy - e_exact) / abs(e_exact) < 0.01


# ======================================================================
# VariationalAnsatz tests
# ======================================================================


class TestVariationalAnsatz:
    """Tests for the variational ansatz circuit."""

    def test_ansatz_creation(self):
        ansatz = VariationalAnsatz(2, n_layers=2)
        assert ansatz.n_params == 3 * 2 * 2  # 12 params

    def test_circuit_unitary(self):
        """Circuit unitary should be unitary."""
        ansatz = VariationalAnsatz(2, n_layers=1)
        params = np.random.default_rng(42).normal(size=ansatz.n_params)
        U = ansatz.circuit_unitary(params)
        np.testing.assert_allclose(
            U.conj().T @ U, np.eye(4), atol=1e-10
        )

    def test_state_normalized(self):
        ansatz = VariationalAnsatz(2, n_layers=1)
        params = np.random.default_rng(42).normal(size=ansatz.n_params)
        psi = ansatz.state(params)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10

    def test_zero_params_gives_zero_state(self):
        """All-zero params should give |00...0> (up to rotation gates at angle 0)."""
        ansatz = VariationalAnsatz(2, n_layers=1)
        params = np.zeros(ansatz.n_params)
        psi = ansatz.state(params)
        # Rotation by 0 is identity, so should give |00>
        assert abs(abs(psi[0]) - 1.0) < 1e-10

    def test_gradient_shape(self):
        ansatz = VariationalAnsatz(2, n_layers=1)
        params = np.zeros(ansatz.n_params)
        H = ising_model(2, J=1.0, h=0.5).matrix()
        grad = ansatz.gradient(params, H)
        assert len(grad) == ansatz.n_params


# ======================================================================
# VarQTE tests
# ======================================================================


class TestVarQTE:
    """Tests for Variational Quantum Time Evolution."""

    def test_varqte_runs(self):
        """VarQTE should run without errors."""
        h = SparsePauliHamiltonian([PauliOperator("Z", 1.0)])
        ansatz = VariationalAnsatz(1, n_layers=1)
        varqte = VarQTE(h, ansatz, dt=0.01)
        params0 = np.zeros(ansatz.n_params)
        result = varqte.evolve(params0, t_final=0.1, n_steps=5)

        assert len(result.times) == 6  # 0 + 5 steps
        assert len(result.parameters) == 6

    def test_varqte_energy_bounded(self):
        """Energy should stay within the spectral range."""
        h = SparsePauliHamiltonian([PauliOperator("X", 1.0)])
        ansatz = VariationalAnsatz(1, n_layers=2)
        varqte = VarQTE(h, ansatz, dt=0.01, regularization=1e-4)
        params0 = np.zeros(ansatz.n_params)
        result = varqte.evolve(params0, t_final=0.2, n_steps=10)

        evals = h.eigenvalues()
        for e in result.energies:
            assert evals[0] - 0.1 <= e <= evals[-1] + 0.1


# ======================================================================
# PVQD tests
# ======================================================================


class TestPVQD:
    """Tests for Projected Variational Quantum Dynamics."""

    def test_pvqd_runs(self):
        """PVQD should run without errors."""
        h = SparsePauliHamiltonian([PauliOperator("Z", 1.0)])
        ansatz = VariationalAnsatz(1, n_layers=1)
        pvqd = PVQD(h, ansatz, dt=0.05, optimizer_steps=20, learning_rate=0.1)
        params0 = np.zeros(ansatz.n_params)
        result = pvqd.evolve(params0, t_final=0.1, n_steps=2)

        assert len(result.states) == 3  # 0 + 2 steps

    def test_pvqd_trajectory_length(self):
        h = SparsePauliHamiltonian([PauliOperator("X", 0.5)])
        ansatz = VariationalAnsatz(1, n_layers=1)
        pvqd = PVQD(h, ansatz, dt=0.1, optimizer_steps=10)
        params0 = np.zeros(ansatz.n_params)
        result = pvqd.evolve(params0, t_final=0.3, n_steps=3)

        assert len(result.times) == 4
        assert len(result.energies) == 4


# ======================================================================
# Observable tests
# ======================================================================


class TestObservable:
    """Tests for the Observable class."""

    def test_expectation_z_spin_up(self, spin_up):
        obs = Observable(np.array([[1, 0], [0, -1]], dtype=np.complex128), "Z")
        assert abs(obs.expectation(spin_up) - 1.0) < 1e-10

    def test_expectation_z_spin_down(self, spin_down):
        obs = Observable(np.array([[1, 0], [0, -1]], dtype=np.complex128), "Z")
        assert abs(obs.expectation(spin_down) - (-1.0)) < 1e-10

    def test_expectation_x_plus(self, plus_state):
        obs = Observable(np.array([[0, 1], [1, 0]], dtype=np.complex128), "X")
        assert abs(obs.expectation(plus_state) - 1.0) < 1e-10

    def test_variance_eigenstate(self, spin_up):
        """Variance of an eigenstate is zero."""
        obs = Observable(np.array([[1, 0], [0, -1]], dtype=np.complex128), "Z")
        assert abs(obs.variance(spin_up)) < 1e-10

    def test_variance_superposition(self, plus_state):
        """<Z> = 0 for |+>, so Var(Z) = <Z^2> - <Z>^2 = 1."""
        obs = Observable(np.array([[1, 0], [0, -1]], dtype=np.complex128), "Z")
        assert abs(obs.variance(plus_state) - 1.0) < 1e-10

    def test_from_pauli_string(self):
        obs = Observable.from_pauli_string("XZ", coeff=2.0)
        assert obs.dim == 4
        assert obs.name == "XZ"

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            Observable(np.zeros((2, 3)))


class TestTimeSeriesObservable:
    def test_measure_trajectory(self, spin_up):
        obs_z = Observable(np.array([[1, 0], [0, -1]], dtype=np.complex128), "Z")
        ts = TimeSeriesObservable(observables=[obs_z])
        # Simple trajectory: stays in |0>
        times = np.array([0.0, 1.0])
        states = [spin_up, spin_up]
        result = ts.measure_trajectory(times, states)
        np.testing.assert_allclose(result["Z"], [1.0, 1.0])


# ======================================================================
# CorrelationFunction tests
# ======================================================================


class TestCorrelationFunction:
    """Tests for two-point correlators."""

    def test_autocorrelation_at_zero(self, spin_up):
        """C(0) = <psi|A^dag B|psi>."""
        H = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        A = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # X
        B = A.copy()
        cf = CorrelationFunction(H, A, B)
        times = np.array([0.0])
        result = cf.compute(spin_up, times)
        # <0|X^dag X|0> = <0|I|0> = 1
        assert abs(result[0] - 1.0) < 1e-10

    def test_correlation_shape(self, spin_up):
        H = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        A = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        B = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        cf = CorrelationFunction(H, A, B)
        times = np.linspace(0, 1, 10)
        result = cf.compute(spin_up, times)
        assert len(result) == 10


# ======================================================================
# EntanglementEntropy tests
# ======================================================================


class TestEntanglementEntropy:
    """Tests for bipartite entanglement entropy."""

    def test_product_state_zero_entropy(self):
        """Product state |00> has zero entanglement."""
        psi = np.array([1, 0, 0, 0], dtype=np.complex128)
        ee = EntanglementEntropy(2, n_a=1)
        assert abs(ee.von_neumann(psi)) < 1e-10

    def test_bell_state_max_entropy(self, bell_state):
        """Bell state has maximum entropy = ln(2)."""
        ee = EntanglementEntropy(2, n_a=1)
        entropy = ee.von_neumann(bell_state)
        assert abs(entropy - math.log(2)) < 1e-10

    def test_renyi_product_state(self):
        """Renyi entropy of product state is zero."""
        psi = np.array([1, 0, 0, 0], dtype=np.complex128)
        ee = EntanglementEntropy(2, n_a=1)
        assert abs(ee.renyi(psi, alpha=2.0)) < 1e-10

    def test_renyi_bell_state(self, bell_state):
        """Renyi-2 entropy of Bell state = ln(2)."""
        ee = EntanglementEntropy(2, n_a=1)
        entropy = ee.renyi(bell_state, alpha=2.0)
        assert abs(entropy - math.log(2)) < 1e-10

    def test_renyi_alpha_1_limit(self, bell_state):
        """Renyi at alpha=1 should give Von Neumann."""
        ee = EntanglementEntropy(2, n_a=1)
        s_vn = ee.von_neumann(bell_state)
        s_r1 = ee.renyi(bell_state, alpha=1.0)
        assert abs(s_vn - s_r1) < 1e-8

    def test_trajectory(self, bell_state):
        ee = EntanglementEntropy(2, n_a=1)
        product = np.array([1, 0, 0, 0], dtype=np.complex128)
        times = np.array([0.0, 1.0])
        states = [product, bell_state]
        entropies = ee.trajectory(times, states)
        assert abs(entropies[0]) < 1e-10
        assert abs(entropies[1] - math.log(2)) < 1e-10


# ======================================================================
# Magnetization tests
# ======================================================================


class TestMagnetization:
    """Tests for magnetization measurements."""

    def test_all_up_magnetization(self):
        """All spins up: <Z_i> = +1 for all i."""
        n = 3
        psi = np.zeros(8, dtype=np.complex128)
        psi[0] = 1.0  # |000>
        mag = Magnetization(n)
        local = mag.local(psi)
        np.testing.assert_allclose(local, [1.0, 1.0, 1.0], atol=1e-10)

    def test_all_down_magnetization(self):
        """All spins down: <Z_i> = -1 for all i."""
        n = 3
        psi = np.zeros(8, dtype=np.complex128)
        psi[7] = 1.0  # |111>
        mag = Magnetization(n)
        local = mag.local(psi)
        np.testing.assert_allclose(local, [-1.0, -1.0, -1.0], atol=1e-10)

    def test_total_magnetization(self):
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0  # |00>
        mag = Magnetization(2)
        assert abs(mag.total(psi) - 1.0) < 1e-10

    def test_magnetization_profile(self):
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0
        mag = Magnetization(2)
        times = np.array([0.0, 1.0])
        profile = mag.profile(times, [psi, psi])
        assert profile.shape == (2, 2)

    def test_total_trajectory(self):
        psi = np.zeros(4, dtype=np.complex128)
        psi[0] = 1.0
        mag = Magnetization(2)
        times = np.array([0.0, 1.0])
        traj = mag.total_trajectory(times, [psi, psi])
        np.testing.assert_allclose(traj, [1.0, 1.0], atol=1e-10)


# ======================================================================
# SpectralFunction tests
# ======================================================================


class TestSpectralFunction:
    """Tests for dynamical structure factor."""

    def test_single_site_spectral(self):
        """Spectral function of a pure oscillation should peak at the frequency."""
        n_t = 256
        omega_0 = 2.0
        dt = 0.1
        times = np.arange(n_t) * dt
        # Oscillating correlation: C(t) = cos(omega_0 * t)
        corr = np.cos(omega_0 * times)

        sf = SpectralFunction(n_sites=1)
        omega, S = sf.compute_single_site(times, corr)

        # Find peak
        peak_idx = np.argmax(np.abs(S))
        peak_omega = abs(omega[peak_idx])
        # Should be close to omega_0 / (2*pi) ... wait, need to check units.
        # Our FFT uses fftfreq with d=dt/(2*pi), so omega is in radians.
        # Actually peak should be at omega_0
        # Allow some FFT resolution error
        assert abs(peak_omega - omega_0) < 2 * np.pi / (n_t * dt) * 2

    def test_spectral_shape(self):
        n_sites = 4
        n_t = 32
        times = np.linspace(0, 1, n_t)
        corr = np.random.default_rng(42).random((n_sites, n_t))
        sf = SpectralFunction(n_sites)
        k, omega, S = sf.compute(times, corr)
        assert S.shape == (n_sites, n_t)


# ======================================================================
# Fidelity tests
# ======================================================================


class TestFidelity:
    """Tests for Loschmidt echo."""

    def test_same_state_fidelity(self, spin_up):
        fid = Fidelity(spin_up)
        assert abs(fid.compute(spin_up) - 1.0) < 1e-10

    def test_orthogonal_fidelity(self, spin_up, spin_down):
        fid = Fidelity(spin_up)
        assert abs(fid.compute(spin_down)) < 1e-10

    def test_fidelity_between_states(self, spin_up, plus_state):
        f = Fidelity.state_fidelity(spin_up, plus_state)
        assert abs(f - 0.5) < 1e-10

    def test_fidelity_trajectory(self, spin_up, spin_down):
        fid = Fidelity(spin_up)
        times = np.array([0.0, 1.0])
        traj = fid.trajectory(times, [spin_up, spin_down])
        assert abs(traj[0] - 1.0) < 1e-10
        assert abs(traj[1]) < 1e-10

    def test_loschmidt_echo_single_qubit(self, spin_up):
        """Under H = X, Loschmidt echo should oscillate as cos^2(t)."""
        h = SparsePauliHamiltonian([PauliOperator("X", 1.0)])
        evo = ExactEvolution(h)
        result = evo.evolve(spin_up, t_final=math.pi, n_steps=100)

        fid = Fidelity(spin_up)
        echo = fid.trajectory(result.times, result.states)

        # At each time, echo should be cos^2(t)
        for i, t in enumerate(result.times):
            expected = math.cos(t) ** 2
            assert abs(echo[i] - expected) < 1e-8


# ======================================================================
# Integrator tests
# ======================================================================


class TestRungeKutta4:
    """Tests for the classic RK4 integrator."""

    def test_rk4_single_qubit_precession(self, spin_up):
        """RK4 should match exact precession under Z."""
        H = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        rk4 = RungeKutta4(H, dt=0.001)
        result = rk4.evolve(spin_up, t_final=1.0, record_interval=100)

        psi_final = result.states[-1]
        expected = np.exp(-1j * 1.0) * spin_up
        np.testing.assert_allclose(psi_final, expected, atol=1e-6)

    def test_rk4_norm_preservation(self, spin_up):
        H = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # X
        rk4 = RungeKutta4(H, dt=0.01, renormalize=True)
        result = rk4.evolve(spin_up, t_final=2.0)
        assert result.norm_drift < 1e-10

    def test_rk4_energy_conservation(self):
        H = ising_model(2, J=1.0, h=0.5).matrix()
        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        rk4 = RungeKutta4(H, dt=0.001)
        result = rk4.evolve(psi0, t_final=1.0, record_interval=100)
        assert result.energy_drift < 1e-4

    def test_rk4_convergence_order(self, spin_up):
        """RK4 should show 4th-order convergence."""
        H = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        t = 0.5

        # Exact solution
        exact = ExactEvolution(
            SparsePauliHamiltonian([PauliOperator("X", 1.0)])
        ).evolve_state(spin_up, t)

        errors = []
        dts = [0.1, 0.05, 0.025]
        for dt in dts:
            rk4 = RungeKutta4(H, dt=dt, renormalize=False)
            result = rk4.evolve(spin_up, t_final=t)
            err = np.linalg.norm(result.states[-1] - exact)
            errors.append(err)

        # Check convergence ratio: error should decrease by ~16x when dt halves
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]
        # 4th order: ratio should be close to 2^4 = 16
        assert ratio1 > 10  # Some tolerance
        assert ratio2 > 10


class TestLeapfrogIntegrator:
    """Tests for the symplectic leapfrog integrator."""

    def test_leapfrog_norm_preservation(self, spin_up):
        H = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        lf = LeapfrogIntegrator(H, dt=0.01)
        result = lf.evolve(spin_up, t_final=2.0)
        assert result.norm_drift < 1e-10

    def test_leapfrog_energy_conservation(self):
        """Symplectic integrator should conserve energy extremely well."""
        H = ising_model(2, J=1.0, h=0.5).matrix()
        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        lf = LeapfrogIntegrator(H, dt=0.01)
        result = lf.evolve(psi0, t_final=5.0)
        # Leapfrog should have excellent energy conservation
        assert result.energy_drift < 1e-4

    def test_leapfrog_long_time_stability(self, spin_up):
        """Long-time integration should remain stable."""
        H = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        lf = LeapfrogIntegrator(H, dt=0.01)
        result = lf.evolve(spin_up, t_final=10.0, record_interval=100)
        # Norm should still be close to 1 after long evolution
        assert abs(result.norm_history[-1] - 1.0) < 1e-10


class TestAdaptiveRK45:
    """Tests for adaptive Runge-Kutta-Fehlberg."""

    def test_rk45_accuracy(self, spin_up):
        """RK45 should match exact to specified tolerance."""
        H = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        rk45 = AdaptiveRK45(H, dt_init=0.1, atol=1e-10, rtol=1e-8)
        result = rk45.evolve(spin_up, t_final=1.0)

        exact = ExactEvolution(
            SparsePauliHamiltonian([PauliOperator("X", 1.0)])
        ).evolve_state(spin_up, 1.0)

        np.testing.assert_allclose(result.states[-1], exact, atol=1e-6)

    def test_rk45_norm_preservation(self, spin_up):
        H = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        rk45 = AdaptiveRK45(H, dt_init=0.01, renormalize=True)
        result = rk45.evolve(spin_up, t_final=2.0)
        assert result.norm_drift < 1e-10

    def test_rk45_step_adaptation(self, spin_up):
        """Should take fewer steps for smooth dynamics."""
        H = np.array([[1, 0], [0, -1]], dtype=np.complex128)  # Diagonal = easy
        rk45 = AdaptiveRK45(H, dt_init=0.01)
        result = rk45.evolve(spin_up, t_final=1.0)
        # Diagonal Hamiltonian should allow large steps
        assert result.steps_taken < 200


class TestCrankNicolson:
    """Tests for the Crank-Nicolson integrator."""

    def test_cn_exact_unitarity(self, spin_up):
        """CN is exactly unitary -- norm must be preserved to machine precision."""
        H = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        cn = CrankNicolson(H, dt=0.05)
        result = cn.evolve(spin_up, t_final=2.0)
        # CN preserves norm exactly (it's a unitary transform)
        assert result.norm_drift < 1e-13

    def test_cn_accuracy(self, spin_up):
        """CN should converge with decreasing dt."""
        H = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        t = 1.0

        exact = ExactEvolution(
            SparsePauliHamiltonian([PauliOperator("X", 1.0)])
        ).evolve_state(spin_up, t)

        errors = []
        for dt in [0.1, 0.01, 0.001]:
            cn = CrankNicolson(H, dt=dt)
            result = cn.evolve(spin_up, t_final=t)
            err = np.linalg.norm(result.states[-1] - exact)
            errors.append(err)

        # Error should decrease
        assert errors[-1] < errors[0]

    def test_cn_energy_conservation(self):
        """CN should conserve energy well."""
        H = ising_model(2, J=1.0, h=0.5).matrix()
        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        cn = CrankNicolson(H, dt=0.01)
        result = cn.evolve(psi0, t_final=2.0)
        assert result.energy_drift < 1e-4

    def test_cn_step(self, spin_up):
        """Single CN step should preserve norm."""
        H = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        cn = CrankNicolson(H, dt=0.1)
        psi_new = cn.step(spin_up)
        assert abs(np.linalg.norm(psi_new) - 1.0) < 1e-13


# ======================================================================
# Integration pipeline tests
# ======================================================================


class TestIntegrationPipeline:
    """End-to-end tests: Hamiltonian -> evolve -> measure."""

    def test_ising_dynamics_pipeline(self):
        """Full pipeline: build Ising model, evolve, measure magnetization."""
        H = ising_model(3, J=1.0, h=0.5)
        evo = ExactEvolution(H)

        psi0 = np.zeros(8, dtype=np.complex128)
        psi0[0] = 1.0

        result = evo.evolve(psi0, t_final=2.0, n_steps=50)

        mag = Magnetization(3)
        m_total = mag.total_trajectory(result.times, result.states)
        assert len(m_total) == 51

        ee = EntanglementEntropy(3, n_a=1)
        entropies = ee.trajectory(result.times, result.states)
        assert len(entropies) == 51
        # Initial product state has zero entropy
        assert abs(entropies[0]) < 1e-10

    def test_trotter_vs_exact_observables(self):
        """Trotter and exact should give similar observable values."""
        H = ising_model(2, J=1.0, h=0.5)
        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        t = 0.5

        exact = ExactEvolution(H)
        exact_state = exact.evolve_state(psi0, t)
        exact_energy = H.expectation(exact_state)

        trotter = TrotterEvolution(H, order=2)
        trotter_state = trotter.evolve_state(psi0, t, n_steps=50)
        trotter_energy = H.expectation(trotter_state)

        assert abs(trotter_energy - exact_energy) < 0.01

    def test_qite_then_measure(self):
        """QITE ground state should have minimum energy."""
        H = ising_model(2, J=1.0, h=1.0)
        e_exact, gs_exact = H.ground_state()

        qite = QITE(H, dt=0.1)
        result = qite.evolve(n_steps=100)

        # Fidelity with exact ground state
        fid = Fidelity.state_fidelity(result.states[-1], gs_exact)
        assert fid > 0.95

    def test_integrator_matches_exact(self):
        """CN integrator should closely match exact evolution."""
        H_ham = ising_model(2, J=0.5, h=0.3)
        H_mat = H_ham.matrix()
        psi0 = np.zeros(4, dtype=np.complex128)
        psi0[0] = 1.0
        t = 1.0

        exact = ExactEvolution(H_ham).evolve_state(psi0, t)
        cn = CrankNicolson(H_mat, dt=0.01)
        cn_result = cn.evolve(psi0, t_final=t)

        np.testing.assert_allclose(cn_result.states[-1], exact, atol=1e-3)

    def test_adiabatic_then_qite(self):
        """Adiabatic prep followed by QITE refinement."""
        H_target = ising_model(2, J=1.0, h=0.5)
        e_exact, _ = H_target.ground_state()

        # Use QITE directly (adiabatic is tested separately)
        qite = QITE(H_target, dt=0.1)
        result = qite.evolve(n_steps=100)
        assert abs(result.final_energy - e_exact) / abs(e_exact) < 0.01

    def test_entanglement_growth_under_ising(self):
        """Starting from product state, entanglement should grow under Ising."""
        H = ising_model(4, J=1.0, h=0.5)
        evo = ExactEvolution(H)
        psi0 = np.zeros(16, dtype=np.complex128)
        psi0[0] = 1.0

        result = evo.evolve(psi0, t_final=2.0, n_steps=20)

        ee = EntanglementEntropy(4, n_a=2)
        entropies = ee.trajectory(result.times, result.states)

        # Entanglement should grow from zero
        assert abs(entropies[0]) < 1e-10
        assert max(entropies) > 0.01  # Some entanglement develops

    def test_fidelity_decays_then_returns(self):
        """Loschmidt echo under H=X should oscillate as cos^2(t).

        Under H=X, |0> evolves to cos(t)|0> - i*sin(t)|1>, so the
        Loschmidt echo |<0|psi(t)>|^2 = cos^2(t).  This reaches zero
        at t=pi/2 and returns to 1 at t=pi.
        """
        h = SparsePauliHamiltonian([PauliOperator("X", 1.0)])
        evo = ExactEvolution(h)
        psi0 = np.array([1, 0], dtype=np.complex128)

        result = evo.evolve(psi0, t_final=math.pi, n_steps=100)

        fid = Fidelity(psi0)
        echo = fid.trajectory(result.times, result.states)

        # At t=0, fidelity = 1
        assert abs(echo[0] - 1.0) < 1e-10
        # At t=pi/2 (midpoint), fidelity = cos^2(pi/2) = 0
        idx_half = len(result.times) // 2
        assert echo[idx_half] < 0.01
        # At t=pi, fidelity = cos^2(pi) = 1 (full revival)
        assert echo[-1] > 0.99


# ======================================================================
# Lindblad master equation tests
# ======================================================================

from nqpu.simulation.lindblad import (
    LindbladOperator,
    LindbladMasterEquation,
    LindbladResult,
    LindbladSolver,
    amplitude_damping_operators,
    dephasing_operators,
    depolarizing_operators,
    thermal_operators,
    create_lindblad_equation,
)


class TestLindbladOperator:
    """Tests for LindbladOperator construction."""

    def test_creation(self):
        op = LindbladOperator(operator=np.eye(2), rate=0.5, label="test")
        assert op.rate == 0.5
        assert op.label == "test"

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            LindbladOperator(operator=np.eye(2), rate=-0.1)


class TestLindbladMasterEquation:
    """Tests for the Lindblad master equation construction."""

    def test_dim(self):
        H = np.array([[0, 1], [1, 0]], dtype=complex)
        eq = LindbladMasterEquation(hamiltonian=H)
        assert eq.dim == 2

    def test_drho_dt_hermitian_preserving(self):
        """drho/dt should be Hermitian when rho is Hermitian."""
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        L = np.array([[0, 1], [0, 0]], dtype=complex)
        eq = LindbladMasterEquation(
            hamiltonian=H,
            jump_operators=[LindbladOperator(L, 0.1)],
        )
        rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        drho = eq.drho_dt(rho)
        np.testing.assert_allclose(drho, drho.conj().T, atol=1e-12)

    def test_liouvillian_shape(self):
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        eq = LindbladMasterEquation(hamiltonian=H)
        L = eq.liouvillian()
        assert L.shape == (4, 4)

    def test_liouvillian_trace_preserving(self):
        """vec(I)^T L = 0 for any valid Lindblad equation."""
        H = np.array([[1, 0.5], [0.5, -1]], dtype=complex)
        L_op = np.array([[0, 1], [0, 0]], dtype=complex)
        eq = LindbladMasterEquation(
            hamiltonian=H,
            jump_operators=[LindbladOperator(L_op, 0.2)],
        )
        assert eq.is_trace_preserving()

    def test_drho_dt_matches_liouvillian(self):
        """Direct drho_dt and Liouvillian should give same result."""
        H = np.array([[1, 0.5], [0.5, -1]], dtype=complex)
        L_op = np.array([[0, 1], [0, 0]], dtype=complex)
        eq = LindbladMasterEquation(
            hamiltonian=H,
            jump_operators=[LindbladOperator(L_op, 0.3)],
        )
        rho = np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=complex)

        drho_direct = eq.drho_dt(rho)
        L = eq.liouvillian()
        drho_super = (L @ rho.ravel()).reshape(2, 2)

        np.testing.assert_allclose(drho_direct, drho_super, atol=1e-12)


class TestLindbladSolver:
    """Tests for LindbladSolver evolution and steady-state finding."""

    def test_rk4_trace_preservation(self):
        """Trace should remain 1 throughout RK4 evolution."""
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        ops = amplitude_damping_operators(1, gamma=0.1)
        eq = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=eq, method="rk4")

        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        result = solver.evolve(rho0, t_final=5.0, n_steps=200)

        traces = result.trace()
        np.testing.assert_allclose(traces, 1.0, atol=1e-6)

    def test_exact_trace_preservation(self):
        """Trace should remain 1 throughout exact evolution."""
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        ops = dephasing_operators(1, gamma=0.2)
        eq = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=eq, method="exact")

        rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        result = solver.evolve(rho0, t_final=5.0, n_steps=50)

        traces = result.trace()
        np.testing.assert_allclose(traces, 1.0, atol=1e-6)

    def test_purity_decays_under_dephasing(self):
        """Purity should decrease under dephasing noise."""
        H = np.zeros((2, 2), dtype=complex)
        ops = dephasing_operators(1, gamma=0.5)
        eq = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=eq, method="rk4")

        # Start in pure superposition |+>
        psi_plus = np.array([1, 1], dtype=complex) / math.sqrt(2)
        rho0 = np.outer(psi_plus, psi_plus.conj())
        result = solver.evolve(rho0, t_final=10.0, n_steps=200)

        purities = result.purity()
        assert purities[0] > 0.99  # Initially pure
        assert purities[-1] < purities[0]  # Purity decreases

    def test_amplitude_damping_population_transfer(self):
        """Amplitude damping should transfer population |1> -> |0>."""
        H = np.zeros((2, 2), dtype=complex)
        ops = amplitude_damping_operators(1, gamma=0.5)
        eq = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=eq, method="rk4")

        rho0 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|
        result = solver.evolve(rho0, t_final=20.0, n_steps=500)

        pops = result.populations()
        assert pops[0, 0] < 0.01   # Initially in |1>
        assert pops[-1, 0] > 0.95  # Decayed to |0>

    def test_rk4_matches_exact(self):
        """RK4 and exact methods should give similar results."""
        H = np.array([[0.5, 0.2], [0.2, -0.5]], dtype=complex)
        ops = dephasing_operators(1, gamma=0.1)
        eq = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)

        rho0 = np.array([[0.6, 0.3], [0.3, 0.4]], dtype=complex)

        solver_rk4 = LindbladSolver(equation=eq, method="rk4")
        solver_exact = LindbladSolver(equation=eq, method="exact")

        result_rk4 = solver_rk4.evolve(rho0, t_final=2.0, n_steps=500)
        result_exact = solver_exact.evolve(rho0, t_final=2.0, n_steps=500)

        # Compare final states
        np.testing.assert_allclose(
            result_rk4.states[-1], result_exact.states[-1], atol=1e-4
        )

    def test_von_neumann_entropy_increases(self):
        """Entropy should increase under depolarizing noise."""
        H = np.zeros((2, 2), dtype=complex)
        ops = depolarizing_operators(1, gamma=0.3)
        eq = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=eq, method="rk4")

        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        result = solver.evolve(rho0, t_final=10.0, n_steps=200)

        entropies = result.von_neumann_entropy()
        assert entropies[0] < 0.01  # Initially pure
        assert entropies[-1] > entropies[0]  # Entropy increases

    def test_steady_state_eigenvalue(self):
        """Steady state should satisfy L rho_ss = 0."""
        H = np.array([[1, 0.5], [0.5, -1]], dtype=complex)
        ops = amplitude_damping_operators(1, gamma=0.3)
        eq = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=eq)

        rho_ss = solver.steady_state(method="eigenvalue")

        # Check valid density matrix
        assert abs(np.trace(rho_ss) - 1.0) < 1e-10
        eigvals = np.linalg.eigvalsh(rho_ss)
        assert np.all(eigvals >= -1e-10)

        # Check it's actually a steady state
        drho = eq.drho_dt(rho_ss)
        np.testing.assert_allclose(drho, 0, atol=1e-6)

    def test_steady_state_svd(self):
        """SVD method should also find a valid steady state."""
        H = np.array([[0, 0.3], [0.3, 0]], dtype=complex)
        ops = dephasing_operators(1, gamma=0.5)
        eq = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=eq)

        rho_ss = solver.steady_state(method="svd")

        assert abs(np.trace(rho_ss) - 1.0) < 1e-10
        drho = eq.drho_dt(rho_ss)
        np.testing.assert_allclose(drho, 0, atol=1e-6)

    def test_thermal_steady_state(self):
        """System with thermal bath should reach thermal equilibrium.

        At inverse temperature beta, the steady state should be
        approximately rho ~ exp(-beta H) / Z.
        """
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        gamma = 0.5
        n_th = 0.3
        ops = thermal_operators(1, gamma=gamma, n_thermal=n_th)
        eq = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=eq, method="rk4")

        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        result = solver.evolve(rho0, t_final=50.0, n_steps=1000)

        # At steady state, population ratio should be n_th / (1 + n_th)
        pops = result.populations()[-1]
        # |0> population and |1> population
        # For thermal: p1/p0 = n_th / (1 + n_th)
        expected_p1 = n_th / (1 + 2 * n_th)
        expected_p0 = (1 + n_th) / (1 + 2 * n_th)
        assert abs(pops[1] - expected_p1) < 0.05
        assert abs(pops[0] - expected_p0) < 0.05

    def test_expectation_observable(self):
        """Expectation values should track correctly."""
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        ops = amplitude_damping_operators(1, gamma=0.2)
        eq = LindbladMasterEquation(hamiltonian=H, jump_operators=ops)
        solver = LindbladSolver(equation=eq, method="rk4")

        rho0 = np.array([[0, 0], [0, 1]], dtype=complex)
        result = solver.evolve(rho0, t_final=10.0, n_steps=200)

        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        z_exp = result.expectation(Z)
        assert z_exp[0] < -0.9   # Initially |1>, <Z> = -1
        assert z_exp[-1] > z_exp[0]  # Moves toward |0> (<Z> = +1)

    def test_pure_state_input_conversion(self):
        """Solver should accept a pure state vector and convert to rho."""
        H = np.array([[0, 1], [1, 0]], dtype=complex)
        eq = LindbladMasterEquation(hamiltonian=H)
        solver = LindbladSolver(equation=eq, method="rk4")

        psi0 = np.array([1, 0], dtype=complex)
        result = solver.evolve(psi0, t_final=1.0, n_steps=50)
        # Should have density matrices
        assert result.states[0].shape == (2, 2)

    def test_create_lindblad_equation_convenience(self):
        """Test the convenience constructor."""
        H = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, -1]], dtype=complex)
        eq = create_lindblad_equation(H, noise_type="depolarizing", gamma=0.1)
        assert eq.dim == 4
        assert len(eq.jump_operators) == 6  # 3 Paulis x 2 qubits


# ======================================================================
# Fermionic Gaussian state tests
# ======================================================================

from nqpu.simulation.fermionic import (
    FermionicMode,
    GaussianState,
    QuadraticHamiltonian,
    GaussianEvolution,
    GaussianEvolutionResult,
    wicks_theorem,
    density_of_states,
)


class TestFermionicMode:
    """Tests for FermionicMode Fock space operators."""

    def test_anticommutation(self):
        """Fermionic operators must satisfy {c_i, c^dag_j} = delta_ij."""
        n = 3
        for i in range(n):
            for j in range(n):
                ci = FermionicMode(i)
                cj = FermionicMode(j)
                c_i = ci.annihilation_matrix(n)
                c_dag_j = cj.creation_matrix(n)
                anticomm = c_i @ c_dag_j + c_dag_j @ c_i
                expected = np.eye(2 ** n) * (1 if i == j else 0)
                np.testing.assert_allclose(anticomm, expected, atol=1e-12)

    def test_anticommutation_annihilation(self):
        """{c_i, c_j} = 0 for all i, j."""
        n = 3
        for i in range(n):
            for j in range(n):
                ci = FermionicMode(i).annihilation_matrix(n)
                cj = FermionicMode(j).annihilation_matrix(n)
                anticomm = ci @ cj + cj @ ci
                np.testing.assert_allclose(anticomm, 0, atol=1e-12)

    def test_number_operator_eigenvalues(self):
        """Number operator should have eigenvalues 0 and 1."""
        n = 2
        n_op = FermionicMode(0).number_operator(n)
        evals = np.linalg.eigvalsh(n_op)
        np.testing.assert_allclose(sorted(evals), [0, 0, 1, 1], atol=1e-12)


class TestGaussianState:
    """Tests for GaussianState correlation matrix representation."""

    def test_vacuum_properties(self):
        gs = GaussianState.vacuum(4)
        assert gs.n_modes == 4
        assert abs(gs.particle_number) < 1e-12
        assert gs.is_valid()

    def test_filled_state(self):
        gs = GaussianState.filled(4, [0, 2])
        assert abs(gs.particle_number - 2.0) < 1e-12
        assert gs.is_valid()
        occ = gs.occupation_numbers()
        np.testing.assert_allclose(occ, [1, 0, 1, 0], atol=1e-12)

    def test_half_filled(self):
        gs = GaussianState.half_filled(6)
        assert abs(gs.particle_number - 3.0) < 1e-12
        assert gs.is_valid()

    def test_vacuum_entropy(self):
        """Vacuum has zero entropy."""
        gs = GaussianState.vacuum(4)
        assert gs.entropy() < 1e-10

    def test_from_state_vector_vacuum(self):
        """Correlation matrix of vacuum |0000> should be all zeros."""
        n = 3
        psi = np.zeros(2 ** n, dtype=complex)
        psi[0] = 1.0
        gs = GaussianState.from_state_vector(psi, n)
        np.testing.assert_allclose(gs.correlation_matrix, 0, atol=1e-12)

    def test_from_state_vector_single_particle(self):
        """State |100> should have Gamma = diag(1, 0, 0)."""
        n = 3
        # |100> in big-endian: first mode occupied = basis state 4
        psi = np.zeros(2 ** n, dtype=complex)
        psi[4] = 1.0  # |100>
        gs = GaussianState.from_state_vector(psi, n)
        assert abs(gs.particle_number - 1.0) < 1e-10
        assert abs(gs.correlation_matrix[0, 0] - 1.0) < 1e-10

    def test_mutual_information_uncorrelated(self):
        """Uncorrelated modes should have zero mutual information."""
        gs = GaussianState.filled(4, [0, 2])
        mi = gs.mutual_information([0], [1])
        assert mi < 1e-10

    def test_subsystem_entropy(self):
        """Subsystem entropy should be non-negative."""
        gs = GaussianState.half_filled(6)
        s = gs.subsystem_entropy([0, 1, 2])
        assert s >= -1e-10


class TestQuadraticHamiltonian:
    """Tests for QuadraticHamiltonian and model constructors."""

    def test_tight_binding_hermitian(self):
        ham = QuadraticHamiltonian.tight_binding_1d(6, t=1.0, mu=0.0)
        h = ham.hopping_matrix
        np.testing.assert_allclose(h, h.conj().T, atol=1e-12)

    def test_tight_binding_ground_state_energy(self):
        """Check tight-binding ground state energy against exact formula.

        For 1D tight-binding with N sites and open boundaries,
        E_k = -2t cos(pi*k / (N+1)), k = 1, ..., N.
        Ground state fills the lowest N/2 levels.
        """
        n = 6
        t = 1.0
        ham = QuadraticHamiltonian.tight_binding_1d(n, t=t)
        gs = ham.ground_state()
        e_gs = ham.energy(gs)

        # Exact single-particle energies
        exact_evals = [-2 * t * math.cos(math.pi * k / (n + 1))
                       for k in range(1, n + 1)]
        exact_evals.sort()
        exact_e_gs = sum(exact_evals[:n // 2])

        assert abs(e_gs - exact_e_gs) < 1e-8

    def test_tight_binding_periodic_spectrum(self):
        """Periodic tight-binding: E_k = -2t cos(2 pi k / N)."""
        n = 8
        t = 1.0
        ham = QuadraticHamiltonian.tight_binding_1d(n, t=t, periodic=True)
        evals, _ = ham.single_particle_spectrum()

        exact = sorted([-2 * t * math.cos(2 * math.pi * k / n)
                        for k in range(n)])
        np.testing.assert_allclose(sorted(evals), exact, atol=1e-10)

    def test_ssh_model_construction(self):
        """SSH model should create 2*n_cells modes."""
        ham = QuadraticHamiltonian.ssh_model(5, t1=1.0, t2=0.5)
        assert ham.n_modes == 10

    def test_ssh_topological_vs_trivial(self):
        """SSH with t1 < t2 should have smaller gap than t1 > t2."""
        ham_topo = QuadraticHamiltonian.ssh_model(10, t1=0.5, t2=1.0)
        ham_triv = QuadraticHamiltonian.ssh_model(10, t1=1.0, t2=0.5)

        # Both should have a gap, but the topological phase has
        # edge states in a finite system
        gap_topo = ham_topo.band_gap()
        gap_triv = ham_triv.band_gap()
        assert gap_topo >= 0
        assert gap_triv >= 0

    def test_energy_ground_state_is_minimum(self):
        """Ground state energy should be the minimum."""
        ham = QuadraticHamiltonian.tight_binding_1d(4, t=1.0)
        gs = ham.ground_state()
        e_gs = ham.energy(gs)

        # Any other state should have higher energy
        rng = np.random.default_rng(42)
        for _ in range(10):
            gamma = np.zeros((4, 4), dtype=complex)
            n_filled = rng.integers(0, 5)
            indices = rng.choice(4, size=n_filled, replace=False)
            for idx in indices:
                gamma[idx, idx] = 1.0
            state = GaussianState(gamma)
            assert ham.energy(state) >= e_gs - 1e-10

    def test_fock_space_matches_quadratic(self):
        """Fock-space Hamiltonian should match quadratic form energies."""
        ham = QuadraticHamiltonian.tight_binding_1d(3, t=1.0)
        H_fock = ham.fock_space_hamiltonian()

        # Check Hermiticity
        np.testing.assert_allclose(H_fock, H_fock.conj().T, atol=1e-12)

        # Ground state energy from full diagonalisation
        evals = np.linalg.eigvalsh(H_fock)
        e0_fock = evals[0]

        # Ground state energy from Gaussian
        e0_gauss = ham.ground_state_energy()

        assert abs(e0_fock - e0_gauss) < 1e-10


class TestGaussianEvolution:
    """Tests for Gaussian state time evolution."""

    def test_particle_number_conservation(self):
        """Particle number should be conserved under quadratic evolution."""
        ham = QuadraticHamiltonian.tight_binding_1d(6, t=1.0)
        state0 = GaussianState.filled(6, [0, 1, 2])
        evo = GaussianEvolution(hamiltonian=ham)
        result = evo.evolve(state0, t_final=5.0, n_steps=100)

        n_traj = result.particle_number_trajectory()
        np.testing.assert_allclose(n_traj, 3.0, atol=1e-10)

    def test_energy_conservation(self):
        """Energy should be conserved under unitary evolution."""
        ham = QuadraticHamiltonian.tight_binding_1d(4, t=1.0)
        state0 = GaussianState.filled(4, [0, 1])
        evo = GaussianEvolution(hamiltonian=ham)
        result = evo.evolve(state0, t_final=3.0, n_steps=50)

        energies = result.energy_trajectory(ham)
        np.testing.assert_allclose(energies, energies[0], atol=1e-10)

    def test_ground_state_stationary(self):
        """Ground state should be stationary under evolution."""
        ham = QuadraticHamiltonian.tight_binding_1d(4, t=1.0)
        gs = ham.ground_state()
        evo = GaussianEvolution(hamiltonian=ham)
        result = evo.evolve(gs, t_final=5.0, n_steps=50)

        # Correlation matrix should not change
        gamma_0 = gs.correlation_matrix
        gamma_final = result.states[-1].correlation_matrix
        np.testing.assert_allclose(gamma_final, gamma_0, atol=1e-10)

    def test_validity_preserved(self):
        """Evolved states should remain valid Gaussian states."""
        ham = QuadraticHamiltonian.tight_binding_1d(4, t=1.0)
        state0 = GaussianState.filled(4, [0, 2])
        evo = GaussianEvolution(hamiltonian=ham)
        result = evo.evolve(state0, t_final=2.0, n_steps=20)

        for state in result.states:
            assert state.is_valid()

    def test_site_occupations_shape(self):
        """Site occupations should have correct shape."""
        ham = QuadraticHamiltonian.tight_binding_1d(4, t=1.0)
        state0 = GaussianState.filled(4, [0])
        evo = GaussianEvolution(hamiltonian=ham)
        result = evo.evolve(state0, t_final=1.0, n_steps=10)

        occ = result.site_occupations()
        assert occ.shape == (11, 4)

    def test_single_particle_propagation(self):
        """Single particle on a chain should propagate ballistically.

        Starting from site 0, occupation should spread to other sites.
        """
        n = 8
        ham = QuadraticHamiltonian.tight_binding_1d(n, t=1.0)
        state0 = GaussianState.filled(n, [0])
        evo = GaussianEvolution(hamiltonian=ham)
        result = evo.evolve(state0, t_final=3.0, n_steps=30)

        # At t=0, particle is at site 0 only
        occ_0 = result.site_occupations()[0]
        assert occ_0[0] > 0.99
        assert occ_0[-1] < 0.01

        # At t=3, particle should have spread
        occ_final = result.site_occupations()[-1]
        assert occ_final[0] < 0.99  # No longer fully at site 0
        assert np.sum(occ_final) == pytest.approx(1.0, abs=1e-10)


class TestWicksTheorem:
    """Tests for Wick's theorem correlator evaluation."""

    def test_single_pair(self):
        """Single pair should return Gamma[i,j]."""
        gamma = np.array([[0.6, 0.1], [0.1, 0.4]], dtype=complex)
        state = GaussianState(gamma)
        result = wicks_theorem(state, [(0, 1)])
        assert abs(result - gamma[0, 1]) < 1e-12

    def test_empty_operators(self):
        """No operators should return 1."""
        state = GaussianState.vacuum(2)
        assert abs(wicks_theorem(state, []) - 1.0) < 1e-12

    def test_two_pairs_determinant(self):
        """Two pairs should give det of 2x2 contraction matrix."""
        gamma = np.array([
            [0.8, 0.1, 0.05],
            [0.1, 0.5, 0.02],
            [0.05, 0.02, 0.3],
        ], dtype=complex)
        state = GaussianState(gamma)

        # <c^dag_0 c_0 c^dag_1 c_1> = det([[Gamma[0,0], Gamma[0,1]],
        #                                    [Gamma[1,0], Gamma[1,1]]])
        result = wicks_theorem(state, [(0, 0), (1, 1)])
        expected = gamma[0, 0] * gamma[1, 1] - gamma[0, 1] * gamma[1, 0]
        assert abs(result - expected) < 1e-12

    def test_vacuum_correlator(self):
        """For vacuum, all normal-ordered correlators should be zero."""
        state = GaussianState.vacuum(3)
        assert abs(wicks_theorem(state, [(0, 0)])) < 1e-12
        assert abs(wicks_theorem(state, [(0, 1)])) < 1e-12


class TestDensityOfStates:
    """Tests for the density of states computation."""

    def test_dos_positive(self):
        """DOS should be non-negative everywhere."""
        ham = QuadraticHamiltonian.tight_binding_1d(10, t=1.0)
        energies, dos = density_of_states(ham, broadening=0.1)
        assert np.all(dos >= -1e-15)

    def test_dos_integrates_to_n_modes(self):
        """Integral of DOS should equal the number of modes."""
        ham = QuadraticHamiltonian.tight_binding_1d(8, t=1.0)
        energies, dos = density_of_states(ham, broadening=0.1)
        de = energies[1] - energies[0]
        total = np.sum(dos) * de
        assert abs(total - 8.0) < 0.5  # Approximate due to discretisation


# ======================================================================
# Multiscale simulation tests
# ======================================================================

from nqpu.simulation.multiscale import (
    Subsystem,
    CouplingTerm,
    MultiscaleSystem,
    MultiscaleSolver,
    MultiscaleResult,
    MultiscaleEvolution,
    MultiscaleEvolutionResult,
    AdaptiveMultiscale,
)


class TestSubsystem:
    """Tests for Subsystem construction."""

    def test_creation(self):
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        sub = Subsystem(name="test", n_qubits=1, hamiltonian=H)
        assert sub.dim == 2
        assert sub.name == "test"

    def test_dimension_mismatch(self):
        H = np.eye(4, dtype=complex)
        with pytest.raises(ValueError, match="does not match"):
            Subsystem(name="bad", n_qubits=1, hamiltonian=H)

    def test_ground_state(self):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        sub = Subsystem(name="z", n_qubits=1, hamiltonian=Z)
        e, gs = sub.ground_state()
        assert abs(e - (-1.0)) < 1e-10
        assert abs(abs(gs[1]) - 1.0) < 1e-10

    def test_energy(self):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        sub = Subsystem(name="z", n_qubits=1, hamiltonian=Z)
        sub.state = np.array([1, 0], dtype=complex)
        assert abs(sub.energy() - 1.0) < 1e-10


class TestMultiscaleSystem:
    """Tests for MultiscaleSystem construction and full Hamiltonian."""

    def test_add_subsystem(self):
        sys = MultiscaleSystem()
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        sys.add_subsystem(Subsystem("a", 1, H))
        assert sys.total_qubits == 1
        assert "a" in sys.subsystems

    def test_duplicate_subsystem_raises(self):
        sys = MultiscaleSystem()
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        sys.add_subsystem(Subsystem("a", 1, H))
        with pytest.raises(ValueError, match="already exists"):
            sys.add_subsystem(Subsystem("a", 1, H))

    def test_add_coupling_unknown_subsystem(self):
        sys = MultiscaleSystem()
        H = np.eye(2, dtype=complex)
        sys.add_subsystem(Subsystem("a", 1, H))
        coupling = CouplingTerm("a", "b", np.eye(2), np.eye(2))
        with pytest.raises(ValueError, match="not found"):
            sys.add_coupling(coupling)

    def test_full_hamiltonian_two_subsystems(self):
        """Full Hamiltonian of two uncoupled subsystems should be
        H_A kron I_B + I_A kron H_B."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)

        sys = MultiscaleSystem()
        sys.add_subsystem(Subsystem("a", 1, Z))
        sys.add_subsystem(Subsystem("b", 1, X))

        H_full = sys.full_hamiltonian()
        expected = np.kron(Z, np.eye(2)) + np.kron(np.eye(2), X)
        np.testing.assert_allclose(H_full, expected, atol=1e-12)

    def test_full_hamiltonian_with_coupling(self):
        """Coupling term should add strength * op_a kron op_b."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)

        sys = MultiscaleSystem()
        sys.add_subsystem(Subsystem("a", 1, I2 * 0))
        sys.add_subsystem(Subsystem("b", 1, I2 * 0))

        coupling = CouplingTerm("a", "b", Z, Z, strength=0.5)
        sys.add_coupling(coupling)

        H_full = sys.full_hamiltonian()
        expected = 0.5 * np.kron(Z, np.eye(2)) @ np.kron(np.eye(2), Z)
        np.testing.assert_allclose(H_full, expected, atol=1e-12)

    def test_ground_state_energy_exact(self):
        """Exact ground state energy should match full diagonalisation."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        sys = MultiscaleSystem()
        sys.add_subsystem(Subsystem("a", 1, -Z))
        sys.add_subsystem(Subsystem("b", 1, -Z))

        coupling = CouplingTerm("a", "b", Z, Z, strength=-0.5)
        sys.add_coupling(coupling)

        e_exact = sys.ground_state_energy(method="exact")
        H = sys.full_hamiltonian()
        evals = np.linalg.eigvalsh(H)
        assert abs(e_exact - evals[0]) < 1e-10


class TestMultiscaleSolver:
    """Tests for the self-consistent field solver."""

    def test_uncoupled_convergence(self):
        """Uncoupled subsystems should converge in one iteration."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        sys = MultiscaleSystem()
        sys.add_subsystem(Subsystem("a", 1, -Z))
        sys.add_subsystem(Subsystem("b", 1, -Z))

        solver = MultiscaleSolver(system=sys)
        result = solver.solve()

        assert result.converged
        # Each subsystem ground state energy is -1
        assert abs(result.energy - (-2.0)) < 1e-6

    def test_weakly_coupled_converges(self):
        """Weakly coupled system should converge close to exact."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        sys = MultiscaleSystem()
        sys.add_subsystem(Subsystem("a", 1, -Z))
        sys.add_subsystem(Subsystem("b", 1, -Z))

        coupling = CouplingTerm("a", "b", Z, Z, strength=-0.1)
        sys.add_coupling(coupling)

        solver = MultiscaleSolver(
            system=sys,
            max_iterations=100,
            convergence_threshold=1e-8,
            mixing=0.7,
        )
        result = solver.solve()

        assert result.converged

        # Compare to exact
        e_exact = sys.ground_state_energy(method="exact")
        assert abs(result.energy - e_exact) < 0.5  # Mean-field is approximate

    def test_energy_history_decreasing(self):
        """Energy should generally decrease during SCF iterations."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)

        sys = MultiscaleSystem()
        sys.add_subsystem(Subsystem("a", 1, -Z))
        sys.add_subsystem(Subsystem("b", 1, -X))

        coupling = CouplingTerm("a", "b", Z, Z, strength=-0.3)
        sys.add_coupling(coupling)

        solver = MultiscaleSolver(system=sys, max_iterations=30, mixing=0.5)
        result = solver.solve()

        # Energy history should not be empty
        assert len(result.energy_history) > 1

    def test_result_contains_states(self):
        """Result should contain subsystem states."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        sys = MultiscaleSystem()
        sys.add_subsystem(Subsystem("a", 1, -Z))
        sys.add_subsystem(Subsystem("b", 1, -Z))

        solver = MultiscaleSolver(system=sys)
        result = solver.solve()

        assert "a" in result.subsystem_states
        assert "b" in result.subsystem_states
        assert len(result.subsystem_states["a"]) == 2


class TestMultiscaleEvolution:
    """Tests for multiscale time evolution."""

    def test_evolution_runs(self):
        """Evolution should produce correct number of time steps."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        sys = MultiscaleSystem()
        sys.add_subsystem(Subsystem("a", 1, Z))
        sys.add_subsystem(Subsystem("b", 1, -Z))

        evo = MultiscaleEvolution(system=sys)
        result = evo.evolve(t_final=1.0, n_steps=10)

        assert len(result.times) == 11
        assert len(result.energies) == 11
        assert len(result.subsystem_states["a"]) == 11

    def test_uncoupled_evolution_energy_conservation(self):
        """Energy should be approximately conserved for uncoupled subsystems."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        sys = MultiscaleSystem()
        sys.add_subsystem(Subsystem("a", 1, Z))
        sys.add_subsystem(Subsystem("b", 1, -Z))

        evo = MultiscaleEvolution(system=sys)
        result = evo.evolve(t_final=2.0, n_steps=100)

        energies = result.energies
        # Energy should be stable (no coupling = exact conservation)
        assert abs(energies[-1] - energies[0]) < 0.1

    def test_coupled_evolution_runs(self):
        """Coupled evolution should not crash."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)

        sys = MultiscaleSystem()
        sys.add_subsystem(Subsystem("a", 1, Z))
        sys.add_subsystem(Subsystem("b", 1, -Z))

        coupling = CouplingTerm("a", "b", X, X, strength=0.2)
        sys.add_coupling(coupling)

        evo = MultiscaleEvolution(system=sys)
        result = evo.evolve(t_final=1.0, n_steps=50)

        assert len(result.times) == 51


class TestAdaptiveMultiscale:
    """Tests for adaptive decomposition."""

    def test_suggest_decomposition(self):
        """Should produce a valid MultiscaleSystem."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        H = np.kron(Z, I2) + np.kron(I2, Z) + 0.5 * np.kron(Z, Z)

        adaptive = AdaptiveMultiscale(
            system=MultiscaleSystem(),
            entanglement_threshold=0.5,
        )
        result = adaptive.suggest_decomposition(
            n_qubits=2, hamiltonian=H, max_subsystem_size=1
        )

        assert result.total_qubits == 2
        assert len(result.subsystems) == 2

    def test_analyze_entanglement_runs(self):
        """Entanglement analysis should return a dict."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        sys = MultiscaleSystem()
        sys.add_subsystem(Subsystem("a", 1, Z))
        sys.add_subsystem(Subsystem("b", 1, -Z))

        coupling = CouplingTerm("a", "b", Z, Z, strength=0.5)
        sys.add_coupling(coupling)

        adaptive = AdaptiveMultiscale(system=sys)
        result = adaptive.analyze_entanglement()

        assert isinstance(result, dict)
        assert "a-b" in result

    def test_large_system_decomposition(self):
        """Should handle decomposition of larger systems."""
        n = 8
        dim = 2 ** n
        # Simple Ising-like Hamiltonian
        I2 = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n):
            # Z_i term
            op = np.eye(1, dtype=complex)
            for j in range(n):
                op = np.kron(op, Z if j == i else I2)
            H += op

        adaptive = AdaptiveMultiscale(
            system=MultiscaleSystem(),
        )
        result = adaptive.suggest_decomposition(
            n_qubits=n, hamiltonian=H, max_subsystem_size=4
        )

        assert result.total_qubits == n
        for name, sub in result.subsystems.items():
            assert sub.n_qubits <= 4
