"""Comprehensive tests for the nQPU quantum chemistry module.

Tests cover all five new modules of the chem package:

  - molecular.py: Atom, Molecule, BasisSet, predefined molecules
  - integrals.py: overlap, kinetic, nuclear, ERI, Boys function
  - fermion.py: FermionicHamiltonian, JW/BK/parity mappings,
                QubitHamiltonian
  - ansatz.py: HartreeFockState, UCCSD, UCCD, kUpCCGSD,
               HardwareEfficient
  - vqe_driver.py: MolecularVQE, PES scan, exact diag

Numerical reference values are taken from:
  - Szabo & Ostlund, *Modern Quantum Chemistry* (1996)
  - Known STO-3G benchmark results for H2, LiH
"""

import math

import numpy as np
import pytest

from nqpu.chem.molecular import (
    Atom,
    BasisSet,
    ANGSTROM_TO_BOHR,
    h2,
    h2o,
    h4_chain,
    h6_ring,
    lih,
    beh2,
)
from nqpu.chem.molecular import Molecule
from nqpu.chem.integrals import (
    boys_function,
    compute_one_electron_integrals,
    compute_two_electron_integrals,
    overlap_integral,
    kinetic_integral,
    nuclear_attraction_integral,
    electron_repulsion_integral,
)
from nqpu.chem.fermion import (
    FermionicHamiltonian,
    FermionicTerm,
    PauliString,
    QubitHamiltonian,
    jordan_wigner,
    bravyi_kitaev,
    parity_mapping,
)
from nqpu.chem.ansatz import (
    HartreeFockState,
    UCCSD,
    UCCD,
    kUpCCGSD,
    HardwareEfficient,
)
from nqpu.chem.vqe_driver import MolecularVQE


# ======================================================================
# Molecular geometry tests
# ======================================================================


class TestAtom:
    """Tests for the Atom class."""

    def test_atom_creation(self):
        a = Atom("H", (0.0, 0.0, 0.0))
        assert a.symbol == "H"
        assert a.position == (0.0, 0.0, 0.0)

    def test_atomic_number(self):
        assert Atom("H", (0, 0, 0)).atomic_number == 1
        assert Atom("Li", (0, 0, 0)).atomic_number == 3
        assert Atom("O", (0, 0, 0)).atomic_number == 8

    def test_nuclear_charge(self):
        assert Atom("H", (0, 0, 0)).nuclear_charge == 1.0
        assert Atom("C", (0, 0, 0)).nuclear_charge == 6.0

    def test_position_bohr(self):
        a = Atom("H", (1.0, 0.0, 0.0))
        px, py, pz = a.position_bohr
        assert abs(px - ANGSTROM_TO_BOHR) < 1e-10
        assert abs(py) < 1e-10
        assert abs(pz) < 1e-10

    def test_unsupported_element(self):
        a = Atom("Unobtanium", (0, 0, 0))
        with pytest.raises(ValueError, match="Unsupported element"):
            _ = a.atomic_number


class TestMolecule:
    """Tests for the Molecule class."""

    def test_from_atoms(self):
        mol = Molecule.from_atoms([
            Atom("H", (0, 0, 0)),
            Atom("H", (0, 0, 0.74)),
        ])
        assert mol.num_atoms == 2
        assert mol.num_electrons == 2
        assert mol.charge == 0
        assert mol.multiplicity == 1

    def test_empty_molecule_raises(self):
        with pytest.raises(ValueError, match="at least one atom"):
            Molecule.from_atoms([])

    def test_from_xyz(self):
        xyz = """2
H2 molecule
H  0.0  0.0  0.0
H  0.0  0.0  0.74
"""
        mol = Molecule.from_xyz(xyz)
        assert mol.num_atoms == 2
        assert mol.symbols == ["H", "H"]

    def test_from_xyz_no_header(self):
        xyz = """H  0.0  0.0  0.0
H  0.0  0.0  0.74
"""
        mol = Molecule.from_xyz(xyz)
        assert mol.num_atoms == 2

    def test_num_electrons_with_charge(self):
        mol = Molecule.from_atoms([Atom("Li", (0, 0, 0))], charge=1)
        assert mol.num_electrons == 2  # Li has 3, minus charge 1

    def test_formula(self):
        mol = h2o()
        assert mol.formula == "H2O"

    def test_symmetry_h2(self):
        mol = h2()
        assert mol.detect_symmetry() == "Dinfh"

    def test_symmetry_lih(self):
        mol = lih()
        assert mol.detect_symmetry() == "Cinfv"


class TestPredefinedMolecules:
    """Tests for predefined molecule factory functions."""

    def test_h2_electrons(self):
        mol = h2()
        assert mol.num_electrons == 2

    def test_h2_custom_bond(self):
        mol = h2(bond_length=1.0)
        d = np.linalg.norm(
            np.array(mol.atoms[1].position) - np.array(mol.atoms[0].position)
        )
        assert abs(d - 1.0) < 1e-10

    def test_lih_electrons(self):
        mol = lih()
        assert mol.num_electrons == 4

    def test_h2o_electrons(self):
        mol = h2o()
        assert mol.num_electrons == 10

    def test_beh2_electrons(self):
        mol = beh2()
        assert mol.num_electrons == 6

    def test_h4_chain_electrons(self):
        mol = h4_chain()
        assert mol.num_electrons == 4

    def test_h6_ring_electrons(self):
        mol = h6_ring()
        assert mol.num_electrons == 6

    def test_h6_ring_geometry(self):
        mol = h6_ring(radius=1.5)
        positions = np.array([a.position for a in mol.atoms])
        # All atoms should be at the same radius
        distances = np.linalg.norm(positions[:, :2], axis=1)
        np.testing.assert_allclose(distances, 1.5, atol=1e-10)


class TestNuclearRepulsion:
    """Tests for nuclear repulsion energy."""

    def test_h2_nuclear_repulsion(self):
        """H2 at 0.74 Angstrom: E_nuc ~ 0.7137 Hartree."""
        mol = h2(bond_length=0.74)
        e_nuc = mol.nuclear_repulsion_energy()
        # Known value: Z_A * Z_B / R_AB
        # R_AB = 0.74 * 1.8897 = 1.3984 Bohr
        # E_nuc = 1 / 1.3984 = 0.7151 (approx)
        assert e_nuc > 0.0
        assert abs(e_nuc - 0.7151) < 0.01

    def test_nuclear_repulsion_positive(self):
        mol = lih()
        assert mol.nuclear_repulsion_energy() > 0.0

    def test_nuclear_repulsion_increases_closer(self):
        e1 = h2(bond_length=1.0).nuclear_repulsion_energy()
        e2 = h2(bond_length=0.5).nuclear_repulsion_energy()
        assert e2 > e1


# ======================================================================
# Basis set tests
# ======================================================================


class TestBasisSet:
    """Tests for BasisSet construction."""

    def test_sto3g_h2(self):
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        assert basis.num_functions == 2  # one s-type per H

    def test_sto3g_lih(self):
        mol = lih()
        basis = BasisSet("sto-3g", mol)
        # Li: 1s + 2s = 2, H: 1s = 1 => total = 3
        assert basis.num_functions == 3

    def test_unsupported_basis(self):
        with pytest.raises(ValueError, match="Unsupported basis"):
            BasisSet("cc-pvtz", h2())

    def test_basis_functions_have_correct_centers(self):
        mol = h2(bond_length=0.74)
        basis = BasisSet("sto-3g", mol)
        # Each function should be centered on its atom
        assert basis.functions[0].atom_index == 0
        assert basis.functions[1].atom_index == 1


# ======================================================================
# Integral tests
# ======================================================================


class TestBoysFunction:
    """Tests for the Boys function implementation."""

    def test_boys_f0_at_zero(self):
        """F_0(0) = 1."""
        assert abs(boys_function(0, 0.0) - 1.0) < 1e-12

    def test_boys_f1_at_zero(self):
        """F_1(0) = 1/3."""
        assert abs(boys_function(1, 0.0) - 1.0 / 3.0) < 1e-12

    def test_boys_f0_small(self):
        """F_0(x) for small x ~ 1 - x/3 + x^2/10 - ..."""
        x = 0.1
        # Numerical integration reference via trapezoidal rule
        t = np.linspace(0, 1, 10000)
        dt = t[1] - t[0]
        integrand = np.exp(-x * t ** 2)
        ref = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dt)
        assert abs(boys_function(0, x) - ref) < 1e-4

    def test_boys_large_argument(self):
        """F_0(x) for large x ~ sqrt(pi/4x)."""
        x = 50.0
        expected = math.sqrt(math.pi / (4 * x))
        assert abs(boys_function(0, x) - expected) < 0.01 * expected


class TestOverlapIntegrals:
    """Tests for overlap integrals."""

    def test_self_overlap(self):
        """Self-overlap of a normalized basis function should be ~1."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        s_00 = overlap_integral(basis.functions[0], basis.functions[0])
        assert abs(s_00 - 1.0) < 0.01

    def test_h2_overlap_positive(self):
        """Overlap between H1 and H2 s-functions should be positive."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        s_01 = overlap_integral(basis.functions[0], basis.functions[1])
        assert s_01 > 0.0

    def test_overlap_symmetry(self):
        """S_ij = S_ji."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        s_01 = overlap_integral(basis.functions[0], basis.functions[1])
        s_10 = overlap_integral(basis.functions[1], basis.functions[0])
        assert abs(s_01 - s_10) < 1e-12

    def test_overlap_matrix_positive_definite(self):
        """Overlap matrix S must be positive definite."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        S, _, _ = compute_one_electron_integrals(mol, basis)
        eigenvalues = np.linalg.eigvalsh(S)
        assert np.all(eigenvalues > 0)


class TestKineticIntegrals:
    """Tests for kinetic energy integrals."""

    def test_kinetic_diagonal_positive(self):
        """Diagonal kinetic integrals must be positive."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        t_00 = kinetic_integral(basis.functions[0], basis.functions[0])
        assert t_00 > 0.0

    def test_kinetic_symmetry(self):
        """T_ij = T_ji."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        t_01 = kinetic_integral(basis.functions[0], basis.functions[1])
        t_10 = kinetic_integral(basis.functions[1], basis.functions[0])
        assert abs(t_01 - t_10) < 1e-12


class TestNuclearAttractionIntegrals:
    """Tests for nuclear attraction integrals."""

    def test_nuclear_attraction_negative(self):
        """Nuclear attraction integrals should be negative (attractive)."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        nuclei = [(a.position_bohr, a.nuclear_charge) for a in mol.atoms]
        v_00 = nuclear_attraction_integral(
            basis.functions[0], basis.functions[0], nuclei
        )
        assert v_00 < 0.0


class TestTwoElectronIntegrals:
    """Tests for electron repulsion integrals (ERI)."""

    def test_eri_positive(self):
        """ERI (aa|aa) should be positive (electron repulsion)."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        bf = basis.functions[0]
        eri_val = electron_repulsion_integral(bf, bf, bf, bf)
        assert eri_val > 0.0

    def test_eri_symmetry(self):
        """(pq|rs) = (qp|rs) = (rs|pq)."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        b0, b1 = basis.functions[0], basis.functions[1]
        v1 = electron_repulsion_integral(b0, b1, b0, b1)
        v2 = electron_repulsion_integral(b1, b0, b0, b1)
        v3 = electron_repulsion_integral(b0, b1, b1, b0)
        assert abs(v1 - v2) < 1e-10
        assert abs(v1 - v3) < 1e-10


class TestOneElectronIntegrals:
    """Tests for the full one-electron integral matrices."""

    def test_matrices_shape(self):
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        S, T, H_core = compute_one_electron_integrals(mol, basis)
        n = basis.num_functions
        assert S.shape == (n, n)
        assert T.shape == (n, n)
        assert H_core.shape == (n, n)

    def test_matrices_symmetric(self):
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        S, T, H_core = compute_one_electron_integrals(mol, basis)
        np.testing.assert_allclose(S, S.T, atol=1e-12)
        np.testing.assert_allclose(T, T.T, atol=1e-12)
        np.testing.assert_allclose(H_core, H_core.T, atol=1e-12)


class TestTwoElectronIntegralTensor:
    """Tests for the full two-electron tensor."""

    def test_tensor_shape(self):
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        eri = compute_two_electron_integrals(mol, basis)
        n = basis.num_functions
        assert eri.shape == (n, n, n, n)

    def test_tensor_8fold_symmetry(self):
        """(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq) etc."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        eri = compute_two_electron_integrals(mol, basis)
        n = basis.num_functions
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        val = eri[p, q, r, s]
                        assert abs(val - eri[q, p, r, s]) < 1e-10
                        assert abs(val - eri[p, q, s, r]) < 1e-10
                        assert abs(val - eri[r, s, p, q]) < 1e-10


# ======================================================================
# Fermionic Hamiltonian tests
# ======================================================================


class TestFermionicHamiltonian:
    """Tests for second-quantized Hamiltonian construction."""

    def test_from_integrals_creates_terms(self):
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        _, _, H_core = compute_one_electron_integrals(mol, basis)
        eri = compute_two_electron_integrals(mol, basis)
        nuc = mol.nuclear_repulsion_energy()

        ham = FermionicHamiltonian.from_integrals(H_core, eri, nuc)
        assert len(ham.terms) > 0
        assert ham.nuclear_repulsion == pytest.approx(nuc)
        assert ham.n_spin_orbitals == 4  # 2 spatial * 2 spin

    def test_spin_orbital_count(self):
        mol = lih()
        basis = BasisSet("sto-3g", mol)
        _, _, H_core = compute_one_electron_integrals(mol, basis)
        eri = compute_two_electron_integrals(mol, basis)

        ham = FermionicHamiltonian.from_integrals(H_core, eri)
        assert ham.n_spin_orbitals == 6  # 3 spatial orbitals * 2


class TestFermionAnticommutation:
    """Tests verifying fermion anticommutation via JW transformation."""

    def test_number_operator_jw(self):
        """Number operator n_0 = a+_0 a_0 should map to (I - Z_0) / 2."""
        ham = FermionicHamiltonian()
        ham.n_spin_orbitals = 2
        ham.terms = [FermionicTerm(
            coefficient=1.0, operators=[(0, True), (0, False)]
        )]

        qham = jordan_wigner(ham)
        mat = qham.to_matrix(2)

        # Expected: (I - Z) / 2 tensored with I
        # = diag(0, 0, 1, 1) for qubit ordering where qubit 0 is least significant
        expected_diag = [0, 1, 0, 1]
        actual_diag = np.real(np.diag(mat))
        np.testing.assert_allclose(actual_diag, expected_diag, atol=1e-10)

    def test_a_dagger_0_a_1_jw(self):
        """a+_0 a_1 should produce X and Y Pauli terms."""
        ham = FermionicHamiltonian()
        ham.n_spin_orbitals = 2
        ham.nuclear_repulsion = 0.0
        ham.terms = [FermionicTerm(
            coefficient=1.0, operators=[(0, True), (1, False)]
        )]

        qham = jordan_wigner(ham).simplify()

        # a+_0 a_1 in JW for adjacent qubits (0,1):
        # = 0.5 * (X0 X1 + Y0 Y1) (which is not Hermitian -- that's correct,
        # because a+_0 a_1 is not Hermitian by itself)
        mat = qham.to_matrix(2)

        # This operator maps |10> (orbital 1 occupied) to |01> (orbital 0 occupied)
        # In LSB convention: |10> has index 2 (qubit 1 = |1>), |01> has index 1 (qubit 0 = |1>)
        # So mat[1, 2] should be nonzero: <01| a+_0 a_1 |10> = 1
        assert abs(mat[1, 2]) > 0.1  # maps |10> -> |01>

        # Verify a+_0 a_1 + h.c. is Hermitian
        ham_hc = FermionicHamiltonian()
        ham_hc.n_spin_orbitals = 2
        ham_hc.nuclear_repulsion = 0.0
        ham_hc.terms = [
            FermionicTerm(coefficient=1.0, operators=[(0, True), (1, False)]),
            FermionicTerm(coefficient=1.0, operators=[(1, True), (0, False)]),
        ]
        qham_hc = jordan_wigner(ham_hc)
        mat_hc = qham_hc.to_matrix(2)
        np.testing.assert_allclose(mat_hc, mat_hc.conj().T, atol=1e-10)

    def test_anticommutation_relation(self):
        """Verify {a_i, a+_j} = delta_ij via JW matrices."""
        n = 2  # 2 spin orbitals

        # Build a_0 and a+_0 matrices
        dim = 2 ** n

        # a_0 in JW
        ham_a0 = FermionicHamiltonian()
        ham_a0.n_spin_orbitals = n
        ham_a0.terms = [FermionicTerm(coefficient=1.0, operators=[(0, False)])]
        a0_terms = jordan_wigner(ham_a0)
        a0_mat = a0_terms.to_matrix(n)

        # a+_0 in JW
        ham_ad0 = FermionicHamiltonian()
        ham_ad0.n_spin_orbitals = n
        ham_ad0.terms = [FermionicTerm(coefficient=1.0, operators=[(0, True)])]
        ad0_terms = jordan_wigner(ham_ad0)
        ad0_mat = ad0_terms.to_matrix(n)

        # {a_0, a+_0} = a_0 a+_0 + a+_0 a_0 = I
        anticommutator = a0_mat @ ad0_mat + ad0_mat @ a0_mat
        np.testing.assert_allclose(anticommutator, np.eye(dim), atol=1e-10)


# ======================================================================
# Qubit Hamiltonian tests
# ======================================================================


class TestQubitHamiltonian:
    """Tests for QubitHamiltonian."""

    def test_matrix_hermitian(self):
        """Qubit Hamiltonian matrix must be Hermitian."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        _, _, H_core = compute_one_electron_integrals(mol, basis)
        eri = compute_two_electron_integrals(mol, basis)
        nuc = mol.nuclear_repulsion_energy()

        fham = FermionicHamiltonian.from_integrals(H_core, eri, nuc)
        qham = jordan_wigner(fham)
        mat = qham.to_matrix()
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-10)

    def test_expectation_matches_matrix(self):
        """<psi|H|psi> via expectation() should match matrix multiplication."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        _, _, H_core = compute_one_electron_integrals(mol, basis)
        eri = compute_two_electron_integrals(mol, basis)
        nuc = mol.nuclear_repulsion_energy()

        fham = FermionicHamiltonian.from_integrals(H_core, eri, nuc)
        qham = jordan_wigner(fham)
        n = qham.num_qubits()
        mat = qham.to_matrix(n)

        # Random state
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
        psi /= np.linalg.norm(psi)

        exp_direct = float(np.real(psi.conj() @ mat @ psi))
        exp_method = qham.expectation(psi)
        assert abs(exp_direct - exp_method) < 1e-10

    def test_simplify_combines_terms(self):
        """Simplify should combine duplicate Pauli strings."""
        qham = QubitHamiltonian()
        ps = PauliString.from_dict({0: "Z"})
        qham.add_term(0.5, ps)
        qham.add_term(0.3, ps)
        simplified = qham.simplify()
        assert simplified.num_terms() == 1
        assert abs(simplified.terms[0][0] - 0.8) < 1e-12

    def test_addition(self):
        qh1 = QubitHamiltonian([(1.0, PauliString.from_dict({0: "X"}))])
        qh2 = QubitHamiltonian([(2.0, PauliString.from_dict({1: "Z"}))])
        qh3 = qh1 + qh2
        assert qh3.num_terms() == 2

    def test_scalar_multiplication(self):
        qh = QubitHamiltonian([(1.0, PauliString.from_dict({0: "X"}))])
        qh2 = 3.0 * qh
        assert abs(qh2.terms[0][0] - 3.0) < 1e-12


# ======================================================================
# Fermion-to-qubit mapping tests
# ======================================================================


class TestJordanWigner:
    """Tests for Jordan-Wigner transformation."""

    def test_h2_ground_state_energy(self):
        """H2 STO-3G exact ground state ~ -1.137 Hartree.

        The second-quantized Hamiltonian requires integrals in an
        orthonormal (MO) basis.  The MolecularVQE driver handles the
        RHF + MO transformation automatically.
        """
        vqe = MolecularVQE(h2(bond_length=0.74), basis="sto-3g")
        ground_energy = vqe.exact_ground_state_energy()

        # STO-3G H2 ground state energy is approximately -1.1373 Hartree
        assert abs(ground_energy - (-1.1373)) < 0.01


class TestBravyiKitaev:
    """Tests for Bravyi-Kitaev transformation."""

    def test_bk_preserves_eigenvalues(self):
        """BK eigenvalues should match JW eigenvalues."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        _, _, H_core = compute_one_electron_integrals(mol, basis)
        eri = compute_two_electron_integrals(mol, basis)
        nuc = mol.nuclear_repulsion_energy()

        fham = FermionicHamiltonian.from_integrals(H_core, eri, nuc)

        jw_ham = jordan_wigner(fham)
        bk_ham = bravyi_kitaev(fham)

        jw_eigs = sorted(np.linalg.eigvalsh(jw_ham.to_matrix()))
        bk_eigs = sorted(np.linalg.eigvalsh(bk_ham.to_matrix()))

        np.testing.assert_allclose(jw_eigs, bk_eigs, atol=1e-8)


class TestParityMapping:
    """Tests for parity mapping transformation."""

    def test_parity_preserves_eigenvalues(self):
        """Parity mapping eigenvalues should match JW eigenvalues."""
        mol = h2()
        basis = BasisSet("sto-3g", mol)
        _, _, H_core = compute_one_electron_integrals(mol, basis)
        eri = compute_two_electron_integrals(mol, basis)
        nuc = mol.nuclear_repulsion_energy()

        fham = FermionicHamiltonian.from_integrals(H_core, eri, nuc)

        jw_ham = jordan_wigner(fham)
        par_ham = parity_mapping(fham)

        jw_eigs = sorted(np.linalg.eigvalsh(jw_ham.to_matrix()))
        par_eigs = sorted(np.linalg.eigvalsh(par_ham.to_matrix()))

        np.testing.assert_allclose(jw_eigs, par_eigs, atol=1e-8)


# ======================================================================
# Ansatz tests
# ======================================================================


class TestHartreeFockState:
    """Tests for Hartree-Fock reference state preparation."""

    def test_h2_hf_state(self):
        """H2 HF state in JW should be |1100> (2 occupied, 2 virtual)."""
        hf = HartreeFockState(n_electrons=2, n_spin_orbitals=4)
        state = hf.state_vector()

        # In JW: qubits 0,1 are |1>, qubits 2,3 are |0>
        # Binary: 0b0011 = 3
        expected_idx = 3  # |11> on qubits 0,1
        assert abs(state[expected_idx] - 1.0) < 1e-12
        assert abs(np.linalg.norm(state) - 1.0) < 1e-12

    def test_occupation_string(self):
        hf = HartreeFockState(n_electrons=2, n_spin_orbitals=4)
        assert hf.occupation_string == "|1100>"

    def test_lih_hf_state(self):
        """LiH with 4 electrons, 6 spin orbitals."""
        hf = HartreeFockState(n_electrons=4, n_spin_orbitals=6)
        state = hf.state_vector()
        # Qubits 0,1,2,3 occupied = binary 0b001111 = 15
        assert abs(state[15] - 1.0) < 1e-12


class TestUCCSD:
    """Tests for UCCSD ansatz."""

    def test_h2_parameter_count(self):
        """H2 with 2 electrons, 4 spin orbitals: check excitation count."""
        uccsd = UCCSD(n_electrons=2, n_spin_orbitals=4)
        n_singles = len(uccsd.singles)
        n_doubles = len(uccsd.doubles)
        n_params = uccsd.num_parameters()

        # H2: 2 occupied (0a, 0b), 2 virtual (1a, 1b)
        # Singles (spin-conserving): 0a->1a, 0b->1b = 2
        # Doubles: (0a,0b) -> (1a,1b) = 1
        assert n_singles == 2
        assert n_doubles == 1
        assert n_params == 3

    def test_zero_params_gives_hf(self):
        """UCCSD with all-zero params should return HF state."""
        uccsd = UCCSD(n_electrons=2, n_spin_orbitals=4)
        params = np.zeros(uccsd.num_parameters())
        state = uccsd.state_vector(params)

        hf = HartreeFockState(n_electrons=2, n_spin_orbitals=4)
        hf_state = hf.state_vector()

        np.testing.assert_allclose(np.abs(state), np.abs(hf_state), atol=1e-10)

    def test_state_is_normalized(self):
        """UCCSD state should always be normalized."""
        uccsd = UCCSD(n_electrons=2, n_spin_orbitals=4)
        params = np.array([0.1, -0.2, 0.3])
        state = uccsd.state_vector(params)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10


class TestUCCD:
    """Tests for UCCD (doubles only) ansatz."""

    def test_h2_parameter_count(self):
        uccd = UCCD(n_electrons=2, n_spin_orbitals=4)
        assert uccd.num_parameters() == 1  # only doubles

    def test_state_normalized(self):
        uccd = UCCD(n_electrons=2, n_spin_orbitals=4)
        state = uccd.state_vector(np.array([0.5]))
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10


class TestHardwareEfficient:
    """Tests for hardware-efficient ansatz."""

    def test_parameter_count(self):
        he = HardwareEfficient(n_qubits=4, n_layers=2)
        assert he.num_parameters() == 2 * 4 * 2  # 2 angles * n_qubits * n_layers

    def test_state_normalized(self):
        he = HardwareEfficient(n_qubits=4, n_layers=1)
        rng = np.random.default_rng(42)
        params = rng.uniform(-np.pi, np.pi, he.num_parameters())
        state = he.state_vector(params)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_zero_params_gives_zero_state(self):
        """With all-zero params (no initial occupation), should give |0...0>."""
        he = HardwareEfficient(n_qubits=4, n_layers=1)
        params = np.zeros(he.num_parameters())
        state = he.state_vector(params)
        assert abs(state[0] - 1.0) < 1e-10


class TestkUpCCGSD:
    """Tests for k-UpCCGSD ansatz."""

    def test_has_more_params_than_uccsd(self):
        """Generalized excitations should give more parameters."""
        uccsd = UCCSD(n_electrons=2, n_spin_orbitals=4)
        kupcc = kUpCCGSD(n_spin_orbitals=4, k=1, n_electrons=2)
        # kUpCCGSD includes all pairs, not just occupied->virtual
        assert kupcc.num_parameters() >= uccsd.num_parameters()

    def test_state_normalized(self):
        kupcc = kUpCCGSD(n_spin_orbitals=4, k=1, n_electrons=2)
        params = np.zeros(kupcc.num_parameters())
        state = kupcc.state_vector(params)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10


# ======================================================================
# VQE driver tests
# ======================================================================


class TestMolecularVQE:
    """Tests for the end-to-end MolecularVQE driver."""

    def test_h2_exact_diag(self):
        """Exact diag of H2 STO-3G should give ~-1.137 Hartree."""
        vqe = MolecularVQE(h2(bond_length=0.74), basis="sto-3g")
        exact = vqe.exact_ground_state_energy()
        assert abs(exact - (-1.137)) < 0.02

    def test_h2_vqe_converges(self):
        """VQE on H2 should converge to within 10 mHa of exact."""
        mol = h2(bond_length=0.74)
        vqe = MolecularVQE(mol, basis="sto-3g", ansatz="uccsd")
        result = vqe.compute_ground_state(
            optimizer="cobyla", maxiter=300,
        )

        assert result.exact_energy is not None
        error_mha = abs(result.energy - result.exact_energy) * 1000
        # Allow 10 mHa tolerance (chemical accuracy is 1.6 mHa)
        assert error_mha < 10.0, f"VQE error {error_mha:.1f} mHa exceeds 10 mHa"

    def test_vqe_result_fields(self):
        vqe = MolecularVQE(h2(), basis="sto-3g")
        result = vqe.compute_ground_state(maxiter=10)
        assert hasattr(result, "energy")
        assert hasattr(result, "optimal_params")
        assert hasattr(result, "exact_energy")
        assert result.num_function_evals > 0

    def test_num_qubits(self):
        vqe = MolecularVQE(h2(), basis="sto-3g")
        assert vqe.num_qubits == 4  # 2 spatial * 2 spin

    def test_bond_length_affects_energy(self):
        """Different bond lengths should give different energies."""
        e1 = MolecularVQE(h2(0.5)).exact_ground_state_energy()
        e2 = MolecularVQE(h2(0.74)).exact_ground_state_energy()
        e3 = MolecularVQE(h2(2.0)).exact_ground_state_energy()
        # Equilibrium (0.74) should be lowest
        assert e2 < e1
        assert e2 < e3

    def test_pes_scan(self):
        """PES scan should return results for each bond length."""
        vqe = MolecularVQE(h2(), basis="sto-3g")
        results = vqe.potential_energy_surface(
            bond_lengths=[0.5, 0.74, 1.0],
            optimizer="cobyla",
            maxiter=50,
        )
        assert len(results) == 3
        for r in results:
            assert "bond_length" in r
            assert "vqe_energy" in r
            assert "exact_energy" in r


class TestActiveSpace:
    """Tests for active space reduction."""

    def test_lih_frozen_core_reduces_qubits(self):
        """Freezing Li 1s should reduce from 6 to 4 spin orbitals."""
        vqe = MolecularVQE(lih(), basis="sto-3g", frozen_core=1)
        # 3 spatial - 1 frozen = 2 active => 4 spin orbitals
        assert vqe.num_qubits == 4

    def test_frozen_core_preserves_approximate_energy(self):
        """Frozen-core energy should be within ~0.1 Ha of full-space."""
        mol = lih()
        vqe_full = MolecularVQE(mol, basis="sto-3g")
        vqe_frozen = MolecularVQE(mol, basis="sto-3g", frozen_core=1)

        e_full = vqe_full.exact_ground_state_energy()
        e_frozen = vqe_frozen.exact_ground_state_energy()

        # Frozen core is an approximation; should be within ~0.1 Ha
        assert abs(e_full - e_frozen) < 0.15


# ======================================================================
# Integration / smoke tests
# ======================================================================


class TestSmoke:
    """Quick smoke tests ensuring the pipeline does not crash."""

    def test_bk_mapping_pipeline(self):
        """Full pipeline with Bravyi-Kitaev mapping."""
        vqe = MolecularVQE(h2(), basis="sto-3g", mapping="bravyi_kitaev")
        exact = vqe.exact_ground_state_energy()
        assert exact < 0.0  # bound state

    def test_parity_mapping_pipeline(self):
        """Full pipeline with parity mapping."""
        vqe = MolecularVQE(h2(), basis="sto-3g", mapping="parity")
        exact = vqe.exact_ground_state_energy()
        assert exact < 0.0

    def test_uccd_ansatz(self):
        """UCCD ansatz on H2."""
        vqe = MolecularVQE(h2(), basis="sto-3g", ansatz="uccd")
        result = vqe.compute_ground_state(maxiter=50)
        assert result.energy < 0.0

    def test_hardware_efficient_ansatz(self):
        """Hardware-efficient ansatz on H2."""
        vqe = MolecularVQE(h2(), basis="sto-3g", ansatz="hardware_efficient")
        result = vqe.compute_ground_state(maxiter=50)
        assert result.energy is not None

    def test_repr_methods(self):
        """All __repr__ methods should not crash."""
        mol = h2()
        assert "H2" in repr(mol)

        basis = BasisSet("sto-3g", mol)
        assert "sto-3g" in repr(basis)

        vqe = MolecularVQE(mol)
        assert "MolecularVQE" in repr(vqe)
