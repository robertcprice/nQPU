"""Comprehensive tests for the nqpu.chem package.

Tests cover: Atom/Molecule geometry, BasisSet construction,
integral computation, FermionicHamiltonian/QubitHamiltonian,
Jordan-Wigner mapping, ansatze (UCCSD, UCCD, HardwareEfficient,
HartreeFockState), and the MolecularVQE driver.
"""

import math

import numpy as np
import pytest

from nqpu.chem import (
    ANGSTROM_TO_BOHR,
    Atom,
    BasisSet,
    ChemMolecule,
    FermionicHamiltonian,
    FermionicTerm,
    HardwareEfficient,
    HartreeFockState,
    MolecularVQE,
    PauliString,
    PrimitiveGaussian,
    QubitHamiltonian,
    UCCSD,
    UCCD,
    VQEResult,
    beh2,
    boys_function,
    h2,
    h2o,
    h4_chain,
    h6_ring,
    jordan_wigner,
    lih,
    overlap_integral,
)


# ====================================================================
# Fixtures
# ====================================================================


@pytest.fixture
def h2_molecule():
    return h2()


@pytest.fixture
def h2_basis(h2_molecule):
    return BasisSet("sto-3g", h2_molecule)


# ====================================================================
# Atom tests
# ====================================================================


class TestAtom:
    def test_atom_creation(self):
        a = Atom("H", (0.0, 0.0, 0.0))
        assert a.symbol == "H"
        assert a.atomic_number == 1

    def test_position_bohr_conversion(self):
        a = Atom("H", (1.0, 0.0, 0.0))
        pb = a.position_bohr
        assert pb[0] == pytest.approx(ANGSTROM_TO_BOHR)

    def test_nuclear_charge_equals_atomic_number(self):
        li = Atom("Li", (0.0, 0.0, 0.0))
        assert li.nuclear_charge == pytest.approx(3.0)

    def test_unsupported_element_raises(self):
        a = Atom("Unobtainium", (0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="Unsupported"):
            _ = a.atomic_number


# ====================================================================
# Molecule tests
# ====================================================================


class TestMolecule:
    def test_h2_num_atoms(self, h2_molecule):
        assert h2_molecule.num_atoms == 2

    def test_h2_num_electrons(self, h2_molecule):
        assert h2_molecule.num_electrons == 2

    def test_h2_formula(self, h2_molecule):
        assert h2_molecule.formula == "H2"

    def test_nuclear_repulsion_positive(self, h2_molecule):
        e_nuc = h2_molecule.nuclear_repulsion_energy()
        assert e_nuc > 0

    def test_from_xyz_parsing(self):
        xyz = """2
H2 molecule
H 0.0 0.0 -0.37
H 0.0 0.0  0.37
"""
        mol = ChemMolecule.from_xyz(xyz)
        assert mol.num_atoms == 2

    def test_empty_molecule_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ChemMolecule.from_atoms([])

    def test_detect_symmetry_h2(self, h2_molecule):
        sym = h2_molecule.detect_symmetry()
        assert sym == "Dinfh"

    @pytest.mark.parametrize("factory,expected_n", [
        (h2, 2),
        (lih, 2),
        (h2o, 3),
        (beh2, 3),
    ])
    def test_predefined_molecule_atom_count(self, factory, expected_n):
        mol = factory()
        assert mol.num_atoms == expected_n


# ====================================================================
# BasisSet tests
# ====================================================================


class TestBasisSet:
    def test_sto3g_h2_has_two_functions(self, h2_basis):
        assert h2_basis.num_functions == 2

    def test_unsupported_basis_raises(self, h2_molecule):
        with pytest.raises(ValueError, match="Unsupported"):
            BasisSet("cc-pvdz", h2_molecule)

    def test_basis_function_labels(self, h2_basis):
        labels = [bf.label for bf in h2_basis.functions]
        assert all("H" in lbl for lbl in labels)

    def test_num_orbitals_equals_num_functions(self, h2_basis):
        assert h2_basis.num_orbitals() == h2_basis.num_functions


# ====================================================================
# PrimitiveGaussian tests
# ====================================================================


class TestPrimitiveGaussian:
    def test_normalization_positive(self):
        pg = PrimitiveGaussian(exponent=1.0, coefficient=1.0)
        assert pg.normalization > 0

    def test_normalization_scales_with_exponent(self):
        pg1 = PrimitiveGaussian(exponent=1.0, coefficient=1.0)
        pg2 = PrimitiveGaussian(exponent=4.0, coefficient=1.0)
        assert pg2.normalization > pg1.normalization


# ====================================================================
# Integral tests
# ====================================================================


class TestIntegrals:
    def test_boys_function_at_zero(self):
        # F_0(0) = 1/(2*0+1) = 1
        assert boys_function(0, 0.0) == pytest.approx(1.0)

    def test_boys_function_order_1_at_zero(self):
        # F_1(0) = 1/3
        assert boys_function(1, 0.0) == pytest.approx(1.0 / 3.0)

    def test_boys_function_large_x_positive(self):
        val = boys_function(0, 30.0)
        assert val > 0

    def test_overlap_self_positive(self, h2_basis):
        bf = h2_basis.functions[0]
        s = overlap_integral(bf, bf)
        assert s > 0


# ====================================================================
# PauliString and QubitHamiltonian tests
# ====================================================================


class TestPauliString:
    def test_identity(self):
        ps = PauliString.identity()
        assert ps.qubits == []
        assert repr(ps) == "I"

    def test_from_dict(self):
        ps = PauliString.from_dict({0: "X", 2: "Z"})
        assert set(ps.qubits) == {0, 2}

    def test_max_qubit(self):
        ps = PauliString.from_dict({0: "X", 3: "Y"})
        assert ps.max_qubit == 3


class TestQubitHamiltonian:
    def test_add_term_and_num_terms(self):
        qh = QubitHamiltonian()
        qh.add_term(1.0, PauliString.from_dict({0: "Z"}))
        qh.add_term(0.5, PauliString.from_dict({1: "X"}))
        assert qh.num_terms() == 2

    def test_simplify_combines_like_terms(self):
        qh = QubitHamiltonian()
        ps = PauliString.from_dict({0: "Z"})
        qh.add_term(1.0, ps)
        qh.add_term(2.0, ps)
        simplified = qh.simplify()
        assert simplified.num_terms() == 1

    def test_to_matrix_hermitian(self):
        qh = QubitHamiltonian()
        qh.add_term(1.0, PauliString.from_dict({0: "Z"}))
        mat = qh.to_matrix(1)
        assert np.allclose(mat, mat.conj().T)

    def test_expectation_identity(self):
        qh = QubitHamiltonian()
        qh.add_term(3.0, PauliString.identity())
        state = np.array([1.0, 0.0], dtype=np.complex128)
        assert qh.expectation(state) == pytest.approx(3.0)


# ====================================================================
# Fermionic Hamiltonian and Jordan-Wigner tests
# ====================================================================


class TestFermionicHamiltonian:
    def test_from_integrals_creates_terms(self):
        h1 = np.array([[1.0, 0.5], [0.5, 1.0]])
        h2 = np.zeros((2, 2, 2, 2))
        h2[0, 0, 0, 0] = 0.6
        fh = FermionicHamiltonian.from_integrals(h1, h2, nuclear_repulsion=0.5)
        assert fh.nuclear_repulsion == 0.5
        assert len(fh.terms) > 0
        assert fh.n_spin_orbitals == 4

    def test_jordan_wigner_produces_qubit_hamiltonian(self):
        h1 = np.array([[1.0, 0.1], [0.1, 1.2]])
        h2 = np.zeros((2, 2, 2, 2))
        fh = FermionicHamiltonian.from_integrals(h1, h2, nuclear_repulsion=0.3)
        qh = jordan_wigner(fh)
        assert isinstance(qh, QubitHamiltonian)
        assert qh.num_terms() > 0


# ====================================================================
# Ansatz tests
# ====================================================================


class TestHartreeFockState:
    def test_hf_state_norm(self):
        hf = HartreeFockState(n_electrons=2, n_spin_orbitals=4)
        sv = hf.state_vector()
        assert np.linalg.norm(sv) == pytest.approx(1.0)

    def test_hf_state_occupation_string(self):
        hf = HartreeFockState(n_electrons=2, n_spin_orbitals=4)
        occ = hf.occupation_string
        assert occ.startswith("|1100")


class TestUCCSD:
    def test_uccsd_num_parameters_h2(self):
        uccsd = UCCSD(n_electrons=2, n_spin_orbitals=4)
        # H2 in STO-3G: should have singles + doubles
        assert uccsd.num_parameters() > 0

    def test_uccsd_state_vector_normalized(self):
        uccsd = UCCSD(n_electrons=2, n_spin_orbitals=4)
        params = np.zeros(uccsd.num_parameters())
        sv = uccsd.state_vector(params)
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)

    def test_uccsd_wrong_params_raises(self):
        uccsd = UCCSD(n_electrons=2, n_spin_orbitals=4)
        with pytest.raises(ValueError, match="parameters"):
            uccsd.state_vector(np.zeros(100))


class TestUCCD:
    def test_uccd_fewer_params_than_uccsd(self):
        uccsd = UCCSD(n_electrons=2, n_spin_orbitals=4)
        uccd = UCCD(n_electrons=2, n_spin_orbitals=4)
        assert uccd.num_parameters() <= uccsd.num_parameters()


class TestHardwareEfficient:
    def test_hw_eff_num_params(self):
        hw = HardwareEfficient(n_qubits=4, n_layers=2)
        assert hw.num_parameters() == 2 * 4 * 2

    def test_hw_eff_state_normalized(self):
        hw = HardwareEfficient(n_qubits=2, n_layers=1)
        params = np.zeros(hw.num_parameters())
        sv = hw.state_vector(params)
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)

    def test_hw_eff_invalid_entangler_raises(self):
        with pytest.raises(ValueError, match="entangler"):
            HardwareEfficient(n_qubits=2, entangler="swap")


# ====================================================================
# MolecularVQE tests
# ====================================================================


class TestMolecularVQE:
    def test_h2_vqe_creation(self):
        vqe = MolecularVQE(h2(), basis="sto-3g")
        assert vqe.num_qubits == 4
        assert vqe.num_parameters > 0

    def test_exact_energy_reasonable(self):
        vqe = MolecularVQE(h2(), basis="sto-3g")
        exact = vqe.exact_ground_state_energy()
        # H2 STO-3G exact is around -1.137 Hartree
        assert -1.5 < exact < -0.5

    def test_invalid_mapping_raises(self):
        with pytest.raises(ValueError, match="mapping"):
            MolecularVQE(h2(), mapping="bogus")

    def test_invalid_ansatz_raises(self):
        with pytest.raises(ValueError, match="ansatz"):
            MolecularVQE(h2(), ansatz="bogus")

    def test_vqe_compute_returns_result(self):
        vqe = MolecularVQE(h2(), basis="sto-3g")
        result = vqe.compute_ground_state(maxiter=5)
        assert isinstance(result, VQEResult)
        assert result.energy < 0
        assert result.num_function_evals > 0

    def test_vqe_result_chemical_accuracy(self):
        vqe = MolecularVQE(h2(), basis="sto-3g")
        result = vqe.compute_ground_state(maxiter=5)
        acc = result.chemical_accuracy()
        assert acc is not None
        assert acc >= 0
