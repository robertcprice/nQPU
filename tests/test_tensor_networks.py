"""Comprehensive tests for nqpu.tensor_networks -- tensors, MPS, MPO, DMRG,
and TEBD for one-dimensional quantum many-body systems.
"""
from __future__ import annotations

import numpy as np
import pytest

from nqpu.tensor_networks import (
    # Core tensors
    Tensor,
    TensorNetwork,
    contract_pair,
    # MPS
    MPS,
    ProductState,
    GHZState,
    RandomMPS,
    WState,
    # MPO
    MPO,
    IsingMPO,
    HeisenbergMPO,
    IdentityMPO,
    # DMRG
    DMRG,
    DMRGResult,
    dmrg_ground_state,
    # TEBD
    TEBD,
    TEBDResult,
    ImaginaryTEBD,
    NNHamiltonian,
    ising_nn_hamiltonian,
    heisenberg_nn_hamiltonian,
    tebd_evolve,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def identity_tensor():
    """2x2 identity as a named tensor."""
    return Tensor(np.eye(2, dtype=complex), legs=["i", "j"])


@pytest.fixture
def product_state_4():
    """4-site product state |0000>."""
    return ProductState(4)


@pytest.fixture
def ghz_4():
    """4-site GHZ state."""
    return GHZState(4)


@pytest.fixture
def ising_mpo_4():
    """4-site Ising MPO with J=1, h=1."""
    return IsingMPO(4, J=1.0, h=1.0)


# =====================================================================
# Tensor tests
# =====================================================================

class TestTensor:

    def test_creation(self, identity_tensor):
        assert identity_tensor.shape == (2, 2)
        assert identity_tensor.legs == ["i", "j"]
        assert identity_tensor.ndim == 2

    def test_mismatched_legs_raises(self):
        with pytest.raises(ValueError, match="Number of legs"):
            Tensor(np.eye(2), legs=["i", "j", "k"])

    def test_norm(self, identity_tensor):
        expected = np.linalg.norm(np.eye(2))
        assert identity_tensor.norm == pytest.approx(expected, abs=1e-12)

    def test_transpose(self, identity_tensor):
        t = identity_tensor.transpose(["j", "i"])
        assert t.legs == ["j", "i"]
        np.testing.assert_allclose(t.data, np.eye(2).T, atol=1e-12)

    def test_copy_independence(self, identity_tensor):
        copy = identity_tensor.copy()
        copy.data[0, 0] = 999.0
        assert identity_tensor.data[0, 0] != 999.0

    def test_scalar_multiply(self, identity_tensor):
        t2 = identity_tensor * 3.0
        np.testing.assert_allclose(t2.data, 3.0 * np.eye(2), atol=1e-12)

    def test_trace(self):
        t = Tensor(np.eye(3, dtype=complex), legs=["a", "b"])
        result = t.trace("a", "b")
        assert result.data == pytest.approx(3.0, abs=1e-12)

    def test_svd_roundtrip(self):
        data = np.random.default_rng(42).standard_normal((3, 4)) + 0j
        t = Tensor(data, legs=["a", "b"])
        U, S, Vh = t.svd(["a"], ["b"], absorb="none")
        # Reconstruct: U @ diag(S) @ Vh
        reconstructed = (U.data.reshape(-1, len(S)) @ np.diag(S)
                         @ Vh.data.reshape(len(S), -1))
        np.testing.assert_allclose(reconstructed, data, atol=1e-10)

    def test_qr_decomposition(self):
        data = np.random.default_rng(42).standard_normal((4, 3)) + 0j
        t = Tensor(data, legs=["a", "b"])
        Q, R = t.qr(["a"], ["b"])
        reconstructed = Q.data.reshape(-1, Q.shape[-1]) @ R.data.reshape(R.shape[0], -1)
        np.testing.assert_allclose(reconstructed, data, atol=1e-10)


# =====================================================================
# Contract pair tests
# =====================================================================

class TestContractPair:

    def test_auto_contract_shared_legs(self):
        a = Tensor(np.eye(2, dtype=complex), legs=["i", "j"])
        b = Tensor(np.ones((2, 3), dtype=complex), legs=["j", "k"])
        c = contract_pair(a, b)
        assert c.legs == ["i", "k"]
        assert c.shape == (2, 3)

    def test_explicit_legs(self):
        a = Tensor(np.eye(2, dtype=complex), legs=["a", "b"])
        b = Tensor(np.eye(2, dtype=complex), legs=["c", "d"])
        c = contract_pair(a, b, legs_A=["b"], legs_B=["c"])
        assert c.shape == (2, 2)
        assert c.legs == ["a", "d"]


# =====================================================================
# TensorNetwork tests
# =====================================================================

class TestTensorNetwork:

    def test_add_and_contract(self):
        tn = TensorNetwork()
        tn.add("A", Tensor(np.eye(2, dtype=complex), ["i", "j"]))
        tn.add("B", Tensor(np.ones((2, 3), dtype=complex), ["j", "k"]))
        result = tn.contract_all()
        assert result.shape == (2, 3)

    def test_bond_dimensions(self):
        tn = TensorNetwork()
        tn.add("A", Tensor(np.zeros((2, 3), dtype=complex), ["i", "j"]))
        tn.add("B", Tensor(np.zeros((3, 4), dtype=complex), ["j", "k"]))
        bonds = tn.total_bond_dimension()
        assert bonds["j"] == 3

    def test_empty_network_raises(self):
        tn = TensorNetwork()
        with pytest.raises(ValueError, match="empty"):
            tn.contract_all()


# =====================================================================
# MPS tests
# =====================================================================

class TestMPS:

    def test_product_state_properties(self, product_state_4):
        assert product_state_4.n_sites == 4
        assert product_state_4.d == 2
        assert product_state_4.max_bond_dim == 1

    def test_product_state_vector(self, product_state_4):
        psi = product_state_4.to_state_vector()
        assert psi.shape == (16,)
        # |0000> is the first basis vector
        assert abs(psi[0] - 1.0) < 1e-12
        assert np.sum(np.abs(psi[1:])) < 1e-12

    def test_ghz_state_vector(self, ghz_4):
        psi = ghz_4.to_state_vector()
        # GHZ = (|0000> + |1111>) / sqrt(2)
        assert abs(abs(psi[0]) - 1 / np.sqrt(2)) < 1e-10
        assert abs(abs(psi[-1]) - 1 / np.sqrt(2)) < 1e-10

    def test_ghz_bond_dimension(self, ghz_4):
        assert ghz_4.max_bond_dim == 2

    def test_norm_product_state(self, product_state_4):
        assert product_state_4.norm() == pytest.approx(1.0, abs=1e-10)

    def test_inner_product_self(self, ghz_4):
        overlap = ghz_4.inner(ghz_4)
        assert abs(overlap - 1.0) < 1e-8

    def test_from_state_vector_roundtrip(self):
        """Convert a state vector to MPS and back."""
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=3, d=2)
        psi_back = mps.to_state_vector()
        # Allow global phase difference
        overlap = abs(np.dot(psi.conj(), psi_back))
        assert overlap == pytest.approx(1.0, abs=1e-8)

    def test_w_state_properties(self):
        w = WState(4)
        assert w.n_sites == 4
        psi = w.to_state_vector()
        # W state should have 4 non-zero amplitudes
        nonzero = np.abs(psi) > 1e-10
        assert np.sum(nonzero) == 4

    def test_random_mps_normalized(self):
        mps = RandomMPS(4, chi=4, rng=np.random.default_rng(42))
        assert mps.norm() == pytest.approx(1.0, abs=1e-6)

    def test_canonicalize(self, ghz_4):
        canonical = ghz_4.canonicalize(center=2)
        # Norm should be preserved
        assert canonical.norm() == pytest.approx(1.0, abs=1e-6)

    def test_expectation_z_product_state(self, product_state_4):
        """<0000|Z_0|0000> = 1 for Pauli Z."""
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        exp_val = product_state_4.expectation(z, site=0)
        assert exp_val == pytest.approx(1.0, abs=1e-8)

    def test_entanglement_entropy_product_state(self, product_state_4):
        """Product states have zero entanglement entropy."""
        entropy = product_state_4.entanglement_entropy(bond=1)
        assert entropy == pytest.approx(0.0, abs=1e-6)

    def test_entanglement_entropy_ghz(self, ghz_4):
        """GHZ states have log(2) entanglement entropy at every bond."""
        entropy = ghz_4.entanglement_entropy(bond=0)
        assert entropy == pytest.approx(np.log(2), abs=0.1)


# =====================================================================
# MPO tests
# =====================================================================

class TestMPO:

    def test_ising_mpo_construction(self, ising_mpo_4):
        assert ising_mpo_4.n_sites == 4
        assert ising_mpo_4.d == 2

    def test_identity_mpo_expectation(self, product_state_4):
        """<psi|I|psi> = <psi|psi> for identity MPO."""
        identity = IdentityMPO(4)
        exp_val = identity.expectation(product_state_4)
        assert exp_val == pytest.approx(1.0, abs=1e-8)

    def test_identity_mpo_apply(self, product_state_4):
        """I|psi> = |psi>."""
        identity = IdentityMPO(4)
        result = identity.apply(product_state_4)
        # Check that the resulting state vector matches
        psi_original = product_state_4.to_state_vector()
        psi_result = result.to_state_vector()
        overlap = abs(np.dot(psi_original.conj(), psi_result))
        assert overlap == pytest.approx(1.0, abs=1e-6)

    def test_ising_mpo_to_matrix_hermitian(self, ising_mpo_4):
        """Ising Hamiltonian should be Hermitian."""
        mat = ising_mpo_4.to_matrix()
        assert mat.shape == (16, 16)
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-10)

    def test_heisenberg_mpo_hermitian(self):
        mpo = HeisenbergMPO(3, Jx=1.0, Jy=1.0, Jz=1.0)
        mat = mpo.to_matrix()
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-10)


# =====================================================================
# DMRG tests
# =====================================================================

class TestDMRG:

    def test_dmrg_ising_small(self):
        """DMRG on a 4-site Ising model should converge."""
        H = IsingMPO(4, J=1.0, h=1.0)
        result = dmrg_ground_state(H, chi_max=8, n_sweeps=10, tol=1e-6)
        assert isinstance(result, DMRGResult)
        assert result.energy < 0  # Ground state energy is negative
        assert len(result.energies) > 0

    def test_dmrg_energy_below_product_state(self):
        """DMRG energy should be below the product state energy."""
        H = IsingMPO(4, J=1.0, h=0.5)
        product = ProductState(4)
        product_energy = H.expectation(product)
        result = dmrg_ground_state(H, chi_max=8, n_sweeps=10)
        assert result.energy <= product_energy + 1e-6

    def test_dmrg_result_fields(self):
        H = IsingMPO(4, J=1.0, h=1.0)
        result = dmrg_ground_state(H, chi_max=8, n_sweeps=5)
        assert hasattr(result, 'energy')
        assert hasattr(result, 'state')
        assert hasattr(result, 'energies')
        assert isinstance(result.state, MPS)


# =====================================================================
# TEBD tests
# =====================================================================

class TestTEBD:

    def test_ising_nn_hamiltonian_construction(self):
        H = ising_nn_hamiltonian(4, J=1.0, h=1.0)
        assert isinstance(H, NNHamiltonian)
        assert H.n_sites == 4
        assert H.n_bonds == 3

    def test_heisenberg_nn_hamiltonian(self):
        H = heisenberg_nn_hamiltonian(4, Jx=1.0, Jy=1.0, Jz=1.0)
        assert H.n_sites == 4

    def test_tebd_evolve_preserves_norm_real_time(self):
        """Real-time evolution should approximately preserve the norm."""
        mps = ProductState(4)
        H = ising_nn_hamiltonian(4, J=1.0, h=0.5)
        result = tebd_evolve(mps, H, dt=0.01, n_steps=5, chi_max=8)
        assert isinstance(result, TEBDResult)
        final_norm = result.norms[-1]
        assert final_norm == pytest.approx(1.0, abs=0.1)

    def test_imaginary_tebd_lowers_energy(self):
        """Imaginary-time TEBD should lower the energy compared to product state."""
        mps = ProductState(4)
        H_mpo = IsingMPO(4, J=1.0, h=1.0)
        H_nn = ising_nn_hamiltonian(4, J=1.0, h=1.0)
        initial_energy = H_mpo.expectation(mps)

        cooler = ImaginaryTEBD(mps, H_nn, chi_max=8)
        result = cooler.run(dt=0.05, n_steps=20)
        final_state = result.states[-1]
        final_energy = H_mpo.expectation(final_state)
        assert final_energy < initial_energy + 1e-6

    def test_tebd_result_fields(self):
        mps = ProductState(4)
        H = ising_nn_hamiltonian(4, J=1.0, h=1.0)
        result = tebd_evolve(mps, H, dt=0.01, n_steps=3, chi_max=8)
        assert len(result.states) > 0
        assert len(result.times) > 0
        assert len(result.norms) > 0
        assert len(result.entropies) > 0
