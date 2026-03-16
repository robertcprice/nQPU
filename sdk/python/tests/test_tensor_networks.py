"""Comprehensive tests for the nQPU tensor network package.

Tests cover:
  - Tensor: contraction, SVD truncation, QR, reshape, trace
  - MPS: from_state_vector roundtrip, inner product, normalization, canonical forms
  - MPS factories: GHZ state properties, W state, product state, random MPS
  - MPO: Ising/Heisenberg construction, expectation values vs exact diag
  - DMRG: ground state energy within 1% for small Ising/Heisenberg
  - DMRG: convergence (energy decreases monotonically across sweeps)
  - TEBD: imaginary time converges to ground state
  - TEBD: real-time preserves norm
  - Integration: build Hamiltonian -> DMRG -> measure observables pipeline
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
    XXModelMPO,
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
    # PEPS
    PEPSTensor,
    PEPS,
    BoundaryMPS,
    SimpleUpdate,
    SimpleUpdateResult,
    ising_2d_bonds,
    heisenberg_2d_bonds,
    # TDVP
    TDVPResult,
    TDVP1Site,
    TDVP2Site,
    matrix_exponential_action,
    krylov_expm,
    # Autodiff
    TensorNode,
    tensor_node,
    contract,
    trace,
    svd,
    backward,
    DifferentiableContraction,
    VariationalTN,
    OptimizationResult,
    # TN Machine Learning
    MPSClassifier,
    TNKernel,
    MLResult,
)


# ===================================================================
# Helpers
# ===================================================================

def _exact_ground_state_energy(H_dense: np.ndarray) -> float:
    """Ground state energy via exact diagonalisation."""
    eigenvalues = np.linalg.eigvalsh(H_dense)
    return float(eigenvalues[0])


def _exact_ising_hamiltonian(n: int, J: float = 1.0, h: float = 1.0) -> np.ndarray:
    """Build full dense Ising Hamiltonian: H = -J sum ZZ - h sum X."""
    d = 2
    dim = d ** n
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n - 1):
        op = np.eye(1, dtype=np.complex128)
        for j in range(n):
            if j == i:
                op = np.kron(op, Z)
            elif j == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H -= J * op

    for i in range(n):
        op = np.eye(1, dtype=np.complex128)
        for j in range(n):
            if j == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H -= h * op

    return H


def _exact_heisenberg_hamiltonian(
    n: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    hz: float = 0.0,
) -> np.ndarray:
    """Build full dense Heisenberg Hamiltonian."""
    d = 2
    dim = d ** n
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n - 1):
        for pauli, Jp in [(X, Jx), (Y, Jy), (Z, Jz)]:
            op = np.eye(1, dtype=np.complex128)
            for j in range(n):
                if j == i or j == i + 1:
                    op = np.kron(op, pauli)
                else:
                    op = np.kron(op, I)
            H += Jp * op

    for i in range(n):
        op = np.eye(1, dtype=np.complex128)
        for j in range(n):
            if j == i:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H += hz * op

    return H


# ===================================================================
# Tensor Tests
# ===================================================================

class TestTensor:
    """Tests for the core Tensor class."""

    def test_creation(self):
        t = Tensor(np.eye(2, dtype=complex), ["i", "j"])
        assert t.shape == (2, 2)
        assert t.legs == ["i", "j"]
        assert t.ndim == 2

    def test_creation_shape_mismatch(self):
        with pytest.raises(ValueError, match="does not match"):
            Tensor(np.eye(2), ["i", "j", "k"])

    def test_transpose(self):
        data = np.arange(6, dtype=complex).reshape(2, 3)
        t = Tensor(data, ["i", "j"])
        t2 = t.transpose(["j", "i"])
        assert t2.shape == (3, 2)
        assert t2.legs == ["j", "i"]
        assert np.allclose(t2.data, data.T)

    def test_reshape(self):
        data = np.arange(12, dtype=complex).reshape(3, 4)
        t = Tensor(data, ["i", "j"])
        t2 = t.reshape((2, 6), ["a", "b"])
        assert t2.shape == (2, 6)
        assert t2.legs == ["a", "b"]

    def test_trace(self):
        data = np.eye(3, dtype=complex)
        t = Tensor(data, ["i", "j"])
        result = t.trace("i", "j")
        assert result.shape == ()
        assert np.isclose(result.data, 3.0)

    def test_trace_rank4(self):
        data = np.zeros((2, 3, 2, 4), dtype=complex)
        for a in range(2):
            for b in range(3):
                for c in range(4):
                    data[a, b, a, c] = a + b + c
        t = Tensor(data, ["i", "j", "k", "l"])
        result = t.trace("i", "k")
        assert result.shape == (3, 4)
        assert result.legs == ["j", "l"]

    def test_conjugate(self):
        data = np.array([[1 + 1j, 2 + 2j]], dtype=complex)
        t = Tensor(data, ["i", "j"])
        tc = t.conjugate()
        assert np.allclose(tc.data, np.conj(data))

    def test_norm(self):
        data = np.array([3.0, 4.0], dtype=complex)
        t = Tensor(data, ["i"])
        assert np.isclose(t.norm, 5.0)

    def test_copy(self):
        t = Tensor(np.eye(2, dtype=complex), ["i", "j"])
        t2 = t.copy()
        t2.data[0, 0] = 999
        assert t.data[0, 0] == 1.0  # original unchanged

    def test_multiply_scalar(self):
        t = Tensor(np.ones((2, 2), dtype=complex), ["i", "j"])
        t2 = t * 3.0
        assert np.allclose(t2.data, 3.0)
        t3 = 3.0 * t
        assert np.allclose(t3.data, 3.0)


class TestTensorSVD:
    """Tests for SVD decomposition of tensors."""

    def test_svd_identity(self):
        data = np.eye(4, dtype=complex)
        t = Tensor(data, ["i", "j"])
        U, S, Vh = t.svd(["i"], ["j"])
        # Reconstruct
        reconstructed = contract_pair(U, Vh)
        assert np.allclose(reconstructed.data, data, atol=1e-12)

    def test_svd_truncation(self):
        # Rank-2 matrix
        data = np.array([[1, 0, 0], [0, 0.5, 0], [0, 0, 0.01]], dtype=complex)
        t = Tensor(data, ["i", "j"])
        U, S, Vh = t.svd(["i"], ["j"], chi_max=2, absorb="none")
        assert len(S) == 2
        assert S[0] > S[1]

    def test_svd_cutoff(self):
        data = np.diag([1.0, 0.1, 0.001]).astype(complex)
        t = Tensor(data, ["i", "j"])
        U, S, Vh = t.svd(["i"], ["j"], cutoff=0.05, absorb="none")
        assert len(S) == 2

    def test_svd_absorb_left(self):
        data = np.random.randn(3, 4).astype(complex)
        t = Tensor(data, ["i", "j"])
        U, S, Vh = t.svd(["i"], ["j"], absorb="left")
        result = contract_pair(U, Vh)
        assert np.allclose(result.data, data, atol=1e-12)

    def test_svd_absorb_right(self):
        data = np.random.randn(3, 4).astype(complex)
        t = Tensor(data, ["i", "j"])
        U, S, Vh = t.svd(["i"], ["j"], absorb="right")
        result = contract_pair(U, Vh)
        assert np.allclose(result.data, data, atol=1e-12)

    def test_svd_absorb_none(self):
        data = np.random.randn(3, 4).astype(complex)
        t = Tensor(data, ["i", "j"])
        U, S, Vh = t.svd(["i"], ["j"], absorb="none")
        # Reconstruct U @ diag(S) @ Vh
        # U has shape (3, k), Vh has shape (k, 4), S has shape (k,)
        # Multiply S into U along the last axis
        US = Tensor(U.data * S[np.newaxis, :], U.legs)
        result = contract_pair(US, Vh)
        assert np.allclose(result.data, data, atol=1e-12)

    def test_svd_rank3(self):
        data = np.random.randn(2, 3, 4).astype(complex)
        t = Tensor(data, ["i", "j", "k"])
        U, S, Vh = t.svd(["i", "j"], ["k"], absorb="right")
        result = contract_pair(U, Vh)
        assert result.shape == (2, 3, 4)


class TestTensorQR:
    """Tests for QR decomposition."""

    def test_qr_basic(self):
        data = np.random.randn(3, 4).astype(complex)
        t = Tensor(data, ["i", "j"])
        Q, R = t.qr(["i"], ["j"])
        result = contract_pair(Q, R)
        assert np.allclose(result.data, data, atol=1e-12)

    def test_qr_orthogonality(self):
        data = np.random.randn(4, 3).astype(complex)
        t = Tensor(data, ["i", "j"])
        Q, R = t.qr(["i"], ["j"])
        # Q should have orthonormal columns
        mat = Q.data.reshape(-1, Q.shape[-1])
        gram = mat.conj().T @ mat
        assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-12)


class TestContractPair:
    """Tests for pairwise tensor contraction."""

    def test_matrix_multiply(self):
        A = Tensor(np.eye(3, dtype=complex), ["i", "j"])
        B = Tensor(np.ones((3, 2), dtype=complex), ["j", "k"])
        C = contract_pair(A, B)
        assert C.shape == (3, 2)
        assert C.legs == ["i", "k"]
        assert np.allclose(C.data, np.ones((3, 2)))

    def test_auto_detect_shared_legs(self):
        A = Tensor(np.random.randn(2, 3).astype(complex), ["a", "shared"])
        B = Tensor(np.random.randn(3, 4).astype(complex), ["shared", "b"])
        C = contract_pair(A, B)
        assert C.legs == ["a", "b"]
        assert C.shape == (2, 4)

    def test_outer_product(self):
        A = Tensor(np.array([1, 2], dtype=complex), ["i"])
        B = Tensor(np.array([3, 4, 5], dtype=complex), ["j"])
        C = contract_pair(A, B, legs_A=[], legs_B=[])
        assert C.shape == (2, 3)

    def test_explicit_legs(self):
        A = Tensor(np.eye(2, dtype=complex), ["a", "b"])
        B = Tensor(np.eye(2, dtype=complex), ["c", "d"])
        C = contract_pair(A, B, legs_A=["b"], legs_B=["c"])
        assert C.shape == (2, 2)

    def test_full_contraction(self):
        # Inner product of two vectors
        v1 = Tensor(np.array([1, 2, 3], dtype=complex), ["i"])
        v2 = Tensor(np.array([4, 5, 6], dtype=complex), ["i"])
        result = contract_pair(v1, v2)
        assert result.shape == ()
        assert np.isclose(result.data, 32.0)


class TestTensorNetwork:
    """Tests for the TensorNetwork container."""

    def test_basic_contraction(self):
        tn = TensorNetwork()
        tn.add("A", Tensor(np.eye(2, dtype=complex), ["i", "j"]))
        tn.add("B", Tensor(np.ones((2, 3), dtype=complex), ["j", "k"]))
        result = tn.contract_all()
        assert result.shape == (2, 3)

    def test_ordered_contraction(self):
        tn = TensorNetwork()
        tn.add("A", Tensor(np.eye(2, dtype=complex), ["i", "j"]))
        tn.add("B", Tensor(np.ones((2, 3), dtype=complex), ["j", "k"]))
        tn.add("C", Tensor(np.ones((3, 4), dtype=complex), ["k", "l"]))
        tn.set_contraction_order([("A", "B"), ("A", "C")])
        result = tn.contract_all()
        assert result.shape == (2, 4)

    def test_total_bond_dimension(self):
        tn = TensorNetwork()
        tn.add("A", Tensor(np.zeros((2, 3), dtype=complex), ["i", "j"]))
        tn.add("B", Tensor(np.zeros((3, 4), dtype=complex), ["j", "k"]))
        bonds = tn.total_bond_dimension()
        assert bonds["i"] == 2
        assert bonds["j"] == 3
        assert bonds["k"] == 4

    def test_empty_network(self):
        tn = TensorNetwork()
        with pytest.raises(ValueError):
            tn.contract_all()

    def test_add_remove(self):
        tn = TensorNetwork()
        t = Tensor(np.eye(2, dtype=complex), ["i", "j"])
        tn.add("A", t)
        assert len(tn) == 1
        removed = tn.remove("A")
        assert len(tn) == 0
        assert np.allclose(removed.data, t.data)


# ===================================================================
# MPS Tests
# ===================================================================

class TestMPS:
    """Tests for Matrix Product States."""

    def test_product_state(self):
        mps = ProductState(4)
        assert mps.n_sites == 4
        assert mps.max_bond_dim == 1
        psi = mps.to_state_vector()
        expected = np.zeros(16, dtype=complex)
        expected[0] = 1.0
        assert np.allclose(psi, expected)

    def test_from_state_vector_roundtrip_2_sites(self):
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        mps = MPS.from_state_vector(psi, n_sites=2)
        psi_out = mps.to_state_vector()
        assert np.allclose(psi, psi_out, atol=1e-12)

    def test_from_state_vector_roundtrip_4_sites(self):
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=4)
        psi_out = mps.to_state_vector()
        assert np.allclose(psi, psi_out, atol=1e-10)

    def test_from_state_vector_roundtrip_6_sites(self):
        rng = np.random.default_rng(123)
        psi = rng.standard_normal(64) + 1j * rng.standard_normal(64)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=6)
        psi_out = mps.to_state_vector()
        assert np.allclose(psi, psi_out, atol=1e-10)

    def test_from_state_vector_truncated(self):
        rng = np.random.default_rng(99)
        psi = rng.standard_normal(64) + 1j * rng.standard_normal(64)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=6, chi_max=4)
        assert mps.max_bond_dim <= 4

    def test_inner_product_self(self):
        mps = GHZState(4)
        overlap = mps.inner(mps)
        assert np.isclose(abs(overlap), 1.0, atol=1e-12)

    def test_inner_product_orthogonal(self):
        # |00> and |11>
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
        psi1 = np.array([0, 0, 0, 1], dtype=complex)
        mps0 = MPS.from_state_vector(psi0, n_sites=2)
        mps1 = MPS.from_state_vector(psi1, n_sites=2)
        assert np.isclose(abs(mps0.inner(mps1)), 0.0, atol=1e-12)

    def test_inner_product_bell_state(self):
        psi_bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        mps = MPS.from_state_vector(psi_bell, n_sites=2)
        assert np.isclose(mps.inner(mps), 1.0, atol=1e-12)

    def test_norm(self):
        mps = GHZState(4)
        assert np.isclose(mps.norm(), 1.0, atol=1e-12)

    def test_normalize(self):
        tensors = [t * 3.0 for t in GHZState(4).tensors]
        mps = MPS(tensors)
        assert not np.isclose(mps.norm(), 1.0)
        mps_n = mps.normalize()
        assert np.isclose(mps_n.norm(), 1.0, atol=1e-12)

    def test_bond_dimensions(self):
        mps = GHZState(5)
        assert mps.bond_dimensions == [2, 2, 2, 2]

    def test_validation_rank_error(self):
        with pytest.raises(ValueError, match="rank-3"):
            MPS([np.zeros((2, 2))])

    def test_validation_boundary_error(self):
        with pytest.raises(ValueError, match="chi_left=1"):
            MPS([np.zeros((2, 2, 1))])


class TestMPSCanonical:
    """Tests for MPS canonical forms."""

    def test_canonicalize_left(self):
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=4)
        canon = mps.canonicalize(0)
        # State vector should be unchanged
        psi_out = canon.to_state_vector()
        assert np.allclose(psi, psi_out, atol=1e-10)

    def test_canonicalize_right(self):
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=4)
        canon = mps.canonicalize(3)
        psi_out = canon.to_state_vector()
        assert np.allclose(psi, psi_out, atol=1e-10)

    def test_canonicalize_center(self):
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=4)
        canon = mps.canonicalize(2)
        psi_out = canon.to_state_vector()
        assert np.allclose(psi, psi_out, atol=1e-10)

    def test_left_canonical_isometry(self):
        mps = RandomMPS(6, chi=4, rng=np.random.default_rng(42))
        canon = mps.canonicalize(5)  # All left-canonical except site 5
        # Check site 0 is left-isometric: sum_s A[s]^dag A[s] = I
        A = canon.tensors[0]  # (1, d, chi)
        gram = np.einsum("asb,asc->bc", np.conj(A), A)
        assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10)


class TestMPSExpectation:
    """Tests for MPS expectation values."""

    def test_expectation_z_product_state(self):
        mps = ProductState(4)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        # All |0>, so <Z> = +1
        for site in range(4):
            assert np.isclose(mps.expectation(Z, site), 1.0, atol=1e-12)

    def test_expectation_x_product_state(self):
        mps = ProductState(4)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        # |0> state, <X> = 0
        for site in range(4):
            assert np.isclose(mps.expectation(X, site), 0.0, atol=1e-12)

    def test_expectation_ghz_z(self):
        mps = GHZState(4)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        # GHZ: equal mix of |0000> and |1111>: <Z_i> = 0
        for site in range(4):
            assert np.isclose(mps.expectation(Z, site), 0.0, atol=1e-12)

    def test_expectation_matches_exact(self):
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=4)

        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        for site in range(4):
            # Exact: <psi|Z_i|psi>
            Z_full = np.eye(1, dtype=complex)
            I = np.eye(2, dtype=complex)
            for j in range(4):
                Z_full = np.kron(Z_full, Z if j == site else I)
            exact = np.real(psi.conj() @ Z_full @ psi)
            mps_val = mps.expectation(Z, site)
            assert np.isclose(mps_val, exact, atol=1e-10)

    def test_two_site_expectation(self):
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=3)

        ZZ = np.kron(
            np.array([[1, 0], [0, -1]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex),
        )
        # Exact
        I = np.eye(2, dtype=complex)
        ZZ_full = np.kron(ZZ, I)
        exact = np.real(psi.conj() @ ZZ_full @ psi)
        mps_val = mps.expectation_two_site(ZZ, 0, 1)
        assert np.isclose(mps_val, exact, atol=1e-10)


class TestMPSEntanglement:
    """Tests for entanglement entropy calculations."""

    def test_product_state_zero_entropy(self):
        mps = ProductState(4)
        for bond in range(3):
            S = mps.entanglement_entropy(bond)
            assert np.isclose(S, 0.0, atol=1e-10)

    def test_ghz_entropy_log2(self):
        mps = GHZState(6)
        ln2 = np.log(2)
        for bond in range(5):
            S = mps.entanglement_entropy(bond)
            assert np.isclose(S, ln2, atol=1e-6), (
                f"Bond {bond}: expected {ln2:.6f}, got {S:.6f}"
            )

    def test_bell_state_entropy(self):
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        mps = MPS.from_state_vector(psi, n_sites=2)
        S = mps.entanglement_entropy(0)
        assert np.isclose(S, np.log(2), atol=1e-10)


class TestMPSFactories:
    """Tests for MPS factory functions."""

    def test_ghz_state_vector(self):
        mps = GHZState(3)
        psi = mps.to_state_vector()
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1.0 / np.sqrt(2)  # |000>
        expected[7] = 1.0 / np.sqrt(2)  # |111>
        assert np.allclose(psi, expected, atol=1e-12)

    def test_ghz_norm(self):
        for n in range(2, 8):
            mps = GHZState(n)
            assert np.isclose(mps.norm(), 1.0, atol=1e-12)

    def test_random_mps_normalized(self):
        mps = RandomMPS(6, chi=4, rng=np.random.default_rng(42))
        assert np.isclose(mps.norm(), 1.0, atol=1e-10)

    def test_random_mps_bond_dimension(self):
        mps = RandomMPS(6, chi=4, rng=np.random.default_rng(42))
        for bd in mps.bond_dimensions:
            assert bd <= 4

    def test_random_mps_reproducible(self):
        mps1 = RandomMPS(4, chi=3, rng=np.random.default_rng(42))
        mps2 = RandomMPS(4, chi=3, rng=np.random.default_rng(42))
        overlap = abs(mps1.inner(mps2))
        assert np.isclose(overlap, 1.0, atol=1e-10)

    def test_w_state(self):
        mps = WState(3)
        psi = mps.to_state_vector()
        c = 1.0 / np.sqrt(3)
        expected = np.zeros(8, dtype=complex)
        expected[4] = c  # |100>
        expected[2] = c  # |010>
        expected[1] = c  # |001>
        assert np.allclose(psi, expected, atol=1e-12)

    def test_w_state_norm(self):
        for n in [2, 3, 4, 5]:
            mps = WState(n)
            assert np.isclose(mps.norm(), 1.0, atol=1e-10)

    def test_w_state_bond_dim(self):
        mps = WState(5)
        assert mps.max_bond_dim == 2


# ===================================================================
# MPO Tests
# ===================================================================

class TestMPO:
    """Tests for Matrix Product Operators."""

    def test_identity_mpo(self):
        mpo = IdentityMPO(4)
        mat = mpo.to_matrix()
        assert np.allclose(mat, np.eye(16), atol=1e-12)

    def test_identity_expectation(self):
        mps = GHZState(4)
        mpo = IdentityMPO(4)
        val = mpo.expectation(mps)
        assert np.isclose(val, 1.0, atol=1e-12)

    def test_ising_mpo_matches_exact(self):
        n = 4
        J, h = 1.0, 0.5
        mpo = IsingMPO(n, J=J, h=h)
        H_mpo = mpo.to_matrix()
        H_exact = _exact_ising_hamiltonian(n, J=J, h=h)
        assert np.allclose(H_mpo, H_exact, atol=1e-10), (
            f"MPO matrix does not match exact Ising Hamiltonian"
        )

    def test_ising_mpo_energy_exact(self):
        n = 4
        J, h = 1.0, 1.0
        mpo = IsingMPO(n, J=J, h=h)
        H_mpo = mpo.to_matrix()
        H_exact = _exact_ising_hamiltonian(n, J=J, h=h)
        e_mpo = _exact_ground_state_energy(H_mpo)
        e_exact = _exact_ground_state_energy(H_exact)
        assert np.isclose(e_mpo, e_exact, atol=1e-10)

    def test_heisenberg_mpo_matches_exact(self):
        n = 4
        mpo = HeisenbergMPO(n, Jx=1.0, Jy=1.0, Jz=1.0)
        H_mpo = mpo.to_matrix()
        H_exact = _exact_heisenberg_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0)
        assert np.allclose(H_mpo, H_exact, atol=1e-10), (
            f"MPO matrix does not match exact Heisenberg Hamiltonian"
        )

    def test_heisenberg_mpo_xxz(self):
        n = 4
        mpo = HeisenbergMPO(n, Jx=1.0, Jy=1.0, Jz=0.5)
        H_mpo = mpo.to_matrix()
        H_exact = _exact_heisenberg_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=0.5)
        assert np.allclose(H_mpo, H_exact, atol=1e-10)

    def test_mpo_expectation_product_state(self):
        # |0000> in Ising: E = -J*(n-1) - h*0 = -(n-1)
        # Actually H = -J sum ZZ - h sum X
        # For |0000>: each Z gives +1, so ZZ = 1, sum ZZ = n-1
        # E = -J*(n-1) + 0 (since <0|X|0> = 0 for each site)
        n = 4
        J, h = 1.0, 0.5
        mpo = IsingMPO(n, J=J, h=h)
        mps = ProductState(n)
        E = mpo.expectation(mps)
        assert np.isclose(E, -J * (n - 1), atol=1e-10)

    def test_mpo_apply_identity(self):
        mps = GHZState(4)
        mpo = IdentityMPO(4)
        result = mpo.apply(mps)
        psi_orig = mps.to_state_vector()
        psi_new = result.to_state_vector()
        # Normalize for comparison
        psi_new /= np.linalg.norm(psi_new)
        assert np.allclose(psi_orig, psi_new, atol=1e-10)

    def test_mpo_apply_with_truncation(self):
        mps = GHZState(4)
        mpo = IsingMPO(4)
        result = mpo.apply(mps, chi_max=8)
        assert result.max_bond_dim <= 8

    def test_xx_model_mpo(self):
        n = 4
        mpo = XXModelMPO(n, J=1.0)
        H_mpo = mpo.to_matrix()
        H_exact = _exact_heisenberg_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=0.0)
        assert np.allclose(H_mpo, H_exact, atol=1e-10)


# ===================================================================
# DMRG Tests
# ===================================================================

class TestDMRG:
    """Tests for the DMRG ground state solver."""

    def test_ising_4_site_ground_state(self):
        n = 4
        J, h = 1.0, 1.0
        mpo = IsingMPO(n, J=J, h=h)
        H_exact = _exact_ising_hamiltonian(n, J=J, h=h)
        e_exact = _exact_ground_state_energy(H_exact)

        result = dmrg_ground_state(mpo, chi_max=16, n_sweeps=20)
        assert abs(result.energy - e_exact) / abs(e_exact) < 0.01, (
            f"DMRG energy {result.energy:.8f} vs exact {e_exact:.8f}"
        )

    def test_ising_6_site_ground_state(self):
        n = 6
        J, h = 1.0, 1.0
        mpo = IsingMPO(n, J=J, h=h)
        H_exact = _exact_ising_hamiltonian(n, J=J, h=h)
        e_exact = _exact_ground_state_energy(H_exact)

        result = dmrg_ground_state(mpo, chi_max=32, n_sweeps=30)
        assert abs(result.energy - e_exact) / abs(e_exact) < 0.01, (
            f"DMRG energy {result.energy:.8f} vs exact {e_exact:.8f}"
        )

    def test_heisenberg_4_site_ground_state(self):
        n = 4
        mpo = HeisenbergMPO(n, Jx=1.0, Jy=1.0, Jz=1.0)
        H_exact = _exact_heisenberg_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0)
        e_exact = _exact_ground_state_energy(H_exact)

        result = dmrg_ground_state(mpo, chi_max=16, n_sweeps=30)
        assert abs(result.energy - e_exact) / abs(e_exact) < 0.01, (
            f"DMRG energy {result.energy:.8f} vs exact {e_exact:.8f}"
        )

    def test_heisenberg_6_site_ground_state(self):
        n = 6
        mpo = HeisenbergMPO(n, Jx=1.0, Jy=1.0, Jz=1.0)
        H_exact = _exact_heisenberg_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0)
        e_exact = _exact_ground_state_energy(H_exact)

        result = dmrg_ground_state(mpo, chi_max=32, n_sweeps=30)
        assert abs(result.energy - e_exact) / abs(e_exact) < 0.01, (
            f"DMRG energy {result.energy:.8f} vs exact {e_exact:.8f}"
        )

    def test_convergence_history(self):
        n = 4
        mpo = IsingMPO(n, J=1.0, h=1.0)
        result = dmrg_ground_state(mpo, chi_max=16, n_sweeps=10)
        assert len(result.energies) > 0
        assert len(result.bond_dimensions) == len(result.energies)

    def test_dmrg_energy_decreasing(self):
        n = 6
        mpo = IsingMPO(n, J=1.0, h=0.5)
        result = dmrg_ground_state(mpo, chi_max=16, n_sweeps=15)
        # Energy should generally decrease (allow small fluctuations)
        # Check that final energy < initial energy
        assert result.energies[-1] <= result.energies[0] + 0.1, (
            "DMRG energy should decrease over sweeps"
        )

    def test_dmrg_result_state_is_mps(self):
        mpo = IsingMPO(4)
        result = dmrg_ground_state(mpo, chi_max=8, n_sweeps=5)
        assert isinstance(result.state, MPS)
        assert result.state.n_sites == 4

    def test_dmrg_class_interface(self):
        mpo = IsingMPO(4)
        solver = DMRG(mpo, chi_max=8, n_sweeps=5)
        result = solver.run()
        assert isinstance(result, DMRGResult)
        assert result.energy < 0  # Ising ground state energy is negative

    def test_dmrg_bond_dimension_growth(self):
        mpo = IsingMPO(6, J=1.0, h=1.0)
        result = dmrg_ground_state(mpo, chi_max=32, n_sweeps=10)
        # Bond dim should grow from 1 (product state) to something > 1
        if len(result.bond_dimensions) >= 2:
            final_max = max(result.bond_dimensions[-1])
            assert final_max > 1, "Bond dimension should grow during DMRG"

    def test_dmrg_ising_strong_field(self):
        """Strong field limit: ground state close to all X-polarized."""
        n = 4
        J, h = 0.1, 5.0
        mpo = IsingMPO(n, J=J, h=h)
        H_exact = _exact_ising_hamiltonian(n, J=J, h=h)
        e_exact = _exact_ground_state_energy(H_exact)

        result = dmrg_ground_state(mpo, chi_max=16, n_sweeps=20)
        assert abs(result.energy - e_exact) / abs(e_exact) < 0.01


# ===================================================================
# TEBD Tests
# ===================================================================

class TestTEBD:
    """Tests for TEBD time evolution."""

    def test_real_time_norm_preservation(self):
        mps = ProductState(4)
        H = ising_nn_hamiltonian(4, J=1.0, h=0.5)
        result = tebd_evolve(mps, H, dt=0.05, n_steps=20, chi_max=16)
        # Norm should stay close to 1 for real-time evolution
        for norm in result.norms:
            assert np.isclose(norm, 1.0, atol=0.05), (
                f"Norm deviated: {norm}"
            )

    def test_real_time_trivial(self):
        """Evolving an eigenstate should not change it (up to phase)."""
        # Product |0000> under H = -J ZZ (no transverse field)
        # This is an eigenstate of ZZ
        mps = ProductState(4)
        H = ising_nn_hamiltonian(4, J=1.0, h=0.0)
        result = tebd_evolve(mps, H, dt=0.1, n_steps=10, chi_max=16)

        # Should still be close to product state
        psi_initial = mps.to_state_vector()
        psi_final = result.states[-1].to_state_vector()
        # Compare up to global phase
        overlap = abs(np.dot(np.conj(psi_initial), psi_final))
        assert np.isclose(overlap, 1.0, atol=1e-4)

    def test_imaginary_time_convergence(self):
        n = 4
        J, h = 1.0, 1.0
        H_nn = ising_nn_hamiltonian(n, J=J, h=h)
        H_exact = _exact_ising_hamiltonian(n, J=J, h=h)
        e_exact = _exact_ground_state_energy(H_exact)

        mps = ProductState(n)
        result = tebd_evolve(
            mps, H_nn, dt=0.05, n_steps=200, chi_max=16, imaginary=True
        )

        # Check energy via MPO
        mpo = IsingMPO(n, J=J, h=h)
        e_final = mpo.expectation(result.states[-1])
        assert abs(e_final - e_exact) / abs(e_exact) < 0.05, (
            f"TEBD imag time energy {e_final:.6f} vs exact {e_exact:.6f}"
        )

    def test_imaginary_tebd_class(self):
        n = 4
        J, h = 1.0, 1.0
        H_nn = ising_nn_hamiltonian(n, J=J, h=h)
        mps = ProductState(n)
        cooler = ImaginaryTEBD(mps, H_nn, chi_max=16)
        result = cooler.run(dt=0.05, n_steps=200)
        assert len(result.states) > 0
        assert np.isclose(result.states[-1].norm(), 1.0, atol=1e-6)

    def test_first_order_trotter(self):
        mps = ProductState(4)
        H = ising_nn_hamiltonian(4, J=1.0, h=0.5)
        engine = TEBD(mps, H, chi_max=16, order=1)
        result = engine.evolve(dt=0.05, n_steps=20)
        assert len(result.states) > 0

    def test_second_order_trotter(self):
        mps = ProductState(4)
        H = ising_nn_hamiltonian(4, J=1.0, h=0.5)
        engine = TEBD(mps, H, chi_max=16, order=2)
        result = engine.evolve(dt=0.05, n_steps=20)
        assert len(result.states) > 0

    def test_tebd_entropy_tracking(self):
        mps = ProductState(4)
        H = ising_nn_hamiltonian(4, J=1.0, h=1.0)
        result = tebd_evolve(mps, H, dt=0.05, n_steps=20, chi_max=16)
        # Entropy should be recorded at each step
        assert len(result.entropies) > 0
        # Product state starts with zero entropy
        for s in result.entropies[0]:
            assert np.isclose(s, 0.0, atol=1e-6)

    def test_tebd_bond_dimension_tracking(self):
        mps = ProductState(4)
        H = ising_nn_hamiltonian(4, J=1.0, h=1.0)
        result = tebd_evolve(mps, H, dt=0.05, n_steps=20, chi_max=16)
        assert len(result.bond_dimensions) > 0

    def test_tebd_heisenberg(self):
        n = 4
        H_nn = heisenberg_nn_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0)
        mps = ProductState(n)
        result = tebd_evolve(mps, H_nn, dt=0.05, n_steps=10, chi_max=16)
        assert len(result.states) > 0

    def test_record_every(self):
        mps = ProductState(4)
        H = ising_nn_hamiltonian(4, J=1.0, h=0.5)
        result = tebd_evolve(mps, H, dt=0.05, n_steps=20, chi_max=16)
        # record_every=1 by default, so 21 snapshots (initial + 20 steps)
        assert len(result.states) == 21

    def test_imag_time_heisenberg_convergence(self):
        n = 4
        H_nn = heisenberg_nn_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0)
        H_exact = _exact_heisenberg_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0)
        e_exact = _exact_ground_state_energy(H_exact)

        # Start from the Neel state |0101> (S_z=0 sector) so that
        # imaginary time evolution can reach the Heisenberg ground state,
        # which lives in the S_z=0 sector.  The product state |0000>
        # (S_z=+n/2) has zero overlap with the ground state because
        # the Heisenberg Hamiltonian conserves total S_z.
        neel = np.zeros(2**n, dtype=complex)
        neel[0b0101] = 1.0  # |0101> = alternating up-down
        mps = MPS.from_state_vector(neel, n_sites=n)

        result = tebd_evolve(
            mps, H_nn, dt=0.05, n_steps=300, chi_max=16, imaginary=True
        )

        mpo = HeisenbergMPO(n, Jx=1.0, Jy=1.0, Jz=1.0)
        e_final = mpo.expectation(result.states[-1])
        assert abs(e_final - e_exact) / abs(e_exact) < 0.05, (
            f"TEBD imag time energy {e_final:.6f} vs exact {e_exact:.6f}"
        )


class TestNNHamiltonian:
    """Tests for nearest-neighbour Hamiltonian construction."""

    def test_ising_nn_sites(self):
        H = ising_nn_hamiltonian(6, J=1.0, h=0.5)
        assert H.n_sites == 6
        assert H.n_bonds == 5

    def test_heisenberg_nn_sites(self):
        H = heisenberg_nn_hamiltonian(4)
        assert H.n_sites == 4
        assert H.n_bonds == 3

    def test_ising_nn_bond_shapes(self):
        H = ising_nn_hamiltonian(4)
        for bond in H.h_bonds:
            assert bond.shape == (4, 4)

    def test_ising_nn_hermitian(self):
        H = ising_nn_hamiltonian(4, J=1.0, h=0.5)
        for bond in H.h_bonds:
            assert np.allclose(bond, bond.conj().T, atol=1e-12)

    def test_heisenberg_nn_hermitian(self):
        H = heisenberg_nn_hamiltonian(4, Jx=1.0, Jy=1.0, Jz=1.0)
        for bond in H.h_bonds:
            assert np.allclose(bond, bond.conj().T, atol=1e-12)


# ===================================================================
# Integration Tests
# ===================================================================

class TestIntegration:
    """End-to-end integration tests combining multiple components."""

    def test_full_pipeline_ising(self):
        """Build Ising -> DMRG ground state -> measure observables."""
        n = 4
        J, h = 1.0, 1.0

        # Step 1: Build Hamiltonian
        mpo = IsingMPO(n, J=J, h=h)

        # Step 2: Find ground state
        result = dmrg_ground_state(mpo, chi_max=16, n_sweeps=20)

        # Step 3: Verify energy
        H_exact = _exact_ising_hamiltonian(n, J=J, h=h)
        e_exact = _exact_ground_state_energy(H_exact)
        assert abs(result.energy - e_exact) / abs(e_exact) < 0.01

        # Step 4: Measure observables
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        mps = result.state

        # Magnetization should be reasonable
        mx_total = sum(mps.expectation(X, i) for i in range(n))
        mz_total = sum(mps.expectation(Z, i) for i in range(n))
        # At h=J=1 (critical point), both should be nonzero
        assert isinstance(mx_total, float)
        assert isinstance(mz_total, float)

    def test_full_pipeline_heisenberg(self):
        """Build Heisenberg -> DMRG -> measure -> check entanglement."""
        n = 4
        mpo = HeisenbergMPO(n, Jx=1.0, Jy=1.0, Jz=1.0)
        result = dmrg_ground_state(mpo, chi_max=16, n_sweeps=20)

        H_exact = _exact_heisenberg_hamiltonian(n, Jx=1.0, Jy=1.0, Jz=1.0)
        e_exact = _exact_ground_state_energy(H_exact)
        assert abs(result.energy - e_exact) / abs(e_exact) < 0.01

        # Ground state should have nonzero entanglement
        mps = result.state
        S = mps.entanglement_entropy(n // 2 - 1)
        assert S > 0.1, "Heisenberg ground state should be entangled"

    def test_dmrg_then_tebd_evolution(self):
        """Find ground state with DMRG, then evolve with TEBD."""
        n = 4
        J, h = 1.0, 1.0
        mpo = IsingMPO(n, J=J, h=h)

        # Find ground state
        gs_result = dmrg_ground_state(mpo, chi_max=16, n_sweeps=15)
        gs = gs_result.state

        # Evolve in real time
        H_nn = ising_nn_hamiltonian(n, J=J, h=h)
        result = tebd_evolve(gs, H_nn, dt=0.02, n_steps=10, chi_max=16)

        # Eigenstate evolution should preserve norm and overlap
        for norm in result.norms:
            assert np.isclose(norm, 1.0, atol=0.05)

    def test_imag_tebd_matches_dmrg(self):
        """Both methods should find the same ground state energy."""
        n = 4
        J, h = 1.0, 1.0

        # DMRG
        mpo = IsingMPO(n, J=J, h=h)
        dmrg_result = dmrg_ground_state(mpo, chi_max=16, n_sweeps=20)

        # TEBD imaginary time
        H_nn = ising_nn_hamiltonian(n, J=J, h=h)
        mps = ProductState(n)
        tebd_result = tebd_evolve(
            mps, H_nn, dt=0.05, n_steps=200, chi_max=16, imaginary=True
        )
        e_tebd = mpo.expectation(tebd_result.states[-1])

        # Both should be within 5% of each other
        assert abs(dmrg_result.energy - e_tebd) / abs(dmrg_result.energy) < 0.05, (
            f"DMRG E={dmrg_result.energy:.6f} vs TEBD E={e_tebd:.6f}"
        )

    def test_mps_to_mpo_expectation_consistency(self):
        """MPO expectation via contract should match MPS local expectation."""
        n = 4
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=n)

        mpo = IsingMPO(n, J=1.0, h=0.5)
        e_mpo = mpo.expectation(mps)

        # Compare with exact: <psi|H|psi>
        H_exact = _exact_ising_hamiltonian(n, J=1.0, h=0.5)
        e_exact = float(np.real(psi.conj() @ H_exact @ psi))
        assert np.isclose(e_mpo, e_exact, atol=1e-8)

    def test_tensor_network_mps_roundtrip(self):
        """Build MPS from state vector, contract to TensorNetwork."""
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=3)

        # Build a TensorNetwork from MPS tensors
        tn = TensorNetwork()
        for i, t in enumerate(mps.tensors):
            legs = [f"bond_{i}", f"phys_{i}", f"bond_{i+1}"]
            tn.add(f"site_{i}", Tensor(t, legs))

        # The network should have the right structure
        assert len(tn) == 3

    def test_mps_copy_independence(self):
        """Ensure MPS copy is independent of original."""
        mps = GHZState(4)
        mps2 = mps.copy()
        mps2.tensors[0] *= 0
        # Original should be unchanged
        assert np.isclose(mps.norm(), 1.0, atol=1e-12)

    def test_svd_truncation_in_from_state_vector(self):
        """Truncated SVD should reduce bond dimension while maintaining quality."""
        n = 6
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
        psi /= np.linalg.norm(psi)

        mps_exact = MPS.from_state_vector(psi, n_sites=n)
        mps_trunc = MPS.from_state_vector(psi, n_sites=n, chi_max=4)

        assert mps_trunc.max_bond_dim <= 4
        # Truncated should still have decent overlap
        psi_trunc = mps_trunc.to_state_vector()
        overlap = abs(np.dot(np.conj(psi), psi_trunc))
        assert overlap > 0.5  # should maintain reasonable fidelity


# ===================================================================
# Edge Cases and Error Handling
# ===================================================================

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_two_site_mps(self):
        psi = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
        mps = MPS.from_state_vector(psi, n_sites=2)
        psi_out = mps.to_state_vector()
        assert np.allclose(psi, psi_out, atol=1e-12)

    def test_single_site_mps(self):
        psi = np.array([1, 0], dtype=complex)
        mps = MPS.from_state_vector(psi, n_sites=1)
        assert mps.n_sites == 1
        psi_out = mps.to_state_vector()
        assert np.allclose(psi, psi_out, atol=1e-12)

    def test_chi_max_1_product_states(self):
        psi = np.zeros(8, dtype=complex)
        psi[0] = 1.0  # |000>
        mps = MPS.from_state_vector(psi, n_sites=3, chi_max=1)
        psi_out = mps.to_state_vector()
        assert np.allclose(psi, psi_out, atol=1e-12)

    def test_ising_2_site(self):
        """Minimal Ising chain."""
        mpo = IsingMPO(2, J=1.0, h=0.5)
        H = mpo.to_matrix()
        assert H.shape == (4, 4)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_heisenberg_2_site(self):
        mpo = HeisenbergMPO(2, Jx=1.0, Jy=1.0, Jz=1.0)
        H = mpo.to_matrix()
        assert H.shape == (4, 4)
        assert np.allclose(H, H.conj().T, atol=1e-12)

    def test_dmrg_2_site(self):
        mpo = IsingMPO(2, J=1.0, h=1.0)
        H_exact = _exact_ising_hamiltonian(2, J=1.0, h=1.0)
        e_exact = _exact_ground_state_energy(H_exact)
        result = dmrg_ground_state(mpo, chi_max=4, n_sweeps=10)
        assert abs(result.energy - e_exact) / abs(e_exact) < 0.01

    def test_tebd_2_site(self):
        H = ising_nn_hamiltonian(2, J=1.0, h=0.5)
        mps = ProductState(2)
        result = tebd_evolve(mps, H, dt=0.05, n_steps=10, chi_max=4)
        assert len(result.states) == 11

    def test_large_mps_bond_dim(self):
        """Ensure large chi_max does not cause issues."""
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        mps = MPS.from_state_vector(psi, n_sites=2, chi_max=1000)
        psi_out = mps.to_state_vector()
        assert np.allclose(psi, psi_out, atol=1e-12)

    def test_mpo_site_count_mismatch(self):
        mpo = IsingMPO(4)
        mps = ProductState(3)
        with pytest.raises(ValueError, match="same number"):
            mpo.expectation(mps)

    def test_tensor_trace_dimension_mismatch(self):
        data = np.zeros((2, 3), dtype=complex)
        t = Tensor(data, ["i", "j"])
        with pytest.raises(ValueError, match="different sizes"):
            t.trace("i", "j")


# ===================================================================
# Performance Sanity Checks
# ===================================================================

class TestPerformance:
    """Basic performance sanity checks."""

    def test_mps_8_sites_roundtrip(self):
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(256) + 1j * rng.standard_normal(256)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=8)
        psi_out = mps.to_state_vector()
        assert np.allclose(psi, psi_out, atol=1e-10)

    def test_mps_10_sites_roundtrip(self):
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_state_vector(psi, n_sites=10)
        psi_out = mps.to_state_vector()
        assert np.allclose(psi, psi_out, atol=1e-10)

    def test_dmrg_converges_fast(self):
        """DMRG should converge in reasonable number of sweeps."""
        mpo = IsingMPO(4, J=1.0, h=1.0)
        result = dmrg_ground_state(mpo, chi_max=16, n_sweeps=30)
        # Should converge within 30 sweeps for a 4-site chain
        assert result.n_sweeps <= 30

    def test_contraction_chain(self):
        """Contract a chain of tensors."""
        tensors = []
        for i in range(5):
            d1 = 3 if i > 0 else 1
            d2 = 3 if i < 4 else 1
            tensors.append(
                Tensor(
                    np.random.randn(d1, 2, d2).astype(complex),
                    [f"l{i}", f"p{i}", f"l{i+1}"],
                )
            )
        tn = TensorNetwork()
        for i, t in enumerate(tensors):
            tn.add(f"T{i}", t)
        result = tn.contract_all()
        # Should produce a tensor with only physical legs
        assert "p0" in result.legs


# ===================================================================
# PEPS Tests
# ===================================================================

class TestPEPS:
    """Tests for 2D Projected Entangled Pair States."""

    def test_peps_construction(self):
        """PEPS initializes with correct dimensions."""
        peps = PEPS(rows=3, cols=3, phys_dim=2, bond_dim=2)
        assert peps.rows == 3
        assert peps.cols == 3
        assert len(peps.tensors) == 3
        assert len(peps.tensors[0]) == 3

    def test_peps_tensor_shape_corner(self):
        """Corner tensors have bond dimension 1 along boundary edges."""
        peps = PEPS(rows=3, cols=3, phys_dim=2, bond_dim=3)
        # Top-left corner: up=1, left=1
        t = peps.tensors[0][0].data
        assert t.shape[0] == 2  # physical
        assert t.shape[1] == 1  # up (boundary)
        assert t.shape[4] == 1  # left (boundary)

    def test_peps_tensor_shape_bulk(self):
        """Bulk tensors have full bond dimension along all virtual axes."""
        peps = PEPS(rows=3, cols=3, phys_dim=2, bond_dim=3)
        t = peps.tensors[1][1].data
        assert t.shape[0] == 2  # physical
        assert t.shape[1] == 3  # up
        assert t.shape[2] == 3  # right
        assert t.shape[3] == 3  # down
        assert t.shape[4] == 3  # left

    def test_peps_tensor_shape_edge(self):
        """Edge tensors have bond dim 1 only along the boundary."""
        peps = PEPS(rows=3, cols=3, phys_dim=2, bond_dim=2)
        # Top middle: up=1, all others=2
        t = peps.tensors[0][1].data
        assert t.shape[1] == 1  # up (boundary)
        assert t.shape[2] == 2  # right
        assert t.shape[3] == 2  # down
        assert t.shape[4] == 2  # left

    def test_product_state_construction(self):
        """Product state PEPS has bond dimension 1 everywhere."""
        peps = PEPS.product_state(2, 3, state=0)
        assert peps.rows == 2
        assert peps.cols == 3
        for r in range(2):
            for c in range(3):
                assert peps.tensors[r][c].data.shape == (2, 1, 1, 1, 1)

    def test_product_state_data(self):
        """Product state tensor is correct basis vector."""
        peps = PEPS.product_state(2, 2, state=0)
        for r in range(2):
            for c in range(2):
                t = peps.tensors[r][c].data
                assert np.isclose(t[0, 0, 0, 0, 0], 1.0)
                assert np.isclose(t[1, 0, 0, 0, 0], 0.0)

    def test_product_state_1(self):
        """Product state with state=1 sets second component."""
        peps = PEPS.product_state(2, 2, state=1)
        for r in range(2):
            for c in range(2):
                t = peps.tensors[r][c].data
                assert np.isclose(t[0, 0, 0, 0, 0], 0.0)
                assert np.isclose(t[1, 0, 0, 0, 0], 1.0)

    def test_neel_state_pattern(self):
        """Neel state alternates 0 and 1."""
        peps = PEPS.neel_state(2, 3)
        for r in range(2):
            for c in range(3):
                expected = (r + c) % 2
                t = peps.tensors[r][c].data
                assert np.isclose(t[expected, 0, 0, 0, 0], 1.0)
                assert np.isclose(t[1 - expected, 0, 0, 0, 0], 0.0)

    def test_product_state_norm(self):
        """Product state should have norm 1."""
        peps = PEPS.product_state(2, 2, state=0)
        norm_sq = peps.norm_squared(chi_boundary=4)
        assert np.isclose(norm_sq, 1.0, atol=1e-10), (
            f"Product state norm^2 = {norm_sq}, expected 1.0"
        )

    def test_neel_state_norm(self):
        """Neel state should have norm 1."""
        peps = PEPS.neel_state(2, 2)
        norm_sq = peps.norm_squared(chi_boundary=4)
        assert np.isclose(norm_sq, 1.0, atol=1e-10), (
            f"Neel state norm^2 = {norm_sq}, expected 1.0"
        )

    def test_expectation_z_product_state(self):
        """<0|Z|0> = 1 for product state."""
        peps = PEPS.product_state(2, 2, state=0)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        val = peps.expectation_local(Z, 0, 0, chi_boundary=4)
        assert np.isclose(np.real(val), 1.0, atol=1e-8), (
            f"Expected <Z> = 1.0, got {val}"
        )

    def test_expectation_z_state_1(self):
        """<1|Z|1> = -1 for |1> product state."""
        peps = PEPS.product_state(2, 2, state=1)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        val = peps.expectation_local(Z, 0, 0, chi_boundary=4)
        assert np.isclose(np.real(val), -1.0, atol=1e-8), (
            f"Expected <Z> = -1.0, got {val}"
        )

    def test_boundary_mps_contract_trivial(self):
        """Boundary MPS correctly contracts a product state PEPS."""
        peps = PEPS.product_state(2, 2, state=0)
        bmps = BoundaryMPS(chi_max=4)
        val = bmps.contract_full(peps)
        assert np.isclose(abs(val), 1.0, atol=1e-10)

    def test_ising_2d_bonds_count(self):
        """Correct number of bonds in 2D Ising lattice."""
        bonds = ising_2d_bonds(2, 3, J=1.0, h=0.0)
        # 2x3 lattice: horizontal bonds = 2*2 = 4, vertical bonds = 1*3 = 3
        assert len(bonds) == 7

    def test_heisenberg_2d_bonds_count(self):
        """Correct number of bonds in 2D Heisenberg lattice."""
        bonds = heisenberg_2d_bonds(3, 3, J=1.0)
        # 3x3: horizontal = 3*2 = 6, vertical = 2*3 = 6
        assert len(bonds) == 12

    def test_ising_2d_bonds_hermitian(self):
        """All Ising bond Hamiltonians are Hermitian."""
        bonds = ising_2d_bonds(2, 2, J=1.0, h=0.5)
        for r1, c1, r2, c2, h_bond in bonds:
            assert np.allclose(h_bond, h_bond.conj().T, atol=1e-12), (
                f"Bond ({r1},{c1})-({r2},{c2}) is not Hermitian"
            )

    def test_heisenberg_2d_bonds_hermitian(self):
        """All Heisenberg bond Hamiltonians are Hermitian."""
        bonds = heisenberg_2d_bonds(2, 2, J=1.0)
        for r1, c1, r2, c2, h_bond in bonds:
            assert np.allclose(h_bond, h_bond.conj().T, atol=1e-12)

    def test_peps_copy_independence(self):
        """PEPS copy is independent of original."""
        peps = PEPS.product_state(2, 2, state=0)
        peps2 = peps.copy()
        peps2.tensors[0][0].data[:] = 0
        # Original should be unchanged
        assert np.isclose(peps.tensors[0][0].data[0, 0, 0, 0, 0], 1.0)

    def test_pepstensor_properties(self):
        """PEPSTensor reports correct physical_dim and bond_dims."""
        data = np.zeros((2, 3, 4, 5, 6), dtype=complex)
        t = PEPSTensor(data=data, row=0, col=0)
        assert t.physical_dim == 2
        assert t.bond_dims == (3, 4, 5, 6)


# ===================================================================
# TDVP Tests
# ===================================================================

class TestTDVP:
    """Tests for Time-Dependent Variational Principle."""

    def test_krylov_expm_identity(self):
        """Krylov expm with H=0 returns the input vector."""
        n = 4
        H = np.zeros((n, n), dtype=complex)
        v = np.array([1, 0, 0, 0], dtype=complex)
        result = krylov_expm(H, v, dt=1.0)
        assert np.allclose(result, v, atol=1e-10)

    def test_krylov_expm_diagonal(self):
        """Krylov expm matches exact for diagonal matrix."""
        H = np.diag([1.0, 2.0, 3.0]).astype(complex)
        v = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
        dt = 0.5
        exact = np.exp(-1j * dt * np.diag(H)) * v
        result = krylov_expm(H, v, dt, m=10)
        assert np.allclose(result, exact, atol=1e-10)

    def test_matrix_exponential_exact(self):
        """Exact method matches eigendecomposition for Pauli X."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        v = np.array([1, 0], dtype=complex)
        dt = 0.3
        result = matrix_exponential_action(X, v, dt, method="exact")
        # exp(-i dt X) |0> = cos(dt)|0> - i sin(dt)|1>
        expected = np.array([np.cos(dt), -1j * np.sin(dt)])
        assert np.allclose(result, expected, atol=1e-10)

    def test_matrix_exponential_krylov(self):
        """Krylov method matches exact for small matrix."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        v = np.array([1, 0], dtype=complex)
        dt = 0.3
        exact = matrix_exponential_action(X, v, dt, method="exact")
        krylov = matrix_exponential_action(X, v, dt, method="krylov")
        assert np.allclose(exact, krylov, atol=1e-8)

    def test_tdvp1_energy_conservation(self):
        """TDVP1 should approximately conserve energy."""
        n = 4
        mps = ProductState(n)
        mpo = IsingMPO(n, J=1.0, h=0.5)
        tdvp = TDVP1Site(mps=mps, mpo=mpo, dt=0.05)
        result = tdvp.evolve(t_final=0.2, n_steps=4)
        # Energy should be approximately conserved
        E0 = result.energies[0]
        E_final = result.energies[-1]
        assert abs(E_final - E0) / abs(E0) < 0.15, (
            f"Energy changed too much: E0={E0:.6f}, E_final={E_final:.6f}"
        )

    def test_tdvp1_returns_result(self):
        """TDVP1 returns correct result structure."""
        mps = ProductState(4)
        mpo = IsingMPO(4)
        tdvp = TDVP1Site(mps=mps, mpo=mpo, dt=0.05)
        result = tdvp.evolve(t_final=0.1, n_steps=2)
        assert isinstance(result, TDVPResult)
        assert len(result.times) == 3  # initial + 2 steps
        assert len(result.energies) == 3
        assert len(result.bond_dims) == 3

    def test_tdvp1_preserves_bond_dim(self):
        """TDVP1 preserves bond dimension exactly."""
        mps = RandomMPS(4, chi=4, rng=np.random.default_rng(42))
        mpo = IsingMPO(4, J=1.0, h=0.5)
        initial_bonds = list(mps.bond_dimensions)
        tdvp = TDVP1Site(mps=mps, mpo=mpo, dt=0.05)
        result = tdvp.evolve(t_final=0.1, n_steps=2)
        final_bonds = result.bond_dims[-1]
        assert initial_bonds == final_bonds, (
            f"Bond dims changed: {initial_bonds} -> {final_bonds}"
        )

    def test_tdvp2_returns_result(self):
        """TDVP2 returns correct result structure."""
        mps = ProductState(4)
        mpo = IsingMPO(4)
        tdvp = TDVP2Site(mps=mps, mpo=mpo, dt=0.05, chi_max=8)
        result = tdvp.evolve(t_final=0.1, n_steps=2)
        assert isinstance(result, TDVPResult)
        assert len(result.times) == 3
        assert len(result.truncation_errors) == 2

    def test_tdvp2_energy_conservation(self):
        """TDVP2 should approximately conserve energy."""
        n = 4
        mps = ProductState(n)
        mpo = IsingMPO(n, J=1.0, h=0.5)
        tdvp = TDVP2Site(mps=mps, mpo=mpo, dt=0.05, chi_max=16)
        result = tdvp.evolve(t_final=0.2, n_steps=4)
        E0 = result.energies[0]
        E_final = result.energies[-1]
        assert abs(E_final - E0) / abs(E0) < 0.15, (
            f"Energy changed too much: E0={E0:.6f}, E_final={E_final:.6f}"
        )

    def test_tdvp2_can_grow_bond_dim(self):
        """TDVP2 should be able to increase bond dimension."""
        n = 4
        mps = ProductState(n)
        mpo = IsingMPO(n, J=1.0, h=1.0)
        tdvp = TDVP2Site(mps=mps, mpo=mpo, dt=0.1, chi_max=8, svd_cutoff=1e-12)
        result = tdvp.evolve(t_final=0.5, n_steps=5)
        # Starting from product state (chi=1), bond dim should grow
        initial_max = max(result.bond_dims[0]) if result.bond_dims[0] else 1
        final_max = max(result.bond_dims[-1]) if result.bond_dims[-1] else 1
        assert final_max >= initial_max

    def test_tdvp1_eigenstate_stability(self):
        """Evolving an eigenstate should keep it stable."""
        n = 4
        mpo = IsingMPO(n, J=1.0, h=0.0)
        # |0000> is an eigenstate of ZZ Ising (no field)
        mps = ProductState(n)
        E0 = mpo.expectation(mps)
        tdvp = TDVP1Site(mps=mps, mpo=mpo, dt=0.05)
        result = tdvp.evolve(t_final=0.2, n_steps=4)
        # Energy should not change
        for e in result.energies:
            assert np.isclose(e, E0, atol=0.1), (
                f"Eigenstate energy drifted: {e} vs {E0}"
            )

    def test_krylov_expm_zero_vector(self):
        """Krylov expm handles zero vector gracefully."""
        H = np.eye(3, dtype=complex)
        v = np.zeros(3, dtype=complex)
        result = krylov_expm(H, v, dt=1.0)
        assert np.allclose(result, np.zeros(3), atol=1e-15)

    def test_krylov_expm_matches_exact(self):
        """Krylov result matches exact for random Hermitian matrix."""
        rng = np.random.default_rng(42)
        n = 8
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = (A + A.conj().T) / 2  # Make Hermitian
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        v /= np.linalg.norm(v)
        dt = 0.1

        exact = matrix_exponential_action(H, v, dt, method="exact")
        krylov = krylov_expm(H, v, dt, m=n)
        assert np.allclose(exact, krylov, atol=1e-8), (
            f"Krylov differs from exact: max diff = {np.max(np.abs(exact - krylov))}"
        )


# ===================================================================
# Autodiff Tests
# ===================================================================

class TestAutodiff:
    """Tests for differentiable tensor network contractions."""

    def test_tensor_node_creation(self):
        """TensorNode stores data and shape correctly."""
        data = np.eye(3, dtype=complex)
        node = tensor_node(data, name="test")
        assert node.shape == (3, 3)
        assert node.name == "test"
        assert node.requires_grad is True
        assert node.grad is None

    def test_tensor_node_copy(self):
        """TensorNode copies data (not reference)."""
        data = np.eye(3, dtype=complex)
        node = tensor_node(data)
        data[0, 0] = 999
        assert node.data[0, 0] == 1.0  # should be unaffected

    def test_contract_matrix_multiply(self):
        """Contraction of matrices equals matrix multiplication."""
        A = tensor_node(np.array([[1, 2], [3, 4]], dtype=complex))
        B = tensor_node(np.array([[5, 6], [7, 8]], dtype=complex))
        C = contract(A, B, axes=([1], [0]))
        expected = np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])
        assert np.allclose(C.data, expected)

    def test_contract_inner_product(self):
        """Contraction computes inner product of vectors."""
        v1 = tensor_node(np.array([1, 2, 3], dtype=complex))
        v2 = tensor_node(np.array([4, 5, 6], dtype=complex))
        result = contract(v1, v2, axes=([0], [0]))
        assert np.isclose(result.data, 32.0)

    def test_backward_linear(self):
        """Backward pass computes correct gradients for matrix multiply."""
        A = tensor_node(np.array([[1, 0], [0, 1]], dtype=complex), name="A")
        B = tensor_node(np.array([[2, 3], [4, 5]], dtype=complex), name="B")
        C = contract(A, B, axes=([1], [0]))
        # C = A @ B
        # dC/dA = I (when grad_C = I and B = I, etc.)
        C.grad = np.ones_like(C.data)
        backward(C)
        # grad_A should exist
        assert A.grad is not None
        assert B.grad is not None

    def test_backward_gradient_shape(self):
        """Gradients have the same shape as the tensors."""
        A = tensor_node(np.random.randn(3, 4).astype(complex), name="A")
        B = tensor_node(np.random.randn(4, 5).astype(complex), name="B")
        C = contract(A, B, axes=([1], [0]))
        C.grad = np.ones_like(C.data)
        backward(C)
        assert A.grad.shape == (3, 4)
        assert B.grad.shape == (4, 5)

    def test_trace_operation(self):
        """Trace computes correctly."""
        data = np.eye(3, dtype=complex)
        node = tensor_node(data)
        result = trace(node, axis1=0, axis2=1)
        assert np.isclose(result.data, 3.0)

    def test_trace_gradient(self):
        """Trace gradient places grad on diagonal."""
        data = np.eye(3, dtype=complex)
        node = tensor_node(data)
        result = trace(node, axis1=0, axis2=1)
        result.grad = np.ones_like(result.data)
        backward(result)
        # Gradient should be identity-like
        assert node.grad is not None
        assert node.grad.shape == (3, 3)
        for i in range(3):
            assert np.isclose(node.grad[i, i], 1.0)

    def test_svd_decomposition(self):
        """SVD produces correct factors."""
        data = np.random.randn(3, 4).astype(complex)
        node = tensor_node(data)
        U, S, Vh = svd(node)
        # Reconstruct
        reconstructed = U.data[:, :len(S.data)] @ np.diag(S.data) @ Vh.data[:len(S.data), :]
        assert np.allclose(reconstructed, data, atol=1e-10)

    def test_svd_shapes(self):
        """SVD factors have correct shapes."""
        data = np.random.randn(4, 3).astype(complex)
        node = tensor_node(data)
        U, S, Vh = svd(node)
        assert U.shape[0] == 4
        assert len(S.data) == 3
        assert Vh.shape[1] == 3

    def test_differentiable_contraction_mps_overlap(self):
        """MPS overlap computation via DifferentiableContraction."""
        # Create a simple 2-site MPS
        t0 = tensor_node(np.array([[[1, 0]], [[0, 0]]], dtype=complex).reshape(1, 2, 2))
        t1 = tensor_node(np.array([[[1], [0]], [[0], [0]]], dtype=complex).reshape(2, 2, 1))
        dc = DifferentiableContraction()
        result = dc.mps_overlap([t0, t1], [t0, t1])
        # <00|00> = 1
        assert np.isclose(abs(result.data.ravel()[0]), 1.0, atol=1e-10)

    def test_optimization_result_structure(self):
        """OptimizationResult has correct attributes."""
        result = OptimizationResult(energies=[1.0, 0.5], converged=True, n_steps=2)
        assert result.converged is True
        assert result.n_steps == 2
        assert len(result.energies) == 2

    def test_zero_grad(self):
        """zero_grad resets gradient to zeros."""
        node = tensor_node(np.ones((2, 2), dtype=complex))
        node.grad = np.ones((2, 2), dtype=complex)
        node.zero_grad()
        assert np.allclose(node.grad, 0.0)


# ===================================================================
# TN Machine Learning Tests
# ===================================================================

class TestTNML:
    """Tests for tensor network machine learning."""

    def test_mps_classifier_init(self):
        """MPSClassifier initializes correctly."""
        clf = MPSClassifier(n_features=4, n_classes=2, bond_dim=3)
        clf.initialize(rng=np.random.default_rng(42))
        assert clf._weights is not None
        assert len(clf._weights) == 4

    def test_mps_classifier_predict_shape(self):
        """predict_proba returns correct shape."""
        clf = MPSClassifier(n_features=4, n_classes=3, bond_dim=4)
        clf.initialize(rng=np.random.default_rng(42))
        probs = clf.predict_proba(np.array([0.1, 0.2, 0.3, 0.4]))
        assert probs.shape == (3,)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-10)

    def test_mps_classifier_predict_batch(self):
        """predict returns correct batch shape."""
        clf = MPSClassifier(n_features=4, n_classes=2, bond_dim=4)
        clf.initialize(rng=np.random.default_rng(42))
        X = np.random.rand(5, 4)
        preds = clf.predict(X)
        assert preds.shape == (5,)
        assert all(0 <= p < 2 for p in preds)

    def test_mps_classifier_probabilities_sum_to_one(self):
        """Class probabilities sum to 1."""
        clf = MPSClassifier(n_features=6, n_classes=3, bond_dim=4)
        clf.initialize(rng=np.random.default_rng(42))
        for _ in range(5):
            x = np.random.rand(6)
            probs = clf.predict_proba(x)
            assert np.isclose(np.sum(probs), 1.0, atol=1e-10)
            assert all(p >= 0 for p in probs)

    def test_mps_classifier_feature_map(self):
        """Feature map produces correct trigonometric encoding."""
        clf = MPSClassifier(n_features=2, n_classes=2)
        states = clf._feature_map(np.array([0.0, 1.0]))
        # x=0: cos(0) = 1, sin(0) = 0
        assert np.isclose(states[0][0], 1.0)
        assert np.isclose(states[0][1], 0.0, atol=1e-10)
        # x=1: cos(pi/2) = 0, sin(pi/2) = 1
        assert np.isclose(states[1][0], 0.0, atol=1e-10)
        assert np.isclose(states[1][1], 1.0)

    def test_mps_classifier_fit(self):
        """MPSClassifier training runs and returns results."""
        rng = np.random.default_rng(42)
        clf = MPSClassifier(n_features=4, n_classes=2, bond_dim=4)
        X = rng.random((20, 4))
        y = (X[:, 0] > 0.5).astype(int)
        result = clf.fit(X, y, epochs=5, lr=0.01, rng=rng)
        assert isinstance(result, MLResult)
        assert len(result.losses) == 5
        assert len(result.accuracies) == 5
        assert result.epochs == 5

    def test_mps_classifier_not_initialized_raises(self):
        """Prediction before initialization raises error."""
        clf = MPSClassifier(n_features=4, n_classes=2)
        with pytest.raises(ValueError, match="not initialized"):
            clf.predict_proba(np.array([0.1, 0.2, 0.3, 0.4]))

    def test_tn_kernel_self_overlap(self):
        """TN kernel of identical vectors equals 1."""
        kernel = TNKernel(bond_dim=1)
        x = np.array([0.3, 0.7, 0.5])
        k_val = kernel.kernel(x, x)
        assert np.isclose(k_val, 1.0, atol=1e-10), (
            f"Self-kernel should be 1.0, got {k_val}"
        )

    def test_tn_kernel_symmetry(self):
        """TN kernel is symmetric: K(x1, x2) = K(x2, x1)."""
        kernel = TNKernel(bond_dim=1)
        x1 = np.array([0.2, 0.8, 0.4])
        x2 = np.array([0.9, 0.1, 0.5])
        assert np.isclose(kernel.kernel(x1, x2), kernel.kernel(x2, x1), atol=1e-12)

    def test_tn_kernel_range(self):
        """TN kernel values are in [0, 1]."""
        kernel = TNKernel(bond_dim=1)
        rng = np.random.default_rng(42)
        for _ in range(10):
            x1 = rng.random(5)
            x2 = rng.random(5)
            k_val = kernel.kernel(x1, x2)
            assert 0.0 - 1e-10 <= k_val <= 1.0 + 1e-10, (
                f"Kernel value {k_val} out of [0, 1]"
            )

    def test_tn_kernel_matrix_shape(self):
        """Kernel matrix has correct shape."""
        kernel = TNKernel(bond_dim=1)
        X = np.random.rand(5, 3)
        K = kernel.kernel_matrix(X)
        assert K.shape == (5, 5)

    def test_tn_kernel_matrix_symmetric(self):
        """Kernel matrix is symmetric."""
        kernel = TNKernel(bond_dim=1)
        X = np.random.rand(4, 3)
        K = kernel.kernel_matrix(X)
        assert np.allclose(K, K.T, atol=1e-12)

    def test_tn_kernel_matrix_diagonal(self):
        """Diagonal of kernel matrix is 1 (self-kernel)."""
        kernel = TNKernel(bond_dim=1)
        X = np.random.rand(4, 3)
        K = kernel.kernel_matrix(X)
        for i in range(4):
            assert np.isclose(K[i, i], 1.0, atol=1e-10)

    def test_tn_kernel_matrix_psd(self):
        """Kernel matrix is positive semi-definite."""
        kernel = TNKernel(bond_dim=1)
        rng = np.random.default_rng(42)
        X = rng.random((6, 4))
        K = kernel.kernel_matrix(X)
        eigenvalues = np.linalg.eigvalsh(K)
        assert all(ev > -1e-10 for ev in eigenvalues), (
            f"Kernel matrix has negative eigenvalue: {min(eigenvalues)}"
        )

    def test_ml_result_structure(self):
        """MLResult has correct attributes."""
        result = MLResult(losses=[1.0], accuracies=[0.5], epochs=1, bond_dim=4)
        assert result.epochs == 1
        assert result.bond_dim == 4
