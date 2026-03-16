"""TDVP: Time-Dependent Variational Principle for MPS time evolution.

Implements the one-site and two-site TDVP algorithms for evolving
Matrix Product States under a Hamiltonian given as an MPO. TDVP
projects the Schrodinger equation onto the MPS manifold, yielding
a symplectic integrator that preserves the variational structure.

One-site TDVP (TDVP1) preserves the bond dimension exactly, while
two-site TDVP (TDVP2) can dynamically adapt the bond dimension.

The key operation at each step is computing exp(-i dt H_eff) |v>
where H_eff is the effective Hamiltonian in the local tangent space.
This is done either via exact diagonalization (small systems) or
the Krylov (Lanczos) subspace method (larger systems).

Reference: Haegeman et al., "Time-Dependent Variational Principle for
Quantum Lattices", Phys. Rev. B 84, 165139 (2011).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .mps import MPS
from .mpo import MPO


# -------------------------------------------------------------------
# Result container
# -------------------------------------------------------------------

@dataclass
class TDVPResult:
    """Result of TDVP time evolution.

    Attributes
    ----------
    times : ndarray
        Array of time values at each recorded step.
    energies : ndarray
        Energy <psi|H|psi> at each recorded step.
    bond_dims : list[list[int]]
        Bond dimensions at each recorded step.
    truncation_errors : list[float]
        Truncation errors per step (nonzero only for TDVP2).
    """
    times: np.ndarray
    energies: np.ndarray
    bond_dims: List[List[int]]
    truncation_errors: List[float]


# -------------------------------------------------------------------
# Matrix exponential helpers
# -------------------------------------------------------------------

def matrix_exponential_action(H_eff: np.ndarray, v: np.ndarray, dt: float,
                              method: str = "exact") -> np.ndarray:
    """Compute exp(-i dt H) |v> using the specified method.

    Parameters
    ----------
    H_eff : ndarray
        Effective Hamiltonian matrix.
    v : ndarray
        Vector to act on.
    dt : float
        Time step (the exponent is -i * dt * H_eff).
    method : str
        "exact" for dense eigendecomposition, "krylov" for Lanczos.

    Returns
    -------
    ndarray
        The evolved vector exp(-i dt H_eff) |v>.
    """
    if method == "krylov":
        return krylov_expm(H_eff, v, dt)
    else:
        return _exact_expm_action(H_eff, v, dt)


def _exact_expm_action(H_eff: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """Exact exp(-i dt H) |v> via eigendecomposition.

    For small effective Hamiltonians this is perfectly adequate.
    """
    H = np.asarray(H_eff, dtype=np.complex128)
    v = np.asarray(v, dtype=np.complex128).ravel()
    n = H.shape[0]

    if n == 0:
        return v.copy()
    if n == 1:
        return v * np.exp(-1j * dt * H[0, 0])

    # Use eigh for Hermitian matrices (much more stable)
    # Check if H is approximately Hermitian
    if np.allclose(H, H.conj().T, atol=1e-10):
        evals, evecs = np.linalg.eigh(H)
        # exp(-i dt H) |v> = V diag(exp(-i dt lambda)) V^dag |v>
        coeffs = evecs.conj().T @ v
        return evecs @ (np.exp(-1j * dt * evals) * coeffs)
    else:
        # Non-Hermitian: use general eigendecomposition
        evals, evecs = np.linalg.eig(H)
        coeffs = np.linalg.solve(evecs, v)
        return evecs @ (np.exp(-1j * dt * evals) * coeffs)


def krylov_expm(H_eff: np.ndarray, v: np.ndarray, dt: float,
                m: int = 20) -> np.ndarray:
    """Krylov subspace method for exp(-i dt H) |v>.

    Uses the Lanczos algorithm to build a small Krylov subspace and
    computes the matrix exponential in that reduced basis.

    Parameters
    ----------
    H_eff : ndarray
        Effective Hamiltonian matrix (square).
    v : ndarray
        Starting vector.
    dt : float
        Time step.
    m : int
        Krylov subspace dimension (default 20).

    Returns
    -------
    ndarray
        Approximate result of exp(-i dt H_eff) |v>.
    """
    H = np.asarray(H_eff, dtype=np.complex128)
    v = np.asarray(v, dtype=np.complex128).ravel()
    n = len(v)

    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:
        return np.zeros_like(v)

    m = min(m, n)
    V = np.zeros((m + 1, n), dtype=np.complex128)
    alpha = np.zeros(m, dtype=np.float64)
    beta = np.zeros(m + 1, dtype=np.float64)

    V[0] = v / norm_v
    w = H @ V[0]
    alpha[0] = np.real(np.dot(np.conj(V[0]), w))
    w = w - alpha[0] * V[0]

    actual_m = m
    for j in range(1, m):
        beta[j] = np.linalg.norm(w)
        if beta[j] < 1e-14:
            actual_m = j
            break
        V[j] = w / beta[j]
        w = H @ V[j]
        w -= beta[j] * V[j - 1]
        alpha[j] = np.real(np.dot(np.conj(V[j]), w))
        w -= alpha[j] * V[j]

        # Reorthogonalize
        for k in range(j + 1):
            w -= np.dot(np.conj(V[k]), w) * V[k]

    # Build tridiagonal matrix and exponentiate
    T = np.diag(alpha[:actual_m])
    if actual_m > 1:
        T += np.diag(beta[1:actual_m], 1) + np.diag(beta[1:actual_m], -1)

    evals, evecs = np.linalg.eigh(T)
    # exp(-i dt T) @ e1
    e1 = np.zeros(actual_m, dtype=np.complex128)
    e1[0] = 1.0
    exp_coeffs = evecs @ (np.exp(-1j * dt * evals) * (evecs.conj().T @ e1))

    # Transform back to full space
    result = norm_v * (V[:actual_m].T @ exp_coeffs)
    return result


# -------------------------------------------------------------------
# Environment building (shared between TDVP1 and TDVP2)
# -------------------------------------------------------------------

def _build_left_envs(mps: MPS, mpo: MPO) -> List[np.ndarray]:
    """Build all left environment tensors L[0]..L[n].

    L[i] has shape (chi_bra, chi_mpo, chi_ket) and represents the
    contraction of sites 0..i-1.
    """
    n = mps.n_sites
    envs = [None] * (n + 1)
    envs[0] = np.ones((1, 1, 1), dtype=np.complex128)

    for i in range(n):
        envs[i + 1] = np.einsum(
            "abc,ase,bstf,ctg->efg",
            envs[i],
            np.conj(mps.tensors[i]),
            mpo.tensors[i],
            mps.tensors[i],
        )
    return envs


def _build_right_envs(mps: MPS, mpo: MPO) -> List[np.ndarray]:
    """Build all right environment tensors R[0]..R[n].

    R[i] has shape (chi_bra, chi_mpo, chi_ket) and represents the
    contraction of sites i..n-1.
    """
    n = mps.n_sites
    envs = [None] * (n + 1)
    envs[n] = np.ones((1, 1, 1), dtype=np.complex128)

    for i in range(n - 1, -1, -1):
        envs[i] = np.einsum(
            "ase,bstf,ctg,efg->abc",
            np.conj(mps.tensors[i]),
            mpo.tensors[i],
            mps.tensors[i],
            envs[i + 1],
        )
    return envs


def _effective_H_site(L: np.ndarray, W: np.ndarray, R: np.ndarray,
                      chi_l: int, d: int, chi_r: int) -> np.ndarray:
    """Build the effective Hamiltonian matrix for a single site.

    H_eff acts on a vector of shape (chi_l * d * chi_r,).

    L[a, b, c], W[b, s, t, f], R[e, f, g]
    H_eff maps theta[c, t, g] -> result[a, s, e]
    """
    dim = chi_l * d * chi_r
    H_eff = np.zeros((dim, dim), dtype=np.complex128)

    # Build by explicit contraction
    # H_eff[a,s,e, c,t,g] = L[a,b,c] * W[b,s,t,f] * R[e,f,g]
    full = np.einsum("abc,bstf,efg->asectg", L, W, R)
    H_eff = full.reshape(dim, dim)
    return H_eff


def _effective_H_bond(L: np.ndarray, R: np.ndarray,
                      chi_l: int, chi_r: int) -> np.ndarray:
    """Build the effective Hamiltonian for a bond matrix.

    The bond matrix C has shape (chi_l, chi_r), and H_eff_bond acts
    on its vectorized form.

    L[a,b,c] and R[e,f,g] where b and f are the MPO bond indices.
    H_eff_bond maps C[c,g] -> result[a,e] by contracting over MPO index.
    """
    dim = chi_l * chi_r
    # L[a,b,c] * R[e,b,g] -> full[a,e,c,g] (contract over MPO bond b)
    full = np.einsum("abc,ebg->aecg", L, R)
    return full.reshape(dim, dim)


def _effective_H_two_site(L: np.ndarray, W1: np.ndarray, W2: np.ndarray,
                          R: np.ndarray, chi_l: int, d: int,
                          chi_r: int) -> np.ndarray:
    """Build effective Hamiltonian for two-site TDVP.

    Acts on theta with shape (chi_l, d, d, chi_r).
    """
    dim = chi_l * d * d * chi_r
    # H_eff[a,s,t,e, c,u,v,g] = L[a,b,c] * W1[b,s,u,f] * W2[f,t,v,h] * R[e,h,g]
    full = np.einsum("abc,bsuf,ftvh,ehg->astecuvg", L, W1, W2, R)
    return full.reshape(dim, dim)


# -------------------------------------------------------------------
# TDVP1 (One-site)
# -------------------------------------------------------------------

@dataclass
class TDVP1Site:
    """One-site TDVP algorithm.

    Preserves bond dimension exactly. Evolves each site tensor forward
    in time, then evolves the bond matrix backward, alternating
    left-right sweeps. This produces a second-order integrator.

    Parameters
    ----------
    mps : MPS
        The initial MPS state (will be copied).
    mpo : MPO
        The Hamiltonian as an MPO.
    dt : float
        Time step (default 0.01).
    """
    mps: MPS
    mpo: MPO
    dt: float = 0.01

    def __post_init__(self):
        self.mps = self.mps.copy()
        self._n = self.mps.n_sites
        self._d = self.mps.d
        self._L_envs = None
        self._R_envs = None

    def _build_environments(self):
        """Build left and right environment tensors."""
        self._L_envs = _build_left_envs(self.mps, self.mpo)
        self._R_envs = _build_right_envs(self.mps, self.mpo)

    def _update_left_env(self, site: int):
        """Update left environment after modifying site tensor."""
        self._L_envs[site + 1] = np.einsum(
            "abc,ase,bstf,ctg->efg",
            self._L_envs[site],
            np.conj(self.mps.tensors[site]),
            self.mpo.tensors[site],
            self.mps.tensors[site],
        )

    def _update_right_env(self, site: int):
        """Update right environment after modifying site tensor."""
        self._R_envs[site] = np.einsum(
            "ase,bstf,ctg,efg->abc",
            np.conj(self.mps.tensors[site]),
            self.mpo.tensors[site],
            self.mps.tensors[site],
            self._R_envs[site + 1],
        )

    def _evolve_site(self, site: int, dt: float):
        """Evolve single site tensor forward in time by dt.

        Builds effective Hamiltonian H_eff from L, W, R environments
        and applies exp(-i dt H_eff) to the site tensor.
        """
        A = self.mps.tensors[site]
        chi_l, d, chi_r = A.shape
        v = A.ravel()

        L = self._L_envs[site]
        W = self.mpo.tensors[site]
        R = self._R_envs[site + 1]

        H_eff = _effective_H_site(L, W, R, chi_l, d, chi_r)
        v_new = _exact_expm_action(H_eff, v, dt)
        self.mps.tensors[site] = v_new.reshape(chi_l, d, chi_r)

    def _evolve_bond(self, bond: int, dt: float):
        """Evolve bond matrix backward in time by dt.

        The bond matrix C sits between sites 'bond' and 'bond+1'.
        We extract it via QR of the left site, evolve it backward
        with exp(+i dt H_eff_bond), then absorb it back.
        """
        A = self.mps.tensors[bond]
        chi_l, d, chi_r = A.shape

        # QR decomposition: A = Q * C
        mat = A.reshape(chi_l * d, chi_r)
        Q, C = np.linalg.qr(mat)
        new_chi = Q.shape[1]

        # Temporarily update site tensor to Q
        self.mps.tensors[bond] = Q.reshape(chi_l, d, new_chi)
        self._update_left_env(bond)

        # Evolve C backward (note: backward means +i dt, not -i dt)
        L = self._L_envs[bond + 1]
        R = self._R_envs[bond + 1]

        chi_c_l = C.shape[0]
        chi_c_r = C.shape[1]
        H_bond = _effective_H_bond(L, R, chi_c_l, chi_c_r)

        c_vec = C.ravel()
        # Backward evolution: exp(+i dt H)
        c_new = _exact_expm_action(H_bond, c_vec, -dt)
        C_new = c_new.reshape(chi_c_l, chi_c_r)

        # Absorb C into next site
        self.mps.tensors[bond + 1] = np.einsum(
            "ij,jsk->isk", C_new, self.mps.tensors[bond + 1]
        )

    def _evolve_bond_right(self, bond: int, dt: float):
        """Evolve bond matrix backward during right-to-left sweep.

        Uses RQ decomposition on the right site.
        """
        A = self.mps.tensors[bond + 1]
        chi_l, d, chi_r = A.shape

        # LQ decomposition: A = C * Q
        mat = A.reshape(chi_l, d * chi_r)
        Q, R = np.linalg.qr(mat.T)
        C = R.T  # (chi_l, new_chi)
        Qm = Q.T  # (new_chi, d * chi_r)
        new_chi = Qm.shape[0]

        # Temporarily update right site tensor
        self.mps.tensors[bond + 1] = Qm.reshape(new_chi, d, chi_r)
        self._update_right_env(bond + 1)

        # Evolve C backward
        L = self._L_envs[bond + 1]
        R = self._R_envs[bond + 1]

        chi_c_l = C.shape[0]
        chi_c_r = C.shape[1]
        H_bond = _effective_H_bond(L, R, chi_c_l, chi_c_r)

        c_vec = C.ravel()
        c_new = _exact_expm_action(H_bond, c_vec, -dt)
        C_new = c_new.reshape(chi_c_l, chi_c_r)

        # Absorb C into left site
        self.mps.tensors[bond] = np.einsum(
            "asd,de->ase", self.mps.tensors[bond], C_new
        )

    def sweep_right(self, dt: float):
        """Left-to-right TDVP sweep.

        For each site i = 0, 1, ..., n-2:
          1. Evolve site i forward by dt/2
          2. QR decompose and evolve bond backward by dt/2
        Then evolve site n-1 forward by dt/2.
        """
        n = self._n
        for i in range(n - 1):
            self._evolve_site(i, dt / 2)
            self._evolve_bond(i, dt / 2)
            self._update_left_env(i)

        # Last site: forward only
        self._evolve_site(n - 1, dt / 2)
        self._update_left_env(n - 1)

    def sweep_left(self, dt: float):
        """Right-to-left TDVP sweep.

        For each site i = n-1, n-2, ..., 1:
          1. Evolve site i forward by dt/2
          2. LQ decompose and evolve bond backward by dt/2
        Then evolve site 0 forward by dt/2.
        """
        n = self._n
        for i in range(n - 1, 0, -1):
            self._evolve_site(i, dt / 2)
            self._evolve_bond_right(i - 1, dt / 2)
            self._update_right_env(i)

        # First site: forward only
        self._evolve_site(0, dt / 2)
        self._update_right_env(0)

    def evolve(self, t_final: float, n_steps: int = None) -> TDVPResult:
        """Full TDVP1 time evolution from t=0 to t=t_final.

        Uses a second-order integrator: right sweep + left sweep per step.

        Parameters
        ----------
        t_final : float
            Total evolution time.
        n_steps : int, optional
            Number of time steps. If None, computed from self.dt.

        Returns
        -------
        TDVPResult
            Evolution results with energy and bond dimension history.
        """
        if n_steps is None:
            n_steps = max(1, int(np.ceil(t_final / self.dt)))
        dt = t_final / n_steps

        times = np.zeros(n_steps + 1)
        energies = np.zeros(n_steps + 1)
        bond_dims = []
        trunc_errors = []

        # Initial measurements
        self._build_environments()
        energies[0] = self.mpo.expectation(self.mps)
        bond_dims.append(list(self.mps.bond_dimensions))

        for step in range(n_steps):
            self._build_environments()
            self.sweep_right(dt)
            self._build_environments()
            self.sweep_left(dt)

            times[step + 1] = (step + 1) * dt
            energies[step + 1] = self.mpo.expectation(self.mps)
            bond_dims.append(list(self.mps.bond_dimensions))
            trunc_errors.append(0.0)  # TDVP1 has no truncation

        return TDVPResult(
            times=times,
            energies=energies,
            bond_dims=bond_dims,
            truncation_errors=trunc_errors,
        )


# -------------------------------------------------------------------
# TDVP2 (Two-site)
# -------------------------------------------------------------------

@dataclass
class TDVP2Site:
    """Two-site TDVP algorithm.

    Can dynamically grow bond dimension. More expensive but allows
    adaptation of the MPS bond structure. At each step, optimizes a
    two-site tensor then SVD-truncates to the maximum bond dimension.

    Parameters
    ----------
    mps : MPS
        The initial MPS state (will be copied).
    mpo : MPO
        The Hamiltonian as an MPO.
    dt : float
        Time step (default 0.01).
    chi_max : int
        Maximum bond dimension (default 32).
    svd_cutoff : float
        SVD truncation cutoff (default 1e-10).
    """
    mps: MPS
    mpo: MPO
    dt: float = 0.01
    chi_max: int = 32
    svd_cutoff: float = 1e-10

    def __post_init__(self):
        self.mps = self.mps.copy()
        self._n = self.mps.n_sites
        self._d = self.mps.d
        self._L_envs = None
        self._R_envs = None

    def _build_environments(self):
        """Build left and right environment tensors."""
        self._L_envs = _build_left_envs(self.mps, self.mpo)
        self._R_envs = _build_right_envs(self.mps, self.mpo)

    def _update_left_env(self, site: int):
        """Update left environment after modifying site tensor."""
        self._L_envs[site + 1] = np.einsum(
            "abc,ase,bstf,ctg->efg",
            self._L_envs[site],
            np.conj(self.mps.tensors[site]),
            self.mpo.tensors[site],
            self.mps.tensors[site],
        )

    def _update_right_env(self, site: int):
        """Update right environment after modifying site tensor."""
        self._R_envs[site] = np.einsum(
            "ase,bstf,ctg,efg->abc",
            np.conj(self.mps.tensors[site]),
            self.mpo.tensors[site],
            self.mps.tensors[site],
            self._R_envs[site + 1],
        )

    def _evolve_two_sites_right(self, site: int, dt: float) -> float:
        """Evolve two-site tensor for right sweep and truncate via SVD.

        Returns truncation error.
        """
        d = self._d
        A1 = self.mps.tensors[site]
        A2 = self.mps.tensors[site + 1]
        chi_l = A1.shape[0]
        chi_r = A2.shape[2]

        # Form two-site tensor
        theta = np.einsum("asd,dtc->astc", A1, A2)
        v = theta.ravel()

        # Build effective Hamiltonian
        L = self._L_envs[site]
        W1 = self.mpo.tensors[site]
        W2 = self.mpo.tensors[site + 1]
        R = self._R_envs[site + 2]

        H_eff = _effective_H_two_site(L, W1, W2, R, chi_l, d, chi_r)
        v_new = _exact_expm_action(H_eff, v, dt)
        theta_new = v_new.reshape(chi_l, d, d, chi_r)

        # SVD truncation
        mat = theta_new.reshape(chi_l * d, d * chi_r)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        keep = len(S)
        if self.svd_cutoff > 0:
            keep = max(int(np.sum(S > self.svd_cutoff)), 1)
        keep = min(keep, self.chi_max)

        trunc_error = float(np.sum(S[keep:] ** 2))
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        # Absorb S into right (orthogonality center moves right)
        self.mps.tensors[site] = U.reshape(chi_l, d, keep)
        self.mps.tensors[site + 1] = (np.diag(S) @ Vh).reshape(keep, d, chi_r)

        return trunc_error

    def _evolve_two_sites_left(self, site: int, dt: float) -> float:
        """Evolve two-site tensor for left sweep and truncate via SVD.

        Returns truncation error.
        """
        d = self._d
        A1 = self.mps.tensors[site]
        A2 = self.mps.tensors[site + 1]
        chi_l = A1.shape[0]
        chi_r = A2.shape[2]

        theta = np.einsum("asd,dtc->astc", A1, A2)
        v = theta.ravel()

        L = self._L_envs[site]
        W1 = self.mpo.tensors[site]
        W2 = self.mpo.tensors[site + 1]
        R = self._R_envs[site + 2]

        H_eff = _effective_H_two_site(L, W1, W2, R, chi_l, d, chi_r)
        v_new = _exact_expm_action(H_eff, v, dt)
        theta_new = v_new.reshape(chi_l, d, d, chi_r)

        mat = theta_new.reshape(chi_l * d, d * chi_r)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        keep = len(S)
        if self.svd_cutoff > 0:
            keep = max(int(np.sum(S > self.svd_cutoff)), 1)
        keep = min(keep, self.chi_max)

        trunc_error = float(np.sum(S[keep:] ** 2))
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        # Absorb S into left (orthogonality center moves left)
        self.mps.tensors[site] = (U @ np.diag(S)).reshape(chi_l, d, keep)
        self.mps.tensors[site + 1] = Vh.reshape(keep, d, chi_r)

        return trunc_error

    def evolve(self, t_final: float, n_steps: int = None) -> TDVPResult:
        """Full 2-site TDVP time evolution.

        Uses alternating right and left sweeps with two-site updates.

        Parameters
        ----------
        t_final : float
            Total evolution time.
        n_steps : int, optional
            Number of time steps. If None, computed from self.dt.

        Returns
        -------
        TDVPResult
            Evolution results.
        """
        if n_steps is None:
            n_steps = max(1, int(np.ceil(t_final / self.dt)))
        dt = t_final / n_steps

        times = np.zeros(n_steps + 1)
        energies = np.zeros(n_steps + 1)
        bond_dims = []
        trunc_errors = []

        self._build_environments()
        energies[0] = self.mpo.expectation(self.mps)
        bond_dims.append(list(self.mps.bond_dimensions))

        for step in range(n_steps):
            max_trunc = 0.0
            n = self._n

            # Right sweep
            self._build_environments()
            for i in range(n - 1):
                trunc = self._evolve_two_sites_right(i, dt / 2)
                max_trunc = max(max_trunc, trunc)
                self._update_left_env(i)

            # Left sweep
            self._build_environments()
            for i in range(n - 2, -1, -1):
                trunc = self._evolve_two_sites_left(i, dt / 2)
                max_trunc = max(max_trunc, trunc)
                self._update_right_env(i + 1)

            times[step + 1] = (step + 1) * dt
            energies[step + 1] = self.mpo.expectation(self.mps)
            bond_dims.append(list(self.mps.bond_dimensions))
            trunc_errors.append(max_trunc)

        return TDVPResult(
            times=times,
            energies=energies,
            bond_dims=bond_dims,
            truncation_errors=trunc_errors,
        )
