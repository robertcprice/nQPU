"""Density Matrix Renormalization Group (DMRG) ground-state solver.

Implements the two-site DMRG algorithm for finding ground states of
one-dimensional quantum systems expressed as MPOs.  The algorithm
sweeps left-right and right-left through the MPS chain, solving a
local eigenvalue problem at each bond using the Lanczos eigensolver,
then truncating with SVD to control the bond dimension.

Key features:
  - Two-site DMRG with dynamic bond dimension growth
  - Lanczos eigensolver for the local effective Hamiltonian
  - Left/right environment caching for efficient sweeps
  - Convergence tracking on energy and truncation error
  - Convenience function ``dmrg_ground_state`` for quick usage

References:
  - White, S.R., Phys. Rev. Lett. 69, 2863 (1992)
  - Schollwoeck, U., Ann. Phys. 326, 96 (2011)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .mps import MPS, ProductState
from .mpo import MPO


# -------------------------------------------------------------------
# Result container
# -------------------------------------------------------------------

@dataclass
class DMRGResult:
    """Results from a DMRG ground-state search.

    Attributes
    ----------
    energy : float
        Final ground state energy.
    state : MPS
        The ground-state MPS.
    energies : list[float]
        Energy at each sweep.
    bond_dimensions : list[list[int]]
        Bond dimensions at each sweep.
    converged : bool
        Whether the energy converged within tolerance.
    n_sweeps : int
        Number of sweeps performed.
    truncation_errors : list[float]
        Maximum truncation error per sweep.
    """
    energy: float
    state: MPS
    energies: List[float] = field(default_factory=list)
    bond_dimensions: List[List[int]] = field(default_factory=list)
    converged: bool = False
    n_sweeps: int = 0
    truncation_errors: List[float] = field(default_factory=list)


# -------------------------------------------------------------------
# Lanczos eigensolver
# -------------------------------------------------------------------

def _lanczos_ground(
    matvec,
    v0: NDArray,
    n_iter: int = 30,
) -> Tuple[float, NDArray]:
    """Find the lowest eigenvalue/eigenvector using the Lanczos algorithm.

    Parameters
    ----------
    matvec : callable
        Function applying the matrix to a vector.
    v0 : 1-D array
        Initial vector (will be normalized).
    n_iter : int
        Number of Lanczos iterations.

    Returns
    -------
    eigenvalue : float
        Lowest eigenvalue.
    eigenvector : 1-D array
        Corresponding eigenvector in the original basis.
    """
    n = v0.size
    n_iter = min(n_iter, n)

    # Normalize initial vector with a small random perturbation.
    # The perturbation prevents the Krylov subspace from collapsing
    # when v0 happens to be (close to) an eigenvector, which would
    # cause Lanczos to miss the true ground state.
    rng = np.random.default_rng()
    v = v0.astype(np.complex128).copy()
    v += 1e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        norm = np.linalg.norm(v)
    v /= norm

    alpha = np.zeros(n_iter, dtype=np.float64)
    beta = np.zeros(n_iter, dtype=np.float64)
    V = np.zeros((n_iter, n), dtype=np.complex128)
    V[0] = v

    w = matvec(v)
    alpha[0] = np.real(np.dot(np.conj(v), w))
    w = w - alpha[0] * v

    for j in range(1, n_iter):
        beta[j] = np.linalg.norm(w)
        if beta[j] < 1e-14:
            # Invariant subspace found
            n_iter = j
            break
        V[j] = w / beta[j]
        w = matvec(V[j])
        w -= beta[j] * V[j - 1]
        alpha[j] = np.real(np.dot(np.conj(V[j]), w))
        w -= alpha[j] * V[j]

        # Reorthogonalize against all previous vectors
        for k in range(j + 1):
            w -= np.dot(np.conj(V[k]), w) * V[k]

    # Solve tridiagonal eigenvalue problem
    T = np.diag(alpha[:n_iter]) + np.diag(beta[1:n_iter], 1) + np.diag(beta[1:n_iter], -1)
    evals, evecs = np.linalg.eigh(T)

    # Lowest eigenvalue
    idx = 0
    e0 = evals[idx]
    # Transform back to original basis
    psi = V[:n_iter].T @ evecs[:, idx]

    # Normalize
    norm = np.linalg.norm(psi)
    if norm > 1e-15:
        psi /= norm

    return float(e0), psi


# -------------------------------------------------------------------
# Environment construction
# -------------------------------------------------------------------

def _build_left_envs(mps: MPS, mpo: MPO) -> List[NDArray]:
    """Build left environment tensors for all bonds.

    L[i] has shape (chi_bra, chi_mpo, chi_ket) and represents the
    contraction of sites 0..i-1.

    L[0] = trivial (1,1,1) tensor.
    """
    n = mps.n_sites
    envs = [None] * (n + 1)
    envs[0] = np.ones((1, 1, 1), dtype=np.complex128)

    for i in range(n):
        # L[i][a,b,c] * conj(mps[i])[a,s,a'] * mpo[i][b,s,t,b'] * mps[i][c,t,c']
        # -> L[i+1][a',b',c']
        envs[i + 1] = np.einsum(
            "abc,ase,bstf,ctg->efg",
            envs[i],
            np.conj(mps.tensors[i]),
            mpo.tensors[i],
            mps.tensors[i],
        )
    return envs


def _build_right_envs(mps: MPS, mpo: MPO) -> List[NDArray]:
    """Build right environment tensors for all bonds.

    R[i] has shape (chi_bra, chi_mpo, chi_ket) and represents the
    contraction of sites i..n-1.

    R[n] = trivial (1,1,1) tensor.
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


def _effective_hamiltonian_two_site(
    L: NDArray,
    W1: NDArray,
    W2: NDArray,
    R: NDArray,
    d: int,
) -> callable:
    """Build the effective Hamiltonian for a two-site DMRG step.

    The effective Hamiltonian acts on a vector of shape
    (chi_left * d * d * chi_right,) representing the two-site tensor.

    Returns a matvec function.

    Index conventions:
      L[a, b, c]    -- left env:  bra_L, mpo_L, ket_L
      W1[b, s, u, e] -- MPO site i: mpo_L, phys_bra, phys_ket, mpo_mid
      W2[e, t, v, f] -- MPO site i+1: mpo_mid, phys_bra, phys_ket, mpo_R
      R[g, f, h]    -- right env: bra_R, mpo_R, ket_R
      theta[c, u, v, h] -- ket two-site: ket_L, phys_i, phys_{i+1}, ket_R

    Result: [a, s, t, g]  -- bra_L, phys_bra_i, phys_bra_{i+1}, bra_R
    """
    chi_l = L.shape[2]   # ket_L dimension
    chi_r = R.shape[2]   # ket_R dimension

    def matvec(v):
        theta = v.reshape(chi_l, d, d, chi_r)

        # Contract step by step to avoid huge intermediate tensors:
        # 1. theta[c, u, v, h] * R[g, f, h] -> tR[c, u, v, g, f]
        tR = np.einsum("cuvh,gfh->cuvgf", theta, R)
        # 2. tR * W2[e, t, v, f] -> tRW2[c, u, g, e, t]  (contract v, f)
        tRW = np.einsum("cuvgf,etvf->cuget", tR, W2)
        # 3. tRW * W1[b, s, u, e] -> tRWW[c, g, t, b, s]  (contract u, e)
        tRWW = np.einsum("cuget,bsue->cgtbs", tRW, W1)
        # 4. tRWW * L[a, b, c] -> result[g, t, s, a] -> reshape to [a, s, t, g]
        result = np.einsum("cgtbs,abc->astg", tRWW, L)

        return result.ravel()

    return matvec


# -------------------------------------------------------------------
# DMRG class
# -------------------------------------------------------------------

class DMRG:
    """Two-site DMRG ground state finder.

    Parameters
    ----------
    mpo : MPO
        The Hamiltonian as a matrix product operator.
    chi_max : int
        Maximum bond dimension.
    n_sweeps : int
        Maximum number of left-right sweeps.
    tol : float
        Energy convergence tolerance.
    lanczos_iter : int
        Number of Lanczos iterations per local solve.
    """

    def __init__(
        self,
        mpo: MPO,
        chi_max: int = 32,
        n_sweeps: int = 20,
        tol: float = 1e-8,
        lanczos_iter: int = 30,
    ) -> None:
        self.mpo = mpo
        self.chi_max = chi_max
        self.n_sweeps = n_sweeps
        self.tol = tol
        self.lanczos_iter = lanczos_iter

    def run(self, initial_state: Optional[MPS] = None) -> DMRGResult:
        """Execute the DMRG algorithm.

        Parameters
        ----------
        initial_state : MPS, optional
            Initial guess.  If ``None``, starts from a product state.

        Returns
        -------
        DMRGResult
            The optimisation result containing the ground state.
        """
        n = self.mpo.n_sites
        d = self.mpo.d

        if initial_state is not None:
            mps = initial_state.copy()
        else:
            mps = ProductState(n, d=d)

        # Canonicalize to the left (center at site 0)
        mps = mps.canonicalize(0)

        # Build right environments
        R_envs = _build_right_envs(mps, self.mpo)
        L_envs = [None] * (n + 1)
        L_envs[0] = np.ones((1, 1, 1), dtype=np.complex128)

        energies = []
        bond_dims_history = []
        trunc_errors = []
        converged = False

        for sweep in range(self.n_sweeps):
            max_trunc = 0.0

            # --- Right sweep: sites 0,1 then 1,2 then ... n-3,n-2 ---
            for i in range(n - 1):
                energy, trunc = self._update_two_site(
                    mps, i, L_envs[i], R_envs[i + 2],
                    direction="right",
                )
                max_trunc = max(max_trunc, trunc)

                # Update left environment for site i
                L_envs[i + 1] = np.einsum(
                    "abc,ase,bstf,ctg->efg",
                    L_envs[i],
                    np.conj(mps.tensors[i]),
                    self.mpo.tensors[i],
                    mps.tensors[i],
                )

            # --- Left sweep: sites n-2,n-1 then n-3,n-2 then ... 1,0 ---
            # Rebuild right environments from the right end
            R_envs[n] = np.ones((1, 1, 1), dtype=np.complex128)

            for i in range(n - 2, -1, -1):
                energy, trunc = self._update_two_site(
                    mps, i, L_envs[i], R_envs[i + 2],
                    direction="left",
                )
                max_trunc = max(max_trunc, trunc)

                # Update right environment for site i+1
                R_envs[i + 1] = np.einsum(
                    "ase,bstf,ctg,efg->abc",
                    np.conj(mps.tensors[i + 1]),
                    self.mpo.tensors[i + 1],
                    mps.tensors[i + 1],
                    R_envs[i + 2],
                )

            energies.append(energy)
            bond_dims_history.append(list(mps.bond_dimensions))
            trunc_errors.append(max_trunc)

            # Check convergence
            if len(energies) >= 2 and abs(energies[-1] - energies[-2]) < self.tol:
                converged = True
                break

        return DMRGResult(
            energy=energies[-1] if energies else 0.0,
            state=mps,
            energies=energies,
            bond_dimensions=bond_dims_history,
            converged=converged,
            n_sweeps=len(energies),
            truncation_errors=trunc_errors,
        )

    def _update_two_site(
        self,
        mps: MPS,
        i: int,
        L: NDArray,
        R: NDArray,
        direction: str,
    ) -> Tuple[float, float]:
        """Perform one two-site DMRG update at bond (i, i+1).

        Parameters
        ----------
        mps : MPS
            Current MPS state (modified in-place).
        i : int
            Left site index.
        L : ndarray
            Left environment, shape (chi_bra, chi_mpo, chi_ket).
        R : ndarray
            Right environment, shape (chi_bra, chi_mpo, chi_ket).
        direction : str
            "right" or "left" — determines where to put the
            orthogonality center after SVD.

        Returns
        -------
        energy : float
            The local eigenvalue found.
        trunc_error : float
            The truncation error (sum of discarded singular values squared).
        """
        d = self.mpo.d
        W1 = self.mpo.tensors[i]
        W2 = self.mpo.tensors[i + 1]

        chi_l = mps.tensors[i].shape[0]
        chi_r = mps.tensors[i + 1].shape[2]

        # Initial guess: contract the two site tensors
        theta0 = np.einsum(
            "asd,dtc->astc", mps.tensors[i], mps.tensors[i + 1]
        ).ravel()

        # Build effective Hamiltonian and solve
        matvec = _effective_hamiltonian_two_site(L, W1, W2, R, d)
        energy, theta = _lanczos_ground(matvec, theta0, self.lanczos_iter)

        # Reshape and SVD
        theta = theta.reshape(chi_l, d, d, chi_r)

        # SVD: group (chi_l, d) vs (d, chi_r)
        mat = theta.reshape(chi_l * d, d * chi_r)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # Truncate
        keep = min(len(S), self.chi_max)
        trunc_error = float(np.sum(S[keep:] ** 2))

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        # Normalize singular values
        norm = np.linalg.norm(S)
        if norm > 1e-15:
            S /= norm

        if direction == "right":
            # Absorb S into Vh (orthogonality center moves right)
            mps.tensors[i] = U.reshape(chi_l, d, keep)
            mps.tensors[i + 1] = (np.diag(S) @ Vh).reshape(keep, d, chi_r)
        else:
            # Absorb S into U (orthogonality center moves left)
            mps.tensors[i] = (U @ np.diag(S)).reshape(chi_l, d, keep)
            mps.tensors[i + 1] = Vh.reshape(keep, d, chi_r)

        return energy, trunc_error


# -------------------------------------------------------------------
# Convenience function
# -------------------------------------------------------------------

def dmrg_ground_state(
    mpo: MPO,
    chi_max: int = 32,
    n_sweeps: int = 20,
    tol: float = 1e-8,
    initial_state: Optional[MPS] = None,
) -> DMRGResult:
    """Find the ground state of an MPO Hamiltonian using DMRG.

    This is a convenience wrapper around the ``DMRG`` class.

    Parameters
    ----------
    mpo : MPO
        Hamiltonian as a matrix product operator.
    chi_max : int
        Maximum bond dimension (default 32).
    n_sweeps : int
        Maximum number of sweeps (default 20).
    tol : float
        Energy convergence tolerance.
    initial_state : MPS, optional
        Initial guess for the ground state.

    Returns
    -------
    DMRGResult
        The result containing ground-state energy and MPS.

    Examples
    --------
    >>> from nqpu.tensor_networks import IsingMPO, dmrg_ground_state
    >>> H = IsingMPO(6, J=1.0, h=1.0)
    >>> result = dmrg_ground_state(H, chi_max=16, n_sweeps=20)
    >>> print(f"E0 = {result.energy:.6f}")
    """
    solver = DMRG(mpo, chi_max=chi_max, n_sweeps=n_sweeps, tol=tol)
    return solver.run(initial_state=initial_state)
