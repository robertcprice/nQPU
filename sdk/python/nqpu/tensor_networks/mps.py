"""Matrix Product States (MPS) for one-dimensional quantum systems.

An MPS represents an n-site quantum state as a chain of rank-3 tensors:

    |psi> = sum_{s1,...,sn} A[1]^{s1} A[2]^{s2} ... A[n]^{sn} |s1,...,sn>

where each A[k] has shape ``(chi_left, d, chi_right)`` with ``d`` the
local Hilbert space dimension (default 2 for qubits) and ``chi`` the
bond dimension controlling the entanglement capacity.

Key features:
  - Exact conversion to/from state vectors via SVD decomposition
  - Inner products (overlaps) via sequential left-to-right contraction
  - Local expectation values and entanglement entropy
  - Left/right canonical forms with center-site gauge
  - Factory functions for common states: product, GHZ, random MPS
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# -------------------------------------------------------------------
# MPS
# -------------------------------------------------------------------

class MPS:
    """Matrix Product State representation of a 1-D quantum state.

    Each tensor ``tensors[i]`` has shape ``(chi_left, d, chi_right)``
    where ``d`` is the physical dimension.  Boundary tensors have
    ``chi_left=1`` (site 0) and ``chi_right=1`` (site n-1).

    Parameters
    ----------
    tensors : list[ndarray]
        Rank-3 tensors for each site.
    d : int
        Physical dimension (default 2 for qubits).
    """

    def __init__(self, tensors: List[NDArray], d: int = 2) -> None:
        self.tensors: List[NDArray[np.complexfloating]] = [
            np.asarray(t, dtype=np.complex128) for t in tensors
        ]
        self.d = d
        self._validate()

    def _validate(self) -> None:
        for i, t in enumerate(self.tensors):
            if t.ndim != 3:
                raise ValueError(
                    f"Tensor at site {i} must be rank-3, got rank-{t.ndim}"
                )
            if t.shape[1] != self.d:
                raise ValueError(
                    f"Physical dimension mismatch at site {i}: "
                    f"expected {self.d}, got {t.shape[1]}"
                )
        # Check bond dimension consistency
        for i in range(len(self.tensors) - 1):
            if self.tensors[i].shape[2] != self.tensors[i + 1].shape[0]:
                raise ValueError(
                    f"Bond dimension mismatch between sites {i} and {i+1}: "
                    f"{self.tensors[i].shape[2]} != {self.tensors[i+1].shape[0]}"
                )
        if self.tensors[0].shape[0] != 1:
            raise ValueError(
                f"Left boundary must have chi_left=1, got {self.tensors[0].shape[0]}"
            )
        if self.tensors[-1].shape[2] != 1:
            raise ValueError(
                f"Right boundary must have chi_right=1, got {self.tensors[-1].shape[2]}"
            )

    # -- Properties ---------------------------------------------------

    @property
    def n_sites(self) -> int:
        return len(self.tensors)

    @property
    def bond_dimensions(self) -> List[int]:
        """Bond dimensions chi_1, ..., chi_{n-1} between consecutive sites."""
        return [self.tensors[i].shape[2] for i in range(self.n_sites - 1)]

    @property
    def max_bond_dim(self) -> int:
        if not self.bond_dimensions:
            return 1
        return max(self.bond_dimensions)

    # -- Conversions --------------------------------------------------

    @staticmethod
    def from_state_vector(
        psi: NDArray,
        n_sites: int,
        chi_max: Optional[int] = None,
        d: int = 2,
        cutoff: float = 0.0,
    ) -> "MPS":
        """Decompose a full state vector into an MPS via successive SVDs.

        Parameters
        ----------
        psi : 1-D array
            State vector of length ``d**n_sites``.
        n_sites : int
            Number of sites.
        chi_max : int, optional
            Maximum bond dimension.  If ``None``, keep all singular values
            (exact representation up to numerical precision).
        d : int
            Physical dimension per site (default 2).
        cutoff : float
            Discard singular values below this threshold.

        Returns
        -------
        MPS
            The resulting matrix product state.
        """
        psi = np.asarray(psi, dtype=np.complex128).ravel()
        expected = d ** n_sites
        if psi.size != expected:
            raise ValueError(
                f"State vector size {psi.size} does not match d^n = {d}^{n_sites} = {expected}"
            )

        tensors: List[NDArray] = []
        remaining = psi.copy()
        chi_left = 1

        for i in range(n_sites - 1):
            # Reshape into (chi_left * d, d^(n-i-1))
            right_dim = d ** (n_sites - i - 1)
            mat = remaining.reshape(chi_left * d, right_dim)

            U, S, Vh = np.linalg.svd(mat, full_matrices=False)

            # Truncation
            if cutoff > 0:
                keep = max(int(np.sum(S > cutoff)), 1)
            else:
                keep = len(S)
            if chi_max is not None:
                keep = min(keep, chi_max)

            U = U[:, :keep]
            S = S[:keep]
            Vh = Vh[:keep, :]

            # Site tensor: (chi_left, d, chi_right)
            A = U.reshape(chi_left, d, keep)
            tensors.append(A)

            # Remaining state: absorb S into Vh
            remaining = np.diag(S) @ Vh
            chi_left = keep

        # Last site
        tensors.append(remaining.reshape(chi_left, d, 1))
        return MPS(tensors, d=d)

    def to_state_vector(self) -> NDArray[np.complexfloating]:
        """Contract MPS back into a full state vector.

        Returns
        -------
        1-D array
            State vector of length ``d**n_sites``.
        """
        # Start from the left
        result = self.tensors[0]  # (1, d, chi_1)
        for i in range(1, self.n_sites):
            # result shape: (1, d^i, chi_i)
            # contract with tensors[i]: (chi_i, d, chi_{i+1})
            # result[a, S, b] * T[b, s, c] -> new[a, S, s, c]
            result = np.einsum("asb,bpc->aspc", result, self.tensors[i])
            shape = result.shape
            result = result.reshape(shape[0], shape[1] * shape[2], shape[3])

        # result shape: (1, d^n, 1)
        return result.reshape(-1)

    # -- Inner product and norms --------------------------------------

    def inner(self, other: "MPS") -> complex:
        """Compute the overlap <self|other>.

        Uses sequential left-to-right contraction with transfer matrices.

        Parameters
        ----------
        other : MPS
            The ket state.

        Returns
        -------
        complex
            The inner product <self|other>.
        """
        if self.n_sites != other.n_sites:
            raise ValueError("MPS must have the same number of sites")

        # Initialize: contract first site
        # <bra| = conj(A_bra) with shape (chi_L_bra, d, chi_R_bra)
        # |ket> = A_ket with shape (chi_L_ket, d, chi_R_ket)
        # Transfer: sum_s conj(bra[a,s,b]) * ket[c,s,d] -> T[b,d]
        T = np.einsum(
            "asb,asc->bc",
            np.conj(self.tensors[0]),
            other.tensors[0],
        )

        for i in range(1, self.n_sites):
            # T[b,d] * conj(bra[b,s,e]) * ket[d,s,f] -> new_T[e,f]
            T = np.einsum(
                "bd,bse,dsf->ef",
                T,
                np.conj(self.tensors[i]),
                other.tensors[i],
            )

        return complex(T.item())

    def norm(self) -> float:
        """Compute the norm sqrt(<self|self>)."""
        return float(np.sqrt(abs(self.inner(self))))

    def normalize(self) -> "MPS":
        """Return a normalized copy of this MPS.

        The normalization factor is absorbed into the last tensor.

        Returns
        -------
        MPS
            Normalized MPS with <psi|psi> = 1.
        """
        n = self.norm()
        if n < 1e-15:
            raise ValueError("Cannot normalize a zero MPS")
        new_tensors = [t.copy() for t in self.tensors]
        new_tensors[-1] = new_tensors[-1] / n
        return MPS(new_tensors, d=self.d)

    # -- Canonical forms ----------------------------------------------

    def canonicalize(self, center: int) -> "MPS":
        """Return a copy in mixed-canonical form with orthogonality center.

        Sites 0..center-1 are left-canonical (isometries from left),
        sites center+1..n-1 are right-canonical (isometries from right).

        Parameters
        ----------
        center : int
            The orthogonality center site index.

        Returns
        -------
        MPS
            A new MPS in mixed-canonical form.
        """
        tensors = [t.copy() for t in self.tensors]
        n = len(tensors)

        # Left sweep: QR from site 0 to center-1
        for i in range(center):
            chi_l, d, chi_r = tensors[i].shape
            mat = tensors[i].reshape(chi_l * d, chi_r)
            Q, R = np.linalg.qr(mat)
            new_chi = Q.shape[1]
            tensors[i] = Q.reshape(chi_l, d, new_chi)
            # Absorb R into the next site
            tensors[i + 1] = np.einsum("ij,jsk->isk", R, tensors[i + 1])

        # Right sweep: RQ from site n-1 to center+1
        for i in range(n - 1, center, -1):
            chi_l, d, chi_r = tensors[i].shape
            mat = tensors[i].reshape(chi_l, d * chi_r)
            Q, R = np.linalg.qr(mat.T)
            # Q^T R^T = mat, so mat = R.T @ Q.T
            # We want LQ decomposition: mat = L Q where L = R.T, Q = Q.T
            L = R.T
            Qm = Q.T
            new_chi = Qm.shape[0]
            tensors[i] = Qm.reshape(new_chi, d, chi_r)
            # Absorb L into the previous site
            tensors[i - 1] = np.einsum("asd,de->ase", tensors[i - 1], L)

        return MPS(tensors, d=self.d)

    # -- Observables --------------------------------------------------

    def expectation(self, operator: NDArray, site: int) -> float:
        """Compute <psi| O_site |psi> for a local operator.

        Parameters
        ----------
        operator : 2-D array
            Local operator acting on site ``site``, shape ``(d, d)``.
        site : int
            The site index where the operator acts.

        Returns
        -------
        float
            The expectation value (real part).
        """
        op = np.asarray(operator, dtype=np.complex128)
        if op.shape != (self.d, self.d):
            raise ValueError(f"Operator must be ({self.d},{self.d}), got {op.shape}")

        # Build transfer matrices left-to-right
        # T starts as identity in bond space
        T = np.ones((1, 1), dtype=np.complex128)

        for i in range(self.n_sites):
            if i == site:
                # Insert operator: sum_s,sp conj(A[a,s,b]) * O[s,sp] * A[c,sp,d]
                T = np.einsum(
                    "ac,asb,st,ctd->bd",
                    T,
                    np.conj(self.tensors[i]),
                    op,
                    self.tensors[i],
                )
            else:
                T = np.einsum(
                    "ac,asb,csd->bd",
                    T,
                    np.conj(self.tensors[i]),
                    self.tensors[i],
                )

        return float(np.real(T.item()))

    def expectation_two_site(
        self,
        operator: NDArray,
        site1: int,
        site2: int,
    ) -> float:
        """Compute <psi| O_{site1,site2} |psi> for a two-site operator.

        Parameters
        ----------
        operator : 2-D array
            Two-site operator, shape ``(d^2, d^2)``.
        site1, site2 : int
            The two site indices (must satisfy site1 < site2).

        Returns
        -------
        float
            The expectation value (real part).
        """
        if site1 >= site2:
            raise ValueError("site1 must be less than site2")

        op = np.asarray(operator, dtype=np.complex128).reshape(
            self.d, self.d, self.d, self.d
        )

        T = np.ones((1, 1), dtype=np.complex128)

        for i in range(self.n_sites):
            if i == site1:
                # Partial contraction: insert first index of the operator
                # T[a,c] * conj(A[a,s1,b]) * A[c,t1,d] -> T_partial[b,d,s1,t1]
                T = np.einsum(
                    "ac,asb,ctd->bdst",
                    T,
                    np.conj(self.tensors[i]),
                    self.tensors[i],
                )
            elif i == site2:
                # Complete the operator contraction
                # T_partial[a,c,s1,t1] * conj(A[a,s2,b]) * op[s1,s2,t1,t2] * A[c,t2,d] -> T[b,d]
                T = np.einsum(
                    "acst,aub,sutv,cvd->bd",
                    T,
                    np.conj(self.tensors[i]),
                    op,
                    self.tensors[i],
                )
            elif i > site1 and i < site2:
                # Between the two sites: identity on physical, carry partial
                T = np.einsum(
                    "acst,apb,cpd->bdst",
                    T,
                    np.conj(self.tensors[i]),
                    self.tensors[i],
                )
            else:
                # Normal transfer matrix
                if T.ndim == 2:
                    T = np.einsum(
                        "ac,asb,csd->bd",
                        T,
                        np.conj(self.tensors[i]),
                        self.tensors[i],
                    )
                else:
                    T = np.einsum(
                        "acst,apb,cpd->bdst",
                        T,
                        np.conj(self.tensors[i]),
                        self.tensors[i],
                    )

        return float(np.real(T.item()))

    def entanglement_entropy(self, bond: int) -> float:
        """Von Neumann entanglement entropy across a bond.

        The MPS is canonicalized at the bond, and the entropy is computed
        from the singular values of the bond matrix.

        Parameters
        ----------
        bond : int
            Bond index (0-based). Bond ``i`` is between sites ``i`` and ``i+1``.

        Returns
        -------
        float
            Von Neumann entropy S = -sum_i lambda_i^2 * log(lambda_i^2).
        """
        if bond < 0 or bond >= self.n_sites - 1:
            raise ValueError(f"Bond {bond} out of range [0, {self.n_sites - 2}]")

        # Canonicalize at bond+1 (the right site of the bond)
        canonical = self.canonicalize(bond + 1)

        # Get the singular values by SVD of the center tensor
        # or equivalently, the bond matrix between site bond and bond+1
        # after canonicalization.
        # Actually: take the left-canonical part up to site bond and do SVD
        # on the center tensor.
        t = canonical.tensors[bond + 1]
        chi_l, d, chi_r = t.shape
        mat = t.reshape(chi_l, d * chi_r)
        _, S, _ = np.linalg.svd(mat, full_matrices=False)

        # Compute entropy from singular values
        probs = S ** 2
        probs = probs[probs > 1e-30]  # avoid log(0)
        total = np.sum(probs)
        if total > 0:
            probs = probs / total
        return float(-np.sum(probs * np.log(probs)))

    # -- Copying -------------------------------------------------------

    def copy(self) -> "MPS":
        """Deep copy of the MPS."""
        return MPS([t.copy() for t in self.tensors], d=self.d)

    def __repr__(self) -> str:
        bonds = self.bond_dimensions
        return (
            f"MPS(n_sites={self.n_sites}, d={self.d}, "
            f"bond_dims={bonds}, max_chi={self.max_bond_dim})"
        )


# -------------------------------------------------------------------
# Factory functions for common states
# -------------------------------------------------------------------

def ProductState(n_sites: int, d: int = 2) -> MPS:
    """Create all-|0> product state as MPS (chi=1).

    Parameters
    ----------
    n_sites : int
        Number of sites.
    d : int
        Physical dimension (default 2).

    Returns
    -------
    MPS
        Product state |00...0>.
    """
    tensors = []
    for _ in range(n_sites):
        t = np.zeros((1, d, 1), dtype=np.complex128)
        t[0, 0, 0] = 1.0
        tensors.append(t)
    return MPS(tensors, d=d)


def GHZState(n_sites: int) -> MPS:
    """Create GHZ state (|00...0> + |11...1>) / sqrt(2) as MPS (chi=2).

    The GHZ state has entanglement entropy log(2) at every bond.

    Parameters
    ----------
    n_sites : int
        Number of sites (must be >= 2).

    Returns
    -------
    MPS
        GHZ state with bond dimension 2.
    """
    if n_sites < 2:
        raise ValueError("GHZ state requires at least 2 sites")

    tensors = []

    # First site: shape (1, 2, 2)
    A0 = np.zeros((1, 2, 2), dtype=np.complex128)
    A0[0, 0, 0] = 1.0 / np.sqrt(2)
    A0[0, 1, 1] = 1.0 / np.sqrt(2)
    tensors.append(A0)

    # Middle sites: shape (2, 2, 2)
    for _ in range(n_sites - 2):
        Am = np.zeros((2, 2, 2), dtype=np.complex128)
        Am[0, 0, 0] = 1.0
        Am[1, 1, 1] = 1.0
        tensors.append(Am)

    # Last site: shape (2, 2, 1)
    An = np.zeros((2, 2, 1), dtype=np.complex128)
    An[0, 0, 0] = 1.0
    An[1, 1, 0] = 1.0
    tensors.append(An)

    return MPS(tensors, d=2)


def RandomMPS(
    n_sites: int,
    chi: int,
    d: int = 2,
    rng: Optional[np.random.Generator] = None,
    normalize: bool = True,
) -> MPS:
    """Create a random MPS with specified bond dimension.

    Parameters
    ----------
    n_sites : int
        Number of sites.
    chi : int
        Internal bond dimension.
    d : int
        Physical dimension (default 2).
    rng : numpy random Generator, optional
        Random number generator for reproducibility.
    normalize : bool
        Whether to normalize the resulting MPS (default True).

    Returns
    -------
    MPS
        Random MPS.
    """
    if rng is None:
        rng = np.random.default_rng()

    tensors = []
    bond_dims = [1]
    for i in range(1, n_sites):
        bd = min(chi, d ** i, d ** (n_sites - i))
        bond_dims.append(bd)
    bond_dims.append(1)

    for i in range(n_sites):
        chi_l = bond_dims[i]
        chi_r = bond_dims[i + 1]
        real = rng.standard_normal((chi_l, d, chi_r))
        imag = rng.standard_normal((chi_l, d, chi_r))
        tensors.append((real + 1j * imag) / np.sqrt(2))

    mps = MPS(tensors, d=d)
    if normalize:
        mps = mps.normalize()
    return mps


def WState(n_sites: int) -> MPS:
    """Create the W state as an MPS.

    The W state is (|100...0> + |010...0> + ... + |000...1>) / sqrt(n).

    Parameters
    ----------
    n_sites : int
        Number of sites (must be >= 2).

    Returns
    -------
    MPS
        W state with bond dimension 2.
    """
    if n_sites < 2:
        raise ValueError("W state requires at least 2 sites")

    c = 1.0 / np.sqrt(n_sites)
    tensors = []

    # First site: shape (1, 2, 2)
    # Two bond states: "no excitation yet" (0) and "excitation placed" (1)
    A0 = np.zeros((1, 2, 2), dtype=np.complex128)
    A0[0, 0, 0] = 1.0   # |0>, no excitation yet
    A0[0, 1, 1] = c      # |1>, excitation placed
    tensors.append(A0)

    # Middle sites: shape (2, 2, 2)
    for _ in range(n_sites - 2):
        Am = np.zeros((2, 2, 2), dtype=np.complex128)
        Am[0, 0, 0] = 1.0   # pass through, no excitation
        Am[0, 1, 1] = c     # place excitation here
        Am[1, 0, 1] = 1.0   # excitation already placed, propagate
        tensors.append(Am)

    # Last site: shape (2, 2, 1)
    An = np.zeros((2, 2, 1), dtype=np.complex128)
    An[0, 1, 0] = c     # place excitation at last site
    An[1, 0, 0] = 1.0   # excitation was already placed
    tensors.append(An)

    return MPS(tensors, d=2)
