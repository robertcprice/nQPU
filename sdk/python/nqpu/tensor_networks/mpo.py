"""Matrix Product Operators (MPO) for representing quantum Hamiltonians.

An MPO represents a many-body operator as a chain of rank-4 tensors:

    O = sum_{s,s'} W[1]^{s1,s1'} W[2]^{s2,s2'} ... W[n]^{sn,sn'}
                   |s1><s1'| (x) |s2><s2'| (x) ... (x) |sn><sn'|

where each W[k] has shape ``(chi_left, d, d, chi_right)`` with the
two physical indices being the bra/ket indices.

Key features:
  - Efficient MPO construction for standard Hamiltonians (Ising, Heisenberg)
  - MPO-MPS contraction for applying operators to states
  - Expectation values <psi|H|psi> via MPO sandwich contraction
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .mps import MPS


# -------------------------------------------------------------------
# Pauli matrices (shared constants)
# -------------------------------------------------------------------

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_Sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # S+
_Sm = np.array([[0, 0], [1, 0]], dtype=np.complex128)  # S-


# -------------------------------------------------------------------
# MPO
# -------------------------------------------------------------------

class MPO:
    """Matrix Product Operator for representing many-body operators.

    Each tensor ``tensors[i]`` has shape ``(chi_left, d, d, chi_right)``
    where ``d`` is the physical dimension.  The two middle indices are
    (bra, ket) or equivalently (output, input) physical indices.

    Parameters
    ----------
    tensors : list[ndarray]
        Rank-4 tensors for each site.
    d : int
        Physical dimension (default 2).
    """

    def __init__(self, tensors: List[NDArray], d: int = 2) -> None:
        self.tensors: List[NDArray[np.complexfloating]] = [
            np.asarray(t, dtype=np.complex128) for t in tensors
        ]
        self.d = d
        self._validate()

    def _validate(self) -> None:
        for i, t in enumerate(self.tensors):
            if t.ndim != 4:
                raise ValueError(
                    f"MPO tensor at site {i} must be rank-4, got rank-{t.ndim}"
                )
            if t.shape[1] != self.d or t.shape[2] != self.d:
                raise ValueError(
                    f"Physical dimension mismatch at site {i}: "
                    f"expected ({self.d},{self.d}), got ({t.shape[1]},{t.shape[2]})"
                )
        for i in range(len(self.tensors) - 1):
            if self.tensors[i].shape[3] != self.tensors[i + 1].shape[0]:
                raise ValueError(
                    f"Bond dimension mismatch between sites {i} and {i+1}"
                )

    # -- Properties ---------------------------------------------------

    @property
    def n_sites(self) -> int:
        return len(self.tensors)

    @property
    def bond_dimensions(self) -> List[int]:
        return [self.tensors[i].shape[3] for i in range(self.n_sites - 1)]

    # -- Operations ---------------------------------------------------

    def apply(
        self,
        mps: MPS,
        chi_max: Optional[int] = None,
        cutoff: float = 1e-14,
    ) -> MPS:
        """Apply this MPO to an MPS, returning a new MPS.

        The result is compressed via SVD sweeps if ``chi_max`` is given.

        Parameters
        ----------
        mps : MPS
            The input MPS state.
        chi_max : int, optional
            Maximum bond dimension of the output MPS.
        cutoff : float
            SVD truncation cutoff.

        Returns
        -------
        MPS
            The operator applied to the state: O|psi>.
        """
        if self.n_sites != mps.n_sites:
            raise ValueError("MPO and MPS must have the same number of sites")

        # Exact contraction: merge MPO and MPS bond indices
        new_tensors = []
        for i in range(self.n_sites):
            # MPS tensor: (chi_mps_l, d, chi_mps_r)
            A = mps.tensors[i]
            # MPO tensor: (chi_mpo_l, d_out, d_in, chi_mpo_r)
            W = self.tensors[i]

            # Contract over physical/ket index (u):
            # W[a,s,u,b] * A[c,u,e] -> T[a,s,b,c,e]
            T = np.einsum("asub,cue->asbce", W, A)

            # Reshape: (chi_mpo_l * chi_mps_l, d_out, chi_mpo_r * chi_mps_r)
            chi_mpo_l, d_out, d_in, chi_mpo_r = W.shape
            chi_mps_l, _, chi_mps_r = A.shape
            # Current shape: (chi_mpo_l, d_out, chi_mpo_r, chi_mps_l, chi_mps_r)
            # Reorder to: (chi_mpo_l, chi_mps_l, d_out, chi_mpo_r, chi_mps_r)
            T = T.transpose(0, 3, 1, 2, 4)
            T = T.reshape(chi_mpo_l * chi_mps_l, d_out, chi_mpo_r * chi_mps_r)
            new_tensors.append(T)

        result = MPS(new_tensors, d=self.d)

        # Compress if needed
        if chi_max is not None:
            result = _compress_mps(result, chi_max, cutoff)

        return result

    def expectation(self, mps: MPS) -> float:
        """Compute <psi|O|psi> via MPO-MPS sandwich contraction.

        Parameters
        ----------
        mps : MPS
            The state to evaluate.

        Returns
        -------
        float
            The expectation value (real part).
        """
        if self.n_sites != mps.n_sites:
            raise ValueError("MPO and MPS must have the same number of sites")

        # Left-to-right transfer matrix contraction
        # E has indices (bra_bond, mpo_bond, ket_bond)
        E = np.ones((1, 1, 1), dtype=np.complex128)

        for i in range(self.n_sites):
            # E[a,b,c] * conj(bra[a,s,a']) * W[b,s,t,b'] * ket[c,t,c']
            # -> E'[a',b',c']
            E = np.einsum(
                "abc,ase,bstf,ctg->efg",
                E,
                np.conj(mps.tensors[i]),
                self.tensors[i],
                mps.tensors[i],
            )

        return float(np.real(E.item()))

    def to_matrix(self) -> NDArray[np.complexfloating]:
        """Contract the MPO into a full dense matrix.

        Returns
        -------
        2-D array
            Dense matrix of shape (d^n, d^n).
        """
        n = self.n_sites
        d = self.d

        # Start with first tensor
        result = self.tensors[0]  # (chi_l, d, d, chi_r)
        for i in range(1, n):
            # result: (..., d_out_prev, d_in_prev, chi) x tensor: (chi, d_out, d_in, chi')
            result = np.einsum("...a,abcd->...bcd", result, self.tensors[i])

        # result shape: (1, d, d, d, d, ..., 1) with 2*n physical indices
        # Remove boundary dimensions
        shape = result.shape
        # After squeezing, we have d_out_1, d_in_1, d_out_2, d_in_2, ...
        result = result.reshape([s for s in shape if s != 1] if 1 in shape else shape)

        # Need to carefully reshape
        # Current layout: chi_l=1 removed, then pairs of (d_out_i, d_in_i), chi_r=1 removed
        # Actually let me redo this more carefully
        result = self.tensors[0]  # (1, d, d, chi_r)
        for i in range(1, n):
            # Contract bond dimension
            result = np.tensordot(result, self.tensors[i], axes=([-1], [0]))
            # result gains two new physical indices

        # Now shape is (1, d, d, d, d, ..., d, d, 1)
        # = (1, [d_out_0, d_in_0, d_out_1, d_in_1, ..., d_out_{n-1}, d_in_{n-1}], 1)
        result = result.squeeze()  # remove boundary bond dims of 1

        # Reorder: group all d_out indices first, then all d_in indices
        # Current order: out_0, in_0, out_1, in_1, ..., out_{n-1}, in_{n-1}
        # Want: out_0, out_1, ..., out_{n-1}, in_0, in_1, ..., in_{n-1}
        perm = list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2))
        result = result.transpose(perm)
        return result.reshape(d ** n, d ** n)

    def __repr__(self) -> str:
        return (
            f"MPO(n_sites={self.n_sites}, d={self.d}, "
            f"bond_dims={self.bond_dimensions})"
        )


# -------------------------------------------------------------------
# MPS compression utility
# -------------------------------------------------------------------

def _compress_mps(mps: MPS, chi_max: int, cutoff: float = 1e-14) -> MPS:
    """Compress an MPS via left-right SVD sweeps.

    Parameters
    ----------
    mps : MPS
        Input MPS (may have large bond dimensions).
    chi_max : int
        Maximum bond dimension after compression.
    cutoff : float
        Singular value cutoff.

    Returns
    -------
    MPS
        Compressed MPS.
    """
    n = mps.n_sites
    d = mps.d
    tensors = [t.copy() for t in mps.tensors]

    # Left sweep: QR
    for i in range(n - 1):
        chi_l, dd, chi_r = tensors[i].shape
        mat = tensors[i].reshape(chi_l * dd, chi_r)
        Q, R = np.linalg.qr(mat)
        new_chi = Q.shape[1]
        tensors[i] = Q.reshape(chi_l, dd, new_chi)
        tensors[i + 1] = np.einsum("ij,jsk->isk", R, tensors[i + 1])

    # Right sweep: SVD with truncation
    for i in range(n - 1, 0, -1):
        chi_l, dd, chi_r = tensors[i].shape
        mat = tensors[i].reshape(chi_l, dd * chi_r)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        keep = len(S)
        if cutoff > 0:
            keep = max(int(np.sum(S > cutoff)), 1)
        keep = min(keep, chi_max)

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        tensors[i] = Vh.reshape(keep, dd, chi_r)
        tensors[i - 1] = np.einsum("asd,de->ase", tensors[i - 1], U @ np.diag(S))

    return MPS(tensors, d=d)


# -------------------------------------------------------------------
# Standard Hamiltonian MPOs
# -------------------------------------------------------------------

def IsingMPO(n: int, J: float = 1.0, h: float = 1.0) -> MPO:
    """Transverse-field Ising model as MPO.

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i

    The MPO has bond dimension 3 with the structure::

        W = | I    0   0  |
            | Z    0   0  |
            | -hX  -JZ  I  |

    Parameters
    ----------
    n : int
        Number of sites.
    J : float
        ZZ coupling strength (default 1.0).
    h : float
        Transverse field strength (default 1.0).

    Returns
    -------
    MPO
        The Ising Hamiltonian as an MPO.
    """
    d = 2
    D = 3  # MPO bond dimension

    tensors = []
    for i in range(n):
        W = np.zeros((D, d, d, D), dtype=np.complex128)

        # Row 0: I
        W[0, :, :, 0] = _I
        # Row 1: Z
        W[1, :, :, 0] = _Z
        # Row 2 (operator row):
        W[2, :, :, 0] = -h * _X
        W[2, :, :, 1] = -J * _Z
        W[2, :, :, 2] = _I

        if i == 0:
            # First site: take row 2 only (start vector)
            tensors.append(W[2:3, :, :, :])
        elif i == n - 1:
            # Last site: take column 0 only (end vector)
            tensors.append(W[:, :, :, 0:1])
        else:
            tensors.append(W)

    return MPO(tensors, d=d)


def HeisenbergMPO(
    n: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    hz: float = 0.0,
) -> MPO:
    """Heisenberg XXZ chain as MPO.

    H = sum_i [ Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1} ]
        + hz sum_i Z_i

    The MPO has bond dimension 5.

    Parameters
    ----------
    n : int
        Number of sites.
    Jx, Jy, Jz : float
        Exchange couplings (default all 1.0 for isotropic Heisenberg).
    hz : float
        External field along Z (default 0.0).

    Returns
    -------
    MPO
        The Heisenberg Hamiltonian as an MPO.
    """
    d = 2
    D = 5

    tensors = []
    for i in range(n):
        W = np.zeros((D, d, d, D), dtype=np.complex128)

        # W structure (rows = left bond, cols = right bond):
        # | I     0    0    0    0 |
        # | Sp    0    0    0    0 |
        # | Sm    0    0    0    0 |
        # | Z     0    0    0    0 |
        # | hz*Z  a*Sm b*Sp Jz*Z I |
        #
        # where a = Jx/2 + Jy/2 (from S+S- + S-S+)
        # and   b = Jx/2 - Jy/2 ... but actually
        # XX + YY = 2(S+S- + S-S+) so Jx XX + Jy YY needs care.
        #
        # Actually: X = S+ + S-, Y = -i(S+ - S-)
        # XX = S+S- + S-S+ + S+S+ + S-S-
        # YY = S+S- + S-S+ - S+S+ - S-S-
        # So Jx*XX + Jy*YY = (Jx+Jy)(S+S- + S-S+) + (Jx-Jy)(S+S+ + S-S-)
        #
        # For Heisenberg (Jx=Jy), the S+S+ terms cancel.
        # General case needs D=7 to include S+S+ and S-S- terms.
        # For simplicity, use the Sp/Sm decomposition:
        # H = sum_i [a (Sp_i Sm_{i+1} + Sm_i Sp_{i+1}) + b (Sp_i Sp_{i+1} + Sm_i Sm_{i+1}) + Jz Z_i Z_{i+1}] + hz Z_i
        # where a = (Jx + Jy)/2, b = (Jx - Jy)/2

        a = Jx + Jy
        b = Jx - Jy

        if abs(b) < 1e-15:
            # Standard Heisenberg / XXZ: D=5 suffices
            W[0, :, :, 0] = _I
            W[1, :, :, 0] = _Sp
            W[2, :, :, 0] = _Sm
            W[3, :, :, 0] = _Z
            W[4, :, :, 0] = hz * _Z
            W[4, :, :, 1] = a * _Sm
            W[4, :, :, 2] = a * _Sp
            W[4, :, :, 3] = Jz * _Z
            W[4, :, :, 4] = _I
        else:
            # General XYZ: still fit in D=5 by using X and Y directly
            # Use the S+/S- decomposition with extra terms
            # For now, approximate: this covers the standard cases
            W[0, :, :, 0] = _I
            W[1, :, :, 0] = _Sp
            W[2, :, :, 0] = _Sm
            W[3, :, :, 0] = _Z
            W[4, :, :, 0] = hz * _Z
            W[4, :, :, 1] = a * _Sm + b * _Sp
            W[4, :, :, 2] = a * _Sp + b * _Sm
            W[4, :, :, 3] = Jz * _Z
            W[4, :, :, 4] = _I

        if i == 0:
            tensors.append(W[4:5, :, :, :])
        elif i == n - 1:
            tensors.append(W[:, :, :, 0:1])
        else:
            tensors.append(W)

    return MPO(tensors, d=d)


def IdentityMPO(n: int, d: int = 2) -> MPO:
    """Identity operator as MPO (bond dimension 1).

    Parameters
    ----------
    n : int
        Number of sites.
    d : int
        Physical dimension.

    Returns
    -------
    MPO
        The identity operator.
    """
    tensors = []
    for _ in range(n):
        W = np.zeros((1, d, d, 1), dtype=np.complex128)
        W[0, :, :, 0] = np.eye(d)
        tensors.append(W)
    return MPO(tensors, d=d)


def XXModelMPO(n: int, J: float = 1.0, h: float = 0.0) -> MPO:
    """XX model: H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1}) + h sum_i Z_i.

    This is equivalent to the Heisenberg model with Jx=Jy=J, Jz=0.

    Parameters
    ----------
    n : int
        Number of sites.
    J : float
        XX+YY coupling.
    h : float
        Z-field.

    Returns
    -------
    MPO
    """
    return HeisenbergMPO(n, Jx=J, Jy=J, Jz=0.0, hz=h)
