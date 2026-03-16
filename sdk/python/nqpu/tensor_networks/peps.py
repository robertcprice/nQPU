"""2D PEPS: Projected Entangled Pair States for 2D quantum systems.

Implements PEPS (Projected Entangled Pair States) for representing
quantum states on two-dimensional lattices. Each site tensor has one
physical index and up to four virtual bond indices connecting to
nearest neighbours (up, right, down, left).

Key features:
  - PEPS construction on rectangular lattices with configurable bond dimension
  - Product state and Neel state factory methods
  - Boundary MPS contraction for approximate norm and expectation values
  - Simple update algorithm for imaginary time evolution (ground state prep)
  - Ising and Heisenberg 2D bond Hamiltonians

The boundary MPS method contracts the 2D tensor network row by row,
building an effective MPS representation of the boundary that is
truncated at each step to keep the computation tractable.

References:
  - Verstraete, F. & Cirac, J.I., cond-mat/0407066 (2004)
  - Jordan, J. et al., Phys. Rev. Lett. 101, 250602 (2008)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# -------------------------------------------------------------------
# Pauli matrices
# -------------------------------------------------------------------

_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# -------------------------------------------------------------------
# PEPSTensor
# -------------------------------------------------------------------

@dataclass
class PEPSTensor:
    """Single PEPS tensor with physical and virtual indices.

    The tensor has shape (phys, up, right, down, left) for bulk sites,
    with boundary sites having dimension 1 along absent directions.

    Parameters
    ----------
    data : ndarray
        The tensor data with shape (phys, up, right, down, left).
    row : int
        Row index in the lattice.
    col : int
        Column index in the lattice.
    """
    data: np.ndarray
    row: int
    col: int

    @property
    def physical_dim(self) -> int:
        """Physical dimension (first index)."""
        return self.data.shape[0]

    @property
    def bond_dims(self) -> tuple:
        """Virtual bond dimensions (up, right, down, left)."""
        return self.data.shape[1:]

    def __repr__(self) -> str:
        return (
            f"PEPSTensor(row={self.row}, col={self.col}, "
            f"phys={self.physical_dim}, bonds={self.bond_dims})"
        )


# -------------------------------------------------------------------
# PEPS
# -------------------------------------------------------------------

@dataclass
class PEPS:
    """2D Projected Entangled Pair State on a rectangular lattice.

    Each site has a tensor with one physical index (dimension d)
    and up to 4 virtual indices (bond dimension D) connecting to
    neighbours in the up, right, down, and left directions.

    Parameters
    ----------
    rows : int
        Number of rows in the lattice.
    cols : int
        Number of columns in the lattice.
    phys_dim : int
        Physical dimension at each site (default 2 for qubits).
    bond_dim : int
        Maximum virtual bond dimension (default 2).
    tensors : list[list[PEPSTensor]], optional
        Pre-built tensor grid. If None, random tensors are generated.
    """
    rows: int
    cols: int
    phys_dim: int = 2
    bond_dim: int = 2
    tensors: Optional[List[List[PEPSTensor]]] = None

    def __post_init__(self):
        if self.tensors is None:
            self._init_random()

    def _init_random(self, rng=None):
        """Initialize with random tensors respecting boundary conditions.

        Boundary sites have bond dimension 1 along edges that have no
        neighbour. Interior bonds have dimension ``self.bond_dim``.
        """
        if rng is None:
            rng = np.random.default_rng(0)

        d = self.phys_dim
        D = self.bond_dim
        self.tensors = []

        for r in range(self.rows):
            row_tensors = []
            for c in range(self.cols):
                d_up = D if r > 0 else 1
                d_right = D if c < self.cols - 1 else 1
                d_down = D if r < self.rows - 1 else 1
                d_left = D if c > 0 else 1

                shape = (d, d_up, d_right, d_down, d_left)
                data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
                data /= np.linalg.norm(data) + 1e-15
                row_tensors.append(PEPSTensor(data=data, row=r, col=c))
            self.tensors.append(row_tensors)

    @staticmethod
    def product_state(rows: int, cols: int, state: int = 0, phys_dim: int = 2) -> 'PEPS':
        """Create product state PEPS (all sites in same computational basis state).

        Parameters
        ----------
        rows, cols : int
            Lattice dimensions.
        state : int
            Which computational basis state (0 for |0>, 1 for |1>, etc.).
        phys_dim : int
            Physical dimension.

        Returns
        -------
        PEPS
            Product state with bond dimension 1.
        """
        tensor_grid = []
        for r in range(rows):
            row_tensors = []
            for c in range(cols):
                data = np.zeros((phys_dim, 1, 1, 1, 1), dtype=np.complex128)
                data[state, 0, 0, 0, 0] = 1.0
                row_tensors.append(PEPSTensor(data=data, row=r, col=c))
            tensor_grid.append(row_tensors)
        return PEPS(rows=rows, cols=cols, phys_dim=phys_dim, bond_dim=1,
                    tensors=tensor_grid)

    @staticmethod
    def neel_state(rows: int, cols: int) -> 'PEPS':
        """Create antiferromagnetic Neel state on the 2D lattice.

        Site (r, c) is in state |0> if (r + c) is even, |1> if odd.

        Parameters
        ----------
        rows, cols : int
            Lattice dimensions.

        Returns
        -------
        PEPS
            Neel state with bond dimension 1 and physical dimension 2.
        """
        tensor_grid = []
        for r in range(rows):
            row_tensors = []
            for c in range(cols):
                data = np.zeros((2, 1, 1, 1, 1), dtype=np.complex128)
                s = (r + c) % 2
                data[s, 0, 0, 0, 0] = 1.0
                row_tensors.append(PEPSTensor(data=data, row=r, col=c))
            tensor_grid.append(row_tensors)
        return PEPS(rows=rows, cols=cols, phys_dim=2, bond_dim=1,
                    tensors=tensor_grid)

    def get_tensor(self, r: int, c: int) -> np.ndarray:
        """Get raw tensor data at site (r, c)."""
        return self.tensors[r][c].data

    def set_tensor(self, r: int, c: int, data: np.ndarray):
        """Set raw tensor data at site (r, c)."""
        self.tensors[r][c] = PEPSTensor(data=data, row=r, col=c)

    def norm_squared(self, chi_boundary: int = 4) -> float:
        """Compute <psi|psi> using boundary MPS contraction.

        The double-layer tensor network is contracted row by row
        using a boundary MPS with truncated bond dimension.

        Parameters
        ----------
        chi_boundary : int
            Maximum bond dimension of the boundary MPS.

        Returns
        -------
        float
            The norm squared of the PEPS.
        """
        bmps = BoundaryMPS(chi_max=chi_boundary)
        return abs(bmps.contract_full(self))

    def expectation_local(self, operator: np.ndarray, row: int, col: int,
                          chi_boundary: int = 4) -> complex:
        """Compute <psi|O_site|psi> for a single-site operator.

        Inserts the operator at site (row, col) and contracts the
        modified double-layer network using boundary MPS.

        Parameters
        ----------
        operator : ndarray
            Local operator, shape (d, d).
        row, col : int
            Site where the operator acts.
        chi_boundary : int
            Maximum boundary MPS bond dimension.

        Returns
        -------
        complex
            The expectation value.
        """
        op = np.asarray(operator, dtype=np.complex128)
        norm_sq = self.norm_squared(chi_boundary)
        if abs(norm_sq) < 1e-30:
            return 0.0

        # Build modified PEPS with operator inserted
        bmps = BoundaryMPS(chi_max=chi_boundary)
        numerator = bmps.contract_with_operator(self, op, row, col)
        return numerator / norm_sq

    def copy(self) -> 'PEPS':
        """Deep copy of the PEPS."""
        tensor_grid = []
        for r in range(self.rows):
            row_tensors = []
            for c in range(self.cols):
                t = self.tensors[r][c]
                row_tensors.append(PEPSTensor(data=t.data.copy(), row=r, col=c))
            tensor_grid.append(row_tensors)
        return PEPS(rows=self.rows, cols=self.cols, phys_dim=self.phys_dim,
                    bond_dim=self.bond_dim, tensors=tensor_grid)

    def __repr__(self) -> str:
        return (
            f"PEPS(rows={self.rows}, cols={self.cols}, "
            f"phys_dim={self.phys_dim}, bond_dim={self.bond_dim})"
        )


# -------------------------------------------------------------------
# Double-layer transfer tensor
# -------------------------------------------------------------------

def _double_layer_tensor(bra: np.ndarray, ket: np.ndarray) -> np.ndarray:
    """Build the double-layer tensor at one site by contracting physical index.

    bra has shape (d, u_b, r_b, d_b, l_b)  -- conjugated
    ket has shape (d, u_k, r_k, d_k, l_k)

    Result has shape (u_b, u_k, r_b, r_k, d_b, d_k, l_b, l_k)
    obtained by summing over the physical index.
    """
    # Contract over physical index (axis 0)
    return np.einsum("pabcd,pefgh->aebfcgdh", np.conj(bra), ket)


def _double_layer_tensor_with_op(bra: np.ndarray, ket: np.ndarray,
                                  op: np.ndarray) -> np.ndarray:
    """Double-layer tensor with operator inserted on the physical index.

    op has shape (d, d): <bra| op |ket>
    Computes sum_pq conj(bra_p) * op_{pq} * ket_q.

    Result has same shape as _double_layer_tensor.
    """
    return np.einsum("pabcd,pq,qefgh->aebfcgdh", np.conj(bra), op, ket)


# -------------------------------------------------------------------
# BoundaryMPS
# -------------------------------------------------------------------

@dataclass
class BoundaryMPS:
    """Boundary MPS used for approximate PEPS contraction.

    Contract a 2D tensor network by building boundary MPS row by row.
    Starting from the top, each row is absorbed into the boundary MPS,
    which is truncated to keep its bond dimension manageable.

    Parameters
    ----------
    chi_max : int
        Maximum bond dimension of the boundary MPS.
    """
    chi_max: int = 8

    def _initial_boundary(self, ncols: int) -> List[np.ndarray]:
        """Create trivial initial boundary (row of scalar 1s).

        Each boundary tensor has shape (bond_left, combined_virtual, bond_right).
        For the initial boundary, everything is dimension 1.
        """
        boundary = []
        for c in range(ncols):
            boundary.append(np.ones((1, 1, 1), dtype=np.complex128))
        return boundary

    def _absorb_row(self, boundary: List[np.ndarray],
                    row_dl: List[np.ndarray]) -> List[np.ndarray]:
        """Absorb one row of double-layer tensors into the boundary MPS.

        Parameters
        ----------
        boundary : list of ndarray
            Current boundary MPS. Each tensor has shape
            (bond_left, combined_up, bond_right).
        row_dl : list of ndarray
            Double-layer tensors for this row. Each has shape
            (u_b*u_k, r_b*r_k, d_b*d_k, l_b*l_k) -- combined indices.

        Returns
        -------
        list of ndarray
            Updated boundary MPS with shape
            (new_bond_left, combined_down, new_bond_right).
        """
        ncols = len(boundary)
        new_boundary = []

        for c in range(ncols):
            bnd = boundary[c]    # (bl, cu, br)
            dl = row_dl[c]       # (cu, cr, cd, cl)

            # Contract boundary with double-layer over the up index (cu)
            # bnd[a, u, b] * dl[u, r, d, l] -> merged[a, b, r, d, l]
            merged = np.einsum("aub,urdl->abrdl", bnd, dl)

            # Reshape: new_bl = bl * cl, new_br = br * cr
            bl = merged.shape[0]
            br = merged.shape[1]
            cr = merged.shape[2]
            cd = merged.shape[3]
            cl = merged.shape[4]

            # Reorder to (bl, cl, cd, br, cr) then reshape
            merged = merged.transpose(0, 4, 3, 1, 2)  # (bl, cl, cd, br, cr)
            new_tensor = merged.reshape(bl * cl, cd, br * cr)
            new_boundary.append(new_tensor)

        # Truncate bond dimensions via SVD sweep
        new_boundary = self._truncate_boundary(new_boundary)
        return new_boundary

    def _truncate_boundary(self, boundary: List[np.ndarray]) -> List[np.ndarray]:
        """Truncate boundary MPS bond dimensions via left-right SVD sweep.

        Parameters
        ----------
        boundary : list of ndarray
            Boundary MPS tensors with potentially large bond dimensions.

        Returns
        -------
        list of ndarray
            Truncated boundary MPS.
        """
        ncols = len(boundary)
        if ncols <= 1:
            return boundary

        tensors = [t.copy() for t in boundary]

        # Left sweep: QR
        for c in range(ncols - 1):
            bl, cd, br = tensors[c].shape
            mat = tensors[c].reshape(bl * cd, br)
            Q, R = np.linalg.qr(mat)
            new_chi = min(Q.shape[1], self.chi_max)
            Q = Q[:, :new_chi]
            R = R[:new_chi, :]
            tensors[c] = Q.reshape(bl, cd, new_chi)
            # Absorb remainder into next tensor
            tensors[c + 1] = np.einsum("ij,jdk->idk", R, tensors[c + 1])

        # Right sweep: SVD with truncation
        for c in range(ncols - 1, 0, -1):
            bl, cd, br = tensors[c].shape
            mat = tensors[c].reshape(bl, cd * br)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)

            keep = min(len(S), self.chi_max)
            U = U[:, :keep]
            S = S[:keep]
            Vh = Vh[:keep, :]

            tensors[c] = Vh.reshape(keep, cd, br)
            tensors[c - 1] = np.einsum("adb,bc->adc", tensors[c - 1], U @ np.diag(S))

        return tensors

    def _contract_boundary_to_scalar(self, boundary: List[np.ndarray]) -> complex:
        """Contract a boundary MPS to a scalar (trace over all indices).

        The boundary should have combined_down dimension = 1 (bottom row
        already absorbed) or we trace over the down indices.
        """
        ncols = len(boundary)
        if ncols == 0:
            return 1.0 + 0j

        # Contract left to right, tracing over the "down" (middle) index
        # Each tensor: (bl, cd, br)
        vec = boundary[0]  # (bl, cd, br)
        # Trace over the down index
        vec = np.einsum("adb->ab", vec)  # (bl, br)

        for c in range(1, ncols):
            t = boundary[c]  # (bl, cd, br)
            t_traced = np.einsum("adb->ab", t)  # (bl, br)
            # Contract: vec[..., br_prev] * t_traced[bl_cur=br_prev, br_cur]
            vec = vec @ t_traced

        return complex(vec.item()) if vec.size == 1 else complex(np.trace(vec))

    def contract_full(self, peps: PEPS) -> complex:
        """Full PEPS norm contraction via boundary MPS.

        Contracts <psi|psi> by building double-layer tensors row by
        row and absorbing them into a boundary MPS from top to bottom.

        Parameters
        ----------
        peps : PEPS
            The PEPS to contract.

        Returns
        -------
        complex
            The value <psi|psi>.
        """
        boundary = self._initial_boundary(peps.cols)

        for r in range(peps.rows):
            row_dl = []
            for c in range(peps.cols):
                t = peps.tensors[r][c].data
                dl = _double_layer_tensor(t, t)
                # dl shape: (u_b, u_k, r_b, r_k, d_b, d_k, l_b, l_k)
                # Combine pairs: (u_b*u_k, r_b*r_k, d_b*d_k, l_b*l_k)
                s = dl.shape
                dl = dl.reshape(s[0]*s[1], s[2]*s[3], s[4]*s[5], s[6]*s[7])
                row_dl.append(dl)

            boundary = self._absorb_row(boundary, row_dl)

        return self._contract_boundary_to_scalar(boundary)

    def contract_with_operator(self, peps: PEPS, op: np.ndarray,
                               op_row: int, op_col: int) -> complex:
        """Contract <psi|O|psi> with single-site operator at (op_row, op_col).

        Same as contract_full but inserts the operator at the specified site.

        Parameters
        ----------
        peps : PEPS
            The PEPS state.
        op : ndarray
            Single-site operator, shape (d, d).
        op_row, op_col : int
            Site where the operator is inserted.

        Returns
        -------
        complex
            The value <psi|O|psi>.
        """
        boundary = self._initial_boundary(peps.cols)

        for r in range(peps.rows):
            row_dl = []
            for c in range(peps.cols):
                t = peps.tensors[r][c].data
                if r == op_row and c == op_col:
                    dl = _double_layer_tensor_with_op(t, t, op)
                else:
                    dl = _double_layer_tensor(t, t)
                s = dl.shape
                dl = dl.reshape(s[0]*s[1], s[2]*s[3], s[4]*s[5], s[6]*s[7])
                row_dl.append(dl)

            boundary = self._absorb_row(boundary, row_dl)

        return self._contract_boundary_to_scalar(boundary)


# -------------------------------------------------------------------
# Simple Update
# -------------------------------------------------------------------

@dataclass
class SimpleUpdateResult:
    """Result from the simple update algorithm.

    Attributes
    ----------
    energies : list[float]
        Energy estimates at periodic intervals.
    converged : bool
        Whether the algorithm converged.
    peps : PEPS
        The final optimized PEPS.
    n_steps : int
        Number of steps performed.
    """
    energies: List[float]
    converged: bool
    peps: PEPS
    n_steps: int


@dataclass
class SimpleUpdate:
    """Simple update algorithm for PEPS imaginary time evolution.

    Uses local SVD-based truncation (no environment) for efficiency.
    Less accurate than full update but much faster.

    The algorithm applies imaginary time gates exp(-dt * h_bond) to
    each bond Hamiltonian, then truncates the resulting bond dimension
    via SVD to maintain a manageable PEPS representation.

    Parameters
    ----------
    peps : PEPS
        The initial PEPS state.
    hamiltonian_bonds : list
        List of (row1, col1, row2, col2, h_bond) tuples specifying
        the two-site bond Hamiltonians on the lattice.
    bond_dim : int
        Maximum bond dimension after truncation.
    """
    peps: PEPS
    hamiltonian_bonds: list
    bond_dim: int = 4

    def _get_bond_direction(self, r1: int, c1: int, r2: int, c2: int) -> Tuple[int, int]:
        """Get the virtual index directions for a bond between two sites.

        Returns (dir1, dir2) where dir is the axis in the PEPS tensor
        (0=up, 1=right, 2=down, 3=left) offset by +1 for the physical index.
        """
        if r2 == r1 and c2 == c1 + 1:  # horizontal: site1-right, site2-left
            return 2, 4  # right index of site1, left index of site2
        elif r2 == r1 + 1 and c2 == c1:  # vertical: site1-down, site2-up
            return 3, 1  # down index of site1, up index of site2
        elif r2 == r1 and c2 == c1 - 1:
            return 4, 2
        elif r2 == r1 - 1 and c2 == c1:
            return 1, 3
        else:
            raise ValueError(
                f"Sites ({r1},{c1}) and ({r2},{c2}) are not nearest neighbours"
            )

    def _apply_gate_to_bond(self, r1: int, c1: int, r2: int, c2: int,
                            gate: np.ndarray):
        """Apply a two-site gate to a bond and truncate.

        The gate has shape (d^2, d^2) acting on the physical indices
        of the two sites.
        """
        d = self.peps.phys_dim
        dir1, dir2 = self._get_bond_direction(r1, c1, r2, c2)

        t1 = self.peps.get_tensor(r1, c1)  # (d, u, r, d_down, l)
        t2 = self.peps.get_tensor(r2, c2)

        # Contract the two tensors along their shared bond
        # t1 has the bond at axis dir1, t2 at axis dir2
        # Form a combined tensor by contracting these axes
        theta = np.tensordot(t1, t2, axes=([dir1], [dir2]))
        # theta has all remaining axes of t1 then all remaining axes of t2

        # Apply gate on the two physical indices (axis 0 of t1 and the
        # first axis from t2's remaining axes)
        # After tensordot, physical axes are at positions 0 (from t1) and
        # ndim(t1)-1 (from t2, since dir1 was removed)
        ndim1 = t1.ndim
        phys1_axis = 0
        phys2_axis = ndim1 - 1  # t2's physical index position in theta

        gate_4 = gate.reshape(d, d, d, d)
        # Move physical axes to front for gate application
        axes_order = list(range(theta.ndim))
        axes_order.remove(phys1_axis)
        axes_order.remove(phys2_axis)
        theta = np.moveaxis(theta, [phys1_axis, phys2_axis], [0, 1])

        # Apply gate: gate[p_out, q_out, p_in, q_in] * theta[p_in, q_in, ...]
        theta = np.einsum("pqrs,rs...->pq...", gate_4, theta)

        # SVD to split back into two tensors
        # Group: (phys1, other_t1_axes) vs (phys2, other_t2_axes)
        remaining_shape = theta.shape[2:]
        # First half of remaining belongs to t1, second half to t2
        n_t1_remaining = ndim1 - 2  # t1 axes minus physical and bond
        n_t2_remaining = t2.ndim - 2

        left_shape = (d,) + remaining_shape[:n_t1_remaining]
        right_shape = (d,) + remaining_shape[n_t1_remaining:]

        left_dim = int(np.prod(left_shape))
        right_dim = int(np.prod(right_shape))

        mat = theta.reshape(left_dim, right_dim)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # Truncate
        keep = min(len(S), self.bond_dim)
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        # Normalize singular values
        norm = np.linalg.norm(S)
        if norm > 1e-15:
            S /= norm

        # Absorb sqrt(S) into both sides
        sqrt_S = np.sqrt(S)
        U_new = U @ np.diag(sqrt_S)
        Vh_new = np.diag(sqrt_S) @ Vh

        # Reshape back to tensor form
        t1_new = U_new.reshape(*left_shape, keep)
        t2_new = Vh_new.reshape(keep, *right_shape)

        # Move the bond axis back to its original position in t1
        # Currently: (phys, other_axes..., bond)
        # Need: (phys, u, r, d, l) with bond at position dir1
        # The other axes are in the same order as the original minus the bond
        perm1 = list(range(t1_new.ndim))
        # Bond is at the last position, needs to go to dir1
        perm1.remove(t1_new.ndim - 1)
        perm1.insert(dir1, t1_new.ndim - 1)
        t1_new = t1_new.transpose(perm1)

        # For t2: bond is at position 0, physical is at 1
        # Currently: (bond, phys, other_axes...)
        # Need: (phys, u, r, d, l) with bond at position dir2
        perm2 = list(range(t2_new.ndim))
        # Move phys from 1 to 0
        perm2.remove(1)
        perm2.insert(0, 1)
        # Now move bond from its current position to dir2
        bond_pos = perm2.index(0)
        perm2.remove(0)
        perm2.insert(dir2, 0)
        t2_new = t2_new.transpose(perm2)

        self.peps.set_tensor(r1, c1, t1_new)
        self.peps.set_tensor(r2, c2, t2_new)

    def step(self, dt: float):
        """One imaginary time step using simple update.

        Applies exp(-dt * h_bond) to each bond in the Hamiltonian.

        Parameters
        ----------
        dt : float
            Imaginary time step size.
        """
        for bond_spec in self.hamiltonian_bonds:
            r1, c1, r2, c2, h_bond = bond_spec
            h = np.asarray(h_bond, dtype=np.complex128)
            # Gate = exp(-dt * H_bond) via eigendecomposition
            evals, evecs = np.linalg.eigh(h)
            gate = (evecs * np.exp(-dt * evals)) @ evecs.conj().T
            self._apply_gate_to_bond(r1, c1, r2, c2, gate)

    def run(self, dt: float = 0.01, n_steps: int = 100,
            measure_every: int = 10) -> SimpleUpdateResult:
        """Run simple update for ground state preparation.

        Parameters
        ----------
        dt : float
            Imaginary time step size.
        n_steps : int
            Number of imaginary time steps.
        measure_every : int
            Record energy estimate every this many steps.

        Returns
        -------
        SimpleUpdateResult
            The result containing energy history and final PEPS.
        """
        energies = []
        converged = False

        for step in range(n_steps):
            self.step(dt)

            if (step + 1) % measure_every == 0:
                try:
                    e = self._estimate_energy()
                    energies.append(e)
                    if len(energies) >= 2 and abs(energies[-1] - energies[-2]) < 1e-8:
                        converged = True
                        break
                except Exception:
                    pass

        return SimpleUpdateResult(
            energies=energies,
            converged=converged,
            peps=self.peps.copy(),
            n_steps=step + 1 if 'step' in dir() else n_steps,
        )

    def _estimate_energy(self) -> float:
        """Estimate total energy from bond Hamiltonians.

        Uses boundary MPS contraction for each bond term. This is
        expensive; for production use, one would cache environments.
        """
        total = 0.0
        bmps = BoundaryMPS(chi_max=4)
        norm_sq = abs(bmps.contract_full(self.peps))
        if norm_sq < 1e-30:
            return 0.0

        for bond_spec in self.hamiltonian_bonds:
            r1, c1, r2, c2, h_bond = bond_spec
            # Approximate: use single-site Z expectation for diagonal terms
            # For simplicity, use the diagonal part of the bond Hamiltonian
            h = np.asarray(h_bond, dtype=np.complex128)
            d = self.peps.phys_dim
            # Trace over one pair of physical indices to get single-site contribution
            h_reshaped = h.reshape(d, d, d, d)
            # Partial trace over site 2 for site 1 operator
            op1 = np.einsum("abcb->ac", h_reshaped) / d
            val = bmps.contract_with_operator(self.peps, op1, r1, c1) / norm_sq
            total += float(np.real(val))

        return total


# -------------------------------------------------------------------
# Standard 2D bond Hamiltonians
# -------------------------------------------------------------------

def ising_2d_bonds(rows: int, cols: int, J: float = 1.0, h: float = 0.0) -> list:
    """Generate Ising model bond Hamiltonians for 2D lattice.

    H = -J sum_{<ij>} Z_i Z_j - h sum_i X_i

    The single-site transverse field terms are distributed equally
    among the bonds touching each site.

    Parameters
    ----------
    rows, cols : int
        Lattice dimensions.
    J : float
        ZZ coupling (default 1.0).
    h : float
        Transverse field (default 0.0).

    Returns
    -------
    list
        List of (row1, col1, row2, col2, h_bond) tuples.
    """
    bonds = []

    # Count number of bonds per site for field distribution
    coordination = np.zeros((rows, cols), dtype=int)
    bond_list = []
    for r in range(rows):
        for c in range(cols):
            if c < cols - 1:  # horizontal bond
                bond_list.append((r, c, r, c + 1))
                coordination[r, c] += 1
                coordination[r, c + 1] += 1
            if r < rows - 1:  # vertical bond
                bond_list.append((r, c, r + 1, c))
                coordination[r, c] += 1
                coordination[r + 1, c] += 1

    for r1, c1, r2, c2 in bond_list:
        h_bond = -J * np.kron(_Z, _Z)
        # Distribute transverse field
        if h != 0.0:
            z1 = coordination[r1, c1]
            z2 = coordination[r2, c2]
            if z1 > 0:
                h_bond -= (h / z1) * np.kron(_X, _I2)
            if z2 > 0:
                h_bond -= (h / z2) * np.kron(_I2, _X)
        bonds.append((r1, c1, r2, c2, h_bond))

    return bonds


def heisenberg_2d_bonds(rows: int, cols: int, J: float = 1.0) -> list:
    """Generate Heisenberg model bond Hamiltonians for 2D lattice.

    H = J sum_{<ij>} (X_i X_j + Y_i Y_j + Z_i Z_j)

    Parameters
    ----------
    rows, cols : int
        Lattice dimensions.
    J : float
        Exchange coupling (default 1.0).

    Returns
    -------
    list
        List of (row1, col1, row2, col2, h_bond) tuples.
    """
    h_heis = J * (np.kron(_X, _X) + np.kron(_Y, _Y) + np.kron(_Z, _Z))
    bonds = []

    for r in range(rows):
        for c in range(cols):
            if c < cols - 1:
                bonds.append((r, c, r, c + 1, h_heis.copy()))
            if r < rows - 1:
                bonds.append((r, c, r + 1, c, h_heis.copy()))

    return bonds
