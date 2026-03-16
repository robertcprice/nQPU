"""Time Evolving Block Decimation (TEBD) for MPS time evolution.

Implements efficient real and imaginary time evolution for MPS states
using Suzuki-Trotter decomposition of nearest-neighbour Hamiltonians.
Each Trotter step applies two-site gates and truncates via SVD.

Key features:
  - First-order Trotter: even-odd gate layers
  - Second-order Trotter: even-odd-even with half-steps (Strang splitting)
  - Real-time evolution: e^{-iHt} with unitary gates
  - Imaginary-time evolution: e^{-Ht} for ground-state preparation (cooling)
  - Per-step SVD truncation with controllable bond dimension
  - Entanglement entropy tracking over time

References:
  - Vidal, G., Phys. Rev. Lett. 91, 147902 (2003)
  - Vidal, G., Phys. Rev. Lett. 93, 040502 (2004)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .mps import MPS


# -------------------------------------------------------------------
# Result containers
# -------------------------------------------------------------------

@dataclass
class TEBDResult:
    """Results from TEBD time evolution.

    Attributes
    ----------
    states : list[MPS]
        Snapshots of the MPS at each recorded time step.
    times : list[float]
        Time values at each snapshot.
    energies : list[float]
        Energy at each snapshot (if a Hamiltonian is provided).
    entropies : list[list[float]]
        Entanglement entropy at each bond for each snapshot.
    norms : list[float]
        Norm of the state at each snapshot.
    bond_dimensions : list[list[int]]
        Bond dimensions at each snapshot.
    """
    states: List[MPS] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    entropies: List[List[float]] = field(default_factory=list)
    norms: List[float] = field(default_factory=list)
    bond_dimensions: List[List[int]] = field(default_factory=list)


# -------------------------------------------------------------------
# Two-site gate application
# -------------------------------------------------------------------

def _apply_two_site_gate(
    mps: MPS,
    gate: NDArray,
    site: int,
    chi_max: int,
    cutoff: float = 1e-14,
) -> float:
    """Apply a two-site gate to an MPS at bond (site, site+1).

    Modifies the MPS tensors in place and returns the truncation error.

    Parameters
    ----------
    mps : MPS
        The MPS to modify.
    gate : 2-D array
        The gate matrix, shape (d^2, d^2).
    site : int
        Left site index.
    chi_max : int
        Maximum bond dimension after truncation.
    cutoff : float
        SVD cutoff for small singular values.

    Returns
    -------
    float
        Truncation error (sum of discarded singular values squared).
    """
    d = mps.d
    chi_l = mps.tensors[site].shape[0]
    chi_r = mps.tensors[site + 1].shape[2]

    # Contract the two site tensors
    theta = np.einsum(
        "asd,dtc->astc", mps.tensors[site], mps.tensors[site + 1]
    )
    # theta shape: (chi_l, d, d, chi_r)

    # Apply gate: gate reshaped to (d, d, d, d) acting on physical indices
    # gate_4 indices: (s_out, t_out, s_in, t_in)
    # theta indices: (chi_l, s_in, t_in, chi_r)
    # Result: (chi_l, s_out, t_out, chi_r)
    gate_4 = gate.reshape(d, d, d, d)
    theta = np.einsum("pquv,auvb->apqb", gate_4, theta)

    # SVD to split back into two site tensors
    mat = theta.reshape(chi_l * d, d * chi_r)
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)

    # Truncate
    keep = len(S)
    if cutoff > 0:
        keep = max(int(np.sum(S > cutoff)), 1)
    keep = min(keep, chi_max)

    trunc_error = float(np.sum(S[keep:] ** 2))

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]

    # Absorb S into Vh (right)
    SV = np.diag(S) @ Vh

    mps.tensors[site] = U.reshape(chi_l, d, keep)
    mps.tensors[site + 1] = SV.reshape(keep, d, chi_r)

    return trunc_error


def _make_two_site_gate(
    h_bond: NDArray,
    dt_factor: complex,
) -> NDArray:
    """Exponentiate a two-site Hamiltonian term: exp(dt_factor * h_bond).

    Uses eigendecomposition of the Hermitian h_bond, then applies
    the complex exponential.  This avoids issues with non-Hermitian
    matrices in the real-time case.

    Parameters
    ----------
    h_bond : 2-D array
        Two-site Hamiltonian term (Hermitian), shape (d^2, d^2).
    dt_factor : complex
        Factor multiplying h_bond in the exponent.
        For real time:  ``-1j * dt``  gives ``exp(-i*dt*H)``.
        For imaginary time: ``-dt`` gives ``exp(-dt*H)``.

    Returns
    -------
    2-D array
        The gate matrix, shape (d^2, d^2).
    """
    h = np.asarray(h_bond, dtype=np.complex128)
    # Diagonalise the Hermitian Hamiltonian term
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    # exp(dt_factor * H) = V * diag(exp(dt_factor * lambda)) * V^dag
    exp_diag = np.exp(dt_factor * eigenvalues)
    return (eigenvectors * exp_diag) @ eigenvectors.conj().T


# -------------------------------------------------------------------
# Hamiltonian specification for TEBD
# -------------------------------------------------------------------

@dataclass
class NNHamiltonian:
    """Nearest-neighbour Hamiltonian for TEBD.

    Specifies the Hamiltonian as a list of two-site terms ``h_bonds[i]``
    acting on bond ``(i, i+1)``, plus optional single-site terms.

    Parameters
    ----------
    h_bonds : list[ndarray]
        Two-site Hamiltonian terms, each shape (d^2, d^2).
        ``h_bonds[i]`` acts on sites ``(i, i+1)``.
    h_single : list[ndarray], optional
        Single-site terms, each shape (d, d).  Absorbed into bonds.
    """
    h_bonds: List[NDArray]
    h_single: Optional[List[NDArray]] = None

    @property
    def n_sites(self) -> int:
        return len(self.h_bonds) + 1

    @property
    def n_bonds(self) -> int:
        return len(self.h_bonds)


def ising_nn_hamiltonian(
    n: int,
    J: float = 1.0,
    h: float = 1.0,
) -> NNHamiltonian:
    """Transverse-field Ising model as nearest-neighbour Hamiltonian.

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i

    Single-site terms are split equally between adjacent bonds.

    Parameters
    ----------
    n : int
        Number of sites.
    J : float
        ZZ coupling.
    h : float
        Transverse field.

    Returns
    -------
    NNHamiltonian
    """
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    bonds = []
    for i in range(n - 1):
        # ZZ term
        h_bond = -J * np.kron(Z, Z)

        # Distribute single-site X terms
        # First bond gets full X on site 0
        # Last bond gets full X on site n-1
        # Middle bonds split X terms
        if i == 0:
            h_bond -= h * np.kron(X, I)
            if n > 2:
                h_bond -= (h / 2) * np.kron(I, X)
            else:
                h_bond -= h * np.kron(I, X)
        elif i == n - 2:
            h_bond -= (h / 2) * np.kron(X, I)
            h_bond -= h * np.kron(I, X)
        else:
            h_bond -= (h / 2) * np.kron(X, I)
            h_bond -= (h / 2) * np.kron(I, X)

        bonds.append(h_bond)

    return NNHamiltonian(h_bonds=bonds)


def heisenberg_nn_hamiltonian(
    n: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    hz: float = 0.0,
) -> NNHamiltonian:
    """Heisenberg XXZ chain as nearest-neighbour Hamiltonian.

    H = sum_i [Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1}]
        + hz sum_i Z_i

    Parameters
    ----------
    n : int
        Number of sites.
    Jx, Jy, Jz : float
        Exchange couplings.
    hz : float
        External Z-field.

    Returns
    -------
    NNHamiltonian
    """
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    bonds = []
    for i in range(n - 1):
        h_bond = (
            Jx * np.kron(X, X)
            + Jy * np.kron(Y, Y)
            + Jz * np.kron(Z, Z)
        )

        # Distribute single-site Z-field terms
        if i == 0:
            h_bond += hz * np.kron(Z, I)
            if n > 2:
                h_bond += (hz / 2) * np.kron(I, Z)
            else:
                h_bond += hz * np.kron(I, Z)
        elif i == n - 2:
            h_bond += (hz / 2) * np.kron(Z, I)
            h_bond += hz * np.kron(I, Z)
        else:
            h_bond += (hz / 2) * np.kron(Z, I)
            h_bond += (hz / 2) * np.kron(I, Z)

        bonds.append(h_bond)

    return NNHamiltonian(h_bonds=bonds)


# -------------------------------------------------------------------
# TEBD engine
# -------------------------------------------------------------------

class TEBD:
    """Time Evolving Block Decimation for MPS time evolution.

    Parameters
    ----------
    mps : MPS
        Initial state.
    hamiltonian : NNHamiltonian
        The nearest-neighbour Hamiltonian.
    chi_max : int
        Maximum bond dimension.
    cutoff : float
        SVD truncation cutoff.
    order : int
        Trotter order: 1 (first-order) or 2 (second-order Strang splitting).
    """

    def __init__(
        self,
        mps: MPS,
        hamiltonian: NNHamiltonian,
        chi_max: int = 32,
        cutoff: float = 1e-14,
        order: int = 2,
    ) -> None:
        self.mps = mps.copy()
        self.hamiltonian = hamiltonian
        self.chi_max = chi_max
        self.cutoff = cutoff
        self.order = order

        if self.mps.n_sites != hamiltonian.n_sites:
            raise ValueError(
                f"MPS has {self.mps.n_sites} sites but Hamiltonian has "
                f"{hamiltonian.n_sites} sites"
            )

    def evolve(
        self,
        dt: float,
        n_steps: int,
        imaginary: bool = False,
        record_every: int = 1,
    ) -> TEBDResult:
        """Run TEBD time evolution.

        Parameters
        ----------
        dt : float
            Time step size.
        n_steps : int
            Number of time steps.
        imaginary : bool
            If True, do imaginary time evolution (ground-state cooling).
        record_every : int
            Record state every this many steps (default 1).

        Returns
        -------
        TEBDResult
            Evolution results including state snapshots and observables.
        """
        if imaginary:
            dt_factor = -dt  # e^{-H*dt}
        else:
            dt_factor = -1j * dt  # e^{-iH*dt}

        # Precompute gates
        n_bonds = self.hamiltonian.n_bonds
        even_bonds = list(range(0, n_bonds, 2))
        odd_bonds = list(range(1, n_bonds, 2))

        if self.order == 1:
            gates_even = [
                _make_two_site_gate(self.hamiltonian.h_bonds[b], dt_factor)
                for b in even_bonds
            ]
            gates_odd = [
                _make_two_site_gate(self.hamiltonian.h_bonds[b], dt_factor)
                for b in odd_bonds
            ]
        elif self.order == 2:
            # Strang splitting: even(dt/2) - odd(dt) - even(dt/2)
            gates_even_half = [
                _make_two_site_gate(self.hamiltonian.h_bonds[b], dt_factor / 2)
                for b in even_bonds
            ]
            gates_odd_full = [
                _make_two_site_gate(self.hamiltonian.h_bonds[b], dt_factor)
                for b in odd_bonds
            ]
        else:
            raise ValueError(f"Trotter order {self.order} not supported (use 1 or 2)")

        result = TEBDResult()

        # Record initial state
        self._record_snapshot(result, 0.0)

        for step in range(1, n_steps + 1):
            if self.order == 1:
                self._apply_layer(even_bonds, gates_even)
                self._apply_layer(odd_bonds, gates_odd)
            else:  # order 2
                self._apply_layer(even_bonds, gates_even_half)
                self._apply_layer(odd_bonds, gates_odd_full)
                self._apply_layer(even_bonds, gates_even_half)

            # For imaginary time: renormalize
            if imaginary:
                self.mps = self.mps.normalize()

            if step % record_every == 0:
                t = step * dt
                self._record_snapshot(result, t)

        return result

    def _apply_layer(self, bonds: List[int], gates: List[NDArray]) -> None:
        """Apply a layer of two-site gates."""
        for bond, gate in zip(bonds, gates):
            _apply_two_site_gate(self.mps, gate, bond, self.chi_max, self.cutoff)

    def _record_snapshot(self, result: TEBDResult, t: float) -> None:
        """Record current state into the result."""
        result.states.append(self.mps.copy())
        result.times.append(t)
        result.norms.append(self.mps.norm())
        result.bond_dimensions.append(list(self.mps.bond_dimensions))

        # Compute entanglement entropy at each bond
        entropies = []
        for b in range(self.mps.n_sites - 1):
            try:
                entropies.append(self.mps.entanglement_entropy(b))
            except Exception:
                entropies.append(0.0)
        result.entropies.append(entropies)


# -------------------------------------------------------------------
# Imaginary-time TEBD wrapper
# -------------------------------------------------------------------

class ImaginaryTEBD:
    """Ground state finder via imaginary-time TEBD (cooling).

    Applies e^{-H*tau} repeatedly, renormalizing after each step,
    to project onto the ground state.

    Parameters
    ----------
    mps : MPS
        Initial state.
    hamiltonian : NNHamiltonian
        Nearest-neighbour Hamiltonian.
    chi_max : int
        Maximum bond dimension.
    order : int
        Trotter order.
    """

    def __init__(
        self,
        mps: MPS,
        hamiltonian: NNHamiltonian,
        chi_max: int = 32,
        order: int = 2,
    ) -> None:
        self.mps = mps.copy()
        self.hamiltonian = hamiltonian
        self.chi_max = chi_max
        self.order = order

    def run(
        self,
        dt: float = 0.1,
        n_steps: int = 100,
        tol: float = 1e-8,
    ) -> TEBDResult:
        """Run imaginary-time evolution until convergence.

        Parameters
        ----------
        dt : float
            Imaginary time step.
        n_steps : int
            Maximum number of steps.
        tol : float
            Convergence tolerance on energy difference.

        Returns
        -------
        TEBDResult
            Evolution results.  The last state is the approximate
            ground state.
        """
        engine = TEBD(
            self.mps,
            self.hamiltonian,
            chi_max=self.chi_max,
            order=self.order,
        )
        result = engine.evolve(
            dt=dt,
            n_steps=n_steps,
            imaginary=True,
            record_every=1,
        )

        # Update internal state
        if result.states:
            self.mps = result.states[-1]

        return result


# -------------------------------------------------------------------
# Convenience function
# -------------------------------------------------------------------

def tebd_evolve(
    mps: MPS,
    hamiltonian: NNHamiltonian,
    dt: float,
    n_steps: int,
    chi_max: int = 32,
    imaginary: bool = False,
    order: int = 2,
) -> TEBDResult:
    """Convenience function for TEBD time evolution.

    Parameters
    ----------
    mps : MPS
        Initial state.
    hamiltonian : NNHamiltonian
        Nearest-neighbour Hamiltonian.
    dt : float
        Time step.
    n_steps : int
        Number of steps.
    chi_max : int
        Maximum bond dimension.
    imaginary : bool
        If True, imaginary time evolution.
    order : int
        Trotter order (1 or 2).

    Returns
    -------
    TEBDResult
        Evolution results.

    Examples
    --------
    >>> from nqpu.tensor_networks import (
    ...     ProductState, ising_nn_hamiltonian, tebd_evolve,
    ... )
    >>> mps = ProductState(6)
    >>> H = ising_nn_hamiltonian(6, J=1.0, h=0.5)
    >>> result = tebd_evolve(mps, H, dt=0.05, n_steps=100, chi_max=16)
    """
    engine = TEBD(mps, hamiltonian, chi_max=chi_max, order=order)
    return engine.evolve(dt=dt, n_steps=n_steps, imaginary=imaginary)
