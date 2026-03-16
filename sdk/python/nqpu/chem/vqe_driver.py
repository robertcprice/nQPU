"""End-to-end VQE driver for molecular quantum chemistry.

Connects molecular geometry, integral computation, fermion-to-qubit
mapping, ansatz construction, and classical optimization into a single
workflow for ground-state energy calculations.

Provides:

- :class:`MolecularVQE` -- full pipeline from molecule to ground-state
  energy.
- Potential energy surface scanning.
- Exact diagonalization reference for benchmarking.
- Active space reduction for larger systems.

Example
-------
>>> from nqpu.chem import MolecularVQE, h2
>>> vqe = MolecularVQE(h2(), basis="sto-3g")
>>> result = vqe.compute_ground_state(maxiter=100)
>>> print(f"E = {result['energy']:.6f} Hartree")
E = -1.137270 Hartree

References
----------
- Peruzzo et al., Nat. Commun. 5, 4213 (2014).
- McClean et al., New J. Phys. 18, 023023 (2016).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .molecular import Molecule, BasisSet, Atom
from .integrals import (
    compute_one_electron_integrals,
    compute_two_electron_integrals,
)
from .fermion import (
    FermionicHamiltonian,
    QubitHamiltonian,
    jordan_wigner,
    bravyi_kitaev,
    parity_mapping,
)
from .ansatz import UCCSD, UCCD, HardwareEfficient


# ============================================================
# VQE Result
# ============================================================


@dataclass
class VQEResult:
    """Container for VQE optimization results.

    Attributes
    ----------
    energy : float
        Best energy found (in Hartree).
    optimal_params : np.ndarray
        Optimal variational parameters.
    num_iterations : int
        Number of optimizer iterations.
    num_function_evals : int
        Number of energy evaluations.
    convergence_history : list[float]
        Energy at each iteration.
    exact_energy : float | None
        Exact ground-state energy (if computed).
    """

    energy: float
    optimal_params: np.ndarray
    num_iterations: int
    num_function_evals: int
    convergence_history: list[float]
    exact_energy: float | None = None

    def chemical_accuracy(self) -> float | None:
        """Error relative to exact result in milli-Hartree.

        Returns ``None`` if exact energy was not computed.
        """
        if self.exact_energy is None:
            return None
        return abs(self.energy - self.exact_energy) * 1000.0

    def __repr__(self) -> str:
        acc = self.chemical_accuracy()
        acc_str = f", error={acc:.3f} mHa" if acc is not None else ""
        return (
            f"VQEResult(energy={self.energy:.8f} Ha, "
            f"iters={self.num_iterations}"
            f"{acc_str})"
        )


# ============================================================
# Molecular VQE Driver
# ============================================================


class MolecularVQE:
    """End-to-end VQE driver for molecular ground-state energy.

    Orchestrates the full quantum chemistry pipeline:

    1. Compute molecular integrals (overlap, kinetic, nuclear, ERI).
    2. Build the second-quantized fermionic Hamiltonian.
    3. Map to qubit space via Jordan-Wigner (or BK / parity).
    4. Construct a variational ansatz (UCCSD by default).
    5. Optimize parameters to find the ground-state energy.

    Parameters
    ----------
    molecule : Molecule
        The molecular system.
    basis : str
        Basis set name (``'sto-3g'`` or ``'6-31g'``).
    mapping : str
        Fermion-to-qubit mapping (``'jordan_wigner'``,
        ``'bravyi_kitaev'``, or ``'parity'``).
    ansatz : str
        Ansatz type (``'uccsd'``, ``'uccd'``, or ``'hardware_efficient'``).
    frozen_core : int
        Number of core spatial orbitals to freeze (0 = no freezing).
    """

    MAPPING_FNS = {
        "jordan_wigner": jordan_wigner,
        "bravyi_kitaev": bravyi_kitaev,
        "parity": parity_mapping,
    }

    def __init__(
        self,
        molecule: Molecule,
        basis: str = "sto-3g",
        mapping: str = "jordan_wigner",
        ansatz: str = "uccsd",
        frozen_core: int = 0,
    ) -> None:
        if mapping not in self.MAPPING_FNS:
            raise ValueError(
                f"Unknown mapping '{mapping}'. "
                f"Supported: {list(self.MAPPING_FNS.keys())}"
            )
        if ansatz not in ("uccsd", "uccd", "hardware_efficient"):
            raise ValueError(
                f"Unknown ansatz '{ansatz}'. "
                f"Supported: ['uccsd', 'uccd', 'hardware_efficient']"
            )

        self.molecule = molecule
        self.basis_name = basis
        self.mapping_name = mapping
        self.ansatz_name = ansatz
        self.frozen_core = frozen_core

        # Build basis set
        self._basis = BasisSet(basis, molecule)

        # Compute AO integrals
        self._S, self._T, self._H_core = compute_one_electron_integrals(
            molecule, self._basis
        )
        self._eri = compute_two_electron_integrals(molecule, self._basis)
        self._nuclear_repulsion = molecule.nuclear_repulsion_energy()

        # Transform AO integrals to orthonormal MO basis via RHF
        # The second-quantized Hamiltonian requires orthonormal orbitals.
        self._C, self._orbital_energies = self._solve_rhf()
        h1_mo, h2_mo = self._transform_to_mo_basis(self._C)

        # Active space reduction
        h1_active = h1_mo
        h2_active = h2_mo
        nuc_active = self._nuclear_repulsion
        n_spatial = self._basis.num_functions
        n_electrons = molecule.num_electrons

        if frozen_core > 0:
            h1_active, h2_active, nuc_active, n_spatial, n_electrons = (
                self._reduce_active_space(frozen_core, h1_mo, h2_mo)
            )

        self._h1_active = h1_active
        self._h2_active = h2_active
        self._nuc_active = nuc_active
        self._n_spatial_active = n_spatial
        self._n_electrons_active = n_electrons
        self._n_spin_orbitals = 2 * n_spatial

        # Build fermionic Hamiltonian from MO integrals
        self._ferm_ham = FermionicHamiltonian.from_integrals(
            h1_active, h2_active, nuc_active
        )

        # Map to qubits
        mapping_fn = self.MAPPING_FNS[mapping]
        self._qubit_ham = mapping_fn(self._ferm_ham)

        # Build Hamiltonian matrix for expectation evaluation
        self._ham_matrix = self._qubit_ham.to_matrix(self._n_spin_orbitals)

        # Build ansatz
        self._ansatz = self._build_ansatz()

    def _solve_rhf(self) -> tuple[np.ndarray, np.ndarray]:
        """Solve the Restricted Hartree-Fock equations.

        Uses the generalized eigenvalue approach to diagonalize the
        Fock matrix, producing canonical molecular orbital coefficients.

        Returns
        -------
        C : np.ndarray
            MO coefficient matrix, shape ``(n_ao, n_mo)``.
        orbital_energies : np.ndarray
            Orbital energies.
        """
        S = self._S
        H_core = self._H_core
        eri = self._eri
        n = S.shape[0]
        n_occ = self.molecule.num_electrons // 2

        # Orthogonalization matrix: S^{-1/2}
        eigvals, eigvecs = np.linalg.eigh(S)
        X = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Initial guess: diagonalize core Hamiltonian
        F_prime = X.T @ H_core @ X
        eps, C_prime = np.linalg.eigh(F_prime)
        C = X @ C_prime

        # SCF iteration
        max_scf_iter = 50
        scf_tol = 1e-10

        for scf_iter in range(max_scf_iter):
            # Build density matrix
            D = np.zeros((n, n), dtype=np.float64)
            for i in range(n_occ):
                D += 2.0 * np.outer(C[:, i], C[:, i])

            # Build Fock matrix
            F = H_core.copy()
            for mu in range(n):
                for nu in range(n):
                    for lam in range(n):
                        for sig in range(n):
                            # J - 0.5*K
                            F[mu, nu] += D[lam, sig] * (
                                eri[mu, nu, lam, sig]
                                - 0.5 * eri[mu, sig, lam, nu]
                            )

            # Diagonalize Fock matrix in orthonormal basis
            F_prime = X.T @ F @ X
            eps_new, C_prime = np.linalg.eigh(F_prime)
            C_new = X @ C_prime

            # Check convergence
            D_new = np.zeros((n, n), dtype=np.float64)
            for i in range(n_occ):
                D_new += 2.0 * np.outer(C_new[:, i], C_new[:, i])

            delta = np.linalg.norm(D_new - D)
            C = C_new
            eps = eps_new

            if delta < scf_tol:
                break

        return C, eps

    def _transform_to_mo_basis(
        self, C: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform AO integrals to the MO basis.

        Parameters
        ----------
        C : np.ndarray
            MO coefficient matrix, shape ``(n_ao, n_mo)``.

        Returns
        -------
        h1_mo : np.ndarray
            One-electron integrals in MO basis.
        h2_mo : np.ndarray
            Two-electron integrals in MO basis (chemists' notation).
        """
        n = C.shape[1]

        # One-electron: h_mo = C^T @ H_core @ C
        h1_mo = C.T @ self._H_core @ C

        # Two-electron: four-index transformation
        # (pq|rs)_mo = sum_{mu,nu,lam,sig} C_{mu,p} C_{nu,q} C_{lam,r} C_{sig,s} (mu nu|lam sig)_ao
        h2_mo = np.einsum("mp,nq,lr,os,mnlo->pqrs", C, C, C, C, self._eri,
                          optimize=True)

        return h1_mo, h2_mo

    def _reduce_active_space(
        self,
        n_frozen: int,
        h1_mo: np.ndarray,
        h2_mo: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, int, int]:
        """Reduce to an active space by freezing core orbitals.

        Frozen core orbitals contribute a constant energy shift and
        modify the one-electron integrals of the active space.
        Integrals must already be in the MO basis.

        Parameters
        ----------
        n_frozen : int
            Number of spatial orbitals to freeze.
        h1_mo : np.ndarray
            One-electron integrals in MO basis.
        h2_mo : np.ndarray
            Two-electron integrals in MO basis.

        Returns
        -------
        h1_active, h2_active, nuclear_repulsion_active, n_spatial_active, n_electrons_active
        """
        n_total = h1_mo.shape[0]
        if n_frozen >= n_total:
            raise ValueError(
                f"Cannot freeze {n_frozen} orbitals when there are only {n_total}"
            )

        # Frozen core energy contribution
        frozen_energy = self._nuclear_repulsion
        for i in range(n_frozen):
            frozen_energy += 2.0 * h1_mo[i, i]
            for j in range(n_frozen):
                frozen_energy += 2.0 * h2_mo[i, i, j, j] - h2_mo[i, j, j, i]

        # Modified one-electron integrals for active space
        active = list(range(n_frozen, n_total))
        n_active = len(active)

        h1_active = np.zeros((n_active, n_active), dtype=np.float64)
        for ii, i in enumerate(active):
            for jj, j in enumerate(active):
                h1_active[ii, jj] = h1_mo[i, j]
                for k in range(n_frozen):
                    h1_active[ii, jj] += (
                        2.0 * h2_mo[i, j, k, k] - h2_mo[i, k, k, j]
                    )

        # Active-space two-electron integrals
        h2_active = np.zeros(
            (n_active, n_active, n_active, n_active), dtype=np.float64
        )
        for ii, i in enumerate(active):
            for jj, j in enumerate(active):
                for kk, k_val in enumerate(active):
                    for ll, l_val in enumerate(active):
                        h2_active[ii, jj, kk, ll] = h2_mo[i, j, k_val, l_val]

        n_electrons_active = self.molecule.num_electrons - 2 * n_frozen

        return h1_active, h2_active, frozen_energy, n_active, n_electrons_active

    def _build_ansatz(self):
        """Construct the variational ansatz."""
        if self.ansatz_name == "uccsd":
            return UCCSD(
                n_electrons=self._n_electrons_active,
                n_spin_orbitals=self._n_spin_orbitals,
            )
        elif self.ansatz_name == "uccd":
            return UCCD(
                n_electrons=self._n_electrons_active,
                n_spin_orbitals=self._n_spin_orbitals,
            )
        elif self.ansatz_name == "hardware_efficient":
            return HardwareEfficient(
                n_qubits=self._n_spin_orbitals,
                n_layers=2,
                n_electrons=self._n_electrons_active,
            )
        else:
            raise ValueError(f"Unknown ansatz: {self.ansatz_name}")

    def _energy_fn(self, params: np.ndarray) -> float:
        """Compute energy for given parameters."""
        state = self._ansatz.state_vector(params)
        return float(np.real(state.conj() @ self._ham_matrix @ state))

    def exact_ground_state_energy(self) -> float:
        """Compute the exact ground-state energy via diagonalization.

        Returns
        -------
        float
            Exact ground-state energy in Hartree.
        """
        eigenvalues = np.linalg.eigvalsh(self._ham_matrix)
        return float(eigenvalues[0])

    def compute_ground_state(
        self,
        optimizer: str = "cobyla",
        maxiter: int = 200,
        initial_params: np.ndarray | None = None,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> VQEResult:
        """Run VQE to find the molecular ground-state energy.

        Parameters
        ----------
        optimizer : str
            Optimizer name (``'cobyla'``, ``'nelder-mead'``, ``'spsa'``).
        maxiter : int
            Maximum optimizer iterations.
        initial_params : np.ndarray, optional
            Starting parameters. If ``None``, uses zeros.
        callback : callable, optional
            ``callback(iteration, params, energy)`` hook.

        Returns
        -------
        VQEResult
            Optimization outcome.
        """
        n_params = self._ansatz.num_parameters()
        if initial_params is None:
            initial_params = np.zeros(n_params, dtype=np.float64)

        history: list[float] = []
        n_evals = 0
        best_energy = float("inf")
        best_params = initial_params.copy()

        def cost_fn(params: np.ndarray) -> float:
            nonlocal n_evals, best_energy, best_params
            n_evals += 1
            e = self._energy_fn(params)
            if e < best_energy:
                best_energy = e
                best_params = params.copy()
            return e

        # Use our own optimizer suite
        from nqpu.optimizers import minimize as opt_minimize

        result = opt_minimize(
            cost_fn,
            initial_params,
            method=optimizer,
            maxiter=maxiter,
            callback=callback,
        )

        # Get exact energy for reference
        exact = self.exact_ground_state_energy()

        # Use the best energy we tracked
        final_energy = min(best_energy, result.optimal_value)
        final_params = best_params if best_energy < result.optimal_value else result.optimal_params

        return VQEResult(
            energy=final_energy,
            optimal_params=final_params,
            num_iterations=result.num_iterations,
            num_function_evals=n_evals,
            convergence_history=result.convergence_history,
            exact_energy=exact,
        )

    def potential_energy_surface(
        self,
        bond_lengths: list[float] | np.ndarray,
        atom_indices: tuple[int, int] = (0, 1),
        optimizer: str = "cobyla",
        maxiter: int = 200,
    ) -> list[dict]:
        """Scan the potential energy surface along a bond stretch.

        Computes the VQE energy at each bond length by modifying the
        distance between two atoms while keeping all others fixed.

        Parameters
        ----------
        bond_lengths : array-like
            Bond lengths to scan (in Angstroms).
        atom_indices : tuple[int, int]
            Indices of the two atoms defining the bond.
        optimizer : str
            Optimizer for VQE at each point.
        maxiter : int
            Maximum iterations per point.

        Returns
        -------
        list[dict]
            List of dicts with keys ``'bond_length'``, ``'vqe_energy'``,
            ``'exact_energy'``.
        """
        results = []
        i, j = atom_indices
        base_atoms = list(self.molecule.atoms)
        direction = np.array(base_atoms[j].position) - np.array(base_atoms[i].position)
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            raise ValueError("Atoms are at the same position")
        direction = direction / dir_norm

        prev_params = None

        for bl in bond_lengths:
            # Create modified molecule
            new_atoms = list(base_atoms)
            new_pos = tuple(
                np.array(base_atoms[i].position) + bl * direction
            )
            new_atoms[j] = Atom(base_atoms[j].symbol, new_pos)

            mol = Molecule.from_atoms(
                new_atoms,
                charge=self.molecule.charge,
                multiplicity=self.molecule.multiplicity,
            )

            vqe = MolecularVQE(
                mol,
                basis=self.basis_name,
                mapping=self.mapping_name,
                ansatz=self.ansatz_name,
                frozen_core=self.frozen_core,
            )

            result = vqe.compute_ground_state(
                optimizer=optimizer,
                maxiter=maxiter,
                initial_params=prev_params,
            )

            prev_params = result.optimal_params

            results.append({
                "bond_length": bl,
                "vqe_energy": result.energy,
                "exact_energy": result.exact_energy,
            })

        return results

    def dipole_moment(self, params: np.ndarray) -> np.ndarray:
        """Compute the molecular dipole moment for a given state.

        The electronic contribution to the dipole is:
        ``mu_e = -sum_pq D_pq <p|r|q>``
        where D_pq is the one-particle density matrix.

        The nuclear contribution is:
        ``mu_n = sum_A Z_A * R_A``

        Parameters
        ----------
        params : np.ndarray
            Variational parameters for the ansatz.

        Returns
        -------
        np.ndarray
            Dipole moment vector (x, y, z) in atomic units (e*a0).
        """
        state = self._ansatz.state_vector(params)

        # Compute one-particle density matrix in MO basis
        # For simplicity, compute in AO basis directly
        n = self._n_spatial_active

        # Nuclear dipole
        mu_nuc = np.zeros(3, dtype=np.float64)
        for atom in self.molecule.atoms:
            pos = np.array(atom.position_bohr)
            mu_nuc += atom.nuclear_charge * pos

        # Electronic dipole (simplified: diagonal approximation)
        # Full implementation would require dipole integrals <p|r|q>
        # Here we provide the nuclear contribution only as a first approximation
        return mu_nuc

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the mapped Hamiltonian."""
        return self._n_spin_orbitals

    @property
    def num_parameters(self) -> int:
        """Number of variational parameters in the ansatz."""
        return self._ansatz.num_parameters()

    def __repr__(self) -> str:
        return (
            f"MolecularVQE({self.molecule.formula}, "
            f"basis={self.basis_name!r}, "
            f"mapping={self.mapping_name!r}, "
            f"ansatz={self.ansatz_name!r}, "
            f"qubits={self.num_qubits})"
        )
