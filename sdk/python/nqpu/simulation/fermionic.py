"""Fermionic Gaussian state simulation with O(n^3) evolution.

Provides efficient simulation of non-interacting (quadratic) fermionic
systems using the Gaussian state formalism.  A Gaussian state is fully
characterised by its two-point correlation matrix Gamma_ij = <c^dag_i c_j>,
and time evolution under a quadratic Hamiltonian H = sum_ij h_ij c^dag_i c_j
reduces to O(n^3) matrix operations rather than O(2^n) full state evolution.

Features:
  - Fermionic creation/annihilation operators in Fock space (for validation).
  - Gaussian states with correlation matrix representation.
  - Quadratic Hamiltonians: tight-binding, SSH model, custom hopping.
  - Efficient O(n^3) time evolution of correlation matrices.
  - Ground state computation via single-particle spectrum.
  - Wick's theorem for multi-point correlator evaluation.
  - Entanglement entropy from correlation matrix eigenvalues.

References:
    - Peschel, J. Phys. A 36, L205 (2003) [Gaussian state entropy]
    - Bravyi, Quantum Inf. Comp. 5, 216 (2005) [Gaussian state formalism]
    - Su, Schrieffer & Heeger, Phys. Rev. Lett. 42, 1698 (1979) [SSH model]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# FermionicMode
# ---------------------------------------------------------------------------


@dataclass
class FermionicMode:
    """Single fermionic mode with creation/annihilation operators.

    Builds full 2^n Fock-space representations of fermionic operators
    using Jordan-Wigner encoding for validation against Gaussian
    state results.

    Parameters
    ----------
    index : int
        Mode index (0-based).
    """

    index: int

    def creation_matrix(self, n_modes: int) -> np.ndarray:
        """Build creation operator c^dag_i in Fock space.

        Uses Jordan-Wigner encoding: c^dag_i = (prod_{j<i} Z_j) sigma+_i

        Parameters
        ----------
        n_modes : int
            Total number of fermionic modes.

        Returns
        -------
        np.ndarray
            Matrix of shape (2^n_modes, 2^n_modes).
        """
        dim = 2 ** n_modes
        result = np.zeros((dim, dim), dtype=np.complex128)

        for basis in range(dim):
            # Check if mode i is unoccupied
            if (basis >> (n_modes - 1 - self.index)) & 1 == 0:
                # Create particle at mode i
                new_basis = basis | (1 << (n_modes - 1 - self.index))
                # Jordan-Wigner sign: count occupied modes with j < i
                sign = 1
                for j in range(self.index):
                    if (basis >> (n_modes - 1 - j)) & 1:
                        sign *= -1
                result[new_basis, basis] = sign

        return result

    def annihilation_matrix(self, n_modes: int) -> np.ndarray:
        """Build annihilation operator c_i in Fock space.

        Parameters
        ----------
        n_modes : int
            Total number of fermionic modes.

        Returns
        -------
        np.ndarray
            Matrix of shape (2^n_modes, 2^n_modes).
        """
        return self.creation_matrix(n_modes).conj().T

    def number_operator(self, n_modes: int) -> np.ndarray:
        """Build number operator n_i = c^dag_i c_i.

        Parameters
        ----------
        n_modes : int
            Total number of fermionic modes.

        Returns
        -------
        np.ndarray
            Diagonal matrix of shape (2^n_modes, 2^n_modes).
        """
        c_dag = self.creation_matrix(n_modes)
        c = self.annihilation_matrix(n_modes)
        return c_dag @ c


# ---------------------------------------------------------------------------
# GaussianState
# ---------------------------------------------------------------------------


@dataclass
class GaussianState:
    """Fermionic Gaussian state defined by its correlation matrix.

    A Gaussian state is fully characterised by the 2-point correlation
    matrix Gamma_ij = <c^dag_i c_j>.  All higher-order correlators
    decompose into products of 2-point functions via Wick's theorem.

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Hermitian matrix of shape (n_modes, n_modes) with eigenvalues
        in [0, 1].
    """

    correlation_matrix: np.ndarray

    def __post_init__(self) -> None:
        self.correlation_matrix = np.asarray(
            self.correlation_matrix, dtype=np.complex128
        )

    @property
    def n_modes(self) -> int:
        """Number of fermionic modes."""
        return self.correlation_matrix.shape[0]

    @property
    def particle_number(self) -> float:
        """Expected total particle number <N> = Tr(Gamma)."""
        return float(np.real(np.trace(self.correlation_matrix)))

    @staticmethod
    def vacuum(n_modes: int) -> "GaussianState":
        """Create vacuum state (no particles).

        Returns
        -------
        GaussianState
            State with Gamma = 0.
        """
        return GaussianState(np.zeros((n_modes, n_modes), dtype=np.complex128))

    @staticmethod
    def filled(n_modes: int, filled_modes: list) -> "GaussianState":
        """Create state with specified modes occupied.

        Parameters
        ----------
        n_modes : int
            Total number of modes.
        filled_modes : list[int]
            Indices of occupied modes.

        Returns
        -------
        GaussianState
        """
        gamma = np.zeros((n_modes, n_modes), dtype=np.complex128)
        for i in filled_modes:
            gamma[i, i] = 1.0
        return GaussianState(gamma)

    @staticmethod
    def half_filled(n_modes: int) -> "GaussianState":
        """Create half-filled state (first n/2 modes occupied).

        Returns
        -------
        GaussianState
        """
        n_filled = n_modes // 2
        return GaussianState.filled(n_modes, list(range(n_filled)))

    @staticmethod
    def from_state_vector(psi: np.ndarray, n_modes: int) -> "GaussianState":
        """Construct Gaussian state from a Fock-space state vector.

        Computes the correlation matrix Gamma_ij = <psi|c^dag_i c_j|psi>
        from a full state vector.

        Parameters
        ----------
        psi : np.ndarray
            State vector of length 2^n_modes.
        n_modes : int
            Number of fermionic modes.

        Returns
        -------
        GaussianState
        """
        psi = np.asarray(psi, dtype=np.complex128).ravel()
        gamma = np.zeros((n_modes, n_modes), dtype=np.complex128)

        for i in range(n_modes):
            ci = FermionicMode(i)
            c_dag_i = ci.creation_matrix(n_modes)
            for j in range(n_modes):
                cj = FermionicMode(j)
                c_j = cj.annihilation_matrix(n_modes)
                gamma[i, j] = psi.conj() @ (c_dag_i @ c_j) @ psi

        return GaussianState(gamma)

    def entropy(self) -> float:
        """Von Neumann entropy from correlation matrix eigenvalues.

        For a Gaussian state, S = -sum_k [n_k log(n_k) + (1-n_k) log(1-n_k)]
        where n_k are the eigenvalues of the correlation matrix.

        Returns
        -------
        float
            Entropy in nats.
        """
        eigvals = np.linalg.eigvalsh(self.correlation_matrix)
        eigvals = np.real(eigvals)
        eigvals = np.clip(eigvals, 1e-15, 1 - 1e-15)
        return float(-np.sum(
            eigvals * np.log(eigvals)
            + (1 - eigvals) * np.log(1 - eigvals)
        ))

    def subsystem_entropy(self, subsystem: list) -> float:
        """Entanglement entropy of a subsystem.

        Parameters
        ----------
        subsystem : list[int]
            Mode indices forming the subsystem.

        Returns
        -------
        float
            Von Neumann entropy of the reduced state on the subsystem.
        """
        sub_gamma = self.correlation_matrix[np.ix_(subsystem, subsystem)]
        eigvals = np.linalg.eigvalsh(sub_gamma)
        eigvals = np.real(eigvals)
        eigvals = np.clip(eigvals, 1e-15, 1 - 1e-15)
        return float(-np.sum(
            eigvals * np.log(eigvals)
            + (1 - eigvals) * np.log(1 - eigvals)
        ))

    def mutual_information(
        self, subsystem_a: list, subsystem_b: list
    ) -> float:
        """Mutual information I(A:B) between two subsystems.

        I(A:B) = S(A) + S(B) - S(AB)

        Parameters
        ----------
        subsystem_a : list[int]
            Mode indices for subsystem A.
        subsystem_b : list[int]
            Mode indices for subsystem B.

        Returns
        -------
        float
            Mutual information in nats.
        """
        s_a = self.subsystem_entropy(subsystem_a)
        s_b = self.subsystem_entropy(subsystem_b)
        s_ab = self.subsystem_entropy(subsystem_a + subsystem_b)
        return max(0.0, s_a + s_b - s_ab)

    def is_valid(self, atol: float = 1e-10) -> bool:
        """Check whether this is a valid fermionic Gaussian state.

        The correlation matrix must be Hermitian with eigenvalues in [0, 1].
        """
        gamma = self.correlation_matrix
        # Hermiticity
        if not np.allclose(gamma, gamma.conj().T, atol=atol):
            return False
        # Eigenvalue bounds
        eigvals = np.linalg.eigvalsh(gamma)
        return bool(np.all(eigvals >= -atol) and np.all(eigvals <= 1 + atol))

    def occupation_numbers(self) -> np.ndarray:
        """Diagonal elements of the correlation matrix (site occupations).

        Returns
        -------
        np.ndarray
            Array of shape (n_modes,) with occupation numbers.
        """
        return np.real(np.diag(self.correlation_matrix))


# ---------------------------------------------------------------------------
# QuadraticHamiltonian
# ---------------------------------------------------------------------------


@dataclass
class QuadraticHamiltonian:
    """Quadratic fermionic Hamiltonian H = sum_ij h_ij c^dag_i c_j.

    The single-particle hopping matrix h_ij fully specifies the
    Hamiltonian.  The many-body spectrum is determined by the
    single-particle eigenvalues.

    Parameters
    ----------
    hopping_matrix : np.ndarray
        Hermitian matrix h_ij of shape (n_modes, n_modes).
    """

    hopping_matrix: np.ndarray

    def __post_init__(self) -> None:
        self.hopping_matrix = np.asarray(
            self.hopping_matrix, dtype=np.complex128
        )

    @property
    def n_modes(self) -> int:
        """Number of fermionic modes."""
        return self.hopping_matrix.shape[0]

    @staticmethod
    def tight_binding_1d(
        n_sites: int,
        t: float = 1.0,
        mu: float = 0.0,
        periodic: bool = False,
    ) -> "QuadraticHamiltonian":
        """1D tight-binding chain.

        H = -t sum_<ij> (c^dag_i c_j + h.c.) - mu sum_i n_i

        Parameters
        ----------
        n_sites : int
            Number of lattice sites.
        t : float
            Hopping amplitude.
        mu : float
            Chemical potential.
        periodic : bool
            Periodic boundary conditions.

        Returns
        -------
        QuadraticHamiltonian
        """
        h = np.zeros((n_sites, n_sites), dtype=np.complex128)

        # Nearest-neighbour hopping
        for i in range(n_sites - 1):
            h[i, i + 1] = -t
            h[i + 1, i] = -t

        # Periodic boundary
        if periodic and n_sites > 2:
            h[0, n_sites - 1] = -t
            h[n_sites - 1, 0] = -t

        # Chemical potential
        for i in range(n_sites):
            h[i, i] = -mu

        return QuadraticHamiltonian(h)

    @staticmethod
    def ssh_model(
        n_cells: int,
        t1: float = 1.0,
        t2: float = 0.5,
    ) -> "QuadraticHamiltonian":
        """Su-Schrieffer-Heeger model.

        A 1D chain with alternating hopping amplitudes t1 (intra-cell)
        and t2 (inter-cell), exhibiting topological band structure.

        H = sum_i [ t1 (c^dag_{2i} c_{2i+1} + h.c.)
                   + t2 (c^dag_{2i+1} c_{2i+2} + h.c.) ]

        Parameters
        ----------
        n_cells : int
            Number of unit cells (total sites = 2 * n_cells).
        t1 : float
            Intra-cell hopping.
        t2 : float
            Inter-cell hopping.

        Returns
        -------
        QuadraticHamiltonian
        """
        n_sites = 2 * n_cells
        h = np.zeros((n_sites, n_sites), dtype=np.complex128)

        for cell in range(n_cells):
            a = 2 * cell      # A sublattice
            b = 2 * cell + 1  # B sublattice

            # Intra-cell hopping
            h[a, b] = -t1
            h[b, a] = -t1

            # Inter-cell hopping
            if cell < n_cells - 1:
                b_current = 2 * cell + 1
                a_next = 2 * (cell + 1)
                h[b_current, a_next] = -t2
                h[a_next, b_current] = -t2

        return QuadraticHamiltonian(h)

    @staticmethod
    def random(
        n_modes: int,
        seed: Optional[int] = None,
    ) -> "QuadraticHamiltonian":
        """Generate a random quadratic Hamiltonian.

        Parameters
        ----------
        n_modes : int
            Number of modes.
        seed : int or None
            Random seed.

        Returns
        -------
        QuadraticHamiltonian
        """
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(n_modes, n_modes)) + \
            1j * rng.normal(size=(n_modes, n_modes))
        h = (A + A.conj().T) / 2.0
        return QuadraticHamiltonian(h)

    def energy(self, state: GaussianState) -> float:
        """Compute <H> = Tr(h * Gamma).

        Parameters
        ----------
        state : GaussianState
            The Gaussian state to evaluate.

        Returns
        -------
        float
            Energy expectation value.
        """
        return float(np.real(
            np.trace(self.hopping_matrix @ state.correlation_matrix)
        ))

    def single_particle_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalise the hopping matrix.

        Returns
        -------
        eigenvalues : np.ndarray
            Single-particle energies (sorted ascending).
        eigenvectors : np.ndarray
            Column eigenvectors.
        """
        evals, evecs = np.linalg.eigh(self.hopping_matrix)
        return evals, evecs

    def ground_state(self) -> GaussianState:
        """Find ground state by filling negative-energy single-particle states.

        The many-body ground state of a quadratic Hamiltonian is a Slater
        determinant formed by occupying all single-particle states with
        negative energy.

        Returns
        -------
        GaussianState
            The ground state.
        """
        evals, evecs = self.single_particle_spectrum()

        # Fill modes with negative (or zero) energy
        occupied_mask = evals <= 0
        if not np.any(occupied_mask):
            # All positive: vacuum is ground state
            return GaussianState.vacuum(self.n_modes)

        # Correlation matrix: Gamma = sum_{k: occupied} |phi_k><phi_k|
        occupied_vecs = evecs[:, occupied_mask]
        gamma = occupied_vecs @ occupied_vecs.conj().T

        return GaussianState(gamma)

    def ground_state_energy(self) -> float:
        """Ground state energy = sum of negative single-particle energies.

        Returns
        -------
        float
        """
        evals, _ = self.single_particle_spectrum()
        return float(np.sum(evals[evals <= 0]))

    def band_gap(self) -> float:
        """Energy gap between highest occupied and lowest unoccupied level.

        Returns
        -------
        float
            Band gap (zero if metallic).
        """
        evals, _ = self.single_particle_spectrum()
        negative = evals[evals <= 0]
        positive = evals[evals > 0]
        if len(negative) == 0 or len(positive) == 0:
            return 0.0
        return float(positive[0] - negative[-1])

    def fock_space_hamiltonian(self) -> np.ndarray:
        """Build the full 2^n Fock-space Hamiltonian (for validation).

        Returns
        -------
        np.ndarray
            Hermitian matrix of shape (2^n, 2^n).
        """
        n = self.n_modes
        dim = 2 ** n
        H = np.zeros((dim, dim), dtype=np.complex128)

        for i in range(n):
            for j in range(n):
                if abs(self.hopping_matrix[i, j]) > 1e-15:
                    ci = FermionicMode(i)
                    cj = FermionicMode(j)
                    H += self.hopping_matrix[i, j] * (
                        ci.creation_matrix(n) @ cj.annihilation_matrix(n)
                    )

        return H


# ---------------------------------------------------------------------------
# GaussianEvolutionResult
# ---------------------------------------------------------------------------


@dataclass
class GaussianEvolutionResult:
    """Result of Gaussian state time evolution.

    Attributes
    ----------
    times : np.ndarray
        Time points.
    states : list[GaussianState]
        Gaussian states at each time.
    """

    times: np.ndarray
    states: List[GaussianState]

    def particle_number_trajectory(self) -> np.ndarray:
        """Expected particle number at each time step.

        Returns
        -------
        np.ndarray
        """
        return np.array([s.particle_number for s in self.states])

    def entropy_trajectory(self) -> np.ndarray:
        """Von Neumann entropy at each time step.

        Returns
        -------
        np.ndarray
        """
        return np.array([s.entropy() for s in self.states])

    def site_occupations(self) -> np.ndarray:
        """Occupation number at each site over time.

        Returns
        -------
        np.ndarray
            Array of shape (n_times, n_modes).
        """
        return np.array([s.occupation_numbers() for s in self.states])

    def energy_trajectory(self, hamiltonian: "QuadraticHamiltonian") -> np.ndarray:
        """Energy expectation at each time step.

        Parameters
        ----------
        hamiltonian : QuadraticHamiltonian
            The Hamiltonian for energy evaluation.

        Returns
        -------
        np.ndarray
        """
        return np.array([hamiltonian.energy(s) for s in self.states])


# ---------------------------------------------------------------------------
# GaussianEvolution
# ---------------------------------------------------------------------------


@dataclass
class GaussianEvolution:
    """Time evolution of Gaussian states under quadratic Hamiltonians.

    The key insight: for a quadratic Hamiltonian H = sum h_ij c^dag_i c_j,
    the time evolution of the correlation matrix is:

        Gamma(t) = U(t) Gamma(0) U^dag(t)

    where U(t) = exp(-i h t) is the single-particle unitary.
    This is O(n^3) per step instead of O(2^n).

    Parameters
    ----------
    hamiltonian : QuadraticHamiltonian
        The quadratic Hamiltonian governing evolution.
    """

    hamiltonian: QuadraticHamiltonian

    def evolve(
        self,
        state: GaussianState,
        t_final: float,
        n_steps: int = 100,
    ) -> GaussianEvolutionResult:
        """Evolve Gaussian state from t=0 to t=t_final.

        Uses eigendecomposition of the hopping matrix for exact
        time evolution: U(t) = V diag(e^{-i eps_k t}) V^dag.

        Parameters
        ----------
        state : GaussianState
            Initial Gaussian state.
        t_final : float
            Final time.
        n_steps : int
            Number of time steps to record.

        Returns
        -------
        GaussianEvolutionResult
        """
        h = self.hamiltonian.hopping_matrix
        evals, evecs = np.linalg.eigh(h)

        times = np.linspace(0, t_final, n_steps + 1)
        states = []

        gamma_0 = state.correlation_matrix.copy()

        for t in times:
            # U(t) = V diag(e^{-i eps t}) V^dag
            phases = np.exp(-1j * evals * t)
            U_t = (evecs * phases) @ evecs.conj().T

            # Gamma(t) = U(t) Gamma(0) U^dag(t)
            gamma_t = U_t @ gamma_0 @ U_t.conj().T

            # Enforce Hermiticity (numerical cleanup)
            gamma_t = 0.5 * (gamma_t + gamma_t.conj().T)

            states.append(GaussianState(gamma_t))

        return GaussianEvolutionResult(times=times, states=states)

    def evolve_single_step(
        self,
        state: GaussianState,
        dt: float,
    ) -> GaussianState:
        """Evolve by a single time step dt.

        Parameters
        ----------
        state : GaussianState
            Current state.
        dt : float
            Time step.

        Returns
        -------
        GaussianState
            Evolved state.
        """
        h = self.hamiltonian.hopping_matrix
        evals, evecs = np.linalg.eigh(h)
        phases = np.exp(-1j * evals * dt)
        U = (evecs * phases) @ evecs.conj().T
        gamma_new = U @ state.correlation_matrix @ U.conj().T
        gamma_new = 0.5 * (gamma_new + gamma_new.conj().T)
        return GaussianState(gamma_new)


# ---------------------------------------------------------------------------
# Wick's theorem
# ---------------------------------------------------------------------------


def wicks_theorem(
    state: GaussianState,
    operators: list,
) -> complex:
    """Evaluate multi-point correlator using Wick's theorem.

    For a Gaussian state, any n-point function of creation and
    annihilation operators decomposes into a sum of products of
    2-point functions.  Specifically:

        <c^dag_{i1} c_{j1} c^dag_{i2} c_{j2} ... >
        = det(M)

    where M_ab = <c^dag_{ia} c_{jb}> = Gamma[ia, jb].

    Parameters
    ----------
    state : GaussianState
        The Gaussian state.
    operators : list[tuple]
        List of operator pairs. Each element is a tuple (i, j)
        representing a c^dag_i c_j pair.  The full correlator is
        <prod_k c^dag_{ik} c_{jk}>.

    Returns
    -------
    complex
        The expectation value.

    Notes
    -----
    For n pairs, this computes the determinant of an n x n matrix,
    which is the Pfaffian structure underlying Wick's theorem.
    """
    if len(operators) == 0:
        return 1.0 + 0j

    if len(operators) == 1:
        i, j = operators[0]
        return state.correlation_matrix[i, j]

    # Build the contraction matrix
    n_pairs = len(operators)
    M = np.zeros((n_pairs, n_pairs), dtype=np.complex128)
    for a, (i_a, _) in enumerate(operators):
        for b, (_, j_b) in enumerate(operators):
            M[a, b] = state.correlation_matrix[i_a, j_b]

    return np.linalg.det(M)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def density_of_states(
    hamiltonian: QuadraticHamiltonian,
    energies: Optional[np.ndarray] = None,
    broadening: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the density of states via Gaussian broadening.

    Parameters
    ----------
    hamiltonian : QuadraticHamiltonian
        The single-particle Hamiltonian.
    energies : np.ndarray or None
        Energy grid.  Default: auto-computed from spectrum.
    broadening : float
        Gaussian broadening width (sigma).

    Returns
    -------
    energies : np.ndarray
        Energy grid.
    dos : np.ndarray
        Density of states at each energy point.
    """
    evals, _ = hamiltonian.single_particle_spectrum()

    if energies is None:
        e_min = evals[0] - 3 * broadening
        e_max = evals[-1] + 3 * broadening
        energies = np.linspace(e_min, e_max, 500)

    dos = np.zeros_like(energies)
    for eps_k in evals:
        dos += np.exp(-0.5 * ((energies - eps_k) / broadening) ** 2)
    dos /= broadening * math.sqrt(2 * math.pi)

    return energies, dos


def correlation_length(state: GaussianState) -> float:
    """Estimate the correlation length from the correlation matrix.

    Fits the off-diagonal decay of |Gamma_{0,r}| to an exponential
    exp(-r/xi) and returns xi.

    Parameters
    ----------
    state : GaussianState
        The Gaussian state.

    Returns
    -------
    float
        Estimated correlation length (in units of the lattice spacing).
    """
    n = state.n_modes
    if n < 4:
        return float("inf")

    gamma = state.correlation_matrix
    # Look at correlations from site 0
    correlators = np.abs(gamma[0, :])

    # Find non-zero correlators at distance > 0
    distances = np.arange(1, n)
    values = correlators[1:]

    # Filter out near-zero values
    mask = values > 1e-15
    if np.sum(mask) < 2:
        return float("inf")

    distances = distances[mask]
    values = values[mask]

    # Fit log|C(r)| = -r/xi + const
    log_values = np.log(values)
    # Linear regression
    if len(distances) < 2:
        return float("inf")

    A = np.vstack([distances, np.ones(len(distances))]).T
    result = np.linalg.lstsq(A, log_values, rcond=None)
    slope = result[0][0]

    if slope >= -1e-10:
        return float("inf")

    return float(-1.0 / slope)
