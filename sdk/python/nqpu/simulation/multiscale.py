"""Multiscale quantum simulation: hybrid exact + mean-field methods.

Enables simulation of larger quantum systems by decomposing them into
subsystems that are solved individually and coupled via self-consistent
mean-field interactions.  This is analogous to ONIOM, QM/MM, and DMET
approaches in quantum chemistry.

Features:
  - System decomposition into subsystems with independent Hamiltonians.
  - Inter-subsystem coupling via operator tensor products.
  - Self-consistent field (SCF) solver for ground states.
  - Split-operator time evolution for dynamics.
  - Adaptive method selection based on entanglement estimates.
  - Full Hamiltonian construction for validation against exact results.

References:
    - Knizia & Chan, Phys. Rev. Lett. 109, 186404 (2012) [DMET]
    - Svensson et al., J. Phys. Chem. 100, 19357 (1996) [ONIOM]
    - Suzuki, J. Math. Phys. 32, 400 (1991) [Split-operator]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helper: matrix exponential via eigendecomposition (Hermitian)
# ---------------------------------------------------------------------------


def _matrix_exp_hermitian(H: np.ndarray, t: float) -> np.ndarray:
    """Compute e^{-iHt} for Hermitian H."""
    evals, evecs = np.linalg.eigh(H)
    phases = np.exp(-1j * evals * t)
    return (evecs * phases) @ evecs.conj().T


# ---------------------------------------------------------------------------
# Subsystem
# ---------------------------------------------------------------------------


@dataclass
class Subsystem:
    """A subsystem in the multiscale simulation.

    Parameters
    ----------
    name : str
        Unique identifier for this subsystem.
    n_qubits : int
        Number of qubits in this subsystem.
    hamiltonian : np.ndarray
        Local Hamiltonian matrix of shape (2^n, 2^n).
    method : str
        Simulation method: ``"exact"`` for full diagonalisation or
        ``"approximate"`` for simplified treatment.
    state : np.ndarray or None
        Current state vector.  Initialised during solving.
    """

    name: str
    n_qubits: int
    hamiltonian: np.ndarray
    method: str = "exact"
    state: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.hamiltonian = np.asarray(self.hamiltonian, dtype=np.complex128)
        d = 2 ** self.n_qubits
        if self.hamiltonian.shape != (d, d):
            raise ValueError(
                f"Hamiltonian shape {self.hamiltonian.shape} does not match "
                f"n_qubits={self.n_qubits} (expected ({d}, {d}))."
            )
        if self.state is None:
            self.state = np.zeros(d, dtype=np.complex128)
            self.state[0] = 1.0

    @property
    def dim(self) -> int:
        """Hilbert space dimension."""
        return 2 ** self.n_qubits

    def energy(self) -> float:
        """Compute <psi|H_local|psi> for current state."""
        if self.state is None:
            return 0.0
        psi = np.asarray(self.state, dtype=np.complex128).ravel()
        return float(np.real(psi.conj() @ self.hamiltonian @ psi))

    def ground_state(self) -> Tuple[float, np.ndarray]:
        """Compute ground state of local Hamiltonian."""
        evals, evecs = np.linalg.eigh(self.hamiltonian)
        return float(evals[0]), evecs[:, 0]


# ---------------------------------------------------------------------------
# CouplingTerm
# ---------------------------------------------------------------------------


@dataclass
class CouplingTerm:
    """Coupling between two subsystems.

    Represents an interaction of the form:
        strength * (operator_a tensor operator_b)

    where operator_a acts on subsystem_a and operator_b on subsystem_b.

    Parameters
    ----------
    subsystem_a : str
        Name of the first subsystem.
    subsystem_b : str
        Name of the second subsystem.
    operator_a : np.ndarray
        Operator acting on subsystem A.
    operator_b : np.ndarray
        Operator acting on subsystem B.
    strength : float
        Coupling strength.
    """

    subsystem_a: str
    subsystem_b: str
    operator_a: np.ndarray
    operator_b: np.ndarray
    strength: float = 1.0

    def __post_init__(self) -> None:
        self.operator_a = np.asarray(self.operator_a, dtype=np.complex128)
        self.operator_b = np.asarray(self.operator_b, dtype=np.complex128)


# ---------------------------------------------------------------------------
# MultiscaleSystem
# ---------------------------------------------------------------------------


@dataclass
class MultiscaleSystem:
    """System decomposed into subsystems with inter-subsystem couplings.

    Enables simulation of large systems by treating different regions
    with different levels of approximation and coupling them via
    mean-field interactions.

    Parameters
    ----------
    subsystems : dict
        Mapping from name to Subsystem.
    couplings : list[CouplingTerm]
        Inter-subsystem coupling terms.
    """

    subsystems: Dict[str, Subsystem] = field(default_factory=dict)
    couplings: List[CouplingTerm] = field(default_factory=list)

    def add_subsystem(self, subsystem: Subsystem) -> None:
        """Add a subsystem to the multiscale system.

        Parameters
        ----------
        subsystem : Subsystem
            The subsystem to add.
        """
        if subsystem.name in self.subsystems:
            raise ValueError(
                f"Subsystem '{subsystem.name}' already exists."
            )
        self.subsystems[subsystem.name] = subsystem

    def add_coupling(self, coupling: CouplingTerm) -> None:
        """Add a coupling between two subsystems.

        Parameters
        ----------
        coupling : CouplingTerm
            The coupling to add.
        """
        if coupling.subsystem_a not in self.subsystems:
            raise ValueError(
                f"Subsystem '{coupling.subsystem_a}' not found."
            )
        if coupling.subsystem_b not in self.subsystems:
            raise ValueError(
                f"Subsystem '{coupling.subsystem_b}' not found."
            )
        self.couplings.append(coupling)

    @property
    def total_qubits(self) -> int:
        """Total number of qubits across all subsystems."""
        return sum(s.n_qubits for s in self.subsystems.values())

    @property
    def total_dim(self) -> int:
        """Total Hilbert space dimension."""
        return 2 ** self.total_qubits

    @property
    def subsystem_names(self) -> List[str]:
        """Ordered list of subsystem names."""
        return list(self.subsystems.keys())

    def _subsystem_offset(self, name: str) -> int:
        """Compute qubit offset for a named subsystem."""
        offset = 0
        for n in self.subsystem_names:
            if n == name:
                return offset
            offset += self.subsystems[n].n_qubits
        raise ValueError(f"Subsystem '{name}' not found.")

    def _embed_operator(
        self,
        op: np.ndarray,
        subsystem_name: str,
    ) -> np.ndarray:
        """Embed a subsystem operator into the full Hilbert space.

        Parameters
        ----------
        op : np.ndarray
            Operator acting on a single subsystem.
        subsystem_name : str
            Name of the subsystem.

        Returns
        -------
        np.ndarray
            Full-space operator.
        """
        result = None
        for name in self.subsystem_names:
            sub = self.subsystems[name]
            if name == subsystem_name:
                part = op
            else:
                part = np.eye(sub.dim, dtype=np.complex128)
            if result is None:
                result = part
            else:
                result = np.kron(result, part)
        return result

    def full_hamiltonian(self) -> np.ndarray:
        """Build the full Hamiltonian including all subsystems and couplings.

        This is primarily for validation against exact diagonalisation
        of small systems.

        Returns
        -------
        np.ndarray
            Hermitian matrix of shape (total_dim, total_dim).
        """
        total_dim = self.total_dim
        H = np.zeros((total_dim, total_dim), dtype=np.complex128)

        # Local Hamiltonians
        for name, sub in self.subsystems.items():
            H += self._embed_operator(sub.hamiltonian, name)

        # Coupling terms
        for coupling in self.couplings:
            op_a = self._embed_operator(
                coupling.operator_a, coupling.subsystem_a
            )
            op_b = self._embed_operator(
                coupling.operator_b, coupling.subsystem_b
            )
            H += coupling.strength * (op_a @ op_b)

        return H

    def ground_state_energy(self, method: str = "exact") -> float:
        """Compute ground state energy.

        Parameters
        ----------
        method : str
            ``"exact"`` for full diagonalisation, ``"scf"`` for
            self-consistent field.

        Returns
        -------
        float
            Ground state energy.
        """
        if method == "exact":
            H = self.full_hamiltonian()
            evals = np.linalg.eigvalsh(H)
            return float(evals[0])
        elif method == "scf":
            solver = MultiscaleSolver(self)
            result = solver.solve()
            return result.energy
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'exact' or 'scf'."
            )


# ---------------------------------------------------------------------------
# MultiscaleResult
# ---------------------------------------------------------------------------


@dataclass
class MultiscaleResult:
    """Result of a multiscale calculation.

    Attributes
    ----------
    energy : float
        Total energy of the system.
    subsystem_energies : dict
        Energy of each subsystem.
    converged : bool
        Whether the SCF iteration converged.
    iterations : int
        Number of SCF iterations performed.
    energy_history : list[float]
        Total energy at each iteration.
    subsystem_states : dict
        Final state vectors for each subsystem.
    """

    energy: float
    subsystem_energies: Dict[str, float]
    converged: bool
    iterations: int
    energy_history: List[float]
    subsystem_states: Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# MultiscaleSolver
# ---------------------------------------------------------------------------


@dataclass
class MultiscaleSolver:
    """Solve multiscale system using self-consistent field approach.

    Iteratively solves each subsystem in the mean field generated by
    all other subsystems until energy convergence.

    Parameters
    ----------
    system : MultiscaleSystem
        The multiscale system to solve.
    max_iterations : int
        Maximum number of SCF iterations.
    convergence_threshold : float
        Energy convergence threshold.
    mixing : float
        Damping parameter for SCF mixing (0 < mixing <= 1).
        Smaller values improve stability at the cost of slower convergence.
    """

    system: MultiscaleSystem
    max_iterations: int = 50
    convergence_threshold: float = 1e-6
    mixing: float = 0.5

    def solve(self) -> MultiscaleResult:
        """Self-consistent solution of coupled subsystems.

        Algorithm:
        1. Initialise each subsystem in its local ground state.
        2. For each subsystem, compute the mean-field contribution
           from couplings to other subsystems.
        3. Solve each subsystem with the effective Hamiltonian:
           H_eff_A = H_A + sum_k V_k * <psi_B|O_B|psi_B>
        4. Repeat until total energy converges.

        Returns
        -------
        MultiscaleResult
        """
        system = self.system

        # Step 1: Initialise subsystem states
        for name, sub in system.subsystems.items():
            _, gs = sub.ground_state()
            sub.state = gs

        energy_history = []
        converged = False

        for iteration in range(self.max_iterations):
            # Step 2-3: Update each subsystem
            old_states = {
                name: sub.state.copy()
                for name, sub in system.subsystems.items()
            }

            for name in system.subsystem_names:
                sub = system.subsystems[name]
                effective_field = self._compute_effective_field(name)
                H_eff = sub.hamiltonian + effective_field

                # Solve for ground state of effective Hamiltonian
                evals, evecs = np.linalg.eigh(H_eff)
                new_state = evecs[:, 0]

                # Apply mixing for stability
                if self.mixing < 1.0 and sub.state is not None:
                    # Mix old and new states (as density matrices for stability)
                    rho_old = np.outer(sub.state, sub.state.conj())
                    rho_new = np.outer(new_state, new_state.conj())
                    rho_mixed = self.mixing * rho_new + (1 - self.mixing) * rho_old

                    # Extract dominant eigenvector
                    evals_mix, evecs_mix = np.linalg.eigh(rho_mixed)
                    sub.state = evecs_mix[:, -1]  # largest eigenvalue
                    sub.state = sub.state / np.linalg.norm(sub.state)
                else:
                    sub.state = new_state

            # Compute total energy
            total_energy = self._compute_total_energy()
            energy_history.append(total_energy)

            # Check convergence
            if iteration > 0:
                delta_e = abs(energy_history[-1] - energy_history[-2])
                if delta_e < self.convergence_threshold:
                    converged = True
                    break

        # Collect results
        subsystem_energies = {}
        subsystem_states = {}
        for name, sub in system.subsystems.items():
            subsystem_energies[name] = sub.energy()
            subsystem_states[name] = sub.state.copy()

        return MultiscaleResult(
            energy=energy_history[-1] if energy_history else 0.0,
            subsystem_energies=subsystem_energies,
            converged=converged,
            iterations=len(energy_history),
            energy_history=energy_history,
            subsystem_states=subsystem_states,
        )

    def _compute_effective_field(self, target_name: str) -> np.ndarray:
        """Compute mean-field coupling to target subsystem.

        For each coupling term between target and another subsystem,
        compute <psi_other|O_other|psi_other> * strength * O_target.

        Parameters
        ----------
        target_name : str
            Name of the target subsystem.

        Returns
        -------
        np.ndarray
            Effective field Hamiltonian on the target subsystem.
        """
        system = self.system
        target = system.subsystems[target_name]
        d = target.dim
        effective = np.zeros((d, d), dtype=np.complex128)

        for coupling in system.couplings:
            if coupling.subsystem_a == target_name:
                other_name = coupling.subsystem_b
                op_target = coupling.operator_a
                op_other = coupling.operator_b
            elif coupling.subsystem_b == target_name:
                other_name = coupling.subsystem_a
                op_target = coupling.operator_b
                op_other = coupling.operator_a
            else:
                continue

            other = system.subsystems[other_name]
            if other.state is None:
                continue

            # Mean-field: <psi_other|O_other|psi_other>
            psi_other = np.asarray(other.state, dtype=np.complex128).ravel()
            expectation = float(np.real(
                psi_other.conj() @ op_other @ psi_other
            ))

            effective += coupling.strength * expectation * op_target

        return effective

    def _compute_total_energy(self) -> float:
        """Compute total energy including local and coupling contributions.

        Returns
        -------
        float
            Total energy.
        """
        system = self.system
        total = 0.0

        # Local energies
        for name, sub in system.subsystems.items():
            total += sub.energy()

        # Coupling energies
        for coupling in system.couplings:
            sub_a = system.subsystems[coupling.subsystem_a]
            sub_b = system.subsystems[coupling.subsystem_b]

            if sub_a.state is None or sub_b.state is None:
                continue

            psi_a = np.asarray(sub_a.state, dtype=np.complex128).ravel()
            psi_b = np.asarray(sub_b.state, dtype=np.complex128).ravel()

            exp_a = float(np.real(
                psi_a.conj() @ coupling.operator_a @ psi_a
            ))
            exp_b = float(np.real(
                psi_b.conj() @ coupling.operator_b @ psi_b
            ))

            total += coupling.strength * exp_a * exp_b

        return total

    def _solve_subsystem(
        self,
        sub: Subsystem,
        effective_field: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Solve single subsystem with effective field.

        Parameters
        ----------
        sub : Subsystem
            The subsystem to solve.
        effective_field : np.ndarray
            Mean-field correction to the local Hamiltonian.

        Returns
        -------
        energy : float
            Ground state energy.
        state : np.ndarray
            Ground state vector.
        """
        H_eff = sub.hamiltonian + effective_field
        evals, evecs = np.linalg.eigh(H_eff)
        return float(evals[0]), evecs[:, 0]


# ---------------------------------------------------------------------------
# MultiscaleEvolutionResult
# ---------------------------------------------------------------------------


@dataclass
class MultiscaleEvolutionResult:
    """Result of multiscale time evolution.

    Attributes
    ----------
    times : np.ndarray
        Time points.
    energies : np.ndarray
        Total energy at each time step.
    subsystem_states : dict
        Name -> list of state vectors over time.
    """

    times: np.ndarray
    energies: np.ndarray
    subsystem_states: Dict[str, List[np.ndarray]]


# ---------------------------------------------------------------------------
# MultiscaleEvolution
# ---------------------------------------------------------------------------


@dataclass
class MultiscaleEvolution:
    """Time evolution of multiscale system using split-operator method.

    Alternates between evolving individual subsystems under their local
    Hamiltonians and applying coupling interactions as mean-field kicks.

    This is a first-order split:
        U(dt) ~ prod_k U_coupling_k(dt) * prod_s U_local_s(dt)

    Parameters
    ----------
    system : MultiscaleSystem
        The multiscale system to evolve.
    """

    system: MultiscaleSystem

    def evolve(
        self,
        t_final: float,
        n_steps: int = 100,
    ) -> MultiscaleEvolutionResult:
        """Split-operator evolution.

        Parameters
        ----------
        t_final : float
            Final time.
        n_steps : int
            Number of time steps.

        Returns
        -------
        MultiscaleEvolutionResult
        """
        dt = t_final / n_steps
        times = np.linspace(0, t_final, n_steps + 1)

        # Initialise subsystem state tracking
        subsystem_states = {
            name: [sub.state.copy()]
            for name, sub in self.system.subsystems.items()
        }
        energies = [self._compute_total_energy()]

        for step in range(n_steps):
            # Step 1: Evolve local Hamiltonians
            for name, sub in self.system.subsystems.items():
                U_local = _matrix_exp_hermitian(sub.hamiltonian, dt)
                sub.state = U_local @ sub.state
                sub.state = sub.state / np.linalg.norm(sub.state)

            # Step 2: Apply coupling kicks (mean-field approximation)
            self._apply_coupling_kicks(dt)

            # Record
            for name, sub in self.system.subsystems.items():
                subsystem_states[name].append(sub.state.copy())
            energies.append(self._compute_total_energy())

        return MultiscaleEvolutionResult(
            times=times,
            energies=np.array(energies),
            subsystem_states=subsystem_states,
        )

    def _apply_coupling_kicks(self, dt: float) -> None:
        """Apply mean-field coupling corrections.

        For each coupling, compute the effective Hamiltonian on each
        subsystem and apply a unitary kick.
        """
        for coupling in self.system.couplings:
            sub_a = self.system.subsystems[coupling.subsystem_a]
            sub_b = self.system.subsystems[coupling.subsystem_b]

            if sub_a.state is None or sub_b.state is None:
                continue

            psi_a = np.asarray(sub_a.state, dtype=np.complex128).ravel()
            psi_b = np.asarray(sub_b.state, dtype=np.complex128).ravel()

            # Mean-field on A from B
            exp_b = float(np.real(
                psi_b.conj() @ coupling.operator_b @ psi_b
            ))
            H_eff_a = coupling.strength * exp_b * coupling.operator_a
            U_a = _matrix_exp_hermitian(H_eff_a, dt)
            sub_a.state = U_a @ sub_a.state
            sub_a.state = sub_a.state / np.linalg.norm(sub_a.state)

            # Mean-field on B from A
            # Use updated psi_a
            psi_a = np.asarray(sub_a.state, dtype=np.complex128).ravel()
            exp_a = float(np.real(
                psi_a.conj() @ coupling.operator_a @ psi_a
            ))
            H_eff_b = coupling.strength * exp_a * coupling.operator_b
            U_b = _matrix_exp_hermitian(H_eff_b, dt)
            sub_b.state = U_b @ sub_b.state
            sub_b.state = sub_b.state / np.linalg.norm(sub_b.state)

    def _compute_total_energy(self) -> float:
        """Compute total energy."""
        total = 0.0
        for name, sub in self.system.subsystems.items():
            total += sub.energy()

        for coupling in self.system.couplings:
            sub_a = self.system.subsystems[coupling.subsystem_a]
            sub_b = self.system.subsystems[coupling.subsystem_b]

            if sub_a.state is None or sub_b.state is None:
                continue

            psi_a = np.asarray(sub_a.state, dtype=np.complex128).ravel()
            psi_b = np.asarray(sub_b.state, dtype=np.complex128).ravel()

            exp_a = float(np.real(
                psi_a.conj() @ coupling.operator_a @ psi_a
            ))
            exp_b = float(np.real(
                psi_b.conj() @ coupling.operator_b @ psi_b
            ))

            total += coupling.strength * exp_a * exp_b

        return total


# ---------------------------------------------------------------------------
# AdaptiveMultiscale
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveMultiscale:
    """Adaptive method selection based on entanglement.

    Analyses a monolithic Hamiltonian and suggests how to decompose it
    into subsystems for multiscale simulation, based on the entanglement
    structure of approximate ground states.

    Parameters
    ----------
    system : MultiscaleSystem
        The system to analyse (used for accessing subsystem info).
    entanglement_threshold : float
        Entanglement entropy threshold for deciding subsystem boundaries.
    """

    system: MultiscaleSystem
    entanglement_threshold: float = 0.5

    def analyze_entanglement(self) -> Dict[str, float]:
        """Compute inter-subsystem entanglement.

        For each pair of subsystems, estimates the entanglement entropy
        between them using the coupling strength as a proxy.

        Returns
        -------
        dict
            Mapping from "subsystem_a-subsystem_b" to entanglement estimate.
        """
        entanglements = {}

        for coupling in self.system.couplings:
            key = f"{coupling.subsystem_a}-{coupling.subsystem_b}"

            sub_a = self.system.subsystems[coupling.subsystem_a]
            sub_b = self.system.subsystems[coupling.subsystem_b]

            if sub_a.state is None or sub_b.state is None:
                entanglements[key] = 0.0
                continue

            # Estimate entanglement via coupling strength and correlations
            psi_a = np.asarray(sub_a.state, dtype=np.complex128).ravel()
            psi_b = np.asarray(sub_b.state, dtype=np.complex128).ravel()

            # Operator variance as entanglement proxy
            exp_a = float(np.real(
                psi_a.conj() @ coupling.operator_a @ psi_a
            ))
            exp_a2 = float(np.real(
                psi_a.conj()
                @ (coupling.operator_a @ coupling.operator_a)
                @ psi_a
            ))
            var_a = max(0.0, exp_a2 - exp_a ** 2)

            exp_b = float(np.real(
                psi_b.conj() @ coupling.operator_b @ psi_b
            ))
            exp_b2 = float(np.real(
                psi_b.conj()
                @ (coupling.operator_b @ coupling.operator_b)
                @ psi_b
            ))
            var_b = max(0.0, exp_b2 - exp_b ** 2)

            # Entanglement estimate (product of variances * coupling)
            entanglements[key] = (
                abs(coupling.strength) * math.sqrt(var_a * var_b)
            )

        return entanglements

    def suggest_decomposition(
        self,
        n_qubits: int,
        hamiltonian: np.ndarray,
        max_subsystem_size: int = 6,
    ) -> MultiscaleSystem:
        """Auto-decompose a system into subsystems.

        Uses a simple contiguous-block decomposition that splits the
        qubit register into chunks of at most max_subsystem_size qubits.

        Parameters
        ----------
        n_qubits : int
            Total number of qubits.
        hamiltonian : np.ndarray
            Full Hamiltonian matrix of shape (2^n, 2^n).
        max_subsystem_size : int
            Maximum number of qubits per subsystem.

        Returns
        -------
        MultiscaleSystem
            Decomposed system.
        """
        hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)

        # Determine partition
        n_subsystems = max(1, math.ceil(n_qubits / max_subsystem_size))
        sizes = self._partition_sizes(n_qubits, n_subsystems, max_subsystem_size)

        system = MultiscaleSystem()

        # Create subsystems with local Hamiltonians
        offset = 0
        subsystem_info = []
        for idx, size in enumerate(sizes):
            name = f"sub_{idx}"
            d_sub = 2 ** size
            d_total = 2 ** n_qubits

            # Extract local Hamiltonian by partial trace over other subsystems
            H_local = self._extract_local_hamiltonian(
                hamiltonian, n_qubits, offset, size
            )

            sub = Subsystem(
                name=name,
                n_qubits=size,
                hamiltonian=H_local,
                method="exact",
            )
            system.add_subsystem(sub)
            subsystem_info.append((name, offset, size))
            offset += size

        # Create coupling terms between adjacent subsystems
        for i in range(len(subsystem_info) - 1):
            name_a, offset_a, size_a = subsystem_info[i]
            name_b, offset_b, size_b = subsystem_info[i + 1]

            # Use Z-Z coupling at the boundary
            d_a = 2 ** size_a
            d_b = 2 ** size_b

            # Z operator on the last qubit of subsystem A
            Z_a = self._z_on_qubit(size_a, size_a - 1)
            # Z operator on the first qubit of subsystem B
            Z_b = self._z_on_qubit(size_b, 0)

            # Estimate coupling strength from the Hamiltonian
            strength = self._estimate_boundary_coupling(
                hamiltonian, n_qubits,
                offset_a + size_a - 1,  # last qubit of A
                offset_b,               # first qubit of B
            )

            coupling = CouplingTerm(
                subsystem_a=name_a,
                subsystem_b=name_b,
                operator_a=Z_a,
                operator_b=Z_b,
                strength=strength,
            )
            system.add_coupling(coupling)

        return system

    @staticmethod
    def _partition_sizes(
        n_qubits: int,
        n_subsystems: int,
        max_size: int,
    ) -> List[int]:
        """Compute subsystem sizes for an even partition."""
        base = n_qubits // n_subsystems
        remainder = n_qubits % n_subsystems
        sizes = []
        for i in range(n_subsystems):
            s = base + (1 if i < remainder else 0)
            s = min(s, max_size)
            sizes.append(s)
        # Ensure we cover all qubits
        total = sum(sizes)
        if total < n_qubits:
            sizes[-1] += n_qubits - total
        return sizes

    @staticmethod
    def _z_on_qubit(n_qubits: int, qubit: int) -> np.ndarray:
        """Build Z operator on a specific qubit."""
        I = np.eye(2, dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        result = np.eye(1, dtype=np.complex128)
        for q in range(n_qubits):
            result = np.kron(result, Z if q == qubit else I)
        return result

    @staticmethod
    def _extract_local_hamiltonian(
        H_full: np.ndarray,
        n_qubits: int,
        offset: int,
        size: int,
    ) -> np.ndarray:
        """Extract local Hamiltonian by partial trace.

        Approximates the local Hamiltonian by tracing out all other
        subsystems from the full Hamiltonian, assuming a product state
        (|0>^{other}) for the environment.
        """
        d_total = 2 ** n_qubits
        d_sub = 2 ** size
        d_before = 2 ** offset
        d_after = 2 ** (n_qubits - offset - size)

        # For computational basis |0> in before and after
        # H_local[i,j] = <0_before, i, 0_after | H | 0_before, j, 0_after>
        H_local = np.zeros((d_sub, d_sub), dtype=np.complex128)

        for i in range(d_sub):
            for j in range(d_sub):
                # Full basis index: before=0, subsystem=i/j, after=0
                row = 0 * (d_sub * d_after) + i * d_after + 0
                col = 0 * (d_sub * d_after) + j * d_after + 0
                H_local[i, j] = H_full[row, col]

        return H_local

    @staticmethod
    def _estimate_boundary_coupling(
        H_full: np.ndarray,
        n_qubits: int,
        qubit_a: int,
        qubit_b: int,
    ) -> float:
        """Estimate coupling strength between two boundary qubits.

        Computes the ZZ interaction coefficient by looking at the
        appropriate matrix elements.
        """
        d = 2 ** n_qubits
        I = np.eye(2, dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        # Build ZZ operator on qubits a and b
        ZZ = np.eye(1, dtype=np.complex128)
        for q in range(n_qubits):
            if q == qubit_a or q == qubit_b:
                ZZ = np.kron(ZZ, Z)
            else:
                ZZ = np.kron(ZZ, I)

        # ZZ coefficient in the Hamiltonian = Tr(ZZ * H) / d
        coeff = np.real(np.trace(ZZ @ H_full)) / d
        return float(coeff)
