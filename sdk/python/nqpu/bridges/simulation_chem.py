"""Simulation-chemistry bridge: open quantum chemistry via Lindblad dynamics.

Extends molecular simulations with environment coupling (decoherence, thermal
relaxation) using the Lindblad master equation from nqpu.simulation.

This enables analysis of how environmental noise affects quantum chemical
computations, which is critical for understanding:
  - VQE fidelity under realistic noise conditions
  - Natural molecular decoherence timescales
  - Temperature effects on molecular quantum states

References:
  - Breuer & Petruccione, *The Theory of Open Quantum Systems* (2002)
  - Kandala et al., Nature 549, 242 (2017) -- VQE with noise
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nqpu.simulation import (
    LindbladOperator,
    LindbladMasterEquation,
    LindbladSolver,
    dephasing_operators,
    thermal_operators,
    SparsePauliHamiltonian,
    PauliOperator,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """Matrix square root via eigendecomposition.

    Parameters
    ----------
    M : np.ndarray
        Positive semi-definite Hermitian matrix.

    Returns
    -------
    np.ndarray
        Matrix square root such that sqrt(M) @ sqrt(M) ~ M.
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T


# ---------------------------------------------------------------------------
# OpenQuantumChemistry
# ---------------------------------------------------------------------------


@dataclass
class OpenQuantumChemistry:
    """Open quantum system chemistry: molecular Hamiltonians with environment.

    Models how environmental decoherence affects quantum chemical computations,
    particularly relevant for understanding:
    - VQE fidelity under noise
    - Natural molecular decoherence timescales
    - Temperature effects on molecular quantum states

    The molecular Hamiltonian is specified as a list of (pauli_label, coefficient)
    tuples and coupled to dephasing and amplitude damping noise channels.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the qubit-mapped molecular Hamiltonian.
    hamiltonian_terms : list[tuple[str, float]]
        Pauli decomposition of the molecular Hamiltonian as
        (label, coefficient) pairs.
    temperature_K : float
        Environment temperature in Kelvin.
    dephasing_rate : float
        Per-qubit dephasing rate.
    relaxation_rate : float
        Per-qubit amplitude damping rate.
    """

    n_qubits: int
    hamiltonian_terms: list[tuple[str, float]]
    temperature_K: float = 300.0
    dephasing_rate: float = 0.01
    relaxation_rate: float = 0.001

    @classmethod
    def from_h2(cls, bond_length: float = 0.735, **kwargs) -> OpenQuantumChemistry:
        """Create H2 molecule Hamiltonian in minimal basis.

        Uses the standard 2-qubit Jordan-Wigner mapped H2 Hamiltonian
        in the STO-3G basis.  Coefficients are approximate and
        interpolated around the equilibrium bond length.

        Parameters
        ----------
        bond_length : float
            H-H bond length in Angstroms.
        **kwargs
            Passed through to the constructor (temperature_K, dephasing_rate, etc.).

        Returns
        -------
        OpenQuantumChemistry
        """
        # Standard H2 STO-3G coefficients (approximate for demonstration)
        g0 = -0.8105 + 0.1714 * (bond_length - 0.735)
        g1 = 0.1721
        g2 = -0.2257
        g3 = 0.1721
        g4 = 0.1689
        g5 = 0.1205
        terms = [
            ("II", g0),
            ("IZ", g1),
            ("ZI", g2),
            ("ZZ", g3),
            ("XX", g4),
            ("YY", g5),
        ]
        return cls(n_qubits=2, hamiltonian_terms=terms, **kwargs)

    def hamiltonian(self) -> np.ndarray:
        """Build the molecular Hamiltonian matrix.

        Returns
        -------
        np.ndarray
            Hermitian matrix of shape (2^n, 2^n).
        """
        n = self.n_qubits
        H = np.zeros((2**n, 2**n), dtype=complex)
        for label, coeff in self.hamiltonian_terms:
            op = PauliOperator(label, coeff=coeff)
            H += op.matrix()
        return H

    def decoherence_operators(self) -> list[LindbladOperator]:
        """Build decoherence operators for the molecular environment.

        Returns per-qubit dephasing (Z noise) and amplitude damping
        (energy relaxation) operators embedded in the full Hilbert space.

        Returns
        -------
        list[LindbladOperator]
        """
        ops: list[LindbladOperator] = []
        n = self.n_qubits
        dim = 2**n
        # Per-qubit dephasing via Z/2
        for i in range(n):
            label = "I" * i + "Z" + "I" * (n - i - 1)
            ops.append(
                LindbladOperator(
                    PauliOperator(label).matrix() / 2,
                    rate=self.dephasing_rate,
                    label=f"dephase_q{i}",
                )
            )
        # Amplitude damping (thermal relaxation) on each qubit
        if self.relaxation_rate > 0:
            for i in range(n):
                lowering = np.zeros((dim, dim), dtype=complex)
                for basis in range(dim):
                    if (basis >> (n - 1 - i)) & 1:
                        target = basis & ~(1 << (n - 1 - i))
                        lowering[target, basis] = 1.0
                ops.append(
                    LindbladOperator(
                        lowering * np.sqrt(self.relaxation_rate),
                        rate=1.0,
                        label=f"relax_q{i}",
                    )
                )
        return ops

    def evolve(
        self,
        t_final: float = 10.0,
        n_steps: int = 100,
        initial_state: np.ndarray | None = None,
    ) -> dict:
        """Evolve the molecular system with decoherence.

        Starts from the ground state (or a provided initial state) and
        evolves under the Lindblad master equation, tracking energy,
        purity, and fidelity against the noiseless ground state.

        Parameters
        ----------
        t_final : float
            Total evolution time.
        n_steps : int
            Number of integration steps.
        initial_state : np.ndarray or None
            Initial state vector or density matrix.  Defaults to the
            noiseless ground state.

        Returns
        -------
        dict
            Keys: times, energies, purities, fidelities, ground_state_energy,
            decoherence_time.
        """
        H = self.hamiltonian()
        dim = 2**self.n_qubits

        if initial_state is None:
            # Start from noiseless ground state
            eigvals, eigvecs = np.linalg.eigh(H)
            psi0 = eigvecs[:, 0]
            rho0 = np.outer(psi0, psi0.conj())
        elif initial_state.ndim == 1:
            rho0 = np.outer(initial_state, initial_state.conj())
        else:
            rho0 = initial_state.copy()

        ops = self.decoherence_operators()
        lme = LindbladMasterEquation(H, ops)
        solver = LindbladSolver(lme, method="rk4")
        result = solver.evolve(rho0, t_final=t_final, n_steps=n_steps)

        # Reference ground state for fidelity comparison
        eigvals, eigvecs = np.linalg.eigh(H)
        ground_rho = np.outer(eigvecs[:, 0], eigvecs[:, 0].conj())

        energies: list[float] = []
        purities: list[float] = []
        fidelities: list[float] = []
        for rho_t in result.states:
            energies.append(float(np.real(np.trace(H @ rho_t))))
            purities.append(float(np.real(np.trace(rho_t @ rho_t))))
            # Fidelity: Tr(sqrt(sqrt(ground) rho sqrt(ground)))^2
            # For a pure reference state, simplifies to <psi|rho|psi>
            fidelities.append(float(np.real(np.trace(ground_rho @ rho_t))))

        return {
            "times": result.times,
            "energies": np.array(energies),
            "purities": np.array(purities),
            "fidelities": np.array(fidelities),
            "ground_state_energy": float(eigvals[0]),
            "decoherence_time": self._estimate_decoherence(
                np.array(purities), result.times
            ),
        }

    @staticmethod
    def _estimate_decoherence(purities: np.ndarray, times: np.ndarray) -> float:
        """Estimate decoherence time from purity decay.

        Finds the time at which purity drops to 1/e of the way from its
        initial value to the maximally mixed value (0.5 for a single qubit).

        Parameters
        ----------
        purities : np.ndarray
            Purity values at each time step.
        times : np.ndarray
            Corresponding time values.

        Returns
        -------
        float
            Estimated decoherence time.
        """
        if len(purities) < 2 or purities[0] < 0.5:
            return 0.0
        threshold = 0.5 + (purities[0] - 0.5) / np.e
        for i in range(1, len(purities)):
            if purities[i] < threshold:
                return float(times[i])
        return float(times[-1])


# ---------------------------------------------------------------------------
# DecoherenceAnalysis
# ---------------------------------------------------------------------------


@dataclass
class DecoherenceAnalysis:
    """Analyse decoherence impact on molecular quantum computations.

    Provides scanning tools to systematically study how noise parameters
    affect ground-state energy accuracy and state purity.
    """

    @staticmethod
    def h2_decoherence_scan(
        dephasing_rates: np.ndarray | None = None,
    ) -> dict:
        """Scan decoherence impact on H2 ground state energy.

        Evolves the H2 molecule at multiple dephasing rates and measures
        the energy error and purity degradation at each rate.

        Parameters
        ----------
        dephasing_rates : np.ndarray or None
            Array of dephasing rates to scan.  Defaults to a log-spaced
            range from 1e-4 to 1e-1.

        Returns
        -------
        dict
            Keys: rates, results (list of per-rate dicts), ground_energy.
        """
        if dephasing_rates is None:
            dephasing_rates = np.logspace(-4, -1, 10)
        results: list[dict] = []
        for rate in dephasing_rates:
            mol = OpenQuantumChemistry.from_h2(
                dephasing_rate=float(rate),
                relaxation_rate=float(rate) / 10,
            )
            evo = mol.evolve(t_final=20.0, n_steps=50)
            results.append(
                {
                    "dephasing_rate": float(rate),
                    "final_energy": float(evo["energies"][-1]),
                    "final_purity": float(evo["purities"][-1]),
                    "ground_energy": evo["ground_state_energy"],
                    "energy_error": float(
                        abs(evo["energies"][-1] - evo["ground_state_energy"])
                    ),
                    "decoherence_time": evo["decoherence_time"],
                }
            )
        return {
            "rates": dephasing_rates,
            "results": results,
            "ground_energy": results[0]["ground_energy"],
        }
