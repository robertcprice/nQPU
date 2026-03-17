"""Noise-aware VQE benchmarking: how hardware noise degrades chemistry accuracy.

Runs VQE-style molecular ground-state computations across different hardware
noise profiles using the Lindblad master equation, producing energy accuracy
vs noise rate curves that show where each QPU architecture breaks down for
quantum chemistry.

This bridges nqpu.simulation (Lindblad solver), nqpu.bridges.simulation_chem
(open quantum chemistry), and nqpu.emulator (hardware profiles) to answer:
"How noisy can my QPU be before my VQE chemistry result becomes meaningless?"

Example:
    from nqpu.bridges import VQENoiseBenchmark

    bench = VQENoiseBenchmark()
    result = bench.h2_benchmark()
    for r in result["results"]:
        print(f"{r['profile']}: energy_error={r['energy_error']:.6f}, "
              f"purity={r['final_purity']:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nqpu.simulation import (
    LindbladOperator,
    LindbladMasterEquation,
    LindbladSolver,
    PauliOperator,
    SparsePauliHamiltonian,
)


def _h2_hamiltonian(bond_length: float = 0.735) -> np.ndarray:
    """Build H2 molecular Hamiltonian in minimal STO-3G basis (2 qubits).

    Parameters
    ----------
    bond_length : float
        H-H bond length in Angstroms.

    Returns
    -------
    np.ndarray
        4x4 Hermitian matrix.
    """
    g0 = -0.8105 + 0.1714 * (bond_length - 0.735)
    terms = [
        ("II", g0), ("IZ", 0.1721), ("ZI", -0.2257),
        ("ZZ", 0.1721), ("XX", 0.1689), ("YY", 0.1205),
    ]
    H = np.zeros((4, 4), dtype=complex)
    for label, coeff in terms:
        H += PauliOperator(label, coeff=coeff).matrix()
    return H


def _lih_hamiltonian() -> np.ndarray:
    """Simplified 2-qubit LiH Hamiltonian (Jordan-Wigner, minimal basis).

    Returns
    -------
    np.ndarray
        4x4 Hermitian matrix.
    """
    terms = [
        ("II", -7.4984), ("IZ", 0.2180), ("ZI", -0.3906),
        ("ZZ", 0.0112), ("XX", 0.0919), ("YY", 0.0919),
    ]
    H = np.zeros((4, 4), dtype=complex)
    for label, coeff in terms:
        H += PauliOperator(label, coeff=coeff).matrix()
    return H


def _decoherence_operators(
    n_qubits: int,
    dephasing_rate: float,
    relaxation_rate: float,
) -> list[LindbladOperator]:
    """Build per-qubit dephasing + amplitude damping operators.

    Parameters
    ----------
    n_qubits : int
    dephasing_rate : float
    relaxation_rate : float

    Returns
    -------
    list[LindbladOperator]
    """
    dim = 2**n_qubits
    ops: list[LindbladOperator] = []
    for i in range(n_qubits):
        label = "I" * i + "Z" + "I" * (n_qubits - i - 1)
        ops.append(LindbladOperator(
            PauliOperator(label).matrix() / 2,
            rate=dephasing_rate,
            label=f"dephase_q{i}",
        ))
        if relaxation_rate > 0:
            lowering = np.zeros((dim, dim), dtype=complex)
            for basis in range(dim):
                if (basis >> (n_qubits - 1 - i)) & 1:
                    target = basis & ~(1 << (n_qubits - 1 - i))
                    lowering[target, basis] = 1.0
            ops.append(LindbladOperator(
                lowering * np.sqrt(relaxation_rate),
                rate=1.0,
                label=f"relax_q{i}",
            ))
    return ops


@dataclass
class VQENoiseResult:
    """Result of a single VQE noise benchmark run.

    Attributes
    ----------
    profile_name : str
        Hardware profile name or noise label.
    dephasing_rate : float
        Dephasing rate used.
    relaxation_rate : float
        Relaxation rate used.
    ground_energy : float
        Exact ground-state energy (noiseless reference).
    final_energy : float
        Energy at end of noisy evolution.
    energy_error : float
        Absolute energy error vs noiseless ground state.
    final_purity : float
        Purity of the final state (1 = pure, 1/d = maximally mixed).
    final_fidelity : float
        Fidelity against the noiseless ground state.
    decoherence_time : float
        Estimated time to lose ground-state character.
    chemical_accuracy : bool
        Whether energy error < 1.6 mHa (chemical accuracy threshold).
    """

    profile_name: str
    dephasing_rate: float
    relaxation_rate: float
    ground_energy: float
    final_energy: float
    energy_error: float
    final_purity: float
    final_fidelity: float
    decoherence_time: float
    chemical_accuracy: bool


class VQENoiseBenchmark:
    """Noise-aware VQE benchmarking across hardware profiles.

    Evolves molecular ground states under Lindblad dynamics calibrated
    to real hardware noise parameters, measuring how quickly each QPU's
    noise degrades the VQE energy estimate.

    Parameters
    ----------
    t_final : float
        Evolution time for noise simulation.
    n_steps : int
        Integration steps.
    """

    CHEMICAL_ACCURACY_HA = 0.0016  # 1.6 mHa

    def __init__(self, t_final: float = 10.0, n_steps: int = 50) -> None:
        self.t_final = t_final
        self.n_steps = n_steps

    def benchmark_molecule(
        self,
        H: np.ndarray,
        molecule_name: str,
        noise_configs: list[dict] | None = None,
    ) -> dict:
        """Run VQE noise benchmark for a molecular Hamiltonian.

        Parameters
        ----------
        H : np.ndarray
            Molecular Hamiltonian matrix.
        molecule_name : str
            Human-readable molecule name.
        noise_configs : list[dict] or None
            List of noise configurations, each with keys:
            name, dephasing_rate, relaxation_rate. Defaults to 9
            hardware-calibrated profiles.

        Returns
        -------
        dict
            Keys: molecule, ground_energy, results, chemical_accuracy_threshold.
        """
        if noise_configs is None:
            noise_configs = self._hardware_noise_configs()

        n_qubits = int(np.log2(H.shape[0]))
        eigvals, eigvecs = np.linalg.eigh(H)
        ground_energy = float(eigvals[0])
        ground_state = eigvecs[:, 0]
        rho_ground = np.outer(ground_state, ground_state.conj())

        results: list[VQENoiseResult] = []
        for cfg in noise_configs:
            result = self._run_single(
                H, rho_ground, ground_energy, n_qubits, cfg
            )
            results.append(result)

        return {
            "molecule": molecule_name,
            "ground_energy": ground_energy,
            "results": results,
            "chemical_accuracy_threshold": self.CHEMICAL_ACCURACY_HA,
        }

    def h2_benchmark(self, bond_length: float = 0.735) -> dict:
        """Run H2 molecule benchmark across all hardware noise profiles.

        Parameters
        ----------
        bond_length : float
            H-H bond length in Angstroms.

        Returns
        -------
        dict
            Benchmark results (see benchmark_molecule).
        """
        return self.benchmark_molecule(
            _h2_hamiltonian(bond_length), f"H2 (d={bond_length} A)"
        )

    def lih_benchmark(self) -> dict:
        """Run simplified LiH benchmark across all hardware noise profiles.

        Returns
        -------
        dict
            Benchmark results.
        """
        return self.benchmark_molecule(_lih_hamiltonian(), "LiH (minimal)")

    def noise_sweep(
        self,
        H: np.ndarray,
        dephasing_rates: np.ndarray | None = None,
    ) -> dict:
        """Sweep dephasing rate and measure energy error.

        Parameters
        ----------
        H : np.ndarray
            Molecular Hamiltonian.
        dephasing_rates : np.ndarray or None
            Rates to scan. Defaults to logspace(-5, -1, 15).

        Returns
        -------
        dict
            Keys: rates, energy_errors, purities, fidelities,
            chemical_accuracy_limit.
        """
        if dephasing_rates is None:
            dephasing_rates = np.logspace(-5, -1, 15)

        n_qubits = int(np.log2(H.shape[0]))
        eigvals, eigvecs = np.linalg.eigh(H)
        ground_energy = float(eigvals[0])
        ground_state = eigvecs[:, 0]
        rho_ground = np.outer(ground_state, ground_state.conj())

        errors, purities, fidelities = [], [], []
        for rate in dephasing_rates:
            cfg = {"name": f"rate={rate:.2e}", "dephasing_rate": float(rate),
                   "relaxation_rate": float(rate) / 10}
            r = self._run_single(H, rho_ground, ground_energy, n_qubits, cfg)
            errors.append(r.energy_error)
            purities.append(r.final_purity)
            fidelities.append(r.final_fidelity)

        # Find chemical accuracy limit
        ca_limit = float(dephasing_rates[-1])
        for i, err in enumerate(errors):
            if err > self.CHEMICAL_ACCURACY_HA:
                ca_limit = float(dephasing_rates[max(0, i - 1)])
                break

        return {
            "rates": dephasing_rates,
            "energy_errors": np.array(errors),
            "purities": np.array(purities),
            "fidelities": np.array(fidelities),
            "chemical_accuracy_limit": ca_limit,
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_single(
        self,
        H: np.ndarray,
        rho_ground: np.ndarray,
        ground_energy: float,
        n_qubits: int,
        cfg: dict,
    ) -> VQENoiseResult:
        """Run a single noise simulation."""
        ops = _decoherence_operators(
            n_qubits, cfg["dephasing_rate"], cfg["relaxation_rate"]
        )
        lme = LindbladMasterEquation(H, ops)
        solver = LindbladSolver(lme, method="rk4")
        result = solver.evolve(rho_ground.copy(), t_final=self.t_final, n_steps=self.n_steps)

        rho_final = result.states[-1]
        final_energy = float(np.real(np.trace(H @ rho_final)))
        final_purity = float(np.real(np.trace(rho_final @ rho_final)))
        final_fidelity = float(np.real(np.trace(rho_ground @ rho_final)))
        energy_error = abs(final_energy - ground_energy)

        # Estimate decoherence time from fidelity decay
        decoherence_time = self._estimate_decoherence_time(
            result.states, rho_ground, result.times
        )

        return VQENoiseResult(
            profile_name=cfg["name"],
            dephasing_rate=cfg["dephasing_rate"],
            relaxation_rate=cfg["relaxation_rate"],
            ground_energy=ground_energy,
            final_energy=final_energy,
            energy_error=energy_error,
            final_purity=final_purity,
            final_fidelity=final_fidelity,
            decoherence_time=decoherence_time,
            chemical_accuracy=energy_error < self.CHEMICAL_ACCURACY_HA,
        )

    @staticmethod
    def _estimate_decoherence_time(
        states: list,
        rho_ground: np.ndarray,
        times: np.ndarray,
    ) -> float:
        """Find time at which fidelity drops to 1/e of initial."""
        threshold = 1.0 / np.e
        for i, rho_t in enumerate(states):
            fid = float(np.real(np.trace(rho_ground @ rho_t)))
            if fid < threshold:
                return float(times[i])
        return float(times[-1])

    @staticmethod
    def _hardware_noise_configs() -> list[dict]:
        """Generate noise configs calibrated to real hardware profiles."""
        from nqpu.emulator import HardwareProfile

        configs = []
        for profile in HardwareProfile:
            spec = profile.spec
            # Convert gate fidelities to effective dephasing/relaxation rates
            # Approximate: error_rate ~ gamma * t_gate
            t_gate = spec.two_qubit_gate_us
            error_2q = 1.0 - spec.two_qubit_fidelity
            dephasing = error_2q / max(t_gate, 0.001) * 0.1  # Scale factor
            relaxation = dephasing / 10
            configs.append({
                "name": spec.name,
                "dephasing_rate": dephasing,
                "relaxation_rate": relaxation,
            })
        return configs
