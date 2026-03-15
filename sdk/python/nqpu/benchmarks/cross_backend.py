"""Cross-backend benchmarking: trapped-ion vs superconducting.

Runs identical quantum circuits on both hardware backends, comparing:
- Ideal fidelity (should be identical)
- Noisy fidelity (hardware-dependent error rates)
- Native gate counts (compilation efficiency)
- Estimated circuit duration
- QCVV metrics (RB error rates, Bell/GHZ fidelity)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nqpu.ion_trap import TrappedIonSimulator, TrapConfig, IonSpecies
from nqpu.superconducting import (
    TransmonSimulator,
    ChipConfig,
    DevicePresets,
    TransmonQCVV,
)


@dataclass
class CircuitBenchmark:
    """Results from running a single circuit on one backend."""
    backend_name: str
    execution_mode: str
    probabilities: np.ndarray
    fidelity_vs_ideal: float
    num_gates_1q: int
    num_gates_2q: int
    wall_time_ms: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendComparison:
    """Comparison of a circuit across multiple backends."""
    circuit_name: str
    num_qubits: int
    results: dict[str, CircuitBenchmark]

    def summary(self) -> dict[str, Any]:
        """Generate a summary table."""
        rows = {}
        for name, res in self.results.items():
            rows[name] = {
                "fidelity": f"{res.fidelity_vs_ideal:.4f}",
                "1Q_gates": res.num_gates_1q,
                "2Q_gates": res.num_gates_2q,
                "wall_ms": f"{res.wall_time_ms:.2f}",
            }
        return {
            "circuit": self.circuit_name,
            "num_qubits": self.num_qubits,
            "backends": rows,
        }


class CrossBackendBenchmark:
    """Cross-backend benchmarking suite.

    Runs identical circuits on trapped-ion and superconducting backends
    in both ideal and noisy modes.

    Parameters
    ----------
    num_qubits : int
        Number of qubits for the benchmark circuits.
    ion_species : IonSpecies
        Ion species for the trapped-ion backend.
    sc_preset : DevicePresets
        Device preset for the superconducting backend.
    """

    def __init__(
        self,
        num_qubits: int = 3,
        ion_species: IonSpecies | None = None,
        sc_preset: DevicePresets = DevicePresets.IBM_HERON,
    ) -> None:
        self.num_qubits = num_qubits

        # Trapped-ion config
        if ion_species is None:
            ion_species = IonSpecies.YB171
        self.ion_config = TrapConfig(n_ions=num_qubits, species=ion_species)

        # Superconducting config
        self.sc_config = sc_preset.build(num_qubits=num_qubits)

    # ------------------------------------------------------------------
    # Circuit definitions
    # ------------------------------------------------------------------

    @staticmethod
    def _bell_circuit_ion(sim: TrappedIonSimulator) -> None:
        """Bell state on trapped-ion."""
        sim.h(0)
        sim.cnot(0, 1)

    @staticmethod
    def _bell_circuit_sc(sim: TransmonSimulator) -> None:
        """Bell state on superconducting."""
        sim.h(0)
        sim.cnot(0, 1)

    @staticmethod
    def _ghz_circuit_ion(sim: TrappedIonSimulator, n: int) -> None:
        """GHZ state on trapped-ion."""
        sim.h(0)
        for i in range(1, n):
            sim.cnot(0, i)

    @staticmethod
    def _ghz_circuit_sc(sim: TransmonSimulator, n: int) -> None:
        """GHZ state on superconducting."""
        sim.h(0)
        for i in range(1, n):
            sim.cnot(0, i)

    @staticmethod
    def _qft_circuit_ion(sim: TrappedIonSimulator, n: int) -> None:
        """QFT on trapped-ion."""
        for i in range(n):
            sim.h(i)
            for j in range(i + 1, n):
                angle = math.pi / (2 ** (j - i))
                sim.rz(j, angle)

    @staticmethod
    def _qft_circuit_sc(sim: TransmonSimulator, n: int) -> None:
        """QFT on superconducting."""
        for i in range(n):
            sim.h(i)
            for j in range(i + 1, n):
                angle = math.pi / (2 ** (j - i))
                sim.rz(j, angle)

    @staticmethod
    def _random_circuit_ion(sim: TrappedIonSimulator, n: int, depth: int = 5) -> None:
        """Random circuit on trapped-ion."""
        rng = np.random.RandomState(42)
        for _ in range(depth):
            for q in range(n):
                sim.rx(q, rng.uniform(0, 2 * math.pi))
                sim.rz(q, rng.uniform(0, 2 * math.pi))
            for q in range(0, n - 1, 2):
                sim.cnot(q, q + 1)

    @staticmethod
    def _random_circuit_sc(sim: TransmonSimulator, n: int, depth: int = 5) -> None:
        """Random circuit on superconducting."""
        rng = np.random.RandomState(42)
        for _ in range(depth):
            for q in range(n):
                sim.rx(q, rng.uniform(0, 2 * math.pi))
                sim.rz(q, rng.uniform(0, 2 * math.pi))
            for q in range(0, n - 1, 2):
                sim.cnot(q, q + 1)

    # ------------------------------------------------------------------
    # Benchmark runners
    # ------------------------------------------------------------------

    def _run_ion(
        self, circuit_fn, mode: str, label: str
    ) -> CircuitBenchmark:
        """Run circuit on trapped-ion backend."""
        t0 = time.time()
        sim = TrappedIonSimulator(self.ion_config, execution_mode=mode)
        circuit_fn(sim)

        # In noisy mode the state vector is not available -- use
        # the density matrix diagonal to get measurement probabilities.
        if mode == "noisy":
            dm = sim.density_matrix()
            probs = np.real(np.diag(dm))
            probs = np.maximum(probs, 0.0)
            psum = probs.sum()
            if psum > 0:
                probs /= psum
        else:
            probs = np.array([abs(x)**2 for x in sim.statevector()])

        dt = (time.time() - t0) * 1000
        stats = sim.circuit_stats()

        # Ideal reference
        ideal_sim = TrappedIonSimulator(self.ion_config, execution_mode="ideal")
        circuit_fn(ideal_sim)
        ideal_probs = np.array([abs(x)**2 for x in ideal_sim.statevector()])
        fidelity = float(np.sum(np.sqrt(probs * ideal_probs)) ** 2)

        return CircuitBenchmark(
            backend_name=f"trapped_ion_{label}",
            execution_mode=mode,
            probabilities=probs,
            fidelity_vs_ideal=fidelity if mode != "ideal" else 1.0,
            num_gates_1q=stats.single_qubit_gates,
            num_gates_2q=stats.two_qubit_gates,
            wall_time_ms=dt,
        )

    def _run_sc(
        self, circuit_fn, mode: str, label: str
    ) -> CircuitBenchmark:
        """Run circuit on superconducting backend."""
        t0 = time.time()
        sim = TransmonSimulator(self.sc_config, execution_mode=mode)
        circuit_fn(sim)
        probs = sim.probabilities()
        dt = (time.time() - t0) * 1000
        stats = sim.circuit_stats()

        # Ideal reference
        ideal_sim = TransmonSimulator(self.sc_config, execution_mode="ideal")
        circuit_fn(ideal_sim)
        ideal_probs = ideal_sim.probabilities()
        fidelity = float(np.sum(np.sqrt(probs * ideal_probs)) ** 2)

        return CircuitBenchmark(
            backend_name=f"superconducting_{label}",
            execution_mode=mode,
            probabilities=probs,
            fidelity_vs_ideal=fidelity if mode != "ideal" else 1.0,
            num_gates_1q=stats.single_qubit_gates,
            num_gates_2q=stats.two_qubit_gates,
            wall_time_ms=dt,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def benchmark_bell(self) -> BackendComparison:
        """Benchmark Bell state preparation across backends."""
        results = {}
        for mode in ("ideal", "noisy"):
            results[f"ion_{mode}"] = self._run_ion(
                self._bell_circuit_ion, mode, mode
            )
            results[f"sc_{mode}"] = self._run_sc(
                self._bell_circuit_sc, mode, mode
            )
        return BackendComparison("Bell State", 2, results)

    def benchmark_ghz(self) -> BackendComparison:
        """Benchmark GHZ state preparation."""
        n = self.num_qubits
        results = {}
        for mode in ("ideal", "noisy"):
            results[f"ion_{mode}"] = self._run_ion(
                lambda sim: self._ghz_circuit_ion(sim, n), mode, mode
            )
            results[f"sc_{mode}"] = self._run_sc(
                lambda sim: self._ghz_circuit_sc(sim, n), mode, mode
            )
        return BackendComparison("GHZ State", n, results)

    def benchmark_qft(self) -> BackendComparison:
        """Benchmark Quantum Fourier Transform."""
        n = self.num_qubits
        results = {}
        for mode in ("ideal", "noisy"):
            results[f"ion_{mode}"] = self._run_ion(
                lambda sim: self._qft_circuit_ion(sim, n), mode, mode
            )
            results[f"sc_{mode}"] = self._run_sc(
                lambda sim: self._qft_circuit_sc(sim, n), mode, mode
            )
        return BackendComparison("QFT", n, results)

    def benchmark_random(self, depth: int = 5) -> BackendComparison:
        """Benchmark random circuits."""
        n = self.num_qubits
        results = {}
        for mode in ("ideal", "noisy"):
            results[f"ion_{mode}"] = self._run_ion(
                lambda sim: self._random_circuit_ion(sim, n, depth), mode, mode
            )
            results[f"sc_{mode}"] = self._run_sc(
                lambda sim: self._random_circuit_sc(sim, n, depth), mode, mode
            )
        return BackendComparison("Random Circuit", n, results)

    def run_all(self) -> list[BackendComparison]:
        """Run all benchmark circuits."""
        return [
            self.benchmark_bell(),
            self.benchmark_ghz(),
            self.benchmark_qft(),
            self.benchmark_random(),
        ]

    @staticmethod
    def print_report(comparisons: list[BackendComparison]) -> None:
        """Print a formatted benchmark report."""
        print("=" * 72)
        print("CROSS-BACKEND BENCHMARK REPORT")
        print("Trapped-Ion vs Superconducting Transmon")
        print("=" * 72)

        for comp in comparisons:
            print(f"\n--- {comp.circuit_name} ({comp.num_qubits}Q) ---")
            print(f"{'Backend':<25} {'Fidelity':>10} {'1Q Gates':>10} "
                  f"{'2Q Gates':>10} {'Time(ms)':>10}")
            print("-" * 65)
            for name, res in comp.results.items():
                print(f"{name:<25} {res.fidelity_vs_ideal:>10.4f} "
                      f"{res.num_gates_1q:>10} {res.num_gates_2q:>10} "
                      f"{res.wall_time_ms:>10.2f}")

        print("\n" + "=" * 72)


# ------------------------------------------------------------------
# Digital twin construction and validation
# ------------------------------------------------------------------

def build_digital_twin(
    frequencies_ghz: list[float],
    t1_us: list[float],
    t2_us: list[float],
    readout_fidelities: list[float],
    edges: list[tuple[int, int]],
    coupling_mhz: float = 3.0,
    native_gate: str = "ecr",
    two_qubit_fidelity: float = 0.995,
) -> ChipConfig:
    """Build a digital twin from hardware calibration data.

    Parameters
    ----------
    frequencies_ghz : list
        Per-qubit transition frequencies.
    t1_us : list
        Per-qubit T1 values.
    t2_us : list
        Per-qubit T2 values.
    readout_fidelities : list
        Per-qubit readout fidelities.
    edges : list of (int, int)
        Coupled qubit pairs.
    coupling_mhz : float
        Uniform coupling strength.
    native_gate : str
        Native 2Q gate type.
    two_qubit_fidelity : float
        Average 2Q gate fidelity.

    Returns
    -------
    ChipConfig
        Processor configuration matching the hardware.
    """
    from nqpu.superconducting import (
        TransmonQubit, ChipTopology, ChipConfig, NativeGateFamily
    )

    n = len(frequencies_ghz)
    qubits = []
    for i in range(n):
        qubits.append(TransmonQubit(
            frequency_ghz=frequencies_ghz[i],
            t1_us=t1_us[i],
            t2_us=t2_us[i],
            readout_fidelity=readout_fidelities[i],
        ))

    couplings = {(min(a, b), max(a, b)): coupling_mhz for a, b in edges}
    topo = ChipTopology(
        num_qubits=n,
        edges=edges,
        coupling_mhz=couplings,
    )

    gate_family = NativeGateFamily(native_gate)

    return ChipConfig(
        topology=topo,
        qubits=qubits,
        native_2q_gate=gate_family,
        two_qubit_fidelity=two_qubit_fidelity,
    )
