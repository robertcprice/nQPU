"""Cross-backend benchmarking: trapped-ion vs superconducting vs neutral atom.

Runs identical quantum circuits on all three hardware backends, comparing:
- Ideal fidelity (should be identical)
- Noisy fidelity (hardware-dependent error rates)
- Native gate counts (compilation efficiency)
- Estimated circuit duration
- Multi-qubit gate advantage (Toffoli benchmark for neutral atom CCZ)

Includes standard benchmark circuit generators (Bell, GHZ, QFT, random
Clifford, Toffoli-heavy, QAOA, supremacy), a backend adapter layer for
uniform access, analysis functions, and a recommendation engine that
suggests the optimal backend for a given circuit workload.

Example::

    from nqpu.benchmarks import CrossBackendBenchmark
    bench = CrossBackendBenchmark(num_qubits=4)
    results = bench.run_all()
    bench.print_report(results)
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from nqpu.ion_trap import TrappedIonSimulator, TrapConfig, IonSpecies
from nqpu.superconducting import (
    TransmonSimulator,
    ChipConfig,
    DevicePresets,
)
from nqpu.neutral_atom import (
    NeutralAtomSimulator,
    ArrayConfig,
    AtomSpecies,
)


# ======================================================================
# Gate-list circuit representation
# ======================================================================

# A gate is a tuple: (gate_name, qubit_indices..., [optional_params...])
# gate_name: 'h', 'x', 'y', 'z', 'cx', 'cz', 'ccx', 'rx', 'ry', 'rz',
#            's', 't', 'swap'
Gate = tuple


def _circuit_depth(circuit: list[Gate], n_qubits: int) -> int:
    """Compute circuit depth (longest qubit timeline).

    Each gate is placed in the earliest layer where all its qubits are free.
    """
    qubit_layer: dict[int, int] = {q: 0 for q in range(n_qubits)}
    for gate in circuit:
        name = str(gate[0]).lower()
        if name in ("h", "x", "y", "z", "s", "t"):
            q = int(gate[1])
            qubit_layer[q] = qubit_layer.get(q, 0) + 1
        elif name in ("rx", "ry", "rz"):
            q = int(gate[1])
            qubit_layer[q] = qubit_layer.get(q, 0) + 1
        elif name in ("cx", "cz", "swap"):
            q0, q1 = int(gate[1]), int(gate[2])
            layer = max(qubit_layer.get(q0, 0), qubit_layer.get(q1, 0)) + 1
            qubit_layer[q0] = layer
            qubit_layer[q1] = layer
        elif name in ("ccx", "ccz"):
            q0, q1, q2 = int(gate[1]), int(gate[2]), int(gate[3])
            layer = max(
                qubit_layer.get(q0, 0),
                qubit_layer.get(q1, 0),
                qubit_layer.get(q2, 0),
            ) + 1
            qubit_layer[q0] = layer
            qubit_layer[q1] = layer
            qubit_layer[q2] = layer
    return max(qubit_layer.values()) if qubit_layer else 0


def _count_entangling_gates(circuit: list[Gate]) -> int:
    """Count two-qubit and three-qubit entangling gates."""
    count = 0
    for gate in circuit:
        name = str(gate[0]).lower()
        if name in ("cx", "cz", "swap", "ccx", "ccz"):
            count += 1
    return count


def _count_three_qubit_gates(circuit: list[Gate]) -> int:
    """Count three-qubit gates (Toffoli / CCZ)."""
    count = 0
    for gate in circuit:
        name = str(gate[0]).lower()
        if name in ("ccx", "ccz"):
            count += 1
    return count


# ======================================================================
# Standard benchmark circuits (gate-list based)
# ======================================================================


class BenchmarkCircuit:
    """A benchmark circuit definition with metadata and standard generators.

    Represents a quantum circuit as a list of gate tuples, together with
    descriptive metadata.  Provides class-method factories for all
    standard benchmark circuits.

    Attributes
    ----------
    name : str
        Human-readable circuit name.
    description : str
        What the circuit tests.
    n_qubits : int
        Number of qubits required.
    gates : list[Gate]
        Circuit as a list of ``(gate_name, qubit, ..., [params...])`` tuples.
    """

    def __init__(
        self,
        name: str,
        description: str,
        n_qubits: int,
        gates: list[Gate],
    ) -> None:
        self.name = name
        self.description = description
        self.n_qubits = n_qubits
        self.gates = list(gates)

    # --- computed properties ---

    @property
    def gate_count(self) -> int:
        """Total number of gates."""
        return len(self.gates)

    @property
    def depth(self) -> int:
        """Circuit depth (longest qubit timeline)."""
        return _circuit_depth(self.gates, self.n_qubits)

    @property
    def entangling_gate_count(self) -> int:
        """Number of multi-qubit entangling gates."""
        return _count_entangling_gates(self.gates)

    @property
    def three_qubit_gate_count(self) -> int:
        """Number of three-qubit gates (Toffoli / CCZ)."""
        return _count_three_qubit_gates(self.gates)

    @property
    def toffoli_fraction(self) -> float:
        """Fraction of gates that are three-qubit."""
        if self.gate_count == 0:
            return 0.0
        return self.three_qubit_gate_count / self.gate_count

    # --- standard circuit generators ---

    @classmethod
    def bell_state(cls, n_pairs: int = 1) -> BenchmarkCircuit:
        """Generate Bell pair preparation circuit.

        Parameters
        ----------
        n_pairs : int
            Number of Bell pairs.  Total qubits = ``2 * n_pairs``.
        """
        n_qubits = 2 * n_pairs
        gates: list[Gate] = []
        for i in range(n_pairs):
            q0, q1 = 2 * i, 2 * i + 1
            gates.append(("h", q0))
            gates.append(("cx", q0, q1))
        return cls(
            name="Bell State",
            description=f"{n_pairs} Bell pair(s)",
            n_qubits=n_qubits,
            gates=gates,
        )

    @classmethod
    def ghz_state(cls, n_qubits: int) -> BenchmarkCircuit:
        """Generate GHZ state preparation circuit.

        Parameters
        ----------
        n_qubits : int
            Number of qubits (>= 2).
        """
        if n_qubits < 2:
            raise ValueError("GHZ state requires >= 2 qubits")
        gates: list[Gate] = [("h", 0)]
        for i in range(n_qubits - 1):
            gates.append(("cx", i, i + 1))
        return cls(
            name="GHZ State",
            description=f"{n_qubits}-qubit GHZ",
            n_qubits=n_qubits,
            gates=gates,
        )

    @classmethod
    def qft_circuit(cls, n_qubits: int) -> BenchmarkCircuit:
        """Generate Quantum Fourier Transform circuit.

        Parameters
        ----------
        n_qubits : int
            Number of qubits (>= 1).
        """
        gates: list[Gate] = []
        for i in range(n_qubits):
            gates.append(("h", i))
            for j in range(i + 1, n_qubits):
                angle = math.pi / (2 ** (j - i))
                # Controlled-Rz decomposition
                gates.append(("cx", j, i))
                gates.append(("rz", i, -angle / 2))
                gates.append(("cx", j, i))
                gates.append(("rz", i, angle / 2))
        # Swap to reverse qubit order
        for i in range(n_qubits // 2):
            j = n_qubits - 1 - i
            if i < j:
                gates.append(("swap", i, j))
        return cls(
            name="QFT",
            description=f"{n_qubits}-qubit QFT",
            n_qubits=n_qubits,
            gates=gates,
        )

    @classmethod
    def random_clifford(
        cls,
        n_qubits: int,
        depth: int,
        seed: int | None = None,
    ) -> BenchmarkCircuit:
        """Generate a random Clifford circuit.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        depth : int
            Number of layers.
        seed : int, optional
            Random seed for reproducibility.
        """
        rng = np.random.RandomState(seed)
        sq_gates = ["h", "s", "x", "y", "z"]
        gates: list[Gate] = []
        for _ in range(depth):
            for q in range(n_qubits):
                g = sq_gates[rng.randint(len(sq_gates))]
                gates.append((g, q))
            pairs = list(range(0, n_qubits - 1))
            rng.shuffle(pairs)
            for q in pairs[: max(1, n_qubits // 2)]:
                gates.append(("cx", q, q + 1))
        return cls(
            name="Random Clifford",
            description=f"{n_qubits}Q depth-{depth} random Clifford",
            n_qubits=n_qubits,
            gates=gates,
        )

    @classmethod
    def toffoli_heavy(
        cls, n_qubits: int, n_toffolis: int | None = None
    ) -> BenchmarkCircuit:
        """Generate a Toffoli-heavy circuit.

        Showcases the native CCZ advantage of neutral-atom backends.

        Parameters
        ----------
        n_qubits : int
            Number of qubits (>= 3).
        n_toffolis : int, optional
            Number of Toffoli gates.  Default: ``n_qubits - 2``.
        """
        if n_qubits < 3:
            raise ValueError("Toffoli-heavy circuit requires >= 3 qubits")
        if n_toffolis is None:
            n_toffolis = n_qubits - 2
        gates: list[Gate] = []
        for q in range(n_qubits):
            gates.append(("h", q))
        for i in range(n_toffolis):
            q0 = i % n_qubits
            q1 = (i + 1) % n_qubits
            q2 = (i + 2) % n_qubits
            gates.append(("ccx", q0, q1, q2))
        for q in range(n_qubits):
            gates.append(("h", q))
        return cls(
            name="Toffoli Circuit",
            description=f"{n_qubits}Q with {n_toffolis} Toffoli gates",
            n_qubits=n_qubits,
            gates=gates,
        )

    @classmethod
    def qaoa_layer(cls, n_qubits: int, p: int = 1) -> BenchmarkCircuit:
        """Generate a QAOA ansatz circuit (MaxCut on a chain).

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        p : int
            Number of QAOA layers.
        """
        gates: list[Gate] = []
        for q in range(n_qubits):
            gates.append(("h", q))
        for layer in range(p):
            gamma = math.pi / (4 * (layer + 1))
            beta = math.pi / (8 * (layer + 1))
            for i in range(n_qubits - 1):
                gates.append(("cx", i, i + 1))
                gates.append(("rz", i + 1, 2 * gamma))
                gates.append(("cx", i, i + 1))
            for q in range(n_qubits):
                gates.append(("rx", q, 2 * beta))
        return cls(
            name="QAOA",
            description=f"{n_qubits}Q QAOA p={p}",
            n_qubits=n_qubits,
            gates=gates,
        )

    @classmethod
    def supremacy_circuit(
        cls,
        n_qubits: int,
        depth: int,
        seed: int | None = None,
    ) -> BenchmarkCircuit:
        """Generate a random circuit for quantum volume estimation.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        depth : int
            Number of layers.
        seed : int, optional
            Random seed.
        """
        rng = np.random.RandomState(seed)
        gates: list[Gate] = []
        for layer in range(depth):
            for q in range(n_qubits):
                theta = rng.uniform(0, 2 * math.pi)
                phi = rng.uniform(0, 2 * math.pi)
                gates.append(("rz", q, theta))
                gates.append(("rx", q, phi))
            start = layer % 2
            for i in range(start, n_qubits - 1, 2):
                gates.append(("cz", i, i + 1))
        return cls(
            name="Supremacy Circuit",
            description=f"{n_qubits}Q depth-{depth} random circuit",
            n_qubits=n_qubits,
            gates=gates,
        )

    def __repr__(self) -> str:
        return (
            f"BenchmarkCircuit(name={self.name!r}, n_qubits={self.n_qubits}, "
            f"gates={self.gate_count}, depth={self.depth})"
        )


# ======================================================================
# Backend adapter layer
# ======================================================================


class BackendAdapter(ABC):
    """Abstract interface for wrapping a backend simulator.

    Subclasses handle configuration, gate compilation, and noise model
    differences so that the benchmark orchestrator treats all backends
    identically.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name string."""

    @abstractmethod
    def create_simulator(self, n_qubits: int, mode: str) -> Any:
        """Create and return a simulator instance."""

    @abstractmethod
    def run_circuit(self, simulator: Any, circuit: list[Gate]) -> None:
        """Execute the gate list on the simulator (resets first)."""

    @abstractmethod
    def get_probabilities(self, simulator: Any) -> np.ndarray:
        """Return measurement probabilities for all basis states."""

    @abstractmethod
    def get_native_gate_count(self, simulator: Any) -> dict[str, int]:
        """Return native gate counts as ``{category: count}``."""

    @abstractmethod
    def get_fidelity_estimate(self, simulator: Any) -> float:
        """Return the noise-model fidelity estimate."""


def _apply_gate_to_sim(sim: Any, gate: Gate, backend: str) -> None:
    """Apply a single gate tuple to a simulator.

    Handles the union of all gate names across all backends.
    For ``ccx`` / ``ccz`` on backends without native support, the
    Toffoli is decomposed into six CNOTs plus single-qubit gates.
    """
    name = str(gate[0]).lower()

    if name == "h":
        sim.h(int(gate[1]))
    elif name == "x":
        sim.x(int(gate[1]))
    elif name == "y":
        sim.y(int(gate[1]))
    elif name == "z":
        sim.z(int(gate[1]))
    elif name == "s":
        sim.rz(int(gate[1]), math.pi / 2)
    elif name == "t":
        sim.rz(int(gate[1]), math.pi / 4)
    elif name == "rx":
        sim.rx(int(gate[1]), float(gate[2]))
    elif name == "ry":
        sim.ry(int(gate[1]), float(gate[2]))
    elif name == "rz":
        sim.rz(int(gate[1]), float(gate[2]))
    elif name in ("cx", "cnot"):
        sim.cnot(int(gate[1]), int(gate[2]))
    elif name == "cz":
        sim.cz(int(gate[1]), int(gate[2]))
    elif name == "swap":
        q0, q1 = int(gate[1]), int(gate[2])
        if backend == "superconducting" and hasattr(sim, "swap"):
            sim.swap(q0, q1)
        else:
            sim.cnot(q0, q1)
            sim.cnot(q1, q0)
            sim.cnot(q0, q1)
    elif name == "ccx":
        q0, q1, q2 = int(gate[1]), int(gate[2]), int(gate[3])
        if backend == "neutral_atom":
            sim.toffoli(q0, q1, q2)
        else:
            _toffoli_decomposed(sim, q0, q1, q2)
    elif name == "ccz":
        q0, q1, q2 = int(gate[1]), int(gate[2]), int(gate[3])
        if backend == "neutral_atom":
            sim.ccz(q0, q1, q2)
        else:
            sim.h(q2)
            _toffoli_decomposed(sim, q0, q1, q2)
            sim.h(q2)
    else:
        raise ValueError(f"Unsupported gate '{name}' for backend '{backend}'")


def _toffoli_decomposed(sim: Any, q0: int, q1: int, q2: int) -> None:
    """Standard Toffoli decomposition into 6 CNOTs + single-qubit gates."""
    sim.h(q2)
    sim.cnot(q1, q2)
    sim.rz(q2, -math.pi / 4)
    sim.cnot(q0, q2)
    sim.rz(q2, math.pi / 4)
    sim.cnot(q1, q2)
    sim.rz(q2, -math.pi / 4)
    sim.cnot(q0, q2)
    sim.rz(q2, math.pi / 4)
    sim.rz(q1, math.pi / 4)
    sim.h(q2)
    sim.cnot(q0, q1)
    sim.rz(q0, math.pi / 4)
    sim.rz(q1, -math.pi / 4)
    sim.cnot(q0, q1)


class IonTrapAdapter(BackendAdapter):
    """Adapter for the trapped-ion backend."""

    def __init__(self, species: IonSpecies | None = None) -> None:
        self._species = species if species is not None else IonSpecies.YB171

    @property
    def name(self) -> str:
        return "trapped_ion"

    def create_simulator(self, n_qubits: int, mode: str) -> TrappedIonSimulator:
        config = TrapConfig(n_ions=n_qubits, species=self._species)
        return TrappedIonSimulator(config, execution_mode=mode)

    def run_circuit(self, simulator: TrappedIonSimulator, circuit: list[Gate]) -> None:
        simulator.reset()
        for gate in circuit:
            _apply_gate_to_sim(simulator, gate, "trapped_ion")

    def get_probabilities(self, simulator: TrappedIonSimulator) -> np.ndarray:
        if simulator.execution_mode == "noisy":
            dm = simulator.density_matrix()
            probs = np.real(np.diag(dm))
            probs = np.maximum(probs, 0.0)
            total = np.sum(probs)
            if total > 0 and abs(total - 1.0) > 1e-10:
                probs /= total
            return probs
        return np.abs(simulator.statevector()) ** 2

    def get_native_gate_count(self, simulator: TrappedIonSimulator) -> dict[str, int]:
        stats = simulator.circuit_stats()
        return {
            "single_qubit": stats.single_qubit_gates,
            "two_qubit": stats.two_qubit_gates,
            "total": stats.total_gates,
        }

    def get_fidelity_estimate(self, simulator: TrappedIonSimulator) -> float:
        return simulator.fidelity_estimate()


class NeutralAtomAdapter(BackendAdapter):
    """Adapter for the neutral-atom (Rydberg blockade) backend."""

    def __init__(self, species: AtomSpecies | None = None) -> None:
        self._species = species if species is not None else AtomSpecies.RB87

    @property
    def name(self) -> str:
        return "neutral_atom"

    def create_simulator(self, n_qubits: int, mode: str) -> NeutralAtomSimulator:
        config = ArrayConfig(n_atoms=n_qubits, species=self._species)
        return NeutralAtomSimulator(config, execution_mode=mode)

    def run_circuit(
        self, simulator: NeutralAtomSimulator, circuit: list[Gate]
    ) -> None:
        simulator.reset()
        for gate in circuit:
            _apply_gate_to_sim(simulator, gate, "neutral_atom")

    def get_probabilities(self, simulator: NeutralAtomSimulator) -> np.ndarray:
        if simulator.execution_mode == "noisy":
            dm = simulator.density_matrix()
            probs = np.real(np.diag(dm))
            probs = np.maximum(probs, 0.0)
            total = np.sum(probs)
            if total > 0 and abs(total - 1.0) > 1e-10:
                probs /= total
            return probs
        return np.abs(simulator.statevector()) ** 2

    def get_native_gate_count(
        self, simulator: NeutralAtomSimulator
    ) -> dict[str, int]:
        stats = simulator.circuit_stats()
        return {
            "single_qubit": stats.single_qubit_gates,
            "two_qubit": stats.two_qubit_gates,
            "three_qubit": stats.three_qubit_gates,
            "total": stats.total_gates,
        }

    def get_fidelity_estimate(self, simulator: NeutralAtomSimulator) -> float:
        return simulator.fidelity_estimate()


class SuperconductingAdapter(BackendAdapter):
    """Adapter for the superconducting transmon backend."""

    def __init__(self, preset: DevicePresets | None = None) -> None:
        self._preset = preset if preset is not None else DevicePresets.IBM_HERON

    @property
    def name(self) -> str:
        return "superconducting"

    def create_simulator(self, n_qubits: int, mode: str) -> TransmonSimulator:
        config = self._preset.build(num_qubits=n_qubits)
        return TransmonSimulator(config, execution_mode=mode)

    def run_circuit(
        self, simulator: TransmonSimulator, circuit: list[Gate]
    ) -> None:
        simulator.reset()
        for gate in circuit:
            _apply_gate_to_sim(simulator, gate, "superconducting")

    def get_probabilities(self, simulator: TransmonSimulator) -> np.ndarray:
        return simulator.probabilities()

    def get_native_gate_count(self, simulator: TransmonSimulator) -> dict[str, int]:
        stats = simulator.circuit_stats()
        return {
            "single_qubit": stats.single_qubit_gates,
            "two_qubit": stats.two_qubit_gates,
            "total": stats.total_gates,
        }

    def get_fidelity_estimate(self, simulator: TransmonSimulator) -> float:
        return simulator.fidelity_estimate()


# ======================================================================
# Results dataclass (backward-compatible name: CircuitBenchmark)
# ======================================================================


@dataclass
class CircuitBenchmark:
    """Results from running a single circuit on one backend.

    The name ``CircuitBenchmark`` is kept for backward compatibility with
    the existing test suite.  For the circuit *definition* class, see
    :class:`BenchmarkCircuit`.
    """

    backend_name: str
    execution_mode: str
    probabilities: np.ndarray
    fidelity_vs_ideal: float
    num_gates_1q: int
    num_gates_2q: int
    wall_time_ms: float
    num_gates_3q: int = 0
    native_gate_counts: dict[str, int] = field(default_factory=dict)
    estimated_fidelity: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Comparison container
# ======================================================================


@dataclass
class BackendComparison:
    """Comparison of a circuit across multiple backends.

    Attributes
    ----------
    circuit_name : str
        Name of the benchmark circuit.
    num_qubits : int
        Number of qubits.
    results : dict[str, CircuitBenchmark]
        Mapping from backend key to per-backend result.
    """

    circuit_name: str
    num_qubits: int
    results: dict[str, CircuitBenchmark] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Generate a summary dict."""
        rows: dict[str, dict[str, Any]] = {}
        for name, res in self.results.items():
            rows[name] = {
                "fidelity": res.fidelity_vs_ideal,
                "1Q_gates": res.num_gates_1q,
                "2Q_gates": res.num_gates_2q,
                "wall_ms": res.wall_time_ms,
            }
        return {
            "circuit": self.circuit_name,
            "num_qubits": self.num_qubits,
            "backends": rows,
        }

    def best_backend_for(self, metric: str) -> str:
        """Return the backend key with the best value for *metric*.

        Parameters
        ----------
        metric : str
            One of ``'fidelity'`` (higher is better), ``'gate_count'`` or
            ``'wall_time'`` (lower is better).
        """
        if not self.results:
            raise ValueError("No results to compare")
        if metric == "fidelity":
            return max(
                self.results, key=lambda k: self.results[k].fidelity_vs_ideal
            )
        elif metric == "gate_count":
            return min(
                self.results,
                key=lambda k: (
                    self.results[k].num_gates_1q
                    + self.results[k].num_gates_2q
                    + self.results[k].num_gates_3q
                ),
            )
        elif metric == "wall_time":
            return min(
                self.results, key=lambda k: self.results[k].wall_time_ms
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def fidelity_ranking(self) -> list[tuple[str, float]]:
        """Return backends ranked by fidelity (highest first)."""
        return sorted(
            [(k, v.fidelity_vs_ideal) for k, v in self.results.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for serialisation."""
        return {
            "circuit_name": self.circuit_name,
            "num_qubits": self.num_qubits,
            "results": {
                k: {
                    "backend_name": v.backend_name,
                    "execution_mode": v.execution_mode,
                    "fidelity_vs_ideal": v.fidelity_vs_ideal,
                    "num_gates_1q": v.num_gates_1q,
                    "num_gates_2q": v.num_gates_2q,
                    "num_gates_3q": v.num_gates_3q,
                    "wall_time_ms": v.wall_time_ms,
                }
                for k, v in self.results.items()
            },
        }


# ======================================================================
# Main orchestrator
# ======================================================================


class CrossBackendBenchmark:
    """Cross-backend benchmarking suite.

    Runs identical circuits on trapped-ion, superconducting, and neutral-atom
    backends in both ideal and noisy modes.

    Parameters
    ----------
    num_qubits : int
        Number of qubits for the benchmark circuits.
    ion_species : IonSpecies, optional
        Ion species for the trapped-ion backend.
    sc_preset : DevicePresets, optional
        Device preset for the superconducting backend.
    na_species : AtomSpecies, optional
        Atom species for the neutral-atom backend.
    """

    def __init__(
        self,
        num_qubits: int = 3,
        ion_species: IonSpecies | None = None,
        sc_preset: DevicePresets = DevicePresets.IBM_HERON,
        na_species: AtomSpecies | None = None,
    ) -> None:
        self.num_qubits = num_qubits

        if ion_species is None:
            ion_species = IonSpecies.YB171
        if na_species is None:
            na_species = AtomSpecies.RB87

        # Eagerly build configs (backward-compat test access)
        self.ion_config = TrapConfig(n_ions=num_qubits, species=ion_species)
        self.sc_config = sc_preset.build(num_qubits=num_qubits)
        self.na_config = ArrayConfig(n_atoms=num_qubits, species=na_species)

        # Backend adapters
        self._adapters: dict[str, BackendAdapter] = {
            "ion": IonTrapAdapter(species=ion_species),
            "sc": SuperconductingAdapter(preset=sc_preset),
            "na": NeutralAtomAdapter(species=na_species),
        }

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------

    def add_backend(self, name: str, adapter: BackendAdapter) -> None:
        """Register a custom backend adapter.

        Parameters
        ----------
        name : str
            Short backend key (used as prefix in result keys).
        adapter : BackendAdapter
            Adapter wrapping the backend simulator.
        """
        self._adapters[name] = adapter

    def register_default_backends(self) -> None:
        """Re-register the three default backends."""
        # Already done in __init__; provided for interface completeness.
        pass

    # ------------------------------------------------------------------
    # Generic adapter-based execution
    # ------------------------------------------------------------------

    def _run_via_adapter(
        self,
        adapter: BackendAdapter,
        circuit: BenchmarkCircuit,
        mode: str,
        label: str,
    ) -> CircuitBenchmark:
        """Run a ``BenchmarkCircuit`` through an adapter."""
        t0 = time.time()
        sim = adapter.create_simulator(circuit.n_qubits, mode)
        adapter.run_circuit(sim, circuit.gates)
        probs = adapter.get_probabilities(sim)
        dt = (time.time() - t0) * 1000.0

        native = adapter.get_native_gate_count(sim)

        if mode == "noisy":
            ideal_sim = adapter.create_simulator(circuit.n_qubits, "ideal")
            adapter.run_circuit(ideal_sim, circuit.gates)
            ideal_probs = adapter.get_probabilities(ideal_sim)
            fidelity = float(np.sum(np.sqrt(probs * ideal_probs)) ** 2)
        else:
            fidelity = 1.0

        return CircuitBenchmark(
            backend_name=f"{adapter.name}_{label}",
            execution_mode=mode,
            probabilities=probs,
            fidelity_vs_ideal=fidelity,
            num_gates_1q=native.get("single_qubit", 0),
            num_gates_2q=native.get("two_qubit", 0),
            num_gates_3q=native.get("three_qubit", 0),
            wall_time_ms=dt,
            native_gate_counts=native,
            estimated_fidelity=adapter.get_fidelity_estimate(sim),
        )

    def run_circuit(
        self,
        circuit: BenchmarkCircuit,
        backends: list[str] | None = None,
        modes: list[str] | None = None,
    ) -> BackendComparison:
        """Run a benchmark circuit across selected backends and modes.

        Parameters
        ----------
        circuit : BenchmarkCircuit
            Benchmark circuit to run.
        backends : list[str], optional
            Backend keys.  Default: all registered.
        modes : list[str], optional
            Execution modes.  Default: ``['ideal', 'noisy']``.

        Returns
        -------
        BackendComparison
        """
        if modes is None:
            modes = ["ideal", "noisy"]
        if backends is None:
            backends = list(self._adapters.keys())

        results: dict[str, CircuitBenchmark] = {}
        for bk in backends:
            adapter = self._adapters[bk]
            for mode in modes:
                key = f"{bk}_{mode}"
                results[key] = self._run_via_adapter(
                    adapter, circuit, mode, mode
                )
        return BackendComparison(circuit.name, circuit.n_qubits, results)

    def run_scaling_analysis(
        self,
        circuit_gen: Callable[[int], BenchmarkCircuit],
        qubit_range: range | list[int],
        backends: list[str] | None = None,
    ) -> dict[str, list[BackendComparison]]:
        """Analyse how backends scale with qubit count.

        Parameters
        ----------
        circuit_gen : callable
            ``f(n_qubits) -> BenchmarkCircuit``.
        qubit_range : range or list
            Qubit counts to test.
        backends : list[str], optional
            Backend keys.

        Returns
        -------
        dict[str, list[BackendComparison]]
        """
        results: dict[str, list[BackendComparison]] = {}
        for n in qubit_range:
            circuit = circuit_gen(n)
            comp = self.run_circuit(circuit, backends=backends)
            results[str(n)] = [comp]
        return results

    def recommend_backend(self, circuit: BenchmarkCircuit) -> str:
        """Suggest the optimal backend for the given circuit.

        Heuristics:
        - Toffoli-heavy (>10% three-qubit gates) -> neutral atom (native CCZ)
        - High entanglement density (>30%) -> trapped ion (all-to-all)
        - Otherwise -> superconducting (fast parallel gates)
        """
        if circuit.toffoli_fraction > 0.10:
            return "na"
        ent_ratio = circuit.entangling_gate_count / max(1, circuit.gate_count)
        if ent_ratio > 0.3:
            return "ion"
        return "sc"

    # ------------------------------------------------------------------
    # Legacy circuit definitions (used by existing test_cross_backend.py)
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
    def _random_circuit_ion(
        sim: TrappedIonSimulator, n: int, depth: int = 5
    ) -> None:
        """Random circuit on trapped-ion."""
        rng = np.random.RandomState(42)
        for _ in range(depth):
            for q in range(n):
                sim.rx(q, rng.uniform(0, 2 * math.pi))
                sim.rz(q, rng.uniform(0, 2 * math.pi))
            for q in range(0, n - 1, 2):
                sim.cnot(q, q + 1)

    @staticmethod
    def _random_circuit_sc(
        sim: TransmonSimulator, n: int, depth: int = 5
    ) -> None:
        """Random circuit on superconducting."""
        rng = np.random.RandomState(42)
        for _ in range(depth):
            for q in range(n):
                sim.rx(q, rng.uniform(0, 2 * math.pi))
                sim.rz(q, rng.uniform(0, 2 * math.pi))
            for q in range(0, n - 1, 2):
                sim.cnot(q, q + 1)

    @staticmethod
    def _bell_circuit_na(sim: NeutralAtomSimulator) -> None:
        """Bell state on neutral atom."""
        sim.h(0)
        sim.cnot(0, 1)

    @staticmethod
    def _ghz_circuit_na(sim: NeutralAtomSimulator, n: int) -> None:
        """GHZ state on neutral atom."""
        sim.h(0)
        for i in range(1, n):
            sim.cnot(0, i)

    @staticmethod
    def _qft_circuit_na(sim: NeutralAtomSimulator, n: int) -> None:
        """QFT on neutral atom."""
        for i in range(n):
            sim.h(i)
            for j in range(i + 1, n):
                angle = math.pi / (2 ** (j - i))
                sim.rz(j, angle)

    @staticmethod
    def _random_circuit_na(
        sim: NeutralAtomSimulator, n: int, depth: int = 5
    ) -> None:
        """Random circuit on neutral atom."""
        rng = np.random.RandomState(42)
        for _ in range(depth):
            for q in range(n):
                sim.rx(q, rng.uniform(0, 2 * math.pi))
                sim.rz(q, rng.uniform(0, 2 * math.pi))
            for q in range(0, n - 1, 2):
                sim.cnot(q, q + 1)

    @staticmethod
    def _toffoli_circuit_ion(sim: TrappedIonSimulator, n: int) -> None:
        """Toffoli-heavy circuit on trapped-ion (decomposed)."""
        sim.h(0)
        for q in range(0, n - 2):
            _toffoli_decomposed(sim, q, q + 1, q + 2)

    @staticmethod
    def _toffoli_circuit_sc(sim: TransmonSimulator, n: int) -> None:
        """Toffoli-heavy circuit on superconducting (decomposed)."""
        sim.h(0)
        for q in range(0, n - 2):
            _toffoli_decomposed(sim, q, q + 1, q + 2)

    @staticmethod
    def _toffoli_circuit_na(sim: NeutralAtomSimulator, n: int) -> None:
        """Toffoli-heavy circuit on neutral atom (native CCZ)."""
        sim.h(0)
        for q in range(0, n - 2):
            sim.toffoli(q, q + 1, q + 2)

    # ------------------------------------------------------------------
    # Legacy benchmark runners (direct simulator construction)
    # ------------------------------------------------------------------

    def _run_ion(
        self, circuit_fn: Callable, mode: str, label: str
    ) -> CircuitBenchmark:
        """Run circuit on trapped-ion backend."""
        t0 = time.time()
        sim = TrappedIonSimulator(self.ion_config, execution_mode=mode)
        circuit_fn(sim)

        if mode == "noisy":
            dm = sim.density_matrix()
            probs = np.real(np.diag(dm))
            probs = np.maximum(probs, 0.0)
            psum = probs.sum()
            if psum > 0:
                probs /= psum
        else:
            probs = np.abs(sim.statevector()) ** 2

        dt = (time.time() - t0) * 1000
        stats = sim.circuit_stats()

        ideal_sim = TrappedIonSimulator(self.ion_config, execution_mode="ideal")
        circuit_fn(ideal_sim)
        ideal_probs = np.abs(ideal_sim.statevector()) ** 2
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
        self, circuit_fn: Callable, mode: str, label: str
    ) -> CircuitBenchmark:
        """Run circuit on superconducting backend."""
        t0 = time.time()
        sim = TransmonSimulator(self.sc_config, execution_mode=mode)
        circuit_fn(sim)
        probs = sim.probabilities()
        dt = (time.time() - t0) * 1000
        stats = sim.circuit_stats()

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

    def _run_na(
        self, circuit_fn: Callable, mode: str, label: str
    ) -> CircuitBenchmark:
        """Run circuit on neutral-atom backend."""
        t0 = time.time()
        sim = NeutralAtomSimulator(self.na_config, execution_mode=mode)
        circuit_fn(sim)

        if mode == "noisy":
            dm = sim.density_matrix()
            probs = np.real(np.diag(dm))
            probs = np.maximum(probs, 0.0)
            psum = probs.sum()
            if psum > 0:
                probs /= psum
        else:
            probs = np.abs(sim.statevector()) ** 2

        dt = (time.time() - t0) * 1000
        stats = sim.circuit_stats()

        ideal_sim = NeutralAtomSimulator(self.na_config, execution_mode="ideal")
        circuit_fn(ideal_sim)
        ideal_probs = np.abs(ideal_sim.statevector()) ** 2
        fidelity = float(np.sum(np.sqrt(probs * ideal_probs)) ** 2)

        extra: dict[str, Any] = {}
        if hasattr(stats, "three_qubit_gates"):
            extra["three_qubit_gates"] = stats.three_qubit_gates

        return CircuitBenchmark(
            backend_name=f"neutral_atom_{label}",
            execution_mode=mode,
            probabilities=probs,
            fidelity_vs_ideal=fidelity if mode != "ideal" else 1.0,
            num_gates_1q=stats.single_qubit_gates,
            num_gates_2q=stats.two_qubit_gates,
            wall_time_ms=dt,
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Public legacy benchmark API
    # ------------------------------------------------------------------

    def benchmark_bell(self) -> BackendComparison:
        """Benchmark Bell state preparation across backends."""
        results: dict[str, CircuitBenchmark] = {}
        for mode in ("ideal", "noisy"):
            results[f"ion_{mode}"] = self._run_ion(
                self._bell_circuit_ion, mode, mode
            )
            results[f"sc_{mode}"] = self._run_sc(
                self._bell_circuit_sc, mode, mode
            )
            results[f"na_{mode}"] = self._run_na(
                self._bell_circuit_na, mode, mode
            )
        return BackendComparison("Bell State", self.num_qubits, results)

    def benchmark_ghz(self) -> BackendComparison:
        """Benchmark GHZ state preparation."""
        n = self.num_qubits
        results: dict[str, CircuitBenchmark] = {}
        for mode in ("ideal", "noisy"):
            results[f"ion_{mode}"] = self._run_ion(
                lambda sim: self._ghz_circuit_ion(sim, n), mode, mode
            )
            results[f"sc_{mode}"] = self._run_sc(
                lambda sim: self._ghz_circuit_sc(sim, n), mode, mode
            )
            results[f"na_{mode}"] = self._run_na(
                lambda sim: self._ghz_circuit_na(sim, n), mode, mode
            )
        return BackendComparison("GHZ State", n, results)

    def benchmark_qft(self) -> BackendComparison:
        """Benchmark Quantum Fourier Transform."""
        n = self.num_qubits
        results: dict[str, CircuitBenchmark] = {}
        for mode in ("ideal", "noisy"):
            results[f"ion_{mode}"] = self._run_ion(
                lambda sim: self._qft_circuit_ion(sim, n), mode, mode
            )
            results[f"sc_{mode}"] = self._run_sc(
                lambda sim: self._qft_circuit_sc(sim, n), mode, mode
            )
            results[f"na_{mode}"] = self._run_na(
                lambda sim: self._qft_circuit_na(sim, n), mode, mode
            )
        return BackendComparison("QFT", n, results)

    def benchmark_random(self, depth: int = 5) -> BackendComparison:
        """Benchmark random circuits."""
        n = self.num_qubits
        results: dict[str, CircuitBenchmark] = {}
        for mode in ("ideal", "noisy"):
            results[f"ion_{mode}"] = self._run_ion(
                lambda sim: self._random_circuit_ion(sim, n, depth),
                mode,
                mode,
            )
            results[f"sc_{mode}"] = self._run_sc(
                lambda sim: self._random_circuit_sc(sim, n, depth),
                mode,
                mode,
            )
            results[f"na_{mode}"] = self._run_na(
                lambda sim: self._random_circuit_na(sim, n, depth),
                mode,
                mode,
            )
        return BackendComparison("Random Circuit", n, results)

    def benchmark_toffoli(self) -> BackendComparison:
        """Benchmark Toffoli-heavy circuit (neutral-atom advantage)."""
        n = max(self.num_qubits, 3)
        results: dict[str, CircuitBenchmark] = {}
        for mode in ("ideal", "noisy"):
            results[f"ion_{mode}"] = self._run_ion(
                lambda sim: self._toffoli_circuit_ion(sim, n), mode, mode
            )
            results[f"sc_{mode}"] = self._run_sc(
                lambda sim: self._toffoli_circuit_sc(sim, n), mode, mode
            )
            results[f"na_{mode}"] = self._run_na(
                lambda sim: self._toffoli_circuit_na(sim, n), mode, mode
            )
        return BackendComparison("Toffoli Circuit", n, results)

    def run_all(self) -> list[BackendComparison]:
        """Run all benchmark circuits."""
        return [
            self.benchmark_bell(),
            self.benchmark_ghz(),
            self.benchmark_qft(),
            self.benchmark_random(),
            self.benchmark_toffoli(),
        ]

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(comparisons: list[BackendComparison]) -> str:
        """Print a formatted benchmark report.

        Returns the report string.
        """
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append("CROSS-BACKEND BENCHMARK REPORT")
        lines.append(
            "Trapped-Ion vs Superconducting vs Neutral Atom"
        )
        lines.append("=" * 78)

        for comp in comparisons:
            lines.append(
                f"\n--- {comp.circuit_name} ({comp.num_qubits}Q) ---"
            )
            lines.append(
                f"{'Backend':<25} {'Fidelity':>10} {'1Q Gates':>10} "
                f"{'2Q Gates':>10} {'Time(ms)':>10}"
            )
            lines.append("-" * 65)
            for name, res in comp.results.items():
                extra_str = ""
                if res.extra.get("three_qubit_gates", 0) > 0:
                    extra_str = (
                        f"  [3Q: {res.extra['three_qubit_gates']}]"
                    )
                lines.append(
                    f"{name:<25} {res.fidelity_vs_ideal:>10.4f} "
                    f"{res.num_gates_1q:>10} {res.num_gates_2q:>10} "
                    f"{res.wall_time_ms:>10.2f}{extra_str}"
                )

        lines.append("\n" + "=" * 78)

        report = "\n".join(lines)
        print(report)
        return report


# ======================================================================
# Analysis functions
# ======================================================================


def fidelity_comparison_chart(results: list[BackendComparison]) -> str:
    """Generate a formatted fidelity comparison table.

    Parameters
    ----------
    results : list[BackendComparison]
        Benchmark results.

    Returns
    -------
    str
        Formatted table string.
    """
    all_keys: set[str] = set()
    for comp in results:
        all_keys.update(comp.results.keys())
    sorted_keys = sorted(all_keys)

    header = f"{'Circuit':<20}"
    for key in sorted_keys:
        header += f" {key:>15}"
    lines = [header, "-" * len(header)]

    for comp in results:
        row = f"{comp.circuit_name:<20}"
        for key in sorted_keys:
            if key in comp.results:
                fid = comp.results[key].fidelity_vs_ideal
                row += f" {fid:>15.4f}"
            else:
                row += f" {'N/A':>15}"
        lines.append(row)

    return "\n".join(lines)


def gate_overhead_analysis(
    results: list[BackendComparison],
) -> dict[str, dict[str, int]]:
    """Analyse native gate overhead per backend across benchmarks.

    Returns
    -------
    dict
        ``{circuit_name: {backend_key: total_native_gates}}``.
    """
    analysis: dict[str, dict[str, int]] = {}
    for comp in results:
        circuit_data: dict[str, int] = {}
        for key, res in comp.results.items():
            circuit_data[key] = (
                res.num_gates_1q + res.num_gates_2q + res.num_gates_3q
            )
        analysis[comp.circuit_name] = circuit_data
    return analysis


def scaling_summary(
    scaling_results: dict[str, list[BackendComparison]],
) -> str:
    """Format a scaling analysis summary.

    Parameters
    ----------
    scaling_results : dict
        Output of ``run_scaling_analysis()``.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines: list[str] = []
    lines.append("Scaling Analysis Summary")
    lines.append("=" * 60)

    for n_str, comps in sorted(
        scaling_results.items(), key=lambda x: int(x[0])
    ):
        lines.append(f"\n  n_qubits = {n_str}")
        for comp in comps:
            for key, res in sorted(comp.results.items()):
                lines.append(
                    f"    {key:<20} fidelity={res.fidelity_vs_ideal:.4f} "
                    f"gates={res.num_gates_1q + res.num_gates_2q}"
                )

    return "\n".join(lines)


def toffoli_advantage_report(
    results: list[BackendComparison],
) -> str:
    """Generate a report highlighting the neutral-atom CCZ advantage.

    Parameters
    ----------
    results : list[BackendComparison]
        Should include a Toffoli benchmark.

    Returns
    -------
    str
        Formatted report string.
    """
    lines: list[str] = []
    lines.append("Toffoli / CCZ Advantage Report")
    lines.append("=" * 60)

    toffoli_comps = [
        c for c in results if "toffoli" in c.circuit_name.lower()
    ]

    if not toffoli_comps:
        lines.append("  No Toffoli benchmarks found in results.")
        return "\n".join(lines)

    for comp in toffoli_comps:
        lines.append(f"\n  Circuit: {comp.circuit_name}")
        lines.append(f"  Qubits:  {comp.num_qubits}")
        lines.append("")

        na_keys = [
            k for k in comp.results if "na_" in k or "neutral" in k
        ]
        other_keys = [k for k in comp.results if k not in na_keys]

        for key in sorted(comp.results.keys()):
            res = comp.results[key]
            total = (
                res.num_gates_1q + res.num_gates_2q + res.num_gates_3q
            )
            lines.append(
                f"    {key:<25} 1Q={res.num_gates_1q:>4} "
                f"2Q={res.num_gates_2q:>4} 3Q={res.num_gates_3q:>4} "
                f"total={total:>5}"
            )

        if na_keys and other_keys:
            na_ent = min(
                comp.results[k].num_gates_2q
                + comp.results[k].num_gates_3q
                for k in na_keys
            )
            other_ent = min(
                comp.results[k].num_gates_2q
                + comp.results[k].num_gates_3q
                for k in other_keys
            )
            if other_ent > 0:
                reduction = 1.0 - (na_ent / other_ent)
                lines.append(
                    f"\n    Entangling gate reduction (neutral atom): "
                    f"{reduction:.0%}"
                )

    return "\n".join(lines)


# ======================================================================
# Digital twin builder
# ======================================================================


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
        TransmonQubit,
        ChipTopology,
        NativeGateFamily,
    )

    n = len(frequencies_ghz)
    qubits = []
    for i in range(n):
        qubits.append(
            TransmonQubit(
                frequency_ghz=frequencies_ghz[i],
                t1_us=t1_us[i],
                t2_us=t2_us[i],
                readout_fidelity=readout_fidelities[i],
            )
        )

    couplings = {
        (min(a, b), max(a, b)): coupling_mhz for a, b in edges
    }
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
