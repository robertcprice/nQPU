"""QPU emulator: unified interface to all hardware backends.

The :class:`QPU` class is the primary entry point.  It wraps the nQPU
hardware backend simulators (trapped-ion, superconducting, neutral-atom)
behind a single interface driven by real hardware profiles.

Typical usage::

    from nqpu.emulator import QPU, HardwareProfile

    qpu = QPU(HardwareProfile.IONQ_ARIA)
    job = qpu.run([("h", 0), ("cx", 0, 1)], shots=1000)
    print(job.result.counts)
"""

from __future__ import annotations

import math
import uuid
from typing import Optional

import numpy as np

from .hardware import HardwareFamily, HardwareProfile, HardwareSpec
from .job import Counts, EmulatorResult, Job

# Gate sets used for classification in fidelity / runtime estimation.
_1Q_GATES = frozenset(
    ("h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz")
)
_2Q_GATES = frozenset(("cx", "cnot", "cz", "swap"))
_3Q_GATES = frozenset(("ccx", "toffoli", "ccz"))


class QPU:
    """Emulated quantum processing unit.

    Wraps the nQPU hardware backend simulators (trapped-ion, superconducting,
    neutral-atom) behind a unified interface driven by real hardware profiles.

    Parameters
    ----------
    profile : HardwareProfile
        Hardware profile to emulate.
    noise : bool
        Whether to apply the hardware noise model.  Default ``True``.
    seed : int, optional
        Random seed for measurement sampling.
    max_qubits : int, optional
        Override the maximum qubit count (useful for smaller test
        simulations on profiles with large qubit counts).
    """

    def __init__(
        self,
        profile: HardwareProfile,
        noise: bool = True,
        seed: int | None = None,
        max_qubits: int | None = None,
    ) -> None:
        self.profile = profile
        self.spec: HardwareSpec = (
            profile.spec if isinstance(profile, HardwareProfile) else profile
        )
        self.noise = noise
        self.seed = seed
        self._max_qubits = max_qubits
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------ #
    #  Public properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        """Human-readable device name."""
        return self.spec.name

    @property
    def num_qubits(self) -> int:
        """Maximum qubit count available on this emulated QPU."""
        if self._max_qubits is not None:
            return min(self._max_qubits, self.spec.num_qubits)
        return self.spec.num_qubits

    # ------------------------------------------------------------------ #
    #  Public methods                                                      #
    # ------------------------------------------------------------------ #

    def run(self, circuit, shots: int = 1024) -> Job:
        """Run a circuit on the emulated QPU.

        Parameters
        ----------
        circuit : QuantumCircuit or list[tuple]
            Circuit to execute.  Accepts either a
            :class:`nqpu.transpiler.QuantumCircuit` or a raw gate list of
            ``(gate_name, qubit, ..., [params])`` tuples.
        shots : int
            Number of measurement shots.  Use ``0`` for statevector mode.

        Returns
        -------
        Job
            Completed job containing an :class:`EmulatorResult` on success,
            or an error message on failure.
        """
        gate_list = self._to_gate_list(circuit)
        n_qubits = self._infer_qubits(gate_list)

        if n_qubits > self.num_qubits:
            return Job(
                job_id=str(uuid.uuid4()),
                status="failed",
                error=(
                    f"Circuit requires {n_qubits} qubits but {self.spec.name} "
                    f"supports max {self.num_qubits}"
                ),
            )

        mode = "noisy" if self.noise else "ideal"

        try:
            sim = self._create_simulator(n_qubits, mode)
            self._execute_gates(sim, gate_list)

            fidelity = self._estimate_fidelity(gate_list, n_qubits)
            depth = self._circuit_depth(gate_list, n_qubits)
            native_count = self._count_native_gates(gate_list)
            runtime = self._estimate_runtime(gate_list)

            if shots == 0:
                sv = self._get_statevector(sim)
                result = EmulatorResult(
                    counts=Counts(),
                    statevector=sv,
                    fidelity_estimate=fidelity,
                    circuit_depth=depth,
                    native_gate_count=native_count,
                    estimated_runtime_us=runtime,
                    hardware_profile=self.spec.name,
                )
            else:
                probs = self._get_probabilities(sim)
                counts = self._sample_counts(probs, n_qubits, shots)
                result = EmulatorResult(
                    counts=counts,
                    fidelity_estimate=fidelity,
                    circuit_depth=depth,
                    native_gate_count=native_count,
                    estimated_runtime_us=runtime,
                    hardware_profile=self.spec.name,
                )

            return Job(
                job_id=str(uuid.uuid4()),
                status="completed",
                result=result,
            )
        except Exception as exc:
            return Job(
                job_id=str(uuid.uuid4()),
                status="failed",
                error=str(exc),
            )

    def info(self) -> dict:
        """Return hardware specification summary as a plain dictionary."""
        s = self.spec
        return {
            "name": s.name,
            "family": s.family.value,
            "num_qubits": s.num_qubits,
            "connectivity": s.connectivity,
            "1q_fidelity": s.single_qubit_fidelity,
            "2q_fidelity": s.two_qubit_fidelity,
            "readout_fidelity": s.readout_fidelity,
            "native_2q_gate": s.native_2q_gate,
            "native_3q_gate": s.native_3q_gate,
            "t1_us": s.t1_us,
            "t2_us": s.t2_us,
            "max_circuit_depth": s.max_circuit_depth,
        }

    @classmethod
    def compare(
        cls,
        circuit,
        profiles: list[HardwareProfile] | None = None,
        shots: int = 1024,
        seed: int = 42,
    ) -> dict[str, EmulatorResult]:
        """Compare a circuit across multiple hardware profiles.

        Parameters
        ----------
        circuit
            Circuit to compare (same types accepted by :meth:`run`).
        profiles : list of HardwareProfile, optional
            Profiles to compare.  Defaults to one representative from
            each hardware family (IonQ Aria, IBM Heron, QuEra Aquila).
        shots : int
            Shots per run.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict[str, EmulatorResult]
            Mapping from profile name to :class:`EmulatorResult`.
            Only includes profiles whose jobs completed successfully.
        """
        if profiles is None:
            profiles = [
                HardwareProfile.IONQ_ARIA,
                HardwareProfile.IBM_HERON,
                HardwareProfile.QUERA_AQUILA,
            ]
        results: dict[str, EmulatorResult] = {}
        for profile in profiles:
            qpu = cls(profile, noise=True, seed=seed)
            job = qpu.run(circuit, shots=shots)
            if job.successful():
                assert job.result is not None
                results[profile.spec.name] = job.result
        return results

    # ------------------------------------------------------------------ #
    #  Backend dispatch                                                    #
    # ------------------------------------------------------------------ #

    def _create_simulator(self, n_qubits: int, mode: str):
        """Create a backend simulator based on hardware family."""
        family = self.spec.family

        if family == HardwareFamily.TRAPPED_ION:
            from nqpu.ion_trap import IonSpecies, TrapConfig, TrappedIonSimulator

            config = TrapConfig(n_ions=n_qubits, species=IonSpecies.YB171)
            return TrappedIonSimulator(config, execution_mode=mode)

        if family == HardwareFamily.SUPERCONDUCTING:
            from nqpu.superconducting import DevicePresets, TransmonSimulator

            preset = self._pick_sc_preset()
            config = preset.build(num_qubits=n_qubits)
            return TransmonSimulator(config, execution_mode=mode)

        if family == HardwareFamily.NEUTRAL_ATOM:
            from nqpu.neutral_atom import (
                ArrayConfig,
                AtomSpecies,
                NeutralAtomSimulator,
            )

            config = ArrayConfig(n_atoms=n_qubits, species=AtomSpecies.RB87)
            return NeutralAtomSimulator(config, execution_mode=mode)

        raise ValueError(f"Unsupported hardware family: {family}")

    def _pick_sc_preset(self):
        """Map the current hardware profile to a superconducting DevicePreset."""
        from nqpu.superconducting import DevicePresets

        mapping = {
            "IBM Eagle (127Q)": DevicePresets.IBM_EAGLE,
            "IBM Heron (133Q)": DevicePresets.IBM_HERON,
            "Google Sycamore (72Q)": DevicePresets.GOOGLE_SYCAMORE,
            "Rigetti Ankaa-2 (84Q)": DevicePresets.RIGETTI_ANKAA,
        }
        return mapping.get(self.spec.name, DevicePresets.IBM_HERON)

    # ------------------------------------------------------------------ #
    #  Gate execution                                                      #
    # ------------------------------------------------------------------ #

    def _execute_gates(self, sim, gate_list: list) -> None:
        """Execute the full gate list on the simulator."""
        backend = self.spec.family.value
        sim.reset()
        for gate in gate_list:
            self._apply_gate(sim, gate, backend)

    def _apply_gate(self, sim, gate: tuple, backend: str) -> None:
        """Apply a single gate to the simulator.

        Dispatches standard gate names to the simulator's native methods.
        For gates that require decomposition (SWAP, Toffoli, CCZ), the
        decomposition is performed inline.
        """
        name = str(gate[0]).lower()

        # -- single-qubit gates --
        if name == "h":
            sim.h(int(gate[1]))
        elif name == "x":
            sim.x(int(gate[1]))
        elif name == "y":
            sim.y(int(gate[1]))
        elif name == "z":
            sim.z(int(gate[1]))
        elif name in ("s", "sdg"):
            angle = math.pi / 2 if name == "s" else -math.pi / 2
            sim.rz(int(gate[1]), angle)
        elif name in ("t", "tdg"):
            angle = math.pi / 4 if name == "t" else -math.pi / 4
            sim.rz(int(gate[1]), angle)
        elif name == "sx":
            sim.rx(int(gate[1]), math.pi / 2)
        elif name == "rx":
            sim.rx(int(gate[1]), float(gate[2]))
        elif name == "ry":
            sim.ry(int(gate[1]), float(gate[2]))
        elif name == "rz":
            sim.rz(int(gate[1]), float(gate[2]))

        # -- two-qubit gates --
        elif name in ("cx", "cnot"):
            sim.cnot(int(gate[1]), int(gate[2]))
        elif name == "cz":
            sim.cz(int(gate[1]), int(gate[2]))
        elif name == "swap":
            q0, q1 = int(gate[1]), int(gate[2])
            sim.cnot(q0, q1)
            sim.cnot(q1, q0)
            sim.cnot(q0, q1)

        # -- three-qubit gates --
        elif name in ("ccx", "toffoli"):
            q0, q1, q2 = int(gate[1]), int(gate[2]), int(gate[3])
            if backend == "neutral_atom" and hasattr(sim, "toffoli"):
                sim.toffoli(q0, q1, q2)
            else:
                self._decompose_toffoli(sim, q0, q1, q2)
        elif name == "ccz":
            q0, q1, q2 = int(gate[1]), int(gate[2]), int(gate[3])
            if backend == "neutral_atom" and hasattr(sim, "ccz"):
                sim.ccz(q0, q1, q2)
            else:
                sim.h(q2)
                self._decompose_toffoli(sim, q0, q1, q2)
                sim.h(q2)
        else:
            raise ValueError(f"Unsupported gate: {name}")

    @staticmethod
    def _decompose_toffoli(sim, q0: int, q1: int, q2: int) -> None:
        """Standard Toffoli decomposition into 1Q + CNOT gates."""
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

    # ------------------------------------------------------------------ #
    #  State extraction                                                    #
    # ------------------------------------------------------------------ #

    def _get_probabilities(self, sim) -> np.ndarray:
        """Get measurement probabilities from the simulator state.

        Tries ``density_matrix`` (for noisy mode), ``probabilities``,
        then falls back to computing from the statevector.
        """
        if self.noise and hasattr(sim, "density_matrix"):
            try:
                dm = sim.density_matrix()
                probs = np.real(np.diag(dm))
                probs = np.maximum(probs, 0.0)
                total = np.sum(probs)
                if total > 0 and abs(total - 1.0) > 1e-10:
                    probs /= total
                return probs
            except (AttributeError, Exception):
                pass

        if hasattr(sim, "probabilities"):
            return sim.probabilities()

        sv = sim.statevector()
        return np.abs(sv) ** 2

    @staticmethod
    def _get_statevector(sim) -> np.ndarray:
        """Get the statevector from the simulator."""
        return sim.statevector()

    def _sample_counts(
        self, probs: np.ndarray, n_qubits: int, shots: int
    ) -> Counts:
        """Sample measurement outcomes from a probability distribution.

        Applies readout error when noise is enabled by mixing the ideal
        distribution with a uniform distribution weighted by the readout
        error rate.
        """
        if self.noise:
            err = self.spec.error_per_readout
            if err > 0:
                probs = probs * (1 - err) + err / len(probs)
                probs = probs / probs.sum()

        indices = self._rng.choice(len(probs), size=shots, p=probs)
        counts = Counts()
        for idx in indices:
            bitstring = format(idx, f"0{n_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    #  Analysis / estimation helpers                                       #
    # ------------------------------------------------------------------ #

    def _estimate_fidelity(self, gate_list: list, n_qubits: int) -> float:
        """Estimate overall circuit fidelity from gate error rates.

        Uses the multiplicative independent-error model:
        ``F = F_1q^n_1q * F_2q^n_2q * F_ro^n_qubits``.  For non-neutral-atom
        backends, Toffoli gates are accounted for via their CNOT decomposition
        (6 CNOTs per Toffoli).
        """
        if not self.noise:
            return 1.0

        n_1q = sum(1 for g in gate_list if str(g[0]).lower() in _1Q_GATES)
        n_2q = sum(1 for g in gate_list if str(g[0]).lower() in _2Q_GATES)
        n_3q = sum(1 for g in gate_list if str(g[0]).lower() in _3Q_GATES)

        # Non-neutral-atom backends decompose Toffoli into ~6 CNOTs
        if self.spec.family != HardwareFamily.NEUTRAL_ATOM:
            n_2q += n_3q * 6
            n_3q = 0

        fidelity = (
            self.spec.single_qubit_fidelity ** n_1q
            * self.spec.two_qubit_fidelity ** n_2q
            * self.spec.readout_fidelity ** n_qubits
        )
        return fidelity

    def _estimate_runtime(self, gate_list: list) -> float:
        """Estimate hardware wall-clock runtime in microseconds.

        Simple additive model: sum of individual gate durations plus one
        readout at the end.
        """
        s = self.spec
        n_1q = sum(1 for g in gate_list if str(g[0]).lower() in _1Q_GATES)
        n_2q = sum(1 for g in gate_list if str(g[0]).lower() in _2Q_GATES)
        return n_1q * s.single_qubit_gate_us + n_2q * s.two_qubit_gate_us + s.readout_us

    @staticmethod
    def _circuit_depth(gate_list: list, n_qubits: int) -> int:
        """Compute circuit depth using a per-qubit layer tracker."""
        if n_qubits == 0:
            return 0

        qubit_layer = [0] * n_qubits

        for gate in gate_list:
            name = str(gate[0]).lower()
            if name in _1Q_GATES:
                q = int(gate[1])
                if q < n_qubits:
                    qubit_layer[q] += 1
            elif name in _2Q_GATES:
                q0, q1 = int(gate[1]), int(gate[2])
                if q0 < n_qubits and q1 < n_qubits:
                    layer = max(qubit_layer[q0], qubit_layer[q1]) + 1
                    qubit_layer[q0] = layer
                    qubit_layer[q1] = layer
            elif name in _3Q_GATES:
                q0, q1, q2 = int(gate[1]), int(gate[2]), int(gate[3])
                if q0 < n_qubits and q1 < n_qubits and q2 < n_qubits:
                    layer = max(
                        qubit_layer[q0], qubit_layer[q1], qubit_layer[q2]
                    ) + 1
                    qubit_layer[q0] = layer
                    qubit_layer[q1] = layer
                    qubit_layer[q2] = layer

        return max(qubit_layer) if qubit_layer else 0

    @staticmethod
    def _count_native_gates(gate_list: list) -> int:
        """Count total native gates after decomposition.

        SWAP decomposes to 3 CNOTs.  Toffoli on non-neutral-atom backends
        decomposes to ~15 primitive gates.  On neutral-atom backends with
        native CCZ / Toffoli, the 3Q gate counts as 1.
        """
        count = 0
        for gate in gate_list:
            name = str(gate[0]).lower()
            if name in _1Q_GATES:
                count += 1
            elif name in ("cx", "cnot", "cz"):
                count += 1
            elif name == "swap":
                count += 3
            elif name in _3Q_GATES:
                # Neutral-atom can execute natively; others decompose.
                # Conservative count: use 15 for decomposed Toffoli.
                count += 15
        return count

    # ------------------------------------------------------------------ #
    #  Circuit conversion                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_gate_list(circuit) -> list:
        """Convert a QuantumCircuit or raw gate list to gate tuples.

        Accepts:
        - ``list[tuple]``: returned as-is.
        - Object with a ``.gates`` attribute (e.g.
          :class:`nqpu.transpiler.QuantumCircuit`): each gate is converted
          to ``(name, *qubits, *params)``.

        Raises
        ------
        TypeError
            If the circuit type is not recognized.
        """
        if isinstance(circuit, list):
            return circuit
        # Support nqpu.transpiler.QuantumCircuit
        if hasattr(circuit, "gates"):
            return [
                (g.name.lower(), *g.qubits, *g.params) for g in circuit.gates
            ]
        raise TypeError(
            f"Unsupported circuit type: {type(circuit).__name__}. "
            "Pass a list of gate tuples or a nqpu.transpiler.QuantumCircuit."
        )

    @staticmethod
    def _infer_qubits(gate_list: list) -> int:
        """Infer the number of qubits from gate indices.

        Scans all gates for the highest qubit index and returns
        ``max_index + 1``.
        """
        max_q = -1
        for gate in gate_list:
            for i in range(1, len(gate)):
                try:
                    q = int(gate[i])
                    max_q = max(max_q, q)
                except (ValueError, TypeError):
                    break
        return max_q + 1 if max_q >= 0 else 0

    # ------------------------------------------------------------------ #
    #  Dunder methods                                                      #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"QPU({self.spec.name!r}, noise={self.noise})"
