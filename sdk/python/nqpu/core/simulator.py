"""Stable gate-based simulator API with optional Rust acceleration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from nqpu._compat import get_rust_bindings


class Backend(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class GateOperation:
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


@dataclass
class SimulationResult:
    counts: dict[str, int]
    shots: int
    backend: str
    engine: str


Result = SimulationResult


class QuantumCircuit:
    """Small, stable circuit builder used by the public Python API."""

    def __init__(self, num_qubits: int):
        if num_qubits < 1:
            raise ValueError("num_qubits must be positive")
        self.num_qubits = num_qubits
        self._gates: list[GateOperation] = []
        self._measure_all = False

    @property
    def gates(self) -> tuple[GateOperation, ...]:
        return tuple(self._gates)

    def _append(self, name: str, *qubits: int, params: tuple[float, ...] = ()) -> None:
        for qubit in qubits:
            if qubit < 0 or qubit >= self.num_qubits:
                raise ValueError(f"qubit index {qubit} out of range for {self.num_qubits} qubits")
        self._gates.append(GateOperation(name=name, qubits=tuple(qubits), params=params))

    def h(self, qubit: int) -> None:
        self._append("h", qubit)

    def x(self, qubit: int) -> None:
        self._append("x", qubit)

    def y(self, qubit: int) -> None:
        self._append("y", qubit)

    def z(self, qubit: int) -> None:
        self._append("z", qubit)

    def s(self, qubit: int) -> None:
        self._append("s", qubit)

    def t(self, qubit: int) -> None:
        self._append("t", qubit)

    def rx(self, qubit: int, theta: float) -> None:
        self._append("rx", qubit, params=(float(theta),))

    def ry(self, qubit: int, theta: float) -> None:
        self._append("ry", qubit, params=(float(theta),))

    def rz(self, qubit: int, theta: float) -> None:
        self._append("rz", qubit, params=(float(theta),))

    def cx(self, control: int, target: int) -> None:
        if control == target:
            raise ValueError("control and target must be different")
        self._append("cx", control, target)

    def cnot(self, control: int, target: int) -> None:
        self.cx(control, target)

    def cz(self, control: int, target: int) -> None:
        if control == target:
            raise ValueError("control and target must be different")
        self._append("cz", control, target)

    def swap(self, a: int, b: int) -> None:
        if a == b:
            raise ValueError("swap targets must be different")
        self._append("swap", a, b)

    def measure_all(self) -> "QuantumCircuit":
        self._measure_all = True
        return self

    def __repr__(self) -> str:
        return f"<QuantumCircuit {self.num_qubits} qubits, {len(self._gates)} gates>"


class NQPUBackend:
    """Backend facade that prefers the Rust extension and falls back to NumPy."""

    def __init__(
        self,
        backend: Backend | str = Backend.AUTO,
        *,
        gpu: bool = False,
        seed: int | None = None,
    ) -> None:
        if isinstance(backend, str):
            backend = Backend(backend)
        if gpu and backend is Backend.AUTO:
            backend = Backend.GPU
        self.backend = backend
        self.gpu = gpu or backend is Backend.GPU
        self._rng = np.random.default_rng(seed)
        self._bindings = get_rust_bindings()

    @property
    def uses_rust(self) -> bool:
        return self._bindings is not None

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> SimulationResult:
        if shots < 1:
            raise ValueError("shots must be positive")
        if self._bindings is not None:
            try:
                return self._run_with_rust(circuit, shots)
            except Exception:
                pass
        return self._run_with_numpy(circuit, shots)

    def _run_with_rust(self, circuit: QuantumCircuit, shots: int) -> SimulationResult:
        assert self._bindings is not None
        bindings = self._bindings
        rust_circuit = bindings.QuantumCircuit(circuit.num_qubits)

        for gate in circuit.gates:
            name = gate.name
            if name == "h":
                rust_circuit.h(gate.qubits[0])
            elif name == "x":
                rust_circuit.x(gate.qubits[0])
            elif name == "y":
                rust_circuit.y(gate.qubits[0])
            elif name == "z":
                rust_circuit.z(gate.qubits[0])
            elif name == "s":
                rust_circuit.s(gate.qubits[0])
            elif name == "t":
                rust_circuit.t(gate.qubits[0])
            elif name == "rx":
                rust_circuit.rx(gate.qubits[0], gate.params[0])
            elif name == "ry":
                rust_circuit.ry(gate.qubits[0], gate.params[0])
            elif name == "rz":
                rust_circuit.rz(gate.qubits[0], gate.params[0])
            elif name == "cx":
                rust_circuit.cx(gate.qubits[0], gate.qubits[1])
            elif name == "cz":
                rust_circuit.cz(gate.qubits[0], gate.qubits[1])
            elif name == "swap":
                rust_circuit.swap(gate.qubits[0], gate.qubits[1])

        simulator: Any
        if hasattr(bindings, "Backend") and self.backend is not Backend.AUTO:
            rust_backend = getattr(bindings.Backend, self.backend.name, None)
            simulator = bindings.Simulator(rust_backend) if rust_backend is not None else bindings.Simulator()
        else:
            simulator = bindings.Simulator()

        raw_result = simulator.run(rust_circuit, shots)
        counts = raw_result.counts if hasattr(raw_result, "counts") else raw_result.results()
        return SimulationResult(
            counts=dict(counts),
            shots=shots,
            backend=self.backend.value,
            engine="rust",
        )

    def _run_with_numpy(self, circuit: QuantumCircuit, shots: int) -> SimulationResult:
        state = np.zeros(1 << circuit.num_qubits, dtype=np.complex128)
        state[0] = 1.0

        for gate in circuit.gates:
            if len(gate.qubits) == 1:
                state = _apply_single_qubit_gate(state, circuit.num_qubits, gate)
            else:
                state = _apply_two_qubit_gate(state, gate)

        probabilities = np.abs(state) ** 2
        probabilities = probabilities / probabilities.sum()
        outcomes = self._rng.choice(len(state), size=shots, p=probabilities)
        counts = Counter(format(index, f"0{circuit.num_qubits}b") for index in outcomes)
        return SimulationResult(
            counts=dict(sorted(counts.items())),
            shots=shots,
            backend=self.backend.value,
            engine="numpy",
        )


QuantumBackend = NQPUBackend


def _apply_single_qubit_gate(state: np.ndarray, num_qubits: int, gate: GateOperation) -> np.ndarray:
    del num_qubits
    matrix = _single_qubit_matrix(gate.name, gate.params)
    target = gate.qubits[0]
    updated = state.copy()
    stride = 1 << target
    period = stride << 1
    for base in range(0, len(state), period):
        for offset in range(stride):
            i0 = base + offset
            i1 = i0 + stride
            a0 = state[i0]
            a1 = state[i1]
            updated[i0] = matrix[0, 0] * a0 + matrix[0, 1] * a1
            updated[i1] = matrix[1, 0] * a0 + matrix[1, 1] * a1
    return updated


def _apply_two_qubit_gate(state: np.ndarray, gate: GateOperation) -> np.ndarray:
    updated = state.copy()
    q0, q1 = gate.qubits
    mask0 = 1 << q0
    mask1 = 1 << q1

    if gate.name == "cx":
        control_mask, target_mask = mask0, mask1
        for index in range(len(state)):
            if (index & control_mask) and not (index & target_mask):
                partner = index | target_mask
                updated[index], updated[partner] = updated[partner], updated[index]
        return updated

    if gate.name == "cz":
        for index in range(len(state)):
            if (index & mask0) and (index & mask1):
                updated[index] *= -1.0
        return updated

    if gate.name == "swap":
        for index in range(len(state)):
            bit0 = bool(index & mask0)
            bit1 = bool(index & mask1)
            if not bit0 and bit1:
                partner = index ^ (mask0 | mask1)
                updated[index], updated[partner] = updated[partner], updated[index]
        return updated

    raise ValueError(f"unsupported two-qubit gate: {gate.name}")


def _single_qubit_matrix(name: str, params: tuple[float, ...]) -> np.ndarray:
    if name == "h":
        return (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    if name == "x":
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    if name == "y":
        return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    if name == "z":
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    if name == "s":
        return np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    if name == "t":
        return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]], dtype=np.complex128)
    if name == "rx":
        theta = params[0]
        c = np.cos(theta / 2.0)
        s = -1.0j * np.sin(theta / 2.0)
        return np.array([[c, s], [s, c]], dtype=np.complex128)
    if name == "ry":
        theta = params[0]
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if name == "rz":
        theta = params[0]
        phase = np.exp(0.5j * theta)
        return np.array([[np.conjugate(phase), 0.0], [0.0, phase]], dtype=np.complex128)
    raise ValueError(f"unsupported single-qubit gate: {name}")
