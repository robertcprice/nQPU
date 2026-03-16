"""Quantum circuit representation and standard gate library.

Provides a lightweight, numpy-only circuit abstraction used throughout
the transpiler pipeline.  The design mirrors the Rust ``LogicalGate``
representation in ``sdk/rust/src/circuits/synthesis/transpiler.rs`` while
exposing a Pythonic interface suited for rapid prototyping.

Key classes
-----------
Gate
    A single quantum gate with a name, target qubits, and optional
    continuous parameters (rotation angles).
QuantumCircuit
    An ordered sequence of ``Gate`` objects with a fixed qubit count.
    Supports depth / gate-count queries and small-circuit unitary
    simulation via ``to_matrix()``.
CircuitStats
    Summary statistics (gate counts by type, depth, two-qubit gate count).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------
# Gate matrices (2x2 and 4x4)
# ------------------------------------------------------------------

_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_Sdg = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
_T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)
_Tdg = np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128)
_SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128) / 2


def _rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """General single-qubit unitary U3(theta, phi, lambda)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=np.complex128,
    )


# Two-qubit gate matrices (4x4, computational basis order |00>, |01>, |10>, |11>)
_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)
_CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)
_SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
)

# Three-qubit (8x8) Toffoli
_TOFFOLI = np.eye(8, dtype=np.complex128)
_TOFFOLI[6, 6] = 0
_TOFFOLI[7, 7] = 0
_TOFFOLI[6, 7] = 1
_TOFFOLI[7, 6] = 1


def _gate_matrix(name: str, params: Tuple[float, ...]) -> np.ndarray:
    """Return the unitary matrix for a named gate."""
    name_lower = name.lower()
    if name_lower == "h":
        return _H.copy()
    if name_lower == "x":
        return _X.copy()
    if name_lower == "y":
        return _Y.copy()
    if name_lower == "z":
        return _Z.copy()
    if name_lower == "s":
        return _S.copy()
    if name_lower == "sdg":
        return _Sdg.copy()
    if name_lower == "t":
        return _T.copy()
    if name_lower == "tdg":
        return _Tdg.copy()
    if name_lower == "sx":
        return _SX.copy()
    if name_lower == "rx":
        return _rx(params[0])
    if name_lower == "ry":
        return _ry(params[0])
    if name_lower == "rz":
        return _rz(params[0])
    if name_lower == "u3":
        return _u3(params[0], params[1], params[2])
    if name_lower in ("cx", "cnot"):
        return _CNOT.copy()
    if name_lower == "cz":
        return _CZ.copy()
    if name_lower == "swap":
        return _SWAP.copy()
    if name_lower in ("ccx", "toffoli"):
        return _TOFFOLI.copy()
    if name_lower == "id":
        return _I2.copy()
    raise ValueError(f"Unknown gate: {name}")


# ------------------------------------------------------------------
# Gate
# ------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """A single quantum gate instruction.

    Parameters
    ----------
    name : str
        Gate name (e.g. ``"H"``, ``"CNOT"``, ``"Rz"``).
    qubits : tuple[int, ...]
        Target qubit indices.
    params : tuple[float, ...]
        Continuous parameters (rotation angles).  Empty for fixed gates.
    """

    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()

    # -- convenience properties ------------------------------------------

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def is_single_qubit(self) -> bool:
        return len(self.qubits) == 1

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2

    @property
    def is_three_qubit(self) -> bool:
        return len(self.qubits) == 3

    @property
    def is_parametric(self) -> bool:
        return len(self.params) > 0

    def matrix(self) -> np.ndarray:
        """Return the unitary matrix for this gate."""
        return _gate_matrix(self.name, self.params)

    # -- inverse ---------------------------------------------------------

    def inverse(self) -> Gate:
        """Return the inverse (adjoint) gate."""
        name = self.name.lower()
        # Self-inverse gates
        if name in ("h", "x", "y", "z", "cx", "cnot", "cz", "swap", "ccx", "toffoli", "id"):
            return Gate(self.name, self.qubits, self.params)
        if name == "s":
            return Gate("Sdg", self.qubits)
        if name == "sdg":
            return Gate("S", self.qubits)
        if name == "t":
            return Gate("Tdg", self.qubits)
        if name == "tdg":
            return Gate("T", self.qubits)
        # Parametric inverses: negate angles
        if name in ("rx", "ry", "rz"):
            return Gate(self.name, self.qubits, (-self.params[0],))
        if name == "u3":
            # U3^dag = U3(-theta, -lambda, -phi)
            return Gate(
                "U3",
                self.qubits,
                (-self.params[0], -self.params[2], -self.params[1]),
            )
        raise ValueError(f"inverse not implemented for gate: {self.name}")

    def __repr__(self) -> str:
        q_str = ", ".join(str(q) for q in self.qubits)
        if self.params:
            p_str = ", ".join(f"{p:.4f}" for p in self.params)
            return f"{self.name}({p_str}) @ q[{q_str}]"
        return f"{self.name} @ q[{q_str}]"


# ------------------------------------------------------------------
# Standard gate constructors
# ------------------------------------------------------------------

def H(qubit: int) -> Gate:
    return Gate("H", (qubit,))


def X(qubit: int) -> Gate:
    return Gate("X", (qubit,))


def Y(qubit: int) -> Gate:
    return Gate("Y", (qubit,))


def Z(qubit: int) -> Gate:
    return Gate("Z", (qubit,))


def S(qubit: int) -> Gate:
    return Gate("S", (qubit,))


def Sdg(qubit: int) -> Gate:
    return Gate("Sdg", (qubit,))


def T(qubit: int) -> Gate:
    return Gate("T", (qubit,))


def Tdg(qubit: int) -> Gate:
    return Gate("Tdg", (qubit,))


def SX(qubit: int) -> Gate:
    return Gate("SX", (qubit,))


def Id(qubit: int) -> Gate:
    return Gate("Id", (qubit,))


def Rx(qubit: int, theta: float) -> Gate:
    return Gate("Rx", (qubit,), (theta,))


def Ry(qubit: int, theta: float) -> Gate:
    return Gate("Ry", (qubit,), (theta,))


def Rz(qubit: int, theta: float) -> Gate:
    return Gate("Rz", (qubit,), (theta,))


def U3(qubit: int, theta: float, phi: float, lam: float) -> Gate:
    return Gate("U3", (qubit,), (theta, phi, lam))


def CNOT(control: int, target: int) -> Gate:
    return Gate("CNOT", (control, target))


def CX(control: int, target: int) -> Gate:
    return Gate("CX", (control, target))


def CZ(control: int, target: int) -> Gate:
    return Gate("CZ", (control, target))


def SWAP(q0: int, q1: int) -> Gate:
    return Gate("SWAP", (q0, q1))


def Toffoli(q0: int, q1: int, q2: int) -> Gate:
    return Gate("CCX", (q0, q1, q2))


def CCX(q0: int, q1: int, q2: int) -> Gate:
    return Gate("CCX", (q0, q1, q2))


# ------------------------------------------------------------------
# CircuitStats
# ------------------------------------------------------------------

@dataclass
class CircuitStats:
    """Summary statistics for a quantum circuit."""

    num_qubits: int = 0
    depth: int = 0
    gate_count: int = 0
    ops_count: Dict[str, int] = field(default_factory=dict)
    two_qubit_gate_count: int = 0
    three_qubit_gate_count: int = 0
    single_qubit_gate_count: int = 0


# ------------------------------------------------------------------
# QuantumCircuit
# ------------------------------------------------------------------

class QuantumCircuit:
    """Ordered list of gates acting on a fixed number of qubits.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    """

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = num_qubits
        self.gates: List[Gate] = []

    # -- mutation --------------------------------------------------------

    def add_gate(self, gate: Gate) -> "QuantumCircuit":
        """Append a gate.  Returns ``self`` for chaining."""
        for q in gate.qubits:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(
                    f"Qubit index {q} out of range for {self.num_qubits}-qubit circuit"
                )
        self.gates.append(gate)
        return self

    # convenience shortcuts
    def h(self, q: int) -> "QuantumCircuit":
        return self.add_gate(H(q))

    def x(self, q: int) -> "QuantumCircuit":
        return self.add_gate(X(q))

    def y(self, q: int) -> "QuantumCircuit":
        return self.add_gate(Y(q))

    def z(self, q: int) -> "QuantumCircuit":
        return self.add_gate(Z(q))

    def s(self, q: int) -> "QuantumCircuit":
        return self.add_gate(S(q))

    def sdg(self, q: int) -> "QuantumCircuit":
        return self.add_gate(Sdg(q))

    def t(self, q: int) -> "QuantumCircuit":
        return self.add_gate(T(q))

    def tdg(self, q: int) -> "QuantumCircuit":
        return self.add_gate(Tdg(q))

    def sx(self, q: int) -> "QuantumCircuit":
        return self.add_gate(SX(q))

    def rx(self, q: int, theta: float) -> "QuantumCircuit":
        return self.add_gate(Rx(q, theta))

    def ry(self, q: int, theta: float) -> "QuantumCircuit":
        return self.add_gate(Ry(q, theta))

    def rz(self, q: int, theta: float) -> "QuantumCircuit":
        return self.add_gate(Rz(q, theta))

    def u3(self, q: int, theta: float, phi: float, lam: float) -> "QuantumCircuit":
        return self.add_gate(U3(q, theta, phi, lam))

    def cx(self, ctrl: int, tgt: int) -> "QuantumCircuit":
        return self.add_gate(CX(ctrl, tgt))

    def cnot(self, ctrl: int, tgt: int) -> "QuantumCircuit":
        return self.add_gate(CNOT(ctrl, tgt))

    def cz(self, q0: int, q1: int) -> "QuantumCircuit":
        return self.add_gate(CZ(q0, q1))

    def swap(self, q0: int, q1: int) -> "QuantumCircuit":
        return self.add_gate(SWAP(q0, q1))

    def ccx(self, q0: int, q1: int, q2: int) -> "QuantumCircuit":
        return self.add_gate(CCX(q0, q1, q2))

    def toffoli(self, q0: int, q1: int, q2: int) -> "QuantumCircuit":
        return self.add_gate(Toffoli(q0, q1, q2))

    # -- queries ---------------------------------------------------------

    def gate_count(self) -> int:
        """Total number of gates."""
        return len(self.gates)

    def depth(self) -> int:
        """Circuit depth (longest path through qubit wires).

        A greedy layer-assignment: each gate is placed at the earliest
        layer where all its qubits are free.
        """
        if not self.gates:
            return 0
        qubit_depth = [0] * self.num_qubits
        for gate in self.gates:
            layer = max(qubit_depth[q] for q in gate.qubits) + 1
            for q in gate.qubits:
                qubit_depth[q] = layer
        return max(qubit_depth)

    def count_ops(self) -> Dict[str, int]:
        """Gate counts keyed by gate name."""
        counts: Dict[str, int] = {}
        for gate in self.gates:
            counts[gate.name] = counts.get(gate.name, 0) + 1
        return counts

    def stats(self) -> CircuitStats:
        """Compute comprehensive circuit statistics."""
        ops = self.count_ops()
        single = sum(1 for g in self.gates if g.is_single_qubit)
        two = sum(1 for g in self.gates if g.is_two_qubit)
        three = sum(1 for g in self.gates if g.is_three_qubit)
        return CircuitStats(
            num_qubits=self.num_qubits,
            depth=self.depth(),
            gate_count=len(self.gates),
            ops_count=ops,
            two_qubit_gate_count=two,
            three_qubit_gate_count=three,
            single_qubit_gate_count=single,
        )

    # -- unitary simulation (small circuits only) -----------------------

    def to_matrix(self, validate: bool = True) -> np.ndarray:
        """Compute the full unitary matrix of the circuit.

        Only practical for circuits with <= ~10 qubits due to exponential
        matrix size.

        Parameters
        ----------
        validate : bool
            If *True* (default), raise for circuits larger than 12 qubits.
        """
        n = self.num_qubits
        if validate and n > 12:
            raise ValueError(
                f"to_matrix() is impractical for {n}-qubit circuits (limit 12)"
            )

        dim = 1 << n
        result = np.eye(dim, dtype=np.complex128)

        for gate in self.gates:
            mat = gate.matrix()
            full = _embed_gate(mat, gate.qubits, n)
            result = full @ result

        return result

    # -- copy / reverse --------------------------------------------------

    def copy(self) -> "QuantumCircuit":
        """Deep copy of the circuit."""
        qc = QuantumCircuit(self.num_qubits)
        qc.gates = list(self.gates)
        return qc

    def inverse(self) -> "QuantumCircuit":
        """Return the inverse circuit (gates in reverse order, each inverted)."""
        qc = QuantumCircuit(self.num_qubits)
        for gate in reversed(self.gates):
            qc.add_gate(gate.inverse())
        return qc

    def __len__(self) -> int:
        return len(self.gates)

    def __iter__(self):
        return iter(self.gates)

    def __repr__(self) -> str:
        return f"QuantumCircuit(num_qubits={self.num_qubits}, gates={len(self.gates)})"


# ------------------------------------------------------------------
# Unitary embedding helper
# ------------------------------------------------------------------

def _embed_gate(
    mat: np.ndarray,
    qubits: Tuple[int, ...],
    num_qubits: int,
) -> np.ndarray:
    """Embed a gate matrix into the full 2^n Hilbert space.

    Uses the tensor-product / permutation approach: reorder the
    computational-basis indices so that the target qubits are in the
    least-significant positions, apply the gate, then reorder back.
    """
    n = num_qubits
    dim = 1 << n
    nq = len(qubits)

    # Build the full unitary by explicit index manipulation.
    full = np.zeros((dim, dim), dtype=np.complex128)
    gate_dim = 1 << nq

    for row in range(dim):
        for col in range(dim):
            # Extract the bits corresponding to the target qubits
            row_target = 0
            col_target = 0
            for k, q in enumerate(qubits):
                bit_pos = n - 1 - q  # big-endian convention
                row_target |= ((row >> bit_pos) & 1) << (nq - 1 - k)
                col_target |= ((col >> bit_pos) & 1) << (nq - 1 - k)

            # The non-target bits must be identical for a non-zero entry
            row_other = row
            col_other = col
            for q in qubits:
                bit_pos = n - 1 - q
                mask = ~(1 << bit_pos)
                row_other &= mask
                col_other &= mask
            if row_other != col_other:
                continue

            full[row, col] += mat[row_target, col_target]

    return full
