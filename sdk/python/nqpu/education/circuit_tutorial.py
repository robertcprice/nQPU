"""Interactive quantum circuit tutorials.

Gate-by-gate quantum circuit builder with matrix representations, Bloch sphere
descriptions, measurement probabilities, and step-by-step algorithm walkthroughs.

All simulation is pure numpy statevector manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LessonStep:
    """A single step in a circuit lesson."""

    instruction: str
    hint: str
    gate_sequence: List[Tuple]  # [(gate_name, qubits, params), ...]
    explanation: str
    state_after: Optional[np.ndarray] = None


@dataclass
class CircuitLesson:
    """A complete circuit lesson with multiple steps."""

    title: str
    description: str
    n_qubits: int
    steps: List[LessonStep]

    def run(self) -> "TutorialResult":
        """Run lesson, applying each step's gates and showing results."""
        dim = 2 ** self.n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        step_results: list = []
        for i, step in enumerate(self.steps):
            state_before = state.copy()
            for gate_name, qubits, params in step.gate_sequence:
                state = _apply_gate(state, self.n_qubits, gate_name, qubits, params)
            probs = np.abs(state) ** 2
            step.state_after = state.copy()
            step_results.append({
                "step": i + 1,
                "instruction": step.instruction,
                "state_before": state_before,
                "state_after": state.copy(),
                "probabilities": probs,
                "explanation": step.explanation,
            })
        return TutorialResult(
            lesson_title=self.title,
            steps_completed=len(self.steps),
            total_steps=len(self.steps),
            final_state=state.copy(),
            step_results=step_results,
        )


@dataclass
class TutorialResult:
    """Result of running a tutorial."""

    lesson_title: str
    steps_completed: int
    total_steps: int
    final_state: np.ndarray
    step_results: list = field(default_factory=list)

    def summary(self) -> str:
        """ASCII summary of the tutorial progress."""
        lines = [
            f"=== {self.lesson_title} ===",
            f"Steps completed: {self.steps_completed}/{self.total_steps}",
            "",
        ]
        for sr in self.step_results:
            lines.append(f"Step {sr['step']}: {sr['instruction']}")
            probs = sr["probabilities"]
            n_qubits = int(np.log2(len(probs)))
            for idx, p in enumerate(probs):
                if p > 1e-10:
                    label = format(idx, f"0{n_qubits}b")
                    lines.append(f"  |{label}> : {p:.4f}")
            lines.append(f"  -> {sr['explanation']}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate library helpers
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)
_CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1],
], dtype=complex)
_SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=complex)
_TOFFOLI = np.eye(8, dtype=complex)
_TOFFOLI[6, 6] = 0
_TOFFOLI[7, 7] = 0
_TOFFOLI[6, 7] = 1
_TOFFOLI[7, 6] = 1


def _rx(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)],
    ], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)],
    ], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)],
    ], dtype=complex)


_GATE_INFO: Dict[str, Tuple[np.ndarray, str, str]] = {
    "I": (_I2, "Identity gate: leaves the qubit unchanged.",
          "No rotation on the Bloch sphere."),
    "X": (_X, "Pauli-X (NOT) gate: flips |0> to |1> and vice versa.",
          "180-degree rotation about the X axis of the Bloch sphere."),
    "Y": (_Y, "Pauli-Y gate: flips with a phase of i.",
          "180-degree rotation about the Y axis of the Bloch sphere."),
    "Z": (_Z, "Pauli-Z gate: adds a phase of -1 to |1>.",
          "180-degree rotation about the Z axis of the Bloch sphere."),
    "H": (_H, "Hadamard gate: creates equal superposition from computational basis.",
          "Rotation that maps |0> to |+> and |1> to |-> on the Bloch sphere."),
    "S": (_S, "S (phase) gate: adds a phase of i to |1>.",
          "90-degree rotation about the Z axis of the Bloch sphere."),
    "T": (_T, "T gate: adds a phase of exp(i*pi/4) to |1>.",
          "45-degree rotation about the Z axis of the Bloch sphere."),
    "CNOT": (_CNOT, "Controlled-NOT: flips target qubit when control is |1>.",
             "Entangling gate; no single-qubit Bloch sphere interpretation."),
    "CZ": (_CZ, "Controlled-Z: applies Z to target when control is |1>.",
           "Entangling gate; symmetric between control and target."),
    "SWAP": (_SWAP, "SWAP gate: exchanges the states of two qubits.",
             "Swaps the Bloch vectors of the two qubits."),
    "Toffoli": (_TOFFOLI, "Toffoli (CCX): flips target when both controls are |1>.",
                "Three-qubit gate; no simple Bloch sphere picture."),
}

_PARAM_GATES: Dict[str, Tuple[Callable, str, str]] = {
    "Rx": (_rx, "Rx(theta): rotation about X axis by angle theta.",
           "Rotation by theta about the X axis of the Bloch sphere."),
    "Ry": (_ry, "Ry(theta): rotation about Y axis by angle theta.",
           "Rotation by theta about the Y axis of the Bloch sphere."),
    "Rz": (_rz, "Rz(theta): rotation about Z axis by angle theta.",
           "Rotation by theta about the Z axis of the Bloch sphere."),
}


def _apply_single_qubit_gate(state: np.ndarray, n_qubits: int, gate: np.ndarray, qubit: int) -> np.ndarray:
    """Apply a 2x2 gate to a specific qubit in the statevector."""
    dim = 2 ** n_qubits
    new_state = np.zeros(dim, dtype=complex)
    for i in range(dim):
        bit = (i >> (n_qubits - 1 - qubit)) & 1
        if bit == 0:
            j = i | (1 << (n_qubits - 1 - qubit))
            new_state[i] += gate[0, 0] * state[i] + gate[0, 1] * state[j]
            new_state[j] += gate[1, 0] * state[i] + gate[1, 1] * state[j]
    return new_state


def _apply_two_qubit_gate(state: np.ndarray, n_qubits: int, gate: np.ndarray, q0: int, q1: int) -> np.ndarray:
    """Apply a 4x4 gate to two qubits. q0 is the more significant qubit in the 4x4 matrix."""
    dim = 2 ** n_qubits
    new_state = np.zeros(dim, dtype=complex)
    shift0 = n_qubits - 1 - q0
    shift1 = n_qubits - 1 - q1
    visited = set()
    for i in range(dim):
        b0 = (i >> shift0) & 1
        b1 = (i >> shift1) & 1
        base = i & ~(1 << shift0) & ~(1 << shift1)
        if base in visited:
            continue
        visited.add(base)
        indices = []
        for a in range(2):
            for b in range(2):
                idx = base | (a << shift0) | (b << shift1)
                indices.append(idx)
        sub = state[indices]
        out = gate @ sub
        for k, idx in enumerate(indices):
            new_state[idx] = out[k]
    return new_state


def _apply_three_qubit_gate(state: np.ndarray, n_qubits: int, gate: np.ndarray, q0: int, q1: int, q2: int) -> np.ndarray:
    """Apply an 8x8 gate to three qubits."""
    dim = 2 ** n_qubits
    new_state = np.zeros(dim, dtype=complex)
    shift0 = n_qubits - 1 - q0
    shift1 = n_qubits - 1 - q1
    shift2 = n_qubits - 1 - q2
    visited = set()
    for i in range(dim):
        base = i & ~(1 << shift0) & ~(1 << shift1) & ~(1 << shift2)
        if base in visited:
            continue
        visited.add(base)
        indices = []
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    idx = base | (a << shift0) | (b << shift1) | (c << shift2)
                    indices.append(idx)
        sub = state[indices]
        out = gate @ sub
        for k, idx in enumerate(indices):
            new_state[idx] = out[k]
    return new_state


def _apply_gate(state: np.ndarray, n_qubits: int, gate_name: str, qubits: list, params: dict = None) -> np.ndarray:
    """Apply a named gate to the statevector."""
    if params is None:
        params = {}
    if gate_name in _PARAM_GATES:
        theta = params.get("theta", 0.0)
        mat = _PARAM_GATES[gate_name][0](theta)
        return _apply_single_qubit_gate(state, n_qubits, mat, qubits[0])
    if gate_name not in _GATE_INFO:
        raise ValueError(f"Unknown gate: {gate_name}")
    mat = _GATE_INFO[gate_name][0]
    if mat.shape == (2, 2):
        return _apply_single_qubit_gate(state, n_qubits, mat, qubits[0])
    elif mat.shape == (4, 4):
        return _apply_two_qubit_gate(state, n_qubits, mat, qubits[0], qubits[1])
    elif mat.shape == (8, 8):
        return _apply_three_qubit_gate(state, n_qubits, mat, qubits[0], qubits[1], qubits[2])
    raise ValueError(f"Unsupported gate size: {mat.shape}")


# ---------------------------------------------------------------------------
# GateTutor
# ---------------------------------------------------------------------------

class GateTutor:
    """Interactive gate-by-gate tutor.

    Teaches: X, Y, Z, H, S, T, CNOT, CZ, SWAP, Toffoli, Rx, Ry, Rz.
    For each gate shows the matrix, description, Bloch sphere perspective,
    state evolution, and measurement probabilities.
    """

    def __init__(self, n_qubits: int = 2):
        self.n_qubits = n_qubits
        self.state = np.zeros(2 ** n_qubits, dtype=complex)
        self.state[0] = 1.0
        self._gates = self._build_gate_library()

    def _build_gate_library(self) -> dict:
        """Build dictionary of all gate matrices with descriptions."""
        lib = {}
        for name, (mat, desc, bloch) in _GATE_INFO.items():
            lib[name] = {"matrix": mat, "description": desc, "bloch_info": bloch}
        for name, (fn, desc, bloch) in _PARAM_GATES.items():
            lib[name] = {"matrix_fn": fn, "description": desc, "bloch_info": bloch}
        return lib

    def apply(self, gate_name: str, qubits: list, params: dict = None) -> dict:
        """Apply gate and return educational info."""
        if params is None:
            params = {}
        if gate_name not in self._gates:
            raise ValueError(f"Unknown gate '{gate_name}'. Use list_gates() to see available gates.")
        info = self._gates[gate_name]
        state_before = self.state.copy()
        self.state = _apply_gate(self.state, self.n_qubits, gate_name, qubits, params)
        if "matrix_fn" in info:
            theta = params.get("theta", 0.0)
            matrix = info["matrix_fn"](theta)
        else:
            matrix = info["matrix"]
        return {
            "gate_name": gate_name,
            "matrix": matrix,
            "description": info["description"],
            "state_before": state_before,
            "state_after": self.state.copy(),
            "probabilities": np.abs(self.state) ** 2,
            "bloch_info": info["bloch_info"],
        }

    def demonstrate_gate(self, gate_name: str) -> str:
        """Full demonstration of a gate on standard inputs."""
        if gate_name not in self._gates:
            raise ValueError(f"Unknown gate '{gate_name}'.")
        info = self._gates[gate_name]
        lines = [f"=== {gate_name} Gate ===", "", info["description"], ""]
        if "matrix" in info:
            mat = info["matrix"]
        else:
            mat = info["matrix_fn"](np.pi / 4)
            lines.append(f"(Shown for theta = pi/4)")
        lines.append("Matrix:")
        for row in mat:
            parts = []
            for v in row:
                if abs(v.imag) < 1e-12:
                    parts.append(f"{v.real:7.4f}")
                else:
                    parts.append(f"{v.real:+.3f}{v.imag:+.3f}i")
            lines.append("  [" + ", ".join(parts) + "]")
        lines.append("")
        lines.append(f"Bloch sphere: {info['bloch_info']}")
        if mat.shape == (2, 2):
            lines.append("")
            lines.append("Effect on standard states:")
            for label, sv in [
                ("|0>", np.array([1, 0], dtype=complex)),
                ("|1>", np.array([0, 1], dtype=complex)),
                ("|+>", np.array([1, 1], dtype=complex) / np.sqrt(2)),
            ]:
                out = mat @ sv
                probs = np.abs(out) ** 2
                lines.append(f"  {label} -> P(|0>)={probs[0]:.4f}, P(|1>)={probs[1]:.4f}")
        return "\n".join(lines)

    def list_gates(self) -> str:
        """List all available gates with brief descriptions."""
        lines = ["Available gates:", ""]
        for name, info in self._gates.items():
            desc = info["description"].split(".")[0] + "."
            lines.append(f"  {name:10s} {desc}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset to |0...0> state."""
        self.state = np.zeros(2 ** self.n_qubits, dtype=complex)
        self.state[0] = 1.0


# ---------------------------------------------------------------------------
# EntanglementTutor
# ---------------------------------------------------------------------------

class EntanglementTutor:
    """Teaches entanglement concepts step by step.

    Lessons cover Bell states, GHZ states, W states, entanglement measures,
    the no-cloning theorem, and quantum teleportation.
    """

    def bell_state_lesson(self) -> CircuitLesson:
        """Create a Bell state |Phi+> = (|00> + |11>) / sqrt(2)."""
        return CircuitLesson(
            title="Bell State Creation",
            description="Create maximally entangled Bell state |Phi+> using H and CNOT.",
            n_qubits=2,
            steps=[
                LessonStep(
                    instruction="Apply Hadamard to qubit 0 to create superposition.",
                    hint="H|0> = (|0>+|1>)/sqrt(2)",
                    gate_sequence=[("H", [0], {})],
                    explanation="Qubit 0 is now in superposition: (|0>+|1>)/sqrt(2). The system is |+0> = (|00>+|10>)/sqrt(2).",
                ),
                LessonStep(
                    instruction="Apply CNOT with qubit 0 as control and qubit 1 as target.",
                    hint="CNOT flips the target when the control is |1>.",
                    gate_sequence=[("CNOT", [0, 1], {})],
                    explanation="The CNOT entangles the qubits: |00>+|10> becomes |00>+|11> (normalized). This is the Bell state |Phi+>.",
                ),
            ],
        )

    def ghz_lesson(self, n: int = 3) -> CircuitLesson:
        """Create an n-qubit GHZ state (|00...0> + |11...1>) / sqrt(2)."""
        steps = [
            LessonStep(
                instruction="Apply Hadamard to qubit 0.",
                hint="Creates initial superposition on the first qubit.",
                gate_sequence=[("H", [0], {})],
                explanation="Qubit 0 is now in superposition, other qubits remain |0>.",
            ),
        ]
        for i in range(1, n):
            steps.append(LessonStep(
                instruction=f"Apply CNOT with control=0, target={i}.",
                hint=f"This entangles qubit {i} with qubit 0.",
                gate_sequence=[("CNOT", [0, i], {})],
                explanation=f"Qubit {i} is now entangled. The |1...1> branch has grown by one qubit.",
            ))
        return CircuitLesson(
            title=f"{n}-qubit GHZ State",
            description=f"Create a {n}-qubit GHZ state with Hadamard + cascading CNOTs.",
            n_qubits=n,
            steps=steps,
        )

    def w_state_lesson(self) -> CircuitLesson:
        """Create the 3-qubit W state (|001>+|010>+|100>)/sqrt(3)."""
        theta1 = np.arccos(np.sqrt(1 / 3))
        theta2 = np.pi / 4
        return CircuitLesson(
            title="W State Creation",
            description="Create the W state |W> = (|001>+|010>+|100>)/sqrt(3), which has different entanglement properties than GHZ.",
            n_qubits=3,
            steps=[
                LessonStep(
                    instruction="Apply Ry(arccos(sqrt(1/3))) to qubit 0.",
                    hint="This sets qubit 0 amplitude so that P(|1>)=1/3.",
                    gate_sequence=[("Ry", [0], {"theta": 2 * theta1})],
                    explanation="Qubit 0 now has amplitude sqrt(1/3) for |1> and sqrt(2/3) for |0>.",
                ),
                LessonStep(
                    instruction="Apply controlled-H (via CNOT pattern) to distribute amplitude.",
                    hint="Use CNOT(0,1) and then conditional rotation on qubit 1.",
                    gate_sequence=[
                        ("CNOT", [0, 1], {}),
                        ("CNOT", [0, 2], {}),
                        ("X", [0], {}),
                        ("CNOT", [0, 1], {}),
                        ("Ry", [1], {"theta": 2 * theta2}),
                        ("CNOT", [1, 2], {}),
                        ("X", [1], {}),
                    ],
                    explanation="Through careful gate decomposition we distribute the single-excitation across all qubits, approximating the W state.",
                ),
            ],
        )

    def entanglement_measures_lesson(self) -> CircuitLesson:
        """Demonstrate entanglement measures on Bell state."""
        return CircuitLesson(
            title="Entanglement Measures",
            description="Create a Bell state and discuss its entanglement measures (concurrence=1, von Neumann entropy=1 bit).",
            n_qubits=2,
            steps=[
                LessonStep(
                    instruction="Create Bell state |Phi+>.",
                    hint="H on qubit 0, then CNOT(0,1).",
                    gate_sequence=[("H", [0], {}), ("CNOT", [0, 1], {})],
                    explanation="For |Phi+>, the reduced density matrix of either qubit is maximally mixed (I/2). Concurrence = 1.0, von Neumann entropy = log2(2) = 1 bit. This is maximally entangled.",
                ),
                LessonStep(
                    instruction="Compare with a product state: reset and apply H to qubit 0 only.",
                    hint="A product state has zero entanglement.",
                    gate_sequence=[],
                    explanation="Without CNOT, the state (|0>+|1>)/sqrt(2) x |0> is a product state. Concurrence = 0, entropy = 0. Separable states have no entanglement.",
                ),
            ],
        )

    def no_cloning_lesson(self) -> CircuitLesson:
        """Demonstrate the no-cloning theorem."""
        return CircuitLesson(
            title="No-Cloning Theorem",
            description="Show that CNOT does not clone arbitrary quantum states.",
            n_qubits=2,
            steps=[
                LessonStep(
                    instruction="Start with |00> and apply CNOT. This 'copies' |0>.",
                    hint="CNOT(0,1) on |00> gives |00>.",
                    gate_sequence=[("CNOT", [0, 1], {})],
                    explanation="CNOT|00> = |00>. It looks like cloning worked for |0>.",
                ),
                LessonStep(
                    instruction="Reset. Put qubit 0 in superposition |+>, then try to 'clone' with CNOT.",
                    hint="If cloning worked, we'd get |+>|+>. But we get a Bell state instead.",
                    gate_sequence=[
                        ("H", [0], {}),
                        ("CNOT", [0, 1], {}),
                    ],
                    explanation="We get (|00>+|11>)/sqrt(2), the Bell state -- NOT |+>|+> = (|00>+|01>+|10>+|11>)/2. CNOT entangles rather than clones. This is the no-cloning theorem in action: no unitary can clone an unknown quantum state.",
                ),
            ],
        )

    def teleportation_lesson(self) -> CircuitLesson:
        """Quantum teleportation protocol walkthrough."""
        return CircuitLesson(
            title="Quantum Teleportation",
            description="Teleport qubit 0's state to qubit 2 using a Bell pair (qubits 1,2) and classical communication.",
            n_qubits=3,
            steps=[
                LessonStep(
                    instruction="Prepare the state to teleport on qubit 0: apply H to create |+>.",
                    hint="We will teleport the state H|0> = |+>.",
                    gate_sequence=[("H", [0], {})],
                    explanation="Qubit 0 is now in state |+> = (|0>+|1>)/sqrt(2). This is the state we want to teleport to qubit 2.",
                ),
                LessonStep(
                    instruction="Create a Bell pair between qubits 1 and 2.",
                    hint="H on qubit 1, then CNOT(1,2).",
                    gate_sequence=[("H", [1], {}), ("CNOT", [1, 2], {})],
                    explanation="Qubits 1 and 2 are now in the Bell state |Phi+>. Qubit 0 is still |+>.",
                ),
                LessonStep(
                    instruction="Alice performs Bell measurement: CNOT(0,1) then H(0).",
                    hint="This disentangles qubit 0 from the system and encodes info in qubits 0,1.",
                    gate_sequence=[("CNOT", [0, 1], {}), ("H", [0], {})],
                    explanation="After Bell measurement gates, the state is restructured. Measuring qubits 0 and 1 (classically) tells Bob which correction to apply. For outcome 00, qubit 2 is already in state |+>.",
                ),
            ],
        )


# ---------------------------------------------------------------------------
# AlgorithmWalkthrough
# ---------------------------------------------------------------------------

class AlgorithmWalkthrough:
    """Step-by-step walkthroughs of famous quantum algorithms.

    Each walkthrough returns a CircuitLesson that shows the state vector
    at every step.
    """

    def deutsch_jozsa(self, oracle_type: str = "balanced") -> CircuitLesson:
        """Deutsch-Jozsa algorithm for 1-bit function (2 qubits).

        Parameters
        ----------
        oracle_type : str
            'constant' (f(x)=0 for all x) or 'balanced' (f(0)=0, f(1)=1).
        """
        if oracle_type == "constant":
            oracle_gates: list = []
            oracle_desc = "Constant oracle: does nothing (f(x)=0 for all x)."
        else:
            oracle_gates = [("CNOT", [0, 1], {})]
            oracle_desc = "Balanced oracle: CNOT flips ancilla when input is |1>."

        return CircuitLesson(
            title=f"Deutsch-Jozsa ({oracle_type})",
            description="Determine if a boolean function is constant or balanced with a single query.",
            n_qubits=2,
            steps=[
                LessonStep(
                    instruction="Prepare ancilla qubit 1 in |1> state.",
                    hint="Apply X to qubit 1.",
                    gate_sequence=[("X", [1], {})],
                    explanation="State is now |01> (qubit 0 = |0>, qubit 1 = |1>).",
                ),
                LessonStep(
                    instruction="Apply Hadamard to both qubits.",
                    hint="This creates superposition on input and phase kickback state on ancilla.",
                    gate_sequence=[("H", [0], {}), ("H", [1], {})],
                    explanation="State is |+>|-> = (|0>+|1>)(|0>-|1>)/2. The ancilla in |-> enables phase kickback.",
                ),
                LessonStep(
                    instruction="Apply the oracle.",
                    hint=oracle_desc,
                    gate_sequence=oracle_gates,
                    explanation=f"Oracle applied ({oracle_type}). For balanced oracle, phase kickback flips sign of |1> component on qubit 0.",
                ),
                LessonStep(
                    instruction="Apply Hadamard to qubit 0 and measure.",
                    hint="H maps |+> to |0> and |-> to |1>.",
                    gate_sequence=[("H", [0], {})],
                    explanation=f"For {oracle_type} oracle: measuring qubit 0 gives {'|0> (constant)' if oracle_type == 'constant' else '|1> (balanced)'}. One query suffices!",
                ),
            ],
        )

    def bernstein_vazirani(self, secret: str = "101") -> CircuitLesson:
        """Bernstein-Vazirani algorithm to find a secret string.

        Parameters
        ----------
        secret : str
            Binary string like '101'. The oracle computes f(x) = s . x (mod 2).
        """
        n = len(secret)
        total_qubits = n + 1

        oracle_gates = []
        for i, bit in enumerate(secret):
            if bit == "1":
                oracle_gates.append(("CNOT", [i, n], {}))

        return CircuitLesson(
            title=f"Bernstein-Vazirani (s={secret})",
            description=f"Find the secret string s={secret} with a single oracle query.",
            n_qubits=total_qubits,
            steps=[
                LessonStep(
                    instruction="Prepare ancilla in |1> and apply H to all qubits.",
                    hint="Put ancilla in |-> for phase kickback.",
                    gate_sequence=[("X", [n], {})] + [("H", [i], {}) for i in range(total_qubits)],
                    explanation="All input qubits in |+>, ancilla in |->. Ready for oracle query.",
                ),
                LessonStep(
                    instruction="Apply the oracle (CNOT for each 1-bit in the secret).",
                    hint=f"Secret is {secret}: CNOT from each '1' position to ancilla.",
                    gate_sequence=oracle_gates,
                    explanation="Phase kickback encodes the secret string into the phases of input qubits.",
                ),
                LessonStep(
                    instruction="Apply Hadamard to all input qubits.",
                    hint="H converts phase information back to computational basis.",
                    gate_sequence=[("H", [i], {}) for i in range(n)],
                    explanation=f"Measuring input qubits now yields the secret string s = {secret} with certainty!",
                ),
            ],
        )

    def grover_2qubit(self, target: int = 3) -> CircuitLesson:
        """Grover's algorithm on 2 qubits (single iteration suffices).

        Parameters
        ----------
        target : int
            Target state index (0-3).
        """
        n = 2
        target_bits = format(target, f"0{n}b")

        oracle_x_before = []
        for i, bit in enumerate(target_bits):
            if bit == "0":
                oracle_x_before.append(("X", [i], {}))
        oracle_x_after = list(oracle_x_before)

        return CircuitLesson(
            title=f"Grover's Search (target=|{target_bits}>)",
            description=f"Find |{target_bits}> in a 2-qubit search space with one oracle call.",
            n_qubits=n,
            steps=[
                LessonStep(
                    instruction="Apply Hadamard to all qubits to create uniform superposition.",
                    hint="Equal amplitude on all 4 computational basis states.",
                    gate_sequence=[("H", [i], {}) for i in range(n)],
                    explanation="State is |++> = (|00>+|01>+|10>+|11>)/2. Each state has probability 1/4.",
                ),
                LessonStep(
                    instruction="Apply the oracle: mark the target state with a phase flip.",
                    hint=f"Oracle negates amplitude of |{target_bits}>. Use X gates + CZ.",
                    gate_sequence=oracle_x_before + [("CZ", [0, 1], {})] + oracle_x_after,
                    explanation=f"The amplitude of |{target_bits}> is now negative. All other amplitudes unchanged.",
                ),
                LessonStep(
                    instruction="Apply the diffusion operator (inversion about mean).",
                    hint="H on all, X on all, CZ, X on all, H on all.",
                    gate_sequence=(
                        [("H", [i], {}) for i in range(n)]
                        + [("X", [i], {}) for i in range(n)]
                        + [("CZ", [0, 1], {})]
                        + [("X", [i], {}) for i in range(n)]
                        + [("H", [i], {}) for i in range(n)]
                    ),
                    explanation=f"Diffusion amplifies the marked state. For 2 qubits, one iteration gives probability 1 for |{target_bits}>!",
                ),
            ],
        )

    def qft_walkthrough(self, n: int = 3) -> CircuitLesson:
        """Quantum Fourier Transform walkthrough.

        Parameters
        ----------
        n : int
            Number of qubits.
        """
        steps = [
            LessonStep(
                instruction="Start from |1> on qubit 0 (others |0>) to see QFT effect.",
                hint="QFT of |1> gives equal superposition with phases.",
                gate_sequence=[("X", [0], {})],
                explanation=f"State is |1{'0' * (n - 1)}>. We will apply QFT to transform this.",
            ),
        ]
        for i in range(n):
            step_gates: List[Tuple] = [("H", [i], {})]
            desc_parts = [f"H on qubit {i}"]
            for j in range(i + 1, n):
                k = j - i + 1
                angle = np.pi / (2 ** (k - 1))
                step_gates.append(("Rz", [i], {"theta": angle}))
                desc_parts.append(f"controlled-R{k} phase on qubit {i}")
            steps.append(LessonStep(
                instruction=f"QFT stage {i + 1}: Apply H and controlled phase rotations for qubit {i}.",
                hint="Each stage combines H with increasingly fine phase rotations.",
                gate_sequence=step_gates,
                explanation=f"Applied {', '.join(desc_parts)}. Each qubit gains phase information from higher qubits.",
            ))

        return CircuitLesson(
            title=f"{n}-qubit QFT",
            description="Quantum Fourier Transform: the quantum analog of the discrete Fourier transform.",
            n_qubits=n,
            steps=steps,
        )

    def phase_estimation(self, eigenvalue: float = 0.25) -> CircuitLesson:
        """Quantum Phase Estimation walkthrough.

        Parameters
        ----------
        eigenvalue : float
            The phase to estimate (eigenvalue = exp(2*pi*i*eigenvalue)).
        """
        n_counting = 3
        total_qubits = n_counting + 1

        steps = [
            LessonStep(
                instruction="Prepare eigenstate |1> on the last qubit.",
                hint="The eigenstate of Z is |1> with eigenvalue -1 = exp(i*pi).",
                gate_sequence=[("X", [n_counting], {})],
                explanation="Target qubit is in eigenstate |1>.",
            ),
            LessonStep(
                instruction="Apply Hadamard to all counting qubits.",
                hint="This creates uniform superposition on the counting register.",
                gate_sequence=[("H", [i], {}) for i in range(n_counting)],
                explanation="Counting register in uniform superposition. Ready for controlled-U operations.",
            ),
        ]

        for i in range(n_counting):
            reps = 2 ** i
            angle = 2 * np.pi * eigenvalue * reps
            steps.append(LessonStep(
                instruction=f"Apply controlled-Rz({reps} * 2*pi*{eigenvalue}) from counting qubit {i}.",
                hint=f"Controlled phase rotation encodes eigenvalue into counting register phases.",
                gate_sequence=[("Rz", [i], {"theta": angle})],
                explanation=f"Counting qubit {i} picks up phase proportional to 2^{i} * eigenvalue.",
            ))

        inv_qft_gates: list = []
        for i in range(n_counting - 1, -1, -1):
            for j in range(n_counting - 1, i, -1):
                k = j - i + 1
                angle = -np.pi / (2 ** (k - 1))
                inv_qft_gates.append(("Rz", [i], {"theta": angle}))
            inv_qft_gates.append(("H", [i], {}))
        steps.append(LessonStep(
            instruction="Apply inverse QFT to the counting register.",
            hint="Inverse QFT converts phase information to computational basis.",
            gate_sequence=inv_qft_gates,
            explanation=f"After inverse QFT, measuring the counting register gives the binary representation of the phase {eigenvalue}.",
        ))

        return CircuitLesson(
            title=f"Phase Estimation (eigenvalue={eigenvalue})",
            description="Estimate the eigenvalue phase of a unitary operator to n-bit precision.",
            n_qubits=total_qubits,
            steps=steps,
        )
