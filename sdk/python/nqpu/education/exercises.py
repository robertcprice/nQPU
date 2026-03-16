"""Self-paced quantum computing exercises with solution checkers.

Each exercise defines a target quantum state and a checker function that
verifies a student's state vector (up to global phase).

All simulation is pure numpy -- no scipy or external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .circuit_tutorial import _apply_gate


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Exercise:
    """A self-contained quantum computing exercise."""

    id: str
    title: str
    description: str
    category: str  # "gates", "entanglement", "algorithms", "qec"
    difficulty: int  # 1-5
    n_qubits: int
    hints: List[str]
    checker: Callable  # function(student_state) -> bool
    solution_explanation: str


@dataclass
class ExerciseResult:
    """Result of running an exercise."""

    exercise_id: str
    passed: bool
    feedback: str
    attempts: int = 1


@dataclass
class ExerciseSet:
    """Collection of exercises in a category."""

    name: str
    exercises: List[Exercise]

    def run_all(self, solutions: dict) -> List[ExerciseResult]:
        """Run all exercises with provided solutions.

        Parameters
        ----------
        solutions : dict
            Mapping from exercise.id to np.ndarray state vectors.
        """
        results = []
        for ex in self.exercises:
            if ex.id in solutions:
                result = run_exercise(ex, solutions[ex.id])
            else:
                result = ExerciseResult(
                    exercise_id=ex.id,
                    passed=False,
                    feedback=f"No solution provided for '{ex.title}'.",
                )
            results.append(result)
        return results

    def progress_report(self, results: List[ExerciseResult]) -> str:
        """ASCII progress report."""
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        lines = [
            f"=== {self.name} Progress ===",
            f"Passed: {passed}/{total}",
            "",
        ]
        for r in results:
            ex = next((e for e in self.exercises if e.id == r.exercise_id), None)
            title = ex.title if ex else r.exercise_id
            status = "[PASS]" if r.passed else "[FAIL]"
            lines.append(f"  {status} {title}")
            if not r.passed:
                lines.append(f"         {r.feedback}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checker utilities
# ---------------------------------------------------------------------------

def _states_equal_up_to_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> bool:
    """Check if two state vectors are equal up to a global phase."""
    if a.shape != b.shape:
        return False
    # Normalize
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return na < 1e-12 and nb < 1e-12
    a_n = a / na
    b_n = b / nb
    # overlap = |<a|b>|^2
    overlap = abs(np.vdot(a_n, b_n)) ** 2
    return overlap > 1.0 - atol


def run_exercise(exercise: Exercise, state: np.ndarray) -> ExerciseResult:
    """Check a student's solution state against the exercise checker."""
    try:
        passed = exercise.checker(state)
    except Exception as e:
        return ExerciseResult(
            exercise_id=exercise.id,
            passed=False,
            feedback=f"Error while checking solution: {e}",
        )
    if passed:
        feedback = f"Correct! {exercise.solution_explanation}"
    else:
        feedback = f"Not quite right. Hint: {exercise.hints[0] if exercise.hints else 'Try again.'}"
    return ExerciseResult(
        exercise_id=exercise.id,
        passed=passed,
        feedback=feedback,
    )


def check_solution(exercise: Exercise, state: np.ndarray) -> bool:
    """Quick check if state is correct."""
    try:
        return exercise.checker(state)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helper: build a state by applying gates from |0...0>
# ---------------------------------------------------------------------------

def _build_state(n_qubits: int, gates: list) -> np.ndarray:
    """Build a state vector by applying a sequence of gates to |0...0>."""
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[0] = 1.0
    for gate_name, qubits, params in gates:
        state = _apply_gate(state, n_qubits, gate_name, qubits, params)
    return state


# ---------------------------------------------------------------------------
# Gate exercises
# ---------------------------------------------------------------------------

def gate_exercises() -> ExerciseSet:
    """8 exercises on quantum gates."""
    exercises = []

    # 1. Create |1> from |0> (X gate)
    target_1 = np.array([0, 1], dtype=complex)
    exercises.append(Exercise(
        id="gate_01",
        title="Create |1> from |0>",
        description="Apply a gate to transform |0> into |1>.",
        category="gates",
        difficulty=1,
        n_qubits=1,
        hints=["The Pauli-X gate acts like a classical NOT gate."],
        checker=lambda s, t=target_1: _states_equal_up_to_phase(s, t),
        solution_explanation="X|0> = |1>. The X gate flips the qubit state.",
    ))

    # 2. Create |+> from |0> (H gate)
    target_2 = np.array([1, 1], dtype=complex) / np.sqrt(2)
    exercises.append(Exercise(
        id="gate_02",
        title="Create |+> from |0>",
        description="Apply a gate to transform |0> into |+> = (|0>+|1>)/sqrt(2).",
        category="gates",
        difficulty=1,
        n_qubits=1,
        hints=["The Hadamard gate creates superposition."],
        checker=lambda s, t=target_2: _states_equal_up_to_phase(s, t),
        solution_explanation="H|0> = |+>. The Hadamard creates equal superposition.",
    ))

    # 3. Create |-> from |0> (X then H)
    target_3 = np.array([1, -1], dtype=complex) / np.sqrt(2)
    exercises.append(Exercise(
        id="gate_03",
        title="Create |-> from |0>",
        description="Apply gates to transform |0> into |-> = (|0>-|1>)/sqrt(2).",
        category="gates",
        difficulty=2,
        n_qubits=1,
        hints=["First flip to |1>, then apply Hadamard.", "H|1> = |->"],
        checker=lambda s, t=target_3: _states_equal_up_to_phase(s, t),
        solution_explanation="H(X|0>) = H|1> = |->.",
    ))

    # 4. Create |i> = (|0>+i|1>)/sqrt(2) (H then S)
    target_4 = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    exercises.append(Exercise(
        id="gate_04",
        title="Create (|0>+i|1>)/sqrt(2)",
        description="Create the state (|0>+i|1>)/sqrt(2) from |0>.",
        category="gates",
        difficulty=2,
        n_qubits=1,
        hints=["First create |+>, then add phase with S gate.", "S|+> = (|0>+i|1>)/sqrt(2)"],
        checker=lambda s, t=target_4: _states_equal_up_to_phase(s, t),
        solution_explanation="S(H|0>) = S|+> = (|0>+i|1>)/sqrt(2). S adds phase i to |1>.",
    ))

    # 5. Apply T gate to |+> and verify phase
    target_5 = np.array([1, np.exp(1j * np.pi / 4)], dtype=complex) / np.sqrt(2)
    exercises.append(Exercise(
        id="gate_05",
        title="T gate on |+>",
        description="Create T|+> = (|0>+exp(i*pi/4)|1>)/sqrt(2).",
        category="gates",
        difficulty=2,
        n_qubits=1,
        hints=["Create |+> with H, then apply T."],
        checker=lambda s, t=target_5: _states_equal_up_to_phase(s, t),
        solution_explanation="T(H|0>) = (|0>+exp(i*pi/4)|1>)/sqrt(2). T adds a pi/4 phase to |1>.",
    ))

    # 6. Controlled NOT: |10> -> |11>
    target_6 = np.array([0, 0, 0, 1], dtype=complex)
    exercises.append(Exercise(
        id="gate_06",
        title="CNOT: |10> to |11>",
        description="Start with |10> and use CNOT to produce |11>.",
        category="gates",
        difficulty=2,
        n_qubits=2,
        hints=["First prepare |10> by applying X to qubit 0.", "CNOT flips target when control is |1>."],
        checker=lambda s, t=target_6: _states_equal_up_to_phase(s, t),
        solution_explanation="X on q0 gives |10>, then CNOT(0,1) flips q1 -> |11>.",
    ))

    # 7. SWAP two qubits: |10> -> |01>
    target_7 = np.array([0, 1, 0, 0], dtype=complex)
    exercises.append(Exercise(
        id="gate_07",
        title="SWAP: |10> to |01>",
        description="Start with |10> and SWAP the qubits to get |01>.",
        category="gates",
        difficulty=2,
        n_qubits=2,
        hints=["Prepare |10>, then apply SWAP."],
        checker=lambda s, t=target_7: _states_equal_up_to_phase(s, t),
        solution_explanation="SWAP|10> = |01>. The SWAP gate exchanges qubit states.",
    ))

    # 8. Create (|01> + |10>)/sqrt(2) using CNOT
    target_8 = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    exercises.append(Exercise(
        id="gate_08",
        title="Create (|01>+|10>)/sqrt(2)",
        description="Create the state (|01>+|10>)/sqrt(2) -- one of the Bell states.",
        category="gates",
        difficulty=3,
        n_qubits=2,
        hints=[
            "Start by flipping qubit 1 to get |01>.",
            "Then apply H to qubit 0 and CNOT(0,1).",
            "This is |Psi+>.",
        ],
        checker=lambda s, t=target_8: _states_equal_up_to_phase(s, t),
        solution_explanation="X on q1 -> |01>. H on q0 -> (|0>+|1>)|1>/sqrt(2) = (|01>+|11>)/sqrt(2). CNOT(0,1) -> (|01>+|10>)/sqrt(2).",
    ))

    return ExerciseSet(name="Quantum Gates", exercises=exercises)


# ---------------------------------------------------------------------------
# Entanglement exercises
# ---------------------------------------------------------------------------

def entanglement_exercises() -> ExerciseSet:
    """6 exercises on entanglement."""
    exercises = []

    # 1. Bell state |Phi+> = (|00>+|11>)/sqrt(2)
    target_1 = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    exercises.append(Exercise(
        id="ent_01",
        title="Bell state |Phi+>",
        description="Create |Phi+> = (|00>+|11>)/sqrt(2).",
        category="entanglement",
        difficulty=2,
        n_qubits=2,
        hints=["Apply H to qubit 0, then CNOT(0,1)."],
        checker=lambda s, t=target_1: _states_equal_up_to_phase(s, t),
        solution_explanation="H on q0 then CNOT(0,1): |00> -> (|0>+|1>)|0>/sqrt(2) -> (|00>+|11>)/sqrt(2).",
    ))

    # 2. Bell state |Psi-> = (|01>-|10>)/sqrt(2)
    target_2 = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    exercises.append(Exercise(
        id="ent_02",
        title="Bell state |Psi->",
        description="Create |Psi-> = (|01>-|10>)/sqrt(2).",
        category="entanglement",
        difficulty=3,
        n_qubits=2,
        hints=[
            "Start with |01>, apply H to qubit 0, then CNOT(0,1).",
            "Then apply Z to qubit 0 for the minus sign.",
        ],
        checker=lambda s, t=target_2: _states_equal_up_to_phase(s, t),
        solution_explanation="X on q1, H on q0, CNOT(0,1), Z on q0 gives (|01>-|10>)/sqrt(2).",
    ))

    # 3. 3-qubit GHZ state
    target_3 = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex) / np.sqrt(2)
    exercises.append(Exercise(
        id="ent_03",
        title="3-qubit GHZ state",
        description="Create (|000>+|111>)/sqrt(2).",
        category="entanglement",
        difficulty=3,
        n_qubits=3,
        hints=["H on qubit 0, CNOT(0,1), CNOT(0,2)."],
        checker=lambda s, t=target_3: _states_equal_up_to_phase(s, t),
        solution_explanation="H(q0), CNOT(0,1), CNOT(0,2) -> (|000>+|111>)/sqrt(2).",
    ))

    # 4. W state (|001>+|010>+|100>)/sqrt(3)
    target_4 = np.array([0, 1, 1, 0, 1, 0, 0, 0], dtype=complex) / np.sqrt(3)
    exercises.append(Exercise(
        id="ent_04",
        title="W state",
        description="Create the W state (|001>+|010>+|100>)/sqrt(3).",
        category="entanglement",
        difficulty=4,
        n_qubits=3,
        hints=[
            "The W state has exactly one qubit in |1>.",
            "Use rotation gates to distribute amplitudes.",
            "Ry(2*arccos(sqrt(1/3))) on q0 sets up the right amplitude.",
        ],
        checker=lambda s, t=target_4: _states_equal_up_to_phase(s, t),
        solution_explanation="The W state requires careful amplitude distribution. Unlike GHZ, losing one qubit preserves entanglement of the remaining two.",
    ))

    # 5. Verify maximal entanglement (compute entropy)
    def check_entropy(state: np.ndarray) -> bool:
        """Check that the state has von Neumann entropy = 1 bit for qubit 0."""
        if len(state) != 4:
            return False
        # Partial trace over qubit 1
        rho = np.outer(state, state.conj())
        rho_A = np.array([
            [rho[0, 0] + rho[1, 1], rho[0, 2] + rho[1, 3]],
            [rho[2, 0] + rho[3, 1], rho[2, 2] + rho[3, 3]],
        ])
        eigvals = np.linalg.eigvalsh(rho_A)
        eigvals = eigvals[eigvals > 1e-15]
        entropy = -np.sum(eigvals * np.log2(eigvals))
        return abs(entropy - 1.0) < 0.01

    exercises.append(Exercise(
        id="ent_05",
        title="Maximally entangled state",
        description="Create any 2-qubit state with von Neumann entropy = 1 bit.",
        category="entanglement",
        difficulty=3,
        n_qubits=2,
        hints=["Any Bell state is maximally entangled.", "H on q0, CNOT(0,1)."],
        checker=check_entropy,
        solution_explanation="Any Bell state has entropy = 1 bit. The reduced density matrix of either qubit is I/2.",
    ))

    # 6. Quantum teleportation circuit
    def check_teleportation(state: np.ndarray) -> bool:
        """Check that qubit 2 ends up in |+> state (teleportation of |+>)."""
        if len(state) != 8:
            return False
        # Trace out qubits 0 and 1, check qubit 2
        rho = np.outer(state, state.conj())
        # Partial trace: sum over indices for qubits 0,1
        rho_2 = np.zeros((2, 2), dtype=complex)
        for a in range(2):
            for b in range(2):
                for i in range(2):
                    for j in range(2):
                        idx_i = a * 4 + b * 2 + i
                        idx_j = a * 4 + b * 2 + j
                        rho_2[i, j] += rho[idx_i, idx_j]
        # Check if rho_2 is close to |+><+|
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        target_rho = np.outer(plus, plus.conj())
        return np.allclose(rho_2, target_rho, atol=0.05)

    exercises.append(Exercise(
        id="ent_06",
        title="Quantum teleportation",
        description="Teleport |+> from qubit 0 to qubit 2 using a Bell pair on qubits 1,2.",
        category="entanglement",
        difficulty=5,
        n_qubits=3,
        hints=[
            "1. Prepare |+> on qubit 0.",
            "2. Create Bell pair on qubits 1,2: H(1), CNOT(1,2).",
            "3. Bell measurement: CNOT(0,1), H(0).",
            "For outcome 00, qubit 2 is already in |+>.",
        ],
        checker=check_teleportation,
        solution_explanation="Teleportation: H(0), H(1), CNOT(1,2), CNOT(0,1), H(0). Qubit 2 receives the state.",
    ))

    return ExerciseSet(name="Entanglement", exercises=exercises)


# ---------------------------------------------------------------------------
# Algorithm exercises
# ---------------------------------------------------------------------------

def algorithm_exercises() -> ExerciseSet:
    """6 exercises on quantum algorithms."""
    exercises = []

    # 1. Deutsch's algorithm
    def check_deutsch(state: np.ndarray) -> bool:
        if len(state) != 4:
            return False
        probs = np.abs(state) ** 2
        # Qubit 0 should be |1> (balanced oracle)
        return probs[2] + probs[3] > 0.95

    exercises.append(Exercise(
        id="algo_01",
        title="Deutsch's Algorithm (balanced)",
        description="Run Deutsch's algorithm with the balanced oracle (CNOT). Qubit 0 should measure |1>.",
        category="algorithms",
        difficulty=3,
        n_qubits=2,
        hints=[
            "X on q1, H on both, CNOT(0,1) as oracle, H on q0.",
            "Measure q0: |1> means balanced.",
        ],
        checker=check_deutsch,
        solution_explanation="X(1), H(0), H(1), CNOT(0,1), H(0). Qubit 0 is |1> -> function is balanced.",
    ))

    # 2. Bernstein-Vazirani: find secret "11"
    def check_bv(state: np.ndarray) -> bool:
        if len(state) != 8:
            return False
        probs = np.abs(state) ** 2
        # Qubits 0,1 should be |11>, ancilla is whatever
        # |110> = index 6, |111> = index 7
        return probs[6] + probs[7] > 0.95

    exercises.append(Exercise(
        id="algo_02",
        title="Bernstein-Vazirani (s=11)",
        description="Find secret string s=11 using BV algorithm. Input qubits should measure |11>.",
        category="algorithms",
        difficulty=3,
        n_qubits=3,
        hints=[
            "X on ancilla (q2), H on all, oracle: CNOT(0,2) and CNOT(1,2), H on q0 and q1.",
        ],
        checker=check_bv,
        solution_explanation="BV algorithm: prepare, query, Hadamard. Input register reveals s=11.",
    ))

    # 3. Grover on 2 qubits: amplify |11>
    target_grover = _build_state(2, [
        ("H", [0], {}), ("H", [1], {}),
        ("CZ", [0, 1], {}),
        ("H", [0], {}), ("H", [1], {}),
        ("X", [0], {}), ("X", [1], {}),
        ("CZ", [0, 1], {}),
        ("X", [0], {}), ("X", [1], {}),
        ("H", [0], {}), ("H", [1], {}),
    ])

    def check_grover(state: np.ndarray) -> bool:
        if len(state) != 4:
            return False
        probs = np.abs(state) ** 2
        return probs[3] > 0.95

    exercises.append(Exercise(
        id="algo_03",
        title="Grover's Search for |11>",
        description="Use Grover's algorithm to find |11> in a 2-qubit search space.",
        category="algorithms",
        difficulty=4,
        n_qubits=2,
        hints=[
            "H on both, oracle (CZ), diffusion operator.",
            "Diffusion: H, X, CZ, X, H on both qubits.",
        ],
        checker=check_grover,
        solution_explanation="One Grover iteration on 2 qubits gives probability 1 for the target state.",
    ))

    # 4. QFT on 2 qubits from |10>
    def check_qft_2(state: np.ndarray) -> bool:
        if len(state) != 4:
            return False
        # QFT|10> should give equal amplitudes with specific phases
        expected = np.array([1, -1j, -1, 1j], dtype=complex) / 2
        return _states_equal_up_to_phase(state, expected)

    exercises.append(Exercise(
        id="algo_04",
        title="2-qubit QFT",
        description="Apply the 2-qubit QFT to |10>. Result: (|0>-i|1>-|2>+i|3>)/2.",
        category="algorithms",
        difficulty=4,
        n_qubits=2,
        hints=[
            "Prepare |10>. QFT: H on q0, controlled-S (Rz(pi/2)) from q1 to q0, H on q1, then SWAP.",
            "Simplified: X(0), H(0), Rz(pi/2)(0), H(1).",
        ],
        checker=check_qft_2,
        solution_explanation="QFT maps |k> to sum_j exp(2*pi*i*j*k/N)|j> / sqrt(N). For |10>, phases are 1, -i, -1, i.",
    ))

    # 5. Phase estimation with eigenvalue 1/4
    def check_pe(state: np.ndarray) -> bool:
        if len(state) < 8:
            return False
        probs = np.abs(state) ** 2
        # With 2 counting qubits, eigenvalue 1/4 -> measurement should show |01> on counting register
        # For 3 qubits total: |010> or |011> depending on eigenstate
        # Just check that the counting register has significant |01x> probability
        return probs[2] + probs[3] > 0.4

    exercises.append(Exercise(
        id="algo_05",
        title="Phase Estimation (eigenvalue 1/4)",
        description="Use QPE with 2 counting qubits to estimate eigenvalue exp(2*pi*i/4) of T gate.",
        category="algorithms",
        difficulty=5,
        n_qubits=3,
        hints=[
            "Prepare eigenstate |1> on q2.",
            "H on counting qubits q0, q1.",
            "Controlled-T^(2^k) from each counting qubit.",
            "Inverse QFT on counting register.",
        ],
        checker=check_pe,
        solution_explanation="QPE outputs the binary fraction 0.01 = 1/4 on the counting register.",
    ))

    # 6. Variational ansatz: minimize <psi|Z|psi>
    def check_variational(state: np.ndarray) -> bool:
        if len(state) != 2:
            return False
        # Minimum of <Z> = -1 when state = |1>
        probs = np.abs(state) ** 2
        expectation_Z = probs[0] - probs[1]
        return expectation_Z < -0.95

    exercises.append(Exercise(
        id="algo_06",
        title="Variational: minimize <Z>",
        description="Find the 1-qubit state that minimizes <psi|Z|psi>. Minimum is -1.",
        category="algorithms",
        difficulty=2,
        n_qubits=1,
        hints=[
            "Z has eigenvalues +1 (|0>) and -1 (|1>).",
            "The minimum eigenvalue state is |1>.",
        ],
        checker=check_variational,
        solution_explanation="The ground state of Z is |1> with eigenvalue -1. Apply X to |0> to get |1>.",
    ))

    return ExerciseSet(name="Quantum Algorithms", exercises=exercises)


# ---------------------------------------------------------------------------
# QEC exercises
# ---------------------------------------------------------------------------

def qec_exercises() -> ExerciseSet:
    """4 exercises on error correction."""
    exercises = []

    # 1. Encode |1> in 3-qubit bit-flip code
    target_1 = np.zeros(8, dtype=complex)
    target_1[0b111] = 1.0
    exercises.append(Exercise(
        id="qec_01",
        title="Encode |1> in bit-flip code",
        description="Encode logical |1> into 3-qubit bit-flip code: |1_L> = |111>.",
        category="qec",
        difficulty=2,
        n_qubits=3,
        hints=["Apply X to all three qubits."],
        checker=lambda s, t=target_1: _states_equal_up_to_phase(s, t),
        solution_explanation="|1_L> = |111>. In the bit-flip code, |0_L>=|000>, |1_L>=|111>.",
    ))

    # 2. Detect bit-flip error location from syndrome
    # Student provides the 3-qubit state after error; checker verifies syndrome detection
    def check_syndrome_detection(state: np.ndarray) -> bool:
        """Check that student correctly identified error on qubit 1 by correcting it."""
        if len(state) != 8:
            return False
        # After correction, state should be |111> (logical |1>)
        target = np.zeros(8, dtype=complex)
        target[0b111] = 1.0
        return _states_equal_up_to_phase(state, target)

    exercises.append(Exercise(
        id="qec_02",
        title="Correct bit-flip error",
        description="Given |101> (bit-flip error on qubit 1 of |111>), correct it back to |111>.",
        category="qec",
        difficulty=2,
        n_qubits=3,
        hints=[
            "Syndrome (1,1) indicates error on qubit 1.",
            "Apply X to qubit 1 to correct.",
        ],
        checker=check_syndrome_detection,
        solution_explanation="Syndrome (1,1) -> error on qubit 1. Apply X(1): |101> -> |111>.",
    ))

    # 3. Shor code: encode |0> in logical zero
    def check_shor_encode(state: np.ndarray) -> bool:
        """Check that the state is a valid Shor code |0_L>."""
        if len(state) != 512:  # 9 qubits
            return False
        # |0_L> = (|000>+|111>)^tensor3 / 2sqrt(2)
        target = np.zeros(512, dtype=complex)
        # Generate all codewords for |0_L>
        block_patterns = [0b000, 0b111]
        for b0 in block_patterns:
            for b1 in block_patterns:
                for b2 in block_patterns:
                    idx = (b0 << 6) | (b1 << 3) | b2
                    target[idx] = 1.0
        target /= np.linalg.norm(target)
        return _states_equal_up_to_phase(state, target)

    exercises.append(Exercise(
        id="qec_03",
        title="Shor code: encode |0>",
        description="Encode logical |0> in the 9-qubit Shor code. |0_L> = (|000>+|111>)^3 / 2sqrt(2).",
        category="qec",
        difficulty=4,
        n_qubits=9,
        hints=[
            "|0_L> is the tensor product of three copies of (|000>+|111>)/sqrt(2).",
            "For each block: H on first qubit, CNOT to second and third.",
        ],
        checker=check_shor_encode,
        solution_explanation="Each 3-qubit block is (|000>+|111>)/sqrt(2), created by H and two CNOTs.",
    ))

    # 4. Steane code: identify syndrome for X error on qubit 2
    def check_steane_syndrome(state: np.ndarray) -> bool:
        """Verify the student created the right syndrome pattern."""
        # The syndrome for X error on qubit 2 (0-indexed) should be
        # Z-syndrome = 011 (qubit 2 = index 3 in 1-indexed = binary 011)
        # We check by looking at the state with X error applied
        if not isinstance(state, np.ndarray):
            return False
        if len(state) == 3:
            # Student provided syndrome bits directly
            return tuple(int(round(x)) for x in state) == (0, 1, 1)
        if len(state) == 6:
            # Full syndrome (x_syndrome, z_syndrome)
            z_syn = tuple(int(round(x)) for x in state[3:6])
            return z_syn == (1, 1, 0)
        return False

    exercises.append(Exercise(
        id="qec_04",
        title="Steane code: X error syndrome",
        description="What is the Z-syndrome (3 bits) for an X error on qubit 2 of the Steane code? Provide as array [s0,s1,s2].",
        category="qec",
        difficulty=3,
        n_qubits=7,
        hints=[
            "Z-stabilizers detect X errors.",
            "Qubit 2 (0-indexed) -> syndrome = binary(2+1) = binary(3) = 011.",
            "Provide the 3-bit Z-syndrome as a numpy array.",
        ],
        checker=check_steane_syndrome,
        solution_explanation="X error on qubit 2 -> Z-syndrome = 011 (binary for 3). The syndrome is qubit_index + 1 in binary.",
    ))

    return ExerciseSet(name="Quantum Error Correction", exercises=exercises)
