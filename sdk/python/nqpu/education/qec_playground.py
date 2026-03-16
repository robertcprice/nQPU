"""Interactive quantum error correction exploration.

Provides an environment for learning QEC concepts: encoding, error injection,
syndrome measurement, correction, and decoding -- with ASCII visualizations.

All simulation is pure numpy statevector manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .circuit_tutorial import (
    CircuitLesson,
    LessonStep,
    TutorialResult,
    _apply_gate,
)


# ---------------------------------------------------------------------------
# Gate matrices used in QEC
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _kron(*mats: np.ndarray) -> np.ndarray:
    """Multi-argument Kronecker product."""
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


# ---------------------------------------------------------------------------
# QECPlayground
# ---------------------------------------------------------------------------

class QECPlayground:
    """Interactive QEC exploration environment.

    Supports bit_flip (3-qubit), phase_flip (3-qubit), shor (9-qubit),
    steane (7-qubit), and surface (small conceptual) codes.
    """

    _CODES = ("bit_flip", "phase_flip", "shor", "steane")

    def encode(self, logical_state: np.ndarray, code: str = "bit_flip") -> np.ndarray:
        """Encode a logical qubit into a QEC code.

        Parameters
        ----------
        logical_state : np.ndarray
            2-element vector [alpha, beta] for alpha|0>+beta|1>.
        code : str
            One of 'bit_flip', 'phase_flip', 'shor', 'steane'.
        """
        alpha, beta = complex(logical_state[0]), complex(logical_state[1])
        if code == "bit_flip":
            # |0_L> = |000>, |1_L> = |111>
            state = np.zeros(8, dtype=complex)
            state[0b000] = alpha
            state[0b111] = beta
            return state
        elif code == "phase_flip":
            # |0_L> = |+++>, |1_L> = |--->
            plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
            minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
            state_0L = _kron(plus, plus, plus)
            state_1L = _kron(minus, minus, minus)
            return alpha * state_0L + beta * state_1L
        elif code == "shor":
            # |0_L> = (|000>+|111>)^3 / 2sqrt(2)
            # |1_L> = (|000>-|111>)^3 / 2sqrt(2)
            plus3 = (np.zeros(8, dtype=complex))
            plus3[0b000] = 1.0
            plus3[0b111] = 1.0
            plus3 /= np.sqrt(2)
            minus3 = np.zeros(8, dtype=complex)
            minus3[0b000] = 1.0
            minus3[0b111] = -1.0
            minus3 /= np.sqrt(2)
            state_0L = _kron(plus3, plus3, plus3)
            state_1L = _kron(minus3, minus3, minus3)
            return alpha * state_0L + beta * state_1L
        elif code == "steane":
            # [[7,1,3]] code: |0_L> and |1_L> from stabilizer formalism
            state_0L = np.zeros(128, dtype=complex)
            state_1L = np.zeros(128, dtype=complex)
            # Steane code codewords (even-weight binary strings from [7,4,3] Hamming code)
            codewords_0 = [
                0b0000000, 0b1010101, 0b0110011, 0b1100110,
                0b0001111, 0b1011010, 0b0111100, 0b1101001,
            ]
            codewords_1 = [cw ^ 0b1111111 for cw in codewords_0]
            for cw in codewords_0:
                state_0L[cw] = 1.0
            state_0L /= np.linalg.norm(state_0L)
            for cw in codewords_1:
                state_1L[cw] = 1.0
            state_1L /= np.linalg.norm(state_1L)
            return alpha * state_0L + beta * state_1L
        raise ValueError(f"Unknown code: {code}. Use one of {self._CODES}.")

    def inject_error(self, state: np.ndarray, error_type: str, qubit: int, code: str) -> np.ndarray:
        """Inject a specific error and return the corrupted state.

        Parameters
        ----------
        error_type : str
            'X' (bit-flip), 'Z' (phase-flip), or 'Y' (both).
        qubit : int
            Physical qubit index to apply the error to.
        """
        n_qubits = int(np.log2(len(state)))
        error_map = {"X": _X, "Z": _Z, "Y": 1j * _X @ _Z}
        if error_type not in error_map:
            raise ValueError(f"Unknown error type: {error_type}. Use 'X', 'Z', or 'Y'.")
        error_op = error_map[error_type]
        # Build full operator: I x ... x error x ... x I
        ops = [_I] * n_qubits
        ops[qubit] = error_op
        full_op = ops[0]
        for op in ops[1:]:
            full_op = np.kron(full_op, op)
        return full_op @ state

    def measure_syndrome(self, state: np.ndarray, code: str) -> dict:
        """Measure syndrome bits and explain what they mean.

        Returns dict with 'syndrome', 'interpretation', 'error_location', 'correction'.
        """
        if code == "bit_flip":
            return self._syndrome_bit_flip(state)
        elif code == "phase_flip":
            return self._syndrome_phase_flip(state)
        elif code == "shor":
            return self._syndrome_shor(state)
        elif code == "steane":
            return self._syndrome_steane(state)
        raise ValueError(f"Unknown code: {code}")

    def correct(self, state: np.ndarray, syndrome: tuple, code: str) -> np.ndarray:
        """Apply correction based on syndrome."""
        if code == "bit_flip":
            return self._correct_bit_flip(state, syndrome)
        elif code == "phase_flip":
            return self._correct_phase_flip(state, syndrome)
        elif code == "shor":
            return self._correct_shor(state, syndrome)
        elif code == "steane":
            return self._correct_steane(state, syndrome)
        raise ValueError(f"Unknown code: {code}")

    def decode(self, state: np.ndarray, code: str) -> np.ndarray:
        """Decode back to logical qubit."""
        if code == "bit_flip":
            # Project onto logical subspace and extract alpha, beta
            alpha = state[0b000]
            beta = state[0b111]
            norm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
            if norm < 1e-15:
                return np.array([0, 0], dtype=complex)
            return np.array([alpha, beta], dtype=complex) / norm
        elif code == "phase_flip":
            # Transform to computational basis first (H on each qubit)
            n_q = 3
            s = state.copy()
            for q in range(n_q):
                s = _apply_gate(s, n_q, "H", [q])
            alpha = s[0b000]
            beta = s[0b111]
            norm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
            if norm < 1e-15:
                return np.array([0, 0], dtype=complex)
            return np.array([alpha, beta], dtype=complex) / norm
        elif code == "shor":
            alpha = state[0]
            n_q = 9
            idx_111 = 0b111111111
            beta = state[idx_111] if idx_111 < len(state) else 0.0
            # |0_L> coefficient from the +++ pattern
            idx_0L = 0  # |000000000>
            idx_1L = (1 << n_q) - 1  # |111111111>
            coeff_0 = state[idx_0L]
            coeff_1 = state[idx_1L]
            # Normalization accounting for code structure
            alpha_est = coeff_0 * (2 * np.sqrt(2))
            beta_est = coeff_1 * (2 * np.sqrt(2))
            norm = np.sqrt(abs(alpha_est) ** 2 + abs(beta_est) ** 2)
            if norm < 1e-15:
                return np.array([0, 0], dtype=complex)
            return np.array([alpha_est, beta_est], dtype=complex) / norm
        elif code == "steane":
            # Project onto logical subspace
            codewords_0 = [
                0b0000000, 0b1010101, 0b0110011, 0b1100110,
                0b0001111, 0b1011010, 0b0111100, 0b1101001,
            ]
            alpha = sum(state[cw] for cw in codewords_0)
            codewords_1 = [cw ^ 0b1111111 for cw in codewords_0]
            beta = sum(state[cw] for cw in codewords_1)
            # Normalize (each logical state has 8 codewords with equal amplitude)
            alpha /= np.sqrt(8)
            beta /= np.sqrt(8)
            norm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
            if norm < 1e-15:
                return np.array([0, 0], dtype=complex)
            return np.array([alpha, beta], dtype=complex) / norm
        raise ValueError(f"Unknown code: {code}")

    # ---- Bit-flip syndrome / correction ------------------------------------

    def _syndrome_bit_flip(self, state: np.ndarray) -> dict:
        """Syndrome for 3-qubit bit-flip code."""
        probs = np.abs(state) ** 2
        # Syndrome bits: s1 = q0 XOR q1, s2 = q1 XOR q2
        s1 = 0.0
        s2 = 0.0
        for i in range(8):
            b0 = (i >> 2) & 1
            b1 = (i >> 1) & 1
            b2 = i & 1
            s1 += probs[i] * (b0 ^ b1)
            s2 += probs[i] * (b1 ^ b2)
        s1_bit = 1 if s1 > 0.5 else 0
        s2_bit = 1 if s2 > 0.5 else 0
        syndrome = (s1_bit, s2_bit)
        lookup = {
            (0, 0): (None, "No error detected."),
            (1, 0): (0, "Error on qubit 0 (q0 differs from q1 but q1 matches q2)."),
            (1, 1): (1, "Error on qubit 1 (q1 differs from both q0 and q2)."),
            (0, 1): (2, "Error on qubit 2 (q2 differs from q1 but q0 matches q1)."),
        }
        loc, interp = lookup[syndrome]
        correction = f"Apply X to qubit {loc}." if loc is not None else "No correction needed."
        return {
            "syndrome": syndrome,
            "interpretation": interp,
            "error_location": loc,
            "correction": correction,
        }

    def _correct_bit_flip(self, state: np.ndarray, syndrome: tuple) -> np.ndarray:
        lookup = {(0, 0): None, (1, 0): 0, (1, 1): 1, (0, 1): 2}
        loc = lookup.get(syndrome)
        if loc is None:
            return state.copy()
        return self.inject_error(state, "X", loc, "bit_flip")

    # ---- Phase-flip syndrome / correction ----------------------------------

    def _syndrome_phase_flip(self, state: np.ndarray) -> dict:
        """Syndrome for 3-qubit phase-flip code (work in Hadamard basis)."""
        # Transform to computational basis
        s = state.copy()
        for q in range(3):
            s = _apply_gate(s, 3, "H", [q])
        return self._syndrome_bit_flip(s)

    def _correct_phase_flip(self, state: np.ndarray, syndrome: tuple) -> np.ndarray:
        lookup = {(0, 0): None, (1, 0): 0, (1, 1): 1, (0, 1): 2}
        loc = lookup.get(syndrome)
        if loc is None:
            return state.copy()
        return self.inject_error(state, "Z", loc, "phase_flip")

    # ---- Shor code syndrome / correction -----------------------------------

    def _syndrome_shor(self, state: np.ndarray) -> dict:
        """Simplified Shor code syndrome."""
        # The Shor code has 8 syndrome bits.  For educational purposes we
        # check each 3-qubit block for bit-flip and the inter-block phase.
        n_q = 9
        probs = np.abs(state) ** 2

        bit_syndromes = []
        for block in range(3):
            offset = block * 3
            s1 = 0.0
            s2 = 0.0
            for i in range(2 ** n_q):
                b0 = (i >> (n_q - 1 - offset)) & 1
                b1 = (i >> (n_q - 1 - offset - 1)) & 1
                b2 = (i >> (n_q - 1 - offset - 2)) & 1
                s1 += probs[i] * (b0 ^ b1)
                s2 += probs[i] * (b1 ^ b2)
            bit_syndromes.append((1 if s1 > 0.5 else 0, 1 if s2 > 0.5 else 0))

        syndrome = tuple(s for pair in bit_syndromes for s in pair)
        # Determine error location
        error_loc = None
        error_type = "none"
        for block_idx, (s1, s2) in enumerate(bit_syndromes):
            if (s1, s2) != (0, 0):
                lookup = {(1, 0): 0, (1, 1): 1, (0, 1): 2}
                within = lookup.get((s1, s2), None)
                if within is not None:
                    error_loc = block_idx * 3 + within
                    error_type = "X"
        interp = f"Bit-flip error detected on qubit {error_loc}." if error_loc is not None else "No bit-flip error detected."
        correction = f"Apply X to qubit {error_loc}." if error_loc is not None else "No correction needed."
        return {
            "syndrome": syndrome,
            "interpretation": interp,
            "error_location": error_loc,
            "correction": correction,
        }

    def _correct_shor(self, state: np.ndarray, syndrome: tuple) -> np.ndarray:
        bit_syndromes = [(syndrome[i], syndrome[i + 1]) for i in range(0, 6, 2)]
        for block_idx, (s1, s2) in enumerate(bit_syndromes):
            if (s1, s2) != (0, 0):
                lookup = {(1, 0): 0, (1, 1): 1, (0, 1): 2}
                within = lookup.get((s1, s2))
                if within is not None:
                    loc = block_idx * 3 + within
                    state = self.inject_error(state, "X", loc, "shor")
        return state

    # ---- Steane code syndrome / correction ---------------------------------

    def _syndrome_steane(self, state: np.ndarray) -> dict:
        """Syndrome for [[7,1,3]] Steane code."""
        n_q = 7
        probs = np.abs(state) ** 2
        # X-stabilizers (detect Z errors): check parity of qubit subsets
        x_checks = [
            [0, 2, 4, 6],  # S1
            [1, 2, 5, 6],  # S2
            [3, 4, 5, 6],  # S3
        ]
        # Z-stabilizers (detect X errors): same pattern
        z_checks = [
            [0, 2, 4, 6],
            [1, 2, 5, 6],
            [3, 4, 5, 6],
        ]

        x_syndrome = []
        for check_qubits in x_checks:
            parity = 0.0
            for i in range(2 ** n_q):
                p = 0
                for q in check_qubits:
                    p ^= (i >> (n_q - 1 - q)) & 1
                parity += probs[i] * p
            x_syndrome.append(1 if parity > 0.5 else 0)

        z_syndrome = []
        # For Z error detection we need phase information
        # Use state overlap approach: check if stabilizer expectation is negative
        for check_qubits in z_checks:
            # Apply X to check qubits and see if state maps to itself
            s = state.copy()
            for q in check_qubits:
                s = _apply_gate(s, n_q, "X", [q])
            overlap = np.real(np.vdot(state, s))
            z_syndrome.append(0 if overlap > 0 else 1)

        syndrome = tuple(x_syndrome + z_syndrome)

        # Decode syndrome to error location
        x_syn_val = x_syndrome[0] + 2 * x_syndrome[1] + 4 * x_syndrome[2]
        z_syn_val = z_syndrome[0] + 2 * z_syndrome[1] + 4 * z_syndrome[2]

        error_loc = None
        error_type = None
        if x_syn_val > 0:
            error_loc = x_syn_val - 1
            error_type = "Z"
        if z_syn_val > 0:
            loc = z_syn_val - 1
            if error_loc is not None and loc == error_loc:
                error_type = "Y"
            elif error_loc is None:
                error_loc = loc
                error_type = "X"

        if error_loc is not None:
            interp = f"{error_type} error on qubit {error_loc}."
            correction = f"Apply {error_type} to qubit {error_loc}."
        else:
            interp = "No error detected."
            correction = "No correction needed."

        return {
            "syndrome": syndrome,
            "interpretation": interp,
            "error_location": error_loc,
            "correction": correction,
        }

    def _correct_steane(self, state: np.ndarray, syndrome: tuple) -> np.ndarray:
        x_syndrome = list(syndrome[:3])
        z_syndrome = list(syndrome[3:])
        x_syn_val = x_syndrome[0] + 2 * x_syndrome[1] + 4 * x_syndrome[2]
        z_syn_val = z_syndrome[0] + 2 * z_syndrome[1] + 4 * z_syndrome[2]
        if x_syn_val > 0:
            loc = x_syn_val - 1
            state = self.inject_error(state, "Z", loc, "steane")
        if z_syn_val > 0:
            loc = z_syn_val - 1
            state = self.inject_error(state, "X", loc, "steane")
        return state


# ---------------------------------------------------------------------------
# SyndromeVisualizer
# ---------------------------------------------------------------------------

class SyndromeVisualizer:
    """ASCII visualization of syndrome measurement."""

    def draw_syndrome_table(self, code: str) -> str:
        """Show all syndromes and their error associations."""
        if code == "bit_flip":
            lines = [
                "3-Qubit Bit-Flip Code Syndrome Table",
                "=" * 45,
                f"{'Syndrome':>10} | {'Error Location':>15} | {'Correction'}",
                "-" * 45,
                f"{'(0, 0)':>10} | {'None':>15} | No correction",
                f"{'(1, 0)':>10} | {'Qubit 0':>15} | Apply X to q0",
                f"{'(1, 1)':>10} | {'Qubit 1':>15} | Apply X to q1",
                f"{'(0, 1)':>10} | {'Qubit 2':>15} | Apply X to q2",
            ]
            return "\n".join(lines)
        elif code == "phase_flip":
            lines = [
                "3-Qubit Phase-Flip Code Syndrome Table",
                "=" * 45,
                f"{'Syndrome':>10} | {'Error Location':>15} | {'Correction'}",
                "-" * 45,
                f"{'(0, 0)':>10} | {'None':>15} | No correction",
                f"{'(1, 0)':>10} | {'Qubit 0':>15} | Apply Z to q0",
                f"{'(1, 1)':>10} | {'Qubit 1':>15} | Apply Z to q1",
                f"{'(0, 1)':>10} | {'Qubit 2':>15} | Apply Z to q2",
            ]
            return "\n".join(lines)
        elif code == "steane":
            lines = [
                "[[7,1,3]] Steane Code Syndrome Table",
                "=" * 55,
                "X-syndrome (3 bits) detects Z errors",
                "Z-syndrome (3 bits) detects X errors",
                f"{'X-syn':>7} {'Z-syn':>7} | {'Error':>20} | {'Correction'}",
                "-" * 55,
                f"{'000':>7} {'000':>7} | {'None':>20} | No correction",
            ]
            for i in range(1, 8):
                bits = format(i, "03b")
                lines.append(f"  {bits:>5} {'000':>7} | {'Z on qubit ' + str(i - 1):>20} | Apply Z to q{i - 1}")
            return "\n".join(lines)
        return f"Syndrome table not available for code '{code}'."

    def draw_error_chain(self, state_history: List[dict]) -> str:
        """Show encode -> error -> syndrome -> correct -> decode chain."""
        lines = ["Error Correction Chain", "=" * 50]
        for i, entry in enumerate(state_history):
            stage = entry.get("stage", f"Step {i}")
            info = entry.get("info", "")
            lines.append(f"  [{stage}] {info}")
            if "state" in entry:
                state = entry["state"]
                probs = np.abs(state) ** 2
                n_q = int(np.log2(len(state)))
                nonzero = [(format(j, f"0{n_q}b"), probs[j]) for j in range(len(probs)) if probs[j] > 0.01]
                for label, p in nonzero[:6]:
                    lines.append(f"    |{label}> : {p:.4f}")
            lines.append("    |")
            lines.append("    v")
        return "\n".join(lines[:-2])  # Remove trailing arrow


# ---------------------------------------------------------------------------
# DecoderRace
# ---------------------------------------------------------------------------

class DecoderRace:
    """Compare decoder performance on random errors."""

    def race(self, code: str = "bit_flip", n_errors: int = 100, seed: int = 42) -> dict:
        """Run random errors through the playground and measure success rate.

        Returns dict with 'code', 'n_errors', 'successes', 'failures', 'success_rate'.
        """
        rng = np.random.default_rng(seed)
        pg = QECPlayground()
        successes = 0
        failures = 0

        for _ in range(n_errors):
            # Random logical state
            alpha = rng.normal() + 1j * rng.normal()
            beta = rng.normal() + 1j * rng.normal()
            norm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
            alpha /= norm
            beta /= norm
            logical = np.array([alpha, beta], dtype=complex)

            encoded = pg.encode(logical, code)
            n_q = int(np.log2(len(encoded)))

            # Random single-qubit error
            error_types = ["X"] if code == "bit_flip" else ["Z"] if code == "phase_flip" else ["X", "Z"]
            etype = rng.choice(error_types)
            equbit = int(rng.integers(0, n_q))

            corrupted = pg.inject_error(encoded, etype, equbit, code)
            syndrome_info = pg.measure_syndrome(corrupted, code)
            corrected = pg.correct(corrupted, syndrome_info["syndrome"], code)
            decoded = pg.decode(corrected, code)

            # Check fidelity (up to global phase)
            overlap = abs(np.vdot(logical, decoded)) ** 2
            if overlap > 0.99:
                successes += 1
            else:
                failures += 1

        return {
            "code": code,
            "n_errors": n_errors,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / n_errors,
        }


# ---------------------------------------------------------------------------
# Pre-built QEC lessons
# ---------------------------------------------------------------------------

class BitFlipLesson:
    """3-qubit bit-flip code lesson."""

    def run(self) -> TutorialResult:
        return CircuitLesson(
            title="3-Qubit Bit-Flip Code",
            description="Encode a logical qubit, inject a bit-flip error, detect it via syndrome, and correct.",
            n_qubits=3,
            steps=[
                LessonStep(
                    instruction="Encode logical |1> into 3-qubit bit-flip code: |1_L> = |111>.",
                    hint="Apply X to all three qubits.",
                    gate_sequence=[("X", [0], {}), ("X", [1], {}), ("X", [2], {})],
                    explanation="The logical |1> is encoded as |111>. Any single bit-flip will be detectable.",
                ),
                LessonStep(
                    instruction="Inject a bit-flip error on qubit 1.",
                    hint="Apply X to qubit 1.",
                    gate_sequence=[("X", [1], {})],
                    explanation="State is now |101>. One qubit disagrees with the majority.",
                ),
                LessonStep(
                    instruction="Syndrome measurement would reveal (1,1) -> error on qubit 1. Correct by flipping qubit 1 back.",
                    hint="Apply X to qubit 1.",
                    gate_sequence=[("X", [1], {})],
                    explanation="State is restored to |111> = |1_L>. The error is corrected!",
                ),
            ],
        ).run()


class PhaseFlipLesson:
    """3-qubit phase-flip code lesson."""

    def run(self) -> TutorialResult:
        return CircuitLesson(
            title="3-Qubit Phase-Flip Code",
            description="Encode into phase-flip code, inject phase error, detect and correct.",
            n_qubits=3,
            steps=[
                LessonStep(
                    instruction="Encode |0_L> = |+++>. Apply H to all qubits.",
                    hint="H|0> = |+>.",
                    gate_sequence=[("H", [0], {}), ("H", [1], {}), ("H", [2], {})],
                    explanation="State is |+++>. This encodes logical |0> in the phase-flip code.",
                ),
                LessonStep(
                    instruction="Inject a phase-flip error on qubit 2.",
                    hint="Z|+> = |->.",
                    gate_sequence=[("Z", [2], {})],
                    explanation="State is |++->. Qubit 2 has been flipped in the Hadamard basis.",
                ),
                LessonStep(
                    instruction="Detect and correct: apply Z to qubit 2.",
                    hint="Z|-> = |+>.",
                    gate_sequence=[("Z", [2], {})],
                    explanation="State is restored to |+++> = |0_L>. Phase error corrected!",
                ),
            ],
        ).run()


class ShorCodeLesson:
    """9-qubit Shor code lesson (corrects any single-qubit error)."""

    def run(self) -> TutorialResult:
        return CircuitLesson(
            title="9-Qubit Shor Code",
            description="The Shor code combines bit-flip and phase-flip protection. It uses 9 physical qubits to protect 1 logical qubit against any single-qubit error.",
            n_qubits=3,
            steps=[
                LessonStep(
                    instruction="Understand the structure: 3 blocks of 3 qubits each. Within each block, bit-flip protection. Between blocks, phase-flip protection.",
                    hint="Think of it as a phase-flip code where each 'qubit' is itself a bit-flip code.",
                    gate_sequence=[("H", [0], {}), ("H", [1], {}), ("H", [2], {})],
                    explanation="The Shor code concatenates: inner code = 3-qubit bit-flip, outer code = 3-qubit phase-flip. This is the first code ever discovered that can correct arbitrary single-qubit errors (1995).",
                ),
                LessonStep(
                    instruction="Simulate error detection within one block.",
                    hint="A bit-flip on any qubit within a block is caught by that block's syndrome.",
                    gate_sequence=[("X", [1], {}), ("X", [1], {})],
                    explanation="Each block of 3 qubits uses majority voting. A phase error across blocks is detected by comparing block parities. Together: full single-qubit error correction.",
                ),
            ],
        ).run()


class SteaneCodeLesson:
    """7-qubit Steane code lesson (CSS code)."""

    def run(self) -> TutorialResult:
        return CircuitLesson(
            title="[[7,1,3]] Steane Code",
            description="The Steane code is a CSS (Calderbank-Shor-Steane) code built from the classical [7,4,3] Hamming code. It encodes 1 logical qubit in 7 physical qubits and corrects any single-qubit error.",
            n_qubits=3,
            steps=[
                LessonStep(
                    instruction="Study the stabilizer generators. The Steane code has 6 stabilizers: 3 X-type and 3 Z-type.",
                    hint="X-stabilizers detect Z errors, Z-stabilizers detect X errors.",
                    gate_sequence=[("H", [0], {}), ("H", [1], {}), ("H", [2], {})],
                    explanation="X-stabilizers: X on qubits {0,2,4,6}, {1,2,5,6}, {3,4,5,6}. Z-stabilizers: same qubit sets but with Z. The 3-bit syndrome from each set identifies which qubit has an error (binary code: syndrome = qubit_index + 1).",
                ),
                LessonStep(
                    instruction="The Steane code is transversal: logical X = X^7, logical Z = Z^7, logical H = H^7.",
                    hint="Transversal gates don't spread errors between qubits.",
                    gate_sequence=[("X", [0], {}), ("X", [1], {}), ("X", [2], {})],
                    explanation="Transversality is key for fault tolerance. Each logical gate operates qubit-by-qubit, so a single fault stays a single fault. The Steane code supports transversal {H, S, CNOT} -- enough for Clifford group operations.",
                ),
            ],
        ).run()


class SurfaceCodeLesson:
    """Intro to surface codes (conceptual, small example)."""

    def run(self) -> TutorialResult:
        return CircuitLesson(
            title="Introduction to Surface Codes",
            description="Surface codes arrange qubits on a 2D lattice with local stabilizer checks. They have the highest known threshold (~1%) among topological codes.",
            n_qubits=2,
            steps=[
                LessonStep(
                    instruction="Visualize: data qubits on edges, X-stabilizers on faces, Z-stabilizers on vertices of a planar lattice.",
                    hint="A distance-d surface code uses O(d^2) physical qubits.",
                    gate_sequence=[("H", [0], {}), ("CNOT", [0, 1], {})],
                    explanation="For distance d=3 (smallest useful): 9 data qubits + 8 syndrome qubits = 17 total. Errors form chains; only chains connecting opposite boundaries cause logical errors.",
                ),
                LessonStep(
                    instruction="Error correction: measure all stabilizers each round. Chain-like errors are corrected by minimum-weight perfect matching (MWPM).",
                    hint="The threshold theorem: if physical error rate < ~1%, logical error rate decreases exponentially with d.",
                    gate_sequence=[("H", [0], {})],
                    explanation="Surface codes are the leading candidate for scalable quantum computing. Google's 2023 experiment showed below-threshold performance. Key advantage: only nearest-neighbor interactions required.",
                ),
            ],
        ).run()
