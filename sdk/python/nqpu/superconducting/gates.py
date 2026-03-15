"""Native gate set and compilation for superconducting transmon processors.

Each vendor family has a different native gate set:
    - IBM:     {Rz, SX, X, ECR}
    - Google:  {Rz, sqrt(iSWAP), Phased-XZ}
    - Rigetti: {Rz, RX, CZ}

Standard gates (H, CNOT, CZ, etc.) are compiled into the native set
for accurate noise simulation and resource counting.

References:
    - Sheldon et al., PRA 93, 060302 (2016) [ECR gate]
    - Arute et al., Nature 574, 505 (2019) [sqrt(iSWAP)]
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .chip import NativeGateFamily


class NativeGateType(enum.Enum):
    """Native gate types for superconducting hardware."""
    RZ = "rz"
    SX = "sx"
    X = "x"
    ECR = "ecr"
    SQRT_ISWAP = "sqrt_iswap"
    CZ = "cz"


@dataclass(frozen=True)
class GateInstruction:
    """A native gate instruction in the compiled circuit.

    Parameters
    ----------
    gate_type : NativeGateType
        The native gate being applied.
    qubits : tuple of int
        Target qubit(s).
    angle : float, optional
        Rotation angle for parametric gates (Rz).
    """

    gate_type: NativeGateType
    qubits: tuple[int, ...]
    angle: float = 0.0

    @property
    def is_two_qubit(self) -> bool:
        return self.gate_type in (NativeGateType.ECR, NativeGateType.SQRT_ISWAP,
                                   NativeGateType.CZ)


class TransmonGateSet:
    """Compiler from standard gates to native transmon gate set.

    Decomposes standard gates (H, CNOT, CZ, SWAP, Rx, Ry, etc.) into
    the native gate family of the target processor.
    """

    def __init__(self, native_2q: str = "ecr") -> None:
        self.native_2q = native_2q

    # ------------------------------------------------------------------
    # Single-qubit decompositions (shared across vendors)
    # ------------------------------------------------------------------

    def compile_h(self, qubit: int) -> list[GateInstruction]:
        """H = Rz(pi) SX Rz(pi)."""
        return [
            GateInstruction(NativeGateType.RZ, (qubit,), math.pi),
            GateInstruction(NativeGateType.SX, (qubit,)),
            GateInstruction(NativeGateType.RZ, (qubit,), math.pi),
        ]

    def compile_x(self, qubit: int) -> list[GateInstruction]:
        return [GateInstruction(NativeGateType.X, (qubit,))]

    def compile_rx(self, qubit: int, theta: float) -> list[GateInstruction]:
        """Rx(theta) = Rz(-pi/2) SX Rz(pi/2) scaled."""
        return [
            GateInstruction(NativeGateType.RZ, (qubit,), -math.pi / 2),
            GateInstruction(NativeGateType.SX, (qubit,)),
            GateInstruction(NativeGateType.RZ, (qubit,), theta + math.pi / 2),
        ]

    def compile_ry(self, qubit: int, theta: float) -> list[GateInstruction]:
        """Ry(theta) = SX Rz(theta) SXdg."""
        return [
            GateInstruction(NativeGateType.SX, (qubit,)),
            GateInstruction(NativeGateType.RZ, (qubit,), theta),
            GateInstruction(NativeGateType.RZ, (qubit,), -math.pi / 2),
            GateInstruction(NativeGateType.SX, (qubit,)),
            GateInstruction(NativeGateType.RZ, (qubit,), math.pi / 2),
        ]

    def compile_rz(self, qubit: int, theta: float) -> list[GateInstruction]:
        return [GateInstruction(NativeGateType.RZ, (qubit,), theta)]

    # ------------------------------------------------------------------
    # Two-qubit decompositions (vendor-specific)
    # ------------------------------------------------------------------

    def compile_cnot(self, control: int, target: int) -> list[GateInstruction]:
        """Compile CNOT into native 2Q gate + single-qubit corrections."""
        if self.native_2q == "ecr":
            return self._cnot_via_ecr(control, target)
        elif self.native_2q == "sqrt_iswap":
            return self._cnot_via_sqrt_iswap(control, target)
        else:
            return self._cnot_via_cz(control, target)

    def compile_cz(self, q0: int, q1: int) -> list[GateInstruction]:
        """CZ = CNOT with Hadamards on target."""
        if self.native_2q == "cz":
            return [GateInstruction(NativeGateType.CZ, (q0, q1))]
        cnot = self.compile_cnot(q0, q1)
        return (
            self.compile_h(q1)
            + cnot
            + self.compile_h(q1)
        )

    def _cnot_via_ecr(self, ctrl: int, tgt: int) -> list[GateInstruction]:
        """CNOT via ECR: Rz(-pi/2)_t ECR Rz(pi/2)_c SX_c."""
        return [
            GateInstruction(NativeGateType.RZ, (tgt,), -math.pi / 2),
            GateInstruction(NativeGateType.ECR, (ctrl, tgt)),
            GateInstruction(NativeGateType.RZ, (ctrl,), math.pi / 2),
            GateInstruction(NativeGateType.SX, (ctrl,)),
        ]

    def _cnot_via_sqrt_iswap(self, ctrl: int, tgt: int) -> list[GateInstruction]:
        """CNOT via two sqrt(iSWAP) gates + single-qubit corrections."""
        return [
            GateInstruction(NativeGateType.RZ, (tgt,), math.pi / 2),
            GateInstruction(NativeGateType.SQRT_ISWAP, (ctrl, tgt)),
            GateInstruction(NativeGateType.RZ, (ctrl,), math.pi),
            GateInstruction(NativeGateType.SQRT_ISWAP, (ctrl, tgt)),
            GateInstruction(NativeGateType.RZ, (ctrl,), math.pi / 2),
            GateInstruction(NativeGateType.RZ, (tgt,), math.pi / 2),
        ]

    def _cnot_via_cz(self, ctrl: int, tgt: int) -> list[GateInstruction]:
        """CNOT = H_t CZ H_t."""
        return (
            self.compile_h(tgt)
            + [GateInstruction(NativeGateType.CZ, (ctrl, tgt))]
            + self.compile_h(tgt)
        )

    # ------------------------------------------------------------------
    # Gate matrices for ideal simulation
    # ------------------------------------------------------------------

    @staticmethod
    def h_matrix() -> np.ndarray:
        s = 1.0 / math.sqrt(2.0)
        return np.array([[s, s], [s, -s]], dtype=np.complex128)

    @staticmethod
    def x_matrix() -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)

    @staticmethod
    def sx_matrix() -> np.ndarray:
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]],
                               dtype=np.complex128)

    @staticmethod
    def rz_matrix(theta: float) -> np.ndarray:
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)],
        ], dtype=np.complex128)

    @staticmethod
    def cnot_matrix() -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=np.complex128)

    @staticmethod
    def cz_matrix() -> np.ndarray:
        return np.diag([1, 1, 1, -1]).astype(np.complex128)

    @staticmethod
    def swap_matrix() -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.complex128)
