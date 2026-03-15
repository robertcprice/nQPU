"""Native gate set for neutral-atom quantum computers using Rydberg blockade.

Implements the native gate set for neutral-atom platforms following:
- Levine et al., Phys. Rev. Lett. 123, 170503 (2019) [CZ gate]
- Jandura & Pupillo, Phys. Rev. Lett. 130, 193602 (2023) [time-optimal pulses]
- Evered et al., Nature 622, 268 (2023) [high-fidelity CZ]
- Graham et al., Nature 604, 457 (2022) [native multi-qubit gates]

Native gate set
---------------
1. **Rz(theta)**: Virtual Z rotation via frame tracking (zero error).
2. **Rxy(theta, phi)**: Arbitrary single-qubit rotation in the XY plane,
   driven by focused or global Raman/microwave beams.
3. **CZ**: Controlled-Z via Rydberg blockade (native two-qubit gate).
   Implemented as: pi-pulse(control) -> 2*pi-pulse(target) -> pi-pulse(control).
4. **CCZ**: Native three-qubit gate enabled by Rydberg blockade extending
   to three atoms (Graham et al., 2022).
5. **Global rotation**: Simultaneous rotation of all atoms via global beam.

Compilation strategies decompose standard gates into the native set to
minimise the number of CZ gates, since Rydberg entangling operations
dominate error budgets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class NativeGateType(Enum):
    """Enumeration of native neutral-atom gate types."""

    RZ = auto()  # Virtual Z rotation (zero error)
    RXY = auto()  # Rabi rotation in XY plane: R(theta, phi)
    RX = auto()  # X rotation (special case of RXY with phi=0)
    RY = auto()  # Y rotation (special case of RXY with phi=pi/2)
    CZ = auto()  # Rydberg blockade CZ gate
    CCZ = auto()  # Native three-qubit CCZ gate
    GLOBAL_R = auto()  # Global rotation applied to all qubits


@dataclass(frozen=True)
class GateInstruction:
    """A single gate instruction in the native gate set.

    Parameters
    ----------
    gate_type : NativeGateType
        Which native gate to apply.
    qubits : tuple[int, ...]
        Target qubit indices.
    params : tuple[float, ...]
        Gate parameters (angles in radians).
    """

    gate_type: NativeGateType
    qubits: tuple[int, ...]
    params: tuple[float, ...]

    def __repr__(self) -> str:
        param_str = ", ".join(f"{p:.4f}" for p in self.params)
        qubit_str = ", ".join(str(q) for q in self.qubits)
        return f"{self.gate_type.name}({param_str}) @ q[{qubit_str}]"


class NeutralAtomGateSet:
    """Native gate set for neutral-atom quantum computers.

    Native gates
    ------------
    Rz(theta)
        Z rotation via software frame tracking.  Zero physical error.

    Rxy(theta, phi)
        Single-qubit rotation by angle theta about the axis
        cos(phi)*X + sin(phi)*Y.  Physically realised via focused
        Raman or microwave beam.

    CZ
        Controlled-Z gate via three-pulse Rydberg blockade sequence:
        (1) pi-pulse excites control to Rydberg state,
        (2) 2*pi-pulse on target (blockaded if control is |1>),
        (3) pi-pulse de-excites control.

    CCZ
        Native three-qubit Toffoli-phase gate using the multi-body
        Rydberg blockade.  Requires all three atoms within mutual
        blockade radius.
    """

    # ==================================================================
    # Single-qubit gate matrices
    # ==================================================================

    @staticmethod
    def rz_matrix(theta: float) -> np.ndarray:
        """Z rotation matrix Rz(theta) = exp(-i*theta/2 * Z).

        Parameters
        ----------
        theta : float
            Rotation angle in radians.
        """
        return np.array(
            [
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)],
            ],
            dtype=np.complex128,
        )

    @staticmethod
    def rx_matrix(theta: float) -> np.ndarray:
        """X rotation matrix Rx(theta) = exp(-i*theta/2 * X)."""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [[c, -1j * s], [-1j * s, c]], dtype=np.complex128
        )

    @staticmethod
    def ry_matrix(theta: float) -> np.ndarray:
        """Y rotation matrix Ry(theta) = exp(-i*theta/2 * Y)."""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def rxy_matrix(theta: float, phi: float) -> np.ndarray:
        """Arbitrary single-qubit rotation Rxy(theta, phi).

        R(theta, phi) = exp(-i * theta/2 * (cos(phi)*X + sin(phi)*Y))

        Parameters
        ----------
        theta : float
            Rotation angle in radians.
        phi : float
            Axis angle in the XY plane, in radians.
        """
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return np.array(
            [
                [c, -1j * s * np.exp(-1j * phi)],
                [-1j * s * np.exp(1j * phi), c],
            ],
            dtype=np.complex128,
        )

    # ==================================================================
    # Two-qubit gate matrices
    # ==================================================================

    @staticmethod
    def cz_matrix() -> np.ndarray:
        """CZ gate matrix (4x4).

        CZ = diag(1, 1, 1, -1)

        The native Rydberg blockade gate: when both atoms are in |1>,
        the Rydberg blockade prevents the target's 2*pi pulse from
        completing, acquiring a pi phase.
        """
        return np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)

    # ==================================================================
    # Three-qubit gate matrices
    # ==================================================================

    @staticmethod
    def ccz_matrix() -> np.ndarray:
        """CCZ gate matrix (8x8).

        CCZ = diag(1, 1, 1, 1, 1, 1, 1, -1)

        Native three-qubit gate enabled by Rydberg blockade.  The
        |111> state acquires a -1 phase because all three atoms
        mutually blockade.

        This is equivalent to the Toffoli gate (up to Hadamards on
        the target), but is natively available without decomposition
        in neutral-atom hardware.
        """
        phases = np.ones(8, dtype=np.complex128)
        phases[7] = -1.0
        return np.diag(phases)

    # ==================================================================
    # Gate compilation: standard gates -> native gates
    # ==================================================================

    @staticmethod
    def compile_h(qubit: int) -> list[GateInstruction]:
        """Compile Hadamard into native gates.

        H = Rz(pi) * Ry(pi/2)

        Uses one virtual Rz and one physical Ry rotation.
        """
        return [
            GateInstruction(NativeGateType.RY, (qubit,), (math.pi / 2,)),
            GateInstruction(NativeGateType.RZ, (qubit,), (math.pi,)),
        ]

    @staticmethod
    def compile_x(qubit: int) -> list[GateInstruction]:
        """X = Rxy(pi, 0)."""
        return [GateInstruction(NativeGateType.RXY, (qubit,), (math.pi, 0.0))]

    @staticmethod
    def compile_y(qubit: int) -> list[GateInstruction]:
        """Y = Rxy(pi, pi/2)."""
        return [
            GateInstruction(
                NativeGateType.RXY, (qubit,), (math.pi, math.pi / 2)
            )
        ]

    @staticmethod
    def compile_z(qubit: int) -> list[GateInstruction]:
        """Z = Rz(pi) (virtual gate)."""
        return [GateInstruction(NativeGateType.RZ, (qubit,), (math.pi,))]

    @staticmethod
    def compile_rx(qubit: int, theta: float) -> list[GateInstruction]:
        """Rx(theta) = Rxy(theta, 0)."""
        return [GateInstruction(NativeGateType.RXY, (qubit,), (theta, 0.0))]

    @staticmethod
    def compile_ry(qubit: int, theta: float) -> list[GateInstruction]:
        """Ry(theta) = Rxy(theta, pi/2)."""
        return [
            GateInstruction(
                NativeGateType.RXY, (qubit,), (theta, math.pi / 2)
            )
        ]

    @staticmethod
    def compile_rz(qubit: int, theta: float) -> list[GateInstruction]:
        """Rz(theta) -- already native (virtual gate)."""
        return [GateInstruction(NativeGateType.RZ, (qubit,), (theta,))]

    @staticmethod
    def compile_cnot(control: int, target: int) -> list[GateInstruction]:
        """Compile CNOT into native gates.

        CNOT = (I x H) * CZ * (I x H)

        Uses exactly 1 CZ gate plus 2 Hadamards on the target.

        Parameters
        ----------
        control : int
            Control qubit index.
        target : int
            Target qubit index.
        """
        return [
            # H on target
            GateInstruction(NativeGateType.RY, (target,), (math.pi / 2,)),
            GateInstruction(NativeGateType.RZ, (target,), (math.pi,)),
            # CZ
            GateInstruction(NativeGateType.CZ, (control, target), ()),
            # H on target
            GateInstruction(NativeGateType.RY, (target,), (math.pi / 2,)),
            GateInstruction(NativeGateType.RZ, (target,), (math.pi,)),
        ]

    @staticmethod
    def compile_cz(control: int, target: int) -> list[GateInstruction]:
        """CZ -- already native.

        Parameters
        ----------
        control : int
            Control qubit index.
        target : int
            Target qubit index.
        """
        return [GateInstruction(NativeGateType.CZ, (control, target), ())]

    @staticmethod
    def compile_ccz(
        qubit_a: int, qubit_b: int, qubit_c: int
    ) -> list[GateInstruction]:
        """CCZ -- native three-qubit gate.

        Parameters
        ----------
        qubit_a, qubit_b, qubit_c : int
            The three qubit indices (all must be within mutual blockade
            radius).
        """
        return [
            GateInstruction(
                NativeGateType.CCZ, (qubit_a, qubit_b, qubit_c), ()
            )
        ]

    @staticmethod
    def compile_toffoli(
        control_a: int, control_b: int, target: int
    ) -> list[GateInstruction]:
        """Compile Toffoli gate using native CCZ + Hadamards.

        Toffoli = (I x I x H) * CCZ * (I x I x H)

        This is dramatically more efficient than the trapped-ion
        decomposition which requires 6 entangling gates, since
        neutral atoms can natively perform CCZ.

        Parameters
        ----------
        control_a, control_b : int
            Control qubit indices.
        target : int
            Target qubit index.
        """
        return [
            # H on target
            GateInstruction(NativeGateType.RY, (target,), (math.pi / 2,)),
            GateInstruction(NativeGateType.RZ, (target,), (math.pi,)),
            # Native CCZ
            GateInstruction(
                NativeGateType.CCZ, (control_a, control_b, target), ()
            ),
            # H on target
            GateInstruction(NativeGateType.RY, (target,), (math.pi / 2,)),
            GateInstruction(NativeGateType.RZ, (target,), (math.pi,)),
        ]

    @staticmethod
    def compile_swap(qubit_a: int, qubit_b: int) -> list[GateInstruction]:
        """Compile SWAP = 3 CNOTs = 3 CZ gates + single-qubit gates."""
        instructions: list[GateInstruction] = []
        instructions.extend(NeutralAtomGateSet.compile_cnot(qubit_a, qubit_b))
        instructions.extend(NeutralAtomGateSet.compile_cnot(qubit_b, qubit_a))
        instructions.extend(NeutralAtomGateSet.compile_cnot(qubit_a, qubit_b))
        return instructions

    # ==================================================================
    # Arbitrary unitary decomposition
    # ==================================================================

    @staticmethod
    def compile_arbitrary_unitary(
        u: np.ndarray, qubit: int
    ) -> list[GateInstruction]:
        """Decompose an arbitrary single-qubit unitary via ZYZ decomposition.

        Any U in SU(2) can be written as:
            U = exp(i*alpha) * Rz(beta) * Ry(gamma) * Rz(delta)

        Parameters
        ----------
        u : np.ndarray
            2x2 unitary matrix.
        qubit : int
            Target qubit index.

        Returns
        -------
        list[GateInstruction]
            Sequence of native Rz and Ry gates.
        """
        if u.shape != (2, 2):
            raise ValueError("Expected 2x2 unitary matrix")

        # Extract ZYZ angles
        det = np.linalg.det(u)
        phase = np.angle(det) / 2.0
        u_su2 = u * np.exp(-1j * phase)

        cos_gamma_half = min(1.0, max(0.0, abs(u_su2[0, 0])))
        sin_gamma_half = min(1.0, max(0.0, abs(u_su2[1, 0])))
        gamma = 2.0 * math.atan2(sin_gamma_half, cos_gamma_half)

        if abs(cos_gamma_half) < 1e-10:
            beta = 0.0
            delta = np.angle(u_su2[1, 0]) - np.angle(u_su2[0, 1]) - math.pi
        elif abs(sin_gamma_half) < 1e-10:
            beta = 0.0
            delta = 2.0 * np.angle(u_su2[1, 1])
        else:
            beta_plus_delta = 2.0 * np.angle(u_su2[1, 1])
            beta_minus_delta = 2.0 * np.angle(u_su2[1, 0])
            beta = (beta_plus_delta + beta_minus_delta) / 2.0
            delta = (beta_plus_delta - beta_minus_delta) / 2.0

        instructions: list[GateInstruction] = []
        if abs(delta) > 1e-10:
            instructions.append(
                GateInstruction(NativeGateType.RZ, (qubit,), (delta,))
            )
        if abs(gamma) > 1e-10:
            instructions.append(
                GateInstruction(NativeGateType.RY, (qubit,), (gamma,))
            )
        if abs(beta) > 1e-10:
            instructions.append(
                GateInstruction(NativeGateType.RZ, (qubit,), (beta,))
            )
        return instructions

    # ==================================================================
    # Gate count analysis
    # ==================================================================

    @staticmethod
    def cz_gate_count(instructions: list[GateInstruction]) -> int:
        """Count the number of CZ (entangling) gates in an instruction list."""
        return sum(
            1
            for inst in instructions
            if inst.gate_type == NativeGateType.CZ
        )

    @staticmethod
    def ccz_gate_count(instructions: list[GateInstruction]) -> int:
        """Count the number of CCZ (three-qubit) gates."""
        return sum(
            1
            for inst in instructions
            if inst.gate_type == NativeGateType.CCZ
        )

    @staticmethod
    def entangling_gate_count(instructions: list[GateInstruction]) -> int:
        """Count all entangling gates (CZ + CCZ)."""
        return sum(
            1
            for inst in instructions
            if inst.gate_type in (NativeGateType.CZ, NativeGateType.CCZ)
        )

    @staticmethod
    def single_qubit_gate_count(instructions: list[GateInstruction]) -> int:
        """Count single-qubit gates in an instruction list."""
        return sum(
            1
            for inst in instructions
            if inst.gate_type
            in (
                NativeGateType.RZ,
                NativeGateType.RXY,
                NativeGateType.RX,
                NativeGateType.RY,
                NativeGateType.GLOBAL_R,
            )
        )
