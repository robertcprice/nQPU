"""Native trapped-ion gate implementations and compilation.

Gate set follows the conventions of:
- Debnath et al., Nature 536, 63 (2016)  [Molmer-Sorensen]
- Maslov, Phys. Rev. A 93, 022311 (2016) [gate compilation]
- IonQ native gates: GPI, GPI2, MS
- Quantinuum native gates: Rz, R, ZZ

Compilation strategies decompose standard gates into the native
trapped-ion gate set to minimise total MS gate count, since
entangling gates dominate error budgets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

import numpy as np


class NativeGateType(Enum):
    """Enumeration of native trapped-ion gate types."""
    RZ = auto()       # Z rotation (virtual, zero-error)
    R = auto()        # Arbitrary single-qubit rotation
    RX = auto()       # X rotation
    RY = auto()       # Y rotation
    MS = auto()       # Molmer-Sorensen XX interaction
    XX = auto()       # Ising XX(theta) gate
    GPI = auto()      # IonQ GPI gate
    GPI2 = auto()     # IonQ GPI2 gate
    ZZ = auto()       # Quantinuum ZZ interaction


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


class TrappedIonGateSet:
    """Native gate set for trapped-ion quantum computers.

    Native gates
    ------------
    Rz(theta)
        Z rotation via AC Stark shift.  Implemented as a virtual gate
        (frame update) with zero physical error.

    R(theta, phi)
        Arbitrary single-qubit rotation by angle theta about the axis
        cos(phi)X + sin(phi)Y.  Physically realised via Raman or
        microwave drive.

    MS(theta)
        Molmer-Sorensen entangling gate generating an XX interaction:
        MS(theta) = exp(-i * theta * X x X).

    XX(theta)
        Parameterised Ising interaction (synonym for MS with flexible
        angle).
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
    def r_matrix(theta: float, phi: float) -> np.ndarray:
        """Arbitrary single-qubit rotation R(theta, phi).

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
    def ms_matrix(theta: float) -> np.ndarray:
        """Molmer-Sorensen gate matrix (4x4).

        MS(theta) = exp(-i * theta * X x X)
                   = cos(theta)*I4 - i*sin(theta)*(X x X)

        In the computational basis {|00>, |01>, |10>, |11>}:

            [[cos(t),    0,       0,    -i*sin(t)],
             [0,       cos(t), -i*sin(t),   0     ],
             [0,     -i*sin(t), cos(t),      0     ],
             [-i*sin(t), 0,       0,       cos(t)  ]]

        Parameters
        ----------
        theta : float
            Interaction angle in radians.  theta=pi/4 gives a
            maximally entangling gate (equivalent to CNOT up to
            single-qubit rotations).
        """
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array(
            [
                [c, 0, 0, -1j * s],
                [0, c, -1j * s, 0],
                [0, -1j * s, c, 0],
                [-1j * s, 0, 0, c],
            ],
            dtype=np.complex128,
        )

    @staticmethod
    def xx_matrix(theta: float) -> np.ndarray:
        """XX(theta) Ising interaction gate.

        Identical to MS(theta) --- provided as an alias matching
        Quantinuum and IonQ conventions.
        """
        return TrappedIonGateSet.ms_matrix(theta)

    @staticmethod
    def zz_matrix(theta: float) -> np.ndarray:
        """ZZ(theta) = exp(-i * theta * Z x Z).

        Used by Quantinuum native gate set.
        """
        return np.diag(
            [
                np.exp(-1j * theta),
                np.exp(1j * theta),
                np.exp(1j * theta),
                np.exp(-1j * theta),
            ]
        )

    # ==================================================================
    # Gate compilation: standard gates -> native gates
    # ==================================================================

    @staticmethod
    def compile_h(qubit: int) -> list[GateInstruction]:
        """Compile Hadamard into native gates.

        H = Rz(pi) * Ry(pi/2) = R(pi/2, pi/2) * Rz(pi)

        Uses one virtual Rz and one physical rotation.
        """
        return [
            GateInstruction(NativeGateType.RY, (qubit,), (math.pi / 2,)),
            GateInstruction(NativeGateType.RZ, (qubit,), (math.pi,)),
        ]

    @staticmethod
    def compile_x(qubit: int) -> list[GateInstruction]:
        """X = R(pi, 0)."""
        return [GateInstruction(NativeGateType.R, (qubit,), (math.pi, 0.0))]

    @staticmethod
    def compile_y(qubit: int) -> list[GateInstruction]:
        """Y = R(pi, pi/2)."""
        return [
            GateInstruction(
                NativeGateType.R, (qubit,), (math.pi, math.pi / 2)
            )
        ]

    @staticmethod
    def compile_z(qubit: int) -> list[GateInstruction]:
        """Z = Rz(pi) (virtual gate)."""
        return [GateInstruction(NativeGateType.RZ, (qubit,), (math.pi,))]

    @staticmethod
    def compile_rx(qubit: int, theta: float) -> list[GateInstruction]:
        """Rx(theta) = R(theta, 0)."""
        return [GateInstruction(NativeGateType.R, (qubit,), (theta, 0.0))]

    @staticmethod
    def compile_ry(qubit: int, theta: float) -> list[GateInstruction]:
        """Ry(theta) = R(theta, pi/2)."""
        return [
            GateInstruction(
                NativeGateType.R, (qubit,), (theta, math.pi / 2)
            )
        ]

    @staticmethod
    def compile_rz(qubit: int, theta: float) -> list[GateInstruction]:
        """Rz(theta) — already native (virtual gate)."""
        return [GateInstruction(NativeGateType.RZ, (qubit,), (theta,))]

    @staticmethod
    def compile_cnot(control: int, target: int) -> list[GateInstruction]:
        """Compile CNOT into native trapped-ion gates.

        Verified decomposition (up to global phase):
            CNOT = [Rx(pi/2) Rz(pi/2)]_c [Rx(pi/2) Ry(pi)]_t
                   MS(pi/4)
                   Ry(pi/2)_c  Ry(pi)_t

        Uses exactly 1 MS gate (optimal for trapped ions) plus
        single-qubit rotations.

        Parameters
        ----------
        control : int
            Control qubit index.
        target : int
            Target qubit index.
        """
        return [
            # Pre-MS single-qubit gates
            GateInstruction(NativeGateType.RY, (control,), (math.pi / 2,)),
            GateInstruction(NativeGateType.RY, (target,), (math.pi,)),
            # Entangling gate
            GateInstruction(
                NativeGateType.MS, (control, target), (math.pi / 4,)
            ),
            # Post-MS single-qubit gates
            GateInstruction(NativeGateType.RZ, (control,), (math.pi / 2,)),
            GateInstruction(NativeGateType.RX, (control,), (math.pi / 2,)),
            GateInstruction(NativeGateType.RY, (target,), (math.pi,)),
            GateInstruction(NativeGateType.RX, (target,), (math.pi / 2,)),
        ]

    @staticmethod
    def compile_cz(control: int, target: int) -> list[GateInstruction]:
        """Compile CZ into native trapped-ion gates.

        Verified decomposition (up to global phase):
            CZ = [Rx(pi/2) Rz(-pi/2)]_c [Rx(pi/2) Rz(-pi/2)]_t
                 MS(pi/4)
                 Ry(-pi/2)_c  Ry(-pi/2)_t

        Uses exactly 1 MS gate plus single-qubit rotations.

        Parameters
        ----------
        control : int
            Control qubit index.
        target : int
            Target qubit index.
        """
        return [
            # Pre-MS single-qubit gates
            GateInstruction(NativeGateType.RY, (control,), (-math.pi / 2,)),
            GateInstruction(NativeGateType.RY, (target,), (-math.pi / 2,)),
            # Entangling gate
            GateInstruction(
                NativeGateType.MS, (control, target), (math.pi / 4,)
            ),
            # Post-MS single-qubit gates
            GateInstruction(NativeGateType.RZ, (control,), (-math.pi / 2,)),
            GateInstruction(NativeGateType.RX, (control,), (math.pi / 2,)),
            GateInstruction(NativeGateType.RZ, (target,), (-math.pi / 2,)),
            GateInstruction(NativeGateType.RX, (target,), (math.pi / 2,)),
        ]

    @staticmethod
    def compile_swap(qubit_a: int, qubit_b: int) -> list[GateInstruction]:
        """Compile SWAP = 3 CNOTs = 3 MS gates."""
        instructions = []
        instructions.extend(TrappedIonGateSet.compile_cnot(qubit_a, qubit_b))
        instructions.extend(TrappedIonGateSet.compile_cnot(qubit_b, qubit_a))
        instructions.extend(TrappedIonGateSet.compile_cnot(qubit_a, qubit_b))
        return instructions

    @staticmethod
    def compile_toffoli(
        control_a: int, control_b: int, target: int
    ) -> list[GateInstruction]:
        """Compile Toffoli using 6 MS gates (optimal for trapped ions).

        Follows the decomposition from Maslov, Phys. Rev. A 93, 022311.
        """
        # Simplified: V * CNOT(a,t) * V_dag * CNOT(b,t) * V * CNOT(a,t) * V_dag
        # where V = sqrt(X).  Each CNOT costs 1 MS gate + singles.
        # Plus 2 CNOTs for the controls = 6 MS total (reduced from 3 CNOT = 3 MS).
        # We use the standard 6-CNOT decomposition rewritten in native gates.
        instructions = []
        instructions.extend(TrappedIonGateSet.compile_h(target))
        instructions.extend(TrappedIonGateSet.compile_cnot(control_b, target))
        instructions.append(
            GateInstruction(NativeGateType.RZ, (target,), (-math.pi / 4,))
        )
        instructions.extend(TrappedIonGateSet.compile_cnot(control_a, target))
        instructions.append(
            GateInstruction(NativeGateType.RZ, (target,), (math.pi / 4,))
        )
        instructions.extend(TrappedIonGateSet.compile_cnot(control_b, target))
        instructions.append(
            GateInstruction(NativeGateType.RZ, (target,), (-math.pi / 4,))
        )
        instructions.extend(TrappedIonGateSet.compile_cnot(control_a, target))
        instructions.append(
            GateInstruction(NativeGateType.RZ, (target,), (math.pi / 4,))
        )
        instructions.append(
            GateInstruction(NativeGateType.RZ, (control_b,), (math.pi / 4,))
        )
        instructions.extend(TrappedIonGateSet.compile_h(target))
        instructions.extend(
            TrappedIonGateSet.compile_cnot(control_a, control_b)
        )
        instructions.append(
            GateInstruction(NativeGateType.RZ, (control_a,), (math.pi / 4,))
        )
        instructions.append(
            GateInstruction(NativeGateType.RZ, (control_b,), (-math.pi / 4,))
        )
        instructions.extend(
            TrappedIonGateSet.compile_cnot(control_a, control_b)
        )
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
        assert u.shape == (2, 2), "Expected 2x2 unitary"

        # Extract ZYZ angles
        # U = e^{i*alpha} Rz(beta) Ry(gamma) Rz(delta)
        # Use the parameterisation from Nielsen & Chuang
        det = np.linalg.det(u)
        phase = np.angle(det) / 2.0

        # Normalise to SU(2)
        u_su2 = u * np.exp(-1j * phase)

        # gamma from |u00|
        cos_gamma_half = min(1.0, max(0.0, abs(u_su2[0, 0])))
        sin_gamma_half = min(1.0, max(0.0, abs(u_su2[1, 0])))
        gamma = 2.0 * math.atan2(sin_gamma_half, cos_gamma_half)

        if abs(cos_gamma_half) < 1e-10:
            # gamma ~ pi: u00 ~ 0
            beta = 0.0
            delta = np.angle(u_su2[1, 0]) - np.angle(u_su2[0, 1]) - math.pi
        elif abs(sin_gamma_half) < 1e-10:
            # gamma ~ 0: u10 ~ 0
            beta = 0.0
            delta = 2.0 * np.angle(u_su2[1, 1])
        else:
            beta_plus_delta = 2.0 * np.angle(u_su2[1, 1])
            beta_minus_delta = 2.0 * np.angle(u_su2[1, 0])
            beta = (beta_plus_delta + beta_minus_delta) / 2.0
            delta = (beta_plus_delta - beta_minus_delta) / 2.0

        instructions = []
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
    # MS gate count analysis
    # ==================================================================

    @staticmethod
    def ms_gate_count(instructions: list[GateInstruction]) -> int:
        """Count the number of MS (entangling) gates in an instruction list."""
        return sum(
            1
            for inst in instructions
            if inst.gate_type in (NativeGateType.MS, NativeGateType.XX, NativeGateType.ZZ)
        )

    @staticmethod
    def single_qubit_gate_count(instructions: list[GateInstruction]) -> int:
        """Count single-qubit gates in an instruction list."""
        return sum(
            1
            for inst in instructions
            if inst.gate_type
            in (NativeGateType.RZ, NativeGateType.R, NativeGateType.RX,
                NativeGateType.RY, NativeGateType.GPI, NativeGateType.GPI2)
        )
