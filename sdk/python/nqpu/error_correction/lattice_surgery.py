"""Lattice surgery operations for logical qubit manipulation.

Implements the core primitives for fault-tolerant computation via
lattice surgery on surface code patches:

  - :class:`LogicalQubit` -- encoded qubit with Pauli frame tracking
  - :class:`LatticeSurgery` -- merge/split operations for logical CNOT
  - :class:`MagicStateDistillation` -- T-state distillation protocols
  - :func:`estimate_resources` -- physical qubit and time cost estimation

References:
  - Horsman et al., "Surface code quantum computing by lattice surgery"
    (NJP 2012)
  - Litinski, "A Game of Surface Codes" (Quantum 2019)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ #
# Pauli Frame
# ------------------------------------------------------------------ #

class PauliFrame:
    """Classical tracking of Pauli corrections on logical qubits.

    In lattice surgery, many gates produce byproduct Pauli operators
    that can be tracked classically and applied (or compensated) at
    measurement time rather than physically corrected.

    Attributes
    ----------
    num_qubits : int
        Number of logical qubits being tracked.
    x_frame : np.ndarray
        Binary vector tracking pending X corrections.
    z_frame : np.ndarray
        Binary vector tracking pending Z corrections.
    """

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        self.x_frame = np.zeros(num_qubits, dtype=np.int8)
        self.z_frame = np.zeros(num_qubits, dtype=np.int8)

    def apply_x(self, qubit: int) -> None:
        """Record a pending X correction on a logical qubit."""
        self.x_frame[qubit] = (self.x_frame[qubit] + 1) % 2

    def apply_z(self, qubit: int) -> None:
        """Record a pending Z correction on a logical qubit."""
        self.z_frame[qubit] = (self.z_frame[qubit] + 1) % 2

    def apply_cnot(self, control: int, target: int) -> None:
        """Update frame for a logical CNOT.

        CNOT propagation rules:
          - X on control -> X on both control and target
          - Z on target -> Z on both control and target
        """
        # X propagates forward
        if self.x_frame[control]:
            self.x_frame[target] = (self.x_frame[target] + 1) % 2
        # Z propagates backward
        if self.z_frame[target]:
            self.z_frame[control] = (self.z_frame[control] + 1) % 2

    def apply_hadamard(self, qubit: int) -> None:
        """Update frame for a logical Hadamard (swap X and Z)."""
        self.x_frame[qubit], self.z_frame[qubit] = (
            self.z_frame[qubit],
            self.x_frame[qubit],
        )

    def apply_s(self, qubit: int) -> None:
        """Update frame for a logical S gate.

        S gate: X -> Y = iXZ, Z -> Z
        So X frame bit generates a Z frame bit.
        """
        if self.x_frame[qubit]:
            self.z_frame[qubit] = (self.z_frame[qubit] + 1) % 2

    def measure_correction(self, qubit: int, basis: str = "Z") -> int:
        """Get measurement correction sign for a logical qubit.

        Returns 0 if no correction needed, 1 if result should be flipped.
        """
        if basis == "Z":
            return int(self.x_frame[qubit])
        elif basis == "X":
            return int(self.z_frame[qubit])
        else:
            return int((self.x_frame[qubit] + self.z_frame[qubit]) % 2)

    def reset(self, qubit: int) -> None:
        """Clear all pending corrections for a qubit."""
        self.x_frame[qubit] = 0
        self.z_frame[qubit] = 0

    def copy(self) -> "PauliFrame":
        """Return a copy of this frame."""
        frame = PauliFrame(self.num_qubits)
        frame.x_frame = self.x_frame.copy()
        frame.z_frame = self.z_frame.copy()
        return frame


# ------------------------------------------------------------------ #
# Logical Qubit
# ------------------------------------------------------------------ #

@dataclass
class LogicalQubit:
    """A logical qubit encoded in a surface code patch.

    Attributes
    ----------
    qubit_id : int
        Unique identifier for this logical qubit.
    distance : int
        Code distance of the surface code patch.
    state : str
        Current logical state description (for tracking).
    position : tuple of int
        (row, col) position of the patch on the lattice.
    is_ancilla : bool
        Whether this patch is used as an ancilla for surgery.
    """

    qubit_id: int
    distance: int
    state: str = "|0>"
    position: Tuple[int, int] = (0, 0)
    is_ancilla: bool = False

    @property
    def physical_qubits(self) -> int:
        """Number of physical data qubits in this patch."""
        return self.distance * self.distance

    @property
    def total_physical_qubits(self) -> int:
        """Total physical qubits including syndrome ancillas."""
        d = self.distance
        # Data qubits + approximately same number of ancillas
        return 2 * d * d - 1


# ------------------------------------------------------------------ #
# Lattice Surgery
# ------------------------------------------------------------------ #

class MergeType(Enum):
    """Type of lattice surgery merge."""
    XX = "XX"
    ZZ = "ZZ"


@dataclass
class SurgeryOperation:
    """Record of a lattice surgery operation.

    Attributes
    ----------
    op_type : str
        Type of operation ("merge", "split", "cnot", "hadamard", "s", "t").
    qubits : list of int
        Logical qubit IDs involved.
    merge_type : MergeType or None
        For merge/split: XX or ZZ merge.
    measurement_result : int
        Measurement outcome (0 or 1) from the merge.
    code_cycles : int
        Number of QEC code cycles for this operation.
    """

    op_type: str
    qubits: List[int]
    merge_type: Optional[MergeType] = None
    measurement_result: int = 0
    code_cycles: int = 1


class LatticeSurgery:
    """Lattice surgery engine for logical operations between surface code patches.

    Parameters
    ----------
    num_logical_qubits : int
        Number of logical qubits in the computation.
    distance : int
        Code distance for all patches.
    seed : int or None
        Random seed for simulated measurement outcomes.
    """

    def __init__(
        self,
        num_logical_qubits: int,
        distance: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.distance = distance
        self.rng = np.random.default_rng(seed)
        self.frame = PauliFrame(num_logical_qubits)
        self.operations: List[SurgeryOperation] = []

        # Create logical qubit patches
        self.qubits: List[LogicalQubit] = []
        for i in range(num_logical_qubits):
            q = LogicalQubit(
                qubit_id=i,
                distance=distance,
                position=(0, i * (distance + 1)),
            )
            self.qubits.append(q)

        # Ancilla pool
        self._next_ancilla_id = num_logical_qubits
        self.ancilla_pool: List[LogicalQubit] = []

    def _get_ancilla(self) -> LogicalQubit:
        """Get an ancilla patch, creating one if needed."""
        if self.ancilla_pool:
            return self.ancilla_pool.pop()
        anc = LogicalQubit(
            qubit_id=self._next_ancilla_id,
            distance=self.distance,
            is_ancilla=True,
        )
        self._next_ancilla_id += 1
        return anc

    def _return_ancilla(self, anc: LogicalQubit) -> None:
        """Return an ancilla to the pool."""
        self.ancilla_pool.append(anc)

    def merge(
        self, qubit_a: int, qubit_b: int, merge_type: MergeType = MergeType.ZZ
    ) -> int:
        """Perform a lattice surgery merge.

        Measures the joint operator (XX or ZZ) between two patches by
        merging their shared boundary for ``d`` code cycles, then
        measuring the merged stabilizers.

        Parameters
        ----------
        qubit_a, qubit_b : int
            Logical qubit IDs to merge.
        merge_type : MergeType
            XX or ZZ merge.

        Returns
        -------
        int
            Measurement result (0 or 1).
        """
        result = int(self.rng.integers(0, 2))
        op = SurgeryOperation(
            op_type="merge",
            qubits=[qubit_a, qubit_b],
            merge_type=merge_type,
            measurement_result=result,
            code_cycles=self.distance,
        )
        self.operations.append(op)
        return result

    def split(self, qubit_a: int, qubit_b: int) -> None:
        """Split a merged patch back into two separate patches.

        Takes ``d`` code cycles to re-establish independent stabilizers.
        """
        op = SurgeryOperation(
            op_type="split",
            qubits=[qubit_a, qubit_b],
            code_cycles=self.distance,
        )
        self.operations.append(op)

    def logical_cnot(self, control: int, target: int) -> None:
        """Perform a logical CNOT via lattice surgery.

        Protocol (Horsman et al.):
        1. Prepare ancilla in |+>
        2. ZZ merge(ancilla, target) -> measure m1
        3. Split
        4. XX merge(control, ancilla) -> measure m2
        5. Split
        6. Apply Pauli corrections based on m1, m2

        Total: 4d code cycles.
        """
        anc = self._get_ancilla()
        anc_id = anc.qubit_id

        # Temporarily expand frame if needed
        max_id = max(control, target, anc_id) + 1
        if max_id > self.frame.num_qubits:
            old = self.frame
            self.frame = PauliFrame(max_id)
            self.frame.x_frame[: old.num_qubits] = old.x_frame
            self.frame.z_frame[: old.num_qubits] = old.z_frame

        m1 = self.merge(anc_id, target, MergeType.ZZ)
        self.split(anc_id, target)
        m2 = self.merge(control, anc_id, MergeType.XX)
        self.split(control, anc_id)

        # Pauli corrections based on measurement outcomes
        if m1:
            self.frame.apply_x(target)
        if m2:
            self.frame.apply_z(control)

        self.frame.apply_cnot(control, target)

        op = SurgeryOperation(
            op_type="cnot",
            qubits=[control, target],
            code_cycles=4 * self.distance,
        )
        self.operations.append(op)

        self._return_ancilla(anc)

    def logical_hadamard(self, qubit: int) -> None:
        """Perform a logical Hadamard by transposing patch boundaries.

        In lattice surgery, Hadamard swaps rough and smooth boundaries,
        which swaps X and Z logical operators.  Takes d code cycles.
        """
        self.frame.apply_hadamard(qubit)
        op = SurgeryOperation(
            op_type="hadamard",
            qubits=[qubit],
            code_cycles=self.distance,
        )
        self.operations.append(op)

    def logical_s(self, qubit: int) -> None:
        """Perform a logical S gate via magic state injection.

        Requires a |Y> magic state and teleportation.
        """
        self.frame.apply_s(qubit)
        op = SurgeryOperation(
            op_type="s",
            qubits=[qubit],
            code_cycles=2 * self.distance,
        )
        self.operations.append(op)

    def logical_t(self, qubit: int) -> None:
        """Perform a logical T gate via magic state injection.

        Requires a distilled |T> = (|0> + e^{i pi/4}|1>)/sqrt(2)
        magic state and gate teleportation.
        """
        op = SurgeryOperation(
            op_type="t",
            qubits=[qubit],
            code_cycles=2 * self.distance,
        )
        self.operations.append(op)

    def total_code_cycles(self) -> int:
        """Total number of code cycles across all operations."""
        return sum(op.code_cycles for op in self.operations)

    def total_physical_qubits(self) -> int:
        """Total physical qubits needed (data + ancilla patches)."""
        num_patches = len(self.qubits) + 1  # +1 for ancilla
        return num_patches * (2 * self.distance ** 2 - 1)


# ------------------------------------------------------------------ #
# Magic State Distillation
# ------------------------------------------------------------------ #

class MagicStateDistillation:
    """Magic state distillation protocols for non-Clifford gates.

    Implements the 15-to-1 distillation protocol: takes 15 noisy T states
    and produces 1 higher-fidelity T state, using a [[15,1,3]] Reed-Muller
    code.

    Parameters
    ----------
    input_error_rate : float
        Error rate of the input noisy T states.
    num_levels : int
        Number of distillation levels (each reduces error cubically).
    """

    def __init__(
        self, input_error_rate: float = 0.01, num_levels: int = 1
    ) -> None:
        self.input_error_rate = input_error_rate
        self.num_levels = num_levels

    def output_error_rate(self) -> float:
        """Compute the output T-state error rate after distillation.

        The 15-to-1 protocol has output error ~ 35 * p_in^3 per level.
        """
        p = self.input_error_rate
        for _ in range(self.num_levels):
            p = 35.0 * p ** 3
        return p

    def output_fidelity(self) -> float:
        """Fidelity of the distilled T state."""
        return 1.0 - self.output_error_rate()

    def input_states_needed(self) -> int:
        """Total number of raw T states consumed."""
        return 15 ** self.num_levels

    def success_probability(self) -> float:
        """Probability that the distillation succeeds (post-selected).

        Approximate: 1 - 15*p per level (probability of no detectable error).
        """
        p = self.input_error_rate
        prob = 1.0
        for _ in range(self.num_levels):
            prob *= max(0.0, 1.0 - 15.0 * p)
            p = 35.0 * p ** 3
        return prob

    def physical_qubit_cost(self, code_distance: int) -> int:
        """Estimate physical qubit cost for the distillation factory.

        Each level requires 15 patches for input + workspace.
        """
        qubits_per_patch = 2 * code_distance ** 2 - 1
        total = 0
        for level in range(self.num_levels):
            patches_this_level = 15 + 1  # 15 input + 1 output
            total += patches_this_level * qubits_per_patch
        return total

    def time_cost_cycles(self, code_distance: int) -> int:
        """Estimate time cost in code cycles.

        Each distillation level takes ~5d code cycles.
        """
        return self.num_levels * 5 * code_distance


# ------------------------------------------------------------------ #
# Resource Estimation
# ------------------------------------------------------------------ #

@dataclass
class ResourceEstimate:
    """Resource estimate for a fault-tolerant computation.

    Attributes
    ----------
    num_logical_qubits : int
        Number of logical qubits.
    code_distance : int
        Surface code distance.
    physical_qubits : int
        Total physical qubits needed.
    num_t_gates : int
        Number of T gates in the circuit.
    num_cnot_gates : int
        Number of CNOT gates.
    num_clifford_gates : int
        Number of Clifford gates (H, S, CNOT).
    total_code_cycles : int
        Total code cycles for the computation.
    distillation_qubits : int
        Physical qubits dedicated to magic state factories.
    wall_time_us : float
        Estimated wall-clock time in microseconds (assuming 1us cycle time).
    logical_error_per_cycle : float
        Expected logical error probability per code cycle.
    total_logical_error : float
        Expected total logical error probability.
    """

    num_logical_qubits: int
    code_distance: int
    physical_qubits: int
    num_t_gates: int = 0
    num_cnot_gates: int = 0
    num_clifford_gates: int = 0
    total_code_cycles: int = 0
    distillation_qubits: int = 0
    wall_time_us: float = 0.0
    logical_error_per_cycle: float = 0.0
    total_logical_error: float = 0.0


def estimate_resources(
    num_logical_qubits: int,
    num_t_gates: int,
    num_cnot_gates: int = 0,
    num_clifford_gates: int = 0,
    physical_error_rate: float = 1e-3,
    target_logical_error: float = 1e-6,
    cycle_time_us: float = 1.0,
    distillation_levels: int = 1,
) -> ResourceEstimate:
    """Estimate physical resources for a fault-tolerant quantum computation.

    Uses the heuristic that the logical error per code cycle scales as
    ``d * (100 * p_phys)^((d+1)/2)`` for a distance-d surface code.

    Parameters
    ----------
    num_logical_qubits : int
        Number of logical qubits.
    num_t_gates : int
        Number of T gates in the circuit.
    num_cnot_gates : int
        Number of CNOT gates.
    num_clifford_gates : int
        Number of Clifford gates.
    physical_error_rate : float
        Physical gate error rate.
    target_logical_error : float
        Target total logical error budget.
    cycle_time_us : float
        Duration of one code cycle in microseconds.
    distillation_levels : int
        Number of magic state distillation levels.

    Returns
    -------
    ResourceEstimate
    """
    # Estimate total code cycles
    total_gates = num_t_gates + num_cnot_gates + num_clifford_gates
    if total_gates == 0:
        total_gates = 1

    # Find minimum code distance
    # Logical error per cycle ~ d * (100 * p)^((d+1)/2)
    # Need: cycles * error_per_cycle < target
    p = physical_error_rate
    best_d = 3

    for d in range(3, 101, 2):
        # Approximate total code cycles
        cycles = total_gates * 4 * d  # each gate ~ 4d cycles
        error_per_cycle = d * (100 * p) ** ((d + 1) / 2)
        total_error = cycles * error_per_cycle

        if total_error < target_logical_error:
            best_d = d
            break
    else:
        best_d = 99  # very high distance needed

    d = best_d
    total_cycles = total_gates * 4 * d

    # Physical qubits
    qubits_per_patch = 2 * d * d - 1
    data_qubits = num_logical_qubits * qubits_per_patch

    # Distillation factory
    distill = MagicStateDistillation(
        input_error_rate=p, num_levels=distillation_levels
    )
    distill_qubits = distill.physical_qubit_cost(d) if num_t_gates > 0 else 0

    total_physical = data_qubits + distill_qubits

    # Logical error
    error_per_cycle = d * (100 * p) ** ((d + 1) / 2)
    total_logical_error = total_cycles * error_per_cycle

    return ResourceEstimate(
        num_logical_qubits=num_logical_qubits,
        code_distance=d,
        physical_qubits=total_physical,
        num_t_gates=num_t_gates,
        num_cnot_gates=num_cnot_gates,
        num_clifford_gates=num_clifford_gates,
        total_code_cycles=total_cycles,
        distillation_qubits=distill_qubits,
        wall_time_us=total_cycles * cycle_time_us,
        logical_error_per_cycle=error_per_cycle,
        total_logical_error=total_logical_error,
    )
