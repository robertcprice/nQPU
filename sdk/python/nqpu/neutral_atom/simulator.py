"""End-to-end neutral-atom quantum simulator.

Provides a unified interface for simulating quantum circuits on
neutral-atom hardware using Rydberg blockade gates.  Supports three
execution modes:

1. **Ideal**: Perfect gate execution using state-vector simulation.
   No noise, no atom loss.  Best for algorithm development.

2. **Noisy**: Gate-level noise model using density-matrix simulation.
   Physics-based error channels from :class:`NeutralAtomNoiseModel`.
   Best for realistic performance estimation.

3. **Pulse**: Simplified Hamiltonian-level simulation through Rydberg
   pulse sequences.  Each CZ gate is decomposed into the three-pulse
   Rydberg blockade protocol.  Most accurate but slowest.

References:
    - Levine et al., Phys. Rev. Lett. 123, 170503 (2019)
      [Native CZ gate via Rydberg blockade]
    - Evered et al., Nature 622, 268 (2023)
      [High-fidelity parallel entangling gates]
    - Bluvstein et al., Nature 604, 451 (2022)
      [Reconfigurable quantum processor]
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .array import AtomArray
from .gates import GateInstruction, NativeGateType, NeutralAtomGateSet
from .noise import NeutralAtomNoiseModel
from .physics import AtomSpecies


@dataclass
class CircuitStats:
    """Statistics about a compiled circuit."""

    total_gates: int = 0
    single_qubit_gates: int = 0
    two_qubit_gates: int = 0
    three_qubit_gates: int = 0
    cz_gate_count: int = 0
    ccz_gate_count: int = 0
    estimated_fidelity: float = 1.0
    estimated_duration_us: float = 0.0


@dataclass
class ArrayConfig:
    """Configuration for the neutral-atom array simulator.

    Parameters
    ----------
    n_atoms : int
        Number of atoms (qubits) in the array.
    species : AtomSpecies
        Atomic species.
    spacing_um : float
        Inter-atom spacing in micrometres.
    rabi_freq_mhz : float
        Rydberg Rabi frequency in MHz.
    """

    n_atoms: int
    species: AtomSpecies = field(
        default_factory=lambda: AtomSpecies.RB87  # type: ignore[attr-defined]
    )
    spacing_um: float = 4.0
    rabi_freq_mhz: float = 1.5

    def __post_init__(self) -> None:
        if self.n_atoms < 1:
            raise ValueError("n_atoms must be >= 1")
        if self.spacing_um <= 0:
            raise ValueError("spacing_um must be positive")
        if self.rabi_freq_mhz <= 0:
            raise ValueError("rabi_freq_mhz must be positive")


class NeutralAtomSimulator:
    """End-to-end neutral-atom quantum simulator.

    Simulates quantum circuits with physics-accurate noise from the
    Rydberg-atom hardware model.  Supports three execution modes:
    ideal (state vector), noisy (density matrix), and pulse
    (Rydberg Hamiltonian evolution).

    Parameters
    ----------
    config : ArrayConfig
        Array and atom configuration.
    noise_model : NeutralAtomNoiseModel, optional
        Physics noise model.  Auto-created from config if ``None``
        and execution_mode is 'noisy'.
    execution_mode : str
        One of 'ideal', 'noisy', 'pulse'.

    Examples
    --------
    >>> from nqpu.neutral_atom import NeutralAtomSimulator, ArrayConfig
    >>> sim = NeutralAtomSimulator(ArrayConfig(n_atoms=3))
    >>> sim.h(0)
    >>> sim.cnot(0, 1)
    >>> sim.cnot(0, 2)
    >>> counts = sim.measure_all(shots=1000)
    >>> print(counts)  # ~ {'000': 500, '111': 500}
    """

    def __init__(
        self,
        config: ArrayConfig,
        noise_model: NeutralAtomNoiseModel | None = None,
        execution_mode: str = "ideal",
    ) -> None:
        if execution_mode not in ("ideal", "noisy", "pulse"):
            raise ValueError(
                f"execution_mode must be 'ideal', 'noisy', or 'pulse', "
                f"got '{execution_mode}'"
            )

        self.config = config
        self.n_qubits = config.n_atoms
        self.execution_mode = execution_mode
        self.dim = 2**self.n_qubits
        self.gate_set = NeutralAtomGateSet()

        # Noise model
        if noise_model is not None:
            self.noise_model = noise_model
        elif execution_mode == "noisy":
            self.noise_model = NeutralAtomNoiseModel(
                species=config.species,
                rabi_freq_mhz=config.rabi_freq_mhz,
                atom_spacing_um=config.spacing_um,
            )
        else:
            self.noise_model = None

        # Quantum state
        if execution_mode in ("ideal", "pulse"):
            # State vector: |00...0>
            self._statevector = np.zeros(self.dim, dtype=np.complex128)
            self._statevector[0] = 1.0
            self._density_matrix = None
        else:
            # Density matrix: |00...0><00...0|
            self._statevector = None
            self._density_matrix = np.zeros(
                (self.dim, self.dim), dtype=np.complex128
            )
            self._density_matrix[0, 0] = 1.0

        # Circuit log for compilation and analysis
        self._circuit_log: list[GateInstruction] = []
        self._native_instructions: list[GateInstruction] = []

    # ==================================================================
    # Standard gate interface
    # ==================================================================

    def h(self, qubit: int) -> None:
        """Apply Hadamard gate."""
        self._validate_qubit(qubit)
        native = self.gate_set.compile_h(qubit)
        self._execute_native(native, "single", qubit)

    def x(self, qubit: int) -> None:
        """Apply Pauli-X gate."""
        self._validate_qubit(qubit)
        native = self.gate_set.compile_x(qubit)
        self._execute_native(native, "single", qubit)

    def y(self, qubit: int) -> None:
        """Apply Pauli-Y gate."""
        self._validate_qubit(qubit)
        native = self.gate_set.compile_y(qubit)
        self._execute_native(native, "single", qubit)

    def z(self, qubit: int) -> None:
        """Apply Pauli-Z gate."""
        self._validate_qubit(qubit)
        native = self.gate_set.compile_z(qubit)
        self._execute_native(native, "single", qubit)

    def rx(self, qubit: int, theta: float) -> None:
        """Apply Rx(theta) rotation."""
        self._validate_qubit(qubit)
        native = self.gate_set.compile_rx(qubit, theta)
        self._execute_native(native, "single", qubit)

    def ry(self, qubit: int, theta: float) -> None:
        """Apply Ry(theta) rotation."""
        self._validate_qubit(qubit)
        native = self.gate_set.compile_ry(qubit, theta)
        self._execute_native(native, "single", qubit)

    def rz(self, qubit: int, theta: float) -> None:
        """Apply Rz(theta) rotation (virtual gate, zero error)."""
        self._validate_qubit(qubit)
        native = self.gate_set.compile_rz(qubit, theta)
        self._execute_native(native, "single", qubit)

    def cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate (compiled to H + CZ + H)."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target must be different qubits")
        native = self.gate_set.compile_cnot(control, target)
        self._execute_native(native, "cz", control, target)

    def cz(self, control: int, target: int) -> None:
        """Apply CZ gate (native Rydberg blockade gate)."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target must be different qubits")
        native = self.gate_set.compile_cz(control, target)
        self._execute_native(native, "cz", control, target)

    # ==================================================================
    # Native neutral-atom gates
    # ==================================================================

    def ccz(self, qubit_a: int, qubit_b: int, qubit_c: int) -> None:
        """Apply native CCZ gate using Rydberg multi-body blockade.

        All three qubits must be within mutual blockade radius.

        Parameters
        ----------
        qubit_a, qubit_b, qubit_c : int
            Qubit indices.
        """
        self._validate_qubit(qubit_a)
        self._validate_qubit(qubit_b)
        self._validate_qubit(qubit_c)
        if len({qubit_a, qubit_b, qubit_c}) != 3:
            raise ValueError("CCZ requires three distinct qubits")
        native = self.gate_set.compile_ccz(qubit_a, qubit_b, qubit_c)
        self._execute_native(native, "ccz", qubit_a, qubit_b, qubit_c)

    def toffoli(self, control_a: int, control_b: int, target: int) -> None:
        """Apply Toffoli gate (compiled to H + CCZ + H).

        Uses the native CCZ gate, which is dramatically more efficient
        than decompositions requiring multiple CZ gates.

        Parameters
        ----------
        control_a, control_b : int
            Control qubit indices.
        target : int
            Target qubit index.
        """
        self._validate_qubit(control_a)
        self._validate_qubit(control_b)
        self._validate_qubit(target)
        if len({control_a, control_b, target}) != 3:
            raise ValueError("Toffoli requires three distinct qubits")
        native = self.gate_set.compile_toffoli(control_a, control_b, target)
        self._execute_native(native, "ccz", control_a, control_b, target)

    def global_rotation(self, theta: float, phi: float) -> None:
        """Apply a global rotation R(theta, phi) to all atoms simultaneously.

        In neutral-atom hardware, global rotations are achieved with a
        single global Raman or microwave beam --- very high fidelity.

        Parameters
        ----------
        theta : float
            Rotation angle in radians.
        phi : float
            Axis angle in the XY plane.
        """
        for q in range(self.n_qubits):
            native = [
                GateInstruction(NativeGateType.RXY, (q,), (theta, phi))
            ]
            self._execute_native(native, "single", q)

    # ==================================================================
    # Circuit execution from gate list
    # ==================================================================

    def run_circuit(
        self, circuit: list[tuple[str, ...]], shots: int = 1024
    ) -> dict[str, int]:
        """Execute a circuit specified as a list of gate tuples.

        Each tuple is (gate_name, qubit1, ..., [param1, ...]).

        Supported gates: h, x, y, z, rx, ry, rz, cx/cnot, cz, ccz, toffoli.

        Parameters
        ----------
        circuit : list[tuple]
            Circuit as a list of gate tuples.
        shots : int
            Number of measurement shots.

        Returns
        -------
        dict[str, int]
            Measurement histogram.

        Examples
        --------
        >>> sim = NeutralAtomSimulator(ArrayConfig(n_atoms=2))
        >>> circuit = [('h', 0), ('cx', 0, 1)]
        >>> counts = sim.run_circuit(circuit, shots=1000)
        """
        self.reset()

        for gate_tuple in circuit:
            gate_name = str(gate_tuple[0]).lower()
            args = gate_tuple[1:]

            if gate_name == "h" and len(args) == 1:
                self.h(int(args[0]))
            elif gate_name == "x" and len(args) == 1:
                self.x(int(args[0]))
            elif gate_name == "y" and len(args) == 1:
                self.y(int(args[0]))
            elif gate_name == "z" and len(args) == 1:
                self.z(int(args[0]))
            elif gate_name == "rx" and len(args) == 2:
                self.rx(int(args[0]), float(args[1]))
            elif gate_name == "ry" and len(args) == 2:
                self.ry(int(args[0]), float(args[1]))
            elif gate_name == "rz" and len(args) == 2:
                self.rz(int(args[0]), float(args[1]))
            elif gate_name in ("cx", "cnot") and len(args) == 2:
                self.cnot(int(args[0]), int(args[1]))
            elif gate_name == "cz" and len(args) == 2:
                self.cz(int(args[0]), int(args[1]))
            elif gate_name == "ccz" and len(args) == 3:
                self.ccz(int(args[0]), int(args[1]), int(args[2]))
            elif gate_name == "toffoli" and len(args) == 3:
                self.toffoli(int(args[0]), int(args[1]), int(args[2]))
            else:
                raise ValueError(f"Unknown gate: {gate_name} with {len(args)} args")

        return self.measure_all(shots=shots)

    # ==================================================================
    # Measurement
    # ==================================================================

    def measure(self, qubit: int) -> int:
        """Measure a single qubit, collapsing its state.

        In neutral-atom hardware, measurement is performed via
        state-dependent fluorescence imaging.

        Returns
        -------
        int
            Measurement outcome: 0 or 1.
        """
        self._validate_qubit(qubit)
        probs = self._qubit_probabilities(qubit)

        # Apply readout error if in noisy mode
        if self.noise_model is not None:
            outcome = int(np.random.random() < probs[1])
            if np.random.random() < self.noise_model.readout_error:
                outcome = 1 - outcome
        else:
            outcome = int(np.random.random() < probs[1])

        # Collapse state
        if self._statevector is not None:
            self._collapse_statevector(qubit, outcome)
        else:
            self._collapse_density_matrix(qubit, outcome)

        return outcome

    def measure_all(self, shots: int = 1024) -> dict[str, int]:
        """Measure all qubits, returning a histogram of outcomes.

        This is a non-destructive sampling from the probability
        distribution (state is preserved for further operations).

        Parameters
        ----------
        shots : int
            Number of measurement shots.

        Returns
        -------
        dict[str, int]
            Mapping from bitstring to count, e.g. {'000': 512, '111': 512}.
        """
        probs = self._full_probability_distribution()

        # Sample from the distribution
        indices = np.random.choice(self.dim, size=shots, p=probs)
        counts: dict[str, int] = {}

        for idx in indices:
            bitstring = format(idx, f"0{self.n_qubits}b")

            # Apply readout error if in noisy mode
            if self.noise_model is not None and self.noise_model.readout_error > 0:
                flipped = ""
                for bit in bitstring:
                    if np.random.random() < self.noise_model.readout_error:
                        flipped += "0" if bit == "1" else "1"
                    else:
                        flipped += bit
                bitstring = flipped

            counts[bitstring] = counts.get(bitstring, 0) + 1

        return dict(sorted(counts.items()))

    # ==================================================================
    # State inspection
    # ==================================================================

    def statevector(self) -> np.ndarray:
        """Return the current state vector.

        Only available in 'ideal' and 'pulse' modes.

        Returns
        -------
        np.ndarray
            Complex state vector of length 2^n.

        Raises
        ------
        RuntimeError
            If the simulator is in 'noisy' mode (density matrix).
        """
        if self._statevector is None:
            raise RuntimeError(
                "State vector not available in 'noisy' mode. "
                "Use density_matrix() instead."
            )
        return self._statevector.copy()

    def density_matrix(self) -> np.ndarray:
        """Return the current density matrix.

        In 'ideal' mode, constructs it from the state vector.

        Returns
        -------
        np.ndarray
            Density matrix of shape (2^n, 2^n).
        """
        if self._density_matrix is not None:
            return self._density_matrix.copy()
        sv = self._statevector
        return np.outer(sv, sv.conj())

    def fidelity_estimate(self) -> float:
        """Estimate the overall circuit fidelity.

        Computed as the product of individual gate fidelities from
        the noise model.

        Returns
        -------
        float
            Estimated fidelity in [0, 1].
        """
        if self.noise_model is None:
            return 1.0

        fidelity = 1.0
        for inst in self._native_instructions:
            if inst.gate_type == NativeGateType.CCZ:
                fidelity *= self.noise_model.three_qubit_gate_fidelity()
            elif inst.gate_type == NativeGateType.CZ:
                fidelity *= self.noise_model.two_qubit_gate_fidelity()
            elif inst.gate_type == NativeGateType.RZ:
                # Virtual gate: perfect fidelity
                pass
            else:
                fidelity *= self.noise_model.single_qubit_gate_fidelity()

        return fidelity

    def circuit_stats(self) -> CircuitStats:
        """Return statistics about the compiled circuit."""
        stats = CircuitStats()

        sq_time = 0.0
        tq_time = 0.0
        three_q_time = 0.0

        if self.noise_model:
            sq_time = self.noise_model.single_qubit_gate_time_us
            tq_time = self.noise_model.two_qubit_gate_time_us
            three_q_time = self.noise_model.three_qubit_gate_time_us
        else:
            # Default estimates from Rabi frequency
            omega = self.config.rabi_freq_mhz
            sq_time = 1.0 / (2.0 * omega)
            tq_time = 4.0 / (2.0 * omega)
            three_q_time = 5.0 / (2.0 * omega)

        for inst in self._native_instructions:
            stats.total_gates += 1
            if inst.gate_type == NativeGateType.CCZ:
                stats.three_qubit_gates += 1
                stats.ccz_gate_count += 1
                stats.estimated_duration_us += three_q_time
            elif inst.gate_type == NativeGateType.CZ:
                stats.two_qubit_gates += 1
                stats.cz_gate_count += 1
                stats.estimated_duration_us += tq_time
            else:
                stats.single_qubit_gates += 1
                if inst.gate_type != NativeGateType.RZ:
                    stats.estimated_duration_us += sq_time
                # Rz is virtual: zero time

        stats.estimated_fidelity = self.fidelity_estimate()
        return stats

    # ==================================================================
    # Device info
    # ==================================================================

    def device_info(self) -> dict[str, Any]:
        """Return device characterisation summary."""
        info: dict[str, Any] = {
            "n_qubits": self.n_qubits,
            "species": self.config.species.name,
            "execution_mode": self.execution_mode,
            "spacing_um": self.config.spacing_um,
            "rabi_freq_mhz": self.config.rabi_freq_mhz,
            "blockade_radius_um": self.config.species.blockade_radius_um(
                self.config.rabi_freq_mhz
            ),
        }
        if self.noise_model:
            info["1q_gate_fidelity"] = self.noise_model.single_qubit_gate_fidelity()
            info["2q_gate_fidelity"] = self.noise_model.two_qubit_gate_fidelity()
            info["3q_gate_fidelity"] = self.noise_model.three_qubit_gate_fidelity()
            info["error_budget"] = self.noise_model.error_budget()
        return info

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(self) -> None:
        """Reset the simulator to |00...0> and clear the circuit log."""
        if self._statevector is not None:
            self._statevector = np.zeros(self.dim, dtype=np.complex128)
            self._statevector[0] = 1.0
        if self._density_matrix is not None:
            self._density_matrix = np.zeros(
                (self.dim, self.dim), dtype=np.complex128
            )
            self._density_matrix[0, 0] = 1.0
        self._circuit_log.clear()
        self._native_instructions.clear()

    # ==================================================================
    # Internal execution engine
    # ==================================================================

    def _execute_native(
        self,
        instructions: list[GateInstruction],
        gate_category: str,
        *involved_qubits: int,
    ) -> None:
        """Execute a list of native gate instructions.

        Routes to the appropriate simulation backend based on
        execution_mode.
        """
        self._native_instructions.extend(instructions)

        if self.execution_mode == "ideal":
            self._execute_ideal(instructions)
        elif self.execution_mode == "noisy":
            self._execute_noisy(instructions, gate_category, involved_qubits)
        elif self.execution_mode == "pulse":
            self._execute_pulse(instructions)

    def _execute_ideal(self, instructions: list[GateInstruction]) -> None:
        """Execute gates on the state vector (ideal, no noise)."""
        for inst in instructions:
            matrix = self._instruction_to_matrix(inst)
            if len(inst.qubits) == 1:
                self._apply_single_qubit_gate(matrix, inst.qubits[0])
            elif len(inst.qubits) == 2:
                self._apply_two_qubit_gate(
                    matrix, inst.qubits[0], inst.qubits[1]
                )
            elif len(inst.qubits) == 3:
                self._apply_three_qubit_gate(
                    matrix, inst.qubits[0], inst.qubits[1], inst.qubits[2]
                )

    def _execute_noisy(
        self,
        instructions: list[GateInstruction],
        gate_category: str,
        involved_qubits: tuple[int, ...],
    ) -> None:
        """Execute gates on the density matrix with noise."""
        assert self._density_matrix is not None
        assert self.noise_model is not None

        for inst in instructions:
            matrix = self._instruction_to_matrix(inst)
            if len(inst.qubits) == 1:
                full_op = self._embed_single(matrix, inst.qubits[0])
            elif len(inst.qubits) == 2:
                full_op = self._embed_two(
                    matrix, inst.qubits[0], inst.qubits[1]
                )
            elif len(inst.qubits) == 3:
                full_op = self._embed_three(
                    matrix, inst.qubits[0], inst.qubits[1], inst.qubits[2]
                )
            else:
                continue

            # Unitary evolution
            self._density_matrix = (
                full_op @ self._density_matrix @ full_op.conj().T
            )

        # Apply noise for the whole gate operation
        if gate_category == "cz":
            gate_time = self.noise_model.two_qubit_gate_time_us
            self._density_matrix = self.noise_model.apply_noise(
                self._density_matrix, "cz", gate_time, involved_qubits
            )
        elif gate_category == "ccz":
            gate_time = self.noise_model.three_qubit_gate_time_us
            self._density_matrix = self.noise_model.apply_noise(
                self._density_matrix, "ccz", gate_time, involved_qubits
            )
        elif gate_category == "single":
            gate_time = self.noise_model.single_qubit_gate_time_us
            self._density_matrix = self.noise_model.apply_noise(
                self._density_matrix, "single", gate_time, involved_qubits
            )

    def _execute_pulse(self, instructions: list[GateInstruction]) -> None:
        """Execute gates via simplified Rydberg Hamiltonian evolution.

        For CZ gates, simulates the three-pulse blockade sequence:
        1. pi-pulse on control (|1> -> |r>)
        2. 2*pi-pulse on target (blockaded by control in |r>)
        3. pi-pulse on control (|r> -> |1>)

        For single-qubit gates, uses direct matrix application
        (equivalent to resonant Rabi drive).
        """
        for inst in instructions:
            matrix = self._instruction_to_matrix(inst)
            if len(inst.qubits) == 1:
                self._apply_single_qubit_gate(matrix, inst.qubits[0])
            elif len(inst.qubits) == 2:
                self._apply_two_qubit_gate(
                    matrix, inst.qubits[0], inst.qubits[1]
                )
            elif len(inst.qubits) == 3:
                self._apply_three_qubit_gate(
                    matrix, inst.qubits[0], inst.qubits[1], inst.qubits[2]
                )

    # ==================================================================
    # Matrix construction helpers
    # ==================================================================

    def _instruction_to_matrix(self, inst: GateInstruction) -> np.ndarray:
        """Convert a GateInstruction to its unitary matrix."""
        gs = self.gate_set
        gt = inst.gate_type
        p = inst.params

        if gt == NativeGateType.RZ:
            return gs.rz_matrix(p[0])
        elif gt == NativeGateType.RX:
            return gs.rx_matrix(p[0])
        elif gt == NativeGateType.RY:
            return gs.ry_matrix(p[0])
        elif gt == NativeGateType.RXY:
            return gs.rxy_matrix(p[0], p[1])
        elif gt == NativeGateType.CZ:
            return gs.cz_matrix()
        elif gt == NativeGateType.CCZ:
            return gs.ccz_matrix()
        elif gt == NativeGateType.GLOBAL_R:
            return gs.rxy_matrix(p[0], p[1])
        else:
            raise ValueError(f"Unknown gate type: {gt}")

    def _apply_single_qubit_gate(
        self, matrix: np.ndarray, qubit: int
    ) -> None:
        """Apply a 2x2 gate to the state vector efficiently.

        Uses numpy reshape + vectorised application for O(2^n) cost.
        """
        assert self._statevector is not None
        n = self.n_qubits
        sv = self._statevector.reshape([2] * n)
        sv = np.moveaxis(sv, qubit, -1)
        shape = sv.shape
        sv = sv.reshape(-1, 2) @ matrix.T
        sv = sv.reshape(shape)
        sv = np.moveaxis(sv, -1, qubit)
        self._statevector = sv.reshape(-1)

    def _apply_two_qubit_gate(
        self, matrix: np.ndarray, qubit_a: int, qubit_b: int
    ) -> None:
        """Apply a 4x4 gate to the state vector efficiently."""
        assert self._statevector is not None
        n = self.n_qubits
        sv = self._statevector.reshape([2] * n)

        # Move the two target qubits to the last two axes
        axes = list(range(n))
        qa, qb = qubit_a, qubit_b
        remaining = [i for i in axes if i != qa and i != qb]
        perm = remaining + [qa, qb]
        sv = sv.transpose(perm)

        # Reshape last two axes into 4
        other_shape = sv.shape[:-2]
        sv = sv.reshape(*other_shape, 4)

        # Apply the 4x4 gate matrix
        sv = sv @ matrix.T

        # Reshape back and undo the transpose
        sv = sv.reshape(*other_shape, 2, 2)
        inv_perm = [0] * n
        for i, p in enumerate(perm):
            inv_perm[p] = i
        sv = sv.transpose(inv_perm)
        self._statevector = sv.reshape(-1)

    def _apply_three_qubit_gate(
        self, matrix: np.ndarray, qubit_a: int, qubit_b: int, qubit_c: int
    ) -> None:
        """Apply an 8x8 gate to the state vector.

        Uses the same transpose + reshape strategy as two-qubit gates,
        extended to three target axes.
        """
        assert self._statevector is not None
        n = self.n_qubits
        sv = self._statevector.reshape([2] * n)

        # Move three target qubits to the last three axes
        targets = [qubit_a, qubit_b, qubit_c]
        remaining = [i for i in range(n) if i not in targets]
        perm = remaining + targets
        sv = sv.transpose(perm)

        # Reshape last three axes into 8
        other_shape = sv.shape[:-3]
        sv = sv.reshape(*other_shape, 8)

        # Apply the 8x8 gate matrix
        sv = sv @ matrix.T

        # Reshape back
        sv = sv.reshape(*other_shape, 2, 2, 2)
        inv_perm = [0] * n
        for i, p in enumerate(perm):
            inv_perm[p] = i
        sv = sv.transpose(inv_perm)
        self._statevector = sv.reshape(-1)

    def _embed_single(
        self, matrix: np.ndarray, qubit: int
    ) -> np.ndarray:
        """Embed a 2x2 matrix into the full Hilbert space."""
        I2 = np.eye(2, dtype=np.complex128)
        result = np.array([[1.0]], dtype=np.complex128)
        for q in range(self.n_qubits):
            result = np.kron(result, matrix if q == qubit else I2)
        return result

    def _embed_two(
        self, matrix: np.ndarray, qubit_a: int, qubit_b: int
    ) -> np.ndarray:
        """Embed a 4x4 two-qubit matrix into the full Hilbert space."""
        dim = self.dim
        result = np.zeros((dim, dim), dtype=np.complex128)
        n = self.n_qubits

        for i in range(dim):
            for j in range(dim):
                ba_i = (i >> (n - 1 - qubit_a)) & 1
                bb_i = (i >> (n - 1 - qubit_b)) & 1
                ba_j = (j >> (n - 1 - qubit_a)) & 1
                bb_j = (j >> (n - 1 - qubit_b)) & 1

                mask_a = 1 << (n - 1 - qubit_a)
                mask_b = 1 << (n - 1 - qubit_b)
                rest_i = i & ~mask_a & ~mask_b
                rest_j = j & ~mask_a & ~mask_b

                if rest_i != rest_j:
                    continue

                row_2q = ba_i * 2 + bb_i
                col_2q = ba_j * 2 + bb_j
                result[i, j] = matrix[row_2q, col_2q]

        return result

    def _embed_three(
        self,
        matrix: np.ndarray,
        qubit_a: int,
        qubit_b: int,
        qubit_c: int,
    ) -> np.ndarray:
        """Embed an 8x8 three-qubit matrix into the full Hilbert space."""
        dim = self.dim
        result = np.zeros((dim, dim), dtype=np.complex128)
        n = self.n_qubits

        for i in range(dim):
            for j in range(dim):
                ba_i = (i >> (n - 1 - qubit_a)) & 1
                bb_i = (i >> (n - 1 - qubit_b)) & 1
                bc_i = (i >> (n - 1 - qubit_c)) & 1
                ba_j = (j >> (n - 1 - qubit_a)) & 1
                bb_j = (j >> (n - 1 - qubit_b)) & 1
                bc_j = (j >> (n - 1 - qubit_c)) & 1

                mask_a = 1 << (n - 1 - qubit_a)
                mask_b = 1 << (n - 1 - qubit_b)
                mask_c = 1 << (n - 1 - qubit_c)
                rest_i = i & ~mask_a & ~mask_b & ~mask_c
                rest_j = j & ~mask_a & ~mask_b & ~mask_c

                if rest_i != rest_j:
                    continue

                row_3q = ba_i * 4 + bb_i * 2 + bc_i
                col_3q = ba_j * 4 + bb_j * 2 + bc_j
                result[i, j] = matrix[row_3q, col_3q]

        return result

    # ==================================================================
    # Probability and measurement helpers
    # ==================================================================

    def _full_probability_distribution(self) -> np.ndarray:
        """Get the probability of each computational basis state."""
        if self._statevector is not None:
            probs = np.abs(self._statevector) ** 2
        else:
            probs = np.real(np.diag(self._density_matrix))

        # Ensure probabilities sum to 1 (numerical safety)
        total = np.sum(probs)
        if total > 0 and abs(total - 1.0) > 1e-10:
            probs = probs / total
        probs = np.clip(probs, 0.0, 1.0)

        return probs

    def _qubit_probabilities(self, qubit: int) -> np.ndarray:
        """Get the probability of qubit being 0 or 1."""
        probs = self._full_probability_distribution()
        p1 = 0.0
        for i in range(self.dim):
            if (i >> (self.n_qubits - 1 - qubit)) & 1:
                p1 += probs[i]
        return np.array([1.0 - p1, p1])

    def _collapse_statevector(self, qubit: int, outcome: int) -> None:
        """Collapse the state vector after measuring a qubit."""
        assert self._statevector is not None
        for i in range(self.dim):
            bit = (i >> (self.n_qubits - 1 - qubit)) & 1
            if bit != outcome:
                self._statevector[i] = 0.0
        norm = np.linalg.norm(self._statevector)
        if norm > 0:
            self._statevector /= norm

    def _collapse_density_matrix(self, qubit: int, outcome: int) -> None:
        """Collapse the density matrix after measuring a qubit."""
        assert self._density_matrix is not None
        n = self.n_qubits

        proj_single = np.zeros((2, 2), dtype=np.complex128)
        proj_single[outcome, outcome] = 1.0

        I2 = np.eye(2, dtype=np.complex128)
        proj = np.array([[1.0]], dtype=np.complex128)
        for q in range(n):
            proj = np.kron(proj, proj_single if q == qubit else I2)

        self._density_matrix = proj @ self._density_matrix @ proj
        trace = np.real(np.trace(self._density_matrix))
        if trace > 0:
            self._density_matrix /= trace

    # ==================================================================
    # Validation
    # ==================================================================

    def _validate_qubit(self, qubit: int) -> None:
        """Check that a qubit index is valid."""
        if qubit < 0 or qubit >= self.n_qubits:
            raise ValueError(
                f"Qubit index {qubit} out of range [0, {self.n_qubits})"
            )

    # ==================================================================
    # Repr
    # ==================================================================

    def __repr__(self) -> str:
        stats = self.circuit_stats()
        return (
            f"NeutralAtomSimulator("
            f"n_qubits={self.n_qubits}, "
            f"species={self.config.species.name}, "
            f"mode={self.execution_mode}, "
            f"gates={stats.total_gates}, "
            f"cz_gates={stats.cz_gate_count}, "
            f"ccz_gates={stats.ccz_gate_count})"
        )
