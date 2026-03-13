"""End-to-end trapped-ion quantum simulator.

Provides a unified interface that bridges the three abstraction layers
(digital, analog, atomic) into a single simulator with three execution
modes:

1. **Ideal**: Perfect gate execution using state-vector simulation.
   No noise, no decoherence.  Best for algorithm development.

2. **Noisy**: Gate-level noise model using density-matrix simulation.
   Physics-based error channels from :class:`TrappedIonNoiseModel`.
   Best for realistic performance estimation.

3. **Pulse**: Full Hamiltonian-level simulation through the analog
   layer.  Each gate is decomposed into laser pulse sequences and
   time-evolved under the system Hamiltonian.  Most accurate but
   slowest.

References:
    - Wright et al., Nature Communications 10, 5464 (2019)
      [Benchmarking trapped-ion gates]
    - Debnath et al., Nature 536, 63 (2016)
      [Programmable trapped-ion quantum computer]
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .analog import AnalogCircuit, PulseSequence
from .gates import GateInstruction, NativeGateType, TrappedIonGateSet
from .noise import TrappedIonNoiseModel
from .trap import TrapConfig


@dataclass
class CircuitStats:
    """Statistics about a compiled circuit."""
    total_gates: int = 0
    single_qubit_gates: int = 0
    two_qubit_gates: int = 0
    ms_gate_count: int = 0
    estimated_fidelity: float = 1.0
    estimated_duration_us: float = 0.0


class TrappedIonSimulator:
    """End-to-end trapped-ion quantum simulator.

    Simulates quantum circuits with physics-accurate noise from the
    trapped-ion hardware model.  Supports three execution modes:
    ideal (state vector), noisy (density matrix), and pulse
    (Hamiltonian evolution).

    Parameters
    ----------
    config : TrapConfig
        Trap and ion configuration.
    noise_model : TrappedIonNoiseModel, optional
        Physics noise model.  Auto-created from config if ``None``
        and execution_mode is 'noisy'.
    execution_mode : str
        One of 'ideal', 'noisy', 'pulse'.

    Examples
    --------
    >>> from nqpu.ion_trap import TrappedIonSimulator, TrapConfig
    >>> sim = TrappedIonSimulator(TrapConfig(n_ions=3))
    >>> sim.h(0)
    >>> sim.cnot(0, 1)
    >>> sim.cnot(0, 2)
    >>> counts = sim.measure_all(shots=1000)
    >>> print(counts)  # ~ {'000': 500, '111': 500}
    """

    def __init__(
        self,
        config: TrapConfig,
        noise_model: TrappedIonNoiseModel | None = None,
        execution_mode: str = "ideal",
    ) -> None:
        if execution_mode not in ("ideal", "noisy", "pulse"):
            raise ValueError(
                f"execution_mode must be 'ideal', 'noisy', or 'pulse', "
                f"got '{execution_mode}'"
            )

        self.config = config
        self.n_qubits = config.n_ions
        self.execution_mode = execution_mode
        self.dim = 2 ** self.n_qubits
        self.gate_set = TrappedIonGateSet()

        # Noise model
        if noise_model is not None:
            self.noise_model = noise_model
        elif execution_mode == "noisy":
            self.noise_model = TrappedIonNoiseModel(config)
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

        # Analog layer (used in pulse mode)
        self._analog_circuit: AnalogCircuit | None = None
        if execution_mode == "pulse":
            self._analog_circuit = AnalogCircuit(self.n_qubits, config)

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
        """Apply CNOT gate (compiled to 1 MS + single-qubit gates)."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target must be different qubits")
        native = self.gate_set.compile_cnot(control, target)
        self._execute_native(native, "ms", control, target)

    def cz(self, control: int, target: int) -> None:
        """Apply CZ gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target must be different qubits")
        native = self.gate_set.compile_cz(control, target)
        self._execute_native(native, "ms", control, target)

    # ==================================================================
    # Native ion trap gates
    # ==================================================================

    def ms(
        self, ion_a: int, ion_b: int, theta: float = math.pi / 4
    ) -> None:
        """Apply Molmer-Sorensen gate MS(theta) on two ions.

        Parameters
        ----------
        ion_a, ion_b : int
            Ion indices.
        theta : float
            Interaction angle (default pi/4 for maximally entangling).
        """
        self._validate_qubit(ion_a)
        self._validate_qubit(ion_b)
        native = [
            GateInstruction(NativeGateType.MS, (ion_a, ion_b), (theta,))
        ]
        self._execute_native(native, "ms", ion_a, ion_b)

    def xx(self, ion_a: int, ion_b: int, theta: float) -> None:
        """Apply XX(theta) Ising interaction gate."""
        self.ms(ion_a, ion_b, theta)

    def global_rotation(self, theta: float, phi: float) -> None:
        """Apply a global rotation R(theta, phi) to all ions simultaneously.

        In trapped-ion hardware, global rotations are achieved with a
        single beam illuminating all ions --- very high fidelity and
        no additional time cost.

        Parameters
        ----------
        theta : float
            Rotation angle in radians.
        phi : float
            Axis angle in the XY plane.
        """
        for q in range(self.n_qubits):
            native = [GateInstruction(NativeGateType.R, (q,), (theta, phi))]
            self._execute_native(native, "single", q)

    # ==================================================================
    # Measurement
    # ==================================================================

    def measure(self, qubit: int) -> int:
        """Measure a single qubit, collapsing its state.

        Returns
        -------
        int
            Measurement outcome: 0 or 1.
        """
        self._validate_qubit(qubit)
        probs = self._qubit_probabilities(qubit)

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
            counts[bitstring] = counts.get(bitstring, 0) + 1

        # Sort by bitstring
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
            if inst.gate_type in (
                NativeGateType.MS,
                NativeGateType.XX,
                NativeGateType.ZZ,
            ):
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
        sq_time = 1.0  # microseconds
        tq_time = 100.0

        if self.noise_model:
            sq_time = self.noise_model.single_qubit_gate_time_us
            tq_time = self.noise_model.two_qubit_gate_time_us

        for inst in self._native_instructions:
            stats.total_gates += 1
            if inst.gate_type in (
                NativeGateType.MS,
                NativeGateType.XX,
                NativeGateType.ZZ,
            ):
                stats.two_qubit_gates += 1
                stats.ms_gate_count += 1
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
        """Return device characterisation summary.

        Includes mode frequencies, gate fidelities, and trap parameters.
        """
        freqs, modes = self.config.normal_modes()
        info: dict[str, Any] = {
            "n_qubits": self.n_qubits,
            "species": self.config.species.name,
            "execution_mode": self.execution_mode,
            "axial_freq_mhz": self.config.axial_freq_mhz,
            "radial_freq_mhz": self.config.radial_freq_mhz,
            "lamb_dicke": self.config.lamb_dicke,
            "com_mode_freq_mhz": float(freqs[0]) if len(freqs) > 0 else None,
            "mode_frequencies_mhz": freqs.tolist(),
        }
        if self.noise_model:
            info["1q_gate_fidelity"] = self.noise_model.single_qubit_gate_fidelity()
            info["2q_gate_fidelity"] = self.noise_model.two_qubit_gate_fidelity()
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
        if self._analog_circuit is not None:
            self._analog_circuit = AnalogCircuit(self.n_qubits, self.config)

    # ==================================================================
    # Circuit compilation from QASM
    # ==================================================================

    def compile_circuit(self, qasm_str: str) -> list[GateInstruction]:
        """Compile a minimal OpenQASM 2.0 string into native gate instructions.

        Supports: h, x, y, z, rx, ry, rz, cx, cz, measure.

        Parameters
        ----------
        qasm_str : str
            OpenQASM 2.0 circuit string.

        Returns
        -------
        list[GateInstruction]
            Compiled native gate sequence.
        """
        instructions: list[GateInstruction] = []

        for line in qasm_str.strip().split("\n"):
            line = line.strip().rstrip(";")
            if not line or line.startswith("//") or line.startswith("OPENQASM"):
                continue
            if line.startswith("include") or line.startswith("qreg") or line.startswith("creg"):
                continue

            # Parse gate and arguments
            if "(" in line:
                gate_part, rest = line.split("(", 1)
                param_str, qubit_part = rest.split(")", 1)
                gate_name = gate_part.strip()
                params = [float(p.strip().replace("pi", str(math.pi)))
                          for p in param_str.split(",")]
            else:
                parts = line.split()
                gate_name = parts[0]
                qubit_part = " ".join(parts[1:])
                params = []

            # Extract qubit indices
            qubits = []
            for token in qubit_part.replace(",", " ").split():
                token = token.strip()
                if token.startswith("q[") and token.endswith("]"):
                    qubits.append(int(token[2:-1]))
                elif token.startswith("q"):
                    try:
                        qubits.append(int(token[1:]))
                    except ValueError:
                        pass

            # Compile to native gates
            if gate_name == "h" and len(qubits) == 1:
                instructions.extend(self.gate_set.compile_h(qubits[0]))
            elif gate_name == "x" and len(qubits) == 1:
                instructions.extend(self.gate_set.compile_x(qubits[0]))
            elif gate_name == "y" and len(qubits) == 1:
                instructions.extend(self.gate_set.compile_y(qubits[0]))
            elif gate_name == "z" and len(qubits) == 1:
                instructions.extend(self.gate_set.compile_z(qubits[0]))
            elif gate_name == "rx" and len(qubits) == 1 and len(params) == 1:
                instructions.extend(
                    self.gate_set.compile_rx(qubits[0], params[0])
                )
            elif gate_name == "ry" and len(qubits) == 1 and len(params) == 1:
                instructions.extend(
                    self.gate_set.compile_ry(qubits[0], params[0])
                )
            elif gate_name == "rz" and len(qubits) == 1 and len(params) == 1:
                instructions.extend(
                    self.gate_set.compile_rz(qubits[0], params[0])
                )
            elif gate_name in ("cx", "cnot") and len(qubits) == 2:
                instructions.extend(
                    self.gate_set.compile_cnot(qubits[0], qubits[1])
                )
            elif gate_name == "cz" and len(qubits) == 2:
                instructions.extend(
                    self.gate_set.compile_cz(qubits[0], qubits[1])
                )

        return instructions

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
                self._apply_two_qubit_gate(matrix, inst.qubits[0], inst.qubits[1])

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
            else:
                full_op = self._embed_two(matrix, inst.qubits[0], inst.qubits[1])

            # Unitary evolution
            self._density_matrix = full_op @ self._density_matrix @ full_op.conj().T

        # Apply noise for the whole gate operation
        if gate_category == "ms":
            gate_time = self.noise_model.two_qubit_gate_time_us
            self._density_matrix = self.noise_model.apply_noise(
                self._density_matrix, "ms", gate_time, involved_qubits
            )
        elif gate_category == "single":
            gate_time = self.noise_model.single_qubit_gate_time_us
            self._density_matrix = self.noise_model.apply_noise(
                self._density_matrix, "single", gate_time, involved_qubits
            )

    def _execute_pulse(self, instructions: list[GateInstruction]) -> None:
        """Execute gates via Hamiltonian evolution (pulse mode)."""
        assert self._analog_circuit is not None

        for inst in instructions:
            if inst.gate_type == NativeGateType.MS:
                theta = inst.params[0]
                # MS gate: need Omega * t = theta, with typical Omega ~ 0.01 MHz
                rabi = 0.01  # MHz
                duration = theta / (2.0 * math.pi * rabi)
                self._analog_circuit.add_ms_interaction(
                    inst.qubits[0],
                    inst.qubits[1],
                    rabi_freq_mhz=rabi,
                    detuning_mhz=self.config.axial_freq_mhz,
                    duration_us=duration,
                )
            elif inst.gate_type == NativeGateType.RZ:
                theta = inst.params[0]
                # Stark shift: delta * t = theta
                shift = 1.0  # MHz
                duration = theta / (2.0 * math.pi * shift)
                self._analog_circuit.add_stark_shift(
                    inst.qubits[0],
                    shift_mhz=shift,
                    duration_us=abs(duration),
                )
            elif inst.gate_type in (NativeGateType.R, NativeGateType.RX, NativeGateType.RY):
                if inst.gate_type == NativeGateType.R:
                    theta, phi = inst.params[0], inst.params[1]
                elif inst.gate_type == NativeGateType.RX:
                    theta, phi = inst.params[0], 0.0
                else:  # RY
                    theta, phi = inst.params[0], math.pi / 2.0
                rabi = 1.0  # MHz
                duration = abs(theta) / (2.0 * math.pi * rabi)
                self._analog_circuit.add_rabi_drive(
                    inst.qubits[0],
                    rabi_freq_mhz=rabi,
                    phase=phi,
                    duration_us=duration,
                )

        # Propagate the state through the analog circuit
        self._statevector = self._analog_circuit.simulate(
            initial_state=self._statevector
        )
        # Reset the analog circuit for next gate batch
        self._analog_circuit = AnalogCircuit(self.n_qubits, self.config)

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
        elif gt == NativeGateType.R:
            return gs.r_matrix(p[0], p[1])
        elif gt == NativeGateType.MS:
            return gs.ms_matrix(p[0])
        elif gt == NativeGateType.XX:
            return gs.xx_matrix(p[0])
        elif gt == NativeGateType.ZZ:
            return gs.zz_matrix(p[0])
        else:
            raise ValueError(f"Unknown gate type: {gt}")

    def _apply_single_qubit_gate(
        self, matrix: np.ndarray, qubit: int
    ) -> None:
        """Apply a 2x2 gate to the state vector efficiently.

        Uses numpy reshape + einsum for vectorized O(2^n) application.
        """
        assert self._statevector is not None
        n = self.n_qubits
        sv = self._statevector.reshape([2] * n)
        # Contract gate matrix with the qubit axis
        sv = np.moveaxis(sv, qubit, -1)
        shape = sv.shape
        sv = sv.reshape(-1, 2) @ matrix.T
        sv = sv.reshape(shape)
        sv = np.moveaxis(sv, -1, qubit)
        self._statevector = sv.reshape(-1)

    def _apply_two_qubit_gate(
        self, matrix: np.ndarray, qubit_a: int, qubit_b: int
    ) -> None:
        """Apply a 4x4 gate to the state vector efficiently.

        Uses numpy reshape + vectorized indexing for O(2^N) application
        without building the full 2^N x 2^N embedding.
        """
        assert self._statevector is not None
        n = self.n_qubits
        sv = self._statevector.reshape([2] * n)

        # Move the two target qubits to the last two axes
        axes = list(range(n))
        # Sort so we remove higher index first to avoid shift
        qa, qb = qubit_a, qubit_b
        remaining = [i for i in axes if i != qa and i != qb]
        perm = remaining + [qa, qb]
        sv = sv.transpose(perm)

        # Now shape is (*other_dims, 2, 2) — reshape last two into 4
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

    # ==================================================================
    # Probability and measurement helpers
    # ==================================================================

    def _full_probability_distribution(self) -> np.ndarray:
        """Get the probability of each computational basis state."""
        if self._statevector is not None:
            return np.abs(self._statevector) ** 2
        else:
            return np.real(np.diag(self._density_matrix))

    def _qubit_probabilities(self, qubit: int) -> np.ndarray:
        """Get the probability of qubit being 0 or 1."""
        probs = self._full_probability_distribution()
        p1 = 0.0
        step = 2 ** (self.n_qubits - 1 - qubit)
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
        # Renormalise
        norm = np.linalg.norm(self._statevector)
        if norm > 0:
            self._statevector /= norm

    def _collapse_density_matrix(self, qubit: int, outcome: int) -> None:
        """Collapse the density matrix after measuring a qubit."""
        assert self._density_matrix is not None
        n = self.n_qubits

        # Build projector |outcome><outcome| on the target qubit
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
            f"TrappedIonSimulator("
            f"n_qubits={self.n_qubits}, "
            f"species={self.config.species.name}, "
            f"mode={self.execution_mode}, "
            f"gates={stats.total_gates}, "
            f"ms_gates={stats.ms_gate_count})"
        )
