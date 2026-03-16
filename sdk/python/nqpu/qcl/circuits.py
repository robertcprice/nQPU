"""Parameterized quantum circuits for quantum circuit learning.

Provides data-encoding circuits, trainable ansatz circuits, and a combined
:class:`CircuitTemplate` that wires encoding and ansatz into a full QCL
circuit.  All circuits operate on a built-in statevector simulator so the
package has zero external quantum-framework dependencies.

Design principles
-----------------
- Every circuit is fully described by a list of :class:`GateOp` instructions.
- Parameters are identified by string names and can be bound to concrete
  float values to produce a statevector.
- The statevector simulator is deliberately minimal: single-qubit rotations,
  CNOT, and CZ are sufficient for all supported ansaetze.

References
----------
- Havlicek et al., Nature 567, 209 (2019) [IQP encoding, quantum kernels]
- Schuld et al., Phys. Rev. A 101, 032308 (2020) [data re-uploading]
- Sim et al., Adv. Quantum Technol. 2, 1900070 (2019) [expressibility]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import numpy as np


# ------------------------------------------------------------------
# Gate definitions
# ------------------------------------------------------------------


class GateType(Enum):
    """Supported gate types for the parameterized circuit."""

    RX = auto()
    RY = auto()
    RZ = auto()
    CNOT = auto()
    CZ = auto()
    H = auto()
    X = auto()
    Z = auto()


@dataclass(frozen=True)
class GateOp:
    """A single gate operation in a parameterized circuit.

    Attributes
    ----------
    gate : GateType
        Which gate to apply.
    qubits : tuple[int, ...]
        Target qubit indices (1 for single-qubit, 2 for two-qubit gates).
    param_name : str or None
        Name of the parameter this gate depends on, or ``None`` for fixed gates.
    param_value : float or None
        Fixed angle for non-parameterized rotation gates.
    """

    gate: GateType
    qubits: tuple[int, ...]
    param_name: str | None = None
    param_value: float | None = None


# ------------------------------------------------------------------
# Statevector simulator
# ------------------------------------------------------------------


class StatevectorSimulator:
    """Minimal statevector simulator for parameterized circuits.

    Operates on a complex128 statevector of 2^n amplitudes.  Supports
    single-qubit rotations (RX, RY, RZ), Hadamard, X, Z, CNOT, and CZ.
    """

    @staticmethod
    def initial_state(n_qubits: int) -> np.ndarray:
        """Return the |00...0> state for *n_qubits*."""
        state = np.zeros(1 << n_qubits, dtype=np.complex128)
        state[0] = 1.0
        return state

    @staticmethod
    def _apply_single(
        state: np.ndarray, n_qubits: int, qubit: int, matrix: np.ndarray
    ) -> np.ndarray:
        """Apply a 2x2 unitary to a single qubit."""
        dim = 1 << n_qubits
        new_state = np.zeros(dim, dtype=np.complex128)
        step = 1 << qubit
        for i in range(dim):
            if i & step == 0:
                j = i | step
                a, b = state[i], state[j]
                new_state[i] += matrix[0, 0] * a + matrix[0, 1] * b
                new_state[j] += matrix[1, 0] * a + matrix[1, 1] * b
        return new_state

    @staticmethod
    def _apply_cnot(
        state: np.ndarray, n_qubits: int, control: int, target: int
    ) -> np.ndarray:
        """Apply CNOT: flip target when control is |1>."""
        dim = 1 << n_qubits
        new_state = state.copy()
        c_step = 1 << control
        t_step = 1 << target
        for i in range(dim):
            if (i & c_step) != 0 and (i & t_step) == 0:
                j = i | t_step
                new_state[i], new_state[j] = state[j], state[i]
        return new_state

    @staticmethod
    def _apply_cz(
        state: np.ndarray, n_qubits: int, q0: int, q1: int
    ) -> np.ndarray:
        """Apply CZ: phase flip when both qubits are |1>."""
        dim = 1 << n_qubits
        new_state = state.copy()
        s0 = 1 << q0
        s1 = 1 << q1
        for i in range(dim):
            if (i & s0) != 0 and (i & s1) != 0:
                new_state[i] = -state[i]
        return new_state

    @staticmethod
    def _rotation_matrix(gate: GateType, angle: float) -> np.ndarray:
        """Return the 2x2 rotation matrix for a rotation gate."""
        c = math.cos(angle / 2)
        s = math.sin(angle / 2)
        if gate == GateType.RX:
            return np.array(
                [[c, -1j * s], [-1j * s, c]], dtype=np.complex128
            )
        elif gate == GateType.RY:
            return np.array([[c, -s], [s, c]], dtype=np.complex128)
        elif gate == GateType.RZ:
            return np.array(
                [[c - 1j * s, 0], [0, c + 1j * s]], dtype=np.complex128
            )
        raise ValueError(f"Not a rotation gate: {gate}")

    _H_MATRIX = np.array(
        [[1, 1], [1, -1]], dtype=np.complex128
    ) / math.sqrt(2)
    _X_MATRIX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    _Z_MATRIX = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    @classmethod
    def run(
        cls,
        n_qubits: int,
        ops: list[GateOp],
        param_bindings: dict[str, float],
    ) -> np.ndarray:
        """Execute a circuit and return the final statevector.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        ops : list[GateOp]
            Gate operations to apply in order.
        param_bindings : dict[str, float]
            Map from parameter names to float values.

        Returns
        -------
        np.ndarray
            Complex statevector of length 2^n_qubits.

        Raises
        ------
        ValueError
            If a required parameter is missing from *param_bindings*.
        """
        state = cls.initial_state(n_qubits)

        for op in ops:
            if op.gate in (GateType.RX, GateType.RY, GateType.RZ):
                if op.param_name is not None:
                    if op.param_name not in param_bindings:
                        raise ValueError(
                            f"Missing parameter binding for '{op.param_name}'"
                        )
                    angle = param_bindings[op.param_name]
                elif op.param_value is not None:
                    angle = op.param_value
                else:
                    raise ValueError(
                        f"Rotation gate on qubit {op.qubits} has no parameter"
                    )
                mat = cls._rotation_matrix(op.gate, angle)
                state = cls._apply_single(state, n_qubits, op.qubits[0], mat)

            elif op.gate == GateType.H:
                state = cls._apply_single(
                    state, n_qubits, op.qubits[0], cls._H_MATRIX
                )

            elif op.gate == GateType.X:
                state = cls._apply_single(
                    state, n_qubits, op.qubits[0], cls._X_MATRIX
                )

            elif op.gate == GateType.Z:
                state = cls._apply_single(
                    state, n_qubits, op.qubits[0], cls._Z_MATRIX
                )

            elif op.gate == GateType.CNOT:
                state = cls._apply_cnot(
                    state, n_qubits, op.qubits[0], op.qubits[1]
                )

            elif op.gate == GateType.CZ:
                state = cls._apply_cz(
                    state, n_qubits, op.qubits[0], op.qubits[1]
                )

        return state


# ------------------------------------------------------------------
# Base circuit class
# ------------------------------------------------------------------


class ParameterizedCircuit:
    """A quantum circuit with named parameters and gate operations.

    This is the fundamental building block: a sequence of gates that may
    depend on named parameters.  Subclasses (encoding circuits, ansatz
    circuits) populate the operation list via their constructors.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    """

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self.ops: list[GateOp] = []
        self._param_names: set[str] = set()

    @property
    def parameter_names(self) -> list[str]:
        """Sorted list of all parameter names in this circuit."""
        return sorted(self._param_names)

    @property
    def n_params(self) -> int:
        """Number of unique parameters."""
        return len(self._param_names)

    def add_gate(self, op: GateOp) -> None:
        """Append a gate operation."""
        self.ops.append(op)
        if op.param_name is not None:
            self._param_names.add(op.param_name)

    def run(self, param_bindings: dict[str, float]) -> np.ndarray:
        """Execute the circuit and return the statevector."""
        return StatevectorSimulator.run(self.n_qubits, self.ops, param_bindings)

    def expectation_z(
        self, param_bindings: dict[str, float], qubit: int = 0
    ) -> float:
        """Compute <Z> expectation value on a single qubit.

        Parameters
        ----------
        param_bindings : dict
            Parameter name -> value mapping.
        qubit : int
            Which qubit to measure Z on.

        Returns
        -------
        float
            Expectation value in [-1, 1].
        """
        state = self.run(param_bindings)
        n = self.n_qubits
        dim = 1 << n
        exp_val = 0.0
        step = 1 << qubit
        for i in range(dim):
            p = abs(state[i]) ** 2
            if i & step:
                exp_val -= p
            else:
                exp_val += p
        return exp_val

    def probabilities(self, param_bindings: dict[str, float]) -> np.ndarray:
        """Compute measurement probabilities in the computational basis.

        Returns
        -------
        np.ndarray
            Probability vector of length 2^n_qubits.
        """
        state = self.run(param_bindings)
        return np.abs(state) ** 2


# ------------------------------------------------------------------
# Data encoding circuits
# ------------------------------------------------------------------


class DataEncodingCircuit(ParameterizedCircuit):
    """Base class for circuits that encode classical data into quantum states.

    Subclasses generate gate operations that map an input feature vector
    ``x`` to parameter bindings.
    """

    def __init__(self, n_qubits: int) -> None:
        super().__init__(n_qubits)

    def feature_bindings(self, x: np.ndarray) -> dict[str, float]:
        """Map a classical feature vector to parameter bindings.

        Parameters
        ----------
        x : np.ndarray
            Feature vector.

        Returns
        -------
        dict[str, float]
            Parameter bindings for the encoding gates.
        """
        raise NotImplementedError

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode a feature vector and return the statevector."""
        return self.run(self.feature_bindings(x))


class AngleEncoding(DataEncodingCircuit):
    """Encode features as rotation angles on individual qubits.

    Each feature ``x[i]`` is mapped to a rotation on qubit *i*.
    If there are more features than qubits, features are wrapped
    (qubit ``i`` gets ``x[i] + x[i + n_qubits] + ...``).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    rotation : str
        Rotation axis: ``'Y'`` (default) or ``'Z'``.
    """

    def __init__(self, n_qubits: int, rotation: str = "Y") -> None:
        super().__init__(n_qubits)
        if rotation.upper() not in ("Y", "Z"):
            raise ValueError(f"rotation must be 'Y' or 'Z', got '{rotation}'")
        self.rotation = rotation.upper()
        gate = GateType.RY if self.rotation == "Y" else GateType.RZ
        for q in range(n_qubits):
            name = f"x_{q}"
            self.add_gate(GateOp(gate=gate, qubits=(q,), param_name=name))

    def feature_bindings(self, x: np.ndarray) -> dict[str, float]:
        """Map features to rotation angles with wrapping."""
        x = np.asarray(x, dtype=np.float64).ravel()
        bindings: dict[str, float] = {}
        for q in range(self.n_qubits):
            total_angle = 0.0
            idx = q
            while idx < len(x):
                total_angle += x[idx]
                idx += self.n_qubits
            bindings[f"x_{q}"] = total_angle
        return bindings


class AmplitudeEncoding(DataEncodingCircuit):
    """Encode a normalized feature vector as quantum amplitudes.

    The input vector is normalized to unit norm and directly set as the
    statevector amplitudes.  This is not implementable with local gates
    in general, but is useful as an idealized encoding for kernel methods.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.  Feature dimension must be <= 2^n_qubits.
    """

    def __init__(self, n_qubits: int) -> None:
        super().__init__(n_qubits)
        # No gates: we override run() directly.

    def feature_bindings(self, x: np.ndarray) -> dict[str, float]:
        """Amplitude encoding uses no parameterized gates."""
        return {}

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode feature vector as amplitudes (normalized).

        Parameters
        ----------
        x : np.ndarray
            Feature vector of length <= 2^n_qubits.

        Returns
        -------
        np.ndarray
            Statevector with x as amplitudes (zero-padded, normalized).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        dim = 1 << self.n_qubits
        if len(x) > dim:
            raise ValueError(
                f"Feature dimension {len(x)} exceeds 2^{self.n_qubits} = {dim}"
            )
        state = np.zeros(dim, dtype=np.complex128)
        state[: len(x)] = x
        norm = np.linalg.norm(state)
        if norm < 1e-15:
            raise ValueError("Feature vector has near-zero norm")
        state /= norm
        return state


class IQPEncoding(DataEncodingCircuit):
    """Instantaneous Quantum Polynomial (IQP) encoding.

    Applies Hadamard gates, then RZ rotations encoding features, then
    ZZ interaction terms between adjacent qubits encoding product features.
    This creates entanglement-dependent feature maps used in quantum
    kernel methods (Havlicek et al., Nature 2019).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_repeats : int
        Number of times to repeat the encoding block.
    """

    def __init__(self, n_qubits: int, n_repeats: int = 1) -> None:
        super().__init__(n_qubits)
        if n_repeats < 1:
            raise ValueError("n_repeats must be >= 1")
        self.n_repeats = n_repeats

        for rep in range(n_repeats):
            # Hadamard layer
            for q in range(n_qubits):
                self.add_gate(GateOp(gate=GateType.H, qubits=(q,)))

            # Single-qubit RZ encoding
            for q in range(n_qubits):
                self.add_gate(
                    GateOp(
                        gate=GateType.RZ,
                        qubits=(q,),
                        param_name=f"iqp_x_{q}_r{rep}",
                    )
                )

            # Two-qubit ZZ interaction: implemented as CNOT-RZ-CNOT
            for q in range(n_qubits - 1):
                self.add_gate(
                    GateOp(
                        gate=GateType.CNOT, qubits=(q, q + 1)
                    )
                )
                self.add_gate(
                    GateOp(
                        gate=GateType.RZ,
                        qubits=(q + 1,),
                        param_name=f"iqp_xx_{q}_{q+1}_r{rep}",
                    )
                )
                self.add_gate(
                    GateOp(
                        gate=GateType.CNOT, qubits=(q, q + 1)
                    )
                )

    def feature_bindings(self, x: np.ndarray) -> dict[str, float]:
        """Map features to IQP encoding parameters.

        Single-qubit terms use x[i], two-qubit terms use x[i]*x[j].
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        bindings: dict[str, float] = {}
        for rep in range(self.n_repeats):
            for q in range(self.n_qubits):
                val = x[q] if q < len(x) else 0.0
                bindings[f"iqp_x_{q}_r{rep}"] = val
            for q in range(self.n_qubits - 1):
                v0 = x[q] if q < len(x) else 0.0
                v1 = x[q + 1] if (q + 1) < len(x) else 0.0
                bindings[f"iqp_xx_{q}_{q+1}_r{rep}"] = v0 * v1
        return bindings


# ------------------------------------------------------------------
# Ansatz circuits (trainable)
# ------------------------------------------------------------------


class AnsatzCircuit(ParameterizedCircuit):
    """Base class for trainable ansatz circuits.

    Subclasses build a fixed circuit structure with named parameters
    that are optimized during training.
    """

    def __init__(self, n_qubits: int, n_layers: int = 1) -> None:
        super().__init__(n_qubits)
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_layers = n_layers

    def param_vector_to_bindings(self, params: np.ndarray) -> dict[str, float]:
        """Convert a flat parameter vector to named bindings.

        Parameters are assigned to parameter names in sorted order.
        """
        names = self.parameter_names
        if len(params) != len(names):
            raise ValueError(
                f"Expected {len(names)} parameters, got {len(params)}"
            )
        return dict(zip(names, params))

    def bindings_to_param_vector(self, bindings: dict[str, float]) -> np.ndarray:
        """Convert named bindings to a flat parameter vector (sorted order)."""
        names = self.parameter_names
        return np.array([bindings[n] for n in names], dtype=np.float64)


class HardwareEfficientAnsatz(AnsatzCircuit):
    """Hardware-efficient ansatz with RY-RZ rotation layers and CNOT entanglement.

    Each layer consists of:
    1. RY rotation on each qubit
    2. RZ rotation on each qubit
    3. Linear CNOT entangling chain (qubit i -> qubit i+1)

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of ansatz layers.
    """

    def __init__(self, n_qubits: int, n_layers: int = 1) -> None:
        super().__init__(n_qubits, n_layers)
        for layer in range(n_layers):
            # RY rotations
            for q in range(n_qubits):
                self.add_gate(
                    GateOp(
                        gate=GateType.RY,
                        qubits=(q,),
                        param_name=f"ry_{layer}_{q}",
                    )
                )
            # RZ rotations
            for q in range(n_qubits):
                self.add_gate(
                    GateOp(
                        gate=GateType.RZ,
                        qubits=(q,),
                        param_name=f"rz_{layer}_{q}",
                    )
                )
            # CNOT chain
            for q in range(n_qubits - 1):
                self.add_gate(
                    GateOp(gate=GateType.CNOT, qubits=(q, q + 1))
                )


class StronglyEntanglingLayers(AnsatzCircuit):
    """Strongly entangling layers with full rotations and long-range CNOTs.

    Each layer consists of:
    1. RX-RY-RZ rotations on each qubit (3 params per qubit)
    2. CNOT connections with increasing range: layer *l* connects
       qubit *i* to qubit *(i + l + 1) mod n_qubits*.

    This creates a rich entanglement pattern that covers the full
    Hilbert space more efficiently than nearest-neighbor topologies.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be >= 2).
    n_layers : int
        Number of ansatz layers.
    """

    def __init__(self, n_qubits: int, n_layers: int = 1) -> None:
        if n_qubits < 2:
            raise ValueError("StronglyEntanglingLayers requires n_qubits >= 2")
        super().__init__(n_qubits, n_layers)
        for layer in range(n_layers):
            # Full rotations (3 angles per qubit)
            for q in range(n_qubits):
                for axis_idx, gate in enumerate(
                    [GateType.RX, GateType.RY, GateType.RZ]
                ):
                    self.add_gate(
                        GateOp(
                            gate=gate,
                            qubits=(q,),
                            param_name=f"sel_{layer}_{q}_{axis_idx}",
                        )
                    )
            # Long-range CNOT pattern
            step = (layer % (n_qubits - 1)) + 1
            for q in range(n_qubits):
                target = (q + step) % n_qubits
                if target != q:
                    self.add_gate(
                        GateOp(gate=GateType.CNOT, qubits=(q, target))
                    )


class SimplifiedTwoDesign(AnsatzCircuit):
    """Simplified 2-design ansatz (Cerezo et al., 2021).

    Initial layer of RY rotations followed by blocks of controlled-Z
    gates and RY rotations.  This structure approximates a unitary
    2-design and has favorable trainability properties.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of ansatz layers.
    """

    def __init__(self, n_qubits: int, n_layers: int = 1) -> None:
        super().__init__(n_qubits, n_layers)

        # Initial RY layer
        for q in range(n_qubits):
            self.add_gate(
                GateOp(
                    gate=GateType.RY,
                    qubits=(q,),
                    param_name=f"s2d_init_{q}",
                )
            )

        for layer in range(n_layers):
            # CZ entangling block (even pairs, then odd pairs)
            for start in (0, 1):
                for q in range(start, n_qubits - 1, 2):
                    self.add_gate(
                        GateOp(gate=GateType.CZ, qubits=(q, q + 1))
                    )

            # RY rotation layer
            for q in range(n_qubits):
                self.add_gate(
                    GateOp(
                        gate=GateType.RY,
                        qubits=(q,),
                        param_name=f"s2d_{layer}_{q}",
                    )
                )


# ------------------------------------------------------------------
# Combined circuit template
# ------------------------------------------------------------------


class CircuitTemplate:
    """Combine a data encoding circuit and an ansatz into a full QCL circuit.

    This is the main user-facing class for building quantum machine learning
    models.  It manages the encoding of input data and the trainable
    parameters of the ansatz, providing convenience methods for computing
    expectation values and measurement probabilities.

    Parameters
    ----------
    encoding : DataEncodingCircuit
        Circuit for encoding classical data.
    ansatz : AnsatzCircuit
        Trainable circuit with variational parameters.

    Raises
    ------
    ValueError
        If encoding and ansatz have different numbers of qubits.
    """

    def __init__(
        self,
        encoding: DataEncodingCircuit,
        ansatz: AnsatzCircuit,
    ) -> None:
        if encoding.n_qubits != ansatz.n_qubits:
            raise ValueError(
                f"Qubit count mismatch: encoding has {encoding.n_qubits}, "
                f"ansatz has {ansatz.n_qubits}"
            )
        self.encoding = encoding
        self.ansatz = ansatz
        self.n_qubits = encoding.n_qubits

        # Build combined circuit
        self._combined = ParameterizedCircuit(self.n_qubits)
        for op in encoding.ops:
            self._combined.add_gate(op)
        for op in ansatz.ops:
            self._combined.add_gate(op)

    @property
    def n_params(self) -> int:
        """Number of trainable parameters (ansatz only)."""
        return self.ansatz.n_params

    @property
    def parameter_names(self) -> list[str]:
        """Sorted list of trainable parameter names."""
        return self.ansatz.parameter_names

    def _full_bindings(
        self, x: np.ndarray, params: np.ndarray
    ) -> dict[str, float]:
        """Merge encoding bindings with ansatz parameter bindings."""
        bindings = self.encoding.feature_bindings(x)
        bindings.update(self.ansatz.param_vector_to_bindings(params))
        return bindings

    def run(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Execute the full circuit and return the statevector.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector.
        params : np.ndarray
            Trainable parameter vector (length ``n_params``).

        Returns
        -------
        np.ndarray
            Final statevector.
        """
        return self._combined.run(self._full_bindings(x, params))

    def expectation_z(
        self, x: np.ndarray, params: np.ndarray, qubit: int = 0
    ) -> float:
        """Compute <Z> on a qubit after encoding x and applying the ansatz."""
        return self._combined.expectation_z(
            self._full_bindings(x, params), qubit
        )

    def probabilities(
        self, x: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        """Compute measurement probabilities in the computational basis."""
        return self._combined.probabilities(self._full_bindings(x, params))

    def overlap(
        self, x1: np.ndarray, x2: np.ndarray, params: np.ndarray
    ) -> float:
        """Compute |<psi(x1)|psi(x2)>|^2 (fidelity between encoded states).

        Parameters
        ----------
        x1, x2 : np.ndarray
            Two input feature vectors.
        params : np.ndarray
            Shared trainable parameters.

        Returns
        -------
        float
            Fidelity in [0, 1].
        """
        sv1 = self.run(x1, params)
        sv2 = self.run(x2, params)
        return float(abs(np.vdot(sv1, sv2)) ** 2)
