"""Analog layer -- Hamiltonian-level and pulse-level simulation.

Implements the analog and atomic abstraction layers inspired by
Open Quantum Design's full-stack architecture:

- **Analog layer** (openQSIM IR): Continuous Hamiltonian evolution
  U(t) = T exp(-i integral H(t) dt)

- **Atomic layer** (openAPL IR): Laser pulse sequences describing
  physical operations on individual ions.

References:
    - OQD architecture: https://openquantumdesign.org
    - Sorensen & Molmer, Phys. Rev. A 62, 022311 (2000)
    - Leibfried et al., Rev. Mod. Phys. 75, 281 (2003)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .trap import TrapConfig


# ======================================================================
# Pauli matrices (module-level constants)
# ======================================================================
_SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_SIGMA_P = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # sigma+
_SIGMA_M = np.array([[0, 0], [1, 0]], dtype=np.complex128)  # sigma-
_I2 = np.eye(2, dtype=np.complex128)


def _embed_operator(
    op: np.ndarray, target: int, n_qubits: int
) -> np.ndarray:
    """Embed a 2x2 operator into the full Hilbert space."""
    result = np.array([[1.0]], dtype=np.complex128)
    for q in range(n_qubits):
        result = np.kron(result, op if q == target else _I2)
    return result


def _embed_two_qubit_op(
    op: np.ndarray, qubit_a: int, qubit_b: int, n_qubits: int
) -> np.ndarray:
    """Embed a 4x4 two-qubit operator.

    Constructs via tensor products of single-qubit contributions.
    ``op`` is assumed to act on the subspace of (qubit_a, qubit_b).
    """
    dim = 2 ** n_qubits
    # Build index mapping for the two-qubit subspace
    result = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(dim):
        for j in range(dim):
            # Extract the 2-bit values for qubit_a and qubit_b
            ba_i = (i >> (n_qubits - 1 - qubit_a)) & 1
            bb_i = (i >> (n_qubits - 1 - qubit_b)) & 1
            ba_j = (j >> (n_qubits - 1 - qubit_a)) & 1
            bb_j = (j >> (n_qubits - 1 - qubit_b)) & 1

            # Remaining bits must match
            mask_a = 1 << (n_qubits - 1 - qubit_a)
            mask_b = 1 << (n_qubits - 1 - qubit_b)
            rest_i = i & ~mask_a & ~mask_b
            rest_j = j & ~mask_a & ~mask_b

            if rest_i != rest_j:
                continue

            # 2-qubit indices
            row_2q = ba_i * 2 + bb_i
            col_2q = ba_j * 2 + bb_j
            result[i, j] = op[row_2q, col_2q]

    return result


# ======================================================================
# Analog Circuit
# ======================================================================

@dataclass
class EvolutionStep:
    """A single Hamiltonian evolution step."""
    hamiltonian: np.ndarray
    duration_us: float
    label: str = ""


class AnalogCircuit:
    """Analog quantum circuit using Hamiltonian evolution.

    Directly inspired by OQD's analog interface (openQSIM IR).

    Instead of discrete gates, specifies continuous Hamiltonian
    evolution: U(t) = T exp(-i * integral H(t) dt).

    For time-independent segments, the propagator is simply
    U = exp(-i * H * t).

    Parameters
    ----------
    n_ions : int
        Number of ions (qubits).
    config : TrapConfig
        Trap configuration for physical parameters.
    """

    def __init__(self, n_ions: int, config: "TrapConfig") -> None:
        self.n_ions = n_ions
        self.config = config
        self.dim = 2 ** n_ions
        self.steps: list[EvolutionStep] = []

    def add_evolution(
        self,
        hamiltonian: np.ndarray,
        duration_us: float,
        label: str = "",
    ) -> None:
        """Add a time-independent Hamiltonian evolution segment.

        Parameters
        ----------
        hamiltonian : np.ndarray
            Hermitian operator of shape (2^n, 2^n).  Units: MHz
            (so that H*t gives dimensionless phase in radians when
            t is in microseconds and H carries a factor of 2*pi).
        duration_us : float
            Evolution time in microseconds.
        label : str, optional
            Descriptive label for this segment.
        """
        if hamiltonian.shape != (self.dim, self.dim):
            raise ValueError(
                f"Hamiltonian shape {hamiltonian.shape} does not match "
                f"Hilbert space dimension {self.dim}"
            )
        self.steps.append(EvolutionStep(hamiltonian, duration_us, label))

    def add_ms_interaction(
        self,
        ion_a: int,
        ion_b: int,
        rabi_freq_mhz: float,
        detuning_mhz: float,
        duration_us: float,
    ) -> None:
        """Add a Molmer-Sorensen-type XX interaction Hamiltonian.

        H_MS = Omega * (sigma_x^a x sigma_x^b)

        In a real MS gate, the bichromatic drive at omega_qubit +/- delta
        generates an effective spin-spin coupling.  Here we use the
        effective Hamiltonian in the Lamb-Dicke regime.

        Parameters
        ----------
        ion_a, ion_b : int
            Ion indices.
        rabi_freq_mhz : float
            Effective Rabi frequency of the XX interaction in MHz.
        detuning_mhz : float
            Detuning from the motional sideband in MHz (for bookkeeping;
            the effective Hamiltonian absorbs this).
        duration_us : float
            Interaction duration in microseconds.
        """
        # Build XX Hamiltonian
        Xa = _embed_operator(_SIGMA_X, ion_a, self.n_ions)
        Xb = _embed_operator(_SIGMA_X, ion_b, self.n_ions)
        H = rabi_freq_mhz * 2.0 * math.pi * (Xa @ Xb)

        self.add_evolution(
            H,
            duration_us,
            label=f"MS({ion_a},{ion_b}) Omega={rabi_freq_mhz:.3f}MHz",
        )

    def add_rabi_drive(
        self,
        ion: int,
        rabi_freq_mhz: float,
        phase: float,
        duration_us: float,
    ) -> None:
        """Add a resonant Rabi drive on a single ion.

        H = (Omega/2) * (cos(phi)*sigma_x + sin(phi)*sigma_y)

        Parameters
        ----------
        ion : int
            Ion index.
        rabi_freq_mhz : float
            Rabi frequency in MHz.
        phase : float
            Drive phase in radians.
        duration_us : float
            Pulse duration in microseconds.
        """
        driven_op = (
            math.cos(phase) * _SIGMA_X + math.sin(phase) * _SIGMA_Y
        )
        H = (rabi_freq_mhz * 2.0 * math.pi / 2.0) * _embed_operator(
            driven_op, ion, self.n_ions
        )
        self.add_evolution(
            H,
            duration_us,
            label=f"Rabi(ion={ion}) Omega={rabi_freq_mhz:.3f}MHz phi={phase:.3f}",
        )

    def add_stark_shift(
        self,
        ion: int,
        shift_mhz: float,
        duration_us: float,
    ) -> None:
        """Add an AC Stark shift (Z rotation) on a single ion.

        H = (delta/2) * sigma_z

        Parameters
        ----------
        ion : int
            Ion index.
        shift_mhz : float
            Stark shift in MHz.
        duration_us : float
            Duration in microseconds.
        """
        H = (shift_mhz * 2.0 * math.pi / 2.0) * _embed_operator(
            _SIGMA_Z, ion, self.n_ions
        )
        self.add_evolution(H, duration_us, label=f"Stark(ion={ion})")

    def simulate(
        self, initial_state: np.ndarray | None = None, dt_us: float = 0.01
    ) -> np.ndarray:
        """Simulate the full analog circuit.

        For each segment with time-independent Hamiltonian, computes the
        exact propagator U = expm(-i * H * t).  For better accuracy with
        time-dependent Hamiltonians, set ``dt_us`` to subdivide each step
        (Trotter approximation).

        Parameters
        ----------
        initial_state : np.ndarray, optional
            Initial state vector of length 2^n.  Defaults to |00...0>.
        dt_us : float
            Time step for Trotterised evolution (microseconds).

        Returns
        -------
        np.ndarray
            Final state vector.
        """
        if initial_state is None:
            state = np.zeros(self.dim, dtype=np.complex128)
            state[0] = 1.0
        else:
            state = initial_state.astype(np.complex128).copy()

        for step in self.steps:
            n_steps = max(1, int(math.ceil(step.duration_us / dt_us)))
            actual_dt = step.duration_us / n_steps

            # Compute single-step propagator
            U_step = _matrix_exp_hermitian(
                -1j * step.hamiltonian * actual_dt
            )

            for _ in range(n_steps):
                state = U_step @ state

        return state

    def total_duration_us(self) -> float:
        """Total circuit duration in microseconds."""
        return sum(step.duration_us for step in self.steps)

    def __repr__(self) -> str:
        lines = [f"AnalogCircuit(n_ions={self.n_ions}, steps={len(self.steps)})"]
        for i, step in enumerate(self.steps):
            lines.append(
                f"  [{i}] {step.label or 'evolution'}: "
                f"{step.duration_us:.3f} us"
            )
        return "\n".join(lines)


# ======================================================================
# Pulse Sequence (Atomic Layer)
# ======================================================================

@dataclass
class LaserPulse:
    """A single laser pulse on one ion.

    Parameters
    ----------
    ion : int
        Target ion index.
    frequency_mhz : float
        Laser frequency relative to qubit resonance, in MHz.
    amplitude : float
        Normalised pulse amplitude (0 to 1).
    phase : float
        Optical phase in radians.
    duration_us : float
        Pulse duration in microseconds.
    shape : str
        Pulse envelope shape: 'square', 'gaussian', 'sech',
        'blackman', or 'cosine'.
    """
    ion: int
    frequency_mhz: float
    amplitude: float
    phase: float
    duration_us: float
    shape: str = "gaussian"


class PulseSequence:
    """Pulse-level description of laser operations on ions.

    Inspired by OQD's atomic interface (openAPL IR).

    Represents the lowest-level description of quantum operations:
    individual laser pulses with specified frequency, amplitude, phase,
    duration, and envelope shape.

    Parameters
    ----------
    n_ions : int
        Number of ions in the chain.
    """

    def __init__(self, n_ions: int) -> None:
        self.n_ions = n_ions
        self.pulses: list[LaserPulse] = []

    def add_pulse(
        self,
        ion: int,
        frequency_mhz: float,
        amplitude: float,
        phase: float,
        duration_us: float,
        shape: str = "gaussian",
    ) -> None:
        """Add a laser pulse to the sequence.

        Parameters
        ----------
        ion : int
            Target ion index.
        frequency_mhz : float
            Laser frequency offset from qubit resonance in MHz.
        amplitude : float
            Normalised amplitude (0 to 1).
        phase : float
            Optical phase in radians.
        duration_us : float
            Pulse duration in microseconds.
        shape : str
            Envelope shape.
        """
        if ion < 0 or ion >= self.n_ions:
            raise ValueError(f"Ion index {ion} out of range [0, {self.n_ions})")
        if amplitude < 0 or amplitude > 1:
            raise ValueError("Amplitude must be in [0, 1]")
        if shape not in ("square", "gaussian", "sech", "blackman", "cosine"):
            raise ValueError(f"Unknown pulse shape: {shape}")

        self.pulses.append(
            LaserPulse(ion, frequency_mhz, amplitude, phase, duration_us, shape)
        )

    def envelope_function(self, shape: str, t: np.ndarray, duration: float) -> np.ndarray:
        """Compute the pulse envelope at times t.

        Parameters
        ----------
        shape : str
            Envelope type.
        t : np.ndarray
            Time points (0 to duration).
        duration : float
            Total pulse duration.

        Returns
        -------
        np.ndarray
            Envelope values in [0, 1].
        """
        t_norm = t / duration  # normalise to [0, 1]

        if shape == "square":
            return np.ones_like(t_norm)
        elif shape == "gaussian":
            # Gaussian truncated at 3 sigma
            sigma = 1.0 / 6.0  # so 3*sigma = 0.5 -> full width
            return np.exp(-((t_norm - 0.5) ** 2) / (2.0 * sigma ** 2))
        elif shape == "sech":
            # Hyperbolic secant pulse
            return 1.0 / np.cosh(6.0 * (t_norm - 0.5))
        elif shape == "blackman":
            a0, a1, a2 = 0.42, 0.50, 0.08
            return (
                a0
                - a1 * np.cos(2.0 * math.pi * t_norm)
                + a2 * np.cos(4.0 * math.pi * t_norm)
            )
        elif shape == "cosine":
            return np.sin(math.pi * t_norm) ** 2
        else:
            return np.ones_like(t_norm)

    def to_analog_circuit(self, config: "TrapConfig") -> AnalogCircuit:
        """Convert pulse sequence to an AnalogCircuit.

        Each pulse is converted into a Hamiltonian evolution step.
        Resonant pulses (frequency_mhz ~ 0) become Rabi drives.
        Off-resonant pulses become AC Stark shifts.

        Parameters
        ----------
        config : TrapConfig
            Trap configuration.

        Returns
        -------
        AnalogCircuit
            Equivalent analog-level circuit.
        """
        circuit = AnalogCircuit(self.n_ions, config)

        for pulse in self.pulses:
            # Peak Rabi frequency from normalised amplitude
            # Assume max Rabi freq ~ 1 MHz (tunable)
            max_rabi_mhz = 1.0
            rabi = pulse.amplitude * max_rabi_mhz

            if abs(pulse.frequency_mhz) < 0.01:
                # Near-resonant: Rabi drive
                circuit.add_rabi_drive(
                    ion=pulse.ion,
                    rabi_freq_mhz=rabi,
                    phase=pulse.phase,
                    duration_us=pulse.duration_us,
                )
            elif abs(pulse.frequency_mhz) < 10.0:
                # Moderate detuning: could be sideband transition
                # Approximate as Rabi drive with detuned corrections
                circuit.add_rabi_drive(
                    ion=pulse.ion,
                    rabi_freq_mhz=rabi,
                    phase=pulse.phase,
                    duration_us=pulse.duration_us,
                )
            else:
                # Far off-resonant: AC Stark shift
                # Stark shift ~ Omega^2 / (4 * Delta)
                stark = rabi ** 2 / (4.0 * pulse.frequency_mhz)
                circuit.add_stark_shift(
                    ion=pulse.ion,
                    shift_mhz=stark,
                    duration_us=pulse.duration_us,
                )

        return circuit

    def total_duration_us(self) -> float:
        """Total sequence duration (sequential execution)."""
        return sum(p.duration_us for p in self.pulses)

    def __repr__(self) -> str:
        lines = [f"PulseSequence(n_ions={self.n_ions}, pulses={len(self.pulses)})"]
        for i, p in enumerate(self.pulses):
            lines.append(
                f"  [{i}] ion={p.ion} f={p.frequency_mhz:.2f}MHz "
                f"A={p.amplitude:.3f} phi={p.phase:.3f} "
                f"t={p.duration_us:.3f}us shape={p.shape}"
            )
        return "\n".join(lines)


# ======================================================================
# Utility: matrix exponential for Hermitian arguments
# ======================================================================

def _matrix_exp_hermitian(A: np.ndarray) -> np.ndarray:
    """Compute matrix exponential exp(A) where A = -i*H*t.

    Uses eigendecomposition for Hermitian H (so A is anti-Hermitian up
    to a scalar), which is numerically stable and exact.

    Parameters
    ----------
    A : np.ndarray
        Matrix to exponentiate.

    Returns
    -------
    np.ndarray
        exp(A).
    """
    # A = -i*H*t, so H = i*A/t is Hermitian
    # Use scipy-free approach via eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(
        1j * A  # This should be Hermitian if A is anti-Hermitian
    )
    # exp(A) = V * diag(exp(eigenvalues of A)) * V^dag
    # eigenvalues of A = -i * eigenvalues of (i*A) = -i * eigenvalues
    exp_eigenvalues = np.exp(-1j * eigenvalues)
    return (eigenvectors * exp_eigenvalues) @ eigenvectors.conj().T
