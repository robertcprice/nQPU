"""Physics-based noise model for trapped-ion quantum computers.

Models the dominant error sources in trapped-ion systems following:

- Ballance et al., Phys. Rev. Lett. 117, 060504 (2016) [gate errors]
- Brownnutt et al., Rev. Mod. Phys. 87, 1419 (2015) [motional heating]
- Ozeri et al., Phys. Rev. A 75, 042329 (2007) [spontaneous emission]

Noise channels
--------------
1. Motional heating: phonon occupation increases during gates, degrading
   MS gate fidelity.
2. Spontaneous emission: off-resonant photon scattering during Raman
   gates causes decoherence.
3. Magnetic field noise: ambient B-field fluctuations shift qubit
   frequencies, causing dephasing.
4. Laser intensity noise: Rabi frequency fluctuations produce
   over/under-rotation errors.
5. Crosstalk: AC Stark shifts from addressing beams on neighbouring ions.
6. Motional dephasing: trap frequency fluctuations decohere motional
   modes during MS gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .trap import TrapConfig


@dataclass
class TrappedIonNoiseModel:
    """Physics-accurate noise model for trapped-ion systems.

    All error rates are computed from first principles using the trap
    parameters and known physical noise processes.

    Parameters
    ----------
    config : TrapConfig
        Trap and ion configuration.
    magnetic_field_noise_hz : float
        RMS magnetic field noise in Hz (qubit frequency fluctuation).
        Typical: 10-100 Hz for lab environments with mu-metal shielding.
    laser_intensity_noise_frac : float
        Fractional Rabi frequency fluctuation (delta_Omega / Omega).
        Typical: 1e-4 to 1e-2 depending on laser stabilisation.
    crosstalk_fraction : float
        Fraction of addressing beam intensity hitting neighbouring ions.
        Typical: 0.5% to 5% depending on beam waist and ion spacing.
    motional_dephasing_rate_hz : float
        Rate of motional mode frequency fluctuations in Hz.
    """

    config: "TrapConfig"
    magnetic_field_noise_hz: float = 50.0
    laser_intensity_noise_frac: float = 0.001
    crosstalk_fraction: float = 0.01
    motional_dephasing_rate_hz: float = 10.0

    # Typical gate times (can be overridden per-call)
    single_qubit_gate_time_us: float = 1.0
    two_qubit_gate_time_us: float = 100.0

    # ------------------------------------------------------------------
    # Gate fidelity estimators
    # ------------------------------------------------------------------

    def single_qubit_gate_fidelity(
        self, gate_time_us: float | None = None
    ) -> float:
        """Estimate single-qubit gate fidelity including all noise sources.

        Error contributions
        -------------------
        1. Spontaneous emission: p_se = scattering_rate * gate_time
        2. Intensity noise: p_int = (delta_Omega/Omega)^2 * pi^2 / 4
        3. Dephasing: p_deph = (2*pi * B_noise * gate_time)^2 / 2
        4. Crosstalk: p_xt = crosstalk_fraction^2 (leakage to neighbours)

        Returns
        -------
        float
            Estimated gate fidelity in [0, 1].
        """
        t_gate = (gate_time_us or self.single_qubit_gate_time_us) * 1e-6

        species = self.config.species

        # Spontaneous emission
        p_se = species.scattering_rate_per_gate

        # Laser intensity noise -> rotation error
        p_int = (self.laser_intensity_noise_frac * math.pi) ** 2 / 4.0

        # Magnetic dephasing during gate
        p_deph = (
            2.0 * math.pi * self.magnetic_field_noise_hz * t_gate
        ) ** 2 / 2.0

        # Crosstalk
        p_xt = self.crosstalk_fraction ** 2

        total_error = p_se + p_int + p_deph + p_xt
        return max(0.0, 1.0 - total_error)

    def two_qubit_gate_fidelity(
        self,
        gate_time_us: float | None = None,
        motional_mode_idx: int = 0,
    ) -> float:
        """Estimate Molmer-Sorensen gate fidelity.

        Error contributions
        -------------------
        1. Motional heating: p_heat = eta^2 * n_dot * gate_time
        2. Spontaneous emission: 2 * scattering_rate * gate_time
        3. Motional dephasing: (dephasing_rate * gate_time)^2
        4. Off-resonant coupling: ~ (eta * Omega / delta_mode)^2
        5. Intensity noise: 2 * (delta_Omega/Omega)^2

        Parameters
        ----------
        gate_time_us : float, optional
            Override gate duration in microseconds.
        motional_mode_idx : int
            Index of the motional mode used for the MS gate.

        Returns
        -------
        float
            Estimated gate fidelity in [0, 1].
        """
        t_gate = (gate_time_us or self.two_qubit_gate_time_us) * 1e-6

        species = self.config.species
        eta = self.config.lamb_dicke

        # Motional heating error
        n_dot = self.config.heating_rate_quanta_per_s
        p_heat = eta ** 2 * n_dot * t_gate

        # Spontaneous emission (both ions)
        p_se = 2.0 * species.scattering_rate_per_gate

        # Motional dephasing
        p_mdeph = (self.motional_dephasing_rate_hz * t_gate) ** 2

        # Off-resonant coupling to spectator modes
        # Rough estimate: ~ N_ions * (eta / delta_mode)^2
        # where delta_mode is mode splitting
        mode_freqs, _ = self.config.normal_modes()
        if len(mode_freqs) > 1 and motional_mode_idx < len(mode_freqs) - 1:
            delta_mode = abs(
                mode_freqs[motional_mode_idx + 1]
                - mode_freqs[motional_mode_idx]
            )
            delta_mode_hz = delta_mode * 1e6  # MHz -> Hz
            if delta_mode_hz > 0:
                rabi_estimate = 1.0 / (t_gate * 2)  # rough Rabi freq
                p_offres = self.config.n_ions * (
                    eta * rabi_estimate / delta_mode_hz
                ) ** 2
            else:
                p_offres = 0.0
        else:
            p_offres = 0.0

        # Laser intensity noise (both beams)
        p_int = 2.0 * self.laser_intensity_noise_frac ** 2

        total_error = p_heat + p_se + p_mdeph + p_offres + p_int
        return max(0.0, 1.0 - total_error)

    # ------------------------------------------------------------------
    # Noise channels on density matrices
    # ------------------------------------------------------------------

    def apply_noise(
        self,
        density_matrix: np.ndarray,
        gate_type: str,
        gate_time_us: float,
        target_qubits: tuple[int, ...] = (0,),
    ) -> np.ndarray:
        """Apply a noise channel to a density matrix after a gate.

        Implements depolarising + dephasing noise derived from the
        physics-based error model.

        Parameters
        ----------
        density_matrix : np.ndarray
            Density matrix of shape (2^n, 2^n).
        gate_type : str
            One of 'single', 'ms', 'idle'.
        gate_time_us : float
            Duration of the gate in microseconds.
        target_qubits : tuple[int, ...]
            Which qubits the gate acts on (for local noise).

        Returns
        -------
        np.ndarray
            Noisy density matrix.
        """
        rho = density_matrix.copy().astype(np.complex128)
        n_qubits = int(math.log2(rho.shape[0]))

        if gate_type == "single":
            fidelity = self.single_qubit_gate_fidelity(gate_time_us)
            for q in target_qubits:
                rho = self._apply_single_qubit_depolarising(
                    rho, q, n_qubits, 1.0 - fidelity
                )
        elif gate_type == "ms":
            fidelity = self.two_qubit_gate_fidelity(gate_time_us)
            error = 1.0 - fidelity
            # Split between depolarising on each qubit
            for q in target_qubits:
                rho = self._apply_single_qubit_depolarising(
                    rho, q, n_qubits, error / 2.0
                )
        elif gate_type == "idle":
            rho = self.idle_decoherence(rho, gate_time_us)

        return rho

    def idle_decoherence(
        self, density_matrix: np.ndarray, idle_time_us: float
    ) -> np.ndarray:
        """Apply T1/T2 decoherence during idle time.

        Implements amplitude damping (T1) and pure dephasing (T2) on
        each qubit independently.

        Parameters
        ----------
        density_matrix : np.ndarray
            Density matrix of shape (2^n, 2^n).
        idle_time_us : float
            Idle duration in microseconds.

        Returns
        -------
        np.ndarray
            Decohered density matrix.
        """
        rho = density_matrix.copy().astype(np.complex128)
        n_qubits = int(math.log2(rho.shape[0]))
        t = idle_time_us * 1e-6

        t1 = self.config.species.t1_s
        t2 = self.config.species.t2_s

        gamma1 = 1.0 - math.exp(-t / t1) if t1 > 0 else 0.0
        gamma_phi = (
            1.0 - math.exp(-t / t2) * math.exp(t / (2.0 * t1))
            if t2 > 0 and t1 > 0
            else 0.0
        )

        for q in range(n_qubits):
            if gamma1 > 0:
                rho = self._apply_amplitude_damping(rho, q, n_qubits, gamma1)
            if gamma_phi > 0:
                rho = self._apply_dephasing(rho, q, n_qubits, gamma_phi)

        return rho

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def error_budget(self) -> dict[str, float]:
        """Return a breakdown of error contributions.

        Returns
        -------
        dict
            Error budget with individual noise source contributions.
        """
        t1q = self.single_qubit_gate_time_us * 1e-6
        t2q = self.two_qubit_gate_time_us * 1e-6
        species = self.config.species
        eta = self.config.lamb_dicke

        budget = {
            "1q_spontaneous_emission": species.scattering_rate_per_gate,
            "1q_intensity_noise": (self.laser_intensity_noise_frac * math.pi) ** 2 / 4.0,
            "1q_magnetic_dephasing": (2.0 * math.pi * self.magnetic_field_noise_hz * t1q) ** 2 / 2.0,
            "1q_crosstalk": self.crosstalk_fraction ** 2,
            "2q_motional_heating": eta ** 2 * self.config.heating_rate_quanta_per_s * t2q,
            "2q_spontaneous_emission": 2.0 * species.scattering_rate_per_gate,
            "2q_motional_dephasing": (self.motional_dephasing_rate_hz * t2q) ** 2,
            "2q_intensity_noise": 2.0 * self.laser_intensity_noise_frac ** 2,
        }
        budget["1q_total"] = sum(v for k, v in budget.items() if k.startswith("1q_"))
        budget["2q_total"] = sum(v for k, v in budget.items() if k.startswith("2q_"))
        budget["1q_fidelity"] = 1.0 - budget["1q_total"]
        budget["2q_fidelity"] = 1.0 - budget["2q_total"]
        return budget

    # ------------------------------------------------------------------
    # Internal: Kraus channel implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_single_qubit_depolarising(
        rho: np.ndarray, qubit: int, n_qubits: int, p: float
    ) -> np.ndarray:
        """Apply single-qubit depolarising channel with error probability p.

        E(rho) = (1 - p) * rho + (p/3) * (X rho X + Y rho Y + Z rho Z)
        """
        if p <= 0:
            return rho
        p = min(p, 1.0)

        dim = 2 ** n_qubits
        paulis = [
            np.array([[0, 1], [1, 0]], dtype=np.complex128),     # X
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128),  # Y
            np.array([[1, 0], [0, -1]], dtype=np.complex128),     # Z
        ]

        result = (1.0 - p) * rho
        for pauli in paulis:
            op = _embed_single_qubit_op(pauli, qubit, n_qubits)
            result += (p / 3.0) * (op @ rho @ op.conj().T)

        return result

    @staticmethod
    def _apply_amplitude_damping(
        rho: np.ndarray, qubit: int, n_qubits: int, gamma: float
    ) -> np.ndarray:
        """Apply amplitude damping channel (T1 decay).

        Kraus operators:
            K0 = [[1, 0], [0, sqrt(1-gamma)]]
            K1 = [[0, sqrt(gamma)], [0, 0]]
        """
        if gamma <= 0:
            return rho
        gamma = min(gamma, 1.0)

        K0 = np.array(
            [[1, 0], [0, math.sqrt(1 - gamma)]], dtype=np.complex128
        )
        K1 = np.array(
            [[0, math.sqrt(gamma)], [0, 0]], dtype=np.complex128
        )

        K0_full = _embed_single_qubit_op(K0, qubit, n_qubits)
        K1_full = _embed_single_qubit_op(K1, qubit, n_qubits)

        return K0_full @ rho @ K0_full.conj().T + K1_full @ rho @ K1_full.conj().T

    @staticmethod
    def _apply_dephasing(
        rho: np.ndarray, qubit: int, n_qubits: int, gamma: float
    ) -> np.ndarray:
        """Apply pure dephasing channel.

        E(rho) = (1 - gamma) * rho + gamma * Z rho Z
        """
        if gamma <= 0:
            return rho
        gamma = min(gamma, 1.0)

        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        Z_full = _embed_single_qubit_op(Z, qubit, n_qubits)

        return (1.0 - gamma) * rho + gamma * (Z_full @ rho @ Z_full.conj().T)


# ======================================================================
# Utility functions
# ======================================================================

def _embed_single_qubit_op(
    op: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    """Embed a 2x2 operator on a specific qubit into the full Hilbert space.

    Uses the convention that qubit 0 is the most significant bit:
        |q0 q1 ... q_{n-1}>

    Parameters
    ----------
    op : np.ndarray
        2x2 operator.
    qubit : int
        Target qubit index.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    np.ndarray
        (2^n x 2^n) operator.
    """
    I2 = np.eye(2, dtype=np.complex128)
    result = np.array([[1.0]], dtype=np.complex128)
    for q in range(n_qubits):
        if q == qubit:
            result = np.kron(result, op)
        else:
            result = np.kron(result, I2)
    return result
