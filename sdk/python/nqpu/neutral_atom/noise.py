"""Physics-based noise model for neutral-atom quantum computers.

Models the dominant error sources in Rydberg-atom systems following:

- Levine et al., Phys. Rev. Lett. 123, 170503 (2019) [CZ gate errors]
- Evered et al., Nature 622, 268 (2023) [high-fidelity entanglement]
- de Leseleuc et al., Phys. Rev. Lett. 120, 113602 (2018) [Rydberg decay]
- Madjarov et al., Nature Physics 16, 857 (2020) [Sr coherence]

Noise channels
--------------
1. Atom loss: Atoms can be lost from tweezers during gates (vacuum
   collisions, off-resonant scattering, anti-trapping in Rydberg state).
2. Laser intensity noise: Rabi frequency fluctuations produce over/under-
   rotation errors on single-qubit gates.
3. Thermal motion dephasing: Doppler shifts from residual atomic motion
   cause dephasing during Rydberg pulses.
4. Rydberg decay: Finite Rydberg state lifetime causes spontaneous
   emission during entangling gates.
5. Blockade leakage: Imperfect blockade (finite V/Omega ratio) allows
   double Rydberg excitation, reducing CZ fidelity.
6. Crosstalk: Rydberg interactions with non-target atoms outside the
   intended blockade pair.
7. Readout error: Imperfect state-dependent fluorescence detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .physics import AtomSpecies


@dataclass
class NeutralAtomNoiseModel:
    """Physics-accurate noise model for neutral-atom Rydberg systems.

    All error rates are computed from first principles using atom
    parameters and known physical noise processes.

    Parameters
    ----------
    species : AtomSpecies
        Atom species configuration.
    rabi_freq_mhz : float
        Rydberg Rabi frequency in MHz.
    atom_loss_prob : float
        Probability of atom loss per gate operation.
        Typical: 0.1% - 1% per circuit layer.
    laser_intensity_noise_frac : float
        Fractional Rabi frequency fluctuation (delta_Omega / Omega).
        Typical: 1e-3 to 1e-2 depending on laser stabilisation.
    thermal_dephasing_factor : float
        Additional thermal dephasing scaling factor (dimensionless).
    crosstalk_fraction : float
        Fraction of Rydberg interaction leaking to non-target neighbors.
    readout_error : float
        Probability of readout misassignment per qubit.
    atom_spacing_um : float
        Typical inter-atom spacing for blockade error estimation.
    """

    species: "AtomSpecies"
    rabi_freq_mhz: float = 1.5
    atom_loss_prob: float = 0.003
    laser_intensity_noise_frac: float = 0.005
    thermal_dephasing_factor: float = 1.0
    crosstalk_fraction: float = 0.01
    readout_error: float = 0.03
    atom_spacing_um: float = 4.0

    # Typical gate times (derived from Rabi frequency)
    @property
    def single_qubit_gate_time_us(self) -> float:
        """Single-qubit gate time: pi-pulse at Rabi frequency."""
        if self.rabi_freq_mhz <= 0:
            return 1.0
        return 1.0 / (2.0 * self.rabi_freq_mhz)  # pi / (2*pi*Omega)

    @property
    def two_qubit_gate_time_us(self) -> float:
        """CZ gate time: 3 Rydberg pi-pulses (pi + 2*pi + pi = 4*pi total).

        The three-pulse CZ sequence requires ~4 pi-pulse durations.
        """
        if self.rabi_freq_mhz <= 0:
            return 4.0
        return 4.0 / (2.0 * self.rabi_freq_mhz)

    @property
    def three_qubit_gate_time_us(self) -> float:
        """CCZ gate time: approximately 5 Rydberg pi-pulses."""
        if self.rabi_freq_mhz <= 0:
            return 5.0
        return 5.0 / (2.0 * self.rabi_freq_mhz)

    # ------------------------------------------------------------------
    # Gate fidelity estimators
    # ------------------------------------------------------------------

    def single_qubit_gate_fidelity(
        self, gate_time_us: float | None = None
    ) -> float:
        """Estimate single-qubit gate fidelity including all noise sources.

        Error contributions
        -------------------
        1. Laser intensity noise: p_int = (delta_Omega/Omega)^2 * pi^2 / 4
        2. Thermal dephasing: p_thermal = thermal_rate * gate_time
        3. Atom loss: p_loss
        4. Scattering from trap light: p_scatter = scatter_rate * gate_time

        Returns
        -------
        float
            Estimated gate fidelity in [0, 1].
        """
        t_gate_us = gate_time_us if gate_time_us is not None else self.single_qubit_gate_time_us
        t_gate_s = t_gate_us * 1e-6

        # Laser intensity noise -> rotation error
        p_int = (self.laser_intensity_noise_frac * math.pi) ** 2 / 4.0

        # Thermal dephasing
        thermal_rate = (
            self.species.thermal_dephasing_rate_hz(self.rabi_freq_mhz)
            * self.thermal_dephasing_factor
        )
        p_thermal = (thermal_rate * t_gate_s) ** 2 / 2.0

        # Atom loss
        p_loss = self.atom_loss_prob * 0.1  # Lower for single-qubit gates

        # Trap scattering
        p_scatter = self.species.scattering_rate_trap_hz * t_gate_s

        total_error = p_int + p_thermal + p_loss + p_scatter
        return max(0.0, 1.0 - total_error)

    def two_qubit_gate_fidelity(
        self, gate_time_us: float | None = None
    ) -> float:
        """Estimate CZ gate fidelity via Rydberg blockade.

        Error contributions
        -------------------
        1. Rydberg decay: p_decay = gate_time / rydberg_lifetime (both atoms)
        2. Blockade leakage: p_blockade = (Omega / V)^2
        3. Laser intensity noise: 2 * (delta_Omega/Omega)^2
        4. Thermal dephasing: (thermal_rate * gate_time)^2
        5. Atom loss: 2 * atom_loss_prob
        6. Crosstalk: crosstalk_fraction^2

        Returns
        -------
        float
            Estimated gate fidelity in [0, 1].
        """
        t_gate_us = gate_time_us if gate_time_us is not None else self.two_qubit_gate_time_us

        # Rydberg decay (both atoms spend time in Rydberg state)
        p_decay = 2.0 * t_gate_us / self.species.rydberg_lifetime_us

        # Blockade leakage
        omega_hz = self.rabi_freq_mhz * 1e6
        v_hz = self.species.vdw_interaction_hz(self.atom_spacing_um)
        p_blockade = (omega_hz / v_hz) ** 2 if v_hz > 0 else 1.0

        # Laser intensity noise (both beams)
        p_int = 2.0 * self.laser_intensity_noise_frac ** 2

        # Thermal dephasing
        t_gate_s = t_gate_us * 1e-6
        thermal_rate = (
            self.species.thermal_dephasing_rate_hz(self.rabi_freq_mhz)
            * self.thermal_dephasing_factor
        )
        p_thermal = (thermal_rate * t_gate_s) ** 2

        # Atom loss (both atoms)
        p_loss = 2.0 * self.atom_loss_prob

        # Crosstalk
        p_xt = self.crosstalk_fraction ** 2

        total_error = p_decay + p_blockade + p_int + p_thermal + p_loss + p_xt
        return max(0.0, 1.0 - total_error)

    def three_qubit_gate_fidelity(
        self, gate_time_us: float | None = None
    ) -> float:
        """Estimate CCZ gate fidelity.

        Similar to CZ but with three atoms contributing to errors.

        Returns
        -------
        float
            Estimated gate fidelity in [0, 1].
        """
        t_gate_us = gate_time_us if gate_time_us is not None else self.three_qubit_gate_time_us

        # Rydberg decay (three atoms)
        p_decay = 3.0 * t_gate_us / self.species.rydberg_lifetime_us

        # Blockade leakage (worst pair)
        omega_hz = self.rabi_freq_mhz * 1e6
        v_hz = self.species.vdw_interaction_hz(self.atom_spacing_um)
        p_blockade = 3.0 * (omega_hz / v_hz) ** 2 if v_hz > 0 else 1.0

        # Laser intensity noise (all three beams)
        p_int = 3.0 * self.laser_intensity_noise_frac ** 2

        # Atom loss (three atoms)
        p_loss = 3.0 * self.atom_loss_prob

        total_error = p_decay + p_blockade + p_int + p_loss
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

        Implements depolarising + dephasing + atom-loss noise derived
        from the physics-based error model.

        Parameters
        ----------
        density_matrix : np.ndarray
            Density matrix of shape (2^n, 2^n).
        gate_type : str
            One of 'single', 'cz', 'ccz', 'idle'.
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
        elif gate_type == "cz":
            fidelity = self.two_qubit_gate_fidelity(gate_time_us)
            error = 1.0 - fidelity
            for q in target_qubits:
                rho = self._apply_single_qubit_depolarising(
                    rho, q, n_qubits, error / len(target_qubits)
                )
        elif gate_type == "ccz":
            fidelity = self.three_qubit_gate_fidelity(gate_time_us)
            error = 1.0 - fidelity
            for q in target_qubits:
                rho = self._apply_single_qubit_depolarising(
                    rho, q, n_qubits, error / len(target_qubits)
                )
        elif gate_type == "idle":
            rho = self.idle_decoherence(rho, gate_time_us)

        # Apply atom loss channel
        if gate_type != "idle":
            for q in target_qubits:
                rho = self._apply_atom_loss(rho, q, n_qubits)

        return rho

    def idle_decoherence(
        self, density_matrix: np.ndarray, idle_time_us: float
    ) -> np.ndarray:
        """Apply T1/T2 decoherence during idle time.

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

        t1 = self.species.t1_s
        t2 = self.species.t2_s

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
        t1q_s = self.single_qubit_gate_time_us * 1e-6
        t2q_us = self.two_qubit_gate_time_us

        thermal_rate = (
            self.species.thermal_dephasing_rate_hz(self.rabi_freq_mhz)
            * self.thermal_dephasing_factor
        )

        omega_hz = self.rabi_freq_mhz * 1e6
        v_hz = self.species.vdw_interaction_hz(self.atom_spacing_um)

        budget: dict[str, float] = {
            "1q_intensity_noise": (self.laser_intensity_noise_frac * math.pi) ** 2 / 4.0,
            "1q_thermal_dephasing": (thermal_rate * t1q_s) ** 2 / 2.0,
            "1q_atom_loss": self.atom_loss_prob * 0.1,
            "1q_trap_scattering": self.species.scattering_rate_trap_hz * t1q_s,
            "2q_rydberg_decay": 2.0 * t2q_us / self.species.rydberg_lifetime_us,
            "2q_blockade_leakage": (omega_hz / v_hz) ** 2 if v_hz > 0 else 1.0,
            "2q_intensity_noise": 2.0 * self.laser_intensity_noise_frac ** 2,
            "2q_atom_loss": 2.0 * self.atom_loss_prob,
            "2q_crosstalk": self.crosstalk_fraction ** 2,
            "readout_error": self.readout_error,
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

        paulis = [
            np.array([[0, 1], [1, 0]], dtype=np.complex128),  # X
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128),  # Y
            np.array([[1, 0], [0, -1]], dtype=np.complex128),  # Z
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

    def _apply_atom_loss(
        self, rho: np.ndarray, qubit: int, n_qubits: int
    ) -> np.ndarray:
        """Apply atom loss channel.

        With probability atom_loss_prob, the qubit is projected to |0>
        (atom lost = no fluorescence = measured as 0).

        This is modelled as a partial trace + replacement:
            E(rho) = (1 - p) * rho + p * |0><0|_q x Tr_q(rho)
        Simplified here as amplitude damping with high gamma.
        """
        p = self.atom_loss_prob
        if p <= 0:
            return rho
        # Model as strong amplitude damping
        return self._apply_amplitude_damping(rho, qubit, n_qubits, p)


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
