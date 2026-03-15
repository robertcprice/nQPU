"""Physics-based noise model for superconducting transmon processors.

Models the dominant error sources in transmon systems:

1. T1 relaxation: amplitude damping from dielectric loss, Purcell effect
2. T2 dephasing: phase noise from 1/f charge noise, TLS defects
3. ZZ crosstalk: always-on conditional phase from capacitive coupling
4. Leakage: population transfer to |2> during fast gates
5. Readout errors: assignment infidelity and measurement crosstalk
6. TLS defects: two-level systems causing frequency jitter
7. Calibration drift: gate parameter drift over hours

References:
    - Krantz et al., Appl. Phys. Rev. 6, 021318 (2019)
    - Muller et al., Rep. Prog. Phys. 82, 124501 (2019) [TLS]
    - Sheldon et al., PRA 93, 060302 (2016) [CR gate errors]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .chip import ChipConfig


@dataclass
class TransmonNoiseModel:
    """Physics-accurate noise model for superconducting transmon systems.

    Parameters
    ----------
    config : ChipConfig
        Processor configuration with per-qubit parameters.
    enable_t1 : bool
        Enable T1 amplitude damping.
    enable_t2 : bool
        Enable T2 dephasing.
    enable_zz_crosstalk : bool
        Enable always-on ZZ interaction.
    enable_leakage : bool
        Enable leakage to |2>.
    enable_readout_error : bool
        Enable measurement assignment errors.
    """

    config: "ChipConfig"
    enable_t1: bool = True
    enable_t2: bool = True
    enable_zz_crosstalk: bool = True
    enable_leakage: bool = True
    enable_readout_error: bool = True

    @classmethod
    def ideal(cls, config: "ChipConfig") -> TransmonNoiseModel:
        """Create noise model with all sources disabled."""
        return cls(
            config=config,
            enable_t1=False,
            enable_t2=False,
            enable_zz_crosstalk=False,
            enable_leakage=False,
            enable_readout_error=False,
        )

    # ------------------------------------------------------------------
    # Error rate calculations
    # ------------------------------------------------------------------

    def t1_decay_prob(self, qubit: int, gate_time_ns: float) -> float:
        """Probability of T1 decay during a gate."""
        if not self.enable_t1:
            return 0.0
        t1_ns = self.config.qubits[qubit].t1_us * 1000.0
        if t1_ns <= 0:
            return 0.0
        return 1.0 - math.exp(-gate_time_ns / t1_ns)

    def t2_dephase_prob(self, qubit: int, gate_time_ns: float) -> float:
        """Probability of T2 dephasing during a gate."""
        if not self.enable_t2:
            return 0.0
        t2_ns = self.config.qubits[qubit].t2_us * 1000.0
        if t2_ns <= 0:
            return 0.0
        return 1.0 - math.exp(-gate_time_ns / t2_ns)

    def leakage_prob(self, qubit: int) -> float:
        """Probability of leakage to |2> during a single-qubit gate."""
        if not self.enable_leakage:
            return 0.0
        q = self.config.qubits[qubit]
        anharmonicity_ghz = abs(q.anharmonicity_mhz) / 1000.0
        if anharmonicity_ghz <= 0:
            return 0.0
        gate_time_ns = q.gate_time_ns
        # Leakage ~ (Omega / alpha)^2 where Omega ~ pi / t_gate
        rabi_ghz = 0.5 / gate_time_ns  # in GHz
        return (rabi_ghz / anharmonicity_ghz) ** 2

    def zz_coupling_hz(self, qubit_a: int, qubit_b: int) -> float:
        """ZZ coupling strength between two qubits in Hz."""
        if not self.enable_zz_crosstalk:
            return 0.0
        g_mhz = self.config.topology.coupling_strength(qubit_a, qubit_b)
        if g_mhz == 0.0:
            return 0.0
        qa = self.config.qubits[qubit_a]
        qb = self.config.qubits[qubit_b]
        delta_ghz = qa.frequency_ghz - qb.frequency_ghz
        alpha_ghz = abs(qa.anharmonicity_mhz) / 1000.0
        g_ghz = g_mhz / 1000.0
        if abs(delta_ghz) < 1e-6 or abs(delta_ghz - alpha_ghz) < 1e-6:
            return 0.0
        zz_ghz = g_ghz ** 2 * alpha_ghz / (delta_ghz * (delta_ghz - alpha_ghz))
        return abs(zz_ghz) * 1e9

    def single_gate_error(self, qubit: int) -> float:
        """Total error rate for a single-qubit gate."""
        if not any([self.enable_t1, self.enable_t2, self.enable_leakage,
                     self.enable_readout_error, self.enable_zz_crosstalk]):
            return 0.0
        q = self.config.qubits[qubit]
        gate_err = 1.0 - q.single_gate_fidelity
        t1_err = self.t1_decay_prob(qubit, q.gate_time_ns)
        t2_err = self.t2_dephase_prob(qubit, q.gate_time_ns)
        leak_err = self.leakage_prob(qubit)
        return 1.0 - (1.0 - gate_err) * (1.0 - t1_err / 3) * (1.0 - t2_err / 2) * (1.0 - leak_err)

    def two_qubit_gate_error(self, qubit_a: int, qubit_b: int) -> float:
        """Total error rate for a two-qubit gate."""
        if not any([self.enable_t1, self.enable_t2, self.enable_leakage,
                     self.enable_readout_error, self.enable_zz_crosstalk]):
            return 0.0
        gate_err = 1.0 - self.config.two_qubit_fidelity
        t_gate = self.config.two_qubit_gate_time_ns
        t1_a = self.t1_decay_prob(qubit_a, t_gate)
        t1_b = self.t1_decay_prob(qubit_b, t_gate)
        t2_a = self.t2_dephase_prob(qubit_a, t_gate)
        t2_b = self.t2_dephase_prob(qubit_b, t_gate)
        return 1.0 - (1.0 - gate_err) * (1.0 - (t1_a + t1_b) / 3) * (1.0 - (t2_a + t2_b) / 2)

    def readout_confusion(self, qubit: int) -> np.ndarray:
        """2x2 readout confusion matrix for a qubit."""
        if not self.enable_readout_error:
            return np.eye(2)
        f = self.config.qubits[qubit].readout_fidelity
        # Asymmetric: p(1|0) is typically higher than p(0|1)
        p01 = (1.0 - f) * 1.2  # excited-state decay during readout
        p10 = (1.0 - f) * 0.8
        return np.array([[1.0 - p01, p01], [p10, 1.0 - p10]])

    # ------------------------------------------------------------------
    # Noise channel application (Kraus operators)
    # ------------------------------------------------------------------

    def amplitude_damping_kraus(self, qubit: int, gate_time_ns: float) -> list[np.ndarray]:
        """T1 amplitude damping Kraus operators."""
        gamma = self.t1_decay_prob(qubit, gate_time_ns)
        k0 = np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=np.complex128)
        k1 = np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=np.complex128)
        return [k0, k1]

    def dephasing_kraus(self, qubit: int, gate_time_ns: float) -> list[np.ndarray]:
        """T2 pure dephasing Kraus operators."""
        p = self.t2_dephase_prob(qubit, gate_time_ns)
        # Pure dephasing after removing T1 contribution
        t1_p = self.t1_decay_prob(qubit, gate_time_ns)
        pure_p = max(p - t1_p / 2, 0.0)
        k0 = np.array([[math.sqrt(1 - pure_p / 2), 0],
                        [0, math.sqrt(1 - pure_p / 2)]], dtype=np.complex128)
        k1 = np.array([[math.sqrt(pure_p / 2), 0],
                        [0, -math.sqrt(pure_p / 2)]], dtype=np.complex128)
        return [k0, k1]

    def apply_noise_channel(
        self,
        rho: np.ndarray,
        kraus_ops: list[np.ndarray],
        qubit: int,
        n_qubits: int,
    ) -> np.ndarray:
        """Apply a single-qubit Kraus channel to a density matrix."""
        dim = 2 ** n_qubits
        result = np.zeros((dim, dim), dtype=np.complex128)
        for k in kraus_ops:
            # Embed single-qubit Kraus into full space
            full_k = self._embed_operator(k, qubit, n_qubits)
            result += full_k @ rho @ full_k.conj().T
        return result

    def depolarizing_channel(
        self,
        rho: np.ndarray,
        qubit: int,
        n_qubits: int,
        p: float,
    ) -> np.ndarray:
        """Apply single-qubit depolarising channel."""
        if p <= 0:
            return rho
        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        result = (1 - p) * rho
        for pauli in [X, Y, Z]:
            full_p = self._embed_operator(pauli, qubit, n_qubits)
            result += (p / 3) * full_p @ rho @ full_p.conj().T
        return result

    @staticmethod
    def _embed_operator(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
        """Embed a single-qubit operator into the full Hilbert space.

        Uses bit-ordering convention where qubit 0 = LSB (rightmost
        factor in the tensor product).
        """
        result = np.array([1.0], dtype=np.complex128)
        I2 = np.eye(2, dtype=np.complex128)
        # Build from MSB (qubit n-1) to LSB (qubit 0)
        for i in range(n_qubits - 1, -1, -1):
            if i == qubit:
                result = np.kron(result, op)
            else:
                result = np.kron(result, I2)
        return result
