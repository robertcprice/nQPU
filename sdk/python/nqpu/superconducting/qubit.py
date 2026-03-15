"""Transmon qubit physical parameters.

Models the weakly anharmonic oscillator with energy levels:
    E_n = omega_01 * n - (alpha / 2) * n * (n - 1)

References:
    - Koch et al., PRA 76, 042319 (2007)
    - Krantz et al., Appl. Phys. Rev. 6, 021318 (2019)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransmonQubit:
    """Physical parameters for a single transmon qubit.

    Parameters
    ----------
    frequency_ghz : float
        Qubit 0->1 transition frequency in GHz.
    anharmonicity_mhz : float
        Anharmonicity alpha in MHz (negative for transmon).
    t1_us : float
        Energy relaxation time T1 in microseconds.
    t2_us : float
        Dephasing time T2 in microseconds (T2 <= 2*T1).
    single_gate_fidelity : float
        Single-qubit gate fidelity from randomised benchmarking.
    gate_time_ns : float
        Duration of a single-qubit gate in nanoseconds.
    readout_fidelity : float
        Measurement assignment fidelity.
    readout_time_ns : float
        Readout integration time in nanoseconds.
    """

    frequency_ghz: float = 5.0
    anharmonicity_mhz: float = -330.0
    t1_us: float = 100.0
    t2_us: float = 120.0
    single_gate_fidelity: float = 0.9995
    gate_time_ns: float = 25.0
    readout_fidelity: float = 0.99
    readout_time_ns: float = 800.0

    @property
    def frequency_02_ghz(self) -> float:
        """0->2 transition frequency in GHz."""
        return 2.0 * self.frequency_ghz + self.anharmonicity_mhz / 1000.0

    @property
    def pure_dephasing_rate_mhz(self) -> float:
        """Pure dephasing rate 1/T_phi in MHz."""
        t2_inv = 1.0 / self.t2_us if self.t2_us > 0 else 0.0
        t1_inv = 1.0 / (2.0 * self.t1_us) if self.t1_us > 0 else 0.0
        t_phi_inv = max(t2_inv - t1_inv, 0.0)
        return t_phi_inv

    @classmethod
    def typical(cls, index: int = 0) -> TransmonQubit:
        """Create a qubit with typical parameters, staggered in frequency."""
        base_freq = 5.0 + 0.05 * (index % 4)
        return cls(frequency_ghz=base_freq)

    @classmethod
    def ibm_eagle_qubit(cls) -> TransmonQubit:
        """Typical IBM Eagle qubit parameters."""
        return cls(
            frequency_ghz=5.1,
            anharmonicity_mhz=-340.0,
            t1_us=100.0,
            t2_us=120.0,
            single_gate_fidelity=0.9995,
            gate_time_ns=25.0,
            readout_fidelity=0.985,
        )

    @classmethod
    def ibm_heron_qubit(cls) -> TransmonQubit:
        """Typical IBM Heron qubit parameters (improved coherence)."""
        return cls(
            frequency_ghz=4.9,
            anharmonicity_mhz=-320.0,
            t1_us=200.0,
            t2_us=250.0,
            single_gate_fidelity=0.9998,
            gate_time_ns=20.0,
            readout_fidelity=0.995,
        )

    @classmethod
    def google_sycamore_qubit(cls) -> TransmonQubit:
        """Typical Google Sycamore qubit parameters."""
        return cls(
            frequency_ghz=6.0,
            anharmonicity_mhz=-200.0,
            t1_us=15.0,
            t2_us=20.0,
            single_gate_fidelity=0.9985,
            gate_time_ns=25.0,
            readout_fidelity=0.965,
        )
