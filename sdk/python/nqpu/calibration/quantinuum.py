"""Parse Quantinuum H-Series specifications into nQPU device configs.

Quantinuum's H-Series processors use a QCCD (Quantum Charge-Coupled
Device) trapped-ion architecture where ions can be shuttled between
specialised zones for computation, memory, and transport. This module
models the zone layout and published fidelity figures so that nQPU's
transpiler and noise-aware compiler can produce realistic circuit-level
fidelity estimates without requiring a live Quantinuum account.

Presets for H1-1 (20 qubits) and H2-1 (56 qubits) are included.

Example
-------
>>> from nqpu.calibration.quantinuum import h1_1, h2_1
>>> h1 = h1_1()
>>> print(h1.summary())
>>> print(f"Expected fidelity for 100 gates: "
...       f"{h1.expected_circuit_fidelity(50, 30, 20):.4f}")

References
----------
- Quantinuum H-Series technical specifications (2024)
- Pino et al. (2021), "Demonstration of the QCCD trapped-ion quantum computer"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrapZone:
    """Trapped-ion zone in the QCCD architecture.

    Attributes
    ----------
    zone_id : int
        Unique identifier for this zone.
    n_qubits : int
        Number of ion slots in this zone.
    zone_type : str
        One of ``"compute"``, ``"memory"``, or ``"transport"``.
    """

    zone_id: int
    n_qubits: int
    zone_type: str = "compute"


@dataclass
class TrapConfig:
    """Full trapped-ion processor calibration.

    Models a QCCD processor's published performance numbers. All fidelities
    are expressed as probabilities (e.g. 0.998 means 99.8 %). Times are in
    microseconds unless otherwise noted.
    """

    name: str
    n_qubits: int
    zones: List[TrapZone] = field(default_factory=list)
    single_qubit_fidelity: float = 0.9999
    two_qubit_fidelity: float = 0.998
    measurement_fidelity: float = 0.9997
    t1: float = 1e6      # microseconds (~1 second for trapped ions)
    t2: float = 1e5      # microseconds
    gate_time_1q: float = 10.0      # microseconds
    gate_time_2q: float = 200.0     # microseconds
    transport_time: float = 100.0   # inter-zone transport microseconds
    all_to_all: bool = True

    # -- Derived error properties -------------------------------------------

    @property
    def single_qubit_error(self) -> float:
        """Single-qubit gate error rate (1 - fidelity)."""
        return 1.0 - self.single_qubit_fidelity

    @property
    def two_qubit_error(self) -> float:
        """Two-qubit gate error rate (1 - fidelity)."""
        return 1.0 - self.two_qubit_fidelity

    @property
    def measurement_error(self) -> float:
        """Measurement error rate (1 - fidelity)."""
        return 1.0 - self.measurement_fidelity

    @property
    def total_zone_capacity(self) -> int:
        """Total ion slots across all zones."""
        return sum(z.n_qubits for z in self.zones)

    @property
    def compute_zones(self) -> List[TrapZone]:
        """Return only compute zones."""
        return [z for z in self.zones if z.zone_type == "compute"]

    # -- Fidelity estimation ------------------------------------------------

    def expected_circuit_fidelity(
        self,
        n_1q: int,
        n_2q: int,
        n_meas: int,
        n_transports: int = 0,
    ) -> float:
        """Estimate circuit fidelity from gate and measurement counts.

        Uses a simple independent-error model:

            F = F_1q^n_1q * F_2q^n_2q * F_meas^n_meas * F_transport^n_transports

        where ``F_transport`` is estimated from decoherence during the
        inter-zone shuttle time.

        Parameters
        ----------
        n_1q : int
            Number of single-qubit gates.
        n_2q : int
            Number of two-qubit gates.
        n_meas : int
            Number of measurements.
        n_transports : int, optional
            Number of inter-zone ion transports.

        Returns
        -------
        float
            Estimated circuit fidelity in [0, 1].
        """
        f_1q = self.single_qubit_fidelity ** n_1q
        f_2q = self.two_qubit_fidelity ** n_2q
        f_meas = self.measurement_fidelity ** n_meas

        # Transport decoherence: e^(-t_transport / T2) per transport
        if n_transports > 0 and self.t2 > 0:
            transport_fidelity = math.exp(
                -self.transport_time / self.t2
            ) ** n_transports
        else:
            transport_fidelity = 1.0

        return f_1q * f_2q * f_meas * transport_fidelity

    def circuit_time_us(
        self,
        n_1q: int,
        n_2q: int,
        n_meas: int = 0,
        n_transports: int = 0,
    ) -> float:
        """Estimate total circuit execution time in microseconds.

        Assumes sequential execution (worst case).
        """
        return (
            n_1q * self.gate_time_1q
            + n_2q * self.gate_time_2q
            + n_transports * self.transport_time
            + n_meas * 300.0  # typical measurement time for ions
        )

    def decoherence_limit(self) -> int:
        """Maximum number of two-qubit gates before T2 decoherence dominates.

        Returns the number of sequential two-qubit gates whose total
        duration equals T2 / 10 (keeping decoherence loss under ~10 %).
        """
        if self.t2 <= 0 or self.gate_time_2q <= 0:
            return 0
        return int(self.t2 / (10.0 * self.gate_time_2q))

    def summary(self) -> str:
        """Human-readable summary of the trapped-ion processor."""
        zone_info = (
            f"  Zones:             {len(self.zones)} "
            f"({sum(1 for z in self.zones if z.zone_type == 'compute')} compute, "
            f"{sum(1 for z in self.zones if z.zone_type == 'memory')} memory, "
            f"{sum(1 for z in self.zones if z.zone_type == 'transport')} transport)"
        )
        lines = [
            f"TrapConfig: {self.name}",
            f"  Qubits:            {self.n_qubits}",
            zone_info,
            f"  All-to-all:        {self.all_to_all}",
            f"  1Q fidelity:       {self.single_qubit_fidelity:.6f}",
            f"  2Q fidelity:       {self.two_qubit_fidelity:.5f}",
            f"  Measurement:       {self.measurement_fidelity:.6f}",
            f"  T1:                {self.t1:.0f} us",
            f"  T2:                {self.t2:.0f} us",
            f"  Gate time (1Q):    {self.gate_time_1q:.1f} us",
            f"  Gate time (2Q):    {self.gate_time_2q:.1f} us",
            f"  Transport time:    {self.transport_time:.1f} us",
            f"  Decoherence limit: ~{self.decoherence_limit()} 2Q gates",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_quantinuum_specs(specs: dict) -> TrapConfig:
    """Parse a Quantinuum device specification dictionary.

    Parameters
    ----------
    specs : dict
        Dictionary with keys such as ``"name"``, ``"n_qubits"``,
        ``"single_qubit_fidelity"``, ``"two_qubit_fidelity"``, etc.
        Zones can be specified under a ``"zones"`` key as a list of
        dicts with ``"zone_id"``, ``"n_qubits"``, ``"zone_type"``.

    Returns
    -------
    TrapConfig
    """
    zones = []
    for z in specs.get("zones", []):
        zones.append(
            TrapZone(
                zone_id=z.get("zone_id", 0),
                n_qubits=z.get("n_qubits", 0),
                zone_type=z.get("zone_type", "compute"),
            )
        )

    return TrapConfig(
        name=specs.get("name", "unknown"),
        n_qubits=specs.get("n_qubits", 0),
        zones=zones,
        single_qubit_fidelity=specs.get("single_qubit_fidelity", 0.9999),
        two_qubit_fidelity=specs.get("two_qubit_fidelity", 0.998),
        measurement_fidelity=specs.get("measurement_fidelity", 0.9997),
        t1=specs.get("t1", 1e6),
        t2=specs.get("t2", 1e5),
        gate_time_1q=specs.get("gate_time_1q", 10.0),
        gate_time_2q=specs.get("gate_time_2q", 200.0),
        transport_time=specs.get("transport_time", 100.0),
        all_to_all=specs.get("all_to_all", True),
    )


# ---------------------------------------------------------------------------
# Preset devices
# ---------------------------------------------------------------------------

def h1_1() -> TrapConfig:
    """Quantinuum H1-1 (20 qubits) preset.

    Based on published H1-1 specifications (2024) featuring a linear
    QCCD architecture with 5 zones.
    """
    return TrapConfig(
        name="quantinuum_h1_1",
        n_qubits=20,
        zones=[
            TrapZone(zone_id=0, n_qubits=5, zone_type="compute"),
            TrapZone(zone_id=1, n_qubits=5, zone_type="compute"),
            TrapZone(zone_id=2, n_qubits=4, zone_type="compute"),
            TrapZone(zone_id=3, n_qubits=3, zone_type="memory"),
            TrapZone(zone_id=4, n_qubits=3, zone_type="transport"),
        ],
        single_qubit_fidelity=0.99998,
        two_qubit_fidelity=0.998,
        measurement_fidelity=0.9997,
        t1=1e7,           # ~10 seconds
        t2=3e5,           # ~300 ms
        gate_time_1q=10.0,
        gate_time_2q=210.0,
        transport_time=120.0,
        all_to_all=True,
    )


def h2_1() -> TrapConfig:
    """Quantinuum H2-1 (56 qubits) preset.

    Based on published H2-1 specifications (2024) featuring a 2D grid
    QCCD architecture with 7 zones and improved gate fidelities.
    """
    return TrapConfig(
        name="quantinuum_h2_1",
        n_qubits=56,
        zones=[
            TrapZone(zone_id=0, n_qubits=8, zone_type="compute"),
            TrapZone(zone_id=1, n_qubits=8, zone_type="compute"),
            TrapZone(zone_id=2, n_qubits=8, zone_type="compute"),
            TrapZone(zone_id=3, n_qubits=8, zone_type="compute"),
            TrapZone(zone_id=4, n_qubits=8, zone_type="compute"),
            TrapZone(zone_id=5, n_qubits=8, zone_type="memory"),
            TrapZone(zone_id=6, n_qubits=8, zone_type="transport"),
        ],
        single_qubit_fidelity=0.999995,
        two_qubit_fidelity=0.9992,
        measurement_fidelity=0.9998,
        t1=1e7,
        t2=5e5,           # ~500 ms
        gate_time_1q=8.0,
        gate_time_2q=180.0,
        transport_time=80.0,
        all_to_all=True,
    )
