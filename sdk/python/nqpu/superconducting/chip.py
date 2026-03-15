"""Chip topology and processor configuration for superconducting QPUs.

Models the physical layout and connectivity of transmon qubit arrays
including heavy-hex (IBM), grid (Google), and octagonal (Rigetti) topologies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .qubit import TransmonQubit


class TopologyType(enum.Enum):
    """Chip topology types."""
    HEAVY_HEX = "heavy_hex"
    GRID = "grid"
    FULLY_CONNECTED = "fully_connected"
    CUSTOM = "custom"


class NativeGateFamily(enum.Enum):
    """Native two-qubit gate families by vendor."""
    ECR = "ecr"              # IBM: Echoed Cross-Resonance
    SQRT_ISWAP = "sqrt_iswap"  # Google: sqrt(iSWAP)
    CZ = "cz"                # Rigetti: Controlled-Z


@dataclass
class ChipTopology:
    """Physical connectivity graph of a transmon chip.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits on chip.
    edges : list of (int, int)
        Connected qubit pairs.
    coupling_mhz : dict, optional
        Per-edge coupling strengths in MHz.
    topology_type : TopologyType
        Layout family.
    """

    num_qubits: int
    edges: list[tuple[int, int]] = field(default_factory=list)
    coupling_mhz: dict[tuple[int, int], float] = field(default_factory=dict)
    topology_type: TopologyType = TopologyType.CUSTOM

    def neighbors(self, qubit: int) -> list[int]:
        """Return the neighbors of a qubit."""
        result = []
        for a, b in self.edges:
            if a == qubit:
                result.append(b)
            elif b == qubit:
                result.append(a)
        return result

    def coupling_strength(self, a: int, b: int) -> float:
        """Get coupling strength between two qubits in MHz."""
        key = (min(a, b), max(a, b))
        return self.coupling_mhz.get(key, 0.0)

    @classmethod
    def heavy_hex(cls, num_qubits: int, coupling: float = 3.0) -> ChipTopology:
        """Build heavy-hex topology (IBM-style).

        Heavy-hex has degree-2 and degree-3 nodes in a hexagonal pattern.
        """
        edges = []
        couplings = {}
        # Simplified heavy-hex: chain with skip connections
        for i in range(num_qubits - 1):
            edges.append((i, i + 1))
            couplings[(i, i + 1)] = coupling
        # Add skip connections every 4 qubits
        for i in range(0, num_qubits - 3, 4):
            if i + 3 < num_qubits:
                edges.append((i, i + 3))
                couplings[(i, i + 3)] = coupling * 0.8
        return cls(
            num_qubits=num_qubits,
            edges=edges,
            coupling_mhz=couplings,
            topology_type=TopologyType.HEAVY_HEX,
        )

    @classmethod
    def grid(cls, rows: int, cols: int, coupling: float = 4.0) -> ChipTopology:
        """Build 2D grid topology (Google-style)."""
        n = rows * cols
        edges = []
        couplings = {}
        for r in range(rows):
            for c in range(cols):
                q = r * cols + c
                if c + 1 < cols:
                    neighbor = r * cols + c + 1
                    edges.append((q, neighbor))
                    couplings[(q, neighbor)] = coupling
                if r + 1 < rows:
                    neighbor = (r + 1) * cols + c
                    edges.append((q, neighbor))
                    couplings[(q, neighbor)] = coupling
        return cls(
            num_qubits=n,
            edges=edges,
            coupling_mhz=couplings,
            topology_type=TopologyType.GRID,
        )

    @classmethod
    def fully_connected(cls, n: int, coupling: float = 4.0) -> ChipTopology:
        """Build fully-connected topology (for small test circuits)."""
        edges = []
        couplings = {}
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                couplings[(i, j)] = coupling
        return cls(
            num_qubits=n,
            edges=edges,
            coupling_mhz=couplings,
            topology_type=TopologyType.FULLY_CONNECTED,
        )


@dataclass
class ChipConfig:
    """Full processor configuration.

    Parameters
    ----------
    topology : ChipTopology
        Physical connectivity.
    qubits : list of TransmonQubit
        Per-qubit parameters.
    native_2q_gate : NativeGateFamily
        Native two-qubit gate.
    two_qubit_fidelity : float
        Average two-qubit gate fidelity.
    two_qubit_gate_time_ns : float
        Duration of native two-qubit gate.
    temperature_mk : float
        Cryostat base temperature in millikelvin.
    """

    topology: ChipTopology
    qubits: list[TransmonQubit] = field(default_factory=list)
    native_2q_gate: NativeGateFamily = NativeGateFamily.ECR
    two_qubit_fidelity: float = 0.995
    two_qubit_gate_time_ns: float = 200.0
    temperature_mk: float = 15.0

    @property
    def num_qubits(self) -> int:
        return self.topology.num_qubits

    def device_info(self) -> dict[str, Any]:
        """Return device information summary."""
        return {
            "num_qubits": self.num_qubits,
            "topology": self.topology.topology_type.value,
            "native_2q_gate": self.native_2q_gate.value,
            "two_qubit_fidelity": self.two_qubit_fidelity,
            "num_couplers": len(self.topology.edges),
            "temperature_mk": self.temperature_mk,
        }


class DevicePresets(enum.Enum):
    """Pre-calibrated device configurations."""
    IBM_EAGLE = "ibm_eagle"
    IBM_HERON = "ibm_heron"
    GOOGLE_SYCAMORE = "google_sycamore"
    GOOGLE_WILLOW = "google_willow"
    RIGETTI_ANKAA = "rigetti_ankaa"

    def build(self, num_qubits: int | None = None) -> ChipConfig:
        """Construct the chip configuration for this preset."""
        builders = {
            DevicePresets.IBM_EAGLE: _build_ibm_eagle,
            DevicePresets.IBM_HERON: _build_ibm_heron,
            DevicePresets.GOOGLE_SYCAMORE: _build_google_sycamore,
            DevicePresets.GOOGLE_WILLOW: _build_google_willow,
            DevicePresets.RIGETTI_ANKAA: _build_rigetti_ankaa,
        }
        return builders[self](num_qubits)


def _build_ibm_eagle(n: int | None = None) -> ChipConfig:
    n = n or 127
    topo = ChipTopology.heavy_hex(n, coupling=3.0)
    qubits = [TransmonQubit.ibm_eagle_qubit() for _ in range(n)]
    return ChipConfig(
        topology=topo,
        qubits=qubits,
        native_2q_gate=NativeGateFamily.ECR,
        two_qubit_fidelity=0.99,
        two_qubit_gate_time_ns=300.0,
        temperature_mk=15.0,
    )


def _build_ibm_heron(n: int | None = None) -> ChipConfig:
    n = n or 156
    topo = ChipTopology.heavy_hex(n, coupling=3.5)
    qubits = [TransmonQubit.ibm_heron_qubit() for _ in range(n)]
    return ChipConfig(
        topology=topo,
        qubits=qubits,
        native_2q_gate=NativeGateFamily.ECR,
        two_qubit_fidelity=0.995,
        two_qubit_gate_time_ns=200.0,
        temperature_mk=12.0,
    )


def _build_google_sycamore(n: int | None = None) -> ChipConfig:
    n = n or 53
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    topo = ChipTopology.grid(rows, cols, coupling=4.0)
    qubits = [TransmonQubit.google_sycamore_qubit() for _ in range(topo.num_qubits)]
    return ChipConfig(
        topology=topo,
        qubits=qubits,
        native_2q_gate=NativeGateFamily.SQRT_ISWAP,
        two_qubit_fidelity=0.995,
        two_qubit_gate_time_ns=32.0,
        temperature_mk=20.0,
    )


def _build_google_willow(n: int | None = None) -> ChipConfig:
    n = n or 105
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    topo = ChipTopology.grid(rows, cols, coupling=4.5)
    qubits = [TransmonQubit.google_sycamore_qubit() for _ in range(topo.num_qubits)]
    return ChipConfig(
        topology=topo,
        qubits=qubits,
        native_2q_gate=NativeGateFamily.SQRT_ISWAP,
        two_qubit_fidelity=0.997,
        two_qubit_gate_time_ns=25.0,
        temperature_mk=15.0,
    )


def _build_rigetti_ankaa(n: int | None = None) -> ChipConfig:
    n = n or 84
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    topo = ChipTopology.grid(rows, cols, coupling=5.0)
    qubits = [TransmonQubit.typical(i) for i in range(topo.num_qubits)]
    return ChipConfig(
        topology=topo,
        qubits=qubits,
        native_2q_gate=NativeGateFamily.CZ,
        two_qubit_fidelity=0.992,
        two_qubit_gate_time_ns=80.0,
        temperature_mk=10.0,
    )
