"""nQPU Neutral-Atom Backend -- Rydberg blockade quantum simulation.

Full-stack simulation of neutral-atom quantum computers using optical
tweezer arrays and Rydberg blockade entangling gates.

Three execution modes:
  - Ideal: Perfect state-vector simulation for algorithm development
  - Noisy: Density-matrix simulation with physics-based error channels
  - Pulse: Simplified Rydberg Hamiltonian evolution

Example:
    from nqpu.neutral_atom import NeutralAtomSimulator, ArrayConfig, AtomSpecies

    config = ArrayConfig(n_atoms=5, species=AtomSpecies.RB87)
    sim = NeutralAtomSimulator(config)
    sim.h(0)
    sim.cnot(0, 1)
    result = sim.measure_all()
"""

from .physics import AtomSpecies, ALL_SPECIES
from .array import (
    AtomArray,
    ArrayGeometry,
    Zone,
    ZoneConfig,
)
from .gates import (
    NeutralAtomGateSet,
    GateInstruction,
    NativeGateType,
)
from .noise import NeutralAtomNoiseModel
from .devices import DevicePresets, DeviceSpec
from .simulator import NeutralAtomSimulator, ArrayConfig, CircuitStats

__all__ = [
    # Physics
    "AtomSpecies",
    "ALL_SPECIES",
    # Array
    "AtomArray",
    "ArrayGeometry",
    "Zone",
    "ZoneConfig",
    # Gates
    "NeutralAtomGateSet",
    "GateInstruction",
    "NativeGateType",
    # Noise
    "NeutralAtomNoiseModel",
    # Devices
    "DevicePresets",
    "DeviceSpec",
    # Simulator
    "NeutralAtomSimulator",
    "ArrayConfig",
    "CircuitStats",
]
