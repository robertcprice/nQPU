"""nQPU Trapped-Ion Backend -- End-to-end simulation from gates to laser pulses.

Inspired by Open Quantum Design's full-stack quantum computing architecture.

Three abstraction layers:
  - Digital: Standard gate-based quantum circuits
  - Analog: Hamiltonian evolution and pulse sequences
  - Atomic: Laser-ion interactions and motional dynamics

Example:
    from nqpu.ion_trap import TrappedIonSimulator, IonSpecies, TrapConfig

    config = TrapConfig(n_ions=5, species=IonSpecies.YB171)
    sim = TrappedIonSimulator(config)
    sim.h(0)
    sim.cnot(0, 1)
    result = sim.measure_all()
"""

from .species import IonSpecies, ALL_SPECIES
from .trap import TrapConfig, DevicePresets
from .gates import (
    TrappedIonGateSet,
    GateInstruction,
    NativeGateType,
)
from .noise import TrappedIonNoiseModel
from .analog import AnalogCircuit, PulseSequence, LaserPulse
from .simulator import TrappedIonSimulator, CircuitStats

__all__ = [
    # Species
    "IonSpecies",
    "ALL_SPECIES",
    # Trap
    "TrapConfig",
    "DevicePresets",
    # Gates
    "TrappedIonGateSet",
    "GateInstruction",
    "NativeGateType",
    # Noise
    "TrappedIonNoiseModel",
    # Analog layer
    "AnalogCircuit",
    "PulseSequence",
    "LaserPulse",
    # Simulator
    "TrappedIonSimulator",
    "CircuitStats",
]
