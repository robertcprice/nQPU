"""nQPU QPU Emulator -- run quantum circuits on emulated hardware.

Provides a unified interface to emulate quantum computation on realistic
models of real quantum hardware. Pick a hardware profile, submit circuits,
and get measurement results with hardware-accurate noise.

Supported hardware families:
  - Trapped-ion: IonQ Aria, IonQ Forte, Quantinuum H2
  - Superconducting: IBM Eagle, IBM Heron, Google Sycamore, Rigetti Ankaa-2
  - Neutral-atom: QuEra Aquila, Atom Computing Phoenix

Example:
    from nqpu.emulator import QPU, HardwareProfile
    from nqpu.transpiler import QuantumCircuit

    qpu = QPU(HardwareProfile.IONQ_ARIA)
    circ = QuantumCircuit(3)
    circ.h(0).cx(0, 1).cx(1, 2)

    result = qpu.run(circ, shots=1000)
    print(result.counts)
    print(result.fidelity_estimate)

    # Compare backends
    for profile in [HardwareProfile.IBM_HERON, HardwareProfile.IONQ_ARIA,
                    HardwareProfile.QUERA_AQUILA]:
        r = QPU(profile).run(circ, shots=1000)
        print(f"{profile.name}: fidelity={r.fidelity_estimate:.4f}")
"""

from .hardware import HardwareProfile, HardwareSpec, HardwareFamily
from .qpu import QPU
from .job import Job, EmulatorResult, Counts
from .advisor import HardwareAdvisor, Recommendation, CircuitProfile, HardwareScore

__all__ = [
    "QPU",
    "HardwareProfile",
    "HardwareSpec",
    "HardwareFamily",
    "Job",
    "EmulatorResult",
    "Counts",
    "HardwareAdvisor",
    "Recommendation",
    "CircuitProfile",
    "HardwareScore",
]
