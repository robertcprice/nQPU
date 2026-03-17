"""Hardware profiles for QPU emulation.

Each profile captures the key characteristics of a real quantum processor:
qubit count, connectivity, native gate set, gate fidelities, coherence times,
and gate speeds.  These parameters drive the noise model and transpilation.

Supported devices
-----------------
Trapped-ion:
    IonQ Aria (25Q), IonQ Forte (36Q), Quantinuum H2 (56Q)
Superconducting:
    IBM Eagle (127Q), IBM Heron (133Q), Google Sycamore (72Q), Rigetti Ankaa-2 (84Q)
Neutral-atom:
    QuEra Aquila (256Q), Atom Computing Phoenix (1225Q)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HardwareFamily(Enum):
    """Top-level hardware technology family."""

    TRAPPED_ION = "trapped_ion"
    SUPERCONDUCTING = "superconducting"
    NEUTRAL_ATOM = "neutral_atom"


@dataclass(frozen=True)
class HardwareSpec:
    """Specification of a quantum hardware device.

    All timing values are in microseconds.  Fidelities are expressed as
    success probabilities (e.g. 0.999 means 0.1 % error rate).

    Parameters
    ----------
    name : str
        Human-readable device name (e.g. ``"IonQ Aria"``).
    family : HardwareFamily
        Hardware technology family.
    num_qubits : int
        Total number of available qubits.
    connectivity : str
        Qubit connectivity topology.  One of ``"all_to_all"``, ``"grid"``,
        ``"heavy_hex"``, ``"linear"``, ``"reconfigurable"``.
    single_qubit_fidelity : float
        Average single-qubit gate fidelity.
    two_qubit_fidelity : float
        Average two-qubit gate fidelity.
    readout_fidelity : float
        Average measurement fidelity.
    native_1q_gates : tuple[str, ...]
        Names of the native single-qubit gates.
    native_2q_gate : str
        Name of the native two-qubit gate.
    native_3q_gate : str or None
        Name of the native three-qubit gate, if any.
    t1_us : float
        T1 relaxation time in microseconds.
    t2_us : float
        T2 dephasing time in microseconds.
    single_qubit_gate_us : float
        Duration of a single-qubit gate in microseconds.
    two_qubit_gate_us : float
        Duration of a two-qubit gate in microseconds.
    readout_us : float
        Duration of a measurement operation in microseconds.
    clock_rate_mhz : float
        Effective clock rate in MHz.
    max_circuit_depth : int
        Maximum circuit depth before coherence-limited fidelity.
    """

    name: str
    family: HardwareFamily
    num_qubits: int
    # Connectivity
    connectivity: str  # "all_to_all", "grid", "heavy_hex", "linear", "reconfigurable"
    # Gate fidelities (as success probability)
    single_qubit_fidelity: float
    two_qubit_fidelity: float
    readout_fidelity: float
    # Native gates
    native_1q_gates: tuple[str, ...]
    native_2q_gate: str
    native_3q_gate: str | None = None  # For neutral-atom CCZ
    # Timing (microseconds)
    t1_us: float = 1000.0
    t2_us: float = 500.0
    single_qubit_gate_us: float = 1.0
    two_qubit_gate_us: float = 10.0
    readout_us: float = 100.0
    # Speed (operations per second)
    clock_rate_mhz: float = 1.0
    # Max circuit depth before coherence-limited
    max_circuit_depth: int = 1000

    @property
    def error_per_1q(self) -> float:
        """Single-qubit gate error rate."""
        return 1.0 - self.single_qubit_fidelity

    @property
    def error_per_2q(self) -> float:
        """Two-qubit gate error rate."""
        return 1.0 - self.two_qubit_fidelity

    @property
    def error_per_readout(self) -> float:
        """Measurement (readout) error rate."""
        return 1.0 - self.readout_fidelity


class HardwareProfile(Enum):
    """Pre-configured hardware profiles matching real quantum processors.

    Each member wraps a :class:`HardwareSpec` with calibration data drawn
    from published device characterization papers and vendor specifications.

    Access the underlying spec via the :attr:`spec` property::

        >>> HardwareProfile.IONQ_ARIA.spec.num_qubits
        25
    """

    # ---- Trapped-ion ----

    IONQ_ARIA = HardwareSpec(
        name="IonQ Aria",
        family=HardwareFamily.TRAPPED_ION,
        num_qubits=25,
        connectivity="all_to_all",
        single_qubit_fidelity=0.9998,
        two_qubit_fidelity=0.995,
        readout_fidelity=0.997,
        native_1q_gates=("gpi", "gpi2"),
        native_2q_gate="ms",
        t1_us=100_000.0,
        t2_us=1_000.0,
        single_qubit_gate_us=10.0,
        two_qubit_gate_us=200.0,
        readout_us=300.0,
        clock_rate_mhz=0.005,
        max_circuit_depth=200,
    )

    IONQ_FORTE = HardwareSpec(
        name="IonQ Forte",
        family=HardwareFamily.TRAPPED_ION,
        num_qubits=36,
        connectivity="all_to_all",
        single_qubit_fidelity=0.9999,
        two_qubit_fidelity=0.997,
        readout_fidelity=0.998,
        native_1q_gates=("gpi", "gpi2"),
        native_2q_gate="ms",
        t1_us=200_000.0,
        t2_us=2_000.0,
        single_qubit_gate_us=8.0,
        two_qubit_gate_us=150.0,
        readout_us=250.0,
        clock_rate_mhz=0.007,
        max_circuit_depth=300,
    )

    QUANTINUUM_H2 = HardwareSpec(
        name="Quantinuum H2",
        family=HardwareFamily.TRAPPED_ION,
        num_qubits=56,
        connectivity="all_to_all",
        single_qubit_fidelity=0.99998,
        two_qubit_fidelity=0.998,
        readout_fidelity=0.999,
        native_1q_gates=("rz", "u1q"),
        native_2q_gate="zz",
        t1_us=500_000.0,
        t2_us=3_000.0,
        single_qubit_gate_us=5.0,
        two_qubit_gate_us=250.0,
        readout_us=200.0,
        clock_rate_mhz=0.004,
        max_circuit_depth=500,
    )

    # ---- Superconducting ----

    IBM_EAGLE = HardwareSpec(
        name="IBM Eagle (127Q)",
        family=HardwareFamily.SUPERCONDUCTING,
        num_qubits=127,
        connectivity="heavy_hex",
        single_qubit_fidelity=0.9996,
        two_qubit_fidelity=0.99,
        readout_fidelity=0.98,
        native_1q_gates=("sx", "rz", "x"),
        native_2q_gate="ecr",
        t1_us=300.0,
        t2_us=150.0,
        single_qubit_gate_us=0.035,
        two_qubit_gate_us=0.66,
        readout_us=1.0,
        clock_rate_mhz=30.0,
        max_circuit_depth=300,
    )

    IBM_HERON = HardwareSpec(
        name="IBM Heron (133Q)",
        family=HardwareFamily.SUPERCONDUCTING,
        num_qubits=133,
        connectivity="heavy_hex",
        single_qubit_fidelity=0.9998,
        two_qubit_fidelity=0.995,
        readout_fidelity=0.985,
        native_1q_gates=("sx", "rz", "x"),
        native_2q_gate="cz",
        t1_us=400.0,
        t2_us=200.0,
        single_qubit_gate_us=0.03,
        two_qubit_gate_us=0.08,
        readout_us=0.8,
        clock_rate_mhz=35.0,
        max_circuit_depth=500,
    )

    GOOGLE_SYCAMORE = HardwareSpec(
        name="Google Sycamore (72Q)",
        family=HardwareFamily.SUPERCONDUCTING,
        num_qubits=72,
        connectivity="grid",
        single_qubit_fidelity=0.9985,
        two_qubit_fidelity=0.995,
        readout_fidelity=0.962,
        native_1q_gates=("phased_xz",),
        native_2q_gate="syc",
        t1_us=20.0,
        t2_us=10.0,
        single_qubit_gate_us=0.025,
        two_qubit_gate_us=0.032,
        readout_us=1.0,
        clock_rate_mhz=40.0,
        max_circuit_depth=25,
    )

    RIGETTI_ANKAA2 = HardwareSpec(
        name="Rigetti Ankaa-2 (84Q)",
        family=HardwareFamily.SUPERCONDUCTING,
        num_qubits=84,
        connectivity="grid",
        single_qubit_fidelity=0.999,
        two_qubit_fidelity=0.975,
        readout_fidelity=0.975,
        native_1q_gates=("rx", "rz"),
        native_2q_gate="cz",
        t1_us=25.0,
        t2_us=15.0,
        single_qubit_gate_us=0.04,
        two_qubit_gate_us=0.2,
        readout_us=2.0,
        clock_rate_mhz=25.0,
        max_circuit_depth=100,
    )

    # ---- Neutral-atom ----

    QUERA_AQUILA = HardwareSpec(
        name="QuEra Aquila (256Q)",
        family=HardwareFamily.NEUTRAL_ATOM,
        num_qubits=256,
        connectivity="reconfigurable",
        single_qubit_fidelity=0.995,
        two_qubit_fidelity=0.985,
        readout_fidelity=0.97,
        native_1q_gates=("rx", "ry", "rz"),
        native_2q_gate="cz",
        native_3q_gate="ccz",
        t1_us=5_000.0,
        t2_us=1_000.0,
        single_qubit_gate_us=0.5,
        two_qubit_gate_us=1.0,
        readout_us=50.0,
        clock_rate_mhz=1.0,
        max_circuit_depth=200,
    )

    ATOM_COMPUTING = HardwareSpec(
        name="Atom Computing (1225Q)",
        family=HardwareFamily.NEUTRAL_ATOM,
        num_qubits=1225,
        connectivity="reconfigurable",
        single_qubit_fidelity=0.997,
        two_qubit_fidelity=0.99,
        readout_fidelity=0.98,
        native_1q_gates=("rx", "ry", "rz"),
        native_2q_gate="cz",
        native_3q_gate="ccz",
        t1_us=10_000.0,
        t2_us=2_000.0,
        single_qubit_gate_us=0.4,
        two_qubit_gate_us=0.8,
        readout_us=40.0,
        clock_rate_mhz=1.2,
        max_circuit_depth=300,
    )

    @property
    def spec(self) -> HardwareSpec:
        """Return the :class:`HardwareSpec` for this profile."""
        return self.value

    @classmethod
    def by_family(cls, family: HardwareFamily) -> list[HardwareProfile]:
        """Return all profiles belonging to a hardware family.

        Parameters
        ----------
        family : HardwareFamily
            Target family (e.g. ``HardwareFamily.TRAPPED_ION``).

        Returns
        -------
        list[HardwareProfile]
            Matching profiles sorted by qubit count ascending.
        """
        return sorted(
            [p for p in cls if p.value.family == family],
            key=lambda p: p.value.num_qubits,
        )

    @classmethod
    def by_name(cls, name: str) -> HardwareProfile:
        """Look up a profile by its device name or enum name.

        Parameters
        ----------
        name : str
            Device name (e.g. ``"IonQ Aria"``) or enum name
            (e.g. ``"IONQ_ARIA"``).  Case-insensitive.

        Returns
        -------
        HardwareProfile

        Raises
        ------
        ValueError
            If no matching profile is found.
        """
        needle = name.lower()
        for p in cls:
            if p.value.name.lower() == needle or p.name.lower() == needle:
                return p
        valid = ", ".join(p.name for p in cls)
        raise ValueError(
            f"Unknown hardware profile: {name!r}. "
            f"Valid profiles: {valid}"
        )
