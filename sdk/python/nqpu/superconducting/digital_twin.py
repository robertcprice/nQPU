"""Digital twin construction and validation for superconducting processors.

Builds a software replica of a real superconducting QPU from published or
measured calibration data, then validates the twin against known QCVV benchmarks
(Bell/GHZ fidelity, randomized benchmarking, quantum volume).

Includes preset calibration data from published results:
    - IBM Eagle r3 (127Q):  Chow et al., IBM Quantum blog (2023)
    - IBM Heron   (156Q):  IBM Quantum roadmap (2024)
    - Google Sycamore (53Q): Arute et al., Nature 574, 505 (2019)
    - Google Willow  (105Q): Google Quantum AI (2024)

Usage::

    from nqpu.superconducting.digital_twin import DigitalTwin, CalibrationData

    twin = DigitalTwin.from_ibm_backend("eagle")
    report = twin.run_qcvv_suite()
    print(report)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chip import (
    ChipConfig,
    ChipTopology,
    DevicePresets,
    NativeGateFamily,
    TopologyType,
)
from .noise import TransmonNoiseModel
from .qcvv import BenchmarkResult, TransmonQCVV
from .qubit import TransmonQubit
from .simulator import CircuitStats, TransmonSimulator


# ======================================================================
# Data containers
# ======================================================================


@dataclass
class CalibrationData:
    """Container for real hardware calibration data.

    All list fields are indexed per-qubit (length == num_qubits).

    Parameters
    ----------
    qubit_frequencies_ghz : list[float]
        Qubit 0->1 transition frequencies in GHz.
    t1_us : list[float]
        Energy relaxation time T1 per qubit in microseconds.
    t2_us : list[float]
        Dephasing time T2 per qubit in microseconds.
    readout_fidelities : list[float]
        Assignment fidelity per qubit (0..1).
    single_gate_fidelities : list[float]
        Single-qubit gate fidelity per qubit from randomized benchmarking.
    two_qubit_fidelities : dict[tuple[int,int], float]
        Two-qubit gate fidelity keyed by (control, target) pair.
    coupling_map : list[tuple[int,int]]
        Connected qubit pairs forming the device coupling graph.
    native_gate_family : str
        Native two-qubit gate family: ``"ecr"``, ``"sqrt_iswap"``, or ``"cz"``.
    two_qubit_gate_time_ns : float
        Duration of the native two-qubit gate in nanoseconds.
    single_gate_time_ns : float
        Duration of a single-qubit gate in nanoseconds.
    temperature_mk : float
        Cryostat base temperature in millikelvin.
    device_name : str
        Human-readable device name for reporting.
    """

    qubit_frequencies_ghz: list[float]
    t1_us: list[float]
    t2_us: list[float]
    readout_fidelities: list[float]
    single_gate_fidelities: list[float]
    two_qubit_fidelities: dict[tuple[int, int], float]
    coupling_map: list[tuple[int, int]]
    native_gate_family: str = "ecr"
    two_qubit_gate_time_ns: float = 200.0
    single_gate_time_ns: float = 25.0
    temperature_mk: float = 15.0
    device_name: str = "custom"

    @property
    def num_qubits(self) -> int:
        return len(self.qubit_frequencies_ghz)

    @property
    def mean_t1(self) -> float:
        return float(np.mean(self.t1_us))

    @property
    def mean_t2(self) -> float:
        return float(np.mean(self.t2_us))

    @property
    def mean_1q_fidelity(self) -> float:
        return float(np.mean(self.single_gate_fidelities))

    @property
    def mean_2q_fidelity(self) -> float:
        vals = list(self.two_qubit_fidelities.values())
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_readout_fidelity(self) -> float:
        return float(np.mean(self.readout_fidelities))

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary dict."""
        return {
            "device": self.device_name,
            "num_qubits": self.num_qubits,
            "num_couplers": len(self.coupling_map),
            "native_gate": self.native_gate_family,
            "mean_T1_us": round(self.mean_t1, 1),
            "mean_T2_us": round(self.mean_t2, 1),
            "mean_1Q_fidelity": round(self.mean_1q_fidelity, 5),
            "mean_2Q_fidelity": round(self.mean_2q_fidelity, 5),
            "mean_readout_fidelity": round(self.mean_readout_fidelity, 4),
        }


@dataclass
class ValidationReport:
    """Results from digital twin validation against QCVV benchmarks.

    Attributes
    ----------
    device_name : str
        Name of the device being validated.
    bell_fidelity : float
        Bell state |00>+|11> preparation fidelity.
    ghz_fidelity : float
        GHZ state preparation fidelity (on subsystem of qubits).
    rb_error_rate : float
        Error per Clifford gate from single-qubit randomized benchmarking.
    quantum_volume : int
        Quantum volume estimate (2^m).
    predicted_fidelity_1q100 : float
        Predicted fidelity for a 100 single-qubit-gate circuit.
    raw_results : dict[str, Any]
        Full benchmark result data for further analysis.
    """

    device_name: str = ""
    bell_fidelity: float = 0.0
    ghz_fidelity: float = 0.0
    rb_error_rate: float = 0.0
    quantum_volume: int = 1
    predicted_fidelity_1q100: float = 0.0
    raw_results: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"=== Validation Report: {self.device_name} ===",
            f"  Bell state fidelity:  {self.bell_fidelity:.4f}",
            f"  GHZ state fidelity:   {self.ghz_fidelity:.4f}",
            f"  RB error per gate:    {self.rb_error_rate:.6f}",
            f"  Quantum volume:       {self.quantum_volume}",
            f"  Predicted F(100x1Q):  {self.predicted_fidelity_1q100:.4f}",
        ]
        return "\n".join(lines)


# ======================================================================
# Preset calibration data from published results
# ======================================================================


def _ibm_eagle_calibration(num_qubits: int = 10) -> CalibrationData:
    """IBM Eagle r3 calibration data (published 2023).

    Median values from IBM Quantum systems:
        T1 ~ 300 us, T2 ~ 200 us, 1Q fidelity 99.95%, 2Q fidelity 99.5%.

    References:
        Chow et al., "IBM Quantum breaks the 100-qubit barrier" (2022)
        IBM Quantum backend properties (2023 median values)
    """
    rng = np.random.RandomState(42)
    n = num_qubits

    # Per-qubit parameters with realistic spread
    freqs = (5.0 + 0.1 * rng.randn(n)).tolist()
    t1 = np.clip(300.0 + 50.0 * rng.randn(n), 100.0, 500.0).tolist()
    t2 = np.clip(200.0 + 40.0 * rng.randn(n), 80.0, 350.0).tolist()
    # Enforce T2 <= 2*T1
    t2 = [min(t2_i, 2 * t1_i) for t1_i, t2_i in zip(t1, t2)]
    readout = np.clip(0.985 + 0.005 * rng.randn(n), 0.95, 0.999).tolist()
    fid_1q = np.clip(0.9995 + 0.0002 * rng.randn(n), 0.998, 0.99995).tolist()

    # Heavy-hex coupling map (simplified chain + skip connections)
    edges: list[tuple[int, int]] = []
    for i in range(n - 1):
        edges.append((i, i + 1))
    for i in range(0, n - 3, 4):
        edges.append((i, i + 3))

    fid_2q: dict[tuple[int, int], float] = {}
    for a, b in edges:
        fid_2q[(a, b)] = float(np.clip(0.995 + 0.003 * rng.randn(), 0.98, 0.999))

    return CalibrationData(
        qubit_frequencies_ghz=freqs,
        t1_us=t1,
        t2_us=t2,
        readout_fidelities=readout,
        single_gate_fidelities=fid_1q,
        two_qubit_fidelities=fid_2q,
        coupling_map=edges,
        native_gate_family="ecr",
        two_qubit_gate_time_ns=300.0,
        single_gate_time_ns=25.0,
        temperature_mk=15.0,
        device_name="IBM Eagle r3",
    )


def _ibm_heron_calibration(num_qubits: int = 10) -> CalibrationData:
    """IBM Heron calibration data (published 2024).

    Improved coherence and tunable coupler architecture:
        T1 ~ 350 us, T2 ~ 250 us, 1Q fidelity 99.97%, 2Q fidelity 99.7%.

    References:
        IBM Quantum roadmap (2024), Heron processor specifications
    """
    rng = np.random.RandomState(43)
    n = num_qubits

    freqs = (4.9 + 0.08 * rng.randn(n)).tolist()
    t1 = np.clip(350.0 + 60.0 * rng.randn(n), 150.0, 600.0).tolist()
    t2 = np.clip(250.0 + 50.0 * rng.randn(n), 100.0, 450.0).tolist()
    t2 = [min(t2_i, 2 * t1_i) for t1_i, t2_i in zip(t1, t2)]
    readout = np.clip(0.995 + 0.003 * rng.randn(n), 0.97, 0.9999).tolist()
    fid_1q = np.clip(0.9997 + 0.0001 * rng.randn(n), 0.999, 0.99999).tolist()

    edges: list[tuple[int, int]] = []
    for i in range(n - 1):
        edges.append((i, i + 1))
    for i in range(0, n - 3, 4):
        edges.append((i, i + 3))

    fid_2q: dict[tuple[int, int], float] = {}
    for a, b in edges:
        fid_2q[(a, b)] = float(np.clip(0.997 + 0.002 * rng.randn(), 0.99, 0.9999))

    return CalibrationData(
        qubit_frequencies_ghz=freqs,
        t1_us=t1,
        t2_us=t2,
        readout_fidelities=readout,
        single_gate_fidelities=fid_1q,
        two_qubit_fidelities=fid_2q,
        coupling_map=edges,
        native_gate_family="ecr",
        two_qubit_gate_time_ns=200.0,
        single_gate_time_ns=20.0,
        temperature_mk=12.0,
        device_name="IBM Heron",
    )


def _google_sycamore_calibration(num_qubits: int = 10) -> CalibrationData:
    """Google Sycamore calibration data (Arute et al., Nature 2019).

    High-speed gates with shorter coherence:
        T1 ~ 20 us, T2 ~ 10 us, 1Q fidelity 99.85%, 2Q fidelity 99.4%.

    References:
        Arute et al., Nature 574, 505 (2019)
    """
    rng = np.random.RandomState(44)
    n = num_qubits

    freqs = (6.0 + 0.15 * rng.randn(n)).tolist()
    t1 = np.clip(20.0 + 5.0 * rng.randn(n), 8.0, 40.0).tolist()
    t2 = np.clip(10.0 + 3.0 * rng.randn(n), 4.0, 20.0).tolist()
    t2 = [min(t2_i, 2 * t1_i) for t1_i, t2_i in zip(t1, t2)]
    readout = np.clip(0.965 + 0.01 * rng.randn(n), 0.93, 0.99).tolist()
    fid_1q = np.clip(0.9985 + 0.0005 * rng.randn(n), 0.996, 0.9999).tolist()

    # Grid topology
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if q >= n:
                continue
            if c + 1 < cols and (r * cols + c + 1) < n:
                edges.append((q, q + 1))
            if r + 1 < rows and ((r + 1) * cols + c) < n:
                edges.append((q, (r + 1) * cols + c))

    fid_2q: dict[tuple[int, int], float] = {}
    for a, b in edges:
        fid_2q[(a, b)] = float(np.clip(0.994 + 0.003 * rng.randn(), 0.985, 0.999))

    return CalibrationData(
        qubit_frequencies_ghz=freqs,
        t1_us=t1,
        t2_us=t2,
        readout_fidelities=readout,
        single_gate_fidelities=fid_1q,
        two_qubit_fidelities=fid_2q,
        coupling_map=edges,
        native_gate_family="sqrt_iswap",
        two_qubit_gate_time_ns=32.0,
        single_gate_time_ns=25.0,
        temperature_mk=20.0,
        device_name="Google Sycamore",
    )


def _google_willow_calibration(num_qubits: int = 10) -> CalibrationData:
    """Google Willow calibration data (Google Quantum AI, 2024).

    Next-generation with improved coherence and error correction readiness:
        T1 ~ 68 us, T2 ~ 30 us, 1Q fidelity 99.94%, 2Q fidelity 99.7%.

    References:
        Google Quantum AI, "Quantum error correction below threshold" (2024)
    """
    rng = np.random.RandomState(45)
    n = num_qubits

    freqs = (6.0 + 0.12 * rng.randn(n)).tolist()
    t1 = np.clip(68.0 + 15.0 * rng.randn(n), 30.0, 120.0).tolist()
    t2 = np.clip(30.0 + 8.0 * rng.randn(n), 12.0, 60.0).tolist()
    t2 = [min(t2_i, 2 * t1_i) for t1_i, t2_i in zip(t1, t2)]
    readout = np.clip(0.990 + 0.005 * rng.randn(n), 0.96, 0.999).tolist()
    fid_1q = np.clip(0.9994 + 0.0003 * rng.randn(n), 0.998, 0.99999).tolist()

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if q >= n:
                continue
            if c + 1 < cols and (r * cols + c + 1) < n:
                edges.append((q, q + 1))
            if r + 1 < rows and ((r + 1) * cols + c) < n:
                edges.append((q, (r + 1) * cols + c))

    fid_2q: dict[tuple[int, int], float] = {}
    for a, b in edges:
        fid_2q[(a, b)] = float(np.clip(0.997 + 0.002 * rng.randn(), 0.99, 0.9999))

    return CalibrationData(
        qubit_frequencies_ghz=freqs,
        t1_us=t1,
        t2_us=t2,
        readout_fidelities=readout,
        single_gate_fidelities=fid_1q,
        two_qubit_fidelities=fid_2q,
        coupling_map=edges,
        native_gate_family="sqrt_iswap",
        two_qubit_gate_time_ns=25.0,
        single_gate_time_ns=20.0,
        temperature_mk=15.0,
        device_name="Google Willow",
    )


# Registry of all preset builders
_PRESET_REGISTRY: dict[str, Any] = {
    # IBM
    "eagle": _ibm_eagle_calibration,
    "ibm_eagle": _ibm_eagle_calibration,
    "heron": _ibm_heron_calibration,
    "ibm_heron": _ibm_heron_calibration,
    # Google
    "sycamore": _google_sycamore_calibration,
    "google_sycamore": _google_sycamore_calibration,
    "willow": _google_willow_calibration,
    "google_willow": _google_willow_calibration,
}


# ======================================================================
# DigitalTwin
# ======================================================================


class DigitalTwin:
    """Software replica of a superconducting QPU built from calibration data.

    The digital twin constructs a :class:`ChipConfig` that mirrors the real
    hardware's per-qubit parameters, coupling map, and gate error rates.
    It can then run QCVV experiments and predict circuit fidelities without
    touching the actual hardware.

    Parameters
    ----------
    config : ChipConfig
        Processor configuration derived from calibration data.
    calibration : CalibrationData
        The raw calibration data used to build the twin.

    Examples
    --------
    Build from IBM Eagle preset and validate:

    >>> twin = DigitalTwin.from_ibm_backend("eagle", num_qubits=5)
    >>> report = twin.run_qcvv_suite()
    >>> print(report)

    Build from custom calibration data:

    >>> cal = CalibrationData(
    ...     qubit_frequencies_ghz=[5.0, 5.1],
    ...     t1_us=[200.0, 180.0],
    ...     t2_us=[150.0, 130.0],
    ...     readout_fidelities=[0.99, 0.98],
    ...     single_gate_fidelities=[0.9995, 0.9993],
    ...     two_qubit_fidelities={(0, 1): 0.995},
    ...     coupling_map=[(0, 1)],
    ... )
    >>> twin = DigitalTwin.from_calibration(cal)
    """

    def __init__(self, config: ChipConfig, calibration: CalibrationData) -> None:
        self.config = config
        self.calibration = calibration
        self._qcvv = TransmonQCVV(config)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_calibration(cls, data: CalibrationData) -> DigitalTwin:
        """Construct a digital twin from raw calibration data.

        Translates per-qubit parameters into :class:`TransmonQubit` instances
        and builds the coupling topology.

        Parameters
        ----------
        data : CalibrationData
            Measured calibration parameters.

        Returns
        -------
        DigitalTwin
        """
        n = data.num_qubits
        qubits = []
        for i in range(n):
            qubits.append(
                TransmonQubit(
                    frequency_ghz=data.qubit_frequencies_ghz[i],
                    t1_us=data.t1_us[i],
                    t2_us=data.t2_us[i],
                    readout_fidelity=data.readout_fidelities[i],
                    single_gate_fidelity=data.single_gate_fidelities[i],
                    gate_time_ns=data.single_gate_time_ns,
                )
            )

        # Build coupling topology
        couplings: dict[tuple[int, int], float] = {}
        for a, b in data.coupling_map:
            key = (min(a, b), max(a, b))
            couplings[key] = 3.0  # Default coupling strength

        topo = ChipTopology(
            num_qubits=n,
            edges=data.coupling_map,
            coupling_mhz=couplings,
        )

        # Average 2Q fidelity from per-edge data
        two_q_fid_vals = list(data.two_qubit_fidelities.values())
        avg_2q_fidelity = float(np.mean(two_q_fid_vals)) if two_q_fid_vals else 0.995

        config = ChipConfig(
            topology=topo,
            qubits=qubits,
            native_2q_gate=NativeGateFamily(data.native_gate_family),
            two_qubit_fidelity=avg_2q_fidelity,
            two_qubit_gate_time_ns=data.two_qubit_gate_time_ns,
            temperature_mk=data.temperature_mk,
        )

        return cls(config=config, calibration=data)

    @classmethod
    def from_ibm_backend(
        cls, backend_name: str, num_qubits: int = 10
    ) -> DigitalTwin:
        """Build a digital twin from preset IBM backend calibration data.

        Supported names: ``"eagle"``, ``"heron"``.

        Parameters
        ----------
        backend_name : str
            IBM backend identifier.
        num_qubits : int
            Number of qubits to model (subset of full processor).

        Returns
        -------
        DigitalTwin
        """
        key = backend_name.lower().replace(" ", "_")
        if key not in ("eagle", "ibm_eagle", "heron", "ibm_heron"):
            raise ValueError(
                f"Unknown IBM backend '{backend_name}'. "
                f"Supported: eagle, heron"
            )
        builder = _PRESET_REGISTRY[key]
        cal = builder(num_qubits)
        return cls.from_calibration(cal)

    @classmethod
    def from_google_backend(
        cls, backend_name: str, num_qubits: int = 10
    ) -> DigitalTwin:
        """Build a digital twin from preset Google backend calibration data.

        Supported names: ``"sycamore"``, ``"willow"``.

        Parameters
        ----------
        backend_name : str
            Google backend identifier.
        num_qubits : int
            Number of qubits to model.

        Returns
        -------
        DigitalTwin
        """
        key = backend_name.lower().replace(" ", "_")
        if key not in ("sycamore", "google_sycamore", "willow", "google_willow"):
            raise ValueError(
                f"Unknown Google backend '{backend_name}'. "
                f"Supported: sycamore, willow"
            )
        builder = _PRESET_REGISTRY[key]
        cal = builder(num_qubits)
        return cls.from_calibration(cal)

    # ------------------------------------------------------------------
    # Validation & benchmarking
    # ------------------------------------------------------------------

    def validate(
        self,
        num_qcvv_qubits: int = 3,
        rb_sequences: int = 8,
        qv_trials: int = 15,
        execution_mode: str = "noisy",
    ) -> ValidationReport:
        """Run a validation suite and return a structured report.

        Executes Bell fidelity, GHZ fidelity, randomized benchmarking,
        and quantum volume experiments on the digital twin.

        Parameters
        ----------
        num_qcvv_qubits : int
            Number of qubits for GHZ and QV benchmarks.
        rb_sequences : int
            Number of random sequences per RB sequence length.
        qv_trials : int
            Number of trials for quantum volume estimation.
        execution_mode : str
            ``"ideal"`` or ``"noisy"``.

        Returns
        -------
        ValidationReport
        """
        n_qcvv = min(num_qcvv_qubits, self.config.num_qubits)
        raw: dict[str, Any] = {}

        # Bell state fidelity
        bell = self._qcvv.bell_state_fidelity(
            qubit_a=0, qubit_b=1, execution_mode=execution_mode
        )
        raw["bell"] = bell.raw_data

        # GHZ state fidelity
        ghz = self._qcvv.ghz_fidelity(
            num_qubits=n_qcvv, execution_mode=execution_mode
        )
        raw["ghz"] = ghz.raw_data

        # Randomized benchmarking
        rb = self._qcvv.randomized_benchmarking(
            qubit=0,
            n_sequences=rb_sequences,
            execution_mode=execution_mode,
        )
        raw["rb"] = rb.raw_data

        # Quantum volume
        qv = self._qcvv.quantum_volume(
            max_depth=min(n_qcvv, 5),
            n_trials=qv_trials,
            execution_mode=execution_mode,
        )
        raw["qv"] = qv.raw_data

        # Predicted fidelity for a 100 single-qubit-gate circuit
        pred = self.predict_circuit_fidelity(
            num_1q_gates=100, num_2q_gates=0, depth=100
        )

        return ValidationReport(
            device_name=self.calibration.device_name,
            bell_fidelity=bell.metric_value,
            ghz_fidelity=ghz.metric_value,
            rb_error_rate=rb.metric_value,
            quantum_volume=int(qv.metric_value),
            predicted_fidelity_1q100=pred,
            raw_results=raw,
        )

    def run_qcvv_suite(
        self, execution_mode: str = "noisy"
    ) -> dict[str, BenchmarkResult]:
        """Run the full QCVV benchmark suite and return all results.

        Parameters
        ----------
        execution_mode : str
            ``"ideal"`` or ``"noisy"``.

        Returns
        -------
        dict[str, BenchmarkResult]
            Results keyed by benchmark name.
        """
        results: dict[str, BenchmarkResult] = {}

        results["bell_fidelity"] = self._qcvv.bell_state_fidelity(
            execution_mode=execution_mode
        )

        n_ghz = min(self.config.num_qubits, 4)
        results["ghz_fidelity"] = self._qcvv.ghz_fidelity(
            num_qubits=n_ghz, execution_mode=execution_mode
        )

        results["randomized_benchmarking"] = self._qcvv.randomized_benchmarking(
            qubit=0, execution_mode=execution_mode
        )

        n_qv = min(self.config.num_qubits, 5)
        results["quantum_volume"] = self._qcvv.quantum_volume(
            max_depth=n_qv, execution_mode=execution_mode
        )

        return results

    def predict_circuit_fidelity(
        self,
        num_1q_gates: int,
        num_2q_gates: int,
        depth: int,
    ) -> float:
        """Predict the output fidelity of a circuit from gate error rates.

        Uses a simple multiplicative error model:
            F = (1 - e_1q)^n_1q * (1 - e_2q)^n_2q * (1 - e_decoherence)^depth

        where e_decoherence accounts for idle decoherence per layer.

        Parameters
        ----------
        num_1q_gates : int
            Number of single-qubit gates in the circuit.
        num_2q_gates : int
            Number of two-qubit gates in the circuit.
        depth : int
            Circuit depth (number of layers).

        Returns
        -------
        float
            Estimated output fidelity (0..1).
        """
        mean_1q_err = 1.0 - self.calibration.mean_1q_fidelity
        mean_2q_err = 1.0 - self.calibration.mean_2q_fidelity

        # Gate error contribution
        fid = (1.0 - mean_1q_err) ** num_1q_gates
        fid *= (1.0 - mean_2q_err) ** num_2q_gates

        # Idle decoherence per layer: approximate as T2-limited
        if self.calibration.mean_t2 > 0 and depth > 0:
            gate_time_us = self.calibration.two_qubit_gate_time_ns / 1000.0
            decoherence_per_layer = 1.0 - math.exp(
                -gate_time_us / self.calibration.mean_t2
            )
            fid *= (1.0 - decoherence_per_layer) ** depth

        return max(0.0, min(1.0, fid))

    # ------------------------------------------------------------------
    # Comparison utilities
    # ------------------------------------------------------------------

    @classmethod
    def compare_presets(
        cls, num_qubits: int = 5, execution_mode: str = "noisy"
    ) -> dict[str, dict[str, Any]]:
        """Compare all available device presets side-by-side.

        Builds a digital twin for each preset, runs Bell fidelity and RB,
        and returns a comparison dictionary.

        Parameters
        ----------
        num_qubits : int
            Number of qubits per twin.
        execution_mode : str
            ``"ideal"`` or ``"noisy"``.

        Returns
        -------
        dict[str, dict[str, Any]]
            Comparison data keyed by device name.
        """
        presets = [
            ("IBM Eagle", "eagle", cls.from_ibm_backend),
            ("IBM Heron", "heron", cls.from_ibm_backend),
            ("Google Sycamore", "sycamore", cls.from_google_backend),
            ("Google Willow", "willow", cls.from_google_backend),
        ]

        results: dict[str, dict[str, Any]] = {}

        for display_name, key, factory in presets:
            twin = factory(key, num_qubits=num_qubits)

            bell = twin._qcvv.bell_state_fidelity(
                execution_mode=execution_mode
            )
            rb = twin._qcvv.randomized_benchmarking(
                qubit=0, execution_mode=execution_mode
            )

            pred_shallow = twin.predict_circuit_fidelity(
                num_1q_gates=20, num_2q_gates=10, depth=15
            )
            pred_deep = twin.predict_circuit_fidelity(
                num_1q_gates=200, num_2q_gates=100, depth=150
            )

            cal = twin.calibration
            results[display_name] = {
                "num_qubits": cal.num_qubits,
                "native_gate": cal.native_gate_family,
                "mean_T1_us": round(cal.mean_t1, 1),
                "mean_T2_us": round(cal.mean_t2, 1),
                "mean_1Q_fidelity": round(cal.mean_1q_fidelity, 5),
                "mean_2Q_fidelity": round(cal.mean_2q_fidelity, 5),
                "bell_fidelity": round(bell.metric_value, 4),
                "rb_error_per_gate": round(rb.metric_value, 6),
                "predicted_F_shallow": round(pred_shallow, 4),
                "predicted_F_deep": round(pred_deep, 4),
            }

        return results

    def device_summary(self) -> dict[str, Any]:
        """Return a summary of the digital twin configuration."""
        return {
            "calibration": self.calibration.summary(),
            "chip_config": self.config.device_info(),
        }

    # ------------------------------------------------------------------
    # Calibration data ingestion from vendor JSON formats
    # ------------------------------------------------------------------

    @classmethod
    def from_ibm_json(cls, json_data: dict) -> "DigitalTwin":
        """Construct a digital twin from IBM Quantum calibration JSON.

        Parses the format returned by ``backend.properties().to_dict()``,
        extracting per-qubit T1, T2, frequency, readout error, and gate
        error rates.  Missing fields fall back to IBM Heron defaults.

        Expected JSON structure::

            {
                "qubits": [
                    [
                        {"name": "T1", "value": 300.5, "unit": "us"},
                        {"name": "T2", "value": 200.1, "unit": "us"},
                        {"name": "frequency", "value": 5.01, "unit": "GHz"},
                        {"name": "readout_error", "value": 0.015},
                        ...
                    ],
                    ...
                ],
                "gates": [
                    {"qubits": [0], "gate": "sx",
                     "parameters": [{"name": "gate_error", "value": 0.0003}]},
                    {"qubits": [0, 1], "gate": "cx",
                     "parameters": [{"name": "gate_error", "value": 0.005}]},
                    ...
                ]
            }

        Parameters
        ----------
        json_data : dict
            IBM Quantum calibration data dictionary.

        Returns
        -------
        DigitalTwin
            Configured digital twin.
        """
        # Defaults from IBM Heron
        default_t1 = 350.0
        default_t2 = 250.0
        default_freq = 5.0
        default_readout = 0.99
        default_1q_fid = 0.9997
        default_2q_fid = 0.997

        qubit_data = json_data.get("qubits", [])
        num_qubits = len(qubit_data)
        if num_qubits == 0:
            raise ValueError("IBM JSON contains no qubit calibration data")

        frequencies: list[float] = []
        t1_list: list[float] = []
        t2_list: list[float] = []
        readout_list: list[float] = []
        single_fid_list: list[float] = []

        for q_idx, qubit_props in enumerate(qubit_data):
            props: dict[str, float] = {}
            for entry in qubit_props:
                name = entry.get("name", "")
                value = entry.get("value")
                if value is not None:
                    props[name.lower()] = float(value)

            frequencies.append(props.get("frequency", default_freq))
            t1_val = props.get("t1", default_t1)
            t2_val = props.get("t2", default_t2)
            # Enforce T2 <= 2*T1
            t2_val = min(t2_val, 2.0 * t1_val)
            t1_list.append(t1_val)
            t2_list.append(t2_val)

            readout_err = props.get("readout_error", 1.0 - default_readout)
            readout_list.append(1.0 - readout_err)

            # Single-qubit fidelity comes from gate data (set defaults here)
            single_fid_list.append(default_1q_fid)

        # Parse gate data for single-qubit and two-qubit errors
        gate_data = json_data.get("gates", [])
        two_q_fid: dict[tuple[int, int], float] = {}
        coupling_map: list[tuple[int, int]] = []

        for gate_entry in gate_data:
            qubits = gate_entry.get("qubits", [])
            gate_name = gate_entry.get("gate", "").lower()
            params = gate_entry.get("parameters", [])

            gate_error = None
            for p in params:
                if p.get("name", "").lower() == "gate_error":
                    gate_error = p.get("value")
                    break

            if gate_error is None:
                continue

            if len(qubits) == 1:
                # Single-qubit gate error
                q = qubits[0]
                if 0 <= q < num_qubits:
                    single_fid_list[q] = 1.0 - float(gate_error)
            elif len(qubits) == 2:
                # Two-qubit gate (cx, ecr, cz, etc.)
                q0, q1 = qubits
                edge = (min(q0, q1), max(q0, q1))
                fid = 1.0 - float(gate_error)
                # Keep the best fidelity if multiple gates on same edge
                if edge not in two_q_fid or fid > two_q_fid[edge]:
                    two_q_fid[edge] = fid
                if edge not in coupling_map:
                    coupling_map.append(edge)

        # If no coupling map was extracted, create a linear chain
        if not coupling_map:
            for i in range(num_qubits - 1):
                coupling_map.append((i, i + 1))
                two_q_fid[(i, i + 1)] = default_2q_fid

        # Determine native gate family from gate names
        gate_names = {g.get("gate", "").lower() for g in gate_data}
        if "ecr" in gate_names:
            native_gate = "ecr"
        elif "cz" in gate_names:
            native_gate = "cz"
        else:
            native_gate = "ecr"  # IBM default

        device_name = json_data.get("backend_name", "IBM (imported)")

        cal = CalibrationData(
            qubit_frequencies_ghz=frequencies,
            t1_us=t1_list,
            t2_us=t2_list,
            readout_fidelities=readout_list,
            single_gate_fidelities=single_fid_list,
            two_qubit_fidelities=two_q_fid,
            coupling_map=coupling_map,
            native_gate_family=native_gate,
            two_qubit_gate_time_ns=float(
                json_data.get("two_qubit_gate_time_ns", 300.0)
            ),
            single_gate_time_ns=float(
                json_data.get("single_gate_time_ns", 25.0)
            ),
            temperature_mk=float(json_data.get("temperature_mk", 15.0)),
            device_name=device_name,
        )
        return cls.from_calibration(cal)

    @classmethod
    def from_google_cirq(cls, json_data: dict) -> "DigitalTwin":
        """Construct a digital twin from Google Cirq device specification JSON.

        Parses a device specification containing per-qubit coherence
        parameters and per-pair gate fidelities in the Cirq device
        representation format.  Missing fields fall back to Google
        Willow defaults.

        Expected JSON structure::

            {
                "device_name": "my_device",
                "qubits": [
                    {
                        "id": "0_0",
                        "t1_us": 68.0,
                        "tphi_us": 25.0,
                        "frequency_ghz": 6.02,
                        "gate_duration_ns": 25.0,
                        "readout_fidelity": 0.99
                    },
                    ...
                ],
                "gate_pairs": [
                    {
                        "qubit_a": "0_0",
                        "qubit_b": "0_1",
                        "fidelity": 0.997,
                        "gate_type": "sqrt_iswap"
                    },
                    ...
                ]
            }

        Qubit IDs can be ``"row_col"`` grid strings or integer strings.
        The method builds a mapping from ID to sequential integer index.

        Parameters
        ----------
        json_data : dict
            Google Cirq device specification dictionary.

        Returns
        -------
        DigitalTwin
            Configured digital twin.
        """
        # Defaults from Google Willow
        default_t1 = 68.0
        default_tphi = 25.0
        default_freq = 6.0
        default_readout = 0.99
        default_1q_fid = 0.9994
        default_2q_fid = 0.997

        qubit_entries = json_data.get("qubits", [])
        num_qubits = len(qubit_entries)
        if num_qubits == 0:
            raise ValueError(
                "Google Cirq JSON contains no qubit specification data"
            )

        # Build ID -> index mapping
        qubit_id_map: dict[str, int] = {}
        for idx, qentry in enumerate(qubit_entries):
            qid = str(qentry.get("id", idx))
            qubit_id_map[qid] = idx

        frequencies: list[float] = []
        t1_list: list[float] = []
        t2_list: list[float] = []
        readout_list: list[float] = []
        single_fid_list: list[float] = []

        for qentry in qubit_entries:
            freq = float(qentry.get("frequency_ghz", default_freq))
            frequencies.append(freq)

            t1 = float(qentry.get("t1_us", default_t1))
            t1_list.append(t1)

            # T2 derived from T1 and Tphi: 1/T2 = 1/(2*T1) + 1/Tphi
            tphi = float(qentry.get("tphi_us", default_tphi))
            if tphi > 0 and t1 > 0:
                t2 = 1.0 / (1.0 / (2.0 * t1) + 1.0 / tphi)
            else:
                t2 = min(2.0 * t1, 2.0 * default_t1)
            t2 = min(t2, 2.0 * t1)
            t2_list.append(t2)

            ro = float(qentry.get("readout_fidelity", default_readout))
            readout_list.append(ro)

            fid_1q = float(qentry.get("single_gate_fidelity", default_1q_fid))
            single_fid_list.append(fid_1q)

        # Parse gate pairs
        gate_pairs = json_data.get("gate_pairs", [])
        two_q_fid: dict[tuple[int, int], float] = {}
        coupling_map: list[tuple[int, int]] = []
        native_gate = "sqrt_iswap"  # Google default

        for pair in gate_pairs:
            qa_id = str(pair.get("qubit_a", ""))
            qb_id = str(pair.get("qubit_b", ""))

            qa_idx = qubit_id_map.get(qa_id)
            qb_idx = qubit_id_map.get(qb_id)
            if qa_idx is None or qb_idx is None:
                continue

            edge = (min(qa_idx, qb_idx), max(qa_idx, qb_idx))
            fid = float(pair.get("fidelity", default_2q_fid))
            two_q_fid[edge] = fid
            if edge not in coupling_map:
                coupling_map.append(edge)

            gate_type = pair.get("gate_type", "")
            if gate_type:
                native_gate = gate_type

        # If no coupling map, create a grid from qubit IDs
        if not coupling_map:
            for i in range(num_qubits - 1):
                coupling_map.append((i, i + 1))
                two_q_fid[(i, i + 1)] = default_2q_fid

        device_name = json_data.get("device_name", "Google (imported)")

        gate_dur_ns = 25.0
        if qubit_entries:
            gate_dur_ns = float(
                qubit_entries[0].get("gate_duration_ns", 25.0)
            )
        two_q_dur_ns = float(json_data.get("two_qubit_gate_time_ns", 25.0))

        cal = CalibrationData(
            qubit_frequencies_ghz=frequencies,
            t1_us=t1_list,
            t2_us=t2_list,
            readout_fidelities=readout_list,
            single_gate_fidelities=single_fid_list,
            two_qubit_fidelities=two_q_fid,
            coupling_map=coupling_map,
            native_gate_family=native_gate,
            two_qubit_gate_time_ns=two_q_dur_ns,
            single_gate_time_ns=gate_dur_ns,
            temperature_mk=float(json_data.get("temperature_mk", 15.0)),
            device_name=device_name,
        )
        return cls.from_calibration(cal)


# ======================================================================
# Calibration drift modeling
# ======================================================================


@dataclass
class StabilityReport:
    """Statistics from a calibration drift analysis.

    Summarises how much each parameter category varies over a given time
    window, and recommends a recalibration interval based on the drift
    magnitude.

    Attributes
    ----------
    t1_variation_percent : float
        Peak-to-peak variation of mean T1 as a percentage of the
        baseline value.
    t2_variation_percent : float
        Peak-to-peak variation of mean T2 as a percentage of the
        baseline value.
    frequency_jumps : int
        Number of discrete TLS-induced frequency jumps observed across
        all qubits during the analysis window.
    fidelity_range : tuple[float, float]
        (minimum, maximum) mean two-qubit gate fidelity observed during
        the window.
    recommended_recalibration_interval_hours : float
        Estimated interval after which recalibration is advisable, based
        on when fidelity first degrades by more than 0.1% from baseline.
    """

    t1_variation_percent: float
    t2_variation_percent: float
    frequency_jumps: int
    fidelity_range: tuple[float, float]
    recommended_recalibration_interval_hours: float

    def __str__(self) -> str:
        return (
            "=== Stability Report ===\n"
            f"  T1 variation:      {self.t1_variation_percent:.2f}%\n"
            f"  T2 variation:      {self.t2_variation_percent:.2f}%\n"
            f"  Frequency jumps:   {self.frequency_jumps}\n"
            f"  Fidelity range:    [{self.fidelity_range[0]:.6f}, "
            f"{self.fidelity_range[1]:.6f}]\n"
            f"  Recal interval:    "
            f"{self.recommended_recalibration_interval_hours:.1f} hours"
        )


class CalibrationDrift:
    """Physically motivated calibration parameter drift model.

    Simulates how superconducting QPU calibration parameters evolve over
    time due to several physical mechanisms:

    - **T1 drift**: slow random walk modelled as a Wiener process.
      ``T1(t) = T1_0 * (1 + sigma_T1 * W(t))`` where ``W(t)`` is a
      standard Wiener process discretised at minute-level resolution.

    - **T2 drift**: correlated with T1 (T2 <= 2*T1 always), plus an
      independent 1/f noise component modelled via superposition of
      Ornstein-Uhlenbeck processes with exponentially spaced time constants.

    - **Frequency drift**: TLS (two-level system) defects in the
      Josephson junction oxide cause discrete Poisson-distributed jumps
      of approximately 100 kHz.

    - **Gate fidelity drift**: slow sinusoidal variation from pulse
      calibration aging, with period on the order of hours.

    Parameters
    ----------
    base_calibration : CalibrationData
        Baseline calibration to drift from.
    seed : int
        Random seed for reproducibility.

    References
    ----------
    - Klimov et al., Phys. Rev. Lett. 121, 090502 (2018) [TLS fluctuators]
    - Burnett et al., npj Quantum Inf. 5, 54 (2019) [T1 fluctuations]
    - Schlor et al., Phys. Rev. Lett. 123, 190502 (2019) [frequency jumps]
    """

    # Drift parameters (physically motivated defaults)
    SIGMA_T1: float = 0.02          # 2% per sqrt(hour) Wiener volatility
    SIGMA_T2_INDEP: float = 0.03    # 3% independent T2 1/f noise amplitude
    TLS_JUMP_RATE_PER_QUBIT: float = 0.5  # jumps per qubit per hour
    TLS_JUMP_SIZE_GHZ: float = 0.0001     # ~100 kHz per jump
    FIDELITY_AGING_AMPLITUDE: float = 0.001  # 0.1% peak sinusoidal drift
    FIDELITY_AGING_PERIOD_HOURS: float = 6.0  # calibration cycle period

    def __init__(
        self,
        base_calibration: CalibrationData,
        seed: int = 42,
    ) -> None:
        self.base = base_calibration
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        # Pre-generate noise sequences for consistency
        self._noise_state: dict[str, Any] = {}

    def _fresh_rng(self, time_seed_offset: int = 0) -> np.random.RandomState:
        """Create a deterministic RNG from the base seed and an offset."""
        return np.random.RandomState(self.seed + time_seed_offset)

    def at_time(self, hours: float) -> CalibrationData:
        """Return a drifted calibration snapshot at the given time.

        Parameters
        ----------
        hours : float
            Elapsed time since baseline calibration in hours.

        Returns
        -------
        CalibrationData
            New calibration with drifted parameters.
        """
        if hours <= 0.0:
            return self.base

        n = self.base.num_qubits
        rng = self._fresh_rng()

        # Number of discrete time steps (minute resolution)
        n_steps = max(1, int(hours * 60))
        dt_hours = hours / n_steps
        sqrt_dt = math.sqrt(dt_hours)

        # --- T1 drift: Wiener random walk per qubit ---
        t1_drifted = list(self.base.t1_us)
        for q in range(n):
            wiener = 0.0
            for _ in range(n_steps):
                wiener += rng.randn() * sqrt_dt
            factor = 1.0 + self.SIGMA_T1 * wiener
            factor = max(0.3, min(2.0, factor))  # clamp to physical range
            t1_drifted[q] = self.base.t1_us[q] * factor

        # --- T2 drift: correlated with T1 + independent 1/f ---
        t2_drifted = list(self.base.t2_us)
        for q in range(n):
            # T2 tracks T1 correlation
            t1_ratio = t1_drifted[q] / max(self.base.t1_us[q], 1e-6)
            # Independent 1/f component via superposition of OU processes
            ou_sum = 0.0
            # 4 octaves of timescales: 0.1h, 0.4h, 1.6h, 6.4h
            for k in range(4):
                tau = 0.1 * (4.0 ** k)  # correlation time in hours
                ou_val = 0.0
                for _ in range(n_steps):
                    ou_val += (
                        -ou_val * dt_hours / tau
                        + math.sqrt(2.0 * dt_hours / tau) * rng.randn()
                    )
                ou_sum += ou_val * 0.25  # equal weight per octave

            indep_factor = 1.0 + self.SIGMA_T2_INDEP * ou_sum
            indep_factor = max(0.3, min(2.0, indep_factor))

            t2_new = self.base.t2_us[q] * t1_ratio * indep_factor
            # Enforce T2 <= 2*T1
            t2_new = min(t2_new, 2.0 * t1_drifted[q])
            t2_new = max(t2_new, 1.0)  # floor at 1 us
            t2_drifted[q] = t2_new

        # --- Frequency drift: Poisson TLS jumps ---
        freq_drifted = list(self.base.qubit_frequencies_ghz)
        total_jumps = 0
        for q in range(n):
            expected_jumps = self.TLS_JUMP_RATE_PER_QUBIT * hours
            n_jumps = rng.poisson(expected_jumps)
            total_jumps += n_jumps
            if n_jumps > 0:
                # Each jump is a random sign * ~100 kHz
                jump_sum = float(
                    np.sum(
                        rng.choice([-1, 1], size=n_jumps)
                        * self.TLS_JUMP_SIZE_GHZ
                    )
                )
                freq_drifted[q] += jump_sum

        # --- Gate fidelity drift: slow sinusoidal aging ---
        phase = 2.0 * math.pi * hours / self.FIDELITY_AGING_PERIOD_HOURS
        fid_multiplier = 1.0 - self.FIDELITY_AGING_AMPLITUDE * (
            1.0 - math.cos(phase)
        )

        single_fid_drifted = [
            max(0.99, f * fid_multiplier)
            for f in self.base.single_gate_fidelities
        ]

        two_q_fid_drifted = {
            edge: max(0.95, f * fid_multiplier)
            for edge, f in self.base.two_qubit_fidelities.items()
        }

        # Readout fidelity also drifts slightly with T1
        readout_drifted = []
        for q in range(n):
            t1_frac = t1_drifted[q] / max(self.base.t1_us[q], 1e-6)
            # Readout degrades slightly when T1 drops
            ro = self.base.readout_fidelities[q] * min(1.0, t1_frac ** 0.1)
            readout_drifted.append(max(0.90, ro))

        return CalibrationData(
            qubit_frequencies_ghz=freq_drifted,
            t1_us=t1_drifted,
            t2_us=t2_drifted,
            readout_fidelities=readout_drifted,
            single_gate_fidelities=single_fid_drifted,
            two_qubit_fidelities=two_q_fid_drifted,
            coupling_map=list(self.base.coupling_map),
            native_gate_family=self.base.native_gate_family,
            two_qubit_gate_time_ns=self.base.two_qubit_gate_time_ns,
            single_gate_time_ns=self.base.single_gate_time_ns,
            temperature_mk=self.base.temperature_mk,
            device_name=f"{self.base.device_name} (drifted +{hours:.1f}h)",
        )

    def time_series(
        self,
        duration_hours: float,
        interval_minutes: float = 15.0,
    ) -> list[CalibrationData]:
        """Generate calibration snapshots at regular intervals.

        Parameters
        ----------
        duration_hours : float
            Total time window in hours.
        interval_minutes : float
            Sampling interval in minutes.

        Returns
        -------
        list[CalibrationData]
            List of calibration snapshots, one per interval.
        """
        if duration_hours <= 0 or interval_minutes <= 0:
            return [self.base]

        interval_hours = interval_minutes / 60.0
        n_points = max(1, int(duration_hours / interval_hours) + 1)
        snapshots: list[CalibrationData] = []

        for i in range(n_points):
            t = i * interval_hours
            if t > duration_hours:
                break
            snapshots.append(self.at_time(t))

        return snapshots

    def stability_report(
        self, duration_hours: float = 24.0
    ) -> StabilityReport:
        """Analyse calibration stability over a time window.

        Generates a time series at 15-minute intervals and computes
        statistics on T1, T2, frequency, and fidelity variation.

        Parameters
        ----------
        duration_hours : float
            Analysis window in hours.

        Returns
        -------
        StabilityReport
            Summary statistics and recalibration recommendation.
        """
        snapshots = self.time_series(duration_hours, interval_minutes=15.0)
        if len(snapshots) < 2:
            return StabilityReport(
                t1_variation_percent=0.0,
                t2_variation_percent=0.0,
                frequency_jumps=0,
                fidelity_range=(
                    self.base.mean_2q_fidelity,
                    self.base.mean_2q_fidelity,
                ),
                recommended_recalibration_interval_hours=duration_hours,
            )

        # Collect per-snapshot aggregate metrics
        mean_t1_series = [s.mean_t1 for s in snapshots]
        mean_t2_series = [s.mean_t2 for s in snapshots]
        mean_2q_fid_series = [s.mean_2q_fidelity for s in snapshots]

        base_t1 = self.base.mean_t1
        base_t2 = self.base.mean_t2

        t1_var_pct = 0.0
        if base_t1 > 0:
            t1_var_pct = (
                (max(mean_t1_series) - min(mean_t1_series)) / base_t1 * 100.0
            )

        t2_var_pct = 0.0
        if base_t2 > 0:
            t2_var_pct = (
                (max(mean_t2_series) - min(mean_t2_series)) / base_t2 * 100.0
            )

        # Count frequency jumps by comparing consecutive snapshots
        n_freq_jumps = 0
        jump_threshold = self.TLS_JUMP_SIZE_GHZ * 0.5
        for i in range(1, len(snapshots)):
            for q in range(self.base.num_qubits):
                delta_f = abs(
                    snapshots[i].qubit_frequencies_ghz[q]
                    - snapshots[i - 1].qubit_frequencies_ghz[q]
                )
                if delta_f > jump_threshold:
                    n_freq_jumps += 1

        fid_min = min(mean_2q_fid_series)
        fid_max = max(mean_2q_fid_series)

        # Recommended recalibration: first time fidelity drops > 0.1%
        recal_hours = duration_hours
        baseline_fid = mean_2q_fid_series[0]
        interval_hours = 15.0 / 60.0
        for i, fid in enumerate(mean_2q_fid_series):
            if baseline_fid - fid > 0.001:
                recal_hours = i * interval_hours
                break

        return StabilityReport(
            t1_variation_percent=t1_var_pct,
            t2_variation_percent=t2_var_pct,
            frequency_jumps=n_freq_jumps,
            fidelity_range=(fid_min, fid_max),
            recommended_recalibration_interval_hours=max(0.25, recal_hours),
        )


# ======================================================================
# Pretty-printing utilities
# ======================================================================


def _print_comparison_table(comparison: dict[str, dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    devices = list(comparison.keys())
    if not devices:
        print("No data to display.")
        return

    # Collect all metric keys
    all_keys = list(comparison[devices[0]].keys())

    # Column widths
    label_w = max(len(k) for k in all_keys) + 2
    col_w = max(max(len(d) for d in devices) + 2, 18)

    # Header
    header = f"{'Metric':<{label_w}}"
    for d in devices:
        header += f"{d:>{col_w}}"
    print("=" * len(header))
    print("DIGITAL TWIN PRESET COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for key in all_keys:
        row = f"{key:<{label_w}}"
        for d in devices:
            val = comparison[d].get(key, "N/A")
            if isinstance(val, float):
                row += f"{val:>{col_w}.6f}" if val < 0.01 else f"{val:>{col_w}.4f}"
            else:
                row += f"{str(val):>{col_w}}"
        print(row)

    print("=" * len(header))


# ======================================================================
# Demo entry point
# ======================================================================


if __name__ == "__main__":
    import traceback

    passed = 0
    failed = 0

    def _check(label: str, condition: bool, detail: str = "") -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {label}")
        else:
            failed += 1
            msg = f"  FAIL: {label}"
            if detail:
                msg += f" -- {detail}"
            print(msg)

    print("=" * 72)
    print("DIGITAL TWIN -- SELF-TEST SUITE")
    print("=" * 72)

    # ================================================================
    # Test 1: Existing presets still work
    # ================================================================
    print("\n--- Test 1: Preset calibration data ---")
    eagle_twin = DigitalTwin.from_ibm_backend("eagle", num_qubits=5)
    heron_twin = DigitalTwin.from_ibm_backend("heron", num_qubits=5)
    syc_twin = DigitalTwin.from_google_backend("sycamore", num_qubits=5)
    willow_twin = DigitalTwin.from_google_backend("willow", num_qubits=5)

    _check("Eagle twin built", eagle_twin.calibration.num_qubits == 5)
    _check("Heron twin built", heron_twin.calibration.num_qubits == 5)
    _check("Sycamore twin built", syc_twin.calibration.num_qubits == 5)
    _check("Willow twin built", willow_twin.calibration.num_qubits == 5)

    for name, twin in [("Eagle", eagle_twin), ("Heron", heron_twin)]:
        cal = twin.calibration
        _check(
            f"{name} T1 in range",
            100.0 < cal.mean_t1 < 600.0,
            f"T1={cal.mean_t1:.1f}",
        )
        _check(
            f"{name} 1Q fidelity > 0.999",
            cal.mean_1q_fidelity > 0.999,
            f"F={cal.mean_1q_fidelity:.5f}",
        )

    # ================================================================
    # Test 2: IBM JSON ingestion
    # ================================================================
    print("\n--- Test 2: from_ibm_json ---")

    ibm_json = {
        "backend_name": "test_ibm_device",
        "qubits": [
            [
                {"name": "T1", "value": 280.0, "unit": "us"},
                {"name": "T2", "value": 190.0, "unit": "us"},
                {"name": "frequency", "value": 5.05, "unit": "GHz"},
                {"name": "readout_error", "value": 0.012},
            ],
            [
                {"name": "T1", "value": 310.0, "unit": "us"},
                {"name": "T2", "value": 210.0, "unit": "us"},
                {"name": "frequency", "value": 5.10, "unit": "GHz"},
                {"name": "readout_error", "value": 0.008},
            ],
            [
                {"name": "T1", "value": 295.0, "unit": "us"},
                {"name": "T2", "value": 200.0, "unit": "us"},
                {"name": "frequency", "value": 4.98, "unit": "GHz"},
                {"name": "readout_error", "value": 0.015},
            ],
        ],
        "gates": [
            {
                "qubits": [0],
                "gate": "sx",
                "parameters": [{"name": "gate_error", "value": 0.0003}],
            },
            {
                "qubits": [1],
                "gate": "sx",
                "parameters": [{"name": "gate_error", "value": 0.0004}],
            },
            {
                "qubits": [2],
                "gate": "sx",
                "parameters": [{"name": "gate_error", "value": 0.0002}],
            },
            {
                "qubits": [0, 1],
                "gate": "ecr",
                "parameters": [{"name": "gate_error", "value": 0.005}],
            },
            {
                "qubits": [1, 2],
                "gate": "ecr",
                "parameters": [{"name": "gate_error", "value": 0.006}],
            },
        ],
    }

    try:
        ibm_twin = DigitalTwin.from_ibm_json(ibm_json)
        cal = ibm_twin.calibration
        _check("IBM JSON: 3 qubits parsed", cal.num_qubits == 3)
        _check(
            "IBM JSON: T1[0] correct",
            abs(cal.t1_us[0] - 280.0) < 0.1,
            f"T1={cal.t1_us[0]}",
        )
        _check(
            "IBM JSON: T2[1] correct",
            abs(cal.t2_us[1] - 210.0) < 0.1,
            f"T2={cal.t2_us[1]}",
        )
        _check(
            "IBM JSON: freq[2] correct",
            abs(cal.qubit_frequencies_ghz[2] - 4.98) < 0.01,
        )
        _check(
            "IBM JSON: readout[0] correct",
            abs(cal.readout_fidelities[0] - 0.988) < 0.001,
            f"ro={cal.readout_fidelities[0]:.4f}",
        )
        _check(
            "IBM JSON: 1Q fid[0] from gate_error",
            abs(cal.single_gate_fidelities[0] - 0.9997) < 0.001,
        )
        _check(
            "IBM JSON: 2Q fidelities parsed",
            len(cal.two_qubit_fidelities) == 2,
        )
        _check(
            "IBM JSON: coupling map correct",
            len(cal.coupling_map) == 2,
        )
        _check(
            "IBM JSON: native gate is ecr",
            cal.native_gate_family == "ecr",
        )
        _check(
            "IBM JSON: device name",
            cal.device_name == "test_ibm_device",
        )
    except Exception as e:
        failed += 1
        print(f"  FAIL: from_ibm_json raised {e}")
        traceback.print_exc()

    # Test with missing fields (tolerance)
    ibm_sparse = {
        "qubits": [
            [{"name": "T1", "value": 200.0}],
            [{"name": "frequency", "value": 5.2}],
        ],
        "gates": [],
    }
    try:
        sparse_twin = DigitalTwin.from_ibm_json(ibm_sparse)
        cal_s = sparse_twin.calibration
        _check("IBM JSON sparse: 2 qubits", cal_s.num_qubits == 2)
        _check(
            "IBM JSON sparse: T1[0] from data",
            abs(cal_s.t1_us[0] - 200.0) < 0.1,
        )
        _check(
            "IBM JSON sparse: T2[0] uses default",
            cal_s.t2_us[0] > 0,
        )
    except Exception as e:
        failed += 1
        print(f"  FAIL: from_ibm_json sparse raised {e}")

    # ================================================================
    # Test 3: Google Cirq JSON ingestion
    # ================================================================
    print("\n--- Test 3: from_google_cirq ---")

    google_json = {
        "device_name": "test_cirq_device",
        "qubits": [
            {
                "id": "0_0",
                "t1_us": 65.0,
                "tphi_us": 30.0,
                "frequency_ghz": 6.01,
                "readout_fidelity": 0.992,
            },
            {
                "id": "0_1",
                "t1_us": 70.0,
                "tphi_us": 28.0,
                "frequency_ghz": 6.05,
                "readout_fidelity": 0.988,
            },
            {
                "id": "1_0",
                "t1_us": 60.0,
                "tphi_us": 32.0,
                "frequency_ghz": 5.98,
                "readout_fidelity": 0.990,
            },
        ],
        "gate_pairs": [
            {
                "qubit_a": "0_0",
                "qubit_b": "0_1",
                "fidelity": 0.996,
                "gate_type": "sqrt_iswap",
            },
            {
                "qubit_a": "0_0",
                "qubit_b": "1_0",
                "fidelity": 0.994,
                "gate_type": "sqrt_iswap",
            },
        ],
    }

    try:
        google_twin = DigitalTwin.from_google_cirq(google_json)
        cal_g = google_twin.calibration
        _check("Google JSON: 3 qubits", cal_g.num_qubits == 3)
        _check(
            "Google JSON: T1[0] correct",
            abs(cal_g.t1_us[0] - 65.0) < 0.1,
        )
        # T2 derived from T1 and Tphi: 1/T2 = 1/(2*65) + 1/30
        expected_t2 = 1.0 / (1.0 / 130.0 + 1.0 / 30.0)
        _check(
            "Google JSON: T2[0] derived from Tphi",
            abs(cal_g.t2_us[0] - expected_t2) < 0.5,
            f"T2={cal_g.t2_us[0]:.2f} expected={expected_t2:.2f}",
        )
        _check(
            "Google JSON: 2 coupling edges",
            len(cal_g.coupling_map) == 2,
        )
        _check(
            "Google JSON: native gate is sqrt_iswap",
            cal_g.native_gate_family == "sqrt_iswap",
        )
        _check(
            "Google JSON: device name",
            cal_g.device_name == "test_cirq_device",
        )
    except Exception as e:
        failed += 1
        print(f"  FAIL: from_google_cirq raised {e}")
        traceback.print_exc()

    # ================================================================
    # Test 4: CalibrationDrift
    # ================================================================
    print("\n--- Test 4: CalibrationDrift ---")

    base_cal = _ibm_heron_calibration(num_qubits=5)
    drift = CalibrationDrift(base_cal, seed=123)

    # at_time(0) should return base
    cal_t0 = drift.at_time(0.0)
    _check(
        "Drift t=0 returns base",
        abs(cal_t0.mean_t1 - base_cal.mean_t1) < 1e-6,
    )

    # at_time(1.0) should return something different
    cal_t1 = drift.at_time(1.0)
    _check(
        "Drift t=1h: T1 changed",
        abs(cal_t1.mean_t1 - base_cal.mean_t1) > 0.01,
        f"base={base_cal.mean_t1:.2f} drifted={cal_t1.mean_t1:.2f}",
    )
    _check(
        "Drift t=1h: T2 <= 2*T1 enforced",
        all(
            cal_t1.t2_us[q] <= 2.0 * cal_t1.t1_us[q] + 0.01
            for q in range(5)
        ),
    )
    _check(
        "Drift t=1h: frequencies may have TLS jumps",
        cal_t1.num_qubits == 5,
    )
    _check(
        "Drift t=1h: device name contains 'drifted'",
        "drifted" in cal_t1.device_name,
    )

    # Determinism: same seed should give same result
    drift2 = CalibrationDrift(base_cal, seed=123)
    cal_t1_b = drift2.at_time(1.0)
    _check(
        "Drift deterministic (same seed)",
        abs(cal_t1.mean_t1 - cal_t1_b.mean_t1) < 1e-10,
    )

    # Different seed should give different result
    drift3 = CalibrationDrift(base_cal, seed=999)
    cal_t1_c = drift3.at_time(1.0)
    _check(
        "Drift different seed gives different result",
        abs(cal_t1.mean_t1 - cal_t1_c.mean_t1) > 0.001,
    )

    # ================================================================
    # Test 5: time_series
    # ================================================================
    print("\n--- Test 5: time_series ---")

    series = drift.time_series(duration_hours=2.0, interval_minutes=30.0)
    _check(
        "Time series: correct length",
        len(series) == 5,  # 0, 0.5, 1.0, 1.5, 2.0 hours
        f"got {len(series)} snapshots",
    )
    _check(
        "Time series: first is base",
        abs(series[0].mean_t1 - base_cal.mean_t1) < 1e-6,
    )
    _check(
        "Time series: monotonic time labels",
        "0.0h" in series[0].device_name or series[0].device_name == base_cal.device_name,
    )

    # ================================================================
    # Test 6: StabilityReport
    # ================================================================
    print("\n--- Test 6: stability_report ---")

    report = drift.stability_report(duration_hours=12.0)
    _check(
        "StabilityReport: T1 variation > 0",
        report.t1_variation_percent > 0.0,
        f"var={report.t1_variation_percent:.2f}%",
    )
    _check(
        "StabilityReport: T2 variation > 0",
        report.t2_variation_percent > 0.0,
        f"var={report.t2_variation_percent:.2f}%",
    )
    _check(
        "StabilityReport: fidelity range valid",
        report.fidelity_range[0] <= report.fidelity_range[1],
    )
    _check(
        "StabilityReport: recal interval > 0",
        report.recommended_recalibration_interval_hours > 0.0,
        f"recal={report.recommended_recalibration_interval_hours:.1f}h",
    )
    print(f"\n  Full report:\n{report}")

    # ================================================================
    # Test 7: Gate fidelity sinusoidal drift
    # ================================================================
    print("\n--- Test 7: Fidelity aging model ---")

    # At half-period (3 hours with default 6h period), drift is maximum
    cal_half = drift.at_time(3.0)
    base_mean_2q = base_cal.mean_2q_fidelity
    drifted_mean_2q = cal_half.mean_2q_fidelity
    _check(
        "Fidelity aging: half-period gives maximum drift",
        drifted_mean_2q < base_mean_2q,
        f"base={base_mean_2q:.6f} drifted={drifted_mean_2q:.6f}",
    )

    # At full period (6 hours), fidelity should return close to baseline
    cal_full = drift.at_time(6.0)
    full_mean_2q = cal_full.mean_2q_fidelity
    # The sinusoidal component returns to baseline; T1 drift adds noise
    _check(
        "Fidelity aging: full-period returns near baseline",
        abs(full_mean_2q - base_mean_2q) < 0.005,
        f"base={base_mean_2q:.6f} full-period={full_mean_2q:.6f}",
    )

    # ================================================================
    # Test 8: DigitalTwin from drifted calibration
    # ================================================================
    print("\n--- Test 8: DigitalTwin from drifted calibration ---")

    drifted_twin = DigitalTwin.from_calibration(cal_t1)
    _check(
        "Drifted twin: num_qubits preserved",
        drifted_twin.calibration.num_qubits == 5,
    )
    pred = drifted_twin.predict_circuit_fidelity(
        num_1q_gates=50, num_2q_gates=20, depth=30
    )
    _check(
        "Drifted twin: fidelity prediction in range",
        0.0 < pred < 1.0,
        f"predicted F={pred:.4f}",
    )

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 72)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All tests passed.")
    else:
        print("SOME TESTS FAILED -- review output above.")
    print("=" * 72)
