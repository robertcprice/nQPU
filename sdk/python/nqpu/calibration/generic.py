"""Hardware-agnostic calibration dataclass and auto-detection.

Provides ``GenericCalibration``, a technology-neutral representation of
any quantum processor's calibration data. Converters from IBM transmon,
Quantinuum trapped-ion, and QuEra neutral-atom configs are included so
that the rest of nQPU's toolchain (transpiler, noise-aware compiler,
error mitigation) can work uniformly regardless of the backend vendor.

Also includes ``auto_detect_format`` for sniffing raw JSON and
``load_calibration`` for one-call ingestion of any supported format.

Example
-------
>>> from nqpu.calibration import load_calibration, compare_devices
>>> data = {...}  # JSON from any supported vendor
>>> cal = load_calibration(data)
>>> print(cal.summary())

References
----------
- nQPU transpiler and noise-aware routing documentation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from .ibm import TransmonProcessor
    from .quantinuum import TrapConfig
    from .quera import NeutralAtomConfig


# ---------------------------------------------------------------------------
# Generic calibration
# ---------------------------------------------------------------------------

@dataclass
class GenericCalibration:
    """Hardware-agnostic calibration data.

    Provides a uniform interface for qubit quality, gate errors, and
    connectivity regardless of the underlying hardware technology.
    """

    name: str
    technology: str  # "superconducting", "trapped_ion", "neutral_atom", "photonic"
    n_qubits: int
    qubit_t1: Dict[int, float] = field(default_factory=dict)
    qubit_t2: Dict[int, float] = field(default_factory=dict)
    single_qubit_errors: Dict[int, float] = field(default_factory=dict)
    two_qubit_errors: Dict[Tuple[int, int], float] = field(default_factory=dict)
    readout_errors: Dict[int, float] = field(default_factory=dict)
    coupling_map: List[Tuple[int, int]] = field(default_factory=list)
    gate_times: Dict[str, float] = field(default_factory=dict)
    basis_gates: List[str] = field(default_factory=list)
    timestamp: str = ""

    # -- Converters ---------------------------------------------------------

    @staticmethod
    def from_transmon(proc: TransmonProcessor) -> GenericCalibration:
        """Convert an IBM ``TransmonProcessor`` to ``GenericCalibration``."""
        qubit_t1: Dict[int, float] = {}
        qubit_t2: Dict[int, float] = {}
        single_q_err: Dict[int, float] = {}
        readout_err: Dict[int, float] = {}

        for idx, q in proc.qubits.items():
            qubit_t1[idx] = q.t1
            qubit_t2[idx] = q.t2
            readout_err[idx] = q.readout_error

        # Extract per-qubit single-qubit error (average across 1Q gates)
        per_qubit_1q: Dict[int, List[float]] = {}
        two_q_err: Dict[Tuple[int, int], float] = {}
        gate_times: Dict[str, float] = {}

        for g in proc.gates:
            if len(g.qubits) == 1:
                per_qubit_1q.setdefault(g.qubits[0], []).append(g.error_rate)
                gate_times.setdefault(f"{g.gate_type}_1q", g.gate_length)
            elif len(g.qubits) == 2:
                two_q_err[g.qubits] = g.error_rate
                gate_times.setdefault(f"{g.gate_type}_2q", g.gate_length)

        for idx, errs in per_qubit_1q.items():
            single_q_err[idx] = float(np.mean(errs))

        return GenericCalibration(
            name=proc.name,
            technology="superconducting",
            n_qubits=proc.n_qubits,
            qubit_t1=qubit_t1,
            qubit_t2=qubit_t2,
            single_qubit_errors=single_q_err,
            two_qubit_errors=two_q_err,
            readout_errors=readout_err,
            coupling_map=list(proc.coupling_map),
            gate_times=gate_times,
            basis_gates=list(proc.basis_gates),
            timestamp=proc.timestamp,
        )

    @staticmethod
    def from_trap(config: TrapConfig) -> GenericCalibration:
        """Convert a Quantinuum ``TrapConfig`` to ``GenericCalibration``."""
        qubit_t1: Dict[int, float] = {}
        qubit_t2: Dict[int, float] = {}
        single_q_err: Dict[int, float] = {}
        readout_err: Dict[int, float] = {}

        for i in range(config.n_qubits):
            qubit_t1[i] = config.t1
            qubit_t2[i] = config.t2
            single_q_err[i] = config.single_qubit_error
            readout_err[i] = config.measurement_error

        # All-to-all connectivity
        coupling: List[Tuple[int, int]] = []
        two_q_err: Dict[Tuple[int, int], float] = {}
        if config.all_to_all:
            for i in range(config.n_qubits):
                for j in range(i + 1, config.n_qubits):
                    coupling.append((i, j))
                    two_q_err[(i, j)] = config.two_qubit_error

        return GenericCalibration(
            name=config.name,
            technology="trapped_ion",
            n_qubits=config.n_qubits,
            qubit_t1=qubit_t1,
            qubit_t2=qubit_t2,
            single_qubit_errors=single_q_err,
            two_qubit_errors=two_q_err,
            readout_errors=readout_err,
            coupling_map=coupling,
            gate_times={
                "1q": config.gate_time_1q,
                "2q": config.gate_time_2q,
                "transport": config.transport_time,
            },
            basis_gates=["rz", "ry", "rx", "zz"],
        )

    @staticmethod
    def from_neutral_atom(config: NeutralAtomConfig) -> GenericCalibration:
        """Convert a QuEra ``NeutralAtomConfig`` to ``GenericCalibration``."""
        n_occ = config.n_occupied
        qubit_t1: Dict[int, float] = {}
        qubit_t2: Dict[int, float] = {}
        single_q_err: Dict[int, float] = {}
        readout_err: Dict[int, float] = {}

        occupied_indices = [
            s.index for s in config.sites if s.occupied
        ]
        for idx in occupied_indices:
            qubit_t1[idx] = config.t1
            qubit_t2[idx] = config.t1 * 0.5  # approximate T2 ~ T1/2
            single_q_err[idx] = 1.0 - config.rydberg_fidelity
            readout_err[idx] = 1.0 - config.measurement_fidelity

        # Connectivity from blockade graph
        edges = config.connectivity()
        two_q_err: Dict[Tuple[int, int], float] = {}
        for edge in edges:
            two_q_err[edge] = 1.0 - config.rydberg_fidelity

        return GenericCalibration(
            name=config.name,
            technology="neutral_atom",
            n_qubits=n_occ,
            qubit_t1=qubit_t1,
            qubit_t2=qubit_t2,
            single_qubit_errors=single_q_err,
            two_qubit_errors=two_q_err,
            readout_errors=readout_err,
            coupling_map=edges,
            gate_times={"rydberg_pulse": 1.0},
            basis_gates=["rydberg"],
        )

    # -- Quality analysis ---------------------------------------------------

    def best_qubits(self, n: int) -> List[int]:
        """Return the *n* best qubits by combined quality score.

        Scoring uses a weighted combination of T1, readout error,
        and single-qubit gate error (if available).
        """
        scores: Dict[int, float] = {}
        max_t1 = max(self.qubit_t1.values()) if self.qubit_t1 else 1.0
        for i in range(self.n_qubits):
            t1_score = self.qubit_t1.get(i, 0.0) / max_t1 if max_t1 > 0 else 0.0
            ro_score = 1.0 - self.readout_errors.get(i, 0.5)
            sq_score = 1.0 - self.single_qubit_errors.get(i, 0.5)
            scores[i] = (t1_score + ro_score + sq_score) / 3.0

        ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        return ranked[:n]

    def best_pairs(self, n: int) -> List[Tuple[int, int]]:
        """Return the *n* best qubit pairs by two-qubit error rate."""
        if not self.two_qubit_errors:
            return []
        sorted_pairs = sorted(self.two_qubit_errors.items(), key=lambda x: x[1])
        return [pair for pair, _ in sorted_pairs[:n]]

    def overall_quality(self) -> float:
        """Single quality metric in [0, 1] for the entire device.

        Combines median readout error, median two-qubit error, and
        median coherence into a composite score.
        """
        if not self.readout_errors:
            return 0.0

        ro_vals = list(self.readout_errors.values())
        median_ro = float(np.median(ro_vals))

        tq_vals = list(self.two_qubit_errors.values()) if self.two_qubit_errors else [0.5]
        median_tq = float(np.median(tq_vals))

        # Combine: lower errors = higher quality
        ro_score = 1.0 - median_ro
        tq_score = 1.0 - median_tq
        return (ro_score + tq_score) / 2.0

    def summary(self) -> str:
        """Human-readable summary of the generic calibration."""
        lines = [
            f"GenericCalibration: {self.name}",
            f"  Technology:          {self.technology}",
            f"  Qubits:             {self.n_qubits}",
            f"  Coupling edges:     {len(self.coupling_map)}",
            f"  Basis gates:        {', '.join(self.basis_gates) if self.basis_gates else 'N/A'}",
            f"  Overall quality:    {self.overall_quality():.4f}",
        ]
        if self.qubit_t1:
            lines.append(
                f"  Median T1:          {float(np.median(list(self.qubit_t1.values()))):.1f} us"
            )
        if self.readout_errors:
            lines.append(
                f"  Median readout err: {float(np.median(list(self.readout_errors.values()))):.5f}"
            )
        if self.two_qubit_errors:
            lines.append(
                f"  Median 2Q error:    {float(np.median(list(self.two_qubit_errors.values()))):.5f}"
            )
        if self.timestamp:
            lines.append(f"  Timestamp:          {self.timestamp}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-detection and loading
# ---------------------------------------------------------------------------

def auto_detect_format(data: dict) -> str:
    """Auto-detect calibration data format.

    Parameters
    ----------
    data : dict
        Raw calibration JSON dictionary.

    Returns
    -------
    str
        One of ``"ibm_v1"``, ``"ibm_v2"``, ``"quantinuum"``, ``"quera"``,
        or ``"unknown"``.
    """
    # IBM v1: has "backend_name" and list-of-lists "qubits"
    if "backend_name" in data and "qubits" in data:
        qubits = data["qubits"]
        if isinstance(qubits, list) and qubits and isinstance(qubits[0], list):
            return "ibm_v1"
        return "ibm_v2"

    # Quantinuum: has fidelity keys or "zones"
    if "single_qubit_fidelity" in data or "two_qubit_fidelity" in data or "zones" in data:
        return "quantinuum"

    # QuEra: has "rydberg_range" or "atom_loading_fidelity" or "max_atoms"
    if "rydberg_range" in data or "atom_loading_fidelity" in data or "max_atoms" in data:
        return "quera"

    return "unknown"


def load_calibration(data: dict) -> GenericCalibration:
    """Load calibration data from any supported format.

    Auto-detects the vendor format and converts to ``GenericCalibration``.

    Parameters
    ----------
    data : dict
        Raw calibration JSON dictionary.

    Returns
    -------
    GenericCalibration

    Raises
    ------
    ValueError
        If the format cannot be detected.
    """
    from .ibm import parse_ibm_properties, parse_ibm_v2
    from .quantinuum import parse_quantinuum_specs
    from .quera import parse_quera_capabilities

    fmt = auto_detect_format(data)

    if fmt == "ibm_v1":
        proc = parse_ibm_properties(data)
        return GenericCalibration.from_transmon(proc)
    elif fmt == "ibm_v2":
        proc = parse_ibm_v2(data)
        return GenericCalibration.from_transmon(proc)
    elif fmt == "quantinuum":
        config = parse_quantinuum_specs(data)
        return GenericCalibration.from_trap(config)
    elif fmt == "quera":
        config = parse_quera_capabilities(data)
        return GenericCalibration.from_neutral_atom(config)
    else:
        raise ValueError(
            f"Cannot auto-detect calibration format. "
            f"Expected IBM, Quantinuum, or QuEra data. "
            f"Top-level keys: {list(data.keys())[:10]}"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ideal_calibration(n_qubits: int) -> GenericCalibration:
    """Create a perfect (noiseless) calibration for testing.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    GenericCalibration
        A calibration with zero errors and infinite coherence.
    """
    qubit_t1 = {i: 1e9 for i in range(n_qubits)}
    qubit_t2 = {i: 1e9 for i in range(n_qubits)}
    single_q_err = {i: 0.0 for i in range(n_qubits)}
    readout_err = {i: 0.0 for i in range(n_qubits)}

    # Linear coupling
    coupling: List[Tuple[int, int]] = []
    two_q_err: Dict[Tuple[int, int], float] = {}
    for i in range(n_qubits - 1):
        coupling.append((i, i + 1))
        two_q_err[(i, i + 1)] = 0.0

    return GenericCalibration(
        name="ideal",
        technology="ideal",
        n_qubits=n_qubits,
        qubit_t1=qubit_t1,
        qubit_t2=qubit_t2,
        single_qubit_errors=single_q_err,
        two_qubit_errors=two_q_err,
        readout_errors=readout_err,
        coupling_map=coupling,
        gate_times={"1q": 0.0, "2q": 0.0},
        basis_gates=["u3", "cx"],
    )


def compare_devices(devices: List[GenericCalibration]) -> str:
    """Generate a comparison table of multiple devices.

    Parameters
    ----------
    devices : list of GenericCalibration
        Devices to compare.

    Returns
    -------
    str
        Formatted comparison table.
    """
    if not devices:
        return "No devices to compare."

    # Header
    col_width = 20
    header = f"{'Metric':<25}"
    for d in devices:
        header += f"{d.name[:col_width]:>{col_width}}"

    separator = "-" * len(header)
    lines = [header, separator]

    # Rows
    def _row(label: str, values: List[str]) -> str:
        row = f"{label:<25}"
        for v in values:
            row += f"{v:>{col_width}}"
        return row

    lines.append(_row("Technology", [d.technology for d in devices]))
    lines.append(_row("Qubits", [str(d.n_qubits) for d in devices]))
    lines.append(_row("Coupling edges", [str(len(d.coupling_map)) for d in devices]))

    lines.append(
        _row(
            "Median T1 (us)",
            [
                f"{float(np.median(list(d.qubit_t1.values()))):.1f}"
                if d.qubit_t1
                else "N/A"
                for d in devices
            ],
        )
    )
    lines.append(
        _row(
            "Median readout err",
            [
                f"{float(np.median(list(d.readout_errors.values()))):.5f}"
                if d.readout_errors
                else "N/A"
                for d in devices
            ],
        )
    )
    lines.append(
        _row(
            "Median 2Q error",
            [
                f"{float(np.median(list(d.two_qubit_errors.values()))):.5f}"
                if d.two_qubit_errors
                else "N/A"
                for d in devices
            ],
        )
    )
    lines.append(
        _row("Overall quality", [f"{d.overall_quality():.4f}" for d in devices])
    )

    return "\n".join(lines)
