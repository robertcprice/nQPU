"""Export calibration data as reports, diffs, JSON, CSV.

Provides tools for generating human-readable calibration reports,
comparing calibration snapshots over time (to detect hardware
degradation), and serialising calibration data to JSON / CSV for
integration with external tooling.

Example
-------
>>> from nqpu.calibration import ibm_eagle_r3
>>> from nqpu.calibration.generic import GenericCalibration
>>> from nqpu.calibration.exporters import CalibrationReport, to_json
>>> eagle = ibm_eagle_r3()
>>> cal = GenericCalibration.from_transmon(eagle)
>>> report = CalibrationReport()
>>> print(report.generate(cal))
>>> json_str = to_json(cal)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .generic import GenericCalibration


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    """Generate human-readable calibration reports."""

    include_qubit_detail: bool = True
    include_gate_detail: bool = True
    max_qubits_shown: int = 30

    def generate(self, calibration: GenericCalibration) -> str:
        """Full calibration report as formatted string.

        Parameters
        ----------
        calibration : GenericCalibration
            The calibration data to report on.

        Returns
        -------
        str
            Multi-section report.
        """
        sections = [
            self.quality_summary(calibration),
            "",
        ]
        if self.include_qubit_detail:
            sections.append(self.qubit_table(calibration))
            sections.append("")
        if self.include_gate_detail:
            sections.append(self.gate_table(calibration))
        return "\n".join(sections)

    def qubit_table(self, calibration: GenericCalibration) -> str:
        """Format qubit calibration data as an ASCII table.

        Parameters
        ----------
        calibration : GenericCalibration

        Returns
        -------
        str
        """
        header = f"{'Qubit':>6}  {'T1 (us)':>10}  {'T2 (us)':>10}  {'RO err':>10}  {'1Q err':>10}"
        sep = "-" * len(header)
        lines = ["=== Qubit Detail ===", header, sep]

        qubit_indices = sorted(calibration.qubit_t1.keys())
        shown = qubit_indices[: self.max_qubits_shown]

        for i in shown:
            t1 = calibration.qubit_t1.get(i, 0.0)
            t2 = calibration.qubit_t2.get(i, 0.0)
            ro = calibration.readout_errors.get(i, 0.0)
            sq = calibration.single_qubit_errors.get(i, 0.0)
            lines.append(
                f"{i:>6d}  {t1:>10.1f}  {t2:>10.1f}  {ro:>10.5f}  {sq:>10.5f}"
            )

        if len(qubit_indices) > self.max_qubits_shown:
            lines.append(
                f"  ... and {len(qubit_indices) - self.max_qubits_shown} more qubits"
            )

        return "\n".join(lines)

    def gate_table(self, calibration: GenericCalibration) -> str:
        """Format two-qubit gate errors as an ASCII table.

        Parameters
        ----------
        calibration : GenericCalibration

        Returns
        -------
        str
        """
        header = f"{'Pair':>12}  {'Error':>12}"
        sep = "-" * len(header)
        lines = ["=== Two-Qubit Gate Detail ===", header, sep]

        sorted_pairs = sorted(
            calibration.two_qubit_errors.items(), key=lambda x: x[1]
        )
        shown = sorted_pairs[: self.max_qubits_shown]

        for (i, j), err in shown:
            lines.append(f"  ({i:>3d},{j:>3d})  {err:>12.6f}")

        if len(sorted_pairs) > self.max_qubits_shown:
            lines.append(
                f"  ... and {len(sorted_pairs) - self.max_qubits_shown} more pairs"
            )

        return "\n".join(lines)

    def quality_summary(self, calibration: GenericCalibration) -> str:
        """Generate a quality summary section.

        Parameters
        ----------
        calibration : GenericCalibration

        Returns
        -------
        str
        """
        lines = [
            f"=== Calibration Report: {calibration.name} ===",
            f"Technology:        {calibration.technology}",
            f"Qubits:            {calibration.n_qubits}",
            f"Coupling edges:    {len(calibration.coupling_map)}",
            f"Basis gates:       {', '.join(calibration.basis_gates) if calibration.basis_gates else 'N/A'}",
            f"Overall quality:   {calibration.overall_quality():.4f}",
        ]
        if calibration.qubit_t1:
            vals = list(calibration.qubit_t1.values())
            lines.append(
                f"T1 (us):           min={min(vals):.1f}  "
                f"median={float(np.median(vals)):.1f}  "
                f"max={max(vals):.1f}"
            )
        if calibration.qubit_t2:
            vals = list(calibration.qubit_t2.values())
            lines.append(
                f"T2 (us):           min={min(vals):.1f}  "
                f"median={float(np.median(vals)):.1f}  "
                f"max={max(vals):.1f}"
            )
        if calibration.readout_errors:
            vals = list(calibration.readout_errors.values())
            lines.append(
                f"Readout error:     min={min(vals):.5f}  "
                f"median={float(np.median(vals)):.5f}  "
                f"max={max(vals):.5f}"
            )
        if calibration.two_qubit_errors:
            vals = list(calibration.two_qubit_errors.values())
            lines.append(
                f"2Q gate error:     min={min(vals):.5f}  "
                f"median={float(np.median(vals)):.5f}  "
                f"max={max(vals):.5f}"
            )
        if calibration.timestamp:
            lines.append(f"Timestamp:         {calibration.timestamp}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calibration diff
# ---------------------------------------------------------------------------

@dataclass
class CalibrationDiff:
    """Compare two calibration snapshots."""

    def diff(
        self,
        old: GenericCalibration,
        new: GenericCalibration,
    ) -> str:
        """Show what changed between two calibration snapshots.

        Parameters
        ----------
        old : GenericCalibration
            Earlier calibration snapshot.
        new : GenericCalibration
            Later calibration snapshot.

        Returns
        -------
        str
            Human-readable diff summary.
        """
        lines = [
            f"=== Calibration Diff: {old.name} -> {new.name} ===",
            f"Qubits: {old.n_qubits} -> {new.n_qubits}",
        ]

        # T1 changes
        if old.qubit_t1 and new.qubit_t1:
            old_med = float(np.median(list(old.qubit_t1.values())))
            new_med = float(np.median(list(new.qubit_t1.values())))
            delta = new_med - old_med
            direction = "improved" if delta > 0 else "degraded"
            lines.append(
                f"Median T1: {old_med:.1f} -> {new_med:.1f} us ({direction})"
            )

        # Readout error changes
        if old.readout_errors and new.readout_errors:
            old_med = float(np.median(list(old.readout_errors.values())))
            new_med = float(np.median(list(new.readout_errors.values())))
            direction = "improved" if new_med < old_med else "degraded"
            lines.append(
                f"Median readout error: {old_med:.5f} -> {new_med:.5f} ({direction})"
            )

        # Two-qubit error changes
        if old.two_qubit_errors and new.two_qubit_errors:
            old_med = float(np.median(list(old.two_qubit_errors.values())))
            new_med = float(np.median(list(new.two_qubit_errors.values())))
            direction = "improved" if new_med < old_med else "degraded"
            lines.append(
                f"Median 2Q error: {old_med:.5f} -> {new_med:.5f} ({direction})"
            )

        # Degraded / improved qubits
        degraded = self.degraded_qubits(old, new)
        improved = self.improved_qubits(old, new)
        if degraded:
            lines.append(
                f"Degraded qubits ({len(degraded)}): {degraded[:10]}"
                + (" ..." if len(degraded) > 10 else "")
            )
        if improved:
            lines.append(
                f"Improved qubits ({len(improved)}): {improved[:10]}"
                + (" ..." if len(improved) > 10 else "")
            )

        return "\n".join(lines)

    def degraded_qubits(
        self,
        old: GenericCalibration,
        new: GenericCalibration,
        threshold: float = 0.1,
    ) -> list:
        """Find qubits whose readout error increased beyond *threshold*.

        Parameters
        ----------
        old, new : GenericCalibration
            Calibration snapshots.
        threshold : float
            Fractional increase threshold (0.1 = 10 % worse).

        Returns
        -------
        list of int
        """
        degraded: List[int] = []
        for i in old.readout_errors:
            if i in new.readout_errors:
                old_val = old.readout_errors[i]
                new_val = new.readout_errors[i]
                if old_val > 0 and (new_val - old_val) / old_val > threshold:
                    degraded.append(i)
        return sorted(degraded)

    def improved_qubits(
        self,
        old: GenericCalibration,
        new: GenericCalibration,
        threshold: float = 0.1,
    ) -> list:
        """Find qubits whose readout error decreased beyond *threshold*.

        Parameters
        ----------
        old, new : GenericCalibration
            Calibration snapshots.
        threshold : float
            Fractional decrease threshold (0.1 = 10 % better).

        Returns
        -------
        list of int
        """
        improved: List[int] = []
        for i in old.readout_errors:
            if i in new.readout_errors:
                old_val = old.readout_errors[i]
                new_val = new.readout_errors[i]
                if old_val > 0 and (old_val - new_val) / old_val > threshold:
                    improved.append(i)
        return sorted(improved)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _serialise_calibration(calibration: GenericCalibration) -> dict:
    """Convert calibration to a JSON-safe dictionary."""
    d: Dict[str, Any] = {
        "name": calibration.name,
        "technology": calibration.technology,
        "n_qubits": calibration.n_qubits,
        "qubit_t1": {str(k): v for k, v in calibration.qubit_t1.items()},
        "qubit_t2": {str(k): v for k, v in calibration.qubit_t2.items()},
        "single_qubit_errors": {
            str(k): v for k, v in calibration.single_qubit_errors.items()
        },
        "two_qubit_errors": {
            f"{i},{j}": v for (i, j), v in calibration.two_qubit_errors.items()
        },
        "readout_errors": {
            str(k): v for k, v in calibration.readout_errors.items()
        },
        "coupling_map": calibration.coupling_map,
        "gate_times": calibration.gate_times,
        "basis_gates": calibration.basis_gates,
        "timestamp": calibration.timestamp,
    }
    return d


def _deserialise_calibration(d: dict) -> GenericCalibration:
    """Reconstruct GenericCalibration from a JSON-safe dictionary."""
    qubit_t1 = {int(k): v for k, v in d.get("qubit_t1", {}).items()}
    qubit_t2 = {int(k): v for k, v in d.get("qubit_t2", {}).items()}
    single_qubit_errors = {
        int(k): v for k, v in d.get("single_qubit_errors", {}).items()
    }
    two_qubit_errors: Dict[Tuple[int, int], float] = {}
    for k, v in d.get("two_qubit_errors", {}).items():
        parts = k.split(",")
        two_qubit_errors[(int(parts[0]), int(parts[1]))] = v
    readout_errors = {int(k): v for k, v in d.get("readout_errors", {}).items()}
    coupling_map = [tuple(e) for e in d.get("coupling_map", [])]

    return GenericCalibration(
        name=d.get("name", "unknown"),
        technology=d.get("technology", "unknown"),
        n_qubits=d.get("n_qubits", 0),
        qubit_t1=qubit_t1,
        qubit_t2=qubit_t2,
        single_qubit_errors=single_qubit_errors,
        two_qubit_errors=two_qubit_errors,
        readout_errors=readout_errors,
        coupling_map=coupling_map,
        gate_times=d.get("gate_times", {}),
        basis_gates=d.get("basis_gates", []),
        timestamp=d.get("timestamp", ""),
    )


def to_json(calibration: GenericCalibration, indent: int = 2) -> str:
    """Serialise calibration data to JSON string.

    Parameters
    ----------
    calibration : GenericCalibration
    indent : int
        JSON indentation level.

    Returns
    -------
    str
        JSON string.
    """
    return json.dumps(_serialise_calibration(calibration), indent=indent)


def from_json(json_str: str) -> GenericCalibration:
    """Deserialise calibration data from JSON string.

    Parameters
    ----------
    json_str : str
        JSON string produced by ``to_json``.

    Returns
    -------
    GenericCalibration
    """
    d = json.loads(json_str)
    return _deserialise_calibration(d)


def to_csv(calibration: GenericCalibration) -> str:
    """Export per-qubit calibration data as CSV.

    Columns: ``qubit,t1_us,t2_us,readout_error,single_qubit_error``

    Parameters
    ----------
    calibration : GenericCalibration

    Returns
    -------
    str
        CSV-formatted string.
    """
    lines = ["qubit,t1_us,t2_us,readout_error,single_qubit_error"]
    qubit_indices = sorted(
        set(calibration.qubit_t1.keys())
        | set(calibration.readout_errors.keys())
    )
    for i in qubit_indices:
        t1 = calibration.qubit_t1.get(i, 0.0)
        t2 = calibration.qubit_t2.get(i, 0.0)
        ro = calibration.readout_errors.get(i, 0.0)
        sq = calibration.single_qubit_errors.get(i, 0.0)
        lines.append(f"{i},{t1:.4f},{t2:.4f},{ro:.6f},{sq:.6f}")
    return "\n".join(lines)
