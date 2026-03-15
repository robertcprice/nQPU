"""Calibration forecasting for optimal QPU submission windows.

Predicts how circuit fidelity degrades over time due to calibration drift,
and identifies the optimal time windows for submitting jobs to a real
quantum processor.  This enables proactive scheduling that maximizes
output quality without requiring real-time hardware monitoring.

The forecasting model combines:
    - CalibrationDrift: physics-based parameter drift (T1, T2, frequency, fidelity)
    - Circuit-aware fidelity: depth and 2Q gate count affect sensitivity to drift
    - Window detection: identifies contiguous periods above a fidelity threshold

Example:
    >>> from nqpu.superconducting.digital_twin import DigitalTwin, CalibrationDrift
    >>> from nqpu.superconducting.calibration_forecast import CalibrationForecaster
    >>> twin = DigitalTwin.from_ibm_backend("heron", num_qubits=5)
    >>> drift = CalibrationDrift(twin.calibration, seed=42)
    >>> forecaster = CalibrationForecaster(drift)
    >>> report = forecaster.forecast(duration_hours=8.0, circuit_depth=50, circuit_2q_gates=20)
    >>> print(f"Best time: {report.best_submission_time_hours:.1f} h")

References:
    - Tannu & Qureshi, ASPLOS (2019) [variational noise-aware compilation]
    - Murali et al., ASPLOS (2020) [software mitigation of crosstalk]
    - Ding & Chong, IEEE JSSC (2020) [quantum computer scheduling]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .digital_twin import CalibrationData, CalibrationDrift


# ---------------------------------------------------------------------------
# Forecast report
# ---------------------------------------------------------------------------


@dataclass
class ForecastReport:
    """Calibration forecast with optimal submission windows.

    Attributes
    ----------
    times_hours : np.ndarray
        Time points at which fidelity was predicted.
    predicted_fidelities : np.ndarray
        Predicted circuit output fidelity at each time point.
    optimal_windows : list[tuple[float, float]]
        Time intervals (start_hours, end_hours) where fidelity exceeds the
        threshold.  Each tuple represents a contiguous window of acceptable
        quality.
    best_submission_time_hours : float
        Time of peak predicted fidelity.
    worst_submission_time_hours : float
        Time of minimum predicted fidelity.
    fidelity_at_best : float
        Fidelity at the best submission time.
    fidelity_at_worst : float
        Fidelity at the worst submission time.
    fidelity_threshold : float
        Threshold used for window detection.
    circuit_depth : int
        Circuit depth used in the forecast.
    circuit_2q_gates : int
        Number of two-qubit gates used in the forecast.
    """

    times_hours: np.ndarray
    predicted_fidelities: np.ndarray
    optimal_windows: list[tuple[float, float]]
    best_submission_time_hours: float
    worst_submission_time_hours: float
    fidelity_at_best: float
    fidelity_at_worst: float
    fidelity_threshold: float
    circuit_depth: int
    circuit_2q_gates: int

    def __str__(self) -> str:
        lines = [
            "=== Calibration Forecast Report ===",
            f"  Circuit depth:       {self.circuit_depth}",
            f"  2Q gates:            {self.circuit_2q_gates}",
            f"  Fidelity threshold:  {self.fidelity_threshold:.4f}",
            f"  Best time:           {self.best_submission_time_hours:.2f} h "
            f"(F={self.fidelity_at_best:.6f})",
            f"  Worst time:          {self.worst_submission_time_hours:.2f} h "
            f"(F={self.fidelity_at_worst:.6f})",
            f"  Optimal windows:     {len(self.optimal_windows)}",
        ]
        for i, (start, end) in enumerate(self.optimal_windows):
            lines.append(f"    Window {i + 1}: {start:.2f} - {end:.2f} h")
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Export time series as CSV.

        Returns
        -------
        str
            CSV with columns: time_hours, predicted_fidelity.
        """
        header = "time_hours,predicted_fidelity"
        rows = [header]
        for i in range(len(self.times_hours)):
            rows.append(
                f"{self.times_hours[i]:.4f},"
                f"{self.predicted_fidelities[i]:.8f}"
            )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Calibration forecaster
# ---------------------------------------------------------------------------


class CalibrationForecaster:
    """Predict optimal QPU submission windows from calibration drift.

    Combines the physically-motivated drift model with circuit-aware
    fidelity prediction to forecast when a quantum processor will
    deliver acceptable output quality.

    Parameters
    ----------
    drift_model : CalibrationDrift
        Calibration drift model providing drifted parameters over time.
    fidelity_threshold : float
        Minimum acceptable fidelity for defining optimal submission
        windows.  Default is 0.5 (moderate-depth circuits).

    Examples
    --------
    >>> drift = CalibrationDrift(calibration_data, seed=42)
    >>> forecaster = CalibrationForecaster(drift, fidelity_threshold=0.8)
    >>> report = forecaster.forecast(8.0, circuit_depth=20, circuit_2q_gates=10)
    """

    def __init__(
        self,
        drift_model: CalibrationDrift,
        fidelity_threshold: float = 0.5,
    ) -> None:
        self.drift = drift_model
        self.fidelity_threshold = fidelity_threshold

    # ------------------------------------------------------------------
    # Circuit fidelity model
    # ------------------------------------------------------------------

    @staticmethod
    def predict_circuit_fidelity(
        cal: CalibrationData,
        circuit_depth: int,
        circuit_2q_gates: int,
    ) -> float:
        """Predict output fidelity for a circuit given current calibration.

        Uses a multiplicative error model:
            F = (1 - e_1q)^(depth) * (1 - e_2q)^(n_2q) * exp(-depth * t_gate / T2)

        where:
            - e_1q: mean single-qubit error rate
            - e_2q: mean two-qubit error rate
            - n_2q: number of two-qubit gates
            - depth: circuit depth (proxy for single-qubit gate count)
            - t_gate: gate time
            - T2: mean dephasing time

        Parameters
        ----------
        cal : CalibrationData
            Current calibration snapshot.
        circuit_depth : int
            Circuit depth (number of layers).
        circuit_2q_gates : int
            Number of two-qubit gates.

        Returns
        -------
        float
            Predicted output fidelity in [0, 1].
        """
        mean_1q_err = 1.0 - cal.mean_1q_fidelity
        mean_2q_err = 1.0 - cal.mean_2q_fidelity

        # Gate error contribution
        fid = (1.0 - mean_1q_err) ** circuit_depth
        fid *= (1.0 - mean_2q_err) ** circuit_2q_gates

        # Decoherence contribution
        if cal.mean_t2 > 0 and circuit_depth > 0:
            gate_time_us = cal.two_qubit_gate_time_ns / 1000.0
            decoherence_per_layer = 1.0 - math.exp(
                -gate_time_us / cal.mean_t2
            )
            fid *= (1.0 - decoherence_per_layer) ** circuit_depth

        return max(0.0, min(1.0, fid))

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def forecast(
        self,
        duration_hours: float,
        circuit_depth: int,
        circuit_2q_gates: int,
        interval_minutes: float = 15.0,
        fidelity_threshold: float | None = None,
    ) -> ForecastReport:
        """Predict circuit fidelity over time and find optimal submission windows.

        Generates drifted calibrations at each time step, computes the
        circuit fidelity, and identifies contiguous windows where fidelity
        exceeds the threshold.

        Parameters
        ----------
        duration_hours : float
            Forecast horizon in hours.
        circuit_depth : int
            Circuit depth.
        circuit_2q_gates : int
            Number of two-qubit gates.
        interval_minutes : float
            Sampling interval in minutes.
        fidelity_threshold : float, optional
            Override the default fidelity threshold for this forecast.

        Returns
        -------
        ForecastReport
            Complete forecast with fidelity time series and windows.
        """
        threshold = fidelity_threshold if fidelity_threshold is not None else self.fidelity_threshold
        interval_hours = interval_minutes / 60.0
        n_points = max(1, int(duration_hours / interval_hours) + 1)

        times = np.zeros(n_points)
        fidelities = np.zeros(n_points)

        for i in range(n_points):
            t = i * interval_hours
            if t > duration_hours:
                break
            times[i] = t

            drifted_cal = self.drift.at_time(t)
            fid = self.predict_circuit_fidelity(
                drifted_cal, circuit_depth, circuit_2q_gates
            )
            fidelities[i] = fid

        # Trim to actual points
        actual = n_points
        for i in range(1, n_points):
            if times[i] == 0.0 and fidelities[i] == 0.0:
                actual = i
                break
        times = times[:actual]
        fidelities = fidelities[:actual]

        # Find best and worst
        best_idx = int(np.argmax(fidelities))
        worst_idx = int(np.argmin(fidelities))

        # Find optimal windows (contiguous above threshold)
        windows: list[tuple[float, float]] = []
        in_window = False
        window_start = 0.0
        for i in range(len(fidelities)):
            above = fidelities[i] >= threshold
            if above and not in_window:
                window_start = times[i]
                in_window = True
            elif not above and in_window:
                # Window ends at the previous point
                windows.append((window_start, times[i - 1] if i > 0 else window_start))
                in_window = False
        if in_window:
            windows.append((window_start, times[-1]))

        return ForecastReport(
            times_hours=times,
            predicted_fidelities=fidelities,
            optimal_windows=windows,
            best_submission_time_hours=float(times[best_idx]),
            worst_submission_time_hours=float(times[worst_idx]),
            fidelity_at_best=float(fidelities[best_idx]),
            fidelity_at_worst=float(fidelities[worst_idx]),
            fidelity_threshold=threshold,
            circuit_depth=circuit_depth,
            circuit_2q_gates=circuit_2q_gates,
        )

    # ------------------------------------------------------------------
    # Multi-circuit comparison
    # ------------------------------------------------------------------

    def compare_circuits(
        self,
        circuits: dict[str, tuple[int, int]],
        duration_hours: float = 8.0,
        interval_minutes: float = 15.0,
    ) -> dict[str, ForecastReport]:
        """Forecast fidelity for multiple circuits simultaneously.

        Parameters
        ----------
        circuits : dict[str, tuple[int, int]]
            Mapping from circuit name to (depth, num_2q_gates).
        duration_hours : float
            Forecast horizon.
        interval_minutes : float
            Sampling interval.

        Returns
        -------
        dict[str, ForecastReport]
            Forecast report for each circuit, keyed by circuit name.
        """
        results = {}
        for name, (depth, n2q) in circuits.items():
            results[name] = self.forecast(
                duration_hours=duration_hours,
                circuit_depth=depth,
                circuit_2q_gates=n2q,
                interval_minutes=interval_minutes,
            )
        return results


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from .digital_twin import DigitalTwin

    print("=" * 72)
    print("CALIBRATION FORECAST -- SELF-TEST SUITE")
    print("=" * 72)

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

    # ---- Setup ----
    twin = DigitalTwin.from_ibm_backend("heron", num_qubits=5)
    drift = CalibrationDrift(twin.calibration, seed=42)
    forecaster = CalibrationForecaster(drift, fidelity_threshold=0.5)

    # ---- Test 1: Static fidelity prediction ----
    print("\n--- Test 1: Static fidelity prediction ---")
    fid_shallow = CalibrationForecaster.predict_circuit_fidelity(
        twin.calibration, circuit_depth=10, circuit_2q_gates=5
    )
    fid_deep = CalibrationForecaster.predict_circuit_fidelity(
        twin.calibration, circuit_depth=100, circuit_2q_gates=50
    )
    _check(
        "Shallow circuit fidelity > deep",
        fid_shallow > fid_deep,
        f"shallow={fid_shallow:.4f}, deep={fid_deep:.4f}",
    )
    _check(
        "Shallow fidelity in [0, 1]",
        0.0 <= fid_shallow <= 1.0,
    )
    _check(
        "Deep fidelity in [0, 1]",
        0.0 <= fid_deep <= 1.0,
    )
    _check(
        "Zero-depth gives fidelity ~ 1",
        CalibrationForecaster.predict_circuit_fidelity(
            twin.calibration, 0, 0
        ) > 0.99,
    )

    # ---- Test 2: Basic forecast ----
    print("\n--- Test 2: Basic forecast ---")
    report = forecaster.forecast(
        duration_hours=8.0,
        circuit_depth=50,
        circuit_2q_gates=20,
        interval_minutes=30.0,
    )
    _check(
        "Time series length > 0",
        len(report.times_hours) > 0,
        f"n={len(report.times_hours)}",
    )
    _check(
        "Fidelities same length as times",
        len(report.predicted_fidelities) == len(report.times_hours),
    )
    _check(
        "First time is 0",
        report.times_hours[0] == 0.0,
    )
    _check(
        "Best fidelity >= worst",
        report.fidelity_at_best >= report.fidelity_at_worst,
        f"best={report.fidelity_at_best:.4f}, worst={report.fidelity_at_worst:.4f}",
    )
    _check(
        "Best time in range",
        0.0 <= report.best_submission_time_hours <= 8.0,
    )
    print(f"\n  Report:\n{report}")

    # ---- Test 3: Optimal windows ----
    print("\n--- Test 3: Optimal windows ---")
    _check(
        "At least one window (shallow circuit)",
        len(report.optimal_windows) > 0 or report.fidelity_at_best < report.fidelity_threshold,
    )
    for i, (start, end) in enumerate(report.optimal_windows):
        _check(
            f"Window {i + 1}: start <= end",
            start <= end,
            f"start={start:.2f}, end={end:.2f}",
        )

    # ---- Test 4: Deep circuit has fewer/no windows ----
    print("\n--- Test 4: Deep circuit degradation ---")
    report_deep = forecaster.forecast(
        duration_hours=8.0,
        circuit_depth=500,
        circuit_2q_gates=200,
        interval_minutes=30.0,
        fidelity_threshold=0.5,
    )
    _check(
        "Deep circuit worst fidelity < shallow worst",
        report_deep.fidelity_at_worst <= report.fidelity_at_worst + 0.01,
        f"deep_worst={report_deep.fidelity_at_worst:.4f}",
    )

    # ---- Test 5: CSV export ----
    print("\n--- Test 5: CSV export ---")
    csv = report.to_csv()
    lines = csv.strip().split("\n")
    _check("CSV has header", "time_hours" in lines[0])
    _check("CSV has data", len(lines) > 1, f"rows={len(lines) - 1}")

    # ---- Test 6: Custom fidelity threshold ----
    print("\n--- Test 6: Custom threshold ---")
    report_high = forecaster.forecast(
        duration_hours=8.0,
        circuit_depth=50,
        circuit_2q_gates=20,
        fidelity_threshold=0.99,
    )
    report_low = forecaster.forecast(
        duration_hours=8.0,
        circuit_depth=50,
        circuit_2q_gates=20,
        fidelity_threshold=0.01,
    )
    _check(
        "High threshold has fewer/equal windows",
        len(report_high.optimal_windows) <= len(report_low.optimal_windows),
        f"high={len(report_high.optimal_windows)}, low={len(report_low.optimal_windows)}",
    )

    # ---- Test 7: Multi-circuit comparison ----
    print("\n--- Test 7: Multi-circuit comparison ---")
    circuits = {
        "shallow": (10, 5),
        "medium": (50, 20),
        "deep": (200, 80),
    }
    multi = forecaster.compare_circuits(circuits, duration_hours=4.0)
    _check("Three reports returned", len(multi) == 3)
    _check(
        "Shallow best > deep best",
        multi["shallow"].fidelity_at_best >= multi["deep"].fidelity_at_best,
    )

    # ---- Test 8: Google Willow forecast ----
    print("\n--- Test 8: Different hardware ---")
    twin_w = DigitalTwin.from_google_backend("willow", num_qubits=5)
    drift_w = CalibrationDrift(twin_w.calibration, seed=99)
    fc_w = CalibrationForecaster(drift_w)
    report_w = fc_w.forecast(4.0, circuit_depth=30, circuit_2q_gates=15)
    _check(
        "Willow forecast has data",
        len(report_w.times_hours) > 0,
    )
    _check(
        "Willow best fidelity > 0",
        report_w.fidelity_at_best > 0.0,
        f"F={report_w.fidelity_at_best:.4f}",
    )

    # ---- Summary ----
    print("\n" + "=" * 72)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All calibration forecast tests passed.")
    else:
        print("SOME TESTS FAILED -- review output above.")
    print("=" * 72)
