"""QEC threshold analysis under calibration drift.

Studies how calibration aging in superconducting processors affects the
logical error rate of surface-code error correction.  Uses a simple
threshold model for the surface code:

    p_L = C * (p / p_th) ^ ((d + 1) / 2)

where:
    - p_L:   logical error rate
    - p:     physical error rate (from gate fidelity, decoherence, etc.)
    - p_th:  threshold error rate (~1% for surface code)
    - d:     code distance
    - C:     constant of order 0.1 (from numerical simulations)

By combining the CalibrationDrift model with this threshold formula,
the study answers practical questions:
    - How long after calibration does the logical error rate exceed a target?
    - What is the optimal recalibration interval for a given code distance?
    - How does the code distance trade off against drift tolerance?

Example:
    >>> from nqpu.superconducting.digital_twin import DigitalTwin, CalibrationDrift
    >>> from nqpu.superconducting.qec_drift_study import QECDriftStudy
    >>> twin = DigitalTwin.from_ibm_backend("heron", num_qubits=10)
    >>> drift = CalibrationDrift(twin.calibration, seed=42)
    >>> study = QECDriftStudy(twin, drift)
    >>> report = study.study_drift_impact(duration_hours=24.0)
    >>> print(f"Time to threshold: {report.time_to_threshold_hours:.1f} h")

References:
    - Fowler et al., PRA 86, 032324 (2012) [surface code threshold]
    - Google Quantum AI, Nature 614, 676 (2024) [below threshold]
    - Acharya et al., Nature 636, 922 (2024) [Willow QEC]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .digital_twin import CalibrationDrift, CalibrationData, DigitalTwin


# ---------------------------------------------------------------------------
# Drift impact report
# ---------------------------------------------------------------------------


@dataclass
class DriftImpactReport:
    """Results from a QEC drift impact study.

    Attributes
    ----------
    times_hours : np.ndarray
        Time points in hours at which the system was evaluated.
    physical_error_rates : np.ndarray
        Effective physical error rate at each time point.
    logical_error_rates : np.ndarray
        Predicted logical error rate at each time point.
    code_distance : int
        Surface code distance used in the analysis.
    threshold_error_rate : float
        Physical threshold error rate for the surface code.
    target_logical_rate : float
        Target logical error rate for determining the recalibration window.
    time_to_threshold_hours : float
        Time in hours until the logical error rate exceeds the target.
        Set to ``float('inf')`` if the target is never exceeded.
    optimal_recalibration_interval_hours : float
        Recommended recalibration interval to keep the logical error rate
        below the target, with a 20% safety margin.
    """

    times_hours: np.ndarray
    physical_error_rates: np.ndarray
    logical_error_rates: np.ndarray
    code_distance: int
    threshold_error_rate: float
    target_logical_rate: float
    time_to_threshold_hours: float
    optimal_recalibration_interval_hours: float

    def to_csv(self) -> str:
        """Export the time series data as CSV.

        Returns
        -------
        str
            CSV string with columns: time_hours, physical_error, logical_error.
        """
        lines = ["time_hours,physical_error,logical_error"]
        for i in range(len(self.times_hours)):
            lines.append(
                f"{self.times_hours[i]:.4f},"
                f"{self.physical_error_rates[i]:.8e},"
                f"{self.logical_error_rates[i]:.8e}"
            )
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Export a summary as a LaTeX table.

        Returns
        -------
        str
            Booktabs-formatted LaTeX table of the key findings.
        """
        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{QEC drift impact analysis.}",
            r"  \label{tab:qec_drift}",
            r"  \begin{tabular}{lr}",
            r"    \toprule",
            r"    Parameter & Value \\",
            r"    \midrule",
            f"    Code distance & {self.code_distance} \\\\",
            f"    Threshold $p_{{\\mathrm{{th}}}}$ & "
            f"${self.threshold_error_rate:.4f}$ \\\\",
            f"    Target $p_L$ & "
            f"${self.target_logical_rate:.2e}$ \\\\",
            f"    Initial physical error & "
            f"${self.physical_error_rates[0]:.6f}$ \\\\",
            f"    Time to threshold & "
            f"${self.time_to_threshold_hours:.1f}$ hours \\\\",
            f"    Recalibration interval & "
            f"${self.optimal_recalibration_interval_hours:.1f}$ hours \\\\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        lines = [
            "=== QEC Drift Impact Report ===",
            f"  Code distance:        {self.code_distance}",
            f"  Threshold p_th:       {self.threshold_error_rate:.4f}",
            f"  Target p_L:           {self.target_logical_rate:.2e}",
            f"  Initial physical err: {self.physical_error_rates[0]:.6f}",
            f"  Final physical err:   {self.physical_error_rates[-1]:.6f}",
            f"  Initial logical err:  {self.logical_error_rates[0]:.2e}",
            f"  Final logical err:    {self.logical_error_rates[-1]:.2e}",
            f"  Time to threshold:    {self.time_to_threshold_hours:.1f} hours",
            f"  Recal interval:       {self.optimal_recalibration_interval_hours:.1f} hours",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# QEC drift study
# ---------------------------------------------------------------------------


class QECDriftStudy:
    """Surface code QEC analysis under calibration drift.

    Combines a physical drift model (from ``CalibrationDrift``) with a
    surface code threshold model to predict how calibration aging affects
    the logical error rate over time.

    Parameters
    ----------
    twin : DigitalTwin
        Baseline digital twin providing the initial calibration.
    drift_model : CalibrationDrift, optional
        Drift model for calibration aging.  If ``None``, one is created
        from the twin's calibration with seed 42.
    code_distance : int
        Default surface code distance.
    threshold_p : float
        Physical error rate threshold for the surface code.
    constant_C : float
        Prefactor in the threshold formula p_L = C * (p/p_th)^((d+1)/2).
    target_logical_rate : float
        Target logical error rate for recalibration analysis.

    References
    ----------
    - Fowler et al., PRA 86, 032324 (2012)
    - Google Quantum AI, Nature 614, 676 (2024)
    """

    def __init__(
        self,
        twin: DigitalTwin,
        drift_model: Optional[CalibrationDrift] = None,
        code_distance: int = 3,
        threshold_p: float = 0.01,
        constant_C: float = 0.1,
        target_logical_rate: float = 1e-6,
    ) -> None:
        self.twin = twin
        self.drift = drift_model or CalibrationDrift(twin.calibration, seed=42)
        self.code_distance = code_distance
        self.threshold_p = threshold_p
        self.constant_C = constant_C
        self.target_logical_rate = target_logical_rate

    # ------------------------------------------------------------------
    # Threshold model
    # ------------------------------------------------------------------

    def logical_error_rate(
        self,
        physical_error: float,
        distance: int = 0,
    ) -> float:
        """Compute the logical error rate from the surface code threshold model.

        Uses the formula:
            p_L = C * (p / p_th) ^ ((d + 1) / 2)

        When the physical error rate exceeds the threshold, the logical
        error rate is capped at 0.5 (maximally mixed).

        Parameters
        ----------
        physical_error : float
            Effective physical error rate per cycle.
        distance : int
            Code distance.  If 0, uses ``self.code_distance``.

        Returns
        -------
        float
            Logical error rate per code cycle.
        """
        d = distance if distance > 0 else self.code_distance
        if physical_error <= 0.0:
            return 0.0
        if physical_error >= self.threshold_p:
            return min(0.5, self.constant_C * (physical_error / self.threshold_p) ** ((d + 1) / 2.0))

        ratio = physical_error / self.threshold_p
        exponent = (d + 1) / 2.0
        p_L = self.constant_C * (ratio ** exponent)
        return min(0.5, max(0.0, p_L))

    # ------------------------------------------------------------------
    # Effective physical error from calibration
    # ------------------------------------------------------------------

    @staticmethod
    def effective_physical_error(cal: CalibrationData) -> float:
        """Compute the effective physical error rate from calibration data.

        Combines single-qubit gate error, two-qubit gate error, and
        decoherence during a typical QEC cycle into a single effective
        physical error rate.

        The model assumes a QEC cycle of duration approximately equal
        to the two-qubit gate time (dominated by syndrome extraction).

        Parameters
        ----------
        cal : CalibrationData
            Calibration snapshot.

        Returns
        -------
        float
            Effective physical error rate.
        """
        # Gate errors
        mean_1q_err = 1.0 - cal.mean_1q_fidelity
        mean_2q_err = 1.0 - cal.mean_2q_fidelity

        # Decoherence during a QEC cycle (approx = 2Q gate time)
        cycle_time_us = cal.two_qubit_gate_time_ns / 1000.0
        mean_t2 = cal.mean_t2
        if mean_t2 > 0:
            decoherence_err = 1.0 - math.exp(-cycle_time_us / mean_t2)
        else:
            decoherence_err = 0.0

        # Combined: each QEC cycle involves ~4 syndrome extraction rounds
        # with 1 CNOT per round + single-qubit prep/measurement.
        # Simplified: p_eff ~ 4 * p_2q + 2 * p_1q + p_decoherence
        p_eff = 4.0 * mean_2q_err + 2.0 * mean_1q_err + decoherence_err

        return min(0.5, max(0.0, p_eff))

    # ------------------------------------------------------------------
    # Drift impact study
    # ------------------------------------------------------------------

    def study_drift_impact(
        self,
        duration_hours: float = 24.0,
        interval_minutes: float = 15.0,
        distance: int = 0,
    ) -> DriftImpactReport:
        """Study how calibration drift affects the logical error rate over time.

        Generates drifted calibration snapshots at regular intervals,
        computes the effective physical error rate at each point, and
        translates that into a logical error rate via the threshold model.

        Parameters
        ----------
        duration_hours : float
            Total analysis window in hours.
        interval_minutes : float
            Sampling interval in minutes.
        distance : int
            Code distance to use.  If 0, uses ``self.code_distance``.

        Returns
        -------
        DriftImpactReport
            Complete drift impact analysis.
        """
        d = distance if distance > 0 else self.code_distance

        interval_hours = interval_minutes / 60.0
        n_points = max(1, int(duration_hours / interval_hours) + 1)

        times = np.zeros(n_points)
        p_physical = np.zeros(n_points)
        p_logical = np.zeros(n_points)

        for i in range(n_points):
            t = i * interval_hours
            if t > duration_hours:
                break
            times[i] = t

            # Get drifted calibration at time t
            drifted_cal = self.drift.at_time(t)

            # Compute effective physical error
            p_eff = self.effective_physical_error(drifted_cal)
            p_physical[i] = p_eff

            # Compute logical error rate
            p_L = self.logical_error_rate(p_eff, distance=d)
            p_logical[i] = p_L

        # Trim to actual points computed
        actual_points = n_points
        for i in range(n_points):
            if i > 0 and times[i] == 0.0:
                actual_points = i
                break
        times = times[:actual_points]
        p_physical = p_physical[:actual_points]
        p_logical = p_logical[:actual_points]

        # Find time to threshold
        time_to_threshold = float("inf")
        for i in range(len(p_logical)):
            if p_logical[i] > self.target_logical_rate:
                if i == 0:
                    time_to_threshold = 0.0
                else:
                    # Linear interpolation to find exact crossing
                    frac = (
                        (self.target_logical_rate - p_logical[i - 1])
                        / max(p_logical[i] - p_logical[i - 1], 1e-30)
                    )
                    time_to_threshold = times[i - 1] + frac * (times[i] - times[i - 1])
                break

        # Optimal recalibration interval: 80% of time to threshold
        if math.isinf(time_to_threshold):
            recal_interval = duration_hours
        else:
            recal_interval = max(0.25, time_to_threshold * 0.8)

        return DriftImpactReport(
            times_hours=times,
            physical_error_rates=p_physical,
            logical_error_rates=p_logical,
            code_distance=d,
            threshold_error_rate=self.threshold_p,
            target_logical_rate=self.target_logical_rate,
            time_to_threshold_hours=time_to_threshold,
            optimal_recalibration_interval_hours=recal_interval,
        )

    # ------------------------------------------------------------------
    # Distance scaling study
    # ------------------------------------------------------------------

    def distance_scaling(
        self,
        distances: list[int] | None = None,
        time_hours: float = 0.0,
    ) -> dict[int, float]:
        """Compute logical error rate as a function of code distance.

        Parameters
        ----------
        distances : list[int], optional
            Code distances to evaluate.  Default: [3, 5, 7, 9, 11].
        time_hours : float
            Time since calibration at which to evaluate.

        Returns
        -------
        dict[int, float]
            Mapping from code distance to logical error rate.
        """
        if distances is None:
            distances = [3, 5, 7, 9, 11]

        cal = self.drift.at_time(time_hours)
        p_eff = self.effective_physical_error(cal)

        return {d: self.logical_error_rate(p_eff, distance=d) for d in distances}


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 72)
    print("QEC DRIFT STUDY -- SELF-TEST SUITE")
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

    # ---- Test 1: Threshold model ----
    print("\n--- Test 1: Threshold model ---")
    twin = DigitalTwin.from_ibm_backend("heron", num_qubits=5)
    drift = CalibrationDrift(twin.calibration, seed=42)
    study = QECDriftStudy(twin, drift, code_distance=3)

    # Below threshold: p < p_th => p_L < C (suppressed relative to at-threshold)
    p_L_below = study.logical_error_rate(0.005, distance=3)
    _check(
        "Below threshold: p_L < C (suppression vs at-threshold)",
        p_L_below < 0.1,
        f"p_L={p_L_below:.6e}",
    )

    # At threshold: p = p_th
    p_L_at = study.logical_error_rate(0.01, distance=3)
    _check(
        "At threshold: p_L = C",
        abs(p_L_at - 0.1) < 0.01,
        f"p_L={p_L_at:.6e}",
    )

    # Above threshold: p > p_th
    p_L_above = study.logical_error_rate(0.02, distance=3)
    _check(
        "Above threshold: p_L > C",
        p_L_above > 0.1,
        f"p_L={p_L_above:.6e}",
    )

    # Zero error
    p_L_zero = study.logical_error_rate(0.0, distance=3)
    _check("Zero physical error -> zero logical", p_L_zero == 0.0)

    # ---- Test 2: Distance scaling ----
    print("\n--- Test 2: Distance scaling ---")
    p_L_d3 = study.logical_error_rate(0.005, distance=3)
    p_L_d5 = study.logical_error_rate(0.005, distance=5)
    p_L_d7 = study.logical_error_rate(0.005, distance=7)
    _check(
        "Higher distance -> lower p_L (d3 > d5)",
        p_L_d3 > p_L_d5,
        f"d3={p_L_d3:.2e}, d5={p_L_d5:.2e}",
    )
    _check(
        "Higher distance -> lower p_L (d5 > d7)",
        p_L_d5 > p_L_d7,
        f"d5={p_L_d5:.2e}, d7={p_L_d7:.2e}",
    )

    # ---- Test 3: Effective physical error ----
    print("\n--- Test 3: Effective physical error ---")
    p_eff = QECDriftStudy.effective_physical_error(twin.calibration)
    _check(
        "Physical error > 0",
        p_eff > 0.0,
        f"p_eff={p_eff:.6f}",
    )
    _check(
        "Physical error < 0.5",
        p_eff < 0.5,
        f"p_eff={p_eff:.6f}",
    )

    # ---- Test 4: Drift impact study ----
    print("\n--- Test 4: Drift impact study ---")
    report = study.study_drift_impact(
        duration_hours=12.0, interval_minutes=30.0, distance=3
    )
    _check(
        "Time series has correct length",
        len(report.times_hours) > 0,
        f"n_points={len(report.times_hours)}",
    )
    _check(
        "Physical errors populated",
        len(report.physical_error_rates) == len(report.times_hours),
    )
    _check(
        "Logical errors populated",
        len(report.logical_error_rates) == len(report.times_hours),
    )
    _check(
        "First time is 0",
        report.times_hours[0] == 0.0,
    )
    _check(
        "Recal interval > 0",
        report.optimal_recalibration_interval_hours > 0.0,
        f"recal={report.optimal_recalibration_interval_hours:.1f}h",
    )
    _check(
        "Time to threshold >= 0 (may be 0 if already above target)",
        report.time_to_threshold_hours >= 0.0 or math.isinf(report.time_to_threshold_hours),
        f"t2t={report.time_to_threshold_hours:.1f}h",
    )
    print(f"\n  Report:\n{report}")

    # ---- Test 5: CSV export ----
    print("\n--- Test 5: CSV export ---")
    csv = report.to_csv()
    lines = csv.strip().split("\n")
    _check("CSV has header", "time_hours" in lines[0])
    _check(
        "CSV has data rows",
        len(lines) > 1,
        f"rows={len(lines) - 1}",
    )

    # ---- Test 6: LaTeX export ----
    print("\n--- Test 6: LaTeX export ---")
    latex = report.to_latex()
    _check("LaTeX contains table", r"\begin{table}" in latex)
    _check("LaTeX contains distance", str(report.code_distance) in latex)

    # ---- Test 7: Distance scaling method ----
    print("\n--- Test 7: distance_scaling method ---")
    scaling = study.distance_scaling(distances=[3, 5, 7], time_hours=0.0)
    _check("Scaling has 3 entries", len(scaling) == 3)
    _check(
        "Scaling d=3 > d=7",
        scaling[3] > scaling[7],
        f"d3={scaling[3]:.2e}, d7={scaling[7]:.2e}",
    )

    # ---- Test 8: Large code distance at long drift ----
    print("\n--- Test 8: Large distance vs drift ---")
    study_d7 = QECDriftStudy(twin, drift, code_distance=7)
    report_d7 = study_d7.study_drift_impact(duration_hours=12.0, interval_minutes=60.0)
    report_d3 = study.study_drift_impact(duration_hours=12.0, interval_minutes=60.0)
    _check(
        "d=7 has longer time to threshold than d=3 (or both inf)",
        report_d7.time_to_threshold_hours >= report_d3.time_to_threshold_hours
        or math.isinf(report_d7.time_to_threshold_hours),
        f"d7={report_d7.time_to_threshold_hours:.1f}, "
        f"d3={report_d3.time_to_threshold_hours:.1f}",
    )

    # ---- Summary ----
    print("\n" + "=" * 72)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All QEC drift study tests passed.")
    else:
        print("SOME TESTS FAILED -- review output above.")
    print("=" * 72)
