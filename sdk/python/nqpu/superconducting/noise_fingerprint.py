"""Hardware noise fingerprinting for digital twin validation.

Compares real hardware fidelity patterns against digital twin predictions
by constructing multidimensional noise fingerprints that capture the
characteristic error signature of a quantum processor.

A noise fingerprint encodes:
    - Bell state fidelity per qubit pair
    - Single-qubit RB error rates (sorted distribution)
    - T1/T2 distribution statistics (mean, std, min, max)
    - ZZ crosstalk matrix (upper triangle)
    - Leakage rates per qubit (if available)

Two fingerprints can be compared to detect calibration drift, hardware
anomalies, or model-reality gaps.

Example:
    >>> from nqpu.superconducting.digital_twin import DigitalTwin
    >>> from nqpu.superconducting.noise_fingerprint import NoiseFingerprint
    >>> twin = DigitalTwin.from_ibm_backend("heron", num_qubits=5)
    >>> fp = NoiseFingerprint.from_digital_twin(twin)
    >>> print(f"Mean RB error: {fp.rb_error_rates.mean():.6f}")

References:
    - Sarovar et al., Quantum 4, 321 (2020) [noise fingerprinting]
    - Proctor et al., Nat. Phys. 18, 75 (2022) [scalable benchmarking]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .digital_twin import CalibrationData, DigitalTwin
from .noise import TransmonNoiseModel


# ---------------------------------------------------------------------------
# Fingerprint comparison result
# ---------------------------------------------------------------------------

@dataclass
class FingerprintComparison:
    """Result of comparing two noise fingerprints.

    Attributes
    ----------
    max_deviation : float
        Worst-case absolute deviation across all fingerprint components.
    mean_deviation : float
        Average absolute deviation across all components.
    anomalous_qubits : list[int]
        Qubit indices where the deviation exceeds a significance threshold.
    health_score : float
        Composite score in [0, 1] where 1.0 means the fingerprints match
        perfectly and 0.0 means total disagreement.
    drift_detected : bool
        Whether the comparison indicates significant calibration drift.
    recommended_action : str
        Human-readable recommendation based on the comparison.
    component_deviations : dict[str, float]
        Per-component deviation breakdown.
    """

    max_deviation: float
    mean_deviation: float
    anomalous_qubits: list[int]
    health_score: float
    drift_detected: bool
    recommended_action: str
    component_deviations: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "=== Fingerprint Comparison ===",
            f"  Health score:      {self.health_score:.4f}",
            f"  Max deviation:     {self.max_deviation:.6f}",
            f"  Mean deviation:    {self.mean_deviation:.6f}",
            f"  Drift detected:    {self.drift_detected}",
            f"  Anomalous qubits:  {self.anomalous_qubits}",
            f"  Recommendation:    {self.recommended_action}",
        ]
        if self.component_deviations:
            lines.append("  Component deviations:")
            for name, dev in self.component_deviations.items():
                lines.append(f"    {name:<30s} {dev:.6f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Noise fingerprint
# ---------------------------------------------------------------------------

@dataclass
class NoiseFingerprint:
    """Multidimensional noise signature of a quantum processor.

    Captures the characteristic error pattern across all qubits and
    qubit pairs, enabling comparison between digital twin predictions
    and real hardware measurements.

    Attributes
    ----------
    num_qubits : int
        Number of qubits in the fingerprint.
    bell_fidelities : np.ndarray
        Bell state fidelity for each coupled qubit pair.  Shape is
        ``(num_pairs,)`` sorted by pair index.
    bell_pair_labels : list[tuple[int, int]]
        Qubit pair labels corresponding to ``bell_fidelities``.
    rb_error_rates : np.ndarray
        Single-qubit RB error rates, sorted ascending.  Shape ``(num_qubits,)``.
    t1_stats : dict[str, float]
        T1 distribution statistics: mean, std, min, max (in microseconds).
    t2_stats : dict[str, float]
        T2 distribution statistics: mean, std, min, max (in microseconds).
    zz_crosstalk_matrix : np.ndarray
        Upper-triangular ZZ coupling matrix in Hz.
        Shape ``(num_qubits, num_qubits)``.
    leakage_rates : Optional[np.ndarray]
        Per-qubit leakage rates, if available.  Shape ``(num_qubits,)``.
    source_label : str
        Descriptive label for the fingerprint source.
    """

    num_qubits: int
    bell_fidelities: np.ndarray
    bell_pair_labels: list[tuple[int, int]]
    rb_error_rates: np.ndarray
    t1_stats: dict[str, float]
    t2_stats: dict[str, float]
    zz_crosstalk_matrix: np.ndarray
    leakage_rates: Optional[np.ndarray]
    source_label: str = ""

    # ------------------------------------------------------------------
    # Construction from digital twin
    # ------------------------------------------------------------------

    @classmethod
    def from_digital_twin(cls, twin: DigitalTwin) -> NoiseFingerprint:
        """Generate a predicted noise fingerprint from a digital twin.

        Extracts noise characteristics from the twin's calibration data
        and noise model without running full QCVV benchmarks.

        Parameters
        ----------
        twin : DigitalTwin
            Configured digital twin instance.

        Returns
        -------
        NoiseFingerprint
            Predicted noise fingerprint.
        """
        cal = twin.calibration
        config = twin.config
        noise = TransmonNoiseModel(config)
        n = cal.num_qubits

        # Bell fidelities from calibration 2Q fidelity data
        pairs = list(cal.two_qubit_fidelities.keys())
        bell_fids = np.array([cal.two_qubit_fidelities[p] for p in pairs])

        # RB error rates from single-qubit gate fidelities
        rb_errors = np.sort(np.array([1.0 - f for f in cal.single_gate_fidelities]))

        # T1/T2 statistics
        t1_arr = np.array(cal.t1_us)
        t2_arr = np.array(cal.t2_us)
        t1_stats = {
            "mean": float(np.mean(t1_arr)),
            "std": float(np.std(t1_arr)),
            "min": float(np.min(t1_arr)),
            "max": float(np.max(t1_arr)),
        }
        t2_stats = {
            "mean": float(np.mean(t2_arr)),
            "std": float(np.std(t2_arr)),
            "min": float(np.min(t2_arr)),
            "max": float(np.max(t2_arr)),
        }

        # ZZ crosstalk matrix
        zz_matrix = np.zeros((n, n), dtype=np.float64)
        for (a, b) in cal.coupling_map:
            if a < n and b < n:
                zz_val = noise.zz_coupling_hz(min(a, b), max(a, b))
                zz_matrix[min(a, b), max(a, b)] = zz_val

        # Leakage rates from noise model
        leakage = np.array([noise.leakage_prob(q) for q in range(n)])

        return cls(
            num_qubits=n,
            bell_fidelities=bell_fids,
            bell_pair_labels=pairs,
            rb_error_rates=rb_errors,
            t1_stats=t1_stats,
            t2_stats=t2_stats,
            zz_crosstalk_matrix=zz_matrix,
            leakage_rates=leakage,
            source_label=f"digital_twin:{cal.device_name}",
        )

    # ------------------------------------------------------------------
    # Construction from measurement data
    # ------------------------------------------------------------------

    @classmethod
    def from_measurement_data(cls, results: dict) -> NoiseFingerprint:
        """Build a noise fingerprint from experimental measurement data.

        Parameters
        ----------
        results : dict
            Measurement data dictionary with keys:

            - ``"num_qubits"`` (int): Number of qubits.
            - ``"bell_fidelities"`` (dict): Mapping ``"i,j"`` -> fidelity.
            - ``"rb_error_rates"`` (list[float]): Per-qubit RB error rates.
            - ``"t1_us"`` (list[float]): Per-qubit T1 in microseconds.
            - ``"t2_us"`` (list[float]): Per-qubit T2 in microseconds.
            - ``"zz_crosstalk_hz"`` (dict, optional): ``"i,j"`` -> ZZ in Hz.
            - ``"leakage_rates"`` (list[float], optional): Per-qubit leakage.
            - ``"device_name"`` (str, optional): Source device name.

        Returns
        -------
        NoiseFingerprint
            Fingerprint constructed from the measurements.
        """
        n = results["num_qubits"]

        # Parse bell fidelities
        bell_dict = results.get("bell_fidelities", {})
        pairs = []
        bell_vals = []
        for key_str, fid in bell_dict.items():
            parts = key_str.split(",")
            pair = (int(parts[0]), int(parts[1]))
            pairs.append(pair)
            bell_vals.append(fid)
        bell_fids = np.array(bell_vals) if bell_vals else np.array([])

        # RB error rates
        rb_raw = results.get("rb_error_rates", [0.0] * n)
        rb_errors = np.sort(np.array(rb_raw))

        # T1/T2 statistics
        t1_arr = np.array(results.get("t1_us", [100.0] * n))
        t2_arr = np.array(results.get("t2_us", [80.0] * n))
        t1_stats = {
            "mean": float(np.mean(t1_arr)),
            "std": float(np.std(t1_arr)),
            "min": float(np.min(t1_arr)),
            "max": float(np.max(t1_arr)),
        }
        t2_stats = {
            "mean": float(np.mean(t2_arr)),
            "std": float(np.std(t2_arr)),
            "min": float(np.min(t2_arr)),
            "max": float(np.max(t2_arr)),
        }

        # ZZ crosstalk
        zz_matrix = np.zeros((n, n), dtype=np.float64)
        zz_dict = results.get("zz_crosstalk_hz", {})
        for key_str, val in zz_dict.items():
            parts = key_str.split(",")
            a, b = int(parts[0]), int(parts[1])
            if a < n and b < n:
                zz_matrix[min(a, b), max(a, b)] = val

        # Leakage rates
        leak_raw = results.get("leakage_rates")
        leakage = np.array(leak_raw) if leak_raw is not None else None

        device_name = results.get("device_name", "measurement")

        return cls(
            num_qubits=n,
            bell_fidelities=bell_fids,
            bell_pair_labels=pairs,
            rb_error_rates=rb_errors,
            t1_stats=t1_stats,
            t2_stats=t2_stats,
            zz_crosstalk_matrix=zz_matrix,
            leakage_rates=leakage,
            source_label=f"measured:{device_name}",
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        other: NoiseFingerprint,
        anomaly_threshold: float = 0.01,
    ) -> FingerprintComparison:
        """Compare this fingerprint against another.

        Computes component-wise deviations and an aggregate health score.

        Parameters
        ----------
        other : NoiseFingerprint
            The fingerprint to compare against.
        anomaly_threshold : float
            Absolute deviation threshold for flagging a qubit as anomalous.

        Returns
        -------
        FingerprintComparison
            Detailed comparison results.
        """
        deviations: dict[str, float] = {}
        all_devs: list[float] = []

        # Bell fidelity deviation
        if len(self.bell_fidelities) > 0 and len(other.bell_fidelities) > 0:
            min_len = min(len(self.bell_fidelities), len(other.bell_fidelities))
            bell_dev = float(np.mean(np.abs(
                self.bell_fidelities[:min_len] - other.bell_fidelities[:min_len]
            )))
            deviations["bell_fidelity"] = bell_dev
            all_devs.append(bell_dev)

        # RB error rate deviation
        if len(self.rb_error_rates) > 0 and len(other.rb_error_rates) > 0:
            min_len = min(len(self.rb_error_rates), len(other.rb_error_rates))
            rb_dev = float(np.mean(np.abs(
                self.rb_error_rates[:min_len] - other.rb_error_rates[:min_len]
            )))
            deviations["rb_error_rates"] = rb_dev
            all_devs.append(rb_dev)

        # T1 deviation
        t1_dev = abs(self.t1_stats["mean"] - other.t1_stats["mean"])
        t1_base = max(self.t1_stats["mean"], 1e-6)
        t1_rel = t1_dev / t1_base
        deviations["t1_relative"] = t1_rel
        all_devs.append(t1_rel)

        # T2 deviation
        t2_dev = abs(self.t2_stats["mean"] - other.t2_stats["mean"])
        t2_base = max(self.t2_stats["mean"], 1e-6)
        t2_rel = t2_dev / t2_base
        deviations["t2_relative"] = t2_rel
        all_devs.append(t2_rel)

        # ZZ crosstalk deviation (normalized)
        zz_diff = np.abs(self.zz_crosstalk_matrix - other.zz_crosstalk_matrix)
        zz_max = max(
            np.max(np.abs(self.zz_crosstalk_matrix)),
            np.max(np.abs(other.zz_crosstalk_matrix)),
            1.0,
        )
        zz_dev = float(np.mean(zz_diff)) / zz_max
        deviations["zz_crosstalk"] = zz_dev
        all_devs.append(zz_dev)

        # Leakage deviation
        if self.leakage_rates is not None and other.leakage_rates is not None:
            min_len = min(len(self.leakage_rates), len(other.leakage_rates))
            leak_dev = float(np.mean(np.abs(
                self.leakage_rates[:min_len] - other.leakage_rates[:min_len]
            )))
            deviations["leakage"] = leak_dev
            all_devs.append(leak_dev)

        # Aggregate metrics
        max_deviation = max(all_devs) if all_devs else 0.0
        mean_deviation = float(np.mean(all_devs)) if all_devs else 0.0

        # Health score: exponential decay with deviation
        health_score = math.exp(-10.0 * mean_deviation)
        health_score = max(0.0, min(1.0, health_score))

        # Anomalous qubits: where RB deviation exceeds threshold
        anomalous: list[int] = []
        if len(self.rb_error_rates) > 0 and len(other.rb_error_rates) > 0:
            min_len = min(len(self.rb_error_rates), len(other.rb_error_rates))
            per_qubit_dev = np.abs(
                self.rb_error_rates[:min_len] - other.rb_error_rates[:min_len]
            )
            for q in range(min_len):
                if per_qubit_dev[q] > anomaly_threshold:
                    anomalous.append(q)

        # Drift detection
        drift_detected = mean_deviation > 0.005 or max_deviation > 0.02

        # Recommendation
        if health_score > 0.95:
            action = "No action needed. Fingerprints match well."
        elif health_score > 0.80:
            action = "Minor drift detected. Monitor and consider recalibration."
        elif health_score > 0.50:
            action = "Significant drift. Recalibrate within the next hour."
        else:
            action = "Major discrepancy. Immediate recalibration required."

        if anomalous:
            action += f" Qubits {anomalous} show anomalous behavior."

        return FingerprintComparison(
            max_deviation=max_deviation,
            mean_deviation=mean_deviation,
            anomalous_qubits=anomalous,
            health_score=health_score,
            drift_detected=drift_detected,
            recommended_action=action,
            component_deviations=deviations,
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from .digital_twin import CalibrationDrift

    print("=" * 72)
    print("NOISE FINGERPRINT -- SELF-TEST SUITE")
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

    # ---- Test 1: Fingerprint from digital twin ----
    print("\n--- Test 1: from_digital_twin ---")
    twin = DigitalTwin.from_ibm_backend("heron", num_qubits=5)
    fp = NoiseFingerprint.from_digital_twin(twin)
    _check("Num qubits correct", fp.num_qubits == 5)
    _check("Bell fidelities populated", len(fp.bell_fidelities) > 0)
    _check(
        "Bell fidelities in range",
        all(0.9 <= f <= 1.0 for f in fp.bell_fidelities),
    )
    _check("RB error rates populated", len(fp.rb_error_rates) == 5)
    _check("RB errors sorted", all(
        fp.rb_error_rates[i] <= fp.rb_error_rates[i + 1]
        for i in range(len(fp.rb_error_rates) - 1)
    ))
    _check("T1 stats has mean", "mean" in fp.t1_stats)
    _check("T2 stats has std", "std" in fp.t2_stats)
    _check(
        "ZZ matrix is square",
        fp.zz_crosstalk_matrix.shape == (5, 5),
    )
    _check("Leakage rates populated", fp.leakage_rates is not None)
    _check("Source label set", "digital_twin" in fp.source_label)

    # ---- Test 2: Fingerprint from measurement data ----
    print("\n--- Test 2: from_measurement_data ---")
    meas_data = {
        "num_qubits": 3,
        "bell_fidelities": {"0,1": 0.995, "1,2": 0.990},
        "rb_error_rates": [0.0005, 0.0008, 0.0003],
        "t1_us": [300.0, 280.0, 310.0],
        "t2_us": [200.0, 190.0, 210.0],
        "zz_crosstalk_hz": {"0,1": 50000.0, "1,2": 30000.0},
        "leakage_rates": [0.001, 0.002, 0.0015],
        "device_name": "test_device",
    }
    fp_meas = NoiseFingerprint.from_measurement_data(meas_data)
    _check("Measurement FP qubits", fp_meas.num_qubits == 3)
    _check("Measurement FP bell pairs", len(fp_meas.bell_fidelities) == 2)
    _check("Measurement FP RB sorted", fp_meas.rb_error_rates[0] <= fp_meas.rb_error_rates[-1])
    _check(
        "Measurement FP T1 mean",
        abs(fp_meas.t1_stats["mean"] - 296.67) < 1.0,
    )

    # ---- Test 3: Self-comparison (perfect match) ----
    print("\n--- Test 3: Self-comparison ---")
    comp = fp.compare(fp)
    _check(
        "Self-comparison max_deviation ~ 0",
        comp.max_deviation < 1e-10,
        f"max_dev={comp.max_deviation}",
    )
    _check(
        "Self-comparison health_score ~ 1",
        comp.health_score > 0.99,
        f"health={comp.health_score}",
    )
    _check("Self-comparison no drift", not comp.drift_detected)
    _check("Self-comparison no anomalous qubits", len(comp.anomalous_qubits) == 0)

    # ---- Test 4: Comparison with drifted twin ----
    print("\n--- Test 4: Drifted comparison ---")
    drift = CalibrationDrift(twin.calibration, seed=42)
    drifted_cal = drift.at_time(6.0)
    drifted_twin = DigitalTwin.from_calibration(drifted_cal)
    fp_drifted = NoiseFingerprint.from_digital_twin(drifted_twin)

    comp_drift = fp.compare(fp_drifted)
    _check(
        "Drifted max_deviation > 0",
        comp_drift.max_deviation > 0.0,
        f"max_dev={comp_drift.max_deviation:.6f}",
    )
    _check(
        "Drifted health_score < 1",
        comp_drift.health_score < 1.0,
        f"health={comp_drift.health_score:.4f}",
    )
    _check("Drifted has recommendation", len(comp_drift.recommended_action) > 0)
    _check("Component deviations populated", len(comp_drift.component_deviations) > 0)
    print(f"\n  Full comparison:\n{comp_drift}")

    # ---- Test 5: Cross-platform comparison ----
    print("\n--- Test 5: Cross-platform comparison ---")
    twin_willow = DigitalTwin.from_google_backend("willow", num_qubits=5)
    fp_willow = NoiseFingerprint.from_digital_twin(twin_willow)
    comp_cross = fp.compare(fp_willow)
    _check(
        "Cross-platform has large deviation",
        comp_cross.max_deviation > 0.001,
        f"max_dev={comp_cross.max_deviation:.6f}",
    )
    _check(
        "Cross-platform health < self",
        comp_cross.health_score < 1.0,
    )

    # ---- Test 6: FingerprintComparison __str__ ----
    print("\n--- Test 6: String representation ---")
    s = str(comp_drift)
    _check("String contains health score", "Health score" in s)
    _check("String contains recommendation", "Recommendation" in s or "action" in s.lower())

    # ---- Summary ----
    print("\n" + "=" * 72)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All noise fingerprint tests passed.")
    else:
        print("SOME TESTS FAILED -- review output above.")
    print("=" * 72)
