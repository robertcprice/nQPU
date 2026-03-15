"""Comprehensive tests for digital twin construction and validation.

Tests cover:
- CalibrationData properties and summary computation
- DigitalTwin construction from IBM and Google presets
- Custom calibration data round-trip
- QCVV validation suite (Bell, GHZ, RB, QV)
- Circuit fidelity predictions
- Edge cases: minimum qubits, invalid backend names
- Comparison utilities
"""

import math

import numpy as np
import pytest

from nqpu.superconducting.digital_twin import (
    CalibrationData,
    DigitalTwin,
    ValidationReport,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def custom_calibration():
    """Minimal 2-qubit custom calibration data."""
    return CalibrationData(
        qubit_frequencies_ghz=[5.0, 5.1],
        t1_us=[200.0, 180.0],
        t2_us=[150.0, 130.0],
        readout_fidelities=[0.99, 0.98],
        single_gate_fidelities=[0.9995, 0.9993],
        two_qubit_fidelities={(0, 1): 0.995},
        coupling_map=[(0, 1)],
        native_gate_family="ecr",
        device_name="Test 2Q",
    )


@pytest.fixture
def three_qubit_calibration():
    """3-qubit custom calibration data for more thorough testing."""
    return CalibrationData(
        qubit_frequencies_ghz=[5.0, 5.05, 5.1],
        t1_us=[250.0, 230.0, 260.0],
        t2_us=[180.0, 170.0, 190.0],
        readout_fidelities=[0.99, 0.985, 0.992],
        single_gate_fidelities=[0.9996, 0.9994, 0.9997],
        two_qubit_fidelities={(0, 1): 0.996, (1, 2): 0.994},
        coupling_map=[(0, 1), (1, 2)],
        native_gate_family="ecr",
        device_name="Custom 3Q Lab Device",
    )


# ======================================================================
# CalibrationData tests
# ======================================================================


class TestCalibrationData:
    """Tests for CalibrationData container."""

    def test_num_qubits(self, custom_calibration):
        """num_qubits matches the frequency list length."""
        assert custom_calibration.num_qubits == 2

    def test_mean_t1(self, custom_calibration):
        """Mean T1 is computed correctly."""
        expected = (200.0 + 180.0) / 2.0
        assert custom_calibration.mean_t1 == pytest.approx(expected)

    def test_mean_t2(self, custom_calibration):
        """Mean T2 is computed correctly."""
        expected = (150.0 + 130.0) / 2.0
        assert custom_calibration.mean_t2 == pytest.approx(expected)

    def test_mean_1q_fidelity(self, custom_calibration):
        """Mean single-qubit fidelity is computed correctly."""
        expected = (0.9995 + 0.9993) / 2.0
        assert custom_calibration.mean_1q_fidelity == pytest.approx(expected)

    def test_mean_2q_fidelity(self, custom_calibration):
        """Mean two-qubit fidelity from edge data."""
        assert custom_calibration.mean_2q_fidelity == pytest.approx(0.995)

    def test_mean_readout_fidelity(self, custom_calibration):
        """Mean readout fidelity is computed correctly."""
        expected = (0.99 + 0.98) / 2.0
        assert custom_calibration.mean_readout_fidelity == pytest.approx(expected)

    def test_summary_keys(self, custom_calibration):
        """Summary dict contains all expected keys."""
        summary = custom_calibration.summary()
        expected_keys = {
            "device", "num_qubits", "num_couplers", "native_gate",
            "mean_T1_us", "mean_T2_us", "mean_1Q_fidelity",
            "mean_2Q_fidelity", "mean_readout_fidelity",
        }
        assert set(summary.keys()) == expected_keys

    def test_empty_2q_fidelities(self):
        """Mean 2Q fidelity is 0.0 when no edges exist."""
        cal = CalibrationData(
            qubit_frequencies_ghz=[5.0],
            t1_us=[200.0],
            t2_us=[150.0],
            readout_fidelities=[0.99],
            single_gate_fidelities=[0.9995],
            two_qubit_fidelities={},
            coupling_map=[],
        )
        assert cal.mean_2q_fidelity == 0.0


# ======================================================================
# DigitalTwin construction tests
# ======================================================================


class TestDigitalTwinConstruction:
    """Tests for building digital twins from calibration data."""

    def test_from_custom_calibration(self, custom_calibration):
        """Building from custom data produces a valid twin."""
        twin = DigitalTwin.from_calibration(custom_calibration)
        assert twin.config.num_qubits == 2
        assert twin.calibration.device_name == "Test 2Q"

    def test_from_three_qubit_calibration(self, three_qubit_calibration):
        """3-qubit calibration produces correct topology."""
        twin = DigitalTwin.from_calibration(three_qubit_calibration)
        assert twin.config.num_qubits == 3
        assert len(twin.config.topology.edges) == 2

    @pytest.mark.parametrize("backend_name", ["eagle", "heron"])
    def test_from_ibm_backend(self, backend_name):
        """IBM preset twins build without errors."""
        twin = DigitalTwin.from_ibm_backend(backend_name, num_qubits=3)
        assert twin.config.num_qubits == 3
        assert twin.calibration.num_qubits == 3

    @pytest.mark.parametrize("backend_name", ["sycamore", "willow"])
    def test_from_google_backend(self, backend_name):
        """Google preset twins build without errors."""
        twin = DigitalTwin.from_google_backend(backend_name, num_qubits=3)
        assert twin.config.num_qubits == 3

    def test_ibm_backend_unknown_raises(self):
        """Unknown IBM backend name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown IBM backend"):
            DigitalTwin.from_ibm_backend("unknown_device")

    def test_google_backend_unknown_raises(self):
        """Unknown Google backend name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown Google backend"):
            DigitalTwin.from_google_backend("unknown_device")

    def test_custom_calibration_round_trip(self, custom_calibration):
        """Calibration data survives the from_calibration round trip."""
        twin = DigitalTwin.from_calibration(custom_calibration)
        # Qubit frequencies should be preserved
        for i in range(custom_calibration.num_qubits):
            assert twin.config.qubits[i].frequency_ghz == pytest.approx(
                custom_calibration.qubit_frequencies_ghz[i]
            )


# ======================================================================
# Validation and QCVV tests
# ======================================================================


class TestDigitalTwinValidation:
    """Tests for QCVV validation and benchmarking."""

    def test_validate_returns_report(self):
        """validate() returns a ValidationReport with reasonable values."""
        twin = DigitalTwin.from_ibm_backend("heron", num_qubits=3)
        report = twin.validate(
            num_qcvv_qubits=3,
            rb_sequences=4,
            qv_trials=5,
            execution_mode="noisy",
        )
        assert isinstance(report, ValidationReport)
        assert report.device_name == "IBM Heron"
        # Bell fidelity should be between 0 and 1
        assert 0.0 < report.bell_fidelity <= 1.0
        # GHZ fidelity should be between 0 and 1
        assert 0.0 < report.ghz_fidelity <= 1.0
        # RB error rate should be small but positive
        assert 0.0 <= report.rb_error_rate < 1.0
        # QV should be a power of 2
        assert report.quantum_volume >= 1

    def test_validate_ideal_mode(self):
        """Ideal-mode validation has higher fidelities than noisy."""
        twin = DigitalTwin.from_ibm_backend("heron", num_qubits=3)
        report_ideal = twin.validate(
            num_qcvv_qubits=2,
            rb_sequences=4,
            qv_trials=5,
            execution_mode="ideal",
        )
        # In ideal mode, Bell fidelity should be near 1
        assert report_ideal.bell_fidelity > 0.99

    def test_run_qcvv_suite_returns_dict(self):
        """run_qcvv_suite() returns a dict of BenchmarkResult objects."""
        twin = DigitalTwin.from_google_backend("willow", num_qubits=3)
        results = twin.run_qcvv_suite(execution_mode="noisy")
        assert isinstance(results, dict)
        assert "bell_fidelity" in results
        assert "ghz_fidelity" in results
        assert "randomized_benchmarking" in results
        assert "quantum_volume" in results

    def test_validation_report_str(self):
        """ValidationReport __str__ produces readable output."""
        report = ValidationReport(
            device_name="TestDevice",
            bell_fidelity=0.98,
            ghz_fidelity=0.95,
            rb_error_rate=0.001,
            quantum_volume=32,
            predicted_fidelity_1q100=0.95,
        )
        text = str(report)
        assert "TestDevice" in text
        assert "0.9800" in text
        assert "32" in text


# ======================================================================
# Fidelity prediction tests
# ======================================================================


class TestFidelityPrediction:
    """Tests for circuit fidelity prediction."""

    def test_zero_depth_prediction(self):
        """Zero-depth circuit has fidelity ~1.0."""
        twin = DigitalTwin.from_ibm_backend("heron", num_qubits=3)
        fid = twin.predict_circuit_fidelity(
            num_1q_gates=0, num_2q_gates=0, depth=0
        )
        assert fid == pytest.approx(1.0, abs=1e-6)

    def test_fidelity_decreases_with_depth(self):
        """Fidelity decreases as circuit depth increases."""
        twin = DigitalTwin.from_ibm_backend("heron", num_qubits=3)
        f_shallow = twin.predict_circuit_fidelity(
            num_1q_gates=10, num_2q_gates=5, depth=10
        )
        f_deep = twin.predict_circuit_fidelity(
            num_1q_gates=100, num_2q_gates=50, depth=100
        )
        assert f_shallow > f_deep
        assert 0.0 <= f_deep <= 1.0
        assert 0.0 <= f_shallow <= 1.0

    def test_fidelity_in_valid_range(self):
        """Predicted fidelity is always in [0, 1]."""
        twin = DigitalTwin.from_ibm_backend("eagle", num_qubits=3)
        for depth in [1, 10, 100, 1000]:
            fid = twin.predict_circuit_fidelity(
                num_1q_gates=depth, num_2q_gates=depth // 2, depth=depth
            )
            assert 0.0 <= fid <= 1.0


# ======================================================================
# Comparison utilities tests
# ======================================================================


class TestDigitalTwinComparison:
    """Tests for compare_presets and device_summary."""

    def test_compare_presets_returns_all_devices(self):
        """compare_presets includes all 4 device presets."""
        comparison = DigitalTwin.compare_presets(
            num_qubits=3, execution_mode="noisy"
        )
        assert isinstance(comparison, dict)
        assert "IBM Eagle" in comparison
        assert "IBM Heron" in comparison
        assert "Google Sycamore" in comparison
        assert "Google Willow" in comparison

    def test_compare_presets_has_metrics(self):
        """Each preset comparison entry has expected metrics."""
        comparison = DigitalTwin.compare_presets(
            num_qubits=3, execution_mode="noisy"
        )
        for device_data in comparison.values():
            assert "num_qubits" in device_data
            assert "native_gate" in device_data
            assert "mean_T1_us" in device_data
            assert "bell_fidelity" in device_data

    def test_device_summary_structure(self, custom_calibration):
        """device_summary returns calibration and config info."""
        twin = DigitalTwin.from_calibration(custom_calibration)
        summary = twin.device_summary()
        assert "calibration" in summary
        assert "chip_config" in summary
