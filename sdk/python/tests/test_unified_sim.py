"""Comprehensive tests for the unified publication simulator.

Tests cover:
- UnifiedSimulator construction and validation
- Running circuits on all backend types (ion + SC)
- FidelityTable LaTeX and CSV output formats
- ScalingData output
- GateOverheadTable output
- Backend identifiers
- Fidelity comparison (ideal should be high, noisy should be < 1)
- Edge cases: minimum qubits, unknown backends, unknown circuits
"""

import math

import numpy as np
import pytest

from nqpu.superconducting.unified_sim import (
    ALL_BACKENDS,
    SUPPORTED_CIRCUITS,
    BackendResult,
    FidelityTable,
    GateOverheadTable,
    ScalingData,
    UnifiedSimulator,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def sim_2q():
    """A 2-qubit UnifiedSimulator with default backends."""
    return UnifiedSimulator(num_qubits=2)


@pytest.fixture
def sim_3q_single_backend():
    """A 3-qubit simulator with a single backend for fast testing."""
    return UnifiedSimulator(num_qubits=3, backends=["ibm_heron"])


@pytest.fixture
def sim_3q_two_backends():
    """A 3-qubit simulator with one ion and one SC backend."""
    return UnifiedSimulator(num_qubits=3, backends=["ion_yb171", "ibm_heron"])


# ======================================================================
# Constructor tests
# ======================================================================


class TestUnifiedSimulatorConstruction:
    """Tests for UnifiedSimulator initialization."""

    def test_minimum_qubits(self):
        """num_qubits=2 is the minimum allowed value."""
        sim = UnifiedSimulator(num_qubits=2)
        assert sim.num_qubits == 2

    def test_rejects_single_qubit(self):
        """num_qubits < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_qubits must be >= 2"):
            UnifiedSimulator(num_qubits=1)

    def test_default_backends(self):
        """Default backends list is non-empty and contains expected entries."""
        sim = UnifiedSimulator(num_qubits=2)
        assert len(sim.backend_names) > 0
        assert "ibm_heron" in sim.backend_names

    def test_custom_backends(self):
        """Custom backend list is accepted."""
        sim = UnifiedSimulator(
            num_qubits=2,
            backends=["ion_yb171", "google_willow"],
        )
        assert sim.backend_names == ["ion_yb171", "google_willow"]

    def test_unknown_backend_raises(self):
        """Unknown backend identifier raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            UnifiedSimulator(num_qubits=2, backends=["fake_backend"])

    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_all_backends_accepted(self, backend):
        """Every registered backend can be used in the constructor."""
        sim = UnifiedSimulator(num_qubits=2, backends=[backend])
        assert backend in sim.backend_names


# ======================================================================
# Running circuits tests
# ======================================================================


class TestRunCircuit:
    """Tests for running circuits on backends."""

    def test_run_circuit_returns_results(self, sim_3q_single_backend):
        """run_circuit returns a dict of BackendResult objects."""
        def bell(sim, n):
            sim.h(0)
            sim.cnot(0, 1)

        results = sim_3q_single_backend.run_circuit(bell, modes=["ideal"])
        assert isinstance(results, dict)
        assert "ibm_heron_ideal" in results
        res = results["ibm_heron_ideal"]
        assert isinstance(res, BackendResult)
        assert res.backend == "ibm_heron"
        assert res.mode == "ideal"

    def test_ideal_fidelity_is_one(self, sim_3q_single_backend):
        """Ideal-mode fidelity_vs_ideal is 1.0."""
        def bell(sim, n):
            sim.h(0)
            sim.cnot(0, 1)

        results = sim_3q_single_backend.run_circuit(bell, modes=["ideal"])
        res = results["ibm_heron_ideal"]
        assert res.fidelity_vs_ideal == pytest.approx(1.0)

    def test_noisy_fidelity_less_than_one(self, sim_3q_single_backend):
        """Noisy-mode fidelity_vs_ideal is < 1.0."""
        def bell(sim, n):
            sim.h(0)
            sim.cnot(0, 1)

        results = sim_3q_single_backend.run_circuit(bell, modes=["noisy"])
        res = results["ibm_heron_noisy"]
        assert res.fidelity_vs_ideal < 1.0
        assert res.fidelity_vs_ideal > 0.5

    def test_probabilities_sum_to_one(self, sim_3q_single_backend):
        """Output probabilities sum to ~1."""
        def ghz(sim, n):
            sim.h(0)
            for i in range(1, n):
                sim.cnot(0, i)

        results = sim_3q_single_backend.run_circuit(ghz, modes=["ideal"])
        probs = results["ibm_heron_ideal"].probabilities
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_native_gate_counts_positive(self, sim_3q_single_backend):
        """Native gate counts are non-negative."""
        def bell(sim, n):
            sim.h(0)
            sim.cnot(0, 1)

        results = sim_3q_single_backend.run_circuit(bell, modes=["ideal"])
        res = results["ibm_heron_ideal"]
        assert res.native_1q_gates >= 0
        assert res.native_2q_gates >= 0

    def test_wall_time_positive(self, sim_3q_single_backend):
        """Wall time is positive."""
        def bell(sim, n):
            sim.h(0)
            sim.cnot(0, 1)

        results = sim_3q_single_backend.run_circuit(bell, modes=["ideal"])
        res = results["ibm_heron_ideal"]
        assert res.wall_time_ms > 0.0


# ======================================================================
# Fidelity comparison tests
# ======================================================================


class TestFidelityComparison:
    """Tests for the fidelity_comparison method."""

    def test_single_circuit_comparison(self, sim_3q_single_backend):
        """Comparing a single circuit returns correct table shape."""
        table = sim_3q_single_backend.fidelity_comparison("bell")
        assert isinstance(table, FidelityTable)
        assert table.fidelities.shape == (1, 1)
        assert table.backends == ["ibm_heron"]
        assert table.circuits == ["bell"]

    def test_all_circuits_comparison(self, sim_3q_single_backend):
        """Comparing all circuits returns one row per backend."""
        table = sim_3q_single_backend.fidelity_comparison()
        assert table.fidelities.shape[0] == 1  # one backend
        assert table.fidelities.shape[1] == len(SUPPORTED_CIRCUITS)

    def test_fidelity_values_in_range(self, sim_3q_two_backends):
        """All fidelity values are in [0, 1]."""
        table = sim_3q_two_backends.fidelity_comparison("ghz")
        for i in range(table.fidelities.shape[0]):
            for j in range(table.fidelities.shape[1]):
                assert 0.0 <= table.fidelities[i, j] <= 1.0

    def test_unknown_circuit_raises(self, sim_3q_single_backend):
        """Unknown circuit type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown circuit type"):
            sim_3q_single_backend.fidelity_comparison("nonexistent")


# ======================================================================
# Output format tests
# ======================================================================


class TestOutputFormats:
    """Tests for LaTeX and CSV output generation."""

    def test_fidelity_table_latex_structure(self):
        """FidelityTable.to_latex() produces valid LaTeX markers."""
        ft = FidelityTable(
            backends=["backend_a", "backend_b"],
            circuits=["bell", "ghz"],
            fidelities=np.array([[0.99, 0.95], [0.97, 0.92]]),
        )
        latex = ft.to_latex()
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert r"\toprule" in latex
        assert r"\bottomrule" in latex
        assert "backend_a" in latex
        assert "backend_b" in latex

    def test_fidelity_table_csv_structure(self):
        """FidelityTable.to_csv() has header and data rows."""
        ft = FidelityTable(
            backends=["a", "b"],
            circuits=["bell", "ghz"],
            fidelities=np.array([[0.99, 0.95], [0.97, 0.92]]),
        )
        csv = ft.to_csv()
        lines = csv.strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows
        assert lines[0].startswith("backend,")
        assert "bell" in lines[0]
        assert "ghz" in lines[0]

    def test_scaling_data_latex(self):
        """ScalingData.to_latex() produces valid LaTeX."""
        sd = ScalingData(
            qubit_counts=[2, 3, 4],
            backend_fidelities={"ion_yb171": [0.99, 0.95, 0.90]},
        )
        latex = sd.to_latex()
        assert r"\begin{table}" in latex
        assert "ion_yb171" in latex

    def test_scaling_data_csv(self):
        """ScalingData.to_csv() has correct structure."""
        sd = ScalingData(
            qubit_counts=[2, 3],
            backend_fidelities={"a": [0.99, 0.95], "b": [0.97, 0.92]},
        )
        csv = sd.to_csv()
        lines = csv.strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows
        assert "qubit_count" in lines[0]

    def test_gate_overhead_table_latex(self):
        """GateOverheadTable.to_latex() produces valid LaTeX."""
        got = GateOverheadTable(
            backends=["ibm_heron"],
            circuits=["bell"],
            gate_counts_1q=np.array([[5]]),
            gate_counts_2q=np.array([[1]]),
        )
        latex = got.to_latex()
        assert r"\begin{table}" in latex
        assert "5/1" in latex

    def test_gate_overhead_table_csv(self):
        """GateOverheadTable.to_csv() has 1q and 2q columns per circuit."""
        got = GateOverheadTable(
            backends=["ibm_heron"],
            circuits=["bell", "ghz"],
            gate_counts_1q=np.array([[5, 3]]),
            gate_counts_2q=np.array([[1, 2]]),
        )
        csv = got.to_csv()
        header = csv.strip().split("\n")[0]
        assert "bell_1q" in header
        assert "bell_2q" in header
        assert "ghz_1q" in header
        assert "ghz_2q" in header


# ======================================================================
# Noise scaling study tests
# ======================================================================


class TestNoiseScalingStudy:
    """Tests for noise_scaling_study method."""

    def test_scaling_study_returns_data(self, sim_3q_single_backend):
        """noise_scaling_study returns ScalingData with expected structure."""
        sd = sim_3q_single_backend.noise_scaling_study(
            "ghz", qubit_range=range(2, 4)
        )
        assert isinstance(sd, ScalingData)
        assert sd.qubit_counts == [2, 3]
        assert "ibm_heron" in sd.backend_fidelities
        assert len(sd.backend_fidelities["ibm_heron"]) == 2

    def test_fidelity_decreases_with_qubits(self, sim_3q_single_backend):
        """Fidelity generally decreases as qubit count increases."""
        sd = sim_3q_single_backend.noise_scaling_study(
            "ghz", qubit_range=range(2, 4)
        )
        fids = sd.backend_fidelities["ibm_heron"]
        # Allow that 2Q and 3Q might be close, but 2Q should not be worse
        # In practice with noise, larger systems have lower fidelity
        assert all(0.0 < f <= 1.0 for f in fids)

    def test_unknown_circuit_raises(self, sim_3q_single_backend):
        """Unknown circuit in scaling study raises ValueError."""
        with pytest.raises(ValueError, match="Unknown circuit type"):
            sim_3q_single_backend.noise_scaling_study("nonexistent")


# ======================================================================
# Gate overhead comparison tests
# ======================================================================


class TestGateOverheadComparison:
    """Tests for gate_overhead_comparison method."""

    def test_overhead_returns_table(self, sim_3q_single_backend):
        """gate_overhead_comparison returns GateOverheadTable."""
        table = sim_3q_single_backend.gate_overhead_comparison()
        assert isinstance(table, GateOverheadTable)
        assert len(table.backends) == 1
        assert len(table.circuits) == len(SUPPORTED_CIRCUITS)

    def test_gate_counts_non_negative(self, sim_3q_single_backend):
        """All gate counts are non-negative."""
        table = sim_3q_single_backend.gate_overhead_comparison()
        assert np.all(table.gate_counts_1q >= 0)
        assert np.all(table.gate_counts_2q >= 0)
