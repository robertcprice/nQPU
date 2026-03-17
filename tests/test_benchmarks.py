"""Comprehensive tests for nqpu.benchmarks package.

Tests cover:
  - BenchmarkCircuit: all circuit generators, gate counting, depth, properties
  - BackendAdapters: IonTrap, Superconducting, NeutralAtom initialization
  - BackendComparison: summary, ranking, best_backend_for
  - CircuitBenchmark: dataclass fields
  - CrossBackendBenchmark: construction, recommendation, legacy runners, report
  - Analysis functions: fidelity_comparison_chart, gate_overhead_analysis,
                        toffoli_advantage_report, scaling_summary
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nqpu.benchmarks import (
    BackendComparison,
    BenchmarkCircuit,
    CircuitBenchmark,
    CrossBackendBenchmark,
    BackendAdapter,
    IonTrapAdapter,
    NeutralAtomAdapter,
    SuperconductingAdapter,
    build_digital_twin,
    fidelity_comparison_chart,
    gate_overhead_analysis,
    scaling_summary,
    toffoli_advantage_report,
)


# -----------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def bell_circuit() -> BenchmarkCircuit:
    """A simple Bell state circuit."""
    return BenchmarkCircuit.bell_state(n_pairs=1)


@pytest.fixture
def ghz_circuit() -> BenchmarkCircuit:
    """A 4-qubit GHZ circuit."""
    return BenchmarkCircuit.ghz_state(4)


@pytest.fixture
def toffoli_circuit() -> BenchmarkCircuit:
    """A Toffoli-heavy circuit."""
    return BenchmarkCircuit.toffoli_heavy(4)


@pytest.fixture
def sample_comparison() -> BackendComparison:
    """A synthetic BackendComparison for testing analysis functions."""
    results = {
        "ion_ideal": CircuitBenchmark(
            backend_name="trapped_ion_ideal",
            execution_mode="ideal",
            probabilities=np.array([0.5, 0.0, 0.0, 0.5]),
            fidelity_vs_ideal=1.0,
            num_gates_1q=1,
            num_gates_2q=1,
            wall_time_ms=5.0,
        ),
        "sc_ideal": CircuitBenchmark(
            backend_name="superconducting_ideal",
            execution_mode="ideal",
            probabilities=np.array([0.5, 0.0, 0.0, 0.5]),
            fidelity_vs_ideal=1.0,
            num_gates_1q=1,
            num_gates_2q=1,
            wall_time_ms=3.0,
        ),
        "na_noisy": CircuitBenchmark(
            backend_name="neutral_atom_noisy",
            execution_mode="noisy",
            probabilities=np.array([0.48, 0.02, 0.02, 0.48]),
            fidelity_vs_ideal=0.96,
            num_gates_1q=1,
            num_gates_2q=1,
            wall_time_ms=4.0,
            num_gates_3q=2,
        ),
    }
    return BackendComparison(
        circuit_name="Bell State", num_qubits=2, results=results
    )


# -----------------------------------------------------------------------
# BenchmarkCircuit tests
# -----------------------------------------------------------------------


class TestBenchmarkCircuit:
    """Tests for BenchmarkCircuit class and its factory methods."""

    def test_bell_state_basic(self, bell_circuit):
        assert bell_circuit.n_qubits == 2
        assert bell_circuit.gate_count == 2  # H + CX
        assert bell_circuit.name == "Bell State"

    def test_bell_multiple_pairs(self):
        bc = BenchmarkCircuit.bell_state(n_pairs=3)
        assert bc.n_qubits == 6
        assert bc.gate_count == 6  # 3*(H + CX)

    def test_ghz_state(self, ghz_circuit):
        assert ghz_circuit.n_qubits == 4
        # 1 H + 3 CX = 4 gates
        assert ghz_circuit.gate_count == 4
        assert ghz_circuit.entangling_gate_count == 3

    def test_ghz_requires_2_qubits(self):
        with pytest.raises(ValueError, match="GHZ state requires >= 2"):
            BenchmarkCircuit.ghz_state(1)

    def test_qft_circuit(self):
        qft = BenchmarkCircuit.qft_circuit(3)
        assert qft.n_qubits == 3
        assert qft.gate_count > 0
        assert "QFT" in qft.name

    def test_random_clifford(self):
        rc = BenchmarkCircuit.random_clifford(3, depth=2, seed=42)
        assert rc.n_qubits == 3
        assert rc.gate_count > 0
        assert "Clifford" in rc.name

    def test_random_clifford_reproducible(self):
        rc1 = BenchmarkCircuit.random_clifford(3, depth=2, seed=42)
        rc2 = BenchmarkCircuit.random_clifford(3, depth=2, seed=42)
        assert len(rc1.gates) == len(rc2.gates)
        for g1, g2 in zip(rc1.gates, rc2.gates):
            assert g1 == g2

    def test_toffoli_heavy(self, toffoli_circuit):
        assert toffoli_circuit.n_qubits == 4
        assert toffoli_circuit.three_qubit_gate_count > 0
        assert toffoli_circuit.toffoli_fraction > 0

    def test_toffoli_heavy_requires_3_qubits(self):
        with pytest.raises(ValueError, match="Toffoli-heavy circuit requires >= 3"):
            BenchmarkCircuit.toffoli_heavy(2)

    def test_qaoa_layer(self):
        qaoa = BenchmarkCircuit.qaoa_layer(4, p=2)
        assert qaoa.n_qubits == 4
        assert qaoa.gate_count > 0
        assert "QAOA" in qaoa.name

    def test_supremacy_circuit(self):
        sc = BenchmarkCircuit.supremacy_circuit(3, depth=2, seed=42)
        assert sc.n_qubits == 3
        assert sc.gate_count > 0
        assert "Supremacy" in sc.name

    def test_depth_computation(self):
        # Bell state: H(0) then CX(0,1) => depth 2
        bc = BenchmarkCircuit.bell_state(1)
        assert bc.depth == 2

    def test_entangling_gate_count(self, bell_circuit):
        assert bell_circuit.entangling_gate_count == 1  # one CX

    def test_toffoli_fraction_zero(self, bell_circuit):
        assert bell_circuit.toffoli_fraction == 0.0

    def test_toffoli_fraction_nonzero(self, toffoli_circuit):
        assert toffoli_circuit.toffoli_fraction > 0.0

    def test_repr(self, bell_circuit):
        r = repr(bell_circuit)
        assert "Bell State" in r
        assert "n_qubits=2" in r


# -----------------------------------------------------------------------
# BackendComparison tests
# -----------------------------------------------------------------------


class TestBackendComparison:
    """Tests for the BackendComparison dataclass."""

    def test_summary_structure(self, sample_comparison):
        s = sample_comparison.summary()
        assert s["circuit"] == "Bell State"
        assert s["num_qubits"] == 2
        assert "backends" in s
        assert "ion_ideal" in s["backends"]

    def test_best_backend_fidelity(self, sample_comparison):
        best = sample_comparison.best_backend_for("fidelity")
        # Both ideal backends have fidelity 1.0, either is valid
        assert best in ("ion_ideal", "sc_ideal")

    def test_best_backend_wall_time(self, sample_comparison):
        best = sample_comparison.best_backend_for("wall_time")
        assert best == "sc_ideal"  # 3.0 ms is fastest

    def test_best_backend_gate_count(self, sample_comparison):
        best = sample_comparison.best_backend_for("gate_count")
        # ion_ideal and sc_ideal both have 1+1=2, na_noisy has 1+1+2=4
        assert best in ("ion_ideal", "sc_ideal")

    def test_best_backend_unknown_metric_raises(self, sample_comparison):
        with pytest.raises(ValueError, match="Unknown metric"):
            sample_comparison.best_backend_for("unknown")

    def test_best_backend_empty_raises(self):
        comp = BackendComparison("Empty", 2, {})
        with pytest.raises(ValueError, match="No results"):
            comp.best_backend_for("fidelity")

    def test_fidelity_ranking(self, sample_comparison):
        ranking = sample_comparison.fidelity_ranking()
        assert len(ranking) == 3
        # First entries should have fidelity 1.0
        assert ranking[0][1] >= ranking[-1][1]

    def test_to_dict(self, sample_comparison):
        d = sample_comparison.to_dict()
        assert d["circuit_name"] == "Bell State"
        assert "results" in d
        for key in ("ion_ideal", "sc_ideal", "na_noisy"):
            assert key in d["results"]
            assert "fidelity_vs_ideal" in d["results"][key]


# -----------------------------------------------------------------------
# CircuitBenchmark tests
# -----------------------------------------------------------------------


class TestCircuitBenchmark:
    """Tests for the CircuitBenchmark dataclass."""

    def test_default_values(self):
        cb = CircuitBenchmark(
            backend_name="test",
            execution_mode="ideal",
            probabilities=np.array([1.0, 0.0]),
            fidelity_vs_ideal=1.0,
            num_gates_1q=5,
            num_gates_2q=3,
            wall_time_ms=1.0,
        )
        assert cb.num_gates_3q == 0
        assert cb.native_gate_counts == {}
        assert cb.estimated_fidelity == 1.0

    def test_extra_field(self):
        cb = CircuitBenchmark(
            backend_name="test",
            execution_mode="noisy",
            probabilities=np.array([0.5, 0.5]),
            fidelity_vs_ideal=0.95,
            num_gates_1q=2,
            num_gates_2q=1,
            wall_time_ms=2.0,
            extra={"three_qubit_gates": 5},
        )
        assert cb.extra["three_qubit_gates"] == 5


# -----------------------------------------------------------------------
# BackendAdapter tests
# -----------------------------------------------------------------------


class TestBackendAdapters:
    """Tests for backend adapter instantiation and names."""

    def test_ion_trap_adapter_name(self):
        adapter = IonTrapAdapter()
        assert adapter.name == "trapped_ion"

    def test_neutral_atom_adapter_name(self):
        adapter = NeutralAtomAdapter()
        assert adapter.name == "neutral_atom"

    def test_superconducting_adapter_name(self):
        adapter = SuperconductingAdapter()
        assert adapter.name == "superconducting"


# -----------------------------------------------------------------------
# CrossBackendBenchmark tests
# -----------------------------------------------------------------------


class TestCrossBackendBenchmark:
    """Tests for the CrossBackendBenchmark orchestrator."""

    def test_construction(self):
        bench = CrossBackendBenchmark(num_qubits=3)
        assert bench.num_qubits == 3

    def test_recommend_toffoli_heavy(self, toffoli_circuit):
        bench = CrossBackendBenchmark(num_qubits=4)
        rec = bench.recommend_backend(toffoli_circuit)
        assert rec == "na"  # Toffoli-heavy -> neutral atom

    def test_recommend_low_entanglement(self):
        # A circuit with mostly single-qubit gates
        gates = [("h", 0), ("h", 1), ("h", 2), ("h", 3)] * 10
        circuit = BenchmarkCircuit(
            name="SQ Heavy",
            description="single-qubit heavy",
            n_qubits=4,
            gates=gates,
        )
        rec = CrossBackendBenchmark(num_qubits=4).recommend_backend(circuit)
        assert rec == "sc"  # Low entanglement -> superconducting

    def test_recommend_high_entanglement(self):
        # Circuit with > 30% entangling gates
        gates = [("cx", 0, 1)] * 7 + [("h", 0)] * 3
        circuit = BenchmarkCircuit(
            name="Entangled",
            description="high entanglement",
            n_qubits=2,
            gates=gates,
        )
        rec = CrossBackendBenchmark(num_qubits=2).recommend_backend(circuit)
        assert rec == "ion"  # High entanglement -> trapped ion

    def test_benchmark_bell_runs(self):
        bench = CrossBackendBenchmark(num_qubits=3)
        comp = bench.benchmark_bell()
        assert isinstance(comp, BackendComparison)
        assert comp.circuit_name == "Bell State"
        assert len(comp.results) > 0

    def test_run_circuit_with_adapters(self):
        bench = CrossBackendBenchmark(num_qubits=3)
        circuit = BenchmarkCircuit.ghz_state(3)
        comp = bench.run_circuit(circuit, backends=["ion"], modes=["ideal"])
        assert "ion_ideal" in comp.results

    def test_run_all_returns_list(self):
        bench = CrossBackendBenchmark(num_qubits=3)
        results = bench.run_all()
        assert isinstance(results, list)
        assert len(results) == 5  # bell, ghz, qft, random, toffoli

    def test_print_report(self):
        bench = CrossBackendBenchmark(num_qubits=3)
        # Use a single fast benchmark
        comp = bench.benchmark_bell()
        report = bench.print_report([comp])
        assert isinstance(report, str)
        assert "BENCHMARK REPORT" in report


# -----------------------------------------------------------------------
# Analysis function tests
# -----------------------------------------------------------------------


class TestAnalysisFunctions:
    """Tests for the analysis and reporting functions."""

    def test_fidelity_comparison_chart(self, sample_comparison):
        chart = fidelity_comparison_chart([sample_comparison])
        assert isinstance(chart, str)
        assert "Bell State" in chart

    def test_gate_overhead_analysis(self, sample_comparison):
        overhead = gate_overhead_analysis([sample_comparison])
        assert "Bell State" in overhead
        assert "ion_ideal" in overhead["Bell State"]
        # ion_ideal: 1 + 1 = 2 total gates
        assert overhead["Bell State"]["ion_ideal"] == 2

    def test_toffoli_advantage_report_no_toffoli(self, sample_comparison):
        report = toffoli_advantage_report([sample_comparison])
        assert "No Toffoli benchmarks found" in report

    def test_toffoli_advantage_report_with_toffoli(self):
        toffoli_results = {
            "na_noisy": CircuitBenchmark(
                backend_name="neutral_atom_noisy",
                execution_mode="noisy",
                probabilities=np.array([0.5, 0.0, 0.0, 0.5]),
                fidelity_vs_ideal=0.95,
                num_gates_1q=2,
                num_gates_2q=0,
                num_gates_3q=2,
                wall_time_ms=4.0,
            ),
            "ion_noisy": CircuitBenchmark(
                backend_name="trapped_ion_noisy",
                execution_mode="noisy",
                probabilities=np.array([0.5, 0.0, 0.0, 0.5]),
                fidelity_vs_ideal=0.90,
                num_gates_1q=10,
                num_gates_2q=12,
                num_gates_3q=0,
                wall_time_ms=8.0,
            ),
        }
        comp = BackendComparison("Toffoli Circuit", 3, toffoli_results)
        report = toffoli_advantage_report([comp])
        assert "Toffoli" in report
        assert "reduction" in report.lower() or "Entangling" in report

    def test_scaling_summary(self):
        results = {
            "3": [
                BackendComparison(
                    "GHZ",
                    3,
                    {
                        "ion_ideal": CircuitBenchmark(
                            backend_name="ion_ideal",
                            execution_mode="ideal",
                            probabilities=np.array([0.5, 0, 0, 0, 0, 0, 0, 0.5]),
                            fidelity_vs_ideal=1.0,
                            num_gates_1q=1,
                            num_gates_2q=2,
                            wall_time_ms=2.0,
                        ),
                    },
                )
            ]
        }
        summary = scaling_summary(results)
        assert "n_qubits = 3" in summary
        assert "fidelity" in summary


# -----------------------------------------------------------------------
# build_digital_twin test
# -----------------------------------------------------------------------


class TestBuildDigitalTwin:
    """Tests for the build_digital_twin utility."""

    def test_builds_chip_config(self):
        config = build_digital_twin(
            frequencies_ghz=[5.0, 5.1],
            t1_us=[80.0, 90.0],
            t2_us=[50.0, 60.0],
            readout_fidelities=[0.99, 0.98],
            edges=[(0, 1)],
        )
        # Should return a ChipConfig-like object with topology
        assert hasattr(config, "topology")
