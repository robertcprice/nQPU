"""Comprehensive tests for the cross-backend benchmarking module.

Covers:
- BenchmarkCircuit construction and all standard circuit generators
- CircuitBenchmark (result dataclass) creation and field access
- BackendComparison data handling, ranking, and serialisation
- BackendAdapter implementations (ion, neutral atom, superconducting)
- CrossBackendBenchmark orchestration with real and mock backends
- Generic adapter-based run_circuit / run_all
- Scaling analysis
- Backend recommendation heuristics
- Report formatting
- Analysis functions (fidelity chart, gate overhead, Toffoli report)
- Gate count and depth calculations
- Edge cases and error handling
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from nqpu.benchmarks.cross_backend import (
    BackendAdapter,
    BackendComparison,
    BenchmarkCircuit,
    CircuitBenchmark,
    CrossBackendBenchmark,
    IonTrapAdapter,
    NeutralAtomAdapter,
    SuperconductingAdapter,
    _circuit_depth,
    _count_entangling_gates,
    _count_three_qubit_gates,
    _toffoli_decomposed,
    build_digital_twin,
    fidelity_comparison_chart,
    gate_overhead_analysis,
    scaling_summary,
    toffoli_advantage_report,
)


# ======================================================================
# BenchmarkCircuit construction and generators
# ======================================================================


class TestBenchmarkCircuitConstruction:
    """Tests for BenchmarkCircuit construction and properties."""

    def test_manual_construction(self):
        circuit = BenchmarkCircuit(
            name="Test",
            description="A test circuit",
            n_qubits=2,
            gates=[("h", 0), ("cx", 0, 1)],
        )
        assert circuit.name == "Test"
        assert circuit.description == "A test circuit"
        assert circuit.n_qubits == 2
        assert circuit.gate_count == 2

    def test_gate_count_property(self):
        circuit = BenchmarkCircuit("T", "d", 3, [("h", 0), ("x", 1), ("cx", 0, 1)])
        assert circuit.gate_count == 3

    def test_depth_single_qubit_only(self):
        circuit = BenchmarkCircuit("T", "d", 2, [("h", 0), ("x", 0), ("y", 1)])
        assert circuit.depth == 2

    def test_depth_two_qubit_gates(self):
        circuit = BenchmarkCircuit(
            "T", "d", 3, [("h", 0), ("cx", 0, 1), ("cx", 1, 2)]
        )
        assert circuit.depth >= 2

    def test_entangling_gate_count(self):
        circuit = BenchmarkCircuit(
            "T", "d", 3, [("h", 0), ("cx", 0, 1), ("cz", 1, 2), ("x", 2)]
        )
        assert circuit.entangling_gate_count == 2

    def test_three_qubit_gate_count(self):
        circuit = BenchmarkCircuit(
            "T", "d", 3, [("h", 0), ("ccx", 0, 1, 2)]
        )
        assert circuit.three_qubit_gate_count == 1

    def test_toffoli_fraction(self):
        circuit = BenchmarkCircuit(
            "T", "d", 3, [("h", 0), ("ccx", 0, 1, 2), ("x", 1), ("ccx", 0, 1, 2)]
        )
        assert circuit.toffoli_fraction == pytest.approx(0.5)

    def test_toffoli_fraction_zero_gates(self):
        circuit = BenchmarkCircuit("T", "d", 1, [])
        assert circuit.toffoli_fraction == 0.0

    def test_repr(self):
        circuit = BenchmarkCircuit("Bell", "test", 2, [("h", 0), ("cx", 0, 1)])
        r = repr(circuit)
        assert "Bell" in r
        assert "n_qubits=2" in r

    def test_empty_circuit(self):
        circuit = BenchmarkCircuit("Empty", "nothing", 1, [])
        assert circuit.gate_count == 0
        assert circuit.depth == 0
        assert circuit.entangling_gate_count == 0


class TestBenchmarkCircuitGenerators:
    """Tests for standard circuit generators."""

    def test_bell_state_single_pair(self):
        circuit = BenchmarkCircuit.bell_state(n_pairs=1)
        assert circuit.name == "Bell State"
        assert circuit.n_qubits == 2
        assert circuit.gate_count == 2
        assert circuit.gates[0] == ("h", 0)
        assert circuit.gates[1] == ("cx", 0, 1)

    def test_bell_state_multiple_pairs(self):
        circuit = BenchmarkCircuit.bell_state(n_pairs=3)
        assert circuit.n_qubits == 6
        assert circuit.gate_count == 6

    def test_ghz_state_basic(self):
        circuit = BenchmarkCircuit.ghz_state(4)
        assert circuit.n_qubits == 4
        assert circuit.gates[0] == ("h", 0)
        assert circuit.entangling_gate_count == 3

    def test_ghz_state_minimum_qubits(self):
        circuit = BenchmarkCircuit.ghz_state(2)
        assert circuit.n_qubits == 2
        assert circuit.gate_count == 2

    def test_ghz_state_rejects_one_qubit(self):
        with pytest.raises(ValueError, match="requires >= 2"):
            BenchmarkCircuit.ghz_state(1)

    def test_qft_circuit_structure(self):
        circuit = BenchmarkCircuit.qft_circuit(3)
        assert circuit.name == "QFT"
        assert circuit.n_qubits == 3
        assert circuit.gate_count > 0

    def test_qft_circuit_single_qubit(self):
        circuit = BenchmarkCircuit.qft_circuit(1)
        assert circuit.n_qubits == 1
        assert circuit.gate_count == 1
        assert circuit.gates[0] == ("h", 0)

    def test_qft_circuit_has_swaps(self):
        circuit = BenchmarkCircuit.qft_circuit(4)
        swap_gates = [g for g in circuit.gates if g[0] == "swap"]
        assert len(swap_gates) == 2

    def test_random_clifford_reproducible(self):
        c1 = BenchmarkCircuit.random_clifford(3, depth=2, seed=42)
        c2 = BenchmarkCircuit.random_clifford(3, depth=2, seed=42)
        assert c1.gates == c2.gates

    def test_random_clifford_different_seeds(self):
        c1 = BenchmarkCircuit.random_clifford(3, depth=2, seed=1)
        c2 = BenchmarkCircuit.random_clifford(3, depth=2, seed=2)
        assert c1.gates != c2.gates

    def test_random_clifford_has_entangling(self):
        circuit = BenchmarkCircuit.random_clifford(4, depth=3, seed=42)
        assert circuit.entangling_gate_count > 0

    def test_toffoli_heavy_basic(self):
        circuit = BenchmarkCircuit.toffoli_heavy(4)
        assert circuit.n_qubits == 4
        assert circuit.three_qubit_gate_count == 2
        assert circuit.toffoli_fraction > 0

    def test_toffoli_heavy_custom_count(self):
        circuit = BenchmarkCircuit.toffoli_heavy(5, n_toffolis=5)
        assert circuit.three_qubit_gate_count == 5

    def test_toffoli_heavy_rejects_small(self):
        with pytest.raises(ValueError, match="requires >= 3"):
            BenchmarkCircuit.toffoli_heavy(2)

    def test_qaoa_layer_basic(self):
        circuit = BenchmarkCircuit.qaoa_layer(3, p=1)
        assert circuit.name == "QAOA"
        assert circuit.n_qubits == 3
        assert circuit.gate_count > 0

    def test_qaoa_layer_multiple_layers(self):
        c1 = BenchmarkCircuit.qaoa_layer(3, p=1)
        c2 = BenchmarkCircuit.qaoa_layer(3, p=3)
        assert c2.gate_count > c1.gate_count

    def test_qaoa_has_entangling_gates(self):
        circuit = BenchmarkCircuit.qaoa_layer(4, p=2)
        assert circuit.entangling_gate_count > 0

    def test_supremacy_circuit_basic(self):
        circuit = BenchmarkCircuit.supremacy_circuit(4, depth=3, seed=42)
        assert circuit.name == "Supremacy Circuit"
        assert circuit.n_qubits == 4
        assert circuit.gate_count > 0

    def test_supremacy_circuit_reproducible(self):
        c1 = BenchmarkCircuit.supremacy_circuit(3, depth=2, seed=7)
        c2 = BenchmarkCircuit.supremacy_circuit(3, depth=2, seed=7)
        assert c1.gates == c2.gates

    def test_supremacy_circuit_has_cz(self):
        circuit = BenchmarkCircuit.supremacy_circuit(4, depth=3, seed=42)
        cz_gates = [g for g in circuit.gates if g[0] == "cz"]
        assert len(cz_gates) > 0


# ======================================================================
# Circuit utility functions
# ======================================================================


class TestCircuitUtilities:
    """Tests for standalone circuit utility functions."""

    def test_circuit_depth_empty(self):
        assert _circuit_depth([], 2) == 0

    def test_circuit_depth_single_gate(self):
        assert _circuit_depth([("h", 0)], 1) == 1

    def test_circuit_depth_parallel(self):
        depth = _circuit_depth([("h", 0), ("x", 1)], 2)
        assert depth == 1

    def test_circuit_depth_sequential(self):
        depth = _circuit_depth([("h", 0), ("x", 0)], 1)
        assert depth == 2

    def test_count_entangling_no_entangling(self):
        assert _count_entangling_gates([("h", 0), ("x", 1)]) == 0

    def test_count_entangling_mixed(self):
        gates = [("h", 0), ("cx", 0, 1), ("ccx", 0, 1, 2)]
        assert _count_entangling_gates(gates) == 2

    def test_count_three_qubit(self):
        gates = [("ccx", 0, 1, 2), ("cx", 0, 1), ("ccz", 0, 1, 2)]
        assert _count_three_qubit_gates(gates) == 2


# ======================================================================
# CircuitBenchmark (result dataclass)
# ======================================================================


class TestCircuitBenchmarkResult:
    """Tests for the CircuitBenchmark result dataclass."""

    def test_basic_construction(self):
        res = CircuitBenchmark(
            backend_name="test_ideal",
            execution_mode="ideal",
            probabilities=np.array([0.5, 0.5]),
            fidelity_vs_ideal=1.0,
            num_gates_1q=3,
            num_gates_2q=1,
            wall_time_ms=1.5,
        )
        assert res.backend_name == "test_ideal"
        assert res.fidelity_vs_ideal == 1.0
        assert res.num_gates_1q == 3
        assert res.num_gates_2q == 1

    def test_default_fields(self):
        res = CircuitBenchmark(
            backend_name="test",
            execution_mode="ideal",
            probabilities=np.array([1.0]),
            fidelity_vs_ideal=1.0,
            num_gates_1q=0,
            num_gates_2q=0,
            wall_time_ms=0.1,
        )
        assert res.num_gates_3q == 0
        assert res.estimated_fidelity == 1.0
        assert res.extra == {}
        assert res.native_gate_counts == {}

    def test_extra_field(self):
        res = CircuitBenchmark(
            backend_name="test",
            execution_mode="noisy",
            probabilities=np.array([0.6, 0.4]),
            fidelity_vs_ideal=0.9,
            num_gates_1q=2,
            num_gates_2q=1,
            wall_time_ms=2.0,
            extra={"three_qubit_gates": 5},
        )
        assert res.extra["three_qubit_gates"] == 5


# ======================================================================
# BackendComparison
# ======================================================================


class TestBackendComparison:
    """Tests for BackendComparison ranking and serialisation."""

    @pytest.fixture
    def sample_comparison(self):
        return BackendComparison(
            circuit_name="Test Circuit",
            num_qubits=3,
            results={
                "ion_ideal": CircuitBenchmark(
                    backend_name="ion_ideal",
                    execution_mode="ideal",
                    probabilities=np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0]),
                    fidelity_vs_ideal=1.0,
                    num_gates_1q=5,
                    num_gates_2q=3,
                    wall_time_ms=2.0,
                ),
                "sc_ideal": CircuitBenchmark(
                    backend_name="sc_ideal",
                    execution_mode="ideal",
                    probabilities=np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0]),
                    fidelity_vs_ideal=0.95,
                    num_gates_1q=4,
                    num_gates_2q=2,
                    wall_time_ms=1.0,
                ),
                "na_ideal": CircuitBenchmark(
                    backend_name="na_ideal",
                    execution_mode="ideal",
                    probabilities=np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0]),
                    fidelity_vs_ideal=0.98,
                    num_gates_1q=3,
                    num_gates_2q=1,
                    num_gates_3q=1,
                    wall_time_ms=1.5,
                ),
            },
        )

    def test_summary_structure(self, sample_comparison):
        s = sample_comparison.summary()
        assert s["circuit"] == "Test Circuit"
        assert s["num_qubits"] == 3
        assert "ion_ideal" in s["backends"]
        assert "fidelity" in s["backends"]["ion_ideal"]

    def test_best_backend_fidelity(self, sample_comparison):
        best = sample_comparison.best_backend_for("fidelity")
        assert best == "ion_ideal"

    def test_best_backend_gate_count(self, sample_comparison):
        best = sample_comparison.best_backend_for("gate_count")
        assert best == "na_ideal"

    def test_best_backend_wall_time(self, sample_comparison):
        best = sample_comparison.best_backend_for("wall_time")
        assert best == "sc_ideal"

    def test_best_backend_unknown_metric(self, sample_comparison):
        with pytest.raises(ValueError, match="Unknown metric"):
            sample_comparison.best_backend_for("unknown")

    def test_best_backend_empty_results(self):
        comp = BackendComparison("Empty", 2, {})
        with pytest.raises(ValueError, match="No results"):
            comp.best_backend_for("fidelity")

    def test_fidelity_ranking(self, sample_comparison):
        ranking = sample_comparison.fidelity_ranking()
        assert ranking[0][0] == "ion_ideal"
        assert ranking[0][1] == 1.0
        assert ranking[-1][0] == "sc_ideal"

    def test_to_dict(self, sample_comparison):
        d = sample_comparison.to_dict()
        assert d["circuit_name"] == "Test Circuit"
        assert "ion_ideal" in d["results"]
        assert "fidelity_vs_ideal" in d["results"]["ion_ideal"]

    def test_repr(self, sample_comparison):
        r = repr(sample_comparison)
        assert "Test Circuit" in r


# ======================================================================
# Backend adapters with real simulators (small circuits)
# ======================================================================


class TestIonTrapAdapter:
    """Tests for IonTrapAdapter with real simulators."""

    def test_name(self):
        adapter = IonTrapAdapter()
        assert adapter.name == "trapped_ion"

    def test_create_simulator_ideal(self):
        adapter = IonTrapAdapter()
        sim = adapter.create_simulator(2, "ideal")
        assert sim.n_qubits == 2
        assert sim.execution_mode == "ideal"

    def test_run_bell_circuit(self):
        adapter = IonTrapAdapter()
        sim = adapter.create_simulator(2, "ideal")
        adapter.run_circuit(sim, [("h", 0), ("cx", 0, 1)])
        probs = adapter.get_probabilities(sim)
        assert probs[0] == pytest.approx(0.5, abs=0.01)
        assert probs[3] == pytest.approx(0.5, abs=0.01)

    def test_get_native_gate_count(self):
        adapter = IonTrapAdapter()
        sim = adapter.create_simulator(2, "ideal")
        adapter.run_circuit(sim, [("h", 0), ("cx", 0, 1)])
        counts = adapter.get_native_gate_count(sim)
        assert counts["total"] > 0
        assert "single_qubit" in counts
        assert "two_qubit" in counts

    def test_fidelity_estimate_ideal(self):
        adapter = IonTrapAdapter()
        sim = adapter.create_simulator(2, "ideal")
        adapter.run_circuit(sim, [("h", 0)])
        fid = adapter.get_fidelity_estimate(sim)
        assert fid == 1.0

    def test_noisy_mode_probabilities(self):
        adapter = IonTrapAdapter()
        sim = adapter.create_simulator(2, "noisy")
        adapter.run_circuit(sim, [("h", 0), ("cx", 0, 1)])
        probs = adapter.get_probabilities(sim)
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)


class TestNeutralAtomAdapter:
    """Tests for NeutralAtomAdapter with real simulators."""

    def test_name(self):
        adapter = NeutralAtomAdapter()
        assert adapter.name == "neutral_atom"

    def test_create_simulator(self):
        adapter = NeutralAtomAdapter()
        sim = adapter.create_simulator(3, "ideal")
        assert sim.n_qubits == 3

    def test_run_toffoli_native(self):
        adapter = NeutralAtomAdapter()
        sim = adapter.create_simulator(3, "ideal")
        adapter.run_circuit(sim, [("x", 0), ("x", 1), ("ccx", 0, 1, 2)])
        probs = adapter.get_probabilities(sim)
        # |110> -> Toffoli -> |111>
        assert probs[7] == pytest.approx(1.0, abs=0.01)

    def test_native_gate_count_includes_three_qubit(self):
        adapter = NeutralAtomAdapter()
        sim = adapter.create_simulator(3, "ideal")
        adapter.run_circuit(sim, [("ccx", 0, 1, 2)])
        counts = adapter.get_native_gate_count(sim)
        assert counts.get("three_qubit", 0) > 0


class TestSuperconductingAdapter:
    """Tests for SuperconductingAdapter with real simulators."""

    def test_name(self):
        adapter = SuperconductingAdapter()
        assert adapter.name == "superconducting"

    def test_create_simulator(self):
        adapter = SuperconductingAdapter()
        sim = adapter.create_simulator(2, "ideal")
        assert sim.n_qubits == 2

    def test_run_bell_circuit(self):
        adapter = SuperconductingAdapter()
        sim = adapter.create_simulator(2, "ideal")
        adapter.run_circuit(sim, [("h", 0), ("cx", 0, 1)])
        probs = adapter.get_probabilities(sim)
        assert probs[0] == pytest.approx(0.5, abs=0.01)
        assert probs[3] == pytest.approx(0.5, abs=0.01)

    def test_swap_gate(self):
        adapter = SuperconductingAdapter()
        sim = adapter.create_simulator(2, "ideal")
        adapter.run_circuit(sim, [("x", 0), ("swap", 0, 1)])
        probs = adapter.get_probabilities(sim)
        # |01> (qubit 0 = MSB) -> SWAP -> |10> = index 2
        assert probs[2] == pytest.approx(1.0, abs=0.01)


# ======================================================================
# CrossBackendBenchmark orchestrator
# ======================================================================


class TestCrossBackendBenchmarkNew:
    """Tests for the new adapter-based CrossBackendBenchmark methods."""

    @pytest.fixture
    def bench(self):
        return CrossBackendBenchmark(num_qubits=2)

    def test_adapters_registered(self, bench):
        assert "ion" in bench._adapters
        assert "sc" in bench._adapters
        assert "na" in bench._adapters

    def test_add_custom_backend(self, bench):
        mock_adapter = MagicMock(spec=BackendAdapter)
        mock_adapter.name = "custom"
        bench.add_backend("custom", mock_adapter)
        assert "custom" in bench._adapters

    def test_run_circuit_bell(self, bench):
        circuit = BenchmarkCircuit.bell_state(n_pairs=1)
        comp = bench.run_circuit(circuit)
        assert isinstance(comp, BackendComparison)
        assert comp.circuit_name == "Bell State"
        assert len(comp.results) == 6  # 3 backends * 2 modes

    def test_run_circuit_specific_backends(self, bench):
        circuit = BenchmarkCircuit.bell_state(n_pairs=1)
        comp = bench.run_circuit(circuit, backends=["ion"])
        assert len(comp.results) == 2  # ideal + noisy
        assert all("ion" in k for k in comp.results.keys())

    def test_run_circuit_ideal_only(self, bench):
        circuit = BenchmarkCircuit.bell_state(n_pairs=1)
        comp = bench.run_circuit(circuit, modes=["ideal"])
        assert len(comp.results) == 3  # 3 backends
        for res in comp.results.values():
            assert res.fidelity_vs_ideal == pytest.approx(1.0)

    def test_run_circuit_ghz(self):
        bench = CrossBackendBenchmark(num_qubits=3)
        circuit = BenchmarkCircuit.ghz_state(3)
        comp = bench.run_circuit(circuit, modes=["ideal"])
        for res in comp.results.values():
            assert res.fidelity_vs_ideal == pytest.approx(1.0)

    def test_run_circuit_qft(self):
        bench = CrossBackendBenchmark(num_qubits=2)
        circuit = BenchmarkCircuit.qft_circuit(2)
        comp = bench.run_circuit(circuit, modes=["ideal"])
        for res in comp.results.values():
            assert res.fidelity_vs_ideal == pytest.approx(1.0)

    def test_recommend_toffoli_heavy(self):
        bench = CrossBackendBenchmark(num_qubits=4)
        circuit = BenchmarkCircuit.toffoli_heavy(4, n_toffolis=5)
        assert bench.recommend_backend(circuit) == "na"

    def test_recommend_entangling_heavy(self):
        bench = CrossBackendBenchmark(num_qubits=3)
        circuit = BenchmarkCircuit.ghz_state(3)
        assert bench.recommend_backend(circuit) == "ion"

    def test_recommend_parallel_circuit(self):
        bench = CrossBackendBenchmark(num_qubits=4)
        # A circuit with mostly single-qubit gates
        gates = [("h", i) for i in range(4)] + [("rz", i, 0.1) for i in range(4)]
        circuit = BenchmarkCircuit("Parallel", "test", 4, gates)
        assert bench.recommend_backend(circuit) == "sc"


class TestScalingAnalysis:
    """Tests for scaling analysis functionality."""

    def test_scaling_analysis_returns_dict(self):
        bench = CrossBackendBenchmark(num_qubits=2)
        results = bench.run_scaling_analysis(
            circuit_gen=BenchmarkCircuit.ghz_state,
            qubit_range=[2, 3],
            backends=["ion"],
        )
        assert "2" in results
        assert "3" in results
        assert isinstance(results["2"][0], BackendComparison)

    def test_scaling_analysis_increasing_qubits(self):
        bench = CrossBackendBenchmark(num_qubits=2)
        results = bench.run_scaling_analysis(
            circuit_gen=BenchmarkCircuit.ghz_state,
            qubit_range=[2, 3, 4],
            backends=["sc"],
        )
        assert len(results) == 3


# ======================================================================
# Report formatting
# ======================================================================


class TestReportFormatting:
    """Tests for report generation."""

    def test_print_report_returns_string(self):
        bench = CrossBackendBenchmark(num_qubits=2)
        circuit = BenchmarkCircuit.bell_state(n_pairs=1)
        comp = bench.run_circuit(circuit, modes=["ideal"])
        report = CrossBackendBenchmark.print_report([comp])
        assert isinstance(report, str)
        assert "Bell State" in report

    def test_print_report_multiple_circuits(self):
        bench = CrossBackendBenchmark(num_qubits=2)
        c1 = BenchmarkCircuit.bell_state(n_pairs=1)
        c2 = BenchmarkCircuit.ghz_state(2)
        comp1 = bench.run_circuit(c1, modes=["ideal"])
        comp2 = bench.run_circuit(c2, modes=["ideal"])
        report = CrossBackendBenchmark.print_report([comp1, comp2])
        assert "Bell State" in report
        assert "GHZ State" in report


# ======================================================================
# Analysis functions
# ======================================================================


class TestAnalysisFunctions:
    """Tests for standalone analysis functions."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for analysis tests."""
        bench = CrossBackendBenchmark(num_qubits=2)
        c1 = BenchmarkCircuit.bell_state(n_pairs=1)
        return [bench.run_circuit(c1, modes=["ideal"])]

    def test_fidelity_comparison_chart(self, sample_results):
        chart = fidelity_comparison_chart(sample_results)
        assert isinstance(chart, str)
        assert "Bell State" in chart

    def test_gate_overhead_analysis(self, sample_results):
        overhead = gate_overhead_analysis(sample_results)
        assert "Bell State" in overhead
        for key, total in overhead["Bell State"].items():
            assert total >= 0

    def test_scaling_summary_format(self):
        data = {
            "2": [
                BackendComparison(
                    "GHZ",
                    2,
                    {
                        "ion_ideal": CircuitBenchmark(
                            "ion", "ideal", np.array([0.5, 0.5, 0, 0]),
                            1.0, 3, 1, 1.0,
                        )
                    },
                )
            ]
        }
        text = scaling_summary(data)
        assert "n_qubits = 2" in text
        assert "fidelity=" in text

    def test_toffoli_advantage_report_no_toffoli(self, sample_results):
        report = toffoli_advantage_report(sample_results)
        assert "No Toffoli benchmarks" in report

    def test_toffoli_advantage_report_with_toffoli(self):
        bench = CrossBackendBenchmark(num_qubits=3)
        circuit = BenchmarkCircuit.toffoli_heavy(3)
        comp = bench.run_circuit(circuit, modes=["ideal"])
        report = toffoli_advantage_report([comp])
        assert "Toffoli" in report


# ======================================================================
# Digital twin builder
# ======================================================================


class TestBuildDigitalTwin:
    """Tests for the build_digital_twin utility."""

    def test_basic_construction(self):
        config = build_digital_twin(
            frequencies_ghz=[5.0, 5.1],
            t1_us=[100.0, 120.0],
            t2_us=[80.0, 100.0],
            readout_fidelities=[0.99, 0.98],
            edges=[(0, 1)],
        )
        assert config.num_qubits == 2

    def test_three_qubit_twin(self):
        config = build_digital_twin(
            frequencies_ghz=[5.0, 5.1, 5.2],
            t1_us=[100.0, 120.0, 110.0],
            t2_us=[80.0, 100.0, 90.0],
            readout_fidelities=[0.99, 0.98, 0.97],
            edges=[(0, 1), (1, 2)],
        )
        assert config.num_qubits == 3
        assert len(config.topology.edges) == 2

    def test_cz_native_gate(self):
        config = build_digital_twin(
            frequencies_ghz=[5.0, 5.1],
            t1_us=[100.0, 120.0],
            t2_us=[80.0, 100.0],
            readout_fidelities=[0.99, 0.98],
            edges=[(0, 1)],
            native_gate="cz",
        )
        from nqpu.superconducting import NativeGateFamily
        assert config.native_2q_gate == NativeGateFamily.CZ


# ======================================================================
# Gate application integration tests
# ======================================================================


class TestGateApplication:
    """Integration tests for gate application via adapters."""

    def test_s_gate_via_adapter(self):
        adapter = IonTrapAdapter()
        sim = adapter.create_simulator(1, "ideal")
        adapter.run_circuit(sim, [("h", 0), ("s", 0)])
        probs = adapter.get_probabilities(sim)
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_t_gate_via_adapter(self):
        adapter = NeutralAtomAdapter()
        sim = adapter.create_simulator(1, "ideal")
        adapter.run_circuit(sim, [("h", 0), ("t", 0)])
        probs = adapter.get_probabilities(sim)
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_rotation_gates(self):
        adapter = SuperconductingAdapter()
        sim = adapter.create_simulator(1, "ideal")
        adapter.run_circuit(sim, [
            ("rx", 0, math.pi / 4),
            ("ry", 0, math.pi / 4),
            ("rz", 0, math.pi / 4),
        ])
        probs = adapter.get_probabilities(sim)
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_toffoli_decomposed_on_ion(self):
        adapter = IonTrapAdapter()
        sim = adapter.create_simulator(3, "ideal")
        # Apply X to controls, then Toffoli should flip target
        adapter.run_circuit(sim, [("x", 0), ("x", 1), ("ccx", 0, 1, 2)])
        probs = adapter.get_probabilities(sim)
        # |110> -> Toffoli -> |111>
        assert probs[7] == pytest.approx(1.0, abs=0.01)

    def test_toffoli_decomposed_on_sc(self):
        adapter = SuperconductingAdapter()
        sim = adapter.create_simulator(3, "ideal")
        adapter.run_circuit(sim, [("x", 0), ("x", 1), ("ccx", 0, 1, 2)])
        probs = adapter.get_probabilities(sim)
        assert probs[7] == pytest.approx(1.0, abs=0.01)

    def test_swap_decomposed_on_ion(self):
        adapter = IonTrapAdapter()
        sim = adapter.create_simulator(2, "ideal")
        adapter.run_circuit(sim, [("x", 0), ("swap", 0, 1)])
        probs = adapter.get_probabilities(sim)
        # |10> -> SWAP -> |01>
        assert probs[1] == pytest.approx(1.0, abs=0.01)

    def test_cz_gate(self):
        adapter = NeutralAtomAdapter()
        sim = adapter.create_simulator(2, "ideal")
        # CZ: |11> -> -|11>
        adapter.run_circuit(sim, [("h", 0), ("h", 1), ("cz", 0, 1), ("h", 0), ("h", 1)])
        probs = adapter.get_probabilities(sim)
        # H CZ H = CNOT-like behavior
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)


# ======================================================================
# Noisy mode integration
# ======================================================================


class TestNoisyModeIntegration:
    """Integration tests for noisy mode through the adapter layer."""

    def test_noisy_fidelity_less_than_one(self):
        bench = CrossBackendBenchmark(num_qubits=2)
        circuit = BenchmarkCircuit.bell_state(n_pairs=1)
        comp = bench.run_circuit(circuit, modes=["noisy"])
        for res in comp.results.values():
            assert res.fidelity_vs_ideal < 1.0

    def test_noisy_probabilities_sum_to_one(self):
        bench = CrossBackendBenchmark(num_qubits=2)
        circuit = BenchmarkCircuit.bell_state(n_pairs=1)
        comp = bench.run_circuit(circuit, modes=["noisy"])
        for res in comp.results.values():
            total = np.sum(res.probabilities)
            assert total == pytest.approx(1.0, abs=1e-3)

    def test_noisy_wall_time_positive(self):
        bench = CrossBackendBenchmark(num_qubits=2)
        circuit = BenchmarkCircuit.bell_state(n_pairs=1)
        comp = bench.run_circuit(circuit, modes=["noisy"])
        for res in comp.results.values():
            assert res.wall_time_ms > 0.0


# ======================================================================
# Legacy API compatibility
# ======================================================================


class TestLegacyAPI:
    """Verify that the old benchmark_* methods still work."""

    @pytest.fixture
    def bench(self):
        return CrossBackendBenchmark(num_qubits=3)

    def test_legacy_benchmark_bell(self, bench):
        comp = bench.benchmark_bell()
        assert comp.circuit_name == "Bell State"
        assert "ion_ideal" in comp.results
        assert "sc_ideal" in comp.results
        assert "na_ideal" in comp.results

    def test_legacy_benchmark_ghz(self, bench):
        comp = bench.benchmark_ghz()
        assert comp.circuit_name == "GHZ State"

    def test_legacy_benchmark_qft(self, bench):
        comp = bench.benchmark_qft()
        assert comp.circuit_name == "QFT"

    def test_legacy_benchmark_random(self, bench):
        comp = bench.benchmark_random(depth=3)
        assert comp.circuit_name == "Random Circuit"

    def test_legacy_run_all(self, bench):
        results = bench.run_all()
        assert len(results) == 5
        names = [r.circuit_name for r in results]
        assert "Bell State" in names
        assert "Toffoli Circuit" in names

    def test_legacy_run_sc(self, bench):
        res = bench._run_sc(
            CrossBackendBenchmark._bell_circuit_sc, "ideal", "ideal"
        )
        assert isinstance(res, CircuitBenchmark)
        assert res.fidelity_vs_ideal == pytest.approx(1.0)
