"""Comprehensive tests for the nQPU integration dashboard package.

Covers:
- DashboardBenchmark creation and field access
- BenchmarkReport formatting, best_backend, summary, scaling_analysis
- MultiBackendBenchmark: profiles, benchmark_circuit, benchmark_algorithm,
  sweep_qubits, sweep_depth
- Convenience functions: quick_benchmark, full_benchmark, compare_backends
- PipelineConfig defaults and custom settings
- QuantumPipeline full run, individual stages, noise models, optimisation
- Pipeline convenience: quick_run, optimized_run
- CircuitProfile.from_gates analysis for various circuit types
- HardwareProfile fidelity estimation
- DeviceDatabase population, filtering, listing
- HardwareAdvisor recommendations with budget and fidelity constraints
- QECOverhead: surface code, colour code, threshold behaviour
- CostEstimator: cost estimates, provider comparison, budget optimiser,
  break-even analysis
- Convenience: estimate_cost, compare_providers
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nqpu.dashboard.benchmark_dashboard import (
    BenchmarkReport,
    DashboardBenchmark,
    MultiBackendBenchmark,
    compare_backends,
    full_benchmark,
    quick_benchmark,
)
from nqpu.dashboard.pipeline import (
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    QuantumPipeline,
    StageResult,
    optimized_run,
    quick_run,
)
from nqpu.dashboard.hardware_advisor import (
    CircuitProfile,
    DeviceDatabase,
    HardwareAdvisor,
    HardwareProfile,
    HardwareRecommendation,
    recommend_hardware,
)
from nqpu.dashboard.cost_estimator import (
    CloudProvider,
    CostBreakdown,
    CostEstimate,
    CostEstimator,
    QECOverhead,
    compare_providers,
    estimate_cost,
)


# ======================================================================
# DashboardBenchmark
# ======================================================================


class TestDashboardBenchmark:
    """Test DashboardBenchmark dataclass."""

    def test_basic_creation(self):
        b = DashboardBenchmark(
            name="test",
            backend="superconducting",
            n_qubits=5,
            depth=10,
            fidelity=0.95,
            time_seconds=0.001,
            gate_count=50,
            two_qubit_count=20,
        )
        assert b.name == "test"
        assert b.backend == "superconducting"
        assert b.n_qubits == 5
        assert b.fidelity == 0.95

    def test_metadata_default(self):
        b = DashboardBenchmark(
            name="x", backend="y", n_qubits=1, depth=1,
            fidelity=1.0, time_seconds=0.0, gate_count=1, two_qubit_count=0,
        )
        assert b.metadata == {}

    def test_metadata_custom(self):
        b = DashboardBenchmark(
            name="x", backend="y", n_qubits=1, depth=1,
            fidelity=1.0, time_seconds=0.0, gate_count=1, two_qubit_count=0,
            metadata={"algorithm": "ghz"},
        )
        assert b.metadata["algorithm"] == "ghz"

    def test_fidelity_range(self):
        b = DashboardBenchmark(
            name="x", backend="y", n_qubits=3, depth=5,
            fidelity=0.0, time_seconds=0.0, gate_count=10, two_qubit_count=3,
        )
        assert 0.0 <= b.fidelity <= 1.0

    def test_report_formatting(self):
        benchmarks = [
            DashboardBenchmark("a", "sc", 2, 3, 0.99, 0.001, 10, 3),
            DashboardBenchmark("b", "ion", 2, 3, 0.999, 0.01, 10, 3),
        ]
        report = BenchmarkReport(benchmarks=benchmarks, timestamp="2025-01-01")
        table = report.ascii_table()
        assert "sc" in table
        assert "ion" in table
        assert "0.99" in table

    def test_empty_report_ascii(self):
        report = BenchmarkReport(benchmarks=[], timestamp="now")
        assert "No benchmarks" in report.ascii_table()


# ======================================================================
# MultiBackendBenchmark
# ======================================================================


class TestMultiBackendBenchmark:
    """Test MultiBackendBenchmark engine."""

    def test_default_backends(self):
        bench = MultiBackendBenchmark()
        assert "superconducting" in bench.backends
        assert "trapped_ion" in bench.backends
        assert "neutral_atom" in bench.backends

    def test_custom_backends(self):
        bench = MultiBackendBenchmark(backends=["superconducting", "trapped_ion"])
        assert len(bench.backends) == 2

    def test_profiles_built(self):
        bench = MultiBackendBenchmark()
        profiles = bench._backend_profiles
        assert "superconducting" in profiles
        assert profiles["superconducting"]["single_qubit_fidelity"] > 0.99

    def test_benchmark_circuit(self):
        bench = MultiBackendBenchmark(seed=42)
        report = bench.benchmark_circuit(n_qubits=3, depth=5)
        assert isinstance(report, BenchmarkReport)
        assert len(report.benchmarks) == 3  # 3 backends
        for b in report.benchmarks:
            assert b.n_qubits == 3
            assert 0.0 <= b.fidelity <= 1.0
            assert b.gate_count > 0

    def test_benchmark_algorithm_ghz(self):
        bench = MultiBackendBenchmark(seed=42)
        report = bench.benchmark_algorithm("ghz", 4)
        assert len(report.benchmarks) == 3
        for b in report.benchmarks:
            assert "ghz" in b.name
            assert b.n_qubits == 4

    def test_benchmark_algorithm_all(self):
        bench = MultiBackendBenchmark(seed=42)
        for algo in ("ghz", "qft", "grover", "qaoa", "vqe"):
            report = bench.benchmark_algorithm(algo, 3)
            assert len(report.benchmarks) == 3

    def test_benchmark_algorithm_unknown(self):
        bench = MultiBackendBenchmark()
        with pytest.raises(ValueError, match="Unknown algorithm"):
            bench.benchmark_algorithm("nonexistent", 3)

    def test_sweep_qubits(self):
        bench = MultiBackendBenchmark(seed=42)
        report = bench.sweep_qubits(range(2, 5), depth=5)
        assert len(report.benchmarks) == 9  # 3 qubit counts * 3 backends

    def test_sweep_depth(self):
        bench = MultiBackendBenchmark(seed=42)
        report = bench.sweep_depth(n_qubits=3, depth_range=range(2, 5))
        assert len(report.benchmarks) == 9  # 3 depths * 3 backends

    def test_fidelity_decreases_with_depth(self):
        bench = MultiBackendBenchmark(seed=42, backends=["superconducting"])
        r_shallow = bench.benchmark_circuit(3, depth=2)
        r_deep = bench.benchmark_circuit(3, depth=50)
        # Deeper circuit should generally have lower fidelity
        shallow_fid = r_shallow.benchmarks[0].fidelity
        deep_fid = r_deep.benchmarks[0].fidelity
        assert deep_fid <= shallow_fid + 0.01  # small tolerance for jitter

    def test_unknown_backend_fallback(self):
        bench = MultiBackendBenchmark(backends=["custom_backend"])
        assert "custom_backend" in bench._backend_profiles


# ======================================================================
# Benchmark convenience functions
# ======================================================================


class TestBenchmarkFunctions:
    """Test convenience benchmark functions."""

    def test_quick_benchmark(self):
        report = quick_benchmark(n_qubits=3, seed=42)
        assert isinstance(report, BenchmarkReport)
        assert len(report.benchmarks) == 3

    def test_full_benchmark(self):
        report = full_benchmark(seed=42)
        assert isinstance(report, BenchmarkReport)
        assert len(report.benchmarks) > 10

    def test_compare_backends_returns_string(self):
        result = compare_backends(["superconducting", "trapped_ion"], n_qubits=3)
        assert isinstance(result, str)
        assert "superconducting" in result

    def test_benchmark_report_best_backend(self):
        report = quick_benchmark(n_qubits=3)
        best = report.best_backend("fidelity")
        assert best in ("superconducting", "trapped_ion", "neutral_atom")

    def test_benchmark_report_best_backend_unknown_metric(self):
        report = quick_benchmark(n_qubits=3)
        with pytest.raises(ValueError, match="Unknown metric"):
            report.best_backend("nonexistent")

    def test_benchmark_report_summary(self):
        report = quick_benchmark(n_qubits=3)
        summary = report.summary()
        assert "BENCHMARK SUMMARY" in summary

    def test_benchmark_report_scaling(self):
        bench = MultiBackendBenchmark(seed=42)
        report = bench.sweep_qubits(range(2, 5))
        analysis = report.scaling_analysis()
        assert "SCALING ANALYSIS" in analysis


# ======================================================================
# PipelineConfig
# ======================================================================


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_defaults(self):
        cfg = PipelineConfig(n_qubits=3)
        assert cfg.n_qubits == 3
        assert cfg.optimization_level == 1
        assert cfg.target_backend == "generic"
        assert cfg.noise_model is None
        assert cfg.shots == 1024

    def test_custom_config(self):
        cfg = PipelineConfig(
            n_qubits=5,
            optimization_level=3,
            target_backend="superconducting",
            noise_model="depolarizing",
            error_rate=0.01,
            shots=4096,
        )
        assert cfg.optimization_level == 3
        assert cfg.noise_model == "depolarizing"

    def test_seed_reproducibility(self):
        cfg1 = PipelineConfig(n_qubits=2, seed=42)
        cfg2 = PipelineConfig(n_qubits=2, seed=42)
        assert cfg1.seed == cfg2.seed

    def test_zero_optimization(self):
        cfg = PipelineConfig(n_qubits=2, optimization_level=0)
        assert cfg.optimization_level == 0


# ======================================================================
# QuantumPipeline
# ======================================================================


class TestQuantumPipeline:
    """Test QuantumPipeline execution."""

    def test_full_pipeline_bell(self):
        cfg = PipelineConfig(n_qubits=2, shots=1000, seed=42)
        pipe = QuantumPipeline(cfg)
        result = pipe.run([("h", [0], {}), ("cx", [0, 1], {})])
        assert isinstance(result, PipelineResult)
        assert len(result.stage_results) == 6
        assert result.total_time > 0
        # Bell state: should see roughly 50/50 |00> and |11>
        assert "00" in result.final_counts or "11" in result.final_counts

    def test_bell_state_distribution(self):
        cfg = PipelineConfig(n_qubits=2, shots=10000, seed=42)
        pipe = QuantumPipeline(cfg)
        result = pipe.run([("h", [0], {}), ("cx", [0, 1], {})])
        total = sum(result.final_counts.values())
        assert total == 10000
        # Expect ~50% |00> and ~50% |11>
        p00 = result.final_counts.get("00", 0) / total
        p11 = result.final_counts.get("11", 0) / total
        assert p00 == pytest.approx(0.5, abs=0.05)
        assert p11 == pytest.approx(0.5, abs=0.05)

    def test_construct_stage(self):
        cfg = PipelineConfig(n_qubits=2)
        pipe = QuantumPipeline(cfg)
        sr = pipe._stage_construct([("h", [0], {}), ("measure", [0, 1], {})])
        assert sr.success
        assert sr.output["raw_gate_count"] == 2
        assert len(sr.output["parsed_gates"]) == 1
        assert sr.output["measurement_qubits"] == [0, 1]

    def test_analyze_stage(self):
        cfg = PipelineConfig(n_qubits=3)
        pipe = QuantumPipeline(cfg)
        gates = [("h", [0], {}), ("cx", [0, 1], {}), ("cx", [1, 2], {})]
        sr = pipe._stage_analyze(gates)
        assert sr.success
        assert sr.output["single_qubit_gates"] == 1
        assert sr.output["two_qubit_gates"] == 2
        assert sr.output["depth"] > 0

    def test_transpile_stage(self):
        cfg = PipelineConfig(n_qubits=2)
        pipe = QuantumPipeline(cfg)
        gates = [("x", [0], {}), ("s", [0], {}), ("cx", [0, 1], {})]
        sr = pipe._stage_transpile(gates)
        assert sr.success
        transpiled = sr.output["transpiled_gates"]
        # X -> Ry(pi), S -> Rz(pi/2), CX stays
        names = [g[0] for g in transpiled]
        assert "ry" in names
        assert "rz" in names
        assert "cx" in names

    def test_optimize_level_0(self):
        cfg = PipelineConfig(n_qubits=2, optimization_level=0)
        pipe = QuantumPipeline(cfg)
        gates = [("h", [0], {}), ("h", [0], {})]
        sr = pipe._stage_optimize(gates)
        # Level 0 = no optimization, both gates remain
        assert len(sr.output["optimized_gates"]) == 2

    def test_optimize_level_1_cancellation(self):
        cfg = PipelineConfig(n_qubits=2, optimization_level=1)
        pipe = QuantumPipeline(cfg)
        gates = [("h", [0], {}), ("h", [0], {}), ("x", [1], {})]
        sr = pipe._stage_optimize(gates)
        # H-H should cancel, leaving only X
        assert len(sr.output["optimized_gates"]) == 1

    def test_optimize_level_2_merge_rotations(self):
        cfg = PipelineConfig(n_qubits=1, optimization_level=2)
        pipe = QuantumPipeline(cfg)
        gates = [("rz", [0], {"angle": 0.3}), ("rz", [0], {"angle": 0.7})]
        sr = pipe._stage_optimize(gates)
        optimized = sr.output["optimized_gates"]
        assert len(optimized) == 1
        assert optimized[0][2]["angle"] == pytest.approx(1.0)

    def test_optimize_level_3_remove_identity(self):
        cfg = PipelineConfig(n_qubits=1, optimization_level=3)
        pipe = QuantumPipeline(cfg)
        gates = [("rz", [0], {"angle": 0.0}), ("h", [0], {})]
        sr = pipe._stage_optimize(gates)
        optimized = sr.output["optimized_gates"]
        # Zero-angle Rz should be removed
        assert len(optimized) == 1
        assert optimized[0][0] == "h"

    def test_noise_depolarizing(self):
        cfg = PipelineConfig(
            n_qubits=2, noise_model="depolarizing", error_rate=0.1, shots=1000, seed=42,
        )
        pipe = QuantumPipeline(cfg)
        result = pipe.run([("h", [0], {}), ("cx", [0, 1], {})])
        # Depolarizing noise should add some noise to the output
        assert len(result.final_counts) >= 1

    def test_noise_amplitude_damping(self):
        cfg = PipelineConfig(
            n_qubits=2, noise_model="amplitude_damping", error_rate=0.1, shots=1000, seed=42,
        )
        pipe = QuantumPipeline(cfg)
        result = pipe.run([("x", [0], {}), ("x", [1], {})])
        assert len(result.final_counts) >= 1

    def test_pipeline_summary_format(self):
        cfg = PipelineConfig(n_qubits=2, shots=100, seed=42)
        pipe = QuantumPipeline(cfg)
        result = pipe.run([("h", [0], {})])
        summary = result.summary()
        assert "PIPELINE EXECUTION SUMMARY" in summary
        assert "Qubits:" in summary

    def test_pipeline_stage_timings(self):
        cfg = PipelineConfig(n_qubits=2, shots=100, seed=42)
        pipe = QuantumPipeline(cfg)
        result = pipe.run([("h", [0], {})])
        timings = result.stage_timings()
        assert "STAGE TIMINGS" in timings


# ======================================================================
# Pipeline convenience functions
# ======================================================================


class TestPipelineFunctions:
    """Test pipeline convenience functions."""

    def test_quick_run(self):
        result = quick_run(
            gates=[("h", [0], {}), ("cx", [0, 1], {})],
            n_qubits=2,
            shots=500,
        )
        assert isinstance(result, PipelineResult)
        assert sum(result.final_counts.values()) == 500

    def test_optimized_run(self):
        result = optimized_run(
            gates=[("h", [0], {}), ("cx", [0, 1], {})],
            n_qubits=2,
            backend="superconducting",
        )
        assert isinstance(result, PipelineResult)
        assert result.config.optimization_level == 3
        assert result.config.noise_model == "depolarizing"

    def test_optimized_run_trapped_ion(self):
        result = optimized_run(
            gates=[("h", [0], {})],
            n_qubits=1,
            backend="trapped_ion",
        )
        assert result.config.error_rate == 0.0001

    def test_quick_run_single_qubit(self):
        result = quick_run(
            gates=[("x", [0], {})],
            n_qubits=1,
            shots=100,
        )
        # X gate on |0> -> |1>, should measure mostly "1"
        total = sum(result.final_counts.values())
        assert total == 100
        assert result.final_counts.get("1", 0) > 80


# ======================================================================
# CircuitProfile
# ======================================================================


class TestCircuitProfile:
    """Test CircuitProfile analysis."""

    def test_simple_circuit(self):
        gates = [("h", [0], {}), ("cx", [0, 1], {})]
        profile = CircuitProfile.from_gates(gates, n_qubits=2)
        assert profile.n_qubits == 2
        assert profile.single_qubit_gates == 1
        assert profile.two_qubit_gates == 1
        assert profile.depth > 0

    def test_t_gate_counting(self):
        gates = [("t", [0], {}), ("t", [1], {}), ("h", [0], {})]
        profile = CircuitProfile.from_gates(gates, n_qubits=2)
        assert profile.t_gates == 2
        assert profile.single_qubit_gates == 3

    def test_three_qubit_gates(self):
        gates = [("ccx", [0, 1, 2], {})]
        profile = CircuitProfile.from_gates(gates, n_qubits=3)
        assert profile.three_qubit_gates == 1

    def test_measurement_counting(self):
        gates = [("h", [0], {}), ("measure", [0, 1], {})]
        profile = CircuitProfile.from_gates(gates, n_qubits=2)
        assert profile.measurement_count == 2
        assert profile.single_qubit_gates == 1

    def test_connectivity_linear(self):
        gates = [("cx", [0, 1], {}), ("cx", [1, 2], {})]
        profile = CircuitProfile.from_gates(gates, n_qubits=3)
        assert profile.connectivity_required == "linear"

    def test_connectivity_non_linear(self):
        gates = [("cx", [0, 2], {})]
        profile = CircuitProfile.from_gates(gates, n_qubits=3)
        assert profile.connectivity_required in ("2d_grid", "all_to_all")


# ======================================================================
# HardwareProfile
# ======================================================================


class TestHardwareProfile:
    """Test HardwareProfile fidelity estimation."""

    def test_perfect_device(self):
        dev = HardwareProfile(
            name="Perfect", technology="ideal", max_qubits=100,
            connectivity="all_to_all",
            single_qubit_fidelity=1.0, two_qubit_fidelity=1.0,
            readout_fidelity=1.0,
            t1_us=float("inf"), t2_us=float("inf"),
            gate_time_ns=1.0, two_qubit_gate_time_ns=1.0,
            native_gates=["all"], availability="cloud", cost_per_shot=0.0,
        )
        profile = CircuitProfile(
            n_qubits=2, depth=5, single_qubit_gates=10, two_qubit_gates=5,
            three_qubit_gates=0, t_gates=0, measurement_count=2,
            connectivity_required="linear", estimated_runtime_us=1.0,
        )
        fid = dev.estimated_circuit_fidelity(profile)
        assert fid == pytest.approx(1.0)

    def test_noisy_device_lower_fidelity(self):
        dev = HardwareProfile(
            name="Noisy", technology="sc", max_qubits=50,
            connectivity="heavy_hex",
            single_qubit_fidelity=0.999, two_qubit_fidelity=0.99,
            readout_fidelity=0.97,
            t1_us=100.0, t2_us=50.0,
            gate_time_ns=25.0, two_qubit_gate_time_ns=200.0,
            native_gates=["sx", "rz", "cx"], availability="cloud",
            cost_per_shot=0.001,
        )
        profile = CircuitProfile(
            n_qubits=5, depth=20, single_qubit_gates=50, two_qubit_gates=30,
            three_qubit_gates=0, t_gates=5, measurement_count=5,
            connectivity_required="heavy_hex", estimated_runtime_us=10.0,
        )
        fid = dev.estimated_circuit_fidelity(profile)
        assert 0.0 < fid < 1.0

    def test_three_qubit_gate_overhead(self):
        dev = HardwareProfile(
            name="Test", technology="sc", max_qubits=50,
            connectivity="2d_grid",
            single_qubit_fidelity=0.999, two_qubit_fidelity=0.99,
            readout_fidelity=0.99,
            t1_us=100.0, t2_us=80.0,
            gate_time_ns=25.0, two_qubit_gate_time_ns=200.0,
            native_gates=["sx", "cx"], availability="cloud",
            cost_per_shot=0.001,
        )
        profile_no3q = CircuitProfile(
            n_qubits=3, depth=5, single_qubit_gates=5, two_qubit_gates=3,
            three_qubit_gates=0, t_gates=0, measurement_count=3,
            connectivity_required="linear", estimated_runtime_us=1.0,
        )
        profile_with3q = CircuitProfile(
            n_qubits=3, depth=5, single_qubit_gates=5, two_qubit_gates=3,
            three_qubit_gates=5, t_gates=0, measurement_count=3,
            connectivity_required="linear", estimated_runtime_us=1.0,
        )
        fid_no3q = dev.estimated_circuit_fidelity(profile_no3q)
        fid_with3q = dev.estimated_circuit_fidelity(profile_with3q)
        assert fid_with3q < fid_no3q

    def test_photonic_no_decoherence(self):
        dev = HardwareProfile(
            name="Photonic", technology="photonic", max_qubits=100,
            connectivity="linear",
            single_qubit_fidelity=0.999, two_qubit_fidelity=0.98,
            readout_fidelity=0.95,
            t1_us=float("inf"), t2_us=float("inf"),
            gate_time_ns=1.0, two_qubit_gate_time_ns=10.0,
            native_gates=["bs"], availability="cloud", cost_per_shot=0.001,
        )
        profile = CircuitProfile(
            n_qubits=2, depth=100, single_qubit_gates=100, two_qubit_gates=50,
            three_qubit_gates=0, t_gates=0, measurement_count=2,
            connectivity_required="linear", estimated_runtime_us=0.1,
        )
        fid = dev.estimated_circuit_fidelity(profile)
        # No decoherence, so fidelity should not be crushed by runtime
        assert fid > 0.01

    def test_fidelity_bounded(self):
        dev = HardwareProfile(
            name="Test", technology="sc", max_qubits=10,
            connectivity="linear",
            single_qubit_fidelity=0.5, two_qubit_fidelity=0.5,
            readout_fidelity=0.5,
            t1_us=1.0, t2_us=0.5,
            gate_time_ns=1000.0, two_qubit_gate_time_ns=5000.0,
            native_gates=["x"], availability="cloud", cost_per_shot=0.01,
        )
        profile = CircuitProfile(
            n_qubits=5, depth=100, single_qubit_gates=500, two_qubit_gates=200,
            three_qubit_gates=0, t_gates=0, measurement_count=5,
            connectivity_required="linear", estimated_runtime_us=100.0,
        )
        fid = dev.estimated_circuit_fidelity(profile)
        assert 0.0 <= fid <= 1.0


# ======================================================================
# DeviceDatabase
# ======================================================================


class TestDeviceDatabase:
    """Test DeviceDatabase population and filtering."""

    def test_database_populated(self):
        db = DeviceDatabase()
        assert len(db.devices) >= 10

    def test_device_lookup(self):
        db = DeviceDatabase()
        dev = db.get_device("ibm_heron")
        assert "IBM" in dev.name
        assert dev.technology == "superconducting"

    def test_device_not_found(self):
        db = DeviceDatabase()
        with pytest.raises(KeyError):
            db.get_device("nonexistent_device")

    def test_filter_by_qubits(self):
        db = DeviceDatabase()
        large = db.filter_by_qubits(100)
        assert len(large) >= 3
        for d in large:
            assert d.max_qubits >= 100

    def test_filter_by_technology(self):
        db = DeviceDatabase()
        ions = db.filter_by_technology("trapped_ion")
        assert len(ions) >= 2
        for d in ions:
            assert d.technology == "trapped_ion"

    def test_list_devices_format(self):
        db = DeviceDatabase()
        table = db.list_devices()
        assert isinstance(table, str)
        assert "IBM" in table
        assert "IonQ" in table


# ======================================================================
# HardwareAdvisor
# ======================================================================


class TestHardwareAdvisor:
    """Test HardwareAdvisor recommendations."""

    def test_basic_recommendation(self):
        advisor = HardwareAdvisor()
        profile = CircuitProfile(
            n_qubits=5, depth=10, single_qubit_gates=20, two_qubit_gates=10,
            three_qubit_gates=0, t_gates=2, measurement_count=5,
            connectivity_required="linear", estimated_runtime_us=5.0,
        )
        rec = advisor.recommend(profile)
        assert isinstance(rec, HardwareRecommendation)
        assert rec.score > 0
        assert rec.recommended != "none"

    def test_recommendation_with_budget(self):
        advisor = HardwareAdvisor()
        profile = CircuitProfile(
            n_qubits=5, depth=10, single_qubit_gates=20, two_qubit_gates=10,
            three_qubit_gates=0, t_gates=0, measurement_count=5,
            connectivity_required="linear", estimated_runtime_us=5.0,
        )
        rec = advisor.recommend(profile, budget=0.01)
        # Very tight budget should still find something
        # (some providers are cheap enough)
        assert isinstance(rec, HardwareRecommendation)

    def test_recommendation_no_devices(self):
        advisor = HardwareAdvisor()
        profile = CircuitProfile(
            n_qubits=2_000_000, depth=10, single_qubit_gates=20,
            two_qubit_gates=10, three_qubit_gates=0, t_gates=0,
            measurement_count=5, connectivity_required="linear",
            estimated_runtime_us=5.0,
        )
        rec = advisor.recommend(profile)
        # PsiQuantum has 1M qubits, so 2M should exceed all
        assert rec.recommended == "none"

    def test_recommendation_report_format(self):
        rec = recommend_hardware(
            gates=[("h", [0], {}), ("cx", [0, 1], {})],
            n_qubits=2,
        )
        report = rec.report()
        assert "HARDWARE RECOMMENDATION" in report
        assert "Recommended" in report

    def test_compare_options(self):
        advisor = HardwareAdvisor()
        profile = CircuitProfile(
            n_qubits=3, depth=5, single_qubit_gates=10, two_qubit_gates=5,
            three_qubit_gates=0, t_gates=0, measurement_count=3,
            connectivity_required="linear", estimated_runtime_us=2.0,
        )
        comparison = advisor.compare_options(profile)
        assert "HARDWARE COMPARISON" in comparison

    def test_analyze_circuit(self):
        advisor = HardwareAdvisor()
        gates = [("h", [0], {}), ("cx", [0, 1], {}), ("t", [0], {})]
        profile = advisor.analyze_circuit(gates, n_qubits=2)
        assert profile.single_qubit_gates == 2
        assert profile.two_qubit_gates == 1
        assert profile.t_gates == 1

    def test_recommendation_alternatives(self):
        rec = recommend_hardware(
            gates=[("h", [0], {}), ("cx", [0, 1], {})],
            n_qubits=2,
        )
        # Should have alternatives
        assert len(rec.alternatives) >= 1

    def test_min_fidelity_filter(self):
        advisor = HardwareAdvisor()
        profile = CircuitProfile(
            n_qubits=5, depth=10, single_qubit_gates=20, two_qubit_gates=10,
            three_qubit_gates=0, t_gates=0, measurement_count=5,
            connectivity_required="linear", estimated_runtime_us=5.0,
        )
        rec = advisor.recommend(profile, min_fidelity=0.9999)
        # With very high fidelity requirement, may get "none"
        assert isinstance(rec, HardwareRecommendation)


# ======================================================================
# QECOverhead
# ======================================================================


class TestQECOverhead:
    """Test QEC overhead calculations."""

    def test_surface_code_basic(self):
        qec = QECOverhead.surface_code(target_error=1e-10, physical_error=0.001)
        assert qec.code_distance >= 3
        assert qec.code_distance % 2 == 1  # must be odd
        assert qec.physical_qubits_per_logical > 0
        assert qec.logical_error_rate < 0.001

    def test_surface_code_below_threshold(self):
        qec = QECOverhead.surface_code(target_error=1e-6, physical_error=0.02)
        # Above threshold (0.01), QEC doesn't help
        assert qec.code_distance == 0

    def test_color_code_basic(self):
        qec = QECOverhead.color_code(target_error=1e-10, physical_error=0.001)
        assert qec.code_distance >= 3
        assert qec.physical_qubits_per_logical > 0

    def test_color_code_below_threshold(self):
        qec = QECOverhead.color_code(target_error=1e-6, physical_error=0.01)
        # Above colour code threshold (0.0082)
        assert qec.code_distance == 0

    def test_surface_code_distance_increases(self):
        qec_loose = QECOverhead.surface_code(target_error=1e-4, physical_error=0.001)
        qec_tight = QECOverhead.surface_code(target_error=1e-12, physical_error=0.001)
        assert qec_tight.code_distance >= qec_loose.code_distance


# ======================================================================
# CostEstimator
# ======================================================================


class TestCostEstimator:
    """Test CostEstimator engine."""

    def test_basic_estimate(self):
        est = CostEstimator()
        cost = est.estimate(n_qubits=5, depth=10, shots=1024)
        assert isinstance(cost, CostEstimate)
        assert cost.breakdown.total > 0
        assert cost.logical_qubits == 5

    def test_estimate_with_qec(self):
        est = CostEstimator()
        cost = est.estimate(n_qubits=5, depth=10, shots=1024, use_qec=True)
        assert cost.qec_overhead is not None
        assert cost.physical_qubits >= cost.logical_qubits
        assert cost.confidence == "low"

    def test_estimate_specific_provider(self):
        est = CostEstimator()
        cost = est.estimate(n_qubits=5, depth=10, provider="ibm_quantum")
        assert cost.provider == "IBM Quantum"

    def test_provider_comparison(self):
        est = CostEstimator()
        comparison = est.compare_providers(n_qubits=5, depth=10)
        assert isinstance(comparison, str)
        assert "PROVIDER COMPARISON" in comparison
        assert "Cheapest" in comparison

    def test_budget_optimizer(self):
        est = CostEstimator()
        result = est.budget_optimizer(budget=1.0, n_qubits=5, depth=10)
        assert isinstance(result, dict)
        if result:
            assert result["shots"] >= 100
            assert result["total_cost"] <= 1.0

    def test_budget_optimizer_tiny_budget(self):
        est = CostEstimator()
        result = est.budget_optimizer(budget=0.0001, n_qubits=5, depth=10, min_shots=100)
        # Very tiny budget may not find anything
        # Or may find cheapest provider at minimum shots
        assert isinstance(result, dict)

    def test_break_even_analysis(self):
        est = CostEstimator()
        analysis = est.break_even_analysis(
            classical_time_hours=10.0, n_qubits=5, depth=10,
        )
        assert isinstance(analysis, str)
        assert "BREAK-EVEN ANALYSIS" in analysis

    def test_cost_report_format(self):
        est = CostEstimator()
        cost = est.estimate(n_qubits=3, depth=5)
        report = cost.report()
        assert "QUANTUM COST ESTIMATE" in report
        assert "Provider:" in report

    def test_cost_breakdown_ascii(self):
        breakdown = CostBreakdown(
            compute_cost=0.1, error_correction_overhead=0.05,
            classical_processing=0.001, total=0.151,
        )
        text = breakdown.ascii_breakdown()
        assert "COST BREAKDOWN" in text
        assert "TOTAL" in text

    def test_providers_populated(self):
        est = CostEstimator()
        assert len(est.providers) >= 5


# ======================================================================
# Cost convenience functions
# ======================================================================


class TestCostFunctions:
    """Test cost convenience functions."""

    def test_estimate_cost(self):
        est = estimate_cost(n_qubits=5, depth=10, shots=1024)
        assert isinstance(est, CostEstimate)
        assert est.breakdown.total > 0

    def test_compare_providers_func(self):
        result = compare_providers(n_qubits=5, depth=10)
        assert isinstance(result, str)
        assert "IBM" in result or "IonQ" in result or "Amazon" in result

    def test_estimate_cost_large_circuit(self):
        est = estimate_cost(n_qubits=51, depth=100, shots=10000)
        assert est.confidence == "medium"
        assert est.breakdown.total > 0
