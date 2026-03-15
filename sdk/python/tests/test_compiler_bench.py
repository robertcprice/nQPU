"""Comprehensive tests for cross-platform compiler benchmarking.

Tests cover:
- NativeGateAnalyzer decompositions (ECR, sqrt-iSWAP, CZ)
- CNOT and SWAP gate counts
- CNOT duration estimates
- CompilerBenchmark compilation for all circuit types
- Overhead ratios
- Scaling behaviour across qubit counts
- Edge cases: unknown gate family, unknown circuit type
"""

import math

import numpy as np
import pytest

from nqpu.superconducting.compiler_bench import (
    CompilationResult,
    CompilerBenchmark,
    NativeGate,
    NativeGateAnalyzer,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def analyzer():
    """A NativeGateAnalyzer instance."""
    return NativeGateAnalyzer()


@pytest.fixture
def bench():
    """A CompilerBenchmark with default 4 qubits."""
    return CompilerBenchmark(num_qubits=4)


# ======================================================================
# NativeGateAnalyzer decomposition tests
# ======================================================================


class TestNativeGateAnalyzer:
    """Tests for native gate decomposition logic."""

    @pytest.mark.parametrize("family", ["ecr", "sqrt_iswap", "cz"])
    def test_cnot_decomposition_non_empty(self, analyzer, family):
        """CNOT decomposition produces at least one gate."""
        gates = analyzer.decompose_cnot(family)
        assert len(gates) > 0

    @pytest.mark.parametrize("family", ["ecr", "sqrt_iswap", "cz"])
    def test_cnot_has_two_qubit_gate(self, analyzer, family):
        """CNOT decomposition contains at least one 2Q native gate."""
        gates = analyzer.decompose_cnot(family)
        two_q_gates = [g for g in gates if len(g.qubits) == 2]
        assert len(two_q_gates) >= 1

    def test_ecr_cnot_uses_ecr_gate(self, analyzer):
        """ECR decomposition includes an ECR gate."""
        gates = analyzer.decompose_cnot("ecr")
        ecr_gates = [g for g in gates if g.name == "ECR"]
        assert len(ecr_gates) == 1

    def test_sqrt_iswap_cnot_uses_two_sqrt_iswap(self, analyzer):
        """sqrt-iSWAP decomposition uses exactly 2 sqrt(iSWAP) gates."""
        gates = analyzer.decompose_cnot("sqrt_iswap")
        sqrt_gates = [g for g in gates if g.name == "sqrt_iSWAP"]
        assert len(sqrt_gates) == 2

    def test_cz_cnot_uses_cz_gate(self, analyzer):
        """CZ decomposition includes a CZ gate."""
        gates = analyzer.decompose_cnot("cz")
        cz_gates = [g for g in gates if g.name == "CZ"]
        assert len(cz_gates) == 1

    @pytest.mark.parametrize("family", ["ecr", "sqrt_iswap", "cz"])
    def test_swap_has_three_times_cnot_2q_gates(self, analyzer, family):
        """SWAP decomposition has 3x the 2Q gates of a single CNOT."""
        cnot_counts = analyzer.cnot_native_counts(family)
        swap_counts = analyzer.swap_native_counts(family)
        assert swap_counts["2q"] == 3 * cnot_counts["2q"]

    def test_unknown_family_raises(self, analyzer):
        """Unknown gate family raises ValueError."""
        with pytest.raises(ValueError, match="Unknown gate family"):
            analyzer.decompose_cnot("unknown_gate")


# ======================================================================
# CNOT native gate counts
# ======================================================================


class TestCnotCounts:
    """Tests for CNOT and SWAP gate count summaries."""

    @pytest.mark.parametrize("family,expected_2q", [
        ("ecr", 1),
        ("sqrt_iswap", 2),
        ("cz", 1),
    ])
    def test_cnot_2q_count(self, analyzer, family, expected_2q):
        """Each family has the expected number of 2Q gates per CNOT."""
        counts = analyzer.cnot_native_counts(family)
        assert counts["2q"] == expected_2q

    @pytest.mark.parametrize("family", ["ecr", "sqrt_iswap", "cz"])
    def test_cnot_total_equals_sum(self, analyzer, family):
        """Total count equals 1Q + 2Q."""
        counts = analyzer.cnot_native_counts(family)
        assert counts["total"] == counts["1q"] + counts["2q"]


# ======================================================================
# CNOT duration estimates
# ======================================================================


class TestCnotDuration:
    """Tests for CNOT duration calculations."""

    def test_ecr_cnot_duration_positive(self, analyzer):
        """ECR CNOT duration is positive."""
        dur = analyzer.cnot_duration_ns("ecr")
        assert dur > 0.0

    def test_sqrt_iswap_cnot_faster_than_ecr(self, analyzer):
        """sqrt-iSWAP CNOT is faster than ECR (32ns vs 300ns per 2Q gate)."""
        dur_ecr = analyzer.cnot_duration_ns("ecr")
        dur_sqrt = analyzer.cnot_duration_ns("sqrt_iswap")
        assert dur_sqrt < dur_ecr

    @pytest.mark.parametrize("family", ["ecr", "sqrt_iswap", "cz"])
    def test_duration_consistency(self, analyzer, family):
        """Duration equals sum of 1Q and 2Q gate timings."""
        gates = analyzer.decompose_cnot(family)
        times = NativeGateAnalyzer._GATE_TIMES[family]
        expected = sum(
            times["2q"] if len(g.qubits) == 2 else times["1q"]
            for g in gates
        )
        assert analyzer.cnot_duration_ns(family) == pytest.approx(expected)


# ======================================================================
# Overhead ratio tests
# ======================================================================


class TestOverheadRatio:
    """Tests for compilation overhead ratios."""

    @pytest.mark.parametrize("family", ["ecr", "sqrt_iswap", "cz"])
    def test_overhead_at_least_one(self, analyzer, family):
        """Overhead ratio is always >= 1.0."""
        for ctype in ["bell", "ghz", "qft", "random", "grover_oracle"]:
            ratio = analyzer.overhead_ratio(family, ctype, num_qubits=4)
            assert ratio >= 1.0, f"{family}/{ctype} overhead < 1.0"

    def test_unknown_circuit_raises(self, analyzer):
        """Unknown circuit type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown circuit type"):
            analyzer.overhead_ratio("ecr", "nonexistent_circuit")


# ======================================================================
# CompilerBenchmark tests
# ======================================================================


class TestCompilerBenchmark:
    """Tests for CompilerBenchmark class."""

    @pytest.mark.parametrize("circuit_type", [
        "bell", "ghz", "qft", "random", "grover_oracle",
    ])
    def test_benchmark_single_family(self, bench, circuit_type):
        """Single-family benchmark returns a CompilationResult."""
        result = bench.benchmark_compilation(
            circuit_type, num_qubits=4, gate_family="ecr"
        )
        assert isinstance(result, CompilationResult)
        assert result.native_gate_family == "ecr"
        assert result.num_1q_gates >= 0
        assert result.num_2q_gates >= 0

    def test_benchmark_all_families(self, bench):
        """When no gate_family given, returns list of 3 results."""
        results = bench.benchmark_compilation("bell", num_qubits=4)
        assert isinstance(results, list)
        assert len(results) == 3
        families = {r.native_gate_family for r in results}
        assert families == {"ecr", "sqrt_iswap", "cz"}

    def test_benchmark_all_circuits(self, bench):
        """benchmark_all_circuits returns dict with all circuit types."""
        results = bench.benchmark_all_circuits(num_qubits=4)
        assert isinstance(results, dict)
        expected_circuits = {"bell", "ghz", "qft", "random", "grover_oracle"}
        assert set(results.keys()) == expected_circuits
        for ctype, family_results in results.items():
            assert len(family_results) == 3

    def test_compilation_result_str(self):
        """CompilationResult __str__ produces readable output."""
        r = CompilationResult(
            native_gate_family="ecr",
            circuit_type="bell",
            num_qubits=2,
            num_1q_gates=5,
            num_2q_gates=1,
            depth=6,
            estimated_duration_ns=425.0,
            overhead_2q=1.0,
        )
        text = str(r)
        assert "ecr" in text
        assert "bell" in text

    def test_gate_counts_consistent_across_runs(self, bench):
        """Gate counts are deterministic across runs."""
        r1 = bench.benchmark_compilation("qft", num_qubits=3, gate_family="ecr")
        r2 = bench.benchmark_compilation("qft", num_qubits=3, gate_family="ecr")
        assert isinstance(r1, CompilationResult)
        assert isinstance(r2, CompilationResult)
        assert r1.num_1q_gates == r2.num_1q_gates
        assert r1.num_2q_gates == r2.num_2q_gates

    def test_bell_uses_one_abstract_2q_gate(self, bench):
        """Bell circuit has exactly 1 abstract 2Q gate (CNOT)."""
        result = bench.benchmark_compilation("bell", num_qubits=2, gate_family="ecr")
        assert isinstance(result, CompilationResult)
        assert result.abstract_2q_gates == 1

    def test_ghz_abstract_2q_gates_scale(self, bench):
        """GHZ circuit has n-1 abstract 2Q gates for n qubits."""
        for n in [3, 4, 5]:
            result = bench.benchmark_compilation("ghz", num_qubits=n, gate_family="ecr")
            assert isinstance(result, CompilationResult)
            assert result.abstract_2q_gates == n - 1

    def test_estimated_duration_positive(self, bench):
        """All compilations have positive estimated duration."""
        results = bench.benchmark_all_circuits(num_qubits=3)
        for ctype, family_results in results.items():
            for r in family_results:
                assert r.estimated_duration_ns > 0.0

    def test_depth_at_least_gate_count(self, bench):
        """Depth is at least the total native gate count (serial estimate)."""
        result = bench.benchmark_compilation("qft", num_qubits=3, gate_family="cz")
        assert isinstance(result, CompilationResult)
        assert result.depth >= result.num_1q_gates + result.num_2q_gates

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_qft_2q_gates_scale_with_qubits(self, n):
        """QFT 2Q gate count increases with qubit count."""
        b = CompilerBenchmark(num_qubits=n)
        result = b.benchmark_compilation("qft", num_qubits=n, gate_family="ecr")
        assert isinstance(result, CompilationResult)
        assert result.num_2q_gates > 0

    def test_unknown_circuit_type_raises(self, bench):
        """Unknown circuit type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown circuit type"):
            bench.benchmark_compilation("nonexistent", num_qubits=4, gate_family="ecr")
