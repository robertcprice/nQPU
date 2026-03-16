"""Multi-backend benchmarking with ASCII reports.

Provides a ``MultiBackendBenchmark`` that runs standardised quantum circuits
across simulated superconducting, trapped-ion, and neutral-atom backends,
modelling realistic noise characteristics and gate timings.  Results are
collected into ``BenchmarkReport`` objects that can be rendered as ASCII
tables and analysed for scaling behaviour.

Example::

    from nqpu.dashboard import quick_benchmark
    report = quick_benchmark(n_qubits=5)
    print(report.ascii_table())
    print(report.summary())
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class DashboardBenchmark:
    """Single benchmark result."""

    name: str
    backend: str
    n_qubits: int
    depth: int
    fidelity: float
    time_seconds: float
    gate_count: int
    two_qubit_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Collection of benchmark results with analysis."""

    benchmarks: List[DashboardBenchmark]
    timestamp: str

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def ascii_table(self) -> str:
        """Format results as an ASCII table."""
        if not self.benchmarks:
            return "No benchmarks to display."

        col_name = 18
        col_back = 18
        col_num = 8
        col_fid = 10
        col_time = 12
        col_gc = 10
        col_2q = 8

        header = (
            f"{'Benchmark':<{col_name}} "
            f"{'Backend':<{col_back}} "
            f"{'Qubits':>{col_num}} "
            f"{'Fidelity':>{col_fid}} "
            f"{'Time (s)':>{col_time}} "
            f"{'Gates':>{col_gc}} "
            f"{'2Q':>{col_2q}}"
        )
        separator = "-" * len(header)
        lines = [separator, header, separator]

        for b in self.benchmarks:
            lines.append(
                f"{b.name:<{col_name}} "
                f"{b.backend:<{col_back}} "
                f"{b.n_qubits:>{col_num}} "
                f"{b.fidelity:>{col_fid}.6f} "
                f"{b.time_seconds:>{col_time}.6f} "
                f"{b.gate_count:>{col_gc}} "
                f"{b.two_qubit_count:>{col_2q}}"
            )

        lines.append(separator)
        return "\n".join(lines)

    def best_backend(self, metric: str = "fidelity") -> str:
        """Return best backend for given metric.

        Parameters
        ----------
        metric : str
            One of ``'fidelity'`` (higher is better), ``'time'``
            (lower is better), or ``'gate_count'`` (lower is better).
        """
        if not self.benchmarks:
            raise ValueError("No benchmarks to analyse.")

        if metric == "fidelity":
            return max(self.benchmarks, key=lambda b: b.fidelity).backend
        elif metric == "time":
            return min(self.benchmarks, key=lambda b: b.time_seconds).backend
        elif metric == "gate_count":
            return min(self.benchmarks, key=lambda b: b.gate_count).backend
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def summary(self) -> str:
        """Executive summary of benchmark results."""
        if not self.benchmarks:
            return "No benchmarks to summarise."

        backends = sorted(set(b.backend for b in self.benchmarks))
        lines = [
            "=" * 60,
            "BENCHMARK SUMMARY",
            f"Timestamp: {self.timestamp}",
            f"Total benchmarks: {len(self.benchmarks)}",
            f"Backends tested: {', '.join(backends)}",
            "=" * 60,
        ]

        for backend in backends:
            subset = [b for b in self.benchmarks if b.backend == backend]
            avg_fid = np.mean([b.fidelity for b in subset])
            avg_time = np.mean([b.time_seconds for b in subset])
            avg_gates = np.mean([b.gate_count for b in subset])
            lines.append(f"\n  {backend}:")
            lines.append(f"    Avg fidelity:   {avg_fid:.6f}")
            lines.append(f"    Avg time:       {avg_time:.6f} s")
            lines.append(f"    Avg gate count: {avg_gates:.1f}")

        best_fid = self.best_backend("fidelity")
        best_time = self.best_backend("time")
        lines.append(f"\n  Best fidelity:  {best_fid}")
        lines.append(f"  Fastest:        {best_time}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def scaling_analysis(self) -> str:
        """Analyse how metrics scale with problem size."""
        if not self.benchmarks:
            return "No benchmarks for scaling analysis."

        backends = sorted(set(b.backend for b in self.benchmarks))
        lines = [
            "SCALING ANALYSIS",
            "=" * 60,
        ]

        for backend in backends:
            subset = sorted(
                [b for b in self.benchmarks if b.backend == backend],
                key=lambda b: b.n_qubits,
            )
            if len(subset) < 2:
                continue
            lines.append(f"\n  {backend}:")
            lines.append(
                f"    {'Qubits':>8} {'Fidelity':>10} {'Time (s)':>12} {'Gates':>8}"
            )
            for b in subset:
                lines.append(
                    f"    {b.n_qubits:>8} {b.fidelity:>10.6f} "
                    f"{b.time_seconds:>12.6f} {b.gate_count:>8}"
                )
            # Simple linear fit on fidelity vs qubits
            qubits_arr = np.array([b.n_qubits for b in subset], dtype=float)
            fid_arr = np.array([b.fidelity for b in subset], dtype=float)
            if len(qubits_arr) >= 2 and np.std(qubits_arr) > 0:
                slope = float(
                    np.sum((qubits_arr - qubits_arr.mean()) * (fid_arr - fid_arr.mean()))
                    / np.sum((qubits_arr - qubits_arr.mean()) ** 2)
                )
                lines.append(f"    Fidelity slope: {slope:.6f} per qubit")

        return "\n".join(lines)


# ======================================================================
# Backend profiles
# ======================================================================

_DEFAULT_BACKEND_PROFILES: Dict[str, dict] = {
    "superconducting": {
        "gate_time_ns": 25.0,
        "two_qubit_gate_time_ns": 200.0,
        "t1_us": 100.0,
        "t2_us": 80.0,
        "connectivity": "heavy_hex",
        "single_qubit_fidelity": 0.9995,
        "two_qubit_fidelity": 0.995,
        "readout_fidelity": 0.99,
        "max_qubits": 1121,
    },
    "trapped_ion": {
        "gate_time_ns": 10000.0,
        "two_qubit_gate_time_ns": 200000.0,
        "t1_us": 1_000_000.0,
        "t2_us": 500_000.0,
        "connectivity": "all_to_all",
        "single_qubit_fidelity": 0.9999,
        "two_qubit_fidelity": 0.999,
        "readout_fidelity": 0.998,
        "max_qubits": 32,
    },
    "neutral_atom": {
        "gate_time_ns": 500.0,
        "two_qubit_gate_time_ns": 1000.0,
        "t1_us": 5_000.0,
        "t2_us": 2_000.0,
        "connectivity": "2d_grid",
        "single_qubit_fidelity": 0.999,
        "two_qubit_fidelity": 0.995,
        "readout_fidelity": 0.97,
        "max_qubits": 256,
    },
    "photonic": {
        "gate_time_ns": 1.0,
        "two_qubit_gate_time_ns": 10.0,
        "t1_us": float("inf"),
        "t2_us": float("inf"),
        "connectivity": "linear",
        "single_qubit_fidelity": 0.999,
        "two_qubit_fidelity": 0.98,
        "readout_fidelity": 0.95,
        "max_qubits": 216,
    },
}


# ======================================================================
# Multi-backend benchmark engine
# ======================================================================


class MultiBackendBenchmark:
    """Run standardised benchmarks across multiple simulated backends.

    Backends are modelled with realistic noise and gate characteristics:

    - **superconducting**: fast gates, moderate connectivity, T1/T2 noise
    - **trapped_ion**: slow gates, all-to-all connectivity, high fidelity
    - **neutral_atom**: native multi-qubit gates, moderate speed
    - **photonic**: measurement-based, linear optics

    Parameters
    ----------
    backends : list, optional
        Backend names to benchmark. Defaults to the three solid-state
        platforms.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        backends: Optional[List[str]] = None,
        seed: int = 42,
    ) -> None:
        self.backends = backends or ["superconducting", "trapped_ion", "neutral_atom"]
        self.rng = np.random.default_rng(seed)
        self._backend_profiles = self._build_profiles()

    def _build_profiles(self) -> Dict[str, dict]:
        """Build realistic performance profiles for each backend."""
        profiles: Dict[str, dict] = {}
        for name in self.backends:
            if name in _DEFAULT_BACKEND_PROFILES:
                profiles[name] = dict(_DEFAULT_BACKEND_PROFILES[name])
            else:
                # Fallback generic profile
                profiles[name] = {
                    "gate_time_ns": 100.0,
                    "two_qubit_gate_time_ns": 500.0,
                    "t1_us": 50.0,
                    "t2_us": 30.0,
                    "connectivity": "linear",
                    "single_qubit_fidelity": 0.999,
                    "two_qubit_fidelity": 0.99,
                    "readout_fidelity": 0.97,
                    "max_qubits": 100,
                }
        return profiles

    # ------------------------------------------------------------------
    # Circuit generation helpers
    # ------------------------------------------------------------------

    def _generate_random_circuit(
        self, n_qubits: int, depth: int
    ) -> List[tuple]:
        """Generate a random circuit as a list of gate tuples."""
        gates: List[tuple] = []
        sq_names = ["h", "x", "y", "z", "s", "t"]
        for _ in range(depth):
            for q in range(n_qubits):
                g = sq_names[int(self.rng.integers(len(sq_names)))]
                gates.append((g, [q], {}))
            for q in range(0, n_qubits - 1, 2):
                gates.append(("cx", [q, q + 1], {}))
        return gates

    def _generate_algorithm_circuit(
        self, algorithm: str, problem_size: int
    ) -> List[tuple]:
        """Generate gates for a named algorithm."""
        n = problem_size
        gates: List[tuple] = []

        if algorithm == "ghz":
            gates.append(("h", [0], {}))
            for i in range(n - 1):
                gates.append(("cx", [i, i + 1], {}))
        elif algorithm == "qft":
            for i in range(n):
                gates.append(("h", [i], {}))
                for j in range(i + 1, n):
                    angle = math.pi / (2 ** (j - i))
                    gates.append(("cx", [j, i], {}))
                    gates.append(("rz", [i], {"angle": -angle / 2}))
                    gates.append(("cx", [j, i], {}))
                    gates.append(("rz", [i], {"angle": angle / 2}))
        elif algorithm == "grover":
            # Simplified single-iteration Grover oracle + diffusion
            for q in range(n):
                gates.append(("h", [q], {}))
            # Oracle (mark state |11...1>)
            for q in range(n):
                gates.append(("x", [q], {}))
            if n >= 2:
                gates.append(("cx", [0, 1], {}))
            for q in range(n):
                gates.append(("x", [q], {}))
            # Diffusion
            for q in range(n):
                gates.append(("h", [q], {}))
                gates.append(("x", [q], {}))
            if n >= 2:
                gates.append(("cx", [0, 1], {}))
            for q in range(n):
                gates.append(("x", [q], {}))
                gates.append(("h", [q], {}))
        elif algorithm == "qaoa":
            for q in range(n):
                gates.append(("h", [q], {}))
            gamma = math.pi / 4
            beta = math.pi / 8
            for i in range(n - 1):
                gates.append(("cx", [i, i + 1], {}))
                gates.append(("rz", [i + 1], {"angle": 2 * gamma}))
                gates.append(("cx", [i, i + 1], {}))
            for q in range(n):
                gates.append(("rx", [q], {"angle": 2 * beta}))
        elif algorithm == "vqe":
            # Hardware-efficient ansatz: Ry-CX layers
            for q in range(n):
                gates.append(("ry", [q], {"angle": 0.5}))
            for i in range(n - 1):
                gates.append(("cx", [i, i + 1], {}))
            for q in range(n):
                gates.append(("ry", [q], {"angle": 0.3}))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return gates

    # ------------------------------------------------------------------
    # Simulation engine (pure numpy state-vector)
    # ------------------------------------------------------------------

    def _count_gates(
        self, gates: List[tuple]
    ) -> tuple:
        """Count single-qubit and two-qubit gates.

        Returns (total, single_qubit, two_qubit).
        """
        sq = 0
        tq = 0
        for g in gates:
            qubits = g[1]
            if len(qubits) == 1:
                sq += 1
            else:
                tq += 1
        return sq + tq, sq, tq

    def _compute_depth(self, gates: List[tuple], n_qubits: int) -> int:
        """Compute circuit depth."""
        qubit_layer = [0] * n_qubits
        for g in gates:
            qubits = g[1]
            if len(qubits) == 1:
                q = qubits[0]
                if 0 <= q < n_qubits:
                    qubit_layer[q] += 1
            elif len(qubits) >= 2:
                valid = [q for q in qubits if 0 <= q < n_qubits]
                if valid:
                    layer = max(qubit_layer[q] for q in valid) + 1
                    for q in valid:
                        qubit_layer[q] = layer
        return max(qubit_layer) if qubit_layer else 0

    def _simulate_fidelity(
        self,
        backend: str,
        n_qubits: int,
        depth: int,
        gate_count: int,
        two_qubit_count: int,
    ) -> float:
        """Estimate circuit fidelity given backend noise model.

        Uses a simple depolarising noise model:
            F = F_sq^(n_1q) * F_2q^(n_2q) * F_readout^(n_qubits)
            * exp(-depth * gate_time / T2)
        """
        profile = self._backend_profiles[backend]
        f_sq = profile["single_qubit_fidelity"]
        f_2q = profile["two_qubit_fidelity"]
        f_ro = profile["readout_fidelity"]
        t2_us = profile["t2_us"]
        gate_time_ns = profile["gate_time_ns"]
        tq_time_ns = profile["two_qubit_gate_time_ns"]

        single_qubit_count = gate_count - two_qubit_count

        gate_fidelity = (f_sq ** single_qubit_count) * (f_2q ** two_qubit_count)
        readout_fidelity = f_ro ** n_qubits

        # Decoherence: total time in us
        total_time_us = (
            single_qubit_count * gate_time_ns
            + two_qubit_count * tq_time_ns
        ) / 1000.0

        if t2_us > 0 and not math.isinf(t2_us):
            decoherence = math.exp(-total_time_us / t2_us)
        else:
            decoherence = 1.0

        fidelity = gate_fidelity * readout_fidelity * decoherence
        # Add small noise jitter for realism
        jitter = float(self.rng.normal(0, 0.0001))
        fidelity = max(0.0, min(1.0, fidelity + jitter))
        return fidelity

    def _estimate_time(
        self,
        backend: str,
        gate_count: int,
        two_qubit_count: int,
    ) -> float:
        """Estimate execution time in seconds."""
        profile = self._backend_profiles[backend]
        single_count = gate_count - two_qubit_count
        total_ns = (
            single_count * profile["gate_time_ns"]
            + two_qubit_count * profile["two_qubit_gate_time_ns"]
        )
        return total_ns * 1e-9

    # ------------------------------------------------------------------
    # Public benchmark methods
    # ------------------------------------------------------------------

    def benchmark_circuit(
        self,
        n_qubits: int,
        depth: int,
        circuit_type: str = "random",
    ) -> BenchmarkReport:
        """Run a circuit benchmark across all backends.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        depth : int
            Circuit depth.
        circuit_type : str
            Circuit type (``'random'`` only for now).

        Returns
        -------
        BenchmarkReport
        """
        gates = self._generate_random_circuit(n_qubits, depth)
        total, sq, tq = self._count_gates(gates)
        actual_depth = self._compute_depth(gates, n_qubits)

        results: List[DashboardBenchmark] = []
        for backend in self.backends:
            fidelity = self._simulate_fidelity(backend, n_qubits, actual_depth, total, tq)
            exec_time = self._estimate_time(backend, total, tq)
            results.append(
                DashboardBenchmark(
                    name=f"random_{n_qubits}q_d{depth}",
                    backend=backend,
                    n_qubits=n_qubits,
                    depth=actual_depth,
                    fidelity=fidelity,
                    time_seconds=exec_time,
                    gate_count=total,
                    two_qubit_count=tq,
                    metadata={"circuit_type": circuit_type},
                )
            )

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return BenchmarkReport(benchmarks=results, timestamp=timestamp)

    def benchmark_algorithm(
        self,
        algorithm: str,
        problem_size: int,
    ) -> BenchmarkReport:
        """Benchmark a quantum algorithm.

        Parameters
        ----------
        algorithm : str
            One of ``'ghz'``, ``'qft'``, ``'grover'``, ``'qaoa'``, ``'vqe'``.
        problem_size : int
            Number of qubits / problem dimension.

        Returns
        -------
        BenchmarkReport
        """
        gates = self._generate_algorithm_circuit(algorithm, problem_size)
        total, sq, tq = self._count_gates(gates)
        depth = self._compute_depth(gates, problem_size)

        results: List[DashboardBenchmark] = []
        for backend in self.backends:
            fidelity = self._simulate_fidelity(backend, problem_size, depth, total, tq)
            exec_time = self._estimate_time(backend, total, tq)
            results.append(
                DashboardBenchmark(
                    name=f"{algorithm}_{problem_size}q",
                    backend=backend,
                    n_qubits=problem_size,
                    depth=depth,
                    fidelity=fidelity,
                    time_seconds=exec_time,
                    gate_count=total,
                    two_qubit_count=tq,
                    metadata={"algorithm": algorithm},
                )
            )

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return BenchmarkReport(benchmarks=results, timestamp=timestamp)

    def sweep_qubits(
        self,
        qubit_range: range,
        depth: int = 10,
    ) -> BenchmarkReport:
        """Sweep qubit count and report scaling.

        Parameters
        ----------
        qubit_range : range
            Qubit counts to test.
        depth : int
            Circuit depth for each test.

        Returns
        -------
        BenchmarkReport
        """
        all_results: List[DashboardBenchmark] = []
        for n in qubit_range:
            report = self.benchmark_circuit(n, depth)
            all_results.extend(report.benchmarks)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return BenchmarkReport(benchmarks=all_results, timestamp=timestamp)

    def sweep_depth(
        self,
        n_qubits: int,
        depth_range: range,
    ) -> BenchmarkReport:
        """Sweep circuit depth and report fidelity decay.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        depth_range : range
            Depths to test.

        Returns
        -------
        BenchmarkReport
        """
        all_results: List[DashboardBenchmark] = []
        for d in depth_range:
            report = self.benchmark_circuit(n_qubits, d)
            all_results.extend(report.benchmarks)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return BenchmarkReport(benchmarks=all_results, timestamp=timestamp)


# ======================================================================
# Convenience functions
# ======================================================================


def quick_benchmark(n_qubits: int = 5, seed: int = 42) -> BenchmarkReport:
    """Quick 3-backend benchmark for given qubit count.

    Runs a random 10-depth circuit on superconducting, trapped-ion,
    and neutral-atom backend models.
    """
    bench = MultiBackendBenchmark(seed=seed)
    return bench.benchmark_circuit(n_qubits, depth=10)


def full_benchmark(seed: int = 42) -> BenchmarkReport:
    """Comprehensive benchmark suite across all configurations.

    Benchmarks multiple algorithms and qubit counts.
    """
    bench = MultiBackendBenchmark(seed=seed)
    all_results: List[DashboardBenchmark] = []

    for algo in ("ghz", "qft", "qaoa", "vqe"):
        report = bench.benchmark_algorithm(algo, 5)
        all_results.extend(report.benchmarks)

    sweep = bench.sweep_qubits(range(2, 8), depth=5)
    all_results.extend(sweep.benchmarks)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return BenchmarkReport(benchmarks=all_results, timestamp=timestamp)


def compare_backends(
    backends: list,
    n_qubits: int = 5,
) -> str:
    """Quick ASCII comparison of specified backends.

    Parameters
    ----------
    backends : list
        Backend names to compare.
    n_qubits : int
        Qubit count for the benchmark circuit.

    Returns
    -------
    str
        ASCII comparison table.
    """
    bench = MultiBackendBenchmark(backends=backends)
    report = bench.benchmark_circuit(n_qubits, depth=10)
    return report.ascii_table()
