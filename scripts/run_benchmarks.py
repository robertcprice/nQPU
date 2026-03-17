#!/usr/bin/env python3
"""Run cross-backend quantum hardware benchmarks.

Executes all 7 standard benchmark circuits across trapped-ion,
superconducting, and neutral-atom backends at qubit counts 3-8.
Results are written to results/benchmark_results.json.

Usage:
    python scripts/run_benchmarks.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from nqpu.benchmarks import (
    CrossBackendBenchmark,
    BenchmarkCircuit,
    fidelity_comparison_chart,
    gate_overhead_analysis,
    scaling_summary,
    toffoli_advantage_report,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUBIT_RANGE = range(3, 9)
CLIFFORD_DEPTH = 5
CLIFFORD_SEED = 42
QAOA_LAYERS = 2
SUPREMACY_DEPTH = 5
SUPREMACY_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_FILE = OUTPUT_DIR / "benchmark_results.json"


def _build_circuits(n: int) -> list[BenchmarkCircuit]:
    """Build the full suite of 7 benchmark circuits for *n* qubits.

    Parameters
    ----------
    n : int
        Number of qubits (must be >= 3).

    Returns
    -------
    list[BenchmarkCircuit]
        The 7 standard circuits.
    """
    circuits = [
        BenchmarkCircuit.bell_state(n_pairs=max(1, n // 2)),
        BenchmarkCircuit.ghz_state(n),
        BenchmarkCircuit.qft_circuit(n),
        BenchmarkCircuit.random_clifford(n, depth=CLIFFORD_DEPTH, seed=CLIFFORD_SEED),
        BenchmarkCircuit.toffoli_heavy(n),  # n >= 3 guaranteed by QUBIT_RANGE
        BenchmarkCircuit.qaoa_layer(n, p=QAOA_LAYERS),
        BenchmarkCircuit.supremacy_circuit(n, depth=SUPREMACY_DEPTH, seed=SUPREMACY_SEED),
    ]
    return circuits


def run_benchmarks() -> dict:
    """Execute all benchmarks and return the full results dictionary.

    Returns
    -------
    dict
        Nested structure:
        {
            "benchmarks": {qubit_count_str: {circuit_name: comparison_dict}},
            "scaling": {qubit_count_str: [comparison_dict]},
            "metadata": {...}
        }
    """
    all_benchmarks: dict[str, dict] = {}
    all_results_flat: list = []

    total_circuits = len(QUBIT_RANGE) * 7
    completed = 0

    for n in QUBIT_RANGE:
        print(f"\n{'='*60}")
        print(f"  Qubit count: {n}")
        print(f"{'='*60}")

        bench = CrossBackendBenchmark(num_qubits=n)
        circuits = _build_circuits(n)
        qubit_results: dict[str, dict] = {}

        for circuit in circuits:
            t_circuit = time.time()
            comparison = bench.run_circuit(circuit)
            dt_circuit = time.time() - t_circuit
            completed += 1

            qubit_results[circuit.name] = comparison.to_dict()
            all_results_flat.append(comparison)

            n_backends = len(comparison.results)
            print(
                f"  [{completed}/{total_circuits}] {circuit.name:<22} "
                f"({n}Q, {n_backends} backend configs) "
                f"{dt_circuit:.2f}s"
            )

        all_benchmarks[str(n)] = qubit_results

    # ------------------------------------------------------------------
    # Scaling analysis (GHZ circuit across all qubit counts)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Running GHZ scaling analysis ...")
    print(f"{'='*60}")

    # Use a fresh benchmark at the minimum qubit count; run_scaling_analysis
    # creates new CrossBackendBenchmark instances internally per qubit count
    # through the circuit generator, but the bench object's num_qubits is
    # only used for config -- the adapter creates sims with circuit.n_qubits.
    bench_scaling = CrossBackendBenchmark(num_qubits=min(QUBIT_RANGE))
    scaling_raw = bench_scaling.run_scaling_analysis(
        BenchmarkCircuit.ghz_state, QUBIT_RANGE
    )

    # Serialise scaling results
    scaling_serialised: dict[str, list[dict]] = {}
    for n_str, comp_list in scaling_raw.items():
        scaling_serialised[n_str] = [c.to_dict() for c in comp_list]

    return {
        "benchmarks": all_benchmarks,
        "scaling": scaling_serialised,
        "metadata": {
            "qubit_range": [int(n) for n in QUBIT_RANGE],
            "clifford_depth": CLIFFORD_DEPTH,
            "clifford_seed": CLIFFORD_SEED,
            "qaoa_layers": QAOA_LAYERS,
            "supremacy_depth": SUPREMACY_DEPTH,
            "supremacy_seed": SUPREMACY_SEED,
        },
    }


def print_analysis(all_results_flat: list, scaling_raw: dict) -> None:
    """Print human-readable analysis tables.

    Parameters
    ----------
    all_results_flat : list[BackendComparison]
        Flat list of all BackendComparison objects.
    scaling_raw : dict
        Raw scaling results from run_scaling_analysis.
    """
    print(f"\n{'='*60}")
    print("  FIDELITY COMPARISON CHART")
    print(f"{'='*60}")
    print(fidelity_comparison_chart(all_results_flat))

    print(f"\n{'='*60}")
    print("  TOFFOLI / CCZ ADVANTAGE REPORT")
    print(f"{'='*60}")
    print(toffoli_advantage_report(all_results_flat))

    print(f"\n{'='*60}")
    print("  SCALING SUMMARY")
    print(f"{'='*60}")
    # Flatten scaling dict for the summary function
    scaling_flat: dict[str, list] = {}
    for n_str, comp_list in scaling_raw.items():
        scaling_flat[n_str] = comp_list
    print(scaling_summary(scaling_flat))


def main() -> None:
    """Entry point: run benchmarks, save JSON, print analysis."""
    t_start = time.time()

    print("nQPU Cross-Backend Hardware Benchmarking")
    print(f"Qubit range: {list(QUBIT_RANGE)}")
    print(f"Circuits per qubit count: 7")
    print(f"Total circuit evaluations: {len(QUBIT_RANGE) * 7}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run all benchmarks
    results = run_benchmarks()

    # Write results to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {OUTPUT_FILE}")

    # Re-run lightweight objects for analysis printing.
    # We reconstruct BackendComparison objects from the flat benchmark run
    # rather than deserialising, so re-collect from a fresh pass.
    # For efficiency, use the scaling dict we already have.
    bench_for_analysis = CrossBackendBenchmark(num_qubits=5)
    circuits_5q = _build_circuits(5)
    comparisons_5q = [bench_for_analysis.run_circuit(c) for c in circuits_5q]

    scaling_for_print = bench_for_analysis.run_scaling_analysis(
        BenchmarkCircuit.ghz_state, QUBIT_RANGE
    )
    print_analysis(comparisons_5q, scaling_for_print)

    t_total = time.time() - t_start
    print(f"\nTotal wall-clock time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
