#!/usr/bin/env python3
"""Generate publication-quality figures from benchmark results.

Reads results/benchmark_results.json (produced by run_benchmarks.py)
and saves four matplotlib figures to results/figures/.

Figures:
    1. fidelity_comparison.png  -- Noisy fidelity grouped bar chart (5 qubits)
    2. gate_overhead.png        -- Native gate count overhead grouped bars
    3. toffoli_advantage.png    -- Neutral-atom CCZ advantage vs qubit count
    4. scaling_fidelity.png     -- GHZ fidelity scaling line plot (3-8 qubits)

Usage:
    python scripts/generate_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "benchmark_results.json"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# ---------------------------------------------------------------------------
# Backend display configuration
# ---------------------------------------------------------------------------

BACKEND_KEYS_NOISY = ["ion_noisy", "sc_noisy", "na_noisy"]
BACKEND_LABELS = {
    "ion_noisy": "Trapped-Ion",
    "sc_noisy": "Superconducting",
    "na_noisy": "Neutral-Atom",
}
BACKEND_COLORS = {
    "ion_noisy": "#2196F3",
    "sc_noisy": "#FF9800",
    "na_noisy": "#4CAF50",
}


def load_results() -> dict:
    """Load benchmark results from JSON.

    Returns
    -------
    dict
        The full results dictionary written by run_benchmarks.py.

    Raises
    ------
    FileNotFoundError
        If the results file does not exist.
    """
    if not RESULTS_FILE.exists():
        print(
            f"ERROR: Results file not found at {RESULTS_FILE}\n"
            f"Run 'python scripts/run_benchmarks.py' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(RESULTS_FILE) as f:
        return json.load(f)


def _get_qubit_data(benchmarks: dict, target_qubits: str) -> dict:
    """Return the benchmark data for a specific qubit count.

    Falls back to the first available qubit count if the target
    is not present.

    Parameters
    ----------
    benchmarks : dict
        The "benchmarks" sub-dict from the results file.
    target_qubits : str
        Desired qubit count as a string (e.g. "5").

    Returns
    -------
    dict
        ``{circuit_name: comparison_dict}`` for the chosen qubit count.
    """
    if target_qubits in benchmarks:
        return benchmarks[target_qubits]
    # Fallback to the first available
    first_key = sorted(benchmarks.keys(), key=int)[0]
    print(f"WARNING: {target_qubits}-qubit data not found, using {first_key}-qubit data")
    return benchmarks[first_key]


# ======================================================================
# Figure 1: Fidelity Comparison
# ======================================================================


def figure_fidelity_comparison(benchmarks: dict) -> None:
    """Grouped bar chart of noisy fidelity per circuit per backend (5 qubits).

    Parameters
    ----------
    benchmarks : dict
        The "benchmarks" sub-dict from the results file.
    """
    data = _get_qubit_data(benchmarks, "5")
    circuit_names = list(data.keys())
    n_circuits = len(circuit_names)
    n_backends = len(BACKEND_KEYS_NOISY)

    x = np.arange(n_circuits)
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, bk in enumerate(BACKEND_KEYS_NOISY):
        fidelities = []
        for cname in circuit_names:
            comp = data[cname]
            results = comp.get("results", {})
            if bk in results:
                fidelities.append(results[bk].get("fidelity_vs_ideal", 0.0))
            else:
                fidelities.append(0.0)

        offset = (i - (n_backends - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            fidelities,
            bar_width,
            label=BACKEND_LABELS[bk],
            color=BACKEND_COLORS[bk],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Benchmark Circuit", fontsize=12)
    ax.set_ylabel("Noisy Fidelity (Bhattacharyya)", fontsize=12)
    ax.set_title("Noisy Fidelity Comparison (5 Qubits)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(circuit_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    outpath = FIGURES_DIR / "fidelity_comparison.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ======================================================================
# Figure 2: Gate Overhead
# ======================================================================


def figure_gate_overhead(benchmarks: dict) -> None:
    """Grouped bar chart of total native gate counts per backend per circuit.

    Parameters
    ----------
    benchmarks : dict
        The "benchmarks" sub-dict from the results file.
    """
    data = _get_qubit_data(benchmarks, "5")
    circuit_names = list(data.keys())
    n_circuits = len(circuit_names)
    n_backends = len(BACKEND_KEYS_NOISY)

    x = np.arange(n_circuits)
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, bk in enumerate(BACKEND_KEYS_NOISY):
        gate_counts = []
        for cname in circuit_names:
            comp = data[cname]
            results = comp.get("results", {})
            if bk in results:
                r = results[bk]
                total = (
                    r.get("num_gates_1q", 0)
                    + r.get("num_gates_2q", 0)
                    + r.get("num_gates_3q", 0)
                )
                gate_counts.append(total)
            else:
                gate_counts.append(0)

        offset = (i - (n_backends - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            gate_counts,
            bar_width,
            label=BACKEND_LABELS[bk],
            color=BACKEND_COLORS[bk],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Benchmark Circuit", fontsize=12)
    ax.set_ylabel("Total Native Gates", fontsize=12)
    ax.set_title("Native Gate Count Overhead", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(circuit_names, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    outpath = FIGURES_DIR / "gate_overhead.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ======================================================================
# Figure 3: Toffoli Advantage
# ======================================================================


def figure_toffoli_advantage(benchmarks: dict) -> None:
    """Bar chart of Toffoli circuit gate counts across backends vs qubit count.

    Shows how the neutral-atom native CCZ advantage grows with system size.

    Parameters
    ----------
    benchmarks : dict
        The "benchmarks" sub-dict from the results file.
    """
    qubit_counts = sorted(benchmarks.keys(), key=int)
    n_qubits_list = [int(q) for q in qubit_counts]

    # Collect total gate counts for the Toffoli circuit at each qubit count
    backend_totals: dict[str, list[int]] = {bk: [] for bk in BACKEND_KEYS_NOISY}

    for q_str in qubit_counts:
        q_data = benchmarks[q_str]
        # Find the Toffoli circuit (name may be "Toffoli Circuit")
        toffoli_key = None
        for cname in q_data:
            if "toffoli" in cname.lower():
                toffoli_key = cname
                break

        for bk in BACKEND_KEYS_NOISY:
            if toffoli_key and bk in q_data[toffoli_key].get("results", {}):
                r = q_data[toffoli_key]["results"][bk]
                total = (
                    r.get("num_gates_1q", 0)
                    + r.get("num_gates_2q", 0)
                    + r.get("num_gates_3q", 0)
                )
                backend_totals[bk].append(total)
            else:
                backend_totals[bk].append(0)

    x = np.arange(len(n_qubits_list))
    bar_width = 0.25
    n_backends = len(BACKEND_KEYS_NOISY)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, bk in enumerate(BACKEND_KEYS_NOISY):
        offset = (i - (n_backends - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            backend_totals[bk],
            bar_width,
            label=BACKEND_LABELS[bk],
            color=BACKEND_COLORS[bk],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Number of Qubits", fontsize=12)
    ax.set_ylabel("Total Native Gates (Toffoli Circuit)", fontsize=12)
    ax.set_title(
        "Neutral Atom Toffoli/CCZ Gate Advantage",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(n_qubits_list, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    outpath = FIGURES_DIR / "toffoli_advantage.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ======================================================================
# Figure 4: Scaling Fidelity
# ======================================================================


def figure_scaling_fidelity(scaling: dict) -> None:
    """Line plot of GHZ state noisy fidelity vs qubit count per backend.

    Parameters
    ----------
    scaling : dict
        The "scaling" sub-dict from the results file.
    """
    qubit_counts = sorted(scaling.keys(), key=int)
    n_qubits_list = [int(q) for q in qubit_counts]

    backend_fidelities: dict[str, list[float]] = {bk: [] for bk in BACKEND_KEYS_NOISY}

    for q_str in qubit_counts:
        comp_list = scaling[q_str]
        # Each entry is a comparison_dict; take the first one
        if comp_list:
            comp = comp_list[0]
            results = comp.get("results", {})
            for bk in BACKEND_KEYS_NOISY:
                if bk in results:
                    backend_fidelities[bk].append(
                        results[bk].get("fidelity_vs_ideal", 0.0)
                    )
                else:
                    backend_fidelities[bk].append(0.0)
        else:
            for bk in BACKEND_KEYS_NOISY:
                backend_fidelities[bk].append(0.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    marker_styles = {"ion_noisy": "o", "sc_noisy": "s", "na_noisy": "^"}

    for bk in BACKEND_KEYS_NOISY:
        ax.plot(
            n_qubits_list,
            backend_fidelities[bk],
            marker=marker_styles.get(bk, "o"),
            color=BACKEND_COLORS[bk],
            label=BACKEND_LABELS[bk],
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Number of Qubits", fontsize=12)
    ax.set_ylabel("Noisy Fidelity (Bhattacharyya)", fontsize=12)
    ax.set_title("GHZ State Fidelity Scaling", fontsize=14, fontweight="bold")
    ax.set_xticks(n_qubits_list)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    outpath = FIGURES_DIR / "scaling_fidelity.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Load results and generate all four figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading benchmark results ...")
    results = load_results()

    benchmarks = results.get("benchmarks", {})
    scaling = results.get("scaling", {})

    if not benchmarks:
        print("ERROR: No benchmark data found in results file.", file=sys.stderr)
        sys.exit(1)

    print("Generating figures ...")

    figure_fidelity_comparison(benchmarks)
    figure_gate_overhead(benchmarks)
    figure_toffoli_advantage(benchmarks)

    if scaling:
        figure_scaling_fidelity(scaling)
    else:
        print("  WARNING: No scaling data found; skipping scaling_fidelity.png")

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
