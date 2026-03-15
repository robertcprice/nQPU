"""Cross-platform native gate compiler benchmarking.

Compares compilation efficiency across different superconducting processor
architectures, each with a distinct native two-qubit gate:

    - IBM (ECR):           Echoed Cross-Resonance, heavy-hex topology
    - Google (sqrt-iSWAP): Parametric coupler, grid topology
    - Rigetti (CZ):        Tunable coupler, grid/octagonal topology

Metrics measured:
    - Native 1Q and 2Q gate counts after compilation
    - Circuit depth after compilation
    - Estimated circuit duration (ns)
    - Compilation overhead ratio vs. ideal (abstract) gate count

Usage::

    from nqpu.superconducting.compiler_bench import CompilerBenchmark

    bench = CompilerBenchmark(num_qubits=4)
    result = bench.benchmark_compilation("qft", 4)
    print(result)

References:
    - Sheldon et al., PRA 93, 060302 (2016) [ECR gate]
    - Arute et al., Nature 574, 505 (2019) [sqrt(iSWAP)]
    - Reagor et al., Science Advances 4 (2018) [CZ gate]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chip import ChipConfig, DevicePresets, NativeGateFamily
from .gates import GateInstruction, NativeGateType, TransmonGateSet
from .simulator import TransmonSimulator


# ======================================================================
# Data containers
# ======================================================================


@dataclass
class CompilationResult:
    """Result of compiling a circuit to a specific native gate family.

    Attributes
    ----------
    native_gate_family : str
        Native 2Q gate name (``"ecr"``, ``"sqrt_iswap"``, ``"cz"``).
    circuit_type : str
        Type of circuit compiled.
    num_qubits : int
        Number of qubits in the circuit.
    num_1q_gates : int
        Total single-qubit native gates after compilation.
    num_2q_gates : int
        Total two-qubit native gates after compilation.
    depth : int
        Circuit depth after compilation (estimated).
    estimated_duration_ns : float
        Estimated total circuit duration in nanoseconds.
    abstract_1q_gates : int
        Number of single-qubit gates in the abstract (uncompiled) circuit.
    abstract_2q_gates : int
        Number of two-qubit gates in the abstract (uncompiled) circuit.
    overhead_1q : float
        Ratio of native 1Q gates to abstract 1Q gates.
    overhead_2q : float
        Ratio of native 2Q gates to abstract 2Q gates.
    """

    native_gate_family: str
    circuit_type: str
    num_qubits: int
    num_1q_gates: int = 0
    num_2q_gates: int = 0
    depth: int = 0
    estimated_duration_ns: float = 0.0
    abstract_1q_gates: int = 0
    abstract_2q_gates: int = 0
    overhead_1q: float = 1.0
    overhead_2q: float = 1.0

    def __str__(self) -> str:
        return (
            f"CompilationResult("
            f"family={self.native_gate_family}, "
            f"circuit={self.circuit_type}, "
            f"qubits={self.num_qubits}, "
            f"native_1Q={self.num_1q_gates}, "
            f"native_2Q={self.num_2q_gates}, "
            f"depth={self.depth}, "
            f"duration={self.estimated_duration_ns:.0f}ns, "
            f"overhead_2Q={self.overhead_2q:.2f}x)"
        )


@dataclass(frozen=True)
class NativeGate:
    """A single native gate in a decomposition sequence.

    Used to display step-by-step CNOT/SWAP decompositions for each
    gate family.
    """

    name: str
    qubits: tuple[int, ...]
    angle: float = 0.0

    def __str__(self) -> str:
        q_str = ",".join(str(q) for q in self.qubits)
        if self.angle != 0.0:
            return f"{self.name}({self.angle:.4f})[{q_str}]"
        return f"{self.name}[{q_str}]"


# ======================================================================
# Circuit builders
# ======================================================================


def _build_bell(sim: TransmonSimulator) -> tuple[int, int]:
    """Bell state: H(0), CNOT(0,1). Returns (abstract_1q, abstract_2q)."""
    sim.h(0)
    sim.cnot(0, 1)
    return (1, 1)


def _build_ghz(sim: TransmonSimulator, n: int) -> tuple[int, int]:
    """GHZ state: H(0), CNOT(0,i) for i=1..n-1."""
    sim.h(0)
    for i in range(1, n):
        sim.cnot(0, i)
    return (1, n - 1)


def _build_qft(sim: TransmonSimulator, n: int) -> tuple[int, int]:
    """Quantum Fourier Transform.

    QFT on n qubits: H gates + controlled-phase rotations.
    We approximate controlled-Rz as CNOT + Rz for compilation counting.
    """
    gates_1q = 0
    gates_2q = 0
    for i in range(n):
        sim.h(i)
        gates_1q += 1
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            # Controlled-phase via CNOT + Rz decomposition
            sim.cnot(i, j)
            sim.rz(j, angle)
            sim.cnot(i, j)
            gates_1q += 1
            gates_2q += 2
    return (gates_1q, gates_2q)


def _build_random(
    sim: TransmonSimulator, n: int, depth: int = 5, seed: int = 42
) -> tuple[int, int]:
    """Random circuit with alternating 1Q and 2Q layers."""
    rng = np.random.RandomState(seed)
    gates_1q = 0
    gates_2q = 0
    for layer in range(depth):
        for q in range(n):
            sim.rx(q, rng.uniform(0, 2 * math.pi))
            sim.rz(q, rng.uniform(0, 2 * math.pi))
            gates_1q += 2
        offset = layer % 2
        for q in range(offset, n - 1, 2):
            sim.cnot(q, q + 1)
            gates_2q += 1
    return (gates_1q, gates_2q)


def _build_grover_oracle(sim: TransmonSimulator, n: int) -> tuple[int, int]:
    """Grover's diffusion operator for n qubits.

    H^n -> X^n -> multi-controlled-Z -> X^n -> H^n
    Multi-controlled-Z decomposed into CNOT ladder.
    """
    gates_1q = 0
    gates_2q = 0

    # H^n
    for q in range(n):
        sim.h(q)
        gates_1q += 1

    # X^n
    for q in range(n):
        sim.x(q)
        gates_1q += 1

    # Multi-controlled-Z via CNOT ladder
    if n >= 2:
        sim.h(n - 1)
        gates_1q += 1
        for q in range(n - 1):
            sim.cnot(q, n - 1)
            gates_2q += 1
        sim.h(n - 1)
        gates_1q += 1

    # X^n
    for q in range(n):
        sim.x(q)
        gates_1q += 1

    # H^n
    for q in range(n):
        sim.h(q)
        gates_1q += 1

    return (gates_1q, gates_2q)


_CIRCUIT_BUILDERS = {
    "bell": lambda sim, n: _build_bell(sim),
    "ghz": _build_ghz,
    "qft": _build_qft,
    "random": _build_random,
    "grover_oracle": _build_grover_oracle,
}


# ======================================================================
# NativeGateAnalyzer
# ======================================================================


class NativeGateAnalyzer:
    """Analyzes native gate decompositions for different gate families.

    Shows how standard gates (CNOT, SWAP) decompose into each vendor's
    native gate set, and computes overhead ratios for different circuit
    types.

    Examples
    --------
    >>> analyzer = NativeGateAnalyzer()
    >>> for gate in analyzer.decompose_cnot("ecr"):
    ...     print(gate)
    Rz(-1.5708)[1]
    ECR[0,1]
    Rz(1.5708)[0]
    SX[0]
    """

    # Gate timings (ns) per family
    _GATE_TIMES: dict[str, dict[str, float]] = {
        "ecr": {"1q": 25.0, "2q": 300.0},
        "sqrt_iswap": {"1q": 25.0, "2q": 32.0},
        "cz": {"1q": 25.0, "2q": 80.0},
    }

    def decompose_cnot(self, gate_family: str) -> list[NativeGate]:
        """Show CNOT decomposition for a given native gate family.

        Parameters
        ----------
        gate_family : str
            One of ``"ecr"``, ``"sqrt_iswap"``, ``"cz"``.

        Returns
        -------
        list[NativeGate]
            Ordered list of native gates implementing CNOT(0, 1).
        """
        if gate_family == "ecr":
            return self._cnot_via_ecr()
        elif gate_family == "sqrt_iswap":
            return self._cnot_via_sqrt_iswap()
        elif gate_family == "cz":
            return self._cnot_via_cz()
        else:
            raise ValueError(f"Unknown gate family: {gate_family}")

    def decompose_swap(self, gate_family: str) -> list[NativeGate]:
        """Show SWAP decomposition for a given native gate family.

        SWAP = 3 CNOTs, each further decomposed into native gates.

        Parameters
        ----------
        gate_family : str
            One of ``"ecr"``, ``"sqrt_iswap"``, ``"cz"``.

        Returns
        -------
        list[NativeGate]
        """
        # SWAP = CNOT(0,1) CNOT(1,0) CNOT(0,1)
        cnot_01 = self.decompose_cnot(gate_family)
        # For CNOT(1,0) we swap control/target labels
        cnot_10 = self._cnot_reversed(gate_family)
        return cnot_01 + cnot_10 + cnot_01

    def cnot_native_counts(self, gate_family: str) -> dict[str, int]:
        """Count native gates in a single CNOT decomposition.

        Returns
        -------
        dict[str, int]
            Keys: ``"1q"``, ``"2q"``, ``"total"``.
        """
        gates = self.decompose_cnot(gate_family)
        n_2q = sum(1 for g in gates if len(g.qubits) == 2)
        n_1q = len(gates) - n_2q
        return {"1q": n_1q, "2q": n_2q, "total": len(gates)}

    def swap_native_counts(self, gate_family: str) -> dict[str, int]:
        """Count native gates in a SWAP decomposition."""
        gates = self.decompose_swap(gate_family)
        n_2q = sum(1 for g in gates if len(g.qubits) == 2)
        n_1q = len(gates) - n_2q
        return {"1q": n_1q, "2q": n_2q, "total": len(gates)}

    def overhead_ratio(
        self, gate_family: str, circuit_type: str, num_qubits: int = 4
    ) -> float:
        """Compute compilation overhead ratio for a given circuit type.

        The overhead is defined as:
            (native_1q + native_2q) / (abstract_1q + abstract_2q)

        Parameters
        ----------
        gate_family : str
            Native gate family.
        circuit_type : str
            Circuit type (``"bell"``, ``"ghz"``, ``"qft"``, ``"random"``,
            ``"grover_oracle"``).
        num_qubits : int
            Number of qubits.

        Returns
        -------
        float
            Ratio >= 1.0 (higher means more overhead).
        """
        config = self._config_for_family(gate_family, num_qubits)
        sim = TransmonSimulator(config, execution_mode="ideal")

        builder = _CIRCUIT_BUILDERS.get(circuit_type)
        if builder is None:
            raise ValueError(
                f"Unknown circuit type '{circuit_type}'. "
                f"Supported: {list(_CIRCUIT_BUILDERS.keys())}"
            )

        abstract_1q, abstract_2q = builder(sim, num_qubits)
        stats = sim.circuit_stats()

        abstract_total = max(abstract_1q + abstract_2q, 1)
        native_total = stats.native_1q_count + stats.native_2q_count
        return native_total / abstract_total

    def cnot_duration_ns(self, gate_family: str) -> float:
        """Estimated duration of a CNOT in nanoseconds for a gate family.

        Sums gate durations along the critical path of the decomposition.
        """
        gates = self.decompose_cnot(gate_family)
        times = self._GATE_TIMES.get(gate_family, {"1q": 25.0, "2q": 200.0})
        total = 0.0
        for g in gates:
            if len(g.qubits) == 2:
                total += times["2q"]
            else:
                total += times["1q"]
        return total

    # ------------------------------------------------------------------
    # Internal decompositions
    # ------------------------------------------------------------------

    @staticmethod
    def _cnot_via_ecr() -> list[NativeGate]:
        """CNOT(0,1) via ECR: Rz(-pi/2)_t, ECR(c,t), Rz(pi/2)_c, SX_c."""
        return [
            NativeGate("Rz", (1,), -math.pi / 2),
            NativeGate("ECR", (0, 1)),
            NativeGate("Rz", (0,), math.pi / 2),
            NativeGate("SX", (0,)),
        ]

    @staticmethod
    def _cnot_via_sqrt_iswap() -> list[NativeGate]:
        """CNOT(0,1) via 2x sqrt(iSWAP) + 1Q corrections."""
        return [
            NativeGate("Rz", (1,), math.pi / 2),
            NativeGate("sqrt_iSWAP", (0, 1)),
            NativeGate("Rz", (0,), math.pi),
            NativeGate("sqrt_iSWAP", (0, 1)),
            NativeGate("Rz", (0,), math.pi / 2),
            NativeGate("Rz", (1,), math.pi / 2),
        ]

    @staticmethod
    def _cnot_via_cz() -> list[NativeGate]:
        """CNOT(0,1) via CZ: H_t, CZ, H_t (H = Rz.SX.Rz)."""
        return [
            # H on target
            NativeGate("Rz", (1,), math.pi),
            NativeGate("SX", (1,)),
            NativeGate("Rz", (1,), math.pi),
            # CZ
            NativeGate("CZ", (0, 1)),
            # H on target
            NativeGate("Rz", (1,), math.pi),
            NativeGate("SX", (1,)),
            NativeGate("Rz", (1,), math.pi),
        ]

    def _cnot_reversed(self, gate_family: str) -> list[NativeGate]:
        """CNOT(1,0) decomposition -- swap qubit labels."""
        forward = self.decompose_cnot(gate_family)
        reversed_gates: list[NativeGate] = []
        for g in forward:
            new_qubits = tuple(1 - q for q in g.qubits)
            reversed_gates.append(NativeGate(g.name, new_qubits, g.angle))
        return reversed_gates

    @staticmethod
    def _config_for_family(gate_family: str, num_qubits: int) -> ChipConfig:
        """Build a minimal ChipConfig for a given gate family."""
        family_map = {
            "ecr": DevicePresets.IBM_HERON,
            "sqrt_iswap": DevicePresets.GOOGLE_SYCAMORE,
            "cz": DevicePresets.RIGETTI_ANKAA,
        }
        preset = family_map.get(gate_family)
        if preset is None:
            raise ValueError(f"Unknown gate family: {gate_family}")
        return preset.build(num_qubits=num_qubits)


# ======================================================================
# CompilerBenchmark
# ======================================================================


class CompilerBenchmark:
    """Benchmarks native gate compilation across processor architectures.

    For each circuit type and gate family, compiles the circuit and measures
    the resulting native gate count, circuit depth, and estimated duration.

    Parameters
    ----------
    num_qubits : int
        Default number of qubits for benchmarks.

    Examples
    --------
    >>> bench = CompilerBenchmark(num_qubits=4)
    >>> result = bench.benchmark_compilation("qft", 4)
    >>> print(result)

    >>> table = bench.benchmark_all_circuits()
    >>> bench.print_report(table)
    """

    # Gate families to benchmark
    GATE_FAMILIES = ["ecr", "sqrt_iswap", "cz"]
    FAMILY_LABELS = {
        "ecr": "IBM (ECR)",
        "sqrt_iswap": "Google (sqrt-iSWAP)",
        "cz": "Rigetti (CZ)",
    }

    # Gate timing (ns) per family
    _TIMING: dict[str, dict[str, float]] = {
        "ecr": {"1q": 25.0, "2q": 300.0},
        "sqrt_iswap": {"1q": 25.0, "2q": 32.0},
        "cz": {"1q": 25.0, "2q": 80.0},
    }

    def __init__(self, num_qubits: int = 4) -> None:
        self.num_qubits = num_qubits
        self._analyzer = NativeGateAnalyzer()

    def benchmark_compilation(
        self,
        circuit_type: str,
        num_qubits: int | None = None,
        gate_family: str | None = None,
    ) -> CompilationResult | list[CompilationResult]:
        """Benchmark compilation for a circuit type.

        If ``gate_family`` is specified, returns a single result.
        Otherwise, benchmarks all three families and returns a list.

        Parameters
        ----------
        circuit_type : str
            One of ``"bell"``, ``"ghz"``, ``"qft"``, ``"random"``,
            ``"grover_oracle"``.
        num_qubits : int, optional
            Override the default qubit count.
        gate_family : str, optional
            Restrict to a single gate family.

        Returns
        -------
        CompilationResult or list[CompilationResult]
        """
        n = num_qubits or self.num_qubits

        if gate_family is not None:
            return self._compile_single(circuit_type, n, gate_family)

        results = []
        for fam in self.GATE_FAMILIES:
            results.append(self._compile_single(circuit_type, n, fam))
        return results

    def benchmark_all_circuits(
        self, num_qubits: int | None = None
    ) -> dict[str, list[CompilationResult]]:
        """Benchmark all circuit types across all gate families.

        Parameters
        ----------
        num_qubits : int, optional
            Override the default qubit count.

        Returns
        -------
        dict[str, list[CompilationResult]]
            Keyed by circuit type.
        """
        n = num_qubits or self.num_qubits
        results: dict[str, list[CompilationResult]] = {}

        for ctype in _CIRCUIT_BUILDERS:
            family_results = []
            for fam in self.GATE_FAMILIES:
                family_results.append(self._compile_single(ctype, n, fam))
            results[ctype] = family_results

        return results

    def _compile_single(
        self, circuit_type: str, num_qubits: int, gate_family: str
    ) -> CompilationResult:
        """Compile one circuit type on one gate family and collect metrics."""
        builder = _CIRCUIT_BUILDERS.get(circuit_type)
        if builder is None:
            raise ValueError(
                f"Unknown circuit type '{circuit_type}'. "
                f"Supported: {list(_CIRCUIT_BUILDERS.keys())}"
            )

        # Build config for this gate family
        config = NativeGateAnalyzer._config_for_family(gate_family, num_qubits)
        sim = TransmonSimulator(config, execution_mode="ideal")

        # Run the circuit builder
        abstract_1q, abstract_2q = builder(sim, num_qubits)

        # Collect native gate stats
        stats = sim.circuit_stats()
        native_1q = stats.native_1q_count
        native_2q = stats.native_2q_count

        # Estimate depth: naive model -- 2Q gates dominate, assume serial
        # A more accurate model would schedule gates on the coupling graph,
        # but this gives a useful comparative metric.
        depth = native_1q + native_2q  # pessimistic serial estimate

        # Estimate duration
        timing = self._TIMING.get(gate_family, {"1q": 25.0, "2q": 200.0})
        duration = native_1q * timing["1q"] + native_2q * timing["2q"]

        # Overhead ratios
        overhead_1q = native_1q / max(abstract_1q, 1)
        overhead_2q = native_2q / max(abstract_2q, 1)

        return CompilationResult(
            native_gate_family=gate_family,
            circuit_type=circuit_type,
            num_qubits=num_qubits,
            num_1q_gates=native_1q,
            num_2q_gates=native_2q,
            depth=depth,
            estimated_duration_ns=duration,
            abstract_1q_gates=abstract_1q,
            abstract_2q_gates=abstract_2q,
            overhead_1q=overhead_1q,
            overhead_2q=overhead_2q,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(results: dict[str, list[CompilationResult]]) -> None:
        """Print a formatted compilation benchmark report.

        Parameters
        ----------
        results : dict[str, list[CompilationResult]]
            Output from :meth:`benchmark_all_circuits`.
        """
        fam_labels = CompilerBenchmark.FAMILY_LABELS

        print("=" * 90)
        print("NATIVE GATE COMPILATION BENCHMARK")
        print("=" * 90)

        for ctype, family_results in results.items():
            if not family_results:
                continue
            n = family_results[0].num_qubits
            abs_1q = family_results[0].abstract_1q_gates
            abs_2q = family_results[0].abstract_2q_gates

            print(f"\n--- {ctype.upper()} ({n}Q) --- "
                  f"[abstract: {abs_1q} 1Q + {abs_2q} 2Q gates]")
            print(
                f"  {'Gate Family':<25} {'Native 1Q':>10} {'Native 2Q':>10} "
                f"{'Depth':>8} {'Duration(ns)':>13} {'2Q Overhead':>12}"
            )
            print("  " + "-" * 78)

            for r in family_results:
                label = fam_labels.get(r.native_gate_family, r.native_gate_family)
                print(
                    f"  {label:<25} {r.num_1q_gates:>10} {r.num_2q_gates:>10} "
                    f"{r.depth:>8} {r.estimated_duration_ns:>13.0f} "
                    f"{r.overhead_2q:>11.2f}x"
                )

        print("\n" + "=" * 90)

    @staticmethod
    def print_decomposition_report() -> None:
        """Print native gate decompositions for CNOT and SWAP."""
        analyzer = NativeGateAnalyzer()
        families = ["ecr", "sqrt_iswap", "cz"]
        labels = {
            "ecr": "IBM (ECR)",
            "sqrt_iswap": "Google (sqrt-iSWAP)",
            "cz": "Rigetti (CZ)",
        }

        print("=" * 72)
        print("NATIVE GATE DECOMPOSITION REFERENCE")
        print("=" * 72)

        # CNOT decompositions
        print("\n--- CNOT(0, 1) Decomposition ---\n")
        for fam in families:
            gates = analyzer.decompose_cnot(fam)
            counts = analyzer.cnot_native_counts(fam)
            duration = analyzer.cnot_duration_ns(fam)
            print(f"  {labels[fam]}:")
            print(f"    Gates: {counts['1q']} 1Q + {counts['2q']} 2Q "
                  f"= {counts['total']} total, ~{duration:.0f} ns")
            for i, g in enumerate(gates, 1):
                print(f"      {i}. {g}")
            print()

        # SWAP decompositions
        print("--- SWAP(0, 1) Decomposition ---\n")
        for fam in families:
            counts = analyzer.swap_native_counts(fam)
            duration = 3 * analyzer.cnot_duration_ns(fam)
            print(f"  {labels[fam]}: "
                  f"{counts['1q']} 1Q + {counts['2q']} 2Q "
                  f"= {counts['total']} total, ~{duration:.0f} ns")
        print()

        # CNOT duration comparison
        print("--- CNOT Duration Comparison ---\n")
        print(f"  {'Family':<25} {'Duration (ns)':>15}")
        print("  " + "-" * 40)
        for fam in families:
            dur = analyzer.cnot_duration_ns(fam)
            print(f"  {labels[fam]:<25} {dur:>15.0f}")

        print("\n" + "=" * 72)


# ======================================================================
# Demo entry point
# ======================================================================


if __name__ == "__main__":
    print("=" * 72)
    print("COMPILER BENCHMARK DEMO")
    print("nQPU Superconducting Backend")
    print("=" * 72)

    # --- 1. Gate decomposition reference ---
    CompilerBenchmark.print_decomposition_report()

    # --- 2. Compilation benchmarks ---
    print()
    bench = CompilerBenchmark(num_qubits=4)
    all_results = bench.benchmark_all_circuits(num_qubits=4)
    CompilerBenchmark.print_report(all_results)

    # --- 3. Overhead ratios ---
    print("\n--- Overhead Ratios (native / abstract gate count) ---\n")
    analyzer = NativeGateAnalyzer()
    families = ["ecr", "sqrt_iswap", "cz"]
    labels = {
        "ecr": "IBM (ECR)",
        "sqrt_iswap": "Google (sqrt-iSWAP)",
        "cz": "Rigetti (CZ)",
    }
    circuit_types = ["bell", "ghz", "qft", "random", "grover_oracle"]

    print(f"  {'Circuit':<18}", end="")
    for fam in families:
        print(f"{labels[fam]:>22}", end="")
    print()
    print("  " + "-" * 84)

    for ctype in circuit_types:
        row = f"  {ctype:<18}"
        for fam in families:
            ratio = analyzer.overhead_ratio(fam, ctype, num_qubits=4)
            row += f"{ratio:>22.2f}x"
        print(row)

    # --- 4. Scaling analysis ---
    print("\n\n--- 2Q Gate Scaling: QFT Circuit ---\n")
    print(f"  {'Qubits':<10}", end="")
    for fam in families:
        print(f"{labels[fam]:>22}", end="")
    print()
    print("  " + "-" * 76)

    for n in [2, 3, 4, 5, 6]:
        row = f"  {n:<10}"
        for fam in families:
            result = bench.benchmark_compilation("qft", num_qubits=n, gate_family=fam)
            assert isinstance(result, CompilationResult)
            row += f"{result.num_2q_gates:>22}"
        print(row)

    print("\nDone.")
