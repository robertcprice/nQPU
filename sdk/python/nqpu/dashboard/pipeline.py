"""End-to-end quantum computing pipeline.

Provides a ``QuantumPipeline`` that chains circuit construction, analysis,
transpilation, optimisation, noise simulation, and measurement into a
single configurable workflow.  All simulation is pure-numpy statevector.

Example::

    from nqpu.dashboard import quick_run
    result = quick_run(
        gates=[("h", [0], {}), ("cx", [0, 1], {}), ("measure", [0, 1], {})],
        n_qubits=2,
    )
    print(result.summary())
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class PipelineStage:
    """A stage in the quantum pipeline."""

    name: str
    description: str


@dataclass
class StageResult:
    """Result of a single pipeline stage."""

    stage: PipelineStage
    success: bool
    output: dict
    time_seconds: float
    metrics: dict


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    n_qubits: int
    optimization_level: int = 1  # 0=none, 1=light, 2=heavy, 3=aggressive
    target_backend: str = "generic"
    noise_model: Optional[str] = None  # "ideal", "depolarizing", "amplitude_damping"
    error_rate: float = 0.001
    shots: int = 1024
    seed: int = 42


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""

    config: PipelineConfig
    stage_results: List[StageResult]
    final_counts: dict
    total_time: float

    def summary(self) -> str:
        """ASCII summary of pipeline execution."""
        lines = [
            "=" * 60,
            "PIPELINE EXECUTION SUMMARY",
            "=" * 60,
            f"  Qubits:             {self.config.n_qubits}",
            f"  Optimization level: {self.config.optimization_level}",
            f"  Target backend:     {self.config.target_backend}",
            f"  Noise model:        {self.config.noise_model or 'ideal'}",
            f"  Shots:              {self.config.shots}",
            f"  Total time:         {self.total_time:.6f} s",
            "",
            "  Stages:",
        ]
        for sr in self.stage_results:
            status = "OK" if sr.success else "FAIL"
            lines.append(
                f"    [{status}] {sr.stage.name:<20} {sr.time_seconds:.6f} s"
            )
        lines.append("")
        if self.final_counts:
            lines.append("  Top measurement outcomes:")
            sorted_counts = sorted(
                self.final_counts.items(), key=lambda x: x[1], reverse=True
            )
            for bitstring, count in sorted_counts[:8]:
                pct = 100.0 * count / self.config.shots
                lines.append(f"    |{bitstring}> : {count:>6} ({pct:5.1f}%)")
        lines.append("=" * 60)
        return "\n".join(lines)

    def stage_timings(self) -> str:
        """ASCII bar chart of stage timings."""
        if not self.stage_results:
            return "No stages to display."

        max_time = max(sr.time_seconds for sr in self.stage_results)
        if max_time <= 0:
            max_time = 1e-9
        bar_width = 40
        lines = ["STAGE TIMINGS", "-" * 60]
        for sr in self.stage_results:
            bar_len = int(bar_width * sr.time_seconds / max_time)
            bar_len = max(bar_len, 1) if sr.time_seconds > 0 else 0
            bar = "#" * bar_len
            lines.append(
                f"  {sr.stage.name:<16} |{bar:<{bar_width}}| "
                f"{sr.time_seconds:.6f} s"
            )
        return "\n".join(lines)


# ======================================================================
# Gate matrices (pure numpy)
# ======================================================================

_I2 = np.eye(2, dtype=np.complex128)
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)


def _rx(theta: float) -> np.ndarray:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def _apply_single_qubit_gate(
    state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    """Apply a single-qubit gate to the statevector."""
    dim = 2**n_qubits
    new_state = np.zeros(dim, dtype=np.complex128)
    step = 2**qubit
    for i in range(0, dim, 2 * step):
        for j in range(step):
            idx0 = i + j
            idx1 = i + j + step
            a, b = state[idx0], state[idx1]
            new_state[idx0] = gate[0, 0] * a + gate[0, 1] * b
            new_state[idx1] = gate[1, 0] * a + gate[1, 1] * b
    return new_state


def _apply_cnot(
    state: np.ndarray, control: int, target: int, n_qubits: int
) -> np.ndarray:
    """Apply a CNOT gate."""
    dim = 2**n_qubits
    new_state = state.copy()
    for i in range(dim):
        if (i >> control) & 1:
            partner = i ^ (1 << target)
            new_state[i], new_state[partner] = state[partner], state[i]
    # Fix double-swap: only swap when control is set
    new_state2 = state.copy()
    for i in range(dim):
        if (i >> control) & 1:
            partner = i ^ (1 << target)
            new_state2[i] = state[partner]
        else:
            new_state2[i] = state[i]
    return new_state2


# ======================================================================
# Gate name resolution
# ======================================================================

_SINGLE_GATE_MAP = {
    "h": _H,
    "x": _X,
    "y": _Y,
    "z": _Z,
    "s": _S,
    "t": _T,
}


def _resolve_single_gate(name: str, params: dict) -> Optional[np.ndarray]:
    """Resolve a gate name to a 2x2 matrix."""
    lower = name.lower()
    if lower in _SINGLE_GATE_MAP:
        return _SINGLE_GATE_MAP[lower]
    if lower == "rx":
        return _rx(params.get("angle", 0.0))
    if lower == "ry":
        return _ry(params.get("angle", 0.0))
    if lower == "rz":
        return _rz(params.get("angle", 0.0))
    return None


# ======================================================================
# Pipeline implementation
# ======================================================================

# Standard stages
STAGE_CONSTRUCT = PipelineStage("construct", "Build circuit from gate list")
STAGE_ANALYZE = PipelineStage("analyze", "Analyse circuit properties")
STAGE_TRANSPILE = PipelineStage("transpile", "Decompose into basis gates")
STAGE_OPTIMIZE = PipelineStage("optimize", "Optimise gate sequence")
STAGE_SIMULATE = PipelineStage("simulate", "Simulate with optional noise")
STAGE_MEASURE = PipelineStage("measure", "Sample measurements")


class QuantumPipeline:
    """End-to-end quantum computing pipeline.

    Stages:

    1. **Circuit construction** from description or gate list
    2. **Circuit analysis** (depth, gate count, entanglement structure)
    3. **Transpilation** (basis gate decomposition)
    4. **Optimisation** (gate cancellation, commutation)
    5. **Noise simulation** (optional depolarising / amplitude damping)
    6. **Measurement and statistics**

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def run(self, gates: list) -> PipelineResult:
        """Run full pipeline on a gate list.

        Parameters
        ----------
        gates : list
            List of ``(gate_name, qubits, params)`` tuples.
            Example: ``[("h", [0], {}), ("cx", [0, 1], {})]``

        Returns
        -------
        PipelineResult
        """
        t_start = time.monotonic()
        stage_results: List[StageResult] = []

        # Stage 1: Construct
        sr_construct = self._stage_construct(gates)
        stage_results.append(sr_construct)
        parsed_gates = sr_construct.output.get("parsed_gates", gates)

        # Stage 2: Analyze
        sr_analyze = self._stage_analyze(parsed_gates)
        stage_results.append(sr_analyze)

        # Stage 3: Transpile
        sr_transpile = self._stage_transpile(parsed_gates)
        stage_results.append(sr_transpile)
        transpiled_gates = sr_transpile.output.get("transpiled_gates", parsed_gates)

        # Stage 4: Optimize
        sr_optimize = self._stage_optimize(transpiled_gates)
        stage_results.append(sr_optimize)
        optimized_gates = sr_optimize.output.get("optimized_gates", transpiled_gates)

        # Stage 5: Simulate
        sr_simulate = self._stage_simulate(optimized_gates)
        stage_results.append(sr_simulate)
        state = sr_simulate.output.get("state")

        # Stage 6: Measure
        sr_measure = self._stage_measure(state)
        stage_results.append(sr_measure)
        counts = sr_measure.output.get("counts", {})

        total_time = time.monotonic() - t_start
        return PipelineResult(
            config=self.config,
            stage_results=stage_results,
            final_counts=counts,
            total_time=total_time,
        )

    # ------------------------------------------------------------------
    # Individual stages
    # ------------------------------------------------------------------

    def _stage_construct(self, gates: list) -> StageResult:
        """Build parsed gate list from raw input."""
        t0 = time.monotonic()
        parsed: List[Tuple[str, list, dict]] = []
        measurement_qubits: List[int] = []

        for g in gates:
            name = str(g[0]).lower()
            qubits = list(g[1]) if len(g) > 1 else []
            params = dict(g[2]) if len(g) > 2 else {}
            if name == "measure":
                measurement_qubits.extend(qubits)
            else:
                parsed.append((name, qubits, params))

        dt = time.monotonic() - t0
        return StageResult(
            stage=STAGE_CONSTRUCT,
            success=True,
            output={
                "parsed_gates": parsed,
                "measurement_qubits": measurement_qubits,
                "raw_gate_count": len(gates),
            },
            time_seconds=dt,
            metrics={"gate_count": len(parsed), "measurement_count": len(measurement_qubits)},
        )

    def _stage_analyze(self, gates: list) -> StageResult:
        """Analyse circuit properties."""
        t0 = time.monotonic()
        n = self.config.n_qubits
        single_q = 0
        two_q = 0

        qubit_layer = [0] * n
        for g in gates:
            name, qubits, _ = g
            if len(qubits) == 1:
                single_q += 1
                q = qubits[0]
                if 0 <= q < n:
                    qubit_layer[q] += 1
            elif len(qubits) >= 2:
                two_q += 1
                valid = [q for q in qubits if 0 <= q < n]
                if valid:
                    layer = max(qubit_layer[q] for q in valid) + 1
                    for q in valid:
                        qubit_layer[q] = layer

        depth = max(qubit_layer) if qubit_layer else 0

        # Entanglement structure: which pairs are entangled
        entangled_pairs = set()
        for g in gates:
            if len(g[1]) >= 2:
                q0, q1 = g[1][0], g[1][1]
                entangled_pairs.add((min(q0, q1), max(q0, q1)))

        dt = time.monotonic() - t0
        return StageResult(
            stage=STAGE_ANALYZE,
            success=True,
            output={
                "depth": depth,
                "single_qubit_gates": single_q,
                "two_qubit_gates": two_q,
                "total_gates": single_q + two_q,
                "entangled_pairs": list(entangled_pairs),
            },
            time_seconds=dt,
            metrics={
                "depth": depth,
                "gate_count": single_q + two_q,
                "two_qubit_fraction": two_q / max(1, single_q + two_q),
            },
        )

    def _stage_transpile(self, gates: list) -> StageResult:
        """Decompose into basis gates {H, CNOT, Rz, Ry}.

        Non-basis single qubit gates are decomposed:
        - X = Ry(pi)
        - Y = Rz(pi) Ry(pi)
        - Z = Rz(pi)
        - S = Rz(pi/2)
        - T = Rz(pi/4)
        - Rx(t) = Rz(-pi/2) Ry(t) Rz(pi/2)
        """
        t0 = time.monotonic()
        basis = {"h", "cx", "rz", "ry", "cnot"}
        transpiled: List[Tuple[str, list, dict]] = []

        for name, qubits, params in gates:
            lower = name.lower()
            if lower in basis:
                transpiled.append((lower, qubits, params))
            elif lower == "x":
                transpiled.append(("ry", qubits, {"angle": math.pi}))
            elif lower == "y":
                transpiled.append(("rz", qubits, {"angle": math.pi}))
                transpiled.append(("ry", qubits, {"angle": math.pi}))
            elif lower == "z":
                transpiled.append(("rz", qubits, {"angle": math.pi}))
            elif lower == "s":
                transpiled.append(("rz", qubits, {"angle": math.pi / 2}))
            elif lower == "t":
                transpiled.append(("rz", qubits, {"angle": math.pi / 4}))
            elif lower == "rx":
                angle = params.get("angle", 0.0)
                transpiled.append(("rz", qubits, {"angle": -math.pi / 2}))
                transpiled.append(("ry", qubits, {"angle": angle}))
                transpiled.append(("rz", qubits, {"angle": math.pi / 2}))
            else:
                # Pass through unknown gates
                transpiled.append((lower, qubits, params))

        dt = time.monotonic() - t0
        return StageResult(
            stage=STAGE_TRANSPILE,
            success=True,
            output={
                "transpiled_gates": transpiled,
                "original_count": len(gates),
                "transpiled_count": len(transpiled),
            },
            time_seconds=dt,
            metrics={
                "expansion_ratio": len(transpiled) / max(1, len(gates)),
            },
        )

    def _stage_optimize(self, gates: list) -> StageResult:
        """Optimise gate sequence.

        Levels:
        - 0: no optimisation
        - 1: cancel adjacent inverse pairs (H-H, X-X)
        - 2: level 1 + merge adjacent Rz rotations
        - 3: level 2 + remove identity rotations
        """
        t0 = time.monotonic()
        level = self.config.optimization_level

        if level == 0:
            optimized = list(gates)
        else:
            optimized = list(gates)

            if level >= 1:
                optimized = self._cancel_inverse_pairs(optimized)

            if level >= 2:
                optimized = self._merge_rotations(optimized)

            if level >= 3:
                optimized = self._remove_identity_rotations(optimized)

        dt = time.monotonic() - t0
        original_count = len(gates)
        optimized_count = len(optimized)
        reduction = 1.0 - optimized_count / max(1, original_count)

        return StageResult(
            stage=STAGE_OPTIMIZE,
            success=True,
            output={
                "optimized_gates": optimized,
                "original_count": original_count,
                "optimized_count": optimized_count,
            },
            time_seconds=dt,
            metrics={"reduction": reduction, "gates_removed": original_count - optimized_count},
        )

    @staticmethod
    def _cancel_inverse_pairs(gates: list) -> list:
        """Cancel adjacent self-inverse gate pairs (H-H, X-X, Y-Y, Z-Z)."""
        self_inverse = {"h", "x", "y", "z"}
        result = []
        skip_next = False
        for i in range(len(gates)):
            if skip_next:
                skip_next = False
                continue
            if i + 1 < len(gates):
                name_i, qubits_i, _ = gates[i]
                name_j, qubits_j, _ = gates[i + 1]
                if (
                    name_i.lower() in self_inverse
                    and name_i.lower() == name_j.lower()
                    and qubits_i == qubits_j
                ):
                    skip_next = True
                    continue
            result.append(gates[i])
        return result

    @staticmethod
    def _merge_rotations(gates: list) -> list:
        """Merge adjacent Rz (or Ry) rotations on the same qubit."""
        result = []
        i = 0
        while i < len(gates):
            if i + 1 < len(gates):
                name_i, qubits_i, params_i = gates[i]
                name_j, qubits_j, params_j = gates[i + 1]
                if (
                    name_i.lower() == name_j.lower()
                    and name_i.lower() in ("rz", "ry")
                    and qubits_i == qubits_j
                ):
                    merged_angle = params_i.get("angle", 0.0) + params_j.get("angle", 0.0)
                    result.append((name_i, qubits_i, {"angle": merged_angle}))
                    i += 2
                    continue
            result.append(gates[i])
            i += 1
        return result

    @staticmethod
    def _remove_identity_rotations(gates: list, tol: float = 1e-10) -> list:
        """Remove rotations with angle effectively zero (mod 2pi)."""
        result = []
        for name, qubits, params in gates:
            if name.lower() in ("rz", "ry", "rx"):
                angle = params.get("angle", 0.0)
                # Normalize to [-pi, pi]
                reduced = angle % (2 * math.pi)
                if reduced > math.pi:
                    reduced -= 2 * math.pi
                if abs(reduced) < tol:
                    continue
            result.append((name, qubits, params))
        return result

    def _stage_simulate(self, gates: list) -> StageResult:
        """Simulate circuit with optional noise."""
        t0 = time.monotonic()
        n = self.config.n_qubits
        dim = 2**n
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0  # |000...0>

        for name, qubits, params in gates:
            lower = name.lower()
            mat = _resolve_single_gate(lower, params)
            if mat is not None and len(qubits) >= 1:
                state = _apply_single_qubit_gate(state, mat, qubits[0], n)
            elif lower in ("cx", "cnot") and len(qubits) >= 2:
                state = _apply_cnot(state, qubits[0], qubits[1], n)
            # else: skip unknown gates silently

        # Apply noise model
        noise = self.config.noise_model
        if noise and noise != "ideal":
            state = self._apply_noise(state, noise)

        # Normalise
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm

        dt = time.monotonic() - t0
        probs = np.abs(state) ** 2
        return StageResult(
            stage=STAGE_SIMULATE,
            success=True,
            output={"state": state, "probabilities": probs},
            time_seconds=dt,
            metrics={"state_norm": float(np.linalg.norm(state))},
        )

    def _apply_noise(self, state: np.ndarray, noise_model: str) -> np.ndarray:
        """Apply a simple noise model to the statevector."""
        n = self.config.n_qubits
        p = self.config.error_rate

        if noise_model == "depolarizing":
            # Simple depolarising: mix with maximally mixed state
            dim = 2**n
            mixed = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
            state = math.sqrt(1 - p) * state + math.sqrt(p) * mixed
        elif noise_model == "amplitude_damping":
            # Simplified amplitude damping: reduce amplitude of |1> states
            dim = 2**n
            for i in range(dim):
                ones = bin(i).count("1")
                damping = (1 - p) ** (ones / 2)
                state[i] *= damping
        return state

    def _stage_measure(self, state: Optional[np.ndarray]) -> StageResult:
        """Sample measurements from final state."""
        t0 = time.monotonic()
        n = self.config.n_qubits
        shots = self.config.shots

        if state is None:
            dt = time.monotonic() - t0
            return StageResult(
                stage=STAGE_MEASURE,
                success=False,
                output={"counts": {}},
                time_seconds=dt,
                metrics={},
            )

        probs = np.abs(state) ** 2
        total = np.sum(probs)
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(len(probs)) / len(probs)

        # Sample
        dim = 2**n
        outcomes = self._rng.choice(dim, size=shots, p=probs)
        counts: Dict[str, int] = {}
        for outcome in outcomes:
            bitstring = format(int(outcome), f"0{n}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        dt = time.monotonic() - t0
        entropy = 0.0
        for c in counts.values():
            p_i = c / shots
            if p_i > 0:
                entropy -= p_i * math.log2(p_i)

        return StageResult(
            stage=STAGE_MEASURE,
            success=True,
            output={"counts": counts},
            time_seconds=dt,
            metrics={"unique_outcomes": len(counts), "entropy": entropy},
        )


# ======================================================================
# Convenience functions
# ======================================================================


def quick_run(
    gates: list,
    n_qubits: int,
    shots: int = 1024,
) -> PipelineResult:
    """Quick pipeline run with default config.

    Parameters
    ----------
    gates : list
        Gate list as ``(name, qubits, params)`` tuples.
    n_qubits : int
        Number of qubits.
    shots : int
        Measurement shots.
    """
    config = PipelineConfig(n_qubits=n_qubits, shots=shots)
    pipeline = QuantumPipeline(config)
    return pipeline.run(gates)


def optimized_run(
    gates: list,
    n_qubits: int,
    backend: str = "superconducting",
) -> PipelineResult:
    """Optimised pipeline with hardware-aware config.

    Parameters
    ----------
    gates : list
        Gate list.
    n_qubits : int
        Number of qubits.
    backend : str
        Target backend for noise model selection.
    """
    noise = "depolarizing"
    error_rate = 0.001
    if backend == "trapped_ion":
        error_rate = 0.0001
    elif backend == "neutral_atom":
        error_rate = 0.005

    config = PipelineConfig(
        n_qubits=n_qubits,
        optimization_level=3,
        target_backend=backend,
        noise_model=noise,
        error_rate=error_rate,
        shots=4096,
    )
    pipeline = QuantumPipeline(config)
    return pipeline.run(gates)
