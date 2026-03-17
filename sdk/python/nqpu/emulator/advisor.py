"""Hardware Advisor: intelligent QPU selection based on circuit analysis.

Analyzes quantum circuit characteristics (gate mix, depth, qubit count,
Toffoli density, connectivity requirements) and recommends the optimal
hardware platform with detailed reasoning.

This is the "quantum hardware decision engine" -- it answers: "Given this
circuit, which QPU should I run it on, and why?"

Example:
    from nqpu.emulator import HardwareAdvisor, HardwareProfile

    advisor = HardwareAdvisor()
    circuit = [("h", 0), ("cx", 0, 1), ("ccx", 0, 1, 2)]

    rec = advisor.recommend(circuit)
    print(rec.best_profile.name)       # e.g. "QUERA_AQUILA"
    print(rec.reasoning)               # Human-readable explanation
    print(rec.scores)                   # Per-profile score breakdown

    # Or get a full comparison report
    report = advisor.full_report(circuit, shots=1000)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hardware import HardwareFamily, HardwareProfile, HardwareSpec

# Gate classification sets (shared with qpu.py)
_1Q_GATES = frozenset(
    ("h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz")
)
_2Q_GATES = frozenset(("cx", "cnot", "cz", "swap"))
_3Q_GATES = frozenset(("ccx", "toffoli", "ccz"))


@dataclass
class CircuitProfile:
    """Characterisation of a quantum circuit's resource requirements.

    Attributes
    ----------
    n_qubits : int
        Number of qubits required.
    n_1q_gates : int
        Count of single-qubit gates.
    n_2q_gates : int
        Count of two-qubit gates.
    n_3q_gates : int
        Count of three-qubit gates (Toffoli / CCZ).
    depth : int
        Circuit depth.
    toffoli_fraction : float
        Fraction of entangling gates that are 3-qubit (0-1).
    unique_gates : set[str]
        Set of unique gate names used.
    connectivity_required : str
        Minimum connectivity needed: "linear", "grid", or "all_to_all".
    """

    n_qubits: int = 0
    n_1q_gates: int = 0
    n_2q_gates: int = 0
    n_3q_gates: int = 0
    depth: int = 0
    toffoli_fraction: float = 0.0
    unique_gates: set = field(default_factory=set)
    connectivity_required: str = "linear"


@dataclass
class HardwareScore:
    """Scored evaluation of a hardware profile for a specific circuit.

    Attributes
    ----------
    profile : HardwareProfile
        The hardware profile being evaluated.
    total_score : float
        Overall score (0-100, higher is better).
    fidelity_score : float
        Estimated circuit fidelity (0-1).
    speed_score : float
        Normalised speed score (0-100).
    capacity_score : float
        Qubit capacity score (0-100).
    toffoli_score : float
        Toffoli efficiency score (0-100).
    connectivity_score : float
        Connectivity match score (0-100).
    estimated_fidelity : float
        Raw estimated circuit fidelity.
    estimated_runtime_us : float
        Estimated wall-clock runtime in microseconds.
    reasons : list[str]
        Human-readable reasons for or against this hardware.
    """

    profile: HardwareProfile
    total_score: float = 0.0
    fidelity_score: float = 0.0
    speed_score: float = 0.0
    capacity_score: float = 0.0
    toffoli_score: float = 0.0
    connectivity_score: float = 0.0
    estimated_fidelity: float = 0.0
    estimated_runtime_us: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    """Hardware recommendation with reasoning.

    Attributes
    ----------
    best_profile : HardwareProfile
        The recommended hardware profile.
    circuit_profile : CircuitProfile
        Analysis of the input circuit.
    scores : list[HardwareScore]
        All profiles scored and ranked (best first).
    reasoning : str
        Human-readable explanation of the recommendation.
    runner_up : HardwareProfile or None
        Second-best option.
    warnings : list[str]
        Any caveats about the recommendation.
    """

    best_profile: HardwareProfile
    circuit_profile: CircuitProfile
    scores: list[HardwareScore]
    reasoning: str
    runner_up: HardwareProfile | None = None
    warnings: list[str] = field(default_factory=list)


class HardwareAdvisor:
    """Quantum hardware decision engine.

    Analyzes circuit characteristics and scores all available hardware
    profiles to recommend the best platform.

    The scoring model weighs five factors:
      - **Fidelity** (40%): estimated circuit fidelity from gate error rates
      - **Speed** (20%): estimated wall-clock execution time
      - **Capacity** (15%): qubit headroom beyond circuit requirements
      - **Toffoli efficiency** (15%): native 3Q gate advantage for Toffoli-heavy circuits
      - **Connectivity** (10%): match between circuit and hardware topology

    Parameters
    ----------
    weights : dict or None
        Custom scoring weights. Keys: fidelity, speed, capacity, toffoli,
        connectivity. Values should sum to 1.0.
    """

    DEFAULT_WEIGHTS = {
        "fidelity": 0.40,
        "speed": 0.20,
        "capacity": 0.15,
        "toffoli": 0.15,
        "connectivity": 0.10,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    # ------------------------------------------------------------------ #
    #  Circuit analysis                                                    #
    # ------------------------------------------------------------------ #

    def analyze_circuit(self, circuit) -> CircuitProfile:
        """Analyze a circuit's resource requirements.

        Parameters
        ----------
        circuit : list[tuple] or QuantumCircuit
            Circuit to analyze.

        Returns
        -------
        CircuitProfile
        """
        gate_list = self._to_gate_list(circuit)
        n_qubits = self._infer_qubits(gate_list)

        n_1q = sum(1 for g in gate_list if str(g[0]).lower() in _1Q_GATES)
        n_2q = sum(1 for g in gate_list if str(g[0]).lower() in _2Q_GATES)
        n_3q = sum(1 for g in gate_list if str(g[0]).lower() in _3Q_GATES)

        unique = {str(g[0]).lower() for g in gate_list}
        entangling = n_2q + n_3q
        toffoli_frac = n_3q / max(1, entangling)

        # Estimate connectivity requirement from gate patterns
        connectivity = self._infer_connectivity(gate_list, n_qubits)

        depth = self._circuit_depth(gate_list, n_qubits)

        return CircuitProfile(
            n_qubits=n_qubits,
            n_1q_gates=n_1q,
            n_2q_gates=n_2q,
            n_3q_gates=n_3q,
            depth=depth,
            toffoli_fraction=toffoli_frac,
            unique_gates=unique,
            connectivity_required=connectivity,
        )

    # ------------------------------------------------------------------ #
    #  Scoring                                                             #
    # ------------------------------------------------------------------ #

    def score_profile(
        self, profile: HardwareProfile, cp: CircuitProfile
    ) -> HardwareScore:
        """Score a hardware profile against circuit requirements.

        Parameters
        ----------
        profile : HardwareProfile
        cp : CircuitProfile

        Returns
        -------
        HardwareScore
        """
        spec = profile.spec
        reasons: list[str] = []

        # --- Capacity check ---
        if cp.n_qubits > spec.num_qubits:
            return HardwareScore(
                profile=profile,
                total_score=0.0,
                reasons=[f"Insufficient qubits: needs {cp.n_qubits}, has {spec.num_qubits}"],
            )

        # --- Fidelity score ---
        n_2q_effective = cp.n_2q_gates
        if spec.family != HardwareFamily.NEUTRAL_ATOM:
            n_2q_effective += cp.n_3q_gates * 6  # Toffoli decomposition
        n_3q_effective = cp.n_3q_gates if spec.family == HardwareFamily.NEUTRAL_ATOM else 0

        fidelity = (
            spec.single_qubit_fidelity ** cp.n_1q_gates
            * spec.two_qubit_fidelity ** n_2q_effective
            * spec.readout_fidelity ** cp.n_qubits
        )
        fidelity_score = fidelity * 100

        # --- Speed score (inverse of runtime, normalised) ---
        runtime = (
            cp.n_1q_gates * spec.single_qubit_gate_us
            + n_2q_effective * spec.two_qubit_gate_us
            + spec.readout_us
        )
        # Superconducting is fastest (~ns gates), ion-trap slowest (~ms)
        # Normalise: 100 at 1us, 0 at 10ms
        speed_score = max(0.0, min(100.0, 100 * (1.0 - np.log10(max(runtime, 0.1)) / 4.0)))

        # --- Capacity score ---
        headroom = spec.num_qubits / max(1, cp.n_qubits)
        capacity_score = min(100.0, 50 + 10 * np.log2(max(headroom, 1.0)))

        # --- Toffoli efficiency score ---
        if cp.n_3q_gates > 0:
            if spec.native_3q_gate is not None:
                toffoli_score = 100.0
                reasons.append(f"Native {spec.native_3q_gate} gate ({cp.n_3q_gates} Toffoli gates)")
            else:
                # Each Toffoli decomposes to ~6 CNOTs + 9 single-qubit
                overhead = cp.n_3q_gates * 6
                penalty = min(1.0, overhead / max(1, cp.n_2q_gates + overhead))
                toffoli_score = (1.0 - penalty) * 100
                reasons.append(f"Toffoli decomposition adds {overhead} CNOTs")
        else:
            toffoli_score = 50.0  # Neutral -- no Toffoli gates

        # --- Connectivity score ---
        connectivity_score = self._score_connectivity(
            spec.connectivity, cp.connectivity_required
        )
        if connectivity_score < 50:
            reasons.append(f"Connectivity mismatch: needs {cp.connectivity_required}, has {spec.connectivity}")

        # --- Aggregate ---
        w = self.weights
        total = (
            w["fidelity"] * fidelity_score
            + w["speed"] * speed_score
            + w["capacity"] * capacity_score
            + w["toffoli"] * toffoli_score
            + w["connectivity"] * connectivity_score
        )

        # Quality reasons
        if fidelity > 0.95:
            reasons.append(f"High fidelity: {fidelity:.4f}")
        elif fidelity < 0.5:
            reasons.append(f"Low fidelity: {fidelity:.4f}")

        if spec.family == HardwareFamily.TRAPPED_ION:
            reasons.append("All-to-all connectivity (no routing overhead)")
        if spec.family == HardwareFamily.SUPERCONDUCTING:
            reasons.append(f"Fast gates: {spec.two_qubit_gate_us:.3f}us 2Q gate")

        return HardwareScore(
            profile=profile,
            total_score=total,
            fidelity_score=fidelity_score,
            speed_score=speed_score,
            capacity_score=capacity_score,
            toffoli_score=toffoli_score,
            connectivity_score=connectivity_score,
            estimated_fidelity=fidelity,
            estimated_runtime_us=runtime,
            reasons=reasons,
        )

    # ------------------------------------------------------------------ #
    #  Recommendation                                                      #
    # ------------------------------------------------------------------ #

    def recommend(
        self,
        circuit,
        profiles: list[HardwareProfile] | None = None,
    ) -> Recommendation:
        """Recommend the best hardware for a circuit.

        Parameters
        ----------
        circuit : list[tuple] or QuantumCircuit
            Circuit to analyze.
        profiles : list[HardwareProfile] or None
            Profiles to consider. Defaults to all.

        Returns
        -------
        Recommendation
        """
        if profiles is None:
            profiles = list(HardwareProfile)

        cp = self.analyze_circuit(circuit)
        scores = [self.score_profile(p, cp) for p in profiles]
        scores.sort(key=lambda s: s.total_score, reverse=True)

        best = scores[0]
        runner_up = scores[1] if len(scores) > 1 and scores[1].total_score > 0 else None

        warnings: list[str] = []
        if best.estimated_fidelity < 0.5:
            warnings.append("Circuit fidelity is below 50% on all platforms -- consider error mitigation")
        if cp.depth > best.profile.spec.max_circuit_depth:
            warnings.append(
                f"Circuit depth ({cp.depth}) exceeds {best.profile.spec.name} "
                f"max ({best.profile.spec.max_circuit_depth}) -- coherence-limited"
            )
        if best.total_score < 30:
            warnings.append("No platform scored above 30 -- circuit may be too demanding for current hardware")

        reasoning = self._build_reasoning(best, runner_up, cp, warnings)

        return Recommendation(
            best_profile=best.profile,
            circuit_profile=cp,
            scores=scores,
            reasoning=reasoning,
            runner_up=runner_up.profile if runner_up else None,
            warnings=warnings,
        )

    def full_report(
        self,
        circuit,
        shots: int = 1024,
        seed: int = 42,
    ) -> dict:
        """Generate a comprehensive comparison report.

        Runs the recommendation engine and also executes the circuit on
        the top 3 hardware profiles for empirical validation.

        Parameters
        ----------
        circuit
            Circuit to analyze and run.
        shots : int
            Shots per execution.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Keys: recommendation, execution_results, comparison_table.
        """
        from .qpu import QPU

        rec = self.recommend(circuit)
        top_profiles = [s.profile for s in rec.scores[:3] if s.total_score > 0]

        execution_results: dict = {}
        for profile in top_profiles:
            qpu = QPU(profile, noise=True, seed=seed, max_qubits=max(rec.circuit_profile.n_qubits + 2, 8))
            job = qpu.run(circuit, shots=shots)
            if job.successful():
                execution_results[profile.spec.name] = {
                    "counts": dict(job.result.counts),
                    "fidelity_estimate": job.result.fidelity_estimate,
                    "circuit_depth": job.result.circuit_depth,
                    "native_gate_count": job.result.native_gate_count,
                    "runtime_us": job.result.estimated_runtime_us,
                }

        comparison_table = []
        for s in rec.scores:
            if s.total_score > 0:
                comparison_table.append({
                    "profile": s.profile.name,
                    "total_score": round(s.total_score, 1),
                    "fidelity": round(s.estimated_fidelity, 4),
                    "runtime_us": round(s.estimated_runtime_us, 1),
                    "reasons": s.reasons,
                })

        return {
            "recommendation": rec,
            "execution_results": execution_results,
            "comparison_table": comparison_table,
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_reasoning(
        best: HardwareScore,
        runner_up: HardwareScore | None,
        cp: CircuitProfile,
        warnings: list[str],
    ) -> str:
        """Build human-readable recommendation text."""
        lines = []
        spec = best.profile.spec

        lines.append(f"Recommended: {spec.name} (score: {best.total_score:.1f}/100)")
        lines.append("")

        lines.append(f"Circuit: {cp.n_qubits} qubits, depth {cp.depth}, "
                     f"{cp.n_1q_gates} 1Q + {cp.n_2q_gates} 2Q + {cp.n_3q_gates} 3Q gates")
        lines.append(f"Estimated fidelity: {best.estimated_fidelity:.4f}")
        lines.append(f"Estimated runtime: {best.estimated_runtime_us:.1f} us")
        lines.append("")

        lines.append("Reasons:")
        for r in best.reasons:
            lines.append(f"  + {r}")
        lines.append("")

        if runner_up and runner_up.total_score > 0:
            diff = best.total_score - runner_up.total_score
            lines.append(
                f"Runner-up: {runner_up.profile.spec.name} "
                f"(score: {runner_up.total_score:.1f}, -{diff:.1f})"
            )
            for r in runner_up.reasons:
                lines.append(f"  + {r}")
            lines.append("")

        if warnings:
            lines.append("Warnings:")
            for w in warnings:
                lines.append(f"  ! {w}")

        return "\n".join(lines)

    @staticmethod
    def _infer_connectivity(gate_list: list, n_qubits: int) -> str:
        """Infer minimum connectivity from qubit interaction patterns."""
        if n_qubits <= 2:
            return "linear"

        pairs: set[tuple[int, int]] = set()
        for gate in gate_list:
            name = str(gate[0]).lower()
            if name in _2Q_GATES and len(gate) >= 3:
                q0, q1 = int(gate[1]), int(gate[2])
                pairs.add((min(q0, q1), max(q0, q1)))
            elif name in _3Q_GATES and len(gate) >= 4:
                q0, q1, q2 = int(gate[1]), int(gate[2]), int(gate[3])
                for a, b in [(q0, q1), (q0, q2), (q1, q2)]:
                    pairs.add((min(a, b), max(a, b)))

        if not pairs:
            return "linear"

        # Check if all interactions are nearest-neighbour
        nn = all(abs(a - b) <= 1 for a, b in pairs)
        if nn:
            return "linear"

        # Check if interactions fit a 2D grid
        max_gap = max(abs(a - b) for a, b in pairs)
        sqrt_n = int(np.ceil(np.sqrt(n_qubits)))
        if max_gap <= sqrt_n + 1:
            return "grid"

        return "all_to_all"

    @staticmethod
    def _score_connectivity(hardware: str, required: str) -> float:
        """Score hardware connectivity against circuit requirements."""
        hierarchy = {"linear": 0, "grid": 1, "heavy_hex": 1, "reconfigurable": 2, "all_to_all": 3}
        hw_level = hierarchy.get(hardware, 1)
        req_level = hierarchy.get(required, 1)

        if hw_level >= req_level:
            return 100.0
        gap = req_level - hw_level
        return max(0.0, 100 - gap * 40)

    @staticmethod
    def _to_gate_list(circuit) -> list:
        """Convert circuit to gate tuples."""
        if isinstance(circuit, list):
            return circuit
        if hasattr(circuit, "gates"):
            return [(g.name.lower(), *g.qubits, *g.params) for g in circuit.gates]
        raise TypeError(f"Unsupported circuit type: {type(circuit).__name__}")

    @staticmethod
    def _infer_qubits(gate_list: list) -> int:
        """Infer qubit count from gate indices."""
        max_q = -1
        for gate in gate_list:
            for i in range(1, len(gate)):
                try:
                    q = int(gate[i])
                    max_q = max(max_q, q)
                except (ValueError, TypeError):
                    break
        return max_q + 1 if max_q >= 0 else 0

    @staticmethod
    def _circuit_depth(gate_list: list, n_qubits: int) -> int:
        """Compute circuit depth."""
        if n_qubits == 0:
            return 0
        qubit_layer = [0] * n_qubits
        for gate in gate_list:
            name = str(gate[0]).lower()
            if name in _1Q_GATES:
                q = int(gate[1])
                if q < n_qubits:
                    qubit_layer[q] += 1
            elif name in _2Q_GATES and len(gate) >= 3:
                q0, q1 = int(gate[1]), int(gate[2])
                if q0 < n_qubits and q1 < n_qubits:
                    layer = max(qubit_layer[q0], qubit_layer[q1]) + 1
                    qubit_layer[q0] = layer
                    qubit_layer[q1] = layer
            elif name in _3Q_GATES and len(gate) >= 4:
                q0, q1, q2 = int(gate[1]), int(gate[2]), int(gate[3])
                if q0 < n_qubits and q1 < n_qubits and q2 < n_qubits:
                    layer = max(qubit_layer[q0], qubit_layer[q1], qubit_layer[q2]) + 1
                    qubit_layer[q0] = layer
                    qubit_layer[q1] = layer
                    qubit_layer[q2] = layer
        return max(qubit_layer) if qubit_layer else 0
