"""Hardware recommendation engine.

Analyses quantum circuit characteristics and recommends the best
hardware platform from a built-in database of ~12 real quantum devices.
Recommendations account for qubit count, connectivity requirements,
gate fidelities, coherence times, and cost constraints.

Example::

    from nqpu.dashboard import recommend_hardware
    rec = recommend_hardware(
        gates=[("h", [0], {}), ("cx", [0, 1], {}), ("cx", [1, 2], {})],
        n_qubits=3,
    )
    print(rec.report())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class CircuitProfile:
    """Analysis of a quantum circuit's characteristics."""

    n_qubits: int
    depth: int
    single_qubit_gates: int
    two_qubit_gates: int
    three_qubit_gates: int
    t_gates: int
    measurement_count: int
    connectivity_required: str  # "linear", "heavy_hex", "all_to_all", "2d_grid"
    estimated_runtime_us: float

    @staticmethod
    def from_gates(gates: list, n_qubits: int) -> CircuitProfile:
        """Analyse a gate list.

        Parameters
        ----------
        gates : list
            List of ``(gate_name, qubits, params)`` tuples.
        n_qubits : int
            Number of qubits.

        Returns
        -------
        CircuitProfile
        """
        single_q = 0
        two_q = 0
        three_q = 0
        t_count = 0
        measure_count = 0

        qubit_layer = [0] * max(n_qubits, 1)
        qubit_pairs: set = set()

        for g in gates:
            name = str(g[0]).lower()
            qubits = list(g[1]) if len(g) > 1 else []

            if name == "measure":
                measure_count += len(qubits)
                continue

            if len(qubits) == 1:
                single_q += 1
                q = qubits[0]
                if 0 <= q < n_qubits:
                    qubit_layer[q] += 1
                if name == "t":
                    t_count += 1
            elif len(qubits) == 2:
                two_q += 1
                q0, q1 = qubits[0], qubits[1]
                qubit_pairs.add((min(q0, q1), max(q0, q1)))
                valid = [q for q in qubits if 0 <= q < n_qubits]
                if valid:
                    layer = max(qubit_layer[q] for q in valid) + 1
                    for q in valid:
                        qubit_layer[q] = layer
            elif len(qubits) >= 3:
                three_q += 1
                valid = [q for q in qubits if 0 <= q < n_qubits]
                if valid:
                    layer = max(qubit_layer[q] for q in valid) + 1
                    for q in valid:
                        qubit_layer[q] = layer

        depth = max(qubit_layer) if qubit_layer else 0

        # Determine connectivity requirement
        connectivity = _infer_connectivity(qubit_pairs, n_qubits)

        # Rough runtime estimate (assume 100ns per gate)
        estimated_runtime_us = (single_q * 0.025 + two_q * 0.2 + three_q * 1.0)

        return CircuitProfile(
            n_qubits=n_qubits,
            depth=depth,
            single_qubit_gates=single_q,
            two_qubit_gates=two_q,
            three_qubit_gates=three_q,
            t_gates=t_count,
            measurement_count=measure_count,
            connectivity_required=connectivity,
            estimated_runtime_us=estimated_runtime_us,
        )


def _infer_connectivity(pairs: set, n_qubits: int) -> str:
    """Infer the connectivity required by a set of qubit pairs."""
    if not pairs:
        return "linear"

    # Check if all pairs are nearest-neighbor
    all_nn = all(abs(q0 - q1) == 1 for q0, q1 in pairs)
    if all_nn:
        return "linear"

    # Check if there are long-range connections
    max_dist = max(abs(q0 - q1) for q0, q1 in pairs)
    if max_dist >= n_qubits - 1 and len(pairs) > n_qubits:
        return "all_to_all"

    # 2D grid if moderate connectivity
    if max_dist > 1:
        return "2d_grid"

    return "linear"


@dataclass
class HardwareProfile:
    """Specifications of a quantum hardware platform."""

    name: str
    technology: str  # "superconducting", "trapped_ion", "neutral_atom", "photonic"
    max_qubits: int
    connectivity: str
    single_qubit_fidelity: float
    two_qubit_fidelity: float
    readout_fidelity: float
    t1_us: float
    t2_us: float
    gate_time_ns: float
    two_qubit_gate_time_ns: float
    native_gates: list
    availability: str  # "cloud", "on_premise", "research"
    cost_per_shot: float

    def estimated_circuit_fidelity(self, profile: CircuitProfile) -> float:
        """Estimate circuit fidelity on this hardware.

        Uses a simple multiplicative noise model:
            F = F_1q^n_1q * F_2q^n_2q * F_ro^n_qubits * exp(-t/T2)
        """
        f_1q = self.single_qubit_fidelity ** profile.single_qubit_gates
        f_2q = self.two_qubit_fidelity ** profile.two_qubit_gates
        f_ro = self.readout_fidelity ** max(profile.measurement_count, profile.n_qubits)

        # Three-qubit gates: model as 2 two-qubit gates each
        f_3q = self.two_qubit_fidelity ** (2 * profile.three_qubit_gates)

        # Decoherence
        total_time_us = (
            profile.single_qubit_gates * self.gate_time_ns
            + profile.two_qubit_gates * self.two_qubit_gate_time_ns
            + profile.three_qubit_gates * self.two_qubit_gate_time_ns * 2
        ) / 1000.0

        if self.t2_us > 0 and not math.isinf(self.t2_us):
            decoherence = math.exp(-total_time_us / self.t2_us)
        else:
            decoherence = 1.0

        return max(0.0, min(1.0, f_1q * f_2q * f_3q * f_ro * decoherence))


# ======================================================================
# Device database
# ======================================================================


class DeviceDatabase:
    """Database of known quantum devices with specs."""

    def __init__(self) -> None:
        self.devices = self._build_database()

    def _build_database(self) -> Dict[str, HardwareProfile]:
        """Build database of ~12 quantum devices."""
        devices = {}

        devices["ibm_eagle"] = HardwareProfile(
            name="IBM Eagle (127Q)",
            technology="superconducting",
            max_qubits=127,
            connectivity="heavy_hex",
            single_qubit_fidelity=0.9996,
            two_qubit_fidelity=0.99,
            readout_fidelity=0.98,
            t1_us=100.0,
            t2_us=80.0,
            gate_time_ns=35.0,
            two_qubit_gate_time_ns=300.0,
            native_gates=["sx", "rz", "cx"],
            availability="cloud",
            cost_per_shot=0.00015,
        )
        devices["ibm_heron"] = HardwareProfile(
            name="IBM Heron (133Q)",
            technology="superconducting",
            max_qubits=133,
            connectivity="heavy_hex",
            single_qubit_fidelity=0.9998,
            two_qubit_fidelity=0.995,
            readout_fidelity=0.99,
            t1_us=300.0,
            t2_us=200.0,
            gate_time_ns=25.0,
            two_qubit_gate_time_ns=68.0,
            native_gates=["sx", "rz", "ecr"],
            availability="cloud",
            cost_per_shot=0.0002,
        )
        devices["google_sycamore"] = HardwareProfile(
            name="Google Sycamore (53Q)",
            technology="superconducting",
            max_qubits=53,
            connectivity="2d_grid",
            single_qubit_fidelity=0.9985,
            two_qubit_fidelity=0.9935,
            readout_fidelity=0.965,
            t1_us=15.0,
            t2_us=10.0,
            gate_time_ns=25.0,
            two_qubit_gate_time_ns=32.0,
            native_gates=["phased_xz", "sycamore"],
            availability="research",
            cost_per_shot=0.0003,
        )
        devices["ionq_aria"] = HardwareProfile(
            name="IonQ Aria (25Q)",
            technology="trapped_ion",
            max_qubits=25,
            connectivity="all_to_all",
            single_qubit_fidelity=0.9998,
            two_qubit_fidelity=0.995,
            readout_fidelity=0.998,
            t1_us=1_000_000.0,
            t2_us=500_000.0,
            gate_time_ns=10000.0,
            two_qubit_gate_time_ns=200000.0,
            native_gates=["gpi", "gpi2", "ms"],
            availability="cloud",
            cost_per_shot=0.003,
        )
        devices["ionq_forte"] = HardwareProfile(
            name="IonQ Forte (36Q)",
            technology="trapped_ion",
            max_qubits=36,
            connectivity="all_to_all",
            single_qubit_fidelity=0.9999,
            two_qubit_fidelity=0.997,
            readout_fidelity=0.999,
            t1_us=2_000_000.0,
            t2_us=1_000_000.0,
            gate_time_ns=8000.0,
            two_qubit_gate_time_ns=150000.0,
            native_gates=["gpi", "gpi2", "ms"],
            availability="cloud",
            cost_per_shot=0.005,
        )
        devices["quantinuum_h2"] = HardwareProfile(
            name="Quantinuum H2 (56Q)",
            technology="trapped_ion",
            max_qubits=56,
            connectivity="all_to_all",
            single_qubit_fidelity=0.99998,
            two_qubit_fidelity=0.998,
            readout_fidelity=0.9993,
            t1_us=3_000_000.0,
            t2_us=1_500_000.0,
            gate_time_ns=5000.0,
            two_qubit_gate_time_ns=250000.0,
            native_gates=["rz", "u1q", "zz"],
            availability="cloud",
            cost_per_shot=0.01,
        )
        devices["rigetti_aspen_m3"] = HardwareProfile(
            name="Rigetti Aspen-M-3 (80Q)",
            technology="superconducting",
            max_qubits=80,
            connectivity="2d_grid",
            single_qubit_fidelity=0.999,
            two_qubit_fidelity=0.97,
            readout_fidelity=0.95,
            t1_us=20.0,
            t2_us=15.0,
            gate_time_ns=40.0,
            two_qubit_gate_time_ns=200.0,
            native_gates=["rx", "rz", "cz"],
            availability="cloud",
            cost_per_shot=0.00035,
        )
        devices["quera_aquila"] = HardwareProfile(
            name="QuEra Aquila (256Q)",
            technology="neutral_atom",
            max_qubits=256,
            connectivity="2d_grid",
            single_qubit_fidelity=0.999,
            two_qubit_fidelity=0.995,
            readout_fidelity=0.97,
            t1_us=5000.0,
            t2_us=2000.0,
            gate_time_ns=500.0,
            two_qubit_gate_time_ns=1000.0,
            native_gates=["rx", "ry", "cz", "ccz"],
            availability="cloud",
            cost_per_shot=0.0005,
        )
        devices["xanadu_borealis"] = HardwareProfile(
            name="Xanadu Borealis (216 modes)",
            technology="photonic",
            max_qubits=216,
            connectivity="linear",
            single_qubit_fidelity=0.999,
            two_qubit_fidelity=0.98,
            readout_fidelity=0.95,
            t1_us=float("inf"),
            t2_us=float("inf"),
            gate_time_ns=1.0,
            two_qubit_gate_time_ns=10.0,
            native_gates=["squeezing", "displacement", "beamsplitter"],
            availability="cloud",
            cost_per_shot=0.0004,
        )
        devices["atom_computing"] = HardwareProfile(
            name="Atom Computing (1180Q)",
            technology="neutral_atom",
            max_qubits=1180,
            connectivity="2d_grid",
            single_qubit_fidelity=0.998,
            two_qubit_fidelity=0.993,
            readout_fidelity=0.96,
            t1_us=10000.0,
            t2_us=5000.0,
            gate_time_ns=600.0,
            two_qubit_gate_time_ns=1200.0,
            native_gates=["rx", "ry", "cz"],
            availability="research",
            cost_per_shot=0.0006,
        )
        devices["aqt_pine"] = HardwareProfile(
            name="AQT Pine (24Q)",
            technology="trapped_ion",
            max_qubits=24,
            connectivity="all_to_all",
            single_qubit_fidelity=0.9997,
            two_qubit_fidelity=0.993,
            readout_fidelity=0.997,
            t1_us=800_000.0,
            t2_us=400_000.0,
            gate_time_ns=12000.0,
            two_qubit_gate_time_ns=250000.0,
            native_gates=["r", "rz", "ms"],
            availability="cloud",
            cost_per_shot=0.002,
        )
        devices["psiquantum"] = HardwareProfile(
            name="PsiQuantum (Fusion-based)",
            technology="photonic",
            max_qubits=1_000_000,
            connectivity="all_to_all",
            single_qubit_fidelity=0.999,
            two_qubit_fidelity=0.99,
            readout_fidelity=0.98,
            t1_us=float("inf"),
            t2_us=float("inf"),
            gate_time_ns=1.0,
            two_qubit_gate_time_ns=10.0,
            native_gates=["fusion", "type_ii_fusion"],
            availability="research",
            cost_per_shot=0.001,
        )

        return devices

    def list_devices(self) -> str:
        """ASCII table of all devices."""
        col_name = 30
        col_tech = 18
        col_q = 10
        col_conn = 14
        col_f2q = 10
        col_avail = 10

        header = (
            f"{'Device':<{col_name}} "
            f"{'Technology':<{col_tech}} "
            f"{'Qubits':>{col_q}} "
            f"{'Connectivity':<{col_conn}} "
            f"{'2Q Fidelity':>{col_f2q}} "
            f"{'Avail.':<{col_avail}}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for dev in self.devices.values():
            lines.append(
                f"{dev.name:<{col_name}} "
                f"{dev.technology:<{col_tech}} "
                f"{dev.max_qubits:>{col_q}} "
                f"{dev.connectivity:<{col_conn}} "
                f"{dev.two_qubit_fidelity:>{col_f2q}.4f} "
                f"{dev.availability:<{col_avail}}"
            )

        lines.append(sep)
        return "\n".join(lines)

    def get_device(self, name: str) -> HardwareProfile:
        """Get a device by key name.

        Raises ``KeyError`` if not found.
        """
        return self.devices[name]

    def filter_by_qubits(self, min_qubits: int) -> List[HardwareProfile]:
        """Return devices with at least *min_qubits* qubits."""
        return [d for d in self.devices.values() if d.max_qubits >= min_qubits]

    def filter_by_technology(self, tech: str) -> List[HardwareProfile]:
        """Return devices matching the given technology string."""
        lower = tech.lower()
        return [d for d in self.devices.values() if d.technology.lower() == lower]


# ======================================================================
# Recommendation engine
# ======================================================================


@dataclass
class HardwareRecommendation:
    """Hardware recommendation with reasoning."""

    recommended: str
    score: float
    reasoning: List[str]
    alternatives: List[Tuple[str, float, str]]  # (name, score, reason)
    warnings: List[str]
    estimated_fidelity: float
    estimated_cost: float

    def report(self) -> str:
        """ASCII recommendation report."""
        lines = [
            "=" * 60,
            "HARDWARE RECOMMENDATION",
            "=" * 60,
            f"  Recommended: {self.recommended}",
            f"  Score:       {self.score:.4f}",
            f"  Est. fidelity: {self.estimated_fidelity:.6f}",
            f"  Est. cost:     ${self.estimated_cost:.4f}",
            "",
            "  Reasoning:",
        ]
        for r in self.reasoning:
            lines.append(f"    - {r}")

        if self.alternatives:
            lines.append("")
            lines.append("  Alternatives:")
            for name, score, reason in self.alternatives:
                lines.append(f"    {name:<30} score={score:.4f}  ({reason})")

        if self.warnings:
            lines.append("")
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ! {w}")

        lines.append("=" * 60)
        return "\n".join(lines)


class HardwareAdvisor:
    """Recommends hardware based on circuit analysis.

    Parameters
    ----------
    database : DeviceDatabase, optional
        Device database to use.  Creates default if not provided.
    """

    def __init__(self, database: Optional[DeviceDatabase] = None) -> None:
        self.db = database or DeviceDatabase()

    def analyze_circuit(self, gates: list, n_qubits: int) -> CircuitProfile:
        """Analyse circuit characteristics.

        Parameters
        ----------
        gates : list
            Gate list.
        n_qubits : int
            Number of qubits.
        """
        return CircuitProfile.from_gates(gates, n_qubits)

    def recommend(
        self,
        profile: CircuitProfile,
        budget: float = float("inf"),
        min_fidelity: float = 0.0,
    ) -> HardwareRecommendation:
        """Recommend best hardware for this circuit.

        Scoring algorithm:
        - 40% fidelity
        - 20% qubit count headroom
        - 20% connectivity match
        - 10% cost efficiency
        - 10% availability
        """
        candidates: List[Tuple[str, HardwareProfile, float, List[str]]] = []

        for key, dev in self.db.devices.items():
            reasons: List[str] = []
            score = 0.0

            # Qubit count check
            if dev.max_qubits < profile.n_qubits:
                continue

            # Fidelity (40%)
            est_fid = dev.estimated_circuit_fidelity(profile)
            if est_fid < min_fidelity:
                continue
            score += 0.4 * est_fid
            reasons.append(f"Estimated fidelity: {est_fid:.4f}")

            # Qubit headroom (20%)
            headroom = min(1.0, dev.max_qubits / max(1, profile.n_qubits * 2))
            score += 0.2 * headroom
            reasons.append(f"Qubit headroom: {dev.max_qubits}/{profile.n_qubits}")

            # Connectivity match (20%)
            conn_score = _connectivity_score(profile.connectivity_required, dev.connectivity)
            score += 0.2 * conn_score
            if conn_score >= 0.8:
                reasons.append(f"Good connectivity match ({dev.connectivity})")

            # Cost (10%)
            est_cost = dev.cost_per_shot * 1024  # assume 1024 shots
            if est_cost <= budget:
                cost_score = 1.0 - min(1.0, est_cost / max(budget, 0.001))
                score += 0.1 * cost_score
            else:
                continue  # over budget

            # Availability (10%)
            avail_score = {"cloud": 1.0, "on_premise": 0.5, "research": 0.3}.get(
                dev.availability, 0.0
            )
            score += 0.1 * avail_score

            candidates.append((key, dev, score, reasons))

        if not candidates:
            return HardwareRecommendation(
                recommended="none",
                score=0.0,
                reasoning=["No suitable devices found"],
                alternatives=[],
                warnings=["No devices meet the requirements"],
                estimated_fidelity=0.0,
                estimated_cost=0.0,
            )

        # Sort by score descending
        candidates.sort(key=lambda c: c[2], reverse=True)

        best_key, best_dev, best_score, best_reasons = candidates[0]
        best_fid = best_dev.estimated_circuit_fidelity(profile)
        best_cost = best_dev.cost_per_shot * 1024

        alternatives = [
            (dev.name, sc, f"{dev.technology}, {dev.max_qubits}Q")
            for key, dev, sc, _ in candidates[1:4]
        ]

        warnings: List[str] = []
        if best_fid < 0.5:
            warnings.append("Estimated fidelity is below 50% -- results may be unreliable")
        if profile.three_qubit_gates > 0 and best_dev.technology != "neutral_atom":
            warnings.append(
                "Circuit has 3-qubit gates; consider neutral atom for native support"
            )
        if profile.connectivity_required == "all_to_all" and best_dev.connectivity != "all_to_all":
            warnings.append("Circuit requires all-to-all connectivity; SWAP overhead expected")

        return HardwareRecommendation(
            recommended=best_dev.name,
            score=best_score,
            reasoning=best_reasons,
            alternatives=alternatives,
            warnings=warnings,
            estimated_fidelity=best_fid,
            estimated_cost=best_cost,
        )

    def compare_options(self, profile: CircuitProfile) -> str:
        """ASCII comparison of all options for a circuit."""
        lines = [
            "HARDWARE COMPARISON",
            "=" * 80,
            f"  Circuit: {profile.n_qubits}Q, depth={profile.depth}, "
            f"1Q={profile.single_qubit_gates}, 2Q={profile.two_qubit_gates}, "
            f"3Q={profile.three_qubit_gates}",
            "",
        ]

        col_name = 30
        col_fid = 12
        col_cost = 10
        col_tech = 16

        header = (
            f"{'Device':<{col_name}} "
            f"{'Est. Fidelity':>{col_fid}} "
            f"{'Cost ($)':>{col_cost}} "
            f"{'Technology':<{col_tech}}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        scored: List[Tuple[str, HardwareProfile, float]] = []
        for key, dev in self.db.devices.items():
            if dev.max_qubits < profile.n_qubits:
                continue
            fid = dev.estimated_circuit_fidelity(profile)
            scored.append((key, dev, fid))

        scored.sort(key=lambda x: x[2], reverse=True)

        for key, dev, fid in scored:
            cost = dev.cost_per_shot * 1024
            lines.append(
                f"{dev.name:<{col_name}} "
                f"{fid:>{col_fid}.6f} "
                f"{cost:>{col_cost}.4f} "
                f"{dev.technology:<{col_tech}}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)


def _connectivity_score(required: str, available: str) -> float:
    """Score connectivity match (0.0 to 1.0)."""
    if required == available:
        return 1.0
    if available == "all_to_all":
        return 1.0  # all-to-all supports everything
    ranking = {"linear": 0, "heavy_hex": 1, "2d_grid": 2, "all_to_all": 3}
    req_rank = ranking.get(required, 1)
    avail_rank = ranking.get(available, 1)
    if avail_rank >= req_rank:
        return 0.8
    return max(0.0, 0.5 - 0.1 * (req_rank - avail_rank))


# ======================================================================
# Convenience function
# ======================================================================


def recommend_hardware(
    gates: list,
    n_qubits: int,
) -> HardwareRecommendation:
    """Quick hardware recommendation.

    Parameters
    ----------
    gates : list
        Gate list.
    n_qubits : int
        Number of qubits.
    """
    advisor = HardwareAdvisor()
    profile = advisor.analyze_circuit(gates, n_qubits)
    return advisor.recommend(profile)
