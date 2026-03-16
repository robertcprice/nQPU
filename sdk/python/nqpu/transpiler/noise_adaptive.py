"""Noise-adaptive transpilation using hardware calibration data.

Provides calibration-aware routing, decomposition, and fidelity
estimation for quantum circuits.  By incorporating real hardware
error rates, T1/T2 coherence times, and readout errors, the
transpiler can make better decisions about:

- **Initial qubit layout**: Place logical qubits on the best
  physical qubits.
- **SWAP routing**: Choose SWAP paths that minimize accumulated
  error rather than just physical distance.
- **Gate decomposition**: Select the optimal two-qubit basis
  (ZX, ZZ, or iSWAP) based on qubit-pair error rates.
- **Fidelity estimation**: Predict the expected circuit fidelity
  before execution.

All computations are pure numpy with no external dependencies.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# Calibration data
# ------------------------------------------------------------------

@dataclass
class CalibrationData:
    """Hardware calibration snapshot.

    Stores error rates, coherence times, and gate durations for a
    quantum processor.  Can be constructed from ideal values (for
    testing) or from realistic device profiles.

    Attributes
    ----------
    single_qubit_errors : dict[int, float]
        Qubit index -> single-qubit gate error rate.
    two_qubit_errors : dict[tuple[int, int], float]
        (q1, q2) -> two-qubit gate error rate.
    t1_times : dict[int, float]
        Qubit index -> T1 relaxation time in microseconds.
    t2_times : dict[int, float]
        Qubit index -> T2 dephasing time in microseconds.
    readout_errors : dict[int, float]
        Qubit index -> readout error probability.
    gate_times : dict[str, float]
        Gate name -> execution time in nanoseconds.
    """

    single_qubit_errors: Dict[int, float]
    two_qubit_errors: Dict[Tuple[int, int], float]
    t1_times: Dict[int, float]
    t2_times: Dict[int, float]
    readout_errors: Dict[int, float]
    gate_times: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def ideal(n_qubits: int) -> "CalibrationData":
        """Create perfect (zero-error) calibration data for testing.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.

        Returns
        -------
        CalibrationData
            Ideal calibration with all errors set to zero.
        """
        return CalibrationData(
            single_qubit_errors={i: 0.0 for i in range(n_qubits)},
            two_qubit_errors={
                (i, j): 0.0
                for i in range(n_qubits)
                for j in range(i + 1, n_qubits)
            },
            t1_times={i: float("inf") for i in range(n_qubits)},
            t2_times={i: float("inf") for i in range(n_qubits)},
            readout_errors={i: 0.0 for i in range(n_qubits)},
            gate_times={"single": 0.0, "two": 0.0, "readout": 0.0},
        )

    @staticmethod
    def noisy_superconducting(
        n_qubits: int, rng: Optional[np.random.RandomState] = None
    ) -> "CalibrationData":
        """Generate realistic superconducting device calibration.

        Models a transmon-based processor with typical IBM-class error
        rates:
        - Single-qubit errors: ~1e-4 to 5e-4.
        - Two-qubit errors: ~5e-3 to 2e-2.
        - T1: 50-200 microseconds.
        - T2: 30-150 microseconds.
        - Readout errors: ~1% to 5%.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        rng : np.random.RandomState, optional
            Random state for reproducibility.
        """
        if rng is None:
            rng = np.random.RandomState(42)

        sq_errors = {
            i: float(rng.uniform(1e-4, 5e-4)) for i in range(n_qubits)
        }
        tq_errors = {
            (i, j): float(rng.uniform(5e-3, 2e-2))
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        }
        t1 = {i: float(rng.uniform(50, 200)) for i in range(n_qubits)}
        t2 = {
            i: float(rng.uniform(30, min(t1[i], 150)))
            for i in range(n_qubits)
        }
        readout = {
            i: float(rng.uniform(0.01, 0.05)) for i in range(n_qubits)
        }
        gate_times = {"single": 35.0, "two": 300.0, "readout": 5000.0}

        return CalibrationData(
            single_qubit_errors=sq_errors,
            two_qubit_errors=tq_errors,
            t1_times=t1,
            t2_times=t2,
            readout_errors=readout,
            gate_times=gate_times,
        )

    @staticmethod
    def noisy_ion_trap(
        n_qubits: int, rng: Optional[np.random.RandomState] = None
    ) -> "CalibrationData":
        """Generate realistic trapped-ion device calibration.

        Models a device like IonQ Aria with typical error rates:
        - Single-qubit errors: ~1e-5 to 5e-5.
        - Two-qubit errors: ~1e-3 to 5e-3.
        - T1: 1e6 microseconds (effectively infinite).
        - T2: 1e5 to 1e6 microseconds.
        - Readout errors: ~0.1% to 1%.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        rng : np.random.RandomState, optional
            Random state for reproducibility.
        """
        if rng is None:
            rng = np.random.RandomState(42)

        sq_errors = {
            i: float(rng.uniform(1e-5, 5e-5)) for i in range(n_qubits)
        }
        tq_errors = {
            (i, j): float(rng.uniform(1e-3, 5e-3))
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        }
        t1 = {i: 1e6 for i in range(n_qubits)}
        t2 = {
            i: float(rng.uniform(1e5, 1e6)) for i in range(n_qubits)
        }
        readout = {
            i: float(rng.uniform(0.001, 0.01)) for i in range(n_qubits)
        }
        gate_times = {"single": 10000.0, "two": 200000.0, "readout": 200000.0}

        return CalibrationData(
            single_qubit_errors=sq_errors,
            two_qubit_errors=tq_errors,
            t1_times=t1,
            t2_times=t2,
            readout_errors=readout,
            gate_times=gate_times,
        )

    def best_qubits(self, n: int) -> List[int]:
        """Return n qubits with the lowest combined error rates.

        Sorts qubits by a composite score combining single-qubit gate
        error and readout error, then returns the top n.

        Parameters
        ----------
        n : int
            Number of best qubits to return.
        """
        all_qubits = sorted(self.single_qubit_errors.keys())
        scored = [
            (q, self.single_qubit_errors.get(q, 1.0) +
             self.readout_errors.get(q, 1.0))
            for q in all_qubits
        ]
        scored.sort(key=lambda x: x[1])
        return [q for q, _ in scored[:n]]

    def best_pairs(self, n: int) -> List[Tuple[int, int]]:
        """Return n qubit pairs with the lowest two-qubit error rates.

        Parameters
        ----------
        n : int
            Number of best pairs to return.
        """
        pairs = sorted(
            self.two_qubit_errors.items(), key=lambda x: x[1]
        )
        return [pair for pair, _ in pairs[:n]]

    def qubit_score(self, qubit: int) -> float:
        """Compute a quality score for a qubit (lower is better).

        Combines single-qubit error, readout error, and coherence
        times into a single metric.
        """
        sq_err = self.single_qubit_errors.get(qubit, 0.01)
        ro_err = self.readout_errors.get(qubit, 0.05)
        t1 = self.t1_times.get(qubit, 100.0)
        t2 = self.t2_times.get(qubit, 50.0)
        # Normalize coherence to a 0-1 penalty (shorter = worse).
        t1_penalty = 1.0 / max(t1, 1.0)
        t2_penalty = 1.0 / max(t2, 1.0)
        return sq_err + ro_err + t1_penalty + t2_penalty

    def pair_score(self, q1: int, q2: int) -> float:
        """Compute a quality score for a qubit pair (lower is better)."""
        key = (min(q1, q2), max(q1, q2))
        tq_err = self.two_qubit_errors.get(key, 0.05)
        return tq_err + self.qubit_score(q1) + self.qubit_score(q2)


# ------------------------------------------------------------------
# Noise-adaptive router
# ------------------------------------------------------------------

@dataclass
class NoiseAdaptiveResult:
    """Result of noise-adaptive routing.

    Attributes
    ----------
    circuit : object
        The routed circuit (gate list).
    layout : dict[int, int]
        Logical-to-physical qubit mapping.
    expected_fidelity : float
        Estimated circuit fidelity after routing.
    swap_count : int
        Number of SWAP gates inserted.
    """

    circuit: object
    layout: Dict[int, int]
    expected_fidelity: float
    swap_count: int


@dataclass
class NoiseAdaptiveRouter:
    """Route circuits considering qubit error rates.

    Uses calibration data to weight routing decisions: instead of
    minimizing topological distance, minimizes expected error along
    SWAP paths.

    Attributes
    ----------
    calibration : CalibrationData
        Hardware calibration snapshot.
    """

    calibration: CalibrationData

    def noise_weighted_distance(
        self, q1: int, q2: int, coupling_map
    ) -> float:
        """Compute noise-weighted distance between two qubits.

        Instead of hop count, sums the two-qubit error rates along
        the shortest path.

        Parameters
        ----------
        q1, q2 : int
            Physical qubit indices.
        coupling_map
            A CouplingMap instance with ``shortest_path`` method.

        Returns
        -------
        float
            Noise-weighted distance (higher = worse).
        """
        if q1 == q2:
            return 0.0

        path = coupling_map.shortest_path(q1, q2)
        if not path or len(path) < 2:
            return float("inf")

        total_error = 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            key = (min(a, b), max(a, b))
            err = self.calibration.two_qubit_errors.get(key, 0.05)
            total_error += err

        return total_error

    def select_initial_layout(
        self, n_logical: int, coupling_map
    ) -> Dict[int, int]:
        """Choose initial qubit mapping based on calibration.

        Places logical qubits on the highest-quality physical qubits,
        prioritizing low error rates and long coherence times.

        Parameters
        ----------
        n_logical : int
            Number of logical qubits.
        coupling_map
            A CouplingMap instance.

        Returns
        -------
        dict[int, int]
            Mapping from logical qubit to physical qubit.
        """
        n_physical = coupling_map.num_qubits
        best = self.calibration.best_qubits(min(n_logical, n_physical))

        layout: Dict[int, int] = {}
        used: set = set()

        for logical in range(n_logical):
            if logical < len(best):
                phys = best[logical]
                layout[logical] = phys
                used.add(phys)
            else:
                # Fallback: assign unused physical qubits.
                for p in range(n_physical):
                    if p not in used:
                        layout[logical] = p
                        used.add(p)
                        break

        return layout

    def route(self, circuit, coupling_map) -> NoiseAdaptiveResult:
        """Route a circuit minimizing expected error.

        Uses a greedy approach with noise-weighted SWAP selection.

        Parameters
        ----------
        circuit
            A circuit-like object with ``gates``, ``num_qubits``.
        coupling_map
            A CouplingMap instance.

        Returns
        -------
        NoiseAdaptiveResult
            Routing result with fidelity estimate.
        """
        n_logical = circuit.num_qubits
        layout = self.select_initial_layout(n_logical, coupling_map)
        inverse_layout = {v: k for k, v in layout.items()}

        routed_gates: List[dict] = []
        swap_count = 0

        for gate in circuit.gates:
            if hasattr(gate, "is_single_qubit") and gate.is_single_qubit:
                pq = layout[gate.qubits[0]]
                routed_gates.append({
                    "name": gate.name, "qubits": (pq,),
                    "params": gate.params,
                })
            elif hasattr(gate, "is_two_qubit") and gate.is_two_qubit:
                l0, l1 = gate.qubits
                p0 = layout[l0]
                p1 = layout[l1]

                # Insert SWAPs if not adjacent.
                while not coupling_map.are_connected(p0, p1):
                    path = coupling_map.shortest_path(p0, p1)
                    if len(path) < 2:
                        break

                    # Choose the SWAP with lowest error along the path.
                    best_next = path[1]
                    best_err = float("inf")
                    for candidate in coupling_map.neighbors(p0):
                        # Would swapping to this neighbor help?
                        new_dist = self.noise_weighted_distance(
                            candidate, p1, coupling_map
                        )
                        key = (min(p0, candidate), max(p0, candidate))
                        swap_err = self.calibration.two_qubit_errors.get(
                            key, 0.05
                        )
                        total = new_dist + swap_err
                        if total < best_err:
                            best_err = total
                            best_next = candidate

                    routed_gates.append({
                        "name": "SWAP",
                        "qubits": (p0, best_next),
                        "params": (),
                    })
                    # Update layout.
                    lo0 = inverse_layout.get(p0, -1)
                    lo1 = inverse_layout.get(best_next, -1)
                    if lo0 >= 0:
                        layout[lo0] = best_next
                    if lo1 >= 0:
                        layout[lo1] = p0
                    inverse_layout[p0] = lo1
                    inverse_layout[best_next] = lo0

                    swap_count += 1
                    p0 = best_next

                routed_gates.append({
                    "name": gate.name, "qubits": (p0, p1),
                    "params": gate.params,
                })
            else:
                # Three-qubit or other gates: pass through.
                mapped_qubits = tuple(layout[q] for q in gate.qubits)
                routed_gates.append({
                    "name": gate.name, "qubits": mapped_qubits,
                    "params": gate.params,
                })

        # Estimate fidelity.
        fidelity = self._estimate_routed_fidelity(routed_gates)

        return NoiseAdaptiveResult(
            circuit=routed_gates,
            layout=layout,
            expected_fidelity=fidelity,
            swap_count=swap_count,
        )

    def _estimate_routed_fidelity(
        self, routed_gates: List[dict]
    ) -> float:
        """Estimate the fidelity of a routed circuit."""
        fidelity = 1.0
        for gate in routed_gates:
            qubits = gate["qubits"]
            name = gate["name"].lower()
            if name == "swap":
                # SWAP = 3 CNOTs.
                q0, q1 = qubits
                key = (min(q0, q1), max(q0, q1))
                err = self.calibration.two_qubit_errors.get(key, 0.05)
                fidelity *= (1 - err) ** 3
            elif len(qubits) == 2:
                q0, q1 = qubits
                key = (min(q0, q1), max(q0, q1))
                err = self.calibration.two_qubit_errors.get(key, 0.05)
                fidelity *= (1 - err)
            elif len(qubits) == 1:
                err = self.calibration.single_qubit_errors.get(qubits[0], 0.001)
                fidelity *= (1 - err)
        return fidelity


# ------------------------------------------------------------------
# Noise-adaptive decomposer
# ------------------------------------------------------------------

@dataclass
class NoiseAdaptiveDecomposer:
    """Choose gate decomposition based on hardware error rates.

    Selects the optimal two-qubit gate basis for a given qubit pair
    by comparing the expected fidelity of different decomposition
    strategies.

    Attributes
    ----------
    calibration : CalibrationData
        Hardware calibration snapshot.
    """

    calibration: CalibrationData

    def decompose_two_qubit(
        self, gate_type: str, qubits: tuple
    ) -> List[dict]:
        """Choose between ZX, ZZ, or iSWAP decomposition.

        Parameters
        ----------
        gate_type : str
            Original gate type (e.g., ``"CX"``, ``"CZ"``).
        qubits : tuple
            (q0, q1) physical qubit indices.

        Returns
        -------
        list[dict]
            Decomposed gate sequence as dicts with name/qubits/params.
        """
        basis = self.optimal_basis(qubits)
        q0, q1 = qubits

        if gate_type.lower() in ("cx", "cnot"):
            if basis == "CX":
                return [{"name": "CX", "qubits": (q0, q1), "params": ()}]
            elif basis == "CZ":
                return [
                    {"name": "H", "qubits": (q1,), "params": ()},
                    {"name": "CZ", "qubits": (q0, q1), "params": ()},
                    {"name": "H", "qubits": (q1,), "params": ()},
                ]
            else:  # iSWAP basis
                return [
                    {"name": "Rz", "qubits": (q0,), "params": (math.pi / 2,)},
                    {"name": "iSWAP", "qubits": (q0, q1), "params": ()},
                    {"name": "Rz", "qubits": (q1,), "params": (math.pi / 2,)},
                ]

        if gate_type.lower() == "cz":
            if basis == "CZ":
                return [{"name": "CZ", "qubits": (q0, q1), "params": ()}]
            elif basis == "CX":
                return [
                    {"name": "H", "qubits": (q1,), "params": ()},
                    {"name": "CX", "qubits": (q0, q1), "params": ()},
                    {"name": "H", "qubits": (q1,), "params": ()},
                ]
            else:
                return [
                    {"name": "Rz", "qubits": (q0,), "params": (math.pi / 2,)},
                    {"name": "iSWAP", "qubits": (q0, q1), "params": ()},
                    {"name": "Rz", "qubits": (q1,), "params": (-math.pi / 2,)},
                ]

        # Default: pass through.
        return [{"name": gate_type, "qubits": qubits, "params": ()}]

    def optimal_basis(self, qubits: tuple) -> str:
        """Determine the optimal two-qubit basis for the given qubit pair.

        Compares error rates for different native two-qubit gates and
        returns the one with lowest expected error.

        Parameters
        ----------
        qubits : tuple
            (q0, q1) physical qubit indices.

        Returns
        -------
        str
            ``"CX"``, ``"CZ"``, or ``"iSWAP"``.
        """
        key = (min(qubits[0], qubits[1]), max(qubits[0], qubits[1]))
        base_error = self.calibration.two_qubit_errors.get(key, 0.05)

        # Heuristic: CX-native devices have lower error when used directly.
        # CZ-native devices (Google-style) have slightly different profiles.
        # iSWAP-native devices (also Google) may have even lower errors.
        # Without specific per-basis error data, use the base error with
        # small scaling factors.
        cx_error = base_error * 1.0
        cz_error = base_error * 1.05  # CZ typically slightly higher
        iswap_error = base_error * 1.1  # iSWAP typically highest

        errors = {"CX": cx_error, "CZ": cz_error, "iSWAP": iswap_error}
        return min(errors, key=errors.get)


# ------------------------------------------------------------------
# Circuit fidelity estimator
# ------------------------------------------------------------------

@dataclass
class CircuitFidelityEstimator:
    """Estimate expected circuit fidelity from calibration data.

    Provides both aggregate fidelity estimates and per-gate error
    budget breakdowns.

    Attributes
    ----------
    calibration : CalibrationData
        Hardware calibration snapshot.
    """

    calibration: CalibrationData

    def estimate(self, circuit) -> float:
        """Estimate fidelity as product of gate fidelities.

        Parameters
        ----------
        circuit
            A circuit-like object with a ``gates`` attribute.

        Returns
        -------
        float
            Expected fidelity in [0, 1].
        """
        fidelity = 1.0

        for gate in circuit.gates:
            n_q = len(gate.qubits)
            if n_q == 1:
                q = gate.qubits[0]
                err = self.calibration.single_qubit_errors.get(q, 0.001)
                fidelity *= (1.0 - err)
            elif n_q == 2:
                q0, q1 = gate.qubits
                key = (min(q0, q1), max(q0, q1))
                err = self.calibration.two_qubit_errors.get(key, 0.05)
                fidelity *= (1.0 - err)
            elif n_q == 3:
                # Toffoli ~ 6 CNOTs.
                for i in range(len(gate.qubits)):
                    for j in range(i + 1, len(gate.qubits)):
                        qa, qb = gate.qubits[i], gate.qubits[j]
                        key = (min(qa, qb), max(qa, qb))
                        err = self.calibration.two_qubit_errors.get(key, 0.05)
                        fidelity *= (1.0 - err)

        # Readout errors.
        used_qubits = set()
        for gate in circuit.gates:
            for q in gate.qubits:
                used_qubits.add(q)
        for q in used_qubits:
            ro_err = self.calibration.readout_errors.get(q, 0.01)
            fidelity *= (1.0 - ro_err)

        return fidelity

    def error_budget(self, circuit) -> Dict[str, float]:
        """Break down error contributions by gate type.

        Parameters
        ----------
        circuit
            A circuit-like object with a ``gates`` attribute.

        Returns
        -------
        dict[str, float]
            Gate type -> total error contribution.
        """
        budget: Dict[str, float] = {}

        for gate in circuit.gates:
            n_q = len(gate.qubits)
            if n_q == 1:
                q = gate.qubits[0]
                err = self.calibration.single_qubit_errors.get(q, 0.001)
            elif n_q == 2:
                q0, q1 = gate.qubits
                key = (min(q0, q1), max(q0, q1))
                err = self.calibration.two_qubit_errors.get(key, 0.05)
            else:
                err = 0.1  # Rough estimate for multi-qubit gates.

            name = gate.name
            budget[name] = budget.get(name, 0.0) + err

        # Add readout.
        used_qubits = set()
        for gate in circuit.gates:
            for q in gate.qubits:
                used_qubits.add(q)
        ro_total = sum(
            self.calibration.readout_errors.get(q, 0.01) for q in used_qubits
        )
        if ro_total > 0:
            budget["readout"] = ro_total

        return budget

    def suggest_improvements(self, circuit) -> List[str]:
        """Suggest transpilation improvements based on error analysis.

        Parameters
        ----------
        circuit
            A circuit-like object with a ``gates`` attribute.

        Returns
        -------
        list[str]
            Actionable suggestions.
        """
        suggestions: List[str] = []
        budget = self.error_budget(circuit)
        fidelity = self.estimate(circuit)

        if fidelity < 0.5:
            suggestions.append(
                "Circuit fidelity is below 50%. Consider reducing "
                "circuit depth or using error mitigation."
            )

        # Find the dominant error source.
        if budget:
            worst_gate = max(budget, key=budget.get)
            worst_err = budget[worst_gate]
            if worst_gate == "readout":
                suggestions.append(
                    "Readout errors dominate. Consider using readout "
                    "error mitigation (e.g., TREX or M3)."
                )
            elif worst_err > 0.1:
                suggestions.append(
                    f"Gate '{worst_gate}' contributes {worst_err:.3f} total "
                    f"error. Consider alternative decomposition or qubit "
                    f"assignment."
                )

        # Check for high-error qubits.
        used_qubits = set()
        for gate in circuit.gates:
            for q in gate.qubits:
                used_qubits.add(q)

        for q in used_qubits:
            sq_err = self.calibration.single_qubit_errors.get(q, 0.0)
            if sq_err > 0.001:
                suggestions.append(
                    f"Qubit {q} has high single-qubit error "
                    f"({sq_err:.4f}). Consider remapping to a better qubit."
                )
                break  # One suggestion is enough.

        # Check for high-error pairs.
        for gate in circuit.gates:
            if len(gate.qubits) == 2:
                q0, q1 = gate.qubits
                key = (min(q0, q1), max(q0, q1))
                tq_err = self.calibration.two_qubit_errors.get(key, 0.0)
                if tq_err > 0.02:
                    suggestions.append(
                        f"Qubit pair ({q0}, {q1}) has high two-qubit error "
                        f"({tq_err:.4f}). Consider routing through a "
                        f"lower-error path."
                    )
                    break

        # T1/T2 coherence warnings.
        gate_time = self.calibration.gate_times.get("two", 300.0)
        for gate in circuit.gates:
            for q in gate.qubits:
                t2 = self.calibration.t2_times.get(q, 100.0)
                # Rough circuit time estimate.
                n_gates = sum(1 for g in circuit.gates if q in g.qubits)
                circuit_time_us = n_gates * gate_time / 1000.0
                if circuit_time_us > 0.5 * t2:
                    suggestions.append(
                        f"Qubit {q} circuit time ({circuit_time_us:.1f} us) "
                        f"approaches T2 ({t2:.1f} us). Decoherence will "
                        f"significantly impact fidelity."
                    )
                    break
            else:
                continue
            break

        if not suggestions:
            suggestions.append("No obvious improvements detected.")

        return suggestions
