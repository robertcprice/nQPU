"""Qubit routing algorithms for mapping logical circuits to hardware.

Implements SABRE (SWAP-Based BidiREctional heuristic search), trivial
identity routing, and greedy nearest-neighbour routing.  The algorithms
take a logical :class:`~nqpu.transpiler.circuits.QuantumCircuit` and a
:class:`~nqpu.transpiler.coupling.CouplingMap` and produce a routed
circuit where every two-qubit gate acts on physically adjacent qubits.

The SABRE implementation follows:
    Gushu Li, Yufei Ding, Yuan Xie.  *Tackling the Qubit Mapping Problem
    for NISQ-Era Quantum Devices*, ASPLOS 2019.

with the three heuristic variants (basic, look-ahead, decay) from the
Rust counterpart in ``sdk/rust/src/circuits/synthesis/transpiler.rs``.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .circuits import Gate, QuantumCircuit
from .coupling import CouplingMap


# ------------------------------------------------------------------
# Layout (qubit mapping)
# ------------------------------------------------------------------

@dataclass
class Layout:
    """Bidirectional mapping between logical and physical qubits.

    ``logical_to_physical[l] = p`` means logical qubit *l* is placed on
    physical qubit *p*, and vice versa for ``physical_to_logical``.
    """

    logical_to_physical: List[int]
    physical_to_logical: List[int]

    @classmethod
    def trivial(cls, n: int) -> "Layout":
        """Identity layout: logical *i* maps to physical *i*."""
        return cls(
            logical_to_physical=list(range(n)),
            physical_to_logical=list(range(n)),
        )

    @classmethod
    def from_mapping(cls, l2p: List[int], n_physical: int) -> "Layout":
        """Build from a logical-to-physical list.

        Parameters
        ----------
        l2p : list[int]
            ``l2p[logical] = physical``.
        n_physical : int
            Total number of physical qubits.
        """
        p2l = [-1] * n_physical
        for l, p in enumerate(l2p):
            p2l[p] = l
        return cls(logical_to_physical=list(l2p), physical_to_logical=p2l)

    def apply_swap(self, p0: int, p1: int) -> None:
        """Swap two physical qubits in the layout (in-place)."""
        l0 = self.physical_to_logical[p0]
        l1 = self.physical_to_logical[p1]
        self.physical_to_logical[p0] = l1
        self.physical_to_logical[p1] = l0
        if l0 >= 0:
            self.logical_to_physical[l0] = p1
        if l1 >= 0:
            self.logical_to_physical[l1] = p0

    def copy(self) -> "Layout":
        return Layout(
            logical_to_physical=list(self.logical_to_physical),
            physical_to_logical=list(self.physical_to_logical),
        )


# ------------------------------------------------------------------
# Initial layout strategies
# ------------------------------------------------------------------

class InitialLayout:
    """Factory for initial qubit-to-physical mappings."""

    @staticmethod
    def trivial_layout(n_logical: int, n_physical: int) -> Layout:
        """Logical *i* -> physical *i*."""
        l2p = list(range(n_logical))
        return Layout.from_mapping(l2p, n_physical)

    @staticmethod
    def random_layout(
        n_logical: int, n_physical: int, seed: Optional[int] = None
    ) -> Layout:
        """Random bijection from logical qubits to physical qubits."""
        rng = random.Random(seed)
        physicals = list(range(n_physical))
        rng.shuffle(physicals)
        l2p = physicals[:n_logical]
        return Layout.from_mapping(l2p, n_physical)

    @staticmethod
    def frequency_layout(
        circuit: QuantumCircuit, coupling_map: CouplingMap
    ) -> Layout:
        """Assign most-used logical qubits to most-connected physical qubits.

        This is a heuristic that tends to reduce routing overhead for
        circuits with non-uniform qubit usage.
        """
        n_logical = circuit.num_qubits
        n_physical = coupling_map.num_qubits

        # Count two-qubit interactions per logical qubit
        freq = [0] * n_logical
        for gate in circuit.gates:
            if gate.is_two_qubit:
                for q in gate.qubits:
                    freq[q] += 1

        # Sort logical qubits by frequency (descending)
        logical_order = sorted(range(n_logical), key=lambda q: -freq[q])

        # Sort physical qubits by degree (descending)
        physical_order = sorted(
            range(n_physical), key=lambda q: -coupling_map.degree(q)
        )

        l2p = [0] * n_logical
        for rank, lq in enumerate(logical_order):
            l2p[lq] = physical_order[rank]
        return Layout.from_mapping(l2p, n_physical)


# ------------------------------------------------------------------
# Routing result
# ------------------------------------------------------------------

@dataclass
class RoutingResult:
    """Output of a routing pass."""

    circuit: QuantumCircuit
    layout: Layout
    num_swaps_inserted: int
    initial_layout: Layout


# ------------------------------------------------------------------
# DAG helpers
# ------------------------------------------------------------------

def _compute_front_layer(
    gates: List[Gate], executed: List[bool]
) -> List[int]:
    """Compute the front layer: gate indices whose dependencies are all executed."""
    n = len(gates)
    last_on_qubit: Dict[int, int] = {}
    deps: List[List[int]] = [[] for _ in range(n)]
    for i, gate in enumerate(gates):
        for q in gate.qubits:
            if q in last_on_qubit:
                deps[i].append(last_on_qubit[q])
            last_on_qubit[q] = i
    front = []
    for i in range(n):
        if executed[i]:
            continue
        if all(executed[d] for d in deps[i]):
            front.append(i)
    return front


def _compute_extended_set(
    gates: List[Gate], executed: List[bool], front_layer: List[int]
) -> List[int]:
    """Compute the next-layer gates (extended set) after executing front_layer."""
    hyp = list(executed)
    for idx in front_layer:
        hyp[idx] = True
    return _compute_front_layer(gates, hyp)


# ------------------------------------------------------------------
# SABRE heuristic
# ------------------------------------------------------------------

class SabreHeuristic(Enum):
    """Heuristic variant for SABRE SWAP scoring."""
    BASIC = auto()
    LOOK_AHEAD = auto()
    DECAY = auto()


@dataclass
class SabreConfig:
    """Configuration for the SABRE router."""

    num_trials: int = 20
    decay_factor: float = 0.001
    heuristic: SabreHeuristic = SabreHeuristic.DECAY
    seed: int = 42


# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------

class TrivialRouter:
    """Identity routing -- no SWAPs inserted.

    Only works when the circuit already respects the coupling map
    (or when using all-to-all connectivity).
    """

    def route(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
    ) -> RoutingResult:
        layout = InitialLayout.trivial_layout(
            circuit.num_qubits, coupling_map.num_qubits
        )
        routed = circuit.copy()
        return RoutingResult(
            circuit=routed,
            layout=layout.copy(),
            num_swaps_inserted=0,
            initial_layout=layout,
        )


class GreedyRouter:
    """Greedy nearest-neighbour SWAP insertion.

    For each non-adjacent two-qubit gate, inserts SWAPs along the
    shortest path between its qubits until they are adjacent.
    Simple but typically produces more SWAPs than SABRE.
    """

    def route(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
    ) -> RoutingResult:
        n_logical = circuit.num_qubits
        n_physical = coupling_map.num_qubits
        layout = InitialLayout.trivial_layout(n_logical, n_physical)
        initial = layout.copy()

        routed = QuantumCircuit(n_physical)
        num_swaps = 0

        for gate in circuit.gates:
            if gate.is_single_qubit:
                pq = layout.logical_to_physical[gate.qubits[0]]
                routed.add_gate(Gate(gate.name, (pq,), gate.params))
            elif gate.is_two_qubit:
                l0, l1 = gate.qubits
                p0 = layout.logical_to_physical[l0]
                p1 = layout.logical_to_physical[l1]

                # Insert SWAPs to bring qubits adjacent
                while not coupling_map.are_connected(p0, p1):
                    path = coupling_map.shortest_path(p0, p1)
                    if len(path) < 2:
                        break
                    # SWAP p0 with the next qubit on the path
                    next_p = path[1]
                    routed.add_gate(Gate("SWAP", (p0, next_p)))
                    layout.apply_swap(p0, next_p)
                    num_swaps += 1
                    # Update p0 (it moved)
                    p0 = next_p
                    p1 = layout.logical_to_physical[l1]

                routed.add_gate(Gate(gate.name, (p0, p1), gate.params))
            else:
                # Three-qubit: route all pairs (simplified)
                mapped_qubits = tuple(
                    layout.logical_to_physical[q] for q in gate.qubits
                )
                routed.add_gate(Gate(gate.name, mapped_qubits, gate.params))

        return RoutingResult(
            circuit=routed,
            layout=layout,
            num_swaps_inserted=num_swaps,
            initial_layout=initial,
        )


class SABRERouter:
    """SABRE bidirectional routing (Gushu Li et al., ASPLOS 2019).

    Runs multiple trials with random initial layouts and keeps the best
    result.  Each trial performs a forward pass (inserting SWAPs with a
    heuristic score) followed by a backward pass to refine the initial
    layout.  The final forward pass with the refined layout produces the
    routed circuit.
    """

    def __init__(self, config: Optional[SabreConfig] = None) -> None:
        self.config = config or SabreConfig()

    def route(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
    ) -> RoutingResult:
        if not circuit.gates:
            layout = Layout.trivial(coupling_map.num_qubits)
            return RoutingResult(
                circuit=QuantumCircuit(coupling_map.num_qubits),
                layout=layout.copy(),
                num_swaps_inserted=0,
                initial_layout=layout,
            )

        n_logical = circuit.num_qubits
        n_physical = coupling_map.num_qubits

        if n_logical > n_physical:
            raise ValueError(
                f"Circuit needs {n_logical} qubits but device has {n_physical}"
            )

        # Precompute distance matrix for fast lookups
        dist = coupling_map._build_distance_matrix()
        adj = coupling_map._build_adj()

        best: Optional[RoutingResult] = None

        for trial in range(self.config.num_trials):
            trial_seed = self.config.seed + trial
            result = self._single_trial(
                circuit, coupling_map, n_logical, n_physical, dist, adj, trial_seed
            )
            if best is None or result.num_swaps_inserted < best.num_swaps_inserted:
                best = result

        assert best is not None
        return best

    # -- single SABRE trial --------------------------------------------

    def _single_trial(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
        n_logical: int,
        n_physical: int,
        dist: np.ndarray,
        adj: Dict[int, Set[int]],
        seed: int,
    ) -> RoutingResult:
        rng = random.Random(seed)

        # --- Forward pass (with random layout) to get a layout ---
        init_layout = InitialLayout.random_layout(n_logical, n_physical, seed)
        _, fwd_layout, _ = self._sabre_pass(
            circuit.gates, coupling_map, init_layout.copy(), n_physical, dist, adj, rng
        )

        # --- Backward pass (reversed circuit, starting from forward result) ---
        reversed_gates = list(reversed(circuit.gates))
        _, bwd_layout, _ = self._sabre_pass(
            reversed_gates, coupling_map, fwd_layout.copy(), n_physical, dist, adj, rng
        )

        # --- Final forward pass with the refined layout ---
        routed_gates, final_layout, num_swaps = self._sabre_pass(
            circuit.gates, coupling_map, bwd_layout.copy(), n_physical, dist, adj, rng
        )

        routed_circuit = QuantumCircuit(n_physical)
        for g in routed_gates:
            routed_circuit.add_gate(g)

        return RoutingResult(
            circuit=routed_circuit,
            layout=final_layout,
            num_swaps_inserted=num_swaps,
            initial_layout=bwd_layout,
        )

    # -- core SABRE forward pass ---------------------------------------

    def _sabre_pass(
        self,
        gates: List[Gate],
        coupling_map: CouplingMap,
        layout: Layout,
        n_physical: int,
        dist: np.ndarray,
        adj: Dict[int, Set[int]],
        rng: random.Random,
    ) -> Tuple[List[Gate], Layout, int]:
        """One SABRE forward pass.  Returns (routed_gates, layout, num_swaps)."""
        executed = [False] * len(gates)
        routed: List[Gate] = []
        num_swaps = 0
        decay = [1.0] * n_physical
        config = self.config

        iterations = 0
        max_iterations = len(gates) * n_physical * 4  # safety bound

        while iterations < max_iterations:
            iterations += 1
            front = _compute_front_layer(gates, executed)
            if not front:
                break

            # Execute all gates whose qubits are already adjacent
            progress = True
            while progress:
                progress = False
                front = _compute_front_layer(gates, executed)
                for idx in front:
                    gate = gates[idx]
                    if gate.num_qubits <= 1:
                        pq = layout.logical_to_physical[gate.qubits[0]]
                        routed.append(Gate(gate.name, (pq,), gate.params))
                        executed[idx] = True
                        progress = True
                    elif gate.num_qubits == 2:
                        p0 = layout.logical_to_physical[gate.qubits[0]]
                        p1 = layout.logical_to_physical[gate.qubits[1]]
                        if coupling_map.are_connected(p0, p1):
                            routed.append(Gate(gate.name, (p0, p1), gate.params))
                            executed[idx] = True
                            progress = True
                    else:
                        # 3+ qubit gates: check all pairs
                        mapped = tuple(
                            layout.logical_to_physical[q] for q in gate.qubits
                        )
                        all_adj = True
                        for i in range(len(mapped)):
                            for j in range(i + 1, len(mapped)):
                                if not coupling_map.are_connected(mapped[i], mapped[j]):
                                    all_adj = False
                                    break
                            if not all_adj:
                                break
                        if all_adj:
                            routed.append(Gate(gate.name, mapped, gate.params))
                            executed[idx] = True
                            progress = True

            # Check if we are done
            front = _compute_front_layer(gates, executed)
            if not front:
                break

            # Find the best SWAP to insert
            extended = _compute_extended_set(gates, executed, front)

            # Candidate SWAPs: all edges involving qubits in the front layer
            candidate_swaps: Set[Tuple[int, int]] = set()
            for idx in front:
                gate = gates[idx]
                for q in gate.qubits:
                    pq = layout.logical_to_physical[q]
                    for nb in adj.get(pq, set()):
                        swap = (min(pq, nb), max(pq, nb))
                        candidate_swaps.add(swap)

            if not candidate_swaps:
                # No candidates -- should not happen on a connected graph
                break

            best_swap = None
            best_score = float("inf")

            for swap in candidate_swaps:
                score = self._swap_score(
                    swap, front, gates, layout, dist, config, decay, extended
                )
                if score < best_score:
                    best_score = score
                    best_swap = swap

            if best_swap is None:
                break

            # Insert the SWAP
            p0, p1 = best_swap
            routed.append(Gate("SWAP", (p0, p1)))
            layout.apply_swap(p0, p1)
            num_swaps += 1

            # Update decay values
            decay[p0] += config.decay_factor
            decay[p1] += config.decay_factor

        return routed, layout, num_swaps

    # -- SWAP scoring --------------------------------------------------

    def _swap_score(
        self,
        swap: Tuple[int, int],
        front_layer: List[int],
        gates: List[Gate],
        layout: Layout,
        dist: np.ndarray,
        config: SabreConfig,
        decay: List[float],
        extended_set: List[int],
    ) -> float:
        """Compute the heuristic score for a candidate SWAP (lower is better)."""
        # Hypothetical layout after applying the swap
        l2p = list(layout.logical_to_physical)
        p2l = list(layout.physical_to_logical)
        p0, p1 = swap
        lo0, lo1 = p2l[p0], p2l[p1]
        if lo0 >= 0:
            l2p[lo0] = p1
        if lo1 >= 0:
            l2p[lo1] = p0

        # Front layer cost
        front_cost = 0.0
        num_front_2q = 0
        for idx in front_layer:
            gate = gates[idx]
            if gate.num_qubits >= 2:
                hp0 = l2p[gate.qubits[0]]
                hp1 = l2p[gate.qubits[1]]
                d = dist[hp0, hp1]
                if d >= 0:
                    front_cost += d
                else:
                    front_cost += 1000  # unreachable penalty
                num_front_2q += 1
        if num_front_2q > 0:
            front_cost /= num_front_2q

        if config.heuristic == SabreHeuristic.BASIC:
            return front_cost

        # Lookahead cost from extended set
        la_cost = 0.0
        num_ext_2q = 0
        for idx in extended_set:
            gate = gates[idx]
            if gate.num_qubits >= 2:
                hp0 = l2p[gate.qubits[0]]
                hp1 = l2p[gate.qubits[1]]
                d = dist[hp0, hp1]
                if d >= 0:
                    la_cost += d
                else:
                    la_cost += 1000
                num_ext_2q += 1
        if num_ext_2q > 0:
            la_cost /= num_ext_2q

        combined = front_cost + 0.5 * la_cost

        if config.heuristic == SabreHeuristic.LOOK_AHEAD:
            return combined

        # Decay heuristic
        d_val = max(decay[swap[0]], decay[swap[1]])
        return d_val * combined


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def route(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    router: str = "sabre",
    **kwargs,
) -> RoutingResult:
    """Route a circuit for a coupling map.

    Parameters
    ----------
    circuit : QuantumCircuit
        Logical circuit to route.
    coupling_map : CouplingMap
        Target hardware topology.
    router : str
        ``"trivial"``, ``"greedy"``, or ``"sabre"`` (default).
    **kwargs
        Extra keyword arguments forwarded to :class:`SabreConfig`.
    """
    if router == "trivial":
        return TrivialRouter().route(circuit, coupling_map)
    if router == "greedy":
        return GreedyRouter().route(circuit, coupling_map)
    if router == "sabre":
        config = SabreConfig(**kwargs) if kwargs else SabreConfig()
        return SABRERouter(config).route(circuit, coupling_map)
    raise ValueError(f"Unknown router: {router}")
