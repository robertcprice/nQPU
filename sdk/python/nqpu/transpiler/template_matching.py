"""Template-based circuit optimization via DAG pattern matching.

Provides a directed acyclic graph (DAG) representation of quantum
circuits and a template-matching engine that finds and replaces known
gate patterns with more efficient equivalents.

The matching algorithm:

1. Builds a DAG from the circuit gate list, with edges representing
   qubit-wire dependencies.
2. For each template, performs subgraph isomorphism search using a
   backtracking algorithm constrained by gate type and qubit structure.
3. When a match is found, replaces the matched sub-DAG with the
   template's replacement pattern.
4. Iterates until no more templates match (fixpoint).

Includes 20+ built-in optimization templates covering common patterns
such as gate cancellation, rotation merging, and CNOT simplification.

All computations are pure numpy with no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# DAG node
# ------------------------------------------------------------------

@dataclass
class DAGNode:
    """A single node (gate) in the circuit DAG.

    Attributes
    ----------
    gate_type : str
        Gate name (e.g., ``"H"``, ``"CX"``, ``"Rz"``).
    qubits : list
        Target qubit indices.
    params : list
        Gate parameters (rotation angles).
    index : int
        Node index in the DAG.
    """

    gate_type: str
    qubits: list
    params: list = field(default_factory=list)
    index: int = 0

    def __repr__(self) -> str:
        if self.params:
            p = ", ".join(f"{v:.4f}" for v in self.params)
            return f"DAGNode({self.gate_type}({p}), q{self.qubits}, idx={self.index})"
        return f"DAGNode({self.gate_type}, q{self.qubits}, idx={self.index})"


# ------------------------------------------------------------------
# Circuit DAG
# ------------------------------------------------------------------

@dataclass
class CircuitDAG:
    """Directed acyclic graph representation of a quantum circuit.

    Nodes represent gates and directed edges represent qubit-wire
    dependencies: if gate B uses a qubit that gate A last wrote to,
    there is an edge A -> B.

    Attributes
    ----------
    n_qubits : int
        Number of qubits.
    nodes : list[DAGNode]
        Gate nodes in insertion order.
    """

    n_qubits: int
    nodes: List[DAGNode] = field(default_factory=list)
    _succ: Dict[int, List[int]] = field(default_factory=dict, repr=False)
    _pred: Dict[int, List[int]] = field(default_factory=dict, repr=False)
    _last_on_qubit: Dict[int, int] = field(default_factory=dict, repr=False)

    def add_gate(
        self, gate_type: str, qubits: list, params: list = None
    ) -> int:
        """Add a gate node and return its index.

        Automatically creates edges from the last gate on each qubit.
        """
        if params is None:
            params = []
        idx = len(self.nodes)
        node = DAGNode(gate_type=gate_type, qubits=list(qubits),
                       params=list(params), index=idx)
        self.nodes.append(node)
        self._succ[idx] = []
        self._pred[idx] = []

        for q in qubits:
            if q in self._last_on_qubit:
                prev = self._last_on_qubit[q]
                if idx not in self._succ[prev]:
                    self._succ[prev].append(idx)
                if prev not in self._pred[idx]:
                    self._pred[idx].append(prev)
            self._last_on_qubit[q] = idx

        return idx

    def successors(self, node_idx: int) -> List[int]:
        """Return successor node indices (gates that depend on this one)."""
        return list(self._succ.get(node_idx, []))

    def predecessors(self, node_idx: int) -> List[int]:
        """Return predecessor node indices."""
        return list(self._pred.get(node_idx, []))

    def topological_order(self) -> List[int]:
        """Return node indices in topological (dependency-respecting) order."""
        in_degree = {i: len(self._pred.get(i, [])) for i in range(len(self.nodes))}
        queue = [i for i, d in in_degree.items() if d == 0]
        order: List[int] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for s in self._succ.get(node, []):
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)
        return order

    @property
    def depth(self) -> int:
        """Circuit depth (longest path through the DAG)."""
        if not self.nodes:
            return 0
        qubit_depth = [0] * self.n_qubits
        for idx in self.topological_order():
            node = self.nodes[idx]
            layer = max((qubit_depth[q] for q in node.qubits), default=0) + 1
            for q in node.qubits:
                qubit_depth[q] = layer
        return max(qubit_depth) if qubit_depth else 0

    @property
    def gate_count(self) -> int:
        """Total number of gates."""
        return len(self.nodes)

    def copy(self) -> "CircuitDAG":
        """Create a deep copy of this DAG."""
        new_dag = CircuitDAG(n_qubits=self.n_qubits)
        for node in self.nodes:
            new_dag.add_gate(node.gate_type, node.qubits, node.params)
        return new_dag

    def to_gate_list(self) -> List[Tuple[str, list, list]]:
        """Convert to a list of (gate_type, qubits, params) tuples."""
        return [
            (n.gate_type, list(n.qubits), list(n.params))
            for n in self.nodes
        ]

    @staticmethod
    def from_gate_list(
        n_qubits: int, gate_list: List[Tuple[str, list, list]]
    ) -> "CircuitDAG":
        """Build a DAG from a list of (gate_type, qubits, params) tuples."""
        dag = CircuitDAG(n_qubits=n_qubits)
        for gt, qs, ps in gate_list:
            dag.add_gate(gt, qs, ps)
        return dag

    def __repr__(self) -> str:
        return f"CircuitDAG(n_qubits={self.n_qubits}, gates={len(self.nodes)})"


# ------------------------------------------------------------------
# Template
# ------------------------------------------------------------------

@dataclass
class Template:
    """A circuit pattern with its replacement.

    Attributes
    ----------
    name : str
        Human-readable template name.
    pattern : CircuitDAG
        The pattern to match.
    replacement : CircuitDAG
        The replacement circuit.
    """

    name: str
    pattern: CircuitDAG
    replacement: CircuitDAG

    def savings(self) -> int:
        """Gate count reduction from applying this template."""
        return self.pattern.gate_count - self.replacement.gate_count


# ------------------------------------------------------------------
# Template matcher
# ------------------------------------------------------------------

@dataclass
class TemplateMatchResult:
    """Result of template-based optimization.

    Attributes
    ----------
    optimized : CircuitDAG
        The optimized circuit DAG.
    templates_applied : list[str]
        Names of templates that were applied.
    gates_before : int
        Gate count before optimization.
    gates_after : int
        Gate count after optimization.
    iterations : int
        Number of optimization iterations performed.
    """

    optimized: CircuitDAG
    templates_applied: List[str]
    gates_before: int
    gates_after: int
    iterations: int


@dataclass
class TemplateMatcher:
    """Find and replace circuit patterns using template matching.

    Attributes
    ----------
    templates : list[Template]
        Templates to apply during optimization.
    """

    templates: List[Template] = field(default_factory=list)

    def match(
        self, circuit: CircuitDAG, template: Template
    ) -> List[Dict[int, int]]:
        """Find all matches of a template pattern in the circuit.

        Performs subgraph isomorphism search using backtracking,
        constrained by gate type and qubit consistency.

        Parameters
        ----------
        circuit : CircuitDAG
            The circuit to search.
        template : Template
            The template whose pattern to match.

        Returns
        -------
        list[dict[int, int]]
            List of mappings from template node index to circuit node index.
        """
        pattern = template.pattern
        if pattern.gate_count == 0:
            return []
        if pattern.gate_count > circuit.gate_count:
            return []

        matches: List[Dict[int, int]] = []
        pattern_order = pattern.topological_order()

        # Try to match starting from each circuit node.
        for start_idx in range(len(circuit.nodes)):
            mapping: Dict[int, int] = {}
            qubit_map: Dict[int, int] = {}
            self._backtrack_match(
                circuit, pattern, pattern_order, 0,
                mapping, qubit_map, matches
            )

        # De-duplicate matches (same set of circuit nodes).
        unique: List[Dict[int, int]] = []
        seen_sets: List[frozenset] = []
        for m in matches:
            key = frozenset(m.values())
            if key not in seen_sets:
                seen_sets.append(key)
                unique.append(m)

        return unique

    def _backtrack_match(
        self,
        circuit: CircuitDAG,
        pattern: CircuitDAG,
        pattern_order: List[int],
        pos: int,
        mapping: Dict[int, int],
        qubit_map: Dict[int, int],
        results: List[Dict[int, int]],
    ) -> None:
        """Recursive backtracking for subgraph isomorphism."""
        if pos == len(pattern_order):
            results.append(dict(mapping))
            return

        p_idx = pattern_order[pos]
        p_node = pattern.nodes[p_idx]

        for c_idx in range(len(circuit.nodes)):
            if c_idx in mapping.values():
                continue

            c_node = circuit.nodes[c_idx]

            # Gate type must match.
            if c_node.gate_type != p_node.gate_type:
                continue
            # Same number of qubits.
            if len(c_node.qubits) != len(p_node.qubits):
                continue
            # Params must match (for parametric gates, check equality).
            if p_node.params and c_node.params:
                if len(p_node.params) != len(c_node.params):
                    continue
                if not all(
                    abs(a - b) < 1e-10
                    for a, b in zip(p_node.params, c_node.params)
                ):
                    continue
            elif bool(p_node.params) != bool(c_node.params):
                # One has params and the other doesn't.
                # For non-parametric templates, both should have no params.
                if p_node.params:
                    continue

            # Check qubit consistency.
            new_qubit_map = dict(qubit_map)
            consistent = True
            for pq, cq in zip(p_node.qubits, c_node.qubits):
                if pq in new_qubit_map:
                    if new_qubit_map[pq] != cq:
                        consistent = False
                        break
                else:
                    # Check that cq is not already mapped from a
                    # different pattern qubit.
                    if cq in new_qubit_map.values():
                        other_pq = [k for k, v in new_qubit_map.items() if v == cq]
                        if other_pq and other_pq[0] != pq:
                            consistent = False
                            break
                    new_qubit_map[pq] = cq

            if not consistent:
                continue

            # Check dependency consistency: if pattern has edge p_prev -> p_idx,
            # then circuit must have a path from mapping[p_prev] to c_idx.
            dep_ok = True
            for p_pred in pattern.predecessors(p_idx):
                if p_pred in mapping:
                    c_pred = mapping[p_pred]
                    # The circuit predecessor should come before c_idx
                    # in the qubit dependency chain.
                    if c_pred >= c_idx:
                        dep_ok = False
                        break

            if not dep_ok:
                continue

            mapping[p_idx] = c_idx
            self._backtrack_match(
                circuit, pattern, pattern_order, pos + 1,
                mapping, new_qubit_map, results
            )
            del mapping[p_idx]

    def apply_template(
        self,
        circuit: CircuitDAG,
        template: Template,
        mapping: Dict[int, int],
    ) -> CircuitDAG:
        """Replace a matched pattern with the template's replacement.

        Parameters
        ----------
        circuit : CircuitDAG
            The original circuit.
        template : Template
            The template to apply.
        mapping : dict[int, int]
            Mapping from template pattern node index to circuit node index.

        Returns
        -------
        CircuitDAG
            New circuit with the pattern replaced.
        """
        matched_circuit_nodes = set(mapping.values())

        # Build qubit mapping from pattern to circuit.
        qubit_map: Dict[int, int] = {}
        for p_idx, c_idx in mapping.items():
            p_node = template.pattern.nodes[p_idx]
            c_node = circuit.nodes[c_idx]
            for pq, cq in zip(p_node.qubits, c_node.qubits):
                qubit_map[pq] = cq

        # Reconstruct the circuit: keep non-matched gates in order,
        # insert replacement gates where the first matched gate was.
        new_dag = CircuitDAG(n_qubits=circuit.n_qubits)
        first_matched = min(matched_circuit_nodes)
        replacement_inserted = False

        for idx in range(len(circuit.nodes)):
            if idx in matched_circuit_nodes:
                if not replacement_inserted and idx == first_matched:
                    # Insert the replacement.
                    for r_node in template.replacement.nodes:
                        mapped_qubits = [
                            qubit_map.get(q, q) for q in r_node.qubits
                        ]
                        new_dag.add_gate(
                            r_node.gate_type, mapped_qubits, r_node.params
                        )
                    replacement_inserted = True
                # Skip the matched node.
                continue
            else:
                node = circuit.nodes[idx]
                new_dag.add_gate(node.gate_type, node.qubits, node.params)

        return new_dag

    def optimize(
        self, circuit: CircuitDAG, max_iterations: int = 10
    ) -> TemplateMatchResult:
        """Repeatedly apply all templates until no more matches found.

        Parameters
        ----------
        circuit : CircuitDAG
            Circuit to optimize.
        max_iterations : int
            Maximum number of full passes over all templates.

        Returns
        -------
        TemplateMatchResult
            Optimization result with metrics.
        """
        gates_before = circuit.gate_count
        templates_applied: List[str] = []
        current = circuit.copy()
        iteration = 0

        for iteration in range(1, max_iterations + 1):
            changed = False
            for tmpl in self.templates:
                if tmpl.savings() <= 0:
                    continue
                matches = self.match(current, tmpl)
                if matches:
                    # Apply the first match.
                    current = self.apply_template(current, tmpl, matches[0])
                    templates_applied.append(tmpl.name)
                    changed = True
                    break  # Restart from first template.

            if not changed:
                break

        return TemplateMatchResult(
            optimized=current,
            templates_applied=templates_applied,
            gates_before=gates_before,
            gates_after=current.gate_count,
            iterations=iteration,
        )


# ------------------------------------------------------------------
# Built-in templates (20+)
# ------------------------------------------------------------------

def _make_template(
    name: str,
    n_qubits: int,
    pattern_gates: List[Tuple[str, list, list]],
    replacement_gates: List[Tuple[str, list, list]],
) -> Template:
    """Helper to build a Template from gate lists."""
    pattern = CircuitDAG(n_qubits=n_qubits)
    for gt, qs, ps in pattern_gates:
        pattern.add_gate(gt, qs, ps)
    replacement = CircuitDAG(n_qubits=n_qubits)
    for gt, qs, ps in replacement_gates:
        replacement.add_gate(gt, qs, ps)
    return Template(name=name, pattern=pattern, replacement=replacement)


def cnot_cancellation() -> Template:
    """CNOT followed by CNOT = identity."""
    return _make_template(
        "cnot_cancellation", 2,
        [("CX", [0, 1], []), ("CX", [0, 1], [])],
        [],
    )


def hadamard_cancellation() -> Template:
    """H followed by H = identity."""
    return _make_template(
        "hadamard_cancellation", 1,
        [("H", [0], []), ("H", [0], [])],
        [],
    )


def x_cancellation() -> Template:
    """X followed by X = identity."""
    return _make_template(
        "x_cancellation", 1,
        [("X", [0], []), ("X", [0], [])],
        [],
    )


def y_cancellation() -> Template:
    """Y followed by Y = identity."""
    return _make_template(
        "y_cancellation", 1,
        [("Y", [0], []), ("Y", [0], [])],
        [],
    )


def z_cancellation() -> Template:
    """Z followed by Z = identity."""
    return _make_template(
        "z_cancellation", 1,
        [("Z", [0], []), ("Z", [0], [])],
        [],
    )


def s_sdg_cancellation() -> Template:
    """S followed by Sdg = identity."""
    return _make_template(
        "s_sdg_cancellation", 1,
        [("S", [0], []), ("Sdg", [0], [])],
        [],
    )


def sdg_s_cancellation() -> Template:
    """Sdg followed by S = identity."""
    return _make_template(
        "sdg_s_cancellation", 1,
        [("Sdg", [0], []), ("S", [0], [])],
        [],
    )


def t_tdg_cancellation() -> Template:
    """T followed by Tdg = identity."""
    return _make_template(
        "t_tdg_cancellation", 1,
        [("T", [0], []), ("Tdg", [0], [])],
        [],
    )


def tdg_t_cancellation() -> Template:
    """Tdg followed by T = identity."""
    return _make_template(
        "tdg_t_cancellation", 1,
        [("Tdg", [0], []), ("T", [0], [])],
        [],
    )


def cx_direction_reversal() -> Template:
    """H-H-CX-H-H around both qubits reverses CX direction.

    CX(0,1) = H(0) H(1) CX(1,0) H(1) H(0).
    Pattern: H(0) H(1) CX(1,0) H(1) H(0) -> CX(0,1).
    """
    return _make_template(
        "cx_direction_reversal", 2,
        [
            ("H", [0], []), ("H", [1], []),
            ("CX", [1, 0], []),
            ("H", [1], []), ("H", [0], []),
        ],
        [("CX", [0, 1], [])],
    )


def rz_merge() -> Template:
    """Rz(pi/4) Rz(pi/4) = Rz(pi/2).

    Note: Only matches specific angle values. For general angle merging,
    use the RotationMerging optimization pass.
    """
    return _make_template(
        "rz_merge", 1,
        [
            ("Rz", [0], [math.pi / 4]),
            ("Rz", [0], [math.pi / 4]),
        ],
        [("Rz", [0], [math.pi / 2])],
    )


def rx_merge() -> Template:
    """Rx(pi/4) Rx(pi/4) = Rx(pi/2)."""
    return _make_template(
        "rx_merge", 1,
        [
            ("Rx", [0], [math.pi / 4]),
            ("Rx", [0], [math.pi / 4]),
        ],
        [("Rx", [0], [math.pi / 2])],
    )


def t_t_to_s() -> Template:
    """T T = S."""
    return _make_template(
        "t_t_to_s", 1,
        [("T", [0], []), ("T", [0], [])],
        [("S", [0], [])],
    )


def s_s_to_z() -> Template:
    """S S = Z."""
    return _make_template(
        "s_s_to_z", 1,
        [("S", [0], []), ("S", [0], [])],
        [("Z", [0], [])],
    )


def cx_cx_swap() -> Template:
    """CX(0,1) CX(1,0) CX(0,1) = SWAP(0,1)."""
    return _make_template(
        "cx_cx_swap", 2,
        [
            ("CX", [0, 1], []),
            ("CX", [1, 0], []),
            ("CX", [0, 1], []),
        ],
        [("SWAP", [0, 1], [])],
    )


def swap_cancellation() -> Template:
    """SWAP followed by SWAP = identity."""
    return _make_template(
        "swap_cancellation", 2,
        [("SWAP", [0, 1], []), ("SWAP", [0, 1], [])],
        [],
    )


def cz_cancellation() -> Template:
    """CZ followed by CZ = identity."""
    return _make_template(
        "cz_cancellation", 2,
        [("CZ", [0, 1], []), ("CZ", [0, 1], [])],
        [],
    )


def h_cz_h_to_cx() -> Template:
    """H(1) CZ(0,1) H(1) = CX(0,1)."""
    return _make_template(
        "h_cz_h_to_cx", 2,
        [("H", [1], []), ("CZ", [0, 1], []), ("H", [1], [])],
        [("CX", [0, 1], [])],
    )


def tdg_tdg_to_sdg() -> Template:
    """Tdg Tdg = Sdg."""
    return _make_template(
        "tdg_tdg_to_sdg", 1,
        [("Tdg", [0], []), ("Tdg", [0], [])],
        [("Sdg", [0], [])],
    )


def sdg_sdg_to_z() -> Template:
    """Sdg Sdg = Z."""
    return _make_template(
        "sdg_sdg_to_z", 1,
        [("Sdg", [0], []), ("Sdg", [0], [])],
        [("Z", [0], [])],
    )


def h_x_h_to_z() -> Template:
    """H X H = Z."""
    return _make_template(
        "h_x_h_to_z", 1,
        [("H", [0], []), ("X", [0], []), ("H", [0], [])],
        [("Z", [0], [])],
    )


def h_z_h_to_x() -> Template:
    """H Z H = X."""
    return _make_template(
        "h_z_h_to_x", 1,
        [("H", [0], []), ("Z", [0], []), ("H", [0], [])],
        [("X", [0], [])],
    )


def default_templates() -> List[Template]:
    """Return all built-in optimization templates.

    Returns
    -------
    list[Template]
        List of 22 standard templates.
    """
    return [
        cnot_cancellation(),
        hadamard_cancellation(),
        x_cancellation(),
        y_cancellation(),
        z_cancellation(),
        s_sdg_cancellation(),
        sdg_s_cancellation(),
        t_tdg_cancellation(),
        tdg_t_cancellation(),
        cx_direction_reversal(),
        rz_merge(),
        rx_merge(),
        t_t_to_s(),
        s_s_to_z(),
        cx_cx_swap(),
        swap_cancellation(),
        cz_cancellation(),
        h_cz_h_to_cx(),
        tdg_tdg_to_sdg(),
        sdg_sdg_to_z(),
        h_x_h_to_z(),
        h_z_h_to_x(),
    ]
