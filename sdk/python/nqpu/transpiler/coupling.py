"""Hardware coupling map (qubit connectivity graph).

Models the physical topology of a quantum processor as a graph where
nodes are physical qubits and edges represent pairs that can execute
two-qubit gates.  Mirrors the Rust ``CouplingMap`` in
``sdk/rust/src/circuits/synthesis/transpiler.rs``.

Factory methods produce standard topologies (grid, heavy-hex, ring, line)
and named device presets (IBM Eagle, Google Sycamore, Rigetti Aspen).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


@dataclass
class CouplingMap:
    """Undirected graph of qubit connectivity.

    Parameters
    ----------
    num_qubits : int
        Number of physical qubits.
    edges : set[frozenset[int]]
        Set of undirected edges (each a frozenset of two qubit indices).
    """

    num_qubits: int
    edges: Set[FrozenSet[int]] = field(default_factory=set)

    # Cached adjacency list (built lazily).
    _adj: Optional[Dict[int, Set[int]]] = field(
        default=None, repr=False, compare=False
    )
    # Cached all-pairs shortest-path distances.
    _dist: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    # -- construction helpers -------------------------------------------

    def _invalidate_cache(self) -> None:
        self._adj = None
        self._dist = None

    def add_edge(self, q0: int, q1: int) -> None:
        """Add an undirected edge between *q0* and *q1*."""
        if q0 == q1:
            return
        self.edges.add(frozenset((q0, q1)))
        self._invalidate_cache()

    def _build_adj(self) -> Dict[int, Set[int]]:
        if self._adj is not None:
            return self._adj
        adj: Dict[int, Set[int]] = {q: set() for q in range(self.num_qubits)}
        for edge in self.edges:
            a, b = tuple(edge)
            adj[a].add(b)
            adj[b].add(a)
        self._adj = adj
        return adj

    # -- queries --------------------------------------------------------

    def neighbors(self, q: int) -> List[int]:
        """Sorted list of neighbors of qubit *q*."""
        adj = self._build_adj()
        return sorted(adj.get(q, set()))

    def are_connected(self, q0: int, q1: int) -> bool:
        """True if *q0* and *q1* share a direct edge."""
        return frozenset((q0, q1)) in self.edges

    def degree(self, q: int) -> int:
        """Number of neighbors of qubit *q*."""
        return len(self.neighbors(q))

    def num_edges(self) -> int:
        return len(self.edges)

    # -- shortest path (BFS) -------------------------------------------

    def shortest_path(self, src: int, dst: int) -> List[int]:
        """BFS shortest path from *src* to *dst*.  Returns [] if unreachable."""
        if src == dst:
            return [src]
        if src >= self.num_qubits or dst >= self.num_qubits:
            return []
        adj = self._build_adj()
        visited = set()
        parent: Dict[int, int] = {}
        queue = deque([src])
        visited.add(src)
        while queue:
            current = queue.popleft()
            if current == dst:
                path = []
                node = dst
                while node != src:
                    path.append(node)
                    node = parent[node]
                path.append(src)
                path.reverse()
                return path
            for nb in adj.get(current, set()):
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = current
                    queue.append(nb)
        return []

    def distance(self, q0: int, q1: int) -> int:
        """Shortest-path distance.  Returns -1 if unreachable."""
        path = self.shortest_path(q0, q1)
        if not path and q0 != q1:
            return -1
        return max(len(path) - 1, 0)

    def _build_distance_matrix(self) -> np.ndarray:
        """All-pairs shortest-path distances via repeated BFS."""
        if self._dist is not None:
            return self._dist
        n = self.num_qubits
        dist = np.full((n, n), -1, dtype=np.int32)
        adj = self._build_adj()
        for src in range(n):
            visited = {src}
            d = {src: 0}
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nb in adj.get(cur, set()):
                    if nb not in visited:
                        visited.add(nb)
                        d[nb] = d[cur] + 1
                        queue.append(nb)
            for dst, val in d.items():
                dist[src, dst] = val
        self._dist = dist
        return dist

    def distance_matrix(self) -> np.ndarray:
        """Return ``(num_qubits, num_qubits)`` int32 array of pairwise distances."""
        return self._build_distance_matrix().copy()

    def is_connected(self) -> bool:
        """True if all qubits are reachable from qubit 0."""
        if self.num_qubits <= 1:
            return True
        adj = self._build_adj()
        visited: Set[int] = set()
        queue = deque([0])
        visited.add(0)
        while queue:
            cur = queue.popleft()
            for nb in adj.get(cur, set()):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return len(visited) == self.num_qubits

    def diameter(self) -> int:
        """Graph diameter (longest shortest path).  -1 if disconnected."""
        if not self.is_connected():
            return -1
        dist = self._build_distance_matrix()
        return int(dist.max())

    # -- edge list (convenience) ----------------------------------------

    def edge_list(self) -> List[Tuple[int, int]]:
        """Sorted list of ``(a, b)`` tuples with ``a < b``."""
        result = []
        for e in self.edges:
            a, b = sorted(e)
            result.append((a, b))
        result.sort()
        return result

    # -- factory methods ------------------------------------------------

    @classmethod
    def from_edge_list(
        cls, num_qubits: int, edges: List[Tuple[int, int]]
    ) -> "CouplingMap":
        """Create a coupling map from a list of ``(a, b)`` edges."""
        cm = cls(num_qubits=num_qubits)
        for a, b in edges:
            cm.add_edge(a, b)
        return cm

    @classmethod
    def from_line(cls, n: int) -> "CouplingMap":
        """Linear topology: ``0-1-2-..-(n-1)``."""
        cm = cls(num_qubits=n)
        for i in range(n - 1):
            cm.add_edge(i, i + 1)
        return cm

    @classmethod
    def from_ring(cls, n: int) -> "CouplingMap":
        """Ring topology: ``0-1-2-..(n-1)-0``."""
        cm = cls(num_qubits=n)
        for i in range(n - 1):
            cm.add_edge(i, i + 1)
        if n > 2:
            cm.add_edge(n - 1, 0)
        return cm

    @classmethod
    def from_grid(cls, rows: int, cols: int) -> "CouplingMap":
        """2D grid topology with ``rows * cols`` qubits."""
        n = rows * cols
        cm = cls(num_qubits=n)
        for r in range(rows):
            for c in range(cols):
                q = r * cols + c
                if c + 1 < cols:
                    cm.add_edge(q, q + 1)
                if r + 1 < rows:
                    cm.add_edge(q, q + cols)
        return cm

    @classmethod
    def from_heavy_hex(cls, unit_cells: int) -> "CouplingMap":
        """Simplified heavy-hex topology (IBM-style).

        Builds two rows of data qubits connected by bridge qubits,
        matching the Rust ``CouplingMap::heavy_hex`` layout.  The
        resulting qubit count is approximately ``5 * unit_cells + 4``.
        """
        if unit_cells == 0:
            return cls(num_qubits=0)

        row_len = 2 * unit_cells + 1
        edges: List[Tuple[int, int]] = []

        # Row A: qubits 0 .. row_len-1
        for i in range(row_len - 1):
            edges.append((i, i + 1))

        # Bridge qubits connecting row A to row C
        offset_b = row_len
        for i in range(unit_cells):
            bridge = offset_b + i
            top = 2 * i + 1
            edges.append((top, bridge))
            bottom = row_len + unit_cells + 2 * i + 1
            if bottom < row_len + unit_cells + row_len:
                edges.append((bridge, bottom))

        # Row C: qubits (row_len + unit_cells) .. (2*row_len + unit_cells - 1)
        offset_c = row_len + unit_cells
        for i in range(row_len - 1):
            edges.append((offset_c + i, offset_c + i + 1))

        total = offset_c + row_len
        cm = cls(num_qubits=total)
        for a, b in edges:
            if a < total and b < total:
                cm.add_edge(a, b)
        return cm

    @classmethod
    def all_to_all(cls, n: int) -> "CouplingMap":
        """Fully connected topology."""
        cm = cls(num_qubits=n)
        for i in range(n):
            for j in range(i + 1, n):
                cm.add_edge(i, j)
        return cm

    # -- device presets -------------------------------------------------

    @classmethod
    def ibm_eagle(cls) -> "CouplingMap":
        """IBM Eagle-class processor (~127Q heavy-hex)."""
        return cls.from_heavy_hex(15)

    @classmethod
    def google_sycamore(cls) -> "CouplingMap":
        """Google Sycamore-class processor (54Q grid ~ 6x9)."""
        return cls.from_grid(6, 9)

    @classmethod
    def rigetti_aspen(cls) -> "CouplingMap":
        """Rigetti Aspen-class processor (80Q ring-of-octagon, modeled as 8x10 grid)."""
        return cls.from_grid(8, 10)

    @classmethod
    def linear(cls, n: int) -> "CouplingMap":
        """Convenience alias for :meth:`from_line`."""
        return cls.from_line(n)

    def __repr__(self) -> str:
        return (
            f"CouplingMap(num_qubits={self.num_qubits}, "
            f"edges={self.num_edges()})"
        )
