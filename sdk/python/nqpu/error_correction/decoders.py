"""Quantum error correction decoders.

Implements syndrome decoding algorithms with a uniform interface: given a
syndrome vector, produce a correction in symplectic form.

Decoders:
  - :class:`LookupTableDecoder` -- exhaustive syndrome-to-correction map
    for small codes.
  - :class:`MWPMDecoder` -- minimum weight perfect matching for surface
    codes, using a greedy Blossom-style heuristic (pure numpy, no
    external graph library).
  - :class:`UnionFindDecoder` -- near-linear-time decoder for surface
    codes via weighted union-find.
  - :class:`BPDecoder` -- belief propagation (min-sum) for LDPC / QLDPC
    codes.

All decoders share the interface ``decode(syndrome) -> correction``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from .codes import QuantumCode, SurfaceCode


# ------------------------------------------------------------------ #
# Abstract decoder
# ------------------------------------------------------------------ #

class Decoder(ABC):
    """Abstract base for QEC decoders."""

    @abstractmethod
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a syndrome into a correction vector.

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome vector (length ``mx + mz``).

        Returns
        -------
        np.ndarray
            Correction in symplectic form (length ``2*n``).
        """
        ...


# ------------------------------------------------------------------ #
# Lookup Table Decoder
# ------------------------------------------------------------------ #

class LookupTableDecoder(Decoder):
    """Exhaustive syndrome lookup table decoder.

    Pre-computes the minimum-weight correction for every possible syndrome
    by enumerating all single-qubit Pauli errors (weight 1).  Falls back
    to weight-2 errors if needed (configurable).

    Suitable for small codes (n <= ~15).

    Parameters
    ----------
    code : QuantumCode
        The code to decode.
    max_weight : int
        Maximum error weight to include in the lookup table (default 1).
        Higher values give better decoding but exponentially more entries.
    """

    def __init__(self, code: QuantumCode, max_weight: int = 1) -> None:
        self.code = code
        self.max_weight = max_weight
        self.table: Dict[bytes, np.ndarray] = {}
        self._build_table()

    def _build_table(self) -> None:
        n = self.code.n
        # Identity (trivial syndrome -> no correction)
        trivial = self.code.syndrome(np.zeros(2 * n, dtype=np.int8))
        self.table[trivial.tobytes()] = np.zeros(2 * n, dtype=np.int8)

        for w in range(1, self.max_weight + 1):
            self._add_weight(w)

    def _add_weight(self, weight: int) -> None:
        n = self.code.n
        # For weight-1: iterate over single-qubit X, Z, Y errors
        if weight == 1:
            for q in range(n):
                for pauli in ["X", "Z", "Y"]:
                    err = np.zeros(2 * n, dtype=np.int8)
                    if pauli in ("X", "Y"):
                        err[q] = 1
                    if pauli in ("Z", "Y"):
                        err[n + q] = 1
                    syn = self.code.syndrome(err)
                    key = syn.tobytes()
                    if key not in self.table:
                        self.table[key] = err.copy()
        elif weight == 2:
            # Two-qubit errors
            for q1 in range(n):
                for q2 in range(q1 + 1, n):
                    for p1 in ["X", "Z", "Y"]:
                        for p2 in ["X", "Z", "Y"]:
                            err = np.zeros(2 * n, dtype=np.int8)
                            if p1 in ("X", "Y"):
                                err[q1] = 1
                            if p1 in ("Z", "Y"):
                                err[n + q1] = 1
                            if p2 in ("X", "Y"):
                                err[q2] = 1
                            if p2 in ("Z", "Y"):
                                err[n + q2] = 1
                            syn = self.code.syndrome(err)
                            key = syn.tobytes()
                            if key not in self.table:
                                self.table[key] = err.copy()

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        key = syndrome.astype(np.int8).tobytes()
        if key in self.table:
            return self.table[key].copy()
        # Unknown syndrome -- return identity (no correction)
        return np.zeros(2 * self.code.n, dtype=np.int8)


# ------------------------------------------------------------------ #
# MWPM Decoder (greedy matching)
# ------------------------------------------------------------------ #

class MWPMDecoder(Decoder):
    """Minimum Weight Perfect Matching decoder for surface codes.

    Builds the **syndrome graph** from the code's check matrix: nodes are
    checks (plus a virtual boundary node), edges are data qubits shared
    between checks.  Defects are matched greedily by shortest graph
    distance, and the correction is the symmetric difference of the
    shortest paths between matched pairs.

    Parameters
    ----------
    code : QuantumCode
        Any stabilizer code (works best on surface codes).
    """

    def __init__(self, code: QuantumCode) -> None:
        self.code = code
        self.n = code.n
        # Pre-compute syndrome graphs for X and Z check matrices
        self._x_graph = self._build_syndrome_graph(code.Hx)
        self._z_graph = self._build_syndrome_graph(code.Hz)

    @staticmethod
    def _build_syndrome_graph(
        check_matrix: np.ndarray,
    ) -> Dict[int, Dict[int, List[int]]]:
        """Build syndrome graph: adjacency dict of check_idx -> {neighbor -> [qubits]}.

        Includes a virtual boundary node (index = num_checks) connected
        to every check that has weight < max_weight (boundary checks).
        """
        if check_matrix.shape[0] == 0:
            return {}

        m, n = check_matrix.shape
        boundary = m  # virtual boundary node index

        # Build qubit -> checks mapping
        qubit_to_checks: Dict[int, List[int]] = {}
        for i in range(m):
            for q in np.where(check_matrix[i] == 1)[0]:
                qubit_to_checks.setdefault(int(q), []).append(i)

        # Adjacency: for each pair of checks sharing a qubit, record the qubit
        adj: Dict[int, Dict[int, List[int]]] = {i: {} for i in range(m + 1)}

        for q, checks in qubit_to_checks.items():
            if len(checks) == 2:
                c1, c2 = checks[0], checks[1]
                adj[c1].setdefault(c2, []).append(q)
                adj[c2].setdefault(c1, []).append(q)
            elif len(checks) == 1:
                # Boundary qubit: connects check to boundary node
                c = checks[0]
                adj[c].setdefault(boundary, []).append(q)
                adj[boundary].setdefault(c, []).append(q)

        return adj

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        code = self.code
        mx = code.Hx.shape[0]
        sx = syndrome[:mx]
        sz = syndrome[mx:]

        correction = np.zeros(2 * self.n, dtype=np.int8)

        # Decode X-check defects -> Z correction
        if code.Hx.shape[0] > 0:
            z_corr = self._decode_single_type(sx, code.Hx, self._x_graph)
            correction[self.n:] = z_corr

        # Decode Z-check defects -> X correction
        if code.Hz.shape[0] > 0:
            x_corr = self._decode_single_type(sz, code.Hz, self._z_graph)
            correction[:self.n] = x_corr

        return correction

    def _decode_single_type(
        self,
        syndrome_bits: np.ndarray,
        check_matrix: np.ndarray,
        graph: Dict[int, Dict[int, List[int]]],
    ) -> np.ndarray:
        """Decode one error type using greedy matching on the syndrome graph."""
        n = self.n
        m = check_matrix.shape[0]
        boundary = m

        defects = list(np.where(syndrome_bits == 1)[0])
        correction = np.zeros(n, dtype=np.int8)

        if not defects or not graph:
            return correction

        # Compute all-pairs shortest paths (BFS) between defects + boundary
        # For small/moderate codes this is fast enough.
        nodes_of_interest = set(defects)
        nodes_of_interest.add(boundary)

        # BFS shortest paths
        dist_cache: Dict[int, Dict[int, int]] = {}
        path_cache: Dict[int, Dict[int, List[int]]] = {}

        for src in nodes_of_interest:
            dist, parent = self._bfs(src, graph, m + 1)
            dist_cache[src] = dist
            path_cache[src] = parent

        # Greedy nearest-neighbor matching
        unmatched = list(defects)
        pairs = []

        while len(unmatched) >= 2:
            best_dist = float("inf")
            best_i, best_j = 0, 1

            for i in range(len(unmatched)):
                for j in range(i + 1, len(unmatched)):
                    di, dj = unmatched[i], unmatched[j]
                    d_ij = dist_cache.get(di, {}).get(dj, float("inf"))
                    if d_ij < best_dist:
                        best_dist = d_ij
                        best_i, best_j = i, j

                # Also consider matching to boundary
                d_bnd = dist_cache.get(unmatched[i], {}).get(boundary, float("inf"))
                # Boundary matching costs the path to boundary
                if d_bnd < best_dist and len(unmatched) % 2 == 1:
                    best_dist = d_bnd
                    best_i = i
                    best_j = -1  # boundary

            if best_j == -1:
                # Match to boundary
                di = unmatched.pop(best_i)
                path_qubits = self._reconstruct_path(
                    di, boundary, path_cache.get(di, {}), graph
                )
                for q in path_qubits:
                    correction[q] = (correction[q] + 1) % 2
            else:
                # Match two defects
                if best_i < best_j:
                    dj = unmatched.pop(best_j)
                    di = unmatched.pop(best_i)
                else:
                    di = unmatched.pop(best_i)
                    dj = unmatched.pop(best_j)
                path_qubits = self._reconstruct_path(
                    di, dj, path_cache.get(di, {}), graph
                )
                for q in path_qubits:
                    correction[q] = (correction[q] + 1) % 2
                pairs.append((di, dj))

        # Remaining unmatched defect -> match to boundary
        for di in unmatched:
            path_qubits = self._reconstruct_path(
                di, boundary, path_cache.get(di, {}), graph
            )
            for q in path_qubits:
                correction[q] = (correction[q] + 1) % 2

        return correction

    @staticmethod
    def _bfs(
        src: int,
        graph: Dict[int, Dict[int, List[int]]],
        num_nodes: int,
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """BFS shortest path from src in the syndrome graph."""
        dist = {src: 0}
        parent = {src: -1}
        queue = [src]
        head = 0

        while head < len(queue):
            u = queue[head]
            head += 1
            for v in graph.get(u, {}):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)

        return dist, parent

    @staticmethod
    def _reconstruct_path(
        src: int,
        dst: int,
        parent: Dict[int, int],
        graph: Dict[int, Dict[int, List[int]]],
    ) -> List[int]:
        """Reconstruct path and return the data qubits along it."""
        if dst not in parent:
            return []

        # Reconstruct node path
        path_nodes = []
        cur = dst
        while cur != src and cur != -1:
            path_nodes.append(cur)
            cur = parent.get(cur, -1)
        path_nodes.append(src)
        path_nodes.reverse()

        # Collect one qubit per edge
        qubits = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            edge_qubits = graph.get(u, {}).get(v, [])
            if edge_qubits:
                qubits.append(edge_qubits[0])  # pick first shared qubit

        return qubits


# ------------------------------------------------------------------ #
# Union-Find Decoder
# ------------------------------------------------------------------ #

class UnionFindDecoder(Decoder):
    """Union-Find decoder for surface codes.

    Near-linear-time decoder using a weighted union-find data structure
    to cluster defects and apply peeling corrections.  Based on:

        Delfosse & Nickerson, "Almost-linear time decoding algorithm for
        topological codes", arXiv:1709.06218.

    The algorithm works on the syndrome graph (same as MWPM): nodes are
    checks, edges are data qubits shared by two checks (or boundary).
    Clusters grow iteratively until every cluster has even parity or is
    connected to the boundary, then a peeling pass extracts the correction.

    Parameters
    ----------
    code : QuantumCode
        The code to decode.
    """

    def __init__(self, code: QuantumCode) -> None:
        self.code = code
        self.n = code.n

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        code = self.code
        mx = code.Hx.shape[0]
        sx = syndrome[:mx]
        sz = syndrome[mx:]

        correction = np.zeros(2 * self.n, dtype=np.int8)

        if code.Hx.shape[0] > 0:
            z_corr = self._uf_decode(sx, code.Hx)
            correction[self.n:] = z_corr

        if code.Hz.shape[0] > 0:
            x_corr = self._uf_decode(sz, code.Hz)
            correction[:self.n] = x_corr

        return correction

    def _uf_decode(
        self, syndrome_bits: np.ndarray, check_matrix: np.ndarray
    ) -> np.ndarray:
        """Union-Find decoding for one error type using the syndrome graph."""
        n = self.n
        m = check_matrix.shape[0]
        boundary = m  # virtual boundary node

        defects = set(int(i) for i in np.where(syndrome_bits == 1)[0])
        if not defects:
            return np.zeros(n, dtype=np.int8)

        # Build syndrome graph edges: (check_a, check_b, qubit)
        qubit_to_checks: Dict[int, List[int]] = {}
        for i in range(m):
            for q in np.where(check_matrix[i] == 1)[0]:
                qubit_to_checks.setdefault(int(q), []).append(i)

        edges = []  # (node_a, node_b, qubit)
        for q, checks in qubit_to_checks.items():
            if len(checks) == 2:
                edges.append((checks[0], checks[1], q))
            elif len(checks) == 1:
                edges.append((checks[0], boundary, q))

        # Union-Find data structure
        parent = list(range(m + 1))
        uf_rank = [0] * (m + 1)
        cluster_size = [0] * (m + 1)  # growth radius
        cluster_defects = [0] * (m + 1)
        cluster_has_boundary = [False] * (m + 1)

        for d_idx in defects:
            cluster_defects[d_idx] = 1

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(a: int, b: int) -> int:
            ra, rb = find(a), find(b)
            if ra == rb:
                return ra
            if uf_rank[ra] < uf_rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if uf_rank[ra] == uf_rank[rb]:
                uf_rank[ra] += 1
            cluster_defects[ra] += cluster_defects[rb]
            cluster_size[ra] = max(cluster_size[ra], cluster_size[rb])
            cluster_has_boundary[ra] = (
                cluster_has_boundary[ra] or cluster_has_boundary[rb]
            )
            return ra

        # Iterative growth: increase cluster radii until all clusters
        # are neutralized (even defect count or touching boundary).
        max_growth = n  # upper bound on growth rounds
        used_edges: List[int] = []

        for growth_round in range(max_growth):
            # Check if all defect clusters are neutralized
            any_odd = False
            for d_idx in defects:
                root = find(d_idx)
                if cluster_defects[root] % 2 == 1 and not cluster_has_boundary[root]:
                    any_odd = True
                    break

            if not any_odd:
                break

            # Grow all odd-parity clusters by one
            for d_idx in defects:
                root = find(d_idx)
                if cluster_defects[root] % 2 == 1:
                    cluster_size[root] += 1

            # Try to fuse clusters along edges
            for c1, c2, q in edges:
                r1 = find(c1)
                if c2 == boundary:
                    if cluster_size[r1] > 0 and cluster_defects[r1] % 2 == 1:
                        cluster_has_boundary[r1] = True
                        used_edges.append(q)
                    continue

                r2 = find(c2)
                if r1 == r2:
                    continue

                # Fuse if either cluster has grown enough (size > 0)
                if cluster_size[r1] > 0 or cluster_size[r2] > 0:
                    root = union(c1, c2)
                    used_edges.append(q)

        # Peeling phase: build a spanning forest from used edges and peel
        # to get minimum-weight correction.
        # Simple approach: the used_edges form a connected subgraph;
        # we need to find which subset gives the right syndrome.
        # Use the standard peeling: process leaf nodes of the subgraph.
        correction = self._peel(used_edges, qubit_to_checks, defects, m, n)

        return correction

    def _peel(
        self,
        used_edges: List[int],
        qubit_to_checks: Dict[int, List[int]],
        defects: set,
        m: int,
        n: int,
    ) -> np.ndarray:
        """Peeling decoder on the spanning forest of used edges.

        Process leaves of the subgraph induced by used_edges.  If a leaf
        check is a defect, include the edge qubit in the correction and
        toggle the defect status of the neighbor.
        """
        boundary = m
        correction = np.zeros(n, dtype=np.int8)

        # Build adjacency for the subgraph of used edges
        adj: Dict[int, List[Tuple[int, int]]] = {}  # node -> [(neighbor, qubit)]
        for q in used_edges:
            checks = qubit_to_checks.get(q, [])
            if len(checks) == 2:
                c1, c2 = checks[0], checks[1]
            elif len(checks) == 1:
                c1, c2 = checks[0], boundary
            else:
                continue
            adj.setdefault(c1, []).append((c2, q))
            adj.setdefault(c2, []).append((c1, q))

        # Track which edges are still active
        active_edges: Dict[int, set] = {}
        for node in adj:
            active_edges[node] = set()
            for neighbor, q in adj[node]:
                active_edges[node].add((neighbor, q))

        defect_status = {i: 1 for i in defects}
        remaining_nodes = set(adj.keys())

        # Iteratively peel leaves
        changed = True
        while changed:
            changed = False
            leaves = []
            for node in list(remaining_nodes):
                if node in active_edges and len(active_edges[node]) == 1:
                    leaves.append(node)

            for leaf in leaves:
                if leaf not in active_edges or not active_edges[leaf]:
                    continue

                neighbor, q = next(iter(active_edges[leaf]))

                # If leaf is a defect (or has odd syndrome), include edge
                if defect_status.get(leaf, 0) % 2 == 1:
                    correction[q] = (correction[q] + 1) % 2
                    # Toggle neighbor's defect status
                    defect_status[neighbor] = defect_status.get(neighbor, 0) + 1

                # Remove edge from both endpoints
                active_edges[leaf].discard((neighbor, q))
                if neighbor in active_edges:
                    active_edges[neighbor].discard((leaf, q))

                remaining_nodes.discard(leaf)
                changed = True

        return correction


# ------------------------------------------------------------------ #
# Belief Propagation Decoder
# ------------------------------------------------------------------ #

class BPDecoder(Decoder):
    """Belief Propagation (min-sum) decoder for LDPC / QLDPC codes.

    Iterative message-passing decoder on the Tanner graph of the code's
    check matrix.  Uses the min-sum variant for numerical stability.

    Parameters
    ----------
    code : QuantumCode
        The code to decode.
    max_iterations : int
        Maximum number of BP iterations (default 50).
    channel_error_rate : float
        Prior probability of a single-qubit error (default 0.05).
    damping : float
        Message damping factor in [0, 1] for convergence (default 0.5).
    """

    def __init__(
        self,
        code: QuantumCode,
        max_iterations: int = 50,
        channel_error_rate: float = 0.05,
        damping: float = 0.5,
    ) -> None:
        self.code = code
        self.n = code.n
        self.max_iterations = max_iterations
        self.p = channel_error_rate
        self.damping = damping

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        code = self.code
        mx = code.Hx.shape[0]
        sx = syndrome[:mx]
        sz = syndrome[mx:]

        correction = np.zeros(2 * self.n, dtype=np.int8)

        # Decode Z errors from X syndrome
        if code.Hx.shape[0] > 0:
            z_corr = self._bp_decode(sx, code.Hx)
            correction[self.n:] = z_corr

        # Decode X errors from Z syndrome
        if code.Hz.shape[0] > 0:
            x_corr = self._bp_decode(sz, code.Hz)
            correction[:self.n] = x_corr

        return correction

    def _bp_decode(
        self, syndrome_bits: np.ndarray, check_matrix: np.ndarray
    ) -> np.ndarray:
        """Run min-sum BP on one check matrix."""
        m, n = check_matrix.shape
        p = self.p

        # Log-likelihood ratios
        eps = 1e-15
        channel_llr = np.log((1 - p) / (p + eps))

        # Initialize variable-to-check messages
        v2c = np.zeros((m, n), dtype=np.float64)
        for j in range(n):
            v2c[:, j] = channel_llr

        c2v = np.zeros((m, n), dtype=np.float64)

        for iteration in range(self.max_iterations):
            # Check-to-variable messages (min-sum)
            new_c2v = np.zeros((m, n), dtype=np.float64)
            for i in range(m):
                neighbors = np.where(check_matrix[i] == 1)[0]
                s = 1 if syndrome_bits[i] == 0 else -1

                for j in neighbors:
                    other = [k for k in neighbors if k != j]
                    if not other:
                        new_c2v[i, j] = 0.0
                        continue

                    # Product of signs
                    sign = s
                    for k in other:
                        sign *= (1 if v2c[i, k] >= 0 else -1)

                    # Min of absolutes
                    min_abs = min(abs(v2c[i, k]) for k in other)
                    new_c2v[i, j] = sign * min_abs

            # Damping
            c2v = self.damping * new_c2v + (1 - self.damping) * c2v

            # Variable-to-check messages
            new_v2c = np.zeros((m, n), dtype=np.float64)
            for j in range(n):
                checks = np.where(check_matrix[:, j] == 1)[0]
                total = channel_llr + np.sum(c2v[checks, j])

                for i in checks:
                    new_v2c[i, j] = total - c2v[i, j]

            v2c = new_v2c

            # Hard decision
            beliefs = np.full(n, channel_llr, dtype=np.float64)
            for j in range(n):
                checks = np.where(check_matrix[:, j] == 1)[0]
                beliefs[j] += np.sum(c2v[checks, j])

            hard = (beliefs < 0).astype(np.int8)

            # Check convergence
            check_syndrome = (check_matrix @ hard) % 2
            if np.array_equal(check_syndrome, syndrome_bits.astype(np.int8)):
                return hard

        # Return best guess even if not converged
        beliefs = np.full(n, channel_llr, dtype=np.float64)
        for j in range(n):
            checks = np.where(check_matrix[:, j] == 1)[0]
            beliefs[j] += np.sum(c2v[checks, j])
        return (beliefs < 0).astype(np.int8)


# ------------------------------------------------------------------ #
# Decoder Benchmarking
# ------------------------------------------------------------------ #

@dataclass
class DecoderBenchmark:
    """Results of a decoder benchmark run.

    Attributes
    ----------
    logical_error_rate : float
        Fraction of trials where the decoder failed.
    num_trials : int
        Number of Monte Carlo trials.
    num_failures : int
        Number of decoding failures.
    physical_error_rate : float
        Physical error rate used.
    code_params : tuple
        ``(n, k, d)`` of the code.
    decoder_name : str
        Name of the decoder class.
    """

    logical_error_rate: float
    num_trials: int
    num_failures: int
    physical_error_rate: float
    code_params: Tuple[int, int, int]
    decoder_name: str


def benchmark_decoder(
    code: QuantumCode,
    decoder: Decoder,
    physical_error_rate: float,
    num_trials: int = 1000,
    error_type: str = "depolarizing",
    seed: Optional[int] = None,
) -> DecoderBenchmark:
    """Benchmark a decoder via Monte Carlo simulation.

    Parameters
    ----------
    code : QuantumCode
        The code under test.
    decoder : Decoder
        The decoder to benchmark.
    physical_error_rate : float
        Per-qubit physical error probability.
    num_trials : int
        Number of Monte Carlo trials.
    error_type : str
        ``"depolarizing"`` (X/Y/Z with equal probability p/3 each),
        ``"x_only"`` (X errors with probability p), or
        ``"z_only"`` (Z errors with probability p).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    DecoderBenchmark
        Benchmark results.
    """
    rng = np.random.default_rng(seed)
    n = code.n
    failures = 0

    for _ in range(num_trials):
        error = np.zeros(2 * n, dtype=np.int8)

        if error_type == "depolarizing":
            p_each = physical_error_rate / 3.0
            for q in range(n):
                r = rng.random()
                if r < p_each:
                    error[q] = 1  # X
                elif r < 2 * p_each:
                    error[n + q] = 1  # Z
                elif r < 3 * p_each:
                    error[q] = 1  # Y = XZ
                    error[n + q] = 1
        elif error_type == "x_only":
            for q in range(n):
                if rng.random() < physical_error_rate:
                    error[q] = 1
        elif error_type == "z_only":
            for q in range(n):
                if rng.random() < physical_error_rate:
                    error[n + q] = 1
        else:
            raise ValueError(f"Unknown error_type: {error_type}")

        syndrome = code.syndrome(error)

        if not np.any(syndrome):
            # No error detected -- check if a logical error occurred anyway
            if not code.check_correction(error, np.zeros(2 * n, dtype=np.int8)):
                failures += 1
            continue

        correction = decoder.decode(syndrome)

        if not code.check_correction(error, correction):
            failures += 1

    logical_error_rate = failures / num_trials
    return DecoderBenchmark(
        logical_error_rate=logical_error_rate,
        num_trials=num_trials,
        num_failures=failures,
        physical_error_rate=physical_error_rate,
        code_params=code.code_params,
        decoder_name=type(decoder).__name__,
    )
