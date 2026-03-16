"""Combinatorial Optimization -- QUBO formulations and classical solvers.

Provides graph-based combinatorial optimization problems with:
  - QUBO (Quadratic Unconstrained Binary Optimization) formulation
  - Exact brute-force solvers for small instances
  - Heuristic solvers (simulated annealing, 2-opt, greedy + local search)
  - QAOA-inspired classical variational simulation

These problems are the canonical targets for near-term quantum optimization
algorithms (QAOA, VQE, quantum annealing).

References:
    Farhi, Goldstone, Gutmann (2014) - QAOA
    Lucas (2014) - Ising formulations of many NP problems
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Simple weighted graph
# ---------------------------------------------------------------------------

class Graph:
    """Weighted undirected graph for combinatorial problems.

    Vertices are integers 0..n-1.  Edges are stored as an adjacency dict
    with weights.
    """

    def __init__(self, n_vertices: int) -> None:
        self.n = n_vertices
        self.edges: Dict[Tuple[int, int], float] = {}
        self._adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_vertices)}

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        if u == v:
            raise ValueError(f"Self-loops not supported: ({u}, {v})")
        key = (min(u, v), max(u, v))
        self.edges[key] = weight
        self._adj[u].append((v, weight))
        self._adj[v].append((u, weight))

    def neighbors(self, v: int) -> List[Tuple[int, float]]:
        return self._adj[v]

    def weight(self, u: int, v: int) -> float:
        key = (min(u, v), max(u, v))
        return self.edges.get(key, 0.0)

    @staticmethod
    def random_graph(n: int, edge_prob: float = 0.5, seed: int = 42) -> "Graph":
        """Generate a random Erdos-Renyi graph with uniform [0,1] weights."""
        rng = np.random.default_rng(seed)
        g = Graph(n)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < edge_prob:
                    g.add_edge(i, j, weight=float(rng.random()))
        return g

    @staticmethod
    def complete_graph(n: int, seed: int = 42) -> "Graph":
        """Complete graph with random weights (for TSP instances)."""
        rng = np.random.default_rng(seed)
        g = Graph(n)
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(i, j, weight=float(rng.uniform(1.0, 10.0)))
        return g


# ---------------------------------------------------------------------------
# Solver results
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result from a combinatorial optimization solver."""

    solution: np.ndarray  # binary vector or permutation
    objective: float
    method: str
    iterations: int = 0
    history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MaxCut
# ---------------------------------------------------------------------------

class MaxCut:
    """Maximum Cut on a weighted graph.

    Given a graph G=(V,E), partition V into two sets S and S_bar to maximize
    the total weight of edges crossing the partition.

    QUBO formulation: maximize sum_{(i,j) in E} w_ij * x_i * (1 - x_j)
    where x_i in {0, 1} indicates set membership.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def qubo_matrix(self) -> np.ndarray:
        """Return the QUBO matrix Q such that x^T Q x gives the negative cut value.

        We minimize x^T Q x, so Q encodes -MaxCut.
        """
        n = self.graph.n
        q = np.zeros((n, n))
        for (u, v), w in self.graph.edges.items():
            # Contribution: w * x_u * (1 - x_v) + w * x_v * (1 - x_u)
            # = w * (x_u + x_v - 2 * x_u * x_v)
            # For minimization (negate): -w * x_u - w * x_v + 2w * x_u * x_v
            q[u, u] -= w
            q[v, v] -= w
            q[u, v] += w  # Off-diagonal stores 2w split as w + w
            q[v, u] += w
        return q

    def evaluate(self, assignment: np.ndarray) -> float:
        """Compute the cut value for a binary assignment."""
        total = 0.0
        for (u, v), w in self.graph.edges.items():
            if assignment[u] != assignment[v]:
                total += w
        return total

    def brute_force(self) -> OptimizationResult:
        """Exact solver: enumerate all 2^n partitions.

        Only practical for n <= 20 or so.
        """
        n = self.graph.n
        best_val = -1.0
        best_assignment = np.zeros(n, dtype=int)
        for bits in range(1 << n):
            assignment = np.array([(bits >> i) & 1 for i in range(n)], dtype=int)
            val = self.evaluate(assignment)
            if val > best_val:
                best_val = val
                best_assignment = assignment.copy()
        return OptimizationResult(
            solution=best_assignment,
            objective=best_val,
            method="brute_force",
        )

    def simulated_annealing(
        self,
        n_iterations: int = 10000,
        temp_start: float = 10.0,
        temp_end: float = 0.01,
        seed: int = 42,
    ) -> OptimizationResult:
        """Simulated annealing heuristic for MaxCut."""
        rng = np.random.default_rng(seed)
        n = self.graph.n
        current = rng.integers(0, 2, size=n)
        current_val = self.evaluate(current)
        best = current.copy()
        best_val = current_val
        history = [current_val]

        for step in range(n_iterations):
            t = temp_start * (temp_end / temp_start) ** (step / max(n_iterations - 1, 1))
            flip = rng.integers(0, n)
            candidate = current.copy()
            candidate[flip] = 1 - candidate[flip]
            candidate_val = self.evaluate(candidate)
            delta = candidate_val - current_val
            if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-15)):
                current = candidate
                current_val = candidate_val
                if current_val > best_val:
                    best_val = current_val
                    best = current.copy()
            if step % 100 == 0:
                history.append(best_val)

        return OptimizationResult(
            solution=best,
            objective=best_val,
            method="simulated_annealing",
            iterations=n_iterations,
            history=history,
        )

    def qaoa_inspired(
        self,
        p: int = 3,
        n_iterations: int = 200,
        seed: int = 42,
    ) -> OptimizationResult:
        """QAOA-inspired classical variational solver.

        Simulates a simplified p-layer QAOA by classically evaluating
        the expected cut value for parametrized gamma/beta angles and
        optimizing them with coordinate descent.

        This is a classical heuristic inspired by the QAOA circuit structure,
        not a full quantum simulation.
        """
        rng = np.random.default_rng(seed)
        n = self.graph.n

        # Initialize variational parameters
        gammas = rng.uniform(0, np.pi, size=p)
        betas = rng.uniform(0, np.pi / 2, size=p)

        def _evaluate_qaoa_angles(gammas_: np.ndarray, betas_: np.ndarray) -> float:
            """Approximate expected cut via classical sampling."""
            # Use angle-biased random sampling to approximate the QAOA output
            total = 0.0
            n_samples = 100
            for _ in range(n_samples):
                # Generate biased assignment from angles
                probs = np.full(n, 0.5)
                for layer in range(p):
                    # Problem unitary phase: bias toward high-cut vertices
                    for v in range(n):
                        degree_w = sum(w for _, w in self.graph.neighbors(v))
                        probs[v] = 0.5 + 0.5 * np.sin(2 * gammas_[layer] * degree_w / max(n, 1))
                    # Mixer rotation
                    probs = probs * np.cos(betas_[layer]) ** 2 + (1 - probs) * np.sin(betas_[layer]) ** 2
                probs = np.clip(probs, 0.0, 1.0)
                sample = (rng.random(n) < probs).astype(int)
                total += self.evaluate(sample)
            return total / n_samples

        best_val = 0.0
        best_assignment = np.zeros(n, dtype=int)
        history = []

        for iteration in range(n_iterations):
            # Coordinate descent on one parameter
            idx = iteration % (2 * p)
            param_type = "gamma" if idx < p else "beta"
            param_idx = idx if idx < p else idx - p

            best_param_val = 0.0
            best_param = gammas[param_idx] if param_type == "gamma" else betas[param_idx]

            for trial_val in np.linspace(0, np.pi if param_type == "gamma" else np.pi / 2, 20):
                if param_type == "gamma":
                    gammas[param_idx] = trial_val
                else:
                    betas[param_idx] = trial_val
                ev = _evaluate_qaoa_angles(gammas, betas)
                if ev > best_param_val:
                    best_param_val = ev
                    best_param = trial_val

            if param_type == "gamma":
                gammas[param_idx] = best_param
            else:
                betas[param_idx] = best_param

            if best_param_val > best_val:
                best_val = best_param_val
            if iteration % 10 == 0:
                history.append(best_val)

        # Final sampling with optimized parameters
        best_val_final = 0.0
        for _ in range(500):
            probs = np.full(n, 0.5)
            for layer in range(p):
                for v in range(n):
                    degree_w = sum(w for _, w in self.graph.neighbors(v))
                    probs[v] = 0.5 + 0.5 * np.sin(2 * gammas[layer] * degree_w / max(n, 1))
                probs = probs * np.cos(betas[layer]) ** 2 + (1 - probs) * np.sin(betas[layer]) ** 2
            probs = np.clip(probs, 0.0, 1.0)
            sample = (rng.random(n) < probs).astype(int)
            val = self.evaluate(sample)
            if val > best_val_final:
                best_val_final = val
                best_assignment = sample.copy()

        return OptimizationResult(
            solution=best_assignment,
            objective=best_val_final,
            method="qaoa_inspired",
            iterations=n_iterations,
            history=history,
        )


# ---------------------------------------------------------------------------
# Graph Coloring
# ---------------------------------------------------------------------------

class GraphColoring:
    """Vertex coloring with minimum colors.

    QUBO penalty formulation:
        minimize sum_{(u,v) in E} penalty * delta(color_u, color_v)
                 + sum_v penalty * (1 - sum_c x_{v,c})^2
    where x_{v,c} = 1 if vertex v has color c.
    """

    def __init__(self, graph: Graph, n_colors: int) -> None:
        self.graph = graph
        self.n_colors = n_colors

    def qubo_matrix(self, penalty: float = 10.0) -> np.ndarray:
        """QUBO matrix encoding the coloring constraints.

        Variables: x_{v,c} for v in V, c in 0..n_colors-1.
        Index mapping: variable index = v * n_colors + c.
        """
        n = self.graph.n
        k = self.n_colors
        size = n * k
        q = np.zeros((size, size))

        # Constraint 1: each vertex gets exactly one color
        for v in range(n):
            for c1 in range(k):
                idx1 = v * k + c1
                q[idx1, idx1] -= penalty  # linear term from (1 - sum x)^2
                for c2 in range(c1 + 1, k):
                    idx2 = v * k + c2
                    q[idx1, idx2] += 2 * penalty
                    q[idx2, idx1] += 2 * penalty

        # Constraint 2: adjacent vertices different colors
        for (u, v), _w in self.graph.edges.items():
            for c in range(k):
                idx_u = u * k + c
                idx_v = v * k + c
                q[idx_u, idx_v] += penalty
                q[idx_v, idx_u] += penalty

        return q

    def evaluate(self, coloring: np.ndarray) -> Tuple[bool, int]:
        """Check if a coloring is valid and count conflicts.

        Parameters
        ----------
        coloring : array of int, length n_vertices
            Color assignment for each vertex.

        Returns
        -------
        valid : bool
        n_conflicts : int
        """
        conflicts = 0
        for (u, v), _w in self.graph.edges.items():
            if coloring[u] == coloring[v]:
                conflicts += 1
        return conflicts == 0, conflicts

    def greedy(self, order: Optional[np.ndarray] = None) -> OptimizationResult:
        """Greedy coloring with optional vertex ordering."""
        n = self.graph.n
        if order is None:
            # Degree-based ordering (largest degree first)
            degrees = [len(self.graph.neighbors(v)) for v in range(n)]
            order = np.argsort(degrees)[::-1]
        coloring = np.full(n, -1, dtype=int)
        for v in order:
            used = set()
            for u, _w in self.graph.neighbors(v):
                if coloring[u] >= 0:
                    used.add(coloring[u])
            for c in range(self.n_colors):
                if c not in used:
                    coloring[v] = c
                    break
            if coloring[v] < 0:
                coloring[v] = 0  # fallback (will cause conflicts)

        valid, conflicts = self.evaluate(coloring)
        return OptimizationResult(
            solution=coloring,
            objective=float(conflicts),
            method="greedy",
        )

    def greedy_local_search(
        self,
        n_iterations: int = 1000,
        seed: int = 42,
    ) -> OptimizationResult:
        """Greedy initialization followed by local search to reduce conflicts."""
        rng = np.random.default_rng(seed)
        result = self.greedy()
        coloring = result.solution.copy()
        n = self.graph.n
        _, best_conflicts = self.evaluate(coloring)
        history = [float(best_conflicts)]

        for step in range(n_iterations):
            if best_conflicts == 0:
                break
            # Pick a conflicting vertex and try recoloring
            conflicting = []
            for (u, v), _w in self.graph.edges.items():
                if coloring[u] == coloring[v]:
                    conflicting.extend([u, v])
            if not conflicting:
                break
            v = rng.choice(conflicting)
            # Try each color, pick the one with fewest conflicts
            best_color = coloring[v]
            best_c_conflicts = best_conflicts
            for c in range(self.n_colors):
                old = coloring[v]
                coloring[v] = c
                _, new_conflicts = self.evaluate(coloring)
                if new_conflicts < best_c_conflicts:
                    best_c_conflicts = new_conflicts
                    best_color = c
                coloring[v] = old
            coloring[v] = best_color
            best_conflicts = best_c_conflicts
            if step % 50 == 0:
                history.append(float(best_conflicts))

        valid, conflicts = self.evaluate(coloring)
        return OptimizationResult(
            solution=coloring,
            objective=float(conflicts),
            method="greedy_local_search",
            iterations=n_iterations,
            history=history,
        )


# ---------------------------------------------------------------------------
# Traveling Salesman
# ---------------------------------------------------------------------------

class TravelingSalesman:
    """Traveling Salesman Problem on a weighted complete graph.

    QUBO formulation (Karp reduction): minimize tour length subject to:
      - each city visited exactly once
      - each position in tour used exactly once
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def evaluate_tour(self, tour: np.ndarray) -> float:
        """Compute the total tour length (including return to start)."""
        n = len(tour)
        total = 0.0
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            total += self.graph.weight(u, v)
        return total

    def qubo_matrix(self, penalty: float = 100.0) -> np.ndarray:
        """QUBO matrix for TSP.

        Variables: x_{v,p} = 1 if city v is at position p in tour.
        Index: v * n + p.
        """
        n = self.graph.n
        size = n * n
        q = np.zeros((size, size))

        # Objective: tour length
        for (u, v), w in self.graph.edges.items():
            for p in range(n):
                p_next = (p + 1) % n
                idx_u_p = u * n + p
                idx_v_pn = v * n + p_next
                q[idx_u_p, idx_v_pn] += w
                # Also reverse direction
                idx_v_p = v * n + p
                idx_u_pn = u * n + p_next
                q[idx_v_p, idx_u_pn] += w

        # Constraint 1: each city visited exactly once
        for v in range(n):
            for p1 in range(n):
                idx1 = v * n + p1
                q[idx1, idx1] -= penalty
                for p2 in range(p1 + 1, n):
                    idx2 = v * n + p2
                    q[idx1, idx2] += 2 * penalty
                    q[idx2, idx1] += 2 * penalty

        # Constraint 2: each position used exactly once
        for p in range(n):
            for v1 in range(n):
                idx1 = v1 * n + p
                q[idx1, idx1] -= penalty
                for v2 in range(v1 + 1, n):
                    idx2 = v2 * n + p
                    q[idx1, idx2] += 2 * penalty
                    q[idx2, idx1] += 2 * penalty

        return q

    def random_tour(self, seed: int = 42) -> np.ndarray:
        """Generate a random tour."""
        rng = np.random.default_rng(seed)
        tour = np.arange(self.graph.n)
        rng.shuffle(tour)
        return tour

    def two_opt(
        self,
        initial_tour: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
        seed: int = 42,
    ) -> OptimizationResult:
        """2-opt local search for TSP improvement."""
        if initial_tour is None:
            initial_tour = self.random_tour(seed)
        tour = initial_tour.copy()
        n = len(tour)
        best_length = self.evaluate_tour(tour)
        history = [best_length]
        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Reverse segment [i, j]
                    new_tour = tour.copy()
                    new_tour[i:j + 1] = new_tour[i:j + 1][::-1]
                    new_length = self.evaluate_tour(new_tour)
                    if new_length < best_length - 1e-10:
                        tour = new_tour
                        best_length = new_length
                        improved = True
                        break
                if improved:
                    break
            history.append(best_length)

        return OptimizationResult(
            solution=tour,
            objective=best_length,
            method="two_opt",
            iterations=iterations,
            history=history,
        )

    def brute_force(self) -> OptimizationResult:
        """Exact solver for small instances (n! complexity)."""
        n = self.graph.n
        if n > 10:
            raise ValueError(f"Brute-force TSP is impractical for n={n} (max 10)")
        from itertools import permutations

        best_length = float("inf")
        best_tour = np.arange(n)
        # Fix city 0 at start to avoid rotational symmetry
        for perm in permutations(range(1, n)):
            tour = np.array([0] + list(perm))
            length = self.evaluate_tour(tour)
            if length < best_length:
                best_length = length
                best_tour = tour.copy()
        return OptimizationResult(
            solution=best_tour,
            objective=best_length,
            method="brute_force",
        )

    @staticmethod
    def random_instance(n: int, seed: int = 42) -> "TravelingSalesman":
        """Generate a random TSP instance on n cities."""
        return TravelingSalesman(Graph.complete_graph(n, seed=seed))


# ---------------------------------------------------------------------------
# Number Partition
# ---------------------------------------------------------------------------

class NumberPartition:
    """Partition a set of numbers into two subsets of equal sum.

    QUBO formulation: minimize (sum_i s_i * (2*x_i - 1))^2
    where x_i in {0,1} assigns number i to subset 0 or 1.
    """

    def __init__(self, numbers: np.ndarray) -> None:
        self.numbers = np.asarray(numbers, dtype=np.float64)

    def qubo_matrix(self) -> np.ndarray:
        """QUBO matrix for the partition problem."""
        n = len(self.numbers)
        # Expand (sum s_i (2x_i - 1))^2
        # = (sum 2*s_i*x_i - sum s_i)^2
        # = 4 * (sum s_i*x_i)^2 - 4*S*(sum s_i*x_i) + S^2
        # where S = sum s_i
        s = self.numbers
        q = np.zeros((n, n))
        total = np.sum(s)
        for i in range(n):
            q[i, i] = 4 * s[i] ** 2 - 4 * total * s[i]
            for j in range(i + 1, n):
                q[i, j] = 8 * s[i] * s[j]
                q[j, i] = 0  # keep upper triangular for QUBO convention
        return q

    def evaluate(self, assignment: np.ndarray) -> float:
        """Return the absolute difference between subset sums."""
        signed = 2.0 * assignment - 1.0
        return float(abs(np.sum(self.numbers * signed)))

    def brute_force(self) -> OptimizationResult:
        """Exact solver: enumerate all 2^n partitions."""
        n = len(self.numbers)
        best_diff = float("inf")
        best_assignment = np.zeros(n, dtype=int)
        for bits in range(1 << n):
            assignment = np.array([(bits >> i) & 1 for i in range(n)], dtype=int)
            diff = self.evaluate(assignment)
            if diff < best_diff:
                best_diff = diff
                best_assignment = assignment.copy()
        return OptimizationResult(
            solution=best_assignment,
            objective=best_diff,
            method="brute_force",
        )

    def simulated_annealing(
        self,
        n_iterations: int = 5000,
        seed: int = 42,
    ) -> OptimizationResult:
        """SA heuristic for number partition."""
        rng = np.random.default_rng(seed)
        n = len(self.numbers)
        current = rng.integers(0, 2, size=n)
        current_val = self.evaluate(current)
        best = current.copy()
        best_val = current_val
        history = [current_val]

        for step in range(n_iterations):
            t = 10.0 * (0.001 / 10.0) ** (step / max(n_iterations - 1, 1))
            flip = rng.integers(0, n)
            candidate = current.copy()
            candidate[flip] = 1 - candidate[flip]
            candidate_val = self.evaluate(candidate)
            delta = current_val - candidate_val  # minimize
            if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-15)):
                current = candidate
                current_val = candidate_val
                if current_val < best_val:
                    best_val = current_val
                    best = current.copy()
            if step % 100 == 0:
                history.append(best_val)

        return OptimizationResult(
            solution=best,
            objective=best_val,
            method="simulated_annealing",
            iterations=n_iterations,
            history=history,
        )
