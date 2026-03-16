"""QAOA Circuit Builder -- construct QAOA circuits from combinatorial problems.

Builds and simulates Quantum Approximate Optimization Algorithm (QAOA)
circuits for various combinatorial optimization problems. Unlike the
combinatorial module's QAOA-inspired heuristic, this module performs
full statevector simulation of the QAOA quantum circuit.

The QAOA circuit alternates between:
  - Cost operator: exp(-i * gamma * C) where C encodes the objective
  - Mixer operator: exp(-i * beta * B) where B = sum_i X_i

Starting from the uniform superposition |+>^n, QAOA with depth p uses
parameters (gamma_1, ..., gamma_p, beta_1, ..., beta_p).

References:
    Farhi, Goldstone, Gutmann (2014) - QAOA
    Zhou et al. (2020) - QAOA for Constrained Optimization
    Hadfield et al. (2019) - From QAOA to the QAO Ansatz
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# QAOA result types
# ---------------------------------------------------------------------------

@dataclass
class QAOAResult:
    """Result of a QAOA circuit evaluation."""

    expectation: float
    state: np.ndarray
    probabilities: np.ndarray
    best_bitstring: np.ndarray
    best_cost: float


@dataclass
class QAOAOptResult:
    """Result of QAOA parameter optimization."""

    best_gammas: np.ndarray
    best_betas: np.ndarray
    best_expectation: float
    best_bitstring: np.ndarray
    optimization_history: List[float]


# ---------------------------------------------------------------------------
# QAOA Circuit
# ---------------------------------------------------------------------------

class QAOACircuit:
    """QAOA circuit for combinatorial optimization.

    The cost function is specified as a sum of diagonal terms:
        C = sum_k coeff_k * prod_{i in qubits_k} Z_i

    where each term acts on a subset of qubits with a coefficient.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    cost_terms : list of (qubits, coefficient)
        Each term is a tuple of (list of qubit indices, float coefficient).
        The term applies a Z gate on each specified qubit.
    p : int
        QAOA depth (number of alternating layers).
    """

    def __init__(
        self,
        n_qubits: int,
        cost_terms: List[Tuple[list, float]],
        p: int = 1,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("Need at least 1 qubit")
        if n_qubits > 20:
            raise ValueError("Maximum 20 qubits for statevector simulation")
        if p < 1:
            raise ValueError("QAOA depth p must be >= 1")
        self.n_qubits = n_qubits
        self.cost_terms = list(cost_terms)
        self.p = p
        self._dim = 1 << n_qubits

        # Precompute the diagonal cost function values
        self._cost_diagonal = self._build_cost_diagonal()

    def _build_cost_diagonal(self) -> np.ndarray:
        """Precompute diagonal of the cost Hamiltonian.

        For each computational basis state |z>, compute C(z) = sum of
        all cost terms evaluated at the bitstring z.
        """
        dim = self._dim
        diag = np.zeros(dim, dtype=np.float64)

        for qubits, coeff in self.cost_terms:
            for idx in range(dim):
                # Compute product of Z eigenvalues (+1 for |0>, -1 for |1>)
                z_product = 1.0
                for q in qubits:
                    bit = (idx >> q) & 1
                    z_product *= (1 - 2 * bit)  # +1 for 0, -1 for 1
                diag[idx] += coeff * z_product

        return diag

    def cost_operator(self, gamma: float, state: np.ndarray) -> np.ndarray:
        """Apply exp(-i * gamma * C) to state.

        Since C is diagonal, this is element-wise multiplication:
            |psi'> = exp(-i * gamma * C_diag) * |psi>
        """
        phases = np.exp(-1j * gamma * self._cost_diagonal)
        return phases * state

    def mixer_operator(self, beta: float, state: np.ndarray) -> np.ndarray:
        """Apply exp(-i * beta * B) where B = sum X_i.

        The mixer is a product of single-qubit X rotations:
            exp(-i * beta * sum X_i) = prod_i exp(-i * beta * X_i)
                                     = prod_i [cos(beta)*I - i*sin(beta)*X_i]
        """
        n = self.n_qubits
        result = state.copy()

        cb = np.cos(beta)
        sb = np.sin(beta)

        for q in range(n):
            # Apply exp(-i*beta*X) on qubit q
            # This maps |0> -> cos(beta)|0> - i*sin(beta)|1>
            #           |1> -> -i*sin(beta)|0> + cos(beta)|1>
            new_state = np.zeros_like(result)
            stride = 1 << q

            for idx in range(self._dim):
                bit = (idx >> q) & 1
                partner = idx ^ stride  # flip qubit q

                if bit == 0:
                    new_state[idx] += cb * result[idx] - 1j * sb * result[partner]
                else:
                    new_state[idx] += cb * result[idx] - 1j * sb * result[partner]

            result = new_state

        return result

    def _initial_state(self) -> np.ndarray:
        """Create the uniform superposition |+>^n."""
        state = np.ones(self._dim, dtype=np.complex128) / np.sqrt(self._dim)
        return state

    def evaluate(
        self,
        gammas: np.ndarray,
        betas: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
    ) -> QAOAResult:
        """Run QAOA circuit and return expectation value.

        Parameters
        ----------
        gammas : ndarray, shape (p,)
            Cost operator angles.
        betas : ndarray, shape (p,)
            Mixer operator angles.
        initial_state : ndarray, optional
            Initial state vector. Default: uniform superposition.

        Returns
        -------
        QAOAResult with expectation value, state, and best bitstring.
        """
        gammas = np.asarray(gammas, dtype=np.float64)
        betas = np.asarray(betas, dtype=np.float64)

        if len(gammas) != self.p or len(betas) != self.p:
            raise ValueError(
                f"Expected {self.p} gammas and betas, "
                f"got {len(gammas)} and {len(betas)}"
            )

        if initial_state is not None:
            state = np.asarray(initial_state, dtype=np.complex128).copy()
        else:
            state = self._initial_state()

        # Apply p layers of cost + mixer
        for layer in range(self.p):
            state = self.cost_operator(gammas[layer], state)
            state = self.mixer_operator(betas[layer], state)

        # Compute measurement probabilities
        probs = np.abs(state) ** 2

        # Expectation value of the cost Hamiltonian
        expectation = float(np.sum(probs * self._cost_diagonal))

        # Find best bitstring
        best_idx = int(np.argmax(probs))
        best_bitstring = np.array(
            [(best_idx >> q) & 1 for q in range(self.n_qubits)],
            dtype=int,
        )
        best_cost = float(self._cost_diagonal[best_idx])

        return QAOAResult(
            expectation=expectation,
            state=state,
            probabilities=probs,
            best_bitstring=best_bitstring,
            best_cost=best_cost,
        )

    def optimize(
        self,
        n_restarts: int = 5,
        max_iter: int = 100,
        rng: Optional[np.random.Generator] = None,
    ) -> QAOAOptResult:
        """Optimize QAOA parameters using coordinate descent.

        Parameters
        ----------
        n_restarts : int
            Number of random restarts.
        max_iter : int
            Maximum iterations per restart.
        rng : Generator, optional
            Random number generator.

        Returns
        -------
        QAOAOptResult with optimized parameters and history.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        best_expectation = float("-inf")
        best_gammas = np.zeros(self.p)
        best_betas = np.zeros(self.p)
        best_bitstring = np.zeros(self.n_qubits, dtype=int)
        all_history: List[float] = []

        for _restart in range(n_restarts):
            gammas = rng.uniform(0, 2 * np.pi, size=self.p)
            betas = rng.uniform(0, np.pi, size=self.p)
            history: List[float] = []

            for _iteration in range(max_iter):
                improved = False

                # Optimize each parameter by grid search
                for param_idx in range(2 * self.p):
                    is_gamma = param_idx < self.p
                    idx = param_idx if is_gamma else param_idx - self.p
                    max_val = 2 * np.pi if is_gamma else np.pi

                    best_param_val = gammas[idx] if is_gamma else betas[idx]
                    best_eval = float("-inf")

                    for trial in np.linspace(0, max_val, 30):
                        if is_gamma:
                            gammas[idx] = trial
                        else:
                            betas[idx] = trial

                        result = self.evaluate(gammas, betas)
                        # We want to MAXIMIZE the cost expectation
                        if result.expectation > best_eval:
                            best_eval = result.expectation
                            best_param_val = trial

                    if is_gamma:
                        gammas[idx] = best_param_val
                    else:
                        betas[idx] = best_param_val

                result = self.evaluate(gammas, betas)
                history.append(result.expectation)

                if len(history) >= 2 and abs(history[-1] - history[-2]) < 1e-8:
                    break

            if history and history[-1] > best_expectation:
                best_expectation = history[-1]
                best_gammas = gammas.copy()
                best_betas = betas.copy()
                final_result = self.evaluate(best_gammas, best_betas)
                best_bitstring = final_result.best_bitstring
            all_history.extend(history)

        return QAOAOptResult(
            best_gammas=best_gammas,
            best_betas=best_betas,
            best_expectation=best_expectation,
            best_bitstring=best_bitstring,
            optimization_history=all_history,
        )


# ---------------------------------------------------------------------------
# Problem-specific QAOA builders
# ---------------------------------------------------------------------------

def maxcut_qaoa(adjacency_matrix: np.ndarray, p: int = 1) -> QAOACircuit:
    """Build QAOA circuit for MaxCut problem.

    MaxCut cost function:
        C = sum_{(i,j) in E} w_{ij} * (1 - Z_i * Z_j) / 2

    We maximize C, equivalently maximize:
        C' = sum_{(i,j)} w_{ij} * (-Z_i * Z_j) / 2
    (dropping the constant).

    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Weighted adjacency matrix (symmetric).
    p : int
        QAOA depth.
    """
    adj = np.asarray(adjacency_matrix, dtype=np.float64)
    n = adj.shape[0]

    cost_terms: List[Tuple[list, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(adj[i, j]) > 1e-15:
                # -w/2 * Z_i * Z_j  (we negate to maximize cut)
                cost_terms.append(([i, j], -adj[i, j] / 2.0))

    return QAOACircuit(n_qubits=n, cost_terms=cost_terms, p=p)


def graph_coloring_qaoa(
    adjacency_matrix: np.ndarray,
    n_colors: int,
    p: int = 1,
) -> QAOACircuit:
    """Build QAOA circuit for graph coloring.

    Uses a penalty-based encoding: one qubit per (vertex, color) pair.
    Penalty for adjacent vertices having the same color.

    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Adjacency matrix (binary or weighted).
    n_colors : int
        Number of colors.
    p : int
        QAOA depth.
    """
    adj = np.asarray(adjacency_matrix, dtype=np.float64)
    n = adj.shape[0]
    n_qubits = n * n_colors
    penalty = 10.0

    cost_terms: List[Tuple[list, float]] = []

    # Penalty: adjacent vertices with same color
    for i in range(n):
        for j in range(i + 1, n):
            if abs(adj[i, j]) > 1e-15:
                for c in range(n_colors):
                    q_i = i * n_colors + c
                    q_j = j * n_colors + c
                    # Penalty when both are "1" (assigned this color)
                    # Z_i * Z_j term penalizes |11>
                    cost_terms.append(([q_i, q_j], penalty / 4.0))

    # One-color-per-vertex constraint (soft)
    for i in range(n):
        for c1 in range(n_colors):
            for c2 in range(c1 + 1, n_colors):
                q1 = i * n_colors + c1
                q2 = i * n_colors + c2
                cost_terms.append(([q1, q2], penalty / 4.0))

    return QAOACircuit(n_qubits=n_qubits, cost_terms=cost_terms, p=p)


def number_partition_qaoa(numbers: List[float], p: int = 1) -> QAOACircuit:
    """Build QAOA circuit for number partitioning.

    Minimize (sum_i s_i * z_i)^2 where z_i in {+1, -1}.

    Expanding: sum_{i,j} s_i * s_j * Z_i * Z_j.
    We negate to convert to maximization.

    Parameters
    ----------
    numbers : list of float
        Numbers to partition.
    p : int
        QAOA depth.
    """
    nums = np.asarray(numbers, dtype=np.float64)
    n = len(nums)

    cost_terms: List[Tuple[list, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            # -s_i * s_j * Z_i * Z_j (negated for maximization)
            cost_terms.append(([i, j], -nums[i] * nums[j]))

        # Diagonal term: -s_i^2 * Z_i^2 = -s_i^2 (constant, but included
        # as single-qubit term for completeness; Z^2 = I so this is constant)
        # We skip diagonal constants since they don't affect optimization

    return QAOACircuit(n_qubits=n, cost_terms=cost_terms, p=p)


def max_independent_set_qaoa(
    adjacency_matrix: np.ndarray,
    p: int = 1,
) -> QAOACircuit:
    """Build QAOA circuit for maximum independent set.

    Maximize |S| subject to no two adjacent vertices in S.

    Cost: sum_i (1-Z_i)/2  - penalty * sum_{(i,j) in E} (1-Z_i)(1-Z_j)/4

    Parameters
    ----------
    adjacency_matrix : ndarray, shape (n, n)
        Adjacency matrix.
    p : int
        QAOA depth.
    """
    adj = np.asarray(adjacency_matrix, dtype=np.float64)
    n = adj.shape[0]
    penalty = 10.0

    cost_terms: List[Tuple[list, float]] = []

    # Reward for including vertex: -(1/2)*Z_i (negate because (1-Z)/2 rewards |1>)
    for i in range(n):
        cost_terms.append(([i], -0.5))

    # Penalty for adjacent vertices both in set
    for i in range(n):
        for j in range(i + 1, n):
            if abs(adj[i, j]) > 1e-15:
                cost_terms.append(([i, j], penalty / 4.0))

    return QAOACircuit(n_qubits=n, cost_terms=cost_terms, p=p)


def tsp_qaoa(distance_matrix: np.ndarray, p: int = 1) -> QAOACircuit:
    """Build QAOA circuit for traveling salesman problem.

    Uses one-hot encoding: qubit (i*n + j) = 1 if city i is at position j.
    Constraints enforce valid tours; objective minimizes total distance.

    Warning: Requires n^2 qubits, practical only for very small instances.

    Parameters
    ----------
    distance_matrix : ndarray, shape (n, n)
        Distance/cost matrix between cities.
    p : int
        QAOA depth.
    """
    dist = np.asarray(distance_matrix, dtype=np.float64)
    n = dist.shape[0]
    if n > 4:
        raise ValueError(f"TSP QAOA impractical for n={n} (requires {n*n} qubits, max 4)")
    n_qubits = n * n
    penalty = 100.0

    cost_terms: List[Tuple[list, float]] = []

    # Objective: minimize tour distance
    for i in range(n):
        for j in range(n):
            if i != j and abs(dist[i, j]) > 1e-15:
                for pos in range(n):
                    next_pos = (pos + 1) % n
                    q1 = i * n + pos
                    q2 = j * n + next_pos
                    # Reward for both being 1 (contribute to tour cost)
                    # Since we maximize, negate the distance
                    cost_terms.append(([q1, q2], -dist[i, j] / 4.0))

    # Row constraint: each city exactly once
    for i in range(n):
        for p1 in range(n):
            for p2 in range(p1 + 1, n):
                q1 = i * n + p1
                q2 = i * n + p2
                cost_terms.append(([q1, q2], penalty / 4.0))

    # Column constraint: each position exactly once
    for pos in range(n):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                q1 = i1 * n + pos
                q2 = i2 * n + pos
                cost_terms.append(([q1, q2], penalty / 4.0))

    return QAOACircuit(n_qubits=n_qubits, cost_terms=cost_terms, p=p)
