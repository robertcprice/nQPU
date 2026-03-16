"""Quantum-inspired classical optimizers.

Three algorithms that borrow ideas from quantum computing to solve
combinatorial optimization problems on classical hardware:

1. **SimulatedQuantumAnnealing (SQA)** -- Suzuki-Trotter path-integral
   Monte Carlo that mimics quantum annealing via coupled replica spins.
2. **QAOAInspiredOptimizer** -- Classical state-vector simulation of the
   Quantum Approximate Optimization Algorithm with coordinate-descent
   angle tuning.
3. **QuantumWalkOptimizer** -- Continuous-time quantum walk on the
   solution hypercube, using eigendecomposition-based time evolution.

All three accept problems expressed as an Ising Hamiltonian or QUBO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

@dataclass
class IsingProblem:
    """Ising model: E = sum_ij J_ij s_i s_j + sum_i h_i s_i.

    Spins are in {-1, +1}.  ``J`` is the symmetric coupling matrix
    (diagonal is ignored) and ``h`` is the local-field vector.
    """

    J: np.ndarray  # (n, n)
    h: np.ndarray  # (n,)

    def __post_init__(self) -> None:
        self.J = np.asarray(self.J, dtype=np.float64)
        self.h = np.asarray(self.h, dtype=np.float64)
        n = self.h.shape[0]
        if self.J.shape != (n, n):
            raise ValueError(
                f"J shape {self.J.shape} does not match h length {n}"
            )

    @property
    def n(self) -> int:
        """Number of spins."""
        return self.h.shape[0]

    def energy(self, spins: np.ndarray) -> float:
        """Compute energy for spin configuration in {-1, +1}^n."""
        spins = np.asarray(spins, dtype=np.float64)
        return float(spins @ self.J @ spins + self.h @ spins)

    @staticmethod
    def from_qubo(Q: np.ndarray) -> "IsingProblem":
        """Convert QUBO min x^T Q x  (x in {0,1}^n) to Ising.

        Substitution x_i = (1 + s_i) / 2 yields:
            E = (1/4) s^T Q s + (1/2)(Q 1 + diag(Q))^T (...) + const
        We drop the constant (it does not affect the optimum).
        """
        Q = np.asarray(Q, dtype=np.float64)
        n = Q.shape[0]
        # Symmetrise
        Qs = (Q + Q.T) / 2.0
        J = Qs / 4.0
        np.fill_diagonal(J, 0.0)
        h = (Qs.sum(axis=1)) / 2.0
        return IsingProblem(J=J, h=h)

    @staticmethod
    def random(n: int, seed: int = 42) -> "IsingProblem":
        """Random Ising instance for benchmarking."""
        rng = np.random.default_rng(seed)
        J = rng.standard_normal((n, n))
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)
        h = rng.standard_normal(n)
        return IsingProblem(J=J, h=h)

    @staticmethod
    def max_cut(adjacency: np.ndarray) -> "IsingProblem":
        """Ising formulation of Max-Cut.

        Max-Cut(G) = const - (1/2) s^T A s.
        Minimising the Ising energy E = s^T J s (with J = A/4)
        is equivalent to maximising the number of cut edges.
        """
        A = np.asarray(adjacency, dtype=np.float64)
        J = A / 4.0
        np.fill_diagonal(J, 0.0)
        h = np.zeros(A.shape[0], dtype=np.float64)
        return IsingProblem(J=J, h=h)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class SQAResult:
    """Result of Simulated Quantum Annealing."""

    best_bitstring: np.ndarray
    best_energy: float
    energy_history: List[float]
    acceptance_rates: List[float]
    n_replicas: int
    n_sweeps: int


@dataclass
class QAOAInspiredResult:
    """Result of QAOA-inspired optimisation."""

    best_bitstring: np.ndarray
    best_cost: float
    optimal_angles: Tuple[np.ndarray, np.ndarray]  # (gammas, betas)
    state_vector: np.ndarray
    depth: int
    cost_history: List[float]


@dataclass
class QuantumWalkOptimizerResult:
    """Result of quantum-walk-based optimisation."""

    best_solution: np.ndarray
    best_energy: float
    probabilities: np.ndarray
    walk_history: List[float]
    evolution_time: float


# ---------------------------------------------------------------------------
# Simulated Quantum Annealing
# ---------------------------------------------------------------------------

class SimulatedQuantumAnnealing:
    """Path-integral Monte Carlo simulated quantum annealing.

    Uses the Suzuki-Trotter decomposition to represent the transverse-
    field Ising model as a classical system of coupled replica layers.
    """

    def __init__(
        self,
        n_replicas: int = 16,
        n_sweeps: int = 200,
        gamma_start: float = 3.0,
        gamma_end: float = 0.01,
        temperature: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.n_replicas = n_replicas
        self.n_sweeps = n_sweeps
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.temperature = temperature
        self.seed = seed

    def solve(self, problem: IsingProblem) -> SQAResult:
        """Run SQA on the given Ising problem."""
        rng = np.random.default_rng(self.seed)
        n = problem.n
        T = self.temperature
        R = self.n_replicas

        # Initialise replicas: each is a random spin config {-1, +1}^n
        replicas = rng.choice([-1, 1], size=(R, n)).astype(np.float64)

        best_energy = np.inf
        best_spins: np.ndarray = replicas[0].copy()
        energy_history: List[float] = []
        acceptance_rates: List[float] = []

        for sweep in range(self.n_sweeps):
            # Linear schedule for transverse field
            frac = sweep / max(self.n_sweeps - 1, 1)
            gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * frac

            # Inter-replica coupling
            # J_perp = -(T/2) * ln(tanh(Gamma / (R * T)))
            arg = gamma / (R * T)
            # Clamp to avoid numerical issues
            tanh_val = np.tanh(min(arg, 20.0))
            tanh_val = max(tanh_val, 1e-15)
            J_perp = -(T / 2.0) * np.log(tanh_val)

            accepted = 0
            total = 0

            for r in range(R):
                # Intra-replica Metropolis sweep (classical energy)
                for i in rng.permutation(n):
                    s_old = replicas[r, i]
                    # Energy change from flipping spin i in the
                    # classical Ising energy E = s^T J s + h^T s:
                    #   delta_E = -4 s_i (J[i] @ s) - 2 s_i h_i
                    local_field = -2.0 * s_old * (
                        2.0 * (problem.J[i] @ replicas[r]) + problem.h[i]
                    )
                    # Inter-replica coupling contribution
                    # flipping s_i changes coupling to neighbours by
                    # -2 s_i J_perp (s_{r+1,i} + s_{r-1,i})
                    r_up = (r + 1) % R
                    r_down = (r - 1) % R
                    inter = -2.0 * s_old * J_perp * (
                        replicas[r_up, i] + replicas[r_down, i]
                    )
                    delta_e = local_field + inter
                    total += 1
                    if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                        replicas[r, i] = -s_old
                        accepted += 1

            # Track best across replicas
            for r in range(R):
                e = problem.energy(replicas[r])
                if e < best_energy:
                    best_energy = e
                    best_spins = replicas[r].copy()

            energy_history.append(best_energy)
            acceptance_rates.append(accepted / max(total, 1))

        return SQAResult(
            best_bitstring=best_spins,
            best_energy=best_energy,
            energy_history=energy_history,
            acceptance_rates=acceptance_rates,
            n_replicas=R,
            n_sweeps=self.n_sweeps,
        )


# ---------------------------------------------------------------------------
# QAOA-Inspired Optimizer
# ---------------------------------------------------------------------------

class QAOAInspiredOptimizer:
    """Classical state-vector simulation of QAOA.

    For small problems (<= ~20 qubits), exactly simulates the QAOA
    circuit and optimises the variational angles via coordinate descent.
    """

    def __init__(
        self,
        depth: int = 2,
        n_optimization_rounds: int = 10,
        angle_resolution: int = 20,
        seed: int = 42,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.n_optimization_rounds = n_optimization_rounds
        self.angle_resolution = angle_resolution
        self.seed = seed

    def solve(self, problem: IsingProblem) -> QAOAInspiredResult:
        """Run QAOA-inspired optimization."""
        n = problem.n
        if n > 20:
            raise ValueError(
                f"QAOAInspiredOptimizer supports n <= 20, got {n}"
            )
        rng = np.random.default_rng(self.seed)
        N = 1 << n  # 2^n

        # Pre-compute cost vector: C(z) for each bitstring z in {0..2^n-1}
        costs = self._compute_cost_vector(problem, n, N)

        # Initial angles (small random)
        gammas = rng.uniform(0, np.pi, size=self.depth)
        betas = rng.uniform(0, np.pi / 2, size=self.depth)

        cost_history: List[float] = []

        # Coordinate descent optimisation
        angle_grid = np.linspace(0, 2 * np.pi, self.angle_resolution)
        beta_grid = np.linspace(0, np.pi, self.angle_resolution)

        for _round in range(self.n_optimization_rounds):
            # Optimise each gamma_p, beta_p
            for p in range(self.depth):
                # Optimise gamma_p
                best_val = np.inf
                best_angle = gammas[p]
                for g in angle_grid:
                    gammas[p] = g
                    sv = self._run_qaoa(n, N, costs, gammas, betas)
                    probs = np.abs(sv) ** 2
                    expected = float(probs @ costs)
                    if expected < best_val:
                        best_val = expected
                        best_angle = g
                gammas[p] = best_angle

                # Optimise beta_p
                best_val = np.inf
                best_angle = betas[p]
                for b in beta_grid:
                    betas[p] = b
                    sv = self._run_qaoa(n, N, costs, gammas, betas)
                    probs = np.abs(sv) ** 2
                    expected = float(probs @ costs)
                    if expected < best_val:
                        best_val = expected
                        best_angle = b
                betas[p] = best_angle

            sv = self._run_qaoa(n, N, costs, gammas, betas)
            probs = np.abs(sv) ** 2
            cost_history.append(float(probs @ costs))

        # Extract best bitstring
        sv = self._run_qaoa(n, N, costs, gammas, betas)
        probs = np.abs(sv) ** 2
        best_idx = int(np.argmin(costs * probs + (1 - probs) * 1e18))
        # Actually pick highest probability state among low-cost ones
        best_idx = int(np.argmax(probs))
        best_bits = self._int_to_spins(best_idx, n)

        return QAOAInspiredResult(
            best_bitstring=best_bits,
            best_cost=float(costs[best_idx]),
            optimal_angles=(gammas.copy(), betas.copy()),
            state_vector=sv,
            depth=self.depth,
            cost_history=cost_history,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _compute_cost_vector(
        problem: IsingProblem, n: int, N: int
    ) -> np.ndarray:
        """Compute C(z) for all bitstrings z in Ising convention."""
        costs = np.empty(N, dtype=np.float64)
        for z in range(N):
            spins = np.array(
                [1 - 2 * ((z >> i) & 1) for i in range(n)],
                dtype=np.float64,
            )
            costs[z] = problem.energy(spins)
        return costs

    @staticmethod
    def _run_qaoa(
        n: int,
        N: int,
        costs: np.ndarray,
        gammas: np.ndarray,
        betas: np.ndarray,
    ) -> np.ndarray:
        """Simulate the QAOA circuit with given angles."""
        # Start in uniform superposition |+>^n
        state = np.full(N, 1.0 / np.sqrt(N), dtype=np.complex128)

        for gamma, beta in zip(gammas, betas):
            # Phase separator: exp(-i * gamma * C)
            state = state * np.exp(-1j * gamma * costs)

            # Mixer: exp(-i * beta * B) where B = sum_i X_i
            # Apply via Hadamard -> phase -> Hadamard per qubit
            # Equivalent to product of single-qubit X rotations
            for q in range(n):
                # exp(-i * beta * X_q) on each qubit
                cos_b = np.cos(beta)
                sin_b = np.sin(beta)
                mask = 1 << q
                for k in range(N):
                    if k & mask == 0:
                        partner = k | mask
                        a, b_val = state[k], state[partner]
                        state[k] = cos_b * a - 1j * sin_b * b_val
                        state[partner] = -1j * sin_b * a + cos_b * b_val

        return state

    @staticmethod
    def _int_to_spins(z: int, n: int) -> np.ndarray:
        """Convert integer to spin array: bit=0 -> +1, bit=1 -> -1."""
        return np.array(
            [1 - 2 * ((z >> i) & 1) for i in range(n)], dtype=np.float64
        )


# ---------------------------------------------------------------------------
# Quantum Walk Optimizer
# ---------------------------------------------------------------------------

class QuantumWalkOptimizer:
    """Continuous-time quantum walk on the hypercube.

    Hamiltonian: H = -gamma * A + C_diag
    where A is the hypercube adjacency matrix and C_diag is the diagonal
    cost operator.  Evolution via eigendecomposition.
    """

    def __init__(
        self,
        walk_time: float = 5.0,
        n_steps: int = 50,
        gamma: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.walk_time = walk_time
        self.n_steps = n_steps
        self.gamma = gamma
        self.seed = seed

    def solve(self, problem: IsingProblem) -> QuantumWalkOptimizerResult:
        """Run the quantum walk optimizer."""
        n = problem.n
        if n > 16:
            raise ValueError(
                f"QuantumWalkOptimizer supports n <= 16, got {n}"
            )
        N = 1 << n

        # Build cost vector
        costs = np.empty(N, dtype=np.float64)
        for z in range(N):
            spins = np.array(
                [1 - 2 * ((z >> i) & 1) for i in range(n)], dtype=np.float64
            )
            costs[z] = problem.energy(spins)

        # Build hypercube adjacency matrix (N x N)
        A = np.zeros((N, N), dtype=np.float64)
        for z in range(N):
            for q in range(n):
                neighbour = z ^ (1 << q)
                A[z, neighbour] = 1.0

        # Walk Hamiltonian
        H = -self.gamma * A + np.diag(costs)

        # Eigendecomposition for time evolution
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        # Initial state: uniform superposition
        state = np.full(N, 1.0 / np.sqrt(N), dtype=np.complex128)

        dt = self.walk_time / self.n_steps
        walk_history: List[float] = []

        for step in range(self.n_steps):
            t = (step + 1) * dt
            # state(t) = V exp(-i * diag(evals) * t) V^T |init>
            coeffs = eigenvectors.T @ state  # project onto eigenbasis
            evolved_coeffs = coeffs * np.exp(-1j * eigenvalues * t)
            state = eigenvectors @ evolved_coeffs

            probs = np.abs(state) ** 2
            expected_cost = float(probs @ costs)
            walk_history.append(expected_cost)

        probs = np.abs(state) ** 2
        best_idx = int(np.argmax(probs))
        best_spins = np.array(
            [1 - 2 * ((best_idx >> i) & 1) for i in range(n)],
            dtype=np.float64,
        )

        return QuantumWalkOptimizerResult(
            best_solution=best_spins,
            best_energy=float(costs[best_idx]),
            probabilities=probs,
            walk_history=walk_history,
            evolution_time=self.walk_time,
        )
