"""Quantum Evolutionary Game Theory -- replicator dynamics and ESS.

Extends classical evolutionary game theory to quantum strategy spaces.
Players use SU(2) unitary strategies instead of classical mixed strategies,
and population dynamics evolve according to quantum payoff matrices.

Key concepts:
  - Quantum replicator dynamics: dx_i/dt = x_i * (f_i - f_avg)
    where fitness f_i is determined by a quantum game payoff matrix
  - Evolutionary Stable Strategy (ESS): strategy that resists invasion
    by any mutant, extended to quantum strategy spaces
  - Coevolution: mixed populations of quantum and classical strategists

References:
    Iqbal & Toor (2002) - Evolutionary Stable Strategies in Quantum Games
    Nawaz & Toor (2004) - Generalized Quantization Scheme for Two-Person Games
    Li et al. (2009) - Quantum Games and Evolutionary Dynamics
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Population representation
# ---------------------------------------------------------------------------

@dataclass
class QuantumPopulation:
    """Population of quantum strategy players.

    Each strategy is an SU(2) unitary (2x2 complex matrix) and has
    a frequency (population share) in [0, 1]. Frequencies sum to 1.

    Parameters
    ----------
    strategies : list of ndarray
        List of 2x2 SU(2) unitary matrices.
    frequencies : ndarray
        Population share of each strategy, summing to 1.
    """

    strategies: List[np.ndarray]
    frequencies: np.ndarray

    def __post_init__(self) -> None:
        self.frequencies = np.asarray(self.frequencies, dtype=np.float64)
        if len(self.strategies) != len(self.frequencies):
            raise ValueError(
                f"Number of strategies ({len(self.strategies)}) != "
                f"number of frequencies ({len(self.frequencies)})"
            )

    @staticmethod
    def uniform(strategies: List[np.ndarray]) -> "QuantumPopulation":
        """Equal frequency for all strategies."""
        n = len(strategies)
        return QuantumPopulation(
            strategies=list(strategies),
            frequencies=np.ones(n) / n,
        )

    @property
    def n_strategies(self) -> int:
        """Number of distinct strategies in the population."""
        return len(self.strategies)

    def entropy(self) -> float:
        """Shannon entropy of population distribution.

        H = -sum_i x_i * log(x_i) for x_i > 0.
        Maximum entropy is log(n) for n strategies.
        """
        h = 0.0
        for f in self.frequencies:
            if f > 1e-15:
                h -= f * np.log(f)
        return float(h)


# ---------------------------------------------------------------------------
# Strategy utilities
# ---------------------------------------------------------------------------

def _su2_unitary(theta: float, phi: float) -> np.ndarray:
    """Create SU(2) unitary from (theta, phi) parametrization."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    eip = np.exp(1j * phi)
    eim = np.exp(-1j * phi)
    return np.array([
        [eip * c, s],
        [-s, eim * c],
    ], dtype=np.complex128)


def _quantum_payoff(u1: np.ndarray, u2: np.ndarray, payoff_matrix: np.ndarray,
                     gamma: float = np.pi / 2.0) -> float:
    """Compute quantum game payoff for player 1 using strategies u1, u2.

    Uses the EWL protocol with entanglement parameter gamma.
    """
    dim = 4
    # Entangling operator: exp(i*gamma/2 * X x X)
    # Since (X x X)^2 = I: exp(i*t*M) = cos(t)*I + i*sin(t)*M
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    xx = np.kron(x, x)
    c = np.cos(gamma / 2.0)
    s = np.sin(gamma / 2.0)
    j_op = c * np.eye(dim, dtype=np.complex128) + 1j * s * xx
    j_dag = j_op.conj().T

    # |00> initial state
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    state = j_op @ state
    state = np.kron(u1, u2) @ state
    state = j_dag @ state

    probs = np.abs(state) ** 2
    probs = probs.reshape(2, 2)

    # Player 1 payoff
    return float(np.sum(probs * payoff_matrix))


# ---------------------------------------------------------------------------
# Evolution result
# ---------------------------------------------------------------------------

@dataclass
class EvolutionResult:
    """Result of replicator dynamics evolution."""

    times: np.ndarray
    frequency_history: np.ndarray  # shape (n_steps, n_strategies)
    final_population: QuantumPopulation
    converged: bool


# ---------------------------------------------------------------------------
# Replicator dynamics
# ---------------------------------------------------------------------------

class QuantumReplicatorDynamics:
    """Replicator dynamics for quantum games.

    The replicator equation:
        dx_i/dt = x_i * (f_i - f_avg)

    where f_i is the fitness (expected payoff) of strategy i against the
    current population, and f_avg is the population average fitness.

    The payoff_matrix[i, j] gives the payoff to strategy i when playing
    against strategy j in the quantum game.

    Parameters
    ----------
    payoff_matrix : ndarray, shape (n, n)
        Quantum game payoff matrix. payoff_matrix[i, j] is the fitness
        of strategy i against strategy j.
    """

    def __init__(self, payoff_matrix: np.ndarray) -> None:
        self.payoff_matrix = np.asarray(payoff_matrix, dtype=np.float64)
        n = self.payoff_matrix.shape[0]
        if self.payoff_matrix.shape != (n, n):
            raise ValueError("Payoff matrix must be square")

    def fitness(self, population: QuantumPopulation) -> np.ndarray:
        """Compute fitness of each strategy in current population.

        f_i = sum_j payoff_matrix[i, j] * x_j
        """
        return self.payoff_matrix @ population.frequencies

    def average_fitness(self, population: QuantumPopulation) -> float:
        """Compute population average fitness.

        f_avg = sum_i x_i * f_i = x^T * A * x
        """
        f = self.fitness(population)
        return float(population.frequencies @ f)

    def step(self, population: QuantumPopulation, dt: float = 0.01) -> QuantumPopulation:
        """One step of replicator dynamics.

        dx_i/dt = x_i * (f_i - f_avg)
        """
        x = population.frequencies.copy()
        f = self.fitness(population)
        f_avg = float(x @ f)

        # Replicator equation (forward Euler)
        dx = x * (f - f_avg) * dt
        x_new = x + dx

        # Clip and renormalize to maintain valid distribution
        x_new = np.maximum(x_new, 0.0)
        total = np.sum(x_new)
        if total > 1e-15:
            x_new = x_new / total
        else:
            x_new = np.ones_like(x_new) / len(x_new)

        return QuantumPopulation(
            strategies=population.strategies,
            frequencies=x_new,
        )

    def evolve(
        self,
        population: QuantumPopulation,
        t_final: float = 10.0,
        dt: float = 0.01,
    ) -> EvolutionResult:
        """Run replicator dynamics to equilibrium.

        Parameters
        ----------
        population : QuantumPopulation
            Initial population.
        t_final : float
            Maximum simulation time.
        dt : float
            Time step.

        Returns
        -------
        EvolutionResult with time series of population frequencies.
        """
        n_steps = int(t_final / dt)
        n = population.n_strategies
        times = np.zeros(n_steps + 1)
        frequency_history = np.zeros((n_steps + 1, n))

        current = population
        times[0] = 0.0
        frequency_history[0] = current.frequencies.copy()

        converged = False
        actual_steps = n_steps

        for step in range(1, n_steps + 1):
            new_pop = self.step(current, dt)
            times[step] = step * dt
            frequency_history[step] = new_pop.frequencies.copy()

            # Check convergence: frequency change below threshold
            change = np.max(np.abs(new_pop.frequencies - current.frequencies))
            if change < 1e-8:
                converged = True
                actual_steps = step
                break

            current = new_pop

        return EvolutionResult(
            times=times[:actual_steps + 1],
            frequency_history=frequency_history[:actual_steps + 1],
            final_population=current,
            converged=converged,
        )

    def fixed_points(self, resolution: int = 20) -> List[np.ndarray]:
        """Find fixed points of the dynamics by scanning simplex.

        A fixed point satisfies: for all i, either x_i = 0 or f_i = f_avg.

        Parameters
        ----------
        resolution : int
            Number of grid points per simplex dimension.

        Returns
        -------
        List of frequency vectors that are approximate fixed points.
        """
        n = self.payoff_matrix.shape[0]
        if n > 5:
            # For high dimensions, only check vertices and center
            candidates = []
            # Pure strategy fixed points (vertices)
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1.0
                candidates.append(e)
            # Uniform mixed strategy
            candidates.append(np.ones(n) / n)
        else:
            # Grid scan on the simplex
            candidates = []
            # Generate simplex grid points
            grid = np.linspace(0, 1, resolution)
            if n == 2:
                for x0 in grid:
                    candidates.append(np.array([x0, 1.0 - x0]))
            elif n == 3:
                for x0 in grid:
                    for x1 in grid:
                        x2 = 1.0 - x0 - x1
                        if x2 >= -1e-10:
                            candidates.append(np.array([x0, x1, max(x2, 0.0)]))
            else:
                # For n=4,5 use random sampling on simplex
                rng = np.random.default_rng(42)
                for _ in range(resolution ** 2):
                    x = rng.dirichlet(np.ones(n))
                    candidates.append(x)

        fixed = []
        for x in candidates:
            x = np.maximum(x, 0.0)
            total = np.sum(x)
            if total < 1e-15:
                continue
            x = x / total
            pop = QuantumPopulation(
                strategies=[np.eye(2, dtype=np.complex128)] * n,
                frequencies=x,
            )
            f = self.fitness(pop)
            f_avg = float(x @ f)

            # Check fixed point condition: x_i * (f_i - f_avg) ~ 0 for all i
            residual = np.max(np.abs(x * (f - f_avg)))
            if residual < 1e-4:
                # Check not a duplicate
                is_dup = False
                for existing in fixed:
                    if np.max(np.abs(x - existing)) < 0.05:
                        is_dup = True
                        break
                if not is_dup:
                    fixed.append(x.copy())

        return fixed


# ---------------------------------------------------------------------------
# ESS Analyzer
# ---------------------------------------------------------------------------

class ESSAnalyzer:
    """Evolutionary Stable Strategy analyzer for quantum games.

    An ESS is a strategy that, when adopted by the entire population,
    cannot be invaded by any mutant strategy. Formally, strategy s* is
    an ESS if for all mutant strategies s != s*:
        1. E(s*, s*) >= E(s, s*)  [Nash condition]
        2. If E(s*, s*) = E(s, s*), then E(s*, s) > E(s, s)  [stability]
    """

    def is_ess(
        self,
        strategy_idx: int,
        payoff_matrix: np.ndarray,
        population: QuantumPopulation,
    ) -> bool:
        """Check if strategy at index is an Evolutionary Stable Strategy.

        Parameters
        ----------
        strategy_idx : int
            Index of the strategy to check.
        payoff_matrix : ndarray
            Payoff matrix where payoff_matrix[i, j] is fitness of i vs j.
        population : QuantumPopulation
            Current population (for context; uses payoff matrix directly).

        Returns
        -------
        True if the strategy satisfies ESS conditions against all others.
        """
        n = payoff_matrix.shape[0]
        s_star = strategy_idx

        for s in range(n):
            if s == s_star:
                continue

            # Nash condition: E(s*, s*) >= E(s, s*)
            e_star_star = payoff_matrix[s_star, s_star]
            e_s_star = payoff_matrix[s, s_star]

            if e_star_star < e_s_star - 1e-10:
                return False

            # Stability condition: if E(s*, s*) == E(s, s*), then E(s*, s) > E(s, s)
            if abs(e_star_star - e_s_star) < 1e-10:
                e_star_s = payoff_matrix[s_star, s]
                e_s_s = payoff_matrix[s, s]
                if e_star_s <= e_s_s + 1e-10:
                    return False

        return True

    def find_ess(
        self,
        payoff_matrix: np.ndarray,
        n_candidates: int = 50,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        """Find all ESS among the strategies in the payoff matrix.

        Parameters
        ----------
        payoff_matrix : ndarray, shape (n, n)
            Payoff matrix.
        n_candidates : int
            Not used (checks all strategies in the matrix).
        rng : Generator, optional
            Not used (deterministic check).

        Returns
        -------
        List of strategy indices that are ESS.
        """
        n = payoff_matrix.shape[0]
        dummy_pop = QuantumPopulation(
            strategies=[np.eye(2, dtype=np.complex128)] * n,
            frequencies=np.ones(n) / n,
        )

        ess_list = []
        for i in range(n):
            if self.is_ess(i, payoff_matrix, dummy_pop):
                ess_list.append(i)
        return ess_list

    def invasion_fitness(
        self,
        mutant_idx: int,
        resident_idx: int,
        payoff_matrix: np.ndarray,
    ) -> float:
        """Compute fitness of mutant in a resident population.

        When mutant is rare (epsilon -> 0), the invasion fitness is:
            W = E(mutant, resident) - E(resident, resident)

        Positive W means the mutant can invade.
        """
        return float(
            payoff_matrix[mutant_idx, resident_idx]
            - payoff_matrix[resident_idx, resident_idx]
        )

    def pairwise_invasion_matrix(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """Compute the pairwise invasion fitness matrix.

        invasion[i, j] = invasion fitness of strategy i against resident j.
        Positive values mean i can invade a population of j.
        """
        n = payoff_matrix.shape[0]
        invasion = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                invasion[i, j] = self.invasion_fitness(i, j, payoff_matrix)
        return invasion


# ---------------------------------------------------------------------------
# Coevolutionary Dynamics
# ---------------------------------------------------------------------------

@dataclass
class CoevoResult:
    """Result of coevolutionary dynamics."""

    times: np.ndarray
    quantum_fraction_history: np.ndarray
    strategy_frequencies: np.ndarray  # shape (n_steps, n_strategies)
    final_quantum_fraction: float


class CoevolutionaryDynamics:
    """Coevolution of quantum strategies and classical strategies.

    Models a population where a fraction uses quantum strategies and the
    rest use classical. Over time, the quantum fraction evolves based on
    relative fitness.

    Parameters
    ----------
    quantum_payoff : ndarray, shape (n, n)
        Payoff matrix when both players use quantum strategies.
    classical_payoff : ndarray, shape (n, n)
        Payoff matrix when both players use classical strategies.
    quantum_fraction : float
        Initial fraction of quantum players in [0, 1].
    """

    def __init__(
        self,
        quantum_payoff: np.ndarray,
        classical_payoff: np.ndarray,
        quantum_fraction: float = 0.5,
    ) -> None:
        self.quantum_payoff = np.asarray(quantum_payoff, dtype=np.float64)
        self.classical_payoff = np.asarray(classical_payoff, dtype=np.float64)
        if self.quantum_payoff.shape != self.classical_payoff.shape:
            raise ValueError("Payoff matrices must have same shape")
        if not (0.0 <= quantum_fraction <= 1.0):
            raise ValueError("quantum_fraction must be in [0, 1]")
        self.initial_quantum_fraction = quantum_fraction
        self.n_strategies = self.quantum_payoff.shape[0]

    def evolve(
        self,
        n_steps: int = 1000,
        dt: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ) -> CoevoResult:
        """Evolve mixed quantum/classical population.

        At each step:
        1. Compute average fitness of quantum and classical subpopulations
        2. Update quantum fraction based on relative fitness
        3. Within each subpopulation, run replicator dynamics

        Parameters
        ----------
        n_steps : int
            Number of time steps.
        dt : float
            Time step size.
        rng : Generator, optional
            Random number generator (for stochastic perturbations).

        Returns
        -------
        CoevoResult with time series of quantum fraction and frequencies.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n = self.n_strategies
        q_frac = self.initial_quantum_fraction

        # Initialize equal frequencies within each subpopulation
        q_freq = np.ones(n, dtype=np.float64) / n
        c_freq = np.ones(n, dtype=np.float64) / n

        times = np.zeros(n_steps + 1)
        q_frac_history = np.zeros(n_steps + 1)
        freq_history = np.zeros((n_steps + 1, n))

        q_frac_history[0] = q_frac
        freq_history[0] = q_frac * q_freq + (1 - q_frac) * c_freq

        for step in range(1, n_steps + 1):
            times[step] = step * dt

            # Compute average fitness for each subpopulation
            q_fitness = self.quantum_payoff @ q_freq
            q_avg = float(q_freq @ q_fitness)

            c_fitness = self.classical_payoff @ c_freq
            c_avg = float(c_freq @ c_fitness)

            # Update quantum fraction based on relative fitness
            # dq/dt = q * (1 - q) * (f_q - f_c)
            dq = q_frac * (1 - q_frac) * (q_avg - c_avg) * dt
            q_frac = np.clip(q_frac + dq, 0.0, 1.0)

            # Within-subpopulation replicator dynamics
            # Quantum subpopulation
            dq_freq = q_freq * (q_fitness - q_avg) * dt
            q_freq = np.maximum(q_freq + dq_freq, 0.0)
            q_total = np.sum(q_freq)
            if q_total > 1e-15:
                q_freq = q_freq / q_total
            else:
                q_freq = np.ones(n) / n

            # Classical subpopulation
            dc_freq = c_freq * (c_fitness - c_avg) * dt
            c_freq = np.maximum(c_freq + dc_freq, 0.0)
            c_total = np.sum(c_freq)
            if c_total > 1e-15:
                c_freq = c_freq / c_total
            else:
                c_freq = np.ones(n) / n

            q_frac_history[step] = q_frac
            freq_history[step] = q_frac * q_freq + (1 - q_frac) * c_freq

        return CoevoResult(
            times=times,
            quantum_fraction_history=q_frac_history,
            strategy_frequencies=freq_history,
            final_quantum_fraction=float(q_frac),
        )
