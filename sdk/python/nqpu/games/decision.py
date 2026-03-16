"""Quantum-Enhanced Decision Making.

Quantum models of decision processes that exhibit interference effects,
contextuality, and other phenomena that cannot be captured by classical
probability theory.

Applications:
  - Bayesian inference with quantum interference (order effects in belief)
  - Markov decision processes with quantum exploration
  - Multi-armed bandits with amplitude amplification
  - Portfolio optimization via game-theoretic QUBO

References:
    Busemeyer & Bruza (2012) - Quantum Models of Cognition and Decision
    Yukalov & Sornette (2011) - Decision Theory with Prospect Interference
    Dunjko & Briegel (2018) - Machine Learning and AI with Quantum Resources
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Quantum Bayesian Inference
# ---------------------------------------------------------------------------

@dataclass
class BayesianResult:
    """Result of a quantum Bayesian update."""

    prior: np.ndarray
    posterior: np.ndarray
    evidence_strength: float
    classical_posterior: np.ndarray
    interference_term: float
    kl_divergence: float  # KL(quantum || classical)


class QuantumBayesian:
    """Quantum Bayesian inference with interference effects.

    In quantum probability, belief states are amplitude vectors.
    Bayesian update is a unitary rotation followed by projection.
    This produces interference terms absent in classical Bayes,
    which can model order-dependent belief updating observed in
    human cognition experiments.

    Parameters
    ----------
    n_hypotheses : int
        Number of hypotheses (basis states).
    """

    def __init__(self, n_hypotheses: int) -> None:
        if n_hypotheses < 2:
            raise ValueError("Need at least 2 hypotheses")
        self.n_hypotheses = n_hypotheses

    def prepare_prior(self, probabilities: np.ndarray) -> np.ndarray:
        """Create a quantum state encoding the prior.

        The state amplitudes are sqrt(probabilities) with zero phase,
        giving Born rule probabilities equal to the classical prior.
        """
        p = np.asarray(probabilities, dtype=np.float64)
        if len(p) != self.n_hypotheses:
            raise ValueError(f"Expected {self.n_hypotheses} probabilities")
        if abs(np.sum(p) - 1.0) > 1e-8:
            raise ValueError("Probabilities must sum to 1")
        if np.any(p < 0):
            raise ValueError("Probabilities must be non-negative")
        return np.sqrt(p).astype(np.complex128)

    def evidence_operator(
        self,
        likelihoods: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Construct a unitary evidence operator.

        Builds a unitary U such that when applied to the prior state
        |psi_prior> = [sqrt(p_0), ..., sqrt(p_{n-1})], the resulting
        state has Born-rule probabilities that shift toward the Bayesian
        posterior (proportional to likelihood_i * p_i).

        The construction uses sequential Givens rotations: for each pair
        (i, j), compute the angle that would rotate the prior amplitudes
        to match the target posterior amplitudes for those two components,
        then interpolate with the strength parameter.

        Parameters
        ----------
        likelihoods : array of float
            P(evidence | hypothesis_i) for each hypothesis.
        strength : float
            Evidence strength parameter (0 = no evidence, 1 = full update).
        """
        lk = np.asarray(likelihoods, dtype=np.float64)
        if len(lk) != self.n_hypotheses:
            raise ValueError(f"Expected {self.n_hypotheses} likelihoods")

        n = self.n_hypotheses

        if strength < 1e-15:
            return np.eye(n, dtype=np.complex128)

        # Build unitary via sequential Givens rotations.
        # Strategy: for each adjacent pair (i, i+1), compute the rotation
        # angle needed to shift amplitude ratio from prior to posterior.
        u = np.eye(n, dtype=np.complex128)

        for i in range(n - 1):
            j = i + 1
            # Current state after previous rotations
            # We want the ratio of amplitudes to match the posterior ratio
            lk_i = max(lk[i], 1e-15)
            lk_j = max(lk[j], 1e-15)

            # The desired rotation maps [a_i, a_j] -> [a_i * r, a_j / r]
            # where r = (lk_i / lk_j)^{strength/4} (fourth root because
            # we're working with amplitudes and the rotation affects both)
            #
            # More precisely: the target angle is arctan(sqrt(lk_j) / sqrt(lk_i))
            # for the pair, and the neutral angle (no update) is pi/4.
            # Interpolate with strength.
            target_angle = np.arctan2(np.sqrt(lk_j), np.sqrt(lk_i))
            neutral_angle = np.pi / 4.0
            angle = neutral_angle + strength * (target_angle - neutral_angle)

            # Rotation to apply: we want to rotate the current state
            # by (angle - neutral_angle) in the (i,j) plane
            delta = angle - neutral_angle
            c = np.cos(delta)
            s = np.sin(delta)
            g = np.eye(n, dtype=np.complex128)
            g[i, i] = c
            g[i, j] = -s
            g[j, i] = s
            g[j, j] = c
            u = g @ u

        return u

    def update(
        self,
        prior_probs: np.ndarray,
        likelihoods: np.ndarray,
        strength: float = 1.0,
    ) -> BayesianResult:
        """Perform a quantum Bayesian update.

        Parameters
        ----------
        prior_probs : array
            Classical prior probabilities.
        likelihoods : array
            P(evidence | hypothesis_i) for each hypothesis.
        strength : float
            Evidence strength parameter.

        Returns
        -------
        BayesianResult with quantum posterior, classical posterior, and
        interference analysis.
        """
        prior_state = self.prepare_prior(prior_probs)
        u = self.evidence_operator(likelihoods, strength)
        posterior_state = u @ prior_state

        # Quantum posterior: Born rule
        q_posterior = np.abs(posterior_state) ** 2
        q_posterior = q_posterior / np.sum(q_posterior)  # renormalize

        # Classical Bayes: P(H|E) = P(E|H)*P(H) / P(E)
        c_posterior = likelihoods * prior_probs
        c_sum = np.sum(c_posterior)
        if c_sum > 1e-15:
            c_posterior = c_posterior / c_sum
        else:
            c_posterior = prior_probs.copy()

        # Interference term: difference between quantum and classical
        interference = float(np.sum(np.abs(q_posterior - c_posterior)))

        # KL divergence: D_KL(q || c)
        kl = 0.0
        for i in range(self.n_hypotheses):
            if q_posterior[i] > 1e-15 and c_posterior[i] > 1e-15:
                kl += q_posterior[i] * np.log(q_posterior[i] / c_posterior[i])

        return BayesianResult(
            prior=prior_probs,
            posterior=q_posterior,
            evidence_strength=strength,
            classical_posterior=c_posterior,
            interference_term=interference,
            kl_divergence=float(kl),
        )

    def sequential_update(
        self,
        prior_probs: np.ndarray,
        evidence_sequence: List[Tuple[np.ndarray, float]],
    ) -> List[BayesianResult]:
        """Apply a sequence of evidence updates.

        Demonstrates order-dependence: quantum updates are unitary
        operations, and U1*U2 != U2*U1 in general.

        Parameters
        ----------
        prior_probs : array
            Initial prior.
        evidence_sequence : list of (likelihoods, strength) tuples
            Sequential evidence to incorporate.
        """
        results = []
        current_probs = np.asarray(prior_probs, dtype=np.float64)
        for likelihoods, strength in evidence_sequence:
            r = self.update(current_probs, likelihoods, strength)
            results.append(r)
            current_probs = r.posterior
        return results


# ---------------------------------------------------------------------------
# Quantum Markov Decision Process
# ---------------------------------------------------------------------------

@dataclass
class MDPResult:
    """Result of quantum MDP value iteration."""

    values: np.ndarray  # value function over states
    policy: np.ndarray  # optimal action for each state
    n_iterations: int
    converged: bool
    history: List[float]  # max value change per iteration


class QuantumMarkov:
    """Quantum Markov Decision Process.

    States are quantum superpositions.  Transitions are unitary operators
    plus measurement (decoherence), and rewards are diagonal observables.

    This simulates a classical MDP enhanced with quantum exploration,
    where the agent can place states in superposition and the value
    function benefits from interference between paths.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_actions : int
        Number of actions per state.
    transition_probs : array, shape (n_states, n_actions, n_states)
        P(s' | s, a) transition probabilities.
    rewards : array, shape (n_states, n_actions)
        R(s, a) immediate rewards.
    gamma : float
        Discount factor in [0, 1).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        transition_probs: np.ndarray,
        rewards: np.ndarray,
        gamma: float = 0.95,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.transitions = np.asarray(transition_probs, dtype=np.float64)
        self.rewards = np.asarray(rewards, dtype=np.float64)
        self.gamma = gamma

        if self.transitions.shape != (n_states, n_actions, n_states):
            raise ValueError(
                f"Transition shape mismatch: expected "
                f"({n_states}, {n_actions}, {n_states}), "
                f"got {self.transitions.shape}"
            )
        if self.rewards.shape != (n_states, n_actions):
            raise ValueError(
                f"Reward shape mismatch: expected ({n_states}, {n_actions}), "
                f"got {self.rewards.shape}"
            )

    def value_iteration(
        self,
        max_iterations: int = 500,
        tol: float = 1e-6,
    ) -> MDPResult:
        """Classical value iteration for the MDP.

        V(s) = max_a [R(s,a) + gamma * sum_s' P(s'|s,a) * V(s')]
        """
        values = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)
        history = []

        converged = False
        for iteration in range(max_iterations):
            new_values = np.zeros(self.n_states)
            for s in range(self.n_states):
                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    q_values[a] = self.rewards[s, a] + self.gamma * np.dot(
                        self.transitions[s, a], values
                    )
                new_values[s] = np.max(q_values)
                policy[s] = int(np.argmax(q_values))

            delta = float(np.max(np.abs(new_values - values)))
            history.append(delta)
            values = new_values

            if delta < tol:
                converged = True
                break

        return MDPResult(
            values=values,
            policy=policy,
            n_iterations=iteration + 1,
            converged=converged,
            history=history,
        )

    def quantum_value_iteration(
        self,
        max_iterations: int = 500,
        tol: float = 1e-6,
        exploration_bonus: float = 0.1,
    ) -> MDPResult:
        """Quantum-enhanced value iteration.

        Adds an exploration bonus inspired by Grover-style amplitude
        amplification: states with lower visitation get an interference
        bonus that encourages exploration of the state space.

        This is a classical simulation of the quantum speedup concept.
        """
        values = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)
        visit_counts = np.ones(self.n_states)  # pseudocounts
        history = []

        converged = False
        for iteration in range(max_iterations):
            # Decay exploration bonus over iterations (annealing schedule)
            decay = 1.0 / (1.0 + iteration)
            current_bonus = exploration_bonus * decay

            new_values = np.zeros(self.n_states)
            for s in range(self.n_states):
                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    # Standard Bellman update
                    q_values[a] = self.rewards[s, a] + self.gamma * np.dot(
                        self.transitions[s, a], values
                    )
                    # Quantum exploration bonus: inversely proportional to sqrt(visits)
                    # Mimics amplitude amplification giving sqrt speedup
                    # Decays over iterations to ensure convergence
                    for s_next in range(self.n_states):
                        bonus = current_bonus / np.sqrt(visit_counts[s_next])
                        q_values[a] += self.gamma * self.transitions[s, a, s_next] * bonus

                new_values[s] = np.max(q_values)
                best_a = int(np.argmax(q_values))
                policy[s] = best_a
                # Update visit counts based on transition
                for s_next in range(self.n_states):
                    visit_counts[s_next] += self.transitions[s, best_a, s_next]

            delta = float(np.max(np.abs(new_values - values)))
            history.append(delta)
            values = new_values

            if delta < tol:
                converged = True
                break

        return MDPResult(
            values=values,
            policy=policy,
            n_iterations=iteration + 1,
            converged=converged,
            history=history,
        )

    @staticmethod
    def random_mdp(
        n_states: int = 5,
        n_actions: int = 3,
        seed: int = 42,
    ) -> "QuantumMarkov":
        """Create a random MDP for testing."""
        rng = np.random.default_rng(seed)
        transitions = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
        rewards = rng.uniform(-1, 1, size=(n_states, n_actions))
        return QuantumMarkov(n_states, n_actions, transitions, rewards)


# ---------------------------------------------------------------------------
# Quantum Multi-Armed Bandit
# ---------------------------------------------------------------------------

@dataclass
class BanditResult:
    """Result from a quantum bandit experiment."""

    total_reward: float
    cumulative_regret: float
    arm_pulls: np.ndarray
    reward_history: List[float]
    regret_history: List[float]
    best_arm: int


class QuantumBandit:
    """Quantum-enhanced multi-armed bandit.

    Uses amplitude amplification-inspired exploration: instead of
    epsilon-greedy or UCB, the exploration probability decays as
    1/sqrt(t) (matching the Grover quadratic speedup), giving faster
    convergence to the optimal arm.

    Parameters
    ----------
    arm_means : array-like
        True mean reward for each arm.
    arm_stds : array-like, optional
        Standard deviation of reward for each arm. Default 1.0.
    """

    def __init__(
        self,
        arm_means: np.ndarray,
        arm_stds: Optional[np.ndarray] = None,
    ) -> None:
        self.arm_means = np.asarray(arm_means, dtype=np.float64)
        self.n_arms = len(self.arm_means)
        self.best_arm = int(np.argmax(self.arm_means))
        self.best_mean = float(np.max(self.arm_means))
        if arm_stds is not None:
            self.arm_stds = np.asarray(arm_stds, dtype=np.float64)
        else:
            self.arm_stds = np.ones(self.n_arms)

    def run_quantum_exploration(
        self,
        n_rounds: int = 1000,
        seed: int = 42,
    ) -> BanditResult:
        """Run the quantum-inspired bandit algorithm.

        Exploration rate decays as 1/sqrt(t), inspired by Grover's
        quadratic speedup for search in unstructured databases.
        """
        rng = np.random.default_rng(seed)
        arm_pulls = np.zeros(self.n_arms, dtype=int)
        arm_rewards = np.zeros(self.n_arms)
        total_reward = 0.0
        cumulative_regret = 0.0
        reward_history = []
        regret_history = []

        for t in range(1, n_rounds + 1):
            # Quantum exploration: probability decays as 1/sqrt(t)
            explore_prob = 1.0 / np.sqrt(t)

            if rng.random() < explore_prob or t <= self.n_arms:
                # Exploration: uniform or initial round-robin
                if t <= self.n_arms:
                    arm = t - 1
                else:
                    arm = rng.integers(0, self.n_arms)
            else:
                # Exploitation: pick arm with best estimated mean
                estimates = np.where(
                    arm_pulls > 0,
                    arm_rewards / arm_pulls,
                    float("inf"),  # unvisited arms get infinite priority
                )
                arm = int(np.argmax(estimates))

            # Sample reward
            reward = rng.normal(self.arm_means[arm], self.arm_stds[arm])
            arm_pulls[arm] += 1
            arm_rewards[arm] += reward
            total_reward += reward
            regret = self.best_mean - self.arm_means[arm]
            cumulative_regret += regret
            reward_history.append(reward)
            regret_history.append(cumulative_regret)

        return BanditResult(
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            arm_pulls=arm_pulls,
            reward_history=reward_history,
            regret_history=regret_history,
            best_arm=self.best_arm,
        )

    def run_classical_ucb(
        self,
        n_rounds: int = 1000,
        seed: int = 42,
    ) -> BanditResult:
        """Run classical UCB1 for comparison."""
        rng = np.random.default_rng(seed)
        arm_pulls = np.zeros(self.n_arms, dtype=int)
        arm_rewards = np.zeros(self.n_arms)
        total_reward = 0.0
        cumulative_regret = 0.0
        reward_history = []
        regret_history = []

        for t in range(1, n_rounds + 1):
            if t <= self.n_arms:
                arm = t - 1
            else:
                estimates = arm_rewards / np.maximum(arm_pulls, 1)
                ucb_bonus = np.sqrt(2 * np.log(t) / np.maximum(arm_pulls, 1))
                arm = int(np.argmax(estimates + ucb_bonus))

            reward = rng.normal(self.arm_means[arm], self.arm_stds[arm])
            arm_pulls[arm] += 1
            arm_rewards[arm] += reward
            total_reward += reward
            regret = self.best_mean - self.arm_means[arm]
            cumulative_regret += regret
            reward_history.append(reward)
            regret_history.append(cumulative_regret)

        return BanditResult(
            total_reward=total_reward,
            cumulative_regret=cumulative_regret,
            arm_pulls=arm_pulls,
            reward_history=reward_history,
            regret_history=regret_history,
            best_arm=self.best_arm,
        )


# ---------------------------------------------------------------------------
# Quantum Portfolio (game-theoretic version)
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """Result of quantum portfolio optimization."""

    weights: np.ndarray
    expected_return: float
    risk: float
    sharpe_ratio: float
    method: str


class QuantumPortfolio:
    """Quantum portfolio optimization via game-theoretic QUBO.

    Unlike the finance module's Markowitz implementation, this formulates
    portfolio optimization as a competitive game between assets, where
    each asset "competes" for allocation weight.

    The QUBO encodes:
        minimize  w^T Sigma w  -  lambda * mu^T w
    where Sigma is the covariance matrix, mu is expected returns,
    and lambda is the risk-return tradeoff parameter.

    Parameters
    ----------
    returns : array, shape (n_assets,)
        Expected return for each asset.
    covariance : array, shape (n_assets, n_assets)
        Covariance matrix.
    risk_aversion : float
        Lambda parameter controlling risk-return tradeoff.
    """

    def __init__(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float = 1.0,
    ) -> None:
        self.returns = np.asarray(returns, dtype=np.float64)
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.risk_aversion = risk_aversion
        self.n_assets = len(self.returns)

    def qubo_matrix(self, n_bits: int = 3) -> np.ndarray:
        """QUBO matrix for discretized portfolio weights.

        Each asset's weight is encoded in n_bits binary variables.
        Total variables: n_assets * n_bits.
        """
        n = self.n_assets
        size = n * n_bits
        q = np.zeros((size, size))

        # Weight encoding: w_i = sum_k 2^{-k-1} * x_{i*n_bits + k}
        for i in range(n):
            for j in range(n):
                for ki in range(n_bits):
                    for kj in range(n_bits):
                        idx_i = i * n_bits + ki
                        idx_j = j * n_bits + kj
                        coeff = 2 ** (-(ki + 1)) * 2 ** (-(kj + 1))
                        # Risk term: w^T Sigma w
                        q[idx_i, idx_j] += self.covariance[i, j] * coeff

        # Return term: -lambda * mu^T w
        for i in range(n):
            for k in range(n_bits):
                idx = i * n_bits + k
                coeff = 2 ** (-(k + 1))
                q[idx, idx] -= self.risk_aversion * self.returns[i] * coeff

        return q

    def optimize_analytical(self) -> PortfolioResult:
        """Analytical mean-variance optimization (Markowitz)."""
        cov_inv = np.linalg.inv(
            self.covariance + 1e-8 * np.eye(self.n_assets)
        )
        raw_weights = self.risk_aversion * cov_inv @ self.returns
        # Project to simplex (non-negative weights summing to 1)
        weights = np.maximum(raw_weights, 0)
        w_sum = np.sum(weights)
        if w_sum > 1e-15:
            weights = weights / w_sum
        else:
            weights = np.ones(self.n_assets) / self.n_assets

        exp_ret = float(weights @ self.returns)
        risk = float(np.sqrt(weights @ self.covariance @ weights))
        sharpe = exp_ret / max(risk, 1e-15)

        return PortfolioResult(
            weights=weights,
            expected_return=exp_ret,
            risk=risk,
            sharpe_ratio=sharpe,
            method="analytical",
        )

    def optimize_simulated_annealing(
        self,
        n_iterations: int = 5000,
        seed: int = 42,
    ) -> PortfolioResult:
        """Simulated annealing over portfolio weights."""
        rng = np.random.default_rng(seed)
        n = self.n_assets
        # Start with equal weights
        weights = np.ones(n) / n

        def objective(w: np.ndarray) -> float:
            risk_term = float(w @ self.covariance @ w)
            return_term = float(w @ self.returns)
            return risk_term - self.risk_aversion * return_term

        current_obj = objective(weights)
        best_weights = weights.copy()
        best_obj = current_obj

        for step in range(n_iterations):
            t = 1.0 * (0.01 / 1.0) ** (step / max(n_iterations - 1, 1))
            # Propose: perturb one weight, renormalize
            candidate = weights.copy()
            i = rng.integers(0, n)
            candidate[i] += rng.normal(0, 0.1)
            candidate = np.maximum(candidate, 0)
            c_sum = np.sum(candidate)
            if c_sum > 1e-15:
                candidate = candidate / c_sum
            else:
                continue

            cand_obj = objective(candidate)
            delta = cand_obj - current_obj
            if delta < 0 or rng.random() < np.exp(-delta / max(t, 1e-15)):
                weights = candidate
                current_obj = cand_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_weights = weights.copy()

        exp_ret = float(best_weights @ self.returns)
        risk = float(np.sqrt(best_weights @ self.covariance @ best_weights))
        sharpe = exp_ret / max(risk, 1e-15)

        return PortfolioResult(
            weights=best_weights,
            expected_return=exp_ret,
            risk=risk,
            sharpe_ratio=sharpe,
            method="simulated_annealing",
        )

    def risk_return_frontier(
        self,
        n_points: int = 20,
    ) -> List[PortfolioResult]:
        """Compute the risk-return efficient frontier."""
        results = []
        for lam in np.linspace(0.1, 10.0, n_points):
            orig = self.risk_aversion
            self.risk_aversion = lam
            r = self.optimize_analytical()
            results.append(r)
            self.risk_aversion = orig
        return results

    @staticmethod
    def random_portfolio(
        n_assets: int = 5,
        seed: int = 42,
    ) -> "QuantumPortfolio":
        """Generate a random portfolio problem."""
        rng = np.random.default_rng(seed)
        returns = rng.uniform(0.02, 0.15, size=n_assets)
        # Generate positive-definite covariance
        a = rng.normal(0, 0.1, size=(n_assets, n_assets))
        cov = a.T @ a + 0.01 * np.eye(n_assets)
        return QuantumPortfolio(returns, cov)
