"""Quantum Game Theory -- 2-player quantum games with entangled strategies.

Implements the Eisert-Wilkens-Lewenstein (EWL) protocol for quantizing
classical 2x2 games. Players choose unitary strategies from SU(2),
a shared entangling operator J(gamma) creates correlations, and payoffs
are computed from measurement probabilities on the final state.

Key result: In the quantum Prisoner's Dilemma, the "quantum miracle move"
Q = U(0, pi/2) is a Nash equilibrium that achieves the Pareto-optimal
cooperative payoff (3,3), resolving the classical dilemma where rational
players are stuck at (1,1).

References:
    Eisert, Wilkens, Lewenstein (1999) - Quantum Games and Quantum Strategies
    Meyer (1999) - Quantum Strategies
    Marinatto, Weber (2000) - Quantum Approach to Static Games
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Strategy representation
# ---------------------------------------------------------------------------

@dataclass
class QuantumStrategy:
    """A quantum strategy in SU(2), parametrized by (theta, phi).

    The unitary is:
        U(theta, phi) = [[e^{i*phi} cos(theta/2),   sin(theta/2)],
                         [-sin(theta/2),             e^{-i*phi} cos(theta/2)]]

    Classical strategies are special cases:
        Cooperate (Identity): theta=0, phi=0
        Defect (X gate):      theta=pi, phi=0
    """

    theta: float
    phi: float
    name: str = ""

    def unitary(self) -> np.ndarray:
        """Return the 2x2 unitary matrix."""
        c = np.cos(self.theta / 2.0)
        s = np.sin(self.theta / 2.0)
        eip = np.exp(1j * self.phi)
        eim = np.exp(-1j * self.phi)
        return np.array([
            [eip * c, s],
            [-s, eim * c],
        ], dtype=np.complex128)


def cooperate() -> QuantumStrategy:
    """Classical cooperate strategy (identity)."""
    return QuantumStrategy(theta=0.0, phi=0.0, name="Cooperate")


def defect() -> QuantumStrategy:
    """Classical defect strategy (X gate / bit-flip)."""
    return QuantumStrategy(theta=np.pi, phi=0.0, name="Defect")


def hadamard_strategy() -> QuantumStrategy:
    """Hadamard strategy -- equal superposition of C and D."""
    return QuantumStrategy(theta=np.pi / 2.0, phi=0.0, name="Hadamard")


def quantum_miracle_move() -> QuantumStrategy:
    """The quantum miracle move Q = U(0, pi/2).

    In the maximally-entangled quantum Prisoner's Dilemma, (Q, Q) is a
    Nash equilibrium with payoff (3, 3) -- the Pareto-optimal cooperative
    outcome that classical players cannot reach.
    """
    return QuantumStrategy(theta=0.0, phi=np.pi / 2.0, name="Quantum Miracle (Q)")


# ---------------------------------------------------------------------------
# Game result
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    """Outcome of playing a quantum game."""

    payoff_p1: float
    payoff_p2: float
    outcome_probs: np.ndarray  # shape (2, 2): [action_p1][action_p2]
    is_pareto_optimal: bool
    classical_nash_p1: float
    classical_nash_p2: float
    quantum_advantage_p1: float
    quantum_advantage_p2: float
    strategy_p1: Optional[QuantumStrategy] = None
    strategy_p2: Optional[QuantumStrategy] = None


# ---------------------------------------------------------------------------
# Core quantum game engine
# ---------------------------------------------------------------------------

class QuantumGame:
    """A 2-player quantum game using the EWL protocol.

    Parameters
    ----------
    name : str
        Human-readable name.
    payoff_p1 : array-like, shape (2, 2)
        Payoff matrix for player 1.  payoff_p1[i][j] is P1's payoff when
        P1 plays action i and P2 plays action j.  Convention: 0 = first
        action (e.g. Cooperate), 1 = second action (e.g. Defect).
    payoff_p2 : array-like, shape (2, 2)
        Payoff matrix for player 2.
    gamma : float
        Entanglement parameter. gamma=0 recovers the classical game,
        gamma=pi/2 is maximal entanglement.
    """

    def __init__(
        self,
        name: str,
        payoff_p1: np.ndarray,
        payoff_p2: np.ndarray,
        gamma: float = np.pi / 2.0,
    ) -> None:
        self.name = name
        self.payoff_p1 = np.asarray(payoff_p1, dtype=np.float64)
        self.payoff_p2 = np.asarray(payoff_p2, dtype=np.float64)
        self.gamma = gamma

    # -- EWL protocol implementation ----------------------------------------

    def _entangler(self) -> np.ndarray:
        """J(gamma) = exp(i * gamma/2 * X x X) as a 4x4 matrix."""
        c = np.cos(self.gamma / 2.0)
        s = np.sin(self.gamma / 2.0)
        # X x X swaps |00><->|11> and |01><->|10>
        j = np.zeros((4, 4), dtype=np.complex128)
        j[0, 0] = c
        j[0, 3] = 1j * s
        j[1, 1] = c
        j[1, 2] = 1j * s
        j[2, 1] = 1j * s
        j[2, 2] = c
        j[3, 0] = 1j * s
        j[3, 3] = c
        return j

    def _disentangler(self) -> np.ndarray:
        """J_dagger(gamma) = exp(-i * gamma/2 * X x X)."""
        return self._entangler().conj().T

    def play(
        self,
        strategy_p1: QuantumStrategy,
        strategy_p2: QuantumStrategy,
    ) -> GameResult:
        """Execute the EWL protocol and return payoffs.

        Steps:
            1. Start in |00>
            2. Apply entangler J(gamma)
            3. Player 1 applies U1 on qubit 0, Player 2 applies U2 on qubit 1
            4. Apply disentangler J_dagger
            5. Measure in computational basis
        """
        # |00> state
        state = np.zeros(4, dtype=np.complex128)
        state[0] = 1.0

        # Step 2: entangle
        state = self._entangler() @ state

        # Step 3: local strategies  U1 x U2
        u1 = strategy_p1.unitary()
        u2 = strategy_p2.unitary()
        u_total = np.kron(u1, u2)
        state = u_total @ state

        # Step 4: disentangle
        state = self._disentangler() @ state

        # Step 5: measurement probabilities
        probs = np.abs(state) ** 2
        # |00>=CC, |01>=CD, |10>=DC, |11>=DD
        outcome_probs = probs.reshape(2, 2)

        payoff_1 = float(np.sum(outcome_probs * self.payoff_p1))
        payoff_2 = float(np.sum(outcome_probs * self.payoff_p2))

        # Classical Nash analysis
        cn1 = self._classical_nash_payoff(player=1)
        cn2 = self._classical_nash_payoff(player=2)

        # Pareto optimality
        classical_outcomes = [
            (self.payoff_p1[i, j], self.payoff_p2[i, j])
            for i in range(2)
            for j in range(2)
        ]
        is_pareto = not any(
            p1 >= payoff_1 and p2 >= payoff_2 and (p1 > payoff_1 or p2 > payoff_2)
            for p1, p2 in classical_outcomes
        )

        return GameResult(
            payoff_p1=payoff_1,
            payoff_p2=payoff_2,
            outcome_probs=outcome_probs,
            is_pareto_optimal=is_pareto,
            classical_nash_p1=cn1,
            classical_nash_p2=cn2,
            quantum_advantage_p1=payoff_1 - cn1,
            quantum_advantage_p2=payoff_2 - cn2,
            strategy_p1=strategy_p1,
            strategy_p2=strategy_p2,
        )

    def _classical_nash_payoff(self, player: int) -> float:
        """Find the best pure-strategy Nash equilibrium payoff for *player*."""
        payoff_mat = self.payoff_p1 if player == 1 else self.payoff_p2
        best = float("-inf")
        for i in range(2):
            for j in range(2):
                other_i = 1 - i
                other_j = 1 - j
                p1_no_deviate = self.payoff_p1[i, j] >= self.payoff_p1[other_i, j]
                p2_no_deviate = self.payoff_p2[i, j] >= self.payoff_p2[i, other_j]
                if p1_no_deviate and p2_no_deviate:
                    best = max(best, float(payoff_mat[i, j]))
        return best

    # -- Strategy search ----------------------------------------------------

    def best_response_p1(
        self,
        strategy_p2: QuantumStrategy,
        grid_steps: int = 50,
    ) -> Tuple[QuantumStrategy, float]:
        """Grid-search for P1's best response to a fixed P2 strategy."""
        best_payoff = float("-inf")
        best_strat = cooperate()
        for ti in range(grid_steps + 1):
            theta = np.pi * ti / grid_steps
            for pi_ in range(grid_steps + 1):
                phi = np.pi * pi_ / grid_steps
                s = QuantumStrategy(theta=theta, phi=phi, name="search")
                r = self.play(s, strategy_p2)
                if r.payoff_p1 > best_payoff:
                    best_payoff = r.payoff_p1
                    best_strat = QuantumStrategy(theta=theta, phi=phi, name="BR-P1")
        return best_strat, best_payoff

    def best_response_p2(
        self,
        strategy_p1: QuantumStrategy,
        grid_steps: int = 50,
    ) -> Tuple[QuantumStrategy, float]:
        """Grid-search for P2's best response to a fixed P1 strategy."""
        best_payoff = float("-inf")
        best_strat = cooperate()
        for ti in range(grid_steps + 1):
            theta = np.pi * ti / grid_steps
            for pi_ in range(grid_steps + 1):
                phi = np.pi * pi_ / grid_steps
                s = QuantumStrategy(theta=theta, phi=phi, name="search")
                r = self.play(strategy_p1, s)
                if r.payoff_p2 > best_payoff:
                    best_payoff = r.payoff_p2
                    best_strat = QuantumStrategy(theta=theta, phi=phi, name="BR-P2")
        return best_strat, best_payoff

    def find_nash_equilibrium(
        self,
        max_iterations: int = 20,
        grid_steps: int = 50,
    ) -> Tuple[QuantumStrategy, QuantumStrategy, GameResult]:
        """Iterated best-response search for a quantum Nash equilibrium."""
        s1 = cooperate()
        s2 = cooperate()
        for _ in range(max_iterations):
            new_s1, _ = self.best_response_p1(s2, grid_steps)
            new_s2, _ = self.best_response_p2(new_s1, grid_steps)
            if (
                abs(new_s1.theta - s1.theta) < 1e-4
                and abs(new_s1.phi - s1.phi) < 1e-4
                and abs(new_s2.theta - s2.theta) < 1e-4
                and abs(new_s2.phi - s2.phi) < 1e-4
            ):
                s1, s2 = new_s1, new_s2
                break
            s1, s2 = new_s1, new_s2
        return s1, s2, self.play(s1, s2)

    # -- Payoff analysis helpers --------------------------------------------

    def payoff_matrix_data(
        self,
        grid_steps: int = 20,
    ) -> dict:
        """Compute payoff landscape over the strategy space.

        Returns a dict with 'theta', 'phi', 'payoff_p1', 'payoff_p2' arrays
        suitable for heatmap visualization.
        """
        n = grid_steps + 1
        thetas = np.linspace(0, np.pi, n)
        phis = np.linspace(0, np.pi, n)
        p1_grid = np.zeros((n, n))
        p2_grid = np.zeros((n, n))
        fixed_s2 = defect()
        for i, th in enumerate(thetas):
            for j, ph in enumerate(phis):
                s1 = QuantumStrategy(theta=th, phi=ph)
                r = self.play(s1, fixed_s2)
                p1_grid[i, j] = r.payoff_p1
                p2_grid[i, j] = r.payoff_p2
        return {
            "theta": thetas,
            "phi": phis,
            "payoff_p1": p1_grid,
            "payoff_p2": p2_grid,
        }


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------

@dataclass
class TournamentResult:
    """Round-robin tournament result."""

    rankings: List[Tuple[str, float]]  # (strategy_name, total_payoff) sorted desc
    matchups: np.ndarray  # shape (n, n) payoff matrix


class QuantumTournament:
    """Round-robin tournament of quantum strategies in a given game."""

    def __init__(self, game: QuantumGame, strategies: List[QuantumStrategy]) -> None:
        self.game = game
        self.strategies = strategies

    def run(self) -> TournamentResult:
        n = len(self.strategies)
        matchups = np.zeros((n, n))
        payoffs = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    r = self.game.play(self.strategies[i], self.strategies[j])
                    matchups[i, j] = r.payoff_p1
                    payoffs[i] += r.payoff_p1
        rankings = sorted(
            [(s.name, float(payoffs[k])) for k, s in enumerate(self.strategies)],
            key=lambda x: -x[1],
        )
        return TournamentResult(rankings=rankings, matchups=matchups)


# ---------------------------------------------------------------------------
# Pre-built games
# ---------------------------------------------------------------------------

class PrisonersDilemma(QuantumGame):
    """Quantum Prisoner's Dilemma (Eisert et al. 1999).

    Payoffs: (C,C)=(3,3), (C,D)=(0,5), (D,C)=(5,0), (D,D)=(1,1).
    Classical Nash equilibrium: (D,D) with payoff (1,1).
    Quantum Nash equilibrium (maximal gamma): (Q,Q) with payoff (3,3).
    """

    def __init__(self, gamma: float = np.pi / 2.0) -> None:
        super().__init__(
            name="Prisoner's Dilemma",
            payoff_p1=np.array([[3.0, 0.0], [5.0, 1.0]]),
            payoff_p2=np.array([[3.0, 5.0], [0.0, 1.0]]),
            gamma=gamma,
        )


class BattleOfSexes(QuantumGame):
    """Quantum Battle of the Sexes -- coordination game.

    P1 prefers (A,A)=(3,2), P2 prefers (B,B)=(2,3).
    Classical: two pure-strategy NE, one mixed-strategy NE.
    Quantum: unique fair equilibrium with symmetric payoff.
    """

    def __init__(self, gamma: float = np.pi / 2.0) -> None:
        super().__init__(
            name="Battle of the Sexes",
            payoff_p1=np.array([[3.0, 0.0], [0.0, 2.0]]),
            payoff_p2=np.array([[2.0, 0.0], [0.0, 3.0]]),
            gamma=gamma,
        )


class Chicken(QuantumGame):
    """Quantum Chicken (Hawk-Dove) game -- anti-coordination.

    Payoffs: (Dove,Dove)=(3,3), (Dove,Hawk)=(1,4),
             (Hawk,Dove)=(4,1), (Hawk,Hawk)=(0,0).
    """

    def __init__(self, gamma: float = np.pi / 2.0) -> None:
        super().__init__(
            name="Chicken",
            payoff_p1=np.array([[3.0, 1.0], [4.0, 0.0]]),
            payoff_p2=np.array([[3.0, 4.0], [1.0, 0.0]]),
            gamma=gamma,
        )


class MatchingPennies(QuantumGame):
    """Quantum Matching Pennies -- zero-sum game.

    P1 wins if both choose same, P2 wins if different.
    Classical: unique mixed-strategy NE with expected payoff 0.
    Quantum: richer equilibrium structure.
    """

    def __init__(self, gamma: float = np.pi / 2.0) -> None:
        super().__init__(
            name="Matching Pennies",
            payoff_p1=np.array([[1.0, -1.0], [-1.0, 1.0]]),
            payoff_p2=np.array([[-1.0, 1.0], [1.0, -1.0]]),
            gamma=gamma,
        )
