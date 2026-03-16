"""N-Player Quantum Games -- public goods, minority game, EWL extensions.

Generalizes 2-player quantum game theory to N players using the
N-player Eisert-Wilkens-Lewenstein (EWL) protocol. Each player applies
an SU(2) unitary strategy to their qubit in an N-qubit entangled state.

Key results:
  - N-player Prisoner's Dilemma: quantum strategies can resolve the
    free-rider problem in public goods provision
  - Minority game: quantum entanglement enables better coordination
    than classical mixed strategies
  - Quantum bargaining: fair division achievable via entangled strategies

References:
    Benjamin & Hayden (2001) - Multiplayer Quantum Games
    Eisert et al. (1999) - Quantum Games and Quantum Strategies
    Du et al. (2002) - Experimental Realization of Quantum Games
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable


# ---------------------------------------------------------------------------
# Strategy representation
# ---------------------------------------------------------------------------

@dataclass
class NPlayerStrategy:
    """Strategy for an N-player quantum game.

    Parametrized as a 2x2 SU(2) unitary matrix:
        U(theta, phi) = [[e^{i*phi} cos(theta/2),   sin(theta/2)],
                         [-sin(theta/2),             e^{-i*phi} cos(theta/2)]]

    Classical strategies:
        Cooperate (Identity): theta=0, phi=0
        Defect (X-like):      theta=pi, phi=0
    """

    theta: float = 0.0
    phi: float = 0.0
    player_id: int = 0
    label: str = ""

    def unitary(self) -> np.ndarray:
        """2x2 SU(2) strategy matrix."""
        c = np.cos(self.theta / 2.0)
        s = np.sin(self.theta / 2.0)
        eip = np.exp(1j * self.phi)
        eim = np.exp(-1j * self.phi)
        return np.array([
            [eip * c, s],
            [-s, eim * c],
        ], dtype=np.complex128)


def n_cooperate(player_id: int = 0) -> NPlayerStrategy:
    """Classical cooperate strategy (identity) for N-player games."""
    return NPlayerStrategy(theta=0.0, phi=0.0, player_id=player_id, label="Cooperate")


def n_defect(player_id: int = 0) -> NPlayerStrategy:
    """Classical defect strategy (X gate) for N-player games."""
    return NPlayerStrategy(theta=np.pi, phi=0.0, player_id=player_id, label="Defect")


def n_quantum(player_id: int = 0) -> NPlayerStrategy:
    """Quantum miracle move Q = U(0, pi/2) for N-player games."""
    return NPlayerStrategy(theta=0.0, phi=np.pi / 2.0, player_id=player_id, label="Quantum")


# ---------------------------------------------------------------------------
# Game result
# ---------------------------------------------------------------------------

@dataclass
class NPlayerResult:
    """Result of an N-player game."""

    payoffs: np.ndarray  # shape (n_players,)
    strategies: List[NPlayerStrategy]
    final_state: np.ndarray
    measurement_probs: np.ndarray


# ---------------------------------------------------------------------------
# N-Player EWL protocol
# ---------------------------------------------------------------------------

def _multi_kron(matrices: List[np.ndarray]) -> np.ndarray:
    """Compute the Kronecker product of a list of matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


class NPlayerEWL:
    """N-player Eisert-Wilkens-Lewenstein quantum game.

    Generalizes 2-player EWL to N players with entangling operator J.

    The protocol:
        1. Start in |00...0> (N qubits)
        2. Apply entangling operator J = exp(i*gamma/2 * X^{otimes n})
        3. Each player applies their SU(2) strategy to their qubit
        4. Apply disentangling operator J^dagger
        5. Measure in computational basis, compute payoffs

    Parameters
    ----------
    n_players : int
        Number of players (and qubits).
    payoff_tensor : ndarray, shape (2,)*n_players + (n_players,)
        Payoff tensor. Entry [i0, i1, ..., i_{n-1}, k] is the payoff
        to player k when the outcome is (i0, i1, ..., i_{n-1}).
    gamma : float
        Entanglement parameter. 0 = classical, pi/2 = maximal.
    """

    def __init__(
        self,
        n_players: int,
        payoff_tensor: np.ndarray,
        gamma: float = np.pi / 2,
    ) -> None:
        if n_players < 2:
            raise ValueError("Need at least 2 players")
        if n_players > 10:
            raise ValueError("Maximum 10 players (state vector grows as 2^n)")
        self.n_players = n_players
        self.payoff_tensor = np.asarray(payoff_tensor, dtype=np.float64)
        self.gamma = gamma

        expected_shape = tuple([2] * n_players) + (n_players,)
        if self.payoff_tensor.shape != expected_shape:
            raise ValueError(
                f"Payoff tensor shape {self.payoff_tensor.shape} != "
                f"expected {expected_shape}"
            )

    def _entangling_operator(self) -> np.ndarray:
        """Build N-qubit entangling operator J = exp(i*gamma/2 * X^{otimes n}).

        X^{otimes n} has eigenvalues +1 and -1. We construct the matrix
        explicitly and exponentiate via spectral decomposition:
            exp(i*theta*M) = cos(theta)*I + i*sin(theta)*M
        when M^2 = I (which holds for X^{otimes n}).
        """
        n = self.n_players
        dim = 1 << n

        # Build X^{otimes n}
        x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        x_n = _multi_kron([x] * n)

        # Since (X^{otimes n})^2 = I, use:
        # exp(i*gamma/2 * X^n) = cos(gamma/2)*I + i*sin(gamma/2)*X^n
        c = np.cos(self.gamma / 2.0)
        s = np.sin(self.gamma / 2.0)
        j_op = c * np.eye(dim, dtype=np.complex128) + 1j * s * x_n
        return j_op

    def _disentangling_operator(self) -> np.ndarray:
        """J^dagger = exp(-i*gamma/2 * X^{otimes n})."""
        return self._entangling_operator().conj().T

    def play(self, strategies: List[NPlayerStrategy]) -> NPlayerResult:
        """Play the game with given strategies.

        Parameters
        ----------
        strategies : list of NPlayerStrategy
            One strategy per player.

        Returns
        -------
        NPlayerResult with payoffs, final state, and measurement probabilities.
        """
        if len(strategies) != self.n_players:
            raise ValueError(
                f"Expected {self.n_players} strategies, got {len(strategies)}"
            )

        n = self.n_players
        dim = 1 << n

        # |00...0> initial state
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0

        # Step 2: entangle
        state = self._entangling_operator() @ state

        # Step 3: each player applies their strategy (tensor product)
        unitaries = [s.unitary() for s in strategies]
        u_total = _multi_kron(unitaries)
        state = u_total @ state

        # Step 4: disentangle
        state = self._disentangling_operator() @ state

        # Step 5: measurement probabilities
        probs = np.abs(state) ** 2
        # Ensure normalization
        probs = probs / np.sum(probs)

        # Compute payoffs: sum over all outcomes
        payoffs = np.zeros(n, dtype=np.float64)
        for idx in range(dim):
            # Convert index to binary tuple
            outcome = tuple((idx >> q) & 1 for q in range(n))
            for k in range(n):
                payoffs[k] += probs[idx] * self.payoff_tensor[outcome + (k,)]

        return NPlayerResult(
            payoffs=payoffs,
            strategies=list(strategies),
            final_state=state,
            measurement_probs=probs,
        )

    def nash_search(
        self,
        resolution: int = 20,
        rng: Optional[np.random.Generator] = None,
    ) -> List[List[NPlayerStrategy]]:
        """Brute force search for approximate Nash equilibria on strategy grid.

        For each player, tries all strategies on a theta-phi grid.
        A strategy profile is a Nash equilibrium if no player can improve
        their payoff by unilaterally deviating.

        Parameters
        ----------
        resolution : int
            Grid resolution per dimension.
        rng : numpy Generator, optional
            Not used (deterministic grid search), kept for API consistency.

        Returns
        -------
        List of strategy profiles that are approximate Nash equilibria.
        """
        n = self.n_players
        thetas = np.linspace(0, np.pi, resolution)
        phis = np.linspace(0, np.pi, resolution)

        # For tractability with N players, do iterated best-response
        # starting from multiple initial strategy profiles
        nash_candidates: List[List[NPlayerStrategy]] = []

        # Start from a few canonical strategy profiles
        starts = []
        # All cooperate
        starts.append([n_cooperate(i) for i in range(n)])
        # All defect
        starts.append([n_defect(i) for i in range(n)])
        # All quantum
        starts.append([n_quantum(i) for i in range(n)])

        for initial in starts:
            current = list(initial)
            converged = False

            for _iteration in range(10):
                changed = False
                for player in range(n):
                    best_payoff = float("-inf")
                    best_strat = current[player]

                    for ti in range(len(thetas)):
                        for pi_ in range(len(phis)):
                            candidate = NPlayerStrategy(
                                theta=thetas[ti],
                                phi=phis[pi_],
                                player_id=player,
                                label="search",
                            )
                            test_strats = list(current)
                            test_strats[player] = candidate
                            result = self.play(test_strats)
                            if result.payoffs[player] > best_payoff + 1e-8:
                                best_payoff = result.payoffs[player]
                                best_strat = candidate
                                changed = True

                    current[player] = best_strat

                if not changed:
                    converged = True
                    break

            # Check if this is actually a Nash equilibrium
            is_nash = True
            base_result = self.play(current)
            for player in range(n):
                for ti in range(len(thetas)):
                    for pi_ in range(len(phis)):
                        deviant = NPlayerStrategy(
                            theta=thetas[ti],
                            phi=phis[pi_],
                            player_id=player,
                        )
                        test = list(current)
                        test[player] = deviant
                        dev_result = self.play(test)
                        if dev_result.payoffs[player] > base_result.payoffs[player] + 1e-4:
                            is_nash = False
                            break
                    if not is_nash:
                        break
                if not is_nash:
                    break

            if is_nash:
                # Check we haven't already found an equivalent NE
                is_duplicate = False
                for existing in nash_candidates:
                    if all(
                        abs(current[i].theta - existing[i].theta) < 0.1
                        and abs(current[i].phi - existing[i].phi) < 0.1
                        for i in range(n)
                    ):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    nash_candidates.append(current)

        return nash_candidates


# ---------------------------------------------------------------------------
# Public Goods Game
# ---------------------------------------------------------------------------

class PublicGoodsGame:
    """Quantum public goods game.

    N players choose contribution levels. Total contributions multiplied
    by factor r and shared equally. Quantum version uses entangled strategies.

    Classical payoff when player i plays action a_i in {0=not contribute, 1=contribute}:
        payoff_i = endowment - a_i * endowment + (r * sum(a_j) * endowment) / n

    Parameters
    ----------
    n_players : int
        Number of players.
    multiplication_factor : float
        Public goods multiplication factor r. Must satisfy 1 < r < n for
        the social dilemma to exist.
    endowment : float
        Initial endowment per player.
    """

    def __init__(
        self,
        n_players: int,
        multiplication_factor: float = 2.0,
        endowment: float = 1.0,
    ) -> None:
        if n_players < 2:
            raise ValueError("Need at least 2 players")
        if n_players > 10:
            raise ValueError("Maximum 10 players")
        self.n_players = n_players
        self.multiplication_factor = multiplication_factor
        self.endowment = endowment
        self._ewl = NPlayerEWL(
            n_players=n_players,
            payoff_tensor=self._payoff_tensor(),
        )

    def _payoff_tensor(self) -> np.ndarray:
        """Build payoff tensor for public goods game.

        For each outcome (a_0, ..., a_{n-1}) where a_i in {0, 1}:
            payoff_i = endowment * (1 - a_i) + (r * sum(a_j) * endowment) / n
        """
        n = self.n_players
        r = self.multiplication_factor
        e = self.endowment
        shape = tuple([2] * n) + (n,)
        tensor = np.zeros(shape, dtype=np.float64)

        for idx in range(1 << n):
            outcome = tuple((idx >> q) & 1 for q in range(n))
            total_contributions = sum(outcome)
            public_return = r * total_contributions * e / n
            for k in range(n):
                private = e * (1 - outcome[k])
                tensor[outcome + (k,)] = private + public_return

        return tensor

    def play(self, strategies: List[NPlayerStrategy]) -> NPlayerResult:
        """Play quantum public goods game."""
        return self._ewl.play(strategies)

    def social_optimum(self) -> float:
        """Compute the social optimum payoff (per player).

        When all players contribute, each gets r * endowment.
        """
        return self.multiplication_factor * self.endowment

    def free_rider_payoff(self) -> float:
        """Payoff when all cooperate except one defector.

        The defector keeps their endowment AND gets the public return.
        """
        n = self.n_players
        r = self.multiplication_factor
        e = self.endowment
        contributors = n - 1
        public_return = r * contributors * e / n
        return e + public_return  # keeps endowment + gets share

    def cooperator_payoff_with_one_defector(self) -> float:
        """Payoff for cooperators when one player defects."""
        n = self.n_players
        r = self.multiplication_factor
        e = self.endowment
        contributors = n - 1
        public_return = r * contributors * e / n
        return public_return  # gave up endowment, gets share


# ---------------------------------------------------------------------------
# Minority Game
# ---------------------------------------------------------------------------

class MinorityGame:
    """Quantum minority game.

    Players choose one of two options (0 or 1). Those in the minority win
    (payoff = 1), those in the majority lose (payoff = 0). If tied, nobody
    wins.

    Quantum strategies can achieve better coordination than classical
    mixed strategies, approaching the theoretical maximum where the
    minority has exactly floor(n/2) players.

    Parameters
    ----------
    n_players : int
        Number of players (should be odd for a clear minority).
    """

    def __init__(self, n_players: int) -> None:
        if n_players < 3:
            raise ValueError("Need at least 3 players for minority game")
        if n_players > 10:
            raise ValueError("Maximum 10 players")
        self.n_players = n_players
        self._ewl = NPlayerEWL(
            n_players=n_players,
            payoff_tensor=self._payoff_tensor(),
        )

    def _payoff_tensor(self) -> np.ndarray:
        """Build minority game payoff tensor.

        For each outcome, players in the strict minority get payoff 1,
        others get 0. If exactly tied (even n), nobody wins.
        """
        n = self.n_players
        shape = tuple([2] * n) + (n,)
        tensor = np.zeros(shape, dtype=np.float64)

        for idx in range(1 << n):
            outcome = tuple((idx >> q) & 1 for q in range(n))
            n_ones = sum(outcome)
            n_zeros = n - n_ones

            if n_ones == n_zeros:
                # Tied: nobody wins
                continue
            elif n_ones < n_zeros:
                # Ones are minority
                for k in range(n):
                    if outcome[k] == 1:
                        tensor[outcome + (k,)] = 1.0
            else:
                # Zeros are minority
                for k in range(n):
                    if outcome[k] == 0:
                        tensor[outcome + (k,)] = 1.0

        return tensor

    def play(self, strategies: List[NPlayerStrategy]) -> NPlayerResult:
        """Play quantum minority game."""
        return self._ewl.play(strategies)

    def classical_expected_payoff(self) -> float:
        """Expected payoff under random classical play.

        Each player independently chooses 0 or 1 with probability 1/2.
        """
        n = self.n_players
        total_expected = 0.0

        for idx in range(1 << n):
            prob = 1.0 / (1 << n)
            outcome = tuple((idx >> q) & 1 for q in range(n))
            n_ones = sum(outcome)
            n_zeros = n - n_ones

            if n_ones < n_zeros:
                total_expected += prob * n_ones
            elif n_zeros < n_ones:
                total_expected += prob * n_zeros
            # If tied, no contribution

        # Per-player expected payoff
        return total_expected / n

    def optimal_minority_size(self) -> int:
        """Optimal minority size: floor(n/2)."""
        return self.n_players // 2


# ---------------------------------------------------------------------------
# Quantum Bargaining
# ---------------------------------------------------------------------------

class QuantumBargaining:
    """Quantum bargaining/negotiation game.

    N players bargain over a divisible resource. Each player's strategy
    determines their demand. If total demands exceed the resource, all
    players get 0 (disagreement). Otherwise, each gets their demand.

    The quantum version uses entanglement to enable fair coordination:
    quantum strategies can achieve the Nash bargaining solution.

    Parameters
    ----------
    n_players : int
        Number of players.
    resource : float
        Total divisible resource.
    """

    def __init__(
        self,
        n_players: int,
        resource: float = 1.0,
    ) -> None:
        if n_players < 2:
            raise ValueError("Need at least 2 players")
        if n_players > 10:
            raise ValueError("Maximum 10 players")
        self.n_players = n_players
        self.resource = resource
        self._ewl = NPlayerEWL(
            n_players=n_players,
            payoff_tensor=self._payoff_tensor(),
        )

    def _payoff_tensor(self) -> np.ndarray:
        """Build bargaining payoff tensor.

        Action 0 = demand fair share (resource/n)
        Action 1 = demand entire resource

        If total demands <= resource, each gets their demand.
        If total demands > resource, all get 0.
        """
        n = self.n_players
        fair_share = self.resource / n
        shape = tuple([2] * n) + (n,)
        tensor = np.zeros(shape, dtype=np.float64)

        for idx in range(1 << n):
            outcome = tuple((idx >> q) & 1 for q in range(n))
            demands = []
            for k in range(n):
                if outcome[k] == 0:
                    demands.append(fair_share)
                else:
                    demands.append(self.resource)

            total_demand = sum(demands)
            if total_demand <= self.resource + 1e-10:
                for k in range(n):
                    tensor[outcome + (k,)] = demands[k]
            # else: all get 0 (disagreement)

        return tensor

    def play(self, strategies: List[NPlayerStrategy]) -> NPlayerResult:
        """Play quantum bargaining game."""
        return self._ewl.play(strategies)

    def nash_bargaining_solution(self) -> np.ndarray:
        """Compute classical Nash bargaining solution.

        With symmetric players and equal disagreement payoffs (0),
        the NBS gives equal shares: resource / n for each player.
        """
        return np.full(self.n_players, self.resource / self.n_players)

    def total_payoff(self, result: NPlayerResult) -> float:
        """Compute total payoff across all players."""
        return float(np.sum(result.payoffs))

    def is_efficient(self, result: NPlayerResult, tol: float = 0.01) -> bool:
        """Check if outcome is Pareto-efficient (total payoff close to resource)."""
        return float(np.sum(result.payoffs)) >= self.resource * (1.0 - tol)
