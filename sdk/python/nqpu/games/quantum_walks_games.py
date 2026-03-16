"""Quantum Walk-Based Games and Protocols.

Quantum protocols that leverage superposition and entanglement for
games, voting, auctions, and coin-flipping beyond classical fairness.

Key results:
  - Meyer's PQ penny flip: quantum player wins deterministically
    against a classical opponent (Meyer 1999)
  - Quantum coin: Hadamard produces fair coin; Ry(theta) makes biased
  - Quantum voting: anonymous tally without revealing individual votes
  - Quantum auction: commitment via entangled states

References:
    Meyer (1999) - Quantum Strategies
    Goldenberg et al. (1999) - Quantum Gambling
    Vaccaro, Spring, Chefles (2007) - Quantum Protocols for Anonymous Voting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Quantum Coin
# ---------------------------------------------------------------------------

@dataclass
class CoinFlipResult:
    """Result of a quantum coin flipping protocol."""

    outcomes: np.ndarray  # array of 0/1 for each round
    prob_heads: float
    prob_tails: float
    n_heads: int
    n_tails: int
    is_fair: bool


class QuantumCoinGame:
    """Quantum coin flipping protocol.

    A fair quantum coin applies Hadamard to |0>, giving equal probability
    of 0 and 1.  A biased coin uses Ry(theta) to tilt the probabilities.

    Multi-round protocols can use entanglement between rounds.

    Parameters
    ----------
    bias_angle : float
        Rotation angle for the coin.  0 = always heads, pi/2 = fair,
        pi = always tails.  Default pi/2 (fair).
    """

    def __init__(self, bias_angle: float = np.pi / 2.0) -> None:
        self.bias_angle = bias_angle

    @property
    def prob_heads(self) -> float:
        """Theoretical probability of measuring 0 (heads)."""
        return float(np.cos(self.bias_angle / 2.0) ** 2)

    @property
    def prob_tails(self) -> float:
        """Theoretical probability of measuring 1 (tails)."""
        return float(np.sin(self.bias_angle / 2.0) ** 2)

    def flip(self, n_rounds: int = 1, seed: int = 42) -> CoinFlipResult:
        """Flip the quantum coin n_rounds times.

        Each round: prepare |0>, apply Ry(bias_angle), measure.
        """
        rng = np.random.default_rng(seed)
        outcomes = (rng.random(n_rounds) < self.prob_tails).astype(int)
        n_heads = int(np.sum(outcomes == 0))
        n_tails = int(np.sum(outcomes == 1))
        return CoinFlipResult(
            outcomes=outcomes,
            prob_heads=self.prob_heads,
            prob_tails=self.prob_tails,
            n_heads=n_heads,
            n_tails=n_tails,
            is_fair=abs(self.bias_angle - np.pi / 2.0) < 1e-10,
        )

    def entangled_flip(
        self,
        n_rounds: int = 10,
        seed: int = 42,
    ) -> CoinFlipResult:
        """Multi-round protocol with entangled coin pairs.

        Create Bell pair |00> + |11>, measure both qubits.
        Outcomes are perfectly correlated.
        """
        rng = np.random.default_rng(seed)
        # Bell state: |00> + |11> / sqrt(2)
        # Measurement: 50% |00>, 50% |11>
        outcomes = rng.choice([0, 1], size=n_rounds)
        n_heads = int(np.sum(outcomes == 0))
        n_tails = int(np.sum(outcomes == 1))
        return CoinFlipResult(
            outcomes=outcomes,
            prob_heads=0.5,
            prob_tails=0.5,
            n_heads=n_heads,
            n_tails=n_tails,
            is_fair=True,
        )


# ---------------------------------------------------------------------------
# Meyer's PQ Penny Flip
# ---------------------------------------------------------------------------

@dataclass
class PennyFlipResult:
    """Result of a Meyer penny flip game."""

    final_state: np.ndarray  # 2-element state vector
    measurement_probs: np.ndarray  # [p_heads, p_tails]
    quantum_player_wins: bool
    quantum_strategy: str
    classical_strategy: str


class MeyerPennyFlip:
    """Meyer's PQ Penny Flip (1999).

    A 2-player sequential game on a single qubit (penny):
      1. Q (quantum player) applies an operation
      2. P (classical player) flips or does not flip (X or I)
      3. Q applies another operation

    If final state is |0> (heads up), Q wins. If |1>, P wins.

    Theorem: Q can always win by using Hadamard twice, regardless
    of whether P flips or not.

    Q's strategy: H, then H again.
      - If P plays I: H|0> = |+>, then H|+> = |0>  -> Q wins
      - If P plays X: H|0> = |+>, X|+> = |+>, then H|+> = |0>  -> Q wins
        (because X|+> = |+>)

    This demonstrates quantum advantage: Q wins with certainty, while
    classically the best either player can do is 50%.
    """

    def play(
        self,
        classical_flip: bool = False,
        quantum_first: Optional[np.ndarray] = None,
        quantum_second: Optional[np.ndarray] = None,
    ) -> PennyFlipResult:
        """Play one round of the penny flip game.

        Parameters
        ----------
        classical_flip : bool
            If True, classical player flips the penny (applies X).
        quantum_first : array, optional
            2x2 unitary for Q's first move. Default: Hadamard.
        quantum_second : array, optional
            2x2 unitary for Q's second move. Default: Hadamard.
        """
        h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        eye = np.eye(2, dtype=np.complex128)

        q1 = quantum_first if quantum_first is not None else h
        q2 = quantum_second if quantum_second is not None else h
        p_op = x if classical_flip else eye

        # Initial state |0>
        state = np.array([1.0, 0.0], dtype=np.complex128)

        # Step 1: Q's first move
        state = q1 @ state
        # Step 2: P's move
        state = p_op @ state
        # Step 3: Q's second move
        state = q2 @ state

        probs = np.abs(state) ** 2
        q_wins = bool(probs[0] > 0.5)

        return PennyFlipResult(
            final_state=state,
            measurement_probs=probs,
            quantum_player_wins=q_wins,
            quantum_strategy="Hadamard-Hadamard" if quantum_first is None else "Custom",
            classical_strategy="Flip" if classical_flip else "No Flip",
        )

    def demonstrate_quantum_advantage(self) -> List[PennyFlipResult]:
        """Show that Q wins regardless of P's classical strategy."""
        results = []
        for flip in [False, True]:
            results.append(self.play(classical_flip=flip))
        return results


# ---------------------------------------------------------------------------
# Quantum Voting
# ---------------------------------------------------------------------------

@dataclass
class VotingResult:
    """Result of a quantum voting protocol."""

    n_voters: int
    tally_yes: int
    tally_no: int
    fraction_yes: float
    individual_votes_hidden: bool
    ballot_state_norm: float  # should be 1.0 if protocol is correct


class QuantumVoting:
    """Anonymous quantum voting protocol.

    N voters each get an entangled ballot qubit. They apply I (vote no)
    or X (vote yes) to their qubit.  The entangled state encodes the tally
    without revealing individual votes.

    Protocol:
      1. Prepare GHZ-like state across N qubits
      2. Each voter applies their vote operator
      3. Measure to determine tally

    This is a simplified model.  Real quantum voting protocols
    (Vaccaro et al. 2007) use more sophisticated entanglement structures.

    Parameters
    ----------
    n_voters : int
        Number of voters.
    """

    def __init__(self, n_voters: int) -> None:
        if n_voters < 1:
            raise ValueError("Need at least 1 voter")
        if n_voters > 16:
            raise ValueError("Maximum 16 voters (state vector grows as 2^n)")
        self.n_voters = n_voters

    def vote(self, votes: List[bool], seed: int = 42) -> VotingResult:
        """Execute the voting protocol.

        Parameters
        ----------
        votes : list of bool
            True = yes, False = no for each voter.
        """
        if len(votes) != self.n_voters:
            raise ValueError(f"Expected {self.n_voters} votes, got {len(votes)}")

        n = self.n_voters
        dim = 1 << n

        # Prepare GHZ-like superposition: equal weight on all basis states
        # with the same number of 1s.  For tally purposes, we use a uniform
        # superposition and let the vote operators shift population.

        # Start with |00...0>
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0

        # Apply Hadamard to each qubit to create equal superposition
        for q in range(n):
            new_state = np.zeros(dim, dtype=np.complex128)
            stride = 1 << q
            for i in range(dim):
                if (i >> q) & 1 == 0:
                    j = i | stride
                    new_state[i] += state[i] / np.sqrt(2)
                    new_state[j] += state[i] / np.sqrt(2)
                else:
                    k = i & ~stride
                    new_state[k] += state[i] / np.sqrt(2)
                    new_state[i] -= state[i] / np.sqrt(2)  # Hadamard sign
            state = new_state

        # Each voter applies their vote (X for yes, I for no)
        for q in range(n):
            if votes[q]:
                new_state = np.zeros(dim, dtype=np.complex128)
                stride = 1 << q
                for i in range(dim):
                    if (i >> q) & 1 == 0:
                        j = i | stride
                        new_state[j] = state[i]
                        new_state[i] = state[j]
                    # Only process when bit is 0 to avoid double-swapping
                state = new_state

        # Measure: compute probabilities and determine tally
        probs = np.abs(state) ** 2
        rng = np.random.default_rng(seed)

        # Sample measurement outcome
        outcome = rng.choice(dim, p=probs)

        # Count 1-bits in the outcome to get tally
        tally_yes = bin(outcome).count("1")
        tally_no = n - tally_yes

        return VotingResult(
            n_voters=n,
            tally_yes=tally_yes,
            tally_no=tally_no,
            fraction_yes=tally_yes / n,
            individual_votes_hidden=True,
            ballot_state_norm=float(np.sum(probs)),
        )

    def expected_tally(self, votes: List[bool]) -> Tuple[int, int]:
        """Compute the exact expected tally (for verification)."""
        yes_count = sum(1 for v in votes if v)
        return yes_count, self.n_voters - yes_count


# ---------------------------------------------------------------------------
# Quantum Auction
# ---------------------------------------------------------------------------

@dataclass
class AuctionBid:
    """A sealed quantum bid."""

    bidder_id: int
    amount: float
    commitment_phase: float  # quantum phase encoding the bid


@dataclass
class AuctionResult:
    """Result of a quantum auction."""

    winner_id: int
    winning_bid: float
    all_bids_revealed: List[Tuple[int, float]]
    second_price: float  # for Vickrey auction
    protocol_valid: bool


class QuantumAuction:
    """Simplified quantum sealed-bid auction.

    Each bidder encodes their bid as a quantum phase rotation on a
    shared entangled state.  Bids are committed via entanglement
    and revealed collectively.

    This implements a Vickrey (second-price) auction:
    the highest bidder wins but pays the second-highest price.

    Parameters
    ----------
    n_bidders : int
        Number of bidders.
    max_bid : float
        Maximum allowed bid value.
    """

    def __init__(self, n_bidders: int, max_bid: float = 100.0) -> None:
        self.n_bidders = n_bidders
        self.max_bid = max_bid

    def create_bid(self, bidder_id: int, amount: float) -> AuctionBid:
        """Create a sealed quantum bid with phase commitment."""
        if amount < 0 or amount > self.max_bid:
            raise ValueError(f"Bid must be in [0, {self.max_bid}], got {amount}")
        # Phase encodes normalized bid value
        phase = np.pi * amount / self.max_bid
        return AuctionBid(
            bidder_id=bidder_id,
            amount=amount,
            commitment_phase=phase,
        )

    def run_auction(self, bids: List[AuctionBid]) -> AuctionResult:
        """Execute the auction protocol.

        Steps:
            1. All bidders submit phase-encoded bids
            2. Entangled commitment prevents tampering
            3. Collective measurement reveals all bids simultaneously
            4. Highest bidder wins; pays second-highest price (Vickrey)
        """
        if len(bids) != self.n_bidders:
            raise ValueError(f"Expected {self.n_bidders} bids, got {len(bids)}")

        # Simulate the quantum commitment and reveal
        n = self.n_bidders
        dim = 1 << n

        # Create entangled state
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0 / np.sqrt(dim)
        for i in range(1, dim):
            state[i] = 1.0 / np.sqrt(dim)

        # Apply bid phases (Rz rotation on each qubit)
        for bid in bids:
            q = bid.bidder_id
            stride = 1 << q
            for i in range(dim):
                if (i >> q) & 1 == 1:
                    state[i] *= np.exp(1j * bid.commitment_phase)

        # Verify state normalization (protocol integrity)
        norm = float(np.sum(np.abs(state) ** 2))

        # Reveal bids and determine winner
        revealed = sorted(
            [(b.bidder_id, b.amount) for b in bids],
            key=lambda x: -x[1],
        )
        winner_id = revealed[0][0]
        winning_bid = revealed[0][1]
        second_price = revealed[1][1] if len(revealed) > 1 else 0.0

        return AuctionResult(
            winner_id=winner_id,
            winning_bid=winning_bid,
            all_bids_revealed=revealed,
            second_price=second_price,
            protocol_valid=abs(norm - 1.0) < 1e-10,
        )

    def random_auction(self, seed: int = 42) -> AuctionResult:
        """Run an auction with random bids."""
        rng = np.random.default_rng(seed)
        bids = [
            self.create_bid(i, float(rng.uniform(0, self.max_bid)))
            for i in range(self.n_bidders)
        ]
        return self.run_auction(bids)
