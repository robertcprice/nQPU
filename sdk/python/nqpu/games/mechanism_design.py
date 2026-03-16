"""Quantum Mechanism Design -- VCG auctions and incentive compatibility.

Applies quantum information processing to mechanism design, the field
of economics concerned with designing institutions/rules that achieve
desired outcomes even when participants act strategically.

Key components:
  - Quantum valuations: player preferences encoded as quantum superpositions
  - Quantum VCG mechanism: truthful allocation with quantum-enhanced welfare
  - Revenue-optimal auctions: quantum strategies for revenue maximization
  - Incentive analysis: quantifying how well mechanisms resist manipulation

References:
    Vickrey (1961) - Counterspeculation, Auctions, and Competitive Sealed Tenders
    Clarke (1971) - Multipart Pricing of Public Goods
    Groves (1973) - Incentives in Teams
    Chen et al. (2008) - Quantum Auction
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Quantum Valuation
# ---------------------------------------------------------------------------

@dataclass
class QuantumValuation:
    """Player's valuation in quantum superposition.

    A quantum valuation represents uncertainty about a player's true value
    using quantum amplitudes. The Born rule gives the probability of each
    valuation level, and interference between amplitudes enables strategies
    not possible with classical probability distributions.

    Parameters
    ----------
    values : ndarray
        Possible valuation levels.
    amplitudes : ndarray
        Complex amplitudes (squared magnitudes give probabilities).
    player_id : int
        Player identifier.
    """

    values: np.ndarray
    amplitudes: np.ndarray
    player_id: int = 0

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        self.amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
        if len(self.values) != len(self.amplitudes):
            raise ValueError(
                f"values length ({len(self.values)}) != "
                f"amplitudes length ({len(self.amplitudes)})"
            )
        # Normalize amplitudes
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 1e-15:
            self.amplitudes = self.amplitudes / norm

    def expected_value(self) -> float:
        """Expected valuation under Born rule probabilities."""
        probs = np.abs(self.amplitudes) ** 2
        return float(np.sum(probs * self.values))

    def variance(self) -> float:
        """Variance of the valuation distribution."""
        probs = np.abs(self.amplitudes) ** 2
        mean = np.sum(probs * self.values)
        return float(np.sum(probs * (self.values - mean) ** 2))

    def sample(self, rng: Optional[np.random.Generator] = None) -> float:
        """Sample a valuation according to Born rule probabilities.

        Parameters
        ----------
        rng : Generator, optional
            Random number generator. Uses default_rng(42) if None.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        probs = np.abs(self.amplitudes) ** 2
        probs = probs / np.sum(probs)  # Ensure normalization
        idx = rng.choice(len(self.values), p=probs)
        return float(self.values[idx])

    def probabilities(self) -> np.ndarray:
        """Born rule probability distribution over values."""
        probs = np.abs(self.amplitudes) ** 2
        return probs / np.sum(probs)


# ---------------------------------------------------------------------------
# Allocation result
# ---------------------------------------------------------------------------

@dataclass
class AllocationResult:
    """Result of a mechanism allocation."""

    allocation: np.ndarray  # who gets what (binary or fractional)
    social_welfare: float
    payments: np.ndarray  # payment from each player


# ---------------------------------------------------------------------------
# Quantum VCG Mechanism
# ---------------------------------------------------------------------------

class QuantumVCG:
    """Quantum Vickrey-Clarke-Groves mechanism.

    Truthful mechanism where quantum valuations are processed to determine
    allocation and payments. The VCG mechanism has the dominant strategy
    property: truthful reporting is optimal regardless of other players.

    VCG payment for player i:
        p_i = max welfare without i - (total welfare - i's allocation value)

    Parameters
    ----------
    n_players : int
        Number of players/bidders.
    n_items : int
        Number of items to allocate. Default 1 (single-item auction).
    """

    def __init__(self, n_players: int, n_items: int = 1) -> None:
        if n_players < 2:
            raise ValueError("Need at least 2 players")
        self.n_players = n_players
        self.n_items = n_items

    def allocate(self, valuations: List[QuantumValuation]) -> AllocationResult:
        """Determine socially optimal allocation.

        Uses expected values from quantum valuations to compute the
        welfare-maximizing allocation, then computes VCG payments.

        Parameters
        ----------
        valuations : list of QuantumValuation
            One valuation per player.

        Returns
        -------
        AllocationResult with allocation, welfare, and payments.
        """
        if len(valuations) != self.n_players:
            raise ValueError(
                f"Expected {self.n_players} valuations, got {len(valuations)}"
            )

        # Compute expected values
        expected = np.array([v.expected_value() for v in valuations])

        # Allocate items to players with highest expected values
        allocation = np.zeros(self.n_players, dtype=np.float64)
        # Sort players by expected value (descending)
        order = np.argsort(-expected)
        for i in range(min(self.n_items, self.n_players)):
            allocation[order[i]] = 1.0

        # Compute social welfare
        social_welfare = float(np.sum(allocation * expected))

        # Compute VCG payments
        payments = self.compute_payments(valuations, allocation, expected)

        return AllocationResult(
            allocation=allocation,
            social_welfare=social_welfare,
            payments=payments,
        )

    def compute_payments(
        self,
        valuations: List[QuantumValuation],
        allocation: np.ndarray,
        expected: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute VCG payments for each player.

        VCG payment for player i:
            p_i = W_{-i}^* - W_{-i}

        where W_{-i}^* is the maximum welfare achievable without player i,
        and W_{-i} is the welfare of other players in the current allocation.

        Parameters
        ----------
        valuations : list of QuantumValuation
            Player valuations.
        allocation : ndarray
            Current allocation.
        expected : ndarray, optional
            Expected values (computed if not provided).
        """
        if expected is None:
            expected = np.array([v.expected_value() for v in valuations])

        payments = np.zeros(self.n_players, dtype=np.float64)

        for i in range(self.n_players):
            if allocation[i] < 1e-10:
                # Player i doesn't get an item, pays nothing
                payments[i] = 0.0
                continue

            # Welfare of others in current allocation
            welfare_others = float(np.sum(allocation * expected)) - expected[i]

            # Maximum welfare without player i
            other_expected = expected.copy()
            other_expected[i] = -np.inf
            other_order = np.argsort(-other_expected)
            welfare_without_i = 0.0
            for j in range(min(self.n_items, self.n_players - 1)):
                idx = other_order[j]
                if idx != i and expected[idx] > 0:
                    welfare_without_i += expected[idx]

            # VCG payment: externality that player i imposes on others
            payments[i] = max(0.0, welfare_without_i - welfare_others)

        return payments

    def is_truthful(
        self,
        valuations: List[QuantumValuation],
        n_tests: int = 100,
        rng: Optional[np.random.Generator] = None,
    ) -> bool:
        """Test incentive compatibility via random deviations.

        For each player, check that truthful reporting gives at least as
        much utility as any random deviation.

        Parameters
        ----------
        valuations : list of QuantumValuation
            True valuations.
        n_tests : int
            Number of random deviations to test per player.
        rng : Generator, optional
            Random number generator.

        Returns
        -------
        True if no profitable deviation found.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Truthful allocation
        truth_result = self.allocate(valuations)
        truth_expected = np.array([v.expected_value() for v in valuations])

        for i in range(self.n_players):
            # Utility under truthful reporting
            truth_utility = (
                truth_result.allocation[i] * truth_expected[i]
                - truth_result.payments[i]
            )

            # Test random deviations
            for _ in range(n_tests):
                # Create deviated valuation
                dev_values = valuations[i].values.copy()
                dev_amps = rng.standard_normal(len(dev_values)) + \
                    1j * rng.standard_normal(len(dev_values))
                dev_amps = dev_amps / np.sqrt(np.sum(np.abs(dev_amps) ** 2))

                dev_valuation = QuantumValuation(
                    values=dev_values,
                    amplitudes=dev_amps,
                    player_id=i,
                )

                # Replace player i's valuation
                dev_valuations = list(valuations)
                dev_valuations[i] = dev_valuation
                dev_result = self.allocate(dev_valuations)

                # Utility under deviation (using TRUE expected value, not reported)
                dev_utility = (
                    dev_result.allocation[i] * truth_expected[i]
                    - dev_result.payments[i]
                )

                if dev_utility > truth_utility + 1e-8:
                    return False

        return True


# ---------------------------------------------------------------------------
# Auction outcome
# ---------------------------------------------------------------------------

@dataclass
class AuctionOutcome:
    """Outcome of a quantum-enhanced auction."""

    winner: int
    payment: float
    revenue: float
    social_welfare: float


# ---------------------------------------------------------------------------
# Revenue-Optimal Mechanism
# ---------------------------------------------------------------------------

class QuantumRevenueMechanism:
    """Revenue-optimal mechanism using quantum strategies.

    Implements a second-price auction with a reserve price, enhanced
    with quantum valuations that can represent uncertainty about bidder
    types more richly than classical distributions.

    Parameters
    ----------
    n_players : int
        Number of bidders.
    reserve_price : float
        Minimum price for the item to be sold.
    """

    def __init__(self, n_players: int, reserve_price: float = 0.0) -> None:
        if n_players < 1:
            raise ValueError("Need at least 1 player")
        self.n_players = n_players
        self.reserve_price = reserve_price

    def run_auction(
        self,
        valuations: List[QuantumValuation],
        rng: Optional[np.random.Generator] = None,
    ) -> AuctionOutcome:
        """Run quantum-enhanced auction.

        Each bidder's quantum valuation is sampled (simulating measurement).
        The highest bidder above the reserve price wins, paying the maximum
        of the second-highest bid and the reserve price.

        Parameters
        ----------
        valuations : list of QuantumValuation
            Bidder valuations.
        rng : Generator, optional
            Random number generator for sampling.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        if len(valuations) != self.n_players:
            raise ValueError(
                f"Expected {self.n_players} valuations, got {len(valuations)}"
            )

        # Sample bids from quantum valuations
        bids = np.array([v.sample(rng) for v in valuations])

        # Find winner (highest bid above reserve)
        valid_mask = bids >= self.reserve_price
        if not np.any(valid_mask):
            # No valid bids: no winner
            return AuctionOutcome(
                winner=-1,
                payment=0.0,
                revenue=0.0,
                social_welfare=0.0,
            )

        winner = int(np.argmax(np.where(valid_mask, bids, -np.inf)))
        winning_bid = bids[winner]

        # Second price: maximum of second-highest valid bid and reserve
        sorted_valid = np.sort(bids[valid_mask])[::-1]
        if len(sorted_valid) >= 2:
            second_price = max(sorted_valid[1], self.reserve_price)
        else:
            second_price = self.reserve_price

        payment = second_price
        social_welfare = float(winning_bid)

        return AuctionOutcome(
            winner=winner,
            payment=payment,
            revenue=payment,
            social_welfare=social_welfare,
        )

    def expected_revenue(
        self,
        value_distributions: List[np.ndarray],
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Estimate expected revenue via Monte Carlo sampling.

        Parameters
        ----------
        value_distributions : list of ndarray
            Each entry is an array of possible values for a bidder.
            Uniform probability over the values is assumed.
        n_samples : int
            Number of auction simulations.
        rng : Generator, optional
            Random number generator.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        total_revenue = 0.0
        for _ in range(n_samples):
            # Create random quantum valuations from distributions
            valuations = []
            for i, vals in enumerate(value_distributions):
                vals_arr = np.asarray(vals, dtype=np.float64)
                n_vals = len(vals_arr)
                amps = np.ones(n_vals, dtype=np.complex128) / np.sqrt(n_vals)
                valuations.append(QuantumValuation(
                    values=vals_arr,
                    amplitudes=amps,
                    player_id=i,
                ))
            outcome = self.run_auction(valuations, rng)
            total_revenue += outcome.revenue

        return total_revenue / n_samples


# ---------------------------------------------------------------------------
# Incentive Analyzer
# ---------------------------------------------------------------------------

class IncentiveAnalyzer:
    """Analyze incentive properties of quantum mechanisms.

    Provides tools to quantify how well a mechanism resists strategic
    manipulation, including incentive compatibility scores and individual
    rationality checks.
    """

    def incentive_compatibility_score(
        self,
        mechanism: QuantumVCG,
        valuations: List[QuantumValuation],
        n_deviations: int = 50,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Score from 0 to 1 how incentive-compatible the mechanism is.

        Tests random deviations for each player and computes the fraction
        that do NOT improve utility (higher is more incentive-compatible).

        Parameters
        ----------
        mechanism : QuantumVCG
            The mechanism to analyze.
        valuations : list of QuantumValuation
            True valuations.
        n_deviations : int
            Number of random deviations per player.
        rng : Generator, optional
            Random number generator.

        Returns
        -------
        Score in [0, 1] where 1.0 = perfectly incentive-compatible.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        truth_result = mechanism.allocate(valuations)
        truth_expected = np.array([v.expected_value() for v in valuations])

        total_tests = 0
        non_profitable = 0

        for i in range(mechanism.n_players):
            truth_utility = (
                truth_result.allocation[i] * truth_expected[i]
                - truth_result.payments[i]
            )

            for _ in range(n_deviations):
                total_tests += 1

                dev_amps = rng.standard_normal(len(valuations[i].values)) + \
                    1j * rng.standard_normal(len(valuations[i].values))
                norm = np.sqrt(np.sum(np.abs(dev_amps) ** 2))
                if norm > 1e-15:
                    dev_amps = dev_amps / norm

                dev_val = QuantumValuation(
                    values=valuations[i].values.copy(),
                    amplitudes=dev_amps,
                    player_id=i,
                )

                dev_vals = list(valuations)
                dev_vals[i] = dev_val
                dev_result = mechanism.allocate(dev_vals)

                dev_utility = (
                    dev_result.allocation[i] * truth_expected[i]
                    - dev_result.payments[i]
                )

                if dev_utility <= truth_utility + 1e-8:
                    non_profitable += 1

        return non_profitable / max(total_tests, 1)

    def individual_rationality_check(
        self,
        mechanism: QuantumVCG,
        valuations: List[QuantumValuation],
    ) -> bool:
        """Check if all players have non-negative utility.

        A mechanism is individually rational if no player is worse off
        by participating than by not participating (utility >= 0).

        Parameters
        ----------
        mechanism : QuantumVCG
            The mechanism to check.
        valuations : list of QuantumValuation
            Player valuations.

        Returns
        -------
        True if all players have non-negative utility.
        """
        result = mechanism.allocate(valuations)
        expected = np.array([v.expected_value() for v in valuations])

        for i in range(mechanism.n_players):
            utility = result.allocation[i] * expected[i] - result.payments[i]
            if utility < -1e-8:
                return False

        return True

    def efficiency_ratio(
        self,
        mechanism: QuantumVCG,
        valuations: List[QuantumValuation],
    ) -> float:
        """Compute the efficiency ratio of the mechanism.

        Ratio of achieved social welfare to the maximum possible welfare.

        Returns
        -------
        Efficiency ratio in [0, 1].
        """
        result = mechanism.allocate(valuations)
        expected = np.array([v.expected_value() for v in valuations])

        # Maximum possible welfare: give items to highest-valued players
        sorted_vals = np.sort(expected)[::-1]
        max_welfare = float(np.sum(sorted_vals[:mechanism.n_items]))

        if max_welfare < 1e-15:
            return 1.0
        return float(result.social_welfare / max_welfare)
