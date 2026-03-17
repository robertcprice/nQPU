"""Comprehensive tests for nqpu.games -- quantum game theory, combinatorial
optimization, walk-based games, and decision theory.
"""
from __future__ import annotations

import numpy as np
import pytest

from nqpu.games import (
    # Quantum Games
    QuantumStrategy,
    GameResult,
    QuantumGame,
    QuantumTournament,
    PrisonersDilemma,
    BattleOfSexes,
    Chicken,
    MatchingPennies,
    cooperate,
    defect,
    hadamard_strategy,
    quantum_miracle_move,
    # Combinatorial
    Graph,
    OptimizationResult,
    MaxCut,
    GraphColoring,
    TravelingSalesman,
    NumberPartition,
    # Walk Games
    CoinFlipResult,
    QuantumCoinGame,
    MeyerPennyFlip,
    QuantumVoting,
    QuantumAuction,
    AuctionBid,
    # Decision
    QuantumBayesian,
    QuantumMarkov,
    QuantumBandit,
    QuantumPortfolio,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def pd_game():
    """Maximally-entangled Prisoner's Dilemma."""
    return PrisonersDilemma(gamma=np.pi / 2.0)


@pytest.fixture
def small_graph():
    """Triangle graph with unit weights for MaxCut / coloring tests."""
    g = Graph(3)
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 1.0)
    g.add_edge(0, 2, 1.0)
    return g


@pytest.fixture
def random_graph():
    """Deterministic random graph for reproducible tests."""
    return Graph.random_graph(5, edge_prob=0.6, seed=42)


# =====================================================================
# Quantum Strategy tests
# =====================================================================

class TestQuantumStrategy:

    def test_cooperate_is_identity(self):
        s = cooperate()
        u = s.unitary()
        assert u.shape == (2, 2)
        np.testing.assert_allclose(u, np.eye(2), atol=1e-12)

    def test_defect_is_x_gate(self):
        s = defect()
        u = s.unitary()
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        # X gate up to global phase:
        np.testing.assert_allclose(np.abs(u), np.abs(x), atol=1e-12)

    def test_unitary_is_unitary(self):
        """Every quantum strategy should produce a unitary matrix."""
        for s in [cooperate(), defect(), hadamard_strategy(), quantum_miracle_move()]:
            u = s.unitary()
            product = u @ u.conj().T
            np.testing.assert_allclose(product, np.eye(2), atol=1e-12)


# =====================================================================
# Prisoner's Dilemma tests
# =====================================================================

class TestPrisonersDilemma:

    def test_classical_defect_defect(self, pd_game):
        """Classical (D,D) in zero-entanglement game gives payoff (1,1)."""
        game = PrisonersDilemma(gamma=0.0)
        result = game.play(defect(), defect())
        assert isinstance(result, GameResult)
        assert abs(result.payoff_p1 - 1.0) < 0.1
        assert abs(result.payoff_p2 - 1.0) < 0.1

    def test_quantum_miracle_payoff(self, pd_game):
        """(Q, Q) in max-entanglement game achieves Pareto-optimal (3,3)."""
        result = pd_game.play(quantum_miracle_move(), quantum_miracle_move())
        assert abs(result.payoff_p1 - 3.0) < 0.1
        assert abs(result.payoff_p2 - 3.0) < 0.1
        assert result.is_pareto_optimal

    def test_quantum_advantage_positive(self, pd_game):
        """Quantum miracle move yields positive advantage over classical Nash."""
        result = pd_game.play(quantum_miracle_move(), quantum_miracle_move())
        assert result.quantum_advantage_p1 > 0.0
        assert result.quantum_advantage_p2 > 0.0

    def test_outcome_probs_sum_to_one(self, pd_game):
        result = pd_game.play(hadamard_strategy(), cooperate())
        total = result.outcome_probs.sum()
        assert abs(total - 1.0) < 1e-10

    def test_strategies_stored_in_result(self, pd_game):
        s1 = cooperate()
        s2 = defect()
        result = pd_game.play(s1, s2)
        assert result.strategy_p1 is s1
        assert result.strategy_p2 is s2


# =====================================================================
# Pre-built games
# =====================================================================

class TestPrebuiltGames:

    @pytest.mark.parametrize("game_cls", [
        PrisonersDilemma, BattleOfSexes, Chicken, MatchingPennies,
    ])
    def test_game_instantiates(self, game_cls):
        game = game_cls()
        assert game.name
        assert game.payoff_p1.shape == (2, 2)

    @pytest.mark.parametrize("game_cls", [
        PrisonersDilemma, BattleOfSexes, Chicken, MatchingPennies,
    ])
    def test_play_returns_game_result(self, game_cls):
        game = game_cls()
        result = game.play(cooperate(), defect())
        assert isinstance(result, GameResult)


# =====================================================================
# Tournament tests
# =====================================================================

class TestQuantumTournament:

    def test_tournament_runs(self, pd_game):
        strategies = [cooperate(), defect(), hadamard_strategy(), quantum_miracle_move()]
        t = QuantumTournament(pd_game, strategies)
        result = t.run()
        assert len(result.rankings) == 4
        assert result.matchups.shape == (4, 4)

    def test_tournament_rankings_sorted(self, pd_game):
        strategies = [cooperate(), defect(), quantum_miracle_move()]
        t = QuantumTournament(pd_game, strategies)
        result = t.run()
        payoffs = [p for _, p in result.rankings]
        assert payoffs == sorted(payoffs, reverse=True)


# =====================================================================
# Combinatorial: Graph
# =====================================================================

class TestGraph:

    def test_graph_creation(self, small_graph):
        assert small_graph.n == 3
        assert len(small_graph.edges) == 3

    def test_self_loop_raises(self):
        g = Graph(3)
        with pytest.raises(ValueError, match="Self-loops"):
            g.add_edge(0, 0)

    def test_random_graph_deterministic(self):
        g1 = Graph.random_graph(5, seed=123)
        g2 = Graph.random_graph(5, seed=123)
        assert g1.edges.keys() == g2.edges.keys()

    def test_complete_graph(self):
        g = Graph.complete_graph(4, seed=42)
        expected_edges = 4 * 3 // 2  # C(4,2) = 6
        assert len(g.edges) == expected_edges


# =====================================================================
# Combinatorial: MaxCut
# =====================================================================

class TestMaxCut:

    def test_brute_force_triangle(self, small_graph):
        mc = MaxCut(small_graph)
        result = mc.brute_force()
        assert isinstance(result, OptimizationResult)
        # Triangle with unit weights: max cut = 2
        assert result.objective == 2.0

    def test_evaluate_known_assignment(self, small_graph):
        mc = MaxCut(small_graph)
        # [0,1,0] cuts edges (0,1) and (1,2) -> cut value 2
        val = mc.evaluate(np.array([0, 1, 0]))
        assert val == 2.0

    def test_qubo_matrix_shape(self, small_graph):
        mc = MaxCut(small_graph)
        q = mc.qubo_matrix()
        assert q.shape == (3, 3)

    def test_simulated_annealing_finds_good_cut(self, random_graph):
        mc = MaxCut(random_graph)
        bf = mc.brute_force()
        sa = mc.simulated_annealing(n_iterations=5000, seed=42)
        # SA should find at least 50% of optimal
        assert sa.objective >= bf.objective * 0.5


# =====================================================================
# Combinatorial: Graph Coloring
# =====================================================================

class TestGraphColoring:

    def test_greedy_triangle_3colors(self, small_graph):
        gc = GraphColoring(small_graph, n_colors=3)
        result = gc.greedy()
        valid, conflicts = gc.evaluate(result.solution)
        assert valid
        assert conflicts == 0

    def test_greedy_triangle_2colors_fails(self, small_graph):
        gc = GraphColoring(small_graph, n_colors=2)
        result = gc.greedy()
        # Triangle cannot be 2-colored -> must have conflicts
        _, conflicts = gc.evaluate(result.solution)
        assert conflicts > 0


# =====================================================================
# Combinatorial: TSP
# =====================================================================

class TestTSP:

    def test_two_opt_improves(self):
        tsp = TravelingSalesman.random_instance(5, seed=42)
        initial_tour = tsp.random_tour(seed=10)
        initial_len = tsp.evaluate_tour(initial_tour)
        result = tsp.two_opt(initial_tour=initial_tour, max_iterations=500)
        assert result.objective <= initial_len + 1e-10

    def test_brute_force_small(self):
        tsp = TravelingSalesman.random_instance(4, seed=42)
        result = tsp.brute_force()
        assert len(result.solution) == 4
        # All cities visited exactly once
        assert set(result.solution) == {0, 1, 2, 3}


# =====================================================================
# Combinatorial: NumberPartition
# =====================================================================

class TestNumberPartition:

    def test_brute_force_perfect_split(self):
        nums = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
        np_ = NumberPartition(nums)
        result = np_.brute_force()
        # Total = 20, perfect split: 10-10, diff = 0
        assert result.objective == 0.0

    def test_evaluate_assignment(self):
        nums = np.array([3.0, 1.0, 2.0])
        np_ = NumberPartition(nums)
        # Assignment [1,0,0] -> subset0={1,2}, subset1={3} -> diff=|3-3|=0
        diff = np_.evaluate(np.array([1, 0, 0]))
        assert abs(diff) < 1e-10


# =====================================================================
# Walk Games: Quantum Coin
# =====================================================================

class TestQuantumCoinGame:

    def test_fair_coin(self):
        coin = QuantumCoinGame(bias_angle=np.pi / 2.0)
        assert coin.prob_heads == pytest.approx(0.5, abs=1e-10)
        assert coin.prob_tails == pytest.approx(0.5, abs=1e-10)

    def test_biased_coin(self):
        coin = QuantumCoinGame(bias_angle=0.0)
        assert coin.prob_heads == pytest.approx(1.0, abs=1e-10)
        result = coin.flip(n_rounds=100, seed=42)
        assert result.n_heads == 100

    def test_flip_returns_result(self):
        coin = QuantumCoinGame()
        result = coin.flip(n_rounds=50, seed=42)
        assert isinstance(result, CoinFlipResult)
        assert result.n_heads + result.n_tails == 50


# =====================================================================
# Walk Games: Meyer Penny Flip
# =====================================================================

class TestMeyerPennyFlip:

    def test_quantum_always_wins_no_flip(self):
        game = MeyerPennyFlip()
        result = game.play(classical_flip=False)
        assert result.quantum_player_wins

    def test_quantum_always_wins_with_flip(self):
        game = MeyerPennyFlip()
        result = game.play(classical_flip=True)
        assert result.quantum_player_wins

    def test_demonstrate_advantage(self):
        game = MeyerPennyFlip()
        results = game.demonstrate_quantum_advantage()
        assert len(results) == 2
        assert all(r.quantum_player_wins for r in results)


# =====================================================================
# Walk Games: Voting
# =====================================================================

class TestQuantumVoting:

    def test_expected_tally(self):
        v = QuantumVoting(n_voters=3)
        yes, no = v.expected_tally([True, True, False])
        assert yes == 2
        assert no == 1

    def test_voting_result_valid(self):
        v = QuantumVoting(n_voters=4)
        result = v.vote([True, False, True, False], seed=42)
        assert result.n_voters == 4
        assert result.tally_yes + result.tally_no == 4
        assert result.individual_votes_hidden
        assert abs(result.ballot_state_norm - 1.0) < 1e-8


# =====================================================================
# Walk Games: Auction
# =====================================================================

class TestQuantumAuction:

    def test_auction_vickrey_pricing(self):
        auction = QuantumAuction(n_bidders=3, max_bid=100.0)
        bids = [
            auction.create_bid(0, 50.0),
            auction.create_bid(1, 80.0),
            auction.create_bid(2, 30.0),
        ]
        result = auction.run_auction(bids)
        assert result.winner_id == 1
        assert result.winning_bid == 80.0
        assert result.second_price == 50.0
        assert result.protocol_valid

    def test_invalid_bid_raises(self):
        auction = QuantumAuction(n_bidders=2, max_bid=10.0)
        with pytest.raises(ValueError):
            auction.create_bid(0, 15.0)


# =====================================================================
# Decision: Bayesian
# =====================================================================

class TestQuantumBayesian:

    def test_uniform_prior_with_no_evidence(self):
        qb = QuantumBayesian(n_hypotheses=3)
        prior = np.array([1 / 3, 1 / 3, 1 / 3])
        result = qb.update(prior, np.array([1.0, 1.0, 1.0]), strength=0.0)
        np.testing.assert_allclose(result.posterior, prior, atol=1e-6)

    def test_posterior_sums_to_one(self):
        qb = QuantumBayesian(n_hypotheses=4)
        prior = np.array([0.4, 0.3, 0.2, 0.1])
        lk = np.array([0.1, 0.5, 0.8, 0.2])
        result = qb.update(prior, lk, strength=1.0)
        assert abs(result.posterior.sum() - 1.0) < 1e-6


# =====================================================================
# Decision: MDP
# =====================================================================

class TestQuantumMarkov:

    def test_value_iteration_converges(self):
        mdp = QuantumMarkov.random_mdp(n_states=4, n_actions=2, seed=42)
        result = mdp.value_iteration(max_iterations=500, tol=1e-6)
        assert result.converged
        assert len(result.values) == 4
        assert len(result.policy) == 4

    def test_quantum_value_iteration(self):
        mdp = QuantumMarkov.random_mdp(n_states=4, n_actions=2, seed=42)
        result = mdp.quantum_value_iteration(max_iterations=500)
        assert result.converged
        assert result.values.shape == (4,)


# =====================================================================
# Decision: Bandit
# =====================================================================

class TestQuantumBandit:

    def test_bandit_runs(self):
        bandit = QuantumBandit(arm_means=np.array([0.1, 0.5, 0.3]))
        result = bandit.run_quantum_exploration(n_rounds=200, seed=42)
        assert result.total_reward != 0.0
        assert len(result.reward_history) == 200

    def test_best_arm_identified(self):
        bandit = QuantumBandit(arm_means=np.array([0.1, 0.9, 0.2]))
        assert bandit.best_arm == 1


# =====================================================================
# Decision: Portfolio
# =====================================================================

class TestQuantumPortfolio:

    def test_analytical_weights_sum_to_one(self):
        pf = QuantumPortfolio.random_portfolio(n_assets=4, seed=42)
        result = pf.optimize_analytical()
        assert abs(result.weights.sum() - 1.0) < 1e-8

    def test_risk_return_frontier(self):
        pf = QuantumPortfolio.random_portfolio(n_assets=3, seed=42)
        frontier = pf.risk_return_frontier(n_points=5)
        assert len(frontier) == 5
        for r in frontier:
            assert abs(r.weights.sum() - 1.0) < 1e-8
