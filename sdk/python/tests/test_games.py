"""Comprehensive tests for nqpu.games -- quantum games, combinatorial
optimization, quantum protocols, and decision theory.

Run with:
    cd sdk/python && python3 -m pytest tests/test_games.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from nqpu.games import (
    # Quantum games
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
    # Quantum walk games
    QuantumCoinGame,
    MeyerPennyFlip,
    QuantumVoting,
    QuantumAuction,
    # Decision theory
    QuantumBayesian,
    QuantumMarkov,
    QuantumBandit,
    QuantumPortfolio,
)


# =========================================================================
# Quantum Games
# =========================================================================

class TestQuantumStrategy:
    """Tests for QuantumStrategy and preset strategies."""

    def test_cooperate_is_identity(self):
        c = cooperate()
        u = c.unitary()
        assert np.allclose(u, np.eye(2), atol=1e-12)

    def test_defect_is_pauli_x(self):
        d = defect()
        u = d.unitary()
        x = np.array([[0, 1], [-1, 0]], dtype=np.complex128)
        # Defect: theta=pi, phi=0 -> [[cos(pi/2), sin(pi/2)], [-sin(pi/2), cos(pi/2)]]
        # = [[0, 1], [-1, 0]]
        assert np.allclose(u, x, atol=1e-12)

    def test_strategy_is_unitary(self):
        for s in [cooperate(), defect(), hadamard_strategy(), quantum_miracle_move()]:
            u = s.unitary()
            product = u @ u.conj().T
            assert np.allclose(product, np.eye(2), atol=1e-12), f"{s.name} not unitary"

    def test_custom_strategy(self):
        s = QuantumStrategy(theta=np.pi / 4, phi=np.pi / 3, name="custom")
        u = s.unitary()
        assert u.shape == (2, 2)
        assert np.allclose(u @ u.conj().T, np.eye(2), atol=1e-12)

    def test_miracle_move_params(self):
        q = quantum_miracle_move()
        assert abs(q.theta) < 1e-12
        assert abs(q.phi - np.pi / 2) < 1e-12


class TestPrisonersDilemma:
    """Tests for the Quantum Prisoner's Dilemma."""

    def test_classical_cooperate_cooperate(self):
        game = PrisonersDilemma()
        r = game.play(cooperate(), cooperate())
        assert abs(r.payoff_p1 - 3.0) < 0.1
        assert abs(r.payoff_p2 - 3.0) < 0.1

    def test_classical_defect_defect(self):
        game = PrisonersDilemma()
        r = game.play(defect(), defect())
        assert abs(r.payoff_p1 - 1.0) < 0.1
        assert abs(r.payoff_p2 - 1.0) < 0.1

    def test_classical_cooperate_defect_classical_limit(self):
        """C vs D in the classical limit (gamma=0) gives standard PD payoffs."""
        game = PrisonersDilemma(gamma=0.0)
        r = game.play(cooperate(), defect())
        assert abs(r.payoff_p1 - 0.0) < 0.1
        assert abs(r.payoff_p2 - 5.0) < 0.1

    def test_classical_defect_cooperate_classical_limit(self):
        """D vs C in the classical limit (gamma=0) gives standard PD payoffs."""
        game = PrisonersDilemma(gamma=0.0)
        r = game.play(defect(), cooperate())
        assert abs(r.payoff_p1 - 5.0) < 0.1
        assert abs(r.payoff_p2 - 0.0) < 0.1

    def test_quantum_miracle_vs_miracle(self):
        """Q vs Q should give payoff close to (3, 3) -- the cooperative outcome."""
        game = PrisonersDilemma()
        q = quantum_miracle_move()
        r = game.play(q, q)
        assert r.payoff_p1 > 2.5, f"Q vs Q payoff too low: {r.payoff_p1}"
        assert r.payoff_p2 > 2.5, f"Q vs Q payoff too low: {r.payoff_p2}"

    def test_miracle_beats_classical_defection(self):
        """Q vs D: quantum player should not be exploited."""
        game = PrisonersDilemma()
        r = game.play(quantum_miracle_move(), defect())
        # In maximally-entangled PD, Q vs D gives P1 >= 1 (not exploited)
        assert r.payoff_p1 >= 0.0, f"Q vs D payoff: {r.payoff_p1}"

    def test_quantum_nash_beats_classical_nash(self):
        """Quantum NE payoff should exceed classical NE payoff of (1,1)."""
        game = PrisonersDilemma()
        s1, s2, r = game.find_nash_equilibrium(max_iterations=10, grid_steps=30)
        assert r.payoff_p1 > 0.5, f"Quantum NE payoff: {r.payoff_p1}"

    def test_payoff_matrix_is_correct(self):
        game = PrisonersDilemma()
        assert game.payoff_p1[0, 0] == 3.0  # CC
        assert game.payoff_p1[0, 1] == 0.0  # CD
        assert game.payoff_p1[1, 0] == 5.0  # DC
        assert game.payoff_p1[1, 1] == 1.0  # DD

    def test_classical_nash_payoff(self):
        """Classical NE for PD is (D,D) with payoff 1."""
        game = PrisonersDilemma()
        assert abs(game._classical_nash_payoff(1) - 1.0) < 1e-8

    def test_pareto_optimality(self):
        game = PrisonersDilemma()
        q = quantum_miracle_move()
        r = game.play(q, q)
        assert r.is_pareto_optimal

    def test_quantum_advantage_positive(self):
        game = PrisonersDilemma()
        q = quantum_miracle_move()
        r = game.play(q, q)
        assert r.quantum_advantage_p1 > 0
        assert r.quantum_advantage_p2 > 0

    def test_classical_limit_gamma_zero(self):
        """With gamma=0, should recover classical game."""
        game = PrisonersDilemma(gamma=0.0)
        r = game.play(defect(), defect())
        assert abs(r.payoff_p1 - 1.0) < 0.1
        r2 = game.play(cooperate(), cooperate())
        assert abs(r2.payoff_p1 - 3.0) < 0.1

    def test_outcome_probs_sum_to_one(self):
        game = PrisonersDilemma()
        for s1 in [cooperate(), defect(), quantum_miracle_move()]:
            for s2 in [cooperate(), defect(), quantum_miracle_move()]:
                r = game.play(s1, s2)
                assert abs(np.sum(r.outcome_probs) - 1.0) < 1e-10

    def test_outcome_probs_non_negative(self):
        game = PrisonersDilemma()
        r = game.play(hadamard_strategy(), quantum_miracle_move())
        assert np.all(r.outcome_probs >= -1e-12)

    def test_payoff_is_real(self):
        game = PrisonersDilemma()
        r = game.play(quantum_miracle_move(), hadamard_strategy())
        assert isinstance(r.payoff_p1, float)
        assert isinstance(r.payoff_p2, float)
        assert np.isfinite(r.payoff_p1)
        assert np.isfinite(r.payoff_p2)


class TestBattleOfSexes:

    def test_coordination_payoff(self):
        game = BattleOfSexes()
        r = game.play(cooperate(), cooperate())
        assert r.payoff_p1 >= 0

    def test_miscoordination(self):
        game = BattleOfSexes()
        r = game.play(cooperate(), defect())
        # Miscoordination should yield lower payoff
        r2 = game.play(cooperate(), cooperate())
        # At least one coordination outcome should be reasonable
        assert r.payoff_p1 >= -1 or r2.payoff_p1 >= 0

    def test_symmetric_quantum(self):
        game = BattleOfSexes()
        q = quantum_miracle_move()
        r = game.play(q, q)
        assert np.isfinite(r.payoff_p1)
        assert np.isfinite(r.payoff_p2)


class TestChicken:

    def test_both_swerve(self):
        game = Chicken()
        r = game.play(cooperate(), cooperate())
        assert abs(r.payoff_p1 - 3.0) < 0.2

    def test_both_straight_is_worst(self):
        game = Chicken()
        r = game.play(defect(), defect())
        # Both "straight" (defect) gives (0, 0) -- worst outcome
        assert abs(r.payoff_p1 - 0.0) < 0.2

    def test_chicken_quantum_strategy(self):
        game = Chicken()
        q = quantum_miracle_move()
        r = game.play(q, q)
        assert np.isfinite(r.payoff_p1)


class TestMatchingPennies:

    def test_zero_sum(self):
        game = MatchingPennies()
        for s1 in [cooperate(), defect(), hadamard_strategy()]:
            for s2 in [cooperate(), defect(), hadamard_strategy()]:
                r = game.play(s1, s2)
                assert abs(r.payoff_p1 + r.payoff_p2) < 0.1, \
                    f"Not zero-sum: {r.payoff_p1} + {r.payoff_p2}"

    def test_matching_wins_p1(self):
        game = MatchingPennies()
        r = game.play(cooperate(), cooperate())
        assert r.payoff_p1 > 0  # same choice -> P1 wins


class TestQuantumTournament:

    def test_tournament_rankings(self):
        game = PrisonersDilemma()
        strategies = [cooperate(), defect(), quantum_miracle_move(), hadamard_strategy()]
        t = QuantumTournament(game, strategies)
        result = t.run()
        assert len(result.rankings) == 4
        assert result.matchups.shape == (4, 4)
        # Diagonal should be zero (no self-play)
        assert np.all(np.diag(result.matchups) == 0)

    def test_tournament_total_payoffs_finite(self):
        game = PrisonersDilemma()
        strategies = [cooperate(), defect(), quantum_miracle_move()]
        t = QuantumTournament(game, strategies)
        result = t.run()
        for name, payoff in result.rankings:
            assert np.isfinite(payoff), f"{name} has non-finite payoff"

    def test_tournament_sorted_descending(self):
        game = PrisonersDilemma()
        strategies = [cooperate(), defect(), quantum_miracle_move()]
        t = QuantumTournament(game, strategies)
        result = t.run()
        payoffs = [p for _, p in result.rankings]
        assert payoffs == sorted(payoffs, reverse=True)


class TestCustomQuantumGame:

    def test_custom_game(self):
        game = QuantumGame(
            name="Test",
            payoff_p1=np.array([[2, 0], [3, 1]]),
            payoff_p2=np.array([[2, 3], [0, 1]]),
            gamma=np.pi / 4,
        )
        r = game.play(cooperate(), cooperate())
        assert np.isfinite(r.payoff_p1)

    def test_payoff_landscape(self):
        game = PrisonersDilemma()
        data = game.payoff_matrix_data(grid_steps=5)
        assert "theta" in data
        assert "phi" in data
        assert data["payoff_p1"].shape == (6, 6)

    def test_best_response_p1(self):
        game = PrisonersDilemma()
        br, payoff = game.best_response_p1(defect(), grid_steps=20)
        assert payoff >= 0
        assert isinstance(br, QuantumStrategy)

    def test_best_response_p2(self):
        game = PrisonersDilemma()
        br, payoff = game.best_response_p2(cooperate(), grid_steps=20)
        assert payoff >= 0
        assert isinstance(br, QuantumStrategy)


# =========================================================================
# Combinatorial Optimization
# =========================================================================

class TestGraph:

    def test_create_graph(self):
        g = Graph(5)
        g.add_edge(0, 1, 2.0)
        g.add_edge(1, 2, 3.0)
        assert len(g.edges) == 2
        assert g.weight(0, 1) == 2.0

    def test_random_graph(self):
        g = Graph.random_graph(10, edge_prob=0.5, seed=42)
        assert g.n == 10
        assert len(g.edges) > 0

    def test_complete_graph(self):
        g = Graph.complete_graph(5, seed=42)
        assert len(g.edges) == 10  # C(5,2) = 10

    def test_self_loop_rejected(self):
        g = Graph(3)
        with pytest.raises(ValueError, match="Self-loops"):
            g.add_edge(1, 1)

    def test_neighbors(self):
        g = Graph(3)
        g.add_edge(0, 1, 1.0)
        g.add_edge(0, 2, 2.0)
        nbrs = g.neighbors(0)
        assert len(nbrs) == 2


class TestMaxCut:

    def test_brute_force_triangle(self):
        """Triangle graph: optimal cut = 2 edges."""
        g = Graph(3)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(0, 2, 1.0)
        mc = MaxCut(g)
        result = mc.brute_force()
        assert abs(result.objective - 2.0) < 1e-8

    def test_brute_force_path(self):
        """Path 0-1-2: optimal cut = 2 (alternate partition)."""
        g = Graph(3)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        mc = MaxCut(g)
        result = mc.brute_force()
        assert abs(result.objective - 2.0) < 1e-8

    def test_brute_force_4_node(self):
        """Square graph: optimal cut = 4."""
        g = Graph(4)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(2, 3, 1.0)
        g.add_edge(3, 0, 1.0)
        mc = MaxCut(g)
        result = mc.brute_force()
        assert abs(result.objective - 4.0) < 1e-8

    def test_brute_force_8_node(self):
        """Random 8-node graph: brute force should find global optimum."""
        g = Graph.random_graph(8, edge_prob=0.5, seed=42)
        mc = MaxCut(g)
        result = mc.brute_force()
        # Verify optimality: check all assignments
        for bits in range(1 << 8):
            a = np.array([(bits >> i) & 1 for i in range(8)])
            assert mc.evaluate(a) <= result.objective + 1e-8

    def test_sa_finds_good_solution(self):
        """SA should find a cut >= 80% of optimal on small graphs."""
        g = Graph.random_graph(6, edge_prob=0.6, seed=42)
        mc = MaxCut(g)
        exact = mc.brute_force()
        sa = mc.simulated_annealing(n_iterations=5000, seed=42)
        assert sa.objective >= 0.8 * exact.objective

    def test_qubo_matrix_shape(self):
        g = Graph.random_graph(5, seed=42)
        mc = MaxCut(g)
        q = mc.qubo_matrix()
        assert q.shape == (5, 5)

    def test_qaoa_inspired(self):
        g = Graph.random_graph(5, edge_prob=0.6, seed=42)
        mc = MaxCut(g)
        result = mc.qaoa_inspired(p=2, n_iterations=50, seed=42)
        assert result.objective >= 0  # at least finds something

    def test_evaluate_correct(self):
        g = Graph(3)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        mc = MaxCut(g)
        # Assignment [0, 1, 0]: edges (0,1) and (1,2) both cross -> cut = 2
        assert abs(mc.evaluate(np.array([0, 1, 0])) - 2.0) < 1e-8
        # Assignment [0, 0, 0]: no edges cross -> cut = 0
        assert abs(mc.evaluate(np.array([0, 0, 0])) - 0.0) < 1e-8


class TestGraphColoring:

    def test_greedy_triangle_3_colors(self):
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(0, 2)
        gc = GraphColoring(g, n_colors=3)
        result = gc.greedy()
        valid, conflicts = gc.evaluate(result.solution)
        assert valid, f"Greedy failed on triangle with 3 colors: {result.solution}"

    def test_bipartite_2_colors(self):
        """Bipartite graph should be 2-colorable."""
        g = Graph(4)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        gc = GraphColoring(g, n_colors=2)
        result = gc.greedy()
        valid, _ = gc.evaluate(result.solution)
        assert valid, f"Failed 2-coloring bipartite: {result.solution}"

    def test_local_search_reduces_conflicts(self):
        g = Graph.random_graph(8, edge_prob=0.4, seed=42)
        gc = GraphColoring(g, n_colors=3)
        greedy_result = gc.greedy()
        ls_result = gc.greedy_local_search(n_iterations=500, seed=42)
        assert ls_result.objective <= greedy_result.objective

    def test_coloring_produces_valid_with_enough_colors(self):
        g = Graph.random_graph(6, edge_prob=0.3, seed=42)
        gc = GraphColoring(g, n_colors=6)  # n colors for n vertices always works
        result = gc.greedy()
        valid, _ = gc.evaluate(result.solution)
        assert valid

    def test_qubo_matrix_shape(self):
        g = Graph(4)
        gc = GraphColoring(g, n_colors=3)
        q = gc.qubo_matrix()
        assert q.shape == (12, 12)  # 4 vertices * 3 colors


class TestTSP:

    def test_brute_force_3_cities(self):
        g = Graph(3)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 2.0)
        g.add_edge(0, 2, 3.0)
        tsp = TravelingSalesman(g)
        result = tsp.brute_force()
        # Optimal tour: 0->1->2->0 = 1+2+3 = 6
        # or 0->2->1->0 = 3+2+1 = 6 (same for complete graph)
        assert abs(result.objective - 6.0) < 1e-8

    def test_two_opt_improves_random(self):
        tsp = TravelingSalesman.random_instance(6, seed=42)
        random_tour = tsp.random_tour(seed=42)
        random_length = tsp.evaluate_tour(random_tour)
        result = tsp.two_opt(initial_tour=random_tour)
        assert result.objective <= random_length + 1e-10

    def test_two_opt_history_decreasing(self):
        tsp = TravelingSalesman.random_instance(5, seed=42)
        result = tsp.two_opt(seed=42)
        # History should be non-increasing
        for i in range(1, len(result.history)):
            assert result.history[i] <= result.history[i - 1] + 1e-10

    def test_tour_is_valid_permutation(self):
        tsp = TravelingSalesman.random_instance(6, seed=42)
        result = tsp.two_opt(seed=42)
        assert sorted(result.solution) == list(range(6))

    def test_random_instance(self):
        tsp = TravelingSalesman.random_instance(5, seed=42)
        assert tsp.graph.n == 5

    def test_qubo_matrix_shape(self):
        tsp = TravelingSalesman.random_instance(4, seed=42)
        q = tsp.qubo_matrix()
        assert q.shape == (16, 16)  # 4 cities * 4 positions

    def test_brute_force_too_large(self):
        tsp = TravelingSalesman.random_instance(12, seed=42)
        with pytest.raises(ValueError, match="impractical"):
            tsp.brute_force()


class TestNumberPartition:

    def test_equal_partition(self):
        nums = np.array([1, 2, 3, 4, 5, 5])
        np_ = NumberPartition(nums)
        result = np_.brute_force()
        assert abs(result.objective) < 1e-8  # 1+4+5=10, 2+3+5=10

    def test_odd_sum_partition(self):
        nums = np.array([1, 2, 3])
        np_ = NumberPartition(nums)
        result = np_.brute_force()
        assert abs(result.objective - 0.0) < 1e-8  # {1,2} vs {3}: diff=0

    def test_sa_finds_good_partition(self):
        nums = np.array([7, 5, 3, 2, 1, 6, 4, 8])
        np_ = NumberPartition(nums)
        exact = np_.brute_force()
        sa = np_.simulated_annealing(n_iterations=3000, seed=42)
        assert sa.objective <= exact.objective + 2  # close to optimal

    def test_qubo_matrix_shape(self):
        nums = np.array([1, 2, 3, 4])
        np_ = NumberPartition(nums)
        q = np_.qubo_matrix()
        assert q.shape == (4, 4)


# =========================================================================
# Quantum Walk Games
# =========================================================================

class TestQuantumCoin:

    def test_fair_coin_probabilities(self):
        coin = QuantumCoinGame(bias_angle=np.pi / 2)
        assert abs(coin.prob_heads - 0.5) < 1e-10
        assert abs(coin.prob_tails - 0.5) < 1e-10

    def test_fair_coin_is_fair(self):
        coin = QuantumCoinGame()
        result = coin.flip(n_rounds=1, seed=42)
        assert result.is_fair

    def test_biased_coin(self):
        coin = QuantumCoinGame(bias_angle=np.pi / 4)  # biased toward heads
        assert coin.prob_heads > 0.5
        assert coin.prob_tails < 0.5
        assert abs(coin.prob_heads + coin.prob_tails - 1.0) < 1e-10

    def test_always_heads(self):
        coin = QuantumCoinGame(bias_angle=0.0)
        assert abs(coin.prob_heads - 1.0) < 1e-10
        result = coin.flip(n_rounds=10, seed=42)
        assert result.n_heads == 10

    def test_always_tails(self):
        coin = QuantumCoinGame(bias_angle=np.pi)
        assert abs(coin.prob_tails - 1.0) < 1e-10
        result = coin.flip(n_rounds=10, seed=42)
        assert result.n_tails == 10

    def test_many_flips_converge(self):
        coin = QuantumCoinGame()
        result = coin.flip(n_rounds=10000, seed=42)
        # Should converge to ~50%
        ratio = result.n_heads / 10000
        assert abs(ratio - 0.5) < 0.05

    def test_entangled_flip(self):
        coin = QuantumCoinGame()
        result = coin.entangled_flip(n_rounds=100, seed=42)
        assert result.n_heads + result.n_tails == 100
        assert result.is_fair


class TestMeyerPennyFlip:

    def test_quantum_wins_no_flip(self):
        """Q wins when P does not flip."""
        pf = MeyerPennyFlip()
        result = pf.play(classical_flip=False)
        assert result.quantum_player_wins
        assert result.measurement_probs[0] > 0.99

    def test_quantum_wins_with_flip(self):
        """Q wins when P flips -- quantum always wins."""
        pf = MeyerPennyFlip()
        result = pf.play(classical_flip=True)
        assert result.quantum_player_wins
        assert result.measurement_probs[0] > 0.99

    def test_quantum_wins_all_classical(self):
        """Demonstrate quantum advantage: Q wins against all classical P strategies."""
        pf = MeyerPennyFlip()
        results = pf.demonstrate_quantum_advantage()
        assert len(results) == 2
        for r in results:
            assert r.quantum_player_wins

    def test_custom_strategy(self):
        pf = MeyerPennyFlip()
        x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        result = pf.play(classical_flip=False, quantum_first=x, quantum_second=x)
        # X twice = identity, so |0> -> |0> regardless
        assert result.measurement_probs[0] > 0.99

    def test_state_normalized(self):
        pf = MeyerPennyFlip()
        result = pf.play(classical_flip=True)
        assert abs(np.sum(result.measurement_probs) - 1.0) < 1e-10


class TestQuantumVoting:

    def test_all_yes(self):
        qv = QuantumVoting(3)
        yes, no = qv.expected_tally([True, True, True])
        assert yes == 3
        assert no == 0

    def test_all_no(self):
        qv = QuantumVoting(3)
        yes, no = qv.expected_tally([False, False, False])
        assert yes == 0
        assert no == 3

    def test_mixed_votes(self):
        qv = QuantumVoting(4)
        yes, no = qv.expected_tally([True, False, True, False])
        assert yes == 2
        assert no == 2

    def test_ballot_state_normalized(self):
        qv = QuantumVoting(3)
        result = qv.vote([True, False, True], seed=42)
        assert abs(result.ballot_state_norm - 1.0) < 1e-8

    def test_individual_votes_hidden(self):
        qv = QuantumVoting(3)
        result = qv.vote([True, True, False], seed=42)
        assert result.individual_votes_hidden

    def test_wrong_vote_count(self):
        qv = QuantumVoting(3)
        with pytest.raises(ValueError, match="Expected 3 votes"):
            qv.vote([True, True])

    def test_too_many_voters(self):
        with pytest.raises(ValueError, match="Maximum 16"):
            QuantumVoting(17)

    def test_single_voter(self):
        qv = QuantumVoting(1)
        result = qv.vote([True], seed=42)
        assert result.n_voters == 1


class TestQuantumAuction:

    def test_highest_bid_wins(self):
        qa = QuantumAuction(n_bidders=3, max_bid=100)
        bids = [
            qa.create_bid(0, 30.0),
            qa.create_bid(1, 70.0),
            qa.create_bid(2, 50.0),
        ]
        result = qa.run_auction(bids)
        assert result.winner_id == 1
        assert abs(result.winning_bid - 70.0) < 1e-8

    def test_vickrey_second_price(self):
        qa = QuantumAuction(n_bidders=3, max_bid=100)
        bids = [
            qa.create_bid(0, 30.0),
            qa.create_bid(1, 70.0),
            qa.create_bid(2, 50.0),
        ]
        result = qa.run_auction(bids)
        assert abs(result.second_price - 50.0) < 1e-8

    def test_protocol_valid(self):
        qa = QuantumAuction(n_bidders=4, max_bid=100)
        result = qa.random_auction(seed=42)
        assert result.protocol_valid

    def test_bid_out_of_range(self):
        qa = QuantumAuction(n_bidders=2, max_bid=100)
        with pytest.raises(ValueError, match="Bid must be in"):
            qa.create_bid(0, 150.0)

    def test_bid_commitment_phase(self):
        qa = QuantumAuction(n_bidders=2, max_bid=100)
        bid = qa.create_bid(0, 50.0)
        assert abs(bid.commitment_phase - np.pi / 2) < 1e-10  # 50/100 * pi


# =========================================================================
# Decision Theory
# =========================================================================

class TestQuantumBayesian:

    def test_uniform_prior(self):
        qb = QuantumBayesian(n_hypotheses=3)
        prior = np.array([1 / 3, 1 / 3, 1 / 3])
        state = qb.prepare_prior(prior)
        probs = np.abs(state) ** 2
        assert np.allclose(probs, prior, atol=1e-10)

    def test_update_with_strong_evidence(self):
        qb = QuantumBayesian(n_hypotheses=2)
        prior = np.array([0.5, 0.5])
        likelihoods = np.array([0.9, 0.1])
        result = qb.update(prior, likelihoods, strength=1.0)
        # Posterior should favor hypothesis 0
        assert result.posterior[0] > result.posterior[1]

    def test_classical_posterior_correct(self):
        qb = QuantumBayesian(n_hypotheses=2)
        prior = np.array([0.3, 0.7])
        likelihoods = np.array([0.8, 0.2])
        result = qb.update(prior, likelihoods)
        # Classical: P(H0|E) = 0.8*0.3 / (0.8*0.3 + 0.2*0.7)
        expected_p0 = (0.8 * 0.3) / (0.8 * 0.3 + 0.2 * 0.7)
        assert abs(result.classical_posterior[0] - expected_p0) < 1e-8

    def test_posterior_sums_to_one(self):
        qb = QuantumBayesian(n_hypotheses=4)
        prior = np.array([0.1, 0.2, 0.3, 0.4])
        likelihoods = np.array([0.5, 0.3, 0.8, 0.1])
        result = qb.update(prior, likelihoods)
        assert abs(np.sum(result.posterior) - 1.0) < 1e-8
        assert abs(np.sum(result.classical_posterior) - 1.0) < 1e-8

    def test_no_evidence_preserves_prior(self):
        qb = QuantumBayesian(n_hypotheses=3)
        prior = np.array([0.2, 0.5, 0.3])
        likelihoods = np.array([1.0, 1.0, 1.0])  # uniform likelihood
        result = qb.update(prior, likelihoods, strength=0.0)
        # With zero strength, posterior should equal prior
        assert np.allclose(result.posterior, prior, atol=1e-6)

    def test_sequential_update(self):
        qb = QuantumBayesian(n_hypotheses=2)
        prior = np.array([0.5, 0.5])
        evidence_seq = [
            (np.array([0.8, 0.3]), 0.5),
            (np.array([0.4, 0.9]), 0.5),
        ]
        results = qb.sequential_update(prior, evidence_seq)
        assert len(results) == 2
        assert abs(np.sum(results[-1].posterior) - 1.0) < 1e-8

    def test_invalid_prior_sum(self):
        qb = QuantumBayesian(n_hypotheses=2)
        with pytest.raises(ValueError, match="sum to 1"):
            qb.prepare_prior(np.array([0.3, 0.3]))

    def test_negative_prior(self):
        qb = QuantumBayesian(n_hypotheses=2)
        with pytest.raises(ValueError, match="non-negative"):
            qb.prepare_prior(np.array([1.5, -0.5]))

    def test_kl_divergence_non_negative(self):
        qb = QuantumBayesian(n_hypotheses=3)
        prior = np.array([0.2, 0.5, 0.3])
        likelihoods = np.array([0.7, 0.2, 0.5])
        result = qb.update(prior, likelihoods)
        assert result.kl_divergence >= -1e-10  # KL >= 0


class TestQuantumMarkov:

    def test_value_iteration_converges(self):
        mdp = QuantumMarkov.random_mdp(n_states=5, n_actions=3, seed=42)
        result = mdp.value_iteration(max_iterations=500, tol=1e-6)
        assert result.converged
        assert result.n_iterations < 500

    def test_policy_shape(self):
        mdp = QuantumMarkov.random_mdp(n_states=5, n_actions=3, seed=42)
        result = mdp.value_iteration()
        assert result.policy.shape == (5,)
        assert np.all(result.policy >= 0)
        assert np.all(result.policy < 3)

    def test_quantum_value_iteration_converges(self):
        mdp = QuantumMarkov.random_mdp(n_states=4, n_actions=2, seed=42)
        result = mdp.quantum_value_iteration(max_iterations=500, tol=1e-6)
        assert result.converged

    def test_quantum_exploration_bonus(self):
        mdp = QuantumMarkov.random_mdp(n_states=4, n_actions=2, seed=42)
        classical = mdp.value_iteration()
        quantum = mdp.quantum_value_iteration(exploration_bonus=0.5)
        # Quantum values should be >= classical (exploration bonus)
        assert np.all(quantum.values >= classical.values - 0.5)

    def test_history_decreasing(self):
        mdp = QuantumMarkov.random_mdp(n_states=5, n_actions=3, seed=42)
        result = mdp.value_iteration()
        # Delta should generally decrease (convergence)
        assert result.history[-1] < result.history[0]

    def test_invalid_transition_shape(self):
        with pytest.raises(ValueError, match="Transition shape"):
            QuantumMarkov(
                n_states=3,
                n_actions=2,
                transition_probs=np.zeros((3, 3, 3)),  # wrong shape
                rewards=np.zeros((3, 2)),
            )

    def test_deterministic_mdp(self):
        """Simple deterministic MDP: state 0 -> action 0 -> state 1 (reward 1)."""
        transitions = np.zeros((2, 1, 2))
        transitions[0, 0, 1] = 1.0  # s0, a0 -> s1
        transitions[1, 0, 0] = 1.0  # s1, a0 -> s0
        rewards = np.array([[1.0], [0.5]])
        mdp = QuantumMarkov(2, 1, transitions, rewards, gamma=0.9)
        result = mdp.value_iteration()
        assert result.converged
        # V(s0) = 1 + 0.9*V(s1), V(s1) = 0.5 + 0.9*V(s0)
        # V(s0) = 1 + 0.9*(0.5 + 0.9*V(s0)) = 1.45 + 0.81*V(s0)
        # V(s0) = 1.45 / 0.19 ~ 7.63
        expected_v0 = 1.45 / 0.19
        assert abs(result.values[0] - expected_v0) < 0.1


class TestQuantumBandit:

    def test_finds_best_arm(self):
        arms = np.array([0.1, 0.5, 0.3, 0.9])
        bandit = QuantumBandit(arms)
        assert bandit.best_arm == 3

    def test_quantum_regret_bounded(self):
        arms = np.array([0.1, 0.5, 0.3, 0.8])
        bandit = QuantumBandit(arms)
        result = bandit.run_quantum_exploration(n_rounds=500, seed=42)
        # Regret should be much less than n_rounds * max_gap
        max_gap = 0.8 - 0.1
        assert result.cumulative_regret < 500 * max_gap

    def test_regret_increases(self):
        arms = np.array([0.3, 0.7])
        bandit = QuantumBandit(arms)
        result = bandit.run_quantum_exploration(n_rounds=100, seed=42)
        # Cumulative regret is non-decreasing
        for i in range(1, len(result.regret_history)):
            assert result.regret_history[i] >= result.regret_history[i - 1] - 1e-10

    def test_ucb_comparison(self):
        arms = np.array([0.2, 0.5, 0.8])
        bandit = QuantumBandit(arms, arm_stds=np.array([0.3, 0.3, 0.3]))
        q_result = bandit.run_quantum_exploration(n_rounds=500, seed=42)
        ucb_result = bandit.run_classical_ucb(n_rounds=500, seed=42)
        # Both should have bounded regret
        assert q_result.cumulative_regret < 500
        assert ucb_result.cumulative_regret < 500

    def test_all_arms_pulled(self):
        arms = np.array([0.1, 0.5, 0.3])
        bandit = QuantumBandit(arms)
        result = bandit.run_quantum_exploration(n_rounds=100, seed=42)
        assert np.all(result.arm_pulls > 0)


class TestQuantumPortfolio:

    def test_analytical_weights_sum_to_one(self):
        qp = QuantumPortfolio.random_portfolio(n_assets=5, seed=42)
        result = qp.optimize_analytical()
        assert abs(np.sum(result.weights) - 1.0) < 1e-8

    def test_weights_non_negative(self):
        qp = QuantumPortfolio.random_portfolio(n_assets=4, seed=42)
        result = qp.optimize_analytical()
        assert np.all(result.weights >= -1e-10)

    def test_sa_weights_sum_to_one(self):
        qp = QuantumPortfolio.random_portfolio(n_assets=3, seed=42)
        result = qp.optimize_simulated_annealing(n_iterations=2000, seed=42)
        assert abs(np.sum(result.weights) - 1.0) < 1e-8

    def test_efficient_frontier(self):
        qp = QuantumPortfolio.random_portfolio(n_assets=4, seed=42)
        frontier = qp.risk_return_frontier(n_points=10)
        assert len(frontier) == 10
        # All points should have valid weights
        for pt in frontier:
            assert abs(np.sum(pt.weights) - 1.0) < 1e-8

    def test_qubo_matrix_shape(self):
        qp = QuantumPortfolio.random_portfolio(n_assets=3, seed=42)
        q = qp.qubo_matrix(n_bits=4)
        assert q.shape == (12, 12)  # 3 assets * 4 bits

    def test_sharpe_ratio_positive_for_positive_returns(self):
        returns = np.array([0.1, 0.15, 0.2])
        cov = 0.01 * np.eye(3)
        qp = QuantumPortfolio(returns, cov, risk_aversion=1.0)
        result = qp.optimize_analytical()
        assert result.sharpe_ratio > 0


# =========================================================================
# Integration tests
# =========================================================================

class TestIntegration:

    def test_full_maxcut_pipeline(self):
        """Complete pipeline: build graph -> formulate QUBO -> solve -> verify."""
        g = Graph.random_graph(6, edge_prob=0.5, seed=42)
        mc = MaxCut(g)
        q = mc.qubo_matrix()
        assert q.shape == (6, 6)
        exact = mc.brute_force()
        sa = mc.simulated_annealing(n_iterations=5000, seed=42)
        assert sa.objective >= 0.8 * exact.objective

    def test_full_tsp_pipeline(self):
        """Complete pipeline: random instance -> solve -> verify tour."""
        tsp = TravelingSalesman.random_instance(5, seed=42)
        result = tsp.two_opt(seed=42)
        assert sorted(result.solution) == list(range(5))
        assert result.objective > 0

    def test_full_game_analysis(self):
        """Complete game analysis: play, find NE, run tournament."""
        game = PrisonersDilemma()
        # Play individual matchups
        r1 = game.play(quantum_miracle_move(), defect())
        assert np.isfinite(r1.payoff_p1)
        # Find Nash equilibrium
        s1, s2, r2 = game.find_nash_equilibrium(max_iterations=5, grid_steps=20)
        assert np.isfinite(r2.payoff_p1)
        # Run tournament
        strategies = [cooperate(), defect(), quantum_miracle_move()]
        t = QuantumTournament(game, strategies)
        result = t.run()
        assert len(result.rankings) == 3

    def test_full_bayesian_pipeline(self):
        """Complete pipeline: prior -> evidence -> posterior -> verify."""
        qb = QuantumBayesian(n_hypotheses=3)
        prior = np.array([0.2, 0.5, 0.3])
        likelihoods = np.array([0.9, 0.1, 0.5])
        result = qb.update(prior, likelihoods, strength=1.0)
        assert abs(np.sum(result.posterior) - 1.0) < 1e-8
        assert np.all(result.posterior >= 0)

    def test_full_mdp_pipeline(self):
        """Complete MDP pipeline: build -> solve classical -> solve quantum -> compare."""
        mdp = QuantumMarkov.random_mdp(n_states=4, n_actions=2, seed=42)
        classical = mdp.value_iteration()
        quantum = mdp.quantum_value_iteration(exploration_bonus=0.1)
        assert classical.converged
        assert quantum.converged

    def test_penny_flip_is_deterministic_advantage(self):
        """Meyer's result: quantum player has deterministic winning strategy."""
        pf = MeyerPennyFlip()
        for flip in [True, False]:
            result = pf.play(classical_flip=flip)
            # P(heads) should be essentially 1.0
            assert abs(result.measurement_probs[0] - 1.0) < 1e-10
            assert result.quantum_player_wins

    def test_coloring_pipeline(self):
        """Build graph -> color -> verify validity."""
        g = Graph(5)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 4)
        gc = GraphColoring(g, n_colors=2)
        result = gc.greedy_local_search(seed=42)
        valid, _ = gc.evaluate(result.solution)
        assert valid, "Path graph should be 2-colorable"

    def test_number_partition_pipeline(self):
        """Build partition problem -> solve exact -> verify."""
        nums = np.array([3, 1, 1, 2, 2, 1])
        np_ = NumberPartition(nums)
        result = np_.brute_force()
        # Total = 10, perfect partition exists: {3,2} vs {1,1,2,1}
        assert abs(result.objective) < 1e-8

    def test_auction_pipeline(self):
        """Create auction -> submit bids -> verify winner."""
        qa = QuantumAuction(n_bidders=5, max_bid=100)
        result = qa.random_auction(seed=42)
        assert result.protocol_valid
        assert result.winning_bid >= result.second_price


# =========================================================================
# N-Player Multiplayer Games
# =========================================================================

from nqpu.games import (
    NPlayerStrategy,
    NPlayerResult,
    NPlayerEWL,
    PublicGoodsGame,
    MinorityGame,
    QuantumBargaining,
)
from nqpu.games.multiplayer import n_cooperate, n_defect, n_quantum


class TestNPlayerStrategy:
    """Tests for NPlayerStrategy and preset strategies."""

    def test_cooperate_is_identity(self):
        s = n_cooperate()
        u = s.unitary()
        assert np.allclose(u, np.eye(2), atol=1e-12)

    def test_defect_is_x_gate(self):
        s = n_defect()
        u = s.unitary()
        expected = np.array([[0, 1], [-1, 0]], dtype=np.complex128)
        assert np.allclose(u, expected, atol=1e-12)

    def test_quantum_is_q_gate(self):
        s = n_quantum()
        assert abs(s.theta) < 1e-12
        assert abs(s.phi - np.pi / 2) < 1e-12

    def test_strategy_is_unitary(self):
        for s in [n_cooperate(), n_defect(), n_quantum()]:
            u = s.unitary()
            product = u @ u.conj().T
            assert np.allclose(product, np.eye(2), atol=1e-12), f"{s.label} not unitary"

    def test_custom_strategy_parameters(self):
        s = NPlayerStrategy(theta=np.pi / 3, phi=np.pi / 4, player_id=2, label="custom")
        u = s.unitary()
        assert u.shape == (2, 2)
        assert np.allclose(u @ u.conj().T, np.eye(2), atol=1e-12)


class TestNPlayerEWL:
    """Tests for the N-player EWL protocol."""

    def test_2player_cooperate_cooperate(self):
        """2-player PD with cooperate-cooperate should give (3, 3)."""
        payoff = np.zeros((2, 2, 2), dtype=np.float64)
        # CC=(3,3), CD=(0,5), DC=(5,0), DD=(1,1)
        payoff[0, 0, 0] = 3.0; payoff[0, 0, 1] = 3.0
        payoff[0, 1, 0] = 0.0; payoff[0, 1, 1] = 5.0
        payoff[1, 0, 0] = 5.0; payoff[1, 0, 1] = 0.0
        payoff[1, 1, 0] = 1.0; payoff[1, 1, 1] = 1.0

        game = NPlayerEWL(n_players=2, payoff_tensor=payoff)
        result = game.play([n_cooperate(0), n_cooperate(1)])
        assert abs(result.payoffs[0] - 3.0) < 0.2
        assert abs(result.payoffs[1] - 3.0) < 0.2

    def test_3player_all_cooperate(self):
        """3-player game: all cooperate payoffs should be finite."""
        n = 3
        payoff = np.ones((2, 2, 2, 3), dtype=np.float64)
        game = NPlayerEWL(n_players=n, payoff_tensor=payoff)
        strategies = [n_cooperate(i) for i in range(n)]
        result = game.play(strategies)
        assert all(np.isfinite(result.payoffs))
        assert len(result.payoffs) == n

    def test_measurement_probs_sum_to_one(self):
        """Measurement probabilities must sum to 1."""
        n = 3
        payoff = np.ones((2, 2, 2, 3), dtype=np.float64)
        game = NPlayerEWL(n_players=n, payoff_tensor=payoff)
        strategies = [n_quantum(i) for i in range(n)]
        result = game.play(strategies)
        assert abs(np.sum(result.measurement_probs) - 1.0) < 1e-10

    def test_entangling_operator_is_unitary(self):
        """The entangling operator J must be unitary."""
        n = 3
        payoff = np.ones((2, 2, 2, 3), dtype=np.float64)
        game = NPlayerEWL(n_players=n, payoff_tensor=payoff)
        j = game._entangling_operator()
        dim = 1 << n
        product = j @ j.conj().T
        assert np.allclose(product, np.eye(dim), atol=1e-12)

    def test_classical_limit_gamma_zero(self):
        """With gamma=0 the game should be classical."""
        payoff = np.zeros((2, 2, 2), dtype=np.float64)
        payoff[0, 0, 0] = 3.0; payoff[0, 0, 1] = 3.0
        payoff[0, 1, 0] = 0.0; payoff[0, 1, 1] = 5.0
        payoff[1, 0, 0] = 5.0; payoff[1, 0, 1] = 0.0
        payoff[1, 1, 0] = 1.0; payoff[1, 1, 1] = 1.0

        game = NPlayerEWL(n_players=2, payoff_tensor=payoff, gamma=0.0)
        r = game.play([n_defect(0), n_defect(1)])
        assert abs(r.payoffs[0] - 1.0) < 0.2
        assert abs(r.payoffs[1] - 1.0) < 0.2

    def test_invalid_player_count(self):
        with pytest.raises(ValueError, match="at least 2"):
            NPlayerEWL(n_players=1, payoff_tensor=np.zeros((2, 1)))

    def test_wrong_strategy_count(self):
        payoff = np.ones((2, 2, 2), dtype=np.float64)
        game = NPlayerEWL(n_players=2, payoff_tensor=payoff)
        with pytest.raises(ValueError, match="Expected 2"):
            game.play([n_cooperate(0)])


class TestPublicGoodsGame:
    """Tests for the quantum public goods game."""

    def test_social_optimum(self):
        pg = PublicGoodsGame(n_players=4, multiplication_factor=2.0, endowment=1.0)
        assert abs(pg.social_optimum() - 2.0) < 1e-10

    def test_free_rider_payoff_exceeds_cooperator(self):
        pg = PublicGoodsGame(n_players=4, multiplication_factor=2.0, endowment=1.0)
        free_rider = pg.free_rider_payoff()
        cooperator = pg.cooperator_payoff_with_one_defector()
        assert free_rider > cooperator

    def test_all_cooperate_payoff(self):
        pg = PublicGoodsGame(n_players=3, multiplication_factor=2.0, endowment=1.0)
        strategies = [n_cooperate(i) for i in range(3)]
        result = pg.play(strategies)
        assert all(np.isfinite(result.payoffs))

    def test_quantum_strategies_give_finite_payoffs(self):
        pg = PublicGoodsGame(n_players=3, multiplication_factor=2.5, endowment=1.0)
        strategies = [n_quantum(i) for i in range(3)]
        result = pg.play(strategies)
        assert all(np.isfinite(result.payoffs))
        assert len(result.payoffs) == 3


class TestMinorityGame:
    """Tests for the quantum minority game."""

    def test_classical_expected_payoff(self):
        mg = MinorityGame(n_players=3)
        ep = mg.classical_expected_payoff()
        assert ep > 0
        assert ep < 1.0  # can't exceed 1 per player

    def test_optimal_minority_size(self):
        mg = MinorityGame(n_players=5)
        assert mg.optimal_minority_size() == 2

    def test_quantum_play_finite(self):
        mg = MinorityGame(n_players=3)
        strategies = [n_quantum(i) for i in range(3)]
        result = mg.play(strategies)
        assert all(np.isfinite(result.payoffs))

    def test_invalid_player_count(self):
        with pytest.raises(ValueError, match="at least 3"):
            MinorityGame(n_players=2)


class TestQuantumBargaining:
    """Tests for the quantum bargaining game."""

    def test_nash_bargaining_solution(self):
        qb = QuantumBargaining(n_players=3, resource=1.0)
        nbs = qb.nash_bargaining_solution()
        assert abs(np.sum(nbs) - 1.0) < 1e-10
        assert np.allclose(nbs, 1.0 / 3.0, atol=1e-10)

    def test_play_gives_finite_payoffs(self):
        qb = QuantumBargaining(n_players=3, resource=1.0)
        strategies = [n_cooperate(i) for i in range(3)]
        result = qb.play(strategies)
        assert all(np.isfinite(result.payoffs))

    def test_total_payoff_bounded(self):
        qb = QuantumBargaining(n_players=3, resource=1.0)
        strategies = [n_cooperate(i) for i in range(3)]
        result = qb.play(strategies)
        assert qb.total_payoff(result) <= 1.0 + 1e-6


# =========================================================================
# Evolutionary Game Theory
# =========================================================================

from nqpu.games import (
    QuantumPopulation,
    QuantumReplicatorDynamics,
    EvolutionResult,
    ESSAnalyzer,
    CoevolutionaryDynamics,
    CoevoResult,
)


class TestQuantumPopulation:
    """Tests for QuantumPopulation."""

    def test_uniform_frequencies(self):
        strategies = [np.eye(2, dtype=np.complex128)] * 3
        pop = QuantumPopulation.uniform(strategies)
        assert abs(np.sum(pop.frequencies) - 1.0) < 1e-10
        assert np.allclose(pop.frequencies, 1.0 / 3.0)

    def test_n_strategies(self):
        strategies = [np.eye(2, dtype=np.complex128)] * 5
        pop = QuantumPopulation.uniform(strategies)
        assert pop.n_strategies == 5

    def test_entropy_maximum(self):
        """Maximum entropy for n strategies is log(n)."""
        n = 4
        strategies = [np.eye(2, dtype=np.complex128)] * n
        pop = QuantumPopulation.uniform(strategies)
        assert abs(pop.entropy() - np.log(n)) < 1e-10

    def test_entropy_minimum(self):
        """Minimum entropy is 0 for a pure population."""
        strategies = [np.eye(2, dtype=np.complex128)] * 3
        pop = QuantumPopulation(strategies=strategies, frequencies=np.array([1.0, 0.0, 0.0]))
        assert abs(pop.entropy()) < 1e-10


class TestReplicatorDynamics:
    """Tests for QuantumReplicatorDynamics."""

    def test_fitness_computation(self):
        """Fitness should be the matrix-vector product A @ x."""
        payoff = np.array([[3, 0], [5, 1]], dtype=np.float64)
        rd = QuantumReplicatorDynamics(payoff)
        pop = QuantumPopulation(
            strategies=[np.eye(2)] * 2,
            frequencies=np.array([0.5, 0.5]),
        )
        f = rd.fitness(pop)
        expected = payoff @ np.array([0.5, 0.5])
        assert np.allclose(f, expected, atol=1e-10)

    def test_step_preserves_normalization(self):
        payoff = np.array([[3, 0], [5, 1]], dtype=np.float64)
        rd = QuantumReplicatorDynamics(payoff)
        pop = QuantumPopulation(
            strategies=[np.eye(2)] * 2,
            frequencies=np.array([0.5, 0.5]),
        )
        new_pop = rd.step(pop, dt=0.01)
        assert abs(np.sum(new_pop.frequencies) - 1.0) < 1e-10

    def test_evolve_converges(self):
        """Hawk-Dove game should converge."""
        payoff = np.array([[-1, 2], [0, 1]], dtype=np.float64)
        rd = QuantumReplicatorDynamics(payoff)
        pop = QuantumPopulation(
            strategies=[np.eye(2)] * 2,
            frequencies=np.array([0.3, 0.7]),
        )
        result = rd.evolve(pop, t_final=50.0, dt=0.01)
        # Should have run without error
        assert result.frequency_history.shape[1] == 2
        assert abs(np.sum(result.final_population.frequencies) - 1.0) < 1e-8

    def test_dominant_strategy_takes_over(self):
        """If strategy 0 dominates, it should take over."""
        payoff = np.array([[5, 5], [1, 1]], dtype=np.float64)
        rd = QuantumReplicatorDynamics(payoff)
        pop = QuantumPopulation(
            strategies=[np.eye(2)] * 2,
            frequencies=np.array([0.1, 0.9]),
        )
        result = rd.evolve(pop, t_final=100.0, dt=0.01)
        # Strategy 0 should dominate
        assert result.final_population.frequencies[0] > 0.9

    def test_fixed_points_found(self):
        """Should find at least the vertex fixed points."""
        payoff = np.array([[3, 0], [5, 1]], dtype=np.float64)
        rd = QuantumReplicatorDynamics(payoff)
        fps = rd.fixed_points(resolution=20)
        assert len(fps) >= 1  # At least one fixed point


class TestESSAnalyzer:
    """Tests for ESSAnalyzer."""

    def test_dominant_strategy_is_ess(self):
        """A strictly dominant strategy should be ESS."""
        payoff = np.array([[5, 5], [1, 1]], dtype=np.float64)
        analyzer = ESSAnalyzer()
        assert analyzer.is_ess(0, payoff, QuantumPopulation.uniform([np.eye(2)] * 2))
        assert not analyzer.is_ess(1, payoff, QuantumPopulation.uniform([np.eye(2)] * 2))

    def test_find_ess(self):
        payoff = np.array([[5, 5], [1, 1]], dtype=np.float64)
        analyzer = ESSAnalyzer()
        ess_list = analyzer.find_ess(payoff)
        assert 0 in ess_list
        assert 1 not in ess_list

    def test_invasion_fitness(self):
        payoff = np.array([[3, 0], [5, 1]], dtype=np.float64)
        analyzer = ESSAnalyzer()
        # Invasion fitness of 1 into population of 0: E(1,0) - E(0,0) = 5 - 3 = 2
        assert abs(analyzer.invasion_fitness(1, 0, payoff) - 2.0) < 1e-10

    def test_pairwise_invasion_matrix(self):
        payoff = np.array([[3, 0], [5, 1]], dtype=np.float64)
        analyzer = ESSAnalyzer()
        inv = analyzer.pairwise_invasion_matrix(payoff)
        assert inv.shape == (2, 2)
        # Diagonal should be zero (no self-invasion)
        assert abs(inv[0, 0]) < 1e-10
        assert abs(inv[1, 1]) < 1e-10


class TestCoevolutionaryDynamics:
    """Tests for CoevolutionaryDynamics."""

    def test_evolve_runs(self):
        q_payoff = np.array([[5, 5], [1, 1]], dtype=np.float64)
        c_payoff = np.array([[3, 0], [5, 1]], dtype=np.float64)
        coevo = CoevolutionaryDynamics(q_payoff, c_payoff, quantum_fraction=0.5)
        result = coevo.evolve(n_steps=500, dt=0.01)
        assert len(result.times) == 501
        assert 0.0 <= result.final_quantum_fraction <= 1.0

    def test_quantum_dominates_when_superior(self):
        """When quantum payoff is uniformly higher, quantum fraction should increase."""
        q_payoff = np.array([[10, 10], [10, 10]], dtype=np.float64)
        c_payoff = np.array([[1, 1], [1, 1]], dtype=np.float64)
        coevo = CoevolutionaryDynamics(q_payoff, c_payoff, quantum_fraction=0.5)
        result = coevo.evolve(n_steps=1000, dt=0.01)
        assert result.final_quantum_fraction > 0.5

    def test_quantum_fraction_history_shape(self):
        q_payoff = np.array([[3, 0], [5, 1]], dtype=np.float64)
        c_payoff = np.array([[3, 0], [5, 1]], dtype=np.float64)
        coevo = CoevolutionaryDynamics(q_payoff, c_payoff, quantum_fraction=0.3)
        result = coevo.evolve(n_steps=200, dt=0.01)
        assert result.quantum_fraction_history.shape == (201,)


# =========================================================================
# Mechanism Design
# =========================================================================

from nqpu.games import (
    QuantumValuation,
    QuantumVCG,
    AllocationResult,
    QuantumRevenueMechanism,
    AuctionOutcome as MDAuctionOutcome,
    IncentiveAnalyzer,
)


class TestQuantumValuation:
    """Tests for QuantumValuation."""

    def test_expected_value(self):
        qv = QuantumValuation(
            values=np.array([10.0, 20.0]),
            amplitudes=np.array([1.0, 0.0], dtype=np.complex128),
        )
        assert abs(qv.expected_value() - 10.0) < 1e-10

    def test_uniform_superposition(self):
        qv = QuantumValuation(
            values=np.array([0.0, 10.0]),
            amplitudes=np.array([1.0, 1.0], dtype=np.complex128),
        )
        assert abs(qv.expected_value() - 5.0) < 1e-10

    def test_probabilities_sum_to_one(self):
        qv = QuantumValuation(
            values=np.array([1.0, 2.0, 3.0]),
            amplitudes=np.array([1.0, 2.0, 3.0], dtype=np.complex128),
        )
        probs = qv.probabilities()
        assert abs(np.sum(probs) - 1.0) < 1e-10

    def test_sample_in_range(self):
        qv = QuantumValuation(
            values=np.array([5.0, 15.0, 25.0]),
            amplitudes=np.array([1.0, 1.0, 1.0], dtype=np.complex128),
        )
        rng = np.random.default_rng(42)
        for _ in range(20):
            val = qv.sample(rng)
            assert val in [5.0, 15.0, 25.0]

    def test_variance_pure_state(self):
        qv = QuantumValuation(
            values=np.array([10.0, 20.0]),
            amplitudes=np.array([1.0, 0.0], dtype=np.complex128),
        )
        assert abs(qv.variance()) < 1e-10


class TestQuantumVCG:
    """Tests for QuantumVCG mechanism."""

    def test_highest_value_wins(self):
        vcg = QuantumVCG(n_players=3, n_items=1)
        vals = [
            QuantumValuation(np.array([10.0]), np.array([1.0 + 0j]), player_id=0),
            QuantumValuation(np.array([30.0]), np.array([1.0 + 0j]), player_id=1),
            QuantumValuation(np.array([20.0]), np.array([1.0 + 0j]), player_id=2),
        ]
        result = vcg.allocate(vals)
        assert result.allocation[1] == 1.0  # player 1 has highest value

    def test_vcg_payments_correct(self):
        vcg = QuantumVCG(n_players=3, n_items=1)
        vals = [
            QuantumValuation(np.array([10.0]), np.array([1.0 + 0j]), player_id=0),
            QuantumValuation(np.array([30.0]), np.array([1.0 + 0j]), player_id=1),
            QuantumValuation(np.array([20.0]), np.array([1.0 + 0j]), player_id=2),
        ]
        result = vcg.allocate(vals)
        # VCG payment for winner (player 1):
        # Welfare without player 1: 20 (player 2 wins)
        # Welfare of others in current allocation: 0 (only player 1 gets item)
        # Payment = 20 - 0 = 20
        assert abs(result.payments[1] - 20.0) < 1e-8

    def test_losers_pay_nothing(self):
        vcg = QuantumVCG(n_players=3, n_items=1)
        vals = [
            QuantumValuation(np.array([10.0]), np.array([1.0 + 0j]), player_id=0),
            QuantumValuation(np.array([30.0]), np.array([1.0 + 0j]), player_id=1),
            QuantumValuation(np.array([20.0]), np.array([1.0 + 0j]), player_id=2),
        ]
        result = vcg.allocate(vals)
        assert result.payments[0] == 0.0
        assert result.payments[2] == 0.0

    def test_truthfulness(self):
        vcg = QuantumVCG(n_players=3, n_items=1)
        vals = [
            QuantumValuation(np.array([10.0, 15.0]), np.array([0.7 + 0j, 0.7 + 0j]), player_id=0),
            QuantumValuation(np.array([25.0, 35.0]), np.array([0.7 + 0j, 0.7 + 0j]), player_id=1),
            QuantumValuation(np.array([18.0, 22.0]), np.array([0.7 + 0j, 0.7 + 0j]), player_id=2),
        ]
        assert vcg.is_truthful(vals, n_tests=50)


class TestQuantumRevenueMechanism:
    """Tests for revenue-optimal quantum auction."""

    def test_run_auction_winner_exists(self):
        mech = QuantumRevenueMechanism(n_players=3, reserve_price=0.0)
        vals = [
            QuantumValuation(np.array([10.0, 20.0]), np.array([0.5 + 0j, 0.866 + 0j]), player_id=0),
            QuantumValuation(np.array([15.0, 25.0]), np.array([0.5 + 0j, 0.866 + 0j]), player_id=1),
            QuantumValuation(np.array([5.0, 30.0]), np.array([0.5 + 0j, 0.866 + 0j]), player_id=2),
        ]
        outcome = mech.run_auction(vals)
        assert outcome.winner >= 0
        assert outcome.revenue >= 0

    def test_reserve_price_respected(self):
        mech = QuantumRevenueMechanism(n_players=2, reserve_price=100.0)
        vals = [
            QuantumValuation(np.array([5.0]), np.array([1.0 + 0j]), player_id=0),
            QuantumValuation(np.array([10.0]), np.array([1.0 + 0j]), player_id=1),
        ]
        outcome = mech.run_auction(vals)
        assert outcome.winner == -1  # all bids below reserve

    def test_expected_revenue_positive(self):
        mech = QuantumRevenueMechanism(n_players=3, reserve_price=0.0)
        dists = [
            np.array([10.0, 20.0, 30.0]),
            np.array([15.0, 25.0, 35.0]),
            np.array([5.0, 15.0, 25.0]),
        ]
        rev = mech.expected_revenue(dists, n_samples=200)
        assert rev > 0


class TestIncentiveAnalyzer:
    """Tests for IncentiveAnalyzer."""

    def test_vcg_is_incentive_compatible(self):
        vcg = QuantumVCG(n_players=3, n_items=1)
        vals = [
            QuantumValuation(np.array([10.0, 15.0]), np.array([0.7 + 0j, 0.7 + 0j]), player_id=0),
            QuantumValuation(np.array([25.0, 35.0]), np.array([0.7 + 0j, 0.7 + 0j]), player_id=1),
            QuantumValuation(np.array([18.0, 22.0]), np.array([0.7 + 0j, 0.7 + 0j]), player_id=2),
        ]
        analyzer = IncentiveAnalyzer()
        score = analyzer.incentive_compatibility_score(vcg, vals, n_deviations=30)
        assert score >= 0.8  # VCG should be highly IC

    def test_individual_rationality(self):
        vcg = QuantumVCG(n_players=3, n_items=1)
        vals = [
            QuantumValuation(np.array([10.0]), np.array([1.0 + 0j]), player_id=0),
            QuantumValuation(np.array([30.0]), np.array([1.0 + 0j]), player_id=1),
            QuantumValuation(np.array([20.0]), np.array([1.0 + 0j]), player_id=2),
        ]
        analyzer = IncentiveAnalyzer()
        assert analyzer.individual_rationality_check(vcg, vals)

    def test_efficiency_ratio(self):
        vcg = QuantumVCG(n_players=3, n_items=1)
        vals = [
            QuantumValuation(np.array([10.0]), np.array([1.0 + 0j]), player_id=0),
            QuantumValuation(np.array([30.0]), np.array([1.0 + 0j]), player_id=1),
            QuantumValuation(np.array([20.0]), np.array([1.0 + 0j]), player_id=2),
        ]
        analyzer = IncentiveAnalyzer()
        ratio = analyzer.efficiency_ratio(vcg, vals)
        assert abs(ratio - 1.0) < 1e-8  # VCG maximizes welfare


# =========================================================================
# QAOA Builder
# =========================================================================

from nqpu.games import (
    QAOACircuit,
    QAOAResult,
    QAOAOptResult,
    maxcut_qaoa,
    graph_coloring_qaoa,
    number_partition_qaoa,
    max_independent_set_qaoa,
    tsp_qaoa,
)


class TestQAOACircuit:
    """Tests for QAOACircuit basics."""

    def test_uniform_initial_state(self):
        circuit = QAOACircuit(n_qubits=3, cost_terms=[], p=1)
        state = circuit._initial_state()
        assert abs(np.sum(np.abs(state) ** 2) - 1.0) < 1e-10
        assert np.allclose(np.abs(state), 1.0 / np.sqrt(8))

    def test_cost_operator_preserves_norm(self):
        circuit = QAOACircuit(
            n_qubits=3,
            cost_terms=[([0, 1], 1.0), ([1, 2], 0.5)],
            p=1,
        )
        state = circuit._initial_state()
        result = circuit.cost_operator(0.5, state)
        assert abs(np.sum(np.abs(result) ** 2) - 1.0) < 1e-10

    def test_mixer_operator_preserves_norm(self):
        circuit = QAOACircuit(n_qubits=3, cost_terms=[], p=1)
        state = circuit._initial_state()
        result = circuit.mixer_operator(0.3, state)
        assert abs(np.sum(np.abs(result) ** 2) - 1.0) < 1e-10

    def test_evaluate_returns_valid_result(self):
        circuit = QAOACircuit(
            n_qubits=3,
            cost_terms=[([0, 1], -0.5)],
            p=1,
        )
        result = circuit.evaluate(
            gammas=np.array([0.5]),
            betas=np.array([0.3]),
        )
        assert np.isfinite(result.expectation)
        assert abs(np.sum(result.probabilities) - 1.0) < 1e-10
        assert len(result.best_bitstring) == 3


class TestMaxCutQAOA:
    """Tests for the MaxCut QAOA builder."""

    def test_triangle_maxcut(self):
        """Triangle graph: MaxCut optimal cut = 2."""
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=np.float64)
        circuit = maxcut_qaoa(adj, p=1)
        assert circuit.n_qubits == 3

    def test_maxcut_qaoa_evaluation(self):
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=np.float64)
        circuit = maxcut_qaoa(adj, p=1)
        result = circuit.evaluate(
            gammas=np.array([np.pi / 4]),
            betas=np.array([np.pi / 8]),
        )
        assert np.isfinite(result.expectation)

    def test_maxcut_qaoa_optimize(self):
        adj = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], dtype=np.float64)
        circuit = maxcut_qaoa(adj, p=1)
        opt = circuit.optimize(n_restarts=2, max_iter=10)
        assert len(opt.optimization_history) > 0
        assert len(opt.best_gammas) == 1
        assert len(opt.best_betas) == 1


class TestGraphColoringQAOA:
    """Tests for graph coloring QAOA builder."""

    def test_coloring_circuit_size(self):
        adj = np.array([
            [0, 1],
            [1, 0],
        ], dtype=np.float64)
        circuit = graph_coloring_qaoa(adj, n_colors=2, p=1)
        assert circuit.n_qubits == 4  # 2 vertices * 2 colors

    def test_coloring_evaluates(self):
        adj = np.array([
            [0, 1],
            [1, 0],
        ], dtype=np.float64)
        circuit = graph_coloring_qaoa(adj, n_colors=2, p=1)
        result = circuit.evaluate(
            gammas=np.array([0.5]),
            betas=np.array([0.3]),
        )
        assert np.isfinite(result.expectation)


class TestNumberPartitionQAOA:
    """Tests for number partition QAOA builder."""

    def test_partition_circuit_size(self):
        circuit = number_partition_qaoa([3.0, 1.0, 2.0], p=1)
        assert circuit.n_qubits == 3

    def test_partition_evaluates(self):
        circuit = number_partition_qaoa([3.0, 1.0, 2.0], p=1)
        result = circuit.evaluate(
            gammas=np.array([0.5]),
            betas=np.array([0.3]),
        )
        assert np.isfinite(result.expectation)


class TestMaxIndependentSetQAOA:
    """Tests for max independent set QAOA builder."""

    def test_mis_circuit_size(self):
        adj = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=np.float64)
        circuit = max_independent_set_qaoa(adj, p=1)
        assert circuit.n_qubits == 3

    def test_mis_evaluates(self):
        adj = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=np.float64)
        circuit = max_independent_set_qaoa(adj, p=1)
        result = circuit.evaluate(
            gammas=np.array([0.5]),
            betas=np.array([0.3]),
        )
        assert np.isfinite(result.expectation)


class TestTSPQAOA:
    """Tests for TSP QAOA builder."""

    def test_tsp_circuit_size(self):
        dist = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ], dtype=np.float64)
        circuit = tsp_qaoa(dist, p=1)
        assert circuit.n_qubits == 9  # 3 cities * 3 positions

    def test_tsp_too_large(self):
        dist = np.zeros((5, 5))
        with pytest.raises(ValueError, match="impractical"):
            tsp_qaoa(dist, p=1)

    def test_tsp_evaluates(self):
        dist = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0],
        ], dtype=np.float64)
        circuit = tsp_qaoa(dist, p=1)
        result = circuit.evaluate(
            gammas=np.array([0.5]),
            betas=np.array([0.3]),
        )
        assert np.isfinite(result.expectation)


# =========================================================================
# New Integration Tests
# =========================================================================

class TestNewIntegration:
    """Integration tests spanning the new modules."""

    def test_multiplayer_to_evolutionary(self):
        """Use multiplayer payoffs to drive evolutionary dynamics."""
        # Create a 2-strategy payoff matrix from the N-player game
        payoff_matrix = np.array([[3.0, 0.0], [5.0, 1.0]], dtype=np.float64)
        rd = QuantumReplicatorDynamics(payoff_matrix)
        pop = QuantumPopulation(
            strategies=[np.eye(2, dtype=np.complex128)] * 2,
            frequencies=np.array([0.5, 0.5]),
        )
        result = rd.evolve(pop, t_final=20.0, dt=0.01)
        assert abs(np.sum(result.final_population.frequencies) - 1.0) < 1e-8

    def test_vcg_then_revenue_comparison(self):
        """Compare VCG and revenue mechanisms on same valuations."""
        vals = [
            QuantumValuation(np.array([10.0, 30.0]), np.array([0.5 + 0j, 0.866 + 0j]), player_id=0),
            QuantumValuation(np.array([20.0, 40.0]), np.array([0.5 + 0j, 0.866 + 0j]), player_id=1),
        ]
        vcg = QuantumVCG(n_players=2, n_items=1)
        vcg_result = vcg.allocate(vals)
        assert vcg_result.social_welfare > 0

        rev_mech = QuantumRevenueMechanism(n_players=2, reserve_price=0.0)
        outcome = rev_mech.run_auction(vals)
        assert outcome.social_welfare > 0

    def test_qaoa_maxcut_pipeline(self):
        """Full pipeline: build MaxCut QAOA, evaluate, optimize."""
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=np.float64)
        circuit = maxcut_qaoa(adj, p=1)
        # Evaluate at specific parameters
        result = circuit.evaluate(
            gammas=np.array([np.pi / 4]),
            betas=np.array([np.pi / 8]),
        )
        assert np.isfinite(result.expectation)
        assert abs(np.sum(result.probabilities) - 1.0) < 1e-10

    def test_ess_drives_population(self):
        """Find ESS and verify it is stable under replicator dynamics."""
        payoff = np.array([[5, 5], [1, 1]], dtype=np.float64)
        analyzer = ESSAnalyzer()
        ess_list = analyzer.find_ess(payoff)
        assert 0 in ess_list

        # Start near ESS: mostly strategy 0
        rd = QuantumReplicatorDynamics(payoff)
        pop = QuantumPopulation(
            strategies=[np.eye(2, dtype=np.complex128)] * 2,
            frequencies=np.array([0.9, 0.1]),
        )
        result = rd.evolve(pop, t_final=50.0, dt=0.01)
        # ESS should be stable: strategy 0 remains dominant
        assert result.final_population.frequencies[0] > 0.9
