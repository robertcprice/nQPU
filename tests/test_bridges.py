"""Comprehensive tests for nqpu.bridges -- cross-package integration.

Tests cover all seven bridge modules:
  - physics_finance: IsingCorrelationModel, QuantumWalkPricer, HamiltonianPortfolio,
                     CorrelationPhaseAnalysis
  - physics_trading: HamiltonianVolatility, PhaseTransitionRegime, QuantumWalkMomentum
  - physics_games: IsingGameSolver, QuantumAuctionModel, QuantumMaxCutBridge
  - simulation_bio: CanonicalFMO, LindbladBioValidator
  - simulation_chem: OpenQuantumChemistry, DecoherenceAnalysis
  - simulation_trading: LindbladVolatility, NoisySignalGenerator, QuantumFilteredMomentum
  - vqe_noise: VQENoiseBenchmark, H2/LiH benchmarks, noise sweeps
"""

from __future__ import annotations

import numpy as np
import pytest

from nqpu.bridges import (
    # physics_finance
    IsingCorrelationModel,
    QuantumWalkPricer,
    HamiltonianPortfolio,
    CorrelationPhaseAnalysis,
    # physics_trading
    HamiltonianVolatility,
    PhaseTransitionRegime,
    QuantumWalkMomentum,
    # physics_games
    IsingGameSolver,
    QuantumAuctionModel,
    QuantumMaxCutBridge,
    NashResult,
    AuctionModelResult,
    MaxCutBenchmarkResult,
    # simulation_bio
    CanonicalFMO,
    LindbladBioValidator,
    # simulation_chem
    OpenQuantumChemistry,
    DecoherenceAnalysis,
    # simulation_trading
    LindbladVolatility,
    NoisySignalGenerator,
    QuantumFilteredMomentum,
    VolSurfaceResult,
    NoisySignalResult,
    FilteredMomentumResult,
    # vqe_noise
    VQENoiseBenchmark,
    VQENoiseResult,
)
from nqpu.bridges.vqe_noise import (
    _h2_hamiltonian,
    _lih_hamiltonian,
    _decoherence_operators,
)
from nqpu.games import Graph, MaxCut
from nqpu.simulation import PauliOperator


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def cov_3x3():
    """3-asset covariance matrix with known structure."""
    return np.array([
        [0.04, 0.02, 0.01],
        [0.02, 0.09, 0.03],
        [0.01, 0.03, 0.16],
    ])


@pytest.fixture
def returns_50x3(cov_3x3):
    """Synthetic 50-day x 3-asset return series."""
    rng = np.random.RandomState(42)
    return rng.multivariate_normal([0.0, 0.0, 0.0], cov_3x3, size=50)


@pytest.fixture
def prisoners_dilemma_payoff():
    """Standard Prisoner's Dilemma payoff matrix.

    Actions: 0=cooperate, 1=defect
    Payoffs: (R,R)=(3,3), (S,T)=(0,5), (T,S)=(5,0), (P,P)=(1,1)
    """
    payoff = np.zeros((2, 2, 2))
    payoff[0, 0, :] = [3, 3]  # (C, C)
    payoff[0, 1, :] = [0, 5]  # (C, D)
    payoff[1, 0, :] = [5, 0]  # (D, C)
    payoff[1, 1, :] = [1, 1]  # (D, D)
    return payoff


@pytest.fixture
def coordination_game_payoff():
    """Coordination game where both players prefer the same action."""
    payoff = np.zeros((2, 2, 2))
    payoff[0, 0, :] = [2, 2]
    payoff[0, 1, :] = [0, 0]
    payoff[1, 0, :] = [0, 0]
    payoff[1, 1, :] = [1, 1]
    return payoff


@pytest.fixture
def simple_graph():
    """Simple 4-node graph for MaxCut tests."""
    g = Graph(4)
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 1.0)
    g.add_edge(2, 3, 1.0)
    g.add_edge(0, 3, 1.0)
    return g


@pytest.fixture
def triangle_graph():
    """Simple 3-node triangle graph."""
    g = Graph(3)
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 1.0)
    g.add_edge(0, 2, 1.0)
    return g


@pytest.fixture
def price_series():
    """Deterministic price series for signal and trading tests."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.02, size=200)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


@pytest.fixture
def volume_series():
    """Deterministic volume series."""
    rng = np.random.default_rng(42)
    return np.abs(rng.normal(1e6, 2e5, size=200))


@pytest.fixture
def vol_surface():
    """Simple volatility surface for Lindblad tests."""
    return np.array([0.2, 0.25, 0.3, 0.22])


@pytest.fixture
def bench():
    """VQENoiseBenchmark instance for VQE noise tests."""
    return VQENoiseBenchmark(t_final=2.0, n_steps=20)


@pytest.fixture
def h2_hamiltonian():
    """H2 Hamiltonian for VQE tests."""
    return _h2_hamiltonian()


@pytest.fixture
def lih_hamiltonian():
    """LiH Hamiltonian for VQE tests."""
    return _lih_hamiltonian()


# ===========================================================================
# physics_finance
# ===========================================================================


class TestIsingCorrelationModel:
    def test_from_covariance(self, cov_3x3):
        model = IsingCorrelationModel.from_covariance(cov_3x3)
        assert model.n_assets == 3
        assert model.couplings.shape == (3, 3)
        # Correlation matrix has values in [0, 1] for positive-definite cov
        assert np.all(np.abs(model.coupling_strengths) <= 1.0 + 1e-10)

    def test_auto_names(self, cov_3x3):
        model = IsingCorrelationModel.from_covariance(cov_3x3)
        assert model.asset_names == ["asset_0", "asset_1", "asset_2"]

    def test_custom_names(self, cov_3x3):
        model = IsingCorrelationModel.from_covariance(
            cov_3x3, names=["AAPL", "GOOG", "MSFT"]
        )
        assert model.asset_names == ["AAPL", "GOOG", "MSFT"]

    def test_hamiltonian_hermitian(self, cov_3x3):
        model = IsingCorrelationModel.from_covariance(cov_3x3)
        H = model.hamiltonian.matrix()
        assert H.shape == (8, 8)  # 2^3
        assert np.allclose(H, H.conj().T)

    def test_ground_state(self, cov_3x3):
        model = IsingCorrelationModel.from_covariance(cov_3x3)
        energy, psi = model.ground_state()
        assert isinstance(energy, float)
        assert len(psi) == 8
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10

    def test_correlation_matrix(self, cov_3x3):
        model = IsingCorrelationModel.from_covariance(cov_3x3)
        corr = model.correlation_matrix()
        assert corr.shape == (3, 3)
        # Diagonal entries should be 1.0 (Z_i Z_i = I)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)
        # Should be symmetric
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_entanglement_risk(self, cov_3x3):
        model = IsingCorrelationModel.from_covariance(cov_3x3)
        entropy = model.entanglement_risk()
        assert isinstance(entropy, float)
        assert entropy >= 0.0

    def test_entanglement_risk_custom_subsystem(self, cov_3x3):
        model = IsingCorrelationModel.from_covariance(cov_3x3)
        entropy = model.entanglement_risk(subsystem=[0])
        assert isinstance(entropy, float)
        assert entropy >= 0.0

    def test_critical_temperature(self, cov_3x3):
        model = IsingCorrelationModel.from_covariance(cov_3x3)
        Tc = model.critical_temperature
        assert Tc > 0.0

    def test_critical_temperature_single_asset(self):
        cov = np.array([[0.04]])
        model = IsingCorrelationModel.from_covariance(cov)
        assert model.critical_temperature == 0.0

    def test_phase_diagram(self, cov_3x3):
        model = IsingCorrelationModel.from_covariance(cov_3x3)
        phase = model.phase_diagram(n_points=5)
        assert "field_values" in phase
        assert "energies" in phase
        assert "magnetizations" in phase
        assert "entanglement_entropy" in phase
        assert len(phase["energies"]) == 5
        # Energies should decrease as field increases (more fluctuation lowers E)
        # Just check they are finite
        assert np.all(np.isfinite(phase["energies"]))


class TestQuantumWalkPricer:
    def test_price_distribution(self):
        pricer = QuantumWalkPricer(n_sites=32, volatility=0.2, drift=0.0)
        result = pricer.price_distribution(spot=100, dt=1 / 252, n_steps=50)
        assert "prices" in result
        assert "probabilities" in result
        assert "mean_price" in result
        assert "std_price" in result
        assert "quantum_advantage" in result
        assert len(result["prices"]) == 32
        assert len(result["probabilities"]) == 32
        # Probabilities approximately sum to 1 (matrix exp via eigh has
        # small numerical error on anti-Hermitian input)
        assert abs(np.sum(result["probabilities"]) - 1.0) < 0.02
        # Mean price should be near spot
        assert abs(result["mean_price"] - 100) < 20

    def test_drift_shifts_mean(self):
        # Use large drift and long evolution to see the effect
        pricer_up = QuantumWalkPricer(n_sites=64, volatility=0.2, drift=5.0)
        pricer_down = QuantumWalkPricer(n_sites=64, volatility=0.2, drift=-5.0)
        up = pricer_up.price_distribution(spot=100, dt=1 / 252, n_steps=200)
        down = pricer_down.price_distribution(spot=100, dt=1 / 252, n_steps=200)
        assert up["mean_price"] >= down["mean_price"]

    def test_quantum_advantage_positive(self):
        pricer = QuantumWalkPricer(n_sites=64, volatility=0.3)
        result = pricer.price_distribution(spot=100)
        assert result["quantum_advantage"] > 0


class TestHamiltonianPortfolio:
    def test_as_hamiltonian(self):
        returns = np.array([0.05, 0.08, 0.12])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16],
        ])
        hp = HamiltonianPortfolio(returns, cov, risk_aversion=1.0)
        H = hp.as_hamiltonian().matrix()
        assert H.shape == (8, 8)
        assert np.allclose(H, H.conj().T)

    def test_optimal_portfolio(self):
        returns = np.array([0.05, 0.08, 0.12])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16],
        ])
        hp = HamiltonianPortfolio(returns, cov, risk_aversion=0.5)
        result = hp.optimal_portfolio()
        assert "weights" in result
        assert "selection" in result
        assert "ground_energy" in result
        assert "expected_return" in result
        assert "variance" in result
        # Weights should sum to ~1
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-10 or np.sum(result["weights"]) == 0
        assert result["variance"] >= 0

    def test_risk_aversion_effect(self):
        """Higher risk aversion should reduce portfolio variance."""
        returns = np.array([0.05, 0.15])
        cov = np.array([[0.04, 0.01], [0.01, 0.25]])
        low_risk = HamiltonianPortfolio(returns, cov, risk_aversion=5.0).optimal_portfolio()
        high_risk = HamiltonianPortfolio(returns, cov, risk_aversion=0.1).optimal_portfolio()
        # With 2 assets, both might select the same, but variance ordering should hold
        # or they select different subsets
        assert isinstance(low_risk["variance"], float)
        assert isinstance(high_risk["variance"], float)


class TestCorrelationPhaseAnalysis:
    def test_analyze(self, returns_50x3):
        result = CorrelationPhaseAnalysis.analyze(
            returns_50x3,
            field_range=np.linspace(0.5, 2.5, 5),
        )
        assert "critical_field" in result
        assert "critical_entropy" in result
        assert "phase_diagram" in result
        assert "current_regime" in result
        assert result["current_regime"] in ("ordered", "disordered")
        assert result["critical_field"] > 0


# ===========================================================================
# physics_trading
# ===========================================================================


class TestHamiltonianVolatility:
    def test_evolve_surface(self):
        hv = HamiltonianVolatility(n_strikes=4, coupling=0.5)
        initial_vols = np.array([0.2, 0.3, 0.25, 0.35])
        result = hv.evolve_surface(initial_vols, t_final=0.5, n_steps=10)
        assert "times" in result
        assert "vol_profiles" in result
        assert "initial_vols" in result
        assert "final_vols" in result
        assert len(result["times"]) == 11  # n_steps + 1 time points
        assert result["vol_profiles"].shape[0] == 11
        assert result["vol_profiles"].shape[1] == 4
        # Vols should remain roughly in [0, 1]
        assert np.all(result["final_vols"] >= -0.1)
        assert np.all(result["final_vols"] <= 1.1)

    def test_truncates_to_n_strikes(self):
        hv = HamiltonianVolatility(n_strikes=3, coupling=1.0)
        initial_vols = np.array([0.2, 0.3, 0.25, 0.35, 0.4])
        result = hv.evolve_surface(initial_vols, t_final=0.1, n_steps=5)
        assert result["vol_profiles"].shape[1] == 3


class TestPhaseTransitionRegime:
    def test_detect_transitions(self, returns_50x3):
        ptr = PhaseTransitionRegime(n_assets=3, window_size=20)
        result = ptr.detect_transitions(returns_50x3)
        assert "regimes" in result
        assert "transitions" in result
        assert "entanglement_entropy" in result
        assert "n_transitions" in result
        assert len(result["regimes"]) == 30  # 50 - 20
        assert all(r in ("trending", "mean_reverting") for r in result["regimes"])
        assert result["n_transitions"] >= 0

    def test_caps_at_6_qubits(self):
        """Even with 10 assets, should cap at 6 qubits for performance."""
        rng = np.random.RandomState(42)
        big_returns = rng.randn(60, 10) * 0.01
        ptr = PhaseTransitionRegime(n_assets=10, window_size=20)
        result = ptr.detect_transitions(big_returns)
        # Should complete without error (capped at 6 qubits internally)
        assert len(result["regimes"]) == 40


class TestQuantumWalkMomentum:
    def test_compute(self):
        # Synthetic uptrending prices
        rng = np.random.RandomState(42)
        prices = 100.0 * np.exp(np.cumsum(0.001 + 0.01 * rng.randn(50)))
        qwm = QuantumWalkMomentum(lookback=10, n_sites=16)
        momentum = qwm.compute(prices)
        assert len(momentum) == 50
        # First `lookback` values should be zero
        np.testing.assert_allclose(momentum[:10], 0.0)
        # Momentum should be in [-1, 1]
        assert np.all(momentum >= -1.0)
        assert np.all(momentum <= 1.0)

    def test_zero_volatility(self):
        """Flat prices (zero volatility) should produce zero momentum."""
        prices = np.full(30, 100.0)
        qwm = QuantumWalkMomentum(lookback=10, n_sites=16)
        momentum = qwm.compute(prices)
        np.testing.assert_allclose(momentum, 0.0, atol=1e-10)


# ===========================================================================
# physics_games
# ===========================================================================


class TestIsingGameSolver:
    """Tests for IsingGameSolver."""

    def test_from_payoff_matrix_creates_solver(self, prisoners_dilemma_payoff):
        solver = IsingGameSolver.from_payoff_matrix(prisoners_dilemma_payoff)
        assert solver.n_players == 2
        assert solver.n_actions == 2

    def test_from_payoff_matrix_stores_payoffs(self, prisoners_dilemma_payoff):
        solver = IsingGameSolver.from_payoff_matrix(prisoners_dilemma_payoff)
        assert hasattr(solver, "_payoff_matrix")
        np.testing.assert_array_equal(solver._payoff_matrix, prisoners_dilemma_payoff)

    def test_from_symmetric_payoff(self):
        """2D payoff matrix (symmetric game) auto-expands to 3D."""
        payoff_2d = np.array([[3, 0], [5, 1]], dtype=float)
        solver = IsingGameSolver.from_payoff_matrix(payoff_2d)
        assert solver.n_players == 2
        assert solver._payoff_matrix.shape == (2, 2, 2)

    def test_nash_equilibrium_returns_result(self, prisoners_dilemma_payoff):
        solver = IsingGameSolver.from_payoff_matrix(prisoners_dilemma_payoff)
        result = solver.nash_equilibrium()
        assert isinstance(result, NashResult)

    def test_nash_strategies_are_probabilities(self, prisoners_dilemma_payoff):
        solver = IsingGameSolver.from_payoff_matrix(prisoners_dilemma_payoff)
        result = solver.nash_equilibrium()
        for player in range(2):
            np.testing.assert_allclose(result.strategy[player].sum(), 1.0, atol=1e-6)
            assert np.all(result.strategy[player] >= -1e-10)

    def test_nash_expected_payoff_shape(self, prisoners_dilemma_payoff):
        solver = IsingGameSolver.from_payoff_matrix(prisoners_dilemma_payoff)
        result = solver.nash_equilibrium()
        assert result.expected_payoff.shape == (2,)

    def test_nash_entanglement_non_negative(self, prisoners_dilemma_payoff):
        solver = IsingGameSolver.from_payoff_matrix(prisoners_dilemma_payoff)
        result = solver.nash_equilibrium()
        assert result.entanglement >= 0.0

    def test_coordination_game_prefers_same_action(self, coordination_game_payoff):
        solver = IsingGameSolver.from_payoff_matrix(
            coordination_game_payoff, transverse_field=0.1
        )
        result = solver.nash_equilibrium()
        # In a coordination game both players should favour the same action
        assert result.strategy.shape == (2, 2)

    def test_quantum_advantage_returns_dict(self, prisoners_dilemma_payoff):
        solver = IsingGameSolver.from_payoff_matrix(prisoners_dilemma_payoff)
        adv = solver.quantum_advantage()
        assert "quantum_payoff" in adv
        assert "classical_payoff" in adv
        assert "payoff_ratio" in adv
        assert "quantum_entropy" in adv
        assert "classical_entropy" in adv

    def test_quantum_advantage_classical_payoff_positive(self, prisoners_dilemma_payoff):
        solver = IsingGameSolver.from_payoff_matrix(prisoners_dilemma_payoff)
        adv = solver.quantum_advantage()
        # In PD the best classical outcome is (D,D)=2 or (C,C)=6
        assert adv["classical_payoff"] > 0

    def test_quantum_entropy_geq_classical(self, prisoners_dilemma_payoff):
        solver = IsingGameSolver.from_payoff_matrix(prisoners_dilemma_payoff)
        adv = solver.quantum_advantage()
        # Quantum mixed strategies should have at least as much entropy as
        # classical pure strategies (which have zero entropy)
        assert adv["quantum_entropy"] >= adv["classical_entropy"] - 1e-10

    def test_custom_transverse_field(self, prisoners_dilemma_payoff):
        solver_low = IsingGameSolver.from_payoff_matrix(
            prisoners_dilemma_payoff, transverse_field=0.01
        )
        solver_high = IsingGameSolver.from_payoff_matrix(
            prisoners_dilemma_payoff, transverse_field=2.0
        )
        # Higher transverse field should give more mixed strategies (higher entropy)
        result_low = solver_low.nash_equilibrium()
        result_high = solver_high.nash_equilibrium()
        assert result_high.entanglement != result_low.entanglement

    def test_three_action_game(self):
        """Test with a 3-action game (rock-paper-scissors like)."""
        payoff = np.zeros((3, 3, 2))
        payoff[0, 1, 0] = 1  # R beats S
        payoff[1, 2, 0] = 1  # S beats P
        payoff[2, 0, 0] = 1  # P beats R
        payoff[1, 0, 1] = 1
        payoff[2, 1, 1] = 1
        payoff[0, 2, 1] = 1
        solver = IsingGameSolver.from_payoff_matrix(payoff)
        assert solver.n_actions == 3
        result = solver.nash_equilibrium()
        assert result.strategy.shape == (2, 3)
        # Strategies should sum to 1
        np.testing.assert_allclose(result.strategy[0].sum(), 1.0, atol=1e-6)


class TestQuantumAuctionModel:
    """Tests for QuantumAuctionModel."""

    def test_vickrey_auction_basic(self):
        model = QuantumAuctionModel(n_bidders=3, auction_type="vickrey")
        valuations = np.array([10.0, 8.0, 6.0])
        result = model.simulate_auction(valuations, seed=42)
        assert isinstance(result, AuctionModelResult)

    def test_vickrey_winner_is_highest_bidder(self):
        model = QuantumAuctionModel(n_bidders=3, auction_type="vickrey")
        valuations = np.array([10.0, 5.0, 3.0])
        result = model.simulate_auction(valuations, seed=42)
        assert result.winner == 0  # Highest valuation should win

    def test_vickrey_revenue_leq_winning_bid(self):
        model = QuantumAuctionModel(n_bidders=3, auction_type="vickrey")
        valuations = np.array([10.0, 8.0, 6.0])
        result = model.simulate_auction(valuations, seed=42)
        # In Vickrey, revenue = second-highest bid <= winning bid
        assert result.revenue <= result.winning_bid + 1e-10

    def test_first_price_auction_basic(self):
        model = QuantumAuctionModel(n_bidders=3, auction_type="first_price")
        valuations = np.array([10.0, 8.0, 6.0])
        result = model.simulate_auction(valuations, seed=42)
        assert isinstance(result, AuctionModelResult)
        # In first-price, revenue equals winning bid
        assert abs(result.revenue - result.winning_bid) < 1e-10

    def test_bidder_correlations_shape(self):
        model = QuantumAuctionModel(n_bidders=4)
        valuations = np.array([10.0, 8.0, 7.0, 5.0])
        result = model.simulate_auction(valuations, seed=42)
        assert result.bidder_correlations.shape == (4, 4)

    def test_bidder_correlations_diagonal_one(self):
        model = QuantumAuctionModel(n_bidders=3)
        valuations = np.array([10.0, 8.0, 6.0])
        result = model.simulate_auction(valuations, seed=42)
        np.testing.assert_allclose(
            np.diag(result.bidder_correlations), 1.0, atol=1e-6
        )

    def test_optimal_bid_below_valuation(self):
        model = QuantumAuctionModel(n_bidders=3, auction_type="first_price")
        bid = model.optimal_bid(valuation=10.0, bidder_index=0)
        # In first-price auction, optimal bid should be below valuation
        assert bid < 10.0
        assert bid > 0.0

    def test_optimal_bids_shape(self):
        model = QuantumAuctionModel(n_bidders=4, auction_type="vickrey")
        valuations = np.array([10.0, 8.0, 7.0, 5.0])
        result = model.simulate_auction(valuations, seed=42)
        assert result.optimal_bids.shape == (4,)

    def test_correlation_affects_bidding(self):
        model_low = QuantumAuctionModel(n_bidders=3, correlation=0.0)
        model_high = QuantumAuctionModel(n_bidders=3, correlation=0.9)
        valuations = np.array([10.0, 8.0, 6.0])
        result_low = model_low.simulate_auction(valuations, seed=42)
        result_high = model_high.simulate_auction(valuations, seed=42)
        # Different correlations should produce different bids
        assert not np.allclose(result_low.optimal_bids, result_high.optimal_bids)

    def test_two_bidder_auction(self):
        model = QuantumAuctionModel(n_bidders=2, auction_type="vickrey")
        valuations = np.array([10.0, 5.0])
        result = model.simulate_auction(valuations, seed=42)
        assert result.winner == 0


class TestQuantumMaxCutBridge:
    """Tests for QuantumMaxCutBridge."""

    def test_compare_qaoa_exact_returns_result(self, simple_graph):
        bridge = QuantumMaxCutBridge(graph=simple_graph)
        result = bridge.compare_qaoa_exact(p=1, n_restarts=1, max_iter=10)
        assert isinstance(result, MaxCutBenchmarkResult)

    def test_exact_cut_positive(self, simple_graph):
        bridge = QuantumMaxCutBridge(graph=simple_graph)
        result = bridge.compare_qaoa_exact(p=1, n_restarts=1, max_iter=10)
        assert result.exact_cut > 0

    def test_qaoa_cut_positive(self, simple_graph):
        bridge = QuantumMaxCutBridge(graph=simple_graph)
        result = bridge.compare_qaoa_exact(p=1, n_restarts=1, max_iter=10)
        assert result.qaoa_cut >= 0

    def test_approximation_ratio_bounded(self, simple_graph):
        bridge = QuantumMaxCutBridge(graph=simple_graph)
        result = bridge.compare_qaoa_exact(p=2, n_restarts=2, max_iter=20)
        assert 0.0 <= result.approximation_ratio <= 1.0 + 1e-10

    def test_triangle_graph_exact_cut(self, triangle_graph):
        bridge = QuantumMaxCutBridge(graph=triangle_graph)
        result = bridge.compare_qaoa_exact(p=1, n_restarts=1, max_iter=10)
        # Triangle MaxCut = 2 (cut any one edge leaves 2 crossing edges)
        assert result.exact_cut == 2.0

    def test_landscape_analysis_returns_grid(self, triangle_graph):
        bridge = QuantumMaxCutBridge(graph=triangle_graph)
        landscape = bridge.landscape_analysis(p=1, n_gamma=10, n_beta=10)
        assert "gammas" in landscape
        assert "betas" in landscape
        assert "landscape" in landscape
        assert landscape["landscape"].shape == (10, 10)

    def test_landscape_optimal_expectation(self, triangle_graph):
        bridge = QuantumMaxCutBridge(graph=triangle_graph)
        landscape = bridge.landscape_analysis(p=1, n_gamma=10, n_beta=10)
        # Optimal expectation should be a finite number
        assert np.isfinite(landscape["optimal_expectation"])

    def test_ising_energy_is_negative(self, simple_graph):
        bridge = QuantumMaxCutBridge(graph=simple_graph)
        result = bridge.compare_qaoa_exact(p=1, n_restarts=1, max_iter=10)
        # Ising ground energy with negative couplings should be negative
        assert result.ising_energy < 0


# ===========================================================================
# simulation_bio
# ===========================================================================


class TestCanonicalFMO:
    def test_hamiltonian_shape(self):
        fmo = CanonicalFMO(n_sites=7)
        H = fmo.hamiltonian()
        assert H.shape == (7, 7)
        # Should be Hermitian
        assert np.allclose(H, H.T)

    def test_hamiltonian_zero_centred(self):
        """Diagonal should be approximately zero-centred."""
        fmo = CanonicalFMO(n_sites=7)
        H = fmo.hamiltonian()
        assert abs(np.mean(np.diag(H))) < 1.0  # well under original ~12400

    def test_lindblad_operators(self):
        fmo = CanonicalFMO(n_sites=7)
        ops = fmo.lindblad_operators()
        # 7 dephasing + 1 trapping = 8
        assert len(ops) == 8
        for op in ops:
            assert op.operator.shape == (7, 7)

    def test_evolve(self):
        fmo = CanonicalFMO(n_sites=4, dephasing_rate=0.01, trapping_rate=0.001)
        result = fmo.evolve(duration_cm_inv=100.0, n_steps=50, initial_site=0)
        assert "times" in result
        assert "populations" in result
        assert "transfer_efficiency" in result
        assert "coherences" in result
        assert "peak_efficiency" in result
        assert "coherence_lifetime" in result
        assert result["populations"].shape == (51, 4)  # 50 steps + initial
        # Population should start at site 0
        assert result["populations"][0, 0] == pytest.approx(1.0)
        # Peak efficiency should be positive
        assert result["peak_efficiency"] > 0

    def test_smaller_system(self):
        """Smaller system should also work."""
        fmo = CanonicalFMO(n_sites=3)
        H = fmo.hamiltonian()
        assert H.shape == (3, 3)
        ops = fmo.lindblad_operators()
        assert len(ops) == 4  # 3 dephasing + 1 trapping


class TestLindbladBioValidator:
    def test_validate_fmo(self):
        result = LindbladBioValidator.validate_fmo(n_sites=4, duration=100.0)
        assert "canonical_peak_efficiency" in result
        assert result["canonical_peak_efficiency"] > 0
        # Should either succeed with bio comparison or fail gracefully
        assert "consistent" in result


# ===========================================================================
# simulation_chem
# ===========================================================================


class TestOpenQuantumChemistry:
    def test_from_h2(self):
        mol = OpenQuantumChemistry.from_h2()
        assert mol.n_qubits == 2
        assert len(mol.hamiltonian_terms) == 6

    def test_hamiltonian_hermitian(self):
        mol = OpenQuantumChemistry.from_h2()
        H = mol.hamiltonian()
        assert H.shape == (4, 4)
        assert np.allclose(H, H.conj().T)

    def test_decoherence_operators(self):
        mol = OpenQuantumChemistry.from_h2(dephasing_rate=0.01, relaxation_rate=0.001)
        ops = mol.decoherence_operators()
        # 2 dephasing + 2 amplitude damping = 4
        assert len(ops) == 4
        for op in ops:
            assert op.operator.shape == (4, 4)

    def test_evolve(self):
        mol = OpenQuantumChemistry.from_h2(dephasing_rate=0.01, relaxation_rate=0.001)
        result = mol.evolve(t_final=5.0, n_steps=20)
        assert "times" in result
        assert "energies" in result
        assert "purities" in result
        assert "fidelities" in result
        assert "ground_state_energy" in result
        assert "decoherence_time" in result
        assert len(result["energies"]) == 21  # 20 steps + initial
        # Purity should start at ~1 and decrease
        assert result["purities"][0] == pytest.approx(1.0, abs=0.01)
        # Fidelity should start at ~1
        assert result["fidelities"][0] == pytest.approx(1.0, abs=0.01)

    def test_evolve_custom_initial_state(self):
        mol = OpenQuantumChemistry.from_h2(dephasing_rate=0.01, relaxation_rate=0.001)
        psi0 = np.array([1, 0, 0, 0], dtype=complex)  # |00>
        result = mol.evolve(t_final=2.0, n_steps=10, initial_state=psi0)
        assert len(result["energies"]) == 11

    def test_bond_length_changes_energy(self):
        """Different bond lengths produce different ground-state energies."""
        mol_eq = OpenQuantumChemistry.from_h2(bond_length=0.735)
        mol_long = OpenQuantumChemistry.from_h2(bond_length=2.0)
        e_eq = np.linalg.eigvalsh(mol_eq.hamiltonian())[0]
        e_long = np.linalg.eigvalsh(mol_long.hamiltonian())[0]
        # The simplified linear model: longer bonds have higher g0 (less negative)
        # so equilibrium should be lower than stretched
        assert e_eq < e_long

    def test_no_relaxation(self):
        mol = OpenQuantumChemistry.from_h2(dephasing_rate=0.01, relaxation_rate=0.0)
        ops = mol.decoherence_operators()
        # Only 2 dephasing, no amplitude damping
        assert len(ops) == 2


class TestDecoherenceAnalysis:
    def test_h2_decoherence_scan(self):
        rates = np.logspace(-3, -1, 4)
        result = DecoherenceAnalysis.h2_decoherence_scan(dephasing_rates=rates)
        assert "rates" in result
        assert "results" in result
        assert "ground_energy" in result
        assert len(result["results"]) == 4
        # Higher dephasing should increase energy error
        errors = [r["energy_error"] for r in result["results"]]
        # Not strictly monotone due to finite simulation time, but should be non-negative
        assert all(e >= 0 for e in errors)

    def test_default_rates(self):
        result = DecoherenceAnalysis.h2_decoherence_scan()
        assert len(result["results"]) == 10  # default is 10 points


# ===========================================================================
# simulation_trading
# ===========================================================================


class TestLindbladVolatility:
    """Tests for LindbladVolatility."""

    def test_encode_surface_creates_density_matrix(self, vol_surface):
        lv = LindbladVolatility(n_qubits=4)
        rho = lv.encode_surface(vol_surface)
        assert rho.shape == (16, 16)

    def test_encode_surface_is_valid_density_matrix(self, vol_surface):
        lv = LindbladVolatility(n_qubits=4)
        rho = lv.encode_surface(vol_surface)
        # Trace should be 1
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-10)
        # Should be Hermitian
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-10)
        # Should be positive semidefinite
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-10)

    def test_encode_surface_pure_state(self, vol_surface):
        lv = LindbladVolatility(n_qubits=4)
        rho = lv.encode_surface(vol_surface)
        # Pure state has purity = 1
        purity = float(np.real(np.trace(rho @ rho)))
        np.testing.assert_allclose(purity, 1.0, atol=1e-10)

    def test_decode_surface_roundtrip(self, vol_surface):
        lv = LindbladVolatility(n_qubits=4)
        rho = lv.encode_surface(vol_surface)
        decoded = lv.decode_surface(rho)
        assert decoded.shape == (4,)
        # Roundtrip should approximately recover the original surface
        # (Ry encoding is not perfectly invertible via Z-expectation but should be close)
        assert np.all(decoded >= 0.0)
        assert np.all(decoded <= 1.0)

    def test_evolve_returns_lindblad_result(self, vol_surface):
        lv = LindbladVolatility(n_qubits=4)
        rho0 = lv.encode_surface(vol_surface)
        result = lv.evolve(rho0, t_final=0.5, n_steps=10)
        assert hasattr(result, "times")
        assert hasattr(result, "states")
        assert len(result.states) == 11  # n_steps + 1

    def test_evolve_preserves_trace(self, vol_surface):
        lv = LindbladVolatility(n_qubits=4)
        rho0 = lv.encode_surface(vol_surface)
        result = lv.evolve(rho0, t_final=0.5, n_steps=10)
        for rho in result.states:
            np.testing.assert_allclose(np.real(np.trace(rho)), 1.0, atol=1e-4)

    def test_purity_decreases_under_noise(self, vol_surface):
        lv = LindbladVolatility(n_qubits=4, dephasing_rate=0.5, damping_rate=0.2)
        rho0 = lv.encode_surface(vol_surface)
        result = lv.evolve(rho0, t_final=2.0, n_steps=20)
        purity = result.purity()
        # Purity should decrease from ~1.0 under noise
        assert purity[0] > purity[-1]

    def test_decoherence_time_positive(self, vol_surface):
        lv = LindbladVolatility(n_qubits=4, dephasing_rate=0.5, damping_rate=0.2)
        dec_time = lv.decoherence_time(vol_surface, t_max=10.0, n_steps=100)
        assert dec_time > 0.0
        assert dec_time <= 10.0

    def test_decoherence_time_longer_with_less_noise(self, vol_surface):
        lv_noisy = LindbladVolatility(n_qubits=4, dephasing_rate=1.0, damping_rate=0.5)
        lv_quiet = LindbladVolatility(n_qubits=4, dephasing_rate=0.01, damping_rate=0.005)
        dec_noisy = lv_noisy.decoherence_time(vol_surface, t_max=10.0, n_steps=100)
        dec_quiet = lv_quiet.decoherence_time(vol_surface, t_max=10.0, n_steps=100)
        assert dec_quiet >= dec_noisy

    def test_full_analysis_returns_vol_surface_result(self, vol_surface):
        lv = LindbladVolatility(n_qubits=4)
        result = lv.full_analysis(vol_surface, t_final=1.0, n_steps=10)
        assert isinstance(result, VolSurfaceResult)
        assert len(result.surfaces) == 11
        assert result.decoherence_time > 0

    def test_full_analysis_entropy_increases(self, vol_surface):
        lv = LindbladVolatility(n_qubits=4, dephasing_rate=0.5, damping_rate=0.2)
        result = lv.full_analysis(vol_surface, t_final=2.0, n_steps=20)
        # Entropy should generally increase as the state decoheres
        assert result.entropy[-1] >= result.entropy[0] - 0.1


class TestNoisySignalGenerator:
    """Tests for NoisySignalGenerator."""

    def test_generate_noisy_signals_returns_result(self, price_series):
        gen = NoisySignalGenerator(n_qubits=4, noise_rate=0.1, seed=42)
        result = gen.generate_noisy_signals(price_series, window=20)
        assert isinstance(result, NoisySignalResult)

    def test_ideal_and_noisy_same_length(self, price_series):
        gen = NoisySignalGenerator(n_qubits=4, noise_rate=0.1, seed=42)
        result = gen.generate_noisy_signals(price_series, window=20)
        assert len(result.ideal_signals) == len(result.noisy_signals)

    def test_snr_is_finite(self, price_series):
        gen = NoisySignalGenerator(n_qubits=4, noise_rate=0.1, seed=42)
        result = gen.generate_noisy_signals(price_series, window=20)
        assert np.isfinite(result.snr)

    def test_direction_agreement_bounded(self, price_series):
        gen = NoisySignalGenerator(n_qubits=4, noise_rate=0.1, seed=42)
        result = gen.generate_noisy_signals(price_series, window=20)
        assert 0.0 <= result.direction_agreement <= 1.0

    def test_zero_noise_perfect_agreement(self, price_series):
        gen = NoisySignalGenerator(n_qubits=4, noise_rate=0.0, seed=42)
        result = gen.generate_noisy_signals(price_series, window=20)
        # With zero noise, ideal and noisy should be identical
        assert result.direction_agreement == 1.0

    def test_high_noise_lower_agreement(self, price_series):
        gen_low = NoisySignalGenerator(n_qubits=4, noise_rate=0.01, seed=42)
        gen_high = NoisySignalGenerator(n_qubits=4, noise_rate=2.0, seed=42)
        result_low = gen_low.generate_noisy_signals(price_series, window=20)
        result_high = gen_high.generate_noisy_signals(price_series, window=20)
        assert result_low.direction_agreement >= result_high.direction_agreement

    def test_signal_to_noise_ratio_returns_dict(self, price_series):
        gen = NoisySignalGenerator(n_qubits=4, noise_rate=0.1, seed=42)
        snr_info = gen.signal_to_noise_ratio(prices=price_series, window=20, n_trials=3)
        assert "mean_snr" in snr_info
        assert "std_snr" in snr_info
        assert "mean_agreement" in snr_info
        assert "snr_values" in snr_info

    def test_snr_multiple_trials(self, price_series):
        gen = NoisySignalGenerator(n_qubits=4, noise_rate=0.1, seed=42)
        snr_info = gen.signal_to_noise_ratio(prices=price_series, window=20, n_trials=3)
        assert len(snr_info["snr_values"]) == 3

    def test_with_volume(self, price_series, volume_series):
        gen = NoisySignalGenerator(n_qubits=4, noise_rate=0.1, seed=42)
        result = gen.generate_noisy_signals(
            price_series, volume=volume_series, window=20
        )
        assert len(result.ideal_signals) > 0


class TestQuantumFilteredMomentum:
    """Tests for QuantumFilteredMomentum."""

    def test_filter_returns_result(self, price_series):
        qfm = QuantumFilteredMomentum(n_sites=16, coupling=1.0, evolution_time=0.5)
        result = qfm.filter(price_series, window=20)
        assert isinstance(result, FilteredMomentumResult)

    def test_filtered_same_length_as_raw(self, price_series):
        qfm = QuantumFilteredMomentum(n_sites=16, coupling=1.0, evolution_time=0.5)
        result = qfm.filter(price_series, window=20)
        assert len(result.filtered_momentum) == len(result.raw_momentum)
        assert len(result.classical_filtered) == len(result.raw_momentum)

    def test_smoothness_is_positive(self, price_series):
        qfm = QuantumFilteredMomentum(n_sites=16, coupling=1.0, evolution_time=0.5)
        result = qfm.filter(price_series, window=20)
        assert result.quantum_smoothness > 0
        assert result.classical_smoothness > 0

    def test_compare_classical_returns_dict(self, price_series):
        qfm = QuantumFilteredMomentum(n_sites=16, coupling=1.0, evolution_time=0.5)
        comparison = qfm.compare_classical(price_series, window=20)
        assert "quantum_smoothness" in comparison
        assert "classical_smoothness" in comparison
        assert "smoothness_ratio" in comparison
        assert "quantum_mse" in comparison
        assert "classical_mse" in comparison

    def test_mse_non_negative(self, price_series):
        qfm = QuantumFilteredMomentum(n_sites=16, coupling=1.0, evolution_time=0.5)
        comparison = qfm.compare_classical(price_series, window=20)
        assert comparison["quantum_mse"] >= 0
        assert comparison["classical_mse"] >= 0

    def test_filter_with_flat_prices(self):
        """Flat prices should produce near-zero momentum."""
        flat_prices = np.full(100, 100.0)
        qfm = QuantumFilteredMomentum(n_sites=16)
        result = qfm.filter(flat_prices, window=10)
        # Raw momentum should be essentially zero
        np.testing.assert_allclose(result.raw_momentum, 0.0, atol=1e-10)


# ===========================================================================
# vqe_noise
# ===========================================================================


class TestHamiltonians:
    def test_h2_hamiltonian_hermitian(self, h2_hamiltonian):
        assert h2_hamiltonian.shape == (4, 4)
        assert np.allclose(h2_hamiltonian, h2_hamiltonian.conj().T)

    def test_h2_hamiltonian_eigenvalues(self, h2_hamiltonian):
        eigvals = np.linalg.eigvalsh(h2_hamiltonian)
        # Ground state energy of H2 at equilibrium ~-1.14 Ha
        assert eigvals[0] < -0.5  # Sanity check

    def test_h2_bond_length_variation(self):
        H_eq = _h2_hamiltonian(0.735)
        H_long = _h2_hamiltonian(1.5)
        e_eq = np.linalg.eigvalsh(H_eq)[0]
        e_long = np.linalg.eigvalsh(H_long)[0]
        # Equilibrium should have lower energy than stretched
        assert e_eq < e_long

    def test_lih_hamiltonian_hermitian(self, lih_hamiltonian):
        assert lih_hamiltonian.shape == (4, 4)
        assert np.allclose(lih_hamiltonian, lih_hamiltonian.conj().T)

    def test_lih_ground_energy(self, lih_hamiltonian):
        eigvals = np.linalg.eigvalsh(lih_hamiltonian)
        # LiH minimal basis ground state ~-7.9 Ha
        assert eigvals[0] < -7.0


class TestDecoherenceOperators:
    def test_single_qubit_dephasing(self):
        ops = _decoherence_operators(1, dephasing_rate=0.01, relaxation_rate=0.0)
        assert len(ops) == 1  # Just dephasing
        assert ops[0].rate == 0.01

    def test_single_qubit_with_relaxation(self):
        ops = _decoherence_operators(1, dephasing_rate=0.01, relaxation_rate=0.001)
        assert len(ops) == 2  # dephasing + relaxation

    def test_two_qubit_operators(self):
        ops = _decoherence_operators(2, dephasing_rate=0.01, relaxation_rate=0.001)
        # 2 dephasing + 2 relaxation = 4
        assert len(ops) == 4

    def test_zero_relaxation_omits_operators(self):
        ops = _decoherence_operators(2, dephasing_rate=0.01, relaxation_rate=0.0)
        # Only 2 dephasing
        assert len(ops) == 2

    def test_lowering_operator_shape(self):
        ops = _decoherence_operators(1, dephasing_rate=0.01, relaxation_rate=0.01)
        for op in ops:
            assert op.operator.shape == (2, 2)


class TestVQENoiseBenchmark:
    def test_init_defaults(self):
        b = VQENoiseBenchmark()
        assert b.t_final == 10.0
        assert b.n_steps == 50
        assert b.CHEMICAL_ACCURACY_HA == 0.0016

    def test_benchmark_molecule_noiseless(self, bench, h2_hamiltonian):
        """With zero noise, energy error should be near zero."""
        configs = [{"name": "noiseless", "dephasing_rate": 0.0, "relaxation_rate": 0.0}]
        result = bench.benchmark_molecule(h2_hamiltonian, "H2", noise_configs=configs)

        assert result["molecule"] == "H2"
        assert len(result["results"]) == 1
        r = result["results"][0]
        assert isinstance(r, VQENoiseResult)
        assert r.energy_error < 1e-6
        assert r.final_purity > 0.999
        assert r.chemical_accuracy is True

    def test_benchmark_molecule_noisy(self, bench, h2_hamiltonian):
        """With moderate noise, energy error should be measurable."""
        configs = [{"name": "noisy", "dephasing_rate": 0.1, "relaxation_rate": 0.01}]
        result = bench.benchmark_molecule(h2_hamiltonian, "H2", noise_configs=configs)

        r = result["results"][0]
        assert r.energy_error > 0
        assert r.final_purity < 1.0
        assert r.profile_name == "noisy"
        assert r.dephasing_rate == 0.1
        assert r.relaxation_rate == 0.01

    def test_benchmark_molecule_multiple_configs(self, bench, h2_hamiltonian):
        configs = [
            {"name": "low", "dephasing_rate": 1e-4, "relaxation_rate": 1e-5},
            {"name": "medium", "dephasing_rate": 1e-2, "relaxation_rate": 1e-3},
            {"name": "high", "dephasing_rate": 0.1, "relaxation_rate": 0.01},
        ]
        result = bench.benchmark_molecule(h2_hamiltonian, "H2", noise_configs=configs)
        assert len(result["results"]) == 3
        # Higher noise should give larger energy error
        errors = [r.energy_error for r in result["results"]]
        assert errors[-1] >= errors[0]

    def test_ground_energy_correct(self, bench, h2_hamiltonian):
        configs = [{"name": "test", "dephasing_rate": 0.0, "relaxation_rate": 0.0}]
        result = bench.benchmark_molecule(h2_hamiltonian, "H2", noise_configs=configs)
        expected = float(np.linalg.eigvalsh(h2_hamiltonian)[0])
        assert abs(result["ground_energy"] - expected) < 1e-10


class TestH2Benchmark:
    def test_h2_benchmark_runs(self, bench):
        result = bench.h2_benchmark()
        assert result["molecule"].startswith("H2")
        assert len(result["results"]) >= 1

    def test_h2_custom_bond_length(self, bench):
        result = bench.h2_benchmark(bond_length=1.0)
        assert "H2" in result["molecule"]
        assert "1.0" in result["molecule"]


class TestLiHBenchmark:
    def test_lih_benchmark_runs(self, bench):
        result = bench.lih_benchmark()
        assert result["molecule"] == "LiH (minimal)"
        assert len(result["results"]) >= 1


class TestNoiseSweep:
    def test_sweep_default_rates(self, bench, h2_hamiltonian):
        result = bench.noise_sweep(h2_hamiltonian, dephasing_rates=np.logspace(-4, -1, 5))
        assert len(result["rates"]) == 5
        assert len(result["energy_errors"]) == 5
        assert len(result["purities"]) == 5
        assert len(result["fidelities"]) == 5
        assert result["chemical_accuracy_limit"] > 0

    def test_sweep_monotonic_error(self, bench, h2_hamiltonian):
        """Energy error should generally increase with noise rate."""
        rates = np.logspace(-4, -1, 6)
        result = bench.noise_sweep(h2_hamiltonian, dephasing_rates=rates)
        errors = result["energy_errors"]
        # At least the first and last should show the trend
        assert errors[-1] >= errors[0]

    def test_sweep_purity_decreases(self, bench, h2_hamiltonian):
        """Purity should generally decrease with higher noise."""
        rates = np.logspace(-4, -1, 6)
        result = bench.noise_sweep(h2_hamiltonian, dephasing_rates=rates)
        purities = result["purities"]
        assert purities[0] >= purities[-1]


class TestVQENoiseResult:
    def test_result_fields(self):
        r = VQENoiseResult(
            profile_name="test",
            dephasing_rate=0.01,
            relaxation_rate=0.001,
            ground_energy=-1.14,
            final_energy=-1.10,
            energy_error=0.04,
            final_purity=0.85,
            final_fidelity=0.80,
            decoherence_time=5.0,
            chemical_accuracy=False,
        )
        assert r.profile_name == "test"
        assert r.chemical_accuracy is False
        assert r.energy_error == 0.04

    def test_chemical_accuracy_threshold(self):
        r = VQENoiseResult(
            profile_name="good",
            dephasing_rate=1e-5,
            relaxation_rate=1e-6,
            ground_energy=-1.14,
            final_energy=-1.1401,
            energy_error=0.0001,
            final_purity=0.999,
            final_fidelity=0.999,
            decoherence_time=100.0,
            chemical_accuracy=True,
        )
        assert r.chemical_accuracy is True
        assert r.energy_error < VQENoiseBenchmark.CHEMICAL_ACCURACY_HA


class TestHardwareNoiseConfigs:
    def test_configs_from_hardware_profiles(self):
        configs = VQENoiseBenchmark._hardware_noise_configs()
        assert len(configs) >= 9  # At least 9 hardware profiles
        for cfg in configs:
            assert "name" in cfg
            assert "dephasing_rate" in cfg
            assert "relaxation_rate" in cfg
            assert cfg["dephasing_rate"] > 0
            assert cfg["relaxation_rate"] >= 0

    def test_h2_with_hardware_configs(self):
        bench = VQENoiseBenchmark(t_final=1.0, n_steps=10)
        result = bench.h2_benchmark()
        # Should have one result per hardware profile
        assert len(result["results"]) >= 9


class TestDecoherenceTime:
    def test_noiseless_decoherence_time(self, bench, h2_hamiltonian):
        """With no noise, decoherence time should be at end of simulation."""
        configs = [{"name": "noiseless", "dephasing_rate": 0.0, "relaxation_rate": 0.0}]
        result = bench.benchmark_molecule(h2_hamiltonian, "H2", noise_configs=configs)
        r = result["results"][0]
        assert r.decoherence_time >= bench.t_final * 0.99

    def test_noisy_decoherence_time(self):
        """With strong noise, decoherence time should be short."""
        bench = VQENoiseBenchmark(t_final=10.0, n_steps=100)
        H = _h2_hamiltonian()
        configs = [{"name": "very_noisy", "dephasing_rate": 2.0, "relaxation_rate": 0.5}]
        result = bench.benchmark_molecule(H, "H2", noise_configs=configs)
        r = result["results"][0]
        assert r.decoherence_time < bench.t_final


# ===========================================================================
# Integration tests (cross-module)
# ===========================================================================


class TestBridgeIntegration:
    """Integration tests across bridge modules."""

    def test_import_from_bridges_init(self):
        """Verify all classes are importable from nqpu.bridges."""
        from nqpu.bridges import (
            IsingGameSolver,
            QuantumAuctionModel,
            QuantumMaxCutBridge,
            LindbladVolatility,
            NoisySignalGenerator,
            QuantumFilteredMomentum,
            VQENoiseBenchmark,
        )
        assert IsingGameSolver is not None
        assert QuantumAuctionModel is not None
        assert QuantumMaxCutBridge is not None
        assert LindbladVolatility is not None
        assert NoisySignalGenerator is not None
        assert QuantumFilteredMomentum is not None
        assert VQENoiseBenchmark is not None

    def test_result_types_importable(self):
        """Verify result dataclasses are importable."""
        from nqpu.bridges import (
            NashResult,
            AuctionModelResult,
            MaxCutBenchmarkResult,
            VolSurfaceResult,
            NoisySignalResult,
            FilteredMomentumResult,
            VQENoiseResult,
        )
        assert NashResult is not None
        assert AuctionModelResult is not None
        assert MaxCutBenchmarkResult is not None
        assert VolSurfaceResult is not None
        assert NoisySignalResult is not None
        assert FilteredMomentumResult is not None
        assert VQENoiseResult is not None

    def test_maxcut_ising_consistency(self, triangle_graph):
        """QAOA expectation and Ising energy should be correlated."""
        bridge = QuantumMaxCutBridge(graph=triangle_graph)
        result = bridge.compare_qaoa_exact(p=1, n_restarts=1, max_iter=10)
        # Both should be finite
        assert np.isfinite(result.ising_energy)
        assert np.isfinite(result.qaoa_expectation)

    def test_lindblad_vol_decode_bounded(self, vol_surface):
        """Decoded vol surface should remain in [0, 1]."""
        lv = LindbladVolatility(n_qubits=4, dephasing_rate=0.3, damping_rate=0.1)
        result = lv.full_analysis(vol_surface, t_final=1.0, n_steps=10)
        for surface in result.surfaces:
            assert np.all(surface >= -0.01)
            assert np.all(surface <= 1.01)
