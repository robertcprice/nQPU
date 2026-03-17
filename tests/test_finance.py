"""Comprehensive tests for the nqpu.finance package.

Tests cover: amplitude estimation (Canonical, Iterative, MaxLikelihood),
option pricing (European call/put, Black-Scholes), portfolio optimization
(QUBO encoding, Ising conversion, classical brute-force, efficient frontier),
and risk analysis (VaR, CVaR, drawdown, Sharpe/Sortino, scenario generation).
"""

from __future__ import annotations

import numpy as np
import pytest

from nqpu.finance import (
    AEResult,
    CanonicalQAE,
    IterativeQAE,
    MaxLikelihoodQAE,
    bernoulli_oracle,
    build_grover_operator,
    apply_grover_power,
    OptionType,
    QAEMethod,
    OptionPricingResult,
    QuantumOptionPricer,
    black_scholes_call,
    black_scholes_put,
    black_scholes_delta,
    PortfolioResult,
    PortfolioOptimizer,
    EfficientFrontierPoint,
    QuboMatrix,
    IsingHamiltonian,
    portfolio_to_qubo,
    qubo_to_ising,
    classical_portfolio_optimize,
    compute_efficient_frontier,
    DistributionModel,
    RiskConfig,
    RiskMetrics,
    RiskAnalyzer,
    compute_var,
    compute_cvar,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    generate_scenarios,
)


# ---- Fixtures ----


@pytest.fixture
def simple_oracle():
    """2x2 Bernoulli oracle with amplitude 0.3."""
    return bernoulli_oracle(0.3), [1]


@pytest.fixture
def portfolio_data():
    """Small 3-asset portfolio problem."""
    returns = np.array([0.10, 0.12, 0.08])
    cov = np.array([
        [0.04, 0.006, 0.002],
        [0.006, 0.09, 0.009],
        [0.002, 0.009, 0.01],
    ])
    return returns, cov


@pytest.fixture
def risk_scenarios():
    """Pre-generated return scenarios for risk tests."""
    rng = np.random.RandomState(42)
    return rng.randn(5000) * 0.02 + 0.0004


# ---- Amplitude Estimation Tests ----


class TestBernoulliOracle:
    """Tests for the Bernoulli oracle constructor."""

    def test_oracle_is_unitary(self):
        oracle = bernoulli_oracle(0.5)
        identity = oracle @ oracle.conj().T
        assert np.allclose(identity, np.eye(2), atol=1e-12)

    def test_oracle_produces_correct_amplitude(self):
        amp = 0.3
        oracle = bernoulli_oracle(amp)
        state = oracle @ np.array([1.0, 0.0], dtype=complex)
        measured_amp = abs(state[1]) ** 2
        assert abs(measured_amp - amp) < 1e-12

    @pytest.mark.parametrize("amp", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_oracle_various_amplitudes(self, amp):
        oracle = bernoulli_oracle(amp)
        state = oracle @ np.array([1.0, 0.0], dtype=complex)
        assert abs(abs(state[1]) ** 2 - amp) < 1e-12

    def test_oracle_invalid_amplitude(self):
        with pytest.raises(ValueError, match="amplitude must be in"):
            bernoulli_oracle(-0.1)
        with pytest.raises(ValueError, match="amplitude must be in"):
            bernoulli_oracle(1.1)


class TestGroverOperator:
    """Tests for Grover operator construction and application."""

    def test_grover_operator_is_unitary(self, simple_oracle):
        oracle, good = simple_oracle
        Q = build_grover_operator(oracle, good)
        identity = Q @ Q.conj().T
        assert np.allclose(identity, np.eye(2), atol=1e-12)

    def test_apply_grover_power_zero(self, simple_oracle):
        oracle, good = simple_oracle
        Q = build_grover_operator(oracle, good)
        state = oracle @ np.array([1.0, 0.0], dtype=complex)
        result = apply_grover_power(state, Q, 0)
        assert np.allclose(result, state, atol=1e-12)


class TestCanonicalQAE:
    """Tests for canonical (QPE-based) amplitude estimation."""

    def test_estimate_near_target(self, simple_oracle):
        oracle, good = simple_oracle
        qae = CanonicalQAE(num_eval_qubits=6)
        result = qae.estimate(oracle, good)
        assert abs(result.estimation - 0.3) < 0.05

    def test_result_has_confidence_interval(self, simple_oracle):
        oracle, good = simple_oracle
        result = CanonicalQAE(num_eval_qubits=6).estimate(oracle, good)
        ci_low, ci_high = result.confidence_interval
        assert ci_low <= ci_high
        assert ci_low >= 0.0
        assert ci_high <= 1.0

    def test_oracle_calls_positive(self, simple_oracle):
        oracle, good = simple_oracle
        result = CanonicalQAE(num_eval_qubits=4).estimate(oracle, good)
        assert result.num_oracle_calls > 0

    def test_invalid_eval_qubits(self):
        with pytest.raises(ValueError, match="num_eval_qubits must be"):
            CanonicalQAE(num_eval_qubits=0)


class TestIterativeQAE:
    """Tests for iterative (QPE-free) amplitude estimation."""

    def test_estimate_near_target(self, simple_oracle):
        oracle, good = simple_oracle
        iqae = IterativeQAE(epsilon=0.05, alpha=0.05)
        result = iqae.estimate(oracle, good)
        assert abs(result.estimation - 0.3) < 0.1

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            IterativeQAE(epsilon=0.0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            IterativeQAE(alpha=1.0)


class TestMaxLikelihoodQAE:
    """Tests for maximum-likelihood amplitude estimation."""

    def test_estimate_near_target(self, simple_oracle):
        oracle, good = simple_oracle
        mlae = MaxLikelihoodQAE(evaluation_schedule=[0, 1, 2, 4, 8], num_shots=200)
        result = mlae.estimate(oracle, good)
        assert abs(result.estimation - 0.3) < 0.05

    def test_exponential_schedule_factory(self):
        mlae = MaxLikelihoodQAE.with_exponential_schedule(max_power_exponent=3)
        assert 0 in mlae.evaluation_schedule
        assert 8 in mlae.evaluation_schedule


# ---- Option Pricing Tests ----


class TestBlackScholes:
    """Tests for Black-Scholes analytical formulas."""

    def test_call_put_parity(self):
        s, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
        call = black_scholes_call(s, k, r, sigma, t)
        put = black_scholes_put(s, k, r, sigma, t)
        parity = call - put - (s - k * np.exp(-r * t))
        assert abs(parity) < 1e-6

    def test_call_at_expiry(self):
        assert black_scholes_call(110.0, 100.0, 0.05, 0.2, 0.0) == pytest.approx(10.0)
        assert black_scholes_call(90.0, 100.0, 0.05, 0.2, 0.0) == pytest.approx(0.0)

    def test_put_at_expiry(self):
        assert black_scholes_put(90.0, 100.0, 0.05, 0.2, 0.0) == pytest.approx(10.0)
        assert black_scholes_put(110.0, 100.0, 0.05, 0.2, 0.0) == pytest.approx(0.0)

    def test_delta_call_bounds(self):
        delta = black_scholes_delta(100.0, 100.0, 0.05, 0.2, 1.0, is_call=True)
        assert 0.0 <= delta <= 1.0

    def test_delta_put_bounds(self):
        delta = black_scholes_delta(100.0, 100.0, 0.05, 0.2, 1.0, is_call=False)
        assert -1.0 <= delta <= 0.0


class TestQuantumOptionPricer:
    """Tests for the main quantum option pricer."""

    def test_european_call_positive_price(self):
        pricer = QuantumOptionPricer(
            spot=100.0, strike=100.0, rate=0.05, volatility=0.2, maturity=1.0,
            option_type=OptionType.EUROPEAN_CALL, num_price_qubits=3,
        )
        result = pricer.price()
        assert result.price > 0
        assert result.analytical_price is not None
        assert result.analytical_price > 0

    def test_european_put_positive_price(self):
        pricer = QuantumOptionPricer(
            spot=100.0, strike=100.0, rate=0.05, volatility=0.2, maturity=1.0,
            option_type=OptionType.EUROPEAN_PUT, num_price_qubits=3,
        )
        result = pricer.price()
        assert result.price >= 0

    def test_invalid_spot(self):
        with pytest.raises(ValueError, match="spot must be positive"):
            QuantumOptionPricer(spot=-1.0)

    def test_invalid_volatility(self):
        with pytest.raises(ValueError, match="volatility must be positive"):
            QuantumOptionPricer(volatility=0.0)


# ---- Portfolio Optimization Tests ----


class TestQuboEncoding:
    """Tests for portfolio-to-QUBO and QUBO-to-Ising encoding."""

    def test_qubo_matrix_shape(self, portfolio_data):
        returns, cov = portfolio_data
        qubo = portfolio_to_qubo(returns, cov)
        n = len(returns)
        assert qubo.matrix.shape == (n, n)
        assert qubo.num_variables == n

    def test_ising_energy_bitstring(self, portfolio_data):
        returns, cov = portfolio_data
        qubo = portfolio_to_qubo(returns, cov)
        ising = qubo_to_ising(qubo)
        bits = np.array([1, 0, 1], dtype=np.int64)
        energy = ising.energy_bitstring(bits)
        assert np.isfinite(energy)

    def test_qubo_to_ising_num_qubits(self, portfolio_data):
        returns, cov = portfolio_data
        qubo = portfolio_to_qubo(returns, cov)
        ising = qubo_to_ising(qubo)
        assert ising.num_qubits == len(returns)


class TestPortfolioOptimizer:
    """Tests for QAOA-based portfolio optimization."""

    def test_classical_brute_force(self, portfolio_data):
        returns, cov = portfolio_data
        result = classical_portfolio_optimize(returns, cov, risk_aversion=0.5, budget=2)
        assert isinstance(result, PortfolioResult)
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-6)
        assert result.variance >= 0

    def test_efficient_frontier(self, portfolio_data):
        returns, cov = portfolio_data
        frontier = compute_efficient_frontier(returns, cov, num_points=5)
        assert len(frontier) == 5
        for pt in frontier:
            assert isinstance(pt, EfficientFrontierPoint)
            assert pt.variance >= 0

    def test_efficient_frontier_variance_ordering(self, portfolio_data):
        returns, cov = portfolio_data
        frontier = compute_efficient_frontier(returns, cov, num_points=10)
        returns_list = [pt.target_return for pt in frontier]
        assert returns_list == sorted(returns_list)


# ---- Risk Analysis Tests ----


class TestRiskMetrics:
    """Tests for classical risk metric computation."""

    def test_var_positive(self, risk_scenarios):
        var = compute_var(risk_scenarios, confidence=0.95)
        assert var > 0

    def test_cvar_greater_than_var(self, risk_scenarios):
        var = compute_var(risk_scenarios, confidence=0.95)
        cvar = compute_cvar(risk_scenarios, var)
        assert cvar >= var

    def test_max_drawdown_bounded(self, risk_scenarios):
        dd = compute_max_drawdown(risk_scenarios)
        assert 0.0 <= dd <= 1.0

    def test_sharpe_ratio_finite(self, risk_scenarios):
        sharpe = compute_sharpe_ratio(risk_scenarios)
        assert np.isfinite(sharpe)

    def test_sortino_ratio_finite(self, risk_scenarios):
        sortino = compute_sortino_ratio(risk_scenarios)
        assert np.isfinite(sortino)

    def test_generate_scenarios_shape(self):
        config = RiskConfig(num_scenarios=1000, seed=42)
        scenarios = generate_scenarios(0.10, 0.20, config)
        assert scenarios.shape == (1000,)

    @pytest.mark.parametrize("dist", [DistributionModel.NORMAL, DistributionModel.STUDENT_T])
    def test_scenario_distribution_models(self, dist):
        config = RiskConfig(num_scenarios=500, distribution=dist, seed=42)
        scenarios = generate_scenarios(0.10, 0.20, config)
        assert len(scenarios) == 500
        assert np.all(np.isfinite(scenarios))


class TestRiskAnalyzer:
    """Tests for the full risk analyzer."""

    def test_classical_analyze(self, portfolio_data):
        returns, cov = portfolio_data
        weights = np.array([0.4, 0.3, 0.3])
        config = RiskConfig(num_scenarios=2000, seed=42)
        analyzer = RiskAnalyzer(config)
        metrics = analyzer.classical_analyze(returns, cov, weights)
        assert isinstance(metrics, RiskMetrics)
        assert metrics.var > 0
        assert metrics.method == "classical"

    def test_analyze_scenarios(self, risk_scenarios):
        config = RiskConfig(confidence_level=0.95, num_qubits=3, seed=42)
        analyzer = RiskAnalyzer(config)
        metrics = analyzer.analyze_scenarios(risk_scenarios)
        assert metrics.var > 0
        assert metrics.method == "QAE-enhanced"
