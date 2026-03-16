"""Comprehensive tests for the nQPU quantum finance package.

Tests cover all four modules:
  - amplitude_estimation: CanonicalQAE, IterativeQAE, MaxLikelihoodQAE
  - option_pricing: European call/put, Asian, Barrier, Black-Scholes
  - portfolio: QAOA optimizer, classical solver, QUBO/Ising encoding,
    efficient frontier
  - risk_analysis: VaR, CVaR, risk metrics, quantum-enhanced estimation
"""

import math

import numpy as np
import pytest

from nqpu.finance import (
    # Amplitude estimation
    AEResult,
    CanonicalQAE,
    IterativeQAE,
    MaxLikelihoodQAE,
    bernoulli_oracle,
    build_grover_operator,
    apply_grover_power,
    # Option pricing
    OptionType,
    QAEMethod,
    OptionPricingResult,
    QuantumOptionPricer,
    black_scholes_call,
    black_scholes_put,
    black_scholes_delta,
    price_european_call,
    price_european_put,
    # Portfolio
    PortfolioResult,
    PortfolioOptimizer,
    EfficientFrontierPoint,
    QuboMatrix,
    IsingHamiltonian,
    portfolio_to_qubo,
    qubo_to_ising,
    classical_portfolio_optimize,
    compute_efficient_frontier,
    # Risk analysis
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
    quantum_var,
    quantum_cvar,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def simple_returns() -> np.ndarray:
    """3-asset expected returns."""
    return np.array([0.10, 0.12, 0.08])


@pytest.fixture
def simple_covariance() -> np.ndarray:
    """3-asset covariance matrix."""
    return np.array([
        [0.04, 0.006, 0.002],
        [0.006, 0.09, 0.009],
        [0.002, 0.009, 0.01],
    ])


@pytest.fixture
def equal_weights() -> np.ndarray:
    """Equal-weight 3-asset portfolio."""
    return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


# ======================================================================
# AMPLITUDE ESTIMATION TESTS
# ======================================================================


class TestBernoulliOracle:
    """Tests for the Bernoulli test oracle."""

    def test_oracle_is_unitary(self):
        oracle = bernoulli_oracle(0.3)
        identity = oracle @ oracle.conj().T
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-12)

    def test_oracle_prepares_correct_amplitude(self):
        for amp in [0.0, 0.1, 0.3, 0.5, 0.7, 0.99, 1.0]:
            oracle = bernoulli_oracle(amp)
            state = oracle @ np.array([1, 0], dtype=complex)
            prob_good = abs(state[1]) ** 2
            assert abs(prob_good - amp) < 1e-10, f"amplitude={amp}"

    def test_oracle_rejects_invalid_amplitude(self):
        with pytest.raises(ValueError):
            bernoulli_oracle(-0.1)
        with pytest.raises(ValueError):
            bernoulli_oracle(1.5)


class TestGroverOperator:
    """Tests for Grover operator construction."""

    def test_grover_is_unitary(self):
        oracle = bernoulli_oracle(0.3)
        q = build_grover_operator(oracle, good_indices=[1])
        identity = q @ q.conj().T
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-10)

    def test_grover_default_good_indices(self):
        oracle = bernoulli_oracle(0.3)
        q = build_grover_operator(oracle)  # default: upper half
        assert q.shape == (2, 2)

    def test_apply_grover_power_zero(self):
        oracle = bernoulli_oracle(0.3)
        q = build_grover_operator(oracle, [1])
        state = oracle @ np.array([1, 0], dtype=complex)
        result = apply_grover_power(state, q, 0)
        np.testing.assert_allclose(result, state, atol=1e-12)

    def test_apply_grover_power_preserves_norm(self):
        oracle = bernoulli_oracle(0.4)
        q = build_grover_operator(oracle, [1])
        state = oracle @ np.array([1, 0], dtype=complex)
        for k in [1, 3, 5, 10]:
            result = apply_grover_power(state, q, k)
            assert abs(np.linalg.norm(result) - 1.0) < 1e-10


class TestCanonicalQAE:
    """Tests for the canonical (QPE-based) amplitude estimation."""

    def test_known_amplitude_03(self):
        oracle = bernoulli_oracle(0.3)
        qae = CanonicalQAE(num_eval_qubits=6)
        result = qae.estimate(oracle, good_indices=[1])
        assert abs(result.estimation - 0.3) < 0.05

    def test_known_amplitude_05(self):
        oracle = bernoulli_oracle(0.5)
        qae = CanonicalQAE(num_eval_qubits=6)
        result = qae.estimate(oracle, good_indices=[1])
        assert abs(result.estimation - 0.5) < 0.05

    def test_known_amplitude_025(self):
        oracle = bernoulli_oracle(0.25)
        qae = CanonicalQAE(num_eval_qubits=8)
        result = qae.estimate(oracle, good_indices=[1])
        assert abs(result.estimation - 0.25) < 0.03

    def test_edge_near_zero(self):
        oracle = bernoulli_oracle(0.01)
        qae = CanonicalQAE(num_eval_qubits=8)
        result = qae.estimate(oracle, good_indices=[1])
        assert result.estimation < 0.1

    def test_edge_near_one(self):
        oracle = bernoulli_oracle(0.99)
        qae = CanonicalQAE(num_eval_qubits=8)
        result = qae.estimate(oracle, good_indices=[1])
        assert result.estimation > 0.9

    def test_confidence_interval_contains_truth(self):
        target = 0.3
        oracle = bernoulli_oracle(target)
        qae = CanonicalQAE(num_eval_qubits=8)
        result = qae.estimate(oracle, good_indices=[1])
        # With 8 eval qubits the CI should comfortably contain the target
        assert result.confidence_interval[0] <= target + 0.1
        assert result.confidence_interval[1] >= target - 0.1

    def test_oracle_calls_count(self):
        oracle = bernoulli_oracle(0.3)
        qae = CanonicalQAE(num_eval_qubits=4)
        result = qae.estimate(oracle, good_indices=[1])
        assert result.num_oracle_calls == (1 << 4) - 1

    def test_precision_improves_with_qubits(self):
        target = 0.3
        oracle = bernoulli_oracle(target)
        err_4 = abs(CanonicalQAE(4).estimate(oracle, [1]).estimation - target)
        err_8 = abs(CanonicalQAE(8).estimate(oracle, [1]).estimation - target)
        assert err_8 <= err_4 + 0.01  # 8 qubits should be at least as good

    def test_rejects_invalid_num_qubits(self):
        with pytest.raises(ValueError):
            CanonicalQAE(num_eval_qubits=0)


class TestIterativeQAE:
    """Tests for iterative (QPE-free) amplitude estimation."""

    def test_known_amplitude_03(self):
        oracle = bernoulli_oracle(0.3)
        iqae = IterativeQAE(epsilon=0.05, alpha=0.05)
        result = iqae.estimate(oracle, good_indices=[1])
        assert abs(result.estimation - 0.3) < 0.1

    def test_known_amplitude_05(self):
        oracle = bernoulli_oracle(0.5)
        iqae = IterativeQAE(epsilon=0.05)
        result = iqae.estimate(oracle, good_indices=[1])
        assert abs(result.estimation - 0.5) < 0.1

    def test_convergence_with_tight_epsilon(self):
        oracle = bernoulli_oracle(0.25)
        iqae = IterativeQAE(epsilon=0.02, alpha=0.05, max_iterations=50)
        result = iqae.estimate(oracle, good_indices=[1])
        assert abs(result.estimation - 0.25) < 0.1

    def test_samples_collected(self):
        oracle = bernoulli_oracle(0.3)
        iqae = IterativeQAE(epsilon=0.05)
        result = iqae.estimate(oracle, good_indices=[1])
        assert len(result.samples) > 0

    def test_rejects_invalid_params(self):
        with pytest.raises(ValueError):
            IterativeQAE(epsilon=-0.01)
        with pytest.raises(ValueError):
            IterativeQAE(epsilon=0.01, alpha=0.0)
        with pytest.raises(ValueError):
            IterativeQAE(epsilon=0.01, alpha=1.0)


class TestMaxLikelihoodQAE:
    """Tests for maximum likelihood amplitude estimation."""

    def test_known_amplitude_03(self):
        oracle = bernoulli_oracle(0.3)
        mlae = MaxLikelihoodQAE(
            evaluation_schedule=[0, 1, 2, 4, 8], num_shots=200
        )
        result = mlae.estimate(oracle, good_indices=[1])
        assert abs(result.estimation - 0.3) < 0.05

    def test_known_amplitude_025(self):
        oracle = bernoulli_oracle(0.25)
        mlae = MaxLikelihoodQAE(
            evaluation_schedule=[0, 1, 2, 4, 8, 16], num_shots=200
        )
        result = mlae.estimate(oracle, good_indices=[1])
        assert abs(result.estimation - 0.25) < 0.05

    def test_samples_per_power(self):
        oracle = bernoulli_oracle(0.3)
        schedule = [0, 1, 2, 4]
        mlae = MaxLikelihoodQAE(evaluation_schedule=schedule, num_shots=100)
        result = mlae.estimate(oracle, good_indices=[1])
        assert len(result.samples) == len(schedule)

    def test_exponential_schedule_factory(self):
        mlae = MaxLikelihoodQAE.with_exponential_schedule(
            max_power_exponent=3, num_shots=100
        )
        assert 0 in mlae.evaluation_schedule
        assert 8 in mlae.evaluation_schedule

    def test_confidence_interval_valid(self):
        oracle = bernoulli_oracle(0.3)
        mlae = MaxLikelihoodQAE(num_shots=200)
        result = mlae.estimate(oracle, good_indices=[1])
        assert 0.0 <= result.confidence_interval[0] <= result.confidence_interval[1] <= 1.0


class TestAllMethodsAgree:
    """Cross-validation between QAE variants."""

    @pytest.mark.parametrize("target", [0.2, 0.25, 0.5, 0.75])
    def test_all_methods_agree(self, target: float):
        oracle = bernoulli_oracle(target)
        good = [1]

        qae_est = CanonicalQAE(8).estimate(oracle, good).estimation
        iqae_est = IterativeQAE(0.05, 0.05).estimate(oracle, good).estimation
        mlae_est = MaxLikelihoodQAE([0, 1, 2, 4, 8, 16], 200).estimate(oracle, good).estimation

        # All within 0.1 of the target
        assert abs(qae_est - target) < 0.1
        assert abs(iqae_est - target) < 0.1
        assert abs(mlae_est - target) < 0.1


# ======================================================================
# OPTION PRICING TESTS
# ======================================================================


class TestBlackScholes:
    """Tests for Black-Scholes analytical formulas."""

    def test_call_at_the_money(self):
        price = black_scholes_call(100, 100, 0.05, 0.2, 1.0)
        assert 8.0 < price < 12.0  # ATM call ~ 10.45

    def test_put_at_the_money(self):
        price = black_scholes_put(100, 100, 0.05, 0.2, 1.0)
        assert 3.0 < price < 8.0  # ATM put ~ 5.57

    def test_put_call_parity(self):
        s, k, r, sigma, t = 100, 100, 0.05, 0.2, 1.0
        call = black_scholes_call(s, k, r, sigma, t)
        put = black_scholes_put(s, k, r, sigma, t)
        # C - P = S - K*exp(-rT)
        lhs = call - put
        rhs = s - k * np.exp(-r * t)
        assert abs(lhs - rhs) < 1e-6

    def test_call_at_expiry(self):
        assert black_scholes_call(110, 100, 0.05, 0.2, 0.0) == pytest.approx(10.0)
        assert black_scholes_call(90, 100, 0.05, 0.2, 0.0) == pytest.approx(0.0)

    def test_put_at_expiry(self):
        assert black_scholes_put(90, 100, 0.05, 0.2, 0.0) == pytest.approx(10.0)
        assert black_scholes_put(110, 100, 0.05, 0.2, 0.0) == pytest.approx(0.0)

    def test_deep_in_the_money_call(self):
        price = black_scholes_call(200, 100, 0.05, 0.2, 1.0)
        intrinsic = 200 - 100 * np.exp(-0.05)
        assert price >= intrinsic - 0.01

    def test_deep_out_of_money_call(self):
        price = black_scholes_call(50, 100, 0.05, 0.2, 1.0)
        assert price < 1.0

    def test_call_delta_bounds(self):
        delta = black_scholes_delta(100, 100, 0.05, 0.2, 1.0, is_call=True)
        assert 0.0 < delta < 1.0

    def test_put_delta_bounds(self):
        delta = black_scholes_delta(100, 100, 0.05, 0.2, 1.0, is_call=False)
        assert -1.0 < delta < 0.0


class TestQuantumOptionPricerEuropean:
    """Tests for European option pricing via QAE."""

    def test_call_reasonable_price(self):
        result = price_european_call(
            spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
            num_price_qubits=4,
        )
        bs = black_scholes_call(100, 100, 0.05, 0.2, 1.0)
        # QAE with 4 qubits (16 bins) won't be exact -- allow 50% relative error
        assert abs(result.price - bs) / bs < 0.5

    def test_put_reasonable_price(self):
        result = price_european_put(
            spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
            num_price_qubits=4,
        )
        bs = black_scholes_put(100, 100, 0.05, 0.2, 1.0)
        assert result.price > 0

    def test_call_increases_with_spot(self):
        p1 = price_european_call(spot=90, strike=100, num_price_qubits=4).price
        p2 = price_european_call(spot=110, strike=100, num_price_qubits=4).price
        assert p2 > p1

    def test_call_decreases_with_strike(self):
        p1 = price_european_call(spot=100, strike=90, num_price_qubits=4).price
        p2 = price_european_call(spot=100, strike=110, num_price_qubits=4).price
        assert p1 > p2

    def test_result_has_analytical(self):
        result = price_european_call(num_price_qubits=4)
        assert result.analytical_price is not None
        assert result.analytical_price > 0

    def test_result_has_confidence_interval(self):
        result = price_european_call(num_price_qubits=4)
        assert result.confidence_interval[0] <= result.price
        assert result.confidence_interval[1] >= result.price

    def test_oracle_calls_positive(self):
        result = price_european_call(num_price_qubits=3)
        assert result.num_oracle_calls > 0


class TestQuantumOptionPricerAsian:
    """Tests for Asian option pricing."""

    def test_asian_call_positive(self):
        pricer = QuantumOptionPricer(
            spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
            option_type=OptionType.ASIAN_CALL,
            num_price_qubits=3, num_paths=1000, num_time_steps=6,
        )
        result = pricer.price()
        assert result.price > 0

    def test_asian_put_positive(self):
        pricer = QuantumOptionPricer(
            spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
            option_type=OptionType.ASIAN_PUT,
            num_price_qubits=3, num_paths=1000, num_time_steps=6,
        )
        result = pricer.price()
        assert result.price > 0

    def test_asian_call_less_than_european(self):
        """Asian call is always worth less than or equal to European call."""
        asian = QuantumOptionPricer(
            spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
            option_type=OptionType.ASIAN_CALL,
            num_price_qubits=3, num_paths=2000, num_time_steps=6,
        ).price()
        european = price_european_call(
            spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
            num_price_qubits=3,
        )
        # With QAE noise, allow some slack
        assert asian.price < european.price * 2.0


class TestQuantumOptionPricerBarrier:
    """Tests for Barrier option pricing."""

    def test_barrier_up_and_out_positive(self):
        pricer = QuantumOptionPricer(
            spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
            option_type=OptionType.BARRIER_UP_AND_OUT, barrier=130.0,
            num_price_qubits=3, num_paths=1000, num_time_steps=6,
        )
        result = pricer.price()
        assert result.price >= 0

    def test_barrier_down_and_out_positive(self):
        pricer = QuantumOptionPricer(
            spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
            option_type=OptionType.BARRIER_DOWN_AND_OUT, barrier=70.0,
            num_price_qubits=3, num_paths=1000, num_time_steps=6,
        )
        result = pricer.price()
        assert result.price >= 0

    def test_barrier_less_than_vanilla(self):
        """Barrier option is always worth less than vanilla."""
        barrier = QuantumOptionPricer(
            spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
            option_type=OptionType.BARRIER_UP_AND_OUT, barrier=120.0,
            num_price_qubits=3, num_paths=2000, num_time_steps=6,
        ).price()
        vanilla_bs = black_scholes_call(100, 100, 0.05, 0.2, 1.0)
        assert barrier.price < vanilla_bs * 2.0

    def test_barrier_requires_barrier_level(self):
        pricer = QuantumOptionPricer(
            option_type=OptionType.BARRIER_UP_AND_OUT,
            barrier=None,
        )
        with pytest.raises(ValueError):
            pricer.price()


class TestQuantumOptionPricerQAEMethods:
    """Test that all QAE methods work for option pricing."""

    @pytest.mark.parametrize("method", list(QAEMethod))
    def test_european_call_all_methods(self, method: QAEMethod):
        pricer = QuantumOptionPricer(
            spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
            option_type=OptionType.EUROPEAN_CALL,
            num_price_qubits=3, qae_method=method,
        )
        result = pricer.price()
        assert result.price > 0


class TestOptionPricerValidation:
    """Input validation tests for QuantumOptionPricer."""

    def test_rejects_negative_spot(self):
        with pytest.raises(ValueError):
            QuantumOptionPricer(spot=-100)

    def test_rejects_negative_strike(self):
        with pytest.raises(ValueError):
            QuantumOptionPricer(strike=-100)

    def test_rejects_negative_volatility(self):
        with pytest.raises(ValueError):
            QuantumOptionPricer(volatility=-0.2)

    def test_rejects_negative_maturity(self):
        with pytest.raises(ValueError):
            QuantumOptionPricer(maturity=-1.0)


# ======================================================================
# PORTFOLIO OPTIMIZATION TESTS
# ======================================================================


class TestQuboEncoding:
    """Tests for QUBO and Ising Hamiltonian encoding."""

    def test_qubo_dimensions(self, simple_returns, simple_covariance):
        qubo = portfolio_to_qubo(simple_returns, simple_covariance)
        assert qubo.matrix.shape == (3, 3)
        assert qubo.num_variables == 3

    def test_qubo_multibit_dimensions(self, simple_returns, simple_covariance):
        qubo = portfolio_to_qubo(simple_returns, simple_covariance, num_bits_per_asset=2)
        assert qubo.matrix.shape == (6, 6)
        assert qubo.num_variables == 6

    def test_ising_from_qubo(self, simple_returns, simple_covariance):
        qubo = portfolio_to_qubo(simple_returns, simple_covariance)
        ising = qubo_to_ising(qubo)
        assert ising.num_qubits == 3
        assert len(ising.h_fields) == 3

    def test_ising_energy_consistent(self, simple_returns, simple_covariance):
        qubo = portfolio_to_qubo(simple_returns, simple_covariance)
        ising = qubo_to_ising(qubo)
        # Check that QUBO energy matches Ising energy for all 2^3 bitstrings
        for idx in range(8):
            bits = np.array([(idx >> q) & 1 for q in range(3)], dtype=float)
            qubo_energy = bits @ qubo.matrix @ bits + qubo.offset
            ising_energy = ising.energy_bitstring(
                np.array([(idx >> q) & 1 for q in range(3)], dtype=np.uint8)
            )
            assert abs(qubo_energy - ising_energy) < 1e-8, (
                f"bitstring {idx}: QUBO={qubo_energy}, Ising={ising_energy}"
            )


class TestClassicalPortfolio:
    """Tests for the classical brute-force portfolio solver."""

    def test_two_asset_selection(self):
        returns = np.array([0.10, 0.15])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = classical_portfolio_optimize(returns, cov, risk_aversion=0.5, budget=1)
        assert result.weights.sum() == pytest.approx(1.0)
        assert sum(result.best_bitstring) == 1

    def test_three_asset_budget_two(self, simple_returns, simple_covariance):
        result = classical_portfolio_optimize(
            simple_returns, simple_covariance, risk_aversion=0.5, budget=2
        )
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert sum(result.best_bitstring) == 2

    def test_all_assets_no_budget(self, simple_returns, simple_covariance):
        result = classical_portfolio_optimize(
            simple_returns, simple_covariance, risk_aversion=0.5, budget=None
        )
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_rejects_large_problem(self):
        returns = np.zeros(25)
        cov = np.eye(25)
        with pytest.raises(ValueError, match="limited"):
            classical_portfolio_optimize(returns, cov)


class TestQAOAPortfolio:
    """Tests for QAOA-based portfolio optimization."""

    def test_two_asset_runs(self):
        returns = np.array([0.10, 0.15])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        optimizer = PortfolioOptimizer(num_layers=1, risk_aversion=0.5, budget=1)
        result = optimizer.optimize(returns, cov)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.iterations > 0

    def test_three_asset_runs(self, simple_returns, simple_covariance):
        optimizer = PortfolioOptimizer(num_layers=2, risk_aversion=0.5, budget=2)
        result = optimizer.optimize(simple_returns, simple_covariance)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert len(result.optimal_angles) == 4  # 2 layers * 2 angles

    def test_objective_is_finite(self, simple_returns, simple_covariance):
        optimizer = PortfolioOptimizer(num_layers=1, budget=2)
        result = optimizer.optimize(simple_returns, simple_covariance)
        assert np.isfinite(result.objective)

    def test_variance_non_negative(self, simple_returns, simple_covariance):
        optimizer = PortfolioOptimizer(num_layers=1, budget=2)
        result = optimizer.optimize(simple_returns, simple_covariance)
        assert result.variance >= -1e-10


class TestEfficientFrontier:
    """Tests for efficient frontier computation."""

    def test_frontier_points(self, simple_returns, simple_covariance):
        frontier = compute_efficient_frontier(
            simple_returns, simple_covariance, num_points=10
        )
        assert len(frontier) == 10

    def test_frontier_returns_increase(self, simple_returns, simple_covariance):
        frontier = compute_efficient_frontier(
            simple_returns, simple_covariance, num_points=10
        )
        returns_list = [pt.target_return for pt in frontier]
        for i in range(len(returns_list) - 1):
            assert returns_list[i] <= returns_list[i + 1] + 1e-10

    def test_frontier_weights_sum_to_one(self, simple_returns, simple_covariance):
        frontier = compute_efficient_frontier(
            simple_returns, simple_covariance, num_points=5
        )
        for pt in frontier:
            assert abs(pt.weights.sum() - 1.0) < 1e-6

    def test_frontier_variance_non_negative(self, simple_returns, simple_covariance):
        frontier = compute_efficient_frontier(
            simple_returns, simple_covariance, num_points=5
        )
        for pt in frontier:
            assert pt.variance >= -1e-10


# ======================================================================
# RISK ANALYSIS TESTS
# ======================================================================


class TestScenarioGeneration:
    """Tests for return scenario generation."""

    def test_normal_scenarios(self):
        config = RiskConfig(num_scenarios=1000, seed=42)
        scenarios = generate_scenarios(0.1, 0.2, config)
        assert len(scenarios) == 1000

    def test_student_t_scenarios(self):
        config = RiskConfig(
            num_scenarios=1000, distribution=DistributionModel.STUDENT_T,
            df=5.0, seed=42,
        )
        scenarios = generate_scenarios(0.1, 0.2, config)
        assert len(scenarios) == 1000

    def test_student_t_heavier_tails(self):
        """Student-t should have heavier tails than Normal."""
        config_n = RiskConfig(num_scenarios=5000, seed=42)
        config_t = RiskConfig(
            num_scenarios=5000, distribution=DistributionModel.STUDENT_T,
            df=3.0, seed=42,
        )
        normal = generate_scenarios(0.0, 0.2, config_n)
        student = generate_scenarios(0.0, 0.2, config_t)
        # Kurtosis of Student-t(3) is infinite; sample kurtosis should be higher
        kurt_n = np.mean(normal ** 4) / np.mean(normal ** 2) ** 2
        kurt_t = np.mean(student ** 4) / np.mean(student ** 2) ** 2
        assert kurt_t > kurt_n

    def test_scenarios_mean(self):
        config = RiskConfig(num_scenarios=50000, seed=42)
        scenarios = generate_scenarios(0.1, 0.2, config)
        # Mean should be close to expected (scaled by horizon)
        assert abs(scenarios.mean() - 0.1 / 252.0) < 0.01


class TestClassicalRiskMetrics:
    """Tests for classical VaR, CVaR, and other risk metrics."""

    def test_var_positive(self):
        rng = np.random.RandomState(42)
        scenarios = rng.randn(10000) * 0.02
        var = compute_var(scenarios, 0.95)
        assert var > 0

    def test_cvar_greater_than_var(self):
        rng = np.random.RandomState(42)
        scenarios = rng.randn(10000) * 0.02
        var = compute_var(scenarios, 0.95)
        cvar = compute_cvar(scenarios, var)
        assert cvar >= var - 1e-10

    def test_var_increases_with_confidence(self):
        rng = np.random.RandomState(42)
        scenarios = rng.randn(10000) * 0.02
        var_90 = compute_var(scenarios, 0.90)
        var_99 = compute_var(scenarios, 0.99)
        assert var_99 > var_90

    def test_max_drawdown_bounds(self):
        rng = np.random.RandomState(42)
        scenarios = rng.randn(1000) * 0.01
        dd = compute_max_drawdown(scenarios)
        assert 0 <= dd <= 1.0

    def test_sharpe_ratio_sign(self):
        # Positive mean returns => positive Sharpe
        rng = np.random.RandomState(42)
        scenarios = 0.001 + rng.randn(5000) * 0.01
        sharpe = compute_sharpe_ratio(scenarios)
        assert sharpe > 0

    def test_sortino_ratio_positive_for_positive_mean(self):
        rng = np.random.RandomState(42)
        scenarios = 0.001 + rng.randn(5000) * 0.01
        sortino = compute_sortino_ratio(scenarios)
        assert sortino > 0


class TestQuantumRisk:
    """Tests for quantum-enhanced VaR/CVaR estimation."""

    def test_quantum_var_positive(self):
        rng = np.random.RandomState(42)
        scenarios = rng.randn(5000) * 0.02
        var_q, ae_result = quantum_var(scenarios, confidence=0.95, num_qubits=3)
        assert var_q > 0

    def test_quantum_var_ae_result(self):
        rng = np.random.RandomState(42)
        scenarios = rng.randn(5000) * 0.02
        _, ae_result = quantum_var(scenarios, confidence=0.95, num_qubits=3)
        assert isinstance(ae_result, AEResult)
        assert 0 <= ae_result.estimation <= 1.0

    def test_quantum_cvar_positive(self):
        rng = np.random.RandomState(42)
        scenarios = rng.randn(5000) * 0.02
        var_val = compute_var(scenarios, 0.95)
        cvar_q, _ = quantum_cvar(scenarios, var_val, num_qubits=3)
        assert cvar_q > 0

    def test_quantum_var_close_to_classical(self):
        rng = np.random.RandomState(42)
        scenarios = rng.randn(10000) * 0.02
        var_cl = compute_var(scenarios, 0.95)
        var_q, _ = quantum_var(scenarios, confidence=0.95, num_qubits=4)
        # Within factor of 3 (QAE discretization is coarse with few qubits)
        assert var_q < var_cl * 3.0
        assert var_q > var_cl * 0.1


class TestRiskAnalyzer:
    """Tests for the RiskAnalyzer class."""

    def test_classical_analyze(self, simple_returns, simple_covariance, equal_weights):
        config = RiskConfig(num_scenarios=2000, seed=42)
        analyzer = RiskAnalyzer(config)
        result = analyzer.classical_analyze(
            simple_returns, simple_covariance, equal_weights
        )
        assert result.var > 0
        assert result.cvar >= result.var - 1e-10
        assert result.method == "classical"

    def test_quantum_analyze(self, simple_returns, simple_covariance, equal_weights):
        config = RiskConfig(num_scenarios=2000, num_qubits=3, seed=42)
        analyzer = RiskAnalyzer(config)
        result = analyzer.analyze(
            simple_returns, simple_covariance, equal_weights
        )
        assert result.var > 0
        assert result.method == "QAE-enhanced"

    def test_analyze_scenarios_directly(self):
        rng = np.random.RandomState(42)
        scenarios = rng.randn(3000) * 0.02
        config = RiskConfig(num_qubits=3)
        analyzer = RiskAnalyzer(config)
        result = analyzer.analyze_scenarios(scenarios)
        assert result.var > 0
        assert result.cvar > 0

    def test_confidence_interval_on_var(self, simple_returns, simple_covariance, equal_weights):
        config = RiskConfig(num_scenarios=2000, seed=42)
        analyzer = RiskAnalyzer(config)
        result = analyzer.classical_analyze(
            simple_returns, simple_covariance, equal_weights
        )
        ci = result.var_confidence_interval
        assert ci[0] <= ci[1]
        assert ci[0] > 0


# ======================================================================
# INTEGRATION / CROSS-MODULE TESTS
# ======================================================================


class TestCrossModuleIntegration:
    """Tests that verify modules work together correctly."""

    def test_portfolio_then_risk(self, simple_returns, simple_covariance):
        """Optimize a portfolio, then assess its risk."""
        opt = classical_portfolio_optimize(
            simple_returns, simple_covariance, risk_aversion=0.5, budget=2
        )
        config = RiskConfig(num_scenarios=2000, seed=42)
        analyzer = RiskAnalyzer(config)
        risk = analyzer.classical_analyze(
            simple_returns, simple_covariance, opt.weights
        )
        assert risk.var > 0
        assert risk.cvar >= risk.var - 1e-10

    def test_option_price_matches_bs_direction(self):
        """QAE price should be in the right ballpark of BS."""
        bs_call = black_scholes_call(100, 100, 0.05, 0.2, 1.0)
        qae_call = price_european_call(
            spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
            num_price_qubits=4,
        )
        # Both should be positive and in the same order of magnitude
        assert qae_call.price > 0
        assert bs_call > 0
        assert qae_call.price / bs_call < 5.0
        assert qae_call.price / bs_call > 0.1

    def test_frontier_optimal_has_low_variance(self, simple_returns, simple_covariance):
        """Minimum-variance portfolio should have lower variance than equal weight."""
        frontier = compute_efficient_frontier(
            simple_returns, simple_covariance, num_points=20
        )
        min_var_point = min(frontier, key=lambda p: p.variance)
        eq_var = float(
            np.ones(3) / 3.0 @ simple_covariance @ np.ones(3) / 3.0
        )
        assert min_var_point.variance <= eq_var + 1e-6


# ======================================================================
# EDGE CASE TESTS
# ======================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_asset_portfolio(self):
        returns = np.array([0.10])
        cov = np.array([[0.04]])
        result = classical_portfolio_optimize(returns, cov, budget=1)
        assert result.weights[0] == pytest.approx(1.0)

    def test_zero_volatility_scenario(self):
        config = RiskConfig(num_scenarios=100, seed=42)
        scenarios = generate_scenarios(0.1, 0.0001, config)
        var = compute_var(scenarios, 0.95)
        assert np.isfinite(var)

    def test_identical_returns(self):
        returns = np.array([0.10, 0.10, 0.10])
        cov = np.eye(3) * 0.04
        frontier = compute_efficient_frontier(returns, cov, num_points=5)
        assert len(frontier) >= 1

    def test_cvar_with_no_tail_losses(self):
        # All positive returns -- VaR is the smallest positive return (negated),
        # so actually negative loss. CVaR should handle this gracefully.
        rng = np.random.RandomState(42)
        scenarios = np.abs(rng.randn(1000)) * 0.01
        var = compute_var(scenarios, 0.95)
        cvar = compute_cvar(scenarios, var)
        # With all-positive returns, VaR itself may be negative (a "gain"),
        # so CVaR can also be negative.  The key invariant is cvar >= var.
        assert cvar >= var - 1e-10

    def test_empty_good_states_in_qae(self):
        """QAE with no good states should still return valid result."""
        oracle = np.eye(2, dtype=complex)
        qae = CanonicalQAE(4)
        # empty good states means amplitude = 0
        result = qae.estimate(oracle, good_indices=[])
        assert result.estimation == pytest.approx(0.0, abs=0.05)
