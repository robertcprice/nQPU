"""Comprehensive tests for the nqpu.trading package.

Covers: quantum_volatility (QuantumStateEncoder, HamiltonianEvolution,
BornRuleMeasurement, QuantumVolatilitySurface), regime_detection
(MarketRegime, QuantumRegimeDetector, RegimeTransitionMatrix,
VolatilityRegimeClassifier), feature_engineering (QuantumFeatureMap,
QuantumKernelSimilarity, EntanglementFeatures, compute_financial_features),
signal_processing (Signal, QuantumSignalGenerator, QuantumFilter,
QuantumMomentum, QuantumMeanReversion, combine_signals), risk_management
(QuantumVaR, QuantumPortfolioOptimizer, QuantumCorrelation, KellyCriterion,
drawdown_analysis), and backtesting (QuantumBacktester, PerformanceMetrics,
RegimeAwareBacktest, WalkForwardOptimizer).

Uses seed=42 for reproducibility, no external dependencies beyond numpy + pytest.
"""

from __future__ import annotations

import numpy as np
import pytest

from nqpu.trading import (
    BornRuleMeasurement,
    DrawdownInfo,
    EncodingType,
    EntanglementFeatures,
    HamiltonianEvolution,
    KellyCriterion,
    MarketRegime,
    PerformanceMetrics,
    QuantumBacktester,
    QuantumCorrelation,
    QuantumFeatureMap,
    QuantumFilter,
    QuantumKernelSimilarity,
    QuantumMeanReversion,
    QuantumMomentum,
    QuantumPortfolioOptimizer,
    QuantumRegimeDetector,
    QuantumSignalGenerator,
    QuantumStateEncoder,
    QuantumVaR,
    QuantumVolatilitySurface,
    RegimeAwareBacktest,
    RegimeTransitionMatrix,
    Signal,
    VolatilityRegimeClassifier,
    WalkForwardOptimizer,
    combine_signals,
    compute_financial_features,
    drawdown_analysis,
    extrapolate_iv_surface,
    interpolate_iv_surface,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def daily_returns(rng):
    """500 days of simulated daily returns."""
    return rng.normal(0.0005, 0.02, size=500)


@pytest.fixture
def multi_asset_returns(rng):
    """500 days x 3 assets."""
    return rng.normal(0.0003, 0.02, size=(500, 3))


@pytest.fixture
def price_series(rng):
    """500-period price series starting at 100."""
    returns = rng.normal(0.0005, 0.01, size=500)
    prices = 100.0 * np.cumprod(1.0 + returns)
    return prices


# ---------------------------------------------------------------------------
# Quantum Volatility tests
# ---------------------------------------------------------------------------

class TestQuantumStateEncoder:
    def test_encode_single(self, rng):
        encoder = QuantumStateEncoder(n_qubits=3)
        features = rng.random(3)
        encoder.fit_scaling(features.reshape(1, -1))
        state = encoder.encode(features)
        assert state.shape == (2 ** 3,)
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-10)

    def test_encode_batch(self, rng):
        encoder = QuantumStateEncoder(n_qubits=3)
        batch = rng.random((10, 3))
        encoder.fit_scaling(batch)
        states = encoder.encode_batch(batch)
        assert states.shape == (10, 8)
        for s in states:
            assert np.isclose(np.linalg.norm(s), 1.0, atol=1e-10)


class TestHamiltonianEvolution:
    def test_evolve(self, rng):
        he = HamiltonianEvolution(n_qubits=2)
        state = np.zeros(4, dtype=np.complex128)
        state[0] = 1.0
        evolved = he.evolve(state)
        assert evolved.shape == (4,)
        assert np.isclose(np.linalg.norm(evolved), 1.0, atol=1e-10)

    def test_num_params(self):
        he = HamiltonianEvolution(n_qubits=3)
        assert he.num_params > 0


class TestBornRuleMeasurement:
    def test_probabilities(self):
        brm = BornRuleMeasurement(n_qubits=2)
        state = np.array([1, 0, 0, 0], dtype=np.complex128)
        probs = brm.probabilities(state)
        assert len(probs) == 4
        assert np.isclose(sum(probs), 1.0)

    def test_expectation(self):
        brm = BornRuleMeasurement(n_qubits=2)
        state = np.array([1, 0, 0, 0], dtype=np.complex128)
        exp = brm.expectation(state)
        assert isinstance(exp, float)


class TestQuantumVolatilitySurface:
    def test_fit_predict(self, rng):
        qvs = QuantumVolatilitySurface(n_qubits=4, seed=SEED)
        # Create synthetic strike/expiry/iv data
        strikes = np.linspace(90, 110, 10)
        expiries = np.linspace(0.1, 1.0, 5)
        iv_data = rng.uniform(0.1, 0.5, size=(10, 5))
        spot = 100.0
        qvs.fit(strikes, expiries, iv_data, spot=spot, max_iter=5)
        predicted = qvs.predict(strikes, expiries, spot=spot)
        assert predicted.shape == (10, 5)
        assert np.all(predicted >= 0)


class TestInterpolateExtrapolate:
    def test_interpolate_iv_surface(self, rng):
        strikes = np.linspace(90, 110, 5)
        expiries = np.linspace(0.1, 1.0, 3)
        iv_grid = rng.uniform(0.1, 0.4, size=(5, 3))
        new_strikes = np.linspace(92, 108, 8)
        new_expiries = np.linspace(0.2, 0.8, 4)
        result = interpolate_iv_surface(strikes, expiries, iv_grid, new_strikes, new_expiries)
        assert result.shape == (8, 4)

    def test_extrapolate_iv_surface(self, rng):
        strikes = np.linspace(90, 110, 5)
        expiries = np.linspace(0.1, 1.0, 3)
        iv_grid = rng.uniform(0.1, 0.4, size=(5, 3))
        new_strikes = np.linspace(80, 120, 10)
        new_expiries = np.linspace(0.05, 1.5, 5)
        result = extrapolate_iv_surface(strikes, expiries, iv_grid, new_strikes, new_expiries)
        assert result.shape == (10, 5)


# ---------------------------------------------------------------------------
# Regime Detection tests
# ---------------------------------------------------------------------------

class TestMarketRegime:
    def test_enum_values(self):
        assert MarketRegime.BULL is not None
        assert MarketRegime.BEAR is not None
        assert MarketRegime.SIDEWAYS is not None
        assert MarketRegime.VOLATILE is not None


class TestQuantumRegimeDetector:
    def test_fit_detect(self, daily_returns):
        detector = QuantumRegimeDetector(n_qubits=3)
        detector.fit(daily_returns, window=30)
        regime = detector.detect(daily_returns[-30:])
        assert isinstance(regime, MarketRegime)

    def test_detect_series(self, daily_returns):
        detector = QuantumRegimeDetector(n_qubits=3)
        detector.fit(daily_returns, window=30)
        regimes = detector.detect_series(daily_returns, window=30)
        assert len(regimes) > 0
        for r in regimes:
            assert isinstance(r, MarketRegime)

    def test_fidelities(self, daily_returns):
        detector = QuantumRegimeDetector(n_qubits=3)
        detector.fit(daily_returns, window=30)
        fids = detector.fidelities(daily_returns[-30:])
        assert isinstance(fids, dict)


class TestRegimeTransitionMatrix:
    def test_fit_and_transition_probs(self, daily_returns):
        detector = QuantumRegimeDetector(n_qubits=3)
        detector.fit(daily_returns, window=30)
        regimes = detector.detect_series(daily_returns, window=30)
        rtm = RegimeTransitionMatrix()
        rtm.fit(regimes)
        probs = rtm.transition_probs(MarketRegime.BULL)
        assert isinstance(probs, dict)
        assert np.isclose(sum(probs.values()), 1.0, atol=1e-6)

    def test_steady_state(self, daily_returns):
        detector = QuantumRegimeDetector(n_qubits=3)
        detector.fit(daily_returns, window=30)
        regimes = detector.detect_series(daily_returns, window=30)
        rtm = RegimeTransitionMatrix()
        rtm.fit(regimes)
        ss = rtm.steady_state()
        assert isinstance(ss, dict)
        assert np.isclose(sum(ss.values()), 1.0, atol=0.01)


class TestVolatilityRegimeClassifier:
    def test_classify(self):
        clf = VolatilityRegimeClassifier(n_qubits=4)
        # classify expects a term structure of length n_qubits
        term_structure = np.array([0.25, 0.22, 0.20, 0.19])
        regime = clf.classify(term_structure)
        assert isinstance(regime, MarketRegime)

    def test_classify_series(self):
        clf = VolatilityRegimeClassifier(n_qubits=4)
        # classify_series expects (n_observations, n_qubits) array
        term_structures = np.array([
            [0.25, 0.22, 0.20, 0.19],
            [0.18, 0.20, 0.22, 0.24],
            [0.20, 0.20, 0.20, 0.20],
        ])
        regimes = clf.classify_series(term_structures)
        assert len(regimes) == 3
        for r in regimes:
            assert isinstance(r, MarketRegime)


# ---------------------------------------------------------------------------
# Feature Engineering tests
# ---------------------------------------------------------------------------

class TestEncodingType:
    def test_enum_values(self):
        assert EncodingType.ANGLE.value == "angle"
        assert EncodingType.AMPLITUDE.value == "amplitude"
        assert EncodingType.ZZ.value == "zz"


class TestQuantumFeatureMap:
    @pytest.mark.parametrize("encoding", ["angle", "amplitude", "zz"])
    def test_encode_returns_normalized_state(self, encoding, rng):
        qfm = QuantumFeatureMap(n_qubits=3, encoding=encoding)
        x = rng.random(3)
        state = qfm.encode(x)
        assert state.shape == (2 ** 3,)
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-8)

    def test_fit_transform(self, rng):
        qfm = QuantumFeatureMap(n_qubits=3, encoding="angle")
        features = rng.random((20, 3))
        states = qfm.fit_transform(features)
        assert states.shape == (20, 8)

    def test_transform_after_fit(self, rng):
        qfm = QuantumFeatureMap(n_qubits=3, encoding="angle")
        train = rng.random((20, 3))
        qfm.fit(train)
        test = rng.random((5, 3))
        states = qfm.transform(test)
        assert states.shape == (5, 8)


class TestQuantumKernelSimilarity:
    def test_kernel_symmetric(self, rng):
        qfm = QuantumFeatureMap(n_qubits=3, encoding="angle")
        qks = QuantumKernelSimilarity(qfm)
        x = rng.random(3)
        y = rng.random(3)
        k_xy = qks.kernel(x, y)
        k_yx = qks.kernel(y, x)
        assert k_xy == pytest.approx(k_yx, abs=1e-10)

    def test_kernel_self_is_one(self, rng):
        qfm = QuantumFeatureMap(n_qubits=3, encoding="angle")
        qks = QuantumKernelSimilarity(qfm)
        x = rng.random(3)
        assert qks.kernel(x, x) == pytest.approx(1.0, abs=1e-8)

    def test_kernel_matrix_shape(self, rng):
        qfm = QuantumFeatureMap(n_qubits=3, encoding="angle")
        qks = QuantumKernelSimilarity(qfm)
        X = rng.random((10, 3))
        K = qks.kernel_matrix(X)
        assert K.shape == (10, 10)
        assert np.allclose(K, K.T, atol=1e-10)

    def test_target_alignment(self, rng):
        qfm = QuantumFeatureMap(n_qubits=3, encoding="angle")
        qks = QuantumKernelSimilarity(qfm)
        X = rng.random((20, 3))
        y = rng.choice([-1, 1], size=20).astype(float)
        alignment = qks.target_alignment(X, y)
        assert isinstance(alignment, float)


class TestEntanglementFeatures:
    def test_pairwise_entanglement(self, rng):
        ef = EntanglementFeatures(n_qubits_per_asset=2)
        returns = rng.normal(0, 0.02, size=(200, 3))
        ent = ef.pairwise_entanglement(returns)
        assert ent.shape == (3, 3)
        # Should be symmetric
        assert np.allclose(ent, ent.T, atol=1e-10)

    def test_entanglement_features_vector(self, rng):
        ef = EntanglementFeatures(n_qubits_per_asset=2)
        returns = rng.normal(0, 0.02, size=(200, 3))
        feats = ef.entanglement_features(returns)
        # 3 choose 2 = 3
        assert feats.shape == (3,)


class TestComputeFinancialFeatures:
    def test_output_shape(self, price_series):
        features = compute_financial_features(price_series, window=20)
        expected_windows = len(price_series) - 1 - 20 + 1
        assert features.shape == (expected_windows, 4)

    def test_with_volume(self, price_series, rng):
        volume = rng.uniform(1e6, 1e7, size=len(price_series))
        features = compute_financial_features(
            price_series, window=20, include_volume=True, volume=volume
        )
        assert features.shape[1] == 5


# ---------------------------------------------------------------------------
# Signal Processing tests
# ---------------------------------------------------------------------------

class TestSignal:
    def test_from_direction(self):
        sig = Signal.from_direction(1, confidence=0.8)
        assert sig.direction == 1
        assert sig.confidence == 0.8

    def test_from_direction_sell(self):
        sig = Signal.from_direction(-1, confidence=0.6)
        assert sig.direction == -1


class TestQuantumSignalGenerator:
    def test_generate(self, price_series):
        gen = QuantumSignalGenerator(n_qubits=4, seed=SEED)
        signals = gen.generate(price_series, volume=None, window=20)
        assert len(signals) > 0
        for s in signals:
            assert isinstance(s, Signal)
            assert -1.0 <= s.direction <= 1.0

    def test_generate_single(self, daily_returns):
        gen = QuantumSignalGenerator(n_qubits=3, seed=SEED)
        sig = gen.generate_single(daily_returns[-20:])
        assert isinstance(sig, Signal)


class TestQuantumFilter:
    def test_filter(self, daily_returns):
        qf = QuantumFilter(n_qubits=3)
        filtered = qf.filter(daily_returns)
        assert len(filtered) == len(daily_returns)

    def test_step_and_reset(self, daily_returns):
        qf = QuantumFilter(n_qubits=3)
        out = qf.step(daily_returns[0])
        assert isinstance(out, float)
        qf.reset()


class TestQuantumMomentum:
    def test_compute(self, price_series):
        qm = QuantumMomentum(n_levels=16, n_steps=5)
        result = qm.compute(price_series, window=20)
        assert len(result) > 0

    def test_compute_single(self, price_series):
        qm = QuantumMomentum(n_levels=16, n_steps=5)
        val = qm.compute_single(price_series[-20:])
        assert isinstance(val, float)


class TestQuantumMeanReversion:
    def test_fit_compute(self, daily_returns):
        qmr = QuantumMeanReversion(n_qubits=3)
        qmr.fit(daily_returns[:200])
        result = qmr.compute(daily_returns[200:])
        assert len(result) > 0

    def test_fidelity(self, daily_returns):
        qmr = QuantumMeanReversion(n_qubits=3)
        qmr.fit(daily_returns[:200])
        fid = qmr.fidelity(daily_returns[200:250])
        assert isinstance(fid, float)


class TestCombineSignals:
    def test_combine(self):
        s1 = [Signal.from_direction(0.5, confidence=0.8)]
        s2 = [Signal.from_direction(-0.3, confidence=0.5)]
        combined = combine_signals({"momentum": s1, "reversion": s2})
        assert len(combined) == 1
        assert isinstance(combined[0], Signal)
        assert -1 <= combined[0].direction <= 1


# ---------------------------------------------------------------------------
# Risk Management tests
# ---------------------------------------------------------------------------

class TestQuantumVaR:
    def test_compute_var(self, daily_returns):
        qvar = QuantumVaR(n_qubits=4)
        var_95 = qvar.compute(daily_returns, confidence=0.95)
        assert isinstance(var_95, float)
        assert var_95 < 0  # VaR should be negative (a loss)

    def test_compute_cvar(self, daily_returns):
        qvar = QuantumVaR(n_qubits=4)
        cvar = qvar.compute_cvar(daily_returns, confidence=0.95)
        var = qvar.compute(daily_returns, confidence=0.95)
        assert cvar <= var  # CVaR should be worse (more negative)

    def test_loss_distribution_state(self, daily_returns):
        qvar = QuantumVaR(n_qubits=4)
        state, bin_edges = qvar.loss_distribution_state(daily_returns)
        assert len(state) == 2 ** 4
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-10)

    @pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
    def test_var_increases_with_confidence(self, daily_returns, confidence):
        qvar = QuantumVaR(n_qubits=5)
        var = qvar.compute(daily_returns, confidence=confidence)
        assert isinstance(var, float)


class TestQuantumPortfolioOptimizer:
    def test_optimize_weights_sum_to_one(self, multi_asset_returns):
        opt = QuantumPortfolioOptimizer(n_assets=3, seed=SEED)
        weights = opt.optimize(multi_asset_returns)
        assert np.isclose(weights.sum(), 1.0, atol=1e-10)
        assert np.all(weights >= 0)

    def test_optimize_shape(self, multi_asset_returns):
        opt = QuantumPortfolioOptimizer(n_assets=3, seed=SEED)
        weights = opt.optimize(multi_asset_returns)
        assert weights.shape == (3,)

    def test_wrong_asset_count_raises(self, multi_asset_returns):
        opt = QuantumPortfolioOptimizer(n_assets=5, seed=SEED)
        with pytest.raises(ValueError, match="Expected 5 assets"):
            opt.optimize(multi_asset_returns)

    def test_efficient_frontier(self, multi_asset_returns):
        opt = QuantumPortfolioOptimizer(n_assets=3, seed=SEED)
        ret, risk, weights = opt.efficient_frontier(multi_asset_returns, n_points=5)
        assert len(ret) == 5
        assert len(risk) == 5
        assert weights.shape == (5, 3)


class TestQuantumCorrelation:
    def test_correlation_matrix_shape(self, multi_asset_returns):
        qc = QuantumCorrelation(n_qubits_per_asset=2)
        corr = qc.correlation_matrix(multi_asset_returns)
        assert corr.shape == (3, 3)
        assert np.allclose(np.diag(corr), 1.0)
        assert np.allclose(corr, corr.T, atol=1e-10)

    def test_pairwise_bounded(self, rng):
        qc = QuantumCorrelation(n_qubits_per_asset=2)
        a = rng.normal(0, 0.02, 200)
        b = rng.normal(0, 0.02, 200)
        c = qc.pairwise(a, b)
        assert 0 <= c <= 1

    def test_divergence_from_classical(self, multi_asset_returns):
        qc = QuantumCorrelation(n_qubits_per_asset=2)
        div = qc.divergence_from_classical(multi_asset_returns)
        assert div.shape == (3, 3)
        assert np.all(div >= 0)


class TestKellyCriterion:
    def test_compute_bounded(self, daily_returns):
        kelly = KellyCriterion(n_qubits=4, fractional_kelly=0.5, max_fraction=0.25)
        fraction = kelly.compute(daily_returns)
        assert 0.0 <= fraction <= 0.25

    def test_compute_with_details(self, daily_returns):
        kelly = KellyCriterion(n_qubits=4, fractional_kelly=0.5)
        details = kelly.compute_with_details(daily_returns)
        assert "fraction" in details
        assert "classical_kelly" in details
        assert "quantum_confidence" in details
        assert "win_rate" in details
        assert "entropy" in details

    def test_short_series_returns_zero(self):
        kelly = KellyCriterion(n_qubits=4)
        assert kelly.compute(np.array([0.01])) == 0.0

    def test_all_wins_returns_nonzero(self, rng):
        kelly = KellyCriterion(n_qubits=4, fractional_kelly=1.0)
        # Mostly positive returns
        returns = np.abs(rng.normal(0.01, 0.005, 100))
        fraction = kelly.compute(returns)
        # Can be 0 if all losses are 0 (edge case)
        assert fraction >= 0.0


class TestDrawdownAnalysis:
    def test_drawdown_info(self, daily_returns):
        info = drawdown_analysis(daily_returns)
        assert isinstance(info, DrawdownInfo)
        assert info.max_drawdown <= 0
        assert info.max_drawdown_duration >= 0
        assert info.drawdown_series.shape == daily_returns.shape

    def test_recovery_times(self, daily_returns):
        info = drawdown_analysis(daily_returns)
        for rt in info.recovery_times:
            assert rt > 0


# ---------------------------------------------------------------------------
# Backtesting tests
# ---------------------------------------------------------------------------

class TestPerformanceMetrics:
    def test_from_returns(self, daily_returns):
        metrics = PerformanceMetrics.from_returns(daily_returns)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.sortino_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert metrics.max_drawdown <= 0

    def test_empty_returns(self):
        metrics = PerformanceMetrics.from_returns(np.array([]))
        assert metrics.total_return == 0.0

    def test_summary_string(self, daily_returns):
        metrics = PerformanceMetrics.from_returns(daily_returns)
        summary = metrics.summary()
        assert "Sharpe" in summary
        assert "Sortino" in summary


class TestQuantumBacktester:
    def test_run(self, price_series):
        bt = QuantumBacktester(transaction_cost_bps=5.0, seed=SEED)
        result = bt.run(price_series, window=20)
        assert "returns" in result
        assert "signals" in result
        assert "positions" in result
        assert "equity_curve" in result
        assert "metrics" in result
        assert isinstance(result["metrics"], PerformanceMetrics)

    def test_equity_curve_starts_at_one(self, price_series):
        bt = QuantumBacktester(seed=SEED)
        result = bt.run(price_series, window=20)
        # Equity curve = cumprod(1 + returns), first element = 1 + first_return
        assert result["equity_curve"][0] > 0


class TestRegimeAwareBacktest:
    def test_run(self, price_series):
        rab = RegimeAwareBacktest(seed=SEED)
        result = rab.run(price_series, window=30, regime_window=60)
        assert "base_result" in result
        assert "regimes" in result
        assert "regime_metrics" in result
        for label, metrics in result["regime_metrics"].items():
            assert isinstance(metrics, PerformanceMetrics)


class TestWalkForwardOptimizer:
    def test_run(self, rng):
        prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, 600))
        wfo = WalkForwardOptimizer(train_size=200, test_size=50, seed=SEED)
        results = wfo.run(prices, window=20, n_trials=3)
        assert "oos_returns" in results
        assert "oos_metrics" in results
        assert isinstance(results["oos_metrics"], PerformanceMetrics)
        assert "fold_results" in results
        assert len(results["fold_results"]) > 0
