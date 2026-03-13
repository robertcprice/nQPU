"""
Quantum Backtesting Framework.

Provides a backtesting engine for quantum-inspired trading strategies
with performance analytics, regime-aware attribution, and walk-forward
optimisation.

The framework decouples signal generation from risk management, allowing
any ``QuantumSignalGenerator`` to be paired with any risk sizing method
(e.g. ``KellyCriterion``) and evaluated across historical data with
realistic transaction costs, slippage, and position constraints.

Example
-------
>>> import numpy as np
>>> from nqpu.trading.backtesting import (
...     QuantumBacktester,
...     PerformanceMetrics,
...     RegimeAwareBacktest,
... )
>>>
>>> prices = np.cumsum(np.random.randn(500)) + 100
>>> bt = QuantumBacktester(transaction_cost_bps=5.0)
>>> result = bt.run(prices, window=20)
>>> metrics = PerformanceMetrics.from_returns(result["returns"])
>>> print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from nqpu.trading.signal_processing import QuantumSignalGenerator, Signal
from nqpu.trading.risk_management import KellyCriterion, drawdown_analysis


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """Comprehensive strategy performance statistics.

    Attributes
    ----------
    total_return : float
    annualised_return : float
    annualised_volatility : float
    sharpe_ratio : float
    sortino_ratio : float
    max_drawdown : float
    max_drawdown_duration : int
    calmar_ratio : float
    win_rate : float
    profit_factor : float
    total_trades : int
    avg_return_per_trade : float
    """

    total_return: float = 0.0
    annualised_return: float = 0.0
    annualised_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_return_per_trade: float = 0.0

    @staticmethod
    def from_returns(
        returns: np.ndarray,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> "PerformanceMetrics":
        """Compute all metrics from a return series.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods,)``
            Strategy returns per period (not cumulative).
        periods_per_year : int
            Trading periods per year (252 for daily).
        risk_free_rate : float
            Annual risk-free rate for Sharpe/Sortino computation.

        Returns
        -------
        PerformanceMetrics

        Example
        -------
        >>> ret = np.random.randn(252) * 0.01
        >>> m = PerformanceMetrics.from_returns(ret)
        >>> assert isinstance(m.sharpe_ratio, float)
        """
        returns = np.asarray(returns, dtype=np.float64)
        n = len(returns)
        if n == 0:
            return PerformanceMetrics()

        # Cumulative and total return.
        cumulative = np.cumprod(1.0 + returns)
        total_return = float(cumulative[-1] - 1.0)

        # Annualised metrics.
        years = n / periods_per_year
        annualised_return = float(
            (1.0 + total_return) ** (1.0 / max(years, 1e-6)) - 1.0
        )
        annualised_vol = float(returns.std() * np.sqrt(periods_per_year))

        # Sharpe ratio.
        excess_return = annualised_return - risk_free_rate
        sharpe = excess_return / (annualised_vol + 1e-15)

        # Sortino ratio (uses downside deviation).
        downside = returns[returns < 0]
        downside_dev = float(
            np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year)
        ) if len(downside) > 0 else 1e-15
        sortino = excess_return / (downside_dev + 1e-15)

        # Drawdown.
        dd_info = drawdown_analysis(returns)
        max_dd = dd_info.max_drawdown
        max_dd_dur = dd_info.max_drawdown_duration

        # Calmar ratio.
        calmar = annualised_return / (abs(max_dd) + 1e-15)

        # Win rate and profit factor.
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        total_trades = int(np.sum(returns != 0))
        win_rate = float(len(wins) / max(total_trades, 1))

        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(losses.sum())) if len(losses) > 0 else 1e-15
        profit_factor = gross_profit / (gross_loss + 1e-15)

        avg_ret = float(returns[returns != 0].mean()) if total_trades > 0 else 0.0

        return PerformanceMetrics(
            total_return=total_return,
            annualised_return=annualised_return,
            annualised_volatility=annualised_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_dur,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_return_per_trade=avg_ret,
        )

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=== Performance Summary ===",
            f"Total Return:          {self.total_return:+.2%}",
            f"Annualised Return:     {self.annualised_return:+.2%}",
            f"Annualised Volatility: {self.annualised_volatility:.2%}",
            f"Sharpe Ratio:          {self.sharpe_ratio:.3f}",
            f"Sortino Ratio:         {self.sortino_ratio:.3f}",
            f"Max Drawdown:          {self.max_drawdown:.2%}",
            f"Max DD Duration:       {self.max_drawdown_duration} periods",
            f"Calmar Ratio:          {self.calmar_ratio:.3f}",
            f"Win Rate:              {self.win_rate:.1%}",
            f"Profit Factor:         {self.profit_factor:.2f}",
            f"Total Trades:          {self.total_trades}",
            f"Avg Return/Trade:      {self.avg_return_per_trade:+.4%}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quantum Backtester
# ---------------------------------------------------------------------------

class QuantumBacktester:
    """Run quantum trading strategies over historical price data.

    Generates signals using a ``QuantumSignalGenerator``, sizes positions
    with a ``KellyCriterion``, and simulates execution with configurable
    transaction costs and position limits.

    Parameters
    ----------
    signal_generator : QuantumSignalGenerator or None
        If ``None``, a default generator is created.
    position_sizer : KellyCriterion or None
        If ``None``, a default sizer is created.
    transaction_cost_bps : float
        Round-trip transaction cost in basis points.
    max_position : float
        Maximum absolute position size in ``[0, 1]``.
    seed : int or None
        Random seed for the signal generator.

    Example
    -------
    >>> bt = QuantumBacktester(transaction_cost_bps=5.0, seed=42)
    >>> prices = np.cumsum(np.random.randn(300)) + 100
    >>> result = bt.run(prices, window=20)
    >>> assert "returns" in result
    >>> assert "signals" in result
    """

    def __init__(
        self,
        signal_generator: Optional[QuantumSignalGenerator] = None,
        position_sizer: Optional[KellyCriterion] = None,
        transaction_cost_bps: float = 5.0,
        max_position: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.signal_generator = signal_generator or QuantumSignalGenerator(
            n_qubits=4, seed=seed
        )
        self.position_sizer = position_sizer or KellyCriterion(
            n_qubits=4, fractional_kelly=0.5
        )
        self.transaction_cost = transaction_cost_bps / 10_000.0
        self.max_position = np.clip(max_position, 0.0, 1.0)

    # -- public API ---------------------------------------------------------

    def run(
        self,
        prices: np.ndarray,
        volume: Optional[np.ndarray] = None,
        window: int = 20,
    ) -> Dict[str, Any]:
        """Execute the backtest.

        Parameters
        ----------
        prices : ndarray of shape ``(n_periods,)``
        volume : ndarray of shape ``(n_periods,)`` or None
        window : int
            Signal lookback window.

        Returns
        -------
        result : dict
            Keys: ``returns``, ``signals``, ``positions``, ``equity_curve``,
            ``metrics``.
        """
        prices = np.asarray(prices, dtype=np.float64)
        log_returns = np.diff(np.log(np.maximum(prices, 1e-12)))

        # Generate signals.
        signals = self.signal_generator.generate(prices, volume, window)

        # Align: signals start at index (window) relative to log_returns.
        # log_returns[i] corresponds to the return from price[i] to price[i+1].
        n_signals = len(signals)
        # Signals cover log_returns[window-1 : window-1 + n_signals].
        start_idx = window - 1

        positions = np.zeros(n_signals)
        strategy_returns = np.zeros(n_signals)

        # Compute position size from a rolling return window.
        for i in range(n_signals):
            ret_idx = start_idx + i

            sig = signals[i]
            direction = sig.direction

            # Size position using Kelly on recent returns.
            lookback_start = max(0, ret_idx - window)
            recent_returns = log_returns[lookback_start:ret_idx]
            if len(recent_returns) >= 5:
                kelly_fraction = self.position_sizer.compute(recent_returns)
            else:
                kelly_fraction = 0.01

            raw_position = direction * kelly_fraction * sig.confidence
            positions[i] = np.clip(
                raw_position, -self.max_position, self.max_position
            )

            # Strategy return = position * market return - transaction cost.
            if ret_idx < len(log_returns):
                market_return = log_returns[ret_idx]
                turnover = abs(positions[i] - (positions[i - 1] if i > 0 else 0.0))
                cost = turnover * self.transaction_cost
                strategy_returns[i] = positions[i] * market_return - cost

        # Equity curve.
        equity_curve = np.cumprod(1.0 + strategy_returns)
        metrics = PerformanceMetrics.from_returns(strategy_returns)

        return {
            "returns": strategy_returns,
            "signals": signals,
            "positions": positions,
            "equity_curve": equity_curve,
            "metrics": metrics,
        }


# ---------------------------------------------------------------------------
# Regime-Aware Backtest
# ---------------------------------------------------------------------------

class RegimeAwareBacktest:
    """Backtest with performance attribution by market regime.

    Wraps a ``QuantumBacktester`` and a regime detector to produce
    per-regime performance breakdowns, revealing whether a strategy
    works in specific market conditions or uniformly.

    Parameters
    ----------
    backtester : QuantumBacktester or None
    n_qubits : int
        Qubits for the regime detector.
    seed : int or None

    Example
    -------
    >>> rab = RegimeAwareBacktest(seed=42)
    >>> prices = np.cumsum(np.random.randn(500)) + 100
    >>> result = rab.run(prices, window=30)
    >>> assert "regime_metrics" in result
    """

    def __init__(
        self,
        backtester: Optional[QuantumBacktester] = None,
        n_qubits: int = 4,
        seed: Optional[int] = None,
    ) -> None:
        self.backtester = backtester or QuantumBacktester(seed=seed)
        self.n_qubits = n_qubits

    # -- public API ---------------------------------------------------------

    def run(
        self,
        prices: np.ndarray,
        volume: Optional[np.ndarray] = None,
        window: int = 20,
        regime_window: int = 60,
    ) -> Dict[str, Any]:
        """Run backtest with regime decomposition.

        Parameters
        ----------
        prices : ndarray of shape ``(n_periods,)``
        volume : ndarray of shape ``(n_periods,)`` or None
        window : int
            Signal lookback.
        regime_window : int
            Regime detection lookback.

        Returns
        -------
        result : dict
            Contains ``base_result``, ``regimes``, ``regime_metrics``.
        """
        from nqpu.trading.regime_detection import QuantumRegimeDetector

        prices = np.asarray(prices, dtype=np.float64)
        base_result = self.backtester.run(prices, volume, window)

        # Detect regimes.
        log_returns = np.diff(np.log(np.maximum(prices, 1e-12)))
        detector = QuantumRegimeDetector(n_qubits=self.n_qubits)
        detector.fit(log_returns, window=regime_window)
        regimes = detector.detect_series(log_returns, window=regime_window)

        # Align regimes with strategy returns.
        strat_returns = base_result["returns"]
        n_common = min(len(regimes), len(strat_returns))

        # Build per-regime return arrays.
        regime_returns: Dict[str, List[float]] = {}
        aligned_regimes: List[str] = []
        for i in range(n_common):
            r_label = str(regimes[i])
            aligned_regimes.append(r_label)
            if r_label not in regime_returns:
                regime_returns[r_label] = []
            regime_returns[r_label].append(strat_returns[i])

        # Compute per-regime metrics.
        regime_metrics: Dict[str, PerformanceMetrics] = {}
        for label, rets in regime_returns.items():
            arr = np.array(rets, dtype=np.float64)
            regime_metrics[label] = PerformanceMetrics.from_returns(arr)

        return {
            "base_result": base_result,
            "regimes": aligned_regimes,
            "regime_metrics": regime_metrics,
        }


# ---------------------------------------------------------------------------
# Walk-Forward Optimiser
# ---------------------------------------------------------------------------

class WalkForwardOptimizer:
    """Walk-forward optimisation with rolling train/test splits.

    Splits historical data into overlapping train/test windows, optimises
    strategy parameters on the training set, and evaluates on the test
    set.  This guards against overfitting by ensuring all performance
    numbers come from out-of-sample data.

    Parameters
    ----------
    train_size : int
        Training window size in periods.
    test_size : int
        Test window size in periods.
    step_size : int or None
        Step between windows.  Defaults to ``test_size``.
    seed : int or None

    Example
    -------
    >>> wfo = WalkForwardOptimizer(train_size=200, test_size=50)
    >>> prices = np.cumsum(np.random.randn(1000)) + 100
    >>> results = wfo.run(prices, window=20, n_trials=10)
    >>> assert "oos_metrics" in results
    """

    def __init__(
        self,
        train_size: int = 200,
        test_size: int = 50,
        step_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.seed = seed

    # -- public API ---------------------------------------------------------

    def run(
        self,
        prices: np.ndarray,
        volume: Optional[np.ndarray] = None,
        window: int = 20,
        n_trials: int = 20,
    ) -> Dict[str, Any]:
        """Execute walk-forward optimisation.

        Parameters
        ----------
        prices : ndarray of shape ``(n_periods,)``
        volume : ndarray of shape ``(n_periods,)`` or None
        window : int
            Signal lookback window.
        n_trials : int
            Number of parameter candidates per training fold.

        Returns
        -------
        result : dict
            Keys: ``oos_returns``, ``oos_metrics``, ``fold_results``,
            ``best_seeds``.
        """
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)

        fold_results: List[Dict[str, Any]] = []
        oos_returns_all: List[np.ndarray] = []
        best_seeds: List[int] = []

        start = 0
        rng = np.random.default_rng(self.seed)

        while start + self.train_size + self.test_size <= n:
            train_end = start + self.train_size
            test_end = train_end + self.test_size

            train_prices = prices[start:train_end]
            test_prices = prices[train_end:test_end]

            train_vol = volume[start:train_end] if volume is not None else None
            test_vol = volume[train_end:test_end] if volume is not None else None

            # Optimise: try different seeds on training data.
            best_sharpe = -np.inf
            best_seed = 0

            for _ in range(n_trials):
                trial_seed = int(rng.integers(0, 2**31))
                bt = QuantumBacktester(seed=trial_seed)
                try:
                    result = bt.run(train_prices, train_vol, window)
                    sharpe = result["metrics"].sharpe_ratio
                except Exception:
                    sharpe = -np.inf

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_seed = trial_seed

            # Test: evaluate best parameters on out-of-sample data.
            bt_test = QuantumBacktester(seed=best_seed)
            try:
                oos_result = bt_test.run(test_prices, test_vol, window)
                oos_returns = oos_result["returns"]
            except Exception:
                oos_returns = np.zeros(0)

            oos_returns_all.append(oos_returns)
            best_seeds.append(best_seed)
            fold_results.append(
                {
                    "train_start": start,
                    "train_end": train_end,
                    "test_start": train_end,
                    "test_end": test_end,
                    "train_sharpe": best_sharpe,
                    "oos_sharpe": (
                        PerformanceMetrics.from_returns(oos_returns).sharpe_ratio
                        if len(oos_returns) > 0
                        else 0.0
                    ),
                }
            )

            start += self.step_size

        # Aggregate out-of-sample results.
        if oos_returns_all:
            all_oos = np.concatenate(oos_returns_all)
        else:
            all_oos = np.zeros(0)

        oos_metrics = (
            PerformanceMetrics.from_returns(all_oos)
            if len(all_oos) > 0
            else PerformanceMetrics()
        )

        return {
            "oos_returns": all_oos,
            "oos_metrics": oos_metrics,
            "fold_results": fold_results,
            "best_seeds": best_seeds,
        }
