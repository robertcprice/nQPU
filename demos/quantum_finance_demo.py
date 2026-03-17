#!/usr/bin/env python3
"""Quantum Finance Demo -- nQPU SDK showcase.

Demonstrates quantum-enhanced financial computation:
1. Option Pricing: QAE vs Black-Scholes comparison
2. Portfolio Optimization: QAOA vs classical brute-force
3. Risk Analysis: Quantum VaR/CVaR computation
4. Trading Signals: Regime detection and backtesting

All data is synthetic (no external dependencies beyond numpy).
Matplotlib is optional -- figures are saved if available.

Usage:
    python demos/quantum_finance_demo.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Matplotlib setup (optional)
# ---------------------------------------------------------------------------

HAS_MPL = False
plt = None


def _init_matplotlib() -> bool:
    global HAS_MPL, plt
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt
        HAS_MPL = True
    except ImportError:
        print("[info] matplotlib not found -- skipping figures\n")
    return HAS_MPL


# ===================================================================
# Section 1: Option Pricing -- QAE vs Black-Scholes
# ===================================================================

def run_option_pricing(save_figures: bool) -> None:
    print("\n" + "=" * 70)
    print("  1. Option Pricing: QAE vs Black-Scholes")
    print("=" * 70)

    from nqpu.finance import (
        QuantumOptionPricer,
        OptionType,
        QAEMethod,
        black_scholes_call,
        black_scholes_put,
        black_scholes_delta,
    )

    # --- Basic European call pricing ---
    pricer = QuantumOptionPricer(
        spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
    )
    result = pricer.price()
    bs_call = black_scholes_call(100, 100, 0.05, 0.2, 1.0)

    print(f"\n  European Call (ATM):")
    print(f"    QAE price:         {result.price:>10.4f}")
    print(f"    Black-Scholes:     {bs_call:>10.4f}")
    print(f"    QAE delta:         {result.delta:>10.4f}")
    print(f"    CI:                [{result.confidence_interval[0]:.4f}, "
          f"{result.confidence_interval[1]:.4f}]")
    print(f"    Oracle calls:      {result.num_oracle_calls}")

    # --- Strike sweep comparison ---
    strikes = list(range(80, 125, 5))
    qae_prices = []
    bs_prices = []

    print(f"\n  {'Strike':>8} {'QAE':>10} {'BS':>10} {'Diff':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    for k in strikes:
        p = QuantumOptionPricer(
            spot=100, strike=k, rate=0.05, volatility=0.2, maturity=1.0,
            _compute_delta=False,
        )
        qae_p = p.price().price
        bs_p = black_scholes_call(100, k, 0.05, 0.2, 1.0)
        qae_prices.append(qae_p)
        bs_prices.append(bs_p)
        print(f"  {k:>8} {qae_p:>10.4f} {bs_p:>10.4f} {qae_p - bs_p:>+10.4f}")

    # --- Exotic options: Asian and Barrier ---
    print(f"\n  Exotic Option Pricing:")

    asian_pricer = QuantumOptionPricer(
        spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
        option_type=OptionType.ASIAN_CALL,
        num_paths=5000, _compute_delta=False,
    )
    asian_result = asian_pricer.price()
    print(f"    Asian Call (QAE):      {asian_result.price:>10.4f}  "
          f"(MC ref: {asian_result.analytical_price:.4f})")

    barrier_pricer = QuantumOptionPricer(
        spot=100, strike=100, rate=0.05, volatility=0.2, maturity=1.0,
        option_type=OptionType.BARRIER_UP_AND_OUT, barrier=120,
        num_paths=5000, _compute_delta=False,
    )
    barrier_result = barrier_pricer.price()
    print(f"    Barrier Up&Out (QAE):  {barrier_result.price:>10.4f}  "
          f"(MC ref: {barrier_result.analytical_price:.4f})")

    # --- Delta surface ---
    spot_range = np.arange(80, 125, 5)
    strike_range = np.arange(85, 120, 5)
    delta_surface = np.zeros((len(spot_range), len(strike_range)))

    for i, s in enumerate(spot_range):
        for j, k in enumerate(strike_range):
            delta_surface[i, j] = black_scholes_delta(
                float(s), float(k), 0.05, 0.2, 1.0, is_call=True
            )

    print(f"\n  Delta Surface (spot x strike) computed: "
          f"{delta_surface.shape[0]}x{delta_surface.shape[1]}")
    print(f"    Delta range: [{delta_surface.min():.4f}, {delta_surface.max():.4f}]")

    # --- Figure ---
    if save_figures and HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Strike sweep
        ax = axes[0]
        ax.plot(strikes, qae_prices, "o-", label="QAE", color="#2196F3")
        ax.plot(strikes, bs_prices, "s--", label="Black-Scholes", color="#FF5722")
        ax.set_xlabel("Strike Price")
        ax.set_ylabel("Option Price")
        ax.set_title("European Call: QAE vs BS")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Pricing error
        ax = axes[1]
        errors = [q - b for q, b in zip(qae_prices, bs_prices)]
        ax.bar(strikes, errors, width=3, color="#4CAF50", alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Strike Price")
        ax.set_ylabel("QAE - BS Difference")
        ax.set_title("Pricing Error by Strike")
        ax.grid(True, alpha=0.3)

        # Delta surface
        ax = axes[2]
        im = ax.imshow(
            delta_surface, aspect="auto", cmap="RdYlGn",
            extent=[strike_range[0], strike_range[-1],
                    spot_range[-1], spot_range[0]],
        )
        ax.set_xlabel("Strike")
        ax.set_ylabel("Spot")
        ax.set_title("Call Delta Surface")
        fig.colorbar(im, ax=ax, label="Delta")

        fig.tight_layout()
        fig.savefig("demos/figures/option_pricing.png", dpi=150)
        plt.close(fig)
        print("  [saved] demos/figures/option_pricing.png")


# ===================================================================
# Section 2: Portfolio Optimization -- QAOA vs Classical
# ===================================================================

def run_portfolio_optimization(save_figures: bool) -> None:
    print("\n" + "=" * 70)
    print("  2. Portfolio Optimization: QAOA vs Classical")
    print("=" * 70)

    from nqpu.finance import (
        PortfolioOptimizer,
        compute_efficient_frontier,
        classical_portfolio_optimize,
    )

    asset_names = ["Tech", "Healthcare", "Energy", "Finance", "Consumer"]
    expected_returns = np.array([0.12, 0.08, 0.10, 0.07, 0.09])
    covariance = np.array([
        [0.04,  0.006, 0.002, 0.005, 0.003],
        [0.006, 0.025, 0.004, 0.003, 0.002],
        [0.002, 0.004, 0.035, 0.002, 0.005],
        [0.005, 0.003, 0.002, 0.02,  0.004],
        [0.003, 0.002, 0.005, 0.004, 0.022],
    ])

    # --- QAOA optimization ---
    optimizer = PortfolioOptimizer(
        num_layers=2, risk_aversion=1.0, budget=3,
    )
    t0 = time.time()
    qaoa_result = optimizer.optimize(expected_returns, covariance)
    qaoa_time = time.time() - t0

    # --- Classical brute-force ---
    t0 = time.time()
    classical = classical_portfolio_optimize(
        expected_returns, covariance, risk_aversion=1.0, budget=3,
    )
    classical_time = time.time() - t0

    # --- Efficient frontier ---
    frontier = compute_efficient_frontier(expected_returns, covariance, num_points=20)

    # --- Results ---
    print(f"\n  {'':>14} {'QAOA':>12} {'Classical':>12}")
    print(f"  {'':>14} {'-'*12} {'-'*12}")
    print(f"  {'Exp. Return':>14} {qaoa_result.expected_return:>12.4f} "
          f"{classical.expected_return:>12.4f}")
    print(f"  {'Variance':>14} {qaoa_result.variance:>12.6f} "
          f"{classical.variance:>12.6f}")
    print(f"  {'Objective':>14} {qaoa_result.objective:>12.6f} "
          f"{classical.objective:>12.6f}")
    print(f"  {'Iterations':>14} {qaoa_result.iterations:>12d} "
          f"{classical.iterations:>12d}")
    print(f"  {'Time (s)':>14} {qaoa_time:>12.3f} {classical_time:>12.3f}")

    print(f"\n  Optimal Weights:")
    print(f"  {'Asset':>12} {'QAOA':>10} {'Classical':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")
    for name, wq, wc in zip(asset_names, qaoa_result.weights, classical.weights):
        print(f"  {name:>12} {wq:>10.4f} {wc:>10.4f}")

    print(f"\n  Selected assets (QAOA): "
          f"{[n for n, b in zip(asset_names, qaoa_result.best_bitstring) if b]}")
    print(f"  Selected assets (Classical): "
          f"{[n for n, b in zip(asset_names, classical.best_bitstring) if b]}")

    print(f"\n  Efficient Frontier ({len(frontier)} points):")
    print(f"  {'Return':>10} {'Std Dev':>10}")
    print(f"  {'-'*10} {'-'*10}")
    for pt in frontier[::4]:
        print(f"  {pt.target_return:>10.4f} {np.sqrt(pt.variance):>10.4f}")

    # --- Figure ---
    if save_figures and HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Efficient frontier
        ax = axes[0]
        f_std = [np.sqrt(pt.variance) for pt in frontier]
        f_ret = [pt.target_return for pt in frontier]
        ax.plot(f_std, f_ret, "b-", linewidth=2, label="Efficient Frontier")

        q_std = np.sqrt(qaoa_result.variance)
        c_std = np.sqrt(classical.variance)
        ax.plot(q_std, qaoa_result.expected_return, "r*",
                markersize=15, label=f"QAOA (obj={qaoa_result.objective:.4f})")
        ax.plot(c_std, classical.expected_return, "g^",
                markersize=12, label=f"Classical (obj={classical.objective:.4f})")

        ax.set_xlabel("Portfolio Std Dev")
        ax.set_ylabel("Expected Return")
        ax.set_title("Efficient Frontier with Optimal Portfolios")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Weight comparison
        ax = axes[1]
        x = np.arange(len(asset_names))
        width = 0.35
        ax.bar(x - width / 2, qaoa_result.weights, width,
               label="QAOA", color="#2196F3", alpha=0.8)
        ax.bar(x + width / 2, classical.weights, width,
               label="Classical", color="#FF5722", alpha=0.8)
        ax.set_xlabel("Asset")
        ax.set_ylabel("Weight")
        ax.set_title("Portfolio Weight Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(asset_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig("demos/figures/portfolio_optimization.png", dpi=150)
        plt.close(fig)
        print("  [saved] demos/figures/portfolio_optimization.png")


# ===================================================================
# Section 3: Risk Analysis -- Quantum VaR / CVaR
# ===================================================================

def run_risk_analysis(save_figures: bool) -> None:
    print("\n" + "=" * 70)
    print("  3. Risk Analysis: Quantum VaR / CVaR")
    print("=" * 70)

    from nqpu.finance import (
        RiskAnalyzer,
        RiskConfig,
        DistributionModel,
        compute_var,
        compute_cvar,
        generate_scenarios,
        quantum_var,
        quantum_cvar,
    )

    expected_returns = np.array([0.12, 0.08, 0.10, 0.07, 0.09])
    covariance = np.array([
        [0.04,  0.006, 0.002, 0.005, 0.003],
        [0.006, 0.025, 0.004, 0.003, 0.002],
        [0.002, 0.004, 0.035, 0.002, 0.005],
        [0.005, 0.003, 0.002, 0.02,  0.004],
        [0.003, 0.002, 0.005, 0.004, 0.022],
    ])
    weights = np.array([0.30, 0.20, 0.25, 0.15, 0.10])
    portfolio_value = 1_000_000

    # --- Normal distribution ---
    config_normal = RiskConfig(
        confidence_level=0.95,
        num_scenarios=10000,
        distribution=DistributionModel.NORMAL,
        num_qubits=4,
        seed=42,
    )

    analyzer_n = RiskAnalyzer(config_normal)
    classical_n = analyzer_n.classical_analyze(expected_returns, covariance, weights)
    quantum_n = analyzer_n.analyze(expected_returns, covariance, weights)

    # --- Student-t distribution ---
    config_t = RiskConfig(
        confidence_level=0.95,
        num_scenarios=10000,
        distribution=DistributionModel.STUDENT_T,
        df=5.0,
        num_qubits=4,
        seed=42,
    )

    analyzer_t = RiskAnalyzer(config_t)
    classical_t = analyzer_t.classical_analyze(expected_returns, covariance, weights)
    quantum_t = analyzer_t.analyze(expected_returns, covariance, weights)

    # --- Results ---
    print(f"\n  Portfolio value: ${portfolio_value:,.0f}")
    print(f"  Weights: {dict(zip(['Tech','HC','Energy','Fin','Cons'], weights))}")

    header = f"  {'Metric':>20} {'Normal-CL':>12} {'Normal-QAE':>12} " \
             f"{'t(5)-CL':>12} {'t(5)-QAE':>12}"
    print(f"\n{header}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    rows = [
        ("VaR (95%)",
         classical_n.var, quantum_n.var,
         classical_t.var, quantum_t.var),
        ("CVaR (95%)",
         classical_n.cvar, quantum_n.cvar,
         classical_t.cvar, quantum_t.cvar),
        ("Sharpe Ratio",
         classical_n.sharpe_ratio, quantum_n.sharpe_ratio,
         classical_t.sharpe_ratio, quantum_t.sharpe_ratio),
        ("Sortino Ratio",
         classical_n.sortino_ratio, quantum_n.sortino_ratio,
         classical_t.sortino_ratio, quantum_t.sortino_ratio),
        ("Max Drawdown",
         classical_n.max_drawdown, quantum_n.max_drawdown,
         classical_t.max_drawdown, quantum_t.max_drawdown),
    ]

    for label, ncl, nqae, tcl, tqae in rows:
        print(f"  {label:>20} {ncl:>12.6f} {nqae:>12.6f} "
              f"{tcl:>12.6f} {tqae:>12.6f}")

    # Dollar amounts
    print(f"\n  Dollar-value Risk (portfolio = ${portfolio_value:,.0f}):")
    print(f"    Normal VaR:    ${classical_n.var * portfolio_value:>12,.0f}  "
          f"(QAE: ${quantum_n.var * portfolio_value:>12,.0f})")
    print(f"    Normal CVaR:   ${classical_n.cvar * portfolio_value:>12,.0f}  "
          f"(QAE: ${quantum_n.cvar * portfolio_value:>12,.0f})")
    print(f"    Student-t VaR: ${classical_t.var * portfolio_value:>12,.0f}  "
          f"(QAE: ${quantum_t.var * portfolio_value:>12,.0f})")
    print(f"    Student-t CVaR:${classical_t.cvar * portfolio_value:>12,.0f}  "
          f"(QAE: ${quantum_t.cvar * portfolio_value:>12,.0f})")

    # --- Figure ---
    if save_figures and HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # VaR/CVaR bar comparison
        ax = axes[0]
        labels = ["VaR\n(Normal)", "CVaR\n(Normal)", "VaR\n(Student-t)", "CVaR\n(Student-t)"]
        classical_vals = [
            classical_n.var * portfolio_value,
            classical_n.cvar * portfolio_value,
            classical_t.var * portfolio_value,
            classical_t.cvar * portfolio_value,
        ]
        quantum_vals = [
            quantum_n.var * portfolio_value,
            quantum_n.cvar * portfolio_value,
            quantum_t.var * portfolio_value,
            quantum_t.cvar * portfolio_value,
        ]

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, classical_vals, width,
               label="Classical MC", color="#2196F3", alpha=0.8)
        ax.bar(x + width / 2, quantum_vals, width,
               label="QAE-Enhanced", color="#FF5722", alpha=0.8)
        ax.set_ylabel("Dollar Risk ($)")
        ax.set_title("Risk Metrics: Classical vs Quantum")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Ratio metrics comparison
        ax = axes[1]
        metric_labels = ["Sharpe", "Sortino"]
        classical_ratios = [classical_n.sharpe_ratio, classical_n.sortino_ratio]
        quantum_ratios = [quantum_n.sharpe_ratio, quantum_n.sortino_ratio]
        classical_t_ratios = [classical_t.sharpe_ratio, classical_t.sortino_ratio]
        quantum_t_ratios = [quantum_t.sharpe_ratio, quantum_t.sortino_ratio]

        x = np.arange(len(metric_labels))
        w = 0.2
        ax.bar(x - 1.5 * w, classical_ratios, w,
               label="Normal CL", color="#2196F3", alpha=0.7)
        ax.bar(x - 0.5 * w, quantum_ratios, w,
               label="Normal QAE", color="#03A9F4", alpha=0.7)
        ax.bar(x + 0.5 * w, classical_t_ratios, w,
               label="Student-t CL", color="#FF5722", alpha=0.7)
        ax.bar(x + 1.5 * w, quantum_t_ratios, w,
               label="Student-t QAE", color="#FF9800", alpha=0.7)
        ax.set_ylabel("Ratio")
        ax.set_title("Risk-Adjusted Return Ratios")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig("demos/figures/risk_analysis.png", dpi=150)
        plt.close(fig)
        print("  [saved] demos/figures/risk_analysis.png")


# ===================================================================
# Section 4: Trading Signals -- Regime Detection & Backtesting
# ===================================================================

def run_trading_signals(save_figures: bool) -> None:
    print("\n" + "=" * 70)
    print("  4. Trading: Regime Detection & Backtesting")
    print("=" * 70)

    from nqpu.trading import (
        QuantumRegimeDetector,
        QuantumSignalGenerator,
        QuantumBacktester,
        QuantumMomentum,
        MarketRegime,
        PerformanceMetrics,
    )

    # --- Synthetic price data (geometric Brownian motion) ---
    np.random.seed(42)
    n_days = 500
    dt = 1 / 252
    mu, sigma = 0.08, 0.25
    daily_returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
    prices = 100.0 * np.cumprod(1 + daily_returns)

    print(f"\n  Synthetic data: {n_days} days, mu={mu}, sigma={sigma}")
    print(f"  Price range: [{prices.min():.2f}, {prices.max():.2f}]")
    print(f"  Final price: {prices[-1]:.2f}")

    # --- Regime detection ---
    log_returns = np.diff(np.log(np.maximum(prices, 1e-12)))
    detector = QuantumRegimeDetector(n_qubits=4)
    detector.fit(log_returns, window=20)
    regimes = detector.detect_series(log_returns, window=20)

    regime_counts = {}
    for r in regimes:
        name = str(r)
        regime_counts[name] = regime_counts.get(name, 0) + 1

    print(f"\n  Regime Distribution ({len(regimes)} windows):")
    for regime, count in sorted(regime_counts.items()):
        pct = 100.0 * count / len(regimes)
        bar = "#" * int(pct / 2)
        print(f"    {regime:>10}: {count:>4} ({pct:5.1f}%) {bar}")

    # --- Signal generation ---
    sig_gen = QuantumSignalGenerator(n_qubits=4, seed=42)
    signals = sig_gen.generate(prices, window=20)

    buy_count = sum(1 for s in signals if s.label == "buy")
    sell_count = sum(1 for s in signals if s.label == "sell")
    hold_count = sum(1 for s in signals if s.label == "hold")
    avg_conf = np.mean([s.confidence for s in signals])

    print(f"\n  Signal Statistics ({len(signals)} signals):")
    print(f"    Buy:  {buy_count:>4} ({100*buy_count/len(signals):.1f}%)")
    print(f"    Sell: {sell_count:>4} ({100*sell_count/len(signals):.1f}%)")
    print(f"    Hold: {hold_count:>4} ({100*hold_count/len(signals):.1f}%)")
    print(f"    Avg confidence: {avg_conf:.3f}")

    # --- Momentum ---
    momentum = QuantumMomentum(n_levels=16, n_steps=5)
    mom_values = momentum.compute(prices, window=20)

    print(f"\n  Quantum Momentum ({len(mom_values)} values):")
    print(f"    Mean:  {mom_values.mean():>+.4f}")
    print(f"    Std:   {mom_values.std():>.4f}")
    print(f"    Range: [{mom_values.min():.4f}, {mom_values.max():.4f}]")

    # --- Backtesting ---
    backtester = QuantumBacktester(
        transaction_cost_bps=10.0, max_position=1.0, seed=42,
    )
    bt_result = backtester.run(prices, window=20)
    perf = bt_result["metrics"]

    print(f"\n  Backtest Results:")
    print(f"    Total Return:      {perf.total_return:>+.2%}")
    print(f"    Annualised Return: {perf.annualised_return:>+.2%}")
    print(f"    Sharpe Ratio:      {perf.sharpe_ratio:>.3f}")
    print(f"    Sortino Ratio:     {perf.sortino_ratio:>.3f}")
    print(f"    Max Drawdown:      {perf.max_drawdown:>.2%}")
    print(f"    Win Rate:          {perf.win_rate:>.1%}")
    print(f"    Profit Factor:     {perf.profit_factor:>.2f}")
    print(f"    Total Trades:      {perf.total_trades}")

    # --- Figure ---
    if save_figures and HAS_MPL:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Price with regime coloring
        ax = axes[0, 0]
        regime_colors = {
            "bull": "#4CAF50",
            "bear": "#F44336",
            "sideways": "#FFC107",
            "volatile": "#9C27B0",
        }
        # Plot price line
        ax.plot(prices, color="black", linewidth=0.8, alpha=0.7)
        # Overlay regime colors as background spans
        offset = len(prices) - len(regimes)
        for i in range(len(regimes) - 1):
            r_name = str(regimes[i])
            color = regime_colors.get(r_name, "#999999")
            ax.axvspan(i + offset, i + offset + 1, alpha=0.15, color=color)
        ax.set_xlabel("Day")
        ax.set_ylabel("Price")
        ax.set_title("Price Series with Quantum Regime Detection")
        # Legend
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor=c, alpha=0.4, label=n.capitalize())
            for n, c in regime_colors.items()
        ]
        ax.legend(handles=legend_patches, loc="upper left", fontsize=8)

        # Equity curve
        ax = axes[0, 1]
        equity = bt_result["equity_curve"]
        ax.plot(equity, color="#2196F3", linewidth=1.5)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
        ax.fill_between(range(len(equity)), 1.0, equity,
                        where=equity >= 1.0, alpha=0.2, color="#4CAF50")
        ax.fill_between(range(len(equity)), 1.0, equity,
                        where=equity < 1.0, alpha=0.2, color="#F44336")
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Equity (normalized)")
        ax.set_title(f"Equity Curve (Sharpe: {perf.sharpe_ratio:.2f})")
        ax.grid(True, alpha=0.3)

        # Momentum
        ax = axes[1, 0]
        ax.plot(mom_values, color="#9C27B0", linewidth=0.8, alpha=0.8)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.fill_between(range(len(mom_values)), 0, mom_values,
                        where=mom_values > 0, alpha=0.3, color="#4CAF50")
        ax.fill_between(range(len(mom_values)), 0, mom_values,
                        where=mom_values < 0, alpha=0.3, color="#F44336")
        ax.set_xlabel("Window")
        ax.set_ylabel("Momentum")
        ax.set_title("Quantum Walk Momentum")
        ax.grid(True, alpha=0.3)

        # Signal distribution
        ax = axes[1, 1]
        directions = [s.direction for s in signals]
        confidences = [s.confidence for s in signals]
        colors = ["#4CAF50" if s.label == "buy"
                  else "#F44336" if s.label == "sell"
                  else "#FFC107" for s in signals]
        ax.scatter(directions, confidences, c=colors, alpha=0.4, s=10)
        ax.set_xlabel("Signal Direction")
        ax.set_ylabel("Confidence")
        ax.set_title("Signal Distribution (Green=Buy, Red=Sell, Yellow=Hold)")
        ax.axvline(x=0.15, color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(x=-0.15, color="gray", linestyle="--", linewidth=0.5)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig("demos/figures/trading_signals.png", dpi=150)
        plt.close(fig)
        print("  [saved] demos/figures/trading_signals.png")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    os.makedirs("demos/figures", exist_ok=True)

    save_figures = _init_matplotlib()

    print("=" * 70)
    print("  nQPU Quantum Finance Demo")
    print("=" * 70)
    print("  Showcasing quantum-enhanced financial computation")
    print("  using simulated quantum circuits (no hardware required).")

    t_start = time.time()

    run_option_pricing(save_figures)
    run_portfolio_optimization(save_figures)
    run_risk_analysis(save_figures)
    run_trading_signals(save_figures)

    elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print(f"  Demo complete!  Total time: {elapsed:.1f}s")
    if save_figures:
        print("  Figures saved to demos/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
