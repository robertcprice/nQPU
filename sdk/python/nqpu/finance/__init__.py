"""nQPU Quantum Finance Module -- quantum algorithms for financial applications.

Provides quantum-enhanced financial computation tools:

- **amplitude_estimation**: Core QAE algorithms (Canonical, Iterative, MLAE)
  used as building blocks for all financial quantum speedups.
- **option_pricing**: European, Asian, and Barrier option pricing via QAE
  with Black-Scholes analytical validation.
- **portfolio**: Markowitz mean-variance portfolio optimization via QAOA,
  QUBO/Ising encoding, and efficient frontier computation.
- **risk_analysis**: Value at Risk (VaR) and Conditional VaR (CVaR) with
  quantum amplitude estimation speedup.

Example
-------
>>> from nqpu.finance import QuantumOptionPricer, OptionType, black_scholes_call
>>> pricer = QuantumOptionPricer(spot=100, strike=100, rate=0.05,
...                              volatility=0.2, maturity=1.0)
>>> result = pricer.price()
>>> bs = black_scholes_call(100, 100, 0.05, 0.2, 1.0)
>>> print(f"QAE: {result.price:.2f}  BS: {bs:.2f}")

References
----------
- Stamatopoulos et al. (2020), "Option Pricing using Quantum Computers"
- Woerner & Egger (2019), "Quantum Risk Analysis"
- Egger et al. (2020), "Quantum Computing for Finance"
- Barkoutsos et al. (2020), "Improving VQO using CVaR"
"""

from __future__ import annotations

# ---- Amplitude Estimation ----
from .amplitude_estimation import (
    AEResult,
    CanonicalQAE,
    IterativeQAE,
    MaxLikelihoodQAE,
    bernoulli_oracle,
    build_grover_operator,
    apply_grover_power,
)

# ---- Option Pricing ----
from .option_pricing import (
    OptionType,
    QAEMethod,
    OptionPricingResult,
    QuantumOptionPricer,
    black_scholes_call,
    black_scholes_put,
    black_scholes_delta,
    price_european_call,
    price_european_put,
)

# ---- Portfolio Optimization ----
from .portfolio import (
    PortfolioResult,
    PortfolioOptimizer,
    EfficientFrontierPoint,
    QuboMatrix,
    IsingHamiltonian,
    portfolio_to_qubo,
    qubo_to_ising,
    classical_portfolio_optimize,
    compute_efficient_frontier,
)

# ---- Risk Analysis ----
from .risk_analysis import (
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


__all__ = [
    # Amplitude Estimation
    "AEResult",
    "CanonicalQAE",
    "IterativeQAE",
    "MaxLikelihoodQAE",
    "bernoulli_oracle",
    "build_grover_operator",
    "apply_grover_power",
    # Option Pricing
    "OptionType",
    "QAEMethod",
    "OptionPricingResult",
    "QuantumOptionPricer",
    "black_scholes_call",
    "black_scholes_put",
    "black_scholes_delta",
    "price_european_call",
    "price_european_put",
    # Portfolio Optimization
    "PortfolioResult",
    "PortfolioOptimizer",
    "EfficientFrontierPoint",
    "QuboMatrix",
    "IsingHamiltonian",
    "portfolio_to_qubo",
    "qubo_to_ising",
    "classical_portfolio_optimize",
    "compute_efficient_frontier",
    # Risk Analysis
    "DistributionModel",
    "RiskConfig",
    "RiskMetrics",
    "RiskAnalyzer",
    "compute_var",
    "compute_cvar",
    "compute_max_drawdown",
    "compute_sharpe_ratio",
    "compute_sortino_ratio",
    "generate_scenarios",
    "quantum_var",
    "quantum_cvar",
]
