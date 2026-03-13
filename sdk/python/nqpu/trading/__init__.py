"""
nQPU Trading -- Quantum-Inspired Trading Tools.

Provides generic, reusable quantum-inspired primitives for quantitative
finance.  Every module is independently useful and relies only on NumPy
(no heavy ML frameworks).

Modules
-------
quantum_volatility
    Quantum state evolution for implied-volatility surface modelling.
regime_detection
    Market regime classification via quantum state overlap and transition
    matrices modelled as quantum channels.
feature_engineering
    Quantum feature maps, kernel similarity, and entanglement-based
    correlation features for financial time series.
signal_processing
    Trading signal generation, quantum-inspired filtering, momentum and
    mean-reversion indicators.
risk_management
    Quantum amplitude estimation VaR, QAOA portfolio optimisation,
    quantum correlation, and Kelly criterion sizing.
backtesting
    Walk-forward backtesting with regime-aware performance attribution.

Example
-------
>>> from nqpu.trading import (
...     QuantumVolatilitySurface,
...     QuantumRegimeDetector,
...     QuantumFeatureMap,
...     QuantumSignalGenerator,
...     QuantumVaR,
...     QuantumBacktester,
... )
"""

from nqpu.trading.quantum_volatility import (
    QuantumStateEncoder,
    HamiltonianEvolution,
    BornRuleMeasurement,
    QuantumVolatilitySurface,
    interpolate_iv_surface,
    extrapolate_iv_surface,
)

from nqpu.trading.regime_detection import (
    MarketRegime,
    QuantumRegimeDetector,
    RegimeTransitionMatrix,
    VolatilityRegimeClassifier,
)

from nqpu.trading.feature_engineering import (
    EncodingType,
    QuantumFeatureMap,
    QuantumKernelSimilarity,
    EntanglementFeatures,
    compute_financial_features,
)

from nqpu.trading.signal_processing import (
    Signal,
    QuantumSignalGenerator,
    QuantumFilter,
    QuantumMomentum,
    QuantumMeanReversion,
    combine_signals,
)

from nqpu.trading.risk_management import (
    QuantumVaR,
    QuantumPortfolioOptimizer,
    QuantumCorrelation,
    KellyCriterion,
    DrawdownInfo,
    drawdown_analysis,
)

from nqpu.trading.backtesting import (
    QuantumBacktester,
    PerformanceMetrics,
    RegimeAwareBacktest,
    WalkForwardOptimizer,
)

__all__ = [
    # quantum_volatility
    "QuantumStateEncoder",
    "HamiltonianEvolution",
    "BornRuleMeasurement",
    "QuantumVolatilitySurface",
    "interpolate_iv_surface",
    "extrapolate_iv_surface",
    # regime_detection
    "MarketRegime",
    "QuantumRegimeDetector",
    "RegimeTransitionMatrix",
    "VolatilityRegimeClassifier",
    # feature_engineering
    "EncodingType",
    "QuantumFeatureMap",
    "QuantumKernelSimilarity",
    "EntanglementFeatures",
    "compute_financial_features",
    # signal_processing
    "Signal",
    "QuantumSignalGenerator",
    "QuantumFilter",
    "QuantumMomentum",
    "QuantumMeanReversion",
    "combine_signals",
    # risk_management
    "QuantumVaR",
    "QuantumPortfolioOptimizer",
    "QuantumCorrelation",
    "KellyCriterion",
    "DrawdownInfo",
    "drawdown_analysis",
    # backtesting
    "QuantumBacktester",
    "PerformanceMetrics",
    "RegimeAwareBacktest",
    "WalkForwardOptimizer",
]
