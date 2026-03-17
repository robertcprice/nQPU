"""nQPU Bridges -- cross-package integration for physics-driven applications.

Connects the foundational physics and simulation modules to domain-specific
applications, enabling physics-inspired algorithms for finance, trading,
chemistry, biology, games, and combinatorial optimisation.

Modules
-------
physics_finance
    Map financial correlations to quantum spin models. Use Ising Hamiltonians
    to model asset return correlations, and quantum walks for price dynamics.
physics_trading
    Hamiltonian volatility evolution and phase-transition regime detection.
    Model market regimes as quantum phases and detect transitions.
physics_games
    Map game-theoretic payoff matrices to Ising Hamiltonians for Nash
    equilibria, quantum auction modelling, and MaxCut QAOA benchmarking.
simulation_bio
    Validate biological quantum models against canonical Lindblad solvers.
    Cross-check the bio module's FMO photosynthesis against simulation.lindblad.
simulation_chem
    Open quantum chemistry via Lindblad master equation. Model decoherence
    effects on molecular systems and multiscale decomposition.
simulation_trading
    Model volatility surface decoherence via Lindblad dynamics, generate
    noisy trading signals, and apply quantum Hamiltonian momentum filtering.
vqe_noise
    Noise-aware VQE benchmarking across hardware profiles. Measures how
    hardware noise degrades molecular ground-state energy accuracy using
    Lindblad master equation dynamics.

Example:
    from nqpu.bridges import IsingCorrelationModel, QuantumWalkPricer

    # Model 5-asset correlations as an Ising spin system
    model = IsingCorrelationModel.from_covariance(cov_matrix, asset_names)
    print(model.coupling_strengths)
    print(model.critical_temperature)

    # Price an option using continuous-time quantum walk
    pricer = QuantumWalkPricer(n_steps=64, volatility=0.2)
    dist = pricer.price_distribution(spot=100, dt=1/252, n_steps=100)
"""

from .physics_finance import (
    IsingCorrelationModel,
    QuantumWalkPricer,
    HamiltonianPortfolio,
    CorrelationPhaseAnalysis,
)
from .physics_trading import (
    HamiltonianVolatility,
    PhaseTransitionRegime,
    QuantumWalkMomentum,
)
from .physics_games import (
    IsingGameSolver,
    QuantumAuctionModel,
    QuantumMaxCutBridge,
    NashResult,
    AuctionModelResult,
    MaxCutBenchmarkResult,
)
from .simulation_bio import (
    LindbladBioValidator,
    CanonicalFMO,
)
from .simulation_chem import (
    OpenQuantumChemistry,
    DecoherenceAnalysis,
)
from .simulation_trading import (
    LindbladVolatility,
    NoisySignalGenerator,
    QuantumFilteredMomentum,
    VolSurfaceResult,
    NoisySignalResult,
    FilteredMomentumResult,
)
from .vqe_noise import (
    VQENoiseBenchmark,
    VQENoiseResult,
)

__all__ = [
    # Physics-Finance
    "IsingCorrelationModel",
    "QuantumWalkPricer",
    "HamiltonianPortfolio",
    "CorrelationPhaseAnalysis",
    # Physics-Trading
    "HamiltonianVolatility",
    "PhaseTransitionRegime",
    "QuantumWalkMomentum",
    # Physics-Games
    "IsingGameSolver",
    "QuantumAuctionModel",
    "QuantumMaxCutBridge",
    "NashResult",
    "AuctionModelResult",
    "MaxCutBenchmarkResult",
    # Simulation-Bio
    "LindbladBioValidator",
    "CanonicalFMO",
    # Simulation-Chem
    "OpenQuantumChemistry",
    "DecoherenceAnalysis",
    # Simulation-Trading
    "LindbladVolatility",
    "NoisySignalGenerator",
    "QuantumFilteredMomentum",
    "VolSurfaceResult",
    "NoisySignalResult",
    "FilteredMomentumResult",
    # VQE Noise Benchmark
    "VQENoiseBenchmark",
    "VQENoiseResult",
]
