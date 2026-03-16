"""nQPU Quantum Games and Decision Theory -- game theory, combinatorial
optimization, and quantum-enhanced decision making.

Provides four major areas:

- **quantum_games**: 2-player quantum games using the Eisert-Wilkens-Lewenstein
  protocol, including Prisoner's Dilemma, Battle of the Sexes, Chicken, and
  Matching Pennies with Nash equilibrium search and tournament play.
- **combinatorial**: QUBO formulations and solvers for MaxCut, Graph Coloring,
  TSP, and Number Partitioning -- the canonical targets for QAOA.
- **quantum_walks_games**: Quantum coin flipping, Meyer's PQ penny flip,
  anonymous voting, and sealed-bid auctions.
- **decision**: Quantum Bayesian inference, Markov decision processes,
  multi-armed bandits, and game-theoretic portfolio optimization.

Example
-------
>>> from nqpu.games import PrisonersDilemma, quantum_miracle_move, defect
>>> game = PrisonersDilemma()
>>> result = game.play(quantum_miracle_move(), defect())
>>> print(f"Q vs D: P1={result.payoff_p1:.2f}, P2={result.payoff_p2:.2f}")

References
----------
- Eisert, Wilkens, Lewenstein (1999) - Quantum Games and Quantum Strategies
- Meyer (1999) - Quantum Strategies
- Farhi, Goldstone, Gutmann (2014) - QAOA
- Busemeyer & Bruza (2012) - Quantum Models of Cognition and Decision
"""

from __future__ import annotations

# ---- Quantum Games ----
from .quantum_games import (
    QuantumStrategy,
    GameResult,
    QuantumGame,
    QuantumTournament,
    TournamentResult,
    PrisonersDilemma,
    BattleOfSexes,
    Chicken,
    MatchingPennies,
    cooperate,
    defect,
    hadamard_strategy,
    quantum_miracle_move,
)

# ---- Combinatorial Optimization ----
from .combinatorial import (
    Graph,
    OptimizationResult,
    MaxCut,
    GraphColoring,
    TravelingSalesman,
    NumberPartition,
)

# ---- Quantum Walk Games ----
from .quantum_walks_games import (
    CoinFlipResult,
    QuantumCoinGame,
    PennyFlipResult,
    MeyerPennyFlip,
    VotingResult,
    QuantumVoting,
    AuctionBid,
    AuctionResult,
    QuantumAuction,
)

# ---- Decision Theory ----
from .decision import (
    BayesianResult,
    QuantumBayesian,
    MDPResult,
    QuantumMarkov,
    BanditResult,
    QuantumBandit,
    PortfolioResult,
    QuantumPortfolio,
)

# ---- Multiplayer Games ----
from .multiplayer import (
    NPlayerStrategy,
    NPlayerResult,
    NPlayerEWL,
    PublicGoodsGame,
    MinorityGame,
    QuantumBargaining,
)

# ---- Evolutionary Game Theory ----
from .evolutionary import (
    QuantumPopulation,
    QuantumReplicatorDynamics,
    EvolutionResult,
    ESSAnalyzer,
    CoevolutionaryDynamics,
    CoevoResult,
)

# ---- Mechanism Design ----
from .mechanism_design import (
    QuantumValuation,
    QuantumVCG,
    AllocationResult,
    QuantumRevenueMechanism,
    AuctionOutcome,
    IncentiveAnalyzer,
)

# ---- QAOA Builder ----
from .qaoa_builder import (
    QAOACircuit,
    QAOAResult,
    QAOAOptResult,
    maxcut_qaoa,
    graph_coloring_qaoa,
    number_partition_qaoa,
    max_independent_set_qaoa,
    tsp_qaoa,
)


__all__ = [
    # Quantum Games
    "QuantumStrategy",
    "GameResult",
    "QuantumGame",
    "QuantumTournament",
    "TournamentResult",
    "PrisonersDilemma",
    "BattleOfSexes",
    "Chicken",
    "MatchingPennies",
    "cooperate",
    "defect",
    "hadamard_strategy",
    "quantum_miracle_move",
    # Combinatorial Optimization
    "Graph",
    "OptimizationResult",
    "MaxCut",
    "GraphColoring",
    "TravelingSalesman",
    "NumberPartition",
    # Quantum Walk Games
    "CoinFlipResult",
    "QuantumCoinGame",
    "PennyFlipResult",
    "MeyerPennyFlip",
    "VotingResult",
    "QuantumVoting",
    "AuctionBid",
    "AuctionResult",
    "QuantumAuction",
    # Decision Theory
    "BayesianResult",
    "QuantumBayesian",
    "MDPResult",
    "QuantumMarkov",
    "BanditResult",
    "QuantumBandit",
    "PortfolioResult",
    "QuantumPortfolio",
    # Multiplayer Games
    "NPlayerStrategy",
    "NPlayerResult",
    "NPlayerEWL",
    "PublicGoodsGame",
    "MinorityGame",
    "QuantumBargaining",
    # Evolutionary Game Theory
    "QuantumPopulation",
    "QuantumReplicatorDynamics",
    "EvolutionResult",
    "ESSAnalyzer",
    "CoevolutionaryDynamics",
    "CoevoResult",
    # Mechanism Design
    "QuantumValuation",
    "QuantumVCG",
    "AllocationResult",
    "QuantumRevenueMechanism",
    "AuctionOutcome",
    "IncentiveAnalyzer",
    # QAOA Builder
    "QAOACircuit",
    "QAOAResult",
    "QAOAOptResult",
    "maxcut_qaoa",
    "graph_coloring_qaoa",
    "number_partition_qaoa",
    "max_independent_set_qaoa",
    "tsp_qaoa",
]
