"""nQPU Quantum Simulation -- Hamiltonian dynamics and time evolution.

Provides a complete toolkit for simulating quantum many-body systems:

1. **Hamiltonians**: Pauli-string construction, standard model Hamiltonians
   (Ising, Heisenberg, Hubbard), exact diagonalisation.

2. **Time Evolution**: Exact matrix exponential, Trotter-Suzuki product
   formulas (1st/2nd/4th order), QDrift randomised compilation, and
   adiabatic evolution with customisable schedules.

3. **Variational Dynamics**: QITE for ground-state preparation, VarQTE
   for real-time variational evolution, PVQD for projected dynamics.

4. **Observables**: Expectation values, correlation functions, entanglement
   entropy (von Neumann / Renyi), magnetisation profiles, spectral
   functions via FFT, and Loschmidt echo (fidelity).

5. **Integrators**: RK4, symplectic leapfrog, adaptive RK45, and
   Crank-Nicolson for direct Schrodinger equation integration.

6. **Lindblad Master Equation**: Open quantum system dynamics with
   standard noise channels (amplitude damping, dephasing, depolarising,
   thermal bath).

7. **Fermionic Gaussian States**: O(n^3) simulation of non-interacting
   fermionic systems with correlation matrix evolution and Wick's theorem.

8. **Multiscale Simulation**: Hybrid exact + mean-field methods for
   large systems via self-consistent subsystem decomposition.

Example:
    from nqpu.simulation import (
        ising_model, ExactEvolution, TrotterEvolution,
        Fidelity, Magnetization,
    )

    H = ising_model(4, J=1.0, h=0.5)
    exact = ExactEvolution(H)

    psi0 = np.zeros(16, dtype=complex)
    psi0[0] = 1.0
    result = exact.evolve(psi0, t_final=2.0, n_steps=50)

    mag = Magnetization(4)
    m_vs_t = mag.total_trajectory(result.times, result.states)
"""

from __future__ import annotations

# -- Hamiltonians -----------------------------------------------------------
from .hamiltonians import (
    PauliOperator,
    SparsePauliHamiltonian,
    ising_model,
    heisenberg_model,
    hubbard_model,
    random_hamiltonian,
)

# -- Time evolution ---------------------------------------------------------
from .time_evolution import (
    EvolutionResult,
    ExactEvolution,
    TrotterEvolution,
    QDrift,
    AdiabaticEvolution,
    SCHEDULE_FUNCTIONS,
)

# -- Variational dynamics ---------------------------------------------------
from .variational_dynamics import (
    VariationalAnsatz,
    VariationalResult,
    QITE,
    VarQTE,
    PVQD,
)

# -- Observables ------------------------------------------------------------
from .observables import (
    Observable,
    TimeSeriesObservable,
    CorrelationFunction,
    EntanglementEntropy,
    Magnetization,
    SpectralFunction,
    Fidelity,
)

# -- Integrators ------------------------------------------------------------
from .integrators import (
    IntegratorResult,
    RungeKutta4,
    LeapfrogIntegrator,
    AdaptiveRK45,
    CrankNicolson,
)

# -- Lindblad master equation -------------------------------------------------
from .lindblad import (
    LindbladOperator,
    LindbladMasterEquation,
    LindbladResult,
    LindbladSolver,
    amplitude_damping_operators,
    dephasing_operators,
    depolarizing_operators,
    thermal_operators,
)

# -- Fermionic Gaussian states ------------------------------------------------
from .fermionic import (
    FermionicMode,
    GaussianState,
    QuadraticHamiltonian,
    GaussianEvolution,
    GaussianEvolutionResult,
    wicks_theorem,
)

# -- Multiscale simulation ----------------------------------------------------
from .multiscale import (
    Subsystem,
    CouplingTerm,
    MultiscaleSystem,
    MultiscaleSolver,
    MultiscaleResult,
    MultiscaleEvolution,
    MultiscaleEvolutionResult,
    AdaptiveMultiscale,
)

__all__ = [
    # Hamiltonians
    "PauliOperator",
    "SparsePauliHamiltonian",
    "ising_model",
    "heisenberg_model",
    "hubbard_model",
    "random_hamiltonian",
    # Time evolution
    "EvolutionResult",
    "ExactEvolution",
    "TrotterEvolution",
    "QDrift",
    "AdiabaticEvolution",
    "SCHEDULE_FUNCTIONS",
    # Variational dynamics
    "VariationalAnsatz",
    "VariationalResult",
    "QITE",
    "VarQTE",
    "PVQD",
    # Observables
    "Observable",
    "TimeSeriesObservable",
    "CorrelationFunction",
    "EntanglementEntropy",
    "Magnetization",
    "SpectralFunction",
    "Fidelity",
    # Integrators
    "IntegratorResult",
    "RungeKutta4",
    "LeapfrogIntegrator",
    "AdaptiveRK45",
    "CrankNicolson",
    # Lindblad master equation
    "LindbladOperator",
    "LindbladMasterEquation",
    "LindbladResult",
    "LindbladSolver",
    "amplitude_damping_operators",
    "dephasing_operators",
    "depolarizing_operators",
    "thermal_operators",
    # Fermionic Gaussian states
    "FermionicMode",
    "GaussianState",
    "QuadraticHamiltonian",
    "GaussianEvolution",
    "GaussianEvolutionResult",
    "wicks_theorem",
    # Multiscale simulation
    "Subsystem",
    "CouplingTerm",
    "MultiscaleSystem",
    "MultiscaleSolver",
    "MultiscaleResult",
    "MultiscaleEvolution",
    "MultiscaleEvolutionResult",
    "AdaptiveMultiscale",
]
