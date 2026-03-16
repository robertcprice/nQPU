"""nQPU Variational Optimizer Suite -- gradient-free, gradient-based, and quantum-specific.

Provides a batteries-included collection of optimizers purpose-built for
variational quantum algorithms (VQE, QAOA, QML).  Three families:

  - **Gradient-free**: COBYLA, Nelder-Mead, SPSA -- robust to shot noise.
  - **Gradient-based**: Adam, L-BFGS-B, GradientDescent -- for analytic
    or finite-difference gradients on ideal simulators.
  - **Quantum utilities**: parameter-shift gradients, natural gradient,
    and a high-level VQEOptimizer convenience class.

Example:
    from nqpu.optimizers import SPSA, minimize

    result = minimize(cost_fn, x0, method="spsa", maxiter=200)
    print(result.optimal_value, result.num_function_evals)
"""

from .base import Optimizer, OptimizerResult
from .gradient_free import COBYLA, NelderMead, SPSA
from .gradient_based import Adam, GradientDescent, LBFGSB
from .quantum import (
    FiniteDifferenceGradient,
    NaturalGradient,
    ParameterShiftGradient,
    VQEOptimizer,
    minimize,
)

__all__ = [
    # Base
    "Optimizer",
    "OptimizerResult",
    # Gradient-free
    "COBYLA",
    "NelderMead",
    "SPSA",
    # Gradient-based
    "Adam",
    "GradientDescent",
    "LBFGSB",
    # Quantum utilities
    "FiniteDifferenceGradient",
    "NaturalGradient",
    "ParameterShiftGradient",
    "VQEOptimizer",
    "minimize",
]
