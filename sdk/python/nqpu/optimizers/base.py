"""Base optimizer abstraction for variational quantum algorithms.

Defines the :class:`OptimizerResult` data container and the
:class:`Optimizer` abstract base class that all concrete optimizers
must implement.

Design principles
-----------------
- Every optimizer returns an :class:`OptimizerResult` so callers
  never need to know *which* optimizer ran.
- The ``minimize`` contract mirrors SciPy's but returns richer
  metadata (convergence history, function-evaluation count).
- Bounds are optional: gradient-free methods that support them
  will honour the constraint; others will silently ignore.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class OptimizerResult:
    """Result container for all optimizers.

    Attributes
    ----------
    optimal_params : np.ndarray
        Parameter vector at the best-found minimum.
    optimal_value : float
        Cost function value at ``optimal_params``.
    num_iterations : int
        Number of optimizer iterations (outer loops).
    num_function_evals : int
        Total number of cost-function evaluations.
    convergence_history : list[float]
        Best cost value recorded at each iteration.
    success : bool
        Whether the optimizer believes it converged.
    message : str
        Human-readable status message.
    """

    optimal_params: np.ndarray
    optimal_value: float
    num_iterations: int
    num_function_evals: int
    convergence_history: list[float] = field(default_factory=list)
    success: bool = True
    message: str = ""

    def __repr__(self) -> str:
        return (
            f"OptimizerResult("
            f"value={self.optimal_value:.6g}, "
            f"iters={self.num_iterations}, "
            f"evals={self.num_function_evals}, "
            f"success={self.success})"
        )


CostFunction = Callable[[np.ndarray], float]
"""Type alias for a scalar cost function of a parameter vector."""

GradientFunction = Callable[[np.ndarray], np.ndarray]
"""Type alias for a gradient function returning a vector."""

Bounds = Sequence[tuple[float | None, float | None]]
"""Per-parameter (lower, upper) bounds.  ``None`` means unbounded."""


class Optimizer(ABC):
    """Abstract base class for all variational optimizers.

    Subclasses must implement :meth:`minimize` and :meth:`name`.

    Parameters
    ----------
    maxiter : int
        Maximum number of optimizer iterations.
    tol : float
        Convergence tolerance on the cost-function value.
    callback : callable, optional
        Called as ``callback(iteration, params, value)`` after each
        iteration.  Useful for live monitoring / early stopping.
    """

    def __init__(
        self,
        maxiter: int = 100,
        tol: float = 1e-8,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> None:
        if maxiter < 1:
            raise ValueError("maxiter must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        self.maxiter = maxiter
        self.tol = tol
        self.callback = callback

    @abstractmethod
    def minimize(
        self,
        cost_fn: CostFunction,
        initial_params: np.ndarray,
        bounds: Bounds | None = None,
    ) -> OptimizerResult:
        """Run the optimizer.

        Parameters
        ----------
        cost_fn : callable
            Scalar objective ``f(params) -> float``.
        initial_params : np.ndarray
            Starting parameter vector.
        bounds : sequence of (low, high) or None
            Per-parameter bounds.  Not all optimizers support this.

        Returns
        -------
        OptimizerResult
            Optimization outcome.
        """

    @abstractmethod
    def name(self) -> str:
        """Return a short human-readable name for this optimizer."""

    def __repr__(self) -> str:
        return f"{self.name()}(maxiter={self.maxiter}, tol={self.tol})"


def _ensure_array(params: np.ndarray | Sequence[float]) -> np.ndarray:
    """Convert *params* to a 1-D float64 ndarray."""
    arr = np.asarray(params, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr
