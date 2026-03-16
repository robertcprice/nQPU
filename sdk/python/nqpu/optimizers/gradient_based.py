"""Gradient-based optimizers for smooth quantum cost landscapes.

Best suited for ideal simulators where analytic or high-quality
finite-difference gradients are available.  On real hardware prefer
the gradient-free optimizers in :mod:`nqpu.optimizers.gradient_free`.

Optimizers
----------
- :class:`Adam` -- Adaptive Moment Estimation (Kingma & Ba, 2014).
- :class:`LBFGSB` -- Limited-memory BFGS with box constraints.
- :class:`GradientDescent` -- SGD with momentum and learning-rate
  schedules.

References
----------
- Kingma & Ba, arXiv:1412.6980 (2014) [Adam]
- Nocedal & Wright, *Numerical Optimization*, 2nd ed. (2006) [L-BFGS-B]
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from .base import (
    Bounds,
    CostFunction,
    GradientFunction,
    Optimizer,
    OptimizerResult,
    _ensure_array,
)


# ------------------------------------------------------------------
# Finite-difference fallback (shared by Adam and GradientDescent)
# ------------------------------------------------------------------


def _finite_difference_gradient(
    cost_fn: CostFunction,
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> tuple[np.ndarray, int]:
    """Central-difference gradient estimate.

    Returns
    -------
    gradient : np.ndarray
    num_evals : int
        Number of cost-function evaluations used (2 * n_params).
    """
    n = len(params)
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        e_i = np.zeros(n, dtype=np.float64)
        e_i[i] = epsilon
        f_plus = cost_fn(params + e_i)
        f_minus = cost_fn(params - e_i)
        grad[i] = (f_plus - f_minus) / (2.0 * epsilon)
    return grad, 2 * n


# ==================================================================
# Adam
# ==================================================================


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation).

    Pure-numpy implementation of the standard Adam algorithm with
    bias-corrected first and second moment estimates.

    If ``gradient_fn`` is not provided, gradients are computed via
    central finite differences.

    Parameters
    ----------
    maxiter : int
        Number of Adam steps.
    lr : float
        Learning rate (step size).
    beta1 : float
        Exponential decay rate for first moment (mean).
    beta2 : float
        Exponential decay rate for second moment (variance).
    eps : float
        Small constant for numerical stability.
    gradient_fn : callable, optional
        ``gradient_fn(params) -> gradient_vector``.  If ``None``,
        central finite differences are used.
    tol : float
        Stop early if the gradient norm falls below this.
    callback : callable, optional
        ``callback(iter, params, value)`` hook.
    """

    def __init__(
        self,
        maxiter: int = 200,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        gradient_fn: GradientFunction | None = None,
        tol: float = 1e-8,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> None:
        super().__init__(maxiter=maxiter, tol=tol, callback=callback)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.gradient_fn = gradient_fn

    def name(self) -> str:
        return "Adam"

    def minimize(
        self,
        cost_fn: CostFunction,
        initial_params: np.ndarray,
        bounds: Bounds | None = None,
    ) -> OptimizerResult:
        theta = _ensure_array(initial_params).copy()
        n = len(theta)
        n_evals = 0

        # Moment estimates
        m = np.zeros(n, dtype=np.float64)
        v = np.zeros(n, dtype=np.float64)

        history: list[float] = []
        best_theta = theta.copy()
        best_value = float(cost_fn(theta))
        n_evals += 1
        history.append(best_value)

        for t in range(1, self.maxiter + 1):
            # Gradient
            if self.gradient_fn is not None:
                g = np.asarray(self.gradient_fn(theta), dtype=np.float64)
            else:
                g, extra_evals = _finite_difference_gradient(cost_fn, theta)
                n_evals += extra_evals

            # Update biased moments
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * g ** 2

            # Bias correction
            m_hat = m / (1.0 - self.beta1 ** t)
            v_hat = v / (1.0 - self.beta2 ** t)

            # Parameter update
            theta = theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Clip to bounds
            if bounds is not None:
                for i, (lo, hi) in enumerate(bounds):
                    if lo is not None:
                        theta[i] = max(theta[i], lo)
                    if hi is not None:
                        theta[i] = min(theta[i], hi)

            # Evaluate
            val = float(cost_fn(theta))
            n_evals += 1
            history.append(val)

            if val < best_value:
                best_value = val
                best_theta = theta.copy()

            if self.callback is not None:
                self.callback(t, theta.copy(), val)

            # Early stopping on gradient norm
            if np.linalg.norm(g) < self.tol:
                return OptimizerResult(
                    optimal_params=best_theta,
                    optimal_value=best_value,
                    num_iterations=t,
                    num_function_evals=n_evals,
                    convergence_history=history,
                    success=True,
                    message=f"Adam converged: gradient norm < {self.tol}",
                )

        return OptimizerResult(
            optimal_params=best_theta,
            optimal_value=best_value,
            num_iterations=self.maxiter,
            num_function_evals=n_evals,
            convergence_history=history,
            success=True,
            message=f"Adam completed {self.maxiter} iterations",
        )


# ==================================================================
# L-BFGS-B
# ==================================================================


class LBFGSB(Optimizer):
    """Limited-memory BFGS with box constraints.

    Wraps ``scipy.optimize.minimize(method='L-BFGS-B')``.  Excellent
    for smooth cost landscapes on ideal simulators where the gradient
    can be computed accurately.

    Parameters
    ----------
    maxiter : int
        Maximum number of L-BFGS-B iterations.
    tol : float
        Convergence tolerance (``ftol`` and ``gtol``).
    gradient_fn : callable, optional
        Analytic gradient.  If ``None``, SciPy's built-in
        finite-difference approximation is used.
    callback : callable, optional
        ``callback(iter, params, value)`` hook.
    """

    def __init__(
        self,
        maxiter: int = 200,
        tol: float = 1e-8,
        gradient_fn: GradientFunction | None = None,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> None:
        super().__init__(maxiter=maxiter, tol=tol, callback=callback)
        self.gradient_fn = gradient_fn

    def name(self) -> str:
        return "L-BFGS-B"

    def minimize(
        self,
        cost_fn: CostFunction,
        initial_params: np.ndarray,
        bounds: Bounds | None = None,
    ) -> OptimizerResult:
        x0 = _ensure_array(initial_params)
        history: list[float] = []
        n_evals = 0
        iteration = 0

        def wrapped(x: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            return float(cost_fn(x))

        def scipy_callback(x: np.ndarray) -> None:
            nonlocal iteration
            val = float(cost_fn(x))
            history.append(val)
            iteration += 1
            if self.callback is not None:
                self.callback(iteration, x.copy(), val)

        scipy_bounds = None
        if bounds is not None:
            scipy_bounds = [
                (lo, hi) for lo, hi in bounds
            ]

        result = scipy_minimize(
            wrapped,
            x0,
            method="L-BFGS-B",
            jac=self.gradient_fn,
            bounds=scipy_bounds,
            options={
                "maxiter": self.maxiter,
                "ftol": self.tol,
                "gtol": self.tol,
            },
            callback=scipy_callback,
        )

        return OptimizerResult(
            optimal_params=np.asarray(result.x, dtype=np.float64),
            optimal_value=float(result.fun),
            num_iterations=iteration,
            num_function_evals=n_evals,
            convergence_history=history,
            success=bool(result.success),
            message=str(result.message),
        )


# ==================================================================
# Gradient Descent with momentum and LR schedules
# ==================================================================


class GradientDescent(Optimizer):
    """Simple gradient descent with optional momentum and LR schedule.

    Pure-numpy implementation supporting four learning-rate schedules:

    - ``'constant'``: fixed learning rate.
    - ``'step'``: multiply LR by ``decay_rate`` every ``decay_steps``.
    - ``'exponential'``: ``lr * exp(-decay_rate * t)``.
    - ``'cosine'``: cosine annealing to zero over ``maxiter``.

    Parameters
    ----------
    maxiter : int
        Number of gradient-descent steps.
    lr : float
        Initial learning rate.
    momentum : float
        Momentum coefficient (0 = no momentum).
    schedule : str
        One of ``'constant'``, ``'step'``, ``'exponential'``,
        ``'cosine'``.
    decay_rate : float
        Decay parameter for ``'step'`` and ``'exponential'`` schedules.
    decay_steps : int
        Steps between LR reductions for the ``'step'`` schedule.
    gradient_fn : callable, optional
        Analytic gradient.  Falls back to central finite differences.
    tol : float
        Stop early if gradient norm falls below this.
    callback : callable, optional
        ``callback(iter, params, value)`` hook.
    """

    VALID_SCHEDULES = ("constant", "step", "exponential", "cosine")

    def __init__(
        self,
        maxiter: int = 200,
        lr: float = 0.01,
        momentum: float = 0.0,
        schedule: str = "constant",
        decay_rate: float = 0.5,
        decay_steps: int = 50,
        gradient_fn: GradientFunction | None = None,
        tol: float = 1e-8,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> None:
        super().__init__(maxiter=maxiter, tol=tol, callback=callback)
        if schedule not in self.VALID_SCHEDULES:
            raise ValueError(
                f"schedule must be one of {self.VALID_SCHEDULES}, got '{schedule}'"
            )
        self.lr = lr
        self.momentum = momentum
        self.schedule = schedule
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.gradient_fn = gradient_fn

    def name(self) -> str:
        return "GradientDescent"

    def _learning_rate(self, step: int) -> float:
        """Compute the learning rate at a given step."""
        if self.schedule == "constant":
            return self.lr
        if self.schedule == "step":
            factor = self.decay_rate ** (step // self.decay_steps)
            return self.lr * factor
        if self.schedule == "exponential":
            return self.lr * math.exp(-self.decay_rate * step)
        if self.schedule == "cosine":
            return self.lr * 0.5 * (1.0 + math.cos(math.pi * step / self.maxiter))
        return self.lr  # pragma: no cover

    def minimize(
        self,
        cost_fn: CostFunction,
        initial_params: np.ndarray,
        bounds: Bounds | None = None,
    ) -> OptimizerResult:
        theta = _ensure_array(initial_params).copy()
        n = len(theta)
        n_evals = 0
        velocity = np.zeros(n, dtype=np.float64)

        history: list[float] = []
        best_theta = theta.copy()
        best_value = float(cost_fn(theta))
        n_evals += 1
        history.append(best_value)

        for step in range(self.maxiter):
            # Gradient
            if self.gradient_fn is not None:
                g = np.asarray(self.gradient_fn(theta), dtype=np.float64)
            else:
                g, extra_evals = _finite_difference_gradient(cost_fn, theta)
                n_evals += extra_evals

            # Momentum update
            current_lr = self._learning_rate(step)
            velocity = self.momentum * velocity - current_lr * g
            theta = theta + velocity

            # Clip to bounds
            if bounds is not None:
                for i, (lo, hi) in enumerate(bounds):
                    if lo is not None:
                        theta[i] = max(theta[i], lo)
                    if hi is not None:
                        theta[i] = min(theta[i], hi)

            # Evaluate
            val = float(cost_fn(theta))
            n_evals += 1
            history.append(val)

            if val < best_value:
                best_value = val
                best_theta = theta.copy()

            if self.callback is not None:
                self.callback(step + 1, theta.copy(), val)

            # Early stopping
            if np.linalg.norm(g) < self.tol:
                return OptimizerResult(
                    optimal_params=best_theta,
                    optimal_value=best_value,
                    num_iterations=step + 1,
                    num_function_evals=n_evals,
                    convergence_history=history,
                    success=True,
                    message=f"GradientDescent converged at step {step + 1}",
                )

        return OptimizerResult(
            optimal_params=best_theta,
            optimal_value=best_value,
            num_iterations=self.maxiter,
            num_function_evals=n_evals,
            convergence_history=history,
            success=True,
            message=f"GradientDescent completed {self.maxiter} iterations",
        )
