"""Gradient-free optimizers for noisy quantum cost functions.

These optimizers require only function evaluations (no gradient) and
are the standard choice for variational circuits executed on real
quantum hardware, where shot noise makes gradient estimation expensive
or unreliable.

Optimizers
----------
- :class:`COBYLA` -- Constrained Optimization BY Linear Approximation.
- :class:`NelderMead` -- Downhill simplex (Nelder-Mead).
- :class:`SPSA` -- Simultaneous Perturbation Stochastic Approximation.

References
----------
- Spall, IEEE Trans. Autom. Control 37, 332 (1992) [SPSA theory]
- Kandala et al., Nature 549, 242 (2017) [SPSA for VQE on hardware]
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from .base import Bounds, CostFunction, Optimizer, OptimizerResult, _ensure_array


class COBYLA(Optimizer):
    """Constrained Optimization BY Linear Approximation.

    Wraps ``scipy.optimize.minimize(method='COBYLA')`` with
    quantum-friendly defaults (larger initial step, generous tolerance).

    Parameters
    ----------
    maxiter : int
        Maximum number of function evaluations.
    rhobeg : float
        Initial step size for the trust region.  Larger values help
        escape shallow local minima typical in variational landscapes.
    tol : float
        Final accuracy on the cost function.
    callback : callable, optional
        ``callback(iter, params, value)`` hook.
    """

    def __init__(
        self,
        maxiter: int = 1000,
        rhobeg: float = 0.5,
        tol: float = 1e-6,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> None:
        super().__init__(maxiter=maxiter, tol=tol, callback=callback)
        self.rhobeg = rhobeg

    def name(self) -> str:
        return "COBYLA"

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
            n_evals_snapshot = n_evals  # noqa: F841
            history.append(val)
            iteration += 1
            if self.callback is not None:
                self.callback(iteration, x.copy(), val)

        # Convert bounds to COBYLA inequality constraints
        constraints = []
        if bounds is not None:
            for i, (lo, hi) in enumerate(bounds):
                if lo is not None:
                    constraints.append(
                        {"type": "ineq", "fun": lambda x, i=i, lo=lo: x[i] - lo}
                    )
                if hi is not None:
                    constraints.append(
                        {"type": "ineq", "fun": lambda x, i=i, hi=hi: hi - x[i]}
                    )

        result = scipy_minimize(
            wrapped,
            x0,
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": self.rhobeg},
            tol=self.tol,
            constraints=constraints if constraints else (),
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


class NelderMead(Optimizer):
    """Nelder-Mead downhill simplex optimizer.

    Wraps ``scipy.optimize.minimize(method='Nelder-Mead')``.  The
    ``adaptive`` flag enables the Gao-Han adaptive simplex variant
    which scales better for high-dimensional parameter spaces (>8
    parameters).

    Parameters
    ----------
    maxiter : int
        Maximum number of function evaluations.
    adaptive : bool
        Use the adaptive Nelder-Mead variant for high dimensions.
    tol : float
        Convergence tolerance (on the simplex diameter).
    callback : callable, optional
        ``callback(iter, params, value)`` hook.
    """

    def __init__(
        self,
        maxiter: int = 1000,
        adaptive: bool = True,
        tol: float = 1e-8,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> None:
        super().__init__(maxiter=maxiter, tol=tol, callback=callback)
        self.adaptive = adaptive

    def name(self) -> str:
        return "Nelder-Mead"

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

        # Nelder-Mead bounds support was added in scipy 1.7
        scipy_bounds = None
        if bounds is not None:
            scipy_bounds = [
                (lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
                for lo, hi in bounds
            ]

        result = scipy_minimize(
            wrapped,
            x0,
            method="Nelder-Mead",
            options={
                "maxiter": self.maxiter,
                "adaptive": self.adaptive,
                "xatol": self.tol,
                "fatol": self.tol,
            },
            bounds=scipy_bounds,
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


class SPSA(Optimizer):
    """Simultaneous Perturbation Stochastic Approximation.

    Pure-numpy implementation of the standard SPSA algorithm, the
    workhorse optimizer for VQE on real quantum hardware.  Requires
    only **two** cost-function evaluations per iteration regardless of
    parameter dimension, making it dramatically cheaper than
    finite-difference methods.

    The gradient estimate at iteration *k* is:

    .. math::

        \\hat{g}_k = \\frac{f(\\theta_k + c_k \\Delta_k)
                           - f(\\theta_k - c_k \\Delta_k)}
                          {2 c_k \\Delta_k}

    where :math:`\\Delta_k` is a Bernoulli +/-1 perturbation vector.

    Parameters
    ----------
    maxiter : int
        Number of SPSA iterations.
    a : float
        Numerator of the gain sequence ``a_k = a / (A + k + 1)^alpha``.
    c : float
        Numerator of the perturbation sequence ``c_k = c / (k + 1)^gamma``.
    A : float
        Stability constant.  Typical: 10% of *maxiter*.
    alpha : float
        Step-size exponent (theory: 0.602).
    gamma : float
        Perturbation exponent (theory: 0.101).
    blocking : bool
        If ``True``, reject parameter updates that increase the cost.
    calibrate : bool
        If ``True``, automatically calibrate *a* from an initial pilot
        gradient estimate so the first step is approximately *a* in
        magnitude (Spall's recommended practice).
    tol : float
        Not used directly (SPSA is fixed-iteration), but stored for
        interface consistency.
    seed : int or None
        RNG seed for reproducibility of perturbation vectors.
    callback : callable, optional
        ``callback(iter, params, value)`` hook.

    References
    ----------
    - Spall, IEEE Trans. Autom. Control 37, 332 (1992).
    - Kandala et al., Nature 549, 242 (2017).
    """

    def __init__(
        self,
        maxiter: int = 200,
        a: float = 0.1,
        c: float = 0.1,
        A: float | None = None,
        alpha: float = 0.602,
        gamma: float = 0.101,
        blocking: bool = False,
        calibrate: bool = False,
        tol: float = 1e-8,
        seed: int | None = None,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> None:
        super().__init__(maxiter=maxiter, tol=tol, callback=callback)
        self.a = a
        self.c = c
        self.A = A if A is not None else 0.1 * maxiter
        self.alpha = alpha
        self.gamma = gamma
        self.blocking = blocking
        self.calibrate = calibrate
        self._rng = np.random.default_rng(seed)

    def name(self) -> str:
        return "SPSA"

    def minimize(
        self,
        cost_fn: CostFunction,
        initial_params: np.ndarray,
        bounds: Bounds | None = None,
    ) -> OptimizerResult:
        theta = _ensure_array(initial_params).copy()
        n_params = len(theta)
        n_evals = 0
        history: list[float] = []

        a = self.a
        c = self.c

        # Optional calibration: rescale *a* so first step ~ target magnitude
        if self.calibrate:
            c0 = c / 1.0**self.gamma  # c_0
            delta = self._bernoulli(n_params)
            f_plus = float(cost_fn(theta + c0 * delta))
            f_minus = float(cost_fn(theta - c0 * delta))
            n_evals += 2
            g0 = (f_plus - f_minus) / (2.0 * c0 * delta)
            g0_norm = np.linalg.norm(g0)
            if g0_norm > 1e-12:
                a = a * (self.A + 1.0) ** self.alpha / g0_norm

        best_theta = theta.copy()
        best_value = float(cost_fn(theta))
        n_evals += 1
        history.append(best_value)

        for k in range(self.maxiter):
            a_k = a / (self.A + k + 1.0) ** self.alpha
            c_k = c / (k + 1.0) ** self.gamma

            # Bernoulli perturbation
            delta = self._bernoulli(n_params)

            # Two-sided gradient estimate
            theta_plus = theta + c_k * delta
            theta_minus = theta - c_k * delta
            f_plus = float(cost_fn(theta_plus))
            f_minus = float(cost_fn(theta_minus))
            n_evals += 2

            g_hat = (f_plus - f_minus) / (2.0 * c_k * delta)

            # Update
            theta_new = theta - a_k * g_hat

            # Clip to bounds if provided
            if bounds is not None:
                for i, (lo, hi) in enumerate(bounds):
                    if lo is not None:
                        theta_new[i] = max(theta_new[i], lo)
                    if hi is not None:
                        theta_new[i] = min(theta_new[i], hi)

            # Blocking: reject if cost increased
            if self.blocking:
                f_new = float(cost_fn(theta_new))
                n_evals += 1
                if f_new <= best_value:
                    theta = theta_new
                    best_value = f_new
                    best_theta = theta.copy()
                # else: keep old theta
            else:
                theta = theta_new
                # Evaluate for tracking (use the mean of plus/minus as
                # a cheap cost estimate to avoid an extra evaluation)
                current_est = 0.5 * (f_plus + f_minus)
                if current_est < best_value:
                    best_value = current_est
                    best_theta = theta.copy()

            history.append(best_value)

            if self.callback is not None:
                self.callback(k + 1, theta.copy(), best_value)

        # Final evaluation at best point
        final_value = float(cost_fn(best_theta))
        n_evals += 1

        return OptimizerResult(
            optimal_params=best_theta,
            optimal_value=final_value,
            num_iterations=self.maxiter,
            num_function_evals=n_evals,
            convergence_history=history,
            success=True,
            message=f"SPSA completed {self.maxiter} iterations",
        )

    def _bernoulli(self, n: int) -> np.ndarray:
        """Draw a Bernoulli +/-1 perturbation vector."""
        return 2.0 * self._rng.integers(0, 2, size=n).astype(np.float64) - 1.0
