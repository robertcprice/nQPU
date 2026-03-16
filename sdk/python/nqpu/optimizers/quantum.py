"""Quantum-specific optimization utilities.

Provides gradient-estimation methods tailored for parameterised quantum
circuits and a high-level :class:`VQEOptimizer` convenience class that
wires together a Hamiltonian, an ansatz, and an optimizer into a single
``optimize()`` call.

Gradient methods
----------------
- :class:`ParameterShiftGradient` -- exact for gates of the form
  ``exp(-i theta G / 2)`` where *G* has eigenvalues +/-1.
- :class:`FiniteDifferenceGradient` -- forward or central difference.
- :class:`NaturalGradient` -- Fubini-Study metric-aware descent.

Convenience
-----------
- :func:`minimize` -- one-liner that selects an optimizer by name.
- :class:`VQEOptimizer` -- full VQE pipeline in a single class.

References
----------
- Mitarai et al., Phys. Rev. A 98, 032309 (2018) [parameter-shift rule]
- Stokes et al., Quantum 4, 269 (2020) [natural gradient on QCs]
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .base import (
    Bounds,
    CostFunction,
    GradientFunction,
    Optimizer,
    OptimizerResult,
    _ensure_array,
)


# ==================================================================
# Gradient estimation helpers
# ==================================================================


class ParameterShiftGradient:
    """Exact gradient via the parameter-shift rule.

    For a parameterised circuit ``U(theta)`` composed of gates
    ``exp(-i theta_j G_j / 2)`` where each ``G_j`` has eigenvalues
    +/-1, the derivative is:

    .. math::

        \\frac{\\partial f}{\\partial \\theta_j}
        = \\frac{f(\\theta + s \\hat{e}_j) - f(\\theta - s \\hat{e}_j)}
               {2 \\sin(s)}

    with shift ``s = pi/2`` (the standard choice).

    Cost: ``2 * n_params`` function evaluations per gradient.

    Parameters
    ----------
    shift : float
        Parameter shift magnitude (default ``pi/2``).
    """

    def __init__(self, shift: float = np.pi / 2) -> None:
        self.shift = shift

    def compute_gradient(
        self, cost_fn: CostFunction, params: np.ndarray
    ) -> np.ndarray:
        """Compute the gradient at *params*.

        Parameters
        ----------
        cost_fn : callable
            ``f(params) -> float``.
        params : np.ndarray
            Current parameter vector.

        Returns
        -------
        np.ndarray
            Gradient vector of the same length as *params*.
        """
        params = _ensure_array(params)
        n = len(params)
        grad = np.zeros(n, dtype=np.float64)
        s = self.shift

        for i in range(n):
            e_i = np.zeros(n, dtype=np.float64)
            e_i[i] = s
            f_plus = cost_fn(params + e_i)
            f_minus = cost_fn(params - e_i)
            grad[i] = (f_plus - f_minus) / (2.0 * np.sin(s))

        return grad


class FiniteDifferenceGradient:
    """Finite-difference gradient estimator.

    Supports forward and central difference modes.

    Parameters
    ----------
    epsilon : float
        Step size for the difference.
    method : str
        ``'central'`` (default, more accurate) or ``'forward'``
        (cheaper but biased).
    """

    def __init__(
        self, epsilon: float = 1e-7, method: str = "central"
    ) -> None:
        if method not in ("central", "forward"):
            raise ValueError(f"method must be 'central' or 'forward', got '{method}'")
        self.epsilon = epsilon
        self.method = method

    def compute_gradient(
        self, cost_fn: CostFunction, params: np.ndarray
    ) -> np.ndarray:
        """Compute the gradient at *params*.

        Parameters
        ----------
        cost_fn : callable
            ``f(params) -> float``.
        params : np.ndarray
            Current parameter vector.

        Returns
        -------
        np.ndarray
            Gradient vector of the same length as *params*.
        """
        params = _ensure_array(params)
        n = len(params)
        grad = np.zeros(n, dtype=np.float64)
        eps = self.epsilon

        if self.method == "central":
            for i in range(n):
                e_i = np.zeros(n, dtype=np.float64)
                e_i[i] = eps
                grad[i] = (cost_fn(params + e_i) - cost_fn(params - e_i)) / (2.0 * eps)
        else:  # forward
            f0 = cost_fn(params)
            for i in range(n):
                e_i = np.zeros(n, dtype=np.float64)
                e_i[i] = eps
                grad[i] = (cost_fn(params + e_i) - f0) / eps

        return grad


class NaturalGradient:
    """Natural gradient descent using the quantum Fisher information metric.

    The natural gradient adjusts the parameter-space gradient by the
    inverse of the Fubini-Study metric tensor (quantum Fisher
    information matrix), leading to more efficient optimisation on
    quantum manifolds:

    .. math::

        \\tilde{g} = F^{-1} g

    where *F* is the QFI matrix and *g* is the Euclidean gradient.

    Parameters
    ----------
    regularization : float
        Tikhonov regularisation parameter added to the diagonal of
        the QFI matrix before inversion, for numerical stability.
    """

    def __init__(self, regularization: float = 1e-4) -> None:
        self.regularization = regularization

    def compute_gradient(
        self,
        cost_fn: CostFunction,
        params: np.ndarray,
        metric_fn: Callable[[np.ndarray], np.ndarray],
        gradient_fn: GradientFunction | None = None,
    ) -> np.ndarray:
        """Compute the natural gradient at *params*.

        Parameters
        ----------
        cost_fn : callable
            ``f(params) -> float``.
        params : np.ndarray
            Current parameter vector.
        metric_fn : callable
            ``metric_fn(params) -> F`` where *F* is the
            ``(n, n)`` quantum Fisher information matrix.
        gradient_fn : callable, optional
            Euclidean gradient.  If ``None``, computed via
            :class:`ParameterShiftGradient`.

        Returns
        -------
        np.ndarray
            Natural gradient vector.
        """
        params = _ensure_array(params)
        n = len(params)

        # Euclidean gradient
        if gradient_fn is not None:
            g = np.asarray(gradient_fn(params), dtype=np.float64)
        else:
            psg = ParameterShiftGradient()
            g = psg.compute_gradient(cost_fn, params)

        # QFI metric
        F = np.asarray(metric_fn(params), dtype=np.float64)
        assert F.shape == (n, n), f"metric_fn must return ({n}, {n}), got {F.shape}"

        # Regularised pseudo-inverse
        F_reg = F + self.regularization * np.eye(n, dtype=np.float64)
        try:
            F_inv = np.linalg.inv(F_reg)
        except np.linalg.LinAlgError:
            F_inv = np.linalg.pinv(F_reg)

        return F_inv @ g


# ==================================================================
# High-level VQE convenience class
# ==================================================================


class VQEOptimizer:
    """High-level Variational Quantum Eigensolver driver.

    Combines a Hamiltonian expectation-value function with a
    parameterised ansatz and an optimizer into a single
    :meth:`optimize` call.

    The user provides an ``ansatz_fn`` that, given a parameter vector,
    returns the expectation value of the Hamiltonian.  This decouples
    the VQEOptimizer from any specific simulator or circuit library.

    Parameters
    ----------
    optimizer : str or Optimizer
        Either a string name (``'spsa'``, ``'cobyla'``,
        ``'nelder-mead'``, ``'adam'``, ``'l-bfgs-b'``,
        ``'gradient-descent'``) or an :class:`Optimizer` instance.
    maxiter : int
        Passed to the optimizer if created from string name.
    callback : callable, optional
        ``callback(iter, params, energy)`` hook for live monitoring.
    """

    OPTIMIZER_MAP: dict[str, type] = {}  # populated at import time

    def __init__(
        self,
        optimizer: str | Optimizer = "spsa",
        maxiter: int = 200,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> None:
        self.maxiter = maxiter
        self.callback = callback

        if isinstance(optimizer, Optimizer):
            self._optimizer = optimizer
        else:
            self._optimizer = self._make_optimizer(optimizer, maxiter, callback)

    def optimize(
        self,
        ansatz_fn: CostFunction,
        initial_params: np.ndarray,
        bounds: Bounds | None = None,
    ) -> OptimizerResult:
        """Run the VQE optimization loop.

        Parameters
        ----------
        ansatz_fn : callable
            ``ansatz_fn(params) -> energy``.  This callable should
            prepare the ansatz circuit, measure the Hamiltonian
            expectation value, and return it as a float.
        initial_params : np.ndarray
            Starting variational parameters.
        bounds : sequence of (lo, hi) or None
            Per-parameter bounds.

        Returns
        -------
        OptimizerResult
            Optimization outcome with ``optimal_value`` being the
            best energy found.
        """
        return self._optimizer.minimize(ansatz_fn, initial_params, bounds=bounds)

    @staticmethod
    def _make_optimizer(
        name: str,
        maxiter: int,
        callback: Callable[[int, np.ndarray, float], None] | None,
    ) -> Optimizer:
        """Instantiate an optimizer from its string name."""
        # Lazy imports to avoid circular dependency at module level
        from .gradient_free import COBYLA, NelderMead, SPSA
        from .gradient_based import Adam, GradientDescent, LBFGSB

        mapping: dict[str, type[Optimizer]] = {
            "spsa": SPSA,
            "cobyla": COBYLA,
            "nelder-mead": NelderMead,
            "adam": Adam,
            "l-bfgs-b": LBFGSB,
            "gradient-descent": GradientDescent,
        }
        key = name.lower()
        if key not in mapping:
            raise ValueError(
                f"Unknown optimizer '{name}'. "
                f"Choose from: {sorted(mapping.keys())}"
            )
        return mapping[key](maxiter=maxiter, callback=callback)


# ==================================================================
# Top-level convenience function
# ==================================================================


def minimize(
    cost_fn: CostFunction,
    x0: np.ndarray,
    method: str = "spsa",
    maxiter: int = 200,
    bounds: Bounds | None = None,
    callback: Callable[[int, np.ndarray, float], None] | None = None,
    **kwargs,
) -> OptimizerResult:
    """One-liner optimization of a quantum cost function.

    Parameters
    ----------
    cost_fn : callable
        Scalar objective ``f(params) -> float``.
    x0 : array-like
        Starting parameter vector.
    method : str
        Optimizer name (see :class:`VQEOptimizer` for valid names).
    maxiter : int
        Maximum iterations.
    bounds : sequence of (lo, hi) or None
        Per-parameter bounds.
    callback : callable, optional
        ``callback(iter, params, value)`` hook.
    **kwargs
        Extra keyword arguments forwarded to the optimizer constructor.

    Returns
    -------
    OptimizerResult
    """
    from .gradient_free import COBYLA, NelderMead, SPSA
    from .gradient_based import Adam, GradientDescent, LBFGSB

    mapping: dict[str, type[Optimizer]] = {
        "spsa": SPSA,
        "cobyla": COBYLA,
        "nelder-mead": NelderMead,
        "adam": Adam,
        "l-bfgs-b": LBFGSB,
        "gradient-descent": GradientDescent,
    }
    key = method.lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {sorted(mapping.keys())}"
        )

    opt = mapping[key](maxiter=maxiter, callback=callback, **kwargs)
    return opt.minimize(cost_fn, _ensure_array(x0), bounds=bounds)
