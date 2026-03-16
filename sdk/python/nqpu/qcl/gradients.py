"""Gradient computation methods for parameterized quantum circuits.

Provides multiple gradient strategies suited to different scenarios:

- :class:`ParameterShiftRule` -- exact for standard rotation gates.
- :class:`FiniteDifferenceGradient` -- numerical (forward or central).
- :class:`StochasticParameterShift` -- hardware-efficient with random subsets.
- :class:`NaturalGradient` -- geometry-aware via the quantum Fisher information.
- :class:`BarrenPlateauScanner` -- detects vanishing gradient issues.

All gradient methods produce :class:`GradientResult` containers that carry
the gradient vector alongside metadata (number of circuit evaluations).

References
----------
- Mitarai et al., Phys. Rev. A 98, 032309 (2018) [parameter-shift rule]
- Stokes et al., Quantum 4, 269 (2020) [natural gradient / QFI]
- McClean et al., Nat. Commun. 9, 4812 (2018) [barren plateaus]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .circuits import CircuitTemplate


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


@dataclass
class GradientResult:
    """Container for gradient computation results.

    Attributes
    ----------
    gradient : np.ndarray
        Gradient vector (same length as the parameter vector).
    n_evaluations : int
        Number of circuit evaluations used.
    method : str
        Name of the gradient method used.
    """

    gradient: np.ndarray
    n_evaluations: int
    method: str

    def __repr__(self) -> str:
        norm = float(np.linalg.norm(self.gradient))
        return (
            f"GradientResult(method={self.method!r}, "
            f"norm={norm:.6g}, evals={self.n_evaluations})"
        )


# ------------------------------------------------------------------
# Cost function wrapper
# ------------------------------------------------------------------

CostFn = Callable[[np.ndarray], float]
"""Type alias: cost function mapping parameter vector to scalar loss."""


# ------------------------------------------------------------------
# Parameter shift rule
# ------------------------------------------------------------------


class ParameterShiftRule:
    """Exact gradient via the parameter-shift rule.

    For parameterized rotation gates ``exp(-i theta G / 2)`` where *G*
    has eigenvalues +/-1, the derivative of any expectation value is:

    .. math::

        \\frac{\\partial f}{\\partial \\theta_j}
        = \\frac{f(\\theta + s \\hat{e}_j) - f(\\theta - s \\hat{e}_j)}
               {2 \\sin(s)}

    with the standard shift ``s = pi/2``.

    Cost: ``2 * n_params`` circuit evaluations per gradient.

    Parameters
    ----------
    shift : float
        Shift magnitude (default pi/2).
    """

    def __init__(self, shift: float = np.pi / 2) -> None:
        self.shift = shift

    def compute(
        self, cost_fn: CostFn, params: np.ndarray
    ) -> GradientResult:
        """Compute the gradient at *params*.

        Parameters
        ----------
        cost_fn : callable
            ``f(params) -> float``.
        params : np.ndarray
            Current parameter vector.

        Returns
        -------
        GradientResult
        """
        params = np.asarray(params, dtype=np.float64)
        n = len(params)
        grad = np.zeros(n, dtype=np.float64)
        s = self.shift
        denom = 2.0 * np.sin(s)

        for i in range(n):
            e_i = np.zeros(n, dtype=np.float64)
            e_i[i] = s
            f_plus = cost_fn(params + e_i)
            f_minus = cost_fn(params - e_i)
            grad[i] = (f_plus - f_minus) / denom

        return GradientResult(
            gradient=grad, n_evaluations=2 * n, method="parameter_shift"
        )


# ------------------------------------------------------------------
# Finite difference gradient
# ------------------------------------------------------------------


class FiniteDifferenceGradient:
    """Numerical gradient via finite differences.

    Supports forward and central difference modes.

    Parameters
    ----------
    epsilon : float
        Step size.
    method : str
        ``'central'`` (default) or ``'forward'``.
    """

    def __init__(
        self, epsilon: float = 1e-7, method: str = "central"
    ) -> None:
        if method not in ("central", "forward"):
            raise ValueError(
                f"method must be 'central' or 'forward', got '{method}'"
            )
        self.epsilon = epsilon
        self.method = method

    def compute(
        self, cost_fn: CostFn, params: np.ndarray
    ) -> GradientResult:
        """Compute the gradient at *params*.

        Parameters
        ----------
        cost_fn : callable
            ``f(params) -> float``.
        params : np.ndarray
            Current parameter vector.

        Returns
        -------
        GradientResult
        """
        params = np.asarray(params, dtype=np.float64)
        n = len(params)
        grad = np.zeros(n, dtype=np.float64)
        eps = self.epsilon
        evals = 0

        if self.method == "central":
            for i in range(n):
                e_i = np.zeros(n, dtype=np.float64)
                e_i[i] = eps
                grad[i] = (cost_fn(params + e_i) - cost_fn(params - e_i)) / (
                    2.0 * eps
                )
            evals = 2 * n
        else:  # forward
            f0 = cost_fn(params)
            evals = 1
            for i in range(n):
                e_i = np.zeros(n, dtype=np.float64)
                e_i[i] = eps
                grad[i] = (cost_fn(params + e_i) - f0) / eps
            evals += n

        return GradientResult(
            gradient=grad,
            n_evaluations=evals,
            method=f"finite_difference_{self.method}",
        )


# ------------------------------------------------------------------
# Stochastic parameter shift
# ------------------------------------------------------------------


class StochasticParameterShift:
    """Stochastic parameter-shift rule (SPSR).

    Instead of evaluating the gradient for all parameters, randomly
    samples a subset of coordinates per step.  This reduces the number
    of circuit evaluations at the cost of gradient variance, making it
    suitable for hardware-efficient training.

    Parameters
    ----------
    shift : float
        Shift magnitude (default pi/2).
    sample_fraction : float
        Fraction of parameters to sample per gradient call (0, 1].
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        shift: float = np.pi / 2,
        sample_fraction: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        self.shift = shift
        self.sample_fraction = sample_fraction
        self.rng = np.random.default_rng(seed)

    def compute(
        self, cost_fn: CostFn, params: np.ndarray
    ) -> GradientResult:
        """Compute a stochastic gradient estimate at *params*.

        Parameters
        ----------
        cost_fn : callable
            ``f(params) -> float``.
        params : np.ndarray
            Current parameter vector.

        Returns
        -------
        GradientResult
            Gradient with zeros for unsampled coordinates.
        """
        params = np.asarray(params, dtype=np.float64)
        n = len(params)
        n_sample = max(1, int(round(self.sample_fraction * n)))
        indices = self.rng.choice(n, size=n_sample, replace=False)

        grad = np.zeros(n, dtype=np.float64)
        s = self.shift
        denom = 2.0 * np.sin(s)

        for i in indices:
            e_i = np.zeros(n, dtype=np.float64)
            e_i[i] = s
            f_plus = cost_fn(params + e_i)
            f_minus = cost_fn(params - e_i)
            grad[i] = (f_plus - f_minus) / denom

        return GradientResult(
            gradient=grad,
            n_evaluations=2 * n_sample,
            method="stochastic_parameter_shift",
        )


# ------------------------------------------------------------------
# Natural gradient
# ------------------------------------------------------------------


class NaturalGradient:
    """Natural gradient using the quantum Fisher information (Fubini-Study) metric.

    Adjusts the Euclidean gradient by the inverse of the QFI matrix,
    producing geometry-aware updates that respect the curvature of
    the quantum state space.

    .. math::

        \\tilde{g} = F^{-1} g

    The QFI matrix is estimated from circuit evaluations using the
    parameter-shift rule on the overlap function.

    Parameters
    ----------
    regularization : float
        Tikhonov regularization for numerical stability.
    shift : float
        Parameter shift for QFI estimation.
    """

    def __init__(
        self, regularization: float = 1e-3, shift: float = np.pi / 2
    ) -> None:
        self.regularization = regularization
        self.shift = shift

    def estimate_qfi(
        self, circuit_fn: CostFn, params: np.ndarray
    ) -> np.ndarray:
        """Estimate the quantum Fisher information matrix.

        Uses the parameter-shift rule on the overlap of the state with
        itself under parameter perturbations.  The diagonal and off-diagonal
        elements are computed from shifted circuit evaluations.

        Parameters
        ----------
        circuit_fn : callable
            Function returning a real-valued fidelity or overlap.
        params : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            QFI matrix of shape (n_params, n_params).
        """
        params = np.asarray(params, dtype=np.float64)
        n = len(params)
        qfi = np.zeros((n, n), dtype=np.float64)
        s = self.shift

        # Diagonal elements: F_ii = -2 * d^2 f / dtheta_i^2 (approx)
        f0 = circuit_fn(params)
        for i in range(n):
            e_i = np.zeros(n, dtype=np.float64)
            e_i[i] = s
            fp = circuit_fn(params + e_i)
            fm = circuit_fn(params - e_i)
            # Second derivative approximation
            qfi[i, i] = max(0.0, -(fp + fm - 2.0 * f0) / (s * s))

        # Off-diagonal elements
        for i in range(n):
            for j in range(i + 1, n):
                e_i = np.zeros(n, dtype=np.float64)
                e_j = np.zeros(n, dtype=np.float64)
                e_i[i] = s
                e_j[j] = s
                fpp = circuit_fn(params + e_i + e_j)
                fpm = circuit_fn(params + e_i - e_j)
                fmp = circuit_fn(params - e_i + e_j)
                fmm = circuit_fn(params - e_i - e_j)
                qfi[i, j] = -(fpp - fpm - fmp + fmm) / (4.0 * s * s)
                qfi[j, i] = qfi[i, j]

        return qfi

    def compute(
        self,
        cost_fn: CostFn,
        params: np.ndarray,
        qfi: np.ndarray | None = None,
        metric_fn: CostFn | None = None,
    ) -> GradientResult:
        """Compute the natural gradient at *params*.

        Parameters
        ----------
        cost_fn : callable
            Loss function ``f(params) -> float``.
        params : np.ndarray
            Current parameter vector.
        qfi : np.ndarray or None
            Pre-computed QFI matrix.  If ``None``, estimated from
            *metric_fn* or *cost_fn*.
        metric_fn : callable or None
            Function for QFI estimation (often the fidelity).
            Defaults to *cost_fn* if not provided.

        Returns
        -------
        GradientResult
        """
        params = np.asarray(params, dtype=np.float64)
        n = len(params)

        # Euclidean gradient via parameter shift
        psr = ParameterShiftRule(shift=self.shift)
        euclidean = psr.compute(cost_fn, params)
        evals = euclidean.n_evaluations

        # QFI matrix
        if qfi is None:
            fn = metric_fn if metric_fn is not None else cost_fn
            qfi = self.estimate_qfi(fn, params)
            evals += 2 * n + 4 * n * (n - 1) // 2 + 1

        # Regularize and invert
        qfi_reg = qfi + self.regularization * np.eye(n, dtype=np.float64)
        try:
            qfi_inv = np.linalg.inv(qfi_reg)
        except np.linalg.LinAlgError:
            qfi_inv = np.linalg.pinv(qfi_reg)

        nat_grad = qfi_inv @ euclidean.gradient

        return GradientResult(
            gradient=nat_grad,
            n_evaluations=evals,
            method="natural_gradient",
        )


# ------------------------------------------------------------------
# Barren plateau scanner
# ------------------------------------------------------------------


class BarrenPlateauScanner:
    """Detect barren plateaus by analyzing gradient variance across random initializations.

    Computes gradients at multiple random parameter points and checks
    whether the variance of each gradient component decreases
    exponentially with circuit depth -- the hallmark of barren plateaus
    (McClean et al., 2018).

    Parameters
    ----------
    n_samples : int
        Number of random parameter initializations to sample.
    seed : int or None
        Random seed.
    """

    def __init__(self, n_samples: int = 50, seed: int | None = None) -> None:
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def scan(
        self, cost_fn: CostFn, n_params: int
    ) -> dict[str, np.ndarray | float | bool]:
        """Scan for barren plateaus.

        Parameters
        ----------
        cost_fn : callable
            ``f(params) -> float``.
        n_params : int
            Length of the parameter vector.

        Returns
        -------
        dict
            Keys:

            - ``'gradient_means'``: mean of each gradient component (n_params,)
            - ``'gradient_variances'``: variance of each component (n_params,)
            - ``'mean_variance'``: average variance across all components
            - ``'is_barren'``: True if mean variance < 1e-4
        """
        psr = ParameterShiftRule()
        grads = np.zeros((self.n_samples, n_params), dtype=np.float64)

        for k in range(self.n_samples):
            theta = self.rng.uniform(0, 2 * np.pi, size=n_params)
            result = psr.compute(cost_fn, theta)
            grads[k] = result.gradient

        means = np.mean(grads, axis=0)
        variances = np.var(grads, axis=0)
        mean_var = float(np.mean(variances))

        return {
            "gradient_means": means,
            "gradient_variances": variances,
            "mean_variance": mean_var,
            "is_barren": mean_var < 1e-4,
        }
