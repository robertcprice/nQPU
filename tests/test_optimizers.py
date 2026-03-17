"""Comprehensive tests for nqpu.optimizers package.

Tests cover:
  - Base classes: OptimizerResult, Optimizer abstract contract
  - Gradient-free: COBYLA, NelderMead, SPSA
  - Gradient-based: Adam, GradientDescent, LBFGSB
  - Quantum utilities: ParameterShiftGradient, FiniteDifferenceGradient,
                       NaturalGradient, VQEOptimizer, minimize
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nqpu.optimizers import (
    Adam,
    COBYLA,
    LBFGSB,
    GradientDescent,
    NelderMead,
    Optimizer,
    OptimizerResult,
    SPSA,
    FiniteDifferenceGradient,
    NaturalGradient,
    ParameterShiftGradient,
    VQEOptimizer,
    minimize,
)


# -----------------------------------------------------------------------
# Shared fixtures and helpers
# -----------------------------------------------------------------------


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function -- classic optimization test.
    Minimum at (1, 1) with value 0.
    """
    return float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)


def quadratic(x: np.ndarray) -> float:
    """Simple quadratic f(x) = sum(x_i^2). Minimum at origin."""
    return float(np.sum(x ** 2))


def quadratic_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of the quadratic: 2*x."""
    return 2.0 * np.asarray(x, dtype=np.float64)


def sinusoidal_cost(x: np.ndarray) -> float:
    """Sinusoidal cost function for testing parameter-shift gradients.
    f(x) = sum(sin(x_i)), derivative = cos(x_i).
    """
    return float(np.sum(np.sin(x)))


@pytest.fixture
def x0_2d() -> np.ndarray:
    """Starting point for 2D optimization."""
    return np.array([0.5, 0.5], dtype=np.float64)


@pytest.fixture
def x0_1d() -> np.ndarray:
    """Starting point for 1D optimization."""
    return np.array([2.0], dtype=np.float64)


# -----------------------------------------------------------------------
# Base class tests
# -----------------------------------------------------------------------


class TestOptimizerResult:
    """Tests for the OptimizerResult dataclass."""

    def test_result_stores_fields(self):
        result = OptimizerResult(
            optimal_params=np.array([1.0, 2.0]),
            optimal_value=0.5,
            num_iterations=10,
            num_function_evals=50,
        )
        assert result.optimal_value == 0.5
        assert result.num_iterations == 10
        assert result.num_function_evals == 50
        assert result.success is True  # default

    def test_repr_contains_key_info(self):
        result = OptimizerResult(
            optimal_params=np.array([0.0]),
            optimal_value=1.23,
            num_iterations=5,
            num_function_evals=20,
        )
        s = repr(result)
        assert "1.23" in s
        assert "5" in s

    def test_convergence_history_default_empty(self):
        result = OptimizerResult(
            optimal_params=np.array([0.0]),
            optimal_value=0.0,
            num_iterations=0,
            num_function_evals=0,
        )
        assert result.convergence_history == []


class TestOptimizerBase:
    """Tests for the Optimizer ABC."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Optimizer()  # type: ignore[abstract]

    def test_invalid_maxiter_raises(self):
        with pytest.raises(ValueError, match="maxiter must be positive"):
            # We need a concrete class to test; SPSA is simplest
            SPSA(maxiter=0)

    def test_negative_tol_raises(self):
        with pytest.raises(ValueError, match="tol must be non-negative"):
            SPSA(tol=-1.0)


# -----------------------------------------------------------------------
# Gradient-free optimizer tests
# -----------------------------------------------------------------------


class TestCOBYLA:
    """Tests for the COBYLA optimizer."""

    def test_name(self):
        opt = COBYLA()
        assert opt.name() == "COBYLA"

    def test_minimizes_quadratic(self, x0_2d):
        opt = COBYLA(maxiter=500, tol=1e-6)
        result = opt.minimize(quadratic, x0_2d)
        assert result.optimal_value < 0.01
        assert result.num_function_evals > 0

    def test_with_bounds(self):
        x0 = np.array([3.0], dtype=np.float64)
        opt = COBYLA(maxiter=200)
        result = opt.minimize(quadratic, x0, bounds=[(1.0, 5.0)])
        # Minimum within bounds should be at x=1
        assert result.optimal_params[0] >= 0.99

    def test_callback_is_called(self, x0_2d):
        calls = []

        def cb(iteration, params, value):
            calls.append((iteration, value))

        opt = COBYLA(maxiter=50, callback=cb)
        opt.minimize(quadratic, x0_2d)
        assert len(calls) > 0


class TestNelderMead:
    """Tests for the Nelder-Mead optimizer."""

    def test_name(self):
        opt = NelderMead()
        assert opt.name() == "Nelder-Mead"

    def test_minimizes_quadratic(self, x0_2d):
        opt = NelderMead(maxiter=500, tol=1e-8)
        result = opt.minimize(quadratic, x0_2d)
        assert result.optimal_value < 0.01

    def test_adaptive_flag(self):
        opt = NelderMead(adaptive=False)
        assert opt.adaptive is False

    def test_convergence_history_populated(self, x0_2d):
        opt = NelderMead(maxiter=100)
        result = opt.minimize(quadratic, x0_2d)
        assert len(result.convergence_history) > 0


class TestSPSA:
    """Tests for the SPSA optimizer."""

    def test_name(self):
        opt = SPSA()
        assert opt.name() == "SPSA"

    def test_minimizes_quadratic(self, x0_2d):
        opt = SPSA(maxiter=300, a=0.2, c=0.1, seed=42)
        result = opt.minimize(quadratic, x0_2d)
        # SPSA may not converge perfectly, but should improve
        assert result.optimal_value < quadratic(x0_2d)

    def test_two_evals_per_iteration(self, x0_1d):
        opt = SPSA(maxiter=10, seed=42)
        result = opt.minimize(quadratic, x0_1d)
        # Should be: 1 (initial) + 10*2 (iterations) + 1 (final) = 22
        assert result.num_function_evals >= 20

    def test_blocking_mode(self, x0_2d):
        opt = SPSA(maxiter=50, blocking=True, seed=42)
        result = opt.minimize(quadratic, x0_2d)
        assert result.success

    def test_calibration_mode(self, x0_2d):
        opt = SPSA(maxiter=50, calibrate=True, seed=42)
        result = opt.minimize(quadratic, x0_2d)
        assert result.num_function_evals > 0

    def test_with_bounds(self, x0_2d):
        opt = SPSA(maxiter=50, seed=42)
        result = opt.minimize(quadratic, x0_2d, bounds=[(-1, 1), (-1, 1)])
        assert all(-1.01 <= p <= 1.01 for p in result.optimal_params)


# -----------------------------------------------------------------------
# Gradient-based optimizer tests
# -----------------------------------------------------------------------


class TestAdam:
    """Tests for the Adam optimizer."""

    def test_name(self):
        opt = Adam()
        assert opt.name() == "Adam"

    def test_minimizes_quadratic_with_gradient(self, x0_2d):
        opt = Adam(maxiter=200, lr=0.1, gradient_fn=quadratic_gradient)
        result = opt.minimize(quadratic, x0_2d)
        assert result.optimal_value < 0.01

    def test_minimizes_without_gradient_fn(self, x0_1d):
        opt = Adam(maxiter=100, lr=0.1)
        result = opt.minimize(quadratic, x0_1d)
        assert result.optimal_value < quadratic(x0_1d)

    def test_convergence_history(self, x0_2d):
        opt = Adam(maxiter=50, lr=0.1, gradient_fn=quadratic_gradient)
        result = opt.minimize(quadratic, x0_2d)
        assert len(result.convergence_history) > 0
        # History should generally decrease
        assert result.convergence_history[-1] <= result.convergence_history[0]


class TestLBFGSB:
    """Tests for the L-BFGS-B optimizer."""

    def test_name(self):
        opt = LBFGSB()
        assert opt.name() == "L-BFGS-B"

    def test_minimizes_quadratic(self, x0_2d):
        opt = LBFGSB(maxiter=100)
        result = opt.minimize(quadratic, x0_2d)
        assert result.optimal_value < 0.01

    def test_with_bounds(self):
        x0 = np.array([3.0, 3.0], dtype=np.float64)
        opt = LBFGSB(maxiter=100)
        result = opt.minimize(quadratic, x0, bounds=[(1.0, 5.0), (1.0, 5.0)])
        assert all(p >= 0.99 for p in result.optimal_params)


class TestGradientDescent:
    """Tests for the GradientDescent optimizer."""

    def test_name(self):
        opt = GradientDescent()
        assert opt.name() == "GradientDescent"

    def test_minimizes_quadratic(self, x0_2d):
        opt = GradientDescent(
            maxiter=200, lr=0.1, gradient_fn=quadratic_gradient
        )
        result = opt.minimize(quadratic, x0_2d)
        assert result.optimal_value < 0.01

    @pytest.mark.parametrize(
        "schedule", ["constant", "step", "exponential", "cosine"]
    )
    def test_lr_schedules(self, schedule, x0_2d):
        opt = GradientDescent(
            maxiter=100,
            lr=0.1,
            schedule=schedule,
            gradient_fn=quadratic_gradient,
        )
        result = opt.minimize(quadratic, x0_2d)
        assert result.optimal_value < quadratic(x0_2d)

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError, match="schedule must be one of"):
            GradientDescent(schedule="invalid")

    def test_momentum(self, x0_2d):
        opt = GradientDescent(
            maxiter=100,
            lr=0.05,
            momentum=0.9,
            gradient_fn=quadratic_gradient,
        )
        result = opt.minimize(quadratic, x0_2d)
        assert result.optimal_value < quadratic(x0_2d)


# -----------------------------------------------------------------------
# Quantum gradient estimation tests
# -----------------------------------------------------------------------


class TestParameterShiftGradient:
    """Tests for the parameter-shift gradient estimator."""

    def test_gradient_of_sin(self):
        psg = ParameterShiftGradient()
        params = np.array([0.0], dtype=np.float64)
        grad = psg.compute_gradient(sinusoidal_cost, params)
        # d/dx sin(x)|_{x=0} = cos(0) = 1
        assert abs(grad[0] - 1.0) < 1e-4

    def test_gradient_at_pi_over_2(self):
        psg = ParameterShiftGradient()
        params = np.array([math.pi / 2], dtype=np.float64)
        grad = psg.compute_gradient(sinusoidal_cost, params)
        # d/dx sin(x)|_{x=pi/2} = cos(pi/2) = 0
        assert abs(grad[0]) < 1e-4

    def test_multidimensional(self):
        psg = ParameterShiftGradient()
        params = np.array([0.0, math.pi / 2], dtype=np.float64)
        grad = psg.compute_gradient(sinusoidal_cost, params)
        assert abs(grad[0] - 1.0) < 1e-4
        assert abs(grad[1]) < 1e-4


class TestFiniteDifferenceGradient:
    """Tests for the finite-difference gradient estimator."""

    def test_central_difference_quadratic(self):
        fdg = FiniteDifferenceGradient(method="central")
        params = np.array([1.0, 2.0], dtype=np.float64)
        grad = fdg.compute_gradient(quadratic, params)
        expected = quadratic_gradient(params)
        assert np.allclose(grad, expected, atol=1e-5)

    def test_forward_difference_quadratic(self):
        fdg = FiniteDifferenceGradient(method="forward")
        params = np.array([1.0, 2.0], dtype=np.float64)
        grad = fdg.compute_gradient(quadratic, params)
        expected = quadratic_gradient(params)
        assert np.allclose(grad, expected, atol=1e-4)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            FiniteDifferenceGradient(method="backward")


class TestNaturalGradient:
    """Tests for the natural gradient estimator."""

    def test_with_identity_metric(self):
        ng = NaturalGradient(regularization=0.0)
        params = np.array([1.0, 2.0], dtype=np.float64)
        metric_fn = lambda x: np.eye(len(x))
        nat_grad = ng.compute_gradient(
            quadratic, params, metric_fn, gradient_fn=quadratic_gradient
        )
        # With identity metric, natural gradient == Euclidean gradient
        expected = quadratic_gradient(params)
        assert np.allclose(nat_grad, expected, atol=1e-5)

    def test_with_scaled_metric(self):
        ng = NaturalGradient(regularization=0.0)
        params = np.array([1.0], dtype=np.float64)
        metric_fn = lambda x: 2.0 * np.eye(len(x))
        nat_grad = ng.compute_gradient(
            quadratic, params, metric_fn, gradient_fn=quadratic_gradient
        )
        # F^{-1} g = (1/2) * g
        expected = quadratic_gradient(params) / 2.0
        assert np.allclose(nat_grad, expected, atol=1e-5)


# -----------------------------------------------------------------------
# VQEOptimizer and minimize convenience tests
# -----------------------------------------------------------------------


class TestVQEOptimizer:
    """Tests for the VQEOptimizer high-level class."""

    def test_optimize_with_string_name(self, x0_2d):
        vqe = VQEOptimizer(optimizer="cobyla", maxiter=200)
        result = vqe.optimize(quadratic, x0_2d)
        assert isinstance(result, OptimizerResult)
        assert result.optimal_value < 0.1

    def test_optimize_with_optimizer_instance(self, x0_2d):
        opt = COBYLA(maxiter=200)
        vqe = VQEOptimizer(optimizer=opt)
        result = vqe.optimize(quadratic, x0_2d)
        assert result.optimal_value < 0.1

    def test_invalid_optimizer_name_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            VQEOptimizer(optimizer="bogus")


class TestMinimizeConvenience:
    """Tests for the top-level minimize function."""

    @pytest.mark.parametrize(
        "method",
        ["spsa", "cobyla", "nelder-mead", "adam", "l-bfgs-b", "gradient-descent"],
    )
    def test_all_methods_run(self, method, x0_2d):
        result = minimize(quadratic, x0_2d, method=method, maxiter=50)
        assert isinstance(result, OptimizerResult)
        assert result.num_function_evals > 0

    def test_invalid_method_raises(self, x0_2d):
        with pytest.raises(ValueError, match="Unknown method"):
            minimize(quadratic, x0_2d, method="bogus")

    def test_with_callback(self, x0_2d):
        calls = []

        def cb(it, params, val):
            calls.append(val)

        minimize(quadratic, x0_2d, method="spsa", maxiter=10, callback=cb, seed=42)
        assert len(calls) > 0
