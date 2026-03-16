"""Comprehensive tests for the nQPU variational optimizer suite.

Tests cover all five modules of the optimizers package:
  - base.py: OptimizerResult data container and Optimizer ABC
  - gradient_free.py: COBYLA, Nelder-Mead, SPSA
  - gradient_based.py: Adam, L-BFGS-B, GradientDescent (with schedules)
  - quantum.py: ParameterShiftGradient, FiniteDifferenceGradient,
                NaturalGradient, VQEOptimizer, minimize convenience fn

All numerical tests use fixed random seeds and generous convergence
tolerances appropriate for stochastic optimization algorithms.
"""

import math

import numpy as np
import pytest

from nqpu.optimizers import (
    COBYLA,
    LBFGSB,
    SPSA,
    Adam,
    FiniteDifferenceGradient,
    GradientDescent,
    NaturalGradient,
    NelderMead,
    Optimizer,
    OptimizerResult,
    ParameterShiftGradient,
    VQEOptimizer,
    minimize,
)


# ======================================================================
# Test objective functions
# ======================================================================


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function: minimum at (1, 1) with value 0."""
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)


def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of the Rosenbrock function."""
    dx0 = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
    dx1 = 200.0 * (x[1] - x[0] ** 2)
    return np.array([dx0, dx1], dtype=np.float64)


def quadratic(x: np.ndarray) -> float:
    """Simple quadratic: f(x) = sum(x_i^2), minimum at origin."""
    return float(np.sum(x ** 2))


def quadratic_gradient(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of the quadratic: 2*x."""
    return 2.0 * np.asarray(x, dtype=np.float64)


def noisy_quadratic(x: np.ndarray, rng: np.random.Generator) -> float:
    """Quadratic with additive Gaussian noise (simulates shot noise)."""
    return float(np.sum(x ** 2) + 0.05 * rng.standard_normal())


def sin_cos(x: np.ndarray) -> float:
    """f(x,y) = sin(x) * cos(y).  Has known analytic gradient."""
    return float(np.sin(x[0]) * np.cos(x[1]))


def sin_cos_gradient(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of sin(x)*cos(y)."""
    return np.array(
        [np.cos(x[0]) * np.cos(x[1]), -np.sin(x[0]) * np.sin(x[1])],
        dtype=np.float64,
    )


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def rosenbrock_x0() -> np.ndarray:
    """Standard Rosenbrock starting point."""
    return np.array([-1.0, 1.0])


@pytest.fixture
def quadratic_x0() -> np.ndarray:
    """Starting point for quadratic tests."""
    return np.array([3.0, -4.0, 2.0])


# ======================================================================
# 1. Gradient-Free Optimizers
# ======================================================================


class TestCOBYLA:
    """Tests for the COBYLA optimizer."""

    def test_cobyla_rosenbrock_converges(self, rosenbrock_x0: np.ndarray) -> None:
        """COBYLA should converge near the Rosenbrock minimum (1, 1)."""
        opt = COBYLA(maxiter=5000, tol=1e-6)
        result = opt.minimize(rosenbrock, rosenbrock_x0)

        assert result.optimal_value < 0.1
        np.testing.assert_allclose(result.optimal_params, [1.0, 1.0], atol=0.2)
        assert result.num_function_evals > 0

    def test_cobyla_name(self) -> None:
        opt = COBYLA()
        assert opt.name() == "COBYLA"
        assert "COBYLA" in repr(opt)

    def test_cobyla_with_bounds(self) -> None:
        """COBYLA should respect inequality constraints derived from bounds."""
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        opt = COBYLA(maxiter=2000, tol=1e-6)
        result = opt.minimize(rosenbrock, np.array([0.5, 0.5]), bounds=bounds)

        for i, (lo, hi) in enumerate(bounds):
            assert result.optimal_params[i] >= lo - 1e-6
            assert result.optimal_params[i] <= hi + 1e-6

    def test_cobyla_callback_fires(self) -> None:
        """Verify the callback is called during optimization."""
        call_log = []

        def cb(iteration, params, value):
            call_log.append((iteration, value))

        opt = COBYLA(maxiter=100, callback=cb)
        opt.minimize(quadratic, np.array([1.0, 1.0]))

        assert len(call_log) > 0
        for it, val in call_log:
            assert isinstance(it, int)
            assert isinstance(val, float)

    def test_cobyla_custom_rhobeg(self) -> None:
        """Different rhobeg values should still converge on a simple problem."""
        for rhobeg in [0.1, 1.0, 2.0]:
            opt = COBYLA(maxiter=2000, rhobeg=rhobeg)
            result = opt.minimize(quadratic, np.array([2.0, 3.0]))
            assert result.optimal_value < 0.1


class TestNelderMead:
    """Tests for the Nelder-Mead optimizer."""

    def test_nelder_mead_rosenbrock_converges(self, rosenbrock_x0: np.ndarray) -> None:
        """Nelder-Mead should converge near the Rosenbrock minimum (1, 1)."""
        opt = NelderMead(maxiter=5000, tol=1e-10)
        result = opt.minimize(rosenbrock, rosenbrock_x0)

        assert result.optimal_value < 0.01
        np.testing.assert_allclose(result.optimal_params, [1.0, 1.0], atol=0.1)
        assert result.num_function_evals > 0

    def test_nelder_mead_name(self) -> None:
        opt = NelderMead()
        assert opt.name() == "Nelder-Mead"

    def test_nelder_mead_adaptive_flag(self) -> None:
        """The adaptive flag should be stored and work without errors."""
        opt_adaptive = NelderMead(adaptive=True)
        opt_standard = NelderMead(adaptive=False)

        assert opt_adaptive.adaptive is True
        assert opt_standard.adaptive is False

        # Both should produce valid results
        x0 = np.array([2.0, 2.0])
        r1 = opt_adaptive.minimize(quadratic, x0)
        r2 = opt_standard.minimize(quadratic, x0)
        assert r1.optimal_value < 1.0
        assert r2.optimal_value < 1.0

    def test_nelder_mead_with_bounds(self) -> None:
        """Nelder-Mead should respect bounds (scipy >= 1.7)."""
        bounds = [(0.5, 5.0), (0.5, 5.0)]
        opt = NelderMead(maxiter=3000)
        result = opt.minimize(quadratic, np.array([3.0, 3.0]), bounds=bounds)

        # Minimum of x^2 on [0.5, 5]^2 is at (0.5, 0.5) = 0.5
        np.testing.assert_allclose(result.optimal_params, [0.5, 0.5], atol=0.05)


class TestSPSA:
    """Tests for the SPSA optimizer."""

    def test_spsa_noisy_quadratic_converges(self) -> None:
        """SPSA should converge on a noisy quadratic function."""
        rng = np.random.default_rng(42)

        def noisy_fn(x):
            return noisy_quadratic(x, rng)

        opt = SPSA(maxiter=500, a=0.5, c=0.2, seed=123)
        result = opt.minimize(noisy_fn, np.array([3.0, -4.0]))

        # SPSA won't be super precise with noise, but should get close
        assert result.optimal_value < 5.0
        assert np.linalg.norm(result.optimal_params) < 3.0
        assert result.num_function_evals > 0

    def test_spsa_deterministic_quadratic(self) -> None:
        """SPSA on a noise-free quadratic should converge well."""
        opt = SPSA(maxiter=300, a=0.3, c=0.1, seed=42)
        result = opt.minimize(quadratic, np.array([5.0, -3.0]))

        assert result.optimal_value < 1.0
        assert result.num_iterations == 300

    def test_spsa_name(self) -> None:
        opt = SPSA()
        assert opt.name() == "SPSA"

    def test_spsa_different_step_sizes(self) -> None:
        """Different SPSA step sizes (a, c) lead to different convergence."""
        results = []
        for a, c in [(0.05, 0.05), (0.3, 0.1), (1.0, 0.5)]:
            opt = SPSA(maxiter=200, a=a, c=c, seed=7)
            result = opt.minimize(quadratic, np.array([3.0, 3.0]))
            results.append(result.optimal_value)

        # All should make some progress from the initial f(3,3)=18
        for val in results:
            assert val < 18.0

    def test_spsa_with_blocking(self) -> None:
        """SPSA with blocking should reject cost-increasing updates."""
        opt = SPSA(maxiter=200, a=0.2, c=0.1, blocking=True, seed=99)
        result = opt.minimize(quadratic, np.array([4.0, -2.0]))

        # Blocking should give monotonically non-increasing history
        for i in range(1, len(result.convergence_history)):
            assert result.convergence_history[i] <= result.convergence_history[i - 1] + 1e-12

    def test_spsa_with_calibration(self) -> None:
        """SPSA with automatic calibration should still converge."""
        opt = SPSA(maxiter=200, a=0.1, c=0.1, calibrate=True, seed=55)
        result = opt.minimize(quadratic, np.array([3.0, -3.0]))

        assert result.optimal_value < 5.0
        # Calibration uses 2 extra evals + 1 initial
        assert result.num_function_evals >= 200 * 2 + 3

    def test_spsa_with_bounds(self) -> None:
        """SPSA should clip parameters to bounds at each step."""
        bounds = [(1.0, 10.0), (1.0, 10.0)]
        opt = SPSA(maxiter=100, a=0.2, c=0.1, seed=42)
        result = opt.minimize(quadratic, np.array([5.0, 5.0]), bounds=bounds)

        # Minimum on [1,10]^2 is at (1, 1) = 2.0
        for i in range(2):
            assert result.optimal_params[i] >= 1.0 - 1e-10
            assert result.optimal_params[i] <= 10.0 + 1e-10

    def test_spsa_reproducibility(self) -> None:
        """Same seed should give identical results."""
        x0 = np.array([2.0, -1.0])
        r1 = SPSA(maxiter=50, seed=42).minimize(quadratic, x0)
        r2 = SPSA(maxiter=50, seed=42).minimize(quadratic, x0)

        np.testing.assert_array_equal(r1.optimal_params, r2.optimal_params)
        assert r1.optimal_value == r2.optimal_value


# ======================================================================
# 2. Gradient-Based Optimizers
# ======================================================================


class TestAdam:
    """Tests for the Adam optimizer."""

    def test_adam_quadratic_converges(self, quadratic_x0: np.ndarray) -> None:
        """Adam should converge to the origin on a quadratic."""
        opt = Adam(maxiter=500, lr=0.1)
        result = opt.minimize(quadratic, quadratic_x0)

        assert result.optimal_value < 0.1
        np.testing.assert_allclose(result.optimal_params, [0.0, 0.0, 0.0], atol=0.5)

    def test_adam_default_hyperparams(self) -> None:
        """Adam with default hyperparameters should work correctly."""
        opt = Adam()
        assert opt.lr == 0.01
        assert opt.beta1 == 0.9
        assert opt.beta2 == 0.999
        assert opt.eps == 1e-8
        assert opt.maxiter == 200
        assert opt.name() == "Adam"

    def test_adam_custom_hyperparams(self) -> None:
        """Adam with custom hyperparameters should converge on quadratic."""
        opt = Adam(maxiter=300, lr=0.05, beta1=0.8, beta2=0.99, eps=1e-6)
        result = opt.minimize(quadratic, np.array([2.0, -3.0]))

        assert result.optimal_value < 0.5
        assert result.num_function_evals > 0

    def test_adam_with_analytic_gradient(self) -> None:
        """Adam should converge faster with an analytic gradient."""
        opt = Adam(maxiter=300, lr=0.05, gradient_fn=quadratic_gradient)
        result = opt.minimize(quadratic, np.array([5.0, -5.0]))

        assert result.optimal_value < 0.1

    def test_adam_convergence_history_length(self) -> None:
        """Convergence history should have one entry per iteration plus initial."""
        opt = Adam(maxiter=50, lr=0.1)
        result = opt.minimize(quadratic, np.array([1.0, 1.0]))

        # History: 1 initial + up to maxiter entries
        assert len(result.convergence_history) >= 2
        assert len(result.convergence_history) <= 51  # 1 initial + 50 iters

    def test_adam_early_stopping(self) -> None:
        """Adam should stop early when gradient norm is below tolerance."""
        opt = Adam(maxiter=10000, lr=0.1, tol=0.1, gradient_fn=quadratic_gradient)
        result = opt.minimize(quadratic, np.array([1.0, 1.0]))

        # Should converge in many fewer than 10000 iterations
        assert result.num_iterations < 10000
        assert "converged" in result.message.lower()


class TestLBFGSB:
    """Tests for the L-BFGS-B optimizer."""

    def test_lbfgsb_rosenbrock(self, rosenbrock_x0: np.ndarray) -> None:
        """L-BFGS-B should converge accurately on the Rosenbrock function."""
        opt = LBFGSB(maxiter=500, tol=1e-12)
        result = opt.minimize(rosenbrock, rosenbrock_x0)

        assert result.optimal_value < 1e-6
        np.testing.assert_allclose(result.optimal_params, [1.0, 1.0], atol=0.01)

    def test_lbfgsb_name(self) -> None:
        opt = LBFGSB()
        assert opt.name() == "L-BFGS-B"

    def test_lbfgsb_with_bounds(self) -> None:
        """L-BFGS-B should respect box constraints."""
        bounds = [(-0.5, 2.0), (-0.5, 2.0)]
        opt = LBFGSB(maxiter=500, tol=1e-10)
        result = opt.minimize(rosenbrock, np.array([0.0, 0.0]), bounds=bounds)

        for i, (lo, hi) in enumerate(bounds):
            assert result.optimal_params[i] >= lo - 1e-10
            assert result.optimal_params[i] <= hi + 1e-10

    def test_lbfgsb_with_analytic_gradient(self) -> None:
        """L-BFGS-B with analytic gradient should converge precisely."""
        opt = LBFGSB(maxiter=200, gradient_fn=rosenbrock_gradient)
        result = opt.minimize(rosenbrock, np.array([0.0, 0.0]))

        assert result.optimal_value < 1e-8
        np.testing.assert_allclose(result.optimal_params, [1.0, 1.0], atol=1e-3)

    def test_lbfgsb_tight_bounds_constrained_minimum(self) -> None:
        """When bounds exclude the true minimum, L-BFGS-B finds constrained min."""
        # Minimize x^2+y^2 but constrain x >= 2, y >= 3
        bounds = [(2.0, 10.0), (3.0, 10.0)]
        opt = LBFGSB(maxiter=200)
        result = opt.minimize(quadratic, np.array([5.0, 5.0]), bounds=bounds)

        np.testing.assert_allclose(result.optimal_params, [2.0, 3.0], atol=0.01)
        np.testing.assert_allclose(result.optimal_value, 13.0, atol=0.1)


class TestGradientDescent:
    """Tests for the GradientDescent optimizer with schedules and momentum."""

    def test_gd_constant_schedule(self) -> None:
        """GradientDescent with constant LR should converge on quadratic."""
        opt = GradientDescent(
            maxiter=500, lr=0.1, schedule="constant",
            gradient_fn=quadratic_gradient,
        )
        result = opt.minimize(quadratic, np.array([3.0, -4.0]))

        assert result.optimal_value < 0.01

    def test_gd_step_decay_schedule(self) -> None:
        """Step decay schedule should reduce LR at defined intervals."""
        opt = GradientDescent(
            maxiter=500, lr=0.1, schedule="step",
            decay_rate=0.5, decay_steps=100,
            gradient_fn=quadratic_gradient,
        )
        result = opt.minimize(quadratic, np.array([3.0, -4.0]))

        assert result.optimal_value < 0.1

        # Check that the LR schedule method works correctly
        assert opt._learning_rate(0) == 0.1     # before first decay
        assert opt._learning_rate(100) == 0.05   # after first decay
        assert opt._learning_rate(200) == 0.025  # after second decay

    def test_gd_cosine_schedule(self) -> None:
        """Cosine annealing schedule should converge on quadratic."""
        opt = GradientDescent(
            maxiter=500, lr=0.2, schedule="cosine",
            gradient_fn=quadratic_gradient,
        )
        result = opt.minimize(quadratic, np.array([3.0, -4.0]))

        assert result.optimal_value < 0.1

        # Cosine LR at step 0 should be full LR, at maxiter should approach 0
        lr_start = opt._learning_rate(0)
        lr_end = opt._learning_rate(opt.maxiter)
        assert abs(lr_start - 0.2) < 1e-10
        assert lr_end < 0.01

    def test_gd_exponential_schedule(self) -> None:
        """Exponential decay schedule should converge on quadratic."""
        opt = GradientDescent(
            maxiter=500, lr=0.2, schedule="exponential",
            decay_rate=0.01, gradient_fn=quadratic_gradient,
        )
        result = opt.minimize(quadratic, np.array([3.0, -4.0]))

        assert result.optimal_value < 1.0

    def test_gd_with_momentum(self) -> None:
        """Momentum should accelerate convergence vs. vanilla GD."""
        x0 = np.array([5.0, -5.0])

        # Without momentum
        opt_no_mom = GradientDescent(
            maxiter=200, lr=0.05, momentum=0.0,
            gradient_fn=quadratic_gradient,
        )
        r_no_mom = opt_no_mom.minimize(quadratic, x0)

        # With momentum
        opt_mom = GradientDescent(
            maxiter=200, lr=0.05, momentum=0.9,
            gradient_fn=quadratic_gradient,
        )
        r_mom = opt_mom.minimize(quadratic, x0)

        # Both should converge, but momentum version should do at least as well
        assert r_no_mom.optimal_value < 1.0
        assert r_mom.optimal_value < 1.0

    def test_gd_invalid_schedule_raises(self) -> None:
        """Passing an invalid schedule name should raise ValueError."""
        with pytest.raises(ValueError, match="schedule must be one of"):
            GradientDescent(schedule="invalid_schedule")

    def test_gd_name(self) -> None:
        opt = GradientDescent()
        assert opt.name() == "GradientDescent"

    def test_gd_early_stopping(self) -> None:
        """GradientDescent should stop early when gradient norm < tol."""
        opt = GradientDescent(
            maxiter=10000, lr=0.1, tol=0.5,
            gradient_fn=quadratic_gradient,
        )
        result = opt.minimize(quadratic, np.array([1.0, 1.0]))

        assert result.num_iterations < 10000
        assert "converged" in result.message.lower()


# ======================================================================
# 3. Quantum-Specific Gradient Tools
# ======================================================================


class TestParameterShiftGradient:
    """Tests for the ParameterShiftGradient estimator."""

    def test_accuracy_sin_cos(self) -> None:
        """Parameter-shift gradient should match the analytic gradient of sin(x)*cos(y)."""
        psg = ParameterShiftGradient()
        params = np.array([0.5, 1.2])

        computed = psg.compute_gradient(sin_cos, params)
        expected = sin_cos_gradient(params)

        np.testing.assert_allclose(computed, expected, atol=1e-10)

    def test_accuracy_at_multiple_points(self) -> None:
        """Check parameter-shift accuracy at several evaluation points."""
        psg = ParameterShiftGradient()

        test_points = [
            np.array([0.0, 0.0]),
            np.array([np.pi / 4, np.pi / 3]),
            np.array([1.5, -0.8]),
            np.array([-2.0, 3.0]),
        ]

        for params in test_points:
            computed = psg.compute_gradient(sin_cos, params)
            expected = sin_cos_gradient(params)
            np.testing.assert_allclose(computed, expected, atol=1e-10)

    def test_custom_shift(self) -> None:
        """A custom shift should still produce a valid gradient for sin-based functions."""
        # The parameter-shift rule with shift s gives exact results for
        # f(theta) = A*sin(theta + phi) + C
        # because d/dtheta = (f(theta+s) - f(theta-s))/(2*sin(s))
        psg = ParameterShiftGradient(shift=np.pi / 4)
        params = np.array([0.7, 1.1])

        computed = psg.compute_gradient(sin_cos, params)
        expected = sin_cos_gradient(params)

        np.testing.assert_allclose(computed, expected, atol=1e-10)

    def test_default_shift(self) -> None:
        """Default shift should be pi/2."""
        psg = ParameterShiftGradient()
        assert abs(psg.shift - np.pi / 2) < 1e-15


class TestFiniteDifferenceGradient:
    """Tests for the FiniteDifferenceGradient estimator."""

    def test_central_difference_accuracy(self) -> None:
        """Central difference should approximate the gradient of sin(x)*cos(y) well."""
        fdg = FiniteDifferenceGradient(epsilon=1e-7, method="central")
        params = np.array([0.5, 1.2])

        computed = fdg.compute_gradient(sin_cos, params)
        expected = sin_cos_gradient(params)

        np.testing.assert_allclose(computed, expected, atol=1e-6)

    def test_forward_difference_accuracy(self) -> None:
        """Forward difference should approximate the gradient (less accurately)."""
        fdg = FiniteDifferenceGradient(epsilon=1e-7, method="forward")
        params = np.array([0.5, 1.2])

        computed = fdg.compute_gradient(sin_cos, params)
        expected = sin_cos_gradient(params)

        # Forward difference is O(epsilon), so less accurate than central O(epsilon^2)
        np.testing.assert_allclose(computed, expected, atol=1e-5)

    def test_central_more_accurate_than_forward(self) -> None:
        """Central difference should be more accurate than forward difference."""
        params = np.array([1.0, 0.5])
        expected = sin_cos_gradient(params)

        central = FiniteDifferenceGradient(epsilon=1e-5, method="central")
        forward = FiniteDifferenceGradient(epsilon=1e-5, method="forward")

        err_central = np.linalg.norm(central.compute_gradient(sin_cos, params) - expected)
        err_forward = np.linalg.norm(forward.compute_gradient(sin_cos, params) - expected)

        assert err_central < err_forward

    def test_invalid_method_raises(self) -> None:
        """Invalid method string should raise ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            FiniteDifferenceGradient(method="backward")

    def test_quadratic_gradient_accuracy(self) -> None:
        """Finite difference on the quadratic should match 2*x closely."""
        fdg = FiniteDifferenceGradient(epsilon=1e-7, method="central")
        params = np.array([3.0, -4.0, 2.0])

        computed = fdg.compute_gradient(quadratic, params)
        expected = quadratic_gradient(params)

        np.testing.assert_allclose(computed, expected, atol=1e-5)


class TestNaturalGradient:
    """Tests for the NaturalGradient descent utility."""

    def test_identity_metric_equals_regular_gradient(self) -> None:
        """With an identity metric, natural gradient = regular gradient."""
        ng = NaturalGradient(regularization=0.0)

        params = np.array([1.0, 2.0])

        def identity_metric(x):
            return np.eye(len(x))

        nat_grad = ng.compute_gradient(
            sin_cos, params,
            metric_fn=identity_metric,
            gradient_fn=sin_cos_gradient,
        )
        regular_grad = sin_cos_gradient(params)

        np.testing.assert_allclose(nat_grad, regular_grad, atol=1e-10)

    def test_scaled_metric(self) -> None:
        """With a scaled identity metric, natural gradient = gradient / scale."""
        ng = NaturalGradient(regularization=0.0)
        params = np.array([1.0, 2.0])
        scale = 4.0

        def scaled_metric(x):
            return scale * np.eye(len(x))

        nat_grad = ng.compute_gradient(
            sin_cos, params,
            metric_fn=scaled_metric,
            gradient_fn=sin_cos_gradient,
        )
        expected = sin_cos_gradient(params) / scale

        np.testing.assert_allclose(nat_grad, expected, atol=1e-10)

    def test_regularization_stabilizes(self) -> None:
        """Regularization should prevent blow-up from near-singular metrics."""
        ng = NaturalGradient(regularization=1e-2)
        params = np.array([1.0, 2.0])

        def near_singular_metric(x):
            # Rank-1 matrix (singular)
            v = np.array([1.0, 0.0])
            return np.outer(v, v)

        # Should not raise, thanks to regularization
        nat_grad = ng.compute_gradient(
            sin_cos, params,
            metric_fn=near_singular_metric,
            gradient_fn=sin_cos_gradient,
        )
        assert np.all(np.isfinite(nat_grad))

    def test_without_gradient_fn_uses_parameter_shift(self) -> None:
        """When gradient_fn is None, should fall back to ParameterShiftGradient."""
        ng = NaturalGradient(regularization=0.0)
        params = np.array([0.5, 1.2])

        def identity_metric(x):
            return np.eye(len(x))

        # Without explicit gradient_fn (uses ParameterShiftGradient internally)
        nat_grad = ng.compute_gradient(
            sin_cos, params,
            metric_fn=identity_metric,
            gradient_fn=None,
        )
        expected = sin_cos_gradient(params)

        np.testing.assert_allclose(nat_grad, expected, atol=1e-10)


# ======================================================================
# 4. OptimizerResult
# ======================================================================


class TestOptimizerResult:
    """Tests for the OptimizerResult dataclass."""

    def test_all_fields_populated(self) -> None:
        """Run an optimizer and verify every field in the result."""
        opt = Adam(maxiter=50, lr=0.1)
        result = opt.minimize(quadratic, np.array([2.0, -1.0]))

        assert isinstance(result.optimal_params, np.ndarray)
        assert result.optimal_params.shape == (2,)
        assert isinstance(result.optimal_value, float)
        assert isinstance(result.num_iterations, int)
        assert isinstance(result.num_function_evals, int)
        assert isinstance(result.convergence_history, list)
        assert isinstance(result.success, bool)
        assert isinstance(result.message, str)

    def test_convergence_history_length(self) -> None:
        """History length should be consistent with the number of iterations."""
        opt = SPSA(maxiter=100, seed=42)
        result = opt.minimize(quadratic, np.array([1.0, 1.0]))

        # SPSA: 1 initial + maxiter entries
        assert len(result.convergence_history) == 101

    def test_num_function_evals_positive(self) -> None:
        """num_function_evals must be positive for any optimizer run."""
        for OptimizerCls, kwargs in [
            (COBYLA, {"maxiter": 50}),
            (NelderMead, {"maxiter": 50}),
            (SPSA, {"maxiter": 50, "seed": 1}),
            (Adam, {"maxiter": 50}),
            (LBFGSB, {"maxiter": 50}),
            (GradientDescent, {"maxiter": 50}),
        ]:
            opt = OptimizerCls(**kwargs)
            result = opt.minimize(quadratic, np.array([1.0, 1.0]))
            assert result.num_function_evals > 0, f"{opt.name()} had 0 evals"

    def test_repr(self) -> None:
        """OptimizerResult repr should include key fields."""
        result = OptimizerResult(
            optimal_params=np.array([1.0, 2.0]),
            optimal_value=0.5,
            num_iterations=10,
            num_function_evals=100,
        )
        text = repr(result)
        assert "0.5" in text
        assert "10" in text
        assert "100" in text

    def test_default_fields(self) -> None:
        """Default fields should have correct values."""
        result = OptimizerResult(
            optimal_params=np.array([0.0]),
            optimal_value=0.0,
            num_iterations=0,
            num_function_evals=0,
        )
        assert result.convergence_history == []
        assert result.success is True
        assert result.message == ""


# ======================================================================
# 5. Integration: VQEOptimizer and minimize()
# ======================================================================


class TestVQEOptimizer:
    """Tests for the VQEOptimizer convenience class."""

    def test_vqe_with_string_optimizer(self) -> None:
        """VQEOptimizer should accept optimizer name as a string."""
        vqe = VQEOptimizer(optimizer="cobyla", maxiter=500)
        result = vqe.optimize(quadratic, np.array([3.0, -2.0]))

        assert result.optimal_value < 0.5
        assert result.num_function_evals > 0

    def test_vqe_with_optimizer_instance(self) -> None:
        """VQEOptimizer should accept a pre-configured Optimizer object."""
        opt = Adam(maxiter=200, lr=0.1)
        vqe = VQEOptimizer(optimizer=opt)
        result = vqe.optimize(quadratic, np.array([2.0, -3.0]))

        assert result.optimal_value < 1.0

    def test_vqe_with_spsa(self) -> None:
        """VQEOptimizer with SPSA should work (common VQE choice)."""
        vqe = VQEOptimizer(optimizer="spsa", maxiter=200)
        result = vqe.optimize(quadratic, np.array([2.0, -2.0]))

        assert result.optimal_value < 5.0

    def test_vqe_invalid_optimizer_raises(self) -> None:
        """Invalid optimizer name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            VQEOptimizer(optimizer="bogus_optimizer")

    def test_vqe_all_string_optimizers(self) -> None:
        """All documented string optimizer names should work."""
        valid_names = ["spsa", "cobyla", "nelder-mead", "adam", "l-bfgs-b", "gradient-descent"]
        for name in valid_names:
            vqe = VQEOptimizer(optimizer=name, maxiter=10)
            result = vqe.optimize(quadratic, np.array([1.0, 1.0]))
            assert isinstance(result, OptimizerResult)

    def test_vqe_with_bounds(self) -> None:
        """VQEOptimizer should forward bounds to the underlying optimizer."""
        bounds = [(0.0, 10.0), (0.0, 10.0)]
        vqe = VQEOptimizer(optimizer="cobyla", maxiter=500)
        result = vqe.optimize(quadratic, np.array([5.0, 5.0]), bounds=bounds)

        assert result.optimal_value < 1.0

    def test_vqe_callback(self) -> None:
        """VQEOptimizer should pass callback through to the optimizer."""
        log = []

        def cb(it, params, val):
            log.append(val)

        vqe = VQEOptimizer(optimizer="cobyla", maxiter=50, callback=cb)
        vqe.optimize(quadratic, np.array([1.0, 1.0]))

        assert len(log) > 0


class TestMinimizeConvenience:
    """Tests for the top-level minimize() function."""

    def test_minimize_default_spsa(self) -> None:
        """minimize() with default method (SPSA) should work."""
        result = minimize(quadratic, np.array([2.0, -1.0]), maxiter=200)
        assert isinstance(result, OptimizerResult)
        assert result.num_function_evals > 0

    def test_minimize_cobyla(self) -> None:
        """minimize() with COBYLA should converge."""
        result = minimize(rosenbrock, np.array([0.0, 0.0]), method="cobyla", maxiter=2000)
        assert result.optimal_value < 0.5

    def test_minimize_adam(self) -> None:
        """minimize() with Adam should converge on quadratic."""
        result = minimize(quadratic, np.array([3.0, 3.0]), method="adam", maxiter=300, lr=0.1)
        assert result.optimal_value < 0.5

    def test_minimize_lbfgsb_with_bounds(self) -> None:
        """minimize() with L-BFGS-B and bounds should respect constraints."""
        bounds = [(0.5, 5.0), (0.5, 5.0)]
        result = minimize(
            quadratic, np.array([3.0, 3.0]),
            method="l-bfgs-b", maxiter=200, bounds=bounds,
        )
        np.testing.assert_allclose(result.optimal_params, [0.5, 0.5], atol=0.05)

    def test_minimize_invalid_method_raises(self) -> None:
        """Invalid method name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            minimize(quadratic, np.array([1.0]), method="fake")

    def test_minimize_extra_kwargs(self) -> None:
        """Extra kwargs should be forwarded to the optimizer constructor."""
        result = minimize(
            quadratic, np.array([2.0, 2.0]),
            method="spsa", maxiter=100, seed=42, a=0.3, c=0.1,
        )
        assert isinstance(result, OptimizerResult)


# ======================================================================
# 6. Base class validation
# ======================================================================


class TestOptimizerBase:
    """Tests for the Optimizer ABC and validation."""

    def test_invalid_maxiter_raises(self) -> None:
        """maxiter < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="maxiter must be positive"):
            COBYLA(maxiter=0)

        with pytest.raises(ValueError, match="maxiter must be positive"):
            Adam(maxiter=-5)

    def test_negative_tol_raises(self) -> None:
        """Negative tol should raise ValueError."""
        with pytest.raises(ValueError, match="tol must be non-negative"):
            NelderMead(tol=-1.0)

    def test_optimizer_is_abstract(self) -> None:
        """Cannot instantiate the abstract Optimizer base class directly."""
        with pytest.raises(TypeError):
            Optimizer()  # type: ignore[abstract]

    def test_repr_format(self) -> None:
        """Optimizer repr should show name, maxiter, and tol."""
        opt = COBYLA(maxiter=500, tol=1e-4)
        text = repr(opt)
        assert "COBYLA" in text
        assert "500" in text
