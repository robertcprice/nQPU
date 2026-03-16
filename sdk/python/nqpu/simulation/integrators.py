"""ODE integrators for quantum dynamics.

Numerical integration methods for the time-dependent Schrodinger equation:

    i d|psi>/dt = H |psi>

with norm-preserving and energy-conserving properties.

Integrators provided:

- **RungeKutta4**: Classic fourth-order explicit RK scheme.
- **LeapfrogIntegrator**: Symplectic second-order (Stormer-Verlet),
  excellent long-time energy conservation.
- **AdaptiveRK45**: Runge-Kutta-Fehlberg with embedded error estimate
  and automatic step-size control.
- **CrankNicolson**: Implicit second-order, unconditionally stable,
  exact norm preservation (unitary).

References:
    - Hairer, Lubich & Wanner, *Geometric Numerical Integration* (2006)
    - Press et al., *Numerical Recipes* (Cambridge, 2007)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Common types and helpers
# ---------------------------------------------------------------------------


@dataclass
class IntegratorResult:
    """Result of an ODE integration run.

    Attributes
    ----------
    times : np.ndarray
        Time points at which the state was recorded.
    states : list[np.ndarray]
        State vectors at each recorded time.
    energy_history : list[float]
        Energy expectation value at each recorded time.
    norm_history : list[float]
        Norm of the state at each recorded time.
    steps_taken : int
        Total number of integration steps.
    """

    times: np.ndarray
    states: List[np.ndarray]
    energy_history: List[float]
    norm_history: List[float]
    steps_taken: int = 0

    @property
    def energy_drift(self) -> float:
        """Relative energy drift: |E_final - E_initial| / |E_initial|."""
        if not self.energy_history or abs(self.energy_history[0]) < 1e-15:
            if not self.energy_history:
                return 0.0
            return abs(self.energy_history[-1] - self.energy_history[0])
        return abs(
            (self.energy_history[-1] - self.energy_history[0])
            / self.energy_history[0]
        )

    @property
    def norm_drift(self) -> float:
        """Maximum deviation of norm from unity."""
        if not self.norm_history:
            return 0.0
        return max(abs(n - 1.0) for n in self.norm_history)


def _compute_energy(state: np.ndarray, H: np.ndarray) -> float:
    """Compute <psi|H|psi> (real part)."""
    return float(np.real(state.conj() @ H @ state))


def _compute_norm(state: np.ndarray) -> float:
    """Compute ||psi||."""
    return float(np.sqrt(np.real(state.conj() @ state)))


# ---------------------------------------------------------------------------
# Schrodinger RHS:  d|psi>/dt = -i H |psi>
# ---------------------------------------------------------------------------


def _schrodinger_rhs(H: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Right-hand side of the Schrodinger equation: -i H |psi>."""
    return -1j * (H @ psi)


# ---------------------------------------------------------------------------
# RungeKutta4
# ---------------------------------------------------------------------------


class RungeKutta4:
    """Classic fourth-order Runge-Kutta integrator.

    Fourth-order accurate with fixed step size.  Not symplectic, so energy
    may drift over long times, but norm-renormalization is applied at each
    step to preserve unitarity.

    Parameters
    ----------
    H : np.ndarray
        Time-independent Hamiltonian matrix.
    dt : float
        Integration time step.
    renormalize : bool
        If True, renormalize state after each step (default True).
    """

    def __init__(
        self,
        H: np.ndarray,
        dt: float = 0.01,
        renormalize: bool = True,
    ) -> None:
        self.H = np.asarray(H, dtype=np.complex128)
        self.dt = dt
        self.renormalize = renormalize

    def step(self, psi: np.ndarray) -> np.ndarray:
        """Advance state by one time step."""
        dt = self.dt
        k1 = _schrodinger_rhs(self.H, psi)
        k2 = _schrodinger_rhs(self.H, psi + 0.5 * dt * k1)
        k3 = _schrodinger_rhs(self.H, psi + 0.5 * dt * k2)
        k4 = _schrodinger_rhs(self.H, psi + dt * k3)

        psi_new = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if self.renormalize:
            psi_new = psi_new / np.linalg.norm(psi_new)

        return psi_new

    def evolve(
        self,
        psi0: np.ndarray,
        t_final: float,
        record_interval: int = 1,
    ) -> IntegratorResult:
        """Evolve from ``psi0`` to ``t_final``.

        Parameters
        ----------
        psi0 : np.ndarray
            Initial state vector.
        t_final : float
            Final time.
        record_interval : int
            Record state every N steps.

        Returns
        -------
        IntegratorResult
        """
        psi = np.asarray(psi0, dtype=np.complex128).copy()
        n_steps = max(1, int(math.ceil(t_final / self.dt)))
        dt_actual = t_final / n_steps

        # Override dt to land exactly on t_final
        original_dt = self.dt
        self.dt = dt_actual

        times = [0.0]
        states = [psi.copy()]
        energies = [_compute_energy(psi, self.H)]
        norms = [_compute_norm(psi)]

        for step_idx in range(1, n_steps + 1):
            psi = self.step(psi)
            if step_idx % record_interval == 0 or step_idx == n_steps:
                t = step_idx * dt_actual
                times.append(t)
                states.append(psi.copy())
                energies.append(_compute_energy(psi, self.H))
                norms.append(_compute_norm(psi))

        self.dt = original_dt

        return IntegratorResult(
            times=np.array(times),
            states=states,
            energy_history=energies,
            norm_history=norms,
            steps_taken=n_steps,
        )


# ---------------------------------------------------------------------------
# LeapfrogIntegrator (Stormer-Verlet)
# ---------------------------------------------------------------------------


class LeapfrogIntegrator:
    """Symplectic leapfrog (Stormer-Verlet) integrator.

    Splits the Schrodinger evolution into real and imaginary parts
    and applies a symplectic leapfrog scheme.  Exactly preserves a
    modified energy (shadow Hamiltonian) and exhibits no secular energy
    drift.

    The state psi = q + i*p is split into real part q and imaginary part p.
    The Schrodinger equation i dpsi/dt = H psi becomes:

        dq/dt =  H_r * p + H_i * q
        dp/dt = -H_r * q + H_i * p

    For real Hamiltonians (H_i = 0) this simplifies to a standard
    symplectic scheme.

    Parameters
    ----------
    H : np.ndarray
        Hamiltonian matrix (must be real-symmetric for energy conservation).
    dt : float
        Time step.
    """

    def __init__(self, H: np.ndarray, dt: float = 0.01) -> None:
        self.H_full = np.asarray(H, dtype=np.complex128)
        self.H_r = np.real(self.H_full)
        self.H_i = np.imag(self.H_full)
        self.dt = dt
        self._has_imaginary = np.any(np.abs(self.H_i) > 1e-15)

    def step(self, psi: np.ndarray) -> np.ndarray:
        """Advance state by one time step using leapfrog."""
        dt = self.dt
        q = np.real(psi).copy()
        p = np.imag(psi).copy()

        if self._has_imaginary:
            # Full complex case
            # Half step for q
            q += 0.5 * dt * (self.H_r @ p + self.H_i @ q)
            # Full step for p
            p += dt * (-self.H_r @ q + self.H_i @ p)
            # Half step for q
            q += 0.5 * dt * (self.H_r @ p + self.H_i @ q)
        else:
            # Real Hamiltonian: pure symplectic
            # Half step for q:   dq/dt = H*p
            q += 0.5 * dt * (self.H_r @ p)
            # Full step for p:   dp/dt = -H*q
            p += dt * (-self.H_r @ q)
            # Half step for q
            q += 0.5 * dt * (self.H_r @ p)

        psi_new = q + 1j * p
        # Renormalize to correct for floating-point drift
        psi_new = psi_new / np.linalg.norm(psi_new)
        return psi_new

    def evolve(
        self,
        psi0: np.ndarray,
        t_final: float,
        record_interval: int = 1,
    ) -> IntegratorResult:
        """Evolve state from t=0 to t=t_final."""
        psi = np.asarray(psi0, dtype=np.complex128).copy()
        n_steps = max(1, int(math.ceil(t_final / self.dt)))
        dt_actual = t_final / n_steps

        original_dt = self.dt
        self.dt = dt_actual

        times = [0.0]
        states = [psi.copy()]
        energies = [_compute_energy(psi, self.H_full)]
        norms = [_compute_norm(psi)]

        for step_idx in range(1, n_steps + 1):
            psi = self.step(psi)
            if step_idx % record_interval == 0 or step_idx == n_steps:
                t = step_idx * dt_actual
                times.append(t)
                states.append(psi.copy())
                energies.append(_compute_energy(psi, self.H_full))
                norms.append(_compute_norm(psi))

        self.dt = original_dt

        return IntegratorResult(
            times=np.array(times),
            states=states,
            energy_history=energies,
            norm_history=norms,
            steps_taken=n_steps,
        )


# ---------------------------------------------------------------------------
# AdaptiveRK45 (Runge-Kutta-Fehlberg)
# ---------------------------------------------------------------------------


class AdaptiveRK45:
    """Runge-Kutta-Fehlberg 4(5) with adaptive step-size control.

    Uses an embedded pair of 4th and 5th order formulas to estimate the
    local truncation error and adapt the step size automatically.

    Parameters
    ----------
    H : np.ndarray
        Hamiltonian matrix.
    dt_init : float
        Initial step size guess.
    atol : float
        Absolute error tolerance.
    rtol : float
        Relative error tolerance.
    dt_min : float
        Minimum allowed step size.
    dt_max : float
        Maximum allowed step size.
    renormalize : bool
        Renormalize after each accepted step.
    """

    # Butcher tableau coefficients for RKF45
    _A = [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]
    _B = [
        [],
        [1 / 4],
        [3 / 32, 9 / 32],
        [1932 / 2197, -7200 / 2197, 7296 / 2197],
        [439 / 216, -8, 3680 / 513, -845 / 4104],
        [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40],
    ]
    # 4th order weights
    _C4 = [25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0]
    # 5th order weights
    _C5 = [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]

    def __init__(
        self,
        H: np.ndarray,
        dt_init: float = 0.01,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        dt_min: float = 1e-12,
        dt_max: float = 1.0,
        renormalize: bool = True,
    ) -> None:
        self.H = np.asarray(H, dtype=np.complex128)
        self.dt = dt_init
        self.atol = atol
        self.rtol = rtol
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.renormalize = renormalize

    def _attempt_step(
        self, psi: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Try a single RKF45 step, return (y4, y5, error_estimate)."""
        k = [None] * 6
        k[0] = _schrodinger_rhs(self.H, psi)

        for i in range(1, 6):
            y_temp = psi.copy()
            for j in range(i):
                y_temp = y_temp + dt * self._B[i][j] * k[j]
            k[i] = _schrodinger_rhs(self.H, y_temp)

        # 4th and 5th order solutions
        y4 = psi.copy()
        y5 = psi.copy()
        for i in range(6):
            y4 = y4 + dt * self._C4[i] * k[i]
            y5 = y5 + dt * self._C5[i] * k[i]

        # Error estimate
        err_vec = y5 - y4
        scale = self.atol + self.rtol * np.maximum(np.abs(y4), np.abs(psi))
        err = float(np.sqrt(np.mean((np.abs(err_vec) / scale) ** 2)))

        return y4, y5, err

    def evolve(
        self,
        psi0: np.ndarray,
        t_final: float,
        record_interval: float = 0.0,
    ) -> IntegratorResult:
        """Evolve with adaptive stepping.

        Parameters
        ----------
        psi0 : np.ndarray
            Initial state.
        t_final : float
            Final time.
        record_interval : float
            Record every this much time (0 = record every accepted step).
        """
        psi = np.asarray(psi0, dtype=np.complex128).copy()
        t = 0.0
        dt = min(self.dt, t_final)

        times = [0.0]
        states = [psi.copy()]
        energies = [_compute_energy(psi, self.H)]
        norms = [_compute_norm(psi)]
        steps = 0
        next_record = record_interval if record_interval > 0 else 0.0

        while t < t_final - 1e-15:
            dt = min(dt, t_final - t)
            dt = max(dt, self.dt_min)

            y4, y5, err = self._attempt_step(psi, dt)
            steps += 1

            if err <= 1.0 or dt <= self.dt_min:
                # Accept step
                psi = y5  # Use 5th order solution
                if self.renormalize:
                    psi = psi / np.linalg.norm(psi)
                t += dt

                # Record
                should_record = False
                if record_interval <= 0:
                    should_record = True
                elif t >= next_record - 1e-15:
                    should_record = True
                    next_record += record_interval

                if should_record or abs(t - t_final) < 1e-15:
                    times.append(t)
                    states.append(psi.copy())
                    energies.append(_compute_energy(psi, self.H))
                    norms.append(_compute_norm(psi))

            # Adjust step size
            if err > 1e-15:
                safety = 0.84
                factor = safety * (1.0 / err) ** 0.2
                factor = max(0.1, min(factor, 5.0))
                dt = dt * factor
            else:
                dt = dt * 2.0

            dt = max(self.dt_min, min(dt, self.dt_max))

        return IntegratorResult(
            times=np.array(times),
            states=states,
            energy_history=energies,
            norm_history=norms,
            steps_taken=steps,
        )


# ---------------------------------------------------------------------------
# CrankNicolson
# ---------------------------------------------------------------------------


class CrankNicolson:
    """Crank-Nicolson (implicit midpoint) integrator.

    Solves the Schrodinger equation using the implicit trapezoidal rule:

        (I + i*dt/2 * H) psi_{n+1} = (I - i*dt/2 * H) psi_n

    This is unconditionally stable, second-order accurate, and exactly
    unitary (preserves norm to machine precision).

    Parameters
    ----------
    H : np.ndarray
        Hamiltonian matrix.
    dt : float
        Time step.
    """

    def __init__(self, H: np.ndarray, dt: float = 0.01) -> None:
        self.H = np.asarray(H, dtype=np.complex128)
        self.dt = dt
        self._update_operators()

    def _update_operators(self) -> None:
        """Precompute LHS^{-1} @ RHS for the implicit step."""
        dim = self.H.shape[0]
        I = np.eye(dim, dtype=np.complex128)
        half_dt_H = 0.5 * self.dt * self.H

        lhs = I + 1j * half_dt_H
        rhs = I - 1j * half_dt_H

        # LHS^{-1} @ RHS is the propagator for one step
        self._propagator = np.linalg.solve(lhs, rhs)

    def step(self, psi: np.ndarray) -> np.ndarray:
        """Advance state by one time step."""
        return self._propagator @ psi

    def evolve(
        self,
        psi0: np.ndarray,
        t_final: float,
        record_interval: int = 1,
    ) -> IntegratorResult:
        """Evolve state from t=0 to t=t_final."""
        psi = np.asarray(psi0, dtype=np.complex128).copy()
        n_steps = max(1, int(math.ceil(t_final / self.dt)))
        dt_actual = t_final / n_steps

        # Rebuild propagator with adjusted dt
        original_dt = self.dt
        self.dt = dt_actual
        self._update_operators()

        times = [0.0]
        states = [psi.copy()]
        energies = [_compute_energy(psi, self.H)]
        norms = [_compute_norm(psi)]

        for step_idx in range(1, n_steps + 1):
            psi = self.step(psi)
            if step_idx % record_interval == 0 or step_idx == n_steps:
                t = step_idx * dt_actual
                times.append(t)
                states.append(psi.copy())
                energies.append(_compute_energy(psi, self.H))
                norms.append(_compute_norm(psi))

        self.dt = original_dt
        self._update_operators()

        return IntegratorResult(
            times=np.array(times),
            states=states,
            energy_history=energies,
            norm_history=norms,
            steps_taken=n_steps,
        )
