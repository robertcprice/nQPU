"""Time evolution methods for quantum Hamiltonians.

Implements multiple approaches to computing e^{-iHt}|psi>:

- **ExactEvolution**: Full matrix exponential via eigendecomposition.
  Exact but O(2^{3n}) cost.  Best for small systems or benchmarking.
- **TrotterEvolution**: Product formula decomposition at 1st, 2nd, and
  4th order.  Error decreases polynomially with step count.
- **QDrift**: Randomized (probabilistic) product formula that samples
  Pauli terms proportional to their weight.
- **AdiabaticEvolution**: Slow interpolation from H_0 to H_1 to prepare
  ground states.

References:
    - Trotter, Proc. Amer. Math. Soc. 10, 545 (1959)
    - Suzuki, Phys. Lett. A 146, 319 (1990)
    - Campbell, Phys. Rev. Lett. 123, 070503 (2019)  [QDrift]
    - Farhi et al., Science 292, 472 (2001)  [Adiabatic]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .hamiltonians import PauliOperator, SparsePauliHamiltonian


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EvolutionResult:
    """Result of a time evolution computation.

    Attributes
    ----------
    times : np.ndarray
        Time points at which states were recorded.
    states : list[np.ndarray]
        State vectors at each recorded time.
    final_state : np.ndarray
        The state at the final time.
    method : str
        Name of the evolution method used.
    metadata : dict
        Additional information (step counts, error estimates, etc.).
    """

    times: np.ndarray
    states: List[np.ndarray]
    final_state: np.ndarray
    method: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Matrix exponential via eigendecomposition (no scipy)
# ---------------------------------------------------------------------------


def _matrix_exp_hermitian(H: np.ndarray, t: float) -> np.ndarray:
    """Compute e^{-i H t} for Hermitian H via eigendecomposition.

    Parameters
    ----------
    H : np.ndarray
        Hermitian matrix.
    t : float
        Time parameter.

    Returns
    -------
    np.ndarray
        Unitary matrix e^{-iHt}.
    """
    evals, evecs = np.linalg.eigh(H)
    phases = np.exp(-1j * evals * t)
    return (evecs * phases) @ evecs.conj().T


def _matrix_exp_pade(M: np.ndarray, order: int = 13) -> np.ndarray:
    """Matrix exponential via scaling-and-squaring with Pade approximant.

    Computes exp(M) for a general complex matrix M using a [p/p] Pade
    approximant with scaling to ensure convergence.

    Parameters
    ----------
    M : np.ndarray
        Square complex matrix.
    order : int
        Pade order (6 or 13).

    Returns
    -------
    np.ndarray
        exp(M).
    """
    n = M.shape[0]
    norm = np.linalg.norm(M, 1)

    # Scale so that norm < 1
    s = max(0, int(math.ceil(math.log2(max(norm, 1e-30)))) + 1)
    A = M / (2 ** s)

    if order <= 6:
        # [3/3] Pade
        I = np.eye(n, dtype=M.dtype)
        A2 = A @ A
        A3 = A2 @ A
        U = A @ (A2 + 60.0 * I)
        V = 12.0 * A2 + 120.0 * I
        # Actually use proper [3/3] coefficients
        U = A3 + 60.0 * A
        V = 12.0 * A2 + 120.0 * I
    else:
        # [6/6] Pade (more accurate)
        I = np.eye(n, dtype=M.dtype)
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A4 @ A2

        b = [
            1.0,
            1.0 / 2,
            1.0 / 9,
            1.0 / 72,
            1.0 / 1008,
            1.0 / 30240,
            1.0 / 1209600,
        ]

        U = A @ (b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I)
        V = b[5] * A6 + b[3] * A4 + b[1] * A2 + I

        # Correct Pade coefficients for [6/6]
        # P = V + U,  Q = V - U,  exp(A) approx P/Q = (V+U)/(V-U)
        # Note: the standard Pade [p/p] for exp has
        # N(x) = sum c_k x^k (even in V, odd in U)

    # exp(A) = (V + U) / (V - U)
    result = np.linalg.solve(V - U, V + U)

    # Undo scaling by repeated squaring
    for _ in range(s):
        result = result @ result

    return result


# ---------------------------------------------------------------------------
# ExactEvolution
# ---------------------------------------------------------------------------


class ExactEvolution:
    """Exact time evolution via matrix exponential.

    Computes e^{-iHt}|psi> using eigendecomposition of the Hamiltonian.
    Exact to machine precision but requires O(2^{3n}) diagonalisation.

    Parameters
    ----------
    hamiltonian : SparsePauliHamiltonian
        The system Hamiltonian.
    """

    def __init__(self, hamiltonian: SparsePauliHamiltonian) -> None:
        self.hamiltonian = hamiltonian
        self.H = hamiltonian.matrix()
        # Pre-diagonalise
        self._evals, self._evecs = np.linalg.eigh(self.H)

    def evolve_state(self, psi0: np.ndarray, t: float) -> np.ndarray:
        """Evolve |psi0> to time t.

        Parameters
        ----------
        psi0 : np.ndarray
            Initial state vector.
        t : float
            Evolution time.

        Returns
        -------
        np.ndarray
            Time-evolved state.
        """
        psi0 = np.asarray(psi0, dtype=np.complex128).ravel()
        # Project onto eigenbasis
        coeffs = self._evecs.conj().T @ psi0
        # Apply phases
        phases = np.exp(-1j * self._evals * t)
        return self._evecs @ (phases * coeffs)

    def evolve(
        self,
        psi0: np.ndarray,
        t_final: float,
        n_steps: int = 100,
    ) -> EvolutionResult:
        """Evolve and record states at uniform time intervals.

        Parameters
        ----------
        psi0 : np.ndarray
            Initial state.
        t_final : float
            Final time.
        n_steps : int
            Number of time steps to record.

        Returns
        -------
        EvolutionResult
        """
        psi0 = np.asarray(psi0, dtype=np.complex128).ravel()
        times = np.linspace(0, t_final, n_steps + 1)
        states = [self.evolve_state(psi0, t) for t in times]

        return EvolutionResult(
            times=times,
            states=states,
            final_state=states[-1],
            method="exact",
        )

    def propagator(self, t: float) -> np.ndarray:
        """Return the unitary propagator U(t) = e^{-iHt}."""
        return _matrix_exp_hermitian(self.H, t)


# ---------------------------------------------------------------------------
# TrotterEvolution
# ---------------------------------------------------------------------------


class TrotterEvolution:
    """Product formula time evolution (Trotter-Suzuki decomposition).

    Decomposes the Hamiltonian H = sum_k H_k into groups of mutually
    commuting or individually exponentiable terms and approximates
    e^{-iHt} as a product of e^{-iH_k dt}.

    Supported orders:
      - 1: Lie-Trotter (first order)
      - 2: Suzuki-Trotter (symmetric, second order)
      - 4: Fourth-order Suzuki formula

    Parameters
    ----------
    hamiltonian : SparsePauliHamiltonian
        The system Hamiltonian.
    order : int
        Trotter order: 1, 2, or 4.
    """

    def __init__(
        self,
        hamiltonian: SparsePauliHamiltonian,
        order: int = 2,
    ) -> None:
        if order not in (1, 2, 4):
            raise ValueError(f"Trotter order must be 1, 2, or 4, got {order}.")
        self.hamiltonian = hamiltonian
        self.order = order

        # Build individual term matrices
        self._term_matrices = [t.matrix() for t in hamiltonian.terms]

    def _first_order_step(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """First-order (Lie-Trotter) step: prod_k e^{-i H_k dt}."""
        for H_k in self._term_matrices:
            U_k = _matrix_exp_hermitian(H_k, dt)
            psi = U_k @ psi
        return psi

    def _second_order_step(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """Second-order (symmetric Suzuki-Trotter) step.

        S_2(dt) = prod_k e^{-i H_k dt/2} * prod_{k reversed} e^{-i H_k dt/2}
        """
        half_dt = dt / 2.0
        # Forward sweep
        for H_k in self._term_matrices:
            psi = _matrix_exp_hermitian(H_k, half_dt) @ psi
        # Backward sweep
        for H_k in reversed(self._term_matrices):
            psi = _matrix_exp_hermitian(H_k, half_dt) @ psi
        return psi

    def _fourth_order_step(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """Fourth-order Suzuki formula.

        S_4(dt) = S_2(p*dt) S_2(p*dt) S_2((1-4p)*dt) S_2(p*dt) S_2(p*dt)
        where p = 1 / (4 - 4^{1/3}).
        """
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        q = 1.0 - 4.0 * p

        psi = self._second_order_step(psi, p * dt)
        psi = self._second_order_step(psi, p * dt)
        psi = self._second_order_step(psi, q * dt)
        psi = self._second_order_step(psi, p * dt)
        psi = self._second_order_step(psi, p * dt)
        return psi

    def _trotter_step(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """Single Trotter step at the configured order."""
        if self.order == 1:
            return self._first_order_step(psi, dt)
        elif self.order == 2:
            return self._second_order_step(psi, dt)
        else:
            return self._fourth_order_step(psi, dt)

    def evolve_state(
        self,
        psi0: np.ndarray,
        t: float,
        n_steps: int = 100,
    ) -> np.ndarray:
        """Evolve state to time t using n_steps Trotter steps.

        Parameters
        ----------
        psi0 : np.ndarray
            Initial state.
        t : float
            Total evolution time.
        n_steps : int
            Number of Trotter steps.

        Returns
        -------
        np.ndarray
            Final state.
        """
        psi = np.asarray(psi0, dtype=np.complex128).ravel()
        dt = t / n_steps

        for _ in range(n_steps):
            psi = self._trotter_step(psi, dt)
            # Renormalize to prevent drift
            psi = psi / np.linalg.norm(psi)

        return psi

    def evolve(
        self,
        psi0: np.ndarray,
        t_final: float,
        n_steps: int = 100,
        record_interval: int = 1,
    ) -> EvolutionResult:
        """Evolve and record trajectory.

        Parameters
        ----------
        psi0 : np.ndarray
            Initial state.
        t_final : float
            Final time.
        n_steps : int
            Total Trotter steps.
        record_interval : int
            Record state every N steps.

        Returns
        -------
        EvolutionResult
        """
        psi = np.asarray(psi0, dtype=np.complex128).ravel()
        dt = t_final / n_steps

        times = [0.0]
        states = [psi.copy()]

        for step_idx in range(1, n_steps + 1):
            psi = self._trotter_step(psi, dt)
            psi = psi / np.linalg.norm(psi)

            if step_idx % record_interval == 0 or step_idx == n_steps:
                times.append(step_idx * dt)
                states.append(psi.copy())

        return EvolutionResult(
            times=np.array(times),
            states=states,
            final_state=states[-1],
            method=f"trotter_order{self.order}",
            metadata={"n_steps": n_steps, "dt": dt},
        )

    @staticmethod
    def estimate_steps(
        hamiltonian: SparsePauliHamiltonian,
        t: float,
        target_error: float,
        order: int = 2,
    ) -> int:
        """Estimate the number of Trotter steps for a target error.

        Uses the bound:
            - Order 1: error ~ (sum ||H_k||)^2 * t^2 / (2*r)
            - Order 2: error ~ (sum ||H_k||)^3 * t^3 / (12*r^2)
            - Order 4: error ~ C * t^5 / r^4

        Parameters
        ----------
        hamiltonian : SparsePauliHamiltonian
            The Hamiltonian to decompose.
        t : float
            Evolution time.
        target_error : float
            Desired accuracy.
        order : int
            Trotter order.

        Returns
        -------
        int
            Estimated number of steps.
        """
        norms = [np.linalg.norm(term.matrix(), 2) for term in hamiltonian.terms]
        total_norm = sum(norms)

        if order == 1:
            # ||error|| <= (Lambda * t)^2 / (2*r) where Lambda = sum of norms
            r = max(1, int(math.ceil((total_norm * t) ** 2 / (2 * target_error))))
        elif order == 2:
            r = max(
                1,
                int(
                    math.ceil(
                        ((total_norm * t) ** 1.5 / (12 * target_error)) ** 0.5
                    )
                ),
            )
        else:  # order 4
            r = max(
                1,
                int(
                    math.ceil(
                        ((total_norm * t) ** 1.25 / target_error) ** 0.25
                    )
                ),
            )

        return r


# ---------------------------------------------------------------------------
# QDrift
# ---------------------------------------------------------------------------


class QDrift:
    """Randomized product formula (QDrift protocol).

    Instead of deterministic Trotter ordering, randomly samples terms
    from the Hamiltonian with probability proportional to |c_k|.
    Each random unitary is e^{-i (lambda/N) * sgn(c_k) * P_k * t}
    where lambda = sum |c_k| and N is the number of samples.

    Converges in expectation regardless of term ordering and is
    particularly efficient when the Hamiltonian has many terms with
    varying magnitudes.

    Parameters
    ----------
    hamiltonian : SparsePauliHamiltonian
        The system Hamiltonian.
    seed : int or None
        Random seed.

    References
    ----------
    Campbell, Phys. Rev. Lett. 123, 070503 (2019).
    """

    def __init__(
        self,
        hamiltonian: SparsePauliHamiltonian,
        seed: Optional[int] = None,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.rng = np.random.default_rng(seed)

        # Precompute sampling distribution
        self._term_matrices = []
        self._weights = []
        self._signs = []

        for term in hamiltonian.terms:
            mat = term.matrix()
            c = term.coeff
            w = abs(c)
            if w < 1e-15:
                continue
            self._term_matrices.append(mat / c if abs(c) > 1e-15 else mat)
            self._weights.append(w)
            self._signs.append(c / w if w > 1e-15 else 1.0)

        total_weight = sum(self._weights)
        self._lambda = total_weight
        self._probs = np.array(self._weights) / total_weight

    def evolve_state(
        self,
        psi0: np.ndarray,
        t: float,
        n_samples: int = 200,
    ) -> np.ndarray:
        """Evolve state using QDrift with n_samples random unitaries.

        Parameters
        ----------
        psi0 : np.ndarray
            Initial state.
        t : float
            Total evolution time.
        n_samples : int
            Number of random unitary samples (more = more accurate).

        Returns
        -------
        np.ndarray
            Final state.
        """
        psi = np.asarray(psi0, dtype=np.complex128).ravel()
        tau = self._lambda * t / n_samples

        for _ in range(n_samples):
            # Sample term index
            idx = self.rng.choice(len(self._term_matrices), p=self._probs)
            # Apply e^{-i * sign * P_k * tau}
            H_k = self._signs[idx] * self._term_matrices[idx]
            U_k = _matrix_exp_hermitian(H_k, tau)
            psi = U_k @ psi

        psi = psi / np.linalg.norm(psi)
        return psi

    def evolve(
        self,
        psi0: np.ndarray,
        t_final: float,
        n_samples: int = 200,
        n_records: int = 10,
    ) -> EvolutionResult:
        """Evolve and record at intermediate times.

        Parameters
        ----------
        psi0 : np.ndarray
            Initial state.
        t_final : float
            Final time.
        n_samples : int
            Total random samples.
        n_records : int
            Number of intermediate recordings.

        Returns
        -------
        EvolutionResult
        """
        psi = np.asarray(psi0, dtype=np.complex128).ravel()
        record_times = np.linspace(0, t_final, n_records + 1)

        times = [0.0]
        states = [psi.copy()]

        samples_per_interval = max(1, n_samples // n_records)
        tau = self._lambda * t_final / n_samples

        sample_idx = 0
        for rec_idx in range(1, n_records + 1):
            samples_this = samples_per_interval
            if rec_idx == n_records:
                samples_this = n_samples - sample_idx

            for _ in range(samples_this):
                idx = self.rng.choice(len(self._term_matrices), p=self._probs)
                H_k = self._signs[idx] * self._term_matrices[idx]
                U_k = _matrix_exp_hermitian(H_k, tau)
                psi = U_k @ psi
                sample_idx += 1

            psi = psi / np.linalg.norm(psi)
            times.append(record_times[rec_idx])
            states.append(psi.copy())

        return EvolutionResult(
            times=np.array(times),
            states=states,
            final_state=states[-1],
            method="qdrift",
            metadata={"n_samples": n_samples, "lambda": self._lambda},
        )


# ---------------------------------------------------------------------------
# AdiabaticEvolution
# ---------------------------------------------------------------------------


def _linear_schedule(s: float) -> float:
    """Linear interpolation schedule."""
    return s


def _polynomial_schedule(s: float, power: float = 3.0) -> float:
    """Polynomial schedule: s^p. Slower start, faster finish."""
    return s ** power


def _exponential_schedule(s: float, rate: float = 5.0) -> float:
    """Exponential schedule: (e^{rate*s} - 1) / (e^{rate} - 1)."""
    if abs(rate) < 1e-10:
        return s
    return (math.exp(rate * s) - 1.0) / (math.exp(rate) - 1.0)


SCHEDULE_FUNCTIONS = {
    "linear": _linear_schedule,
    "polynomial": lambda s: _polynomial_schedule(s, 3.0),
    "exponential": lambda s: _exponential_schedule(s, 5.0),
}


class AdiabaticEvolution:
    """Adiabatic quantum evolution from H_0 to H_1.

    Implements the time-dependent Hamiltonian:

        H(s) = (1 - f(s)) * H_0 + f(s) * H_1

    where s = t / T goes from 0 to 1, f(s) is the schedule function,
    and T is the total evolution time.

    By the adiabatic theorem, if T is large enough compared to the
    inverse spectral gap squared, the system remains in the
    instantaneous ground state.

    Parameters
    ----------
    H_initial : SparsePauliHamiltonian
        Initial Hamiltonian whose ground state is easy to prepare.
    H_final : SparsePauliHamiltonian
        Target Hamiltonian whose ground state we want.
    schedule : str or callable
        Schedule function f(s) mapping [0,1] -> [0,1].
        Built-in options: "linear", "polynomial", "exponential".
    """

    def __init__(
        self,
        H_initial: SparsePauliHamiltonian,
        H_final: SparsePauliHamiltonian,
        schedule: str | Callable[[float], float] = "linear",
    ) -> None:
        self.H_initial = H_initial
        self.H_final = H_final
        self.H0_mat = H_initial.matrix()
        self.H1_mat = H_final.matrix()

        if callable(schedule):
            self.schedule_fn = schedule
        else:
            if schedule not in SCHEDULE_FUNCTIONS:
                raise ValueError(
                    f"Unknown schedule '{schedule}'. "
                    f"Options: {list(SCHEDULE_FUNCTIONS.keys())}"
                )
            self.schedule_fn = SCHEDULE_FUNCTIONS[schedule]

    def hamiltonian_at(self, s: float) -> np.ndarray:
        """Instantaneous Hamiltonian H(s) at schedule parameter s.

        Parameters
        ----------
        s : float
            Schedule parameter in [0, 1].

        Returns
        -------
        np.ndarray
            Hamiltonian matrix at parameter s.
        """
        f = self.schedule_fn(s)
        return (1.0 - f) * self.H0_mat + f * self.H1_mat

    def gap_at(self, s: float) -> float:
        """Spectral gap at schedule parameter s.

        Returns
        -------
        float
            Energy gap between ground and first excited state.
        """
        H_s = self.hamiltonian_at(s)
        evals = np.linalg.eigvalsh(H_s)
        return float(evals[1] - evals[0])

    def minimum_gap(self, n_points: int = 100) -> Tuple[float, float]:
        """Find the minimum spectral gap along the adiabatic path.

        Returns
        -------
        s_min : float
            Schedule parameter at minimum gap.
        gap_min : float
            The minimum gap value.
        """
        s_vals = np.linspace(0, 1, n_points)
        gaps = [self.gap_at(s) for s in s_vals]
        idx = int(np.argmin(gaps))
        return float(s_vals[idx]), float(gaps[idx])

    def evolve(
        self,
        psi0: np.ndarray,
        T: float,
        n_steps: int = 200,
    ) -> EvolutionResult:
        """Run adiabatic evolution.

        Uses a second-order Trotter-like scheme where the Hamiltonian is
        updated at the midpoint of each time step.

        Parameters
        ----------
        psi0 : np.ndarray
            Initial state (should be ground state of H_initial).
        T : float
            Total evolution time.
        n_steps : int
            Number of time steps.

        Returns
        -------
        EvolutionResult
        """
        psi = np.asarray(psi0, dtype=np.complex128).ravel()
        dt = T / n_steps

        times = [0.0]
        states = [psi.copy()]

        for step_idx in range(n_steps):
            # Use midpoint Hamiltonian for better accuracy
            s_mid = (step_idx + 0.5) / n_steps
            H_mid = self.hamiltonian_at(s_mid)
            U = _matrix_exp_hermitian(H_mid, dt)
            psi = U @ psi
            psi = psi / np.linalg.norm(psi)

            times.append((step_idx + 1) * dt)
            states.append(psi.copy())

        return EvolutionResult(
            times=np.array(times),
            states=states,
            final_state=states[-1],
            method="adiabatic",
            metadata={
                "T": T,
                "n_steps": n_steps,
                "schedule": str(self.schedule_fn),
            },
        )

    def estimate_time(self, target_fidelity: float = 0.99) -> float:
        """Estimate the total time T needed for target ground-state fidelity.

        Uses the adiabatic condition: T >> max_s |<1(s)|dH/ds|0(s)>| / gap^2.

        Parameters
        ----------
        target_fidelity : float
            Desired fidelity with the target ground state.

        Returns
        -------
        float
            Estimated total evolution time.
        """
        _, gap_min = self.minimum_gap()
        if gap_min < 1e-10:
            return float("inf")

        # dH/ds norm estimate
        dH_norm = np.linalg.norm(self.H1_mat - self.H0_mat, 2)

        # Adiabatic condition: T >> dH_norm / gap^2
        safety_factor = -math.log(1.0 - target_fidelity + 1e-15) + 1.0
        return safety_factor * dH_norm / (gap_min ** 2)
