"""Observable measurement and analysis for quantum dynamics.

Tools for tracking physical observables throughout time evolution:

- **Observable**: General Hermitian operator measurement.
- **TimeSeriesObservable**: Track an observable at every time step.
- **CorrelationFunction**: Two-point correlators <A(t)B(0)>.
- **EntanglementEntropy**: Von Neumann and Renyi entropies vs time.
- **Magnetization**: Local and total magnetization profiles.
- **SpectralFunction**: Dynamical structure factor S(k, omega) via FFT.
- **Fidelity**: Loschmidt echo |<psi(0)|psi(t)>|^2.

References:
    - Nielsen & Chuang, *Quantum Computation and Quantum Information* (2000)
    - Sachdev, *Quantum Phase Transitions* (2011)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Pauli matrices (shared with hamiltonians.py for convenience)
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


# ---------------------------------------------------------------------------
# Observable
# ---------------------------------------------------------------------------


class Observable:
    """A Hermitian operator whose expectation value can be measured.

    Parameters
    ----------
    matrix : np.ndarray
        Hermitian matrix representation.
    name : str
        Human-readable label.
    """

    def __init__(self, matrix: np.ndarray, name: str = "") -> None:
        self.matrix = np.asarray(matrix, dtype=np.complex128)
        self.name = name
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Observable must be a square matrix.")

    @property
    def dim(self) -> int:
        return self.matrix.shape[0]

    def expectation(self, state: np.ndarray) -> float:
        """Compute <psi|O|psi>."""
        psi = np.asarray(state, dtype=np.complex128).ravel()
        return float(np.real(psi.conj() @ self.matrix @ psi))

    def variance(self, state: np.ndarray) -> float:
        """Compute <O^2> - <O>^2."""
        psi = np.asarray(state, dtype=np.complex128).ravel()
        exp_o = float(np.real(psi.conj() @ self.matrix @ psi))
        o2_psi = self.matrix @ (self.matrix @ psi)
        exp_o2 = float(np.real(psi.conj() @ o2_psi))
        return max(0.0, exp_o2 - exp_o ** 2)

    @classmethod
    def from_pauli_string(cls, label: str, coeff: float = 1.0, name: str = "") -> "Observable":
        """Construct from a Pauli string like ``"XZI"``."""
        mat = _PAULI[label[0]]
        for ch in label[1:]:
            mat = np.kron(mat, _PAULI[ch])
        return cls(coeff * mat, name=name or label)


# ---------------------------------------------------------------------------
# TimeSeriesObservable
# ---------------------------------------------------------------------------


@dataclass
class TimeSeriesObservable:
    """Track one or more observables through a time evolution.

    Parameters
    ----------
    observables : list[Observable]
        Observables to measure at each time step.
    """

    observables: List[Observable] = field(default_factory=list)

    def measure_trajectory(
        self,
        times: np.ndarray,
        states: List[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Measure all observables for a trajectory of states.

        Returns
        -------
        dict[str, np.ndarray]
            Keys are observable names, values are arrays of expectation
            values at each time step.
        """
        result: dict[str, np.ndarray] = {}
        for obs in self.observables:
            values = np.array([obs.expectation(s) for s in states])
            key = obs.name if obs.name else f"obs_{id(obs)}"
            result[key] = values
        return result


# ---------------------------------------------------------------------------
# CorrelationFunction
# ---------------------------------------------------------------------------


class CorrelationFunction:
    """Two-point time-ordered correlation function.

    Computes C(t) = <psi(0)| A^dag(t) B(0) |psi(0)> where
    A(t) = e^{iHt} A e^{-iHt}.

    Parameters
    ----------
    H : np.ndarray
        Hamiltonian matrix.
    A : np.ndarray
        First operator.
    B : np.ndarray
        Second operator.
    """

    def __init__(
        self,
        H: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
    ) -> None:
        self.H = np.asarray(H, dtype=np.complex128)
        self.A = np.asarray(A, dtype=np.complex128)
        self.B = np.asarray(B, dtype=np.complex128)

    def compute(
        self,
        psi0: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Evaluate C(t) at the given time points.

        Uses exact matrix exponentiation via eigendecomposition.

        Parameters
        ----------
        psi0 : np.ndarray
            Initial state.
        times : np.ndarray
            Array of time values.

        Returns
        -------
        np.ndarray
            Complex correlation values C(t).
        """
        psi0 = np.asarray(psi0, dtype=np.complex128).ravel()

        # Diagonalise H for efficient time evolution
        evals, evecs = np.linalg.eigh(self.H)

        # B|psi0>
        b_psi = self.B @ psi0

        correlations = np.zeros(len(times), dtype=np.complex128)

        for idx, t in enumerate(times):
            # e^{-iHt} = V diag(e^{-iE_k t}) V^dag
            phases = np.exp(-1j * evals * t)
            exp_neg = evecs @ np.diag(phases) @ evecs.conj().T
            exp_pos = evecs @ np.diag(np.conj(phases)) @ evecs.conj().T

            # A(t) = e^{iHt} A e^{-iHt}
            a_t = exp_pos @ self.A @ exp_neg

            # C(t) = <psi0| A^dag(t) B |psi0>
            correlations[idx] = psi0.conj() @ (a_t.conj().T @ b_psi)

        return correlations


# ---------------------------------------------------------------------------
# EntanglementEntropy
# ---------------------------------------------------------------------------


class EntanglementEntropy:
    """Bipartite entanglement entropy of a pure state.

    Computes the Von Neumann or Renyi entropy for a bipartition of
    the qubit register into subsystem A (first ``n_a`` qubits) and
    subsystem B (remaining qubits).

    Parameters
    ----------
    n_qubits : int
        Total number of qubits.
    n_a : int
        Number of qubits in subsystem A.  Defaults to n_qubits // 2.
    """

    def __init__(self, n_qubits: int, n_a: Optional[int] = None) -> None:
        self.n_qubits = n_qubits
        self.n_a = n_a if n_a is not None else n_qubits // 2
        self.n_b = n_qubits - self.n_a
        self.dim_a = 2 ** self.n_a
        self.dim_b = 2 ** self.n_b

    def _reduced_density_matrix(self, state: np.ndarray) -> np.ndarray:
        """Trace out subsystem B to get rho_A."""
        psi = np.asarray(state, dtype=np.complex128).ravel()
        psi_mat = psi.reshape(self.dim_a, self.dim_b)
        rho_a = psi_mat @ psi_mat.conj().T
        return rho_a

    def von_neumann(self, state: np.ndarray) -> float:
        """Von Neumann entropy S = -Tr(rho_A log rho_A).

        Parameters
        ----------
        state : np.ndarray
            Pure state vector of length 2^n_qubits.

        Returns
        -------
        float
            Entanglement entropy in nats.
        """
        rho_a = self._reduced_density_matrix(state)
        eigenvalues = np.linalg.eigvalsh(rho_a)
        # Clip small negative values from numerical error
        eigenvalues = np.clip(eigenvalues, 1e-30, None)
        entropy = -float(np.sum(eigenvalues * np.log(eigenvalues)))
        return max(0.0, entropy)

    def renyi(self, state: np.ndarray, alpha: float = 2.0) -> float:
        """Renyi entropy S_alpha = 1/(1-alpha) * log(Tr(rho_A^alpha)).

        Parameters
        ----------
        state : np.ndarray
            Pure state vector.
        alpha : float
            Renyi index (alpha != 1).  Default 2.

        Returns
        -------
        float
        """
        if abs(alpha - 1.0) < 1e-10:
            return self.von_neumann(state)
        rho_a = self._reduced_density_matrix(state)
        eigenvalues = np.linalg.eigvalsh(rho_a)
        eigenvalues = np.clip(eigenvalues, 1e-30, None)
        trace_rho_alpha = float(np.sum(eigenvalues ** alpha))
        return max(0.0, float(np.log(trace_rho_alpha) / (1.0 - alpha)))

    def trajectory(
        self,
        times: np.ndarray,
        states: List[np.ndarray],
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Compute entanglement entropy for a sequence of states.

        Parameters
        ----------
        times : np.ndarray
            Time points (for reference, not used in calculation).
        states : list[np.ndarray]
            State vectors at each time.
        alpha : float or None
            Renyi index.  None (default) uses Von Neumann.

        Returns
        -------
        np.ndarray
            Entropy at each time step.
        """
        func = self.von_neumann if alpha is None else lambda s: self.renyi(s, alpha)
        return np.array([func(s) for s in states])


# ---------------------------------------------------------------------------
# Magnetization
# ---------------------------------------------------------------------------


class Magnetization:
    """Local and total magnetization measurement.

    Computes <Z_i> for each qubit and the total magnetization
    M = (1/n) sum_i <Z_i>.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self._local_ops = []
        for i in range(n_qubits):
            op = np.eye(1, dtype=np.complex128)
            for j in range(n_qubits):
                op = np.kron(op, _Z if j == i else _I)
            self._local_ops.append(op)

    def local(self, state: np.ndarray) -> np.ndarray:
        """Per-qubit magnetization <Z_i>.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_qubits,)`` with local Z expectation values.
        """
        psi = np.asarray(state, dtype=np.complex128).ravel()
        return np.array([
            float(np.real(psi.conj() @ op @ psi))
            for op in self._local_ops
        ])

    def total(self, state: np.ndarray) -> float:
        """Total magnetization M = (1/n) sum_i <Z_i>."""
        return float(np.mean(self.local(state)))

    def profile(
        self,
        times: np.ndarray,
        states: List[np.ndarray],
    ) -> np.ndarray:
        """Compute magnetization profile over time.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(times), n_qubits)`` with local
            magnetizations at each time.
        """
        return np.array([self.local(s) for s in states])

    def total_trajectory(
        self,
        times: np.ndarray,
        states: List[np.ndarray],
    ) -> np.ndarray:
        """Total magnetization vs time.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(times),)`` with total magnetization.
        """
        return np.array([self.total(s) for s in states])


# ---------------------------------------------------------------------------
# SpectralFunction
# ---------------------------------------------------------------------------


class SpectralFunction:
    """Dynamical structure factor S(k, omega) via Fourier transform.

    Computes the spectral function from real-time correlations:

        S(k, omega) = (1/2pi) integral dt e^{i omega t} C_k(t)

    where C_k(t) is the Fourier-transformed two-point correlator in
    momentum space.

    Parameters
    ----------
    n_sites : int
        Number of lattice sites.
    """

    def __init__(self, n_sites: int) -> None:
        self.n_sites = n_sites

    def compute(
        self,
        times: np.ndarray,
        correlations: np.ndarray,
        k_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute S(k, omega) from space-time correlation data.

        Parameters
        ----------
        times : np.ndarray
            Time points, shape ``(n_t,)``.
        correlations : np.ndarray
            Real-space correlations C(r, t) of shape ``(n_sites, n_t)``.
            Row ``r`` is the correlator at spatial separation ``r``.
        k_values : np.ndarray or None
            Momentum values.  Default: 2*pi*m/n_sites for m = 0..n_sites-1.

        Returns
        -------
        k_values : np.ndarray
            Momentum points.
        omega_values : np.ndarray
            Frequency values from FFT.
        S_k_omega : np.ndarray
            Spectral weight, shape ``(len(k_values), len(omega_values))``.
        """
        n_t = len(times)
        dt = times[1] - times[0] if n_t > 1 else 1.0

        if k_values is None:
            k_values = 2.0 * np.pi * np.arange(self.n_sites) / self.n_sites

        omega_values = np.fft.fftfreq(n_t, d=dt / (2 * np.pi))
        omega_values = np.fft.fftshift(omega_values)

        S = np.zeros((len(k_values), n_t), dtype=np.complex128)

        for ik, k in enumerate(k_values):
            # Fourier transform in space: sum_r e^{-ikr} C(r,t)
            c_k_t = np.zeros(n_t, dtype=np.complex128)
            for r in range(correlations.shape[0]):
                c_k_t += np.exp(-1j * k * r) * correlations[r, :]
            # Fourier transform in time
            S[ik, :] = np.fft.fftshift(np.fft.fft(c_k_t)) * dt / (2 * np.pi)

        # Return real part (S is real for Hermitian operators)
        return k_values, omega_values, np.real(S)

    def compute_single_site(
        self,
        times: np.ndarray,
        correlations: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute local spectral function from single-site correlations.

        Parameters
        ----------
        times : np.ndarray
            Time array, shape ``(n_t,)``.
        correlations : np.ndarray
            Time-domain correlation C(t), shape ``(n_t,)``.

        Returns
        -------
        omega_values : np.ndarray
            Frequency grid.
        spectral_weight : np.ndarray
            S(omega) = Re[FT[C(t)]].
        """
        n_t = len(times)
        dt = times[1] - times[0] if n_t > 1 else 1.0
        omega_values = np.fft.fftfreq(n_t, d=dt / (2 * np.pi))
        omega_values = np.fft.fftshift(omega_values)
        S_omega = np.fft.fftshift(np.fft.fft(correlations)) * dt / (2 * np.pi)
        return omega_values, np.real(S_omega)


# ---------------------------------------------------------------------------
# Fidelity (Loschmidt echo)
# ---------------------------------------------------------------------------


class Fidelity:
    """Loschmidt echo and state fidelity tracking.

    Measures |<psi(0)|psi(t)>|^2, which quantifies how much the
    time-evolved state deviates from the initial state.  Also known
    as the survival probability or return probability.

    Parameters
    ----------
    reference : np.ndarray
        Reference state (usually the initial state).
    """

    def __init__(self, reference: np.ndarray) -> None:
        self.reference = np.asarray(reference, dtype=np.complex128).ravel()

    def compute(self, state: np.ndarray) -> float:
        """Fidelity |<ref|state>|^2."""
        psi = np.asarray(state, dtype=np.complex128).ravel()
        overlap = self.reference.conj() @ psi
        return float(np.abs(overlap) ** 2)

    def trajectory(
        self,
        times: np.ndarray,
        states: List[np.ndarray],
    ) -> np.ndarray:
        """Compute fidelity for a trajectory of states.

        Returns
        -------
        np.ndarray
            Fidelity at each time point.
        """
        return np.array([self.compute(s) for s in states])

    @staticmethod
    def state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
        """Fidelity between two pure states: |<psi|phi>|^2."""
        psi = np.asarray(psi, dtype=np.complex128).ravel()
        phi = np.asarray(phi, dtype=np.complex128).ravel()
        return float(np.abs(psi.conj() @ phi) ** 2)
