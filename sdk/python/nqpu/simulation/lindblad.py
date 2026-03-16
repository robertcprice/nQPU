"""Lindblad master equation for open quantum system dynamics.

Simulates the evolution of quantum systems coupled to an environment
using the Lindblad (GKSL) master equation:

    drho/dt = -i[H, rho] + sum_k gamma_k (L_k rho L_k^dag - {L_k^dag L_k, rho}/2)

This is the most general form of a Markovian, trace-preserving,
completely-positive quantum master equation.

Features:
  - Dense and vectorised (superoperator) representations.
  - RK4 integration and exact (Liouvillian diagonalisation) solvers.
  - Steady-state finder via null-space of the Liouvillian.
  - Standard noise channel constructors (amplitude damping, dephasing,
    depolarising, thermal bath).
  - Result analysis: purity, Von Neumann entropy, expectation values,
    populations.

References:
    - Lindblad, Commun. Math. Phys. 48, 119 (1976)
    - Gorini, Kossakowski & Sudarshan, J. Math. Phys. 17, 821 (1976)
    - Breuer & Petruccione, *The Theory of Open Quantum Systems* (2002)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Pauli matrices for noise channel constructors
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Lowering and raising operators
_SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)
_SIGMA_PLUS = np.array([[0, 0], [1, 0]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# LindbladOperator
# ---------------------------------------------------------------------------


@dataclass
class LindbladOperator:
    """Single Lindblad jump operator with rate.

    Parameters
    ----------
    operator : np.ndarray
        The jump operator matrix L_k.
    rate : float
        Dissipation rate gamma_k (must be non-negative).
    label : str
        Human-readable label for this channel.
    """

    operator: np.ndarray
    rate: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        self.operator = np.asarray(self.operator, dtype=np.complex128)
        if self.rate < 0:
            raise ValueError(f"Lindblad rate must be non-negative, got {self.rate}.")


# ---------------------------------------------------------------------------
# LindbladMasterEquation
# ---------------------------------------------------------------------------


@dataclass
class LindbladMasterEquation:
    """Lindblad master equation representation.

    Encodes the generator of Markovian open quantum dynamics:

        drho/dt = -i[H, rho] + sum_k gamma_k (L_k rho L_k^dag
                  - {L_k^dag L_k, rho}/2)

    Supports both matrix-form evaluation (drho_dt) and the full
    Liouvillian superoperator construction for vectorised evolution.

    Parameters
    ----------
    hamiltonian : np.ndarray
        System Hamiltonian (Hermitian matrix).
    jump_operators : list[LindbladOperator]
        Lindblad jump operators with associated rates.
    """

    hamiltonian: np.ndarray
    jump_operators: List[LindbladOperator] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.hamiltonian = np.asarray(self.hamiltonian, dtype=np.complex128)

    @property
    def dim(self) -> int:
        """Hilbert space dimension."""
        return self.hamiltonian.shape[0]

    def liouvillian(self) -> np.ndarray:
        """Build the Liouvillian superoperator L such that
        d vec(rho)/dt = L vec(rho).

        Uses the row-major vectorisation identity (numpy default):
            vec(ABC) = (A kron C^T) vec(B)

        The Liouvillian has dimension d^2 x d^2 where d is the
        Hilbert space dimension.

        Returns
        -------
        np.ndarray
            Complex matrix of shape (d^2, d^2).
        """
        d = self.dim
        I_d = np.eye(d, dtype=np.complex128)
        H = self.hamiltonian

        # Hamiltonian part: -i[H, rho] = -i(H rho - rho H)
        # vec(H rho) = (H kron I) vec(rho)
        # vec(rho H) = (I kron H^T) vec(rho)
        L_H = -1j * (np.kron(H, I_d) - np.kron(I_d, H.T))

        # Dissipator part
        L_D = np.zeros((d * d, d * d), dtype=np.complex128)
        for lop in self.jump_operators:
            Lk = lop.operator
            Lk_dag = Lk.conj().T
            LdL = Lk_dag @ Lk

            # gamma * (L rho L^dag - 0.5 {L^dag L, rho})
            # vec(L rho L^dag) = (L kron L^dag.T) vec(rho)
            #                  = (L kron conj(L)) vec(rho)
            # vec(LdL rho) = (LdL kron I) vec(rho)
            # vec(rho LdL) = (I kron LdL^T) vec(rho)
            L_D += lop.rate * (
                np.kron(Lk, Lk.conj())
                - 0.5 * np.kron(LdL, I_d)
                - 0.5 * np.kron(I_d, LdL.T)
            )

        return L_H + L_D

    def drho_dt(self, rho: np.ndarray) -> np.ndarray:
        """Compute time derivative of density matrix directly.

        This avoids building the full Liouvillian superoperator and is
        more efficient for single evaluations or RK4 stepping.

        Parameters
        ----------
        rho : np.ndarray
            Density matrix of shape (d, d).

        Returns
        -------
        np.ndarray
            Time derivative drho/dt of shape (d, d).
        """
        H = self.hamiltonian
        # Hamiltonian (coherent) part
        result = -1j * (H @ rho - rho @ H)

        # Dissipator part
        for lop in self.jump_operators:
            Lk = lop.operator
            Lk_dag = Lk.conj().T
            LdL = Lk_dag @ Lk
            result += lop.rate * (
                Lk @ rho @ Lk_dag - 0.5 * (LdL @ rho + rho @ LdL)
            )

        return result

    def is_trace_preserving(self, atol: float = 1e-10) -> bool:
        """Check whether the Liouvillian preserves trace.

        The Lindblad form is guaranteed trace-preserving, but this
        serves as a numerical verification.
        """
        L = self.liouvillian()
        d = self.dim
        # Trace preservation: sum_i L_{(i,i), (j,k)} = 0 for all j,k
        # Equivalently: (vec(I))^T L = 0
        vec_I = np.eye(d, dtype=np.complex128).ravel()
        result = vec_I @ L
        return bool(np.allclose(result, 0, atol=atol))

    def is_hermiticity_preserving(self, atol: float = 1e-10) -> bool:
        """Check whether the evolution preserves Hermiticity of rho."""
        d = self.dim
        # Test on a random Hermitian matrix
        rng = np.random.default_rng(42)
        A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        rho_test = (A + A.conj().T) / (2 * d)
        drho = self.drho_dt(rho_test)
        return bool(np.allclose(drho, drho.conj().T, atol=atol))


# ---------------------------------------------------------------------------
# LindbladResult
# ---------------------------------------------------------------------------


@dataclass
class LindbladResult:
    """Result of Lindblad evolution.

    Attributes
    ----------
    times : np.ndarray
        Time points at which states were recorded.
    states : list[np.ndarray]
        Density matrices at each recorded time.
    """

    times: np.ndarray
    states: List[np.ndarray]

    def purity(self) -> np.ndarray:
        """Tr(rho^2) at each time step.

        Purity ranges from 1/d (maximally mixed) to 1 (pure state).
        Under dissipation, purity typically decays from 1.

        Returns
        -------
        np.ndarray
            Purity values at each time step.
        """
        return np.array([
            np.real(np.trace(rho @ rho)) for rho in self.states
        ])

    def von_neumann_entropy(self) -> np.ndarray:
        """Von Neumann entropy S = -Tr(rho log rho) at each step.

        Returns
        -------
        np.ndarray
            Entropy values in nats.
        """
        result = []
        for rho in self.states:
            eigvals = np.linalg.eigvalsh(rho)
            eigvals = eigvals[eigvals > 1e-15]
            s = -np.sum(eigvals * np.log(eigvals))
            result.append(float(np.real(s)))
        return np.array(result)

    def expectation(self, observable: np.ndarray) -> np.ndarray:
        """Compute Tr(O rho) at each time step.

        Parameters
        ----------
        observable : np.ndarray
            Hermitian observable matrix.

        Returns
        -------
        np.ndarray
            Real-valued expectation values.
        """
        observable = np.asarray(observable, dtype=np.complex128)
        return np.array([
            np.real(np.trace(observable @ rho)) for rho in self.states
        ])

    def populations(self) -> np.ndarray:
        """Diagonal elements of rho at each time step.

        Returns
        -------
        np.ndarray
            Array of shape (n_steps, d) with populations.
        """
        return np.array([np.real(np.diag(rho)) for rho in self.states])

    def trace(self) -> np.ndarray:
        """Trace of rho at each step (should be 1 if trace-preserving).

        Returns
        -------
        np.ndarray
            Trace values at each time step.
        """
        return np.array([
            np.real(np.trace(rho)) for rho in self.states
        ])

    def fidelity_with(self, target: np.ndarray) -> np.ndarray:
        """Fidelity F = Tr(target * rho) for a pure target state.

        For mixed-state fidelity, use the Uhlmann fidelity instead.

        Parameters
        ----------
        target : np.ndarray
            Target density matrix (or pure state outer product).

        Returns
        -------
        np.ndarray
            Fidelity values at each time step.
        """
        target = np.asarray(target, dtype=np.complex128)
        return np.array([
            np.real(np.trace(target @ rho)) for rho in self.states
        ])


# ---------------------------------------------------------------------------
# LindbladSolver
# ---------------------------------------------------------------------------


@dataclass
class LindbladSolver:
    """Solve Lindblad master equation using RK4 or exact diagonalisation.

    Parameters
    ----------
    equation : LindbladMasterEquation
        The master equation to solve.
    method : str
        Integration method: ``"rk4"`` for Runge-Kutta 4th order,
        ``"exact"`` for Liouvillian diagonalisation.
    """

    equation: LindbladMasterEquation
    method: str = "rk4"

    def __post_init__(self) -> None:
        if self.method not in ("rk4", "exact"):
            raise ValueError(
                f"Unknown method '{self.method}'. Use 'rk4' or 'exact'."
            )

    def evolve(
        self,
        rho0: np.ndarray,
        t_final: float,
        n_steps: int = 100,
    ) -> LindbladResult:
        """Evolve density matrix from rho0 to t_final.

        Parameters
        ----------
        rho0 : np.ndarray
            Initial density matrix of shape (d, d).
        t_final : float
            Final time.
        n_steps : int
            Number of time steps.

        Returns
        -------
        LindbladResult
            Evolution result with recorded states.
        """
        rho0 = np.asarray(rho0, dtype=np.complex128)
        if rho0.ndim == 1:
            # Interpret as pure state and convert to density matrix
            rho0 = np.outer(rho0, rho0.conj())

        if self.method == "rk4":
            return self._evolve_rk4(rho0, t_final, n_steps)
        else:
            return self._evolve_exact(rho0, t_final, n_steps)

    def _evolve_rk4(
        self,
        rho0: np.ndarray,
        t_final: float,
        n_steps: int,
    ) -> LindbladResult:
        """RK4 integration of the master equation.

        Fourth-order Runge-Kutta applied to drho/dt = L(rho), where L
        is evaluated via the direct matrix form (not the superoperator).
        """
        dt = t_final / n_steps
        times = np.linspace(0, t_final, n_steps + 1)
        states = [rho0.copy()]
        rho = rho0.copy()

        for _ in range(n_steps):
            k1 = self.equation.drho_dt(rho)
            k2 = self.equation.drho_dt(rho + 0.5 * dt * k1)
            k3 = self.equation.drho_dt(rho + 0.5 * dt * k2)
            k4 = self.equation.drho_dt(rho + dt * k3)
            rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            states.append(rho.copy())

        return LindbladResult(times=times, states=states)

    def _evolve_exact(
        self,
        rho0: np.ndarray,
        t_final: float,
        n_steps: int,
    ) -> LindbladResult:
        """Exact evolution using matrix exponential of Liouvillian.

        Diagonalises the Liouvillian L and computes:
            vec(rho(t)) = exp(L*t) vec(rho(0))

        using eigendecomposition: exp(L*t) = V diag(e^{lambda_k t}) V^{-1}.
        """
        d = self.equation.dim
        L = self.equation.liouvillian()

        # Diagonalise the (generally non-Hermitian) Liouvillian
        eigenvalues, V = np.linalg.eig(L)
        V_inv = np.linalg.inv(V)

        vec_rho0 = rho0.ravel()
        coeffs = V_inv @ vec_rho0

        times = np.linspace(0, t_final, n_steps + 1)
        states = []

        for t in times:
            exp_lambda = np.exp(eigenvalues * t)
            vec_rho_t = V @ (exp_lambda * coeffs)
            rho_t = vec_rho_t.reshape(d, d)
            # Enforce Hermiticity (numerical cleanup)
            rho_t = 0.5 * (rho_t + rho_t.conj().T)
            states.append(rho_t)

        return LindbladResult(times=times, states=states)

    def steady_state(self, method: str = "eigenvalue") -> np.ndarray:
        """Find the steady state rho_ss where L rho_ss = 0.

        Parameters
        ----------
        method : str
            Method to use: ``"eigenvalue"`` finds the null eigenvector
            of the Liouvillian; ``"svd"`` uses SVD to find the null space.

        Returns
        -------
        np.ndarray
            Steady-state density matrix.

        Raises
        ------
        RuntimeError
            If no unique steady state is found.
        """
        d = self.equation.dim
        L = self.equation.liouvillian()

        if method == "eigenvalue":
            eigenvalues, eigenvectors = np.linalg.eig(L)

            # Find eigenvalue closest to zero
            idx = np.argmin(np.abs(eigenvalues))
            if np.abs(eigenvalues[idx]) > 1e-6:
                raise RuntimeError(
                    f"No near-zero eigenvalue found (smallest: "
                    f"{np.abs(eigenvalues[idx]):.2e}). "
                    "System may not have a unique steady state."
                )

            vec_rho_ss = eigenvectors[:, idx]
            rho_ss = vec_rho_ss.reshape(d, d)

        elif method == "svd":
            U, S, Vh = np.linalg.svd(L)
            # Null space corresponds to smallest singular value
            idx = np.argmin(S)
            vec_rho_ss = Vh[idx, :].conj()
            rho_ss = vec_rho_ss.reshape(d, d)

        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'eigenvalue' or 'svd'."
            )

        # Enforce Hermiticity
        rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)

        # Normalize to unit trace
        tr = np.trace(rho_ss)
        if np.abs(tr) < 1e-15:
            raise RuntimeError("Steady state has zero trace.")
        rho_ss = rho_ss / tr

        # Ensure positivity by clipping tiny negative eigenvalues
        eigvals, eigvecs = np.linalg.eigh(rho_ss)
        eigvals = np.maximum(eigvals, 0)
        rho_ss = (eigvecs * eigvals) @ eigvecs.conj().T
        rho_ss = rho_ss / np.trace(rho_ss)

        return rho_ss


# ---------------------------------------------------------------------------
# Standard noise channel constructors
# ---------------------------------------------------------------------------


def _embed_single_qubit_operator(
    op_2x2: np.ndarray,
    qubit_index: int,
    n_qubits: int,
) -> np.ndarray:
    """Embed a single-qubit operator into the full n-qubit Hilbert space.

    Parameters
    ----------
    op_2x2 : np.ndarray
        2x2 operator matrix.
    qubit_index : int
        Which qubit the operator acts on (0-indexed).
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    np.ndarray
        Full-space operator of shape (2^n, 2^n).
    """
    result = np.eye(1, dtype=np.complex128)
    for q in range(n_qubits):
        if q == qubit_index:
            result = np.kron(result, op_2x2)
        else:
            result = np.kron(result, _I)
    return result


def amplitude_damping_operators(
    n_qubits: int,
    gamma: float = 0.1,
) -> List[LindbladOperator]:
    """Create amplitude damping jump operators for n qubits.

    Models spontaneous emission (energy relaxation, T1 decay).
    Each qubit independently decays |1> -> |0> at rate gamma.

    The jump operator for each qubit is sqrt(gamma) * sigma_minus.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    gamma : float
        Decay rate (must be non-negative).

    Returns
    -------
    list[LindbladOperator]
        One jump operator per qubit.
    """
    operators = []
    for q in range(n_qubits):
        Lk = _embed_single_qubit_operator(_SIGMA_MINUS, q, n_qubits)
        operators.append(LindbladOperator(
            operator=Lk,
            rate=gamma,
            label=f"amp_damp_q{q}",
        ))
    return operators


def dephasing_operators(
    n_qubits: int,
    gamma: float = 0.1,
) -> List[LindbladOperator]:
    """Create pure dephasing jump operators for n qubits.

    Models loss of quantum coherence without energy exchange (T2 decay).
    The jump operator for each qubit is sqrt(gamma/2) * sigma_z.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    gamma : float
        Dephasing rate.

    Returns
    -------
    list[LindbladOperator]
        One jump operator per qubit.
    """
    operators = []
    for q in range(n_qubits):
        Lk = _embed_single_qubit_operator(_Z, q, n_qubits)
        operators.append(LindbladOperator(
            operator=Lk,
            rate=gamma / 2.0,
            label=f"dephase_q{q}",
        ))
    return operators


def depolarizing_operators(
    n_qubits: int,
    gamma: float = 0.1,
) -> List[LindbladOperator]:
    """Create depolarizing jump operators for n qubits.

    The depolarizing channel drives any state toward the maximally
    mixed state I/d.  It is implemented with three jump operators per
    qubit: sqrt(gamma/4) * {X, Y, Z}.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    gamma : float
        Depolarizing rate.

    Returns
    -------
    list[LindbladOperator]
        Three jump operators per qubit (X, Y, Z channels).
    """
    operators = []
    for q in range(n_qubits):
        for pauli_label, pauli_mat in [("X", _X), ("Y", _Y), ("Z", _Z)]:
            Lk = _embed_single_qubit_operator(pauli_mat, q, n_qubits)
            operators.append(LindbladOperator(
                operator=Lk,
                rate=gamma / 4.0,
                label=f"depol_{pauli_label}_q{q}",
            ))
    return operators


def thermal_operators(
    n_qubits: int,
    gamma: float = 0.1,
    n_thermal: float = 0.1,
) -> List[LindbladOperator]:
    """Create thermal bath (emission + absorption) operators.

    Models coupling to a thermal reservoir with mean occupation number
    n_thermal.  Emission rate is gamma * (1 + n_thermal), absorption
    rate is gamma * n_thermal.

    For n_thermal = 0 this reduces to pure amplitude damping.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    gamma : float
        Base coupling rate.
    n_thermal : float
        Mean thermal photon number (Bose-Einstein occupation).

    Returns
    -------
    list[LindbladOperator]
        Two operators per qubit (emission and absorption).
    """
    if n_thermal < 0:
        raise ValueError(f"Thermal occupation must be non-negative, got {n_thermal}.")

    operators = []
    for q in range(n_qubits):
        # Emission (|1> -> |0>): rate = gamma * (1 + n_th)
        L_emit = _embed_single_qubit_operator(_SIGMA_MINUS, q, n_qubits)
        operators.append(LindbladOperator(
            operator=L_emit,
            rate=gamma * (1.0 + n_thermal),
            label=f"thermal_emit_q{q}",
        ))

        # Absorption (|0> -> |1>): rate = gamma * n_th
        if n_thermal > 1e-15:
            L_absorb = _embed_single_qubit_operator(_SIGMA_PLUS, q, n_qubits)
            operators.append(LindbladOperator(
                operator=L_absorb,
                rate=gamma * n_thermal,
                label=f"thermal_absorb_q{q}",
            ))

    return operators


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def create_lindblad_equation(
    hamiltonian: np.ndarray,
    noise_type: str = "dephasing",
    gamma: float = 0.1,
    n_thermal: float = 0.0,
) -> LindbladMasterEquation:
    """Convenience constructor for common noise models.

    Parameters
    ----------
    hamiltonian : np.ndarray
        System Hamiltonian.
    noise_type : str
        One of: ``"amplitude_damping"``, ``"dephasing"``,
        ``"depolarizing"``, ``"thermal"``.
    gamma : float
        Noise rate.
    n_thermal : float
        Thermal occupation (only used for ``"thermal"``).

    Returns
    -------
    LindbladMasterEquation
    """
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    d = hamiltonian.shape[0]
    n_qubits = int(math.log2(d))
    if 2 ** n_qubits != d:
        raise ValueError(
            f"Hamiltonian dimension {d} is not a power of 2."
        )

    factory = {
        "amplitude_damping": lambda: amplitude_damping_operators(n_qubits, gamma),
        "dephasing": lambda: dephasing_operators(n_qubits, gamma),
        "depolarizing": lambda: depolarizing_operators(n_qubits, gamma),
        "thermal": lambda: thermal_operators(n_qubits, gamma, n_thermal),
    }

    if noise_type not in factory:
        raise ValueError(
            f"Unknown noise type '{noise_type}'. "
            f"Options: {list(factory.keys())}"
        )

    return LindbladMasterEquation(
        hamiltonian=hamiltonian,
        jump_operators=factory[noise_type](),
    )
