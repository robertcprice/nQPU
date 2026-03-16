"""Variational quantum time evolution methods.

Implements variational approaches to quantum dynamics that parametrize
the quantum state and evolve the parameters rather than the full
exponentially-large state vector:

- **QITE** (Quantum Imaginary Time Evolution): Ground state preparation
  via imaginary time evolution using McLachlan's variational principle.
- **VarQTE** (Variational Quantum Time Evolution): Real-time dynamics
  with variational parameter updates.
- **PVQD** (Projected Variational Quantum Dynamics): Step-by-step
  Trotter + variational projection for real-time evolution.

References:
    - Motta et al., Nature Physics 16, 205 (2020) [QITE]
    - Yuan et al., Quantum 3, 191 (2019) [VarQTE]
    - Barison et al., Quantum 5, 512 (2021) [PVQD]
    - McLachlan, Mol. Phys. 8, 39 (1964) [Variational principle]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .hamiltonians import SparsePauliHamiltonian


# ---------------------------------------------------------------------------
# Ansatz circuits for variational dynamics
# ---------------------------------------------------------------------------


class VariationalAnsatz:
    """Parametric ansatz circuit for variational time evolution.

    Implements a hardware-efficient ansatz consisting of alternating
    layers of single-qubit rotations and entangling (CNOT) gates.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of rotation + entangling layers.
    """

    def __init__(self, n_qubits: int, n_layers: int = 2) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits
        # 3 rotation angles per qubit per layer (Rx, Ry, Rz)
        self.n_params = 3 * n_qubits * n_layers

    def _rotation_gate(self, axis: str, theta: float) -> np.ndarray:
        """Single-qubit rotation gate."""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        if axis == "x":
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
        elif axis == "y":
            return np.array([[c, -s], [s, c]], dtype=np.complex128)
        else:  # z
            return np.array(
                [[c - 1j * s, 0], [0, c + 1j * s]], dtype=np.complex128
            )

    def _cnot(self, control: int, target: int) -> np.ndarray:
        """CNOT gate acting on full Hilbert space."""
        dim = self.dim
        U = np.zeros((dim, dim), dtype=np.complex128)
        for basis_state in range(dim):
            ctrl_bit = (basis_state >> (self.n_qubits - 1 - control)) & 1
            if ctrl_bit == 0:
                U[basis_state, basis_state] = 1.0
            else:
                # Flip target bit
                new_state = basis_state ^ (1 << (self.n_qubits - 1 - target))
                U[new_state, basis_state] = 1.0
        return U

    def _single_qubit_gate_full(
        self, qubit: int, gate_2x2: np.ndarray
    ) -> np.ndarray:
        """Embed a 2x2 gate into the full Hilbert space."""
        ops = []
        for q in range(self.n_qubits):
            ops.append(gate_2x2 if q == qubit else np.eye(2, dtype=np.complex128))
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def circuit_unitary(self, params: np.ndarray) -> np.ndarray:
        """Build the unitary matrix for given parameters.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector of length ``n_params``.

        Returns
        -------
        np.ndarray
            Unitary matrix of shape ``(dim, dim)``.
        """
        params = np.asarray(params, dtype=float)
        assert len(params) == self.n_params

        U = np.eye(self.dim, dtype=np.complex128)
        p_idx = 0

        for layer in range(self.n_layers):
            # Single-qubit rotations
            for q in range(self.n_qubits):
                for axis in ["x", "y", "z"]:
                    gate = self._rotation_gate(axis, params[p_idx])
                    U = self._single_qubit_gate_full(q, gate) @ U
                    p_idx += 1

            # Entangling layer (linear chain of CNOTs)
            for q in range(self.n_qubits - 1):
                U = self._cnot(q, q + 1) @ U

        return U

    def state(self, params: np.ndarray) -> np.ndarray:
        """Prepare the ansatz state |psi(theta)> = U(theta)|0...0>.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Prepared state vector.
        """
        psi0 = np.zeros(self.dim, dtype=np.complex128)
        psi0[0] = 1.0
        return self.circuit_unitary(params) @ psi0

    def gradient(
        self, params: np.ndarray, observable: np.ndarray
    ) -> np.ndarray:
        """Compute parameter gradients of <psi(theta)|O|psi(theta)>
        via parameter shift rule.

        Parameters
        ----------
        params : np.ndarray
            Current parameters.
        observable : np.ndarray
            Observable matrix.

        Returns
        -------
        np.ndarray
            Gradient vector.
        """
        params = np.asarray(params, dtype=float)
        grad = np.zeros(len(params))
        shift = math.pi / 2

        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += shift
            psi_plus = self.state(p_plus)
            e_plus = float(np.real(psi_plus.conj() @ observable @ psi_plus))

            p_minus = params.copy()
            p_minus[i] -= shift
            psi_minus = self.state(p_minus)
            e_minus = float(np.real(psi_minus.conj() @ observable @ psi_minus))

            grad[i] = (e_plus - e_minus) / 2.0

        return grad


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class VariationalResult:
    """Result of a variational time evolution.

    Attributes
    ----------
    times : np.ndarray
        Time points.
    states : list[np.ndarray]
        State vectors at each time.
    energies : list[float]
        Energy expectation values.
    parameters : list[np.ndarray]
        Variational parameters at each time step.
    converged : bool
        Whether the algorithm converged.
    final_energy : float
        Energy at the final time.
    """

    times: np.ndarray
    states: List[np.ndarray]
    energies: List[float]
    parameters: List[np.ndarray]
    converged: bool = False
    final_energy: float = 0.0


# ---------------------------------------------------------------------------
# QITE (Quantum Imaginary Time Evolution)
# ---------------------------------------------------------------------------


class QITE:
    """Quantum Imaginary Time Evolution for ground state preparation.

    Implements McLachlan's variational principle for imaginary time:

        d|psi>/d(tau) = -(H - E)|psi>

    where tau is imaginary time and E = <psi|H|psi>.

    The state is normalised at each step, and the energy monotonically
    decreases until the ground state is reached.

    For small systems, this operates directly on the state vector without
    a parametric ansatz, using the exact imaginary time propagator.

    Parameters
    ----------
    hamiltonian : SparsePauliHamiltonian
        The target Hamiltonian.
    dt : float
        Imaginary time step.
    """

    def __init__(
        self,
        hamiltonian: SparsePauliHamiltonian,
        dt: float = 0.05,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.H = hamiltonian.matrix()
        self.dt = dt
        self.dim = hamiltonian.dim

    def step(self, psi: np.ndarray) -> Tuple[np.ndarray, float]:
        """Perform one imaginary time step.

        Applies exp(-H*dt)|psi> and renormalises.

        Parameters
        ----------
        psi : np.ndarray
            Current state.

        Returns
        -------
        psi_new : np.ndarray
            Updated, normalised state.
        energy : float
            Energy <psi_new|H|psi_new>.
        """
        # Exact imaginary-time propagator via eigendecomposition
        evals, evecs = np.linalg.eigh(self.H)
        coeffs = evecs.conj().T @ psi
        # exp(-E_k * dt) -- imaginary time
        weights = np.exp(-evals * self.dt)
        psi_new = evecs @ (weights * coeffs)

        # Normalise
        norm = np.linalg.norm(psi_new)
        if norm < 1e-30:
            raise RuntimeError("State collapsed to zero during QITE.")
        psi_new = psi_new / norm

        energy = float(np.real(psi_new.conj() @ self.H @ psi_new))
        return psi_new, energy

    def evolve(
        self,
        psi0: Optional[np.ndarray] = None,
        n_steps: int = 200,
        tol: float = 1e-8,
    ) -> VariationalResult:
        """Run QITE until convergence or max steps.

        Parameters
        ----------
        psi0 : np.ndarray or None
            Initial state.  If None, uses |+>^n (equal superposition).
        n_steps : int
            Maximum number of imaginary time steps.
        tol : float
            Energy convergence tolerance.

        Returns
        -------
        VariationalResult
        """
        if psi0 is None:
            psi = np.ones(self.dim, dtype=np.complex128) / math.sqrt(self.dim)
        else:
            psi = np.asarray(psi0, dtype=np.complex128).ravel()
            psi = psi / np.linalg.norm(psi)

        times = [0.0]
        states = [psi.copy()]
        energies = [float(np.real(psi.conj() @ self.H @ psi))]
        converged = False

        for step_idx in range(1, n_steps + 1):
            psi, energy = self.step(psi)
            tau = step_idx * self.dt
            times.append(tau)
            states.append(psi.copy())
            energies.append(energy)

            # Check convergence
            if step_idx > 1 and abs(energies[-1] - energies[-2]) < tol:
                converged = True
                break

        return VariationalResult(
            times=np.array(times),
            states=states,
            energies=energies,
            parameters=[],  # No variational parameters in exact QITE
            converged=converged,
            final_energy=energies[-1],
        )


# ---------------------------------------------------------------------------
# VarQTE (Variational Quantum Time Evolution)
# ---------------------------------------------------------------------------


class VarQTE:
    """Variational Quantum Time Evolution for real-time dynamics.

    Uses McLachlan's variational principle for real time:

        A * d(theta)/dt = C

    where A_ij = Re[<d_i psi|d_j psi>] is the quantum Fisher
    information matrix and C_i = -Im[<d_i psi|H|psi>].

    Parameters
    ----------
    hamiltonian : SparsePauliHamiltonian
        System Hamiltonian.
    ansatz : VariationalAnsatz
        Parametric circuit.
    dt : float
        Time step for parameter updates.
    regularization : float
        Tikhonov regularization for the linear system.
    """

    def __init__(
        self,
        hamiltonian: SparsePauliHamiltonian,
        ansatz: VariationalAnsatz,
        dt: float = 0.01,
        regularization: float = 1e-6,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.H = hamiltonian.matrix()
        self.ansatz = ansatz
        self.dt = dt
        self.reg = regularization

    def _compute_qfi_matrix(self, params: np.ndarray) -> np.ndarray:
        """Compute the quantum Fisher information matrix A_ij.

        A_ij = Re[ <d_i psi | d_j psi> ]

        Uses finite-difference approximation for the derivatives.
        """
        n_p = len(params)
        epsilon = 1e-5
        psi0 = self.ansatz.state(params)

        # Compute |d_i psi> via finite differences
        dpsi = []
        for i in range(n_p):
            p_plus = params.copy()
            p_plus[i] += epsilon
            p_minus = params.copy()
            p_minus[i] -= epsilon
            dpsi_i = (self.ansatz.state(p_plus) - self.ansatz.state(p_minus)) / (
                2 * epsilon
            )
            dpsi.append(dpsi_i)

        # Build A matrix
        A = np.zeros((n_p, n_p))
        for i in range(n_p):
            for j in range(n_p):
                A[i, j] = float(np.real(dpsi[i].conj() @ dpsi[j]))

        return A

    def _compute_rhs(self, params: np.ndarray) -> np.ndarray:
        """Compute the right-hand side vector C_i = -Im[<d_i psi|H|psi>]."""
        n_p = len(params)
        epsilon = 1e-5
        psi0 = self.ansatz.state(params)
        H_psi = self.H @ psi0

        C = np.zeros(n_p)
        for i in range(n_p):
            p_plus = params.copy()
            p_plus[i] += epsilon
            p_minus = params.copy()
            p_minus[i] -= epsilon
            dpsi_i = (self.ansatz.state(p_plus) - self.ansatz.state(p_minus)) / (
                2 * epsilon
            )
            C[i] = -float(np.imag(dpsi_i.conj() @ H_psi))

        return C

    def step(self, params: np.ndarray) -> np.ndarray:
        """Perform one variational time step.

        Solves A * dtheta/dt = C and updates theta -> theta + dt * dtheta/dt.

        Parameters
        ----------
        params : np.ndarray
            Current parameters.

        Returns
        -------
        np.ndarray
            Updated parameters.
        """
        A = self._compute_qfi_matrix(params)
        C = self._compute_rhs(params)

        # Regularize
        A += self.reg * np.eye(len(params))

        # Solve linear system
        dtheta_dt = np.linalg.solve(A, C)

        return params + self.dt * dtheta_dt

    def evolve(
        self,
        params0: np.ndarray,
        t_final: float,
        n_steps: Optional[int] = None,
    ) -> VariationalResult:
        """Run variational real-time evolution.

        Parameters
        ----------
        params0 : np.ndarray
            Initial variational parameters.
        t_final : float
            Total evolution time.
        n_steps : int or None
            Number of steps.  Default: int(t_final / dt).

        Returns
        -------
        VariationalResult
        """
        if n_steps is None:
            n_steps = max(1, int(math.ceil(t_final / self.dt)))

        dt_actual = t_final / n_steps
        original_dt = self.dt
        self.dt = dt_actual

        params = np.asarray(params0, dtype=float).copy()

        times = [0.0]
        psi = self.ansatz.state(params)
        states = [psi.copy()]
        energies = [float(np.real(psi.conj() @ self.H @ psi))]
        param_history = [params.copy()]

        for step_idx in range(1, n_steps + 1):
            params = self.step(params)
            psi = self.ansatz.state(params)
            energy = float(np.real(psi.conj() @ self.H @ psi))

            times.append(step_idx * dt_actual)
            states.append(psi.copy())
            energies.append(energy)
            param_history.append(params.copy())

        self.dt = original_dt

        return VariationalResult(
            times=np.array(times),
            states=states,
            energies=energies,
            parameters=param_history,
            converged=True,
            final_energy=energies[-1],
        )


# ---------------------------------------------------------------------------
# PVQD (Projected Variational Quantum Dynamics)
# ---------------------------------------------------------------------------


class PVQD:
    """Projected Variational Quantum Dynamics.

    At each time step:
    1. Apply a Trotter step: |phi> = e^{-iH dt} |psi(theta)>
    2. Optimize parameters to maximize |<psi(theta')|phi>|^2

    This projects the Trotterised evolution back onto the ansatz manifold
    at each step, combining the accuracy of Trotter with the compactness
    of variational representations.

    Parameters
    ----------
    hamiltonian : SparsePauliHamiltonian
        System Hamiltonian.
    ansatz : VariationalAnsatz
        Parametric circuit.
    dt : float
        Time step for Trotter evolution.
    optimizer_steps : int
        Number of gradient descent steps per projection.
    learning_rate : float
        Learning rate for gradient descent.
    """

    def __init__(
        self,
        hamiltonian: SparsePauliHamiltonian,
        ansatz: VariationalAnsatz,
        dt: float = 0.05,
        optimizer_steps: int = 50,
        learning_rate: float = 0.05,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.H = hamiltonian.matrix()
        self.ansatz = ansatz
        self.dt = dt
        self.optimizer_steps = optimizer_steps
        self.lr = learning_rate

        # Pre-diagonalise for efficient exponentiation
        self._evals, self._evecs = np.linalg.eigh(self.H)

    def _trotter_step(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """Exact time evolution step (for reference/target)."""
        phases = np.exp(-1j * self._evals * dt)
        coeffs = self._evecs.conj().T @ psi
        return self._evecs @ (phases * coeffs)

    def _project(
        self, target: np.ndarray, params_init: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Find parameters maximizing fidelity with target state.

        Uses gradient ascent on the fidelity |<psi(theta)|target>|^2.

        Returns
        -------
        params : np.ndarray
            Optimised parameters.
        fidelity : float
            Achieved fidelity.
        """
        params = params_init.copy()
        shift = math.pi / 2

        for _ in range(self.optimizer_steps):
            psi = self.ansatz.state(params)
            current_fidelity = float(np.abs(psi.conj() @ target) ** 2)

            if current_fidelity > 1.0 - 1e-12:
                break

            # Parameter-shift gradient of fidelity
            grad = np.zeros(len(params))
            for i in range(len(params)):
                p_plus = params.copy()
                p_plus[i] += shift
                psi_plus = self.ansatz.state(p_plus)
                f_plus = float(np.abs(psi_plus.conj() @ target) ** 2)

                p_minus = params.copy()
                p_minus[i] -= shift
                psi_minus = self.ansatz.state(p_minus)
                f_minus = float(np.abs(psi_minus.conj() @ target) ** 2)

                grad[i] = (f_plus - f_minus) / 2.0

            # Gradient ascent (maximizing fidelity)
            params = params + self.lr * grad

        psi_final = self.ansatz.state(params)
        final_fidelity = float(np.abs(psi_final.conj() @ target) ** 2)

        return params, final_fidelity

    def evolve(
        self,
        params0: np.ndarray,
        t_final: float,
        n_steps: Optional[int] = None,
    ) -> VariationalResult:
        """Run PVQD evolution.

        Parameters
        ----------
        params0 : np.ndarray
            Initial variational parameters.
        t_final : float
            Total evolution time.
        n_steps : int or None
            Number of Trotter-then-project steps.

        Returns
        -------
        VariationalResult
        """
        if n_steps is None:
            n_steps = max(1, int(math.ceil(t_final / self.dt)))

        dt_actual = t_final / n_steps
        params = np.asarray(params0, dtype=float).copy()

        times = [0.0]
        psi = self.ansatz.state(params)
        states = [psi.copy()]
        energies = [float(np.real(psi.conj() @ self.H @ psi))]
        param_history = [params.copy()]
        fidelities = []

        for step_idx in range(1, n_steps + 1):
            # 1. Trotter step from current variational state
            psi_current = self.ansatz.state(params)
            target = self._trotter_step(psi_current, dt_actual)
            target = target / np.linalg.norm(target)

            # 2. Project back onto ansatz
            params, fidelity = self._project(target, params)
            fidelities.append(fidelity)

            psi = self.ansatz.state(params)
            energy = float(np.real(psi.conj() @ self.H @ psi))

            times.append(step_idx * dt_actual)
            states.append(psi.copy())
            energies.append(energy)
            param_history.append(params.copy())

        return VariationalResult(
            times=np.array(times),
            states=states,
            energies=energies,
            parameters=param_history,
            converged=all(f > 0.9 for f in fidelities) if fidelities else True,
            final_energy=energies[-1],
        )
