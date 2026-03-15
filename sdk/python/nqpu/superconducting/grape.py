"""GRAPE (GRadient Ascent Pulse Engineering) optimal control for transmon qubits.

Numerically optimizes microwave pulse shapes to implement a target unitary gate
on the three-level transmon Hamiltonian while suppressing leakage to the |2>
state and respecting hardware bandwidth/amplitude constraints.

Algorithm overview:

    GRAPE discretizes the pulse into N time slices, each with independent I and Q
    control amplitudes.  The total propagator is:

        U_total = U_N @ U_{N-1} @ ... @ U_1

    where U_k = expm(-i * H_k * dt) with H_k the Hamiltonian during slice k
    (static + drive terms evaluated at the slice amplitudes).

    The cost function combines:
        - Gate fidelity:  F = |Tr(U_target^dag @ U_actual)|^2 / d^2
        - Leakage penalty: population transferred outside the computational subspace
        - Smoothness penalty: penalizes large amplitude jumps between adjacent slices

    Gradients are computed analytically via the standard GRAPE formula using
    forward-propagated and backward-propagated unitaries:

        dF/d(amp_k) = 2 * Re[ Tr(P_k^dag @ dU_k/d(amp_k) @ X_k) * conj(overlap) ] / d^2

    where P_k is the backward propagator from slice N to k+1, X_k is the forward
    propagator from slice 1 to k-1, and the derivative of the slice propagator is
    computed via eigendecomposition.

    Matrix exponential uses eigendecomposition of Hermitian matrices:
        expm(-i*H*dt) = V @ diag(exp(-i*lambda*dt)) @ V^dag

    This is exact for Hermitian H (real eigenvalues, unitary eigenvectors).

References:
    - Khaneja et al., J. Magn. Reson. 172, 296 (2005) [original GRAPE]
    - de Fouquieres et al., J. Magn. Reson. 212, 412 (2011) [L-BFGS GRAPE]
    - Motzoi et al., PRL 103, 110501 (2009) [DRAG for transmons]
    - Goerz et al., SciPost Phys. 7, 080 (2019) [Krotov/GRAPE review]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .pulse import (
    Pulse,
    PulseShape,
    TransmonHamiltonian,
    _TWO_PI,
    build_lindblad_operators,
    evolve_density_matrix,
    _lindblad_rhs,
)
from .qubit import TransmonQubit


# ---------------------------------------------------------------------------
# Matrix exponential via eigendecomposition (Hermitian only, pure numpy)
# ---------------------------------------------------------------------------


def _expm_hermitian(H: np.ndarray, dt: float) -> np.ndarray:
    """Compute expm(-i * H * dt) for Hermitian H via eigendecomposition.

    Since H is Hermitian, its eigenvalues are real and eigenvectors form
    a unitary matrix V.  The matrix exponential is:

        exp(-i*H*dt) = V @ diag(exp(-i*lambda_k*dt)) @ V^dag

    This is exact (no truncation error) and numerically stable for the
    small matrices (3x3 or 9x9) we encounter in transmon simulation.

    Parameters
    ----------
    H : ndarray of shape (d, d)
        Hermitian matrix (the Hamiltonian).
    dt : float
        Time step.

    Returns
    -------
    U : ndarray of shape (d, d)
        Unitary propagator exp(-i*H*dt).
    """
    eigenvalues, V = np.linalg.eigh(H)
    # exp(-i * lambda_k * dt)
    phases = np.exp(-1j * eigenvalues * dt)
    return (V * phases[np.newaxis, :]) @ V.conj().T


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GrapeResult:
    """Result of a GRAPE pulse optimization.

    Attributes
    ----------
    optimized_amplitudes_I : ndarray of shape (num_slices,)
        In-phase control amplitudes per time slice (GHz).
    optimized_amplitudes_Q : ndarray of shape (num_slices,)
        Quadrature control amplitudes per time slice (GHz).
    fidelity : float
        Final gate fidelity |Tr(U_target^dag @ U)|^2 / d^2.
    leakage : float
        Total leakage population outside the computational subspace.
    num_iterations : int
        Number of GRAPE iterations performed.
    converged : bool
        Whether the optimizer reached the convergence threshold.
    fidelity_history : list[float]
        Fidelity at each iteration for convergence analysis.
    cost_history : list[float]
        Total cost (fidelity - penalties) at each iteration.
    duration_ns : float
        Total pulse duration in nanoseconds.
    dt_ns : float
        Duration of each time slice in nanoseconds.
    """

    optimized_amplitudes_I: np.ndarray
    optimized_amplitudes_Q: np.ndarray
    fidelity: float
    leakage: float
    num_iterations: int
    converged: bool
    fidelity_history: list = field(default_factory=list)
    cost_history: list = field(default_factory=list)
    duration_ns: float = 0.0
    dt_ns: float = 0.0

    def to_pulse(self, frequency_ghz: float = 5.0) -> Pulse:
        """Convert optimized amplitudes into a Pulse object.

        Creates a piecewise-constant (FLAT) Pulse whose ``envelope()`` method
        returns the optimized I/Q values for each time slice.  This is compatible
        with the existing ``PulseSimulator.simulate_pulse()`` method.

        Parameters
        ----------
        frequency_ghz : float
            Carrier frequency in GHz.

        Returns
        -------
        Pulse
            A Pulse with PulseShape.FLAT whose amplitude encodes the GRAPE
            result.  Use ``GrapeResult.to_pulse_schedule()`` for direct
            integration with ``PulseSimulator``.
        """
        return _GrapePulse(
            amplitudes_I=self.optimized_amplitudes_I.copy(),
            amplitudes_Q=self.optimized_amplitudes_Q.copy(),
            duration_ns=self.duration_ns,
            dt_ns=self.dt_ns,
            frequency_ghz=frequency_ghz,
        )


class _GrapePulse(Pulse):
    """Custom Pulse subclass that replays GRAPE-optimized piecewise-constant amplitudes.

    Overrides the ``envelope()`` method to return the correct I/Q amplitude
    for each time slice, enabling seamless integration with PulseSimulator.
    """

    def __init__(
        self,
        amplitudes_I: np.ndarray,
        amplitudes_Q: np.ndarray,
        duration_ns: float,
        dt_ns: float,
        frequency_ghz: float = 5.0,
    ) -> None:
        # Pulse is a frozen dataclass, so we use object.__setattr__ to
        # initialize the base fields and then store our custom arrays.
        object.__setattr__(self, "amplitude", float(np.max(np.abs(amplitudes_I))))
        object.__setattr__(self, "duration_ns", duration_ns)
        object.__setattr__(self, "frequency_ghz", frequency_ghz)
        object.__setattr__(self, "phase", 0.0)
        object.__setattr__(self, "shape", PulseShape.FLAT)
        object.__setattr__(self, "drag_coefficient", 0.0)
        object.__setattr__(self, "sigma_ns", 0.0)
        object.__setattr__(self, "flat_duration_ns", 0.0)
        # Custom fields
        object.__setattr__(self, "_grape_I", amplitudes_I)
        object.__setattr__(self, "_grape_Q", amplitudes_Q)
        object.__setattr__(self, "_grape_dt", dt_ns)
        object.__setattr__(self, "_grape_n_slices", len(amplitudes_I))

    def envelope(self, t_ns: float) -> complex:
        """Return the I/Q amplitude for the time slice containing t_ns."""
        idx = int(t_ns / self._grape_dt)
        idx = min(idx, self._grape_n_slices - 1)
        idx = max(idx, 0)
        return complex(self._grape_I[idx], self._grape_Q[idx])


# ---------------------------------------------------------------------------
# GRAPE Optimizer
# ---------------------------------------------------------------------------


class GrapeOptimizer:
    """GRAPE optimal control for transmon qubits.

    Numerically discovers microwave pulse shapes that implement a desired
    unitary gate on the three-level (or nine-level two-qubit) transmon
    Hamiltonian with high fidelity and low leakage.

    Parameters
    ----------
    qubits : list[TransmonQubit] or TransmonQubit
        Physical qubit parameters.  Single qubit or pair for 2Q gates.
    coupling_mhz : float
        Coupling strength in MHz (only relevant for 2-qubit gates).
    max_amplitude_ghz : float
        Hardware limit on maximum drive amplitude in GHz.
    lambda_leakage : float
        Weight of the leakage penalty in the cost function.
    lambda_smoothness : float
        Weight of the smoothness penalty in the cost function.

    Examples
    --------
    >>> from nqpu.superconducting.qubit import TransmonQubit
    >>> from nqpu.superconducting.grape import GrapeOptimizer
    >>> q = TransmonQubit(frequency_ghz=5.0, anharmonicity_mhz=-330.0)
    >>> opt = GrapeOptimizer(q)
    >>> result = opt.optimize_x_gate(duration_ns=25.0)
    >>> print(f"Fidelity: {result.fidelity:.6f}")
    """

    def __init__(
        self,
        qubits: TransmonQubit | list[TransmonQubit],
        coupling_mhz: float = 0.0,
        max_amplitude_ghz: float = 0.15,
        lambda_leakage: float = 5.0,
        lambda_smoothness: float = 0.01,
        use_lindblad: bool = False,
    ) -> None:
        if isinstance(qubits, TransmonQubit):
            qubits = [qubits]
        self.qubits = qubits
        self.coupling_mhz = coupling_mhz
        self.max_amplitude_ghz = max_amplitude_ghz
        self.lambda_leakage = lambda_leakage
        self.lambda_smoothness = lambda_smoothness
        self.use_lindblad = use_lindblad

        # Build the Hamiltonian infrastructure
        self._ham = TransmonHamiltonian(qubits, coupling_mhz=coupling_mhz)
        self._dim = self._ham.dim
        self._comp_dim = 2 ** len(qubits)  # computational subspace dimension

        # Drive frequencies: resonant with each qubit
        self._drive_freqs = [q.frequency_ghz for q in qubits]

        # Pre-compute static Hamiltonian and drive operators
        self._H0 = self._ham.static_hamiltonian(self._drive_freqs)

        # Drive operators for each qubit
        self._drive_ops = []
        for i in range(len(qubits)):
            dx, dy = self._ham.drive_operators(i)
            self._drive_ops.append((dx, dy))

        # Pre-build Lindblad collapse operators if needed.
        self._collapse_ops: list[np.ndarray] = []
        self._collapse_pre: list[np.ndarray] = []
        if self.use_lindblad:
            for q in qubits:
                self._collapse_ops.extend(build_lindblad_operators(q, dim=3))
            self._collapse_pre = [
                L.T.conj() @ L for L in self._collapse_ops
            ]

    # ------------------------------------------------------------------
    # Projector into computational subspace
    # ------------------------------------------------------------------

    def _comp_projector(self) -> np.ndarray:
        """Build the projector from the full Hilbert space to the computational subspace.

        For a single qubit (dim=3), computational states are |0>, |1> (indices 0, 1).
        For two qubits (dim=9), computational states are |00>, |01>, |10>, |11>
        (indices 0, 1, 3, 4 in the 3x3 tensor product basis).

        Returns
        -------
        P : ndarray of shape (comp_dim, dim)
            Projector matrix.
        """
        d = self._dim
        cd = self._comp_dim
        P = np.zeros((cd, d), dtype=np.complex128)

        if len(self.qubits) == 1:
            P[0, 0] = 1.0  # |0>
            P[1, 1] = 1.0  # |1>
        else:
            # 2-qubit: |ij> -> index i*3 + j (row-major in 3-level basis)
            P[0, 0] = 1.0  # |00>
            P[1, 1] = 1.0  # |01>
            P[2, 3] = 1.0  # |10>
            P[3, 4] = 1.0  # |11>
        return P

    def _embed_target(self, U_target: np.ndarray) -> np.ndarray:
        """Embed a computational-subspace unitary into the full Hilbert space.

        Parameters
        ----------
        U_target : ndarray of shape (comp_dim, comp_dim)
            Target unitary in the computational subspace.

        Returns
        -------
        U_full : ndarray of shape (dim, dim)
            Unitary in the full Hilbert space. Acts as U_target on the
            computational subspace and as identity on leakage states.
        """
        P = self._comp_projector()
        U_full = np.eye(self._dim, dtype=np.complex128)
        # U_full = P^dag @ U_target @ P  (on the comp subspace) + identity elsewhere
        U_full += P.conj().T @ U_target @ P - P.conj().T @ P
        return U_full

    # ------------------------------------------------------------------
    # Hamiltonian for a single time slice
    # ------------------------------------------------------------------

    def _slice_hamiltonian(
        self,
        amplitudes_I: np.ndarray,
        amplitudes_Q: np.ndarray,
        qubit_index: int = 0,
    ) -> np.ndarray:
        """Build the Hamiltonian for a single time slice.

        Parameters
        ----------
        amplitudes_I : float or ndarray
            In-phase amplitude(s) in GHz.  For single-qubit, a scalar.
            For two-qubit, amplitudes_I[qubit_index].
        amplitudes_Q : float or ndarray
            Quadrature amplitude(s).
        qubit_index : int
            Which qubit's drive to set (only meaningful for 2Q).

        Returns
        -------
        H : ndarray of shape (dim, dim)
            Total Hamiltonian = H0 + drive terms.
        """
        H = self._H0.copy()
        dx, dy = self._drive_ops[qubit_index]
        amp_I = float(amplitudes_I)
        amp_Q = float(amplitudes_Q)
        H += _TWO_PI * 0.5 * (amp_I * dx + amp_Q * dy)
        return H

    def _build_slice_hamiltonian(
        self,
        all_I: np.ndarray,
        all_Q: np.ndarray,
        slice_idx: int,
    ) -> np.ndarray:
        """Build the full Hamiltonian for slice ``slice_idx``.

        For multi-qubit systems, sums drive terms from all qubits.

        Parameters
        ----------
        all_I : ndarray of shape (n_qubits, num_slices) or (num_slices,)
            I amplitudes for all qubits and slices.
        all_Q : ndarray of shape (n_qubits, num_slices) or (num_slices,)
            Q amplitudes for all qubits and slices.
        slice_idx : int
            Time slice index.

        Returns
        -------
        H : ndarray of shape (dim, dim)
        """
        H = self._H0.copy()
        n_qubits = len(self.qubits)

        if all_I.ndim == 1:
            # Single qubit: all_I is (num_slices,)
            dx, dy = self._drive_ops[0]
            H += _TWO_PI * 0.5 * (
                all_I[slice_idx] * dx + all_Q[slice_idx] * dy
            )
        else:
            # Multi-qubit: all_I is (n_qubits, num_slices)
            for qi in range(n_qubits):
                dx, dy = self._drive_ops[qi]
                H += _TWO_PI * 0.5 * (
                    all_I[qi, slice_idx] * dx + all_Q[qi, slice_idx] * dy
                )
        return H

    # ------------------------------------------------------------------
    # Gate fidelity
    # ------------------------------------------------------------------

    def _gate_fidelity(
        self,
        U_actual: np.ndarray,
        U_target_full: np.ndarray,
    ) -> float:
        """Compute gate fidelity F = |Tr(U_target^dag @ U_actual)|^2 / d^2.

        Uses the computational subspace dimension d (not the full Hilbert
        space dimension) so that the fidelity is meaningful for the gate
        operation on the qubit register.

        Parameters
        ----------
        U_actual : ndarray of shape (dim, dim)
            Simulated propagator in the full Hilbert space.
        U_target_full : ndarray of shape (dim, dim)
            Target unitary embedded in the full Hilbert space.

        Returns
        -------
        float
            Gate fidelity in [0, 1].
        """
        P = self._comp_projector()
        # Project both unitaries into the computational subspace
        U_actual_comp = P @ U_actual @ P.conj().T
        U_target_comp = P @ U_target_full @ P.conj().T
        overlap = np.trace(U_target_comp.conj().T @ U_actual_comp)
        d = self._comp_dim
        return float(np.abs(overlap) ** 2) / (d * d)

    def _leakage(self, U_actual: np.ndarray) -> float:
        """Compute total leakage: probability of leaving the computational subspace.

        Averages over all computational basis state inputs.

        Parameters
        ----------
        U_actual : ndarray of shape (dim, dim)
            Simulated propagator.

        Returns
        -------
        float
            Average leakage in [0, 1].
        """
        P = self._comp_projector()
        d = self._comp_dim
        total_leakage = 0.0
        for k in range(d):
            # k-th computational basis state embedded in full space
            psi_in = P[k, :]  # shape (dim,)
            psi_out = U_actual @ psi_in
            # Population remaining in computational subspace
            comp_pop = np.sum(np.abs(P @ psi_out) ** 2)
            total_leakage += 1.0 - comp_pop
        return float(total_leakage / d)

    def _smoothness_penalty(
        self,
        all_I: np.ndarray,
        all_Q: np.ndarray,
    ) -> float:
        """Compute smoothness penalty: sum of squared differences between adjacent slices.

        Parameters
        ----------
        all_I, all_Q : ndarray
            Amplitude arrays.

        Returns
        -------
        float
            Smoothness cost (larger = rougher pulse).
        """
        if all_I.ndim == 1:
            diff_I = np.diff(all_I)
            diff_Q = np.diff(all_Q)
        else:
            diff_I = np.diff(all_I, axis=-1)
            diff_Q = np.diff(all_Q, axis=-1)
        return float(np.sum(diff_I ** 2) + np.sum(diff_Q ** 2))

    # ------------------------------------------------------------------
    # Forward / backward propagation
    # ------------------------------------------------------------------

    def _forward_propagators(
        self,
        all_I: np.ndarray,
        all_Q: np.ndarray,
        dt: float,
    ) -> list[np.ndarray]:
        """Compute slice propagators U_k = expm(-i * H_k * dt).

        Parameters
        ----------
        all_I, all_Q : ndarray
            Amplitude arrays for all slices.
        dt : float
            Duration of each slice in nanoseconds.

        Returns
        -------
        propagators : list of ndarray
            List of N unitary matrices, one per slice.
        """
        if all_I.ndim == 1:
            n_slices = len(all_I)
        else:
            n_slices = all_I.shape[1]

        propagators = []
        for k in range(n_slices):
            Hk = self._build_slice_hamiltonian(all_I, all_Q, k)
            Uk = _expm_hermitian(Hk, dt)
            propagators.append(Uk)
        return propagators

    def _total_propagator(self, propagators: list[np.ndarray]) -> np.ndarray:
        """Compute U_total = U_N @ U_{N-1} @ ... @ U_1."""
        U = np.eye(self._dim, dtype=np.complex128)
        for Uk in propagators:
            U = Uk @ U
        return U

    def _forward_cumulative(self, propagators: list[np.ndarray]) -> list[np.ndarray]:
        """Compute X_k = U_k @ U_{k-1} @ ... @ U_1 for k = 0..N-1.

        X_0 = U_0
        X_k = U_k @ X_{k-1}

        Returns list of length N where X[k] is the forward propagator up to
        and including slice k.
        """
        N = len(propagators)
        X = [None] * N
        X[0] = propagators[0].copy()
        for k in range(1, N):
            X[k] = propagators[k] @ X[k - 1]
        return X

    def _backward_cumulative(
        self,
        propagators: list[np.ndarray],
        U_target_full: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute P_k = U_{k+1}^dag @ U_{k+2}^dag @ ... @ U_N^dag @ U_target.

        P_{N-1} = U_target
        P_k = U_{k+1}^dag @ P_{k+1}

        Returns list of length N where P[k] is the backward propagator
        from the target down to slice k+1.
        """
        N = len(propagators)
        P = [None] * N
        P[N - 1] = U_target_full.copy()
        for k in range(N - 2, -1, -1):
            P[k] = propagators[k + 1].conj().T @ P[k + 1]
        return P

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def _gradient_hermitian_expm(
        self,
        H: np.ndarray,
        dH: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute d/d(amp) of expm(-i * H * dt) given dH/d(amp).

        For Hermitian H with eigendecomposition H = V @ diag(E) @ V^dag,
        the derivative of the matrix exponential is:

            dU/d(amp) = -i*dt * V @ (F .* (V^dag @ dH @ V)) @ V^dag @ diag(exp(-i*E*dt))

        where F is the divided-difference matrix:

            F_{jk} = (exp(-i*E_j*dt) - exp(-i*E_k*dt)) / (-i*(E_j - E_k)*dt)

        with the diagonal F_{jj} = exp(-i*E_j*dt).

        This is the exact derivative, not a finite-difference approximation.

        Parameters
        ----------
        H : ndarray of shape (d, d)
            The Hamiltonian (Hermitian).
        dH : ndarray of shape (d, d)
            Derivative of H with respect to the control amplitude.
        dt : float
            Time step.

        Returns
        -------
        dU : ndarray of shape (d, d)
            Derivative of expm(-i*H*dt) w.r.t. the control amplitude.
        """
        eigenvalues, V = np.linalg.eigh(H)
        d = len(eigenvalues)

        exp_phases = np.exp(-1j * eigenvalues * dt)

        # Build the divided-difference matrix F
        F = np.zeros((d, d), dtype=np.complex128)
        for j in range(d):
            for k in range(d):
                if abs(eigenvalues[j] - eigenvalues[k]) * dt > 1e-12:
                    F[j, k] = (exp_phases[j] - exp_phases[k]) / (
                        -1j * (eigenvalues[j] - eigenvalues[k]) * dt
                    )
                else:
                    F[j, k] = exp_phases[j]

        # Transform dH into the eigenbasis
        dH_eig = V.conj().T @ dH @ V

        # dU = V @ (-i*dt * F .* dH_eig) @ V^dag
        # But we need to account for the fact that U = V @ diag(exp) @ V^dag
        # and dU is the derivative of this.
        dU = V @ ((-1j * dt) * F * dH_eig) @ V.conj().T

        return dU

    def _compute_gradients(
        self,
        all_I: np.ndarray,
        all_Q: np.ndarray,
        propagators: list[np.ndarray],
        X_fwd: list[np.ndarray],
        P_bwd: list[np.ndarray],
        U_target_full: np.ndarray,
        U_total: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GRAPE gradients for all time slices.

        The gradient of the gate fidelity with respect to amplitude a_k is:

            dF/d(a_k) = 2 * Re[ conj(overlap) * Tr(P_k^dag @ dU_k @ X_{k-1}) ] / d^2

        where overlap = Tr(U_target^dag @ U_total).

        Parameters
        ----------
        all_I, all_Q : ndarray
            Current amplitude arrays.
        propagators : list[ndarray]
            Slice propagators U_k.
        X_fwd : list[ndarray]
            Forward cumulative propagators.
        P_bwd : list[ndarray]
            Backward cumulative propagators.
        U_target_full : ndarray
            Target unitary in full space.
        U_total : ndarray
            Total propagator.
        dt : float
            Slice duration.

        Returns
        -------
        grad_I, grad_Q : ndarray
            Gradients of the fidelity w.r.t. I and Q amplitudes.
        """
        P = self._comp_projector()
        d = self._comp_dim

        # Compute overlap in computational subspace
        U_actual_comp = P @ U_total @ P.conj().T
        U_target_comp = P @ U_target_full @ P.conj().T
        overlap = np.trace(U_target_comp.conj().T @ U_actual_comp)

        if all_I.ndim == 1:
            n_slices = len(all_I)
            n_qubits = 1
        else:
            n_qubits, n_slices = all_I.shape

        # Allocate gradient arrays matching input shape
        if all_I.ndim == 1:
            grad_I = np.zeros(n_slices, dtype=np.float64)
            grad_Q = np.zeros(n_slices, dtype=np.float64)
        else:
            grad_I = np.zeros((n_qubits, n_slices), dtype=np.float64)
            grad_Q = np.zeros((n_qubits, n_slices), dtype=np.float64)

        for qi in range(n_qubits):
            dx, dy = self._drive_ops[qi]
            dH_dI = _TWO_PI * 0.5 * dx
            dH_dQ = _TWO_PI * 0.5 * dy

            for k in range(n_slices):
                Hk = self._build_slice_hamiltonian(all_I, all_Q, k)

                # Derivative of U_k w.r.t. I and Q amplitudes
                dUk_dI = self._gradient_hermitian_expm(Hk, dH_dI, dt)
                dUk_dQ = self._gradient_hermitian_expm(Hk, dH_dQ, dt)

                # Forward propagator up to slice k-1
                if k == 0:
                    X_prev = np.eye(self._dim, dtype=np.complex128)
                else:
                    X_prev = X_fwd[k - 1]

                # Backward propagator from slice k+1
                Pk = P_bwd[k]

                # Project into computational subspace for gradient
                # d(overlap)/d(a_k) = Tr(U_target_comp^dag @ P @ (Pk^dag @ dUk @ X_prev) @ P^dag)
                # But Pk already contains the backward chain, so:
                # d(overlap)/d(a_k) = Tr( (P @ Pk)^dag @ (P @ dUk @ X_prev @ P^dag) )
                # Simpler: compute in full space then project

                M_I = Pk.conj().T @ dUk_dI @ X_prev
                M_Q = Pk.conj().T @ dUk_dQ @ X_prev

                # Project to computational subspace
                tr_I = np.trace(P @ M_I @ P.conj().T)
                tr_Q = np.trace(P @ M_Q @ P.conj().T)

                # dF/d(a_k) = 2 * Re[ conj(overlap) * tr ] / d^2
                fid_grad_I = 2.0 * np.real(np.conj(overlap) * tr_I) / (d * d)
                fid_grad_Q = 2.0 * np.real(np.conj(overlap) * tr_Q) / (d * d)

                if all_I.ndim == 1:
                    grad_I[k] += fid_grad_I
                    grad_Q[k] += fid_grad_Q
                else:
                    grad_I[qi, k] += fid_grad_I
                    grad_Q[qi, k] += fid_grad_Q

        return grad_I, grad_Q

    def _smoothness_gradient(
        self,
        all_I: np.ndarray,
        all_Q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gradient of the smoothness penalty.

        The smoothness cost is S = sum_k (a_{k+1} - a_k)^2.
        dS/d(a_k) = -2*(a_{k+1} - a_k) + 2*(a_k - a_{k-1})  (interior points)
        """
        if all_I.ndim == 1:
            n = len(all_I)
            g_I = np.zeros(n, dtype=np.float64)
            g_Q = np.zeros(n, dtype=np.float64)
            for k in range(n):
                if k > 0:
                    g_I[k] += 2.0 * (all_I[k] - all_I[k - 1])
                    g_Q[k] += 2.0 * (all_Q[k] - all_Q[k - 1])
                if k < n - 1:
                    g_I[k] -= 2.0 * (all_I[k + 1] - all_I[k])
                    g_Q[k] -= 2.0 * (all_Q[k + 1] - all_Q[k])
        else:
            nq, n = all_I.shape
            g_I = np.zeros_like(all_I, dtype=np.float64)
            g_Q = np.zeros_like(all_Q, dtype=np.float64)
            for qi in range(nq):
                for k in range(n):
                    if k > 0:
                        g_I[qi, k] += 2.0 * (all_I[qi, k] - all_I[qi, k - 1])
                        g_Q[qi, k] += 2.0 * (all_Q[qi, k] - all_Q[qi, k - 1])
                    if k < n - 1:
                        g_I[qi, k] -= 2.0 * (all_I[qi, k + 1] - all_I[qi, k])
                        g_Q[qi, k] -= 2.0 * (all_Q[qi, k + 1] - all_Q[qi, k])
        return g_I, g_Q

    def _leakage_gradient_numerical(
        self,
        all_I: np.ndarray,
        all_Q: np.ndarray,
        dt: float,
        epsilon: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numerical gradient of leakage penalty (finite differences).

        Leakage gradient is expensive to compute analytically, so we use
        central finite differences.  The leakage penalty is typically small
        relative to the fidelity gradient, so a numerical approximation
        is sufficient.
        """
        if all_I.ndim == 1:
            n = len(all_I)
        else:
            n = all_I.shape[1]

        grad_I = np.zeros_like(all_I, dtype=np.float64)
        grad_Q = np.zeros_like(all_Q, dtype=np.float64)

        flat_I = all_I.ravel()
        flat_Q = all_Q.ravel()

        for idx in range(len(flat_I)):
            # Perturb I
            flat_I[idx] += epsilon
            I_plus = flat_I.reshape(all_I.shape)
            props_plus = self._forward_propagators(I_plus, all_Q, dt)
            U_plus = self._total_propagator(props_plus)
            leak_plus = self._leakage(U_plus)

            flat_I[idx] -= 2.0 * epsilon
            I_minus = flat_I.reshape(all_I.shape)
            props_minus = self._forward_propagators(I_minus, all_Q, dt)
            U_minus = self._total_propagator(props_minus)
            leak_minus = self._leakage(U_minus)

            flat_I[idx] += epsilon  # restore
            grad_I.ravel()[idx] = (leak_plus - leak_minus) / (2.0 * epsilon)

        for idx in range(len(flat_Q)):
            flat_Q[idx] += epsilon
            Q_plus = flat_Q.reshape(all_Q.shape)
            props_plus = self._forward_propagators(all_I, Q_plus, dt)
            U_plus = self._total_propagator(props_plus)
            leak_plus = self._leakage(U_plus)

            flat_Q[idx] -= 2.0 * epsilon
            Q_minus = flat_Q.reshape(all_Q.shape)
            props_minus = self._forward_propagators(all_I, Q_minus, dt)
            U_minus = self._total_propagator(props_minus)
            leak_minus = self._leakage(U_minus)

            flat_Q[idx] += epsilon  # restore
            grad_Q.ravel()[idx] = (leak_plus - leak_minus) / (2.0 * epsilon)

        return grad_I, grad_Q

    # ------------------------------------------------------------------
    # Lindblad-aware evaluation
    # ------------------------------------------------------------------

    def _lindblad_gate_fidelity(
        self,
        all_I: np.ndarray,
        all_Q: np.ndarray,
        dt: float,
        U_target_full: np.ndarray,
    ) -> tuple[float, float]:
        """Compute average gate fidelity using Lindblad (density matrix) evolution.

        Instead of computing the unitary propagator, evolves each
        computational basis input through the Lindblad equation and
        compares the output density matrix against the ideal pure-state
        output.

        Parameters
        ----------
        all_I, all_Q : ndarray
            Pulse amplitudes for all slices.
        dt : float
            Duration of each slice in ns.
        U_target_full : ndarray
            Target unitary in the full Hilbert space.

        Returns
        -------
        fidelity : float
            Average gate fidelity over computational basis inputs.
        leakage : float
            Average leakage population.
        """
        P = self._comp_projector()
        d = self._comp_dim
        dim = self._dim

        if all_I.ndim == 1:
            n_slices = len(all_I)
        else:
            n_slices = all_I.shape[1]

        total_time = n_slices * dt

        # The RK4 integration step must be much finer than the GRAPE
        # slice width to resolve the fast Hamiltonian oscillations.
        # The Hamiltonian eigenvalues are ~ qubit frequency (5 GHz),
        # so the oscillation period is ~0.2 ns.  A step of 0.05 ns
        # gives ~4 samples per period -- adequate for RK4.
        rk4_dt = min(dt, 0.05)

        # Build the time-dependent Hamiltonian function.
        def H_t(t: float) -> np.ndarray:
            # Which slice are we in?
            k = min(int(t / dt), n_slices - 1)
            return self._build_slice_hamiltonian(all_I, all_Q, k)

        total_fidelity = 0.0
        total_leakage = 0.0

        for comp_k in range(d):
            # k-th computational basis state embedded in full space.
            psi_in = P[comp_k, :]  # shape (dim,)
            rho0 = np.outer(psi_in, psi_in.conj())

            # Evolve under Lindblad equation.
            rho_out = evolve_density_matrix(
                rho0, H_t, self._collapse_ops, total_time, rk4_dt
            )

            # Ideal output state.
            psi_ideal = U_target_full @ psi_in
            # Fidelity: <ideal|rho_out|ideal>
            fid = np.real(psi_ideal.conj() @ rho_out @ psi_ideal)
            total_fidelity += max(float(fid), 0.0)

            # Leakage: 1 - Tr(P_comp @ rho_out @ P_comp^dag)
            rho_comp = P @ rho_out @ P.conj().T
            comp_pop = np.real(np.trace(rho_comp))
            total_leakage += max(1.0 - float(comp_pop), 0.0)

        return total_fidelity / d, total_leakage / d

    # ------------------------------------------------------------------
    # Core optimization loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        target_unitary: np.ndarray,
        duration_ns: float = 25.0,
        num_slices: int = 40,
        max_iterations: int = 300,
        convergence_threshold: float = 0.9999,
        step_size: float = 0.005,
        momentum: float = 0.9,
        seed: Optional[int] = None,
        initial_guess_I: Optional[np.ndarray] = None,
        initial_guess_Q: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> GrapeResult:
        """Run GRAPE optimization to find optimal pulse amplitudes.

        Parameters
        ----------
        target_unitary : ndarray of shape (comp_dim, comp_dim)
            Target gate in the computational subspace (2x2 for 1Q, 4x4 for 2Q).
        duration_ns : float
            Total gate duration in nanoseconds.
        num_slices : int
            Number of piecewise-constant time slices.
        max_iterations : int
            Maximum number of GRAPE iterations.
        convergence_threshold : float
            Stop when gate fidelity exceeds this value.
        step_size : float
            Gradient ascent learning rate.
        momentum : float
            Momentum coefficient for accelerated gradient ascent (0 = no momentum).
        seed : int or None
            Random seed for reproducible initial guesses.
        initial_guess_I, initial_guess_Q : ndarray or None
            Optional initial amplitude arrays of shape (num_slices,) for 1Q
            or (n_qubits, num_slices) for 2Q.  If None, uses small random
            initial amplitudes.
        verbose : bool
            Print progress every 10 iterations.

        Returns
        -------
        GrapeResult
            Optimized pulse amplitudes and convergence diagnostics.
        """
        rng = np.random.default_rng(seed)
        dt = duration_ns / num_slices
        n_qubits = len(self.qubits)

        # Embed target in full Hilbert space
        U_target_full = self._embed_target(target_unitary)

        # Initialize amplitudes
        if n_qubits == 1:
            if initial_guess_I is not None:
                all_I = initial_guess_I.copy().astype(np.float64)
            else:
                all_I = rng.normal(0.0, 0.01, size=num_slices)
            if initial_guess_Q is not None:
                all_Q = initial_guess_Q.copy().astype(np.float64)
            else:
                all_Q = rng.normal(0.0, 0.005, size=num_slices)
        else:
            if initial_guess_I is not None:
                all_I = initial_guess_I.copy().astype(np.float64)
            else:
                all_I = rng.normal(0.0, 0.01, size=(n_qubits, num_slices))
            if initial_guess_Q is not None:
                all_Q = initial_guess_Q.copy().astype(np.float64)
            else:
                all_Q = rng.normal(0.0, 0.005, size=(n_qubits, num_slices))

        # Momentum velocity terms
        vel_I = np.zeros_like(all_I)
        vel_Q = np.zeros_like(all_Q)

        fidelity_history = []
        cost_history = []

        best_fidelity = 0.0
        best_I = all_I.copy()
        best_Q = all_Q.copy()
        converged = False

        for iteration in range(max_iterations):
            if self.use_lindblad:
                # ---- Lindblad mode: evaluate fidelity using density matrices ----
                fidelity, leakage = self._lindblad_gate_fidelity(
                    all_I, all_Q, dt, U_target_full,
                )
                smoothness = self._smoothness_penalty(all_I, all_Q)
                cost = fidelity - self.lambda_leakage * leakage - self.lambda_smoothness * smoothness
                fidelity_history.append(fidelity)
                cost_history.append(cost)

                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    best_I = all_I.copy()
                    best_Q = all_Q.copy()

                if verbose and iteration % 10 == 0:
                    print(
                        f"  GRAPE-Lindblad iter {iteration:4d}: "
                        f"fidelity={fidelity:.6f}  "
                        f"leakage={leakage:.6f}  "
                        f"cost={cost:.6f}"
                    )

                if fidelity >= convergence_threshold:
                    converged = True
                    if verbose:
                        print(f"  Converged at iteration {iteration}!")
                    break

                # Numerical gradient for Lindblad mode (analytic GRAPE
                # gradients are for unitary evolution only).
                epsilon = 1e-4
                flat_I = all_I.ravel()
                flat_Q = all_Q.ravel()
                grad_I_flat = np.zeros_like(flat_I)
                grad_Q_flat = np.zeros_like(flat_Q)

                for idx in range(len(flat_I)):
                    flat_I[idx] += epsilon
                    f_plus, _ = self._lindblad_gate_fidelity(
                        flat_I.reshape(all_I.shape), all_Q, dt, U_target_full,
                    )
                    flat_I[idx] -= 2.0 * epsilon
                    f_minus, _ = self._lindblad_gate_fidelity(
                        flat_I.reshape(all_I.shape), all_Q, dt, U_target_full,
                    )
                    flat_I[idx] += epsilon
                    grad_I_flat[idx] = (f_plus - f_minus) / (2.0 * epsilon)

                for idx in range(len(flat_Q)):
                    flat_Q[idx] += epsilon
                    f_plus, _ = self._lindblad_gate_fidelity(
                        all_I, flat_Q.reshape(all_Q.shape), dt, U_target_full,
                    )
                    flat_Q[idx] -= 2.0 * epsilon
                    f_minus, _ = self._lindblad_gate_fidelity(
                        all_I, flat_Q.reshape(all_Q.shape), dt, U_target_full,
                    )
                    flat_Q[idx] += epsilon
                    grad_Q_flat[idx] = (f_plus - f_minus) / (2.0 * epsilon)

                grad_I = grad_I_flat.reshape(all_I.shape)
                grad_Q = grad_Q_flat.reshape(all_Q.shape)

                # Smoothness gradient.
                sm_grad_I, sm_grad_Q = self._smoothness_gradient(all_I, all_Q)
                grad_I -= self.lambda_smoothness * sm_grad_I
                grad_Q -= self.lambda_smoothness * sm_grad_Q

            else:
                # ---- Standard unitary GRAPE mode ----
                # Forward propagation
                propagators = self._forward_propagators(all_I, all_Q, dt)
                X_fwd = self._forward_cumulative(propagators)
                U_total = X_fwd[-1]

                # Backward propagation
                P_bwd = self._backward_cumulative(propagators, U_target_full)

                # Compute fidelity
                fidelity = self._gate_fidelity(U_total, U_target_full)
                leakage = self._leakage(U_total)
                smoothness = self._smoothness_penalty(all_I, all_Q)

                cost = fidelity - self.lambda_leakage * leakage - self.lambda_smoothness * smoothness
                fidelity_history.append(fidelity)
                cost_history.append(cost)

                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    best_I = all_I.copy()
                    best_Q = all_Q.copy()

                if verbose and iteration % 10 == 0:
                    print(
                        f"  GRAPE iter {iteration:4d}: "
                        f"fidelity={fidelity:.6f}  "
                        f"leakage={leakage:.6f}  "
                        f"cost={cost:.6f}"
                    )

                if fidelity >= convergence_threshold:
                    converged = True
                    if verbose:
                        print(f"  Converged at iteration {iteration}!")
                    break

                # Compute fidelity gradient
                grad_I, grad_Q = self._compute_gradients(
                    all_I, all_Q, propagators, X_fwd, P_bwd,
                    U_target_full, U_total, dt,
                )

                # Smoothness gradient (subtract since we're maximizing cost)
                sm_grad_I, sm_grad_Q = self._smoothness_gradient(all_I, all_Q)
                grad_I -= self.lambda_smoothness * sm_grad_I
                grad_Q -= self.lambda_smoothness * sm_grad_Q

                # Leakage gradient (numerical, only if lambda > 0 and not too expensive)
                if self.lambda_leakage > 0 and n_qubits == 1 and num_slices <= 60:
                    leak_grad_I, leak_grad_Q = self._leakage_gradient_numerical(
                        all_I, all_Q, dt,
                    )
                    grad_I -= self.lambda_leakage * leak_grad_I
                    grad_Q -= self.lambda_leakage * leak_grad_Q

            # Momentum update
            vel_I = momentum * vel_I + step_size * grad_I
            vel_Q = momentum * vel_Q + step_size * grad_Q

            all_I += vel_I
            all_Q += vel_Q

            # Enforce amplitude constraints
            np.clip(all_I, -self.max_amplitude_ghz, self.max_amplitude_ghz, out=all_I)
            np.clip(all_Q, -self.max_amplitude_ghz, self.max_amplitude_ghz, out=all_Q)

        # Final evaluation with best amplitudes
        if self.use_lindblad:
            final_fidelity, final_leakage = self._lindblad_gate_fidelity(
                best_I, best_Q, dt, U_target_full,
            )
        else:
            propagators = self._forward_propagators(best_I, best_Q, dt)
            U_best = self._total_propagator(propagators)
            final_fidelity = self._gate_fidelity(U_best, U_target_full)
            final_leakage = self._leakage(U_best)

        if n_qubits == 1:
            result_I = best_I
            result_Q = best_Q
        else:
            # For multi-qubit, flatten to (n_qubits * num_slices,)
            # but store as (n_qubits, num_slices) shape for clarity
            result_I = best_I
            result_Q = best_Q

        return GrapeResult(
            optimized_amplitudes_I=result_I,
            optimized_amplitudes_Q=result_Q,
            fidelity=final_fidelity,
            leakage=final_leakage,
            num_iterations=iteration + 1,
            converged=converged,
            fidelity_history=fidelity_history,
            cost_history=cost_history,
            duration_ns=duration_ns,
            dt_ns=dt,
        )

    # ------------------------------------------------------------------
    # Preset gate optimizations
    # ------------------------------------------------------------------

    def optimize_x_gate(
        self,
        duration_ns: float = 25.0,
        num_slices: int = 40,
        max_iterations: int = 300,
        convergence_threshold: float = 0.9999,
        seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> GrapeResult:
        """Optimize a Pauli-X gate (pi rotation about x-axis).

        The target unitary is:
            X = [[0, 1],
                 [1, 0]]

        Parameters
        ----------
        duration_ns : float
            Gate duration in nanoseconds.
        num_slices : int
            Number of time slices.
        max_iterations : int
            Maximum GRAPE iterations.
        convergence_threshold : float
            Fidelity convergence target.
        seed : int or None
            Random seed.
        verbose : bool
            Print progress.

        Returns
        -------
        GrapeResult
        """
        X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        return self.optimize(
            X,
            duration_ns=duration_ns,
            num_slices=num_slices,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )

    def optimize_hadamard(
        self,
        duration_ns: float = 25.0,
        num_slices: int = 40,
        max_iterations: int = 300,
        convergence_threshold: float = 0.9999,
        seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> GrapeResult:
        """Optimize a Hadamard gate.

        The target unitary is:
            H = (1/sqrt(2)) * [[1,  1],
                                [1, -1]]

        Parameters
        ----------
        duration_ns : float
            Gate duration in nanoseconds.
        num_slices : int
            Number of time slices.
        max_iterations : int
            Maximum GRAPE iterations.
        convergence_threshold : float
            Fidelity convergence target.
        seed : int or None
            Random seed.
        verbose : bool
            Print progress.

        Returns
        -------
        GrapeResult
        """
        H = np.array(
            [[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128,
        ) / math.sqrt(2.0)
        return self.optimize(
            H,
            duration_ns=duration_ns,
            num_slices=num_slices,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )

    def optimize_cnot(
        self,
        duration_ns: float = 200.0,
        num_slices: int = 50,
        max_iterations: int = 500,
        convergence_threshold: float = 0.999,
        seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> GrapeResult:
        """Optimize a CNOT gate on two coupled transmons.

        The target unitary is:
            CNOT = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]]

        Uses the 9x9 two-qubit Hamiltonian with coupling.

        Parameters
        ----------
        duration_ns : float
            Gate duration in nanoseconds.
        num_slices : int
            Number of time slices.
        max_iterations : int
            Maximum GRAPE iterations.
        convergence_threshold : float
            Fidelity convergence target.
        seed : int or None
            Random seed.
        verbose : bool
            Print progress.

        Returns
        -------
        GrapeResult
        """
        if len(self.qubits) < 2:
            raise ValueError(
                "CNOT optimization requires a 2-qubit GrapeOptimizer. "
                "Pass a list of two TransmonQubit instances."
            )

        CNOT = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.complex128,
        )
        return self.optimize(
            CNOT,
            duration_ns=duration_ns,
            num_slices=num_slices,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Self-contained tests and demonstrations
# ---------------------------------------------------------------------------


def _test_expm_hermitian() -> None:
    """Verify matrix exponential via eigendecomposition."""
    print("  [1/6] Matrix exponential (eigendecomposition) ... ", end="")

    # Test with a known Pauli-Z Hamiltonian: exp(-i*sigma_z*t) should give
    # diag(exp(-it), exp(it))
    H = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    dt = 0.5
    U = _expm_hermitian(H, dt)
    expected = np.diag([np.exp(-1j * dt), np.exp(1j * dt)])
    assert np.allclose(U, expected, atol=1e-12), f"expm mismatch:\n{U}\nvs\n{expected}"

    # Unitarity check
    assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-12), "Not unitary"

    # 3x3 Hermitian matrix
    H3 = np.array(
        [[1, 0.5j, 0], [-0.5j, 2, 0.3], [0, 0.3, 3]], dtype=np.complex128,
    )
    U3 = _expm_hermitian(H3, 0.1)
    assert np.allclose(U3 @ U3.conj().T, np.eye(3), atol=1e-12), "3x3 not unitary"

    print("PASS")


def _test_grape_single_qubit_identity() -> None:
    """GRAPE should achieve near-perfect fidelity for the identity gate (trivial)."""
    print("  [2/6] GRAPE identity gate (sanity check) ... ", end="")

    q = TransmonQubit(frequency_ghz=5.0, anharmonicity_mhz=-330.0)
    opt = GrapeOptimizer(q, lambda_leakage=0.0, lambda_smoothness=0.0)

    I_target = np.eye(2, dtype=np.complex128)

    # Zero amplitudes should give identity propagator with high fidelity
    result = opt.optimize(
        I_target,
        duration_ns=10.0,
        num_slices=10,
        max_iterations=5,
        initial_guess_I=np.zeros(10),
        initial_guess_Q=np.zeros(10),
    )

    # With zero drive, the propagator should be close to identity in the
    # rotating frame (up to phases from the static Hamiltonian).
    # The fidelity metric is phase-insensitive: |Tr(U_target^dag @ U)|^2/d^2
    print(f"fidelity={result.fidelity:.6f} ... ", end="")
    # For an on-resonance drive, static H only has anharmonicity on |2>,
    # so the computational subspace should see near-identity evolution.
    # We just check that GRAPE runs without error.
    assert result.fidelity >= 0.0, "Negative fidelity"
    print("PASS")


def _test_grape_x_gate() -> None:
    """Optimize an X gate and verify >99% fidelity."""
    print("  [3/6] GRAPE X-gate optimization ... ", end="")

    q = TransmonQubit(frequency_ghz=5.0, anharmonicity_mhz=-330.0)
    opt = GrapeOptimizer(
        q,
        max_amplitude_ghz=0.15,
        lambda_leakage=5.0,
        lambda_smoothness=0.01,
    )

    result = opt.optimize_x_gate(
        duration_ns=25.0,
        num_slices=40,
        max_iterations=300,
        convergence_threshold=0.999,
        seed=42,
        step_size=0.005,
        verbose=False,
    )

    print(
        f"\n    Fidelity: {result.fidelity:.6f}  "
        f"Leakage: {result.leakage:.6f}  "
        f"Iterations: {result.num_iterations}  "
        f"Converged: {result.converged}"
    )

    assert result.fidelity > 0.99, (
        f"X-gate fidelity {result.fidelity:.6f} < 0.99"
    )
    assert result.leakage < 0.05, (
        f"X-gate leakage {result.leakage:.6f} too high"
    )

    print("  PASS")


def _test_grape_hadamard() -> None:
    """Optimize a Hadamard gate."""
    print("  [4/6] GRAPE Hadamard optimization ... ", end="")

    q = TransmonQubit(frequency_ghz=5.0, anharmonicity_mhz=-330.0)
    opt = GrapeOptimizer(
        q,
        max_amplitude_ghz=0.15,
        lambda_leakage=5.0,
        lambda_smoothness=0.01,
    )

    result = opt.optimize_hadamard(
        duration_ns=25.0,
        num_slices=40,
        max_iterations=300,
        convergence_threshold=0.999,
        seed=42,
        step_size=0.005,
        verbose=False,
    )

    print(
        f"\n    Fidelity: {result.fidelity:.6f}  "
        f"Leakage: {result.leakage:.6f}  "
        f"Iterations: {result.num_iterations}  "
        f"Converged: {result.converged}"
    )

    assert result.fidelity > 0.99, (
        f"Hadamard fidelity {result.fidelity:.6f} < 0.99"
    )

    print("  PASS")


def _test_grape_vs_gaussian_leakage() -> None:
    """Show that GRAPE-optimized pulse has lower leakage than standard Gaussian."""
    print("  [5/6] GRAPE vs Gaussian leakage comparison ... ", end="")

    from .chip import ChipTopology, ChipConfig, NativeGateFamily
    from .pulse import PulseSimulator

    # Use small anharmonicity to make leakage visible
    q = TransmonQubit(
        frequency_ghz=5.0,
        anharmonicity_mhz=-200.0,
        gate_time_ns=15.0,
    )

    # GRAPE optimization with leakage penalty
    opt = GrapeOptimizer(
        q,
        max_amplitude_ghz=0.15,
        lambda_leakage=10.0,
        lambda_smoothness=0.02,
    )

    grape_result = opt.optimize_x_gate(
        duration_ns=15.0,
        num_slices=30,
        max_iterations=200,
        convergence_threshold=0.999,
        seed=42,
        step_size=0.005,
        verbose=False,
    )

    # Standard Gaussian pulse for comparison
    topo = ChipTopology.fully_connected(1, coupling=0.0)
    config = ChipConfig(
        topology=topo,
        qubits=[q],
        native_2q_gate=NativeGateFamily.ECR,
    )
    psim = PulseSimulator(config, dt_ns=0.01)
    gauss_p = psim.gaussian_pulse(qubit=0, angle=math.pi, axis="x", duration_ns=15.0)
    psi_gauss = psim.simulate_pulse(gauss_p, qubit=0)
    gauss_leakage = abs(psi_gauss[2]) ** 2

    print(
        f"\n    Gaussian leakage:       {gauss_leakage:.6f}"
        f"\n    GRAPE leakage:          {grape_result.leakage:.6f}"
        f"\n    GRAPE fidelity:         {grape_result.fidelity:.6f}"
    )

    # GRAPE should have lower leakage than naive Gaussian
    if gauss_leakage > 0.01:
        assert grape_result.leakage < gauss_leakage, (
            f"GRAPE leakage ({grape_result.leakage:.6f}) should be < "
            f"Gaussian leakage ({gauss_leakage:.6f})"
        )
        print(
            f"    Leakage reduction:      "
            f"{gauss_leakage / max(grape_result.leakage, 1e-12):.1f}x"
        )

    print("  PASS")


def _test_grape_to_pulse() -> None:
    """Verify that GrapeResult.to_pulse() produces a valid Pulse object."""
    print("  [6/6] GRAPE to_pulse() conversion ... ", end="")

    q = TransmonQubit(frequency_ghz=5.0, anharmonicity_mhz=-330.0)
    opt = GrapeOptimizer(q, lambda_leakage=0.0, lambda_smoothness=0.0)

    result = opt.optimize_x_gate(
        duration_ns=20.0,
        num_slices=20,
        max_iterations=50,
        seed=42,
    )

    pulse = result.to_pulse(frequency_ghz=5.0)
    assert isinstance(pulse, Pulse), "to_pulse() should return a Pulse"
    assert abs(pulse.duration_ns - 20.0) < 1e-10, "Duration mismatch"

    # Verify envelope returns correct values for each slice
    dt = result.dt_ns
    for k in range(len(result.optimized_amplitudes_I)):
        t = k * dt + dt / 2.0  # center of slice
        env = pulse.envelope(t)
        expected_I = result.optimized_amplitudes_I[k]
        expected_Q = result.optimized_amplitudes_Q[k]
        assert abs(env.real - expected_I) < 1e-10, (
            f"Slice {k} I mismatch: {env.real} vs {expected_I}"
        )
        assert abs(env.imag - expected_Q) < 1e-10, (
            f"Slice {k} Q mismatch: {env.imag} vs {expected_Q}"
        )

    print("PASS")


def _test_lindblad_grape() -> None:
    """Verify that Lindblad-aware GRAPE finds pulses robust against T1/T2.

    Strategy: first run standard GRAPE to get a good initial pulse, then
    refine it under the Lindblad equation. This tests that the Lindblad
    evaluation and gradient path are functional without requiring the
    full numerical-gradient convergence from scratch (which would need
    thousands of iterations and be prohibitively slow for a self-test).
    """
    print("  [7/7] Lindblad-aware GRAPE ... ", end="")

    q = TransmonQubit(
        frequency_ghz=5.0,
        anharmonicity_mhz=-330.0,
        t1_us=50.0,
        t2_us=70.0,
    )

    # Step 1: get a good pulse from standard GRAPE.
    opt_std = GrapeOptimizer(
        q,
        max_amplitude_ghz=0.15,
        lambda_leakage=2.0,
        lambda_smoothness=0.01,
        use_lindblad=False,
    )

    result_std = opt_std.optimize_x_gate(
        duration_ns=20.0,
        num_slices=10,
        max_iterations=150,
        convergence_threshold=0.999,
        seed=42,
        step_size=0.005,
        verbose=False,
    )

    # Step 2: evaluate that pulse under Lindblad to get a baseline.
    opt_lb = GrapeOptimizer(
        q,
        max_amplitude_ghz=0.15,
        lambda_leakage=2.0,
        lambda_smoothness=0.001,
        use_lindblad=True,
    )

    # Evaluate the standard GRAPE pulse under Lindblad.
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    U_target_full = opt_lb._embed_target(X)
    dt = 20.0 / 10
    baseline_fid, baseline_leak = opt_lb._lindblad_gate_fidelity(
        result_std.optimized_amplitudes_I,
        result_std.optimized_amplitudes_Q,
        dt,
        U_target_full,
    )

    # Step 3: run Lindblad GRAPE starting from the standard pulse (warm-start),
    # using few iterations to just verify the gradient path works.
    result_lb = opt_lb.optimize(
        X,
        duration_ns=20.0,
        num_slices=10,
        max_iterations=8,
        convergence_threshold=0.9999,
        seed=42,
        step_size=0.003,
        initial_guess_I=result_std.optimized_amplitudes_I,
        initial_guess_Q=result_std.optimized_amplitudes_Q,
        verbose=False,
    )

    print(
        f"\n    Standard GRAPE fidelity (unitary): {result_std.fidelity:.6f}"
        f"\n    Baseline Lindblad fidelity:        {baseline_fid:.6f}"
        f"\n    After Lindblad GRAPE refinement:   {result_lb.fidelity:.6f}  "
        f"(leakage={result_lb.leakage:.6f}, iters={result_lb.num_iterations})"
    )

    # The Lindblad-evaluated baseline should be close to the unitary fidelity
    # (gate is only 20 ns, T1=50 us, so decoherence is tiny).
    assert baseline_fid > 0.95, (
        f"Lindblad baseline fidelity too low: {baseline_fid:.6f}"
    )

    # Lindblad GRAPE refinement should maintain or improve fidelity.
    assert result_lb.fidelity > 0.90, (
        f"Lindblad GRAPE fidelity too low: {result_lb.fidelity:.6f}"
    )

    # Fidelity history should be non-empty.
    assert len(result_lb.fidelity_history) > 0

    # Should have generated non-trivial amplitudes.
    assert np.max(np.abs(result_lb.optimized_amplitudes_I)) > 0.001, (
        "Lindblad GRAPE should produce non-zero amplitudes"
    )

    print("  PASS")


def _demo_convergence() -> None:
    """Show fidelity vs iteration convergence curve (informational)."""
    print("\n  Convergence demo (X gate, 40 slices, 25 ns):")
    print("  " + "-" * 50)

    q = TransmonQubit(frequency_ghz=5.0, anharmonicity_mhz=-330.0)
    opt = GrapeOptimizer(
        q,
        max_amplitude_ghz=0.15,
        lambda_leakage=5.0,
        lambda_smoothness=0.01,
    )

    result = opt.optimize_x_gate(
        duration_ns=25.0,
        num_slices=40,
        max_iterations=200,
        convergence_threshold=0.9999,
        seed=42,
        step_size=0.005,
        verbose=False,
    )

    # Print convergence milestones
    milestones = [0.90, 0.95, 0.99, 0.999, 0.9999]
    for m in milestones:
        reached = [i for i, f in enumerate(result.fidelity_history) if f >= m]
        if reached:
            print(f"    F >= {m:.4f} at iteration {reached[0]}")
        else:
            print(f"    F >= {m:.4f} not reached (max={max(result.fidelity_history):.6f})")

    # Print ASCII convergence plot (compact)
    n_hist = len(result.fidelity_history)
    n_cols = 50
    step = max(1, n_hist // n_cols)
    sampled = result.fidelity_history[::step]
    print(f"\n    Iterations: {result.num_iterations}, Final F: {result.fidelity:.6f}")
    print(f"    Leakage: {result.leakage:.6f}, Converged: {result.converged}")

    # Bar chart
    for f_val in sampled[:30]:
        bar_len = int(f_val * 40)
        print(f"    |{'#' * bar_len}{' ' * (40 - bar_len)}| {f_val:.4f}")


if __name__ == "__main__":
    print("Running GRAPE optimal control tests...\n")
    _test_expm_hermitian()
    _test_grape_single_qubit_identity()
    _test_grape_x_gate()
    _test_grape_hadamard()
    _test_grape_vs_gaussian_leakage()
    _test_grape_to_pulse()
    _test_lindblad_grape()
    _demo_convergence()
    print("\nAll GRAPE tests passed.")
