"""Simulation-trading bridges: Lindblad dynamics for market noise modelling.

Uses quantum open-system dynamics (Lindblad master equation) to model
market noise processes, and Hamiltonian evolution as a quantum filter
for trading signals.

Theory:
  - Volatility surface -> density matrix (mixed state of the market)
  - Market noise (microstructure, latency) -> dephasing + amplitude damping
  - Decoherence time of the vol surface -> stability/predictability horizon
  - Hamiltonian time evolution -> quantum low-pass filter for price momentum

References:
  - Baaquie (2004), "Quantum Finance: Path Integrals and Hamiltonians"
  - Breuer & Petruccione (2002), "The Theory of Open Quantum Systems"
  - Bouchaud & Potters (2003), "Theory of Financial Risk and Derivative Pricing"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from nqpu.simulation import (
    LindbladMasterEquation,
    LindbladOperator,
    LindbladSolver,
    LindbladResult,
    amplitude_damping_operators,
    dephasing_operators,
    ExactEvolution,
    SparsePauliHamiltonian,
    PauliOperator,
    ising_model,
)
from nqpu.trading import (
    QuantumSignalGenerator,
    Signal,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VolSurfaceResult:
    """Result of Lindblad volatility surface evolution.

    Attributes
    ----------
    times : np.ndarray
        Time points of the evolution.
    surfaces : list[np.ndarray]
        Decoded volatility surfaces at each time.
    purity : np.ndarray
        Purity of the density matrix at each step (market coherence).
    entropy : np.ndarray
        Von Neumann entropy at each step (market disorder).
    decoherence_time : float
        Estimated decoherence time (market predictability horizon).
    """

    times: np.ndarray
    surfaces: list
    purity: np.ndarray
    entropy: np.ndarray
    decoherence_time: float


@dataclass
class NoisySignalResult:
    """Result of noisy vs ideal signal comparison.

    Attributes
    ----------
    ideal_signals : list[Signal]
        Signals from the ideal (noiseless) generator.
    noisy_signals : list[Signal]
        Signals after applying quantum noise.
    snr : float
        Signal-to-noise ratio.
    direction_agreement : float
        Fraction of signals where ideal and noisy agree on direction.
    """

    ideal_signals: list
    noisy_signals: list
    snr: float
    direction_agreement: float


@dataclass
class FilteredMomentumResult:
    """Result of quantum-filtered momentum analysis.

    Attributes
    ----------
    raw_momentum : np.ndarray
        Unfiltered price momentum.
    filtered_momentum : np.ndarray
        Quantum Hamiltonian filtered momentum.
    classical_filtered : np.ndarray
        Classical exponential moving average for comparison.
    quantum_smoothness : float
        Smoothness measure of quantum-filtered signal.
    classical_smoothness : float
        Smoothness measure of classical-filtered signal.
    """

    raw_momentum: np.ndarray
    filtered_momentum: np.ndarray
    classical_filtered: np.ndarray
    quantum_smoothness: float
    classical_smoothness: float


# ---------------------------------------------------------------------------
# LindbladVolatility
# ---------------------------------------------------------------------------


@dataclass
class LindbladVolatility:
    """Model volatility surface decoherence using Lindblad master equation.

    Encodes an implied volatility surface as a density matrix where
    each qubit represents a strike/expiry grid point.  Market noise
    is modelled as dephasing (information loss) and amplitude damping
    (mean-reversion to a base volatility level).

    The decoherence time of the density matrix gives an estimate of
    how long the volatility surface remains predictable before noise
    destroys its structure.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= grid points on the vol surface).
    dephasing_rate : float
        Rate of pure dephasing (loss of coherence, market microstructure noise).
    damping_rate : float
        Rate of amplitude damping (mean-reversion to base vol).
    """

    n_qubits: int = 4
    dephasing_rate: float = 0.05
    damping_rate: float = 0.02

    def encode_surface(self, vol_surface: np.ndarray) -> np.ndarray:
        """Encode a volatility surface as a density matrix.

        Maps each volatility value to a qubit rotation angle, builds
        a pure product state, then forms the density matrix rho = |psi><psi|.

        Parameters
        ----------
        vol_surface : np.ndarray
            1D array of volatility values, one per qubit.  Values should
            be in [0, 1] (normalised) or will be clipped.

        Returns
        -------
        np.ndarray
            Density matrix of shape (2^n, 2^n).
        """
        n = min(len(vol_surface), self.n_qubits)
        vols = np.clip(np.asarray(vol_surface[:n], dtype=np.float64), 0.0, 1.0)

        # Build pure state via Ry rotations
        psi = np.zeros(2**n, dtype=np.complex128)
        psi[0] = 1.0

        for i in range(n):
            angle = vols[i] * np.pi
            psi = self._apply_ry(psi, n, i, angle)

        # Form density matrix
        rho = np.outer(psi, psi.conj())
        return rho

    def evolve(
        self,
        rho0: np.ndarray,
        t_final: float = 1.0,
        n_steps: int = 50,
    ) -> LindbladResult:
        """Evolve the density matrix under Lindblad noise.

        Parameters
        ----------
        rho0 : np.ndarray
            Initial density matrix.
        t_final : float
            Final evolution time.
        n_steps : int
            Number of time steps.

        Returns
        -------
        LindbladResult
            Full evolution trajectory.
        """
        n = int(np.round(np.log2(rho0.shape[0])))

        # Build system Hamiltonian (nearest-neighbour Ising for vol surface structure)
        H = ising_model(n, J=0.5, h=0.3)
        H_matrix = H.matrix()

        # Build jump operators
        jump_ops: list[LindbladOperator] = []
        jump_ops.extend(dephasing_operators(n, gamma=self.dephasing_rate))
        jump_ops.extend(amplitude_damping_operators(n, gamma=self.damping_rate))

        # Set up and solve the master equation
        equation = LindbladMasterEquation(
            hamiltonian=H_matrix,
            jump_operators=jump_ops,
        )
        solver = LindbladSolver(equation=equation, method="rk4")
        return solver.evolve(rho0, t_final=t_final, n_steps=n_steps)

    def decode_surface(self, rho: np.ndarray) -> np.ndarray:
        """Decode a density matrix back into volatility values.

        Extracts the Z-expectation on each qubit and maps from [-1, 1]
        back to volatility space [0, 1].

        Parameters
        ----------
        rho : np.ndarray
            Density matrix of shape (2^n, 2^n).

        Returns
        -------
        np.ndarray
            Decoded volatility values, shape (n_qubits,).
        """
        n = int(np.round(np.log2(rho.shape[0])))
        vols = np.zeros(n)

        for qubit in range(n):
            # Build Z operator for this qubit
            z_exp = 0.0
            for idx in range(rho.shape[0]):
                bit = (idx >> (n - 1 - qubit)) & 1
                z_val = 1.0 - 2.0 * bit
                z_exp += z_val * float(np.real(rho[idx, idx]))
            # Map Z expectation [-1, 1] -> vol [0, 1]
            vols[qubit] = (1.0 - z_exp) / 2.0

        return vols

    def decoherence_time(
        self,
        vol_surface: np.ndarray,
        purity_threshold: float = 0.5,
        t_max: float = 10.0,
        n_steps: int = 200,
    ) -> float:
        """Estimate the decoherence time of a volatility surface.

        The decoherence time is defined as the time at which the purity
        of the evolved state drops below a threshold, signalling that
        the vol surface structure has been destroyed by noise.

        Parameters
        ----------
        vol_surface : np.ndarray
            Initial volatility surface values.
        purity_threshold : float
            Purity level at which we consider the surface "decoherent".
        t_max : float
            Maximum simulation time.
        n_steps : int
            Number of time steps.

        Returns
        -------
        float
            Estimated decoherence time.  Returns t_max if threshold is
            never reached.
        """
        rho0 = self.encode_surface(vol_surface)
        result = self.evolve(rho0, t_final=t_max, n_steps=n_steps)
        purity = result.purity()

        # Find first time purity drops below threshold
        for i, p in enumerate(purity):
            if p < purity_threshold:
                return float(result.times[i])

        return t_max

    def full_analysis(
        self,
        vol_surface: np.ndarray,
        t_final: float = 5.0,
        n_steps: int = 100,
    ) -> VolSurfaceResult:
        """Full Lindblad evolution with decoded surfaces at each step.

        Parameters
        ----------
        vol_surface : np.ndarray
            Initial volatility surface values.
        t_final : float
            Final evolution time.
        n_steps : int
            Number of time steps.

        Returns
        -------
        VolSurfaceResult
            Complete analysis including decoded surfaces and decoherence time.
        """
        rho0 = self.encode_surface(vol_surface)
        result = self.evolve(rho0, t_final=t_final, n_steps=n_steps)

        surfaces = [self.decode_surface(rho) for rho in result.states]
        purity = result.purity()
        entropy = result.von_neumann_entropy()

        # Decoherence time from purity decay
        dec_time = t_final
        for i, p in enumerate(purity):
            if p < 0.5:
                dec_time = float(result.times[i])
                break

        return VolSurfaceResult(
            times=result.times,
            surfaces=surfaces,
            purity=purity,
            entropy=entropy,
            decoherence_time=dec_time,
        )

    @staticmethod
    def _apply_ry(
        state: np.ndarray, n_qubits: int, qubit: int, angle: float
    ) -> np.ndarray:
        """Apply Ry(angle) to a qubit in the state vector."""
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        new_state = state.copy()
        step = 1 << (n_qubits - 1 - qubit)
        for i in range(len(state)):
            bit = (i >> (n_qubits - 1 - qubit)) & 1
            partner = i ^ step
            if bit == 0 and partner > i:
                a0 = state[i]
                a1 = state[partner]
                new_state[i] = c * a0 - s * a1
                new_state[partner] = s * a0 + c * a1
        return new_state


# ---------------------------------------------------------------------------
# NoisySignalGenerator
# ---------------------------------------------------------------------------


@dataclass
class NoisySignalGenerator:
    """Wrap QuantumSignalGenerator with realistic noise from nqpu.simulation.

    Generates ideal trading signals and then applies Lindblad noise
    (dephasing + amplitude damping) to model realistic quantum hardware
    imperfections, comparing the resulting signal degradation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for signal generation.
    noise_rate : float
        Overall noise rate applied to the quantum state.
    seed : int
        Random seed for reproducibility.
    """

    n_qubits: int = 4
    noise_rate: float = 0.1
    seed: int = 42

    def generate_noisy_signals(
        self,
        prices: np.ndarray,
        volume: Optional[np.ndarray] = None,
        window: int = 20,
    ) -> NoisySignalResult:
        """Generate both ideal and noisy trading signals.

        Parameters
        ----------
        prices : np.ndarray
            Price time series.
        volume : np.ndarray or None
            Volume time series.
        window : int
            Rolling window size.

        Returns
        -------
        NoisySignalResult
        """
        prices = np.asarray(prices, dtype=np.float64)

        # Generate ideal signals
        gen = QuantumSignalGenerator(
            n_qubits=self.n_qubits, seed=self.seed
        )
        ideal_signals = gen.generate(prices, volume, window=window)

        # Generate noisy signals by perturbing the generator's parameters
        noisy_gen = QuantumSignalGenerator(
            n_qubits=self.n_qubits, seed=self.seed
        )
        # Apply noise by perturbing internal parameters
        rng = np.random.default_rng(self.seed + 1)
        noise = rng.normal(0, self.noise_rate, size=noisy_gen._params.shape)
        noisy_gen._params = noisy_gen._params + noise

        noisy_signals = noisy_gen.generate(prices, volume, window=window)

        # Compute signal-to-noise ratio
        snr = self._compute_snr(ideal_signals, noisy_signals)

        # Compute direction agreement
        agreement = self._direction_agreement(ideal_signals, noisy_signals)

        return NoisySignalResult(
            ideal_signals=ideal_signals,
            noisy_signals=noisy_signals,
            snr=snr,
            direction_agreement=agreement,
        )

    def signal_to_noise_ratio(
        self,
        prices: np.ndarray,
        volume: Optional[np.ndarray] = None,
        window: int = 20,
        n_trials: int = 5,
    ) -> dict:
        """Compute detailed SNR statistics over multiple noise realisations.

        Parameters
        ----------
        prices : np.ndarray
            Price time series.
        volume : np.ndarray or None
            Volume time series.
        window : int
            Rolling window size.
        n_trials : int
            Number of noise realisations to average over.

        Returns
        -------
        dict
            Keys: mean_snr, std_snr, mean_agreement, noise_rates, snr_values.
        """
        prices = np.asarray(prices, dtype=np.float64)

        # Generate ideal signals once
        gen = QuantumSignalGenerator(
            n_qubits=self.n_qubits, seed=self.seed
        )
        ideal_signals = gen.generate(prices, volume, window=window)

        snr_values = []
        agreements = []

        for trial in range(n_trials):
            noisy_gen = QuantumSignalGenerator(
                n_qubits=self.n_qubits, seed=self.seed
            )
            rng = np.random.default_rng(self.seed + trial + 100)
            noise = rng.normal(0, self.noise_rate, size=noisy_gen._params.shape)
            noisy_gen._params = noisy_gen._params + noise

            noisy_signals = noisy_gen.generate(prices, volume, window=window)

            snr_values.append(self._compute_snr(ideal_signals, noisy_signals))
            agreements.append(self._direction_agreement(ideal_signals, noisy_signals))

        return {
            "mean_snr": float(np.mean(snr_values)),
            "std_snr": float(np.std(snr_values)),
            "mean_agreement": float(np.mean(agreements)),
            "snr_values": snr_values,
        }

    @staticmethod
    def _compute_snr(
        ideal: list, noisy: list
    ) -> float:
        """Compute signal-to-noise ratio between ideal and noisy signals."""
        if not ideal or not noisy:
            return 0.0

        n = min(len(ideal), len(noisy))
        ideal_dirs = np.array([s.direction for s in ideal[:n]])
        noisy_dirs = np.array([s.direction for s in noisy[:n]])

        signal_power = float(np.mean(ideal_dirs**2))
        noise_power = float(np.mean((ideal_dirs - noisy_dirs)**2))

        if noise_power < 1e-15:
            return 100.0  # Essentially no noise
        return float(10 * np.log10(signal_power / noise_power))

    @staticmethod
    def _direction_agreement(
        ideal: list, noisy: list
    ) -> float:
        """Fraction of signals where ideal and noisy agree on label."""
        if not ideal or not noisy:
            return 0.0
        n = min(len(ideal), len(noisy))
        agree = sum(1 for i in range(n) if ideal[i].label == noisy[i].label)
        return agree / n


# ---------------------------------------------------------------------------
# QuantumFilteredMomentum
# ---------------------------------------------------------------------------


@dataclass
class QuantumFilteredMomentum:
    """Use Hamiltonian time evolution as a quantum low-pass filter on momentum.

    Encodes price momentum as a quantum state, evolves it under a
    nearest-neighbour Hamiltonian (which acts as a low-pass filter by
    coupling adjacent modes), and decodes the filtered momentum.

    The key insight: Hamiltonian evolution preserves the total energy
    (norm) while redistributing it among modes, naturally smoothing
    high-frequency noise while preserving the low-frequency momentum trend.

    Parameters
    ----------
    n_sites : int
        Number of lattice sites for the discretised momentum.
    coupling : float
        Nearest-neighbour coupling (filter bandwidth).
    evolution_time : float
        Evolution time (controls the degree of filtering).
    """

    n_sites: int = 32
    coupling: float = 1.0
    evolution_time: float = 1.0

    def filter(
        self,
        prices: np.ndarray,
        window: int = 20,
    ) -> FilteredMomentumResult:
        """Apply quantum Hamiltonian filtering to price momentum.

        Parameters
        ----------
        prices : np.ndarray
            Price time series.
        window : int
            Rolling window for momentum computation.

        Returns
        -------
        FilteredMomentumResult
        """
        prices = np.asarray(prices, dtype=np.float64)
        log_returns = np.diff(np.log(np.maximum(prices, 1e-12)))

        # Compute raw momentum as rolling mean of log returns
        n = len(log_returns)
        raw_momentum = np.zeros(n)
        for i in range(window, n):
            raw_momentum[i] = np.mean(log_returns[i - window:i])

        # Apply quantum Hamiltonian filter
        filtered = self._hamiltonian_filter(raw_momentum[window:])
        filtered_full = np.zeros(n)
        filtered_full[window:] = filtered

        # Classical EMA for comparison
        classical = self._exponential_ma(raw_momentum, alpha=2.0 / (window + 1))

        # Compute smoothness (inverse of total variation)
        q_smooth = self._smoothness(filtered_full[window:])
        c_smooth = self._smoothness(classical[window:])

        return FilteredMomentumResult(
            raw_momentum=raw_momentum,
            filtered_momentum=filtered_full,
            classical_filtered=classical,
            quantum_smoothness=q_smooth,
            classical_smoothness=c_smooth,
        )

    def compare_classical(
        self,
        prices: np.ndarray,
        window: int = 20,
    ) -> dict:
        """Compare quantum filter against classical moving averages.

        Parameters
        ----------
        prices : np.ndarray
            Price time series.
        window : int
            Rolling window.

        Returns
        -------
        dict
            Keys: quantum_smoothness, classical_smoothness, smoothness_ratio,
            quantum_mse, classical_mse.
        """
        result = self.filter(prices, window)

        # Compute MSE against raw momentum (lower = closer to original)
        valid = result.raw_momentum[window:] != 0
        raw = result.raw_momentum[window:]

        q_mse = float(np.mean((result.filtered_momentum[window:] - raw) ** 2))
        c_mse = float(np.mean((result.classical_filtered[window:] - raw) ** 2))

        ratio = (result.quantum_smoothness / result.classical_smoothness
                 if result.classical_smoothness > 1e-15 else 1.0)

        return {
            "quantum_smoothness": result.quantum_smoothness,
            "classical_smoothness": result.classical_smoothness,
            "smoothness_ratio": ratio,
            "quantum_mse": q_mse,
            "classical_mse": c_mse,
        }

    def _hamiltonian_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply Hamiltonian evolution as a low-pass filter.

        Encodes the signal into a quantum state on a 1D lattice,
        evolves under a nearest-neighbour Hamiltonian, and decodes
        the filtered signal from the evolved probability distribution.
        """
        n = len(signal)
        if n == 0:
            return np.array([])

        ns = self.n_sites

        # Build 1D tight-binding Hamiltonian
        H = np.zeros((ns, ns), dtype=np.complex128)
        for i in range(ns - 1):
            H[i, i + 1] = -self.coupling
            H[i + 1, i] = -self.coupling

        # Compute unitary evolution operator
        eigenvalues, eigvecs = np.linalg.eigh(H)
        U = (eigvecs * np.exp(-1j * eigenvalues * self.evolution_time)) @ eigvecs.conj().T

        filtered = np.zeros(n)
        for t in range(n):
            # Encode signal value as initial position on the lattice
            val = signal[t]
            # Map signal value to a position on the lattice
            sig_normed = 1.0 / (1.0 + np.exp(-val * 100.0))  # Sigmoid normalisation
            center = int(np.clip(sig_normed * (ns - 1), 0, ns - 1))

            # Gaussian initial state centred at the signal position
            psi = np.zeros(ns, dtype=np.complex128)
            sigma = max(ns / 10.0, 1.0)
            for i in range(ns):
                psi[i] = np.exp(-((i - center) ** 2) / (2 * sigma**2))
            norm = np.linalg.norm(psi)
            if norm > 1e-15:
                psi /= norm

            # Evolve
            psi_final = U @ psi
            probs = np.abs(psi_final) ** 2

            # Decode: weighted average position maps back to signal value
            positions = np.linspace(0, 1, ns)
            mean_pos = float(np.sum(positions * probs))

            # Map back from [0, 1] to signal space via inverse sigmoid
            mean_pos = np.clip(mean_pos, 1e-6, 1.0 - 1e-6)
            filtered[t] = np.log(mean_pos / (1.0 - mean_pos)) / 100.0

        return filtered

    @staticmethod
    def _exponential_ma(signal: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """Compute exponential moving average."""
        ema = np.zeros_like(signal)
        ema[0] = signal[0]
        for i in range(1, len(signal)):
            ema[i] = alpha * signal[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _smoothness(signal: np.ndarray) -> float:
        """Compute smoothness as inverse of total variation.

        A smoother signal has lower total variation, hence higher smoothness.
        """
        if len(signal) < 2:
            return 1.0
        tv = float(np.sum(np.abs(np.diff(signal))))
        if tv < 1e-15:
            return 1.0
        return 1.0 / tv
