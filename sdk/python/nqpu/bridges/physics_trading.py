"""Physics-trading bridges: Hamiltonian dynamics for market modelling.

Uses quantum Hamiltonian evolution to model volatility dynamics and
quantum phase transitions to detect market regime changes.

Theory:
  - Implied volatility surface -> quantum spin chain (each site = strike/expiry)
  - Time evolution of the spin chain -> deformation of the vol surface
  - Phase transition in the Ising model -> market regime change
  - Quantum walk spreading -> momentum indicator

References:
  - Baaquie (2004), "Quantum Finance: Path Integrals and Hamiltonians"
  - Orrell (2020), "Quantum Economics and Finance"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nqpu.simulation import (
    ising_model,
    ExactEvolution,
    EvolutionResult,
    Magnetization,
    EntanglementEntropy,
    SparsePauliHamiltonian,
    PauliOperator,
)


# ---------------------------------------------------------------------------
# Private helpers for efficient small-system computations
# ---------------------------------------------------------------------------


def _build_ising_from_cov(
    cov: np.ndarray,
    transverse_field: float = 1.0,
) -> SparsePauliHamiltonian:
    """Build an Ising Hamiltonian from a covariance matrix.

    Converts covariance to correlations and uses them as ZZ couplings.

    Parameters
    ----------
    cov : np.ndarray
        (n, n) covariance matrix.
    transverse_field : float
        Transverse field strength.

    Returns
    -------
    SparsePauliHamiltonian
    """
    n = cov.shape[0]
    std = np.sqrt(np.diag(cov))
    std = np.where(std > 1e-15, std, 1e-15)
    corr = cov / np.outer(std, std)
    terms: list[PauliOperator] = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) > 1e-10:
                label = ["I"] * n
                label[i] = "Z"
                label[j] = "Z"
                terms.append(PauliOperator("".join(label), coeff=-corr[i, j]))
        label = ["I"] * n
        label[i] = "X"
        terms.append(PauliOperator("".join(label), coeff=-transverse_field))
    return SparsePauliHamiltonian(terms)


def _ground_state(
    H: SparsePauliHamiltonian,
    n: int,
) -> tuple[float, np.ndarray]:
    """Compute ground state energy and vector for a Hamiltonian.

    Parameters
    ----------
    H : SparsePauliHamiltonian
        The Hamiltonian.
    n : int
        Number of qubits (used for validation only).

    Returns
    -------
    tuple[float, np.ndarray]
        (energy, statevector).
    """
    mat = H.matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    return float(eigenvalues[0]), eigenvectors[:, 0]


def _magnetization(psi: np.ndarray, n: int) -> float:
    """Compute mean Z magnetization (1/n) sum_i <Z_i>.

    Parameters
    ----------
    psi : np.ndarray
        State vector of length 2^n.
    n : int
        Number of qubits.

    Returns
    -------
    float
        Mean magnetization in [-1, 1].
    """
    mag = Magnetization(n)
    local_mag = mag.local(psi)
    return float(np.mean(local_mag))


def _entanglement_entropy(psi: np.ndarray, n: int) -> float:
    """Compute bipartite entanglement entropy (half-chain cut).

    Parameters
    ----------
    psi : np.ndarray
        State vector of length 2^n.
    n : int
        Number of qubits.

    Returns
    -------
    float
        Von Neumann entanglement entropy (nats).
    """
    ee = EntanglementEntropy(n)
    return ee.von_neumann(psi)


# ---------------------------------------------------------------------------
# HamiltonianVolatility
# ---------------------------------------------------------------------------


@dataclass
class HamiltonianVolatility:
    """Model implied volatility surface evolution using Hamiltonian dynamics.

    Maps the IV surface onto a quantum spin chain where each site
    represents a strike/expiry point.  Time evolution of the system
    models how the vol surface deforms under quantum dynamics.

    Parameters
    ----------
    n_strikes : int
        Number of lattice sites (strike/expiry discretisation).
    coupling : float
        Nearest-neighbour coupling strength J.
    """

    n_strikes: int = 8
    coupling: float = 1.0

    def evolve_surface(
        self,
        initial_vols: np.ndarray,
        t_final: float = 1.0,
        n_steps: int = 50,
    ) -> dict:
        """Evolve an initial volatility profile using quantum dynamics.

        Encodes the volatility values into a quantum state via Ry rotations,
        then evolves under a transverse-field Ising Hamiltonian.  The
        per-site magnetisation at each time step is decoded back into a
        volatility profile.

        Parameters
        ----------
        initial_vols : np.ndarray
            Initial volatility values (one per strike), in [0, 1].
        t_final : float
            Total evolution time.
        n_steps : int
            Number of time steps to record.

        Returns
        -------
        dict
            Keys: times, vol_profiles, initial_vols, final_vols.
        """
        n = min(len(initial_vols), self.n_strikes)
        H = ising_model(n, J=self.coupling, h=0.5)

        # Encode initial vols into quantum state via Ry rotations
        psi0 = np.zeros(2**n, dtype=complex)
        psi0[0] = 1.0  # Start from |00...0>

        for i in range(n):
            angle = initial_vols[i] * np.pi  # Map vol [0,1] -> angle [0, pi]
            cos_half = np.cos(angle / 2)
            sin_half = np.sin(angle / 2)
            new_psi = np.zeros_like(psi0)
            for basis_state in range(2**n):
                bit_i = (basis_state >> (n - 1 - i)) & 1
                if bit_i == 0:
                    flipped = basis_state | (1 << (n - 1 - i))
                    new_psi[basis_state] += cos_half * psi0[basis_state]
                    new_psi[flipped] += sin_half * psi0[basis_state]
                else:
                    unflipped = basis_state & ~(1 << (n - 1 - i))
                    new_psi[basis_state] += cos_half * psi0[basis_state]
                    new_psi[unflipped] -= sin_half * psi0[basis_state]
            psi0 = new_psi

        # Evolve under the Ising Hamiltonian
        evolver = ExactEvolution(H)
        result = evolver.evolve(psi0, t_final=t_final, n_steps=n_steps)

        # Extract per-site magnetisation as evolved vol profile
        mag = Magnetization(n)
        vol_profiles = []
        for state in result.states:
            profile = mag.local(state)
            # Map magnetisation [-1, 1] back to vol space [0, 1]
            vol_profiles.append((1.0 - np.array(profile)) / 2.0)

        return {
            "times": result.times,
            "vol_profiles": np.array(vol_profiles),
            "initial_vols": initial_vols[:n],
            "final_vols": vol_profiles[-1] if vol_profiles else initial_vols[:n],
        }


# ---------------------------------------------------------------------------
# PhaseTransitionRegime
# ---------------------------------------------------------------------------


@dataclass
class PhaseTransitionRegime:
    """Detect market regime transitions using quantum phase transitions.

    Models the market as a transverse-field Ising system where the
    transverse field represents uncertainty.  A phase transition in the
    Ising model corresponds to a regime change in the market:

    - **Ordered phase** (low field): strong correlations -> trending market
    - **Disordered phase** (high field): weak correlations -> mean-reverting

    Parameters
    ----------
    n_assets : int
        Maximum number of assets to include.
    window_size : int
        Rolling window size for correlation estimation.
    """

    n_assets: int
    window_size: int = 50

    def detect_transitions(self, returns: np.ndarray) -> dict:
        """Detect regime transitions in a returns time series.

        Slides a window over the returns, builds an Ising model from
        the windowed covariance, and classifies each window by its
        magnetisation (trending vs mean-reverting).

        Parameters
        ----------
        returns : np.ndarray
            (T, N) matrix of asset returns.

        Returns
        -------
        dict
            Keys: regimes, transitions, entanglement_entropy, n_transitions.
        """
        T, N = returns.shape
        n = min(N, self.n_assets, 6)  # Cap at 6 qubits for performance
        transitions: list[int] = []
        regimes: list[str] = []
        entropies: list[float] = []

        for t in range(self.window_size, T):
            window = returns[t - self.window_size : t, :n]
            cov = np.cov(window, rowvar=False)
            # Ensure positive semi-definite
            cov = (cov + cov.T) / 2
            eigvals = np.linalg.eigvalsh(cov)
            cov += np.eye(n) * max(0, -eigvals.min() + 1e-8)

            model = _build_ising_from_cov(cov, transverse_field=1.0)
            _, psi = _ground_state(model, n)

            # Compute order parameter (magnetisation)
            mz = _magnetization(psi, n)
            regime = "trending" if abs(mz) > 0.5 else "mean_reverting"
            regimes.append(regime)

            # Entanglement entropy as systemic risk
            entropy = _entanglement_entropy(psi, n)
            entropies.append(entropy)

            # Detect transition (regime change)
            if len(regimes) > 1 and regimes[-1] != regimes[-2]:
                transitions.append(t)

        return {
            "regimes": regimes,
            "transitions": transitions,
            "entanglement_entropy": np.array(entropies),
            "n_transitions": len(transitions),
        }


# ---------------------------------------------------------------------------
# QuantumWalkMomentum
# ---------------------------------------------------------------------------


@dataclass
class QuantumWalkMomentum:
    """Quantum walk-based momentum indicator.

    Uses the spreading behaviour of continuous-time quantum walks to
    measure directional momentum in price series.  The drift in the
    quantum walk is set from recent returns, and the resulting asymmetry
    in the probability distribution measures momentum.

    Parameters
    ----------
    lookback : int
        Number of past prices to use for each momentum estimate.
    n_sites : int
        Lattice size for the quantum walk.
    """

    lookback: int = 20
    n_sites: int = 32

    def compute(self, prices: np.ndarray) -> np.ndarray:
        """Compute quantum walk momentum for a price series.

        Parameters
        ----------
        prices : np.ndarray
            1D array of asset prices.

        Returns
        -------
        np.ndarray
            Momentum values in [-1, 1] (same length as prices).
        """
        n = len(prices)
        momentum = np.zeros(n)
        for t in range(self.lookback, n):
            window = prices[t - self.lookback : t]
            log_returns = np.diff(np.log(window))
            drift = float(np.mean(log_returns))
            vol = float(np.std(log_returns))
            if vol < 1e-10:
                momentum[t] = 0.0
                continue
            # Build quantum walk Hamiltonian with drift
            ns = self.n_sites
            H = np.zeros((ns, ns), dtype=complex)
            for i in range(ns - 1):
                H[i, i + 1] = -vol
                H[i + 1, i] = -vol
            for i in range(ns):
                H[i, i] = drift * (i - ns // 2) / ns
            # Localised initial state at centre
            psi = np.zeros(ns, dtype=complex)
            psi[ns // 2] = 1.0
            # Time evolution
            eigenvalues, eigvecs = np.linalg.eigh(H)
            U = eigvecs @ np.diag(np.exp(-1j * eigenvalues)) @ eigvecs.conj().T
            psi_final = U @ psi
            prob = np.abs(psi_final) ** 2
            # Momentum = mean position shift (positive = right/up)
            positions = np.arange(ns) - ns // 2
            momentum[t] = float(np.sum(positions * prob)) / (ns // 2)
        return np.clip(momentum, -1, 1)
