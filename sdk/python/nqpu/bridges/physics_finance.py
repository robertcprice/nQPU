"""Physics-finance bridges: quantum spin models for financial correlations.

Maps financial correlation structures onto quantum many-body Hamiltonians,
enabling physics-inspired analysis of market dynamics.

Theory:
  - N assets -> N spins in a transverse-field Ising model
  - Correlation(i,j) -> coupling J_ij (ferromagnetic = positive correlation)
  - Market volatility -> transverse field h (disorder/uncertainty)
  - Phase transition -> regime change (ordered = trending, disordered = mean-reverting)
  - Entanglement entropy -> systemic correlation risk

References:
  - Orus et al. (2019), "A Practical Introduction to Tensor Networks"
  - Orrell (2020), "Quantum Economics and Finance"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nqpu.simulation import (
    ising_model,
    ExactEvolution,
    SparsePauliHamiltonian,
    PauliOperator,
    Magnetization,
    EntanglementEntropy,
    Fidelity,
)


# ---------------------------------------------------------------------------
# IsingCorrelationModel
# ---------------------------------------------------------------------------


@dataclass
class IsingCorrelationModel:
    """Ising spin model for financial asset correlations.

    Each asset is mapped to a qubit in a transverse-field Ising model.
    The ZZ coupling strengths encode pairwise correlations, and the
    transverse field represents market uncertainty.  The quantum phase
    transition of the model (ordered vs disordered) maps onto market
    regime changes (trending vs mean-reverting).

    Parameters
    ----------
    n_assets : int
        Number of assets (= number of qubits).
    couplings : np.ndarray
        Symmetric (n, n) matrix of ZZ coupling strengths J_ij.
    transverse_field : float
        Strength of the transverse (X) field.
    asset_names : list[str]
        Human-readable names for each asset.
    """

    n_assets: int
    couplings: np.ndarray
    transverse_field: float
    asset_names: list[str]
    _hamiltonian: SparsePauliHamiltonian | None = field(default=None, repr=False)

    @classmethod
    def from_covariance(
        cls,
        cov: np.ndarray,
        names: list[str] | None = None,
        transverse_field: float = 1.0,
    ) -> IsingCorrelationModel:
        """Build an Ising model from an asset covariance matrix.

        Converts covariance to a correlation matrix, then uses correlations
        as ZZ coupling strengths (positive correlation = ferromagnetic coupling).

        Parameters
        ----------
        cov : np.ndarray
            (N, N) covariance matrix of asset returns.
        names : list[str] or None
            Asset labels.  Auto-generated if not provided.
        transverse_field : float
            Transverse field strength (market uncertainty parameter).

        Returns
        -------
        IsingCorrelationModel
        """
        n = cov.shape[0]
        if names is None:
            names = [f"asset_{i}" for i in range(n)]
        # Convert covariance to correlation matrix
        std = np.sqrt(np.diag(cov))
        std = np.where(std > 1e-15, std, 1e-15)  # guard against zero variance
        corr = cov / np.outer(std, std)
        # Extract upper-triangle couplings
        couplings = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                couplings[i, j] = corr[i, j]
                couplings[j, i] = corr[i, j]
        return cls(n, couplings, transverse_field, names)

    @property
    def coupling_strengths(self) -> np.ndarray:
        """Return the J_ij coupling matrix."""
        return self.couplings.copy()

    @property
    def hamiltonian(self) -> SparsePauliHamiltonian:
        """Lazily-built Ising Hamiltonian with custom couplings."""
        if self._hamiltonian is None:
            self._hamiltonian = self._build_hamiltonian()
        return self._hamiltonian

    def _build_hamiltonian(self) -> SparsePauliHamiltonian:
        """Build H = -sum J_ij Z_i Z_j - h sum X_i from the coupling matrix."""
        n = self.n_assets
        terms: list[PauliOperator] = []
        # ZZ couplings
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.couplings[i, j]) > 1e-10:
                    label = ["I"] * n
                    label[i] = "Z"
                    label[j] = "Z"
                    terms.append(
                        PauliOperator("".join(label), coeff=-self.couplings[i, j])
                    )
            # Transverse field on each site
            label = ["I"] * n
            label[i] = "X"
            terms.append(PauliOperator("".join(label), coeff=-self.transverse_field))
        return SparsePauliHamiltonian(terms)

    @property
    def critical_temperature(self) -> float:
        """Estimate the critical temperature from mean coupling strength.

        Uses the mean-field estimate T_c ~ z * J_mean / 2 where z is the
        coordination number (fully connected: z = n - 1).

        Returns
        -------
        float
            Estimated critical temperature in natural units.
        """
        n = self.n_assets
        if n < 2:
            return 0.0
        upper_tri = self.couplings[np.triu_indices(n, k=1)]
        J_mean = float(np.mean(np.abs(upper_tri)))
        z = n - 1  # fully connected
        return z * J_mean / 2.0

    def ground_state(self) -> tuple[float, np.ndarray]:
        """Compute the ground state energy and state vector.

        Returns
        -------
        tuple[float, np.ndarray]
            (energy, statevector) where statevector has length 2^n_assets.
        """
        H_matrix = self.hamiltonian.matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
        return float(eigenvalues[0]), eigenvectors[:, 0]

    def correlation_matrix(self) -> np.ndarray:
        """Compute <Z_i Z_j> correlations from the Ising ground state.

        Returns
        -------
        np.ndarray
            (n, n) quantum correlation matrix where entry (i,j) is
            the expectation value of Z_i Z_j in the ground state.
        """
        _, psi = self.ground_state()
        n = self.n_assets
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr[i, j] = 1.0
                else:
                    label = ["I"] * n
                    label[i] = "Z"
                    label[j] = "Z"
                    op = PauliOperator("".join(label)).matrix()
                    corr[i, j] = float(np.real(psi.conj() @ op @ psi))
        return corr

    def entanglement_risk(self, subsystem: list[int] | None = None) -> float:
        """Compute entanglement entropy of a subsystem (systemic risk measure).

        A higher entanglement entropy indicates stronger quantum correlations
        within the subsystem, interpreted as higher systemic risk.

        Parameters
        ----------
        subsystem : list[int] or None
            Indices of assets forming the subsystem.  Defaults to the
            first half of assets.

        Returns
        -------
        float
            Von Neumann entanglement entropy of the subsystem in bits.
        """
        _, psi = self.ground_state()
        n = self.n_assets
        if subsystem is None:
            subsystem = list(range(n // 2))
        complement = [i for i in range(n) if i not in subsystem]
        dim_a = 2 ** len(subsystem)
        dim_b = 2 ** len(complement)
        # Reorder qubit axes: subsystem first, complement second
        psi_tensor = psi.reshape([2] * n)
        axes_order = list(subsystem) + complement
        psi_tensor = np.transpose(psi_tensor, axes_order)
        psi_matrix = psi_tensor.reshape(dim_a, dim_b)
        # SVD for entanglement spectrum
        s = np.linalg.svd(psi_matrix, compute_uv=False)
        s = s[s > 1e-12]
        entropy = -np.sum(s**2 * np.log2(s**2 + 1e-30))
        return float(entropy)

    def phase_diagram(
        self,
        field_range: np.ndarray | None = None,
        n_points: int = 20,
    ) -> dict:
        """Sweep the transverse field to map the quantum phase diagram.

        For each field value, computes the ground state energy, magnetization,
        and entanglement entropy.  The peak in entanglement entropy locates
        the quantum critical point.

        Parameters
        ----------
        field_range : np.ndarray or None
            Array of transverse field values to sweep.
        n_points : int
            Number of points if field_range is not provided.

        Returns
        -------
        dict
            Keys: field_values, energies, magnetizations, entanglement_entropy.
        """
        if field_range is None:
            field_range = np.linspace(0.1, 3.0, n_points)
        energies = []
        magnetizations = []
        entropies = []
        for h in field_range:
            model = IsingCorrelationModel(
                self.n_assets, self.couplings, float(h), self.asset_names
            )
            energy, psi = model.ground_state()
            energies.append(energy)
            # Compute mean |<Z>| magnetization
            n = self.n_assets
            mz = 0.0
            for i in range(n):
                label = ["I"] * n
                label[i] = "Z"
                op = PauliOperator("".join(label)).matrix()
                mz += float(np.real(psi.conj() @ op @ psi))
            magnetizations.append(abs(mz / n))
            # Entanglement entropy
            entropies.append(model.entanglement_risk())
        return {
            "field_values": field_range,
            "energies": np.array(energies),
            "magnetizations": np.array(magnetizations),
            "entanglement_entropy": np.array(entropies),
        }


# ---------------------------------------------------------------------------
# QuantumWalkPricer
# ---------------------------------------------------------------------------


@dataclass
class QuantumWalkPricer:
    """Price dynamics via continuous-time quantum walk.

    Models asset price as a quantum particle on a discrete 1D lattice.
    The probability distribution from the walk gives the price distribution,
    offering a quantum-native alternative to geometric Brownian motion.

    The quantum walk spreads ballistically (O(t)) rather than diffusively
    (O(sqrt(t))) like a classical random walk, producing heavier tails
    and faster exploration of the price space.

    Parameters
    ----------
    n_sites : int
        Number of lattice sites (price discretisation resolution).
    volatility : float
        Coupling strength between adjacent sites (models volatility).
    drift : float
        On-site potential gradient (models drift / risk-free rate).
    """

    n_sites: int = 64
    volatility: float = 0.2
    drift: float = 0.0

    def _walk_hamiltonian(self) -> np.ndarray:
        """Build the CTQW Hamiltonian (adjacency matrix of line graph with drift)."""
        n = self.n_sites
        H = np.zeros((n, n), dtype=complex)
        for i in range(n - 1):
            H[i, i + 1] = -self.volatility
            H[i + 1, i] = -self.volatility
        # Drift as on-site potential gradient
        for i in range(n):
            H[i, i] = self.drift * (i - n // 2) / n
        return H

    def price_distribution(
        self,
        spot: float,
        dt: float = 1 / 252,
        n_steps: int = 100,
    ) -> dict:
        """Compute the quantum walk price distribution.

        Parameters
        ----------
        spot : float
            Current spot price.
        dt : float
            Time increment per step (default 1/252 for daily).
        n_steps : int
            Number of time steps.

        Returns
        -------
        dict
            Keys: prices, probabilities, mean_price, std_price,
            quantum_advantage.
        """
        H = self._walk_hamiltonian()
        n = self.n_sites
        # Initial state: localised at centre (spot price)
        psi = np.zeros(n, dtype=complex)
        psi[n // 2] = 1.0
        # Time evolution
        t = dt * n_steps
        U = self._matrix_exp(-1j * H * t)
        psi_final = U @ psi
        probs = np.abs(psi_final) ** 2
        # Map lattice sites to prices
        price_min = spot * np.exp(-3 * self.volatility * np.sqrt(t))
        price_max = spot * np.exp(3 * self.volatility * np.sqrt(t))
        prices = np.linspace(price_min, price_max, n)
        mean_price = float(np.sum(prices * probs))
        std_price = float(np.sqrt(np.sum((prices - mean_price) ** 2 * probs)))
        return {
            "prices": prices,
            "probabilities": probs,
            "mean_price": mean_price,
            "std_price": std_price,
            "quantum_advantage": self._quantum_speedup_factor(probs),
        }

    @staticmethod
    def _matrix_exp(M: np.ndarray) -> np.ndarray:
        """Matrix exponential via eigendecomposition."""
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        return eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.conj().T

    @staticmethod
    def _quantum_speedup_factor(probs: np.ndarray) -> float:
        """Estimate quantum speedup from distribution spread vs classical.

        Quantum walk spreads as O(t) vs classical random walk O(sqrt(t)).
        The inverse participation ratio (IPR) measures how spread-out the
        distribution is; lower IPR means wider spread (more quantum advantage).
        """
        ipr = float(np.sum(probs**2))
        return 1.0 / (ipr * len(probs)) if ipr > 0 else 1.0


# ---------------------------------------------------------------------------
# HamiltonianPortfolio
# ---------------------------------------------------------------------------


@dataclass
class HamiltonianPortfolio:
    """Portfolio optimisation via quantum Hamiltonian ground state.

    Encodes the portfolio selection problem as finding the ground state
    of a quantum Hamiltonian where: Z_i = +1 means asset i is selected,
    Z_i = -1 means not selected.

    The Hamiltonian balances expected returns against risk:
        H = lambda * sum C_ij Z_i Z_j - sum r_i Z_i

    Parameters
    ----------
    expected_returns : np.ndarray
        (N,) array of expected asset returns.
    covariance : np.ndarray
        (N, N) covariance matrix of asset returns.
    risk_aversion : float
        Risk-return trade-off parameter (lambda).
    """

    expected_returns: np.ndarray
    covariance: np.ndarray
    risk_aversion: float = 1.0

    def as_hamiltonian(self) -> SparsePauliHamiltonian:
        """Build the portfolio selection Hamiltonian.

        Returns
        -------
        SparsePauliHamiltonian
        """
        n = len(self.expected_returns)
        terms: list[PauliOperator] = []
        # Risk term: lambda * sum C_ij Z_i Z_j
        for i in range(n):
            for j in range(i + 1, n):
                label = ["I"] * n
                label[i] = "Z"
                label[j] = "Z"
                terms.append(
                    PauliOperator(
                        "".join(label),
                        coeff=self.risk_aversion * self.covariance[i, j],
                    )
                )
            # Return term: -r_i Z_i
            label = ["I"] * n
            label[i] = "Z"
            terms.append(
                PauliOperator("".join(label), coeff=-self.expected_returns[i])
            )
        return SparsePauliHamiltonian(terms)

    def optimal_portfolio(self) -> dict:
        """Find the optimal portfolio via exact diagonalisation.

        Returns
        -------
        dict
            Keys: weights, selection, ground_energy, expected_return, variance.
        """
        H = self.as_hamiltonian().matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        ground_energy = float(eigenvalues[0])
        ground_state = eigenvectors[:, 0]
        n = len(self.expected_returns)
        # Decode: find the most probable basis state
        probs = np.abs(ground_state) ** 2
        best_idx = int(np.argmax(probs))
        selection = [(best_idx >> (n - 1 - i)) & 1 for i in range(n)]
        weights = np.array(selection, dtype=float)
        total = weights.sum()
        if total > 0:
            weights /= total
        return {
            "weights": weights,
            "selection": selection,
            "ground_energy": ground_energy,
            "expected_return": float(weights @ self.expected_returns),
            "variance": float(weights @ self.covariance @ weights),
        }


# ---------------------------------------------------------------------------
# CorrelationPhaseAnalysis
# ---------------------------------------------------------------------------


@dataclass
class CorrelationPhaseAnalysis:
    """Detect market regime transitions via quantum phase transitions.

    Analyses a window of asset returns by mapping the correlation structure
    to an Ising model and examining the quantum phase diagram for signatures
    of criticality.

    Parameters
    ----------
    n_assets : int
        Number of assets to include in the analysis.
    """

    n_assets: int

    @staticmethod
    def analyze(
        returns_window: np.ndarray,
        field_range: np.ndarray | None = None,
    ) -> dict:
        """Analyze correlation structure for phase transition signatures.

        Parameters
        ----------
        returns_window : np.ndarray
            (T, N) matrix of asset returns over a time window.
        field_range : np.ndarray or None
            Transverse field values to sweep for the phase diagram.

        Returns
        -------
        dict
            Keys: critical_field, critical_entropy, phase_diagram,
            current_regime.
        """
        cov = np.cov(returns_window, rowvar=False)
        model = IsingCorrelationModel.from_covariance(cov)
        phase = model.phase_diagram(field_range)
        # Find critical point (peak entanglement entropy)
        idx = int(np.argmax(phase["entanglement_entropy"]))
        return {
            "critical_field": float(phase["field_values"][idx]),
            "critical_entropy": float(phase["entanglement_entropy"][idx]),
            "phase_diagram": phase,
            "current_regime": (
                "ordered"
                if model.transverse_field < phase["field_values"][idx]
                else "disordered"
            ),
        }
