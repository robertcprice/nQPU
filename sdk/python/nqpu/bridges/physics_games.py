"""Physics-games bridges: Ising models for game theory and combinatorial optimization.

Maps game-theoretic structures onto quantum many-body Hamiltonians,
enabling physics-inspired analysis of Nash equilibria, auction mechanisms,
and benchmarking of QAOA against exact Ising ground states.

Theory:
  - Payoff matrix -> Ising coupling matrix (mixed-strategy equilibria = ground states)
  - Auction valuations -> entangled quantum states (bidder correlations)
  - MaxCut QUBO -> Ising Hamiltonian (QAOA vs exact ground state benchmarking)

References:
  - Eisert, Wilkens, Lewenstein (1999), "Quantum Games and Quantum Strategies"
  - Lucas (2014), "Ising formulations of many NP problems"
  - Farhi, Goldstone, Gutmann (2014), "QAOA"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from nqpu.simulation import (
    ising_model,
    ExactEvolution,
    SparsePauliHamiltonian,
    PauliOperator,
    Magnetization,
    EntanglementEntropy,
)
from nqpu.games import (
    Graph,
    MaxCut,
    OptimizationResult,
)
from nqpu.games.qaoa_builder import maxcut_qaoa, QAOACircuit


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NashResult:
    """Result of an Ising-based Nash equilibrium computation.

    Attributes
    ----------
    strategy : np.ndarray
        Mixed-strategy probability vector for each player.
    energy : float
        Ground state energy of the Ising model (lower = better equilibrium).
    expected_payoff : np.ndarray
        Expected payoff for each player under the equilibrium strategy.
    entanglement : float
        Entanglement entropy between players (measures strategic correlation).
    """

    strategy: np.ndarray
    energy: float
    expected_payoff: np.ndarray
    entanglement: float


@dataclass
class AuctionModelResult:
    """Result of a quantum auction simulation.

    Attributes
    ----------
    winning_bid : float
        The winning bid amount.
    winner : int
        Index of the winning bidder.
    revenue : float
        Total revenue from the auction.
    bidder_correlations : np.ndarray
        Quantum correlation matrix between bidders.
    optimal_bids : np.ndarray
        Optimal bid vector computed from the quantum model.
    """

    winning_bid: float
    winner: int
    revenue: float
    bidder_correlations: np.ndarray
    optimal_bids: np.ndarray


@dataclass
class MaxCutBenchmarkResult:
    """Result of QAOA vs exact MaxCut benchmarking.

    Attributes
    ----------
    exact_cut : float
        Exact maximum cut value from brute-force or Ising ground state.
    qaoa_cut : float
        Cut value achieved by QAOA.
    approximation_ratio : float
        QAOA cut / exact cut (1.0 = perfect).
    ising_energy : float
        Ground state energy of the Ising formulation.
    qaoa_expectation : float
        QAOA expectation value.
    landscape : dict
        Energy landscape data if landscape_analysis was performed.
    """

    exact_cut: float
    qaoa_cut: float
    approximation_ratio: float
    ising_energy: float
    qaoa_expectation: float
    landscape: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IsingGameSolver
# ---------------------------------------------------------------------------


@dataclass
class IsingGameSolver:
    """Map game theory payoff matrices to Ising Hamiltonians.

    Mixed-strategy Nash equilibria become ground states of a transverse-field
    Ising model where ZZ couplings encode the payoff structure.

    The mapping works as follows:
      - Each player-action pair is a qubit (n_players * n_actions qubits total)
      - ZZ couplings between players encode pairwise payoff interactions
      - The transverse field allows exploration of mixed strategies
      - Ground state amplitudes yield the equilibrium mixed-strategy probabilities

    Parameters
    ----------
    n_players : int
        Number of players in the game.
    n_actions : int
        Number of actions per player.
    transverse_field : float
        Transverse field strength controlling exploration of mixed strategies.
    """

    n_players: int = 2
    n_actions: int = 2
    transverse_field: float = 0.5

    @classmethod
    def from_payoff_matrix(
        cls,
        payoff_matrix: np.ndarray,
        transverse_field: float = 0.5,
    ) -> IsingGameSolver:
        """Build an IsingGameSolver from a payoff matrix.

        Parameters
        ----------
        payoff_matrix : np.ndarray
            For a 2-player game: shape (n_actions, n_actions, n_players)
            where payoff_matrix[i, j, p] is the payoff to player p when
            player 0 plays action i and player 1 plays action j.
        transverse_field : float
            Controls the degree of strategy mixing.

        Returns
        -------
        IsingGameSolver
            Configured solver with the payoff structure stored internally.
        """
        payoff_matrix = np.asarray(payoff_matrix, dtype=np.float64)
        if payoff_matrix.ndim == 2:
            # Symmetric game: same payoff for both players
            n_actions = payoff_matrix.shape[0]
            full = np.zeros((n_actions, n_actions, 2))
            full[:, :, 0] = payoff_matrix
            full[:, :, 1] = payoff_matrix.T
            payoff_matrix = full
        n_actions = payoff_matrix.shape[0]
        n_players = payoff_matrix.shape[2]
        solver = cls(
            n_players=n_players,
            n_actions=n_actions,
            transverse_field=transverse_field,
        )
        solver._payoff_matrix = payoff_matrix
        return solver

    def _build_hamiltonian(self) -> SparsePauliHamiltonian:
        """Build Ising Hamiltonian from payoff structure.

        Maps the payoff matrix to ZZ couplings. Positive payoff entries
        create ferromagnetic couplings (players tend to correlate),
        while negative entries create antiferromagnetic couplings.

        Returns
        -------
        SparsePauliHamiltonian
        """
        n_qubits = self.n_players * self.n_actions
        terms: list[PauliOperator] = []

        if not hasattr(self, "_payoff_matrix"):
            raise ValueError(
                "No payoff matrix set. Use from_payoff_matrix() to create "
                "the solver, or set _payoff_matrix manually."
            )

        payoff = self._payoff_matrix
        # Normalize payoffs to coupling range
        max_payoff = np.abs(payoff).max()
        if max_payoff < 1e-15:
            max_payoff = 1.0
        norm_payoff = payoff / max_payoff

        # ZZ couplings between player-action qubits
        for p1 in range(self.n_players):
            for a1 in range(self.n_actions):
                q1 = p1 * self.n_actions + a1
                for p2 in range(p1 + 1, self.n_players):
                    for a2 in range(self.n_actions):
                        q2 = p2 * self.n_actions + a2
                        # Coupling strength from average payoff of this action pair
                        coupling = (
                            norm_payoff[a1, a2, p1] + norm_payoff[a1, a2, p2]
                        ) / 2.0
                        if abs(coupling) > 1e-10:
                            label = ["I"] * n_qubits
                            label[q1] = "Z"
                            label[q2] = "Z"
                            terms.append(
                                PauliOperator("".join(label), coeff=-coupling)
                            )

        # Transverse field on each qubit
        for q in range(n_qubits):
            label = ["I"] * n_qubits
            label[q] = "X"
            terms.append(
                PauliOperator("".join(label), coeff=-self.transverse_field)
            )

        if not terms:
            # Fallback: pure transverse field
            for q in range(n_qubits):
                label = ["I"] * n_qubits
                label[q] = "X"
                terms.append(
                    PauliOperator("".join(label), coeff=-self.transverse_field)
                )

        return SparsePauliHamiltonian(terms)

    def nash_equilibrium(self) -> NashResult:
        """Compute Nash equilibrium by finding the Ising ground state.

        The ground state amplitudes are mapped to mixed-strategy probabilities
        for each player.

        Returns
        -------
        NashResult
            Nash equilibrium result with strategies, payoffs, and entanglement.
        """
        H = self._build_hamiltonian()
        mat = H.matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(mat)
        energy = float(eigenvalues[0])
        psi = eigenvectors[:, 0]

        n_qubits = self.n_players * self.n_actions
        probs = np.abs(psi) ** 2

        # Extract mixed strategies: marginalize over each player's qubits
        strategies = np.zeros((self.n_players, self.n_actions))
        for player in range(self.n_players):
            for action in range(self.n_actions):
                qubit = player * self.n_actions + action
                # Probability that this qubit is |1> (action selected)
                prob_one = 0.0
                for idx in range(len(probs)):
                    if (idx >> (n_qubits - 1 - qubit)) & 1:
                        prob_one += probs[idx]
                strategies[player, action] = prob_one

            # Normalize to get a probability distribution
            total = strategies[player].sum()
            if total > 1e-15:
                strategies[player] /= total
            else:
                strategies[player] = np.ones(self.n_actions) / self.n_actions

        # Compute expected payoffs
        payoff = self._payoff_matrix
        expected_payoff = np.zeros(self.n_players)
        for a1 in range(self.n_actions):
            for a2 in range(self.n_actions):
                joint_prob = strategies[0, a1] * strategies[1, a2]
                for p in range(self.n_players):
                    expected_payoff[p] += joint_prob * payoff[a1, a2, p]

        # Entanglement entropy between players
        n_a = self.n_actions
        dim_a = 2 ** n_a
        dim_b = 2 ** (n_qubits - n_a)
        psi_matrix = psi.reshape(dim_a, dim_b)
        s = np.linalg.svd(psi_matrix, compute_uv=False)
        s = s[s > 1e-12]
        entanglement = float(-np.sum(s**2 * np.log2(s**2 + 1e-30)))

        return NashResult(
            strategy=strategies,
            energy=energy,
            expected_payoff=expected_payoff,
            entanglement=entanglement,
        )

    def quantum_advantage(self) -> dict:
        """Estimate quantum advantage over classical Nash equilibrium.

        Compares the quantum (Ising ground state) equilibrium to the
        classical best-response dynamics and computes the gain in
        total payoff and strategy diversity.

        Returns
        -------
        dict
            Keys: quantum_payoff, classical_payoff, payoff_ratio,
            quantum_entropy, classical_entropy.
        """
        quantum_result = self.nash_equilibrium()
        quantum_total = float(quantum_result.expected_payoff.sum())

        # Classical: brute-force over pure strategy profiles
        payoff = self._payoff_matrix
        best_total = float("-inf")
        best_strat = np.zeros((self.n_players, self.n_actions))
        for a1 in range(self.n_actions):
            for a2 in range(self.n_actions):
                total = float(payoff[a1, a2, :].sum())
                if total > best_total:
                    best_total = total
                    best_strat = np.zeros((self.n_players, self.n_actions))
                    best_strat[0, a1] = 1.0
                    best_strat[1, a2] = 1.0

        classical_total = best_total

        # Strategy entropy as diversity measure
        def _entropy(probs: np.ndarray) -> float:
            p = probs[probs > 1e-15]
            return float(-np.sum(p * np.log2(p + 1e-30)))

        q_entropy = sum(_entropy(quantum_result.strategy[p])
                        for p in range(self.n_players))
        c_entropy = sum(_entropy(best_strat[p])
                        for p in range(self.n_players))

        payoff_ratio = quantum_total / classical_total if abs(classical_total) > 1e-15 else 1.0

        return {
            "quantum_payoff": quantum_total,
            "classical_payoff": classical_total,
            "payoff_ratio": payoff_ratio,
            "quantum_entropy": q_entropy,
            "classical_entropy": c_entropy,
        }


# ---------------------------------------------------------------------------
# QuantumAuctionModel
# ---------------------------------------------------------------------------


@dataclass
class QuantumAuctionModel:
    """Map auction theory to quantum circuits using entangled bidder states.

    Models bidder valuations as quantum states and uses entanglement to
    capture correlations between bidders. Supports Vickrey (second-price)
    and first-price sealed-bid auction formats.

    The quantum model encodes each bidder as a qubit. Bidder valuations
    set the rotation angles, and entanglement between qubits models
    correlated private values (e.g., common-value component in oil auctions).

    Parameters
    ----------
    n_bidders : int
        Number of bidders.
    auction_type : str
        Auction format: ``"vickrey"`` (second-price) or ``"first_price"``.
    correlation : float
        Bidder correlation strength in [0, 1]. Higher values mean more
        correlated valuations (common-value component).
    """

    n_bidders: int = 4
    auction_type: str = "vickrey"
    correlation: float = 0.3

    def simulate_auction(
        self,
        valuations: np.ndarray,
        seed: int = 42,
    ) -> AuctionModelResult:
        """Simulate an auction with quantum-correlated bidders.

        Parameters
        ----------
        valuations : np.ndarray
            True valuations for each bidder, shape (n_bidders,).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        AuctionModelResult
        """
        valuations = np.asarray(valuations, dtype=np.float64)
        n = min(len(valuations), self.n_bidders)
        rng = np.random.default_rng(seed)

        # Build quantum state encoding bidder valuations
        psi = self._build_bidder_state(valuations[:n])

        # Compute bidder correlation matrix from quantum state
        correlations = self._bidder_correlations(psi, n)

        # Compute optimal bids based on auction type
        optimal_bids = self._compute_optimal_bids(valuations[:n], correlations)

        # Determine winner and revenue
        if self.auction_type == "vickrey":
            # Vickrey: highest bidder wins, pays second-highest bid
            sorted_indices = np.argsort(optimal_bids)[::-1]
            winner = int(sorted_indices[0])
            winning_bid = float(optimal_bids[winner])
            revenue = float(optimal_bids[sorted_indices[1]]) if n > 1 else winning_bid
        else:
            # First-price: highest bidder wins, pays their own bid
            winner = int(np.argmax(optimal_bids))
            winning_bid = float(optimal_bids[winner])
            revenue = winning_bid

        return AuctionModelResult(
            winning_bid=winning_bid,
            winner=winner,
            revenue=revenue,
            bidder_correlations=correlations,
            optimal_bids=optimal_bids,
        )

    def optimal_bid(
        self,
        valuation: float,
        bidder_index: int = 0,
        other_valuations: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the optimal bid for a single bidder.

        Uses the quantum model to account for bidder correlations
        when computing the optimal shading strategy.

        Parameters
        ----------
        valuation : float
            The bidder's true valuation.
        bidder_index : int
            Index of the bidder.
        other_valuations : np.ndarray or None
            Estimated valuations of other bidders. If None, assumes
            uniform distribution centered on the bidder's valuation.

        Returns
        -------
        float
            Optimal bid amount.
        """
        n = self.n_bidders
        if other_valuations is None:
            other_valuations = np.full(n - 1, valuation * 0.9)

        all_vals = np.zeros(n)
        all_vals[bidder_index] = valuation
        idx = 0
        for i in range(n):
            if i != bidder_index:
                all_vals[i] = other_valuations[idx]
                idx += 1

        psi = self._build_bidder_state(all_vals)
        correlations = self._bidder_correlations(psi, n)
        optimal_bids = self._compute_optimal_bids(all_vals, correlations)
        return float(optimal_bids[bidder_index])

    def _build_bidder_state(self, valuations: np.ndarray) -> np.ndarray:
        """Encode bidder valuations into an entangled quantum state.

        Each bidder is a qubit rotated by Ry(pi * v_i / v_max), then
        nearest-neighbour controlled-Ry gates create correlation that
        manifests in Z-basis measurement statistics.
        """
        n = len(valuations)
        v_max = np.max(np.abs(valuations)) + 1e-15

        # Start from |0...0>
        psi = np.zeros(2**n, dtype=np.complex128)
        psi[0] = 1.0

        # Apply Ry rotations based on valuations
        for i in range(n):
            angle = np.pi * valuations[i] / v_max
            psi = self._apply_ry(psi, n, i, angle)

        # Apply controlled-Ry gates for correlations (affects Z-basis statistics)
        if self.correlation > 1e-15:
            for i in range(n - 1):
                psi = self._apply_controlled_ry(
                    psi, n, i, i + 1, self.correlation * np.pi
                )

        # Normalize
        norm = np.linalg.norm(psi)
        if norm > 1e-15:
            psi /= norm

        return psi

    def _bidder_correlations(self, psi: np.ndarray, n: int) -> np.ndarray:
        """Compute ZZ correlation matrix between bidders."""
        probs = np.abs(psi) ** 2
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr[i, j] = 1.0
                else:
                    # <Z_i Z_j>
                    expectation = 0.0
                    for idx in range(len(probs)):
                        bit_i = (idx >> (n - 1 - i)) & 1
                        bit_j = (idx >> (n - 1 - j)) & 1
                        z_i = 1.0 - 2.0 * bit_i
                        z_j = 1.0 - 2.0 * bit_j
                        expectation += z_i * z_j * probs[idx]
                    corr[i, j] = expectation
        return corr

    def _compute_optimal_bids(
        self,
        valuations: np.ndarray,
        correlations: np.ndarray,
    ) -> np.ndarray:
        """Compute optimal bids using quantum correlation information.

        In a first-price auction, optimal shading depends on the number
        of competitors and their correlation with the bidder.
        In a Vickrey auction, truth-telling is optimal with independent
        values, but correlated values modify the strategy.
        """
        n = len(valuations)
        bids = np.zeros(n)

        for i in range(n):
            if self.auction_type == "vickrey":
                # In Vickrey with correlated values, bid slightly below valuation
                # proportional to correlation with other bidders
                mean_corr = float(np.mean(
                    [abs(correlations[i, j]) for j in range(n) if j != i]
                )) if n > 1 else 0.0
                shade = mean_corr * 0.1  # Small shading for correlated Vickrey
                bids[i] = valuations[i] * (1.0 - shade)
            else:
                # First-price: shade bid proportional to 1/(n*competition)
                mean_corr = float(np.mean(
                    [abs(correlations[i, j]) for j in range(n) if j != i]
                )) if n > 1 else 0.0
                # More competition + correlation -> more shading
                shade = (1.0 + mean_corr) / max(n, 2)
                bids[i] = valuations[i] * (1.0 - shade)

        return bids

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

    @staticmethod
    def _apply_controlled_ry(
        state: np.ndarray,
        n_qubits: int,
        control: int,
        target: int,
        angle: float,
    ) -> np.ndarray:
        """Apply controlled-Ry gate: Ry(angle) on target when control is |1>.

        Unlike CZ, this changes Z-basis measurement probabilities,
        creating genuine correlations in the computational basis.
        """
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        new_state = state.copy()
        step = 1 << (n_qubits - 1 - target)
        for i in range(len(state)):
            ctrl_bit = (i >> (n_qubits - 1 - control)) & 1
            tgt_bit = (i >> (n_qubits - 1 - target)) & 1
            if ctrl_bit == 1 and tgt_bit == 0:
                partner = i ^ step
                a0 = state[i]
                a1 = state[partner]
                new_state[i] = c * a0 - s * a1
                new_state[partner] = s * a0 + c * a1
        return new_state


# ---------------------------------------------------------------------------
# QuantumMaxCutBridge
# ---------------------------------------------------------------------------


@dataclass
class QuantumMaxCutBridge:
    """Connect nqpu.games MaxCut with physics Ising model for QAOA benchmarking.

    Provides a unified interface to:
      1. Solve MaxCut exactly via brute force
      2. Solve MaxCut via QAOA simulation
      3. Map the MaxCut problem to an Ising Hamiltonian
      4. Compare results and compute approximation ratios

    Parameters
    ----------
    graph : Graph
        The graph on which to solve MaxCut.
    """

    graph: Graph

    def compare_qaoa_exact(
        self,
        p: int = 2,
        n_restarts: int = 3,
        max_iter: int = 50,
        seed: int = 42,
    ) -> MaxCutBenchmarkResult:
        """Compare QAOA against exact MaxCut solution.

        Parameters
        ----------
        p : int
            QAOA circuit depth.
        n_restarts : int
            Number of random restarts for QAOA optimisation.
        max_iter : int
            Maximum iterations per restart.
        seed : int
            Random seed.

        Returns
        -------
        MaxCutBenchmarkResult
        """
        n = self.graph.n
        maxcut = MaxCut(self.graph)

        # Exact solution
        exact_result = maxcut.brute_force()
        exact_cut = exact_result.objective

        # Build adjacency matrix for QAOA
        adj = np.zeros((n, n))
        for (u, v), w in self.graph.edges.items():
            adj[u, v] = w
            adj[v, u] = w

        # QAOA solution
        circuit = maxcut_qaoa(adj, p=p)
        rng = np.random.default_rng(seed)
        qaoa_opt = circuit.optimize(n_restarts=n_restarts, max_iter=max_iter, rng=rng)

        # Evaluate QAOA solution as a cut
        qaoa_assignment = qaoa_opt.best_bitstring
        qaoa_cut = maxcut.evaluate(qaoa_assignment)

        # Ising ground state energy
        ising_energy = self._ising_ground_energy(adj)

        approx_ratio = qaoa_cut / exact_cut if exact_cut > 1e-15 else 1.0

        return MaxCutBenchmarkResult(
            exact_cut=exact_cut,
            qaoa_cut=qaoa_cut,
            approximation_ratio=approx_ratio,
            ising_energy=ising_energy,
            qaoa_expectation=qaoa_opt.best_expectation,
        )

    def landscape_analysis(
        self,
        p: int = 1,
        n_gamma: int = 20,
        n_beta: int = 20,
    ) -> dict:
        """Compute the QAOA energy landscape for depth p=1.

        Sweeps gamma and beta over a grid and evaluates the QAOA
        expectation value at each point.

        Parameters
        ----------
        p : int
            QAOA depth (typically 1 for landscape visualisation).
        n_gamma : int
            Number of gamma grid points.
        n_beta : int
            Number of beta grid points.

        Returns
        -------
        dict
            Keys: gammas, betas, landscape (2D array), optimal_gamma,
            optimal_beta, optimal_expectation.
        """
        n = self.graph.n
        adj = np.zeros((n, n))
        for (u, v), w in self.graph.edges.items():
            adj[u, v] = w
            adj[v, u] = w

        circuit = maxcut_qaoa(adj, p=p)

        gammas = np.linspace(0, 2 * np.pi, n_gamma)
        betas = np.linspace(0, np.pi, n_beta)
        landscape = np.zeros((n_gamma, n_beta))

        best_exp = float("-inf")
        best_gamma = 0.0
        best_beta = 0.0

        for i, g in enumerate(gammas):
            for j, b in enumerate(betas):
                g_arr = np.array([g] * p)
                b_arr = np.array([b] * p)
                result = circuit.evaluate(g_arr, b_arr)
                landscape[i, j] = result.expectation
                if result.expectation > best_exp:
                    best_exp = result.expectation
                    best_gamma = g
                    best_beta = b

        return {
            "gammas": gammas,
            "betas": betas,
            "landscape": landscape,
            "optimal_gamma": best_gamma,
            "optimal_beta": best_beta,
            "optimal_expectation": best_exp,
        }

    def _ising_ground_energy(self, adj: np.ndarray) -> float:
        """Compute ground state energy of the MaxCut Ising Hamiltonian.

        The MaxCut Ising encoding is:
            H = -sum_{(i,j)} w_ij/2 * Z_i Z_j
        (dropping constant terms).
        """
        n = adj.shape[0]
        terms: list[PauliOperator] = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(adj[i, j]) > 1e-15:
                    label = ["I"] * n
                    label[i] = "Z"
                    label[j] = "Z"
                    terms.append(
                        PauliOperator("".join(label), coeff=-adj[i, j] / 2.0)
                    )
        if not terms:
            return 0.0
        H = SparsePauliHamiltonian(terms)
        eigenvalues = np.linalg.eigvalsh(H.matrix())
        return float(eigenvalues[0])
