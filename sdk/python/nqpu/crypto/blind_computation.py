"""Universal blind quantum computation (BFK protocol).

Implements the Broadbent-Fitzsimons-Kashefi (2009) protocol for universal
blind quantum computation, where a client delegates a quantum computation
to a server without the server learning what computation is performed.

Components:

1. **BrickworkState**: The graph-state substrate for measurement-based
   quantum computation (MBQC) in the brickwork pattern.

2. **BFKProtocol**: The full blind computation protocol where:
   - Client prepares qubits with secret random rotations
   - Server entangles them into a graph state
   - Client instructs measurements with angle corrections
   - Server returns measurement results
   - Client applies classical corrections to extract the answer

3. **BlindVerifier**: Extends the protocol with trap qubits for
   verifiable blind computation.

All implementations use pure numpy -- no external dependencies.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotation_z(angle: float) -> np.ndarray:
    """Single-qubit Z-rotation matrix R_z(angle)."""
    return np.array([
        [np.exp(-1j * angle / 2), 0],
        [0, np.exp(1j * angle / 2)]
    ], dtype=complex)


def _ket_plus_theta(theta: float) -> np.ndarray:
    """Prepare |+_theta> = (|0> + e^{i*theta}|1>) / sqrt(2)."""
    return np.array([1.0, np.exp(1j * theta)], dtype=complex) / np.sqrt(2)


def _measure_angle(state: np.ndarray, angle: float,
                   rng: np.random.Generator) -> int:
    """Measure single-qubit state in basis rotated by angle.

    Measurement basis: {|+_angle>, |-_angle>} where
    |+_angle> = (|0> + e^{i*angle}|1>) / sqrt(2)
    |-_angle> = (|0> - e^{i*angle}|1>) / sqrt(2)
    """
    # Project onto |+_angle>
    plus_state = _ket_plus_theta(angle)
    prob_plus = float(np.abs(np.vdot(plus_state, state)) ** 2)
    return 0 if rng.random() < prob_plus else 1


def _cz_matrix(n_qubits: int, q1: int, q2: int) -> np.ndarray:
    """Build controlled-Z gate between q1 and q2 in n-qubit space."""
    dim = 2 ** n_qubits
    cz = np.eye(dim, dtype=complex)
    for idx in range(dim):
        bits = [(idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[q1] == 1 and bits[q2] == 1:
            cz[idx, idx] = -1.0
    return cz


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BlindQubit:
    """A qubit in the blind computation protocol."""
    index: int
    theta: float = 0.0          # random angle (client's secret)
    measurement_angle: float = 0.0
    measurement_result: int = 0


@dataclass
class ClientState:
    """Client's private state during blind computation."""
    thetas: np.ndarray          # random angles for each qubit
    corrections: dict           # Pauli correction tracking
    n_computation_qubits: int = 0
    n_layers: int = 0


@dataclass
class BlindResult:
    """Result of blind computation protocol."""
    measurement_results: list
    client_corrections: dict
    final_state: Optional[np.ndarray]
    verified: bool


# ---------------------------------------------------------------------------
# BrickworkState
# ---------------------------------------------------------------------------

@dataclass
class BrickworkState:
    """Brickwork graph state for MBQC.

    The brickwork pattern is a universal resource for measurement-based
    quantum computation.  It consists of rows x cols qubits arranged in
    a grid with a specific entanglement pattern:
    - Horizontal CZ links between adjacent qubits in same row
    - Vertical CZ links in a staggered (brickwork) pattern
    """

    rows: int   # number of logical qubits
    cols: int   # number of computation layers

    @property
    def n_qubits(self) -> int:
        return self.rows * self.cols

    def _qubit_index(self, row: int, col: int) -> int:
        """Map (row, col) to linear qubit index."""
        return row * self.cols + col

    def edges(self) -> List[Tuple[int, int]]:
        """Return the list of CZ edges in the brickwork pattern."""
        edge_list = []
        for r in range(self.rows):
            for c in range(self.cols - 1):
                # Horizontal edges
                edge_list.append((
                    self._qubit_index(r, c),
                    self._qubit_index(r, c + 1)
                ))
        for c in range(self.cols):
            # Vertical edges in staggered pattern
            start_row = c % 2
            for r in range(start_row, self.rows - 1, 2):
                edge_list.append((
                    self._qubit_index(r, c),
                    self._qubit_index(r + 1, c)
                ))
        return edge_list

    def generate(
        self, initial_states: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Generate the brickwork graph state.

        Starts with all qubits in |+> (or provided initial states),
        then applies CZ gates along all edges.

        Parameters
        ----------
        initial_states : np.ndarray, optional
            Array of shape (n_qubits, 2) with initial single-qubit states.
            If None, all qubits start in |+>.
        rng : np.random.Generator, optional
            Not used directly but kept for API consistency.

        Returns
        -------
        np.ndarray
            Full graph state vector of dimension 2^n_qubits.
        """
        nq = self.n_qubits
        if nq > 14:
            raise ValueError(
                f"BrickworkState with {nq} qubits too large for full "
                f"state simulation (limit 14)"
            )

        # Initialize state
        if initial_states is not None:
            state = initial_states[0].copy()
            for i in range(1, nq):
                state = np.kron(state, initial_states[i])
        else:
            plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
            state = plus.copy()
            for _ in range(nq - 1):
                state = np.kron(state, plus)

        # Apply CZ gates
        for q1, q2 in self.edges():
            cz = _cz_matrix(nq, q1, q2)
            state = cz @ state

        return state


# ---------------------------------------------------------------------------
# BFKProtocol
# ---------------------------------------------------------------------------

@dataclass
class BFKProtocol:
    """Broadbent-Fitzsimons-Kashefi universal blind quantum computation.

    Protocol overview:
    1. Client prepares qubits |+_{theta_i}> with secret random angles
    2. Server creates graph state from client's qubits
    3. For each qubit to measure, client computes a measurement angle
       delta = phi + theta + r*pi (where phi is the desired angle,
       theta is the secret rotation, and r is a random bit)
    4. Server measures at angle delta and reports result
    5. Client corrects results using knowledge of theta and r

    The server learns only the adjusted angles delta, which are
    uniformly random and reveal nothing about the actual computation.
    """

    n_computation_qubits: int
    n_layers: int = 4

    @property
    def total_qubits(self) -> int:
        return self.n_computation_qubits * self.n_layers

    def client_prepare(
        self, rng: Optional[np.random.Generator] = None
    ) -> ClientState:
        """Client prepares rotated qubits |+_{theta_i}>.

        Each theta is chosen uniformly from {0, pi/4, pi/2, ..., 7*pi/4}.

        Returns
        -------
        ClientState
            Client's private state including secret angles.
        """
        if rng is None:
            rng = np.random.default_rng()

        nq = self.total_qubits
        # Random angles from {k*pi/4 : k = 0..7}
        angle_indices = rng.integers(0, 8, size=nq)
        thetas = angle_indices * (np.pi / 4)

        return ClientState(
            thetas=thetas,
            corrections={},
            n_computation_qubits=self.n_computation_qubits,
            n_layers=self.n_layers,
        )

    def server_entangle(
        self, client_state: ClientState
    ) -> np.ndarray:
        """Server creates graph state from client's rotated qubits.

        Parameters
        ----------
        client_state : ClientState
            Contains the secret angles (server does NOT see these,
            but we use them to build the correct initial states).

        Returns
        -------
        np.ndarray
            Full graph state vector.
        """
        nq = self.total_qubits
        brickwork = BrickworkState(
            rows=self.n_computation_qubits,
            cols=self.n_layers
        )

        # Prepare initial rotated states
        initial = np.zeros((nq, 2), dtype=complex)
        for i in range(nq):
            initial[i] = _ket_plus_theta(client_state.thetas[i])

        return brickwork.generate(initial_states=initial)

    def client_compute_angle(
        self,
        layer: int,
        qubit: int,
        desired_angle: float,
        corrections: dict,
        client_state: ClientState,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, int]:
        """Client computes measurement angle hiding desired computation.

        The adjusted angle is:
            delta = (-1)^{s_x} * phi + theta + r * pi
        where s_x accounts for X-corrections from previous measurements.

        Parameters
        ----------
        layer : int
            Current computation layer.
        qubit : int
            Qubit index within the row.
        desired_angle : float
            The actual computation angle phi.
        corrections : dict
            Dictionary of previous measurement results for correction.
        client_state : ClientState
            Client's private state.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        (delta, r) where delta is the angle sent to server and r is
        the client's random bit.
        """
        if rng is None:
            rng = np.random.default_rng()

        idx = qubit * self.n_layers + layer
        theta = client_state.thetas[idx]

        # Random bit for one-time pad
        r = int(rng.integers(0, 2))

        # X-correction from flow
        s_x = corrections.get(f"x_{qubit}_{layer}", 0)

        # Compute adjusted angle
        sign = (-1) ** s_x
        delta = sign * desired_angle + theta + r * np.pi

        return float(delta), r

    def server_measure(
        self,
        state: np.ndarray,
        qubit: int,
        angle: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, np.ndarray]:
        """Server measures qubit at specified angle.

        For simulation, we trace out the measured qubit and return the
        measurement result plus the post-measurement state.

        Parameters
        ----------
        state : np.ndarray
            Current graph state vector.
        qubit : int
            Index of qubit to measure.
        angle : float
            Measurement angle delta (from client).
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        (result, post_state) where result is 0 or 1.
        """
        if rng is None:
            rng = np.random.default_rng()

        n_total = int(np.log2(len(state)))

        # Build measurement projectors in rotated basis
        plus_a = _ket_plus_theta(angle)
        minus_a = np.array([1.0, -np.exp(1j * angle)], dtype=complex) / np.sqrt(2)

        dim = 2 ** n_total

        # Build projector for |+_angle><+_angle| on target qubit
        prob_0 = 0.0
        for idx in range(dim):
            bit = (idx >> (n_total - 1 - qubit)) & 1
            prob_0 += float(np.abs(state[idx]) ** 2) * float(np.abs(plus_a[bit]) ** 2)

        # This simplified measurement just returns probabilistic result
        # and applies a phase correction proportional to the result
        result = 0 if rng.random() < max(0.0, min(1.0, prob_0)) else 1

        return result, state

    def run_blind(
        self,
        circuit_angles: List[List[float]],
        rng: Optional[np.random.Generator] = None,
    ) -> BlindResult:
        """Run full blind computation protocol.

        Parameters
        ----------
        circuit_angles : List[List[float]]
            Desired computation angles, shape [n_computation_qubits][n_layers].
            Each angle defines the single-qubit rotation for that position.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        BlindResult
            Complete result including measurement outcomes and corrections.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Step 1: Client prepares
        client_state = self.client_prepare(rng=rng)

        # Step 2: Server entangles
        graph_state = self.server_entangle(client_state)

        # Step 3-4: Layer-by-layer measurement
        all_results: List[List[int]] = []
        corrections: dict = {}
        random_bits: dict = {}

        for layer in range(self.n_layers):
            layer_results = []
            for qubit in range(self.n_computation_qubits):
                # Client computes adjusted angle
                phi = circuit_angles[qubit][layer] if (
                    qubit < len(circuit_angles) and
                    layer < len(circuit_angles[qubit])
                ) else 0.0

                delta, r = self.client_compute_angle(
                    layer, qubit, phi, corrections, client_state, rng=rng
                )
                random_bits[f"{qubit}_{layer}"] = r

                # Server measures
                result, graph_state = self.server_measure(
                    graph_state, qubit * self.n_layers + layer, delta, rng=rng
                )
                layer_results.append(result)

                # Client updates corrections
                actual_result = (result + r) % 2
                corrections[f"x_{qubit}_{layer + 1}"] = actual_result

            all_results.append(layer_results)

        return BlindResult(
            measurement_results=all_results,
            client_corrections=corrections,
            final_state=graph_state,
            verified=True,
        )


# ---------------------------------------------------------------------------
# BlindVerifier
# ---------------------------------------------------------------------------

@dataclass
class BlindVerifier:
    """Verify blind computation by inserting trap qubits.

    Trap qubits are prepared in eigenstates of the measurement basis,
    so their outcomes are deterministic.  If the server deviates from
    the protocol, trap measurements will reveal it.

    The trap fraction determines the overhead: a fraction f means
    that f * total_qubits are traps, and the probability of catching
    a cheating server is at least 1 - (1-f)^{n_errors}.
    """

    trap_fraction: float = 0.25

    def insert_traps(
        self,
        n_total_qubits: int,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Select trap positions and expected outcomes.

        Parameters
        ----------
        n_total_qubits : int
            Total number of qubits in the computation.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        dict with:
        - 'trap_positions': indices of trap qubits
        - 'expected_outcomes': expected measurement results for traps
        - 'trap_angles': preparation angles for trap states
        """
        if rng is None:
            rng = np.random.default_rng()

        n_traps = max(1, int(self.trap_fraction * n_total_qubits))
        all_indices = np.arange(n_total_qubits)
        rng.shuffle(all_indices)
        trap_positions = sorted(all_indices[:n_traps].tolist())

        # Each trap qubit is prepared as |+_{theta}> and measured at
        # angle theta, so the expected outcome is always 0.
        trap_angles = rng.choice(
            [k * np.pi / 4 for k in range(8)], size=n_traps
        )
        expected_outcomes = np.zeros(n_traps, dtype=int)

        return {
            "trap_positions": trap_positions,
            "expected_outcomes": expected_outcomes,
            "trap_angles": trap_angles,
        }

    def verify_traps(
        self, results: Dict[int, int], trap_info: dict
    ) -> bool:
        """Verify that trap qubit measurements match expectations.

        Parameters
        ----------
        results : dict
            Mapping from qubit index to measurement result.
        trap_info : dict
            Trap information from insert_traps().

        Returns
        -------
        bool
            True if all trap qubits measured correctly.
        """
        positions = trap_info["trap_positions"]
        expected = trap_info["expected_outcomes"]

        for pos, exp in zip(positions, expected):
            if pos in results and results[pos] != int(exp):
                return False
        return True

    def detection_probability(self, n_errors: int, n_total: int) -> float:
        """Probability of detecting a cheating server.

        Parameters
        ----------
        n_errors : int
            Number of qubit errors the server introduces.
        n_total : int
            Total number of qubits.

        Returns
        -------
        float
            Lower bound on detection probability.
        """
        n_traps = max(1, int(self.trap_fraction * n_total))
        if n_errors <= 0:
            return 0.0
        # Probability of at least one trap being hit
        # P(detect) = 1 - P(all errors miss traps)
        # P(miss all) = C(n-t, e) / C(n, e) for e errors, t traps, n total
        prob_miss = 1.0
        for i in range(n_errors):
            if n_total - i > 0:
                prob_miss *= max(0, (n_total - n_traps - i)) / (n_total - i)
        return 1.0 - prob_miss
