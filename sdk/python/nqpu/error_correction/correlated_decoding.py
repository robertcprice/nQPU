"""Correlated decoding with sliding window and space-time matching.

Implements advanced decoding techniques for quantum error correction
that exploit temporal correlations in syndrome data:

  - :class:`SyndromeHistory` -- time series of syndrome measurements
  - :class:`SlidingWindowDecoder` -- streaming decoder with windowed
    processing for low-latency real-time decoding
  - :class:`SpaceTimeMWPM` -- space-time minimum weight matching that
    treats time as an additional dimension in the matching graph
  - :class:`CorrelatedNoiseModel` -- noise with spatial and temporal
    correlations between error events
  - :class:`DecodingBenchmark` -- comparison framework for correlated
    vs independent decoders

References:
  - Dennis et al., "Topological quantum memory" (JMP 2002) --
    space-time matching concept
  - Skoric et al., "Parallel window decoding enables scalable fault
    tolerant quantum computation" (Nature Communications 2023)
  - Huang et al., "Fault-tolerant weighted union-find decoding on
    the toric code" (PRA 2020)

All implementations are pure numpy with no external dependencies.

Example:
    from nqpu.error_correction.correlated_decoding import (
        SyndromeHistory, SlidingWindowDecoder, CorrelatedNoiseModel,
        DecodingBenchmark,
    )

    # Benchmark correlated decoding
    from nqpu.error_correction import SurfaceCode
    code = SurfaceCode(distance=3)
    noise = CorrelatedNoiseModel(base_rate=0.05, temporal_correlation=0.3)
    bench = DecodingBenchmark(code=code, noise_model=noise)
    result = bench.compare_decoders(n_shots=500, rng=np.random.default_rng(42))
    print(f"Improvement: {result.improvement_factor:.2f}x")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ #
# Syndrome History
# ------------------------------------------------------------------ #

@dataclass
class SyndromeHistory:
    """Time series of syndrome measurements.

    Stores multiple rounds of syndrome extraction results and provides
    utilities for computing syndrome differences (detecting changes
    between rounds).

    In a fault-tolerant decoder, we typically decode the *differences*
    between consecutive syndrome rounds, since a persistent syndrome
    defect indicates an ongoing stabilizer violation while a transient
    defect suggests a measurement error.

    Parameters
    ----------
    syndromes : np.ndarray
        Syndrome data of shape (n_rounds, n_stabilizers).
        Each row is one round of syndrome measurements.
    """

    syndromes: np.ndarray

    def __post_init__(self) -> None:
        if self.syndromes.ndim == 1:
            self.syndromes = self.syndromes.reshape(1, -1)
        self.syndromes = np.asarray(self.syndromes, dtype=np.int8)

    @property
    def n_rounds(self) -> int:
        """Number of syndrome measurement rounds."""
        return self.syndromes.shape[0]

    @property
    def n_stabilizers(self) -> int:
        """Number of stabilizer checks per round."""
        return self.syndromes.shape[1]

    def diff(self) -> np.ndarray:
        """Compute syndrome differences between consecutive rounds.

        Returns
        -------
        np.ndarray
            Differences of shape (n_rounds - 1, n_stabilizers).
            Entry [t, s] = 1 iff syndrome s changed between round t and t+1.
        """
        if self.n_rounds < 2:
            return np.zeros((0, self.n_stabilizers), dtype=np.int8)
        return ((self.syndromes[1:] - self.syndromes[:-1]) % 2).astype(np.int8)

    def diff_with_initial(self) -> np.ndarray:
        """Compute differences including the initial round.

        The first row compares round 0 to the all-zero reference (fresh start).
        Subsequent rows are standard differences.

        Returns
        -------
        np.ndarray
            Differences of shape (n_rounds, n_stabilizers).
        """
        result = np.zeros_like(self.syndromes)
        result[0] = self.syndromes[0]
        if self.n_rounds > 1:
            result[1:] = (self.syndromes[1:] - self.syndromes[:-1]) % 2
        return result.astype(np.int8)

    def get_round(self, t: int) -> np.ndarray:
        """Get syndrome for a specific round."""
        return self.syndromes[t].copy()

    def window(self, start: int, size: int) -> np.ndarray:
        """Extract a time window of syndromes.

        Parameters
        ----------
        start : int
            Starting round index.
        size : int
            Number of rounds in the window.

        Returns
        -------
        np.ndarray
            Syndrome data of shape (size, n_stabilizers).
        """
        end = min(start + size, self.n_rounds)
        return self.syndromes[start:end].copy()


# ------------------------------------------------------------------ #
# Sliding Window Decoder
# ------------------------------------------------------------------ #

@dataclass
class SlidingWindowDecoder:
    """Sliding window decoder for streaming syndrome data.

    Processes syndrome measurements in overlapping windows, committing
    corrections for early rounds while keeping later rounds in a buffer
    for future refinement.  This enables real-time decoding with bounded
    latency.

    The window slides forward by ``commit_size`` rounds at each step.
    The ``window_size`` determines how many rounds of context are used
    for each decoding step.

    Parameters
    ----------
    code : object
        Any code with a ``syndrome(error)`` method and ``n`` attribute.
    window_size : int
        Number of syndrome rounds in each decoding window.
    commit_size : int
        Number of rounds committed per step (must be <= window_size).
    """

    code: Any
    window_size: int = 5
    commit_size: int = 2

    def __post_init__(self) -> None:
        if self.commit_size > self.window_size:
            raise ValueError("commit_size must be <= window_size")

    def decode_stream(
        self, syndrome_history: SyndromeHistory
    ) -> List[np.ndarray]:
        """Process syndrome stream with sliding window.

        Decodes the syndrome history by sliding a window of
        ``window_size`` rounds across the data, committing corrections
        for the first ``commit_size`` rounds in each window.

        Parameters
        ----------
        syndrome_history : SyndromeHistory
            Full syndrome measurement history.

        Returns
        -------
        list of np.ndarray
            One correction vector (symplectic form, length 2*n) per round
            of committed corrections.
        """
        n_rounds = syndrome_history.n_rounds
        n = self.code.n
        corrections: List[np.ndarray] = []
        diffs = syndrome_history.diff_with_initial()

        pos = 0
        while pos < n_rounds:
            # Extract window
            end = min(pos + self.window_size, n_rounds)
            window_diffs = diffs[pos:end]

            # Decode the window: find corrections for each round in the window
            window_corrections = self._decode_window(window_diffs)

            # Commit first commit_size rounds
            commit_end = min(self.commit_size, len(window_corrections))
            for i in range(commit_end):
                corrections.append(window_corrections[i])

            pos += self.commit_size

        return corrections

    def _decode_window(
        self, window_diffs: np.ndarray
    ) -> List[np.ndarray]:
        """Decode a single window of syndrome differences.

        Uses majority voting across the window to distinguish data
        errors from measurement errors: if a syndrome defect persists
        across multiple rounds, it is likely a real data error.

        Parameters
        ----------
        window_diffs : np.ndarray
            Syndrome differences for this window, shape (w, n_stabs).

        Returns
        -------
        list of np.ndarray
            Correction for each round in the window.
        """
        n = self.code.n
        n_rounds = window_diffs.shape[0]
        corrections: List[np.ndarray] = []

        # Accumulated syndrome (running XOR)
        accumulated = np.zeros(window_diffs.shape[1], dtype=np.int8)

        for t in range(n_rounds):
            accumulated = (accumulated + window_diffs[t]) % 2

            # Find defects in the accumulated syndrome
            defects = np.where(accumulated == 1)[0]

            if len(defects) == 0:
                corrections.append(np.zeros(2 * n, dtype=np.int8))
                continue

            # Use the code's check matrices to find a correction
            correction = self._greedy_correction(accumulated)
            corrections.append(correction)

            # After applying correction, the accumulated syndrome should clear
            # (for committed rounds)
            check_syn = self.code.syndrome(correction)
            accumulated = (accumulated + check_syn[:len(accumulated)]) % 2

        return corrections

    def _greedy_correction(self, syndrome: np.ndarray) -> np.ndarray:
        """Find a correction that clears the given syndrome.

        Greedy approach: for each defect, find a single-qubit error
        that flips that syndrome bit and apply it.
        """
        n = self.code.n
        correction = np.zeros(2 * n, dtype=np.int8)
        remaining = syndrome.copy()

        for _ in range(n):
            defects = np.where(remaining == 1)[0]
            if len(defects) == 0:
                break

            # Try single-qubit errors to clear the first defect
            best_err = None
            best_cleared = 0

            for q in range(n):
                for pauli in ["X", "Z", "Y"]:
                    err = np.zeros(2 * n, dtype=np.int8)
                    if pauli in ("X", "Y"):
                        err[q] = 1
                    if pauli in ("Z", "Y"):
                        err[n + q] = 1

                    syn = self.code.syndrome(err)
                    # Truncate syndrome to match remaining length
                    syn_trunc = syn[:len(remaining)]
                    cleared = np.sum((remaining + syn_trunc) % 2 == 0)
                    if cleared > best_cleared:
                        best_cleared = cleared
                        best_err = err.copy()

            if best_err is not None:
                correction = (correction + best_err) % 2
                syn = self.code.syndrome(best_err)
                remaining = (remaining + syn[:len(remaining)]) % 2

        return correction


# ------------------------------------------------------------------ #
# Space-Time MWPM
# ------------------------------------------------------------------ #

@dataclass
class SpaceTimeMWPM:
    """Space-time minimum weight matching for correlated errors.

    Extends standard MWPM decoding by treating time as an additional
    spatial dimension.  Syndrome defects are nodes in a 3D graph
    (space x space x time), and matching considers both spatial and
    temporal distances.

    This allows the decoder to distinguish data errors (which create
    persistent syndrome defects) from measurement errors (which create
    transient defects).

    Parameters
    ----------
    code : object
        Any code with check matrices (Hx, Hz) and n attribute.
    n_rounds : int
        Number of syndrome measurement rounds.
    measurement_error_rate : float
        Probability of a syndrome measurement bit flip.
    """

    code: Any
    n_rounds: int
    measurement_error_rate: float = 0.01

    def build_graph(
        self, syndrome_diffs: np.ndarray
    ) -> Dict[str, Any]:
        """Build space-time matching graph from syndrome differences.

        Nodes are (time, stabilizer_index) pairs where syndrome_diffs
        is nonzero.  Edges connect:
          - Spatial neighbors within the same time slice (weight = 1)
          - Temporal neighbors at the same stabilizer (weight based on
            measurement error rate)
          - Boundary nodes for unmatched defects

        Parameters
        ----------
        syndrome_diffs : np.ndarray
            Syndrome differences, shape (n_rounds, n_stabilizers).

        Returns
        -------
        dict
            Graph with 'nodes', 'edges', and 'weights'.
        """
        nodes: List[Tuple[int, int]] = []
        node_indices: Dict[Tuple[int, int], int] = {}

        # Find all defect locations
        for t in range(syndrome_diffs.shape[0]):
            for s in range(syndrome_diffs.shape[1]):
                if syndrome_diffs[t, s]:
                    idx = len(nodes)
                    nodes.append((t, s))
                    node_indices[(t, s)] = idx

        if not nodes:
            return {"nodes": [], "edges": [], "weights": []}

        # Build edges
        edges: List[Tuple[int, int]] = []
        weights: List[float] = []

        n_nodes = len(nodes)
        meas_weight = -np.log(self.measurement_error_rate + 1e-15)

        for i in range(n_nodes):
            t_i, s_i = nodes[i]
            for j in range(i + 1, n_nodes):
                t_j, s_j = nodes[j]

                # Spatial edge (same time, adjacent stabilizers)
                if t_i == t_j:
                    # Weight based on spatial distance
                    w = abs(s_i - s_j)
                    if w <= 2:  # only connect nearby stabilizers
                        edges.append((i, j))
                        weights.append(float(w))

                # Temporal edge (same stabilizer, adjacent times)
                elif s_i == s_j and abs(t_i - t_j) == 1:
                    edges.append((i, j))
                    weights.append(meas_weight)

        # Add boundary node
        boundary_idx = n_nodes
        for i in range(n_nodes):
            t_i, s_i = nodes[i]
            # Distance to nearest boundary
            if hasattr(self.code, "d"):
                d = self.code.d
            elif hasattr(self.code, "distance"):
                _d = self.code.distance
                d = _d() if callable(_d) else _d
            else:
                d = 3
            boundary_weight = float(min(s_i + 1, d))
            edges.append((i, boundary_idx))
            weights.append(boundary_weight)

        return {
            "nodes": nodes,
            "edges": edges,
            "weights": weights,
            "boundary_idx": boundary_idx,
        }

    def decode(self, syndrome_diffs: np.ndarray) -> np.ndarray:
        """Decode using space-time matching.

        Parameters
        ----------
        syndrome_diffs : np.ndarray
            Syndrome differences, shape (n_rounds, n_stabilizers).

        Returns
        -------
        np.ndarray
            Correction vector in symplectic form (length 2*n).
        """
        n = self.code.n
        correction = np.zeros(2 * n, dtype=np.int8)

        graph = self.build_graph(syndrome_diffs)
        nodes = graph["nodes"]
        if not nodes:
            return correction

        edges = graph["edges"]
        weights = graph["weights"]
        boundary_idx = graph.get("boundary_idx", len(nodes))

        # Greedy matching on the space-time graph
        defect_indices = list(range(len(nodes)))
        matched = set()
        pairs: List[Tuple[int, int]] = []

        # Sort edges by weight
        sorted_edges = sorted(
            range(len(edges)),
            key=lambda i: weights[i],
        )

        for edge_idx in sorted_edges:
            a, b = edges[edge_idx]
            if a in matched or b in matched:
                continue
            if a == boundary_idx or b == boundary_idx:
                # Boundary match
                real_node = a if b == boundary_idx else b
                if real_node not in matched:
                    matched.add(real_node)
                    pairs.append((real_node, boundary_idx))
            else:
                matched.add(a)
                matched.add(b)
                pairs.append((a, b))

        # Convert matched pairs to corrections
        for a, b in pairs:
            if b == boundary_idx:
                # Single defect -> boundary correction
                t, s = nodes[a]
                self._apply_spatial_correction(correction, s)
            else:
                # Matched pair
                t_a, s_a = nodes[a]
                t_b, s_b = nodes[b]
                if t_a == t_b:
                    # Same time: spatial correction
                    self._apply_spatial_correction_between(correction, s_a, s_b)
                # else: temporal match = measurement error, no data correction

        # Match any remaining unmatched defects to boundary
        for i in defect_indices:
            if i not in matched:
                t, s = nodes[i]
                self._apply_spatial_correction(correction, s)

        return correction

    def _apply_spatial_correction(
        self, correction: np.ndarray, stab_idx: int
    ) -> None:
        """Apply a boundary-directed correction for a single defect."""
        n = self.code.n
        # Find a qubit in this stabilizer's support and apply Z error
        if hasattr(self.code, "Hz") and self.code.Hz.shape[0] > 0:
            if stab_idx < self.code.Hz.shape[0]:
                support = np.where(self.code.Hz[stab_idx] == 1)[0]
                if len(support) > 0:
                    q = support[0]
                    correction[q] = (correction[q] + 1) % 2
                    return

        if hasattr(self.code, "Hx") and self.code.Hx.shape[0] > 0:
            # Adjust index for X-checks
            adjusted = stab_idx - (self.code.Hz.shape[0] if hasattr(self.code, "Hz") else 0)
            if 0 <= adjusted < self.code.Hx.shape[0]:
                support = np.where(self.code.Hx[adjusted] == 1)[0]
                if len(support) > 0:
                    q = support[0]
                    correction[n + q] = (correction[n + q] + 1) % 2

    def _apply_spatial_correction_between(
        self,
        correction: np.ndarray,
        stab_a: int,
        stab_b: int,
    ) -> None:
        """Apply correction between two spatially separated defects."""
        # Simple: apply individual corrections for each
        self._apply_spatial_correction(correction, stab_a)
        self._apply_spatial_correction(correction, stab_b)


# ------------------------------------------------------------------ #
# Correlated Noise Model
# ------------------------------------------------------------------ #

@dataclass
class CorrelatedNoiseModel:
    """Noise model with spatial and temporal correlations.

    Generates errors where neighboring qubits and consecutive time
    steps have correlated error patterns.  This models realistic
    noise sources such as crosstalk, cosmic rays, or drifting
    calibration parameters.

    Parameters
    ----------
    base_rate : float
        Base per-qubit error probability.
    spatial_correlation : float
        Probability that an error on qubit i also causes an error on
        adjacent qubits.  Range [0, 1].
    temporal_correlation : float
        Probability that an error at time t persists at time t+1.
        Range [0, 1].
    """

    base_rate: float
    spatial_correlation: float = 0.0
    temporal_correlation: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.base_rate <= 1.0:
            raise ValueError(f"base_rate must be in [0, 1], got {self.base_rate}")
        if not 0.0 <= self.spatial_correlation <= 1.0:
            raise ValueError(
                f"spatial_correlation must be in [0, 1], got {self.spatial_correlation}"
            )
        if not 0.0 <= self.temporal_correlation <= 1.0:
            raise ValueError(
                f"temporal_correlation must be in [0, 1], got {self.temporal_correlation}"
            )

    def sample_correlated_errors(
        self,
        n_qubits: int,
        n_rounds: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample errors with spatial and temporal correlations.

        Parameters
        ----------
        n_qubits : int
            Number of data qubits.
        n_rounds : int
            Number of syndrome measurement rounds.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        np.ndarray
            Error array of shape (n_rounds, 2*n_qubits) in symplectic
            form.  Each row is the cumulative error at that round.
        """
        if rng is None:
            rng = np.random.default_rng()

        errors = np.zeros((n_rounds, 2 * n_qubits), dtype=np.int8)

        prev_error = np.zeros(2 * n_qubits, dtype=np.int8)

        for t in range(n_rounds):
            # Base independent errors (depolarizing)
            new_error = np.zeros(2 * n_qubits, dtype=np.int8)
            p_each = self.base_rate / 3.0

            for q in range(n_qubits):
                r = rng.random()
                if r < p_each:
                    new_error[q] = 1  # X
                elif r < 2 * p_each:
                    new_error[n_qubits + q] = 1  # Z
                elif r < 3 * p_each:
                    new_error[q] = 1  # Y
                    new_error[n_qubits + q] = 1

            # Spatial correlations: spread errors to neighbors
            if self.spatial_correlation > 0:
                spatial_spread = np.zeros_like(new_error)
                for q in range(n_qubits):
                    if new_error[q] or new_error[n_qubits + q]:
                        # Spread to adjacent qubits (1D chain model)
                        for dq in [-1, 1]:
                            neighbor = q + dq
                            if 0 <= neighbor < n_qubits:
                                if rng.random() < self.spatial_correlation:
                                    # Copy error type
                                    spatial_spread[neighbor] = (
                                        spatial_spread[neighbor] + new_error[q]
                                    ) % 2
                                    spatial_spread[n_qubits + neighbor] = (
                                        spatial_spread[n_qubits + neighbor]
                                        + new_error[n_qubits + q]
                                    ) % 2
                new_error = (new_error + spatial_spread) % 2

            # Temporal correlations: errors persist from previous round
            if self.temporal_correlation > 0 and t > 0:
                for q in range(2 * n_qubits):
                    if prev_error[q] and rng.random() < self.temporal_correlation:
                        new_error[q] = (new_error[q] + 1) % 2

            prev_error = new_error.copy()
            errors[t] = new_error

        return errors

    def sample_syndrome_history(
        self,
        code: Any,
        n_rounds: int,
        rng: Optional[np.random.Generator] = None,
        measurement_error_rate: float = 0.0,
    ) -> Tuple[np.ndarray, SyndromeHistory]:
        """Sample correlated errors and corresponding syndrome history.

        Parameters
        ----------
        code : object
            Quantum code with syndrome() and n attributes.
        n_rounds : int
            Number of rounds.
        rng : np.random.Generator or None
            Random number generator.
        measurement_error_rate : float
            Probability of measurement bit flip per syndrome bit per round.

        Returns
        -------
        total_error : np.ndarray
            Accumulated error after all rounds (length 2*n).
        syndrome_history : SyndromeHistory
            History of (possibly noisy) syndromes.
        """
        if rng is None:
            rng = np.random.default_rng()

        errors = self.sample_correlated_errors(code.n, n_rounds, rng)
        syndromes = np.zeros((n_rounds, len(code.syndrome(np.zeros(2 * code.n, dtype=np.int8)))),
                             dtype=np.int8)

        total_error = np.zeros(2 * code.n, dtype=np.int8)

        for t in range(n_rounds):
            total_error = (total_error + errors[t]) % 2
            syn = code.syndrome(total_error)

            # Add measurement noise
            if measurement_error_rate > 0:
                noise = (rng.random(len(syn)) < measurement_error_rate).astype(np.int8)
                syn = (syn + noise) % 2

            syndromes[t] = syn

        return total_error, SyndromeHistory(syndromes=syndromes)


# ------------------------------------------------------------------ #
# Decoding Benchmark
# ------------------------------------------------------------------ #

@dataclass
class BenchmarkResult:
    """Result of a decoder comparison benchmark.

    Attributes
    ----------
    independent_logical_rate : float
        Logical error rate with independent (single-round) decoding.
    correlated_logical_rate : float
        Logical error rate with correlated (sliding window) decoding.
    improvement_factor : float
        Ratio of independent to correlated error rate (>1 means
        correlated is better).
    window_size : int
        Window size used for correlated decoding.
    n_shots : int
        Number of Monte Carlo shots.
    """

    independent_logical_rate: float
    correlated_logical_rate: float
    improvement_factor: float
    window_size: int
    n_shots: int


@dataclass
class DecodingBenchmark:
    """Benchmark correlated vs independent decoding.

    Compares the performance of a sliding window decoder (which uses
    temporal correlations) against a naive single-round decoder on
    the same syndrome data.

    Parameters
    ----------
    code : object
        Quantum error correcting code.
    noise_model : CorrelatedNoiseModel
        Correlated noise model for error generation.
    n_rounds : int
        Number of syndrome rounds per shot.
    window_size : int
        Window size for the sliding window decoder.
    """

    code: Any
    noise_model: CorrelatedNoiseModel
    n_rounds: int = 5
    window_size: int = 3

    def compare_decoders(
        self,
        n_shots: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> BenchmarkResult:
        """Compare sliding window vs single-round decoding.

        For each shot:
        1. Generate correlated errors and syndrome history
        2. Decode with independent (single-round) decoder
        3. Decode with sliding window decoder
        4. Compare logical error rates

        Parameters
        ----------
        n_shots : int
            Number of Monte Carlo shots.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        BenchmarkResult
        """
        if rng is None:
            rng = np.random.default_rng()

        code = self.code
        n = code.n
        window_decoder = SlidingWindowDecoder(
            code=code,
            window_size=self.window_size,
            commit_size=max(1, self.window_size // 2),
        )

        independent_failures = 0
        correlated_failures = 0

        for _ in range(n_shots):
            # Generate correlated errors
            total_error, syn_history = self.noise_model.sample_syndrome_history(
                code=code,
                n_rounds=self.n_rounds,
                rng=rng,
            )

            # Independent decoding: use only the last syndrome
            last_syn = syn_history.get_round(syn_history.n_rounds - 1)
            ind_correction = self._independent_decode(last_syn)

            # Check independent correction
            if not self._check_correction(total_error, ind_correction):
                independent_failures += 1

            # Correlated decoding: use sliding window
            corrections = window_decoder.decode_stream(syn_history)
            corr_correction = np.zeros(2 * n, dtype=np.int8)
            for c in corrections:
                corr_correction = (corr_correction + c) % 2

            # Check correlated correction
            if not self._check_correction(total_error, corr_correction):
                correlated_failures += 1

        ind_rate = independent_failures / n_shots
        corr_rate = correlated_failures / n_shots

        if corr_rate > 0:
            improvement = ind_rate / corr_rate
        elif ind_rate > 0:
            improvement = float("inf")
        else:
            improvement = 1.0

        return BenchmarkResult(
            independent_logical_rate=ind_rate,
            correlated_logical_rate=corr_rate,
            improvement_factor=improvement,
            window_size=self.window_size,
            n_shots=n_shots,
        )

    def _independent_decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Simple independent (single-round) greedy decoder."""
        n = self.code.n
        correction = np.zeros(2 * n, dtype=np.int8)
        remaining = syndrome.copy()

        for _ in range(n):
            defects = np.where(remaining == 1)[0]
            if len(defects) == 0:
                break

            best_err = None
            best_cleared = 0

            for q in range(n):
                for pauli in ["X", "Z"]:
                    err = np.zeros(2 * n, dtype=np.int8)
                    if pauli == "X":
                        err[q] = 1
                    else:
                        err[n + q] = 1

                    syn = self.code.syndrome(err)
                    syn_trunc = syn[:len(remaining)]
                    cleared = np.sum((remaining + syn_trunc) % 2 == 0)
                    if cleared > best_cleared:
                        best_cleared = cleared
                        best_err = err.copy()

            if best_err is not None:
                correction = (correction + best_err) % 2
                syn = self.code.syndrome(best_err)
                remaining = (remaining + syn[:len(remaining)]) % 2

        return correction

    def _check_correction(
        self, error: np.ndarray, correction: np.ndarray
    ) -> bool:
        """Check if error + correction is in the stabilizer group."""
        code = self.code
        residual = (error + correction) % 2
        syn = code.syndrome(residual)
        if np.any(syn):
            return False

        # Check logical operators if available
        if hasattr(code, "logical_x") and hasattr(code, "logical_z"):
            n = code.n
            res_x = residual[:n]
            res_z = residual[n:]

            for lx_op in (
                code.logical_x() if callable(code.logical_x) else [code.logical_x()]
            ):
                lx_x = lx_op[:n]
                lx_z = lx_op[n:]
                ip = (np.dot(res_x, lx_z) + np.dot(res_z, lx_x)) % 2
                if ip:
                    return False

            for lz_op in (
                code.logical_z() if callable(code.logical_z) else [code.logical_z()]
            ):
                lz_x = lz_op[:n]
                lz_z = lz_op[n:]
                ip = (np.dot(res_x, lz_z) + np.dot(res_z, lz_x)) % 2
                if ip:
                    return False

        return True
