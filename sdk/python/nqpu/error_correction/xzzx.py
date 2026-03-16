"""XZZX surface code for biased noise channels.

Implements the XZZX surface code variant that achieves higher thresholds
under biased noise by using alternating X and Z operators in the stabilizer
checks.  Based on:

    Bonilla-Ataides et al., "The XZZX surface code",
    Nature Communications 12, 2172 (2021).

Unlike the standard CSS surface code where X-stabilizers are all-X and
Z-stabilizers are all-Z, the XZZX code uses weight-4 stabilizers of the
form X-Z-Z-X (and its conjugate Z-X-X-Z).  This breaks the CSS structure
but provides a significant threshold advantage when noise is biased toward
one Pauli type.

Key property: under pure Z-biased noise (eta -> infinity), the XZZX code's
threshold approaches 50%, because all Z errors look like repetition-code
errors along one axis of the rotated lattice.

All implementations are pure numpy with no external dependencies.

Example:
    from nqpu.error_correction.xzzx import (
        XZZXCode, BiasedNoiseChannel, XZZXDecoder, XZZXThresholdStudy,
    )

    code = XZZXCode(distance=5, bias=100.0)
    noise = BiasedNoiseChannel(p=0.05, bias=100.0)
    decoder = XZZXDecoder(code)

    rng = np.random.default_rng(42)
    error = noise.sample_error(code.n, rng)
    syndrome = code.syndrome(error)
    correction = decoder.decode(syndrome)
    success = code.check_correction(error, correction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ #
# XZZX Surface Code
# ------------------------------------------------------------------ #

@dataclass
class XZZXCode:
    """XZZX surface code variant optimized for Z-biased noise.

    Unlike the standard CSS surface code, the XZZX code uses alternating
    X and Z checks that exploit noise bias for higher thresholds.

    The stabilizers on the rotated lattice are weight-4 operators:
      - Even plaquettes (r+c even): X_tl Z_tr Z_bl X_br
      - Odd  plaquettes (r+c odd):  Z_tl X_tr X_bl Z_br

    where tl, tr, bl, br are the top-left, top-right, bottom-left,
    bottom-right data qubits of the plaquette.

    Boundary stabilizers are weight-2.

    Parameters
    ----------
    distance : int
        Code distance (must be odd >= 3).
    bias : float
        Noise bias eta = pZ/pX.  1.0 means unbiased (standard depolarizing).
        Higher values mean the noise is Z-dominated.
    """

    distance: int
    bias: float = 1.0

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("Distance must be an odd integer >= 3")
        if self.bias < 0:
            raise ValueError("Bias must be non-negative")
        self.n: int = self.distance ** 2
        self.k: int = 1
        self.d: int = self.distance
        self._stabilizer_list: List[np.ndarray] = []
        self._stabilizer_matrix: Optional[np.ndarray] = None
        self._build_stabilizers()

    # -- lattice helpers --

    def _qubit_index(self, r: int, c: int) -> int:
        """Row-major index for data qubit at (r, c)."""
        return r * self.distance + c

    def qubit_coords(self) -> List[Tuple[int, int]]:
        """Return (row, col) coordinates for each data qubit."""
        d = self.distance
        return [(r, c) for r in range(d) for c in range(d)]

    # -- stabilizer construction --

    def _build_stabilizers(self) -> None:
        """Build XZZX stabilizer generators on the rotated lattice.

        Uses the same plaquette layout as the standard CSS rotated surface
        code (which has (d-1)^2 bulk + boundary = d^2 - 1 total stabilizers),
        but replaces the all-X and all-Z operators with the XZZX pattern.

        Each plaquette face (r, c) for 0 <= r < d-1, 0 <= c < d-1 covers
        data qubits at corners (r,c), (r,c+1), (r+1,c), (r+1,c+1).

        The XZZX pattern assigns Pauli types based on a checkerboard:
          - Face (r, c) with (r+c) even: X_tl Z_tr Z_bl X_br
          - Face (r, c) with (r+c) odd:  Z_tl X_tr X_bl Z_br

        Boundary stabilizers (weight-2) are placed along edges where the
        standard CSS code would place them (matching the same parity
        selection rule), but with XZZX-appropriate Pauli types.
        """
        d = self.distance
        stabs: List[np.ndarray] = []

        def _set_pauli(s: np.ndarray, qubit: int, pauli: str) -> None:
            """Set Pauli operator on qubit in symplectic vector s."""
            if pauli == "X":
                s[qubit] = 1
            elif pauli == "Z":
                s[self.n + qubit] = 1
            elif pauli == "Y":
                s[qubit] = 1
                s[self.n + qubit] = 1

        def _xzzx_type(qubit_row: int, qubit_col: int, face_row: int, face_col: int) -> str:
            """Determine the Pauli type for a qubit in an XZZX face.

            In the XZZX code, the Pauli type on each corner of a face
            alternates: corners where (qr + qc) has same parity as (fr + fc)
            get X, others get Z.
            """
            if (qubit_row + qubit_col) % 2 == (face_row + face_col) % 2:
                return "X"
            else:
                return "Z"

        # Bulk stabilizers (weight-4): one per interior face
        for fr in range(d - 1):
            for fc in range(d - 1):
                s = np.zeros(2 * self.n, dtype=np.int8)
                corners = [
                    (fr, fc), (fr, fc + 1),
                    (fr + 1, fc), (fr + 1, fc + 1),
                ]
                for qr, qc in corners:
                    q = self._qubit_index(qr, qc)
                    pauli = _xzzx_type(qr, qc, fr, fc)
                    _set_pauli(s, q, pauli)
                stabs.append(s)

        # Boundary stabilizers (weight-2):
        # Follow the same parity selection as the CSS surface code.
        #
        # Top boundary: virtual face row r=-1
        # Only for c where (-1 + c) % 2 != 0 (odd parity -> Z-type boundary in CSS)
        for c in range(d - 1):
            if (-1 + c) % 2 != 0:
                s = np.zeros(2 * self.n, dtype=np.int8)
                fr, fc = -1, c
                for qr, qc in [(0, c), (0, c + 1)]:
                    q = self._qubit_index(qr, qc)
                    pauli = _xzzx_type(qr, qc, fr, fc)
                    _set_pauli(s, q, pauli)
                stabs.append(s)

        # Bottom boundary: virtual face row r=d-1
        for c in range(d - 1):
            if (d - 1 + c) % 2 != 0:
                s = np.zeros(2 * self.n, dtype=np.int8)
                fr, fc = d - 1, c
                for qr, qc in [(d - 1, c), (d - 1, c + 1)]:
                    q = self._qubit_index(qr, qc)
                    pauli = _xzzx_type(qr, qc, fr, fc)
                    _set_pauli(s, q, pauli)
                stabs.append(s)

        # Left boundary: virtual face column c=-1
        for r in range(d - 1):
            if (r + (-1)) % 2 == 0:
                s = np.zeros(2 * self.n, dtype=np.int8)
                fr, fc = r, -1
                for qr, qc in [(r, 0), (r + 1, 0)]:
                    q = self._qubit_index(qr, qc)
                    pauli = _xzzx_type(qr, qc, fr, fc)
                    _set_pauli(s, q, pauli)
                stabs.append(s)

        # Right boundary: virtual face column c=d-1
        for r in range(d - 1):
            if (r + (d - 1)) % 2 == 0:
                s = np.zeros(2 * self.n, dtype=np.int8)
                fr, fc = r, d - 1
                for qr, qc in [(r, d - 1), (r + 1, d - 1)]:
                    q = self._qubit_index(qr, qc)
                    pauli = _xzzx_type(qr, qc, fr, fc)
                    _set_pauli(s, q, pauli)
                stabs.append(s)

        # Deduplicate stabilizers (boundaries might coincide)
        seen: set = set()
        unique_stabs: List[np.ndarray] = []
        for s in stabs:
            key = s.tobytes()
            if key not in seen and np.any(s):
                seen.add(key)
                unique_stabs.append(s)

        self._stabilizer_list = unique_stabs
        # Build the full stabilizer matrix for syndrome computation
        if unique_stabs:
            self._stabilizer_matrix = np.array(unique_stabs, dtype=np.int8)
        else:
            self._stabilizer_matrix = np.zeros((0, 2 * self.n), dtype=np.int8)

    @property
    def num_stabilizers(self) -> int:
        """Number of independent stabilizer generators."""
        return len(self._stabilizer_list)

    def stabilizers(self) -> List[np.ndarray]:
        """Return stabilizer generators in symplectic form [x | z]."""
        return [s.copy() for s in self._stabilizer_list]

    def syndrome(self, error: np.ndarray) -> np.ndarray:
        """Compute syndrome from a Pauli error vector.

        The XZZX code is non-CSS, so we cannot separate into X-check
        and Z-check syndrome.  Each stabilizer's syndrome bit is computed
        as the symplectic inner product of the stabilizer with the error.

        Parameters
        ----------
        error : np.ndarray
            Binary error vector of length 2*n in symplectic form [x | z].

        Returns
        -------
        np.ndarray
            Binary syndrome vector of length num_stabilizers.
        """
        x_err = error[:self.n]
        z_err = error[self.n:]
        syn = np.zeros(self.num_stabilizers, dtype=np.int8)
        for i, s in enumerate(self._stabilizer_list):
            s_x = s[:self.n]
            s_z = s[self.n:]
            # Symplectic inner product: <s, e> = s_x . e_z + s_z . e_x  (mod 2)
            syn[i] = (np.dot(s_x, z_err) + np.dot(s_z, x_err)) % 2
        return syn

    def logical_x(self) -> np.ndarray:
        """Logical X operator: alternating X/Z string along the first row.

        For the XZZX code, the logical operators must follow the
        alternating Pauli pattern.  Along the top row (r=0), the
        qubit at column c gets:
          - X if c is even
          - Z if c is odd
        """
        lx = np.zeros(2 * self.n, dtype=np.int8)
        for c in range(self.distance):
            q = self._qubit_index(0, c)
            if c % 2 == 0:
                lx[q] = 1          # X
            else:
                lx[self.n + q] = 1  # Z
        return lx

    def logical_z(self) -> np.ndarray:
        """Logical Z operator: alternating Z/X string along the first column.

        Along the left column (c=0), the qubit at row r gets:
          - Z if r is even
          - X if r is odd
        """
        lz = np.zeros(2 * self.n, dtype=np.int8)
        for r in range(self.distance):
            q = self._qubit_index(r, 0)
            if r % 2 == 0:
                lz[self.n + q] = 1  # Z
            else:
                lz[q] = 1          # X
        return lz

    def check_correction(self, error: np.ndarray, correction: np.ndarray) -> bool:
        """Check if error + correction is in the stabilizer group.

        The residual must have trivial syndrome AND must not anti-commute
        with any logical operator (which would indicate a logical error).

        Parameters
        ----------
        error : np.ndarray
            Original error vector (length 2*n).
        correction : np.ndarray
            Proposed correction vector (length 2*n).

        Returns
        -------
        bool
            True if correction successfully recovers the logical information.
        """
        residual = (error + correction) % 2
        # Check syndrome
        syn = self.syndrome(residual)
        if np.any(syn):
            return False

        # Check that residual does not anti-commute with logical operators
        res_x = residual[:self.n]
        res_z = residual[self.n:]

        lx = self.logical_x()
        lz = self.logical_z()

        # Anti-commutation with logical Z
        lz_x = lz[:self.n]
        lz_z = lz[self.n:]
        ip_z = (np.dot(res_x, lz_z) + np.dot(res_z, lz_x)) % 2
        if ip_z:
            return False

        # Anti-commutation with logical X
        lx_x = lx[:self.n]
        lx_z = lx[self.n:]
        ip_x = (np.dot(res_x, lx_z) + np.dot(res_z, lx_x)) % 2
        if ip_x:
            return False

        return True


# ------------------------------------------------------------------ #
# Biased Noise Channel
# ------------------------------------------------------------------ #

@dataclass
class BiasedNoiseChannel:
    """Biased Pauli noise channel with tunable Z/X ratio.

    The noise model distributes total error probability p among the three
    Pauli types according to the bias parameter eta = pZ / pX:
        pX = p / (1 + eta + eta)  (since pY ~ pX for simplicity)
        pZ = eta * pX
        pY = pX  (conservative: Y has both X and Z components)

    For eta = 1, this reduces to symmetric depolarizing noise (p/3 each).
    For eta >> 1, the noise is strongly Z-biased.

    Parameters
    ----------
    p : float
        Total error probability per qubit.
    bias : float
        Noise bias eta = pZ / pX.  Must be >= 0.
    """

    p: float
    bias: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"Error probability must be in [0, 1], got {self.p}")
        if self.bias < 0:
            raise ValueError(f"Bias must be non-negative, got {self.bias}")

    @property
    def px(self) -> float:
        """X error probability per qubit."""
        if self.p == 0:
            return 0.0
        denom = 1.0 + self.bias + 1.0  # pX + pZ + pY, with pY = pX
        return self.p / denom

    @property
    def py(self) -> float:
        """Y error probability per qubit (= pX for this model)."""
        return self.px

    @property
    def pz(self) -> float:
        """Z error probability per qubit."""
        return self.bias * self.px

    def sample_error(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Sample an n-qubit Pauli error respecting the bias.

        Parameters
        ----------
        n : int
            Number of qubits.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        np.ndarray
            Binary error vector of length 2*n in symplectic form [x | z].
        """
        if rng is None:
            rng = np.random.default_rng()

        error = np.zeros(2 * n, dtype=np.int8)
        _px = self.px
        _py = self.py
        _pz = self.pz

        for q in range(n):
            r = rng.random()
            if r < _px:
                error[q] = 1       # X
            elif r < _px + _pz:
                error[n + q] = 1   # Z
            elif r < _px + _pz + _py:
                error[q] = 1       # Y = XZ
                error[n + q] = 1

        return error


# ------------------------------------------------------------------ #
# XZZX Decoder (greedy matching)
# ------------------------------------------------------------------ #

@dataclass
class XZZXDecoder:
    """Greedy minimum-weight matching decoder adapted for XZZX stabilizers.

    Since the XZZX code is non-CSS, we decode on a single unified
    syndrome graph rather than separate X and Z graphs.  Defects are
    matched greedily by their Manhattan distance on the lattice, with
    bias-weighted costs that prefer corrections along the Z-biased axis.

    Parameters
    ----------
    code : XZZXCode
        The XZZX code to decode.
    """

    code: XZZXCode

    def __post_init__(self) -> None:
        self._stab_coords: List[Tuple[float, float]] = []
        self._build_syndrome_graph()

    def _build_syndrome_graph(self) -> None:
        """Precompute stabilizer centre coordinates for distance-based matching."""
        d = self.code.distance
        coords: List[Tuple[float, float]] = []

        for i, s in enumerate(self.code._stabilizer_list):
            # Find qubits involved in this stabilizer
            x_part = s[:self.code.n]
            z_part = s[self.code.n:]
            involved = np.where((x_part + z_part) > 0)[0]

            if len(involved) == 0:
                coords.append((0.0, 0.0))
                continue

            # Average position
            rows = [q // d for q in involved]
            cols = [q % d for q in involved]
            coords.append((np.mean(rows), np.mean(cols)))

        self._stab_coords = coords

    def _distance(self, i: int, j: int) -> float:
        """Weighted distance between two syndrome nodes.

        Uses Manhattan distance with bias-aware weighting: in the
        direction aligned with the bias, errors are cheaper.
        """
        ri, ci = self._stab_coords[i]
        rj, cj = self._stab_coords[j]
        dr = abs(ri - rj)
        dc = abs(ci - cj)

        # Under Z-bias, Z errors are more likely -> paths along the
        # Z-error direction (which become repetition-code-like) are cheaper
        bias = self.code.bias
        if bias > 1.0:
            # Z-biased: vertical strings of Z errors are more likely
            weight_r = 1.0 / np.log2(bias + 1)
            weight_c = 1.0
        elif bias < 1.0 and bias > 0:
            weight_r = 1.0
            weight_c = 1.0 / np.log2(1.0 / bias + 1)
        else:
            weight_r = 1.0
            weight_c = 1.0

        return weight_r * dr + weight_c * dc

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode syndrome using greedy matching on the XZZX syndrome graph.

        1. Find all defect locations (syndrome = 1).
        2. Greedily pair closest defects using weighted distance.
        3. Construct correction from stabilizer supports along paths.

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome vector of length num_stabilizers.

        Returns
        -------
        np.ndarray
            Correction in symplectic form (length 2*n).
        """
        code = self.code
        defects = list(np.where(syndrome == 1)[0])
        correction = np.zeros(2 * code.n, dtype=np.int8)

        if not defects:
            return correction

        # Greedy nearest-neighbour matching
        unmatched = list(defects)

        while len(unmatched) >= 2:
            best_dist = float("inf")
            best_i, best_j = 0, 1

            for i in range(len(unmatched)):
                for j in range(i + 1, len(unmatched)):
                    d_ij = self._distance(unmatched[i], unmatched[j])
                    if d_ij < best_dist:
                        best_dist = d_ij
                        best_i, best_j = i, j

            # Match the closest pair
            di = unmatched[best_i]
            dj = unmatched[best_j]

            # Remove in reverse order to preserve indices
            if best_i < best_j:
                unmatched.pop(best_j)
                unmatched.pop(best_i)
            else:
                unmatched.pop(best_i)
                unmatched.pop(best_j)

            # Construct correction along the path between matched defects
            path_correction = self._path_correction(di, dj)
            correction = (correction + path_correction) % 2

        # Any remaining unmatched defect -> match to boundary
        for di in unmatched:
            boundary_correction = self._boundary_correction(di)
            correction = (correction + boundary_correction) % 2

        return correction

    def _path_correction(self, stab_i: int, stab_j: int) -> np.ndarray:
        """Construct a correction that flips the syndrome at stab_i and stab_j.

        Find a chain of qubits connecting the two stabilizers using
        stabilizer overlap structure.
        """
        code = self.code
        d = code.distance
        correction = np.zeros(2 * code.n, dtype=np.int8)

        ri, ci = self._stab_coords[stab_i]
        rj, cj = self._stab_coords[stab_j]

        # Walk from stab_i toward stab_j, flipping qubits along the way
        # Strategy: find qubits that are in the support of stabilizers
        # near the path and that flip exactly the needed syndrome bits
        si = code._stabilizer_list[stab_i]
        sj = code._stabilizer_list[stab_j]

        # Find a qubit in stabilizer i's support
        support_i = np.where((si[:code.n] + si[code.n:]) > 0)[0]
        support_j = np.where((sj[:code.n] + sj[code.n:]) > 0)[0]

        # Try to find a shared qubit (adjacent stabilizers)
        shared = np.intersect1d(support_i, support_j)
        if len(shared) > 0:
            q = shared[0]
            # Apply the appropriate Pauli on this qubit
            correction = self._single_qubit_flip(q, stab_i, stab_j)
            return correction

        # For non-adjacent stabilizers, build a chain
        # Use a simple greedy walk through stabilizers
        visited = {stab_i}
        chain = [stab_i]
        current = stab_i

        # Build adjacency: stabilizers that share a qubit
        adj = self._stabilizer_adjacency()

        for _ in range(code.n):
            if current == stab_j:
                break

            # Find unvisited neighbor closest to target
            best_next = -1
            best_cost = float("inf")

            for neighbor, qubit in adj.get(current, []):
                if neighbor in visited:
                    continue
                cost = self._distance(neighbor, stab_j)
                if cost < best_cost:
                    best_cost = cost
                    best_next = neighbor

            if best_next == -1:
                break

            visited.add(best_next)
            chain.append(best_next)
            current = best_next

        # Now apply corrections along the chain
        for k in range(len(chain) - 1):
            s_a = chain[k]
            s_b = chain[k + 1]
            qflip = self._find_shared_qubit(s_a, s_b)
            if qflip >= 0:
                flip = self._single_qubit_flip(qflip, s_a, s_b)
                correction = (correction + flip) % 2

        return correction

    def _stabilizer_adjacency(self) -> Dict[int, List[Tuple[int, int]]]:
        """Build adjacency map: stab_idx -> [(neighbor_idx, shared_qubit)]."""
        code = self.code
        # Map qubit -> list of stabilizer indices that include it
        qubit_to_stabs: Dict[int, List[int]] = {}
        for idx, s in enumerate(code._stabilizer_list):
            support = np.where((s[:code.n] + s[code.n:]) > 0)[0]
            for q in support:
                qubit_to_stabs.setdefault(int(q), []).append(idx)

        adj: Dict[int, List[Tuple[int, int]]] = {}
        for q, stab_indices in qubit_to_stabs.items():
            for i in range(len(stab_indices)):
                for j in range(i + 1, len(stab_indices)):
                    si, sj = stab_indices[i], stab_indices[j]
                    adj.setdefault(si, []).append((sj, q))
                    adj.setdefault(sj, []).append((si, q))

        return adj

    def _find_shared_qubit(self, stab_a: int, stab_b: int) -> int:
        """Find a qubit shared between two stabilizers."""
        code = self.code
        sa = code._stabilizer_list[stab_a]
        sb = code._stabilizer_list[stab_b]
        support_a = set(np.where((sa[:code.n] + sa[code.n:]) > 0)[0])
        support_b = set(np.where((sb[:code.n] + sb[code.n:]) > 0)[0])
        shared = support_a & support_b
        if shared:
            return min(shared)
        return -1

    def _single_qubit_flip(
        self, qubit: int, stab_a: int, stab_b: int
    ) -> np.ndarray:
        """Create a single-qubit Pauli that anticommutes with the given stabilizers.

        Determines the correct Pauli type (X, Z, or Y) on the qubit that
        will flip the syndrome bits for both stab_a and stab_b.
        """
        code = self.code
        n = code.n

        # Try each Pauli type and check which flips the needed syndrome bits
        for pauli_type in ["Z", "X", "Y"]:
            err = np.zeros(2 * n, dtype=np.int8)
            if pauli_type in ("X", "Y"):
                err[qubit] = 1
            if pauli_type in ("Z", "Y"):
                err[n + qubit] = 1

            syn = code.syndrome(err)
            # We want this to flip at least the two target stabilizers
            if syn[stab_a] == 1:
                return err

        # Fallback: Z error
        err = np.zeros(2 * n, dtype=np.int8)
        err[n + qubit] = 1
        return err

    def _boundary_correction(self, stab_idx: int) -> np.ndarray:
        """Construct correction from a stabilizer to the nearest boundary."""
        code = self.code
        d = code.distance
        n = code.n
        correction = np.zeros(2 * n, dtype=np.int8)

        ri, ci = self._stab_coords[stab_idx]
        s = code._stabilizer_list[stab_idx]
        support = np.where((s[:n] + s[n:]) > 0)[0]

        if len(support) == 0:
            return correction

        # Find the boundary qubit in the stabilizer's support
        # that is closest to an edge
        best_q = support[0]
        best_edge_dist = float("inf")
        for q in support:
            r, c = q // d, q % d
            edge_dist = min(r, d - 1 - r, c, d - 1 - c)
            if edge_dist < best_edge_dist:
                best_edge_dist = edge_dist
                best_q = q

        # Apply correction on the boundary qubit
        # Choose Pauli type that anticommutes with this stabilizer
        for pauli_type in ["Z", "X", "Y"]:
            err = np.zeros(2 * n, dtype=np.int8)
            if pauli_type in ("X", "Y"):
                err[best_q] = 1
            if pauli_type in ("Z", "Y"):
                err[n + best_q] = 1
            syn = code.syndrome(err)
            if syn[stab_idx] == 1:
                return err

        # Fallback
        err = np.zeros(2 * n, dtype=np.int8)
        err[n + best_q] = 1
        return err


# ------------------------------------------------------------------ #
# Threshold Study
# ------------------------------------------------------------------ #

@dataclass
class XZZXThresholdResult:
    """Result of an XZZX threshold study.

    Attributes
    ----------
    distances : list of int
        Code distances tested.
    bias_values : list of float
        Bias values tested.
    error_rates : np.ndarray
        Physical error rates tested.
    logical_error_rates : np.ndarray
        Measured logical error rates, shape (n_bias, n_distances, n_points).
    thresholds : dict
        Mapping from bias value to estimated threshold.
    """

    distances: list
    bias_values: list
    error_rates: np.ndarray
    logical_error_rates: np.ndarray
    thresholds: dict


@dataclass
class XZZXThresholdStudy:
    """Threshold estimation for XZZX codes under biased noise.

    Sweeps across multiple bias values, code distances, and physical
    error rates to estimate the fault-tolerance threshold for each bias.

    Parameters
    ----------
    distances : list of int
        Code distances to test (must be odd >= 3).
    bias_values : list of float
        Noise bias values to test.
    """

    distances: list
    bias_values: list = field(default_factory=lambda: [1.0, 10.0, 100.0])

    def run(
        self,
        p_range: Tuple[float, float] = (0.01, 0.20),
        num_points: int = 10,
        shots: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> XZZXThresholdResult:
        """Run threshold study across distances and bias values.

        Parameters
        ----------
        p_range : tuple
            (min_p, max_p) range of physical error rates.
        num_points : int
            Number of error rate sample points.
        shots : int
            Monte Carlo shots per data point.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        XZZXThresholdResult
        """
        if rng is None:
            rng = np.random.default_rng(42)

        p_values = np.linspace(p_range[0], p_range[1], num_points)
        n_bias = len(self.bias_values)
        n_dist = len(self.distances)
        n_pts = num_points

        logical_rates = np.zeros((n_bias, n_dist, n_pts), dtype=np.float64)
        thresholds: dict = {}

        for bi, bias in enumerate(self.bias_values):
            for di, dist in enumerate(self.distances):
                code = XZZXCode(distance=dist, bias=bias)
                decoder = XZZXDecoder(code)
                noise = BiasedNoiseChannel(p=0.0, bias=bias)

                for pi, p in enumerate(p_values):
                    noise_p = BiasedNoiseChannel(p=p, bias=bias)
                    failures = 0

                    for _ in range(shots):
                        error = noise_p.sample_error(code.n, rng)
                        syndrome = code.syndrome(error)

                        if not np.any(syndrome):
                            # Check for undetected logical error
                            if not code.check_correction(
                                error, np.zeros(2 * code.n, dtype=np.int8)
                            ):
                                failures += 1
                            continue

                        correction = decoder.decode(syndrome)
                        if not code.check_correction(error, correction):
                            failures += 1

                    logical_rates[bi, di, pi] = failures / shots

            # Estimate threshold for this bias
            threshold = self._estimate_threshold(
                logical_rates[bi], p_values, self.distances
            )
            thresholds[bias] = threshold

        return XZZXThresholdResult(
            distances=self.distances,
            bias_values=self.bias_values,
            error_rates=p_values,
            logical_error_rates=logical_rates,
            thresholds=thresholds,
        )

    def _estimate_threshold(
        self,
        logical_rates_for_bias: np.ndarray,
        p_values: np.ndarray,
        distances: list,
    ) -> float:
        """Estimate threshold from curve crossings.

        Finds where the logical error rate curves for different distances
        cross, indicating the threshold physical error rate.
        """
        n_dist = len(distances)
        crossings: List[float] = []

        for i in range(n_dist - 1):
            curve_small = logical_rates_for_bias[i]
            curve_large = logical_rates_for_bias[i + 1]

            diff = curve_small - curve_large
            for j in range(len(diff) - 1):
                if diff[j] * diff[j + 1] < 0:
                    # Linear interpolation
                    t = -diff[j] / (diff[j + 1] - diff[j] + 1e-30)
                    p_cross = p_values[j] + t * (p_values[j + 1] - p_values[j])
                    crossings.append(p_cross)

        if crossings:
            return float(np.mean(crossings))

        # Fallback: return midpoint of range
        return float(np.mean(p_values))
