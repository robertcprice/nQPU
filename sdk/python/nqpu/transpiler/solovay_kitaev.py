"""Solovay-Kitaev algorithm: approximate arbitrary SU(2) with {H, T, S}.

Implements the Solovay-Kitaev algorithm for approximating an arbitrary
single-qubit unitary using only gates from the Clifford+T basis
{H, T, S, Tdg, Sdg}.  The algorithm achieves gate count
O(log^c(1/epsilon)) where c ~ 3.97, making it exponentially more
efficient than brute-force search.

Algorithm overview:

1. **Base case**: Search a pre-computed database of short gate sequences
   to find the closest approximation.
2. **Recursive step**: Given approximation U_{n-1} of U, compute the
   error delta = U @ U_{n-1}^dag, decompose delta into a group
   commutator [V, W] = V W V^dag W^dag, recursively approximate V
   and W, and refine: U_n = V_n W_n V_n^dag W_n^dag U_{n-1}.

All computations are pure numpy with no external dependencies.

Reference:
    C. M. Dawson, M. A. Nielsen.
    *The Solovay-Kitaev Algorithm*, Quantum Info. Comput. 6(1), 2006.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# Clifford+T basis gates
# ------------------------------------------------------------------

H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
S_GATE = np.array([[1, 0], [0, 1j]], dtype=complex)
T_DAG = T_GATE.conj().T
S_DAG = S_GATE.conj().T

_GATE_MAP: Dict[str, np.ndarray] = {
    "H": H_GATE,
    "T": T_GATE,
    "S": S_GATE,
    "Tdg": T_DAG,
    "Sdg": S_DAG,
}

_INV_MAP: Dict[str, str] = {
    "H": "H",
    "T": "Tdg",
    "Tdg": "T",
    "S": "Sdg",
    "Sdg": "S",
}


# ------------------------------------------------------------------
# Gate sequence
# ------------------------------------------------------------------

@dataclass
class GateSequence:
    """Sequence of gates from {H, T, S, Tdg, Sdg}.

    Stores both the symbolic gate list and a cached product matrix
    for fast distance computations.
    """

    gates: List[str] = field(default_factory=list)
    _matrix: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def matrix(self) -> np.ndarray:
        """Compute (or return cached) product matrix of the gate sequence."""
        if self._matrix is not None:
            return self._matrix.copy()
        result = np.eye(2, dtype=complex)
        for g in self.gates:
            result = _GATE_MAP[g] @ result
        self._matrix = result
        return result.copy()

    def _invalidate_cache(self) -> None:
        self._matrix = None

    @property
    def t_count(self) -> int:
        """Number of T and Tdg gates (dominant resource cost)."""
        return sum(1 for g in self.gates if g in ("T", "Tdg"))

    @property
    def depth(self) -> int:
        """Total number of gates in the sequence."""
        return len(self.gates)

    def __add__(self, other: "GateSequence") -> "GateSequence":
        """Concatenate two gate sequences.

        Convention: gates listed first are applied first (rightmost in
        matrix product).  So ``self + other`` has matrix
        ``other.matrix() @ self.matrix()``.
        """
        combined = GateSequence(self.gates + other.gates)
        # Compute the combined matrix from the cached matrices.
        if self._matrix is not None and other._matrix is not None:
            combined._matrix = other._matrix @ self._matrix
        return combined

    def inverse(self) -> "GateSequence":
        """Return the inverse sequence: reverse order, invert each gate."""
        inv_gates = [_INV_MAP[g] for g in reversed(self.gates)]
        inv = GateSequence(inv_gates)
        if self._matrix is not None:
            inv._matrix = self._matrix.conj().T
        return inv

    def __repr__(self) -> str:
        if len(self.gates) <= 20:
            return f"GateSequence({' '.join(self.gates)})"
        return f"GateSequence({len(self.gates)} gates, T-count={self.t_count})"


# ------------------------------------------------------------------
# Distance metric
# ------------------------------------------------------------------

def operator_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Trace distance between two 2x2 unitaries (up to global phase).

    Computes sqrt(1 - (|Tr(U V^dag)| / 2)^2), which is zero when
    U and V differ only by a global phase.

    Parameters
    ----------
    u, v : np.ndarray
        2x2 unitary matrices.

    Returns
    -------
    float
        Distance in [0, 1].
    """
    product = u @ v.conj().T
    trace = np.trace(product)
    fidelity = (np.abs(trace) / 2.0) ** 2
    return math.sqrt(max(0.0, 1.0 - fidelity))


def _to_su2(u: np.ndarray) -> np.ndarray:
    """Project a 2x2 unitary to SU(2) by removing the global phase."""
    det = np.linalg.det(u)
    phase = np.sqrt(det)
    if abs(phase) < 1e-15:
        return u.copy()
    result = u / phase
    # Fix the branch: ensure det ~ +1.
    if np.linalg.det(result).real < 0:
        result = -result
    return result


# ------------------------------------------------------------------
# Basic approximations database
# ------------------------------------------------------------------

@dataclass
class BasicApproximations:
    """Database of basic gate sequences for initial approximation.

    Pre-generates all gate sequences up to a given depth and stores
    their product matrices for fast nearest-neighbor lookup.
    """

    sequences: List[GateSequence] = field(default_factory=list)
    matrices: List[np.ndarray] = field(default_factory=list)

    @staticmethod
    def generate(max_depth: int = 4) -> "BasicApproximations":
        """Generate all non-redundant gate sequences up to given depth.

        Applies simple reduction rules to avoid obviously redundant
        sequences (e.g., H-H, T-T-T-T-T-T-T-T = I for T^8).

        Parameters
        ----------
        max_depth : int
            Maximum sequence length.

        Returns
        -------
        BasicApproximations
            Database of sequences and their matrices.
        """
        gate_names = ["H", "T", "S", "Tdg", "Sdg"]

        # Start with the identity.
        sequences: List[GateSequence] = [GateSequence([])]
        matrices: List[np.ndarray] = [np.eye(2, dtype=complex)]

        # BFS expansion.
        current_level: List[GateSequence] = [GateSequence([])]

        for depth in range(max_depth):
            next_level: List[GateSequence] = []
            for seq in current_level:
                for g in gate_names:
                    # Simple redundancy filter: do not add inverse of last gate.
                    if seq.gates and _INV_MAP.get(seq.gates[-1]) == g:
                        continue
                    # Do not repeat the same gate more than 3 times.
                    if (
                        len(seq.gates) >= 3
                        and seq.gates[-1] == g
                        and seq.gates[-2] == g
                        and seq.gates[-3] == g
                    ):
                        continue

                    new_seq = GateSequence(seq.gates + [g])
                    mat = new_seq.matrix()

                    # Check if this matrix is already in the database
                    # (up to global phase). Use a fast check.
                    is_duplicate = False
                    for existing_mat in matrices:
                        if operator_distance(mat, existing_mat) < 1e-10:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        sequences.append(new_seq)
                        matrices.append(mat)
                        next_level.append(new_seq)

            current_level = next_level

        return BasicApproximations(sequences=sequences, matrices=matrices)

    def find_closest(self, target: np.ndarray) -> GateSequence:
        """Find the sequence whose matrix is closest to ``target``.

        Uses a linear scan with operator distance. For production use,
        this would be replaced by a KD-tree on the SU(2) manifold.

        Parameters
        ----------
        target : np.ndarray
            2x2 target unitary.

        Returns
        -------
        GateSequence
            The closest sequence in the database.
        """
        best_dist = float("inf")
        best_seq = self.sequences[0]
        for seq, mat in zip(self.sequences, self.matrices):
            d = operator_distance(target, mat)
            if d < best_dist:
                best_dist = d
                best_seq = seq
        return best_seq


# ------------------------------------------------------------------
# Solovay-Kitaev algorithm
# ------------------------------------------------------------------

@dataclass
class SolovayKitaev:
    """Solovay-Kitaev algorithm for SU(2) approximation.

    Given target U in SU(2), finds a sequence of {H, T, S} gates
    approximating U to precision epsilon with O(log^c(1/epsilon))
    gates where c ~ 3.97.

    Algorithm:

    1. Base case: use ``basic_approximations`` to find the closest
       sequence in the database.
    2. Recursive step: given U_{n-1} (depth n-1 approximation),
       compute delta = U @ U_{n-1}^dag, decompose delta into a
       group commutator V W V^dag W^dag, recursively approximate
       V and W at depth n-1, and combine.

    Parameters
    ----------
    recursion_depth : int
        Number of recursive refinement steps (default 3).
    basic_approximations : BasicApproximations or None
        Pre-computed database. If None, generates one at depth 4.
    """

    recursion_depth: int = 3
    basic_approximations: Optional[BasicApproximations] = None

    def __post_init__(self):
        if self.basic_approximations is None:
            self.basic_approximations = BasicApproximations.generate(max_depth=4)

    def approximate(self, target: np.ndarray) -> GateSequence:
        """Find a gate sequence approximating the target SU(2) matrix.

        Parameters
        ----------
        target : np.ndarray
            2x2 unitary (will be projected to SU(2)).

        Returns
        -------
        GateSequence
            Approximating gate sequence.
        """
        target_su2 = _to_su2(target)
        return self._sk_recursive(target_su2, self.recursion_depth)

    def _sk_recursive(self, target: np.ndarray, depth: int) -> GateSequence:
        """Recursive Solovay-Kitaev step.

        At depth 0, returns the closest sequence from the database.
        At depth n, refines the depth-(n-1) approximation using the
        group commutator decomposition.
        """
        if depth == 0:
            return self.basic_approximations.find_closest(target)

        # Get the (depth-1) approximation.
        u_prev = self._sk_recursive(target, depth - 1)
        u_prev_mat = u_prev.matrix()

        # Error: delta = target @ U_{n-1}^dag, should be close to I.
        delta = target @ u_prev_mat.conj().T

        # If already very close, no need to refine.
        if operator_distance(delta, np.eye(2, dtype=complex)) < 1e-12:
            return u_prev

        # Group commutator decomposition: find V, W such that
        # delta ~ V W V^dag W^dag.
        v_seq, w_seq = self._group_commutator_decompose(delta)

        # Recursively approximate V and W at depth (n-1).
        v_approx = self._sk_recursive(v_seq.matrix(), depth - 1)
        w_approx = self._sk_recursive(w_seq.matrix(), depth - 1)

        # Build the refined approximation:
        # U_n = V_n W_n V_n^dag W_n^dag U_{n-1}
        result = u_prev + w_approx.inverse() + v_approx.inverse() + w_approx + v_approx
        return result

    def _group_commutator_decompose(
        self, delta: np.ndarray
    ) -> Tuple[GateSequence, GateSequence]:
        """Decompose delta (close to I) into a group commutator [V, W].

        For delta in SU(2) near identity, we can write:
            delta = V W V^dag W^dag

        Using the Rodrigues rotation formula on SU(2): if delta is a
        small rotation by angle alpha around axis n, then V and W are
        rotations by angle ~ sqrt(alpha) around appropriately chosen
        axes that satisfy the commutator relation.

        Returns
        -------
        (GateSequence, GateSequence)
            Sequences for V and W.
        """
        delta_su2 = _to_su2(delta)

        # Extract rotation angle from delta.
        trace_val = np.trace(delta_su2)
        cos_half = np.real(trace_val) / 2.0
        cos_half = max(-1.0, min(1.0, cos_half))
        alpha = 2.0 * math.acos(abs(cos_half))

        if alpha < 1e-12:
            # delta ~ identity, return trivial sequences.
            return GateSequence([]), GateSequence([])

        # Extract the rotation axis from the off-diagonal elements.
        # delta = cos(a/2)*I - i*sin(a/2)*(nx*X + ny*Y + nz*Z)
        sin_half = math.sin(alpha / 2.0)
        if abs(sin_half) < 1e-15:
            return GateSequence([]), GateSequence([])

        # Handle the sign of cos_half for the axis extraction.
        sign = 1.0 if cos_half >= 0 else -1.0

        nx = -sign * float(np.imag(delta_su2[0, 1] + delta_su2[1, 0])) / (2 * sin_half)
        ny = -sign * float(np.real(delta_su2[0, 1] - delta_su2[1, 0])) / (2 * sin_half)
        nz = -sign * float(np.imag(delta_su2[0, 0] - delta_su2[1, 1])) / (2 * sin_half)

        # Normalize the axis.
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm < 1e-15:
            return GateSequence([]), GateSequence([])
        nx, ny, nz = nx / norm, ny / norm, nz / norm

        # Choose V and W axes.  We need two axes orthogonal to each
        # other and to n such that the commutator produces a rotation
        # around n.  A simple choice:
        # V rotates around axis v_axis by angle ~ sqrt(alpha),
        # W rotates around axis w_axis by angle ~ sqrt(alpha),
        # where v_axis and w_axis are constructed to produce the
        # correct commutator.

        # Use Euler-angle based approach: pick v_axis perpendicular to n,
        # w_axis = n x v_axis.
        beta = math.sqrt(abs(alpha))
        # Clamp to prevent overflow.
        beta = min(beta, math.pi)

        # Find a vector perpendicular to n.
        if abs(nx) < 0.9:
            perp = np.array([0.0, -nz, ny])
        else:
            perp = np.array([-nz, 0.0, nx])
        perp_norm = np.linalg.norm(perp)
        if perp_norm < 1e-15:
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp = perp / perp_norm

        # Orthogonal axis.
        cross = np.array([
            ny * perp[2] - nz * perp[1],
            nz * perp[0] - nx * perp[2],
            nx * perp[1] - ny * perp[0],
        ])
        cross_norm = np.linalg.norm(cross)
        if cross_norm > 1e-15:
            cross = cross / cross_norm

        # Build V and W as rotation matrices in SU(2).
        v_mat = _axis_angle_to_su2(perp, beta)
        w_mat = _axis_angle_to_su2(cross, beta)

        # Find closest gate sequences in the database.
        v_seq = self.basic_approximations.find_closest(v_mat)
        w_seq = self.basic_approximations.find_closest(w_mat)

        return v_seq, w_seq


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class SKResult:
    """Result of Solovay-Kitaev approximation.

    Attributes
    ----------
    sequence : GateSequence
        The approximating gate sequence.
    target : np.ndarray
        The original target unitary.
    approximation : np.ndarray
        The matrix produced by the gate sequence.
    error : float
        Operator distance between target and approximation.
    t_count : int
        Number of T/Tdg gates in the sequence.
    total_gates : int
        Total number of gates.
    """

    sequence: GateSequence
    target: np.ndarray
    approximation: np.ndarray
    error: float
    t_count: int
    total_gates: int


# ------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------

def _axis_angle_to_su2(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert a rotation axis and angle to a 2x2 SU(2) matrix.

    R = cos(a/2)*I - i*sin(a/2)*(nx*X + ny*Y + nz*Z)
    """
    c = math.cos(angle / 2.0)
    s = math.sin(angle / 2.0)
    nx, ny, nz = axis[0], axis[1], axis[2]
    return np.array([
        [c - 1j * s * nz, -s * ny - 1j * s * nx],
        [s * ny - 1j * s * nx, c + 1j * s * nz],
    ], dtype=complex)


def _rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """Build a single-qubit rotation matrix.

    Parameters
    ----------
    axis : str
        ``"x"``, ``"y"``, or ``"z"``.
    angle : float
        Rotation angle in radians.
    """
    c = math.cos(angle / 2.0)
    s = math.sin(angle / 2.0)
    if axis.lower() == "x":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if axis.lower() == "y":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if axis.lower() == "z":
        return np.array(
            [[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]],
            dtype=complex,
        )
    raise ValueError(f"Unknown axis: {axis}")


def approximate_rotation(
    axis: str, angle: float, recursion_depth: int = 3
) -> SKResult:
    """Approximate a single-qubit rotation with {H, T, S}.

    Parameters
    ----------
    axis : str
        Rotation axis: ``"x"``, ``"y"``, or ``"z"``.
    angle : float
        Rotation angle in radians.
    recursion_depth : int
        SK recursion depth (default 3).

    Returns
    -------
    SKResult
        The approximation result with error metrics.
    """
    target = _rotation_matrix(axis, angle)
    sk = SolovayKitaev(recursion_depth=recursion_depth)
    seq = sk.approximate(target)
    approx_mat = seq.matrix()
    error = operator_distance(target, approx_mat)

    return SKResult(
        sequence=seq,
        target=target,
        approximation=approx_mat,
        error=error,
        t_count=seq.t_count,
        total_gates=seq.depth,
    )


def approximate_u3(
    theta: float, phi: float, lam: float, recursion_depth: int = 3
) -> SKResult:
    """Approximate U3(theta, phi, lambda) with {H, T, S}.

    Parameters
    ----------
    theta, phi, lam : float
        U3 gate parameters.
    recursion_depth : int
        SK recursion depth (default 3).

    Returns
    -------
    SKResult
        The approximation result with error metrics.
    """
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    target = np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
    ], dtype=complex)

    sk = SolovayKitaev(recursion_depth=recursion_depth)
    seq = sk.approximate(target)
    approx_mat = seq.matrix()
    error = operator_distance(target, approx_mat)

    return SKResult(
        sequence=seq,
        target=target,
        approximation=approx_mat,
        error=error,
        t_count=seq.t_count,
        total_gates=seq.depth,
    )
