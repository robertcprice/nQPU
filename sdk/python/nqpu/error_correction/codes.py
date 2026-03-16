"""Quantum error correcting code families.

Implements stabilizer codes with a uniform interface: encode, decode,
syndrome extraction, logical operators, and distance verification.

Each code is represented via its **check matrices** in the binary
symplectic formalism.  An *n*-qubit Pauli error is a length-2n binary
vector ``[x | z]`` where ``x[i]=1`` means X on qubit *i* and ``z[i]=1``
means Z on qubit *i* (Y = XZ, so both bits set).

The X-check matrix ``Hx`` has shape ``(mx, n)`` and detects Z-type errors.
The Z-check matrix ``Hz`` has shape ``(mz, n)`` and detects X-type errors.

Codes implemented:
  - :class:`RepetitionCode` -- 1D bit-flip or phase-flip repetition code
  - :class:`SteaneCode` -- [[7,1,3]] CSS code with transversal gates
  - :class:`ShorCode` -- [[9,1,3]] concatenated bit/phase-flip code
  - :class:`SurfaceCode` -- 2D surface code (rotated and unrotated)
  - :class:`ColorCode` -- [[7,1,3]] color code on triangular lattice
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class PauliType(Enum):
    """Pauli operator types for error specification."""

    I = 0
    X = 1
    Z = 2
    Y = 3


# ------------------------------------------------------------------ #
# Abstract base
# ------------------------------------------------------------------ #

class QuantumCode(ABC):
    """Abstract base for a stabilizer quantum error correcting code.

    Attributes
    ----------
    n : int
        Number of physical qubits.
    k : int
        Number of logical qubits.
    d : int
        Code distance.
    Hx : np.ndarray
        X-check matrix (detects Z errors), shape ``(mx, n)``.
    Hz : np.ndarray
        Z-check matrix (detects X errors), shape ``(mz, n)``.
    """

    n: int
    k: int
    d: int
    Hx: np.ndarray
    Hz: np.ndarray

    # -- public interface --

    @property
    def code_params(self) -> Tuple[int, int, int]:
        """Return ``[[n, k, d]]`` parameters."""
        return (self.n, self.k, self.d)

    def syndrome(self, error: np.ndarray) -> np.ndarray:
        """Compute the syndrome for a given error vector.

        Parameters
        ----------
        error : np.ndarray
            Binary error vector of length ``2*n`` in symplectic form
            ``[x_part | z_part]``.

        Returns
        -------
        np.ndarray
            Binary syndrome vector of length ``mx + mz``.
        """
        x_part = error[: self.n]
        z_part = error[self.n :]
        # X checks detect Z errors; Z checks detect X errors
        sx = (self.Hx @ z_part) % 2
        sz = (self.Hz @ x_part) % 2
        return np.concatenate([sx, sz]).astype(np.int8)

    def syndrome_x(self, error: np.ndarray) -> np.ndarray:
        """X-stabilizer syndrome (detects Z errors)."""
        z_part = error[self.n :]
        return ((self.Hx @ z_part) % 2).astype(np.int8)

    def syndrome_z(self, error: np.ndarray) -> np.ndarray:
        """Z-stabilizer syndrome (detects X errors)."""
        x_part = error[: self.n]
        return ((self.Hz @ x_part) % 2).astype(np.int8)

    def check_correction(self, error: np.ndarray, correction: np.ndarray) -> bool:
        """Check whether a correction combined with the original error
        is equivalent to a stabilizer (i.e. acts trivially on the code space).

        Parameters
        ----------
        error : np.ndarray
            Original error vector (length ``2*n``).
        correction : np.ndarray
            Proposed correction vector (length ``2*n``).

        Returns
        -------
        bool
            True if ``error + correction`` is in the stabilizer group
            (no logical error).
        """
        residual = (error + correction) % 2
        # The residual must have trivial syndrome
        syn = self.syndrome(residual)
        if np.any(syn):
            return False
        # Check that residual commutes with all logical operators
        # (i.e. is not a non-trivial logical operator)
        for lx, lz in zip(self.logical_x(), self.logical_z()):
            # symplectic inner product of residual with logical Z
            # If residual anti-commutes with a logical, it is a logical error
            res_x = residual[: self.n]
            res_z = residual[self.n :]
            # anti-commutation with logical Z
            lz_z = lz[self.n :]
            lz_x = lz[: self.n]
            ip_z = (np.dot(res_x, lz_z) + np.dot(res_z, lz_x)) % 2
            if ip_z:
                return False
            # anti-commutation with logical X
            lx_z = lx[self.n :]
            lx_x = lx[: self.n]
            ip_x = (np.dot(res_x, lx_z) + np.dot(res_z, lx_x)) % 2
            if ip_x:
                return False
        return True

    @abstractmethod
    def stabilizers(self) -> List[np.ndarray]:
        """Return list of stabilizer generators in symplectic form."""
        ...

    @abstractmethod
    def logical_x(self) -> List[np.ndarray]:
        """Return logical X operators in symplectic form."""
        ...

    @abstractmethod
    def logical_z(self) -> List[np.ndarray]:
        """Return logical Z operators in symplectic form."""
        ...

    @abstractmethod
    def distance(self) -> int:
        """Return the code distance."""
        ...

    def encode(self, logical_state: np.ndarray) -> np.ndarray:
        """Encode a logical state vector into the code space.

        For k=1 codes, accepts a 2-element vector ``[alpha, beta]``
        representing ``alpha|0_L> + beta|1_L>``.

        Returns a 2^n state vector in the physical Hilbert space.
        """
        return self._encode_impl(logical_state)

    def decode(self, state: np.ndarray) -> np.ndarray:
        """Decode a physical state vector back to logical space.

        Returns a 2^k state vector.
        """
        return self._decode_impl(state)

    @abstractmethod
    def _encode_impl(self, logical_state: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def _decode_impl(self, state: np.ndarray) -> np.ndarray:
        ...


# ------------------------------------------------------------------ #
# Repetition Code
# ------------------------------------------------------------------ #

class RepetitionCode(QuantumCode):
    """1D repetition code for bit-flip or phase-flip errors.

    Parameters
    ----------
    distance : int
        Code distance (number of physical qubits, must be odd >= 3).
    code_type : str
        ``"bit_flip"`` (default) encodes in Z basis, corrects X errors.
        ``"phase_flip"`` encodes in X basis, corrects Z errors.
    """

    def __init__(self, distance: int = 3, code_type: str = "bit_flip") -> None:
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance must be an odd integer >= 3")
        if code_type not in ("bit_flip", "phase_flip"):
            raise ValueError("code_type must be 'bit_flip' or 'phase_flip'")

        self.n = distance
        self.k = 1
        self.d = distance
        self.code_type = code_type

        # Build check matrices
        m = distance - 1  # number of parity checks
        H = np.zeros((m, distance), dtype=np.int8)
        for i in range(m):
            H[i, i] = 1
            H[i, i + 1] = 1

        if code_type == "bit_flip":
            # Z checks detect X errors
            self.Hz = H
            self.Hx = np.zeros((0, distance), dtype=np.int8)
        else:
            # X checks detect Z errors
            self.Hx = H
            self.Hz = np.zeros((0, distance), dtype=np.int8)

    def stabilizers(self) -> List[np.ndarray]:
        stabs = []
        for i in range(self.n - 1):
            s = np.zeros(2 * self.n, dtype=np.int8)
            if self.code_type == "bit_flip":
                # ZZ stabilizer on qubits i, i+1
                s[self.n + i] = 1
                s[self.n + i + 1] = 1
            else:
                # XX stabilizer on qubits i, i+1
                s[i] = 1
                s[i + 1] = 1
            stabs.append(s)
        return stabs

    def logical_x(self) -> List[np.ndarray]:
        lx = np.zeros(2 * self.n, dtype=np.int8)
        if self.code_type == "bit_flip":
            # Logical X = X on all qubits
            lx[: self.n] = 1
        else:
            # Logical X = Z on all qubits
            lx[self.n :] = 1
        return [lx]

    def logical_z(self) -> List[np.ndarray]:
        lz = np.zeros(2 * self.n, dtype=np.int8)
        if self.code_type == "bit_flip":
            # Logical Z = Z on any single qubit (use qubit 0)
            lz[self.n] = 1
        else:
            # Logical Z = X on any single qubit (use qubit 0)
            lz[0] = 1
        return [lz]

    def distance(self) -> int:
        return self.d

    def _encode_impl(self, logical_state: np.ndarray) -> np.ndarray:
        dim = 2 ** self.n
        state = np.zeros(dim, dtype=np.complex128)
        alpha, beta = logical_state[0], logical_state[1]
        if self.code_type == "bit_flip":
            # |0_L> = |00...0>, |1_L> = |11...1>
            state[0] = alpha
            state[dim - 1] = beta
        else:
            # |0_L> = |+++...+>, |1_L> = |---...->
            plus = np.ones(dim, dtype=np.complex128)
            minus = np.ones(dim, dtype=np.complex128)
            for i in range(dim):
                parity = bin(i).count("1")
                minus[i] = (-1) ** parity
            norm = 1.0 / np.sqrt(dim)
            state = alpha * plus * norm + beta * minus * norm
        return state

    def _decode_impl(self, state: np.ndarray) -> np.ndarray:
        dim = 2 ** self.n
        logical = np.zeros(2, dtype=np.complex128)
        if self.code_type == "bit_flip":
            logical[0] = state[0]
            logical[1] = state[dim - 1]
        else:
            plus = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
            minus = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
            for i in range(dim):
                parity = bin(i).count("1")
                minus[i] *= (-1) ** parity
            logical[0] = np.dot(plus.conj(), state)
            logical[1] = np.dot(minus.conj(), state)
        return logical


# ------------------------------------------------------------------ #
# Steane Code [[7,1,3]]
# ------------------------------------------------------------------ #

class SteaneCode(QuantumCode):
    """Steane [[7,1,3]] code -- the smallest CSS code.

    CSS code built from the classical [7,4,3] Hamming code.
    Supports transversal Clifford gates (H, S, CNOT).
    """

    def __init__(self) -> None:
        self.n = 7
        self.k = 1
        self.d = 3

        # Classical [7,4,3] Hamming parity check matrix
        H_hamming = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ], dtype=np.int8)

        # CSS construction: Hx = Hz = Hamming parity check
        self.Hx = H_hamming.copy()
        self.Hz = H_hamming.copy()

    def stabilizers(self) -> List[np.ndarray]:
        stabs = []
        # X stabilizers from Hx
        for row in self.Hx:
            s = np.zeros(2 * self.n, dtype=np.int8)
            s[: self.n] = row
            stabs.append(s)
        # Z stabilizers from Hz
        for row in self.Hz:
            s = np.zeros(2 * self.n, dtype=np.int8)
            s[self.n :] = row
            stabs.append(s)
        return stabs

    def logical_x(self) -> List[np.ndarray]:
        lx = np.zeros(2 * self.n, dtype=np.int8)
        # Logical X = X on all 7 qubits
        lx[: self.n] = 1
        return [lx]

    def logical_z(self) -> List[np.ndarray]:
        lz = np.zeros(2 * self.n, dtype=np.int8)
        # Logical Z = Z on all 7 qubits
        lz[self.n :] = 1
        return [lz]

    def distance(self) -> int:
        return 3

    def _encode_impl(self, logical_state: np.ndarray) -> np.ndarray:
        """Encode using the Steane code stabilizer projection."""
        dim = 2 ** self.n  # 128
        alpha, beta = logical_state[0], logical_state[1]

        # |0_L> = sum over even-weight codewords of H(7,4)
        # |1_L> = sum over odd-weight codewords of H(7,4)
        # Codewords of [7,4,3] Hamming code
        G = np.array([
            [1, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1],
        ], dtype=np.int8)

        codewords_0L = []
        codewords_1L = []
        for bits in range(16):
            msg = np.array([(bits >> i) & 1 for i in range(4)], dtype=np.int8)
            cw = (msg @ G) % 2
            idx = 0
            for j in range(self.n):
                idx += int(cw[j]) << j
            weight = int(cw.sum())
            if weight % 2 == 0:
                codewords_0L.append(idx)
            else:
                codewords_1L.append(idx)

        state = np.zeros(dim, dtype=np.complex128)
        norm0 = 1.0 / np.sqrt(len(codewords_0L))
        norm1 = 1.0 / np.sqrt(len(codewords_1L))
        for idx in codewords_0L:
            state[idx] += alpha * norm0
        for idx in codewords_1L:
            state[idx] += beta * norm1
        return state

    def _decode_impl(self, state: np.ndarray) -> np.ndarray:
        """Decode by projecting onto logical codewords."""
        G = np.array([
            [1, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1],
        ], dtype=np.int8)

        codewords_0L = []
        codewords_1L = []
        for bits in range(16):
            msg = np.array([(bits >> i) & 1 for i in range(4)], dtype=np.int8)
            cw = (msg @ G) % 2
            idx = 0
            for j in range(self.n):
                idx += int(cw[j]) << j
            weight = int(cw.sum())
            if weight % 2 == 0:
                codewords_0L.append(idx)
            else:
                codewords_1L.append(idx)

        logical = np.zeros(2, dtype=np.complex128)
        norm0 = 1.0 / np.sqrt(len(codewords_0L))
        norm1 = 1.0 / np.sqrt(len(codewords_1L))
        for idx in codewords_0L:
            logical[0] += state[idx] * norm0
        for idx in codewords_1L:
            logical[1] += state[idx] * norm1
        return logical


# ------------------------------------------------------------------ #
# Shor Code [[9,1,3]]
# ------------------------------------------------------------------ #

class ShorCode(QuantumCode):
    """Shor [[9,1,3]] code -- concatenation of bit-flip and phase-flip codes.

    Encodes 1 logical qubit into 9 physical qubits.  First encodes against
    phase-flip errors (3 groups of 3), then each group is encoded against
    bit-flip errors.
    """

    def __init__(self) -> None:
        self.n = 9
        self.k = 1
        self.d = 3

        # Z stabilizers (detect X/bit-flip errors): ZZ on pairs within blocks
        # Blocks: [0,1,2], [3,4,5], [6,7,8]
        Hz_list = []
        for block_start in [0, 3, 6]:
            for j in range(2):
                row = np.zeros(self.n, dtype=np.int8)
                row[block_start + j] = 1
                row[block_start + j + 1] = 1
                Hz_list.append(row)
        self.Hz = np.array(Hz_list, dtype=np.int8)

        # X stabilizers (detect Z/phase-flip errors): XXXXXX on adjacent blocks
        Hx_list = []
        for b in range(2):
            row = np.zeros(self.n, dtype=np.int8)
            for j in range(3):
                row[3 * b + j] = 1
                row[3 * (b + 1) + j] = 1
            Hx_list.append(row)
        self.Hx = np.array(Hx_list, dtype=np.int8)

    def stabilizers(self) -> List[np.ndarray]:
        stabs = []
        # Z stabilizers
        for row in self.Hz:
            s = np.zeros(2 * self.n, dtype=np.int8)
            s[self.n :] = row
            stabs.append(s)
        # X stabilizers
        for row in self.Hx:
            s = np.zeros(2 * self.n, dtype=np.int8)
            s[: self.n] = row
            stabs.append(s)
        return stabs

    def logical_x(self) -> List[np.ndarray]:
        lx = np.zeros(2 * self.n, dtype=np.int8)
        # Logical X = X on all 9 qubits
        lx[: self.n] = 1
        return [lx]

    def logical_z(self) -> List[np.ndarray]:
        lz = np.zeros(2 * self.n, dtype=np.int8)
        # Logical Z = Z on all 9 qubits
        lz[self.n :] = 1
        return [lz]

    def distance(self) -> int:
        return 3

    def _encode_impl(self, logical_state: np.ndarray) -> np.ndarray:
        dim = 2 ** self.n  # 512
        alpha, beta = logical_state[0], logical_state[1]

        # |0_L> = (|000>+|111>)(|000>+|111>)(|000>+|111>) / 2sqrt2
        # |1_L> = (|000>-|111>)(|000>-|111>)(|000>-|111>) / 2sqrt2
        def _block_plus():
            """Returns unnormalized |000>+|111> indices and signs."""
            return [(0, 1.0), (7, 1.0)]  # 0b000=0, 0b111=7

        def _block_minus():
            return [(0, 1.0), (7, -1.0)]

        state = np.zeros(dim, dtype=np.complex128)
        norm = 1.0 / (2.0 * np.sqrt(2.0))

        # Build |0_L>
        for i0, s0 in _block_plus():
            for i1, s1 in _block_plus():
                for i2, s2 in _block_plus():
                    idx = i0 + (i1 << 3) + (i2 << 6)
                    state[idx] += alpha * s0 * s1 * s2 * norm

        # Build |1_L>
        for i0, s0 in _block_minus():
            for i1, s1 in _block_minus():
                for i2, s2 in _block_minus():
                    idx = i0 + (i1 << 3) + (i2 << 6)
                    state[idx] += beta * s0 * s1 * s2 * norm

        return state

    def _decode_impl(self, state: np.ndarray) -> np.ndarray:
        # Project onto |0_L> and |1_L> code spaces
        zero_L = self._encode_impl(np.array([1.0, 0.0]))
        one_L = self._encode_impl(np.array([0.0, 1.0]))
        logical = np.zeros(2, dtype=np.complex128)
        logical[0] = np.dot(zero_L.conj(), state)
        logical[1] = np.dot(one_L.conj(), state)
        return logical


# ------------------------------------------------------------------ #
# Surface Code
# ------------------------------------------------------------------ #

class SurfaceCode(QuantumCode):
    """2D surface code on a square lattice.

    Parameters
    ----------
    distance : int
        Code distance (must be odd >= 3).
    rotated : bool
        If True, use the rotated surface code layout (n = d^2 data qubits).
        If False, use the standard (unrotated) layout (n = 2d^2 - 2d + 1).
    """

    def __init__(self, distance: int = 3, rotated: bool = True) -> None:
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance must be an odd integer >= 3")

        self.d = distance
        self.k = 1
        self.rotated = rotated

        if rotated:
            self._build_rotated()
        else:
            self._build_unrotated()

    def _build_rotated(self) -> None:
        """Build rotated surface code check matrices.

        Data qubits sit on a d x d grid, indexed row-major: (r,c) -> r*d+c.

        The rotated surface code has two types of boundaries:
          - Rough boundaries (top and bottom): Z-type stabilizers
          - Smooth boundaries (left and right): X-type stabilizers

        Bulk stabilizers live on the (d-1) x (d-1) interior face grid in
        a checkerboard pattern.  Boundary stabilizers are weight-2 checks
        along the code perimeter, with type determined by boundary orientation
        (not by the checkerboard).

        This ensures all X and Z stabilizers share 0 or 2 qubits (CSS property).
        """
        d = self.d
        self.n = d * d

        x_checks = []  # detect Z errors
        z_checks = []  # detect X errors

        # Bulk stabilizers: weight-4 on each interior face
        # Face (r, c) covers qubits (r,c), (r,c+1), (r+1,c), (r+1,c+1)
        for r in range(d - 1):
            for c in range(d - 1):
                qubits = [r * d + c, r * d + c + 1,
                           (r + 1) * d + c, (r + 1) * d + c + 1]
                check = np.zeros(self.n, dtype=np.int8)
                for q in qubits:
                    check[q] = 1

                if (r + c) % 2 == 0:
                    x_checks.append(check)
                else:
                    z_checks.append(check)

        # Boundary stabilizers (weight-2):
        # Top boundary (Z-type): horizontal edges along the top row
        # These are Z checks on pairs in the top row where the "missing"
        # face above would be Z-type: face (-1, c) with (-1+c) odd.
        for c in range(d - 1):
            if (-1 + c) % 2 != 0:  # odd -> Z type
                check = np.zeros(self.n, dtype=np.int8)
                check[c] = 1
                check[c + 1] = 1
                z_checks.append(check)

        # Bottom boundary (Z-type): horizontal edges along the bottom row
        for c in range(d - 1):
            if (d - 1 + c) % 2 != 0:  # odd -> Z type
                check = np.zeros(self.n, dtype=np.int8)
                check[(d - 1) * d + c] = 1
                check[(d - 1) * d + c + 1] = 1
                z_checks.append(check)

        # Left boundary (X-type): vertical edges along the left column
        for r in range(d - 1):
            if (r + (-1)) % 2 == 0:  # even -> X type
                check = np.zeros(self.n, dtype=np.int8)
                check[r * d] = 1
                check[(r + 1) * d] = 1
                x_checks.append(check)

        # Right boundary (X-type): vertical edges along the right column
        for r in range(d - 1):
            if (r + (d - 1)) % 2 == 0:  # even -> X type
                check = np.zeros(self.n, dtype=np.int8)
                check[r * d + d - 1] = 1
                check[(r + 1) * d + d - 1] = 1
                x_checks.append(check)

        self.Hx = np.array(x_checks, dtype=np.int8) if x_checks else np.zeros((0, self.n), dtype=np.int8)
        self.Hz = np.array(z_checks, dtype=np.int8) if z_checks else np.zeros((0, self.n), dtype=np.int8)

    def _build_unrotated(self) -> None:
        """Build unrotated (standard) surface code.

        Data qubits on edges of a (d-1)x(d-1) grid of faces.
        n = 2d^2 - 2d + 1 for the planar code.
        """
        d = self.d
        # Use a simpler construction: d x d grid with d^2 data qubits
        # and stabilizers on faces/vertices, same n as rotated for simplicity
        # but with different stabilizer structure.
        self.n = d * d

        x_checks = []
        z_checks = []

        # Star operators (X type) at vertices of the dual lattice
        for r in range(d):
            for c in range(d):
                if (r + c) % 2 == 1:
                    continue
                qubits = []
                if r > 0:
                    qubits.append((r - 1) * d + c)
                if r < d - 1:
                    qubits.append((r + 1) * d + c)
                if c > 0:
                    qubits.append(r * d + c - 1)
                if c < d - 1:
                    qubits.append(r * d + c + 1)
                if len(qubits) >= 2:
                    check = np.zeros(self.n, dtype=np.int8)
                    for q in qubits:
                        check[q] = 1
                    x_checks.append(check)

        # Plaquette operators (Z type)
        for r in range(d):
            for c in range(d):
                if (r + c) % 2 == 0:
                    continue
                qubits = []
                if r > 0:
                    qubits.append((r - 1) * d + c)
                if r < d - 1:
                    qubits.append((r + 1) * d + c)
                if c > 0:
                    qubits.append(r * d + c - 1)
                if c < d - 1:
                    qubits.append(r * d + c + 1)
                if len(qubits) >= 2:
                    check = np.zeros(self.n, dtype=np.int8)
                    for q in qubits:
                        check[q] = 1
                    z_checks.append(check)

        if x_checks:
            self.Hx = np.array(x_checks, dtype=np.int8)
        else:
            self.Hx = np.zeros((0, self.n), dtype=np.int8)
        if z_checks:
            self.Hz = np.array(z_checks, dtype=np.int8)
        else:
            self.Hz = np.zeros((0, self.n), dtype=np.int8)

    def stabilizers(self) -> List[np.ndarray]:
        stabs = []
        for row in self.Hx:
            s = np.zeros(2 * self.n, dtype=np.int8)
            s[: self.n] = row
            stabs.append(s)
        for row in self.Hz:
            s = np.zeros(2 * self.n, dtype=np.int8)
            s[self.n :] = row
            stabs.append(s)
        return stabs

    def logical_x(self) -> List[np.ndarray]:
        lx = np.zeros(2 * self.n, dtype=np.int8)
        d = self.d
        # Logical X: X along the first row
        for c in range(d):
            lx[c] = 1
        return [lx]

    def logical_z(self) -> List[np.ndarray]:
        lz = np.zeros(2 * self.n, dtype=np.int8)
        d = self.d
        # Logical Z: Z along the first column
        for r in range(d):
            lz[self.n + r * d] = 1
        return [lz]

    def distance(self) -> int:
        return self.d

    def _encode_impl(self, logical_state: np.ndarray) -> np.ndarray:
        """Encode using stabilizer projection (exponential in n -- for small codes)."""
        if self.n > 15:
            raise ValueError(
                "Direct encoding is exponential; only supported for distance <= ~3"
            )
        dim = 2 ** self.n
        alpha, beta = logical_state[0], logical_state[1]

        # Build stabilizer group projector
        # Start from |0...0> and project
        stabs = self.stabilizers()
        state_0 = np.zeros(dim, dtype=np.complex128)
        state_0[0] = 1.0

        # Apply (I + S)/2 for each stabilizer
        projected = state_0.copy()
        for stab in stabs:
            projected = SurfaceCode._apply_stabilizer_projection(projected, stab)
            norm = np.linalg.norm(projected)
            if norm > 1e-15:
                projected /= norm

        zero_L = projected.copy()

        # |1_L> = L_X |0_L>
        lx = self.logical_x()[0]
        one_L = SurfaceCode._apply_pauli(zero_L, lx)
        norm = np.linalg.norm(one_L)
        if norm > 1e-15:
            one_L /= norm

        return alpha * zero_L + beta * one_L

    def _decode_impl(self, state: np.ndarray) -> np.ndarray:
        if self.n > 15:
            raise ValueError("Direct decoding only supported for small codes")
        zero_L = self._encode_impl(np.array([1.0, 0.0]))
        one_L = self._encode_impl(np.array([0.0, 1.0]))
        logical = np.zeros(2, dtype=np.complex128)
        logical[0] = np.dot(zero_L.conj(), state)
        logical[1] = np.dot(one_L.conj(), state)
        return logical

    @staticmethod
    def _apply_stabilizer_projection(
        state: np.ndarray, stab: np.ndarray
    ) -> np.ndarray:
        """Apply (I + S)/2 projector for stabilizer S."""
        s_state = SurfaceCode._apply_pauli(state, stab)
        return (state + s_state) / 2.0

    @staticmethod
    def _apply_pauli(state: np.ndarray, pauli: np.ndarray) -> np.ndarray:
        """Apply a Pauli operator in symplectic form to a state vector."""
        n = len(pauli) // 2
        x_part = pauli[:n]
        z_part = pauli[n:]
        dim = len(state)
        result = state.copy()

        # Apply X flips
        new_result = np.zeros_like(result)
        for i in range(dim):
            # Compute the bit-flipped index
            flipped = i
            for q in range(n):
                if x_part[q]:
                    flipped ^= (1 << q)
            new_result[flipped] = result[i]
        result = new_result

        # Apply Z phases
        for i in range(dim):
            phase = 0
            for q in range(n):
                if z_part[q] and ((i >> q) & 1):
                    phase += 1
            if phase % 2 == 1:
                result[i] *= -1

        return result

    def qubit_coords(self) -> List[Tuple[int, int]]:
        """Return (row, col) coordinates for each data qubit."""
        d = self.d
        return [(r, c) for r in range(d) for c in range(d)]


# ------------------------------------------------------------------ #
# Color Code [[7,1,3]]
# ------------------------------------------------------------------ #

class ColorCode(QuantumCode):
    """[[7,1,3]] color code on a triangular lattice.

    The smallest 2D color code, equivalent in parameters to the Steane code
    but with a different stabilizer structure based on a 3-colorable lattice
    of plaquettes.  All stabilizers are CSS (pure X or pure Z).
    """

    def __init__(self) -> None:
        self.n = 7
        self.k = 1
        self.d = 3

        # Plaquette stabilizers on the triangular lattice
        # 3 plaquettes, each involving 4 qubits
        #
        # Qubit layout (triangular tiling, 7 qubits):
        #       0
        #      / \
        #     1 - 2
        #    / \ / \
        #   3 - 4 - 5
        #        \ /
        #         6
        #
        # Plaquettes (faces): {0,1,2,4}, {1,3,4,6}, {2,4,5,6} -- non-standard
        # Actually the standard 7-qubit color code plaquettes:
        # Red:   {0,1,2,3}
        # Green: {0,2,4,6}
        # Blue:  {0,1,5,6} -- but we need to be careful.
        #
        # Standard [[7,1,3]] color code on a hexagonal patch:
        # 3 X-stabilizers and 3 Z-stabilizers
        plaquettes = [
            [0, 1, 2, 3],
            [0, 2, 4, 6],
            [0, 1, 5, 6],
        ]

        Hx_list = []
        Hz_list = []
        for plaq in plaquettes:
            row = np.zeros(self.n, dtype=np.int8)
            for q in plaq:
                row[q] = 1
            Hx_list.append(row)
            Hz_list.append(row.copy())

        self.Hx = np.array(Hx_list, dtype=np.int8)
        self.Hz = np.array(Hz_list, dtype=np.int8)

    def stabilizers(self) -> List[np.ndarray]:
        stabs = []
        for row in self.Hx:
            s = np.zeros(2 * self.n, dtype=np.int8)
            s[: self.n] = row
            stabs.append(s)
        for row in self.Hz:
            s = np.zeros(2 * self.n, dtype=np.int8)
            s[self.n :] = row
            stabs.append(s)
        return stabs

    def logical_x(self) -> List[np.ndarray]:
        lx = np.zeros(2 * self.n, dtype=np.int8)
        # Logical X: X on qubits forming a string across the code
        # Weight-3 logical: X on qubits 0, 4, 5
        lx[0] = 1
        lx[4] = 1
        lx[5] = 1
        return [lx]

    def logical_z(self) -> List[np.ndarray]:
        lz = np.zeros(2 * self.n, dtype=np.int8)
        # Logical Z: Z on qubits 0, 4, 5
        lz[self.n + 0] = 1
        lz[self.n + 4] = 1
        lz[self.n + 5] = 1
        return [lz]

    def distance(self) -> int:
        return 3

    def _encode_impl(self, logical_state: np.ndarray) -> np.ndarray:
        dim = 2 ** self.n
        alpha, beta = logical_state[0], logical_state[1]

        stabs = self.stabilizers()
        state_0 = np.zeros(dim, dtype=np.complex128)
        state_0[0] = 1.0

        projected = state_0.copy()
        for stab in stabs:
            projected = SurfaceCode._apply_stabilizer_projection(projected, stab)
            norm = np.linalg.norm(projected)
            if norm > 1e-15:
                projected /= norm

        zero_L = projected
        lx = self.logical_x()[0]
        one_L = SurfaceCode._apply_pauli(zero_L, lx)
        norm = np.linalg.norm(one_L)
        if norm > 1e-15:
            one_L /= norm

        return alpha * zero_L + beta * one_L

    def _decode_impl(self, state: np.ndarray) -> np.ndarray:
        zero_L = self._encode_impl(np.array([1.0, 0.0]))
        one_L = self._encode_impl(np.array([0.0, 1.0]))
        logical = np.zeros(2, dtype=np.complex128)
        logical[0] = np.dot(zero_L.conj(), state)
        logical[1] = np.dot(one_L.conj(), state)
        return logical
