"""Core tensor operations for tensor network computations.

Provides named-index tensors with contraction, SVD/QR decomposition,
reshape, transpose, and trace operations. These primitives underpin
all higher-level tensor network algorithms (MPS, MPO, DMRG, TEBD).

Key concepts:
  - A ``Tensor`` wraps a numpy ndarray plus a list of named leg labels.
  - Contraction sums over shared legs between two tensors.
  - SVD truncation controls bond dimension (chi_max) and discards
    singular values below a threshold (cutoff).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# -------------------------------------------------------------------
# Tensor
# -------------------------------------------------------------------

class Tensor:
    """Multi-dimensional array with named legs (indices).

    Parameters
    ----------
    data : array-like
        The underlying numerical data (complex128 by default).
    legs : list[str]
        A name for each axis, in order.  Must match ``data.ndim``.

    Examples
    --------
    >>> t = Tensor(np.eye(2, dtype=complex), legs=["i", "j"])
    >>> t.shape
    (2, 2)
    >>> t.legs
    ['i', 'j']
    """

    __slots__ = ("data", "legs")

    def __init__(self, data, legs: List[str]) -> None:
        self.data: NDArray[np.complexfloating] = np.asarray(data, dtype=np.complex128)
        self.legs: List[str] = list(legs)
        if self.data.ndim != len(self.legs):
            raise ValueError(
                f"Number of legs ({len(self.legs)}) does not match "
                f"data dimensions ({self.data.ndim})"
            )

    # -- Properties ---------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def norm(self) -> float:
        """Frobenius norm of the tensor."""
        return float(np.linalg.norm(self.data))

    # -- Core operations ----------------------------------------------

    def transpose(self, new_legs: List[str]) -> "Tensor":
        """Reorder axes to match *new_legs* ordering.

        Parameters
        ----------
        new_legs : list[str]
            Desired leg order.  Must be a permutation of ``self.legs``.

        Returns
        -------
        Tensor
            A new tensor with axes reordered accordingly.
        """
        perm = [self.legs.index(l) for l in new_legs]
        return Tensor(self.data.transpose(perm), new_legs)

    def reshape(self, new_shape: Tuple[int, ...], new_legs: List[str]) -> "Tensor":
        """Reshape the underlying data and assign new leg names.

        Parameters
        ----------
        new_shape : tuple[int, ...]
            New shape compatible with the total number of elements.
        new_legs : list[str]
            Leg names for the reshaped tensor.
        """
        return Tensor(self.data.reshape(new_shape), new_legs)

    def trace(self, leg1: str, leg2: str) -> "Tensor":
        """Contract two legs of the *same* tensor (trace).

        Parameters
        ----------
        leg1, leg2 : str
            The two legs to trace over.  Their dimensions must match.

        Returns
        -------
        Tensor
            Tensor with the traced legs removed.
        """
        i = self.legs.index(leg1)
        j = self.legs.index(leg2)
        if self.data.shape[i] != self.data.shape[j]:
            raise ValueError(
                f"Cannot trace legs of different sizes: "
                f"{leg1}={self.data.shape[i]}, {leg2}={self.data.shape[j]}"
            )
        new_data = np.trace(self.data, axis1=i, axis2=j)
        remaining = [l for k, l in enumerate(self.legs) if k not in (i, j)]
        return Tensor(new_data, remaining)

    def conjugate(self) -> "Tensor":
        """Element-wise complex conjugate."""
        return Tensor(np.conj(self.data), list(self.legs))

    def copy(self) -> "Tensor":
        """Deep copy of the tensor."""
        return Tensor(self.data.copy(), list(self.legs))

    # -- Decompositions -----------------------------------------------

    def svd(
        self,
        left_legs: List[str],
        right_legs: List[str],
        chi_max: Optional[int] = None,
        cutoff: float = 0.0,
        absorb: str = "right",
    ) -> Tuple["Tensor", NDArray[np.floating], "Tensor"]:
        """Singular value decomposition with optional truncation.

        The tensor is reshaped into a matrix ``(left_legs) x (right_legs)``,
        decomposed as ``U S Vh``, then optionally truncated.

        Parameters
        ----------
        left_legs : list[str]
            Legs to group into the row index of the matrix.
        right_legs : list[str]
            Legs to group into the column index.
        chi_max : int, optional
            Keep at most ``chi_max`` singular values.
        cutoff : float
            Discard singular values smaller than ``cutoff``.
        absorb : str
            Where to absorb the singular values: ``"left"`` into U,
            ``"right"`` into Vh, or ``"none"`` to return them separately.

        Returns
        -------
        U : Tensor
            Left unitary with legs ``left_legs + ["svd_inner"]``.
        S : 1-D array
            Retained singular values.
        Vh : Tensor
            Right unitary with legs ``["svd_inner"] + right_legs``.
        """
        # Permute so left legs come first
        perm_order = left_legs + right_legs
        t = self.transpose(perm_order)

        left_dim = int(np.prod([t.data.shape[i] for i in range(len(left_legs))]))
        right_dim = int(np.prod([t.data.shape[i] for i in range(len(left_legs), t.ndim)]))
        mat = t.data.reshape(left_dim, right_dim)

        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # Truncation
        if cutoff > 0:
            keep = np.sum(S > cutoff)
            keep = max(keep, 1)  # keep at least 1
        else:
            keep = len(S)
        if chi_max is not None:
            keep = min(keep, chi_max)

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        # Absorb singular values
        if absorb == "right":
            Vh = np.diag(S) @ Vh
        elif absorb == "left":
            U = U @ np.diag(S)
        elif absorb != "none":
            raise ValueError(f"absorb must be 'left', 'right', or 'none', got {absorb!r}")

        # Reshape back
        left_shapes = [t.data.shape[i] for i in range(len(left_legs))]
        right_shapes = [t.data.shape[i] for i in range(len(left_legs), t.ndim)]

        U_tensor = Tensor(
            U.reshape(*left_shapes, keep),
            left_legs + ["svd_inner"],
        )
        Vh_tensor = Tensor(
            Vh.reshape(keep, *right_shapes),
            ["svd_inner"] + right_legs,
        )
        return U_tensor, S, Vh_tensor

    def qr(
        self,
        left_legs: List[str],
        right_legs: List[str],
    ) -> Tuple["Tensor", "Tensor"]:
        """QR decomposition.

        Parameters
        ----------
        left_legs : list[str]
            Legs grouped into the row index.
        right_legs : list[str]
            Legs grouped into the column index.

        Returns
        -------
        Q : Tensor
            Orthogonal factor with legs ``left_legs + ["qr_inner"]``.
        R : Tensor
            Upper-triangular factor with legs ``["qr_inner"] + right_legs``.
        """
        perm_order = left_legs + right_legs
        t = self.transpose(perm_order)

        left_dim = int(np.prod([t.data.shape[i] for i in range(len(left_legs))]))
        right_dim = int(np.prod([t.data.shape[i] for i in range(len(left_legs), t.ndim)]))
        mat = t.data.reshape(left_dim, right_dim)

        Q, R = np.linalg.qr(mat)
        k = Q.shape[1]

        left_shapes = [t.data.shape[i] for i in range(len(left_legs))]
        right_shapes = [t.data.shape[i] for i in range(len(left_legs), t.ndim)]

        Q_tensor = Tensor(Q.reshape(*left_shapes, k), left_legs + ["qr_inner"])
        R_tensor = Tensor(R.reshape(k, *right_shapes), ["qr_inner"] + right_legs)
        return Q_tensor, R_tensor

    # -- Dunder methods ------------------------------------------------

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, legs={self.legs})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.legs == other.legs and np.allclose(self.data, other.data)

    def __mul__(self, scalar: complex) -> "Tensor":
        return Tensor(self.data * scalar, list(self.legs))

    def __rmul__(self, scalar: complex) -> "Tensor":
        return self.__mul__(scalar)


# -------------------------------------------------------------------
# Pairwise tensor contraction
# -------------------------------------------------------------------

def contract_pair(
    A: Tensor,
    B: Tensor,
    legs_A: Optional[List[str]] = None,
    legs_B: Optional[List[str]] = None,
) -> Tensor:
    """Contract two tensors over specified (or auto-detected shared) legs.

    Parameters
    ----------
    A, B : Tensor
        Tensors to contract.
    legs_A : list[str], optional
        Legs of *A* to sum over.  If ``None``, automatically contracts
        all legs whose names appear in both tensors.
    legs_B : list[str], optional
        Corresponding legs of *B*.  Must pair 1-to-1 with *legs_A*.

    Returns
    -------
    Tensor
        Resulting tensor whose legs are the uncontracted legs of A
        followed by the uncontracted legs of B.

    Examples
    --------
    Contract two matrices sharing leg ``"j"``:

    >>> A = Tensor(np.eye(2, dtype=complex), ["i", "j"])
    >>> B = Tensor(np.ones((2, 3), dtype=complex), ["j", "k"])
    >>> C = contract_pair(A, B)
    >>> C.shape
    (2, 3)
    >>> C.legs
    ['i', 'k']
    """
    if legs_A is None or legs_B is None:
        shared = [l for l in A.legs if l in B.legs]
        legs_A = shared
        legs_B = shared

    if len(legs_A) != len(legs_B):
        raise ValueError("legs_A and legs_B must have the same length")

    axes_A = [A.legs.index(l) for l in legs_A]
    axes_B = [B.legs.index(l) for l in legs_B]

    result_data = np.tensordot(A.data, B.data, axes=(axes_A, axes_B))

    remaining_A = [l for i, l in enumerate(A.legs) if i not in axes_A]
    remaining_B = [l for i, l in enumerate(B.legs) if i not in axes_B]

    return Tensor(result_data, remaining_A + remaining_B)


# -------------------------------------------------------------------
# TensorNetwork
# -------------------------------------------------------------------

@dataclass
class ContractionStep:
    """One step in a contraction schedule."""
    tensor_id_a: int
    tensor_id_b: int
    result_id: int
    contracted_legs: List[str]


class TensorNetwork:
    """A collection of named tensors with an optional contraction order.

    Parameters
    ----------
    tensors : dict[str, Tensor], optional
        Initial tensors keyed by name.

    Examples
    --------
    >>> tn = TensorNetwork()
    >>> tn.add("A", Tensor(np.eye(2, dtype=complex), ["i", "j"]))
    >>> tn.add("B", Tensor(np.ones((2, 3), dtype=complex), ["j", "k"]))
    >>> result = tn.contract_all()
    >>> result.shape
    (2, 3)
    """

    def __init__(self, tensors: Optional[dict] = None) -> None:
        self._tensors: dict[str, Tensor] = dict(tensors) if tensors else {}
        self._contraction_order: List[Tuple[str, str]] = []

    def add(self, name: str, tensor: Tensor) -> None:
        """Add a tensor to the network."""
        self._tensors[name] = tensor

    def remove(self, name: str) -> Tensor:
        """Remove and return a tensor from the network."""
        return self._tensors.pop(name)

    @property
    def tensor_names(self) -> List[str]:
        return list(self._tensors.keys())

    def get(self, name: str) -> Tensor:
        return self._tensors[name]

    def set_contraction_order(self, order: List[Tuple[str, str]]) -> None:
        """Specify pairwise contraction order as list of (name_a, name_b).

        Each pair is contracted in order; the result replaces both tensors
        under the name ``name_a``.
        """
        self._contraction_order = list(order)

    def contract_all(self) -> Tensor:
        """Contract all tensors in the network.

        If a contraction order was set, uses that order. Otherwise falls
        back to greedy sequential contraction.

        Returns
        -------
        Tensor
            The fully contracted result.
        """
        if not self._tensors:
            raise ValueError("TensorNetwork is empty")

        tensors = dict(self._tensors)

        if self._contraction_order:
            for name_a, name_b in self._contraction_order:
                a = tensors.pop(name_a)
                b = tensors.pop(name_b)
                result = contract_pair(a, b)
                tensors[name_a] = result
        else:
            # Greedy: contract in insertion order
            names = list(tensors.keys())
            result = tensors[names[0]]
            for name in names[1:]:
                result = contract_pair(result, tensors[name])
            return result

        # Should be one tensor left
        assert len(tensors) == 1, f"Expected 1 tensor, got {len(tensors)}"
        return next(iter(tensors.values()))

    def total_bond_dimension(self) -> dict:
        """Report bond dimensions for each leg across the network.

        Returns
        -------
        dict[str, int]
            Mapping from leg name to its dimension.
        """
        bonds: dict[str, int] = {}
        for t in self._tensors.values():
            for leg, size in zip(t.legs, t.shape):
                if leg in bonds and bonds[leg] != size:
                    raise ValueError(
                        f"Inconsistent dimension for leg {leg!r}: "
                        f"{bonds[leg]} vs {size}"
                    )
                bonds[leg] = size
        return bonds

    def __repr__(self) -> str:
        names = ", ".join(self._tensors.keys())
        return f"TensorNetwork([{names}])"

    def __len__(self) -> int:
        return len(self._tensors)
