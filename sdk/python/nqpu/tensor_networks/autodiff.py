"""Differentiable tensor network contractions via computation graph.

Provides a lightweight automatic differentiation framework for tensor
networks, enabling gradient-based optimization of MPS parameters. Each
operation records its parents and a backward function, allowing reverse
mode autodiff through arbitrary tensor contraction graphs.

Key features:
  - TensorNode: a differentiable tensor with gradient tracking
  - Differentiable contraction (tensordot), trace, and SVD
  - Reverse-mode backpropagation through the computation graph
  - High-level MPS overlap and expectation value with gradient tracking
  - Variational optimization of tensor network energy

This module enables training tensor networks via gradient descent,
complementing the DMRG approach with a more general optimization
framework useful for custom cost functions and constraints.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable


# -------------------------------------------------------------------
# TensorNode
# -------------------------------------------------------------------

@dataclass
class TensorNode:
    """Node in the computation graph holding a tensor and its gradient.

    Parameters
    ----------
    data : ndarray
        The tensor data.
    grad : ndarray, optional
        Accumulated gradient (same shape as data).
    requires_grad : bool
        Whether this node participates in gradient computation.
    _backward : callable, optional
        Function to propagate gradients to parents.
    _parents : list[TensorNode]
        Parent nodes in the computation graph.
    name : str
        Optional name for debugging.
    """
    data: np.ndarray
    grad: Optional[np.ndarray] = None
    requires_grad: bool = True
    _backward: Optional[Callable] = None
    _parents: List['TensorNode'] = field(default_factory=list)
    name: str = ""

    @property
    def shape(self) -> tuple:
        """Shape of the tensor data."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim

    def zero_grad(self):
        """Reset gradient to zero."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __repr__(self) -> str:
        return f"TensorNode(shape={self.shape}, name='{self.name}', grad={self.grad is not None})"


# -------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------

def tensor_node(data: np.ndarray, requires_grad: bool = True,
                name: str = "") -> TensorNode:
    """Create a differentiable tensor node.

    Parameters
    ----------
    data : ndarray
        Tensor data (will be copied).
    requires_grad : bool
        Whether to track gradients.
    name : str
        Optional name.

    Returns
    -------
    TensorNode
    """
    return TensorNode(
        data=np.asarray(data, dtype=np.complex128).copy(),
        requires_grad=requires_grad,
        name=name,
    )


# -------------------------------------------------------------------
# Differentiable operations
# -------------------------------------------------------------------

def contract(a: TensorNode, b: TensorNode,
             axes: Tuple) -> TensorNode:
    """Differentiable tensor contraction (like np.tensordot with grad tracking).

    Parameters
    ----------
    a, b : TensorNode
        Input tensors.
    axes : tuple
        Axes specification for np.tensordot.

    Returns
    -------
    TensorNode
        Result of contraction with backward function.
    """
    result_data = np.tensordot(a.data, b.data, axes=axes)
    result = TensorNode(
        data=result_data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _parents=[a, b],
    )

    # Normalize axes specification
    if isinstance(axes, int):
        axes_a = list(range(a.ndim - axes, a.ndim))
        axes_b = list(range(axes))
    else:
        axes_a = list(axes[0]) if not isinstance(axes[0], int) else [axes[0]]
        axes_b = list(axes[1]) if not isinstance(axes[1], int) else [axes[1]]

    def _backward():
        if result.grad is None:
            return

        # Gradient w.r.t. a:
        # result = tensordot(a, b, axes) => d_a = tensordot(grad, b*, reverse_axes)
        if a.requires_grad:
            # We need to contract result.grad with b.data over the axes
            # that came from b's non-contracted axes
            # Result axes: [a_remaining] + [b_remaining]
            a_remaining = [i for i in range(a.ndim) if i not in axes_a]
            b_remaining = [i for i in range(b.ndim) if i not in axes_b]

            n_a_remaining = len(a_remaining)
            n_b_remaining = len(b_remaining)

            # grad has shape [a_remaining_dims] + [b_remaining_dims]
            # Contract grad with conj(b) over b_remaining axes
            if n_b_remaining > 0:
                grad_axes = list(range(n_a_remaining, n_a_remaining + n_b_remaining))
                b_conj_axes = b_remaining
                da = np.tensordot(result.grad, np.conj(b.data),
                                  axes=(grad_axes, b_conj_axes))
            else:
                # All of b's axes were contracted
                da = result.grad[..., np.newaxis] * np.conj(b.data) if b.ndim > 0 else result.grad * np.conj(b.data)
                if da.ndim > a.ndim:
                    da = da.reshape(a.shape)

            # Rearrange axes to match a's original order
            # da currently has axes in order [a_remaining] + [axes_a]
            # We need to permute to match a's original axis order
            if n_b_remaining > 0:
                current_order = a_remaining + axes_a
                if sorted(current_order) == list(range(a.ndim)):
                    perm = [current_order.index(i) for i in range(a.ndim)]
                    da = da.transpose(perm)

            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += da.reshape(a.shape)

        if b.requires_grad:
            a_remaining = [i for i in range(a.ndim) if i not in axes_a]
            b_remaining = [i for i in range(b.ndim) if i not in axes_b]

            n_a_remaining = len(a_remaining)
            n_b_remaining = len(b_remaining)

            if n_a_remaining > 0:
                grad_axes = list(range(n_a_remaining))
                a_conj_axes = a_remaining
                db = np.tensordot(np.conj(a.data), result.grad,
                                  axes=(a_conj_axes, grad_axes))
            else:
                db = np.conj(a.data) * result.grad
                if db.ndim > b.ndim:
                    db = db.reshape(b.shape)

            # Rearrange to match b's axis order
            if n_a_remaining > 0:
                current_order = axes_b + b_remaining
                if sorted(current_order) == list(range(b.ndim)):
                    perm = [current_order.index(i) for i in range(b.ndim)]
                    db = db.transpose(perm)

            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            b.grad += db.reshape(b.shape)

    result._backward = _backward
    return result


def trace(a: TensorNode, axis1: int = 0, axis2: int = 1) -> TensorNode:
    """Differentiable trace over two axes.

    Parameters
    ----------
    a : TensorNode
        Input tensor.
    axis1, axis2 : int
        Axes to trace over (must have same dimension).

    Returns
    -------
    TensorNode
        Trace result.
    """
    result_data = np.trace(a.data, axis1=axis1, axis2=axis2)
    result = TensorNode(
        data=result_data,
        requires_grad=a.requires_grad,
        _parents=[a],
    )

    def _backward():
        if result.grad is None or not a.requires_grad:
            return
        if a.grad is None:
            a.grad = np.zeros_like(a.data)

        # Gradient of trace: place grad on the diagonal of the traced axes
        n = a.data.shape[axis1]
        # Build an identity-like tensor and scale by grad
        remaining_shape = result_data.shape
        grad_expanded = result.grad

        # Create the gradient contribution
        # For each element of the result, add to the diagonal of the traced axes
        idx = [slice(None)] * a.ndim
        for i in range(n):
            idx[axis1] = i
            idx[axis2] = i
            a.grad[tuple(idx)] += grad_expanded

    result._backward = _backward
    return result


def svd(a: TensorNode, full_matrices: bool = False) -> Tuple[TensorNode, TensorNode, TensorNode]:
    """Differentiable SVD decomposition.

    Parameters
    ----------
    a : TensorNode
        2D tensor node to decompose.
    full_matrices : bool
        Whether to compute full-size U, Vh.

    Returns
    -------
    U, S, Vh : TensorNode
        The SVD factors with gradient tracking.
    """
    U_data, S_data, Vh_data = np.linalg.svd(a.data, full_matrices=full_matrices)

    U = TensorNode(data=U_data, requires_grad=a.requires_grad, _parents=[a], name="U")
    S_node = TensorNode(data=S_data, requires_grad=a.requires_grad, _parents=[a], name="S")
    Vh = TensorNode(data=Vh_data, requires_grad=a.requires_grad, _parents=[a], name="Vh")

    def _backward_U():
        # SVD backward is complex; provide simplified version for real S
        pass

    def _backward_S():
        pass

    def _backward_Vh():
        pass

    # Simplified backward: propagate through the reconstruction A = U @ diag(S) @ Vh
    def _backward():
        if not a.requires_grad:
            return
        if a.grad is None:
            a.grad = np.zeros_like(a.data)

        # Reconstruct gradient contribution from U, S, Vh gradients
        # dA = dU @ diag(S) @ Vh + U @ diag(dS) @ Vh + U @ diag(S) @ dVh
        k = len(S_data)
        S_mat = np.diag(S_data)

        if U.grad is not None:
            a.grad += U.grad[:, :k] @ S_mat @ Vh_data[:k, :]

        if S_node.grad is not None:
            a.grad += U_data[:, :k] @ np.diag(S_node.grad) @ Vh_data[:k, :]

        if Vh.grad is not None:
            a.grad += U_data[:, :k] @ S_mat @ Vh.grad[:k, :]

    U._backward = _backward
    S_node._backward = None  # backward handled by U
    Vh._backward = None  # backward handled by U

    return U, S_node, Vh


# -------------------------------------------------------------------
# Backpropagation
# -------------------------------------------------------------------

def backward(node: TensorNode):
    """Backpropagate gradients through the computation graph.

    Performs a topological sort of the computation graph, then
    processes nodes in reverse order, calling each node's backward
    function to propagate gradients to its parents.

    Parameters
    ----------
    node : TensorNode
        The output node to differentiate from. Its grad should be set
        (typically to ones for a scalar output).
    """
    # Topological sort via DFS
    visited = set()
    order = []

    def _topo_sort(n: TensorNode):
        nid = id(n)
        if nid in visited:
            return
        visited.add(nid)
        for parent in n._parents:
            _topo_sort(parent)
        order.append(n)

    _topo_sort(node)

    # Initialize gradient for the output node
    if node.grad is None:
        node.grad = np.ones_like(node.data)

    # Reverse pass
    for n in reversed(order):
        if n._backward is not None:
            n._backward()


# -------------------------------------------------------------------
# High-level differentiable tensor network operations
# -------------------------------------------------------------------

@dataclass
class DifferentiableContraction:
    """High-level interface for differentiable tensor network operations.

    Provides gradient-tracked MPS overlap and expectation value
    computations for use in variational optimization.
    """

    def mps_overlap(self, mps1_tensors: List[TensorNode],
                    mps2_tensors: List[TensorNode]) -> TensorNode:
        """Compute <mps1|mps2> with gradient tracking.

        Parameters
        ----------
        mps1_tensors : list[TensorNode]
            Bra MPS tensors (will be conjugated), each shape (chi_l, d, chi_r).
        mps2_tensors : list[TensorNode]
            Ket MPS tensors, each shape (chi_l, d, chi_r).

        Returns
        -------
        TensorNode
            Scalar overlap value.
        """
        n = len(mps1_tensors)

        # Transfer matrix contraction left-to-right
        # T = sum_s conj(bra[a,s,b]) * ket[c,s,d]
        bra0_conj = tensor_node(np.conj(mps1_tensors[0].data), requires_grad=False)
        T = contract(bra0_conj, mps2_tensors[0], axes=([1], [1]))
        # T shape: (1, chi_bra_R, 1, chi_ket_R) -> contract a,c dims
        # Actually: bra0_conj(a,s,b) x ket(c,s,d) over s -> T(a,b,c,d)
        # Reshape to (b, d) by contracting a=c=0 (boundary)
        T_data = T.data.reshape(T.data.shape[0], T.data.shape[1],
                                 T.data.shape[2], T.data.shape[3])
        # For boundary: a and c are both 1-dimensional
        T = tensor_node(T_data[0, :, 0, :], requires_grad=False)

        for i in range(1, n):
            bra_conj = tensor_node(np.conj(mps1_tensors[i].data), requires_grad=False)
            # T[b, d] * conj(bra[b, s, e]) -> T_bra[d, s, e]
            T_bra = contract(T, bra_conj, axes=([0], [0]))
            # T_bra[d, s, e] * ket[d, s, f] -> T_new[e, f]
            T_new = contract(T_bra, mps2_tensors[i], axes=([0, 1], [0, 1]))
            T = T_new

        return T

    def mps_expectation(self, mps_tensors: List[TensorNode],
                        mpo_tensors: List[np.ndarray]) -> TensorNode:
        """Compute <mps|mpo|mps> with gradient tracking on mps_tensors.

        Parameters
        ----------
        mps_tensors : list[TensorNode]
            MPS site tensors with gradient tracking.
        mpo_tensors : list[ndarray]
            MPO tensors (not tracked), each shape (chi_l, d, d, chi_r).

        Returns
        -------
        TensorNode
            Scalar expectation value.
        """
        n = len(mps_tensors)

        # E[a, b, c] = identity start
        E_data = np.ones((1, 1, 1), dtype=np.complex128)

        for i in range(n):
            A = mps_tensors[i]  # TensorNode (chi_l, d, chi_r)
            W = mpo_tensors[i]  # ndarray (chi_mpo_l, d, d, chi_mpo_r)

            # E[a,b,c] * conj(A[a,s,e]) * W[b,s,t,f] * A[c,t,g] -> E'[e,f,g]
            E_new = np.einsum(
                "abc,ase,bstf,ctg->efg",
                E_data,
                np.conj(A.data),
                W,
                A.data,
            )
            E_data = E_new

        result = tensor_node(E_data.ravel(), requires_grad=False)

        # Build backward function for gradient w.r.t. MPS tensors
        # This is a simplified version: recompute environments and
        # calculate gradient for each tensor
        all_E_data = E_data.copy()

        def _backward():
            if result.grad is None:
                return
            grad_scalar = result.grad.ravel()[0]

            # Build left and right environments
            L_envs = [np.ones((1, 1, 1), dtype=np.complex128)]
            for j in range(n):
                L = np.einsum(
                    "abc,ase,bstf,ctg->efg",
                    L_envs[-1],
                    np.conj(mps_tensors[j].data),
                    mpo_tensors[j],
                    mps_tensors[j].data,
                )
                L_envs.append(L)

            R_envs = [None] * (n + 1)
            R_envs[n] = np.ones((1, 1, 1), dtype=np.complex128)
            for j in range(n - 1, -1, -1):
                R_envs[j] = np.einsum(
                    "ase,bstf,ctg,efg->abc",
                    np.conj(mps_tensors[j].data),
                    mpo_tensors[j],
                    mps_tensors[j].data,
                    R_envs[j + 1],
                )

            # Gradient for each tensor
            for j in range(n):
                if not mps_tensors[j].requires_grad:
                    continue

                A = mps_tensors[j].data
                W = mpo_tensors[j]
                L = L_envs[j]
                R = R_envs[j + 1]

                # d<H>/dA*[a,s,e] = L[a,b,c] * W[b,s,t,f] * A[c,t,g] * R[e,f,g]
                # Gradient w.r.t. A (not conj) via chain rule:
                # dE/dA[c,t,g] = conj(L[a,b,c]) * conj(W[b,s,t,f]) * ... complex
                # Simplified: gradient of real part
                grad_A = np.einsum("abc,bstf,efg->ctg", L, W, R)
                # This gives gradient of <H> w.r.t. the ket tensor

                if mps_tensors[j].grad is None:
                    mps_tensors[j].grad = np.zeros_like(A)
                mps_tensors[j].grad += np.real(grad_scalar) * np.conj(grad_A)

        result._backward = _backward
        result.data = all_E_data.ravel()
        return result


# -------------------------------------------------------------------
# Variational optimization
# -------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result of variational optimization.

    Attributes
    ----------
    energies : list[float]
        Energy at each optimization step.
    converged : bool
        Whether the optimization converged.
    n_steps : int
        Number of steps performed.
    """
    energies: List[float]
    converged: bool
    n_steps: int


@dataclass
class VariationalTN:
    """Variational optimization of tensor network using autodiff.

    Optimizes MPS tensors to minimize energy <psi|H|psi>/<psi|psi>
    using gradient descent with the autodiff framework.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent (default 0.01).
    """
    learning_rate: float = 0.01

    def optimize_energy(self, mps_tensors: List[TensorNode],
                        mpo_tensors: List[np.ndarray],
                        n_steps: int = 100,
                        tol: float = 1e-6) -> OptimizationResult:
        """Gradient descent on MPS tensors to minimize energy.

        Minimizes E = <psi|H|psi> / <psi|psi> by gradient descent
        on the MPS tensors.

        Parameters
        ----------
        mps_tensors : list[TensorNode]
            MPS site tensors to optimize.
        mpo_tensors : list[ndarray]
            MPO Hamiltonian tensors.
        n_steps : int
            Maximum number of optimization steps.
        tol : float
            Convergence tolerance on energy change.

        Returns
        -------
        OptimizationResult
        """
        energies = []
        converged = False
        dc = DifferentiableContraction()

        for step in range(n_steps):
            # Zero gradients
            for t in mps_tensors:
                t.zero_grad()

            # Compute energy
            E_node = dc.mps_expectation(mps_tensors, mpo_tensors)

            # Compute norm
            norm_sq = 0.0
            for i in range(len(mps_tensors)):
                T = np.ones((1, 1), dtype=np.complex128)
                for j in range(len(mps_tensors)):
                    T = np.einsum(
                        "ac,asb,csd->bd",
                        T,
                        np.conj(mps_tensors[j].data),
                        mps_tensors[j].data,
                    )
                norm_sq = float(np.real(T.item()))
                break

            if abs(norm_sq) < 1e-30:
                break

            energy = float(np.real(E_node.data.ravel()[0])) / norm_sq
            energies.append(energy)

            # Backward pass
            E_node.grad = np.ones_like(E_node.data)
            if E_node._backward is not None:
                E_node._backward()

            # Gradient step
            for t in mps_tensors:
                if t.requires_grad and t.grad is not None:
                    t.data -= self.learning_rate * t.grad / norm_sq

            # Renormalize MPS
            total_norm = 1.0
            for t in mps_tensors:
                total_norm *= np.linalg.norm(t.data)
            if total_norm > 1e-15:
                factor = total_norm ** (1.0 / len(mps_tensors))
                for t in mps_tensors:
                    n = np.linalg.norm(t.data)
                    if n > 1e-15:
                        t.data *= factor / n

            # Check convergence
            if len(energies) >= 2 and abs(energies[-1] - energies[-2]) < tol:
                converged = True
                break

        return OptimizationResult(
            energies=energies,
            converged=converged,
            n_steps=len(energies),
        )
