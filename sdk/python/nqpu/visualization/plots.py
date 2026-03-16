"""Optional matplotlib wrappers with graceful degradation.

Every plotting function checks for matplotlib availability and raises
a clear ``ImportError`` directing the user to the equivalent pure-ASCII
function when matplotlib is not installed.

This module is safe to import even without matplotlib -- the import
failure is caught at module level.

Example
-------
>>> from nqpu.visualization.plots import QuantumPlotter
>>> plotter = QuantumPlotter()
>>> plotter.plot_probabilities(state, n_qubits=3)  # needs matplotlib
>>> # ASCII fallback:
>>> from nqpu.visualization.state_viz import probability_bar_chart
>>> print(probability_bar_chart(state, 3))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _require_mpl(func_name: str, ascii_alt: str) -> None:
    """Raise ImportError if matplotlib is missing."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            f"matplotlib is required for {func_name}(). "
            f"Install it with `pip install matplotlib`, "
            f"or use the ASCII alternative: {ascii_alt}"
        )


# ---------------------------------------------------------------------------
# QuantumPlotter
# ---------------------------------------------------------------------------


@dataclass
class QuantumPlotter:
    """High-level plotting utilities for quantum data.

    All methods require matplotlib.  If matplotlib is unavailable, they
    raise ``ImportError`` with guidance toward ASCII alternatives.
    """

    figsize: Tuple[float, float] = (8, 5)
    style: Optional[str] = None

    def _apply_style(self) -> None:
        if self.style and HAS_MATPLOTLIB:
            plt.style.use(self.style)

    def plot_probabilities(
        self,
        state: np.ndarray,
        n_qubits: int,
        title: str = "State Probabilities",
        ax: Any = None,
        top_k: Optional[int] = None,
        color: str = "steelblue",
    ) -> Any:
        """Plot measurement probability distribution as a bar chart.

        Parameters
        ----------
        state : np.ndarray
            State vector.
        n_qubits : int
            Number of qubits.
        title : str
            Plot title.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on.
        top_k : int, optional
            Show only the top-k states.
        color : str
            Bar color.

        Returns
        -------
        matplotlib.axes.Axes
        """
        _require_mpl("plot_probabilities", "probability_bar_chart()")
        self._apply_style()

        sv = np.asarray(state, dtype=complex).ravel()
        dim = 2**n_qubits
        probs = np.abs(sv[:dim]) ** 2

        indices = np.arange(dim)
        if top_k is not None:
            order = np.argsort(probs)[::-1][:top_k]
            indices = order
            probs = probs[order]

        labels = [f"|{format(i, f'0{n_qubits}b')}>" for i in indices]

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        ax.bar(range(len(labels)), probs if top_k else probs[indices], color=color)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45 if n_qubits > 3 else 0, ha="right")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        return ax

    def plot_bloch(
        self,
        vectors: list,
        title: str = "Bloch Sphere",
        ax: Any = None,
        colors: Optional[List[str]] = None,
    ) -> Any:
        """Plot Bloch vectors on a 3-D sphere.

        Parameters
        ----------
        vectors : list
            List of ``BlochVector`` objects (or (x, y, z) tuples).
        title : str
            Plot title.
        ax : mpl_toolkits.mplot3d.Axes3D, optional
            Existing 3-D axes.
        colors : list of str, optional
            Per-vector colors.

        Returns
        -------
        mpl_toolkits.mplot3d.Axes3D
        """
        _require_mpl("plot_bloch", "ASCIIBlochSphere().render()")
        self._apply_style()

        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")

        # Draw wireframe sphere
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, alpha=0.1, color="gray")

        # Axes
        ax.plot([-1, 1], [0, 0], [0, 0], "k-", alpha=0.3)
        ax.plot([0, 0], [-1, 1], [0, 0], "k-", alpha=0.3)
        ax.plot([0, 0], [0, 0], [-1, 1], "k-", alpha=0.3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        default_colors = ["red", "blue", "green", "orange", "purple"]
        for idx, vec in enumerate(vectors):
            if hasattr(vec, "x"):
                bx, by, bz = vec.x, vec.y, vec.z
            else:
                bx, by, bz = vec[0], vec[1], vec[2]
            c = (colors[idx] if colors else default_colors[idx % len(default_colors)])
            ax.quiver(0, 0, 0, bx, by, bz, color=c, arrow_length_ratio=0.1)

        ax.set_title(title)
        return ax

    def plot_energy_convergence(
        self,
        energies: list,
        title: str = "Energy Convergence",
        ax: Any = None,
        reference: Optional[float] = None,
    ) -> Any:
        """Plot VQE/QAOA energy convergence over iterations.

        Parameters
        ----------
        energies : list
            Energy values per iteration.
        title : str
            Plot title.
        ax : matplotlib.axes.Axes, optional
            Existing axes.
        reference : float, optional
            Exact ground-state energy (drawn as horizontal dashed line).

        Returns
        -------
        matplotlib.axes.Axes
        """
        _require_mpl("plot_energy_convergence", "ResultFormatter.format_energy()")
        self._apply_style()

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(energies, "b-o", markersize=3, label="Energy")
        if reference is not None:
            ax.axhline(y=reference, color="r", linestyle="--", label=f"Exact = {reference:.4f}")
            ax.legend()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        ax.set_title(title)
        return ax

    def plot_circuit(
        self,
        n_qubits: int,
        gates: list,
        ax: Any = None,
        title: str = "Quantum Circuit",
    ) -> Any:
        """Plot a quantum circuit as a graphical diagram.

        Parameters
        ----------
        n_qubits : int
            Number of qubit wires.
        gates : list
            Gate list ``(name, qubits, params)``.
        ax : matplotlib.axes.Axes, optional
            Existing axes.
        title : str
            Plot title.

        Returns
        -------
        matplotlib.axes.Axes
        """
        _require_mpl("plot_circuit", "draw_circuit()")
        self._apply_style()

        if ax is None:
            width = max(8, len(gates) * 1.2 + 2)
            fig, ax = plt.subplots(figsize=(width, n_qubits * 0.8 + 1))

        # Draw wires
        total_cols = len(gates) + 1
        for q in range(n_qubits):
            ax.plot([0, total_cols], [q, q], "k-", linewidth=0.8)
            ax.text(-0.5, q, f"q{q}", ha="right", va="center", fontsize=10)

        # Draw gates
        for col_idx, gate in enumerate(gates):
            gname = gate[0]
            qubits = gate[1]
            params = gate[2] if len(gate) > 2 else []
            x = col_idx + 0.5

            name_upper = gname.upper()
            if name_upper in ("CNOT", "CX"):
                ctrl, tgt = qubits[0], qubits[1]
                ax.plot([x, x], [ctrl, tgt], "k-", linewidth=1.5)
                ax.plot(x, ctrl, "ko", markersize=8)
                ax.plot(x, tgt, "ko", markersize=12, fillstyle="none")
                ax.plot([x - 0.08, x + 0.08], [tgt, tgt], "k-", linewidth=1.5)
                ax.plot([x, x], [tgt - 0.08, tgt + 0.08], "k-", linewidth=1.5)
            elif name_upper == "SWAP":
                q0, q1 = qubits[0], qubits[1]
                ax.plot([x, x], [q0, q1], "k-", linewidth=1.5)
                for qq in [q0, q1]:
                    ax.plot([x - 0.1, x + 0.1], [qq - 0.1, qq + 0.1], "k-", linewidth=1.5)
                    ax.plot([x - 0.1, x + 0.1], [qq + 0.1, qq - 0.1], "k-", linewidth=1.5)
            elif name_upper in ("M", "MEASURE"):
                for qq in qubits:
                    ax.add_patch(plt.Rectangle((x - 0.2, qq - 0.25), 0.4, 0.5,
                                               fill=True, facecolor="white",
                                               edgecolor="black"))
                    ax.text(x, qq, "M", ha="center", va="center", fontsize=9, fontweight="bold")
            else:
                # Generic gate box
                if params:
                    param_str = ",".join(f"{p:.2g}" for p in params)
                    label = f"{gname}({param_str})"
                else:
                    label = gname
                min_q = min(qubits)
                max_q = max(qubits)
                if len(qubits) > 1:
                    ax.plot([x, x], [min_q, max_q], "k-", linewidth=1.5)
                for qq in qubits:
                    ax.add_patch(plt.Rectangle((x - 0.25, qq - 0.3), 0.5, 0.6,
                                               fill=True, facecolor="lightyellow",
                                               edgecolor="black"))
                    ax.text(x, qq, label, ha="center", va="center", fontsize=7)

        ax.set_xlim(-1, total_cols + 0.5)
        ax.set_ylim(-0.5, n_qubits - 0.5)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.axis("off")
        return ax


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def plot_state(state: np.ndarray, n_qubits: int, **kwargs: Any) -> Any:
    """Plot state probabilities (requires matplotlib)."""
    return QuantumPlotter().plot_probabilities(state, n_qubits, **kwargs)


def plot_density_matrix(rho: np.ndarray, **kwargs: Any) -> Any:
    """Plot density matrix as a colour map (requires matplotlib).

    Parameters
    ----------
    rho : np.ndarray
        Density matrix.
    """
    _require_mpl("plot_density_matrix", "density_matrix_display()")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(np.abs(rho), cmap="Blues", vmin=0)
    axes[0].set_title("Magnitude")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.angle(rho), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Phase")
    plt.colorbar(im1, ax=axes[1])

    return axes
