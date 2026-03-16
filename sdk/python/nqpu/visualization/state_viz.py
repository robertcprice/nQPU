"""State visualization: probability distributions, Hinton diagrams, entanglement maps.

Pure-numpy implementations of common quantum-state visualizations rendered
as ASCII art.  No matplotlib required.

Provides:
  - ProbabilityDisplay: bar charts and histograms from statevectors or counts
  - HintonDiagram: matrix magnitude visualization with block characters
  - EntanglementMap: pairwise entanglement heatmaps (concurrence, mutual info)
  - Convenience functions for one-shot rendering

Example
-------
>>> import numpy as np
>>> from nqpu.visualization.state_viz import probability_bar_chart
>>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
>>> print(probability_bar_chart(bell, n_qubits=2))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _partial_trace_single(
    rho: np.ndarray, n_qubits: int, keep: int
) -> np.ndarray:
    """Trace out all qubits except *keep* from a density matrix.

    Parameters
    ----------
    rho : np.ndarray
        ``2**n_qubits x 2**n_qubits`` density matrix.
    n_qubits : int
        Total number of qubits.
    keep : int
        Index of the qubit to keep (0-indexed).

    Returns
    -------
    np.ndarray
        ``2x2`` reduced density matrix.
    """
    dim = 2**n_qubits
    rho = np.asarray(rho, dtype=complex).reshape((dim, dim))
    # Reshape into tensor with 2 indices per qubit
    shape = [2] * (2 * n_qubits)
    tensor = rho.reshape(shape)
    # Contract all indices except 'keep' (bra) and 'keep + n_qubits' (ket)
    # axes_to_trace: all qubit indices in the first group except 'keep',
    # paired with corresponding second-group index
    axes_first = list(range(n_qubits))
    axes_second = list(range(n_qubits, 2 * n_qubits))
    # Remove the kept qubit from trace-out sets
    trace_first = [a for a in axes_first if a != keep]
    trace_second = [a + n_qubits for a in axes_first if a != keep]
    # Trace pairs: contract trace_first[i] with trace_second[i]
    # Use np.einsum via index manipulation
    result = tensor
    # Sort in descending order so removal indices stay stable
    for f, s in sorted(zip(trace_first, trace_second), reverse=True):
        result = np.trace(result, axis1=f, axis2=s)
    return result.reshape((2, 2))


def _partial_trace_pair(
    rho: np.ndarray, n_qubits: int, keep_a: int, keep_b: int
) -> np.ndarray:
    """Trace out all qubits except *keep_a* and *keep_b*.

    Returns a 4x4 density matrix for the two-qubit subsystem.
    """
    dim = 2**n_qubits
    rho = np.asarray(rho, dtype=complex).reshape((dim, dim))
    kept = sorted([keep_a, keep_b])
    traced = [q for q in range(n_qubits) if q not in kept]

    # Build einsum string
    # Input indices: i0 i1 ... i_{n-1} j0 j1 ... j_{n-1}
    # For traced qubits, set i_k = j_k and sum
    # For kept qubits, keep both i_k and j_k as free indices
    # Result ordering: kept qubits in order

    idx_in_bra = list(range(n_qubits))
    idx_in_ket = list(range(n_qubits, 2 * n_qubits))

    shape = [2] * (2 * n_qubits)
    tensor = rho.reshape(shape)

    # Trace by contracting pairs
    result = tensor
    for t in sorted(traced, reverse=True):
        # After previous traces, indices have shifted -- recompute
        # It's simpler to do sequential traces
        cur_n = result.ndim // 2
        # Find where t maps in current tensor
        remaining = sorted(
            [q for q in range(n_qubits) if q not in traced[:n_qubits]]
            + [q for q in traced if q > t],
            key=lambda x: x,
        )
        # Use direct approach: trace axis t and axis t + cur_n
        result = np.trace(result, axis1=t - sum(1 for x in traced if x < t),
                          axis2=t - sum(1 for x in traced if x < t) + cur_n)
        cur_n -= 1

    return result.reshape((4, 4))


def _statevector_to_density(state: np.ndarray) -> np.ndarray:
    """Convert state vector to density matrix."""
    sv = np.asarray(state, dtype=complex).ravel()
    return np.outer(sv, sv.conj())


def _concurrence_2qubit(rho_4x4: np.ndarray) -> float:
    """Wootters concurrence for a 2-qubit density matrix."""
    rho = np.asarray(rho_4x4, dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    yy = np.kron(sigma_y, sigma_y)
    rho_tilde = yy @ rho.conj() @ yy
    product = rho @ rho_tilde
    eigvals = np.sort(np.real(np.linalg.eigvals(product)))[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    lambdas = np.sqrt(eigvals)
    c = max(0.0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3])
    return float(c)


def _von_neumann_entropy(rho: np.ndarray) -> float:
    """Von Neumann entropy S = -Tr(rho log2 rho)."""
    eigvals = np.real(np.linalg.eigvalsh(rho))
    eigvals = eigvals[eigvals > 1e-15]
    return float(-np.sum(eigvals * np.log2(eigvals)))


def _mutual_information_pair(
    rho_full: np.ndarray, n_qubits: int, qa: int, qb: int
) -> float:
    """Quantum mutual information I(A:B) = S(A) + S(B) - S(AB)."""
    rho_a = _partial_trace_single(rho_full, n_qubits, qa)
    rho_b = _partial_trace_single(rho_full, n_qubits, qb)
    rho_ab = _partial_trace_pair(rho_full, n_qubits, qa, qb)
    sa = _von_neumann_entropy(rho_a)
    sb = _von_neumann_entropy(rho_b)
    sab = _von_neumann_entropy(rho_ab)
    return max(0.0, sa + sb - sab)


# ---------------------------------------------------------------------------
# ProbabilityDisplay
# ---------------------------------------------------------------------------


@dataclass
class ProbabilityDisplay:
    """ASCII bar chart of measurement probabilities.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (determines basis labels).
    width : int
        Maximum bar width in characters.
    """

    n_qubits: int
    width: int = 40

    def bar_chart(
        self, state: np.ndarray, top_k: Optional[int] = None
    ) -> str:
        """Horizontal ASCII bar chart of state-vector probabilities.

        Parameters
        ----------
        state : np.ndarray
            State vector of length ``2**n_qubits``.
        top_k : int, optional
            Show only the top-k highest-probability basis states.
        """
        sv = np.asarray(state, dtype=complex).ravel()
        dim = 2**self.n_qubits
        if len(sv) != dim:
            raise ValueError(
                f"State length {len(sv)} != 2^{self.n_qubits}={dim}"
            )
        probs = np.abs(sv) ** 2

        indices = list(range(dim))
        if top_k is not None:
            indices = sorted(indices, key=lambda i: probs[i], reverse=True)[
                :top_k
            ]
        else:
            # Filter out zero probabilities
            indices = [i for i in indices if probs[i] > 1e-12]
            indices.sort(key=lambda i: probs[i], reverse=True)

        if not indices:
            return "All probabilities are zero."

        max_prob = max(probs[i] for i in indices)
        lines: List[str] = []
        for i in indices:
            label = f"|{format(i, f'0{self.n_qubits}b')}>"
            p = probs[i]
            bar_len = (
                int(self.width * p / max_prob) if max_prob > 0 else 0
            )
            bar = "#" * bar_len
            lines.append(f"  {label} {p:7.4f} |{bar}")
        return "\n".join(lines)

    def histogram(self, counts: dict, width: int = 40) -> str:
        """ASCII histogram from measurement counts dict.

        Parameters
        ----------
        counts : dict
            Mapping ``basis_label -> count``.
        width : int
            Maximum bar width.
        """
        if not counts:
            return "No counts."
        total = sum(counts.values())
        sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        max_count = max(counts.values())
        lines: List[str] = []
        max_label = max(len(str(k)) for k in counts)
        for label, count in sorted_items:
            prob = count / total if total > 0 else 0.0
            bar_len = int(width * count / max_count) if max_count > 0 else 0
            bar = "#" * bar_len
            lines.append(
                f"  {str(label):>{max_label}} : {count:>6} ({prob:6.2%}) |{bar}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hinton Diagram
# ---------------------------------------------------------------------------

_BLOCKS = [" ", "\u2591", "\u2592", "\u2593", "\u2588"]  # light -> full block


@dataclass
class HintonDiagram:
    """Hinton diagram for density matrix visualization.

    Displays matrix elements as characters whose "weight" is proportional
    to the magnitude of each entry.  Positive real parts use ``+`` style
    blocks; negative use ``-`` style.
    """

    max_width: int = 60

    def render(self, matrix: np.ndarray, label: str = "") -> str:
        """ASCII Hinton diagram.

        Parameters
        ----------
        matrix : np.ndarray
            2-D matrix (real or complex).
        label : str
            Optional title.
        """
        m = np.asarray(matrix)
        if m.ndim != 2:
            raise ValueError(f"Expected 2-D matrix, got ndim={m.ndim}")
        rows, cols = m.shape
        mag = np.abs(m)
        max_mag = np.max(mag) if mag.size > 0 else 1.0
        if max_mag < 1e-15:
            max_mag = 1.0

        # Choose block characters based on magnitude
        lines: List[str] = []
        if label:
            lines.append(label)
            lines.append("=" * min(len(label), self.max_width))

        # Column header
        col_labels = [f"{c:>3}" for c in range(cols)]
        lines.append("     " + " ".join(col_labels))

        for r in range(rows):
            row_str = f"{r:>3}: "
            cells: List[str] = []
            for c in range(cols):
                val = m[r, c]
                magnitude = abs(val)
                frac = magnitude / max_mag
                # Pick block level (0-4)
                level = min(int(frac * 4.999), 4)
                block = _BLOCKS[level]
                # Sign indicator
                if np.isrealobj(m):
                    sign = "+" if val >= 0 else "-"
                else:
                    sign = "+" if val.real >= 0 else "-"
                cells.append(f"{sign}{block} ")
            row_str += "".join(cells)
            lines.append(row_str)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entanglement Map
# ---------------------------------------------------------------------------


@dataclass
class EntanglementMap:
    """Visualize entanglement structure of a multi-qubit state.

    Computes pairwise entanglement measures and renders them as
    ASCII heatmaps.
    """

    def concurrence_map(self, state: np.ndarray, n_qubits: int) -> str:
        """ASCII heatmap of pairwise concurrence.

        Parameters
        ----------
        state : np.ndarray
            State vector of length ``2**n_qubits``.
        n_qubits : int
            Number of qubits.

        Returns
        -------
        str
            Labelled heatmap string.
        """
        rho = _statevector_to_density(state)
        values = np.zeros((n_qubits, n_qubits))
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                rho_ij = _partial_trace_pair(rho, n_qubits, i, j)
                c = _concurrence_2qubit(rho_ij)
                values[i, j] = c
                values[j, i] = c
        return self._render_heatmap(values, "Concurrence", n_qubits)

    def mutual_info_map(self, state: np.ndarray, n_qubits: int) -> str:
        """ASCII heatmap of pairwise quantum mutual information.

        Parameters
        ----------
        state : np.ndarray
            State vector of length ``2**n_qubits``.
        n_qubits : int
            Number of qubits.

        Returns
        -------
        str
            Labelled heatmap string.
        """
        rho = _statevector_to_density(state)
        values = np.zeros((n_qubits, n_qubits))
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                mi = _mutual_information_pair(rho, n_qubits, i, j)
                values[i, j] = mi
                values[j, i] = mi
        return self._render_heatmap(values, "Mutual Information", n_qubits)

    @staticmethod
    def _render_heatmap(
        values: np.ndarray, title: str, n_qubits: int
    ) -> str:
        """Render a symmetric matrix as an ASCII heatmap.

        Uses block characters to represent magnitudes.
        """
        heat_chars = " .:oO@#"
        max_val = np.max(values) if values.size > 0 else 1.0
        if max_val < 1e-15:
            max_val = 1.0

        lines: List[str] = [title]
        # Column header
        header = "     " + "  ".join(f"q{j}" for j in range(n_qubits))
        lines.append(header)

        for i in range(n_qubits):
            row = f"q{i}:  "
            cells: List[str] = []
            for j in range(n_qubits):
                if i == j:
                    cells.append(" . ")
                else:
                    frac = values[i, j] / max_val
                    idx = min(int(frac * (len(heat_chars) - 1)), len(heat_chars) - 1)
                    cells.append(f" {heat_chars[idx]} ")
            row += "".join(cells)
            lines.append(row)

        # Legend
        lines.append("")
        legend_parts = []
        for k, ch in enumerate(heat_chars):
            val = k / (len(heat_chars) - 1) * max_val
            legend_parts.append(f"'{ch}'={val:.2f}")
        lines.append("Legend: " + ", ".join(legend_parts))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def probability_bar_chart(
    state: np.ndarray, n_qubits: int, top_k: int = 16
) -> str:
    """One-shot probability bar chart from a statevector.

    Parameters
    ----------
    state : np.ndarray
        State vector.
    n_qubits : int
        Number of qubits.
    top_k : int
        Show only the top-k states.
    """
    return ProbabilityDisplay(n_qubits=n_qubits).bar_chart(state, top_k=top_k)


def state_table(state: np.ndarray, n_qubits: int) -> str:
    """Tabular display of all basis-state amplitudes and probabilities.

    Parameters
    ----------
    state : np.ndarray
        State vector.
    n_qubits : int
        Number of qubits.
    """
    from .formatters import ASCIITable, format_complex

    sv = np.asarray(state, dtype=complex).ravel()
    dim = 2**n_qubits
    if len(sv) != dim:
        raise ValueError(f"State length {len(sv)} != 2^{n_qubits}={dim}")

    tbl = ASCIITable(
        headers=["Basis", "Amplitude", "Probability", "Phase (deg)"],
        alignments=["left", "right", "right", "right"],
    )
    for i in range(dim):
        amp = sv[i]
        prob = abs(amp) ** 2
        phase = math.degrees(math.atan2(amp.imag, amp.real)) if abs(amp) > 1e-12 else 0.0
        basis = f"|{format(i, f'0{n_qubits}b')}>"
        tbl.add_row([basis, format_complex(amp), f"{prob:.6f}", f"{phase:.1f}"])
    return tbl.render()


def density_matrix_display(rho: np.ndarray) -> str:
    """Render a density matrix using a Hinton diagram.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix.
    """
    return HintonDiagram().render(rho, label="Density Matrix")
