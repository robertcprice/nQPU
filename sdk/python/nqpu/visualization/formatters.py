"""ASCII tables, progress bars, and color formatting utilities.

Pure-Python formatting helpers for quantum computation results.
No external dependencies beyond numpy (used only for matrix formatting).

Provides:
  - ASCIITable: Flexible table rendering with alignment control
  - ProgressBar: Configurable ASCII progress indicator
  - ResultFormatter: Quantum-specific result formatting (energy, fidelity, counts)
  - Convenience functions: table(), progress_bar(), format_complex(), format_statevector()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# ASCII Table
# ---------------------------------------------------------------------------


@dataclass
class ASCIITable:
    """Format data as ASCII tables with configurable alignment.

    Parameters
    ----------
    headers : List[str]
        Column header labels.
    alignments : List[str], optional
        Per-column alignment: ``"left"``, ``"right"``, or ``"center"``.
        Defaults to left for all columns.

    Example
    -------
    >>> t = ASCIITable(["Basis", "Amplitude", "Prob"])
    >>> t.add_row(["|00>", "0.707+0j", "0.5"])
    >>> t.add_row(["|11>", "0.707+0j", "0.5"])
    >>> print(t.render())
    """

    headers: List[str]
    alignments: Optional[List[str]] = None
    _rows: List[List[str]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.alignments is None:
            self.alignments = ["left"] * len(self.headers)
        if len(self.alignments) != len(self.headers):
            raise ValueError(
                f"alignments length ({len(self.alignments)}) "
                f"!= headers length ({len(self.headers)})"
            )

    def add_row(self, row: list) -> None:
        """Append a row (auto-converted to strings)."""
        self._rows.append([str(v) for v in row])

    def render(self, rows: Optional[list] = None) -> str:
        """Render the table as a string.

        Parameters
        ----------
        rows : list, optional
            If provided, use these rows instead of internally accumulated ones.
        """
        data = [list(map(str, r)) for r in rows] if rows is not None else self._rows
        if not data and not self.headers:
            return ""

        n_cols = len(self.headers)
        # Compute column widths
        widths = [len(h) for h in self.headers]
        for row in data:
            for i, cell in enumerate(row[:n_cols]):
                widths[i] = max(widths[i], len(cell))

        def _align(text: str, width: int, align: str) -> str:
            if align == "right":
                return text.rjust(width)
            if align == "center":
                return text.center(width)
            return text.ljust(width)

        sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

        lines: List[str] = [sep]
        header_cells = [
            _align(h, widths[i], self.alignments[i])
            for i, h in enumerate(self.headers)
        ]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append(sep.replace("-", "="))

        for row in data:
            # Pad row if shorter than headers
            padded = list(row) + [""] * (n_cols - len(row))
            cells = [
                _align(padded[i], widths[i], self.alignments[i])
                for i in range(n_cols)
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Progress Bar
# ---------------------------------------------------------------------------


@dataclass
class ProgressBar:
    """ASCII progress bar for long-running computations.

    Parameters
    ----------
    total : int
        Total number of steps.
    width : int
        Character width of the bar (excluding brackets/label).
    fill_char : str
        Character used for the filled portion.
    empty_char : str
        Character used for the empty portion.
    """

    total: int
    width: int = 40
    fill_char: str = "#"
    empty_char: str = "-"

    def update(self, current: int) -> str:
        """Return progress bar string for *current* step."""
        return self.format(current)

    def format(self, current: int, label: str = "") -> str:
        """Return formatted progress bar with optional label.

        Example output::

            [##########------------------------------] 25.0% (250/1000) VQE iter
        """
        current = max(0, min(current, self.total))
        if self.total == 0:
            fraction = 1.0
        else:
            fraction = current / self.total
        filled = int(self.width * fraction)
        bar = self.fill_char * filled + self.empty_char * (self.width - filled)
        pct = f"{fraction * 100:5.1f}%"
        counts = f"({current}/{self.total})"
        parts = [f"[{bar}]", pct, counts]
        if label:
            parts.append(label)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Result Formatter
# ---------------------------------------------------------------------------


@dataclass
class ResultFormatter:
    """Format quantum computation results for human-readable display."""

    precision: int = 6

    def format_energy(self, energy: float, reference: Optional[float] = None) -> str:
        """Format energy with optional reference comparison.

        Parameters
        ----------
        energy : float
            Computed energy value (in Hartree or arbitrary units).
        reference : float, optional
            Reference energy for comparison (shows delta and relative error).
        """
        line = f"Energy: {energy:.{self.precision}f}"
        if reference is not None:
            delta = energy - reference
            if abs(reference) > 1e-15:
                rel = abs(delta / reference) * 100
                line += f"  (ref: {reference:.{self.precision}f}, delta: {delta:+.2e}, rel: {rel:.4f}%)"
            else:
                line += f"  (ref: {reference:.{self.precision}f}, delta: {delta:+.2e})"
        return line

    def format_fidelity(self, fidelity: float) -> str:
        """Format fidelity with quality indicator."""
        if fidelity >= 0.999:
            quality = "excellent"
        elif fidelity >= 0.99:
            quality = "good"
        elif fidelity >= 0.95:
            quality = "fair"
        elif fidelity >= 0.9:
            quality = "poor"
        else:
            quality = "very poor"
        return f"Fidelity: {fidelity:.{self.precision}f} ({quality})"

    def format_counts(
        self, counts: dict, total_shots: Optional[int] = None
    ) -> str:
        """Format measurement counts as aligned table.

        Parameters
        ----------
        counts : dict
            Mapping basis-state label -> count.
        total_shots : int, optional
            Total shots for probability calculation. Inferred from sum if omitted.
        """
        if not counts:
            return "No measurements."
        if total_shots is None:
            total_shots = sum(counts.values())
        # Sort by count descending
        sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        max_label = max(len(str(k)) for k in counts)
        max_count = max(len(str(v)) for v in counts.values())
        lines: List[str] = []
        for label, count in sorted_items:
            prob = count / total_shots if total_shots > 0 else 0.0
            bar_len = int(30 * prob)
            bar = "#" * bar_len
            lines.append(
                f"  {str(label):>{max_label}}: {count:>{max_count}}  "
                f"({prob:6.2%}) {bar}"
            )
        return "\n".join(lines)

    def format_matrix(
        self, matrix: np.ndarray, precision: Optional[int] = None
    ) -> str:
        """Format a numpy matrix as a readable string.

        Parameters
        ----------
        matrix : np.ndarray
            2-D array (real or complex).
        precision : int, optional
            Decimal precision (defaults to ``self.precision``).
        """
        p = precision if precision is not None else self.precision
        if np.issubdtype(matrix.dtype, np.complexfloating):
            return _format_complex_matrix(matrix, p)
        return _format_real_matrix(matrix, p)


def _format_real_matrix(m: np.ndarray, prec: int) -> str:
    rows, cols = m.shape
    cells = [[f"{m[r, c]:.{prec}f}" for c in range(cols)] for r in range(rows)]
    widths = [max(len(cells[r][c]) for r in range(rows)) for c in range(cols)]
    lines: List[str] = []
    for r in range(rows):
        line = "  ".join(cells[r][c].rjust(widths[c]) for c in range(cols))
        lines.append(f"[ {line} ]")
    return "\n".join(lines)


def _format_complex_matrix(m: np.ndarray, prec: int) -> str:
    rows, cols = m.shape
    cells = [[format_complex(m[r, c], prec) for c in range(cols)] for r in range(rows)]
    widths = [max(len(cells[r][c]) for r in range(rows)) for c in range(cols)]
    lines: List[str] = []
    for r in range(rows):
        line = "  ".join(cells[r][c].rjust(widths[c]) for c in range(cols))
        lines.append(f"[ {line} ]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def table(headers: list, rows: list, alignments: Optional[list] = None) -> str:
    """One-shot table rendering.

    Parameters
    ----------
    headers : list
        Column headers.
    rows : list
        List of row lists.
    alignments : list, optional
        Per-column alignment strings.

    Returns
    -------
    str
        Rendered ASCII table.
    """
    t = ASCIITable(headers=[str(h) for h in headers], alignments=alignments)
    return t.render(rows)


def progress_bar(
    current: int, total: int, width: int = 40, label: str = ""
) -> str:
    """One-shot progress bar rendering."""
    pb = ProgressBar(total=total, width=width)
    return pb.format(current, label=label)


def format_complex(z: complex, precision: int = 4) -> str:
    """Format a complex number in a+bj notation, suppressing near-zero parts.

    Parameters
    ----------
    z : complex
        Complex number to format.
    precision : int
        Decimal precision for real/imag parts.

    Returns
    -------
    str
        Human-readable complex number string.
    """
    re = z.real
    im = z.imag
    thr = 0.5 * 10 ** (-precision)
    re_zero = abs(re) < thr
    im_zero = abs(im) < thr
    if re_zero and im_zero:
        return f"0.{'0' * precision}"
    if im_zero:
        return f"{re:.{precision}f}"
    if re_zero:
        if abs(abs(im) - 1.0) < thr:
            return f"{'-' if im < 0 else ''}1j"
        return f"{im:.{precision}f}j"
    sign = "+" if im >= 0 else "-"
    return f"{re:.{precision}f}{sign}{abs(im):.{precision}f}j"


def format_statevector(
    state: np.ndarray, n_qubits: int, threshold: float = 0.01
) -> str:
    """Format a statevector in Dirac notation.

    Parameters
    ----------
    state : np.ndarray
        State vector of length ``2**n_qubits``.
    n_qubits : int
        Number of qubits.
    threshold : float
        Amplitude magnitude below which terms are suppressed.

    Returns
    -------
    str
        Dirac-notation representation, e.g. ``0.707|00> + 0.707|11>``.
    """
    dim = 2**n_qubits
    state = np.asarray(state).ravel()
    if len(state) != dim:
        raise ValueError(
            f"State length {len(state)} does not match 2^{n_qubits}={dim}"
        )
    terms: List[str] = []
    for i in range(dim):
        amp = state[i]
        if abs(amp) < threshold:
            continue
        basis = format(i, f"0{n_qubits}b")
        coeff = format_complex(amp, precision=4)
        terms.append(f"{coeff}|{basis}>")
    if not terms:
        return "0"
    return " + ".join(terms)
