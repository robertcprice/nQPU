"""ASCII quantum circuit drawing -- no external dependencies.

Renders quantum circuits as text art suitable for terminal display
and log files.  Supports single- and multi-qubit gates, parameterized
gates, SWAP, controlled gates (with vertical connection lines), and
measurement symbols.

Gate specification format
-------------------------
Each gate is a tuple ``(name, qubits, params)`` where:

- *name*: gate identifier string (``"H"``, ``"CNOT"``, ``"Rz"``, ``"SWAP"``, ``"M"``, ...)
- *qubits*: list of qubit indices the gate acts on
- *params*: list of float parameters (may be empty)

Two styles:

- ``"unicode"`` (default): Uses box-drawing characters for a cleaner look.
- ``"ascii"``: Pure 7-bit ASCII, safe for any terminal/log transport.

Example
-------
>>> from nqpu.visualization.circuit_drawing import draw_circuit
>>> print(draw_circuit(2, [("H", [0], []), ("CNOT", [0, 1], [])]))
q0: --H----*---
           |
q1: -------X---
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi


def _angle_str(radians: float) -> str:
    """Human-friendly angle representation, e.g. pi, pi/2, 0.42."""
    if abs(radians - _PI) < 1e-8:
        return "pi"
    if abs(radians + _PI) < 1e-8:
        return "-pi"
    if abs(radians) < 1e-10:
        return "0"
    for denom in (2, 3, 4, 6, 8):
        for numer in range(-8 * denom, 8 * denom + 1):
            if numer == 0:
                continue
            if abs(radians - numer * _PI / denom) < 1e-8:
                if numer == 1:
                    return f"pi/{denom}"
                if numer == -1:
                    return f"-pi/{denom}"
                return f"{numer}pi/{denom}"
    return f"{radians:.4g}"


# ---------------------------------------------------------------------------
# Style tables
# ---------------------------------------------------------------------------

_STYLES = {
    "unicode": {
        "wire": "\u2500",
        "cross": "\u253c",
        "vline": "\u2502",
        "control": "\u25cf",
        "target_x": "\u2295",
        "swap_x": "\u2573",
        "measure": "M",
        "space": " ",
    },
    "ascii": {
        "wire": "-",
        "cross": "+",
        "vline": "|",
        "control": "*",
        "target_x": "X",
        "swap_x": "x",
        "measure": "M",
        "space": " ",
    },
}


# ---------------------------------------------------------------------------
# CircuitGlyph
# ---------------------------------------------------------------------------


@dataclass
class CircuitGlyph:
    """Visual representation of a single gate in the circuit diagram.

    Attributes
    ----------
    name : str
        Gate identifier (H, CNOT, Rz, ...).
    qubits : list
        Qubit indices the gate acts on.
    params : list
        Numerical parameters (angles, etc.).
    width : int
        Computed display width (set during rendering).
    """

    name: str
    qubits: list
    params: list = field(default_factory=list)
    width: int = 0

    def label(self) -> str:
        """Compact gate label including parameters."""
        if self.name.upper() in ("CNOT", "CX", "CZ", "SWAP", "M", "MEASURE"):
            return self.name.upper()
        if self.params:
            param_str = ",".join(_angle_str(p) for p in self.params)
            return f"{self.name}({param_str})"
        return self.name

    def render(self, style: str = "unicode") -> Dict[int, str]:
        """Return mapping qubit_index -> character(s) for this gate.

        Multi-qubit gates produce entries for every qubit in range
        (including intermediary wires that need vertical connection lines).
        """
        s = _STYLES.get(style, _STYLES["unicode"])
        name_upper = self.name.upper()
        result: Dict[int, str] = {}

        if name_upper in ("M", "MEASURE"):
            for q in self.qubits:
                result[q] = s["measure"]
            self.width = max(len(v) for v in result.values())
            return result

        if name_upper in ("CNOT", "CX"):
            ctrl, tgt = self.qubits[0], self.qubits[1]
            result[ctrl] = s["control"]
            result[tgt] = s["target_x"]
            self.width = 1
            return result

        if name_upper == "CZ":
            ctrl, tgt = self.qubits[0], self.qubits[1]
            result[ctrl] = s["control"]
            result[tgt] = s["control"]
            self.width = 1
            return result

        if name_upper == "SWAP":
            q0, q1 = self.qubits[0], self.qubits[1]
            result[q0] = s["swap_x"]
            result[q1] = s["swap_x"]
            self.width = 1
            return result

        if name_upper == "TOFFOLI" or name_upper == "CCX":
            ctrls = self.qubits[:-1]
            tgt = self.qubits[-1]
            for c in ctrls:
                result[c] = s["control"]
            result[tgt] = s["target_x"]
            self.width = 1
            return result

        if name_upper == "FREDKIN" or name_upper == "CSWAP":
            ctrl = self.qubits[0]
            t0, t1 = self.qubits[1], self.qubits[2]
            result[ctrl] = s["control"]
            result[t0] = s["swap_x"]
            result[t1] = s["swap_x"]
            self.width = 1
            return result

        # Generic controlled gates: C-<name> or controlled_<name>
        if (name_upper.startswith("C") and len(name_upper) > 1
                and name_upper[1:] not in ("X", "Z", "NOT", "SWAP")
                and len(self.qubits) >= 2):
            ctrl = self.qubits[0]
            inner_name = self.name[1:]
            if self.params:
                param_str = ",".join(_angle_str(p) for p in self.params)
                inner_label = f"{inner_name}({param_str})"
            else:
                inner_label = inner_name
            result[ctrl] = s["control"]
            for tq in self.qubits[1:]:
                result[tq] = inner_label
            self.width = max(len(v) for v in result.values())
            return result

        # Single- or multi-qubit named gate
        lbl = self.label()
        for q in self.qubits:
            result[q] = lbl
        self.width = len(lbl)
        return result


# ---------------------------------------------------------------------------
# CircuitDrawer
# ---------------------------------------------------------------------------


@dataclass
class CircuitDrawer:
    """Draw quantum circuits as ASCII art.

    Parameters
    ----------
    n_qubits : int
        Number of qubit wires.
    style : str
        ``"unicode"`` or ``"ascii"``.
    """

    n_qubits: int
    style: str = "unicode"

    def draw(self, gates: list) -> str:
        """Convert gate list to an ASCII circuit diagram string.

        Parameters
        ----------
        gates : list
            Each element is ``(name, qubits, params)`` where *qubits* is a
            list of int and *params* is a list of float.

        Returns
        -------
        str
            Multi-line string with the circuit diagram.
        """
        return "\n".join(self.draw_to_lines(gates))

    def draw_to_lines(self, gates: list) -> List[str]:
        """Return the circuit as a list of strings (one per output line).

        Output has ``2 * n_qubits - 1`` lines: qubit wires interleaved
        with spacer lines for vertical connectors.
        """
        s = _STYLES.get(self.style, _STYLES["unicode"])
        wire = s["wire"]
        vline = s["vline"]
        space = s["space"]

        glyphs = [CircuitGlyph(name=g[0], qubits=list(g[1]),
                                params=list(g[2]) if len(g) > 2 else [])
                   for g in gates]

        # Build columns -- each glyph becomes one column
        # Each column has (2*n_qubits - 1) entries: qubit rows + spacer rows
        n_lines = 2 * self.n_qubits - 1
        columns: List[List[str]] = []

        for glyph in glyphs:
            rendered = glyph.render(self.style)
            col_width = max((len(v) for v in rendered.values()), default=1)
            col_width = max(col_width, 1)

            col: List[str] = []
            # Determine vertical span for multi-qubit gates
            acted_qubits = sorted(glyph.qubits)
            if len(acted_qubits) >= 2:
                min_q, max_q = acted_qubits[0], acted_qubits[-1]
            else:
                min_q = max_q = -1

            for line_idx in range(n_lines):
                if line_idx % 2 == 0:
                    # Qubit wire line
                    q = line_idx // 2
                    if q in rendered:
                        sym = rendered[q]
                        # Pad to column width with wire chars
                        pad_total = col_width - len(sym)
                        pad_left = pad_total // 2
                        pad_right = pad_total - pad_left
                        cell = wire * pad_left + sym + wire * pad_right
                    else:
                        cell = wire * col_width
                    col.append(cell)
                else:
                    # Spacer line between qubit q_above and q_below
                    q_above = line_idx // 2
                    q_below = q_above + 1
                    if (len(acted_qubits) >= 2
                            and q_above >= min_q and q_below <= max_q):
                        # Vertical connector
                        pad_total = col_width - 1
                        pad_left = pad_total // 2
                        pad_right = pad_total - pad_left
                        cell = space * pad_left + vline + space * pad_right
                    else:
                        cell = space * col_width
                    col.append(cell)

            columns.append(col)

        # Assemble final lines
        # Prefix: qubit labels
        max_label = len(f"q{self.n_qubits - 1}")
        prefix_width = max_label + 2  # "qN: "

        result_lines: List[str] = []
        for line_idx in range(n_lines):
            if line_idx % 2 == 0:
                q = line_idx // 2
                prefix = f"q{q}: ".rjust(prefix_width)
            else:
                prefix = " " * prefix_width

            # Join columns separated by wire (for qubit lines) or space (spacer)
            parts: List[str] = []
            sep = wire if (line_idx % 2 == 0) else space
            for ci, col in enumerate(columns):
                parts.append(col[line_idx])
            body = sep.join(parts) if parts else ""
            # Add trailing wire/space
            if line_idx % 2 == 0:
                result_lines.append(prefix + wire + body + wire)
            else:
                result_lines.append(prefix + space + body + space)

        return result_lines


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def draw_circuit(n_qubits: int, gates: list, style: str = "unicode") -> str:
    """Convenience function for circuit drawing.

    Parameters
    ----------
    n_qubits : int
        Number of qubit wires.
    gates : list
        Gate list -- each element ``(name, qubits, params)``.
    style : str
        ``"unicode"`` or ``"ascii"``.

    Returns
    -------
    str
        Multi-line circuit diagram.
    """
    drawer = CircuitDrawer(n_qubits=n_qubits, style=style)
    return drawer.draw(gates)


def gate_to_ascii(gate_name: str, params: Optional[list] = None) -> str:
    """Convert a gate name (with optional params) to its ASCII symbol.

    Parameters
    ----------
    gate_name : str
        Gate identifier.
    params : list, optional
        Numerical parameters.

    Returns
    -------
    str
        ASCII symbol string.
    """
    glyph = CircuitGlyph(
        name=gate_name,
        qubits=[0],
        params=list(params) if params else [],
    )
    return glyph.label()
