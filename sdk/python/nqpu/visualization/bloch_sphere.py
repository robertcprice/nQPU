"""Bloch sphere visualization for single-qubit states.

Provides ``BlochVector`` for converting quantum states (statevectors or
density matrices) to Bloch coordinates, and ``ASCIIBlochSphere`` for
rendering a text-art Bloch sphere with state vectors plotted on it.

Only requires numpy -- no matplotlib needed for the ASCII renderer.

Example
-------
>>> from nqpu.visualization.bloch_sphere import bloch_from_state, ASCIIBlochSphere
>>> import numpy as np
>>> ket_plus = np.array([1, 1]) / np.sqrt(2)
>>> bv = bloch_from_state(ket_plus)
>>> print(bv)
BlochVector(x=1.0000, y=0.0000, z=0.0000)
>>> sphere = ASCIIBlochSphere(radius=8)
>>> print(sphere.render([bv]))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------

_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# BlochVector
# ---------------------------------------------------------------------------


@dataclass
class BlochVector:
    """Bloch vector coordinates for a single-qubit state.

    A pure qubit state ``|psi> = cos(theta/2)|0> + e^{i*phi}sin(theta/2)|1>``
    maps to the point ``(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))``
    on the unit sphere.  Mixed states lie inside the sphere.

    Attributes
    ----------
    x, y, z : float
        Cartesian Bloch coordinates.
    """

    x: float
    y: float
    z: float

    @staticmethod
    def from_statevector(state: np.ndarray) -> "BlochVector":
        """Extract Bloch vector from a 2-element statevector ``[alpha, beta]``.

        Parameters
        ----------
        state : np.ndarray
            Complex array of shape ``(2,)`` or ``(2, 1)``.

        Returns
        -------
        BlochVector
        """
        sv = np.asarray(state, dtype=complex).ravel()
        if len(sv) != 2:
            raise ValueError(
                f"Expected 2-element state vector, got length {len(sv)}"
            )
        # Normalise
        norm = np.linalg.norm(sv)
        if norm < 1e-15:
            raise ValueError("State vector has zero norm")
        sv = sv / norm
        rho = np.outer(sv, sv.conj())
        return BlochVector.from_density_matrix(rho)

    @staticmethod
    def from_density_matrix(rho: np.ndarray) -> "BlochVector":
        """Extract Bloch vector from a 2x2 density matrix.

        Parameters
        ----------
        rho : np.ndarray
            Hermitian 2x2 density matrix.

        Returns
        -------
        BlochVector
        """
        rho = np.asarray(rho, dtype=complex)
        if rho.shape != (2, 2):
            raise ValueError(
                f"Expected 2x2 density matrix, got shape {rho.shape}"
            )
        bx = float(np.real(np.trace(rho @ _PAULI_X)))
        by = float(np.real(np.trace(rho @ _PAULI_Y)))
        bz = float(np.real(np.trace(rho @ _PAULI_Z)))
        return BlochVector(x=bx, y=by, z=bz)

    # -- Spherical coordinates -----------------------------------------------

    @property
    def theta(self) -> float:
        """Polar angle in ``[0, pi]``."""
        r = self.norm
        if r < 1e-15:
            return 0.0
        return float(math.acos(max(-1.0, min(1.0, self.z / r))))

    @property
    def phi(self) -> float:
        """Azimuthal angle in ``(-pi, pi]``."""
        return float(math.atan2(self.y, self.x))

    @property
    def norm(self) -> float:
        """Length of the Bloch vector (1 for pure, <1 for mixed)."""
        return float(math.sqrt(self.x**2 + self.y**2 + self.z**2))

    @property
    def purity(self) -> float:
        """State purity ``(1 + |r|^2) / 2``."""
        return (1.0 + self.norm**2) / 2.0

    def __repr__(self) -> str:
        return (
            f"BlochVector(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"
        )


# ---------------------------------------------------------------------------
# ASCII Bloch Sphere
# ---------------------------------------------------------------------------


@dataclass
class ASCIIBlochSphere:
    """Render a Bloch sphere as ASCII art.

    The projection shows the X-Z plane by default: X is horizontal
    (left-right) and Z is vertical (bottom-top), with the Y axis
    coming "out of the screen".  Points with non-zero Y are projected
    into the X-Z plane and shown slightly dimmer.

    Parameters
    ----------
    radius : int
        Radius of the sphere in character cells (height/width ~2*radius).
    show_axes : bool
        Whether to draw axis labels.
    """

    radius: int = 8
    show_axes: bool = True

    def render(self, vectors: Optional[List[BlochVector]] = None) -> str:
        """Render the sphere with optional state vectors marked.

        Parameters
        ----------
        vectors : list of BlochVector, optional
            Vectors to plot.  Each is projected onto the X-Z plane.

        Returns
        -------
        str
            Multi-line ASCII art string.
        """
        r = self.radius
        # Grid dimensions -- use 2:1 aspect ratio correction
        height = 2 * r + 1
        width = 4 * r + 1
        cx = 2 * r  # center column
        cy = r       # center row (0 = top)

        grid = [[" " for _ in range(width)] for _ in range(height)]

        self._draw_circle(grid, cx, cy, r)
        if self.show_axes:
            self._draw_axes(grid, cx, cy, r)

        if vectors:
            markers = "0123456789abcdefghijklmnopqrstuvwxyz"
            for idx, vec in enumerate(vectors):
                marker = markers[idx % len(markers)]
                self._plot_vector(grid, cx, cy, r, vec, marker)

        return "\n".join("".join(row) for row in grid)

    # -- Internal drawing primitives -----------------------------------------

    def _draw_circle(
        self, grid: list, cx: int, cy: int, r: int
    ) -> None:
        """Draw the outer circle of the Bloch sphere."""
        steps = max(200, 8 * r)
        for i in range(steps):
            angle = 2 * math.pi * i / steps
            # Map unit circle to grid with 2:1 aspect ratio
            gx = round(cx + 2 * r * math.cos(angle))
            gy = round(cy - r * math.sin(angle))
            if 0 <= gy < len(grid) and 0 <= gx < len(grid[0]):
                if grid[gy][gx] == " ":
                    grid[gy][gx] = "."

        # Equator ellipse (Y axis in/out of screen)
        for i in range(steps):
            angle = 2 * math.pi * i / steps
            gx = round(cx + 2 * r * math.cos(angle))
            # Equator is at z=0, projected to cy, with perspective for Y
            gy_offset = round(0.3 * r * math.sin(angle))
            gy = cy + gy_offset
            if 0 <= gy < len(grid) and 0 <= gx < len(grid[0]):
                if grid[gy][gx] == " ":
                    grid[gy][gx] = "-"

    def _draw_axes(
        self, grid: list, cx: int, cy: int, r: int
    ) -> None:
        """Draw X and Z axis lines through the center."""
        height = len(grid)
        width = len(grid[0])

        # Z axis (vertical)
        for y in range(height):
            if 0 <= cx < width and grid[y][cx] == " ":
                grid[y][cx] = "|"

        # X axis (horizontal)
        for x in range(width):
            if 0 <= cy < height and grid[cy][x] == " ":
                grid[cy][x] = "-"

        # Center
        if 0 <= cy < height and 0 <= cx < width:
            grid[cy][cx] = "+"

        # Axis labels
        if cy - r - 1 >= 0 and 0 <= cx < width:
            grid[cy - r - 1][cx] = "Z"
        if cy + r + 1 < height:
            pass  # no room typically
        if 0 <= cy < height and cx + 2 * r + 2 < width:
            grid[cy][cx + 2 * r + 2] = "X"
        if 0 <= cy < height and cx - 2 * r - 2 >= 0:
            grid[cy][cx - 2 * r - 2] = " "

    def _plot_vector(
        self,
        grid: list,
        cx: int,
        cy: int,
        r: int,
        vec: BlochVector,
        marker: str = "*",
    ) -> None:
        """Plot a single Bloch vector onto the grid.

        Projects onto the X-Z plane (Y axis depth).

        Parameters
        ----------
        vec : BlochVector
            The vector to project.
        marker : str
            Single character used as the marker.
        """
        # Clamp to unit sphere for display
        vnorm = vec.norm
        scale = min(vnorm, 1.0)
        if vnorm > 1e-15:
            bx = vec.x / vnorm * scale
            bz = vec.z / vnorm * scale
        else:
            bx, bz = 0.0, 0.0

        gx = round(cx + 2 * r * bx)
        gy = round(cy - r * bz)

        height = len(grid)
        width = len(grid[0])
        if 0 <= gy < height and 0 <= gx < width:
            grid[gy][gx] = marker


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def bloch_from_state(state: np.ndarray) -> BlochVector:
    """Create a ``BlochVector`` from a 2-element statevector.

    Parameters
    ----------
    state : np.ndarray
        Single-qubit statevector ``[alpha, beta]``.
    """
    return BlochVector.from_statevector(state)


def bloch_from_angles(theta: float, phi: float) -> BlochVector:
    """Create a ``BlochVector`` from spherical coordinates.

    Parameters
    ----------
    theta : float
        Polar angle in ``[0, pi]``.
    phi : float
        Azimuthal angle in ``(-pi, pi]``.
    """
    return BlochVector(
        x=math.sin(theta) * math.cos(phi),
        y=math.sin(theta) * math.sin(phi),
        z=math.cos(theta),
    )


def bloch_trajectory(states: list) -> List[BlochVector]:
    """Convert a sequence of statevectors to a Bloch trajectory.

    Parameters
    ----------
    states : list
        Sequence of 2-element statevectors.

    Returns
    -------
    List[BlochVector]
        Corresponding list of Bloch vectors.
    """
    return [BlochVector.from_statevector(np.asarray(s)) for s in states]
