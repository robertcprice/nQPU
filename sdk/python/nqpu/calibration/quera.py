"""Parse QuEra Aquila neutral atom specifications.

QuEra's Aquila is an analog quantum computer based on arrays of neutral
rubidium atoms held in optical tweezers. Qubits interact via the Rydberg
blockade mechanism: when two atoms are closer than the blockade radius
(~9 um), simultaneous Rydberg excitation is suppressed, creating an
effective entangling interaction.

This module models the atom layout geometry, blockade connectivity, and
published fidelity figures. It also provides helper functions for common
lattice geometries (square grid, Kagome, triangular) that are programmable
via the tweezer array.

Example
-------
>>> from nqpu.calibration.quera import aquila, generate_grid
>>> device = aquila()
>>> adj = device.blockade_graph()
>>> print(f"Blockade edges: {int(adj.sum()) // 2}")

References
----------
- Wurtz et al. (2023), "Aquila: QuEra's 256-qubit neutral-atom quantum computer"
- Ebadi et al. (2022), "Quantum optimization of maximum independent set"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AtomSite:
    """Single atom trap site in the tweezer array.

    Attributes
    ----------
    index : int
        Site index in the register.
    x : float
        Horizontal position in micrometers.
    y : float
        Vertical position in micrometers.
    occupied : bool
        Whether the site contains a loaded atom.
    """

    index: int
    x: float  # micrometers
    y: float  # micrometers
    occupied: bool = True


@dataclass
class NeutralAtomConfig:
    """Neutral atom processor calibration.

    Models a Rydberg-blockade neutral atom quantum computer with
    programmable atom positions, global or local drive capabilities,
    and published fidelity specifications.
    """

    name: str
    max_atoms: int
    sites: List[AtomSite] = field(default_factory=list)
    rydberg_range: float = 9.0          # micrometers (blockade radius)
    atom_loading_fidelity: float = 0.97
    measurement_fidelity: float = 0.97
    rydberg_fidelity: float = 0.975
    t1: float = 4e6                     # microseconds (~4 seconds)
    global_drive_only: bool = True      # Aquila limitation: no local addressing

    # -- Connectivity -------------------------------------------------------

    def blockade_graph(self) -> np.ndarray:
        """Compute the adjacency matrix for atoms within blockade radius.

        Returns
        -------
        np.ndarray
            Boolean adjacency matrix of shape ``(n_occupied, n_occupied)``
            where entry ``[i, j]`` is ``True`` if atoms *i* and *j* are
            within the Rydberg blockade radius.
        """
        occupied = [s for s in self.sites if s.occupied]
        n = len(occupied)
        if n == 0:
            return np.zeros((0, 0), dtype=bool)

        coords = np.array([[s.x, s.y] for s in occupied])
        # Pairwise distance matrix
        dx = coords[:, 0:1] - coords[:, 0:1].T
        dy = coords[:, 1:2] - coords[:, 1:2].T
        dist = np.sqrt(dx ** 2 + dy ** 2)

        adj = (dist < self.rydberg_range) & (dist > 0)
        return adj

    def connectivity(self) -> List[Tuple[int, int]]:
        """Return the blockade connectivity as a list of index pairs.

        Returns
        -------
        list of (int, int)
            Pairs of occupied-site indices within blockade radius.
        """
        adj = self.blockade_graph()
        edges: List[Tuple[int, int]] = []
        n = adj.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges

    @property
    def n_occupied(self) -> int:
        """Number of occupied atom sites."""
        return sum(1 for s in self.sites if s.occupied)

    # -- Fidelity estimation ------------------------------------------------

    def expected_fidelity(self, n_pulses: int) -> float:
        """Estimate experiment fidelity from number of Rydberg pulses.

        Uses a simple model:

            F = loading^n_atoms * rydberg^n_pulses * measurement^n_atoms

        Parameters
        ----------
        n_pulses : int
            Number of global Rydberg drive pulses in the program.

        Returns
        -------
        float
            Estimated experiment fidelity in [0, 1].
        """
        n_atoms = self.n_occupied
        f_load = self.atom_loading_fidelity ** n_atoms
        f_rydberg = self.rydberg_fidelity ** n_pulses
        f_meas = self.measurement_fidelity ** n_atoms
        return f_load * f_rydberg * f_meas

    def filling_fraction(self) -> float:
        """Fraction of sites that are occupied."""
        if not self.sites:
            return 0.0
        return self.n_occupied / len(self.sites)

    def summary(self) -> str:
        """Human-readable summary of the neutral atom processor."""
        n_occ = self.n_occupied
        edges = self.connectivity()
        lines = [
            f"NeutralAtomConfig: {self.name}",
            f"  Max atoms:           {self.max_atoms}",
            f"  Sites:               {len(self.sites)}",
            f"  Occupied:            {n_occ}",
            f"  Filling fraction:    {self.filling_fraction():.2%}",
            f"  Blockade radius:     {self.rydberg_range:.1f} um",
            f"  Blockade edges:      {len(edges)}",
            f"  Loading fidelity:    {self.atom_loading_fidelity:.3f}",
            f"  Rydberg fidelity:    {self.rydberg_fidelity:.4f}",
            f"  Measurement:         {self.measurement_fidelity:.3f}",
            f"  T1:                  {self.t1:.0f} us",
            f"  Global drive only:   {self.global_drive_only}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_quera_capabilities(capabilities: dict) -> NeutralAtomConfig:
    """Parse a QuEra device capabilities dictionary.

    Parameters
    ----------
    capabilities : dict
        Dictionary with keys such as ``"name"``, ``"max_atoms"``,
        ``"rydberg_range"``, ``"sites"`` (list of ``{index, x, y}``).

    Returns
    -------
    NeutralAtomConfig
    """
    sites: List[AtomSite] = []
    for s in capabilities.get("sites", []):
        sites.append(
            AtomSite(
                index=s.get("index", 0),
                x=s.get("x", 0.0),
                y=s.get("y", 0.0),
                occupied=s.get("occupied", True),
            )
        )

    return NeutralAtomConfig(
        name=capabilities.get("name", "unknown"),
        max_atoms=capabilities.get("max_atoms", 0),
        sites=sites,
        rydberg_range=capabilities.get("rydberg_range", 9.0),
        atom_loading_fidelity=capabilities.get("atom_loading_fidelity", 0.97),
        measurement_fidelity=capabilities.get("measurement_fidelity", 0.97),
        rydberg_fidelity=capabilities.get("rydberg_fidelity", 0.975),
        t1=capabilities.get("t1", 4e6),
        global_drive_only=capabilities.get("global_drive_only", True),
    )


# ---------------------------------------------------------------------------
# Preset device
# ---------------------------------------------------------------------------

def aquila() -> NeutralAtomConfig:
    """QuEra Aquila (256 atoms) preset.

    Generates a 16x16 square grid with 4 um spacing as a representative
    default layout. Real Aquila programs use custom geometries.
    """
    sites = generate_grid(16, 16, spacing=4.0)
    return NeutralAtomConfig(
        name="quera_aquila",
        max_atoms=256,
        sites=sites,
        rydberg_range=9.0,
        atom_loading_fidelity=0.97,
        measurement_fidelity=0.97,
        rydberg_fidelity=0.975,
        t1=4e6,
        global_drive_only=True,
    )


# ---------------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------------

def generate_grid(
    rows: int,
    cols: int,
    spacing: float = 5.0,
) -> List[AtomSite]:
    """Generate a square grid of atom sites.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    spacing : float
        Distance between adjacent sites in micrometers.

    Returns
    -------
    list of AtomSite
    """
    sites: List[AtomSite] = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            sites.append(
                AtomSite(
                    index=idx,
                    x=c * spacing,
                    y=r * spacing,
                    occupied=True,
                )
            )
            idx += 1
    return sites


def generate_kagome(
    n_cells: int,
    spacing: float = 5.0,
) -> List[AtomSite]:
    """Generate a Kagome lattice of atom sites.

    The Kagome lattice is formed by corner-sharing triangles and is of
    special interest for frustrated magnetism studies on neutral-atom
    quantum simulators.

    Parameters
    ----------
    n_cells : int
        Number of unit cells along each lattice direction.
    spacing : float
        Nearest-neighbour distance in micrometers.

    Returns
    -------
    list of AtomSite
    """
    sites: List[AtomSite] = []
    idx = 0

    # Kagome lattice vectors
    a1 = np.array([2.0 * spacing, 0.0])
    a2 = np.array([spacing, spacing * math.sqrt(3)])

    # Basis positions within a unit cell
    basis = [
        np.array([0.0, 0.0]),
        np.array([spacing, 0.0]),
        np.array([spacing / 2.0, spacing * math.sqrt(3) / 2.0]),
    ]

    for i in range(n_cells):
        for j in range(n_cells):
            origin = i * a1 + j * a2
            for b in basis:
                pos = origin + b
                sites.append(
                    AtomSite(
                        index=idx,
                        x=float(pos[0]),
                        y=float(pos[1]),
                        occupied=True,
                    )
                )
                idx += 1

    return sites


def generate_triangular(
    rows: int,
    cols: int,
    spacing: float = 5.0,
) -> List[AtomSite]:
    """Generate a triangular lattice of atom sites.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    spacing : float
        Nearest-neighbour distance in micrometers.

    Returns
    -------
    list of AtomSite
    """
    sites: List[AtomSite] = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            x = c * spacing + (r % 2) * spacing / 2.0
            y = r * spacing * math.sqrt(3) / 2.0
            sites.append(
                AtomSite(index=idx, x=x, y=y, occupied=True)
            )
            idx += 1
    return sites
