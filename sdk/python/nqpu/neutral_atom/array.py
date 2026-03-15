"""Atom array management for neutral-atom quantum computers.

Handles optical tweezer array geometries, atom loading, sorting,
rearrangement, and zone management for the full experimental cycle.

Supported geometries:
- 1D linear chains
- 2D rectangular and triangular lattices
- Custom coordinate arrays

Zone management follows the architecture used by QuEra, Atom Computing,
and Pasqal devices:
- **Storage zone**: Reservoir atoms awaiting use
- **Entangling zone**: Rydberg interaction region for multi-qubit gates
- **Readout zone**: Fluorescence imaging for qubit measurement
- **Cooling zone**: Laser cooling region for mid-circuit recooling

References:
    - Barredo et al., Science 354, 1021 (2016) [atom sorting]
    - Endres et al., Science 354, 1024 (2016) [defect-free arrays]
    - Bluvstein et al., Nature 604, 451 (2022) [reconfigurable arrays]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from .physics import AtomSpecies


class ArrayGeometry(Enum):
    """Supported atom array geometries."""

    LINEAR = auto()
    RECTANGULAR = auto()
    TRIANGULAR = auto()
    HONEYCOMB = auto()
    KAGOME = auto()
    CUSTOM = auto()


class Zone(Enum):
    """Functional zones in the atom array architecture."""

    STORAGE = auto()
    ENTANGLING = auto()
    READOUT = auto()
    COOLING = auto()


@dataclass
class ZoneConfig:
    """Configuration for a functional zone.

    Parameters
    ----------
    zone_type : Zone
        The type of zone.
    center_um : tuple[float, float]
        Center position of the zone in micrometres (x, y).
    radius_um : float
        Radius of the zone in micrometres.
    """

    zone_type: Zone
    center_um: tuple[float, float] = (0.0, 0.0)
    radius_um: float = 50.0


@dataclass
class AtomArray:
    """Neutral atom array with geometry, loading, and zone management.

    Manages the physical layout of optical tweezers and the state of
    atoms within them.  Supports deterministic loading via atom sorting
    (rearrangement from a stochastically loaded array into a defect-free
    target pattern).

    Parameters
    ----------
    n_sites : int
        Number of tweezer sites in the target array.
    species : AtomSpecies
        Atomic species loaded into the array.
    spacing_um : float
        Nearest-neighbour spacing in micrometres.
    geometry : ArrayGeometry
        Geometry type for the array layout.
    rows : int
        Number of rows for 2D geometries.  Ignored for LINEAR.
    cols : int
        Number of columns for 2D geometries.  Ignored for LINEAR.
    custom_positions : np.ndarray | None
        Custom (x, y) positions in micrometres, shape (n_sites, 2).
        Required when geometry is CUSTOM.
    zones : list[ZoneConfig]
        Functional zone definitions.
    """

    n_sites: int
    species: AtomSpecies = field(
        default_factory=lambda: AtomSpecies.RB87  # type: ignore[attr-defined]
    )
    spacing_um: float = 4.0
    geometry: ArrayGeometry = ArrayGeometry.RECTANGULAR
    rows: int = 0
    cols: int = 0
    custom_positions: np.ndarray | None = None
    zones: list[ZoneConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_sites < 1:
            raise ValueError("n_sites must be >= 1")
        if self.spacing_um <= 0:
            raise ValueError("spacing_um must be positive")

        if self.geometry == ArrayGeometry.CUSTOM:
            if self.custom_positions is None:
                raise ValueError(
                    "custom_positions required for CUSTOM geometry"
                )
            if self.custom_positions.shape != (self.n_sites, 2):
                raise ValueError(
                    f"custom_positions shape must be ({self.n_sites}, 2), "
                    f"got {self.custom_positions.shape}"
                )

        # Generate site positions
        self._positions = self._generate_positions()

        # Atom occupation: True if site is loaded, False if empty
        self._occupied = np.zeros(self.n_sites, dtype=bool)

        # Zone assignments for each site (-1 = unassigned)
        self._zone_assignments = np.full(self.n_sites, -1, dtype=int)

    # ------------------------------------------------------------------
    # Position generation
    # ------------------------------------------------------------------

    def _generate_positions(self) -> np.ndarray:
        """Generate site positions based on geometry.

        Returns
        -------
        np.ndarray, shape (n_sites, 2)
            Positions in micrometres (x, y).
        """
        if self.geometry == ArrayGeometry.CUSTOM:
            assert self.custom_positions is not None
            return self.custom_positions.copy()

        if self.geometry == ArrayGeometry.LINEAR:
            return self._generate_linear()
        elif self.geometry == ArrayGeometry.RECTANGULAR:
            return self._generate_rectangular()
        elif self.geometry == ArrayGeometry.TRIANGULAR:
            return self._generate_triangular()
        elif self.geometry == ArrayGeometry.HONEYCOMB:
            return self._generate_honeycomb()
        elif self.geometry == ArrayGeometry.KAGOME:
            return self._generate_kagome()
        else:
            raise ValueError(f"Unknown geometry: {self.geometry}")

    def _generate_linear(self) -> np.ndarray:
        """Generate 1D linear chain positions."""
        positions = np.zeros((self.n_sites, 2))
        for i in range(self.n_sites):
            positions[i, 0] = i * self.spacing_um
        # Centre the array
        positions[:, 0] -= np.mean(positions[:, 0])
        return positions

    def _generate_rectangular(self) -> np.ndarray:
        """Generate 2D rectangular lattice positions."""
        rows = self.rows if self.rows > 0 else int(math.ceil(math.sqrt(self.n_sites)))
        cols = self.cols if self.cols > 0 else int(math.ceil(self.n_sites / rows))

        positions = []
        for r in range(rows):
            for c in range(cols):
                if len(positions) >= self.n_sites:
                    break
                positions.append([c * self.spacing_um, r * self.spacing_um])
            if len(positions) >= self.n_sites:
                break

        pos = np.array(positions[:self.n_sites])
        pos -= np.mean(pos, axis=0)
        return pos

    def _generate_triangular(self) -> np.ndarray:
        """Generate 2D triangular lattice positions."""
        rows = self.rows if self.rows > 0 else int(math.ceil(math.sqrt(self.n_sites)))
        cols = self.cols if self.cols > 0 else int(math.ceil(self.n_sites / rows))

        positions = []
        for r in range(rows):
            x_offset = (r % 2) * self.spacing_um * 0.5
            y = r * self.spacing_um * math.sqrt(3) / 2.0
            for c in range(cols):
                if len(positions) >= self.n_sites:
                    break
                positions.append([c * self.spacing_um + x_offset, y])
            if len(positions) >= self.n_sites:
                break

        pos = np.array(positions[:self.n_sites])
        pos -= np.mean(pos, axis=0)
        return pos

    def _generate_honeycomb(self) -> np.ndarray:
        """Generate 2D honeycomb lattice positions."""
        # Honeycomb = triangular lattice with 2-site basis
        a = self.spacing_um
        basis = np.array([[0.0, 0.0], [a * 0.5, a * math.sqrt(3) / 6.0]])
        a1 = np.array([a, 0.0])
        a2 = np.array([a * 0.5, a * math.sqrt(3) / 2.0])

        side = int(math.ceil(math.sqrt(self.n_sites / 2)))
        positions = []
        for i in range(-side, side + 1):
            for j in range(-side, side + 1):
                for b in basis:
                    pos = i * a1 + j * a2 + b
                    positions.append(pos)
                    if len(positions) >= self.n_sites:
                        break
                if len(positions) >= self.n_sites:
                    break
            if len(positions) >= self.n_sites:
                break

        pos = np.array(positions[:self.n_sites])
        pos -= np.mean(pos, axis=0)
        return pos

    def _generate_kagome(self) -> np.ndarray:
        """Generate 2D kagome lattice positions."""
        a = self.spacing_um
        # Kagome: triangular lattice with 3-site basis
        basis = np.array([
            [0.0, 0.0],
            [a * 0.5, 0.0],
            [a * 0.25, a * math.sqrt(3) / 4.0],
        ])
        a1 = np.array([a, 0.0])
        a2 = np.array([a * 0.5, a * math.sqrt(3) / 2.0])

        side = int(math.ceil(math.sqrt(self.n_sites / 3)))
        positions = []
        for i in range(side + 1):
            for j in range(side + 1):
                for b in basis:
                    pos = i * a1 + j * a2 + b
                    positions.append(pos)
                    if len(positions) >= self.n_sites:
                        break
                if len(positions) >= self.n_sites:
                    break
            if len(positions) >= self.n_sites:
                break

        pos = np.array(positions[:self.n_sites])
        pos -= np.mean(pos, axis=0)
        return pos

    # ------------------------------------------------------------------
    # Array properties
    # ------------------------------------------------------------------

    @property
    def positions(self) -> np.ndarray:
        """Site positions in micrometres, shape (n_sites, 2)."""
        return self._positions.copy()

    @property
    def occupied(self) -> np.ndarray:
        """Boolean array of site occupation, shape (n_sites,)."""
        return self._occupied.copy()

    @property
    def n_atoms(self) -> int:
        """Number of atoms currently loaded."""
        return int(np.sum(self._occupied))

    def distance(self, site_a: int, site_b: int) -> float:
        """Compute distance between two sites in micrometres.

        Parameters
        ----------
        site_a, site_b : int
            Site indices.

        Returns
        -------
        float
            Distance in micrometres.

        Raises
        ------
        ValueError
            If site indices are out of range.
        """
        self._validate_site(site_a)
        self._validate_site(site_b)
        diff = self._positions[site_a] - self._positions[site_b]
        return float(np.linalg.norm(diff))

    def distance_matrix(self) -> np.ndarray:
        """Compute the full distance matrix between all sites.

        Returns
        -------
        np.ndarray, shape (n_sites, n_sites)
            Pairwise distances in micrometres.
        """
        diff = self._positions[:, np.newaxis, :] - self._positions[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def neighbours(self, site: int, max_distance_um: float | None = None) -> list[int]:
        """Find neighbours of a site within a cutoff distance.

        Parameters
        ----------
        site : int
            Site index.
        max_distance_um : float, optional
            Maximum distance in micrometres.  Defaults to 1.5 * spacing.

        Returns
        -------
        list[int]
            Sorted list of neighbour site indices.
        """
        self._validate_site(site)
        cutoff = max_distance_um if max_distance_um is not None else 1.5 * self.spacing_um
        dists = np.linalg.norm(
            self._positions - self._positions[site], axis=1
        )
        mask = (dists > 0) & (dists <= cutoff)
        return sorted(np.where(mask)[0].tolist())

    # ------------------------------------------------------------------
    # Loading and rearrangement
    # ------------------------------------------------------------------

    def stochastic_load(self, rng: np.random.Generator | None = None) -> int:
        """Simulate stochastic loading with species-dependent probability.

        Each site is independently loaded with probability
        ``species.loading_probability``.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        int
            Number of atoms loaded.
        """
        if rng is None:
            rng = np.random.default_rng()

        self._occupied = rng.random(self.n_sites) < self.species.loading_probability
        return self.n_atoms

    def deterministic_load(self) -> int:
        """Load all sites deterministically (ideal preparation).

        Returns
        -------
        int
            Number of atoms loaded (equals n_sites).
        """
        self._occupied[:] = True
        return self.n_atoms

    def rearrange(self, target_sites: list[int] | None = None) -> int:
        """Rearrange atoms to fill target sites, simulating AOD sorting.

        After stochastic loading, atoms are moved from occupied non-target
        sites to empty target sites.  This models the experimental
        procedure of Barredo et al. (2016) and Endres et al. (2016).

        Parameters
        ----------
        target_sites : list[int], optional
            Sites that should be filled.  If ``None``, fills the first
            ``n_atoms`` sites (compact filling).

        Returns
        -------
        int
            Number of atoms in target sites after rearrangement.

        Raises
        ------
        ValueError
            If not enough atoms to fill all target sites.
        """
        if target_sites is None:
            target_sites = list(range(min(self.n_atoms, self.n_sites)))

        available_atoms = []
        for i in range(self.n_sites):
            if self._occupied[i] and i not in target_sites:
                available_atoms.append(i)

        needed = []
        for t in target_sites:
            if not self._occupied[t]:
                needed.append(t)

        moves = min(len(available_atoms), len(needed))
        for m in range(moves):
            src = available_atoms[m]
            dst = needed[m]
            self._occupied[src] = False
            self._occupied[dst] = True

        filled = sum(1 for t in target_sites if self._occupied[t])
        return filled

    # ------------------------------------------------------------------
    # Zone management
    # ------------------------------------------------------------------

    def assign_zones(self) -> None:
        """Auto-assign sites to zones based on proximity.

        Each site is assigned to the nearest zone whose center is within
        the zone's radius.  Unassigned sites remain -1.
        """
        if not self.zones:
            return

        for i in range(self.n_sites):
            min_dist = float("inf")
            best_zone = -1
            for z_idx, zone_cfg in enumerate(self.zones):
                cx, cy = zone_cfg.center_um
                dx = self._positions[i, 0] - cx
                dy = self._positions[i, 1] - cy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= zone_cfg.radius_um and dist < min_dist:
                    min_dist = dist
                    best_zone = z_idx
            self._zone_assignments[i] = best_zone

    def sites_in_zone(self, zone_index: int) -> list[int]:
        """Return site indices assigned to a given zone.

        Parameters
        ----------
        zone_index : int
            Index into the ``zones`` list.

        Returns
        -------
        list[int]
            Site indices in the specified zone.
        """
        return sorted(np.where(self._zone_assignments == zone_index)[0].tolist())

    # ------------------------------------------------------------------
    # Interaction graph
    # ------------------------------------------------------------------

    def interaction_graph(
        self, rabi_freq_mhz: float = 1.0
    ) -> dict[tuple[int, int], float]:
        """Build the Rydberg interaction graph.

        Returns edges between atom pairs that are within the blockade
        radius, with interaction strengths in MHz.

        Parameters
        ----------
        rabi_freq_mhz : float
            Rabi frequency for blockade radius computation.

        Returns
        -------
        dict[tuple[int, int], float]
            Mapping from (site_i, site_j) to interaction strength in MHz.
        """
        r_b = self.species.blockade_radius_um(rabi_freq_mhz)
        dists = self.distance_matrix()
        graph: dict[tuple[int, int], float] = {}

        for i in range(self.n_sites):
            for j in range(i + 1, self.n_sites):
                if dists[i, j] <= r_b and self._occupied[i] and self._occupied[j]:
                    v_mhz = self.species.vdw_interaction_hz(dists[i, j]) * 1e-6
                    graph[(i, j)] = v_mhz

        return graph

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary of the array configuration."""
        return {
            "n_sites": self.n_sites,
            "n_atoms": self.n_atoms,
            "species": self.species.name,
            "geometry": self.geometry.name,
            "spacing_um": self.spacing_um,
            "n_zones": len(self.zones),
            "fill_fraction": self.n_atoms / self.n_sites if self.n_sites > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_site(self, site: int) -> None:
        """Check that a site index is valid."""
        if site < 0 or site >= self.n_sites:
            raise ValueError(
                f"Site index {site} out of range [0, {self.n_sites})"
            )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AtomArray(n_sites={self.n_sites}, "
            f"species={self.species.name}, "
            f"geometry={self.geometry.name}, "
            f"spacing={self.spacing_um}um, "
            f"loaded={self.n_atoms})"
        )
