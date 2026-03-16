"""Molecular geometry specification and basis set representation.

Defines the core data structures for quantum chemistry calculations:

- :class:`Atom` -- a single atomic center with symbol and 3D position.
- :class:`Molecule` -- a collection of atoms with charge and multiplicity.
- :class:`BasisFunction` -- a contracted Gaussian-type orbital (CGTO).
- :class:`BasisSet` -- a named collection of basis functions for a molecule.

Also provides predefined molecule factories (H2, LiH, H2O, BeH2, H4-chain,
H6-ring) and nuclear repulsion energy calculation.

All distances are in Angstroms, energies in Hartree, unless stated otherwise.

References
----------
- Szabo & Ostlund, *Modern Quantum Chemistry* (1996), Chapter 3.
- Hehre, Stewart & Pople, J. Chem. Phys. 51, 2657 (1969) [STO-3G].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ============================================================
# Constants
# ============================================================

ANGSTROM_TO_BOHR = 1.8897259886
"""Conversion factor from Angstrom to Bohr (atomic units)."""

# Atomic numbers for supported elements
ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6,
    "N": 7, "O": 8, "F": 9, "Ne": 10,
}

# Nuclear charges (same as atomic number for neutral atoms)
NUCLEAR_CHARGES: dict[str, float] = {
    sym: float(z) for sym, z in ATOMIC_NUMBERS.items()
}


# ============================================================
# Atom
# ============================================================


@dataclass(frozen=True)
class Atom:
    """A single atomic center in a molecule.

    Parameters
    ----------
    symbol : str
        Chemical element symbol (e.g. ``'H'``, ``'Li'``, ``'O'``).
    position : tuple[float, float, float]
        Cartesian coordinates in Angstroms.
    """

    symbol: str
    position: tuple[float, float, float]

    @property
    def atomic_number(self) -> int:
        """Return the atomic number for this element."""
        sym = self.symbol.capitalize()
        if sym not in ATOMIC_NUMBERS:
            raise ValueError(f"Unsupported element: {self.symbol}")
        return ATOMIC_NUMBERS[sym]

    @property
    def nuclear_charge(self) -> float:
        """Return the nuclear charge Z."""
        return float(self.atomic_number)

    @property
    def position_bohr(self) -> tuple[float, float, float]:
        """Position converted to Bohr (atomic units)."""
        return (
            self.position[0] * ANGSTROM_TO_BOHR,
            self.position[1] * ANGSTROM_TO_BOHR,
            self.position[2] * ANGSTROM_TO_BOHR,
        )

    def __repr__(self) -> str:
        x, y, z = self.position
        return f"Atom({self.symbol!r}, ({x:.4f}, {y:.4f}, {z:.4f}))"


# ============================================================
# Molecule
# ============================================================


@dataclass
class Molecule:
    """A molecular system for quantum chemistry calculations.

    Attributes
    ----------
    atoms : list[Atom]
        Atomic centers.
    charge : int
        Net molecular charge (0 for neutral).
    multiplicity : int
        Spin multiplicity (1 for singlet, 2 for doublet, etc.).
    """

    atoms: list[Atom]
    charge: int = 0
    multiplicity: int = 1

    @classmethod
    def from_atoms(
        cls,
        atoms: list[Atom],
        charge: int = 0,
        multiplicity: int = 1,
    ) -> Molecule:
        """Build a molecule from a list of Atom objects.

        Parameters
        ----------
        atoms : list[Atom]
            Atomic centers with positions in Angstroms.
        charge : int
            Net molecular charge.
        multiplicity : int
            Spin multiplicity (2S+1).

        Returns
        -------
        Molecule
        """
        if not atoms:
            raise ValueError("Molecule must contain at least one atom")
        if multiplicity < 1:
            raise ValueError("Multiplicity must be >= 1")
        return cls(atoms=list(atoms), charge=charge, multiplicity=multiplicity)

    @classmethod
    def from_xyz(cls, xyz_string: str, charge: int = 0, multiplicity: int = 1) -> Molecule:
        """Parse a molecule from XYZ format string.

        The XYZ format is::

            <num_atoms>
            <comment line>
            <symbol> <x> <y> <z>
            ...

        Alternatively, if the first line is not an integer, all lines
        are treated as ``symbol x y z``.

        Parameters
        ----------
        xyz_string : str
            XYZ-format geometry.
        charge : int
            Net molecular charge.
        multiplicity : int
            Spin multiplicity.

        Returns
        -------
        Molecule
        """
        lines = [ln.strip() for ln in xyz_string.strip().splitlines() if ln.strip()]
        atoms: list[Atom] = []

        # Detect whether the first line is an atom count
        start = 0
        try:
            n_atoms = int(lines[0])
            start = 2  # skip count + comment
        except (ValueError, IndexError):
            start = 0

        for line in lines[start:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            sym = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(Atom(symbol=sym, position=(x, y, z)))

        if not atoms:
            raise ValueError("No atoms found in XYZ string")
        return cls(atoms=atoms, charge=charge, multiplicity=multiplicity)

    # ---- Properties ----

    @property
    def num_atoms(self) -> int:
        """Total number of atoms."""
        return len(self.atoms)

    @property
    def num_electrons(self) -> int:
        """Total number of electrons (accounting for charge)."""
        total_z = sum(a.atomic_number for a in self.atoms)
        return total_z - self.charge

    @property
    def num_alpha(self) -> int:
        """Number of alpha electrons."""
        n_unpaired = self.multiplicity - 1
        return (self.num_electrons + n_unpaired) // 2

    @property
    def num_beta(self) -> int:
        """Number of beta electrons."""
        return self.num_electrons - self.num_alpha

    @property
    def symbols(self) -> list[str]:
        """List of element symbols."""
        return [a.symbol for a in self.atoms]

    @property
    def formula(self) -> str:
        """Molecular formula string (e.g. 'H2O')."""
        from collections import Counter
        counts = Counter(self.symbols)
        parts = []
        for sym in sorted(counts.keys()):
            cnt = counts[sym]
            parts.append(f"{sym}{cnt}" if cnt > 1 else sym)
        return "".join(parts)

    def nuclear_repulsion_energy(self) -> float:
        """Compute the nuclear repulsion energy in Hartree.

        .. math::

            E_{\\mathrm{nuc}} = \\sum_{A<B} \\frac{Z_A Z_B}{R_{AB}}

        where distances are in Bohr.

        Returns
        -------
        float
            Nuclear repulsion energy in Hartree.
        """
        e_nuc = 0.0
        for i in range(len(self.atoms)):
            ri = np.array(self.atoms[i].position_bohr)
            zi = self.atoms[i].nuclear_charge
            for j in range(i + 1, len(self.atoms)):
                rj = np.array(self.atoms[j].position_bohr)
                zj = self.atoms[j].nuclear_charge
                dist = np.linalg.norm(ri - rj)
                if dist < 1e-10:
                    raise ValueError(
                        f"Atoms {i} and {j} are at the same position"
                    )
                e_nuc += zi * zj / dist
        return float(e_nuc)

    def detect_symmetry(self) -> str:
        """Detect simple molecular point group symmetry.

        Returns a string label. Currently detects:
        - ``'C1'`` (no symmetry)
        - ``'Dinfh'`` (linear symmetric, e.g. H2)
        - ``'Cinfv'`` (linear asymmetric, e.g. HF)
        - ``'C2v'`` (planar with C2 axis, e.g. H2O)

        Returns
        -------
        str
            Point group label.
        """
        if self.num_atoms == 1:
            return "Kh"
        if self.num_atoms == 2:
            if self.atoms[0].symbol == self.atoms[1].symbol:
                return "Dinfh"
            return "Cinfv"

        # Check if all atoms are colinear
        positions = np.array([a.position for a in self.atoms])
        if _are_collinear(positions):
            syms = [a.symbol for a in self.atoms]
            if syms == syms[::-1]:
                return "Dinfh"
            return "Cinfv"

        # Check if all atoms are coplanar
        if _are_coplanar(positions):
            return "C2v"

        return "C1"

    def __repr__(self) -> str:
        return (
            f"Molecule({self.formula}, "
            f"charge={self.charge}, "
            f"mult={self.multiplicity}, "
            f"e-={self.num_electrons})"
        )


# ============================================================
# Basis Function and Basis Set
# ============================================================


@dataclass
class PrimitiveGaussian:
    """A single primitive Gaussian function.

    .. math::

        g(r) = N \\exp(-\\alpha |r - R|^2)

    Parameters
    ----------
    exponent : float
        Gaussian exponent alpha.
    coefficient : float
        Contraction coefficient.
    center : tuple[float, float, float]
        Center position in Bohr.
    """

    exponent: float
    coefficient: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def normalization(self) -> float:
        """Normalization constant for an s-type Gaussian.

        .. math::

            N = (2\\alpha/\\pi)^{3/4}
        """
        return (2.0 * self.exponent / math.pi) ** 0.75


@dataclass
class BasisFunction:
    """A contracted Gaussian-type orbital (CGTO).

    A linear combination of :class:`PrimitiveGaussian` functions, all
    centered on the same atom.

    Attributes
    ----------
    primitives : list[PrimitiveGaussian]
        Primitive Gaussians with exponents and contraction coefficients.
    center : tuple[float, float, float]
        Center position in Bohr.
    angular_momentum : int
        Orbital angular momentum quantum number (0 for s, 1 for p, ...).
    atom_index : int
        Index of the atom this function is centered on.
    label : str
        Human-readable label (e.g. ``'H1_1s'``).
    """

    primitives: list[PrimitiveGaussian]
    center: tuple[float, float, float]
    angular_momentum: int = 0
    atom_index: int = 0
    label: str = ""

    def __repr__(self) -> str:
        return f"BasisFunction({self.label}, L={self.angular_momentum}, n_prim={len(self.primitives)})"


# ============================================================
# STO-3G Basis Set Data
# ============================================================

# STO-3G exponents and coefficients from Hehre, Stewart & Pople (1969)
# Format: { element: { orbital_label: [(exponent, coefficient), ...] } }
# All exponents are for the Slater exponent zeta already folded in.

_STO3G_DATA: dict[str, dict[str, list[tuple[float, float]]]] = {
    "H": {
        "1s": [
            (3.42525091, 0.15432897),
            (0.62353064, 0.53532814),
            (0.16885540, 0.44463454),
        ],
    },
    "He": {
        "1s": [
            (6.36242139, 0.15432897),
            (1.15892300, 0.53532814),
            (0.31364979, 0.44463454),
        ],
    },
    "Li": {
        "1s": [
            (16.11957475, 0.15432897),
            (2.93620070, 0.53532814),
            (0.79465050, 0.44463454),
        ],
        "2s": [
            (0.63628970, -0.09996723),
            (0.14786010, 0.39951283),
            (0.04808870, 0.70011547),
        ],
    },
    "Be": {
        "1s": [
            (30.16787069, 0.15432897),
            (5.49531529, 0.53532814),
            (1.48719270, 0.44463454),
        ],
        "2s": [
            (1.31483320, -0.09996723),
            (0.30553890, 0.39951283),
            (0.09937070, 0.70011547),
        ],
    },
    "B": {
        "1s": [
            (48.79111318, 0.15432897),
            (8.88736641, 0.53532814),
            (2.40534070, 0.44463454),
        ],
        "2s": [
            (2.23695610, -0.09996723),
            (0.51982050, 0.39951283),
            (0.16906180, 0.70011547),
        ],
    },
    "C": {
        "1s": [
            (71.61683735, 0.15432897),
            (13.04509632, 0.53532814),
            (3.53051216, 0.44463454),
        ],
        "2s": [
            (2.94124940, -0.09996723),
            (0.68348310, 0.39951283),
            (0.22228990, 0.70011547),
        ],
    },
    "N": {
        "1s": [
            (99.10616896, 0.15432897),
            (18.05231239, 0.53532814),
            (4.88588486, 0.44463454),
        ],
        "2s": [
            (3.78045592, -0.09996723),
            (0.87849660, 0.39951283),
            (0.28571440, 0.70011547),
        ],
    },
    "O": {
        "1s": [
            (130.70932139, 0.15432897),
            (23.80886605, 0.53532814),
            (6.44360831, 0.44463454),
        ],
        "2s": [
            (5.03315132, -0.09996723),
            (1.16959612, 0.39951283),
            (0.38038900, 0.70011547),
        ],
    },
    "F": {
        "1s": [
            (166.67913742, 0.15432897),
            (30.36081233, 0.53532814),
            (8.21682067, 0.44463454),
        ],
        "2s": [
            (6.46480325, -0.09996723),
            (1.50228124, 0.39951283),
            (0.48858849, 0.70011547),
        ],
    },
}


# 6-31G basis data for H (minimal, for testing)
_631G_DATA: dict[str, dict[str, list[tuple[float, float]]]] = {
    "H": {
        "1s_inner": [
            (18.7311370, 0.03349460),
            (2.8253937, 0.23472695),
            (0.6401217, 0.81375733),
        ],
        "1s_outer": [
            (0.1612778, 1.0),
        ],
    },
}


class BasisSet:
    """A named basis set applied to a specific molecule.

    Constructs :class:`BasisFunction` objects for each atom in the
    molecule by looking up tabulated STO-nG or 6-31G parameters.

    Parameters
    ----------
    name : str
        Basis set name (``'sto-3g'`` or ``'6-31g'``).
    molecule : Molecule
        The molecular system.

    Attributes
    ----------
    functions : list[BasisFunction]
        All contracted basis functions for this molecule.
    """

    SUPPORTED = ("sto-3g", "6-31g")

    def __init__(self, name: str, molecule: Molecule) -> None:
        self.name = name.lower()
        if self.name not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported basis set '{name}'. "
                f"Supported: {self.SUPPORTED}"
            )
        self.molecule = molecule
        self.functions: list[BasisFunction] = []
        self._build()

    def _build(self) -> None:
        """Construct basis functions from tabulated data."""
        if self.name == "sto-3g":
            data = _STO3G_DATA
        elif self.name == "6-31g":
            data = _631G_DATA
        else:
            raise ValueError(f"No data for basis set '{self.name}'")

        for idx, atom in enumerate(self.molecule.atoms):
            sym = atom.symbol.capitalize()
            if sym not in data:
                raise ValueError(
                    f"Element {sym} not supported in {self.name} basis set"
                )
            center = atom.position_bohr
            for orbital_label, primitives_data in data[sym].items():
                prims = [
                    PrimitiveGaussian(
                        exponent=exp,
                        coefficient=coeff,
                        center=center,
                    )
                    for exp, coeff in primitives_data
                ]
                bf = BasisFunction(
                    primitives=prims,
                    center=center,
                    angular_momentum=0,  # s-type only in STO-3G minimal
                    atom_index=idx,
                    label=f"{sym}{idx + 1}_{orbital_label}",
                )
                self.functions.append(bf)

    @property
    def num_functions(self) -> int:
        """Total number of basis functions."""
        return len(self.functions)

    def num_orbitals(self) -> int:
        """Number of spatial orbitals (same as num_functions for s-only)."""
        return self.num_functions

    def __repr__(self) -> str:
        return f"BasisSet({self.name!r}, n_funcs={self.num_functions})"


# ============================================================
# Predefined molecules
# ============================================================


def h2(bond_length: float = 0.74) -> Molecule:
    """Create a hydrogen molecule (H2).

    Parameters
    ----------
    bond_length : float
        H-H bond length in Angstroms (default: 0.74 A, equilibrium).

    Returns
    -------
    Molecule
    """
    d = bond_length / 2.0
    return Molecule.from_atoms([
        Atom("H", (0.0, 0.0, -d)),
        Atom("H", (0.0, 0.0, d)),
    ])


def lih(bond_length: float = 1.6) -> Molecule:
    """Create a lithium hydride molecule (LiH).

    Parameters
    ----------
    bond_length : float
        Li-H bond length in Angstroms (default: 1.6 A).

    Returns
    -------
    Molecule
    """
    return Molecule.from_atoms([
        Atom("Li", (0.0, 0.0, 0.0)),
        Atom("H", (0.0, 0.0, bond_length)),
    ])


def h2o() -> Molecule:
    """Create a water molecule (H2O) at experimental geometry.

    Bond length: 0.9572 A, bond angle: 104.52 degrees.

    Returns
    -------
    Molecule
    """
    bl = 0.9572  # O-H bond length in Angstrom
    angle = math.radians(104.52)
    return Molecule.from_atoms([
        Atom("O", (0.0, 0.0, 0.0)),
        Atom("H", (bl * math.sin(angle / 2), 0.0, bl * math.cos(angle / 2))),
        Atom("H", (-bl * math.sin(angle / 2), 0.0, bl * math.cos(angle / 2))),
    ])


def beh2() -> Molecule:
    """Create a beryllium dihydride molecule (BeH2), linear geometry.

    Bond length: 1.326 A.

    Returns
    -------
    Molecule
    """
    bl = 1.326
    return Molecule.from_atoms([
        Atom("H", (0.0, 0.0, -bl)),
        Atom("Be", (0.0, 0.0, 0.0)),
        Atom("H", (0.0, 0.0, bl)),
    ])


def h4_chain(spacing: float = 1.0) -> Molecule:
    """Create a linear chain of four hydrogen atoms.

    Parameters
    ----------
    spacing : float
        Distance between adjacent H atoms in Angstroms.

    Returns
    -------
    Molecule
    """
    atoms = [Atom("H", (0.0, 0.0, i * spacing)) for i in range(4)]
    return Molecule.from_atoms(atoms)


def h6_ring(radius: float = 1.0) -> Molecule:
    """Create a hexagonal ring of six hydrogen atoms.

    Parameters
    ----------
    radius : float
        Radius of the ring in Angstroms.

    Returns
    -------
    Molecule
    """
    atoms = []
    for i in range(6):
        angle = 2.0 * math.pi * i / 6
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        atoms.append(Atom("H", (x, y, 0.0)))
    return Molecule.from_atoms(atoms)


# ============================================================
# Geometry helpers
# ============================================================


def _are_collinear(positions: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if a set of 3D positions are collinear."""
    if len(positions) <= 2:
        return True
    v0 = positions[1] - positions[0]
    v0_norm = np.linalg.norm(v0)
    if v0_norm < tol:
        return True
    v0 = v0 / v0_norm
    for i in range(2, len(positions)):
        vi = positions[i] - positions[0]
        vi_norm = np.linalg.norm(vi)
        if vi_norm < tol:
            continue
        vi = vi / vi_norm
        cross = np.linalg.norm(np.cross(v0, vi))
        if cross > tol:
            return False
    return True


def _are_coplanar(positions: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if a set of 3D positions are coplanar."""
    if len(positions) <= 3:
        return True
    v0 = positions[1] - positions[0]
    v1 = positions[2] - positions[0]
    normal = np.cross(v0, v1)
    n_norm = np.linalg.norm(normal)
    if n_norm < tol:
        return True  # degenerate
    normal = normal / n_norm
    for i in range(3, len(positions)):
        vi = positions[i] - positions[0]
        if abs(np.dot(normal, vi)) > tol:
            return False
    return True
