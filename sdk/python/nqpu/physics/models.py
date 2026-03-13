"""Model Hamiltonians for quantum-physics workflows."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Iterable

import numpy as np


I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron_n(operators: Iterable[np.ndarray]) -> np.ndarray:
    return reduce(np.kron, operators)


def pauli_string_operator(num_sites: int, terms: dict[int, str]) -> np.ndarray:
    operators = [PAULI[terms.get(site, "I")] for site in range(num_sites)]
    return kron_n(operators)


def _bond_pairs(num_sites: int, boundary: str) -> list[tuple[int, int]]:
    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be 'open' or 'periodic'")
    bonds = [(site, site + 1) for site in range(num_sites - 1)]
    if boundary == "periodic" and num_sites > 2:
        bonds.append((num_sites - 1, 0))
    return bonds


@dataclass(frozen=True)
class CustomHamiltonian:
    matrix: np.ndarray
    label: str = "custom"

    @property
    def num_sites(self) -> int:
        dimension = self.matrix.shape[0]
        sites = int(np.log2(dimension))
        if (1 << sites) != dimension or self.matrix.shape != (dimension, dimension):
            raise ValueError("custom Hamiltonian must have a square 2^n x 2^n shape")
        return sites

    @property
    def dimension(self) -> int:
        return self.matrix.shape[0]

    def hamiltonian(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.complex128)

    @property
    def model_name(self) -> str:
        return self.label


@dataclass(frozen=True)
class TransverseFieldIsing1D:
    num_sites: int
    coupling: float = 1.0
    transverse_field: float = 1.0
    longitudinal_field: float = 0.0
    boundary: str = "open"

    @property
    def dimension(self) -> int:
        return 1 << self.num_sites

    @property
    def model_name(self) -> str:
        return "transverse_field_ising_1d"

    def hamiltonian(self) -> np.ndarray:
        hamiltonian = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for left, right in _bond_pairs(self.num_sites, self.boundary):
            hamiltonian -= self.coupling * pauli_string_operator(
                self.num_sites,
                {left: "Z", right: "Z"},
            )
        for site in range(self.num_sites):
            hamiltonian -= self.transverse_field * pauli_string_operator(self.num_sites, {site: "X"})
            if self.longitudinal_field:
                hamiltonian -= self.longitudinal_field * pauli_string_operator(
                    self.num_sites,
                    {site: "Z"},
                )
        return hamiltonian


@dataclass(frozen=True)
class HeisenbergXXZ1D:
    num_sites: int
    coupling_xy: float = 1.0
    anisotropy: float = 1.0
    field_z: float = 0.0
    boundary: str = "open"

    @property
    def dimension(self) -> int:
        return 1 << self.num_sites

    @property
    def model_name(self) -> str:
        return "heisenberg_xxz_1d"

    def hamiltonian(self) -> np.ndarray:
        hamiltonian = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for left, right in _bond_pairs(self.num_sites, self.boundary):
            hamiltonian += self.coupling_xy * pauli_string_operator(
                self.num_sites,
                {left: "X", right: "X"},
            )
            hamiltonian += self.coupling_xy * pauli_string_operator(
                self.num_sites,
                {left: "Y", right: "Y"},
            )
            hamiltonian += self.coupling_xy * self.anisotropy * pauli_string_operator(
                self.num_sites,
                {left: "Z", right: "Z"},
            )
        if self.field_z:
            for site in range(self.num_sites):
                hamiltonian += self.field_z * pauli_string_operator(self.num_sites, {site: "Z"})
        return hamiltonian


@dataclass(frozen=True)
class HeisenbergXYZ1D:
    num_sites: int
    coupling_x: float = 1.0
    coupling_y: float = 1.0
    coupling_z: float = 1.0
    field_z: float = 0.0
    boundary: str = "open"

    @property
    def dimension(self) -> int:
        return 1 << self.num_sites

    @property
    def model_name(self) -> str:
        return "heisenberg_xyz_1d"

    def hamiltonian(self) -> np.ndarray:
        hamiltonian = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for left, right in _bond_pairs(self.num_sites, self.boundary):
            hamiltonian += self.coupling_x * pauli_string_operator(
                self.num_sites,
                {left: "X", right: "X"},
            )
            hamiltonian += self.coupling_y * pauli_string_operator(
                self.num_sites,
                {left: "Y", right: "Y"},
            )
            hamiltonian += self.coupling_z * pauli_string_operator(
                self.num_sites,
                {left: "Z", right: "Z"},
            )
        if self.field_z:
            for site in range(self.num_sites):
                hamiltonian += self.field_z * pauli_string_operator(self.num_sites, {site: "Z"})
        return hamiltonian
