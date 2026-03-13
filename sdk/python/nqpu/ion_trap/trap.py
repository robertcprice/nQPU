"""Trap configuration and equilibrium physics for linear Paul traps.

Implements the classical Coulomb crystal equilibrium and normal-mode
analysis following:

- D.F.V. James, Appl. Phys. B 66, 181 (1998)
- D. Leibfried et al., Rev. Mod. Phys. 75, 281 (2003)

The equilibrium positions are found by minimising the dimensionless
potential:

    V = (1/2) sum_i z_i^2  +  sum_{i<j} 1/|z_i - z_j|

Normal modes are obtained from the Hessian of V evaluated at
equilibrium.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .species import IonSpecies


@dataclass
class TrapConfig:
    """Paul trap configuration for a linear ion chain.

    Parameters
    ----------
    n_ions : int
        Number of ions in the chain (1 -- 50 typically).
    species : IonSpecies
        Atomic species of all ions (homogeneous chain).
    axial_freq_mhz : float
        Axial (longitudinal) secular frequency in MHz.
    radial_freq_mhz : float
        Radial (transverse) secular frequency in MHz.
    heating_rate_quanta_per_s : float
        Motional heating rate in quanta/s for the centre-of-mass mode.
    background_gas_collision_rate : float
        Background gas collision rate in s^-1.
    """

    n_ions: int
    species: IonSpecies = field(default_factory=lambda: IonSpecies.YB171)
    axial_freq_mhz: float = 1.0
    radial_freq_mhz: float = 5.0
    heating_rate_quanta_per_s: float = 100.0
    background_gas_collision_rate: float = 0.001

    def __post_init__(self) -> None:
        if self.n_ions < 1:
            raise ValueError("n_ions must be >= 1")
        if self.n_ions > 100:
            raise ValueError("n_ions > 100 is not supported (chain instability)")
        if self.radial_freq_mhz <= self.axial_freq_mhz:
            raise ValueError(
                "radial_freq_mhz must exceed axial_freq_mhz for a linear chain"
            )

    # ------------------------------------------------------------------
    # Equilibrium positions
    # ------------------------------------------------------------------

    def equilibrium_positions(self) -> np.ndarray:
        """Compute equilibrium positions of the ion crystal.

        Minimises the dimensionless potential
            V = (1/2) sum_i z_i^2 + sum_{i<j} 1/|z_i - z_j|
        via Newton's method on the gradient.

        Returns
        -------
        np.ndarray, shape (n_ions,)
            Equilibrium positions in units of the characteristic length
            scale l = (e^2 / 4*pi*eps0*m*omega_z^2)^{1/3}.
        """
        n = self.n_ions
        if n == 1:
            return np.array([0.0])

        # Initial guess: uniformly spaced
        z = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
        z = z * (n ** (-0.56))  # empirical scaling for better convergence

        max_iter = 200
        tol = 1e-12

        for _ in range(max_iter):
            grad = self._potential_gradient(z)
            hess = self._potential_hessian(z)
            try:
                delta = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if singular
                delta = -np.linalg.lstsq(hess, grad, rcond=None)[0]
            z += delta
            if np.max(np.abs(grad)) < tol:
                break

        # Centre the chain
        z -= np.mean(z)
        return z

    def equilibrium_positions_meters(self) -> np.ndarray:
        """Equilibrium positions in SI metres."""
        l_scale = self.species.length_scale(self.axial_freq_mhz)
        return self.equilibrium_positions() * l_scale

    # ------------------------------------------------------------------
    # Normal modes
    # ------------------------------------------------------------------

    def normal_modes(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute axial normal-mode frequencies and eigenvectors.

        Returns
        -------
        frequencies_mhz : np.ndarray, shape (n_ions,)
            Normal-mode frequencies in MHz, sorted ascending.
        mode_vectors : np.ndarray, shape (n_ions, n_ions)
            Mode participation matrix where ``mode_vectors[m, j]`` is
            the participation of ion *j* in mode *m*.  Rows are
            normalised to unit length.
        """
        z_eq = self.equilibrium_positions()
        A = self._mode_matrix(z_eq)

        eigenvalues, eigenvectors = np.linalg.eigh(A)

        # Eigenvalues are omega_m^2 / omega_z^2; frequencies in MHz
        eigenvalues = np.clip(eigenvalues, 0.0, None)  # numerical safety
        frequencies_mhz = np.sqrt(eigenvalues) * self.axial_freq_mhz

        # Each column of eigenvectors is a mode; transpose so rows = modes
        mode_vectors = eigenvectors.T

        # Sort by ascending frequency
        order = np.argsort(frequencies_mhz)
        frequencies_mhz = frequencies_mhz[order]
        mode_vectors = mode_vectors[order]

        return frequencies_mhz, mode_vectors

    def radial_normal_modes(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute radial (transverse) normal-mode frequencies and eigenvectors.

        The radial mode matrix is:
            A_ij = delta_ij - (omega_z/omega_r)^2 * C_ij
        where C_ij is the Coulomb coupling in the axial mode matrix.

        Returns same format as :meth:`normal_modes`.
        """
        z_eq = self.equilibrium_positions()
        n = self.n_ions
        ratio_sq = (self.axial_freq_mhz / self.radial_freq_mhz) ** 2

        # Build Coulomb coupling matrix (off-diagonal part of axial Hessian)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = abs(z_eq[i] - z_eq[j])
                    C[i, j] = -1.0 / diff ** 3

        # Radial mode matrix
        A = np.eye(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    A[i, j] = -ratio_sq * C[i, j]
                    A[i, i] += ratio_sq * C[i, j]

        eigenvalues, eigenvectors = np.linalg.eigh(A)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        frequencies_mhz = np.sqrt(eigenvalues) * self.radial_freq_mhz

        mode_vectors = eigenvectors.T
        order = np.argsort(frequencies_mhz)
        return frequencies_mhz[order], mode_vectors[order]

    # ------------------------------------------------------------------
    # Helper: characteristic parameters
    # ------------------------------------------------------------------

    @property
    def length_scale_m(self) -> float:
        """Characteristic length scale in metres."""
        return self.species.length_scale(self.axial_freq_mhz)

    @property
    def lamb_dicke(self) -> float:
        """Lamb-Dicke parameter for the COM mode."""
        return self.species.lamb_dicke_parameter(self.axial_freq_mhz)

    @property
    def mean_phonon_number(self) -> float:
        """Thermal phonon number after Doppler cooling.

        n_bar ~ Gamma / (2 * omega) for Doppler limit.
        """
        gamma = self.species.spontaneous_emission_rate
        omega = self.axial_freq_mhz * 1e6 * 2.0 * math.pi
        return gamma / (2.0 * omega)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _potential_gradient(z: np.ndarray) -> np.ndarray:
        """Gradient of the dimensionless potential."""
        n = len(z)
        grad = np.copy(z)  # from harmonic term
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = z[i] - z[j]
                    grad[i] -= 1.0 / (diff * abs(diff))
        return grad

    @staticmethod
    def _potential_hessian(z: np.ndarray) -> np.ndarray:
        """Hessian of the dimensionless potential."""
        n = len(z)
        H = np.eye(n)  # from harmonic term
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = abs(z[i] - z[j])
                    val = 2.0 / diff ** 3
                    H[i, j] -= val
                    H[i, i] += val
        return H

    @staticmethod
    def _mode_matrix(z_eq: np.ndarray) -> np.ndarray:
        """Dimensionless mode matrix A_{ij} from equilibrium positions.

        A_{ii} = 1 + 2 * sum_{j!=i} 1/|z_i - z_j|^3
        A_{ij} = -2 / |z_i - z_j|^3   (i != j)
        """
        n = len(z_eq)
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = abs(z_eq[i] - z_eq[j])
                    coupling = 2.0 / diff ** 3
                    A[i, j] = -coupling
                    A[i, i] += coupling
            A[i, i] += 1.0  # harmonic trap contribution
        return A

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary of trap parameters."""
        freqs, modes = self.normal_modes()
        return {
            "n_ions": self.n_ions,
            "species": self.species.name,
            "axial_freq_mhz": self.axial_freq_mhz,
            "radial_freq_mhz": self.radial_freq_mhz,
            "length_scale_um": self.length_scale_m * 1e6,
            "lamb_dicke": self.lamb_dicke,
            "com_mode_freq_mhz": float(freqs[0]) if len(freqs) > 0 else 0.0,
            "highest_mode_freq_mhz": float(freqs[-1]) if len(freqs) > 0 else 0.0,
            "heating_rate_quanta_per_s": self.heating_rate_quanta_per_s,
        }


# ======================================================================
# Device presets
# ======================================================================

class DevicePresets:
    """Pre-calibrated device configurations for commercial trapped-ion systems."""

    @staticmethod
    def ionq_aria() -> TrapConfig:
        """IonQ Aria: 25-qubit 171Yb+ system.

        Typical published specifications (circa 2023):
        - 25 algorithmic qubits
        - #AQ 25, ~99.5% 1Q fidelity, ~97% 2Q fidelity
        """
        return TrapConfig(
            n_ions=25,
            species=IonSpecies.YB171,
            axial_freq_mhz=0.7,
            radial_freq_mhz=3.5,
            heating_rate_quanta_per_s=50.0,
            background_gas_collision_rate=0.0005,
        )

    @staticmethod
    def ionq_forte() -> TrapConfig:
        """IonQ Forte: 36-qubit 171Yb+ system.

        Typical published specifications (circa 2024):
        - 36 algorithmic qubits
        - #AQ 35, ~99.7% 1Q fidelity, ~99% 2Q fidelity
        """
        return TrapConfig(
            n_ions=36,
            species=IonSpecies.YB171,
            axial_freq_mhz=0.5,
            radial_freq_mhz=4.0,
            heating_rate_quanta_per_s=30.0,
            background_gas_collision_rate=0.0003,
        )

    @staticmethod
    def quantinuum_h1() -> TrapConfig:
        """Quantinuum H1: 20-qubit 171Yb+ QCCD system.

        Typical published specifications (circa 2023):
        - 20 qubits in a racetrack trap
        - ~99.99% 1Q fidelity, ~99.7% 2Q fidelity
        """
        return TrapConfig(
            n_ions=20,
            species=IonSpecies.YB171,
            axial_freq_mhz=1.5,
            radial_freq_mhz=5.0,
            heating_rate_quanta_per_s=20.0,
            background_gas_collision_rate=0.0001,
        )

    @staticmethod
    def quantinuum_h2() -> TrapConfig:
        """Quantinuum H2: 56-qubit 171Yb+ QCCD system.

        Typical published specifications (circa 2024):
        - 56 qubits in a 2D racetrack trap
        - ~99.99% 1Q fidelity, ~99.8% 2Q fidelity
        """
        return TrapConfig(
            n_ions=56,
            species=IonSpecies.YB171,
            axial_freq_mhz=1.2,
            radial_freq_mhz=4.5,
            heating_rate_quanta_per_s=15.0,
            background_gas_collision_rate=0.0001,
        )

    @staticmethod
    def oxford_ionics_demo() -> TrapConfig:
        """Oxford Ionics: 43Ca+ hyperfine qubit demonstrator."""
        return TrapConfig(
            n_ions=8,
            species=IonSpecies.CA43,
            axial_freq_mhz=2.0,
            radial_freq_mhz=5.5,
            heating_rate_quanta_per_s=10.0,
            background_gas_collision_rate=0.0002,
        )
