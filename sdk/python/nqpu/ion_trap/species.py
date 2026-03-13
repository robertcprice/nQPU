"""Ion species database with calibrated physical parameters.

Physical constants sourced from NIST Atomic Spectra Database and the
following references:

- 171Yb+: Olmschenk et al., Phys. Rev. A 76, 052314 (2007)
- 133Ba+: Hucul et al., Phys. Rev. Lett. 119, 100501 (2017)
- 40Ca+:  Haffner et al., Phys. Rep. 469, 155 (2008)
- 43Ca+:  Harty et al., Phys. Rev. Lett. 113, 220501 (2014)
- 88Sr+:  Leschhorn et al., Appl. Phys. B 108, 159 (2012)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Fundamental constants (CODATA 2018)
# ---------------------------------------------------------------------------
AMU_KG = 1.66053906660e-27       # atomic mass unit in kg
HBAR = 1.054571817e-34           # reduced Planck constant, J*s
C_LIGHT = 2.99792458e8           # speed of light, m/s
ELEMENTARY_CHARGE = 1.602176634e-19  # electron charge, C
K_COULOMB = 8.9875517873681764e9 # Coulomb constant, N*m^2/C^2
EPSILON_0 = 8.8541878128e-12     # vacuum permittivity, F/m


@dataclass(frozen=True)
class IonSpecies:
    """Atomic ion species with calibrated physical parameters.

    Parameters
    ----------
    name : str
        Human-readable species label (e.g. '171Yb+').
    mass_amu : float
        Atomic mass in unified atomic mass units.
    charge : int
        Ion charge state (typically +1).
    nuclear_spin : float
        Nuclear spin quantum number I.
    hyperfine_splitting_ghz : float
        Ground-state hyperfine splitting in GHz.  0.0 for optical qubits.
    qubit_wavelength_nm : float
        Primary qubit transition wavelength in nm.
    spontaneous_emission_rate : float
        Excited-state spontaneous decay rate in s^-1.
    branching_ratio : float
        Branching ratio into the cycling transition (0-1).
    scattering_rate_per_gate : float
        Typical off-resonant photon scattering probability per gate.
    t1_s : float
        Longitudinal relaxation time in seconds (qubit T1).
    t2_s : float
        Transverse coherence time in seconds (qubit T2).
    """

    name: str
    mass_amu: float
    charge: int
    nuclear_spin: float
    hyperfine_splitting_ghz: float
    qubit_wavelength_nm: float
    spontaneous_emission_rate: float
    branching_ratio: float
    scattering_rate_per_gate: float
    t1_s: float = 10.0
    t2_s: float = 1.0

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def mass_kg(self) -> float:
        """Mass in kilograms."""
        return self.mass_amu * AMU_KG

    @property
    def qubit_wavelength_m(self) -> float:
        """Qubit transition wavelength in metres."""
        return self.qubit_wavelength_nm * 1e-9

    @property
    def qubit_frequency_hz(self) -> float:
        """Qubit transition frequency in Hz."""
        return C_LIGHT / self.qubit_wavelength_m

    @property
    def wavevector(self) -> float:
        """Transition wavevector k = 2*pi / lambda in m^-1."""
        return 2.0 * math.pi / self.qubit_wavelength_m

    def lamb_dicke_parameter(
        self,
        trap_freq_mhz: float,
        wavelength_nm: float | None = None,
    ) -> float:
        r"""Compute the Lamb--Dicke parameter.

        .. math::
            \eta = k \sqrt{\frac{\hbar}{2 m \omega}}

        Parameters
        ----------
        trap_freq_mhz : float
            Motional (secular) frequency in MHz.
        wavelength_nm : float, optional
            Override wavelength in nm.  Defaults to ``qubit_wavelength_nm``.

        Returns
        -------
        float
            Dimensionless Lamb--Dicke parameter eta.
        """
        wl = (wavelength_nm * 1e-9) if wavelength_nm else self.qubit_wavelength_m
        k = 2.0 * math.pi / wl
        omega = trap_freq_mhz * 1e6 * 2.0 * math.pi  # angular freq in rad/s
        return k * math.sqrt(HBAR / (2.0 * self.mass_kg * omega))

    def length_scale(self, axial_freq_mhz: float) -> float:
        """Characteristic length scale l = (e^2 / 4*pi*eps0 * m * omega_z^2)^(1/3).

        Used for expressing equilibrium positions in natural units.

        Parameters
        ----------
        axial_freq_mhz : float
            Axial trap frequency in MHz.
        """
        omega_z = axial_freq_mhz * 1e6 * 2.0 * math.pi
        numerator = ELEMENTARY_CHARGE ** 2 / (4.0 * math.pi * EPSILON_0)
        denominator = self.mass_kg * omega_z ** 2
        return (numerator / denominator) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Preset species (class-level singletons)
# ---------------------------------------------------------------------------

IonSpecies.YB171 = IonSpecies(  # type: ignore[attr-defined]
    name="171Yb+",
    mass_amu=170.9363258,
    charge=1,
    nuclear_spin=0.5,
    hyperfine_splitting_ghz=12.642812118466,
    qubit_wavelength_nm=369.5262,
    spontaneous_emission_rate=1.18e8,
    branching_ratio=0.9950,
    scattering_rate_per_gate=2e-5,
    t1_s=3600.0,       # hyperfine qubit: hours-scale T1
    t2_s=1.5,          # Ramsey coherence with spin-echo
)

IonSpecies.BA133 = IonSpecies(  # type: ignore[attr-defined]
    name="133Ba+",
    mass_amu=132.9060338,
    charge=1,
    nuclear_spin=0.5,
    hyperfine_splitting_ghz=9.925413,
    qubit_wavelength_nm=493.41,
    spontaneous_emission_rate=9.53e7,
    branching_ratio=0.75,
    scattering_rate_per_gate=3e-5,
    t1_s=3600.0,
    t2_s=1.0,
)

IonSpecies.CA40 = IonSpecies(  # type: ignore[attr-defined]
    name="40Ca+",
    mass_amu=39.962590863,
    charge=1,
    nuclear_spin=0.0,
    hyperfine_splitting_ghz=0.0,           # optical qubit — no hyperfine
    qubit_wavelength_nm=729.147,           # S1/2 -> D5/2 quadrupole transition
    spontaneous_emission_rate=1.35,        # D5/2 metastable: ~1.17 s lifetime
    branching_ratio=0.9347,
    scattering_rate_per_gate=1e-6,
    t1_s=1.168,        # limited by D5/2 lifetime
    t2_s=0.05,         # optical qubit: tens of ms T2
)

IonSpecies.CA43 = IonSpecies(  # type: ignore[attr-defined]
    name="43Ca+",
    mass_amu=42.958766440,
    charge=1,
    nuclear_spin=3.5,
    hyperfine_splitting_ghz=3.225608286,
    qubit_wavelength_nm=397.0,
    spontaneous_emission_rate=1.4e8,
    branching_ratio=0.935,
    scattering_rate_per_gate=1e-5,
    t1_s=3600.0,
    t2_s=50.0,         # clock-state qubit: exceptional T2
)

IonSpecies.SR88 = IonSpecies(  # type: ignore[attr-defined]
    name="88Sr+",
    mass_amu=87.905612,
    charge=1,
    nuclear_spin=0.0,
    hyperfine_splitting_ghz=0.0,           # optical qubit
    qubit_wavelength_nm=674.0,             # S1/2 -> D5/2
    spontaneous_emission_rate=2.56,        # D5/2 metastable: ~0.39 s lifetime
    branching_ratio=0.9446,
    scattering_rate_per_gate=1e-6,
    t1_s=0.391,
    t2_s=0.02,
)


# Convenience list of all built-in species
ALL_SPECIES: list[IonSpecies] = [
    IonSpecies.YB171,
    IonSpecies.BA133,
    IonSpecies.CA40,
    IonSpecies.CA43,
    IonSpecies.SR88,
]
