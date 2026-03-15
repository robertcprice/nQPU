"""Rydberg atom physics for neutral-atom quantum computing.

Provides the foundational physical parameters and interaction models
for neutral-atom arrays using Rydberg blockade gates.

Atom species data sourced from:
- Rb87: Saffman, Walker & Molmer, Rev. Mod. Phys. 82, 2313 (2010)
- Cs133: Levine et al., Phys. Rev. Lett. 123, 170503 (2019)
- Sr87: Madjarov et al., Nature Physics 16, 857 (2020)
- Yb171: Ma et al., Nature 622, 279 (2023)

Rydberg interaction:
    V(r) = C6 / r^6  (van der Waals regime)
    Blockade radius: R_b = (C6 / Omega)^(1/6)

where C6 depends on species, Rydberg state, and polarisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Fundamental constants (CODATA 2018)
# ---------------------------------------------------------------------------
AMU_KG = 1.66053906660e-27  # atomic mass unit in kg
HBAR = 1.054571817e-34  # reduced Planck constant, J*s
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant, J/K
C_LIGHT = 2.99792458e8  # speed of light, m/s
A_BOHR = 5.29177210903e-11  # Bohr radius, m
E_HARTREE = 4.3597447222071e-18  # Hartree energy, J
MICRO = 1e-6  # micro prefix


@dataclass(frozen=True)
class AtomSpecies:
    """Neutral atom species with calibrated physical parameters for Rydberg QC.

    Parameters
    ----------
    name : str
        Human-readable species label (e.g. '87Rb').
    mass_amu : float
        Atomic mass in unified atomic mass units.
    nuclear_spin : float
        Nuclear spin quantum number I.
    wavelength_trap_nm : float
        Typical optical tweezer trapping wavelength in nm.
    wavelength_rydberg_nm : float
        Wavelength for ground-to-Rydberg excitation in nm.
        Two-photon transition: effective wavelength for total process.
    rydberg_state_n : int
        Principal quantum number of the Rydberg state used for gates.
    c6_hz_um6 : float
        C6 dispersion coefficient in Hz * um^6.
        V(r) = h * C6 / r^6 where r is in micrometres.
    rydberg_lifetime_us : float
        Rydberg state lifetime in microseconds (at 300K including
        blackbody radiation).
    scattering_rate_trap_hz : float
        Photon scattering rate from the optical tweezer in Hz.
        Determines heating and atom loss rates.
    ground_state_lifetime_s : float
        Qubit coherence time (T2) in the ground-state manifold, seconds.
    t1_s : float
        Longitudinal relaxation time in seconds (qubit T1).
    t2_s : float
        Transverse coherence time in seconds (qubit T2).
    loading_probability : float
        Single-site loading probability after preparation.
    temperature_uk : float
        Typical atom temperature after cooling, in microkelvin.
    """

    name: str
    mass_amu: float
    nuclear_spin: float
    wavelength_trap_nm: float
    wavelength_rydberg_nm: float
    rydberg_state_n: int
    c6_hz_um6: float
    rydberg_lifetime_us: float
    scattering_rate_trap_hz: float
    ground_state_lifetime_s: float
    t1_s: float = 10.0
    t2_s: float = 1.0
    loading_probability: float = 0.5
    temperature_uk: float = 10.0

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def mass_kg(self) -> float:
        """Mass in kilograms."""
        return self.mass_amu * AMU_KG

    @property
    def c6_joule_m6(self) -> float:
        """C6 coefficient in SI units (J * m^6)."""
        # Convert from Hz * um^6 to J * m^6
        # h * Hz = J, um^6 = (1e-6)^6 m^6 = 1e-36 m^6
        h = 2.0 * math.pi * HBAR
        return h * self.c6_hz_um6 * 1e-36

    def blockade_radius_um(self, rabi_freq_mhz: float) -> float:
        """Compute the Rydberg blockade radius.

        The blockade radius is the distance at which the Rydberg-Rydberg
        interaction energy equals the Rabi frequency:

            R_b = (C6 / hbar * Omega)^(1/6)

        In convenient units:
            R_b = (C6_hz_um6 / Omega_hz)^(1/6) um

        Parameters
        ----------
        rabi_freq_mhz : float
            Rabi frequency of the Rydberg excitation in MHz.

        Returns
        -------
        float
            Blockade radius in micrometres.

        Raises
        ------
        ValueError
            If rabi_freq_mhz is not positive.
        """
        if rabi_freq_mhz <= 0:
            raise ValueError(
                f"Rabi frequency must be positive, got {rabi_freq_mhz}"
            )
        omega_hz = rabi_freq_mhz * 1e6
        return (self.c6_hz_um6 / omega_hz) ** (1.0 / 6.0)

    def vdw_interaction_hz(self, distance_um: float) -> float:
        """Van der Waals interaction energy between two Rydberg atoms.

        V(r) = C6 / r^6  (in Hz)

        Parameters
        ----------
        distance_um : float
            Inter-atomic distance in micrometres.

        Returns
        -------
        float
            Interaction energy in Hz.

        Raises
        ------
        ValueError
            If distance is not positive.
        """
        if distance_um <= 0:
            raise ValueError(
                f"Distance must be positive, got {distance_um}"
            )
        return self.c6_hz_um6 / distance_um**6

    def blockade_fidelity(
        self, distance_um: float, rabi_freq_mhz: float
    ) -> float:
        """Estimate the blockade fidelity for a CZ gate.

        The blockade error scales as (Omega / V)^2 where V is the
        interaction energy and Omega is the Rabi frequency.

        Parameters
        ----------
        distance_um : float
            Inter-atomic distance in micrometres.
        rabi_freq_mhz : float
            Rabi frequency in MHz.

        Returns
        -------
        float
            Estimated fidelity contribution from blockade imperfection.
        """
        omega_hz = rabi_freq_mhz * 1e6
        v_hz = self.vdw_interaction_hz(distance_um)
        if v_hz == 0:
            return 0.0
        blockade_error = (omega_hz / v_hz) ** 2
        return max(0.0, 1.0 - blockade_error)

    def thermal_dephasing_rate_hz(
        self, rabi_freq_mhz: float, trap_freq_khz: float = 100.0
    ) -> float:
        """Dephasing rate from thermal atomic motion.

        For two-photon Rydberg excitation with counter-propagating beams,
        the effective wavevector is the difference k_eff ~ |k1 - k2|.
        This is typically 10-100x smaller than a single-photon k-vector.

        Additionally, atoms confined in optical tweezers experience
        Lamb-Dicke suppression: the effective Doppler shift is reduced
        by the Lamb-Dicke parameter eta = k_eff * x_zpf where
        x_zpf = sqrt(hbar / (2 * m * omega_trap)).

        Parameters
        ----------
        rabi_freq_mhz : float
            Rabi frequency (sets gate time context).
        trap_freq_khz : float
            Trap frequency in kHz (default 100 kHz for typical tweezers).

        Returns
        -------
        float
            Dephasing rate in Hz.
        """
        v_rms = math.sqrt(
            K_BOLTZMANN * self.temperature_uk * 1e-6 / self.mass_kg
        )
        # Two-photon effective k-vector (counter-propagating beams):
        # k_eff ~ 2*pi * |1/lambda_trap - 1/lambda_rydberg|
        # This is much smaller than single-photon k for near-degenerate beams.
        # Typical effective k for Rb (780nm + 480nm two-photon):
        # k_eff ~ 2*pi * (1/480e-9 - 1/780e-9) ~ 5e6 m^-1
        k_trap = 2.0 * math.pi / (self.wavelength_trap_nm * 1e-9)
        k_rydberg = 2.0 * math.pi / (self.wavelength_rydberg_nm * 1e-9)
        k_eff = abs(k_rydberg - k_trap)

        # Lamb-Dicke suppression from trap confinement
        omega_trap = trap_freq_khz * 1e3 * 2.0 * math.pi  # rad/s
        if omega_trap > 0:
            x_zpf = math.sqrt(HBAR / (2.0 * self.mass_kg * omega_trap))
            eta = k_eff * x_zpf
        else:
            eta = 1.0

        # Doppler dephasing rate suppressed by Lamb-Dicke parameter
        doppler_rate = k_eff * v_rms
        return doppler_rate * eta


# ---------------------------------------------------------------------------
# Preset atom species (class-level singletons)
# ---------------------------------------------------------------------------

AtomSpecies.RB87 = AtomSpecies(  # type: ignore[attr-defined]
    name="87Rb",
    mass_amu=86.909180527,
    nuclear_spin=1.5,
    wavelength_trap_nm=810.0,
    wavelength_rydberg_nm=480.0,  # two-photon: 780nm + 480nm
    rydberg_state_n=70,
    c6_hz_um6=862.69e9,  # n=70, |70S> state, Saffman et al.
    rydberg_lifetime_us=150.0,  # at 300K with BBR
    scattering_rate_trap_hz=10.0,
    ground_state_lifetime_s=4.0,  # hyperfine qubit T2 ~ seconds
    t1_s=100.0,  # limited by vacuum lifetime
    t2_s=4.0,  # Ramsey coherence
    loading_probability=0.5,
    temperature_uk=10.0,
)

AtomSpecies.CS133 = AtomSpecies(  # type: ignore[attr-defined]
    name="133Cs",
    mass_amu=132.905451932,
    nuclear_spin=3.5,
    wavelength_trap_nm=1064.0,
    wavelength_rydberg_nm=319.0,  # two-photon: 895nm + 510nm effective
    rydberg_state_n=70,
    c6_hz_um6=1040.0e9,  # n=70, larger polarisability than Rb
    rydberg_lifetime_us=130.0,
    scattering_rate_trap_hz=5.0,
    ground_state_lifetime_s=5.0,
    t1_s=100.0,
    t2_s=5.0,
    loading_probability=0.5,
    temperature_uk=8.0,
)

AtomSpecies.SR87 = AtomSpecies(  # type: ignore[attr-defined]
    name="87Sr",
    mass_amu=86.908877497,
    nuclear_spin=4.5,
    wavelength_trap_nm=813.4,  # magic wavelength for clock transition
    wavelength_rydberg_nm=317.0,  # single-photon Rydberg excitation
    rydberg_state_n=61,
    c6_hz_um6=300.0e9,  # n=61, Madjarov et al.
    rydberg_lifetime_us=180.0,
    scattering_rate_trap_hz=0.1,  # magic wavelength: very low scattering
    ground_state_lifetime_s=100.0,  # nuclear spin qubit: very long T2
    t1_s=1000.0,
    t2_s=100.0,  # clock transition: exceptional coherence
    loading_probability=0.5,
    temperature_uk=5.0,
)

AtomSpecies.YB171 = AtomSpecies(  # type: ignore[attr-defined]
    name="171Yb",
    mass_amu=170.936323,
    nuclear_spin=0.5,
    wavelength_trap_nm=532.0,
    wavelength_rydberg_nm=302.0,  # two-photon via 6P state
    rydberg_state_n=50,
    c6_hz_um6=200.0e9,  # n=50, Ma et al. 2023
    rydberg_lifetime_us=100.0,
    scattering_rate_trap_hz=20.0,
    ground_state_lifetime_s=10.0,
    t1_s=100.0,
    t2_s=10.0,
    loading_probability=0.5,
    temperature_uk=10.0,
)


# Convenience list of all built-in species
ALL_SPECIES: list[AtomSpecies] = [
    AtomSpecies.RB87,
    AtomSpecies.CS133,
    AtomSpecies.SR87,
    AtomSpecies.YB171,
]
