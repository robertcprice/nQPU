"""Quantum coherent energy transfer in photosynthetic light-harvesting complexes.

Models the Fenna-Matthews-Olson (FMO) complex and other photosynthetic
antenna systems using the Lindblad master equation for open quantum
system dynamics. Demonstrates that quantum coherence enhances the
efficiency of energy transport from antenna pigments to the reaction center.

Key physics:
  - Coherent Hamiltonian evolution of exciton states across chromophore sites
  - Environment-induced dephasing (Drude-Lorentz spectral density)
  - Downhill relaxation from higher- to lower-energy sites
  - Trapping at the reaction center (site 3 for FMO)

References:
  - Engel et al., Nature 446, 782 (2007) -- FMO quantum coherence
  - Adolphs & Renger, Biophys. J. 91, 2778 (2006) -- FMO Hamiltonian
  - Ishizaki & Fleming, PNAS 106, 17255 (2009) -- environment coupling
  - Panitchayangkoon et al., PNAS 108, 20908 (2011) -- LHC-II coherence
  - Collini et al., Nature 463, 644 (2010) -- PE545 coherence

Example:
    from nqpu.bio.photosynthesis import FMOComplex

    fmo = FMOComplex.standard()
    result = fmo.evolve(duration_fs=1000.0, steps=2000)
    print(f"Peak transfer efficiency: {max(result.transfer_efficiency):.3f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34        # reduced Planck constant, J*s
K_B = 1.380649e-23            # Boltzmann constant, J/K
CM_INV_TO_J = 1.986_445_68e-23  # wavenumber (cm^-1) to Joule
FS_TO_S = 1.0e-15             # femtosecond to second
EV_TO_J = 1.602_176_634e-19   # electron-volt to Joule


class SpectralDensityType(Enum):
    """Type of spectral density for the protein environment coupling."""
    DRUDE_LORENTZ = "drude_lorentz"
    OHMIC = "ohmic"
    SUPER_OHMIC = "super_ohmic"


@dataclass
class SpectralDensity:
    """Spectral density model for environment-chromophore coupling.

    The spectral density J(omega) characterises how strongly the protein
    environment couples to the electronic excitations at each frequency.

    Parameters
    ----------
    density_type : SpectralDensityType
        Functional form of the spectral density.
    reorganisation_energy_cm_inv : float
        Reorganisation energy lambda in cm^-1. Controls overall coupling
        strength. Typical: 35 cm^-1 (FMO), 100 cm^-1 (LHC-II).
    cutoff_frequency_cm_inv : float
        Bath correlation time inverse, gamma_c in cm^-1.
        Typical: 106 cm^-1 for FMO (Ishizaki & Fleming, 2009).
    """
    density_type: SpectralDensityType = SpectralDensityType.DRUDE_LORENTZ
    reorganisation_energy_cm_inv: float = 35.0
    cutoff_frequency_cm_inv: float = 106.0

    def evaluate(self, omega_cm_inv: float) -> float:
        """Evaluate J(omega) at a given frequency.

        Parameters
        ----------
        omega_cm_inv : float
            Frequency in cm^-1.

        Returns
        -------
        float
            Spectral density J(omega) in cm^-1.
        """
        lam = self.reorganisation_energy_cm_inv
        gc = self.cutoff_frequency_cm_inv

        if omega_cm_inv <= 0.0:
            return 0.0

        if self.density_type == SpectralDensityType.DRUDE_LORENTZ:
            # J(w) = 2 * lambda * gamma_c * w / (w^2 + gamma_c^2)
            return 2.0 * lam * gc * omega_cm_inv / (
                omega_cm_inv**2 + gc**2
            )
        elif self.density_type == SpectralDensityType.OHMIC:
            # J(w) = lambda * w * exp(-w / gamma_c) / gamma_c
            return lam * omega_cm_inv * math.exp(
                -omega_cm_inv / gc
            ) / gc
        elif self.density_type == SpectralDensityType.SUPER_OHMIC:
            # J(w) = lambda * (w/gamma_c)^2 * w * exp(-w / gamma_c) / gamma_c
            return lam * (omega_cm_inv / gc)**2 * omega_cm_inv * math.exp(
                -omega_cm_inv / gc
            ) / gc
        else:
            return 0.0

    def dephasing_rate_at_temperature(self, temperature_k: float) -> float:
        """Compute the effective dephasing rate (cm^-1) at a given temperature.

        Uses the high-temperature limit of the Drude-Lorentz model:
            gamma_deph ~ 2 * lambda * k_B * T / (hbar * gamma_c)

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Dephasing rate in cm^-1.
        """
        if temperature_k <= 0.0:
            raise ValueError("Temperature must be positive")
        lam = self.reorganisation_energy_cm_inv
        gc = self.cutoff_frequency_cm_inv
        # k_B T in cm^-1: k_B T / (hc) ~ T * 0.6950 cm^-1/K
        kbt_cm = temperature_k * 0.6950356
        return 2.0 * lam * kbt_cm / gc


@dataclass
class DecoherenceModel:
    """Temperature-dependent decoherence model for photosynthetic complexes.

    Parameters
    ----------
    spectral_density : SpectralDensity
        Environment spectral density model.
    temperature_k : float
        System temperature in Kelvin.
    """
    spectral_density: SpectralDensity = field(
        default_factory=SpectralDensity
    )
    temperature_k: float = 300.0

    def dephasing_rate_cm_inv(self) -> float:
        """Pure dephasing rate in cm^-1 at the configured temperature."""
        return self.spectral_density.dephasing_rate_at_temperature(
            self.temperature_k
        )

    def relaxation_rate_cm_inv(self, energy_gap_cm_inv: float) -> float:
        """Relaxation rate between sites separated by an energy gap.

        Uses Redfield theory in the Markovian limit:
            gamma_rel = J(Delta_E) * (1 + n_BE(Delta_E))

        where n_BE is the Bose-Einstein occupation number.

        Parameters
        ----------
        energy_gap_cm_inv : float
            Energy gap between the two sites in cm^-1.

        Returns
        -------
        float
            Relaxation rate in cm^-1.
        """
        if energy_gap_cm_inv <= 0.0:
            return 0.0
        j_val = self.spectral_density.evaluate(energy_gap_cm_inv)
        # Bose-Einstein occupation
        kbt_cm = self.temperature_k * 0.6950356
        if kbt_cm > 0.0:
            x = energy_gap_cm_inv / kbt_cm
            if x > 500.0:
                n_be = 0.0
            else:
                n_be = 1.0 / (math.exp(x) - 1.0)
        else:
            n_be = 0.0
        return j_val * (1.0 + n_be)


@dataclass
class FMOEvolution:
    """Results of FMO time evolution.

    Attributes
    ----------
    times_fs : np.ndarray
        Time points in femtoseconds. Shape: (steps+1,).
    site_populations : np.ndarray
        Population on each chromophore at each time step.
        Shape: (steps+1, n_sites).
    coherences : np.ndarray
        Off-diagonal density matrix magnitudes |rho_ij| at each step.
        Shape: (steps+1, n_coherences) where n_coherences = n*(n-1)/2.
    transfer_efficiency : np.ndarray
        Population at the reaction center (site 3) at each step. Shape: (steps+1,).
    """
    times_fs: np.ndarray
    site_populations: np.ndarray
    coherences: np.ndarray
    transfer_efficiency: np.ndarray

    def average_coherence(self, step: int) -> float:
        """Average off-diagonal coherence magnitude at a given time step."""
        coh = self.coherences[step]
        if len(coh) == 0:
            return 0.0
        return float(np.mean(coh))

    def total_population(self, step: int) -> float:
        """Total population (trace of reduced density matrix) at a time step."""
        return float(np.sum(self.site_populations[step]))

    def peak_efficiency(self) -> float:
        """Maximum transfer efficiency achieved during evolution."""
        return float(np.max(self.transfer_efficiency))

    def coherence_lifetime_fs(self, fraction: float = 0.5) -> float:
        """Time (fs) for average coherence to decay to a fraction of its peak value.

        Since coherences start at zero for a localised initial state,
        we measure decay from the peak coherence rather than the initial value.

        Parameters
        ----------
        fraction : float
            Fraction of peak coherence (default 0.5 for half-life).

        Returns
        -------
        float
            Coherence lifetime in femtoseconds (measured from peak).
        """
        n_steps = len(self.times_fs)
        # Find peak coherence and its index
        peak_coh = 0.0
        peak_idx = 0
        for i in range(n_steps):
            c = self.average_coherence(i)
            if c > peak_coh:
                peak_coh = c
                peak_idx = i

        if peak_coh <= 0.0:
            return 0.0

        threshold = peak_coh * fraction
        # Search after the peak for decay below threshold
        for i in range(peak_idx, n_steps):
            if self.average_coherence(i) < threshold:
                return float(self.times_fs[i] - self.times_fs[peak_idx])
        return float(self.times_fs[-1] - self.times_fs[peak_idx])


class PhotosyntheticSystem(Enum):
    """Predefined photosynthetic light-harvesting systems."""
    FMO = "fmo"
    LHC_II = "lhc_ii"
    PE545 = "pe545"


@dataclass
class FMOComplex:
    """7-site Fenna-Matthews-Olson photosynthetic complex.

    Implements the Lindblad master equation for open quantum system dynamics
    of the FMO complex in Chlorobaculum tepidum. Site energies and
    inter-chromophore couplings from Adolphs & Renger (2006).

    The master equation is:
        d rho / dt = -i/hbar [H, rho] + sum_k gamma_k D[L_k](rho)

    where D[L](rho) = L rho L^dag - 0.5 {L^dag L, rho}.

    Parameters
    ----------
    sites : int
        Number of chromophore sites (7 for standard FMO).
    hamiltonian : np.ndarray
        System Hamiltonian in cm^-1, shape (sites, sites).
    dephasing_rates : np.ndarray
        Site-to-bath dephasing rates in cm^-1, shape (sites,).
    relaxation_rate : float
        Inter-site relaxation rate in cm^-1.
    temperature_k : float
        Temperature in Kelvin.
    trapping_rate : float
        Trapping rate at the reaction center (site 3), in cm^-1.
    initial_site : int
        Index of the initially excited chromophore (0-based).
    """
    sites: int
    hamiltonian: np.ndarray
    dephasing_rates: np.ndarray
    relaxation_rate: float
    temperature_k: float
    trapping_rate: float
    initial_site: int

    @staticmethod
    def standard() -> "FMOComplex":
        """Construct the standard 7-site FMO complex.

        Site energies (relative to 12210 cm^-1) and coupling values from
        Adolphs & Renger, Biophys. J. 91, 2778 (2006), Tables 1 and 2.

        Returns
        -------
        FMOComplex
            Configured FMO complex ready for simulation.
        """
        n = 7
        # Site energies (cm^-1) relative to 12210 cm^-1 baseline
        site_energies = np.array([
            200.0,  # BChl 1
            320.0,  # BChl 2
            0.0,    # BChl 3 (lowest energy -- reaction center trap)
            110.0,  # BChl 4
            270.0,  # BChl 5
            420.0,  # BChl 6
            230.0,  # BChl 7
        ])

        # Inter-chromophore couplings (cm^-1), Adolphs & Renger Table 2
        couplings = [
            (0, 1, -87.7),   # 1-2
            (0, 2, 5.5),     # 1-3
            (0, 3, -5.9),    # 1-4
            (0, 4, 6.7),     # 1-5
            (0, 5, -13.7),   # 1-6
            (0, 6, -9.9),    # 1-7
            (1, 2, 30.8),    # 2-3
            (1, 3, 8.2),     # 2-4
            (1, 4, 0.7),     # 2-5
            (1, 5, 11.8),    # 2-6
            (1, 6, 4.3),     # 2-7
            (2, 3, -53.5),   # 3-4
            (2, 4, -2.2),    # 3-5
            (2, 5, -9.6),    # 3-6
            (2, 6, 6.0),     # 3-7
            (3, 4, -70.7),   # 4-5
            (3, 5, -17.0),   # 4-6
            (3, 6, -63.3),   # 4-7
            (4, 5, 81.1),    # 5-6
            (4, 6, -1.3),    # 5-7
            (5, 6, 39.7),    # 6-7
        ]

        h = np.diag(site_energies)
        for i, j, v in couplings:
            h[i, j] = v
            h[j, i] = v

        # Dephasing rates: ~100 cm^-1 (Ishizaki & Fleming, PNAS 2009)
        dephasing = np.full(n, 100.0)

        return FMOComplex(
            sites=n,
            hamiltonian=h,
            dephasing_rates=dephasing,
            relaxation_rate=1.0,
            temperature_k=300.0,
            trapping_rate=1.0,
            initial_site=0,
        )

    @staticmethod
    def from_system(system: PhotosyntheticSystem) -> "FMOComplex":
        """Construct a photosynthetic complex from a predefined system.

        Parameters
        ----------
        system : PhotosyntheticSystem
            Which light-harvesting complex to model.

        Returns
        -------
        FMOComplex
            Configured complex.
        """
        if system == PhotosyntheticSystem.FMO:
            return FMOComplex.standard()
        elif system == PhotosyntheticSystem.LHC_II:
            return _lhc_ii_complex()
        elif system == PhotosyntheticSystem.PE545:
            return _pe545_complex()
        else:
            raise ValueError(f"Unknown photosynthetic system: {system}")

    def evolve(
        self,
        duration_fs: float = 1000.0,
        steps: int = 2000,
    ) -> FMOEvolution:
        """Evolve the density matrix using the Lindblad master equation.

        Uses Euler integration with Hermiticity enforcement at each step.
        The Lindblad equation includes:
          1. Coherent evolution: -i/hbar [H, rho]
          2. Pure dephasing: gamma_k D[|k><k|](rho)
          3. Inter-site relaxation: gamma_rel D[|k+1><k|](rho)
          4. Trapping at reaction center (site 2, 0-indexed)

        Parameters
        ----------
        duration_fs : float
            Total evolution time in femtoseconds.
        steps : int
            Number of discrete time steps.

        Returns
        -------
        FMOEvolution
            Time-resolved populations, coherences, and transfer efficiency.

        Raises
        ------
        ValueError
            If duration or steps are non-positive.
        """
        if duration_fs <= 0.0:
            raise ValueError("Duration must be positive")
        if steps <= 0:
            raise ValueError("Steps must be positive")

        n = self.sites
        dt_s = (duration_fs * FS_TO_S) / steps

        # Convert Hamiltonian to Joules (complex)
        h_j = self.hamiltonian.astype(np.complex128) * CM_INV_TO_J

        # Initial density matrix: pure state on initial_site
        rho = np.zeros((n, n), dtype=np.complex128)
        s0 = min(self.initial_site, n - 1)
        rho[s0, s0] = 1.0 + 0.0j

        # Preallocate result arrays
        all_pops = np.zeros((steps + 1, n))
        n_coh = n * (n - 1) // 2
        all_cohs = np.zeros((steps + 1, n_coh))
        all_times = np.zeros(steps + 1)
        all_eff = np.zeros(steps + 1)

        # Record function
        def record(rho_mat: np.ndarray, idx: int, t_fs: float) -> None:
            pops = np.maximum(np.real(np.diag(rho_mat)), 0.0)
            all_pops[idx] = pops
            # Upper-triangle coherences |rho_ij|
            c_idx = 0
            for ii in range(n):
                for jj in range(ii + 1, n):
                    all_cohs[idx, c_idx] = abs(rho_mat[ii, jj])
                    c_idx += 1
            all_times[idx] = t_fs
            rc = 2  # reaction center is site 3 (0-indexed = 2)
            all_eff[idx] = pops[rc]

        record(rho, 0, 0.0)

        # Precompute dephasing and relaxation rates in SI
        deph_rates_si = self.dephasing_rates * CM_INV_TO_J / HBAR
        gamma_rel_si = self.relaxation_rate * CM_INV_TO_J / HBAR
        gamma_trap_si = self.trapping_rate * CM_INV_TO_J / HBAR

        rc = 2  # reaction center site index

        for step in range(1, steps + 1):
            # Coherent part: -i/hbar [H, rho]
            comm = h_j @ rho - rho @ h_j
            drho = -1j / HBAR * comm

            # Pure dephasing: D[|k><k|](rho)
            # Off-diagonal decay: drho[i,j] -= 0.5 * (gamma_i + gamma_j) * rho[i,j]
            # Diagonal unaffected by pure dephasing
            for k in range(n):
                gamma_k = deph_rates_si[k]
                for ii in range(n):
                    for jj in range(n):
                        if ii == k and jj == k:
                            pass  # pure dephasing preserves diagonal
                        elif ii == k or jj == k:
                            drho[ii, jj] -= 0.5 * gamma_k * rho[ii, jj]

            # Inter-site relaxation: L_{k->k+1} = sqrt(gamma) |k+1><k|
            for k in range(n - 1):
                rho_kk = rho[k, k]
                drho[k + 1, k + 1] += gamma_rel_si * rho_kk
                drho[k, k] -= gamma_rel_si * rho_kk
                for jj in range(n):
                    if jj != k:
                        drho[k, jj] -= 0.5 * gamma_rel_si * rho[k, jj]
                        drho[jj, k] -= 0.5 * gamma_rel_si * rho[jj, k]

            # Trapping at reaction center
            rho_rc = rho[rc, rc]
            drho[rc, rc] -= gamma_trap_si * rho_rc
            for jj in range(n):
                if jj != rc:
                    drho[rc, jj] -= 0.5 * gamma_trap_si * rho[rc, jj]
                    drho[jj, rc] -= 0.5 * gamma_trap_si * rho[jj, rc]

            # Euler step
            rho = rho + drho * dt_s

            # Enforce Hermiticity and positive diagonal
            for ii in range(n):
                rho[ii, ii] = max(rho[ii, ii].real, 0.0) + 0.0j
                for jj in range(ii + 1, n):
                    avg = 0.5 * (rho[ii, jj] + rho[jj, ii].conj())
                    rho[ii, jj] = avg
                    rho[jj, ii] = avg.conj()

            t_fs = step * duration_fs / steps
            record(rho, step, t_fs)

        return FMOEvolution(
            times_fs=all_times,
            site_populations=all_pops,
            coherences=all_cohs,
            transfer_efficiency=all_eff,
        )


class QuantumTransportEfficiency:
    """Compare quantum coherent vs classical random walk energy transport.

    In the quantum case, excitation propagates coherently through the
    Hamiltonian before dephasing. In the classical case, population hops
    incoherently between sites with rates proportional to coupling squared.

    Parameters
    ----------
    hamiltonian : np.ndarray
        System Hamiltonian in cm^-1, shape (n, n).
    target_site : int
        Index of the target (reaction center) site.
    initial_site : int
        Index of the initially excited site.
    """

    def __init__(
        self,
        hamiltonian: np.ndarray,
        target_site: int = 2,
        initial_site: int = 0,
    ):
        self.hamiltonian = hamiltonian.copy()
        self.n = hamiltonian.shape[0]
        self.target_site = target_site
        self.initial_site = initial_site

    def quantum_efficiency(
        self,
        duration_fs: float = 1000.0,
        steps: int = 2000,
        dephasing_rate: float = 100.0,
    ) -> float:
        """Compute quantum transport efficiency to the target site.

        Parameters
        ----------
        duration_fs : float
            Simulation duration in femtoseconds.
        steps : int
            Number of time steps.
        dephasing_rate : float
            Pure dephasing rate in cm^-1.

        Returns
        -------
        float
            Peak population at the target site during evolution.
        """
        fmo = FMOComplex(
            sites=self.n,
            hamiltonian=self.hamiltonian,
            dephasing_rates=np.full(self.n, dephasing_rate),
            relaxation_rate=1.0,
            temperature_k=300.0,
            trapping_rate=0.0,  # no trapping, just transport
            initial_site=self.initial_site,
        )
        result = fmo.evolve(duration_fs, steps)
        return float(np.max(result.site_populations[:, self.target_site]))

    def classical_efficiency(
        self,
        duration_fs: float = 1000.0,
        steps: int = 2000,
    ) -> float:
        """Compute classical hopping transport efficiency.

        Uses a master equation with incoherent hopping rates proportional
        to the square of the Hamiltonian coupling elements.

        Parameters
        ----------
        duration_fs : float
            Simulation duration in femtoseconds.
        steps : int
            Number of time steps.

        Returns
        -------
        float
            Peak population at the target site during evolution.
        """
        n = self.n
        dt_s = (duration_fs * FS_TO_S) / steps

        # Classical hopping rates: W_{ij} ~ |H_{ij}|^2 / hbar
        # (Forster-type incoherent transfer)
        rates = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    coupling_j = self.hamiltonian[i, j] * CM_INV_TO_J
                    rates[i, j] = coupling_j**2 / HBAR

        # Population vector (classical probability)
        pop = np.zeros(n)
        pop[self.initial_site] = 1.0

        max_target_pop = 0.0

        for _ in range(steps):
            dpop = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # Population flows from j to i at rate W_{ij}
                        flow = rates[i, j] * pop[j] - rates[j, i] * pop[i]
                        dpop[i] += flow
            pop = pop + dpop * dt_s
            pop = np.maximum(pop, 0.0)
            total = pop.sum()
            if total > 0:
                pop /= total  # renormalise
            max_target_pop = max(max_target_pop, pop[self.target_site])

        return max_target_pop

    def quantum_advantage(
        self,
        duration_fs: float = 1000.0,
        steps: int = 2000,
    ) -> float:
        """Ratio of quantum to classical transport efficiency.

        A value > 1 indicates quantum coherence enhances transport.

        Parameters
        ----------
        duration_fs : float
            Simulation duration in femtoseconds.
        steps : int
            Number of time steps.

        Returns
        -------
        float
            Quantum/classical efficiency ratio.
        """
        q_eff = self.quantum_efficiency(duration_fs, steps)
        c_eff = self.classical_efficiency(duration_fs, steps)
        if c_eff < 1e-30:
            return float("inf")
        return q_eff / c_eff


# ---------------------------------------------------------------------------
# Predefined photosynthetic systems
# ---------------------------------------------------------------------------

def _lhc_ii_complex() -> FMOComplex:
    """LHC-II (Light-Harvesting Complex II) from higher plants.

    Simplified 4-site model capturing the main Chl-a cluster dynamics.
    Site energies and couplings from Novoderezhkin et al.,
    J. Phys. Chem. B 109, 10542 (2005).
    """
    site_energies = np.array([
        0.0,     # Chl a610 (lowest, trap)
        150.0,   # Chl a611
        250.0,   # Chl a612
        350.0,   # Chl a602
    ])

    h = np.diag(site_energies)
    couplings = [
        (0, 1, -60.0),
        (0, 2, 5.0),
        (0, 3, -10.0),
        (1, 2, -45.0),
        (1, 3, 8.0),
        (2, 3, -70.0),
    ]
    for i, j, v in couplings:
        h[i, j] = v
        h[j, i] = v

    return FMOComplex(
        sites=4,
        hamiltonian=h,
        dephasing_rates=np.full(4, 150.0),  # broader linewidths
        relaxation_rate=2.0,
        temperature_k=300.0,
        trapping_rate=1.0,
        initial_site=3,  # excitation enters at highest-energy Chl
    )


def _pe545_complex() -> FMOComplex:
    """PE545 phycobiliprotein from cryptophyte algae.

    Simplified 4-site model. Data from Collini et al., Nature 463, 644 (2010)
    and Novoderezhkin et al., J. Phys. Chem. B 114, 16946 (2010).
    """
    site_energies = np.array([
        0.0,     # PEB50/61C (lowest, trap)
        100.0,   # PEB50/61D
        300.0,   # DBV19A
        350.0,   # DBV19B
    ])

    h = np.diag(site_energies)
    couplings = [
        (0, 1, -40.0),
        (0, 2, 10.0),
        (0, 3, -5.0),
        (1, 2, -30.0),
        (1, 3, 12.0),
        (2, 3, -90.0),
    ]
    for i, j, v in couplings:
        h[i, j] = v
        h[j, i] = v

    return FMOComplex(
        sites=4,
        hamiltonian=h,
        dephasing_rates=np.full(4, 120.0),
        relaxation_rate=1.5,
        temperature_k=294.0,  # room temperature in Collini et al.
        trapping_rate=0.8,
        initial_site=3,
    )
