"""Quantum vibration theory of olfaction (Turin hypothesis).

Models the proposed mechanism by which olfactory receptors detect molecular
vibrations through inelastic electron tunneling spectroscopy (IETS). An
electron tunnels from a donor site to an acceptor site in the receptor,
with the tunneling rate resonantly enhanced when the odorant's vibrational
frequency matches the donor-acceptor energy gap.

Key physics:
  - Phonon-assisted inelastic electron tunneling (IETS)
  - Fermi golden rule transition rates with Lorentzian broadening
  - Thermal occupation of vibrational modes
  - Isotope effects on vibrational frequencies (deuteration test)
  - Selectivity through resonance matching

References:
  - Turin, Chem. Senses 21, 773 (1996) -- vibration theory of olfaction
  - Turin, J. Theor. Biol. 216, 367 (2002) -- detailed IETS model
  - Brookes et al., Phys. Rev. Lett. 98, 038101 (2007) -- theoretical support
  - Franco et al., PNAS 108, E116 (2011) -- phonon-assisted tunneling model
  - Gane et al., PLoS ONE 8, e55780 (2013) -- deuterium isotope experiment

Example:
    from nqpu.bio.olfaction import QuantumNose, ODORANTS

    nose = QuantumNose.default()
    aceto = ODORANTS["acetophenone"]
    rate = nose.tunneling_rate(aceto.primary_frequency_cm_inv)
    print(f"Detection rate: {rate:.2e} s^-1")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34             # reduced Planck constant, J*s
K_B = 1.380649e-23                # Boltzmann constant, J/K
EV_TO_J = 1.602_176_634e-19       # electron-volt to Joule
CM_INV_TO_J = 1.986_445_68e-23    # wavenumber (cm^-1) to Joule
PI = math.pi


@dataclass
class MolecularVibration:
    """Harmonic oscillator model for an odorant molecular vibration mode.

    Parameters
    ----------
    frequency_cm_inv : float
        Vibrational frequency in cm^-1.
    reduced_mass_amu : float
        Reduced mass of the vibrational mode in atomic mass units.
    ir_intensity : float
        Infrared intensity (arbitrary units, for relative comparison).
    label : str
        Human-readable label for the vibration mode.
    """
    frequency_cm_inv: float
    reduced_mass_amu: float = 1.0
    ir_intensity: float = 1.0
    label: str = ""

    def energy_ev(self) -> float:
        """Vibrational quantum energy in electron-volts."""
        return self.frequency_cm_inv * CM_INV_TO_J / EV_TO_J

    def energy_j(self) -> float:
        """Vibrational quantum energy in Joules."""
        return self.frequency_cm_inv * CM_INV_TO_J

    def thermal_occupation(self, temperature_k: float) -> float:
        """Bose-Einstein thermal occupation number.

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Average phonon number at thermal equilibrium.
        """
        if temperature_k <= 0.0 or self.frequency_cm_inv <= 0.0:
            return 0.0
        hw = self.frequency_cm_inv * CM_INV_TO_J
        x = hw / (K_B * temperature_k)
        if x > 500.0:
            return 0.0
        return 1.0 / (math.exp(x) - 1.0)

    def deuterated(self) -> "MolecularVibration":
        """Return a deuterated version (frequency scaled by sqrt(m_H/m_D)).

        The isotope shift for C-H to C-D vibrations is approximately
        1/sqrt(2) due to the doubled reduced mass.

        Returns
        -------
        MolecularVibration
            New vibration with deuterium-shifted frequency.
        """
        # For C-H stretch: reduced mass ~ m_H. For C-D: ~ m_D = 2*m_H
        # omega ~ sqrt(k/mu), so omega_D = omega_H * sqrt(mu_H/mu_D)
        mass_ratio = math.sqrt(
            self.reduced_mass_amu / (self.reduced_mass_amu * 2.0)
        )
        return MolecularVibration(
            frequency_cm_inv=self.frequency_cm_inv * mass_ratio,
            reduced_mass_amu=self.reduced_mass_amu * 2.0,
            ir_intensity=self.ir_intensity,
            label=f"{self.label} (deuterated)",
        )


@dataclass
class Odorant:
    """Odorant molecule with vibrational spectrum.

    Parameters
    ----------
    name : str
        Molecule name.
    vibrations : list of MolecularVibration
        Active vibrational modes.
    primary_frequency_cm_inv : float
        The dominant vibrational frequency proposed to determine smell.
    molecular_weight : float
        Molecular weight in g/mol.
    """
    name: str
    vibrations: List[MolecularVibration]
    primary_frequency_cm_inv: float
    molecular_weight: float = 0.0

    def deuterated(self) -> "Odorant":
        """Return a fully deuterated version of the odorant.

        Returns
        -------
        Odorant
            Deuterated molecule with shifted vibrational frequencies.
        """
        deut_vibs = [v.deuterated() for v in self.vibrations]
        # Primary frequency also shifts
        primary_shifted = self.primary_frequency_cm_inv / math.sqrt(2.0)
        return Odorant(
            name=f"d-{self.name}",
            vibrations=deut_vibs,
            primary_frequency_cm_inv=primary_shifted,
            molecular_weight=self.molecular_weight,
        )


@dataclass
class OlfactoryReceptor:
    """Model of an olfactory receptor binding pocket.

    Parameters
    ----------
    donor_energy_ev : float
        Energy of the electron donor state (eV).
    acceptor_energy_ev : float
        Energy of the electron acceptor state (eV).
    pocket_size_nm : float
        Approximate binding pocket diameter (nm).
    """
    donor_energy_ev: float = 0.0
    acceptor_energy_ev: float = -0.2
    pocket_size_nm: float = 1.0

    def energy_gap_ev(self) -> float:
        """Energy gap between donor and acceptor in eV."""
        return abs(self.donor_energy_ev - self.acceptor_energy_ev)

    def resonant_frequency_cm_inv(self) -> float:
        """Resonant odorant frequency matching the energy gap (cm^-1)."""
        gap_j = self.energy_gap_ev() * EV_TO_J
        return gap_j / CM_INV_TO_J


@dataclass
class QuantumNose:
    """Quantum olfaction model based on phonon-assisted electron tunneling.

    Implements Turin's inelastic electron tunneling spectroscopy (IETS) model
    for olfactory discrimination. An electron tunnels between donor and
    acceptor sites in the receptor, with the rate enhanced when the odorant's
    vibrational frequency matches the donor-acceptor energy gap.

    Two-site model Hamiltonian:
        H = E_D |D><D| + E_A |A><A| + V (|D><A| + |A><D|)
            + hbar*w_v (a^dag a + 1/2) + lambda (a + a^dag) |A><A|

    Parameters
    ----------
    receptor : OlfactoryReceptor
        The receptor model.
    coupling_ev : float
        Electronic coupling between donor and acceptor (eV).
    phonon_coupling_ev : float
        Electron-phonon (odorant vibration) coupling strength (eV).
    dissipation_rate_ev : float
        Lorentzian broadening of the resonance (eV).
    temperature_k : float
        Temperature in Kelvin.
    """
    receptor: OlfactoryReceptor
    coupling_ev: float = 0.01
    phonon_coupling_ev: float = 0.02
    dissipation_rate_ev: float = 0.005
    temperature_k: float = 300.0

    @staticmethod
    def default() -> "QuantumNose":
        """Create a quantum nose with default parameters.

        Donor-acceptor gap of 0.2 eV corresponds to ~1600 cm^-1,
        in the biologically relevant fingerprint region.

        Returns
        -------
        QuantumNose
            Default nose model.
        """
        return QuantumNose(receptor=OlfactoryReceptor())

    def energy_gap_ev(self) -> float:
        """Energy gap between donor and acceptor (eV)."""
        return self.receptor.energy_gap_ev()

    def resonant_frequency_cm_inv(self) -> float:
        """Resonant odorant frequency matching the energy gap (cm^-1)."""
        return self.receptor.resonant_frequency_cm_inv()

    def tunneling_rate(self, odorant_freq_cm_inv: float) -> float:
        """Compute the IETS tunneling rate for a given odorant frequency.

        Uses Fermi's golden rule with phonon-assisted tunneling:

            Gamma(w) = (2*pi/hbar) * |V|^2 * lambda^2 * (n+1) * L(w)

        where L(w) is a Lorentzian centered at the energy gap frequency
        with width given by the dissipation rate.

        Parameters
        ----------
        odorant_freq_cm_inv : float
            Vibrational frequency of the odorant molecule in cm^-1.

        Returns
        -------
        float
            Tunneling rate in s^-1.

        Raises
        ------
        ValueError
            If frequency is negative.
        """
        if odorant_freq_cm_inv < 0.0:
            raise ValueError(
                f"Vibrational frequency must be non-negative, "
                f"got {odorant_freq_cm_inv}"
            )

        gap_j = self.energy_gap_ev() * EV_TO_J
        hw_j = odorant_freq_cm_inv * CM_INV_TO_J
        v_j = self.coupling_ev * EV_TO_J
        lambda_j = self.phonon_coupling_ev * EV_TO_J
        gamma_j = self.dissipation_rate_ev * EV_TO_J

        # Thermal occupation of vibrational mode
        n_thermal = 0.0
        if self.temperature_k > 0.0 and hw_j > 0.0:
            x = hw_j / (K_B * self.temperature_k)
            if x < 500.0:
                n_thermal = 1.0 / max(math.exp(x) - 1.0, 1e-30)

        # Phonon emission matrix element: |<n+1|a^dag|n>|^2 = n + 1
        matrix_element_sq = n_thermal + 1.0

        # Lorentzian spectral density (broadened delta function)
        detuning = gap_j - hw_j
        lorentzian = (gamma_j / PI) / (detuning**2 + gamma_j**2)

        # Fermi golden rule rate
        rate = (
            (2.0 * PI / HBAR) * v_j**2 * lambda_j**2
            * matrix_element_sq * lorentzian
        )

        return max(rate, 0.0)

    def spectrum(
        self,
        freq_min: float = 500.0,
        freq_max: float = 3500.0,
        steps: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the tunneling rate spectrum across frequencies.

        Parameters
        ----------
        freq_min : float
            Minimum frequency in cm^-1.
        freq_max : float
            Maximum frequency in cm^-1.
        steps : int
            Number of frequency points.

        Returns
        -------
        tuple of np.ndarray
            (frequencies_cm_inv, rates_per_s)
        """
        freqs = np.linspace(freq_min, freq_max, steps)
        rates = np.array([self.tunneling_rate(f) for f in freqs])
        return freqs, rates

    def selectivity(self, detuning_cm_inv: float = 500.0) -> float:
        """Ratio of on-resonance to off-resonance tunneling rate.

        A higher selectivity means better molecular discrimination.

        Parameters
        ----------
        detuning_cm_inv : float
            Frequency offset from resonance in cm^-1.

        Returns
        -------
        float
            On-resonance / off-resonance rate ratio.
        """
        f_res = self.resonant_frequency_cm_inv()
        rate_on = self.tunneling_rate(f_res)
        rate_off = self.tunneling_rate(f_res + detuning_cm_inv)
        if rate_off < 1e-300:
            return float("inf")
        return rate_on / rate_off

    def is_detected(
        self,
        odorant_freq_cm_inv: float,
        threshold_rate: float = 1.0e10,
    ) -> bool:
        """Check whether an odorant frequency triggers detection.

        Parameters
        ----------
        odorant_freq_cm_inv : float
            Vibrational frequency of the odorant in cm^-1.
        threshold_rate : float
            Minimum tunneling rate for detection (s^-1).

        Returns
        -------
        bool
            True if the tunneling rate exceeds the threshold.
        """
        return self.tunneling_rate(odorant_freq_cm_inv) > threshold_rate


class OdorDiscrimination:
    """Compare classical (shape) vs quantum (vibration) olfaction models.

    The classical lock-and-key model predicts that isotopologues
    (e.g., acetophenone and d-acetophenone) smell identical because
    they have the same shape. The quantum vibration model predicts
    they smell different because deuteration shifts vibrational frequencies.

    Parameters
    ----------
    nose : QuantumNose
        The quantum nose model for IETS-based discrimination.
    """

    def __init__(self, nose: Optional[QuantumNose] = None):
        self.nose = nose or QuantumNose.default()

    def quantum_discriminates(
        self,
        molecule_a: Odorant,
        molecule_b: Odorant,
        threshold_ratio: float = 2.0,
    ) -> bool:
        """Test whether the quantum model can discriminate two molecules.

        Molecules are distinguishable if the ratio of their peak
        tunneling rates differs by more than the threshold.

        Parameters
        ----------
        molecule_a : Odorant
            First molecule.
        molecule_b : Odorant
            Second molecule.
        threshold_ratio : float
            Minimum rate ratio for discrimination.

        Returns
        -------
        bool
            True if the quantum model predicts distinguishable smells.
        """
        rate_a = self.nose.tunneling_rate(molecule_a.primary_frequency_cm_inv)
        rate_b = self.nose.tunneling_rate(molecule_b.primary_frequency_cm_inv)
        if min(rate_a, rate_b) < 1e-300:
            return rate_a != rate_b
        ratio = max(rate_a, rate_b) / min(rate_a, rate_b)
        return ratio > threshold_ratio

    def classical_discriminates(
        self,
        molecule_a: Odorant,
        molecule_b: Odorant,
    ) -> bool:
        """Test whether the classical shape model can discriminate two molecules.

        Isotopologues have identical shapes, so the classical model
        cannot distinguish them.

        Parameters
        ----------
        molecule_a : Odorant
            First molecule.
        molecule_b : Odorant
            Second molecule.

        Returns
        -------
        bool
            True if molecules have different molecular weights (proxy for
            different shapes). False for isotopologues.
        """
        # Same molecular weight (within 1%) implies isotopologue
        if molecule_a.molecular_weight > 0 and molecule_b.molecular_weight > 0:
            ratio = (
                molecule_a.molecular_weight / molecule_b.molecular_weight
            )
            if 0.99 < ratio < 1.01:
                # Similar weight, could still differ in shape
                # But isotopologues (d- prefix) are same shape
                if (
                    molecule_a.name.startswith("d-")
                    or molecule_b.name.startswith("d-")
                ):
                    return False
        return True

    def isotope_test(self, molecule: Odorant) -> dict:
        """Run the deuterium isotope discrimination test.

        Creates a deuterated version of the molecule and tests whether
        each model can distinguish them. This is the key experimental
        test of the quantum vibration theory.

        Parameters
        ----------
        molecule : Odorant
            The molecule to test.

        Returns
        -------
        dict
            Results with keys:
            - 'molecule': original molecule name
            - 'deuterated': deuterated molecule name
            - 'freq_normal': original primary frequency (cm^-1)
            - 'freq_deuterated': deuterated primary frequency (cm^-1)
            - 'rate_normal': tunneling rate for normal molecule (s^-1)
            - 'rate_deuterated': tunneling rate for deuterated molecule (s^-1)
            - 'quantum_discriminates': bool
            - 'classical_discriminates': bool
        """
        d_molecule = molecule.deuterated()
        rate_normal = self.nose.tunneling_rate(
            molecule.primary_frequency_cm_inv
        )
        rate_deut = self.nose.tunneling_rate(
            d_molecule.primary_frequency_cm_inv
        )
        return {
            "molecule": molecule.name,
            "deuterated": d_molecule.name,
            "freq_normal": molecule.primary_frequency_cm_inv,
            "freq_deuterated": d_molecule.primary_frequency_cm_inv,
            "rate_normal": rate_normal,
            "rate_deuterated": rate_deut,
            "quantum_discriminates": self.quantum_discriminates(
                molecule, d_molecule
            ),
            "classical_discriminates": self.classical_discriminates(
                molecule, d_molecule
            ),
        }


# ---------------------------------------------------------------------------
# Predefined odorant molecules
# ---------------------------------------------------------------------------

def _acetophenone() -> Odorant:
    """Acetophenone (C6H5COCH3) -- cherry/almond smell.

    Key vibration: C=O stretch at ~1680 cm^-1.
    """
    return Odorant(
        name="acetophenone",
        vibrations=[
            MolecularVibration(1680.0, 6.86, 1.0, "C=O stretch"),
            MolecularVibration(3060.0, 1.08, 0.3, "aromatic C-H stretch"),
            MolecularVibration(1580.0, 6.0, 0.5, "aromatic C=C stretch"),
            MolecularVibration(1260.0, 6.0, 0.4, "C-C stretch"),
        ],
        primary_frequency_cm_inv=1680.0,
        molecular_weight=120.15,
    )


def _benzaldehyde() -> Odorant:
    """Benzaldehyde (C6H5CHO) -- bitter almond smell.

    Key vibration: C=O stretch at ~1700 cm^-1.
    """
    return Odorant(
        name="benzaldehyde",
        vibrations=[
            MolecularVibration(1700.0, 6.86, 1.0, "C=O stretch"),
            MolecularVibration(2820.0, 1.08, 0.5, "aldehyde C-H stretch"),
            MolecularVibration(3060.0, 1.08, 0.3, "aromatic C-H stretch"),
            MolecularVibration(1600.0, 6.0, 0.4, "aromatic C=C stretch"),
        ],
        primary_frequency_cm_inv=1700.0,
        molecular_weight=106.12,
    )


def _musk() -> Odorant:
    """Muscone-like macrocyclic ketone -- musky smell.

    Key vibration: C-H bending modes around ~1400 cm^-1.
    Turin proposed that musk odor is associated with a specific
    vibrational frequency range (1380-1550 cm^-1).
    """
    return Odorant(
        name="muscone",
        vibrations=[
            MolecularVibration(1450.0, 6.5, 1.0, "C-H bend"),
            MolecularVibration(1720.0, 6.86, 0.6, "C=O stretch"),
            MolecularVibration(2930.0, 1.08, 0.4, "alkyl C-H stretch"),
        ],
        primary_frequency_cm_inv=1450.0,
        molecular_weight=238.41,
    )


def _hydrogen_sulfide() -> Odorant:
    """Hydrogen sulfide (H2S) -- rotten egg smell.

    Key vibration: S-H stretch at ~2600 cm^-1.
    The borane/sulfide test is a classic example: boranes (B-H ~2500 cm^-1)
    and thiols (S-H ~2600 cm^-1) have similar vibrational frequencies and
    both have a sulfurous smell, despite completely different chemistry.
    """
    return Odorant(
        name="hydrogen_sulfide",
        vibrations=[
            MolecularVibration(2611.0, 0.97, 1.0, "S-H stretch"),
            MolecularVibration(1183.0, 0.97, 0.3, "H-S-H bend"),
        ],
        primary_frequency_cm_inv=2611.0,
        molecular_weight=34.08,
    )


ODORANTS: Dict[str, Odorant] = {
    "acetophenone": _acetophenone(),
    "benzaldehyde": _benzaldehyde(),
    "muscone": _musk(),
    "hydrogen_sulfide": _hydrogen_sulfide(),
}
"""Predefined odorant molecules.

Keys: 'acetophenone', 'benzaldehyde', 'muscone', 'hydrogen_sulfide'.
"""
