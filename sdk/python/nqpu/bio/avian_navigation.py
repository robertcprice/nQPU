"""Radical pair mechanism for avian magnetoreception.

Models the cryptochrome-based quantum compass in migratory birds. A radical
pair (FAD^.- / Trp^.+) is created in the singlet state and undergoes
singlet-triplet interconversion driven by competing Zeeman and hyperfine
interactions. The singlet product yield depends on the angle of Earth's
magnetic field relative to the radical pair axis, providing directional
information for navigation.

Key physics:
  - Two-electron spin Hamiltonian with Zeeman + hyperfine + exchange terms
  - Anisotropic hyperfine tensor (A_parallel != A_perpendicular)
  - Singlet-triplet interconversion in an 8-dimensional Hilbert space
  - Angular-dependent singlet yield (inclination compass)
  - Decoherence effects on compass sensitivity

References:
  - Schulten et al., Z. Phys. Chem. 111, 1 (1978) -- original RP proposal
  - Ritz et al., Biophys. J. 78, 707 (2000) -- cryptochrome model
  - Hore & Mouritsen, Annu. Rev. Biophys. 45, 299 (2016) -- review
  - Maeda et al., Nature 453, 387 (2008) -- experimental support
  - Xu et al., Nature 594, 535 (2021) -- cryptochrome 4 evidence

Example:
    from nqpu.bio.avian_navigation import RadicalPair, CryptochromeModel

    rp = RadicalPair.cryptochrome()
    yield_0 = rp.singlet_yield(0.0)
    yield_90 = rp.singlet_yield(math.pi / 2)
    print(f"Singlet yield at 0 deg: {yield_0:.3f}")
    print(f"Singlet yield at 90 deg: {yield_90:.3f}")
    print(f"Compass anisotropy: {abs(yield_0 - yield_90):.3f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34            # reduced Planck constant, J*s
BOHR_MAGNETON = 9.274_010_078_3e-24  # Bohr magneton, J/T
G_ELECTRON = 2.002_319_304       # free-electron g-factor
UT_TO_T = 1.0e-6                 # micro-Tesla to Tesla
PI = math.pi


@dataclass
class RadicalPair:
    """Two-electron radical pair in an external magnetic field.

    Models singlet-triplet interconversion under the combined influence
    of the Zeeman interaction (external B field) and hyperfine coupling
    to a single nuclear spin (I=1/2) on one of the radicals.

    The spin Hamiltonian in the 8-dimensional Hilbert space
    (electron_1 x electron_2 x nucleus) is:

        H = omega * cos(theta) * (S1z + S2z)
          + omega * sin(theta) * (S1x + S2x)
          + A_z * S1z * Iz + A_xy * 0.5 * (S1+ I- + S1- I+)
          + J * (S1 . S2)

    where omega = g * mu_B * B / hbar is the Larmor frequency.

    Parameters
    ----------
    hyperfine_coupling_mhz : float
        Isotropic hyperfine coupling constant in MHz.
    hyperfine_anisotropy : float
        Fractional anisotropy of hyperfine tensor. 0 = isotropic,
        ~0.3 = typical for nitrogen in FAD.
        A_z = A * (1 + anisotropy), A_xy = A * (1 - anisotropy/2).
    exchange_coupling_mhz : float
        Exchange coupling J between the two electrons (MHz).
    field_strength_ut : float
        External magnetic field strength in micro-Tesla.
    k_s : float
        Singlet recombination rate (MHz).
    k_t : float
        Triplet recombination rate (MHz).
    lifetime_us : float
        Radical pair lifetime in microseconds.
    """
    hyperfine_coupling_mhz: float = 28.0
    hyperfine_anisotropy: float = 0.3
    exchange_coupling_mhz: float = 0.0
    field_strength_ut: float = 50.0
    k_s: float = 1.0
    k_t: float = 1.0
    lifetime_us: float = 1.0

    @staticmethod
    def cryptochrome() -> "RadicalPair":
        """Construct a radical pair with cryptochrome-typical parameters.

        Based on the FAD-tryptophan radical pair in cryptochrome protein.
        Hyperfine coupling of ~28 MHz corresponds to ~1 mT for the
        nitrogen nucleus. Earth's field is ~50 uT.

        Returns
        -------
        RadicalPair
            Configured for avian cryptochrome.
        """
        return RadicalPair(
            hyperfine_coupling_mhz=28.0,
            hyperfine_anisotropy=0.3,
            exchange_coupling_mhz=0.0,
            field_strength_ut=50.0,
            k_s=1.0,
            k_t=1.0,
            lifetime_us=1.0,
        )

    def singlet_yield(self, field_angle_rad: float) -> float:
        """Compute the singlet yield at a given magnetic field angle.

        Evolves the density matrix of the 8-dimensional spin system
        (2 electrons + 1 nucleus, each spin-1/2) using Heun's method
        (2nd-order Runge-Kutta) with recombination losses.

        The field angle theta is measured from the radical pair symmetry
        axis (the axis connecting the two radicals).

        Parameters
        ----------
        field_angle_rad : float
            Angle between the magnetic field and the radical pair axis,
            in radians.

        Returns
        -------
        float
            Singlet yield Phi_S in [0, 1].
        """
        # Larmor frequency (MHz)
        b = self.field_strength_ut * UT_TO_T
        omega = G_ELECTRON * BOHR_MAGNETON * b / (HBAR * 2.0 * PI * 1.0e6)

        a = self.hyperfine_coupling_mhz
        j = self.exchange_coupling_mhz
        theta = field_angle_rad
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        dim = 8

        # Build 8x8 spin Hamiltonian (complex)
        h = np.zeros((dim, dim), dtype=np.complex128)

        # Helper: Sz eigenvalue for qubit in 3-qubit state
        def sz_diag(qubit: int, state: int) -> float:
            bit = (state >> (2 - qubit)) & 1
            return 0.5 if bit == 0 else -0.5

        # Diagonal Zeeman (Sz components for electrons 1 and 2)
        for s in range(dim):
            s1z = sz_diag(0, s)
            s2z = sz_diag(1, s)
            h[s, s] += omega * cos_t * (s1z + s2z)

        # Off-diagonal Zeeman (Sx components)
        # S1x flips qubit 0
        for s in range(dim):
            flipped = s ^ 4  # flip bit 2 (qubit 0 in MSB ordering)
            h[s, flipped] += 0.5 * omega * sin_t
        # S2x flips qubit 1
        for s in range(dim):
            flipped = s ^ 2  # flip bit 1 (qubit 1)
            h[s, flipped] += 0.5 * omega * sin_t

        # Anisotropic hyperfine: A_z * S1z*Iz + A_xy * 0.5*(S1+I- + S1-I+)
        a_z = a * (1.0 + self.hyperfine_anisotropy)
        a_xy = a * (1.0 - self.hyperfine_anisotropy / 2.0)

        # S1z * Iz (diagonal)
        for s in range(dim):
            s1z = sz_diag(0, s)
            nz = sz_diag(2, s)
            h[s, s] += a_z * s1z * nz

        # S1+I- + S1-I+ (flip-flop): e1 and nucleus exchange spin
        for s in range(dim):
            e1_bit = (s >> 2) & 1
            n_bit = s & 1
            if e1_bit == 1 and n_bit == 0:
                # S1+ I- applies: e1 goes down->up, n goes up->down
                target = (s ^ 4) ^ 1
                h[target, s] += 0.5 * a_xy
                h[s, target] += 0.5 * a_xy

        # Exchange coupling: J * (S1 . S2)
        if abs(j) > 1e-15:
            # S1z*S2z (diagonal)
            for s in range(dim):
                s1z = sz_diag(0, s)
                s2z = sz_diag(1, s)
                h[s, s] += j * s1z * s2z
            # S1+S2- + S1-S2+ (flip-flop between electrons)
            for s in range(dim):
                e1_bit = (s >> 2) & 1
                e2_bit = (s >> 1) & 1
                if e1_bit == 1 and e2_bit == 0:
                    target = (s ^ 4) ^ 2
                    h[target, s] += 0.5 * j
                    h[s, target] += 0.5 * j

        # Singlet projection operator on electrons 1 and 2
        # |S> = (|ud> - |du>) / sqrt(2) for the electron pair
        # In 8-dim basis:
        #   |S, n_up>  = (|udu> - |duu>) / sqrt(2) = (|2> - |4>) / sqrt(2)
        #   |S, n_down> = (|udd> - |dud>) / sqrt(2) = (|3> - |5>) / sqrt(2)
        p_s = np.zeros((dim, dim), dtype=np.complex128)
        # |S, n_up><S, n_up|
        p_s[2, 2] += 0.5
        p_s[2, 4] += -0.5
        p_s[4, 2] += -0.5
        p_s[4, 4] += 0.5
        # |S, n_down><S, n_down|
        p_s[3, 3] += 0.5
        p_s[3, 5] += -0.5
        p_s[5, 3] += -0.5
        p_s[5, 5] += 0.5

        # Initial state: singlet with nuclear spin in maximally mixed state
        # rho(0) = P_S / Tr(P_S) = P_S / 2
        rho = p_s * 0.5

        # Time evolution with Heun's method
        n_steps = 5000
        dt_us = 5.0 * self.lifetime_us / n_steps
        # In MHz*us units, the time step is dimensionless

        k_avg = 0.5 * (self.k_s + self.k_t)
        dk = 0.5 * (self.k_s - self.k_t)
        has_selective = abs(dk) > 1e-15

        singlet_yield_integral = 0.0

        for _ in range(n_steps):
            # Tr(P_S * rho)
            tr_ps_rho = np.real(np.trace(p_s @ rho))
            singlet_yield_integral += self.k_s * tr_ps_rho * dt_us

            # Right-hand side of the master equation
            def rhs(rho_in: np.ndarray) -> np.ndarray:
                comm = h @ rho_in - rho_in @ h
                d = -2.0j * PI * comm - k_avg * rho_in
                if has_selective:
                    anti = p_s @ rho_in + rho_in @ p_s
                    d -= 0.5 * dk * anti
                return d

            # Heun's method (predictor-corrector)
            k1 = rhs(rho)
            rho_pred = rho + k1 * dt_us
            k2 = rhs(rho_pred)
            rho = rho + 0.5 * (k1 + k2) * dt_us

        return max(0.0, min(1.0, singlet_yield_integral))

    def angular_response(
        self,
        n_angles: int = 36,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute singlet yield across field angles [0, pi].

        Parameters
        ----------
        n_angles : int
            Number of angle points.

        Returns
        -------
        tuple of np.ndarray
            (angles_rad, singlet_yields)
        """
        angles = np.linspace(0.0, PI, n_angles)
        yields = np.array([self.singlet_yield(a) for a in angles])
        return angles, yields

    def compass_anisotropy(self, n_angles: int = 36) -> float:
        """Compass sensitivity: max - min singlet yield across angles.

        A larger anisotropy indicates a better magnetic compass.

        Parameters
        ----------
        n_angles : int
            Number of angles to sample.

        Returns
        -------
        float
            Max - min singlet yield.
        """
        _, yields = self.angular_response(n_angles)
        return float(np.max(yields) - np.min(yields))


@dataclass
class CryptochromeModel:
    """Detailed cryptochrome protein model with multiple tryptophan residues.

    In the cryptochrome photocycle, a chain of 3-4 tryptophan (Trp) residues
    transfers an electron to the FAD cofactor. Each Trp radical has distinct
    hyperfine tensors depending on its local protein environment and geometry.

    Parameters
    ----------
    species : str
        Bird species name.
    trp_hyperfine_mhz : list of float
        Hyperfine coupling for each Trp residue in MHz.
    trp_anisotropy : list of float
        Hyperfine anisotropy for each Trp residue.
    trp_distance_nm : list of float
        Distance of each Trp from FAD in nm.
    field_strength_ut : float
        External magnetic field in micro-Tesla.
    """
    species: str
    trp_hyperfine_mhz: List[float]
    trp_anisotropy: List[float]
    trp_distance_nm: List[float]
    field_strength_ut: float = 50.0

    @staticmethod
    def european_robin() -> "CryptochromeModel":
        """Cryptochrome 4 model for the European robin (Erithacus rubecula).

        Based on Xu et al., Nature 594, 535 (2021).
        Three tryptophan residues in the electron transfer chain.

        Returns
        -------
        CryptochromeModel
            Robin cryptochrome model.
        """
        return CryptochromeModel(
            species="European robin",
            trp_hyperfine_mhz=[30.0, 25.0, 20.0],
            trp_anisotropy=[0.35, 0.28, 0.22],
            trp_distance_nm=[0.7, 1.1, 1.5],
        )

    @staticmethod
    def homing_pigeon() -> "CryptochromeModel":
        """Cryptochrome model for the homing pigeon (Columba livia).

        Pigeons have a less sensitive compass than robins, modelled
        with reduced hyperfine anisotropy.

        Returns
        -------
        CryptochromeModel
            Pigeon cryptochrome model.
        """
        return CryptochromeModel(
            species="homing pigeon",
            trp_hyperfine_mhz=[26.0, 22.0, 18.0],
            trp_anisotropy=[0.25, 0.20, 0.15],
            trp_distance_nm=[0.8, 1.2, 1.6],
        )

    def effective_radical_pair(self, trp_index: int = 0) -> RadicalPair:
        """Create a RadicalPair model for a specific Trp residue.

        Parameters
        ----------
        trp_index : int
            Index of the tryptophan residue (0 = closest to FAD).

        Returns
        -------
        RadicalPair
            Radical pair model for the specified Trp-FAD pair.

        Raises
        ------
        IndexError
            If trp_index is out of range.
        """
        if trp_index < 0 or trp_index >= len(self.trp_hyperfine_mhz):
            raise IndexError(
                f"Trp index {trp_index} out of range "
                f"[0, {len(self.trp_hyperfine_mhz) - 1}]"
            )

        # Lifetime decreases with distance (electron transfer rate)
        base_lifetime = 1.0  # us
        distance_factor = (
            self.trp_distance_nm[trp_index] / self.trp_distance_nm[0]
        )
        lifetime = base_lifetime * distance_factor

        return RadicalPair(
            hyperfine_coupling_mhz=self.trp_hyperfine_mhz[trp_index],
            hyperfine_anisotropy=self.trp_anisotropy[trp_index],
            exchange_coupling_mhz=0.0,
            field_strength_ut=self.field_strength_ut,
            k_s=1.0,
            k_t=1.0,
            lifetime_us=lifetime,
        )

    def total_compass_anisotropy(self, n_angles: int = 18) -> float:
        """Combined compass anisotropy from all Trp residues.

        The total anisotropy is the weighted sum of individual radical
        pair anisotropies, weighted by the probability of finding the
        radical at each Trp (approximately equal for the dominant pair).

        Parameters
        ----------
        n_angles : int
            Number of angles per radical pair.

        Returns
        -------
        float
            Combined compass anisotropy.
        """
        total = 0.0
        n_trp = len(self.trp_hyperfine_mhz)
        for i in range(n_trp):
            rp = self.effective_radical_pair(i)
            aniso = rp.compass_anisotropy(n_angles)
            # Weight by 1/distance^2 (approximate radical population)
            weight = 1.0 / self.trp_distance_nm[i] ** 2
            total += aniso * weight
        # Normalise weights
        weight_sum = sum(
            1.0 / d**2 for d in self.trp_distance_nm
        )
        return total / weight_sum


class CompassSensitivity:
    """Analysis tools for radical pair compass sensitivity.

    Parameters
    ----------
    radical_pair : RadicalPair
        The radical pair model to analyse.
    """

    def __init__(self, radical_pair: RadicalPair):
        self.rp = radical_pair

    def angular_resolution(self, n_angles: int = 36) -> float:
        """Estimate angular resolution of the compass in degrees.

        Defined as the angular change needed to produce a 1% change
        in singlet yield.

        Parameters
        ----------
        n_angles : int
            Number of angles to sample for the gradient estimate.

        Returns
        -------
        float
            Angular resolution in degrees. Lower is better.
        """
        angles, yields = self.rp.angular_response(n_angles)
        if len(yields) < 2:
            return 180.0

        # Find maximum gradient (d(yield)/d(angle))
        d_yield = np.diff(yields)
        d_angle = np.diff(angles)
        gradients = np.abs(d_yield / d_angle)
        max_gradient = np.max(gradients)

        if max_gradient < 1e-30:
            return 180.0

        # Angle change for 1% yield change
        delta_angle_rad = 0.01 / max_gradient
        return math.degrees(delta_angle_rad)

    def field_strength_sensitivity(
        self,
        field_min_ut: float = 10.0,
        field_max_ut: float = 100.0,
        n_points: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compass anisotropy vs magnetic field strength.

        Parameters
        ----------
        field_min_ut : float
            Minimum field strength in micro-Tesla.
        field_max_ut : float
            Maximum field strength in micro-Tesla.
        n_points : int
            Number of field strength points.

        Returns
        -------
        tuple of np.ndarray
            (field_strengths_ut, anisotropies)
        """
        fields = np.linspace(field_min_ut, field_max_ut, n_points)
        anisotropies = np.zeros(n_points)
        for i, b in enumerate(fields):
            rp = RadicalPair(
                hyperfine_coupling_mhz=self.rp.hyperfine_coupling_mhz,
                hyperfine_anisotropy=self.rp.hyperfine_anisotropy,
                exchange_coupling_mhz=self.rp.exchange_coupling_mhz,
                field_strength_ut=b,
                k_s=self.rp.k_s,
                k_t=self.rp.k_t,
                lifetime_us=self.rp.lifetime_us,
            )
            anisotropies[i] = rp.compass_anisotropy(12)
        return fields, anisotropies


class DecoherenceEffects:
    """Study how decoherence (spin relaxation) affects compass precision.

    Parameters
    ----------
    base_radical_pair : RadicalPair
        Baseline radical pair without additional decoherence.
    """

    def __init__(self, base_radical_pair: RadicalPair):
        self.base_rp = base_radical_pair

    def anisotropy_vs_lifetime(
        self,
        lifetime_min_us: float = 0.1,
        lifetime_max_us: float = 5.0,
        n_points: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compass anisotropy vs radical pair lifetime.

        Shorter lifetimes reduce the time for singlet-triplet
        interconversion, degrading compass sensitivity.

        Parameters
        ----------
        lifetime_min_us : float
            Minimum lifetime in microseconds.
        lifetime_max_us : float
            Maximum lifetime in microseconds.
        n_points : int
            Number of lifetime points.

        Returns
        -------
        tuple of np.ndarray
            (lifetimes_us, anisotropies)
        """
        lifetimes = np.linspace(lifetime_min_us, lifetime_max_us, n_points)
        anisotropies = np.zeros(n_points)
        for i, lt in enumerate(lifetimes):
            rp = RadicalPair(
                hyperfine_coupling_mhz=self.base_rp.hyperfine_coupling_mhz,
                hyperfine_anisotropy=self.base_rp.hyperfine_anisotropy,
                exchange_coupling_mhz=self.base_rp.exchange_coupling_mhz,
                field_strength_ut=self.base_rp.field_strength_ut,
                k_s=self.base_rp.k_s,
                k_t=self.base_rp.k_t,
                lifetime_us=lt,
            )
            anisotropies[i] = rp.compass_anisotropy(12)
        return lifetimes, anisotropies

    def anisotropy_vs_recombination(
        self,
        k_min: float = 0.1,
        k_max: float = 10.0,
        n_points: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compass anisotropy vs recombination rate asymmetry.

        The ratio k_s/k_t affects the compass because asymmetric
        recombination amplifies the singlet-triplet yield difference.

        Parameters
        ----------
        k_min : float
            Minimum k_s value (MHz), with k_t fixed.
        k_max : float
            Maximum k_s value (MHz).
        n_points : int
            Number of points.

        Returns
        -------
        tuple of np.ndarray
            (k_s_values, anisotropies)
        """
        k_values = np.linspace(k_min, k_max, n_points)
        anisotropies = np.zeros(n_points)
        for i, ks in enumerate(k_values):
            rp = RadicalPair(
                hyperfine_coupling_mhz=self.base_rp.hyperfine_coupling_mhz,
                hyperfine_anisotropy=self.base_rp.hyperfine_anisotropy,
                exchange_coupling_mhz=self.base_rp.exchange_coupling_mhz,
                field_strength_ut=self.base_rp.field_strength_ut,
                k_s=ks,
                k_t=self.base_rp.k_t,
                lifetime_us=self.base_rp.lifetime_us,
            )
            anisotropies[i] = rp.compass_anisotropy(12)
        return k_values, anisotropies
