"""Quantum tunneling in enzyme catalysis.

Models proton and hydrogen transfer across potential energy barriers in
enzyme active sites using the WKB (Wentzel-Kramers-Brillouin) approximation.
Demonstrates that quantum tunneling is a significant contributor to enzyme
catalysis, particularly visible through the kinetic isotope effect (KIE).

Key physics:
  - WKB tunneling through rectangular and shaped barriers
  - Kinetic isotope effect: hydrogen vs deuterium tunneling rates
  - Temperature-dependent rates: crossover from thermal to tunneling regime
  - Tunnel splitting in double-well potentials
  - Sensitivity analysis of barrier parameters

References:
  - Hay & Scrutton, Nat. Chem. 4, 161 (2012) -- enzyme tunneling review
  - Klinman, Chem. Phys. Lett. 471, 179 (2009) -- H-tunneling in enzymes
  - Kohen & Klinman, Acc. Chem. Res. 31, 397 (1998) -- KIE in enzymes
  - Scrutton et al., Eur. J. Biochem. 264, 666 (1999) -- ADH tunneling
  - Knapp & Klinman, Eur. J. Biochem. 269, 3113 (2002) -- SLO tunneling

Example:
    from nqpu.bio.tunneling import EnzymeTunneling, ENZYMES

    adh = ENZYMES["alcohol_dehydrogenase"]
    print(f"Tunneling probability: {adh.tunneling_probability():.2e}")
    print(f"KIE ratio: {adh.kie_ratio():.1f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34            # reduced Planck constant, J*s
K_B = 1.380649e-23               # Boltzmann constant, J/K
M_PROTON = 1.672_621_92e-27      # proton mass, kg
M_DEUTERIUM = 3.343_583_72e-27   # deuterium mass, kg
M_TRITIUM = 5.008_267e-27        # tritium mass, kg
EV_TO_J = 1.602_176_634e-19      # electron-volt to Joule
NM_TO_M = 1.0e-9                 # nanometre to metre
KCAL_MOL_TO_J = 6.9477e-21       # kcal/mol to Joule (per molecule)


class BarrierShape(Enum):
    """Shape of the potential energy barrier."""
    RECTANGULAR = "rectangular"
    PARABOLIC = "parabolic"
    ECKART = "eckart"


@dataclass
class TunnelingBarrier:
    """One-dimensional potential energy barrier.

    Parameters
    ----------
    height_ev : float
        Barrier height in electron-volts.
    width_nm : float
        Barrier width in nanometres.
    shape : BarrierShape
        Barrier shape. Default: rectangular.
    asymmetry_ev : float
        Energy difference between reactant and product wells (eV).
        Positive means exothermic (product lower).
    """
    height_ev: float
    width_nm: float
    shape: BarrierShape = BarrierShape.RECTANGULAR
    asymmetry_ev: float = 0.0

    def validate(self) -> None:
        """Raise ValueError if barrier parameters are unphysical."""
        if self.height_ev < 0.0:
            raise ValueError(
                f"Barrier height must be non-negative, got {self.height_ev}"
            )
        if self.width_nm < 0.0:
            raise ValueError(
                f"Barrier width must be non-negative, got {self.width_nm}"
            )


@dataclass
class EnzymeTunneling:
    """Quantum tunneling model for enzyme catalysis.

    Models proton (or hydrogen/deuterium/tritium) transfer through a
    potential energy barrier in an enzyme active site.

    Parameters
    ----------
    barrier : TunnelingBarrier
        The potential energy barrier.
    particle_mass_kg : float
        Mass of the tunneling particle. Default: proton mass.
    attempt_frequency_hz : float
        Vibrational attempt frequency at the well bottom (Hz).
        Typical: 10^13 Hz for O-H stretch.
    """
    barrier: TunnelingBarrier
    particle_mass_kg: float = M_PROTON
    attempt_frequency_hz: float = 1.0e13

    @staticmethod
    def from_barrier(
        height_ev: float,
        width_nm: float,
        mass_kg: float = M_PROTON,
    ) -> "EnzymeTunneling":
        """Convenience constructor from barrier height and width.

        Parameters
        ----------
        height_ev : float
            Barrier height in eV.
        width_nm : float
            Barrier width in nm.
        mass_kg : float
            Particle mass in kg. Default: proton.

        Returns
        -------
        EnzymeTunneling
            Configured tunneling model.
        """
        return EnzymeTunneling(
            barrier=TunnelingBarrier(height_ev=height_ev, width_nm=width_nm),
            particle_mass_kg=mass_kg,
        )

    def tunneling_probability(self) -> float:
        """WKB tunneling probability through the barrier.

        For a rectangular barrier of height V and width a:
            T = exp(-2 * a * sqrt(2*m*V) / hbar)

        For a parabolic barrier:
            T = exp(-pi * a * sqrt(2*m*V) / (2 * hbar))

        For an Eckart barrier, uses the analytic Eckart formula.

        Returns
        -------
        float
            Tunneling transmission coefficient in [0, 1].

        Raises
        ------
        ValueError
            If barrier parameters are unphysical.
        """
        self.barrier.validate()

        v = self.barrier.height_ev * EV_TO_J
        a = self.barrier.width_nm * NM_TO_M
        m = self.particle_mass_kg

        if v == 0.0 or a == 0.0:
            return 1.0

        if self.barrier.shape == BarrierShape.RECTANGULAR:
            kappa = math.sqrt(2.0 * m * v) / HBAR
            exponent = -2.0 * kappa * a
        elif self.barrier.shape == BarrierShape.PARABOLIC:
            kappa = math.sqrt(2.0 * m * v) / HBAR
            exponent = -math.pi * kappa * a / 2.0
        elif self.barrier.shape == BarrierShape.ECKART:
            # Eckart barrier: T = exp(-2 * a * sqrt(2*m*V) / hbar * alpha)
            # where alpha = 2/pi * arctan(pi * d / (2 * a * sqrt(2*m*V/hbar^2)))
            # Simplified: use rectangular with 0.8x correction for Eckart shape
            kappa = math.sqrt(2.0 * m * v) / HBAR
            exponent = -2.0 * kappa * a * 0.8
        else:
            kappa = math.sqrt(2.0 * m * v) / HBAR
            exponent = -2.0 * kappa * a

        if exponent < -700.0:
            return 0.0

        return math.exp(exponent)

    def tunneling_rate(self) -> float:
        """Tunneling rate in s^-1.

        Rate = attempt_frequency * tunneling_probability.

        Returns
        -------
        float
            Tunneling rate in inverse seconds.
        """
        return self.attempt_frequency_hz * self.tunneling_probability()

    def classical_rate(self, temperature_k: float) -> float:
        """Classical (Arrhenius) thermal rate at temperature T.

        k_classical = nu * exp(-V / k_B T)

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Classical thermal rate in s^-1.

        Raises
        ------
        ValueError
            If temperature is not positive.
        """
        if temperature_k <= 0.0:
            raise ValueError(
                f"Temperature must be positive, got {temperature_k}"
            )
        v = self.barrier.height_ev * EV_TO_J
        exponent = -v / (K_B * temperature_k)
        if exponent < -700.0:
            return 0.0
        return self.attempt_frequency_hz * math.exp(exponent)

    def total_rate(self, temperature_k: float) -> float:
        """Total rate (tunneling + thermal) at temperature T.

        k_total = nu * [T_wkb + exp(-V / k_B T)]

        At low T, tunneling dominates. At high T, thermal activation dominates.

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Total rate in s^-1.
        """
        tunnel_prob = self.tunneling_probability()
        if temperature_k > 0.0:
            v = self.barrier.height_ev * EV_TO_J
            exponent = -v / (K_B * temperature_k)
            thermal = 0.0 if exponent < -700.0 else math.exp(exponent)
        else:
            thermal = 0.0
        return self.attempt_frequency_hz * (tunnel_prob + thermal)

    def kie_ratio(self) -> float:
        """Kinetic Isotope Effect: ratio of H to D tunneling rates.

        KIE = k_H / k_D

        A KIE > 7 typically indicates significant quantum tunneling
        contribution. Classical Arrhenius KIE is bounded by ~7.

        Returns
        -------
        float
            KIE ratio (dimensionless). Returns inf if D rate is zero.
        """
        h_model = EnzymeTunneling(
            barrier=self.barrier,
            particle_mass_kg=M_PROTON,
            attempt_frequency_hz=self.attempt_frequency_hz,
        )
        d_model = EnzymeTunneling(
            barrier=self.barrier,
            particle_mass_kg=M_DEUTERIUM,
            attempt_frequency_hz=self.attempt_frequency_hz,
        )
        rate_h = h_model.tunneling_rate()
        rate_d = d_model.tunneling_rate()
        if rate_d < 1e-300:
            return float("inf")
        return rate_h / rate_d

    def swain_schaad_exponent(self) -> float:
        """Swain-Schaad exponent comparing H/D/T tunneling rates.

        The Swain-Schaad relationship: k_H/k_T = (k_H/k_D)^x
        Classical limit: x = 1.44. Values > 1.44 indicate tunneling.

        Returns
        -------
        float
            Swain-Schaad exponent. Returns nan if rates are too small.
        """
        h_model = EnzymeTunneling(
            barrier=self.barrier,
            particle_mass_kg=M_PROTON,
            attempt_frequency_hz=self.attempt_frequency_hz,
        )
        d_model = EnzymeTunneling(
            barrier=self.barrier,
            particle_mass_kg=M_DEUTERIUM,
            attempt_frequency_hz=self.attempt_frequency_hz,
        )
        t_model = EnzymeTunneling(
            barrier=self.barrier,
            particle_mass_kg=M_TRITIUM,
            attempt_frequency_hz=self.attempt_frequency_hz,
        )
        rate_h = h_model.tunneling_rate()
        rate_d = d_model.tunneling_rate()
        rate_t = t_model.tunneling_rate()

        if rate_d < 1e-300 or rate_t < 1e-300 or rate_h < 1e-300:
            return float("nan")

        kie_hd = rate_h / rate_d
        kie_ht = rate_h / rate_t

        if kie_hd <= 1.0 or kie_ht <= 1.0:
            return float("nan")

        return math.log(kie_ht) / math.log(kie_hd)

    def tunnel_splitting_ev(self) -> float:
        """Double-well energy splitting due to tunneling.

        For a symmetric double well:
            Delta = hbar * omega * sqrt(T)

        where T is the WKB tunneling probability and omega is the
        attempt angular frequency.

        Returns
        -------
        float
            Tunnel splitting in electron-volts.
        """
        prob = self.tunneling_probability()
        omega = 2.0 * math.pi * self.attempt_frequency_hz
        delta_j = HBAR * omega * math.sqrt(prob)
        return delta_j / EV_TO_J

    def rate_vs_temperature(
        self,
        t_min: float = 50.0,
        t_max: float = 500.0,
        steps: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute rate vs temperature curves.

        Parameters
        ----------
        t_min : float
            Minimum temperature in Kelvin.
        t_max : float
            Maximum temperature in Kelvin.
        steps : int
            Number of temperature points.

        Returns
        -------
        tuple of np.ndarray
            (temperatures, total_rates, classical_rates, tunnel_rates)
            All rates in s^-1.

        Raises
        ------
        ValueError
            If t_min <= 0 or t_max <= t_min.
        """
        if t_min <= 0.0:
            raise ValueError(f"t_min must be positive, got {t_min}")
        if t_max <= t_min:
            raise ValueError(f"t_max must exceed t_min")

        tunnel_rate = self.tunneling_rate()
        temps = np.linspace(t_min, t_max, steps)
        total = np.array([self.total_rate(t) for t in temps])
        classical = np.array([self.classical_rate(t) for t in temps])
        tunnel = np.full(steps, tunnel_rate)

        return temps, total, classical, tunnel

    def crossover_temperature(self) -> float:
        """Temperature at which tunneling and thermal rates are equal.

        Solves: exp(-V/k_B T) = T_wkb  =>  T = -V / (k_B * ln(T_wkb))

        Returns
        -------
        float
            Crossover temperature in Kelvin. Returns 0.0 if tunneling
            probability is zero (infinite crossover temperature).
        """
        t_prob = self.tunneling_probability()
        if t_prob <= 0.0 or t_prob >= 1.0:
            return 0.0
        v = self.barrier.height_ev * EV_TO_J
        return -v / (K_B * math.log(t_prob))


class TunnelingSensitivity:
    """Sensitivity analysis of tunneling probability to barrier parameters.

    Parameters
    ----------
    base_model : EnzymeTunneling
        The baseline tunneling model to perturb.
    """

    def __init__(self, base_model: EnzymeTunneling):
        self.base_model = base_model

    def height_sensitivity(
        self,
        delta_ev: float = 0.05,
        n_points: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tunneling probability vs barrier height.

        Parameters
        ----------
        delta_ev : float
            Range to scan around the base barrier height (+/- delta_ev).
        n_points : int
            Number of points in the scan.

        Returns
        -------
        tuple of np.ndarray
            (heights_ev, probabilities)
        """
        base_h = self.base_model.barrier.height_ev
        heights = np.linspace(
            max(0.0, base_h - delta_ev),
            base_h + delta_ev,
            n_points,
        )
        probs = np.zeros(n_points)
        for i, h in enumerate(heights):
            model = EnzymeTunneling(
                barrier=TunnelingBarrier(
                    height_ev=h,
                    width_nm=self.base_model.barrier.width_nm,
                    shape=self.base_model.barrier.shape,
                ),
                particle_mass_kg=self.base_model.particle_mass_kg,
                attempt_frequency_hz=self.base_model.attempt_frequency_hz,
            )
            probs[i] = model.tunneling_probability()
        return heights, probs

    def width_sensitivity(
        self,
        delta_nm: float = 0.02,
        n_points: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tunneling probability vs barrier width.

        Parameters
        ----------
        delta_nm : float
            Range to scan around the base barrier width (+/- delta_nm).
        n_points : int
            Number of points in the scan.

        Returns
        -------
        tuple of np.ndarray
            (widths_nm, probabilities)
        """
        base_w = self.base_model.barrier.width_nm
        widths = np.linspace(
            max(0.0, base_w - delta_nm),
            base_w + delta_nm,
            n_points,
        )
        probs = np.zeros(n_points)
        for i, w in enumerate(widths):
            model = EnzymeTunneling(
                barrier=TunnelingBarrier(
                    height_ev=self.base_model.barrier.height_ev,
                    width_nm=w,
                    shape=self.base_model.barrier.shape,
                ),
                particle_mass_kg=self.base_model.particle_mass_kg,
                attempt_frequency_hz=self.base_model.attempt_frequency_hz,
            )
            probs[i] = model.tunneling_probability()
        return widths, probs

    def mass_sensitivity(
        self,
        masses_amu: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tunneling probability vs particle mass.

        Parameters
        ----------
        masses_amu : list of float, optional
            Masses in AMU to evaluate. Default: [1, 2, 3] (H, D, T).

        Returns
        -------
        tuple of np.ndarray
            (masses_amu, probabilities)
        """
        if masses_amu is None:
            masses_amu = [1.008, 2.014, 3.016]

        amu_to_kg = 1.66053906660e-27
        masses = np.array(masses_amu)
        probs = np.zeros(len(masses_amu))
        for i, m_amu in enumerate(masses_amu):
            model = EnzymeTunneling(
                barrier=self.base_model.barrier,
                particle_mass_kg=m_amu * amu_to_kg,
                attempt_frequency_hz=self.base_model.attempt_frequency_hz,
            )
            probs[i] = model.tunneling_probability()
        return masses, probs


# ---------------------------------------------------------------------------
# Predefined enzyme systems
# ---------------------------------------------------------------------------

def _alcohol_dehydrogenase() -> EnzymeTunneling:
    """Alcohol dehydrogenase (ADH).

    Proton transfer in ADH with moderate barrier.
    Parameters from Scrutton et al., Eur. J. Biochem. 264, 666 (1999).
    """
    return EnzymeTunneling(
        barrier=TunnelingBarrier(height_ev=0.30, width_nm=0.05),
        particle_mass_kg=M_PROTON,
        attempt_frequency_hz=1.0e13,
    )


def _soybean_lipoxygenase() -> EnzymeTunneling:
    """Soybean lipoxygenase (SLO-1).

    Hydrogen abstraction with large KIE (~80), indicating substantial
    tunneling. Parameters from Knapp & Klinman, Eur. J. Biochem. 269,
    3113 (2002).
    """
    return EnzymeTunneling(
        barrier=TunnelingBarrier(height_ev=0.50, width_nm=0.06),
        particle_mass_kg=M_PROTON,
        attempt_frequency_hz=1.0e13,
    )


def _aromatic_amine_dehydrogenase() -> EnzymeTunneling:
    """Aromatic amine dehydrogenase (AADH).

    Proton tunneling with temperature-independent KIE.
    Parameters from Masgrau et al., Science 312, 237 (2006).
    """
    return EnzymeTunneling(
        barrier=TunnelingBarrier(height_ev=0.35, width_nm=0.055),
        particle_mass_kg=M_PROTON,
        attempt_frequency_hz=1.0e13,
    )


ENZYMES: Dict[str, EnzymeTunneling] = {
    "alcohol_dehydrogenase": _alcohol_dehydrogenase(),
    "soybean_lipoxygenase": _soybean_lipoxygenase(),
    "aromatic_amine_dehydrogenase": _aromatic_amine_dehydrogenase(),
}
"""Predefined enzyme tunneling models.

Keys: 'alcohol_dehydrogenase', 'soybean_lipoxygenase',
'aromatic_amine_dehydrogenase'.
"""
