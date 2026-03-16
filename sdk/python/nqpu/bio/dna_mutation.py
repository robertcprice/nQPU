"""Quantum proton tunneling in DNA base pair mutations.

Models spontaneous tautomeric shifts via proton tunneling along the
hydrogen bonds in Watson-Crick base pairs. Following Lowdin's hypothesis
(1963), quantum mechanical tunneling of protons between the normal and
rare tautomeric forms of nucleobases can generate point mutations.

Key physics:
  - Asymmetric double-well potential for proton transfer along H-bonds
  - WKB tunneling through the barrier between tautomeric forms
  - Boltzmann weighting of the energy asymmetry
  - Concerted (simultaneous) double proton transfer in base pairs
  - Temperature-dependent mutation rates

References:
  - Lowdin, Rev. Mod. Phys. 35, 724 (1963) -- original proton tunneling hypothesis
  - Florian et al., JACS 118, 3010 (1996) -- barrier parameters
  - Gorb et al., JPCA 108, 11592 (2004) -- double proton transfer
  - Slocombe et al., Phys. Chem. Chem. Phys. 23, 4141 (2021) -- modern DFT study
  - Brovarets' & Hovorun, J. Biomol. Struct. Dyn. 37, 1880 (2019) -- mutation rates

Example:
    from nqpu.bio.dna_mutation import BasePair, BasePairType

    at = BasePair.from_type(BasePairType.AT)
    gc = BasePair.from_type(BasePairType.GC)
    print(f"A-T tautomer probability at 310K: {at.tautomer_probability(310.0):.2e}")
    print(f"G-C tautomer probability at 310K: {gc.tautomer_probability(310.0):.2e}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34            # reduced Planck constant, J*s
K_B = 1.380649e-23               # Boltzmann constant, J/K
M_PROTON = 1.672_621_92e-27      # proton mass, kg
EV_TO_J = 1.602_176_634e-19      # electron-volt to Joule
NM_TO_M = 1.0e-9                 # nanometre to metre
PI = math.pi


class BasePairType(Enum):
    """Type of Watson-Crick DNA base pair."""
    AT = "A-T"
    GC = "G-C"


@dataclass
class DoubleWellPotential:
    """Asymmetric double-well potential for proton transfer.

    The potential is modelled as:
        V(x) = -a * x^2 + b * x^4 + c * x

    where 'a' and 'b' define the barrier shape and 'c' introduces
    asymmetry between the two wells (normal and tautomeric forms).

    For the simplified WKB calculation, we use the effective barrier
    height and width parameters.

    Parameters
    ----------
    barrier_height_ev : float
        Height of the potential barrier between the two wells (eV).
    barrier_width_nm : float
        Width of the barrier between the two minima (nm).
        This is the N-H...N or O-H...O distance.
    asymmetry_ev : float
        Energy difference between the two wells (eV).
        Positive means the rare tautomer is higher in energy.
    """
    barrier_height_ev: float
    barrier_width_nm: float
    asymmetry_ev: float

    def evaluate(self, x_nm: float) -> float:
        """Evaluate the double-well potential at position x.

        Uses the quartic form: V(x) = -a*x^2 + b*x^4 + c*x
        with parameters derived from barrier height, width, and asymmetry.

        Parameters
        ----------
        x_nm : float
            Position along the proton transfer coordinate (nm).

        Returns
        -------
        float
            Potential energy in eV.
        """
        # Derive quartic coefficients from physical parameters
        w = self.barrier_width_nm / 2.0  # half-width
        if w <= 0:
            return 0.0

        # V(+/-w) = 0 (wells), V(0) = barrier_height
        # -a*w^2 + b*w^4 = 0  =>  a = b*w^2
        # V(0) = 0  =>  need to shift: V(x) = h - a*x^2 + b*x^4
        # V(+/-w) = h - a*w^2 + b*w^4 = 0  => h = a*w^2 - b*w^4
        # a = h/w^2 + b*w^2, substituting: h = (h/w^2 + b*w^2)*w^2 - b*w^4 = h  (identity)
        # Choose b = h/w^4 => a = 2*h/w^2
        h = self.barrier_height_ev
        a = 2.0 * h / (w**2)
        b = h / (w**4)
        c = self.asymmetry_ev / (2.0 * w)  # linear tilt

        return h - a * x_nm**2 + b * x_nm**4 + c * x_nm

    def potential_curve(
        self,
        n_points: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the potential energy curve.

        Parameters
        ----------
        n_points : int
            Number of points in the curve.

        Returns
        -------
        tuple of np.ndarray
            (positions_nm, energies_ev)
        """
        w = self.barrier_width_nm / 2.0
        x = np.linspace(-1.5 * w, 1.5 * w, n_points)
        v = np.array([self.evaluate(xi) for xi in x])
        return x, v


@dataclass
class BasePair:
    """Watson-Crick base pair with proton transfer tunneling model.

    Models spontaneous tautomeric shifts via double proton transfer
    along the hydrogen bonds connecting the two bases.

    Parameters
    ----------
    pair_type : BasePairType
        Type of base pair (A-T or G-C).
    potential : DoubleWellPotential
        The proton transfer potential for a single H-bond.
    n_bonds : int
        Number of hydrogen bonds (2 for A-T, 3 for G-C).
    attempt_frequency_hz : float
        Proton vibrational attempt frequency (Hz).
    """
    pair_type: BasePairType
    potential: DoubleWellPotential
    n_bonds: int
    attempt_frequency_hz: float = 1.0e13

    @staticmethod
    def from_type(bp_type: BasePairType) -> "BasePair":
        """Create a base pair model from the pair type.

        Parameters from Lowdin (1963) and Florian et al., JACS 118, 3010 (1996).

        Parameters
        ----------
        bp_type : BasePairType
            Type of base pair.

        Returns
        -------
        BasePair
            Configured base pair model.
        """
        if bp_type == BasePairType.AT:
            return BasePair(
                pair_type=bp_type,
                potential=DoubleWellPotential(
                    barrier_height_ev=0.40,   # ~0.4 eV for A-T
                    barrier_width_nm=0.070,   # ~0.7 A (N-H...N distance)
                    asymmetry_ev=0.05,        # rare tautomer slightly higher
                ),
                n_bonds=2,
                attempt_frequency_hz=1.0e13,
            )
        elif bp_type == BasePairType.GC:
            return BasePair(
                pair_type=bp_type,
                potential=DoubleWellPotential(
                    barrier_height_ev=0.50,   # ~0.5 eV for G-C
                    barrier_width_nm=0.060,   # slightly narrower
                    asymmetry_ev=0.10,        # larger asymmetry
                ),
                n_bonds=3,
                attempt_frequency_hz=1.0e13,
            )
        else:
            raise ValueError(f"Unknown base pair type: {bp_type}")

    def _wkb_tunneling_probability(self) -> float:
        """WKB tunneling probability for a single hydrogen bond.

        T = exp(-2 * a * sqrt(2*m*V) / hbar)

        Returns
        -------
        float
            Tunneling probability for one proton through one barrier.
        """
        v = self.potential.barrier_height_ev * EV_TO_J
        a = self.potential.barrier_width_nm * NM_TO_M
        m = M_PROTON

        if v <= 0.0 or a <= 0.0:
            return 1.0

        kappa = math.sqrt(2.0 * m * v) / HBAR
        exponent = -2.0 * kappa * a

        if exponent < -700.0:
            return 0.0

        return math.exp(exponent)

    def tautomer_probability(self, temperature_k: float) -> float:
        """Probability of the proton being in the tautomeric (rare) form.

        Combines WKB tunneling probability with Boltzmann weighting of
        the energy asymmetry for a single hydrogen bond.

        P_rare = T * exp(-dE/kT) / (1 + T * exp(-dE/kT))

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Tautomer occupation probability for a single H-bond.

        Raises
        ------
        ValueError
            If temperature is not positive.
        """
        if temperature_k <= 0.0:
            raise ValueError(
                f"Temperature must be positive, got {temperature_k}"
            )

        t_prob = self._wkb_tunneling_probability()

        # Boltzmann factor for the energy asymmetry
        delta_e = self.potential.asymmetry_ev * EV_TO_J
        boltzmann = math.exp(-delta_e / (K_B * temperature_k))

        # Effective population
        tb = t_prob * boltzmann
        return tb / (1.0 + tb)

    def concerted_tautomer_probability(self, temperature_k: float) -> float:
        """Probability that ALL hydrogen bonds simultaneously tunnel.

        For independent bonds: P_total = P_single ^ n_bonds.

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Concerted double proton transfer probability.
        """
        p_single = self.tautomer_probability(temperature_k)
        return p_single ** self.n_bonds

    def mutation_rate(
        self,
        temperature_k: float,
        p_misincorporation: float = 1.0e-4,
    ) -> float:
        """Mutation rate from quantum tunneling (per second).

        Rate = attempt_frequency * P_tautomer * P_misincorporation

        The misincorporation probability accounts for the fact that
        the rare tautomer must be present during DNA replication AND
        escape proofreading to cause a mutation.

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.
        p_misincorporation : float
            Probability that a tautomeric form leads to a replication
            error. Typical: 10^-4 to 10^-5.

        Returns
        -------
        float
            Mutation rate in s^-1.
        """
        p_taut = self.tautomer_probability(temperature_k)
        return self.attempt_frequency_hz * p_taut * p_misincorporation

    def classical_mutation_rate(
        self,
        temperature_k: float,
        p_misincorporation: float = 1.0e-4,
    ) -> float:
        """Classical (Arrhenius, no tunneling) mutation rate.

        Rate = nu * exp(-V/kT) * P_misincorporation

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.
        p_misincorporation : float
            Misincorporation probability.

        Returns
        -------
        float
            Classical mutation rate in s^-1.

        Raises
        ------
        ValueError
            If temperature is not positive.
        """
        if temperature_k <= 0.0:
            raise ValueError(
                f"Temperature must be positive, got {temperature_k}"
            )
        v = self.potential.barrier_height_ev * EV_TO_J
        exponent = -v / (K_B * temperature_k)
        if exponent < -700.0:
            return 0.0
        return (
            self.attempt_frequency_hz
            * math.exp(exponent)
            * p_misincorporation
        )

    def quantum_classical_ratio(self, temperature_k: float) -> float:
        """Ratio of quantum to classical mutation rate.

        A ratio > 1 indicates quantum tunneling enhances mutation rate.

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Quantum/classical mutation rate ratio.
        """
        q_rate = self.mutation_rate(temperature_k)
        c_rate = self.classical_mutation_rate(temperature_k)
        if c_rate < 1e-300:
            return float("inf")
        return q_rate / c_rate

    def tunnel_splitting_ev(self) -> float:
        """Double-well tunnel splitting energy in eV.

        Delta = hbar * omega * sqrt(T_wkb)

        Returns
        -------
        float
            Tunnel splitting in eV.
        """
        prob = self._wkb_tunneling_probability()
        omega = 2.0 * PI * self.attempt_frequency_hz
        delta_j = HBAR * omega * math.sqrt(prob)
        return delta_j / EV_TO_J

    def mutation_rate_vs_temperature(
        self,
        t_min: float = 200.0,
        t_max: float = 400.0,
        steps: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mutation rate vs temperature for quantum and classical models.

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
            (temperatures, quantum_rates, classical_rates)

        Raises
        ------
        ValueError
            If t_min <= 0 or t_max <= t_min.
        """
        if t_min <= 0.0:
            raise ValueError(f"t_min must be positive, got {t_min}")
        if t_max <= t_min:
            raise ValueError("t_max must exceed t_min")

        temps = np.linspace(t_min, t_max, steps)
        q_rates = np.array([self.mutation_rate(t) for t in temps])
        c_rates = np.array([
            self.classical_mutation_rate(t) for t in temps
        ])
        return temps, q_rates, c_rates


class TautomerTunneling:
    """Detailed tautomeric tunneling analysis.

    Provides additional analysis tools beyond the basic BasePair model,
    including instanton approximation and WKB action integrals.

    Parameters
    ----------
    base_pair : BasePair
        The base pair model to analyse.
    """

    def __init__(self, base_pair: BasePair):
        self.bp = base_pair

    def wkb_action(self) -> float:
        """WKB action integral S = integral sqrt(2*m*V(x)) dx.

        For a rectangular barrier: S = a * sqrt(2*m*V)

        Returns
        -------
        float
            WKB action in units of hbar (dimensionless S/hbar).
        """
        v = self.bp.potential.barrier_height_ev * EV_TO_J
        a = self.bp.potential.barrier_width_nm * NM_TO_M
        m = M_PROTON
        return a * math.sqrt(2.0 * m * v) / HBAR

    def instanton_rate(self, temperature_k: float) -> float:
        """Tunneling rate using the instanton (bounce) approximation.

        The instanton rate at finite temperature uses the imaginary-time
        path integral formulation:

            k = A * exp(-S_inst/hbar)

        where S_inst is the instanton action, approximately equal to
        the WKB action for thin barriers.

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Instanton tunneling rate in s^-1.
        """
        s_hbar = self.wkb_action()

        # Prefactor: attempt frequency * quantum correction
        omega = 2.0 * PI * self.bp.attempt_frequency_hz
        kbt = K_B * temperature_k

        # Crossover temperature between thermal and tunneling regimes
        v_j = self.bp.potential.barrier_height_ev * EV_TO_J
        omega_b = math.sqrt(2.0 * v_j / (M_PROTON * (
            self.bp.potential.barrier_width_nm * NM_TO_M / 2.0
        )**2))
        t_crossover = HBAR * omega_b / (2.0 * PI * K_B)

        if temperature_k > t_crossover:
            # Thermal regime: Arrhenius with quantum correction
            exponent = -v_j / kbt
            if exponent < -700.0:
                return 0.0
            correction = s_hbar / (2.0 * PI)
            return omega / (2.0 * PI) * math.exp(exponent) * (1.0 + correction)
        else:
            # Deep tunneling regime: instanton dominates
            if s_hbar > 700.0:
                return 0.0
            return omega / (2.0 * PI) * math.exp(-s_hbar)

    def tunneling_time_fs(self) -> float:
        """Estimate of the quantum tunneling traversal time.

        Uses the Buttiker-Landauer tunneling time:
            tau = m * a / (hbar * kappa)

        where kappa = sqrt(2*m*V)/hbar is the decay constant.

        Returns
        -------
        float
            Tunneling time in femtoseconds.
        """
        v = self.bp.potential.barrier_height_ev * EV_TO_J
        a = self.bp.potential.barrier_width_nm * NM_TO_M
        m = M_PROTON

        if v <= 0.0:
            return 0.0

        kappa = math.sqrt(2.0 * m * v) / HBAR
        tau_s = m * a / (HBAR * kappa)
        return tau_s / 1.0e-15  # convert to fs


class MutationRate:
    """Compare quantum and classical mutation rate predictions.

    Parameters
    ----------
    base_pair : BasePair
        The DNA base pair model.
    """

    def __init__(self, base_pair: BasePair):
        self.bp = base_pair

    def quantum_dominance_temperature(self) -> float:
        """Temperature below which tunneling dominates thermal hopping.

        Solves: T_wkb = exp(-V/kT)  =>  T = -V / (k * ln(T_wkb))

        Returns
        -------
        float
            Crossover temperature in Kelvin.
        """
        t_prob = self.bp._wkb_tunneling_probability()
        if t_prob <= 0.0 or t_prob >= 1.0:
            return 0.0
        v = self.bp.potential.barrier_height_ev * EV_TO_J
        return -v / (K_B * math.log(t_prob))

    def mutation_rate_per_cell_division(
        self,
        temperature_k: float = 310.0,
        genome_size_bp: int = 6_400_000_000,
        division_time_s: float = 86400.0,
    ) -> float:
        """Estimate quantum tunneling mutations per cell division.

        Parameters
        ----------
        temperature_k : float
            Body temperature (default 310 K = 37C).
        genome_size_bp : int
            Number of base pairs in the genome. Default: human (6.4 billion).
        division_time_s : float
            Time for one cell division in seconds. Default: 24 hours.

        Returns
        -------
        float
            Expected number of quantum tunneling mutations per cell division.
        """
        rate_per_bp = self.bp.mutation_rate(temperature_k)
        return rate_per_bp * genome_size_bp * division_time_s

    def compare_rates(
        self,
        temperature_k: float = 310.0,
    ) -> dict:
        """Compare quantum and classical mutation rate predictions.

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        dict
            Comparison results with keys:
            - 'base_pair': base pair type string
            - 'temperature_k': temperature
            - 'quantum_rate': quantum tunneling mutation rate (s^-1)
            - 'classical_rate': classical Arrhenius rate (s^-1)
            - 'ratio': quantum/classical ratio
            - 'tunneling_probability': WKB probability
            - 'tautomer_probability': tautomer occupation
            - 'tunnel_splitting_ev': energy splitting
        """
        return {
            "base_pair": self.bp.pair_type.value,
            "temperature_k": temperature_k,
            "quantum_rate": self.bp.mutation_rate(temperature_k),
            "classical_rate": self.bp.classical_mutation_rate(temperature_k),
            "ratio": self.bp.quantum_classical_ratio(temperature_k),
            "tunneling_probability": self.bp._wkb_tunneling_probability(),
            "tautomer_probability": self.bp.tautomer_probability(temperature_k),
            "tunnel_splitting_ev": self.bp.tunnel_splitting_ev(),
        }
