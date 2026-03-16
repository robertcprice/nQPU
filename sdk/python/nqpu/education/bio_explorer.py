"""Guided quantum biology demonstrations.

Interactive exploration of quantum effects in biological systems: photosynthesis
(FMO complex), enzyme tunneling, DNA mutation, and avian navigation.

All simulation is pure numpy -- no scipy or external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BioDemo:
    """A quantum biology demonstration."""

    title: str
    description: str
    physics: str  # What quantum effect is demonstrated
    parameters: dict = field(default_factory=dict)


@dataclass
class BioDemoResult:
    """Result of a bio demo."""

    demo: BioDemo
    data: dict
    explanation: str

    def summary(self) -> str:
        """ASCII summary of the demonstration results."""
        lines = [
            f"=== {self.demo.title} ===",
            f"Physics: {self.demo.physics}",
            "",
            self.explanation,
            "",
            "Key results:",
        ]
        for key, val in self.data.items():
            if isinstance(val, np.ndarray):
                if val.ndim == 1 and len(val) <= 10:
                    formatted = ", ".join(f"{v:.4f}" for v in val[:10])
                    lines.append(f"  {key}: [{formatted}]")
                else:
                    lines.append(f"  {key}: array shape={val.shape}")
            elif isinstance(val, float):
                lines.append(f"  {key}: {val:.6f}")
            else:
                lines.append(f"  {key}: {val}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matrix exponential via eigendecomposition (pure numpy, no scipy)
# ---------------------------------------------------------------------------

def _matrix_exp(A: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Compute exp(A * t) via eigendecomposition.  Works for any diagonalizable matrix."""
    eigenvalues, eigvecs = np.linalg.eig(A)
    diag = np.diag(np.exp(eigenvalues * t))
    return eigvecs @ diag @ np.linalg.inv(eigvecs)


# ---------------------------------------------------------------------------
# FMO Explorer
# ---------------------------------------------------------------------------

class FMOExplorer:
    """Explore quantum coherence in photosynthesis (FMO complex).

    The FMO complex transfers energy from a light-harvesting antenna to the
    reaction center with near-perfect efficiency.  Quantum coherence may play
    a role in this efficient energy transfer (ENAQT: environment-assisted
    quantum transport).
    """

    def __init__(self, n_sites: int = 7):
        self.n_sites = n_sites
        self.hamiltonian = self._build_fmo_hamiltonian()

    def _build_fmo_hamiltonian(self) -> np.ndarray:
        """Build simplified FMO Hamiltonian with realistic couplings (cm^-1).

        Based on the Adolphs-Renger Hamiltonian for the FMO complex of
        Prosthecochloris aestuarii.
        """
        n = self.n_sites
        H = np.zeros((n, n), dtype=complex)

        # Site energies (diagonal, in cm^-1 relative units)
        if n >= 7:
            site_energies = [12410, 12530, 12210, 12320, 12480, 12630, 12440]
        else:
            site_energies = [12410, 12530, 12210, 12320, 12480, 12630, 12440][:n]

        for i in range(n):
            H[i, i] = site_energies[i]

        # Inter-site couplings (off-diagonal, in cm^-1)
        couplings = {
            (0, 1): -87.7, (0, 2): 5.5, (0, 3): -5.9, (0, 4): 6.7,
            (0, 5): -13.7, (0, 6): -9.9,
            (1, 2): 30.8, (1, 3): 8.2, (1, 4): 0.7, (1, 5): 11.8, (1, 6): 4.3,
            (2, 3): -53.5, (2, 4): -2.2, (2, 5): -9.6, (2, 6): 6.0,
            (3, 4): -70.7, (3, 5): -17.0, (3, 6): -63.3,
            (4, 5): 81.1, (4, 6): -1.3,
            (5, 6): 39.7,
        }
        for (i, j), v in couplings.items():
            if i < n and j < n:
                H[i, j] = v
                H[j, i] = v

        return H

    def demo_coherent_transfer(self, initial_site: int = 0, t_max: float = 1.0, n_steps: int = 200) -> BioDemoResult:
        """Show coherent energy transfer between sites.

        Parameters
        ----------
        initial_site : int
            Site where the excitation starts (0-indexed).
        t_max : float
            Total evolution time in picoseconds.
        n_steps : int
            Number of time steps.
        """
        n = self.n_sites
        psi0 = np.zeros(n, dtype=complex)
        psi0[initial_site] = 1.0

        # Convert Hamiltonian to natural units for evolution
        # H is in cm^-1, time in ps, hbar*c ≈ 5308.8 cm^-1 * ps
        hbar_c = 5308.8  # cm^-1 * ps
        H_scaled = -1j * self.hamiltonian / hbar_c

        times = np.linspace(0, t_max, n_steps)
        populations = np.zeros((n_steps, n))
        for k, t in enumerate(times):
            U = _matrix_exp(H_scaled, t)
            psi_t = U @ psi0
            populations[k] = np.abs(psi_t) ** 2

        # Transfer efficiency: population at site 2 (the exit site) at t_max
        exit_site = min(2, n - 1)
        efficiency = float(populations[-1, exit_site])

        return BioDemoResult(
            demo=BioDemo(
                title="Coherent Energy Transfer in FMO",
                description="Quantum coherent energy transfer from antenna to reaction center.",
                physics="Quantum coherence enables simultaneous exploration of multiple transfer pathways.",
                parameters={"initial_site": initial_site, "t_max": t_max, "n_sites": n},
            ),
            data={
                "times": times,
                "populations": populations,
                "final_populations": populations[-1],
                "exit_site_efficiency": efficiency,
            },
            explanation=(
                f"Starting from site {initial_site}, coherent evolution transfers population "
                f"across {n} chromophore sites. At t={t_max:.1f} ps, exit site {exit_site} "
                f"has population {efficiency:.4f}. The oscillating populations reveal "
                f"quantum coherence -- classical hopping would show monotonic decay."
            ),
        )

    def demo_environment_effect(self, dephasing_rate: float = 0.1, t_max: float = 1.0, n_steps: int = 200) -> BioDemoResult:
        """Show how environment-assisted transport works (ENAQT).

        Moderate dephasing from the protein environment actually *enhances*
        transport by preventing destructive interference at certain sites.
        """
        n = self.n_sites
        hbar_c = 5308.8
        H_scaled = self.hamiltonian / hbar_c

        # Lindblad master equation: drho/dt = -i[H,rho] + gamma * sum_k (L_k rho L_k^dag - {L_k^dag L_k, rho}/2)
        # For pure dephasing: L_k = |k><k|
        # We use a simplified Lindblad evolution in the density matrix formalism

        rho0 = np.zeros((n, n), dtype=complex)
        rho0[0, 0] = 1.0  # start at site 0

        times = np.linspace(0, t_max, n_steps)
        dt = times[1] - times[0]
        rho = rho0.copy()
        populations = np.zeros((n_steps, n))

        for k in range(n_steps):
            populations[k] = np.real(np.diag(rho))
            # Coherent evolution: -i[H, rho]
            commutator = -1j * (H_scaled @ rho - rho @ H_scaled)
            # Dephasing: gamma * (sum_k |k><k| rho |k><k| - rho) for off-diagonals
            dephasing = -dephasing_rate * rho.copy()
            for i in range(n):
                dephasing[i, i] = 0.0  # diagonal elements unaffected
            rho = rho + dt * (commutator + dephasing)
            # Renormalize trace
            rho /= np.trace(rho)

        exit_site = min(2, n - 1)
        efficiency = float(populations[-1, exit_site])

        return BioDemoResult(
            demo=BioDemo(
                title="Environment-Assisted Quantum Transport (ENAQT)",
                description="Protein environment noise enhances transport efficiency.",
                physics="Moderate dephasing breaks destructive interference, enabling ENAQT.",
                parameters={"dephasing_rate": dephasing_rate, "t_max": t_max},
            ),
            data={
                "times": times,
                "populations": populations,
                "final_populations": populations[-1],
                "exit_site_efficiency": efficiency,
                "dephasing_rate": dephasing_rate,
            },
            explanation=(
                f"With dephasing rate gamma={dephasing_rate:.2f}/ps, the environment "
                f"assists transport. Exit site population = {efficiency:.4f}. "
                f"This is ENAQT: noise is not always harmful. At an intermediate dephasing "
                f"rate (the 'Goldilocks zone'), transport efficiency is maximized."
            ),
        )

    def compare_classical_quantum(self, t_max: float = 1.0, n_steps: int = 200) -> BioDemoResult:
        """Compare classical hopping vs quantum coherent transfer efficiency."""
        n = self.n_sites
        exit_site = min(2, n - 1)

        # Quantum coherent transfer
        q_result = self.demo_coherent_transfer(0, t_max, n_steps)
        q_eff = q_result.data["exit_site_efficiency"]

        # Classical hopping (rate equation): dp/dt = R * p
        # Transfer rates from coupling strengths
        R = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    coupling = abs(self.hamiltonian[i, j])
                    R[i, j] = coupling ** 2 * 0.001  # Fermi's golden rule simplified
        for i in range(n):
            R[i, i] = -np.sum(R[:, i])

        # Evolve rate equation
        hbar_c = 5308.8
        p = np.zeros(n)
        p[0] = 1.0
        dt = t_max / n_steps
        classical_pops = np.zeros((n_steps, n))
        for k in range(n_steps):
            classical_pops[k] = p
            p = p + dt * (R @ p)
            p = np.maximum(p, 0)
            s = np.sum(p)
            if s > 0:
                p /= s

        c_eff = float(classical_pops[-1, exit_site])

        return BioDemoResult(
            demo=BioDemo(
                title="Classical vs Quantum Energy Transfer",
                description="Compare incoherent hopping with quantum coherent transfer.",
                physics="Quantum coherence enables superposition of transfer pathways.",
                parameters={"t_max": t_max},
            ),
            data={
                "quantum_efficiency": q_eff,
                "classical_efficiency": c_eff,
                "quantum_speedup": q_eff / max(c_eff, 1e-15),
            },
            explanation=(
                f"Quantum coherent efficiency at exit site: {q_eff:.4f}\n"
                f"Classical hopping efficiency at exit site: {c_eff:.4f}\n"
                f"The quantum mechanism explores multiple pathways simultaneously, "
                f"while classical hopping is limited to sequential jumps."
            ),
        )


# ---------------------------------------------------------------------------
# Tunneling Explorer
# ---------------------------------------------------------------------------

class TunnelingExplorer:
    """Explore proton tunneling in enzyme catalysis.

    Enzymes can catalyze reactions partly through quantum tunneling of
    hydrogen atoms through energy barriers.
    """

    def demo_barrier_crossing(self, barrier_height: float = 1.0, mass: float = 1.0, barrier_width: float = 1.0) -> BioDemoResult:
        """Show tunneling through a rectangular potential barrier.

        Uses the WKB approximation: T ~ exp(-2 * kappa * width)
        where kappa = sqrt(2 * m * (V - E)) / hbar.

        Parameters
        ----------
        barrier_height : float
            Barrier height V in eV.
        mass : float
            Particle mass in proton masses.
        barrier_width : float
            Barrier width in Angstroms.
        """
        # Physical constants in appropriate units
        m_proton = 1.6726e-27  # kg
        hbar = 1.0546e-34  # J*s
        eV = 1.602e-19  # J
        angstrom = 1e-10  # m

        m = mass * m_proton
        V = barrier_height * eV
        w = barrier_width * angstrom

        # Particle energy at thermal energy kT (body temperature)
        kT = 0.0267  # eV at 310 K
        E = kT * eV

        if E >= V:
            T_tunnel = 1.0
            kappa = 0.0
        else:
            kappa = np.sqrt(2 * m * (V - E)) / hbar
            T_tunnel = float(np.exp(-2 * kappa * w))

        # Classical rate (Arrhenius): rate ~ exp(-V/kT)
        T_classical = float(np.exp(-barrier_height / kT))

        return BioDemoResult(
            demo=BioDemo(
                title="Quantum Tunneling Through Enzyme Barrier",
                description="Proton tunneling through a potential barrier in enzyme active site.",
                physics="WKB tunneling probability for a rectangular barrier.",
                parameters={"barrier_height_eV": barrier_height, "mass_proton": mass, "width_A": barrier_width},
            ),
            data={
                "tunneling_probability": T_tunnel,
                "classical_probability": T_classical,
                "kappa_inverse_A": 1.0 / (kappa * angstrom) if kappa > 0 else float("inf"),
                "enhancement": T_tunnel / max(T_classical, 1e-300),
            },
            explanation=(
                f"Barrier: V={barrier_height:.2f} eV, width={barrier_width:.1f} A, mass={mass:.1f} m_p\n"
                f"Tunneling probability (WKB): {T_tunnel:.6e}\n"
                f"Classical (Arrhenius) at 310K: {T_classical:.6e}\n"
                f"Tunneling can be {'significant' if T_tunnel > T_classical else 'negligible compared to thermal activation'}."
            ),
        )

    def demo_kinetic_isotope_effect(self, barrier_height: float = 0.3, barrier_width: float = 0.5) -> BioDemoResult:
        """Compare H vs D tunneling rates (Kinetic Isotope Effect).

        The KIE is the ratio of reaction rates for hydrogen vs deuterium.
        Large KIE (> 7) suggests tunneling contributes significantly.
        """
        h_result = self.demo_barrier_crossing(barrier_height, mass=1.0, barrier_width=barrier_width)
        d_result = self.demo_barrier_crossing(barrier_height, mass=2.0, barrier_width=barrier_width)

        T_H = h_result.data["tunneling_probability"]
        T_D = d_result.data["tunneling_probability"]
        kie = T_H / max(T_D, 1e-300)

        return BioDemoResult(
            demo=BioDemo(
                title="Kinetic Isotope Effect (KIE)",
                description="Compare tunneling rates of H and D through an enzyme barrier.",
                physics="Heavier deuterium tunnels less efficiently than hydrogen.",
                parameters={"barrier_height_eV": barrier_height, "barrier_width_A": barrier_width},
            ),
            data={
                "T_hydrogen": T_H,
                "T_deuterium": T_D,
                "KIE": kie,
            },
            explanation=(
                f"H tunneling probability: {T_H:.6e}\n"
                f"D tunneling probability: {T_D:.6e}\n"
                f"KIE (H/D ratio): {kie:.2f}\n"
                f"A KIE > 7 is strong evidence for tunneling. "
                f"{'This KIE suggests significant tunneling contribution.' if kie > 7 else 'This KIE is in the classical range.'}"
            ),
        )


# ---------------------------------------------------------------------------
# DNA Mutation Explorer
# ---------------------------------------------------------------------------

class DNAMutationExplorer:
    """Explore quantum effects in DNA base pair tautomerism.

    Proton tunneling along hydrogen bonds in Watson-Crick base pairs can
    produce rare tautomeric forms that mispair during replication, causing
    point mutations.
    """

    def demo_tautomer_tunneling(self, barrier: float = 0.4, width: float = 0.7) -> BioDemoResult:
        """Show proton tunneling causing tautomeric shift in base pairs.

        Parameters
        ----------
        barrier : float
            Double-well barrier height in eV.
        width : float
            Barrier width in Angstroms.
        """
        explorer = TunnelingExplorer()
        result = explorer.demo_barrier_crossing(barrier, mass=1.0, barrier_width=width)
        T = result.data["tunneling_probability"]

        return BioDemoResult(
            demo=BioDemo(
                title="Tautomeric Shift via Proton Tunneling",
                description="Proton tunneling along H-bond converts normal to rare tautomer.",
                physics="Double-well potential tunneling in Watson-Crick base pair.",
                parameters={"barrier_eV": barrier, "width_A": width},
            ),
            data={
                "tunneling_probability": T,
                "barrier_eV": barrier,
                "width_A": width,
            },
            explanation=(
                f"In a Watson-Crick base pair (e.g. A-T), the proton sits in a "
                f"double-well potential. Tunneling probability: {T:.6e}.\n"
                f"If the proton tunnels to the other well, the base adopts a rare "
                f"tautomeric form (e.g., enol instead of keto). This rare form "
                f"can mispair during DNA replication, causing a point mutation."
            ),
        )

    def mutation_probability(self, temperature: float = 310.0) -> BioDemoResult:
        """Estimate spontaneous mutation rate from tunneling probability.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin (default: body temperature 310 K).
        """
        kB = 8.617e-5  # eV/K
        kT = kB * temperature

        # Typical values for A-T base pair
        barrier = 0.4  # eV
        width = 0.7  # A

        explorer = TunnelingExplorer()
        result = explorer.demo_barrier_crossing(barrier, mass=1.0, barrier_width=width)
        T_tunnel = result.data["tunneling_probability"]

        # Thermal tautomer population (Boltzmann)
        delta_E = 0.1  # eV energy difference between tautomers
        thermal_pop = float(np.exp(-delta_E / kT))

        # Combined mutation probability per replication
        mutation_prob = T_tunnel * thermal_pop

        return BioDemoResult(
            demo=BioDemo(
                title="Quantum Contribution to Spontaneous Mutation",
                description="Estimate mutation rate from tunneling + thermal population.",
                physics="Combined tunneling and thermal tautomer population.",
                parameters={"temperature_K": temperature},
            ),
            data={
                "tunneling_probability": T_tunnel,
                "thermal_population": thermal_pop,
                "mutation_probability": mutation_prob,
                "temperature_K": temperature,
            },
            explanation=(
                f"At T={temperature:.0f} K:\n"
                f"  Tunneling probability: {T_tunnel:.6e}\n"
                f"  Thermal tautomer population: {thermal_pop:.6e}\n"
                f"  Combined mutation probability: {mutation_prob:.6e}\n"
                f"Observed spontaneous mutation rate ~10^-9 per base pair per replication. "
                f"Quantum tunneling provides a plausible mechanism."
            ),
        )


# ---------------------------------------------------------------------------
# Navigation Explorer
# ---------------------------------------------------------------------------

class NavigationExplorer:
    """Explore the radical pair mechanism in avian navigation.

    Migratory birds may sense the Earth's magnetic field through a quantum
    compass based on radical pair intermediates in the protein cryptochrome.
    The singlet-triplet interconversion rate depends on the orientation of
    the magnetic field relative to the radical pair axis.
    """

    def demo_radical_pair(self, magnetic_field: float = 0.05, t_max: float = 1.0, n_steps: int = 200) -> BioDemoResult:
        """Show singlet-triplet oscillation under magnetic field.

        Parameters
        ----------
        magnetic_field : float
            External magnetic field in mT.
        t_max : float
            Evolution time in microseconds.
        """
        # Simplified 2-spin radical pair model
        # H = omega_1 * S1z + omega_2 * S2z + J * S1.S2
        # where omega = g * mu_B * B / hbar

        # Physical constants
        g = 2.002  # electron g-factor
        mu_B = 9.274e-24  # Bohr magneton (J/T)
        hbar = 1.0546e-34  # J*s

        B = magnetic_field * 1e-3  # convert mT to T
        omega = g * mu_B * B / hbar  # rad/s
        omega_us = omega * 1e-6  # rad/us

        # Hyperfine coupling (simplified: one nuclear spin)
        A = 1.0  # MHz (hyperfine constant)
        A_rad = A * 2 * np.pi  # rad/us

        # Exchange coupling
        J = 0.0  # MHz (assume negligible for cryptochrome)

        # 4x4 Hamiltonian in {|TT>, |TS>, |ST>, |SS>} basis
        # Simplified: just use Zeeman + hyperfine
        # |singlet> = (|01> - |10>) / sqrt(2)
        # |triplet_0> = (|01> + |10>) / sqrt(2)
        H = np.zeros((4, 4), dtype=complex)
        # Zeeman splitting
        H[0, 0] = omega_us  # |00> (T+)
        H[3, 3] = -omega_us  # |11> (T-)
        # Hyperfine mixing between S and T0
        H[1, 2] = A_rad / 2  # |01> <-> |10>
        H[2, 1] = A_rad / 2

        # Initial state: singlet = (|01> - |10>) / sqrt(2)
        psi0 = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)

        # Singlet projector
        P_S = np.outer(psi0, psi0.conj())

        times = np.linspace(0, t_max, n_steps)
        singlet_yield = np.zeros(n_steps)

        for k, t in enumerate(times):
            U = _matrix_exp(-1j * H, t)
            psi_t = U @ psi0
            singlet_yield[k] = float(np.abs(psi_t.conj() @ P_S @ psi_t))

        avg_singlet = float(np.mean(singlet_yield))

        return BioDemoResult(
            demo=BioDemo(
                title="Radical Pair Singlet-Triplet Dynamics",
                description="Singlet-triplet interconversion in cryptochrome radical pair.",
                physics="Magnetic field modulates singlet-triplet mixing via Zeeman effect.",
                parameters={"B_mT": magnetic_field, "t_max_us": t_max},
            ),
            data={
                "times_us": times,
                "singlet_yield": singlet_yield,
                "average_singlet_yield": avg_singlet,
                "magnetic_field_mT": magnetic_field,
            },
            explanation=(
                f"At B={magnetic_field:.3f} mT, the radical pair oscillates between "
                f"singlet and triplet states. Average singlet yield: {avg_singlet:.4f}.\n"
                f"The singlet yield depends on field strength and direction, providing "
                f"the physical basis for a quantum compass."
            ),
        )

    def demo_compass_sensitivity(self, n_angles: int = 36, magnetic_field: float = 0.05) -> BioDemoResult:
        """Show angular dependence of singlet yield (compass function).

        Parameters
        ----------
        n_angles : int
            Number of angles to sample from 0 to 2*pi.
        magnetic_field : float
            Earth's field strength in mT.
        """
        g = 2.002
        mu_B = 9.274e-24
        hbar = 1.0546e-34
        B = magnetic_field * 1e-3
        omega = g * mu_B * B / hbar * 1e-6  # rad/us

        A = 1.0  # MHz hyperfine
        A_rad = A * 2 * np.pi

        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        singlet_yields = np.zeros(n_angles)

        t_max = 1.0
        n_steps = 100

        for ai, angle in enumerate(angles):
            # Field direction modifies effective Zeeman splitting
            omega_eff = omega * np.cos(angle)

            H = np.zeros((4, 4), dtype=complex)
            H[0, 0] = omega_eff
            H[3, 3] = -omega_eff
            H[1, 2] = A_rad / 2
            H[2, 1] = A_rad / 2

            psi0 = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
            P_S = np.outer(psi0, psi0.conj())

            times = np.linspace(0, t_max, n_steps)
            sy = 0.0
            for t in times:
                U = _matrix_exp(-1j * H, t)
                psi_t = U @ psi0
                sy += float(np.abs(psi_t.conj() @ P_S @ psi_t))
            singlet_yields[ai] = sy / n_steps

        anisotropy = float(np.max(singlet_yields) - np.min(singlet_yields))

        return BioDemoResult(
            demo=BioDemo(
                title="Quantum Compass Sensitivity",
                description="Angular dependence of singlet yield as compass function.",
                physics="Anisotropic singlet yield provides directional information.",
                parameters={"B_mT": magnetic_field, "n_angles": n_angles},
            ),
            data={
                "angles_rad": angles,
                "singlet_yields": singlet_yields,
                "anisotropy": anisotropy,
                "max_yield": float(np.max(singlet_yields)),
                "min_yield": float(np.min(singlet_yields)),
            },
            explanation=(
                f"The singlet yield varies with field direction (anisotropy = {anisotropy:.4f}).\n"
                f"Max yield: {np.max(singlet_yields):.4f}, Min yield: {np.min(singlet_yields):.4f}.\n"
                f"This angular dependence is what allows the radical pair to act as a compass. "
                f"Birds can detect changes of a few degrees in field direction."
            ),
        )
