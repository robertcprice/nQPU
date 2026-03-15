"""Device presets for commercial neutral-atom quantum computers.

Pre-calibrated configurations for existing and announced neutral-atom
platforms, with parameters sourced from published specifications and
literature.

Devices:
- QuEra Aquila: 256-atom 87Rb system (Wurtz et al., 2023)
- Atom Computing: 1225-site 171Yb system (Norcia et al., Nature 2023)
- Pasqal Fresnel: 100-atom 87Rb system

References:
    - Wurtz et al., arXiv:2306.11727 (2023) [Aquila]
    - Norcia et al., Nature 622, 279 (2023) [Atom Computing]
    - Pasqal, Fresnel product documentation (2024)
    - Bluvstein et al., Nature 626, 58 (2024) [Harvard/QuEra logical qubits]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .array import ArrayGeometry, AtomArray, Zone, ZoneConfig
from .noise import NeutralAtomNoiseModel
from .physics import AtomSpecies


@dataclass(frozen=True)
class DeviceSpec:
    """Specification for a neutral-atom quantum computing device.

    Parameters
    ----------
    name : str
        Device name.
    vendor : str
        Manufacturer name.
    max_atoms : int
        Maximum number of atoms in the computational register.
    max_sites : int
        Maximum number of tweezer sites available.
    species : AtomSpecies
        Atom species used.
    geometry : ArrayGeometry
        Default array geometry.
    default_spacing_um : float
        Default inter-atom spacing in micrometres.
    rabi_freq_max_mhz : float
        Maximum Rydberg Rabi frequency in MHz.
    rabi_freq_typical_mhz : float
        Typical operating Rabi frequency in MHz.
    single_qubit_fidelity : float
        Published single-qubit gate fidelity.
    two_qubit_fidelity : float
        Published two-qubit gate (CZ) fidelity.
    readout_fidelity : float
        Published measurement fidelity.
    atom_loss_per_gate : float
        Probability of atom loss per gate operation.
    repetition_rate_hz : float
        Experimental cycle repetition rate in Hz.
    connectivity : str
        Connectivity description (e.g. 'reconfigurable', 'nearest-neighbour').
    notes : str
        Additional notes about the device.
    """

    name: str
    vendor: str
    max_atoms: int
    max_sites: int
    species: AtomSpecies
    geometry: ArrayGeometry
    default_spacing_um: float
    rabi_freq_max_mhz: float
    rabi_freq_typical_mhz: float
    single_qubit_fidelity: float
    two_qubit_fidelity: float
    readout_fidelity: float
    atom_loss_per_gate: float
    repetition_rate_hz: float
    connectivity: str
    notes: str = ""

    def blockade_radius_um(self) -> float:
        """Blockade radius at typical Rabi frequency."""
        return self.species.blockade_radius_um(self.rabi_freq_typical_mhz)

    def create_array(
        self,
        n_atoms: int | None = None,
        spacing_um: float | None = None,
    ) -> AtomArray:
        """Create an atom array matching this device's configuration.

        Parameters
        ----------
        n_atoms : int, optional
            Number of atoms.  Defaults to max_atoms.
        spacing_um : float, optional
            Override spacing.  Defaults to device default.

        Returns
        -------
        AtomArray
            Configured atom array.

        Raises
        ------
        ValueError
            If n_atoms exceeds max_atoms.
        """
        n = n_atoms if n_atoms is not None else self.max_atoms
        if n > self.max_atoms:
            raise ValueError(
                f"Requested {n} atoms but device supports max {self.max_atoms}"
            )

        spacing = spacing_um if spacing_um is not None else self.default_spacing_um
        return AtomArray(
            n_sites=n,
            species=self.species,
            spacing_um=spacing,
            geometry=self.geometry,
        )

    def create_noise_model(self) -> NeutralAtomNoiseModel:
        """Create a noise model calibrated to this device's published specs.

        Returns
        -------
        NeutralAtomNoiseModel
            Physics noise model matching device characteristics.
        """
        return NeutralAtomNoiseModel(
            species=self.species,
            rabi_freq_mhz=self.rabi_freq_typical_mhz,
            atom_loss_prob=self.atom_loss_per_gate,
            laser_intensity_noise_frac=_fidelity_to_intensity_noise(
                self.single_qubit_fidelity
            ),
            readout_error=1.0 - self.readout_fidelity,
        )

    def info(self) -> dict[str, Any]:
        """Return device information as a dictionary."""
        return {
            "name": self.name,
            "vendor": self.vendor,
            "max_atoms": self.max_atoms,
            "max_sites": self.max_sites,
            "species": self.species.name,
            "geometry": self.geometry.name,
            "spacing_um": self.default_spacing_um,
            "blockade_radius_um": self.blockade_radius_um(),
            "rabi_freq_typical_mhz": self.rabi_freq_typical_mhz,
            "1q_fidelity": self.single_qubit_fidelity,
            "2q_fidelity": self.two_qubit_fidelity,
            "readout_fidelity": self.readout_fidelity,
            "repetition_rate_hz": self.repetition_rate_hz,
            "connectivity": self.connectivity,
        }


def _fidelity_to_intensity_noise(fidelity: float) -> float:
    """Rough inverse: estimate intensity noise from gate fidelity.

    Assumes intensity noise is dominant single-qubit error source:
        1 - F ~ (delta_Omega/Omega * pi)^2 / 4
    """
    import math

    error = 1.0 - fidelity
    if error <= 0:
        return 0.0
    return math.sqrt(4.0 * error) / math.pi


# ======================================================================
# Device presets
# ======================================================================


class DevicePresets:
    """Pre-calibrated device configurations for commercial neutral-atom systems."""

    @staticmethod
    def quera_aquila() -> DeviceSpec:
        """QuEra Aquila: 256-atom 87Rb analog/digital quantum processor.

        Published specifications (Wurtz et al., 2023; AWS Braket):
        - Up to 256 qubits in a programmable 2D geometry
        - Analog Hamiltonian simulation mode
        - Rydberg blockade-based entanglement
        - 4 um minimum atom spacing
        - Global Rabi drive up to ~15 rad/us
        """
        return DeviceSpec(
            name="Aquila",
            vendor="QuEra",
            max_atoms=256,
            max_sites=256,
            species=AtomSpecies.RB87,  # type: ignore[attr-defined]
            geometry=ArrayGeometry.RECTANGULAR,
            default_spacing_um=4.0,
            rabi_freq_max_mhz=2.4,  # ~15 rad/us / (2*pi)
            rabi_freq_typical_mhz=1.5,
            single_qubit_fidelity=0.995,
            two_qubit_fidelity=0.975,
            readout_fidelity=0.97,
            atom_loss_per_gate=0.003,
            repetition_rate_hz=4.0,
            connectivity="reconfigurable, programmable geometry",
            notes=(
                "First commercially available neutral-atom QPU. "
                "Primary mode: analog Hamiltonian simulation. "
                "Available via Amazon Braket."
            ),
        )

    @staticmethod
    def atom_computing_1225() -> DeviceSpec:
        """Atom Computing: 1225-site 171Yb system.

        Published specifications (Norcia et al., Nature 2023):
        - 1225-site optical tweezer array (35x35)
        - 171Yb atoms with nuclear spin qubit
        - Demonstrated >1000 atoms loaded
        - Mid-circuit readout capability
        """
        return DeviceSpec(
            name="Phoenix",
            vendor="Atom Computing",
            max_atoms=1000,
            max_sites=1225,
            species=AtomSpecies.YB171,  # type: ignore[attr-defined]
            geometry=ArrayGeometry.RECTANGULAR,
            default_spacing_um=3.5,
            rabi_freq_max_mhz=2.0,
            rabi_freq_typical_mhz=1.0,
            single_qubit_fidelity=0.997,
            two_qubit_fidelity=0.990,
            readout_fidelity=0.985,
            atom_loss_per_gate=0.002,
            repetition_rate_hz=2.0,
            connectivity="reconfigurable, mid-circuit measurement",
            notes=(
                "Largest neutral-atom array demonstrated. "
                "171Yb nuclear spin qubit provides long coherence. "
                "Mid-circuit readout and feed-forward demonstrated."
            ),
        )

    @staticmethod
    def pasqal_fresnel() -> DeviceSpec:
        """Pasqal Fresnel: 100-atom 87Rb digital quantum processor.

        Published specifications (Pasqal, 2024):
        - Up to 100 qubits
        - Digital gate-based and analog modes
        - 87Rb atoms with Rydberg entanglement
        - Available via cloud access
        """
        return DeviceSpec(
            name="Fresnel",
            vendor="Pasqal",
            max_atoms=100,
            max_sites=100,
            species=AtomSpecies.RB87,  # type: ignore[attr-defined]
            geometry=ArrayGeometry.RECTANGULAR,
            default_spacing_um=5.0,
            rabi_freq_max_mhz=3.0,
            rabi_freq_typical_mhz=2.0,
            single_qubit_fidelity=0.996,
            two_qubit_fidelity=0.955,
            readout_fidelity=0.96,
            atom_loss_per_gate=0.005,
            repetition_rate_hz=5.0,
            connectivity="reconfigurable, 2D programmable",
            notes=(
                "Pasqal's first commercial QPU. "
                "Supports both analog and digital operation modes. "
                "Developed from CNRS/Browaeys group technology."
            ),
        )

    @staticmethod
    def harvard_quera_48_logical() -> DeviceSpec:
        """Harvard/QuEra 48-logical-qubit system.

        Published specifications (Bluvstein et al., Nature 2024):
        - 280 physical qubits
        - 48 logical qubits demonstrated
        - Transversal CNOT gates between logical qubits
        - Atom transport for reconfigurable connectivity
        - 87Rb with Rydberg state n=70
        """
        return DeviceSpec(
            name="48-Logical",
            vendor="Harvard/QuEra",
            max_atoms=280,
            max_sites=280,
            species=AtomSpecies.RB87,  # type: ignore[attr-defined]
            geometry=ArrayGeometry.RECTANGULAR,
            default_spacing_um=3.5,
            rabi_freq_max_mhz=2.5,
            rabi_freq_typical_mhz=1.5,
            single_qubit_fidelity=0.998,
            two_qubit_fidelity=0.985,
            readout_fidelity=0.98,
            atom_loss_per_gate=0.002,
            repetition_rate_hz=3.0,
            connectivity="reconfigurable, atom transport, logical qubits",
            notes=(
                "Demonstrated 48 error-corrected logical qubits. "
                "Transversal entangling gates between logical qubits. "
                "Atom shuttling enables all-to-all logical connectivity."
            ),
        )

    @staticmethod
    def list_all() -> list[DeviceSpec]:
        """Return a list of all available device presets."""
        return [
            DevicePresets.quera_aquila(),
            DevicePresets.atom_computing_1225(),
            DevicePresets.pasqal_fresnel(),
            DevicePresets.harvard_quera_48_logical(),
        ]
