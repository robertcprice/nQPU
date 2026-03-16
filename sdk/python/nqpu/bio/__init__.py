"""nQPU Quantum Biology -- Quantum effects in biological systems.

Five major quantum biological phenomena are modelled:

1. **Photosynthesis**: Coherent energy transfer in the FMO complex and
   other light-harvesting systems via Lindblad master equation dynamics.

2. **Enzyme Tunneling**: WKB proton tunneling through enzyme active-site
   barriers with kinetic isotope effect (KIE) calculations.

3. **Olfaction**: Turin's vibrational theory of smell using inelastic
   electron tunneling spectroscopy (IETS) in olfactory receptors.

4. **Avian Navigation**: Radical pair mechanism in cryptochrome for
   bird magnetoreception under Earth's magnetic field.

5. **DNA Mutation**: Spontaneous tautomeric shifts via proton tunneling
   along hydrogen bonds in Watson-Crick base pairs.

Example:
    from nqpu.bio import FMOComplex, EnzymeTunneling, RadicalPair

    fmo = FMOComplex.standard()
    result = fmo.evolve(duration_fs=1000.0, steps=2000)

    enzyme = EnzymeTunneling.from_barrier(0.3, 0.05)
    print(f"KIE: {enzyme.kie_ratio():.1f}")

    rp = RadicalPair.cryptochrome()
    print(f"Compass anisotropy: {rp.compass_anisotropy(18):.3f}")
"""

# -- Photosynthesis ----------------------------------------------------------
from .photosynthesis import (
    FMOComplex,
    FMOEvolution,
    PhotosyntheticSystem,
    QuantumTransportEfficiency,
    DecoherenceModel,
    SpectralDensity,
    SpectralDensityType,
)

# -- Enzyme tunneling --------------------------------------------------------
from .tunneling import (
    EnzymeTunneling,
    TunnelingBarrier,
    BarrierShape,
    TunnelingSensitivity,
    ENZYMES,
)

# -- Olfaction ---------------------------------------------------------------
from .olfaction import (
    QuantumNose,
    OlfactoryReceptor,
    MolecularVibration,
    Odorant,
    OdorDiscrimination,
    ODORANTS,
)

# -- Avian navigation --------------------------------------------------------
from .avian_navigation import (
    RadicalPair,
    CryptochromeModel,
    CompassSensitivity,
    DecoherenceEffects,
)

# -- DNA mutation ------------------------------------------------------------
from .dna_mutation import (
    BasePair,
    BasePairType,
    DoubleWellPotential,
    TautomerTunneling,
    MutationRate,
)

__all__ = [
    # Photosynthesis
    "FMOComplex",
    "FMOEvolution",
    "PhotosyntheticSystem",
    "QuantumTransportEfficiency",
    "DecoherenceModel",
    "SpectralDensity",
    "SpectralDensityType",
    # Enzyme tunneling
    "EnzymeTunneling",
    "TunnelingBarrier",
    "BarrierShape",
    "TunnelingSensitivity",
    "ENZYMES",
    # Olfaction
    "QuantumNose",
    "OlfactoryReceptor",
    "MolecularVibration",
    "Odorant",
    "OdorDiscrimination",
    "ODORANTS",
    # Avian navigation
    "RadicalPair",
    "CryptochromeModel",
    "CompassSensitivity",
    "DecoherenceEffects",
    # DNA mutation
    "BasePair",
    "BasePairType",
    "DoubleWellPotential",
    "TautomerTunneling",
    "MutationRate",
]
