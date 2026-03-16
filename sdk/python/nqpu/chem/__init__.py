"""nQPU Quantum Chemistry Module -- molecular Hamiltonians for VQE.

Provides a self-contained quantum chemistry pipeline:

  - **molecular**: Atom/Molecule geometry, basis sets, predefined molecules.
  - **integrals**: Overlap, kinetic, nuclear attraction, and ERI computation.
  - **fermion**: Fermionic Hamiltonian construction, JW/BK/parity mappings.
  - **ansatz**: UCCSD, UCCD, k-UpCCGSD, hardware-efficient ansatze.
  - **vqe_driver**: End-to-end VQE driver for ground-state energy.

Example:
    from nqpu.chem import MolecularVQE, h2

    vqe = MolecularVQE(h2(), basis="sto-3g")
    result = vqe.compute_ground_state()
    print(result.energy)

Legacy chemistry modules (drug design, protein folding, DNA/RNA) are
still available via lazy import for backward compatibility.
"""

from __future__ import annotations

from importlib import import_module

# ---- Direct exports from new quantum chemistry modules ----

from .molecular import (
    Atom,
    BasisFunction,
    BasisSet,
    PrimitiveGaussian,
    # Predefined molecules
    beh2,
    h2,
    h2o,
    h4_chain,
    h6_ring,
    lih,
    ANGSTROM_TO_BOHR,
)

# Rename to avoid collision with lazy-loaded drug-design Molecule
from .molecular import Molecule as ChemMolecule

from .integrals import (
    boys_function,
    compute_one_electron_integrals,
    compute_two_electron_integrals,
    electron_repulsion_integral,
    kinetic_integral,
    nuclear_attraction_integral,
    overlap_integral,
)

from .fermion import (
    FermionicHamiltonian,
    FermionicTerm,
    PauliString,
    QubitHamiltonian,
    bravyi_kitaev,
    jordan_wigner,
    parity_mapping,
)

from .ansatz import (
    UCCSD,
    UCCD,
    HardwareEfficient,
    HartreeFockState,
    kUpCCGSD,
)

from .vqe_driver import (
    MolecularVQE,
    VQEResult,
)


# ---- Lazy exports for legacy modules (drug design, etc.) ----

_LAZY_EXPORTS = {
    "Molecule": ("chem.nqpu_drug_design", "Molecule"),
    "Element": ("chem.nqpu_drug_design", "Element"),
    "BondType": ("chem.nqpu_drug_design", "BondType"),
    "DrugLikenessResult": ("chem.nqpu_drug_design", "DrugLikenessResult"),
    "AdmetResult": ("chem.nqpu_drug_design", "AdmetResult"),
    "DockingResult": ("chem.nqpu_drug_design", "DockingResult"),
    "AminoAcid": ("chem.quantum_protein_folding", "AminoAcid"),
    "Residue": ("chem.quantum_protein_folding", "Residue"),
    "Protein": ("chem.quantum_protein_folding", "Protein"),
    "QuantumProteinFolder": ("chem.quantum_protein_folding", "QuantumProteinFolder"),
    "Substrate": ("chem.quantum_protein_folding", "Substrate"),
    "ActiveSite": ("chem.quantum_protein_folding", "ActiveSite"),
    "Enzyme": ("chem.quantum_protein_folding", "Enzyme"),
    "QuantumElectronTransfer": ("chem.quantum_protein_folding", "QuantumElectronTransfer"),
    "DNA": ("chem.dna_rna_organism", "DNA"),
    "RNA": ("chem.dna_rna_organism", "RNA"),
    "Gene": ("chem.dna_rna_organism", "Gene"),
    "Organism": ("chem.dna_rna_organism", "Organism"),
    "CODON_TABLE": ("chem.dna_rna_organism", "CODON_TABLE"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    # Molecular geometry
    "Atom",
    "ChemMolecule",
    "BasisFunction",
    "BasisSet",
    "PrimitiveGaussian",
    "ANGSTROM_TO_BOHR",
    # Predefined molecules
    "h2",
    "lih",
    "h2o",
    "beh2",
    "h4_chain",
    "h6_ring",
    # Integrals
    "boys_function",
    "overlap_integral",
    "kinetic_integral",
    "nuclear_attraction_integral",
    "electron_repulsion_integral",
    "compute_one_electron_integrals",
    "compute_two_electron_integrals",
    # Fermionic operators
    "FermionicTerm",
    "FermionicHamiltonian",
    "PauliString",
    "QubitHamiltonian",
    "jordan_wigner",
    "bravyi_kitaev",
    "parity_mapping",
    # Ansatze
    "HartreeFockState",
    "UCCSD",
    "UCCD",
    "kUpCCGSD",
    "HardwareEfficient",
    # VQE driver
    "MolecularVQE",
    "VQEResult",
] + list(_LAZY_EXPORTS)
