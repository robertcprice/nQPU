"""Lazy chemistry exports for optional chemistry modules."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
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
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
