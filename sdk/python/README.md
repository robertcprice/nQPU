# NQPU Platform - Python API

Quantum computing platform for drug design, genomics, and quantum biology.

## Directory Structure

```
python/
├── core/                          # NQPU Core Platform
│   ├── quantum_backend.py         # Quantum circuit simulation
│   ├── nqpu_api.py                # REST API server
│   ├── nqpu_dashboard.py          # Web dashboard
│   ├── nqpu_jax.py                # JAX integration
│   └── benchmark_suite.py         # Performance benchmarks
│
├── chem/                          # Chemistry & Drug Design
│   ├── nqpu_drug_design.py        # Quantum drug discovery
│   ├── bio_fingerprints.py        # Molecular fingerprints
│   ├── fingerprint_db.py          # Fingerprint database
│   ├── molecular_dynamics.py      # MD simulations
│   ├── quantum_protein_folding.py # Protein structure
│   └── tests/                     # Chemistry tests
│
├── bio/                           # Quantum Biology
│   ├── quantum_life.py            # Living quantum systems
│   ├── quantum_organism.py        # Quantum organisms
│   └── quantum_genome_tools.py    # Genomic analysis
│
├── demos/                         # Example Applications
│   └── gradio_demo.py             # Gradio interface
│
└── tests/                         # Test Suite
```

## Related Projects

| Project | Location | Purpose |
|---------|----------|---------|
| **NQPU Rust Core** | `../src/` | High-performance quantum simulation |
| **Organic Neural** | `../../../organic_neural/` | Biologically-inspired AI networks |
| **Entropy Bio** | `../../src/entropy/` | Bio signals for AI control |

---

## Drug Design Module (chem/)

### What's Production-Ready vs Experimental

### ✅ Production-Ready (Use These)
- **SMILES Parsing**: Parse molecular structures
- **Property Calculation**: MW, LogP, HBD/HBA, TPSA, SA
- **Drug-Likeness**: Lipinski Rule of Five, QED score
- **ADMET Prediction**: Property predictions
- **Fingerprints**: ECFP4/ECFP6, MACCS, Atom Pair, Topological Torsion
- **Similarity**: Tanimoto and Dice coefficients
- **Fingerprint Database**: Compound search and clustering

### ⚠️ Experimental (Research Only)
- **Quantum-enhanced similarity**: Quantum-inspired math, not actual quantum computing
- **Bio-conditioned fingerprints**: Theoretical concept, no validated use case

## Installation

```bash
# Core dependencies
pip install numpy

# Optional: Web demo
pip install gradio
```

## Quick Start

```python
from nqpu_drug_design import Molecule, ecfp4, maccs_keys, atom_pair_fingerprint, tanimoto_similarity

# Parse a molecule
mol = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O", "Aspirin")

# Calculate properties
print(f"MW: {mol.molecular_weight():.2f} Da")
print(f"LogP: {mol.estimated_log_p():.2f}")
print(f"HBD: {mol.h_bond_donors()}")
print(f"HBA: {mol.h_bond_acceptors()}")
print(f"TPSA: {mol.polar_surface_area():.1f} Å²")

# Generate fingerprints (industry standard)
ecfp = ecfp4(mol, n_bits=2048)
ap = atom_pair_fingerprint(mol)
maccs = maccs_keys(mol)

print(f"ECFP4 bits: {len(ecfp.bits)}")
print(f"Atom Pair bits: {len(ap.bits)}")
print(f"MACCS keys: {maccs.bit_count}")
```

## Fingerprint Types

### ECFP4/ECFP6 (Morgan Circular) - Best for similarity search
```python
fp4 = ecfp4(mol)  # Radius 2
fp6 = ecfp6(mol)  # Radius 3 (larger neighborhood)
```

### MACCS 166-Key - Best for QSAR models
```python
maccs = maccs_keys(mol)  # Pharmacophore features
```

### Atom Pair - Best for scaffold hopping
```python
ap = atom_pair_fingerprint(mol)  # Distance-encoded atom pairs
```

### Topological Torsion - Best for lead optimization
```python
tt = topological_torsion(mol)  # 4-atom sequences
```

## Molecular Similarity

```python
aspirin = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
salicylic = Molecule.from_smiles("OC1=CC=CC=C1C(=O)O")

# Different fingerprints give different similarity values
print(f"ECFP4 similarity: {tanimoto_similarity(ecfp4(aspirin), ecfp4(salicylic)):.3f}")
print(f"Atom Pair similarity: {tanimoto_similarity(atom_pair_fingerprint(aspirin), atom_pair_fingerprint(salicylic)):.3f}")
print(f"MACCS similarity: {tanimoto_similarity(maccs_keys(aspirin), maccs_keys(salicylic)):.3f}")
```

Output:
```
ECFP4 similarity: 0.109
Atom Pair similarity: 0.765
MACCS similarity: 0.700
```

Note: Different fingerprints capture different aspects of molecular structure. ECFP is more sensitive to local environment; Atom Pair captures global topology.

## Fingerprint Database

```python
from fingerprint_db import FingerprintDatabase, load_drug_bank

db = FingerprintDatabase()
db = load_drug_bank(db)  # Load 45 FDA-approved drugs

# Search for similar compounds
similar = db.search_similar(aspirin, k=10, min_similarity=0.3)
for hit in similar:
    print(f"{hit.entry.name}: {hit.similarity:.3f}")
```

## Benchmark Results

Property calculations validated against PubChem on 79 common drugs:

| Property | Accuracy |
|----------|----------|
| Molecular Weight | 100% |
| H-Bond Donors | 97% |
| H-Bond Acceptors | 95% |
| TPSA | 90% |

## License

MIT License
