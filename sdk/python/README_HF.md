---
title: NQPU Drug Design
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Quantum-enhanced drug discovery with industry-standard fingerprints
---

# 🧬 NQPU Drug Design Platform

Quantum-enhanced drug discovery toolkit with industry-standard molecular fingerprints.

## Features

- **SMILES Parsing**: Parse and validate molecular structures
- **Property Calculation**: MW, LogP, HBD/HBA, TPSA, Synthetic Accessibility
- **Drug-Likeness**: Lipinski Rule of Five, QED score
- **ADMET Prediction**: Absorption, toxicity, solubility predictions
- **Molecular Fingerprints**: ECFP4/ECFP6 (Morgan), MACCS 166-key
- **Similarity Search**: Tanimoto and Dice similarity metrics

## Usage

1. Enter a SMILES string (e.g., `CC(=O)Oc1ccccc1C(=O)O` for Aspirin)
2. Click "Analyze" to see comprehensive molecular analysis
3. Use the Similarity tab to compare two molecules

## Example Molecules

- `CC(=O)Oc1ccccc1C(=O)O` - Aspirin
- `Cn1cnc2c1c(=O)n(c(=O)n2C)C` - Caffeine
- `CC(C)CC1=CC=C(C=C1)C(C)C(=O)O` - Ibuprofen
- `CC(=O)NC1=CC=C(C=C1)O` - Paracetamol

## Technical Details

### Fingerprints

- **ECFP4** (Morgan radius 2): Extended Connectivity Fingerprint for similarity searching
- **ECFP6** (Morgan radius 3): Extended neighborhood coverage
- **MACCS 166-Key**: Pharmacophore fingerprint for QSAR modeling

### Drug-Likeness

- **Lipinski Rule of Five**: MW≤500, LogP≤5, HBD≤5, HBA≤10
- **QED Score**: Quantitative Estimate of Drug-likeness (0-1 scale)

## License

MIT License

## Links

- [GitHub](https://github.com/nqpu-metal/nqpu-metal)
- [Documentation](https://github.com/nqpu-metal/nqpu-metal/blob/main/python/README.md)
