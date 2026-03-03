#!/usr/bin/env python3
"""
HuggingFace Spaces entry point for NQPU Drug Design Platform.

This is a simplified version optimized for HF Spaces deployment.
Run locally with: python app.py
"""

import sys
import os

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from typing import Tuple, Optional

# Import drug design module
from nqpu_drug_design import (
    Molecule, evaluate_drug_likeness, predict_admet,
    ecfp4, ecfp6, maccs_keys, tanimoto_similarity,
    DrugDiscoveryPipeline
)

# Example SMILES for demo
EXAMPLE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # Caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CC(=O)NC1=CC=C(C=C1)O",  # Paracetamol
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine (alt)
    "CC(C)(C)NCC(O)C1=CC(=C(C=C1)O)CO",  # Salbutamol
]


def analyze_molecule(smiles: str) -> str:
    """Comprehensive molecule analysis."""
    if not smiles or not smiles.strip():
        return "Please enter a SMILES string."

    smiles = smiles.strip()
    try:
        mol = Molecule.from_smiles(smiles, "Input")
    except Exception as e:
        return f"**Error parsing SMILES**: {str(e)}"

    # Calculate properties
    mw = mol.molecular_weight()
    logp = mol.estimated_log_p()
    hbd = mol.h_bond_donors()
    hba = mol.h_bond_acceptors()
    rotb = mol.rotatable_bonds()
    psa = mol.polar_surface_area()
    sa = mol.synthetic_accessibility()

    # Drug-likeness
    drug_result = evaluate_drug_likeness(mol)

    # Lipinski status
    lipinski_status = "✅ PASSES" if drug_result.lipinski_violations == 0 else f"⚠️ {drug_result.lipinski_violations} VIOLATION(S)"

    # QED interpretation
    if drug_result.qed_score >= 0.7:
        qed_status = "🟢 Excellent"
    elif drug_result.qed_score >= 0.5:
        qed_status = "🟡 Good"
    elif drug_result.qed_score >= 0.3:
        qed_status = "🟠 Moderate"
    else:
        qed_status = "🔴 Poor"

    # ADMET
    admet_results = predict_admet(mol)

    # Fingerprints
    fp4 = ecfp4(mol, n_bits=2048)
    maccs = maccs_keys(mol)

    # Build output
    output = f"""
# 🧬 Analysis Results: `{smiles}`

## 📊 Molecular Properties

| Property | Value | Threshold | Status |
|----------|-------|-----------|--------|
| **Molecular Weight** | {mw:.2f} Da | ≤500 | {"✅" if mw <= 500 else "❌"} |
| **LogP** | {logp:.2f} | ≤5 | {"✅" if logp <= 5 else "❌"} |
| **H-Bond Donors** | {hbd} | ≤5 | {"✅" if hbd <= 5 else "❌"} |
| **H-Bond Acceptors** | {hba} | ≤10 | {"✅" if hba <= 10 else "❌"} |
| **Rotatable Bonds** | {rotb} | ≤10 | {"✅" if rotb <= 10 else "⚠️"} |
| **Polar Surface Area** | {psa:.1f} Å² | ≤140 | {"✅" if psa <= 140 else "⚠️"} |
| **Synthetic Accessibility** | {sa:.1f}/10 | ≤6 | {"✅" if sa <= 6 else "⚠️"} |

## 💊 Drug-Likeness

### Lipinski Rule of Five
**{lipinski_status}**

### QED Score
**{drug_result.qed_score:.3f}** — {qed_status}

## 🧪 ADMET Prediction

| Property | Probability | Status |
|----------|-------------|--------|
"""
    for prop_name, result in admet_results.items():
        status = "✅" if result.passes else "⚠️"
        output += f"| {prop_name.replace('_', ' ').title()} | {result.probability:.2f} | {status} |\n"

    output += f"""
## 🔍 Fingerprints

- **ECFP4 (Morgan R=2)**: {len(fp4.bits)} bits set ({fp4.density:.1%} density)
- **MACCS 166-Keys**: {maccs.bit_count}/166 keys active

---
*Analysis by NQPU Drug Design Platform*
"""
    return output


def calculate_similarity(smiles1: str, smiles2: str) -> str:
    """Calculate molecular similarity."""
    if not smiles1.strip() or not smiles2.strip():
        return "Please enter both SMILES strings."

    try:
        mol1 = Molecule.from_smiles(smiles1.strip(), "Mol1")
        mol2 = Molecule.from_smiles(smiles2.strip(), "Mol2")
    except Exception as e:
        return f"**Error**: {str(e)}"

    # Calculate similarities
    fp4_1, fp4_2 = ecfp4(mol1), ecfp4(mol2)
    fp6_1, fp6_2 = ecfp6(mol1), ecfp6(mol2)
    maccs_1, maccs_2 = maccs_keys(mol1), maccs_keys(mol2)

    tanimoto_4 = tanimoto_similarity(fp4_1, fp4_2)
    tanimoto_6 = tanimoto_similarity(fp6_1, fp6_2)
    tanimoto_maccs = tanimoto_similarity(maccs_1, maccs_2)

    # Interpretation
    if tanimoto_4 >= 0.7:
        verdict = "🟢 Very similar structures"
    elif tanimoto_4 >= 0.5:
        verdict = "🟡 Moderately similar"
    elif tanimoto_4 >= 0.3:
        verdict = "🟠 Low similarity"
    else:
        verdict = "🔴 Different structures"

    return f"""
# 📊 Similarity Analysis

**Molecule 1**: `{smiles1}`
**Molecule 2**: `{smiles2}`

## Tanimoto Similarity

| Fingerprint | Similarity |
|-------------|------------|
| **ECFP4** | {tanimoto_4:.3f} |
| **ECFP6** | {tanimoto_6:.3f} |
| **MACCS** | {tanimoto_maccs:.3f} |

## Verdict
{verdict}
"""


# Create Gradio interface
with gr.Blocks(
    title="NQPU Drug Design",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gradio-container {max-width: 1000px !important}"
) as demo:

    gr.Markdown("""
    # 🧬 NQPU Drug Design Platform

    Quantum-enhanced drug discovery with industry-standard fingerprints (ECFP4, MACCS).

    ### Features: SMILES parsing • Property calculation • Drug-likeness (Lipinski, QED) • ADMET prediction • Molecular similarity
    """)

    with gr.Tabs():
        with gr.TabItem("🔬 Analyze"):
            with gr.Row():
                with gr.Column(scale=1):
                    analyze_input = gr.Textbox(
                        label="SMILES String",
                        placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O",
                        lines=1
                    )
                    gr.Examples(examples=EXAMPLE_SMILES, inputs=analyze_input)
                    analyze_btn = gr.Button("Analyze", variant="primary")
                with gr.Column(scale=2):
                    analyze_output = gr.Markdown()

            analyze_btn.click(analyze_molecule, inputs=[analyze_input], outputs=[analyze_output])

        with gr.TabItem("📊 Similarity"):
            with gr.Row():
                with gr.Column(scale=1):
                    sim1 = gr.Textbox(label="SMILES 1", value="CC(=O)Oc1ccccc1C(=O)O")
                    sim2 = gr.Textbox(label="SMILES 2", value="OC1=CC=CC=C1C(=O)O")
                    sim_btn = gr.Button("Compare", variant="primary")
                with gr.Column(scale=2):
                    sim_output = gr.Markdown()

            sim_btn.click(calculate_similarity, inputs=[sim1, sim2], outputs=[sim_output])

    gr.Markdown("---\n**NQPU Drug Design** | Quantum-enhanced molecular analysis")

# For HuggingFace Spaces
if __name__ == "__main__":
    demo.launch()
