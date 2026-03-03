#!/usr/bin/env python3
"""
Gradio Web Demo for NQPU Drug Design Platform.

Showcases quantum-enhanced drug discovery capabilities including:
- SMILES parsing and property calculation
- Drug-likeness evaluation (Lipinski, QED)
- ADMET prediction
- Industry-standard fingerprints (ECFP4, MACCS)
- Molecular similarity search
- Virtual screening pipeline

Run with: python gradio_demo.py
Or deploy to HuggingFace Spaces.
"""

import sys
sys.path.insert(0, '/Users/bobbyprice/projects/entropy/nqpu_sim/quantum/nqpu-metal/python')

import gradio as gr
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

# Import drug design module
from nqpu_drug_design import (
    Molecule, evaluate_drug_likeness, predict_admet,
    ecfp4, ecfp6, maccs_keys, tanimoto_similarity,
    DrugDiscoveryPipeline, AdmetProperty
)

# Try to import plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_smiles(smiles: str) -> Tuple[Optional[Molecule], str]:
    """Parse SMILES and return molecule or error message."""
    if not smiles or not smiles.strip():
        return None, "Please enter a SMILES string."

    smiles = smiles.strip()
    try:
        mol = Molecule.from_smiles(smiles, "Input")
        return mol, ""
    except Exception as e:
        return None, f"Error parsing SMILES: {str(e)}"


def format_properties(mol: Molecule) -> str:
    """Format molecular properties as markdown."""
    if mol is None:
        return "No molecule loaded."

    mw = mol.molecular_weight()
    logp = mol.estimated_log_p()
    hbd = mol.h_bond_donors()
    hba = mol.h_bond_acceptors()
    rotb = mol.rotatable_bonds()
    psa = mol.polar_surface_area()
    sa = mol.synthetic_accessibility()

    result = f"""
## Molecular Properties

| Property | Value |
|----------|-------|
| **Molecular Weight** | {mw:.2f} Da |
| **LogP (estimated)** | {logp:.2f} |
| **H-Bond Donors** | {hbd} |
| **H-Bond Acceptors** | {hba} |
| **Rotatable Bonds** | {rotb} |
| **Polar Surface Area** | {psa:.1f} Å² |
| **Synthetic Accessibility** | {sa:.1f}/10 |

### Atom Counts
- **Total atoms**: {len(mol.atoms)}
- **Heavy atoms**: {sum(1 for a in mol.atoms if a[0] != 'H')}
- **Ring bonds**: {len(mol.ring_bonds)}
- **Aromatic atoms**: {len(mol.aromatic_atoms)}
"""
    return result


def format_drug_likeness(mol: Molecule) -> str:
    """Format drug-likeness results as markdown."""
    if mol is None:
        return "No molecule loaded."

    result = evaluate_drug_likeness(mol)
    props = result.properties

    # Lipinski Rule of Five
    lipinski_status = "✅ PASSES" if result.lipinski_violations == 0 else f"⚠️ {result.lipinski_violations} VIOLATION(S)"

    lipinski_details = []
    if props.molecular_weight > 500:
        lipinski_details.append(f"- MW > 500 Da ({props.molecular_weight:.1f})")
    if props.log_p > 5:
        lipinski_details.append(f"- LogP > 5 ({props.log_p:.1f})")
    if props.h_bond_donors > 5:
        lipinski_details.append(f"- HBD > 5 ({props.h_bond_donors})")
    if props.h_bond_acceptors > 10:
        lipinski_details.append(f"- HBA > 10 ({props.h_bond_acceptors})")

    # QED score interpretation
    if result.qed_score >= 0.7:
        qed_status = "🟢 Excellent"
    elif result.qed_score >= 0.5:
        qed_status = "🟡 Good"
    elif result.qed_score >= 0.3:
        qed_status = "🟠 Moderate"
    else:
        qed_status = "🔴 Poor"

    output = f"""
## Drug-Likeness Assessment

### Lipinski Rule of Five
{lipinski_status}

{chr(10).join(lipinski_details) if lipinski_details else "All criteria satisfied!"}

### QED Score (Quantitative Estimate of Drug-likeness)
**{result.qed_score:.3f}** — {qed_status}

### Synthetic Accessibility
**{result.synthetic_accessibility:.2f}** (1 = easy, 10 = difficult)

### Property Summary
| Criterion | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| MW ≤ 500 | {props.molecular_weight:.1f} | 500 | {"✅" if props.molecular_weight <= 500 else "❌"} |
| LogP ≤ 5 | {props.log_p:.1f} | 5 | {"✅" if props.log_p <= 5 else "❌"} |
| HBD ≤ 5 | {props.h_bond_donors} | 5 | {"✅" if props.h_bond_donors <= 5 else "❌"} |
| HBA ≤ 10 | {props.h_bond_acceptors} | 10 | {"✅" if props.h_bond_acceptors <= 10 else "❌"} |
"""
    return output


def format_admet(mol: Molecule) -> str:
    """Format ADMET predictions as markdown."""
    if mol is None:
        return "No molecule loaded."

    results = predict_admet(mol)

    output = "## ADMET Predictions\n\n"
    output += "| Property | Probability | Status | Confidence |\n"
    output += "|----------|-------------|--------|------------|\n"

    status_icons = {True: "✅", False: "⚠️"}

    for prop_name, result in results.items():
        status = status_icons[result.passes]
        output += f"| {prop_name.replace('_', ' ').title()} | {result.probability:.2f} | {status} | {result.confidence:.2f} |\n"

    # Summary
    passed = sum(1 for r in results.values() if r.passes)
    total = len(results)

    output += f"\n**Summary**: {passed}/{total} properties pass thresholds\n"

    # Key interpretations
    output += "\n### Interpretation Guide\n"
    output += "- **Absorption**: High probability = good oral bioavailability\n"
    output += "- **BBB Permeability**: High = crosses blood-brain barrier\n"
    output += "- **Toxicity**: Low probability = safer profile\n"
    output += "- **Solubility**: High = better dissolution\n"

    return output


def format_fingerprints(mol: Molecule) -> str:
    """Format fingerprint information as markdown."""
    if mol is None:
        return "No molecule loaded."

    # Generate fingerprints
    fp4 = ecfp4(mol, n_bits=2048)
    fp6 = ecfp6(mol, n_bits=2048)
    maccs = maccs_keys(mol)

    output = """
## Molecular Fingerprints

### ECFP4 (Morgan Radius 2)
"""
    output += f"- **Bits set**: {len(fp4.bits)}/2048 ({fp4.density:.1%} density)\n"
    output += f"- **Bitstring preview**: `{fp4.to_bitstring()[:80]}...`\n"

    output += "\n### ECFP6 (Morgan Radius 3)\n"
    output += f"- **Bits set**: {len(fp6.bits)}/2048 ({fp6.density:.1%} density)\n"

    output += "\n### MACCS 166-Keys\n"
    output += f"- **Keys set**: {maccs.bit_count}/166\n"
    set_keys = [i+1 for i, k in enumerate(maccs.keys) if k]
    output += f"- **Active keys**: {set_keys[:30]}{'...' if len(set_keys) > 30 else ''}\n"

    output += "\n### Key Descriptions\n"
    key_meanings = {
        2: "Contains Carbon",
        7: "Contains Nitrogen",
        8: "Contains Oxygen",
        11: "Has ring",
        16: "Has aromatic ring",
        26: "Has carboxylic acid",
        52: "Has double bond",
    }
    for key_num, meaning in key_meanings.items():
        if maccs.keys[key_num - 1]:
            output += f"- ✅ Key {key_num}: {meaning}\n"

    return output


def calculate_similarity(smiles1: str, smiles2: str) -> str:
    """Calculate similarity between two molecules."""
    mol1, err1 = parse_smiles(smiles1)
    mol2, err2 = parse_smiles(smiles2)

    if err1:
        return f"Error in first SMILES: {err1}"
    if err2:
        return f"Error in second SMILES: {err2}"

    # Calculate all similarity metrics
    fp4_1, fp4_2 = ecfp4(mol1), ecfp4(mol2)
    fp6_1, fp6_2 = ecfp6(mol1), ecfp6(mol2)
    maccs_1, maccs_2 = maccs_keys(mol1), maccs_keys(mol2)

    tanimoto_4 = tanimoto_similarity(fp4_1, fp4_2)
    tanimoto_6 = tanimoto_similarity(fp6_1, fp6_2)
    tanimoto_maccs = tanimoto_similarity(maccs_1, maccs_2)
    dice_4 = fp4_1.dice(fp4_2)

    output = f"""
## Similarity Analysis

**Molecule 1**: {smiles1}
**Molecule 2**: {smiles2}

### Tanimoto Similarity

| Fingerprint | Similarity |
|-------------|------------|
| ECFP4 | {tanimoto_4:.3f} |
| ECFP6 | {tanimoto_6:.3f} |
| MACCS | {tanimoto_maccs:.3f} |

### Dice Similarity
| ECFP4 | {dice_4:.3f}

### Interpretation
- **0.0-0.3**: Very different molecules
- **0.3-0.5**: Low similarity
- **0.5-0.7**: Moderate similarity
- **0.7-0.9**: High similarity
- **0.9-1.0**: Very similar/identical

**Verdict**: {"Similar structures" if tanimoto_4 > 0.5 else "Different structures"}
"""
    return output


def screen_library_ui(smiles_list: str, target_smiles: str) -> str:
    """Screen a library of ligands against a target."""
    if not smiles_list.strip():
        return "Please enter SMILES strings (one per line)."

    # Parse target
    target, err = parse_smiles(target_smiles)
    if err:
        return f"Target error: {err}"

    # Parse library
    ligand_smiles = [s.strip() for s in smiles_list.strip().split('\n') if s.strip()]
    ligands = []
    errors = []

    for i, smi in enumerate(ligand_smiles):
        mol, err = parse_smiles(smi)
        if err:
            errors.append(f"Line {i+1}: {err}")
        else:
            ligands.append((mol, smi))

    if errors:
        return f"Parsing errors:\n" + "\n".join(errors)

    if not ligands:
        return "No valid ligands found."

    # Generate fingerprints and calculate similarities
    target_fp = ecfp4(target)
    results = []

    for mol, smi in ligands:
        fp = ecfp4(mol)
        sim = tanimoto_similarity(target_fp, fp)
        drug_result = evaluate_drug_likeness(mol)
        results.append({
            'smiles': smi,
            'similarity': sim,
            'qed': drug_result.qed_score,
            'lipinski': drug_result.lipinski_violations
        })

    # Sort by similarity
    results.sort(key=lambda x: x['similarity'], reverse=True)

    output = f"""
## Virtual Screening Results

**Target**: {target_smiles}
**Library size**: {len(results)} compounds

### Ranked by ECFP4 Similarity

| Rank | SMILES | Similarity | QED | Lipinski |
|------|--------|------------|-----|----------|
"""
    for i, r in enumerate(results[:20]):
        output += f"| {i+1} | `{r['smiles'][:30]}{'...' if len(r['smiles']) > 30 else ''}` | {r['similarity']:.3f} | {r['qed']:.2f} | {r['lipinski']} |\n"

    if len(results) > 20:
        output += f"\n*... and {len(results) - 20} more*\n"

    return output


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

EXAMPLE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # Caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine (alt)
    "CC(C)(C)NCC(O)C1=CC(=C(C=C1)O)CO",  # Salbutamol
    "CC12CCC(CC1CCC3C2CCC4(C3CCC4=O)C)O",  # Testosterone
    "c1ccc2c(c1)ccc3ccccc32",  # Phenanthrene
    "CC(=O)NC1=CC=C(C=C1)O",  # Paracetamol
]

DESCRIPTION = """
# 🧬 NQPU Drug Design Platform

Quantum-enhanced drug discovery toolkit with industry-standard fingerprints.

### Features
- **SMILES Parsing**: Parse and validate molecular structures
- **Property Calculation**: MW, LogP, HBD/HBA, TPSA, SA
- **Drug-Likeness**: Lipinski Rule of Five, QED score
- **ADMET Prediction**: Absorption, toxicity, solubility
- **Fingerprints**: ECFP4/ECFP6, MACCS 166-key
- **Similarity Search**: Tanimoto and Dice metrics

### Try These Examples
"""

with gr.Blocks(
    title="NQPU Drug Design",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=".gradio-container {max-width: 1200px !important}"
) as demo:

    gr.Markdown(DESCRIPTION)

    # State for molecule
    mol_state = gr.State(None)

    with gr.Tabs():
        # Tab 1: Property Calculator
        with gr.TabItem("🔬 Property Calculator"):
            with gr.Row():
                with gr.Column(scale=1):
                    smiles_input = gr.Textbox(
                        label="SMILES String",
                        placeholder="Enter SMILES (e.g., CC(=O)Oc1ccccc1C(=O)O)",
                        lines=1
                    )
                    gr.Examples(examples=EXAMPLE_SMILES, inputs=smiles_input)
                    analyze_btn = gr.Button("Analyze Molecule", variant="primary")

                with gr.Column(scale=2):
                    props_output = gr.Markdown(label="Properties")

            analyze_btn.click(
                fn=lambda s: format_properties(parse_smiles(s)[0]),
                inputs=[smiles_input],
                outputs=[props_output]
            )

        # Tab 2: Drug-Likeness
        with gr.TabItem("💊 Drug-Likeness"):
            with gr.Row():
                with gr.Column(scale=1):
                    dl_input = gr.Textbox(
                        label="SMILES String",
                        placeholder="Enter SMILES",
                        lines=1
                    )
                    gr.Examples(examples=EXAMPLE_SMILES, inputs=dl_input)
                    dl_btn = gr.Button("Evaluate Drug-Likeness", variant="primary")

                with gr.Column(scale=2):
                    dl_output = gr.Markdown()

            dl_btn.click(
                fn=lambda s: format_drug_likeness(parse_smiles(s)[0]),
                inputs=[dl_input],
                outputs=[dl_output]
            )

        # Tab 3: ADMET
        with gr.TabItem("🧪 ADMET Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    admet_input = gr.Textbox(
                        label="SMILES String",
                        placeholder="Enter SMILES",
                        lines=1
                    )
                    gr.Examples(examples=EXAMPLE_SMILES, inputs=admet_input)
                    admet_btn = gr.Button("Predict ADMET", variant="primary")

                with gr.Column(scale=2):
                    admet_output = gr.Markdown()

            admet_btn.click(
                fn=lambda s: format_admet(parse_smiles(s)[0]),
                inputs=[admet_input],
                outputs=[admet_output]
            )

        # Tab 4: Fingerprints
        with gr.TabItem("🔍 Fingerprints"):
            with gr.Row():
                with gr.Column(scale=1):
                    fp_input = gr.Textbox(
                        label="SMILES String",
                        placeholder="Enter SMILES",
                        lines=1
                    )
                    gr.Examples(examples=EXAMPLE_SMILES, inputs=fp_input)
                    fp_btn = gr.Button("Generate Fingerprints", variant="primary")

                with gr.Column(scale=2):
                    fp_output = gr.Markdown()

            fp_btn.click(
                fn=lambda s: format_fingerprints(parse_smiles(s)[0]),
                inputs=[fp_input],
                outputs=[fp_output]
            )

        # Tab 5: Similarity
        with gr.TabItem("📊 Similarity"):
            with gr.Row():
                with gr.Column(scale=1):
                    sim_input1 = gr.Textbox(
                        label="SMILES 1",
                        placeholder="First molecule",
                        value="CC(=O)Oc1ccccc1C(=O)O"
                    )
                    sim_input2 = gr.Textbox(
                        label="SMILES 2",
                        placeholder="Second molecule",
                        value="OC1=CC=CC=C1C(=O)O"
                    )
                    sim_btn = gr.Button("Calculate Similarity", variant="primary")

                with gr.Column(scale=2):
                    sim_output = gr.Markdown()

            sim_btn.click(
                fn=calculate_similarity,
                inputs=[sim_input1, sim_input2],
                outputs=[sim_output]
            )

        # Tab 6: Virtual Screening
        with gr.TabItem("🎯 Virtual Screening"):
            with gr.Row():
                with gr.Column(scale=1):
                    target_input = gr.Textbox(
                        label="Target SMILES",
                        placeholder="Target molecule",
                        value="CC(=O)Oc1ccccc1C(=O)O"
                    )
                    library_input = gr.Textbox(
                        label="Library SMILES (one per line)",
                        placeholder="Enter SMILES strings...",
                        lines=10,
                        value="\n".join([
                            "OC1=CC=CC=C1C(=O)O",  # Salicylic acid
                            "CC(=O)O",  # Acetic acid
                            "c1ccccc1",  # Benzene
                            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
                            "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # Caffeine
                        ])
                    )
                    screen_btn = gr.Button("Screen Library", variant="primary")

                with gr.Column(scale=2):
                    screen_output = gr.Markdown()

            screen_btn.click(
                fn=screen_library_ui,
                inputs=[library_input, target_input],
                outputs=[screen_output]
            )

    # Footer
    gr.Markdown("""
---
**NQPU Drug Design Platform** | Quantum-enhanced molecular analysis

Built with ❤️ for computational chemistry and drug discovery.
""")


def launch_demo(share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
    """Launch the Gradio demo."""
    print("=" * 60)
    print("NQPU Drug Design Platform - Gradio Demo")
    print("=" * 60)
    print(f"\nStarting server on {server_name}:{server_port}")
    if share:
        print("Public sharing enabled")
    print("\nOpen your browser to http://localhost:7860")
    print("=" * 60)

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NQPU Drug Design Gradio Demo")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    launch_demo(share=args.share, server_name=args.host, server_port=args.port)
