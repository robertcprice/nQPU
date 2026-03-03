#!/usr/bin/env python3
"""
Validation Test Suite for NQPU Drug Design

Compares our implementations against:
1. Known property values from PubChem/DrugBank
2. RDKit reference implementations (if available)
3. Published fingerprint values

Run with: python validation/run_validation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Tuple, Dict

# Import our implementations
from nqpu_drug_design import (
    Molecule, evaluate_drug_likeness,
    ecfp4, ecfp6, maccs_keys, atom_pair_fingerprint, topological_torsion,
    tanimoto_similarity
)

# Try to import RDKit for reference
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys
    HAS_RDKIT = True
    print("✓ RDKit available - will compare against reference")
except ImportError:
    HAS_RDKIT = False
    print("⚠ RDKit not available - comparing against known values only")

# Test dataset
from test_dataset import VALIDATION_SET, DRUG_SET


# ============================================================================
# PROPERTY VALIDATION
# ============================================================================

def validate_molecular_weight():
    """Validate MW calculations against known values."""
    print("\n" + "="*60)
    print("MOLECULAR WEIGHT VALIDATION")
    print("="*60)

    results = []

    for smiles, name, mw_ref, _, _, _, _, _ in VALIDATION_SET:
        if mw_ref is None:
            continue

        try:
            mol = Molecule.from_smiles(smiles, name)
            mw_calc = mol.molecular_weight()
            error = abs(mw_calc - mw_ref)
            rel_error = error / mw_ref * 100

            status = "✓" if rel_error < 1.0 else "✗"
            results.append({
                'name': name,
                'ref': mw_ref,
                'calc': mw_calc,
                'error': error,
                'rel_error': rel_error,
                'status': status
            })

            print(f"{status} {name:20s}: {mw_calc:8.2f} vs {mw_ref:8.2f} ({rel_error:.2f}%)")
        except Exception as e:
            print(f"✗ {name:20s}: ERROR - {e}")
            results.append({'name': name, 'status': 'error', 'error': str(e)})

    # Summary
    passed = sum(1 for r in results if r.get('status') == '✓')
    total = len(results)
    avg_error = np.mean([r.get('rel_error', 0) for r in results if r.get('status') == '✓'])

    print(f"\nSummary: {passed}/{total} passed, avg error: {avg_error:.2f}%")
    return results


def validate_hbd_hba():
    """Validate H-bond donor/acceptor counts."""
    print("\n" + "="*60)
    print("H-BOND DONOR/ACCEPTOR VALIDATION")
    print("="*60)

    # Molecules with known HBD/HBA
    test_cases = [
        ("CCO", "Ethanol", 1, 1),  # 1 OH donor, 1 O acceptor
        ("CC(=O)O", "Acetic acid", 1, 2),  # 1 COOH donor, 2 O acceptors
        ("c1ccccc1O", "Phenol", 1, 1),  # 1 OH
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin", 1, 4),  # 1 COOH, 4 O acceptors
        ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen", 1, 2),
        ("c1cc[nH]c1", "Pyrrole", 1, 0),  # Pyrrole N-H is donor, not acceptor
        ("c1ccncc1", "Pyridine", 0, 1),  # Pyridine N is acceptor only
    ]

    results = []
    for smiles, name, hbd_ref, hba_ref in test_cases:
        try:
            mol = Molecule.from_smiles(smiles, name)
            hbd_calc = mol.h_bond_donors()
            hba_calc = mol.h_bond_acceptors()

            hbd_ok = hbd_calc == hbd_ref
            hba_ok = hba_calc == hba_ref

            status = "✓" if (hbd_ok and hba_ok) else "✗"
            print(f"{status} {name:20s}: HBD {hbd_calc} vs {hbd_ref}, HBA {hba_calc} vs {hba_ref}")

            results.append({
                'name': name,
                'hbd_calc': hbd_calc, 'hbd_ref': hbd_ref,
                'hba_calc': hba_calc, 'hba_ref': hba_ref,
                'status': status
            })
        except Exception as e:
            print(f"✗ {name:20s}: ERROR - {e}")

    passed = sum(1 for r in results if r.get('status') == '✓')
    print(f"\nSummary: {passed}/{len(results)} passed")
    return results


def validate_against_rdkit():
    """Compare our implementations against RDKit."""
    if not HAS_RDKIT:
        print("\n⚠ RDKit not available, skipping comparison")
        return []

    print("\n" + "="*60)
    print("RDKit COMPARISON")
    print("="*60)

    results = []

    for smiles in DRUG_SET[:10]:  # Test first 10
        try:
            # Our implementation
            our_mol = Molecule.from_smiles(smiles, "test")
            our_mw = our_mol.molecular_weight()

            # RDKit
            rdkit_mol = Chem.MolFromSmiles(smiles)
            rdkit_mw = Descriptors.MolWt(rdkit_mol)

            error = abs(our_mw - rdkit_mol)
            rel_error = error / rdkit_mw * 100

            status = "✓" if rel_error < 1.0 else "✗"
            print(f"{status} {smiles[:30]:30s}: ours={our_mw:.2f}, rdkit={rdkit_mw:.2f} ({rel_error:.2f}%)")

            results.append({
                'smiles': smiles,
                'our_mw': our_mw,
                'rdkit_mw': rdkit_mw,
                'rel_error': rel_error,
                'status': status
            })
        except Exception as e:
            print(f"✗ {smiles[:30]:30s}: ERROR - {e}")

    passed = sum(1 for r in results if r.get('status') == '✓')
    print(f"\nSummary: {passed}/{len(results)} within 1% of RDKit")
    return results


# ============================================================================
# FINGERPRINT VALIDATION
# ============================================================================

def validate_fingerprint_properties():
    """Validate fingerprint mathematical properties."""
    print("\n" + "="*60)
    print("FINGERPRINT PROPERTY VALIDATION")
    print("="*60)

    mol = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O", "Aspirin")

    tests_passed = 0
    total_tests = 0

    # Test 1: Self-similarity should be 1.0
    total_tests += 1
    for fp_name, fp_func in [
        ("ECFP4", ecfp4),
        ("Atom Pair", atom_pair_fingerprint),
        ("Topological Torsion", topological_torsion),
    ]:
        fp = fp_func(mol)
        sim = tanimoto_similarity(fp, fp)
        if abs(sim - 1.0) < 0.001:
            print(f"✓ {fp_name}: Self-similarity = {sim:.3f}")
            tests_passed += 1
        else:
            print(f"✗ {fp_name}: Self-similarity = {sim:.3f} (expected 1.0)")

    # Test 2: Similarity symmetry
    total_tests += 1
    mol1 = Molecule.from_smiles("CCO", "Ethanol")
    mol2 = Molecule.from_smiles("CCC", "Propane")

    all_symmetric = True
    for fp_name, fp_func in [
        ("ECFP4", ecfp4),
        ("Atom Pair", atom_pair_fingerprint),
    ]:
        fp1 = fp_func(mol1)
        fp2 = fp_func(mol2)
        sim12 = tanimoto_similarity(fp1, fp2)
        sim21 = tanimoto_similarity(fp2, fp1)

        if abs(sim12 - sim21) < 0.001:
            print(f"✓ {fp_name}: Symmetric ({sim12:.3f} == {sim21:.3f})")
        else:
            print(f"✗ {fp_name}: Not symmetric ({sim12:.3f} != {sim21:.3f})")
            all_symmetric = False

    if all_symmetric:
        tests_passed += 1

    # Test 3: Different molecules should have similarity < 1.0
    total_tests += 1
    all_different = True
    mol_different = Molecule.from_smiles("c1ccccc1", "Benzene")

    for fp_name, fp_func in [("ECFP4", ecfp4), ("Atom Pair", atom_pair_fingerprint)]:
        fp1 = fp_func(mol)
        fp2 = fp_func(mol_different)
        sim = tanimoto_similarity(fp1, fp2)

        if sim < 1.0:
            print(f"✓ {fp_name}: Aspirin vs Benzene = {sim:.3f} (< 1.0)")
        else:
            print(f"✗ {fp_name}: Aspirin vs Benzene = {sim:.3f} (should be < 1.0)")
            all_different = False

    if all_different:
        tests_passed += 1

    print(f"\nSummary: {tests_passed}/{total_tests} property tests passed")
    return tests_passed, total_tests


def validate_fingerprint_discrimination():
    """Test that fingerprints can discriminate similar vs different molecules."""
    print("\n" + "="*60)
    print("FINGERPRINT DISCRIMINATION TEST")
    print("="*60)

    # Similar pair: Aspirin and Salicylic acid
    aspirin = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O", "Aspirin")
    salicylic = Molecule.from_smiles("OC1=CC=CC=C1C(=O)O", "Salicylic")

    # Different pair: Aspirin and Benzene
    benzene = Molecule.from_smiles("c1ccccc1", "Benzene")

    print("\nSimilar pair (Aspirin vs Salicylic Acid):")
    for fp_name, fp_func in [
        ("ECFP4", ecfp4),
        ("Atom Pair", atom_pair_fingerprint),
        ("MACCS", maccs_keys),
        ("Topological Torsion", topological_torsion),
    ]:
        fp1 = fp_func(aspirin)
        fp2 = fp_func(salicylic)
        sim = tanimoto_similarity(fp1, fp2)
        print(f"  {fp_name:20s}: {sim:.3f}")

    print("\nDifferent pair (Aspirin vs Benzene):")
    for fp_name, fp_func in [
        ("ECFP4", ecfp4),
        ("Atom Pair", atom_pair_fingerprint),
        ("MACCS", maccs_keys),
        ("Topological Torsion", topological_torsion),
    ]:
        fp1 = fp_func(aspirin)
        fp2 = fp_func(benzene)
        sim = tanimoto_similarity(fp1, fp2)
        print(f"  {fp_name:20s}: {sim:.3f}")

    # Check discrimination: similar pair should score higher than different pair
    print("\nDiscrimination check (similar > different):")
    discriminated = 0
    total = 0
    for fp_name, fp_func in [
        ("ECFP4", ecfp4),
        ("Atom Pair", atom_pair_fingerprint),
        ("MACCS", maccs_keys),
    ]:
        total += 1
        sim_similar = tanimoto_similarity(fp_func(aspirin), fp_func(salicylic))
        sim_different = tanimoto_similarity(fp_func(aspirin), fp_func(benzene))

        if sim_similar > sim_different:
            print(f"  ✓ {fp_name}: {sim_similar:.3f} > {sim_different:.3f}")
            discriminated += 1
        else:
            print(f"  ✗ {fp_name}: {sim_similar:.3f} <= {sim_different:.3f}")

    print(f"\nSummary: {discriminated}/{total} fingerprints correctly discriminate")
    return discriminated, total


def compare_fingerprints_with_rdkit():
    """Compare fingerprint similarities with RDKit."""
    if not HAS_RDKIT:
        return []

    print("\n" + "="*60)
    print("FINGERPRINT RDKit COMPARISON")
    print("="*60)

    test_pairs = [
        ("CCO", "CCC", "Ethanol vs Propane"),
        ("c1ccccc1", "c1ccccc1C", "Benzene vs Toluene"),
        ("CC(=O)Oc1ccccc1C(=O)O", "OC1=CC=CC=C1C(=O)O", "Aspirin vs Salicylic"),
    ]

    for smi1, smi2, pair_name in test_pairs:
        mol1 = Molecule.from_smiles(smi1)
        mol2 = Molecule.from_smiles(smi2)

        # Our ECFP4
        our_fp1 = ecfp4(mol1)
        our_fp2 = ecfp4(mol2)
        our_sim = tanimoto_similarity(our_fp1, our_fp2)

        # RDKit Morgan
        rdkit_mol1 = Chem.MolFromSmiles(smi1)
        rdkit_mol2 = Chem.MolFromSmiles(smi2)
        rdkit_fp1 = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol1, 2, nBits=2048)
        rdkit_fp2 = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol2, 2, nBits=2048)

        from rdkit import DataStructs
        rdkit_sim = DataStructs.TanimotoSimilarity(rdkit_fp1, rdkit_fp2)

        diff = abs(our_sim - rdkit_sim)
        status = "✓" if diff < 0.2 else "✗"  # Allow 20% difference due to different implementations

        print(f"{status} {pair_name}:")
        print(f"    Ours:   {our_sim:.3f}")
        print(f"    RDKit:  {rdkit_sim:.3f}")
        print(f"    Diff:   {diff:.3f}")


# ============================================================================
# MAIN
# ============================================================================

def run_all_validation():
    """Run all validation tests."""
    print("="*60)
    print("NQPU DRUG DESIGN - VALIDATION TEST SUITE")
    print("="*60)

    all_results = {}

    # Property validation
    all_results['mw'] = validate_molecular_weight()
    all_results['hbd_hba'] = validate_hbd_hba()
    all_results['rdkit_compare'] = validate_against_rdkit()

    # Fingerprint validation
    all_results['fp_properties'] = validate_fingerprint_properties()
    all_results['fp_discrimination'] = validate_fingerprint_discrimination()
    all_results['fp_rdkit'] = compare_fingerprints_with_rdkit()

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    mw_passed = sum(1 for r in all_results['mw'] if r.get('status') == '✓')
    mw_total = len(all_results['mw'])

    hbd_passed = sum(1 for r in all_results['hbd_hba'] if r.get('status') == '✓')
    hbd_total = len(all_results['hbd_hba'])

    fp_prop_passed, fp_prop_total = all_results['fp_properties']
    fp_disc_passed, fp_disc_total = all_results['fp_discrimination']

    print(f"  Molecular Weight:  {mw_passed}/{mw_total} passed")
    print(f"  HBD/HBA:           {hbd_passed}/{hbd_total} passed")
    print(f"  FP Properties:     {fp_prop_passed}/{fp_prop_total} passed")
    print(f"  FP Discrimination: {fp_disc_passed}/{fp_disc_total} passed")

    total_passed = mw_passed + hbd_passed + fp_prop_passed + fp_disc_passed
    total_tests = mw_total + hbd_total + fp_prop_total + fp_disc_total

    print(f"\n  TOTAL:             {total_passed}/{total_tests} tests passed ({100*total_passed/total_tests:.1f}%)")

    return all_results


if __name__ == "__main__":
    results = run_all_validation()
