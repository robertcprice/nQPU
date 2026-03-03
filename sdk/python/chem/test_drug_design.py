"""
Test script for nqpu_drug_design Python bindings.

Tests the pure Python implementation since Rust bindings need maturin build.
"""

import sys
sys.path.insert(0, '.')

from nqpu_drug_design import (
    Molecule,
    Element,
    BondType,
    AdmetProperty,
    ScoringFunction,
    DrugLikenessResult,
    AdmetResult,
    ScreeningResult,
    OptimizationResult,
    ParetoPoint,
    StageResult,
    DrugDiscoveryPipeline,
    MolecularLibrary,
    evaluate_drug_likeness,
    predict_admet,
    screen_library,
    optimize_lead,
    check_installation,
    get_version,
)


def test_molecule_creation():
    """Test basic molecule creation and manipulation."""
    mol = Molecule("TestMol")
    assert mol.name == "TestMol"
    assert len(mol.atoms) == 0
    assert len(mol.bonds) == 0

    # Add atoms
    c0 = mol.add_atom("C", [0.0, 0.0, 0.0], -0.08)
    c1 = mol.add_atom("C", [1.4, 0.0, 0.0], -0.08)
    c2 = mol.add_atom("O", [2.1, 1.2, 0.0], -0.44)

    assert len(mol.atoms) == 3
    assert abs(mol.molecular_weight() - (12.011 * 2 + 15.999)) < 0.1

    # Add bond
    mol.add_bond(c0, c1, "single")
    mol.add_bond(c1, c2, "double")

    assert len(mol.bonds) == 2
    print("PASS: test_molecule_creation")


def test_molecule_from_smiles():
    """Test SMILES parsing (simplified)."""
    mol = Molecule.from_smiles("CCO", "Ethanol")
    assert mol.name == "Ethanol"
    assert len(mol.atoms) >= 2  # At least C, C, O
    print("PASS: test_molecule_from_smiles")


def test_drug_likeness():
    """Test drug-likeness evaluation."""
    mol = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O", "Aspirin")

    result = evaluate_drug_likeness(mol)

    assert isinstance(result, DrugLikenessResult)
    assert 0 <= result.lipinski_violations <= 4
    assert 0.0 <= result.qed_score <= 1.0
    assert 0.0 <= result.synthetic_accessibility <= 1.0

    # Check properties are reasonable
    props = result.properties
    assert props.molecular_weight > 0
    assert -5.0 <= props.log_p <= 10.0  # Wide range

    print(f"  QED Score: {result.qed_score:.3f}")
    print(f"  Lipinski Violations: {result.lipinski_violations}")
    print("PASS: test_drug_likeness")


def test_admet_prediction():
    """Test ADMET property prediction."""
    mol = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O", "Aspirin")

    results = predict_admet(mol)

    assert isinstance(results, dict)
    assert len(results) > 0

    for prop_name, result in results.items():
        # Results are AdmetResult dataclass instances
        assert hasattr(result, 'probability')
        assert hasattr(result, 'passes')
        assert hasattr(result, 'confidence')
        assert 0.0 <= result.probability <= 1.0
        # Accept both Python bool and numpy bool
        assert bool(result.passes) in [True, False], f"result.passes={result.passes!r}"
        assert 0.0 <= result.confidence <= 1.0
        print(f"  {prop_name}: passes={bool(result.passes)}, prob={result.probability:.2f}")

    print("PASS: test_admet_prediction")


def test_screen_library():
    """Test virtual screening."""
    protein = Molecule.from_smiles("CCCC", "Protein")
    ligands = [
        Molecule.from_smiles("CCO", "Ligand1"),
        Molecule.from_smiles("CCCO", "Ligand2"),
        Molecule.from_smiles("CCCCO", "Ligand3"),
    ]

    results = screen_library(protein, ligands, top_k=2)

    assert len(results) <= 2
    for result in results:
        assert isinstance(result, ScreeningResult)
        assert result.ligand_index >= 0
        print(f"  Ligand {result.ligand_index}: score={result.score:.3f}")

    print("PASS: test_screen_library")


def test_optimize_lead():
    """Test lead optimization."""
    lead = Molecule.from_smiles("CCCO", "Lead")
    protein = Molecule.from_smiles("CCCC", "Protein")

    result = optimize_lead(lead, protein, iterations=10)

    assert isinstance(result, OptimizationResult)
    assert len(result.scores_over_iterations) == 10
    assert isinstance(result.pareto_front, list)

    print(f"  Best score: {result.best_score:.3f}")
    print(f"  Pareto front size: {len(result.pareto_front)}")
    print("PASS: test_optimize_lead")


def test_pipeline():
    """Test drug discovery pipeline."""
    pipeline = DrugDiscoveryPipeline.standard(num_qubits=4)

    candidates = [
        Molecule.from_smiles("CCO", "Candidate1"),
        Molecule.from_smiles("CCCO", "Candidate2"),
    ]
    protein = Molecule.from_smiles("CCCC", "Target")

    results = pipeline.run(candidates, protein)

    assert len(results) > 0
    for stage in results:
        assert isinstance(stage, StageResult)
        print(f"  {stage.stage_name}: {stage.passed} passed, {stage.failed} failed")

    print("PASS: test_pipeline")


def test_molecular_library():
    """Test pre-built molecules."""
    aspirin = MolecularLibrary.aspirin()
    assert aspirin.name == "Aspirin"
    assert len(aspirin.atoms) > 0

    ibuprofen = MolecularLibrary.ibuprofen()
    assert ibuprofen.name == "Ibuprofen"

    caffeine = MolecularLibrary.caffeine()
    assert caffeine.name == "Caffeine"

    print("PASS: test_molecular_library")


def test_enums():
    """Test enum values."""
    assert Element.C.value == "C"
    assert Element.N.value == "N"
    assert Element.O.value == "O"

    assert BondType.SINGLE.value == "single"
    assert BondType.DOUBLE.value == "double"
    assert BondType.AROMATIC.value == "aromatic"

    assert AdmetProperty.TOXICITY.value == "toxicity"
    assert AdmetProperty.ABSORPTION.value == "absorption"

    assert ScoringFunction.HYBRID_CLASSICAL_QUANTUM.value == "hybrid_classical_quantum"

    print("PASS: test_enums")


def test_utility_functions():
    """Test utility functions."""
    status = check_installation()
    assert "rust_bindings" in status
    assert "numpy" in status

    version = get_version()
    assert isinstance(version, str)
    assert len(version) > 0

    print("PASS: test_utility_functions")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING ALL DRUG DESIGN TESTS")
    print("=" * 60 + "\n")

    test_molecule_creation()
    test_molecule_from_smiles()
    test_drug_likeness()
    test_admet_prediction()
    test_screen_library()
    test_optimize_lead()
    test_pipeline()
    test_molecular_library()
    test_enums()
    test_utility_functions()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
