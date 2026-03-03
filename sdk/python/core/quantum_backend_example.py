#!/usr/bin/env python3
"""
Example usage of the NQPU Quantum Backend

This script demonstrates:
1. H2 molecule ground state energy calculation using VQE
2. Quantum fingerprint encoding and fidelity
3. Quantum kernel computation for molecule classification
4. Backend status checking
5. Classical fallback when PennyLane unavailable

Run with: python3 quantum_backend_example.py
"""

import numpy as np

# Import quantum backend components
from quantum_backend import (
    check_quantum_backend,
    VQEMolecule,
    QuantumFingerprint,
    QuantumKernel,
    MolecularGeometry,
    ClassicalFallback,
    REFERENCE_ENERGIES,
    HAS_PENNYLANE,
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def example_backend_status():
    """Check and display quantum backend status."""
    print_section("1. Quantum Backend Status")

    status = check_quantum_backend()

    print(f"PennyLane installed: {status['pennylane_installed']}")
    print(f"Version: {status['pennylane_version']}")
    print(f"Available backends: {status['available_backends']}")
    print(f"Recommended: {status['recommended_backend']}")

    if not HAS_PENNYLANE:
        print("\n⚠️  WARNING: PennyLane not installed!")
        print("   Quantum features will use classical fallback.")
        print("   Install with: pip install pennylane pennylane[qchem]")

    return status


def example_h2_vqe():
    """Calculate H2 ground state energy using VQE."""
    print_section("2. H2 Molecule VQE Energy Calculation")

    if not HAS_PENNYLANE:
        print("⚠️  Skipping: PennyLane not installed")
        return

    # Create VQE solver
    print("Creating VQE solver with default.qubit backend...")
    vqe = VQEMolecule(backend='default.qubit')

    # H2 at equilibrium geometry
    bond_length = 0.74  # Angstroms
    print(f"\nComputing H2 ground state at bond length {bond_length} Å")
    print("Reference value: -1.137 Hartree")
    print("\nRunning VQE optimization (50 iterations)...")

    # Run VQE
    energy = vqe.compute_ground_state_energy(
        'H2',
        bond_length=bond_length,
        ansatz='hardware_efficient',
        num_layers=2,
        max_iterations=50,
        verbose=True
    )

    # Compare to reference
    ref_energy = REFERENCE_ENERGIES['H2']['ground_state_energy']
    error = abs(energy - ref_energy)

    print(f"\n📊 Results:")
    print(f"   Computed energy: {energy:.4f} Hartree")
    print(f"   Reference energy: {ref_energy:.4f} Hartree")
    print(f"   Error: {error:.4f} Hartree")

    if error < 0.1:
        print("   ✅ Excellent agreement with reference!")
    elif error < 0.5:
        print("   ✅ Good agreement (hardware-efficient ansatz)")
    else:
        print("   ⚠️  Note: Hardware-efficient ansatz may not reach ground state")
        print("      For better accuracy, use UCCSD ansatz or more iterations")


def example_quantum_fingerprint():
    """Demonstrate quantum fingerprint encoding and fidelity."""
    print_section("3. Quantum Fingerprint Encoding")

    if not HAS_PENNYLANE:
        print("⚠️  Skipping: PennyLane not installed")
        print("\nUsing classical fallback instead:")
        fb = ClassicalFallback()

        # Example molecular features
        features1 = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7])
        features2 = np.array([0.2, 0.4, 0.3, 0.7, 0.2, 0.5, 0.4, 0.6])

        fidelity = fb.angle_encode_fidelity(features1, features2)
        print(f"Classical fidelity approximation: {fidelity:.4f}")
        return

    # Create quantum fingerprint encoder
    num_qubits = 8
    print(f"Creating quantum fingerprint encoder ({num_qubits} qubits, angle encoding)...")
    qfp = QuantumFingerprint(num_qubits=num_qubits, encoding='angle')

    # Example molecular features (normalized to [0, 1])
    features1 = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7])
    features2_similar = np.array([0.11, 0.49, 0.31, 0.79, 0.21, 0.59, 0.41, 0.69])
    features3_different = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

    print("\nEncoding molecular features as quantum states...")

    # Encode features
    state1 = qfp.encode(features1)
    print(f"State 1 shape: {state1.shape}, norm: {np.linalg.norm(state1):.4f}")

    # Compute fidelities
    print("\nComputing quantum fidelities...")

    fidelity_identical = qfp.quantum_fidelity(features1, features1, method='statevector')
    print(f"\n📊 Identical molecules: {fidelity_identical:.4f}")
    print(f"   Expected: 1.0 (same state)")

    fidelity_similar = qfp.quantum_fidelity(features1, features2_similar, method='statevector')
    print(f"\n📊 Similar molecules: {fidelity_similar:.4f}")
    print(f"   Expected: > 0.9 (high similarity)")

    fidelity_different = qfp.quantum_fidelity(features1, features3_different, method='statevector')
    print(f"\n📊 Different molecules: {fidelity_different:.4f}")
    print(f"   Expected: < 0.5 (low similarity)")

    # SWAP test (works on quantum hardware)
    print("\nTesting SWAP test (hardware-compatible method)...")
    fidelity_swap = qfp.quantum_fidelity(features1, features2_similar, method='swap_test')
    print(f"📊 SWAP test fidelity: {fidelity_swap:.4f}")
    print(f"   (Has shot noise variance)")

    # Quantum distance
    distance = qfp.quantum_distance(features1, features3_different)
    print(f"\n📊 Quantum distance (different): {distance:.4f}")


def example_quantum_kernel():
    """Demonstrate quantum kernel computation."""
    print_section("4. Quantum Kernel for Machine Learning")

    if not HAS_PENNYLANE:
        print("⚠️  Skipping: PennyLane not installed")
        print("\nUsing classical RBF kernel fallback:")
        fb = ClassicalFallback()

        # Small dataset
        X = np.random.rand(4, 4)
        K = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                K[i, j] = fb.classical_kernel(X[i], X[j])

        print(f"Classical RBF kernel matrix:\n{K}")
        return

    # Create quantum kernel
    num_qubits = 4
    num_layers = 2
    print(f"Creating quantum kernel ({num_qubits} qubits, {num_layers} layers, ZZ feature map)...")
    qk = QuantumKernel(num_qubits=num_qubits, num_layers=num_layers, feature_map='zz')

    # Generate synthetic molecular features
    # Class 0: "small" molecules (low feature values)
    # Class 1: "large" molecules (high feature values)
    print("\nGenerating synthetic molecular dataset...")
    X_small = np.random.uniform(0.0, 0.3, size=(3, num_qubits))
    X_large = np.random.uniform(0.7, 1.0, size=(3, num_qubits))
    X_train = np.vstack([X_small, X_large])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    print(f"Dataset: 6 molecules (3 small, 3 large)")
    print(f"Feature shape: {X_train.shape}")

    # Compute kernel matrix
    print("\nComputing quantum kernel matrix...")
    K = qk.compute_kernel_matrix(X_train)

    print(f"\n📊 Kernel matrix shape: {K.shape}")
    print(f"   Diagonal (self-similarity): {np.diag(K)}")
    print(f"   Expected: ≈1.0")

    # Check positive semi-definite property
    is_psd = qk.is_positive_semidefinite(K)
    eigenvalues = np.linalg.eigvalsh(K)

    print(f"\n📊 Is positive semi-definite: {is_psd}")
    print(f"   Min eigenvalue: {eigenvalues.min():.6f}")
    print(f"   Expected: ≥0 (for valid kernel)")

    # Visualize kernel structure
    print("\n📊 Kernel matrix (rounded):")
    print(np.round(K, 2))

    # Test molecule classification
    print("\n" + "-" * 70)
    print("Molecule Classification Example")
    print("-" * 70)

    # New "small" molecule
    x_test_small = np.random.uniform(0.0, 0.3, size=num_qubits)

    # New "large" molecule
    x_test_large = np.random.uniform(0.7, 1.0, size=num_qubits)

    print("\nClassifying test molecules using kernel similarity...")

    # Simple kernel-based classification
    # Assign class based on weighted average of training labels
    for name, x_test in [("Small test", x_test_small), ("Large test", x_test_large)]:
        kernel_values = [qk.kernel_circuit(x_test, X_train[i]) for i in range(len(X_train))]
        kernel_values = np.array(kernel_values)

        # Weighted average of labels
        weights = kernel_values / kernel_values.sum()
        prediction = np.dot(weights, y_train)

        predicted_class = 1 if prediction > 0.5 else 0
        confidence = max(prediction, 1 - prediction)

        print(f"\n{name}:")
        print(f"   Kernel values: {np.round(kernel_values, 3)}")
        print(f"   Predicted class: {predicted_class} (confidence: {confidence:.2%})")


def example_molecular_geometry():
    """Demonstrate molecular geometry construction."""
    print_section("5. Molecular Geometry Construction")

    # H2 molecule
    print("\nH2 Molecule:")
    h2 = MolecularGeometry.h2(bond_length=0.74)
    print(f"   Symbols: {h2.symbols}")
    print(f"   Coordinates (Å):\n{h2.coordinates}")
    bond = np.linalg.norm(h2.coordinates[0] - h2.coordinates[1])
    print(f"   Bond length: {bond:.3f} Å")

    # LiH molecule
    print("\nLiH Molecule:")
    lih = MolecularGeometry.lih(bond_length=1.596)
    print(f"   Symbols: {lih.symbols}")
    print(f"   Coordinates (Å):\n{lih.coordinates}")
    bond = np.linalg.norm(lih.coordinates[0] - lih.coordinates[1])
    print(f"   Bond length: {bond:.3f} Å")

    # BeH2 molecule
    print("\nBeH2 Molecule:")
    beh2 = MolecularGeometry.beh2(bond_length=1.326)
    print(f"   Symbols: {beh2.symbols}")
    print(f"   Coordinates (Å):\n{beh2.coordinates}")
    print(f"   Structure: Linear")


def main():
    """Run all examples."""
    print("=" * 70)
    print("NQPU Quantum Backend - Example Usage")
    print("=" * 70)

    # 1. Check backend status
    status = example_backend_status()

    # 2. VQE for H2
    example_h2_vqe()

    # 3. Quantum fingerprints
    example_quantum_fingerprint()

    # 4. Quantum kernels
    example_quantum_kernel()

    # 5. Molecular geometries
    example_molecular_geometry()

    # Summary
    print_section("Summary")

    if HAS_PENNYLANE:
        print("✅ All quantum features available and demonstrated!")
        print("\nKey capabilities:")
        print("   • VQE molecular energy calculation")
        print("   • Quantum state encoding (angle, amplitude, basis)")
        print("   • Quantum fidelity measurement")
        print("   • Quantum kernel for ML")
        print("   • Multiple backend support")

        print("\nNext steps:")
        print("   • Try larger molecules (LiH, BeH2)")
        print("   • Experiment with different ansatz circuits")
        print("   • Use lightning.qubit for faster simulation")
        print("   • Integrate with your drug design pipeline")
    else:
        print("⚠️  PennyLane not installed - classical fallback used")
        print("\nTo enable real quantum computing:")
        print("   pip install pennylane pennylane[qchem]")

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
