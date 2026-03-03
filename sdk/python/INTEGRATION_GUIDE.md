"""
Integration Example: Quantum Backend with NQPU Drug Design

This example shows how to replace the classical "quantum" methods in
nqpu_drug_design.py with REAL quantum computing using the quantum_backend module.

Original classical approach:
    from nqpu_drug_design import QuantumKernel
    kernel = QuantumKernel(num_qubits=8)
    similarity = kernel.compute(fp1, fp2)  # Just numpy operations

New real quantum approach:
    from quantum_backend import QuantumFingerprint
    qfp = QuantumFingerprint(num_qubits=8)
    fidelity = qfp.quantum_fidelity(features1, features2)  # Real quantum circuit!
"""

import numpy as np
from typing import List, Dict, Union

# Import both classical and quantum backends
try:
    from quantum_backend import (
        HAS_PENNYLANE,
        QuantumFingerprint as RealQuantumFingerprint,
        QuantumKernel as RealQuantumKernel,
        check_quantum_backend,
    )
    QUANTUM_AVAILABLE = HAS_PENNYLANE
except ImportError:
    QUANTUM_AVAILABLE = False
    print("Warning: quantum_backend not available, using classical only")

from nqpu_drug_design import Molecule, MolecularFingerprint


class HybridQuantumSimilarity:
    """
    Hybrid quantum-classical molecular similarity.

    Uses real quantum computing when available, falls back to classical
    methods when PennyLane is not installed.

    This replaces the "fake quantum" QuantumKernel in nqpu_drug_design.py
    with ACTUAL quantum circuit execution.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        encoding: str = 'angle',
        backend: str = 'default.qubit',
        prefer_quantum: bool = True
    ):
        """
        Initialize hybrid similarity calculator.

        Args:
            num_qubits: Number of qubits for quantum encoding
            encoding: Encoding method ('angle', 'amplitude', 'basis')
            backend: Quantum backend ('default.qubit', 'lightning.qubit', 'qiskit.aer')
            prefer_quantum: Use quantum when available (default: True)
        """
        self.num_qubits = num_qubits
        self.encoding = encoding
        self.backend = backend
        self.prefer_quantum = prefer_quantum and QUANTUM_AVAILABLE

        if self.prefer_quantum:
            self._quantum_fp = RealQuantumFingerprint(
                num_qubits=num_qubits,
                encoding=encoding,
                backend=backend
            )
            print(f"✅ Using REAL quantum computing ({backend})")
        else:
            self._quantum_fp = None
            print("⚠️  Using classical similarity (quantum not available)")

    def fingerprint_to_features(
        self,
        fp: MolecularFingerprint
    ) -> np.ndarray:
        """
        Convert classical fingerprint to quantum-encodable features.

        Args:
            fp: Classical molecular fingerprint

        Returns:
            Feature array suitable for quantum encoding
        """
        # Convert fingerprint bits to normalized features
        bitstring = fp.to_bitstring()

        # Take first num_qubits bits and normalize to [0, 1]
        features = np.array([
            float(bitstring[i % len(bitstring)])
            for i in range(self.num_qubits)
        ])

        return features

    def compute_similarity(
        self,
        fp1: MolecularFingerprint,
        fp2: MolecularFingerprint
    ) -> float:
        """
        Compute molecular similarity using quantum or classical methods.

        Args:
            fp1: First molecule fingerprint
            fp2: Second molecule fingerprint

        Returns:
            Similarity score in [0, 1]
        """
        if self.prefer_quantum and self._quantum_fp:
            # Real quantum computation
            features1 = self.fingerprint_to_features(fp1)
            features2 = self.fingerprint_to_features(fp2)

            # Use statevector method for exact fidelity
            fidelity = self._quantum_fp.quantum_fidelity(
                features1, features2,
                method='statevector'
            )

            return fidelity
        else:
            # Classical fallback
            return fp1.tanimoto(fp2)

    def compute_similarity_matrix(
        self,
        fingerprints: List[MolecularFingerprint]
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for multiple molecules.

        Args:
            fingerprints: List of molecular fingerprints

        Returns:
            Similarity matrix S where S[i,j] = similarity(fp[i], fp[j])
        """
        n = len(fingerprints)
        S = np.eye(n)  # Diagonal is 1 (self-similarity)

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_similarity(fingerprints[i], fingerprints[j])
                S[i, j] = sim
                S[j, i] = sim

        return S


class QuantumEnhancedDrugScreening:
    """
    Quantum-enhanced virtual screening for drug discovery.

    Uses real quantum computing for molecular similarity and
    classification in the drug screening pipeline.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        backend: str = 'default.qubit'
    ):
        """
        Initialize quantum-enhanced screening.

        Args:
            num_qubits: Number of qubits for quantum encoding
            backend: Quantum backend to use
        """
        self.num_qubits = num_qubits
        self.backend = backend

        if QUANTUM_AVAILABLE:
            self._similarity = HybridQuantumSimilarity(
                num_qubits=num_qubits,
                backend=backend,
                prefer_quantum=True
            )
            self._quantum_kernel = RealQuantumKernel(
                num_qubits=num_qubits,
                num_layers=2
            )
            self.quantum_enabled = True
        else:
            self._similarity = HybridQuantumSimilarity(
                num_qubits=num_qubits,
                prefer_quantum=False
            )
            self._quantum_kernel = None
            self.quantum_enabled = False

    def screen_molecules(
        self,
        target_fingerprint: MolecularFingerprint,
        candidate_fingerprints: List[MolecularFingerprint],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Screen candidate molecules against target using quantum similarity.

        Args:
            target_fingerprint: Target molecule fingerprint
            candidate_fingerprints: List of candidate fingerprints
            top_k: Number of top candidates to return

        Returns:
            List of top candidates with similarity scores
        """
        similarities = []

        for i, candidate_fp in enumerate(candidate_fingerprints):
            # Use quantum or classical similarity
            sim = self._similarity.compute_similarity(
                target_fingerprint,
                candidate_fp
            )
            similarities.append({
                'index': i,
                'similarity': sim,
                'method': 'quantum' if self.quantum_enabled else 'classical'
            })

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        return similarities[:top_k]

    def classify_molecules(
        self,
        train_fingerprints: List[MolecularFingerprint],
        train_labels: List[int],
        test_fingerprints: List[MolecularFingerprint]
    ) -> List[int]:
        """
        Classify molecules using quantum kernel method.

        Args:
            train_fingerprints: Training molecule fingerprints
            train_labels: Training labels (0 or 1)
            test_fingerprints: Test molecule fingerprints

        Returns:
            Predicted labels for test molecules
        """
        if not self.quantum_enabled:
            # Classical fallback: use Tanimoto similarity
            predictions = []
            for test_fp in test_fingerprints:
                # Nearest neighbor classifier
                best_label = 0
                best_sim = -1
                for train_fp, label in zip(train_fingerprints, train_labels):
                    sim = test_fp.tanimoto(train_fp)
                    if sim > best_sim:
                        best_sim = sim
                        best_label = label
                predictions.append(best_label)
            return predictions

        # Quantum kernel classification
        # Convert fingerprints to features
        X_train = np.array([
            self._similarity.fingerprint_to_features(fp)
            for fp in train_fingerprints
        ])
        X_test = np.array([
            self._similarity.fingerprint_to_features(fp)
            for fp in test_fingerprints
        ])
        y_train = np.array(train_labels)

        predictions = []
        for x_test in X_test:
            # Compute kernel values to all training points
            kernel_values = np.array([
                self._quantum_kernel.kernel_circuit(x_test, x_train)
                for x_train in X_train
            ])

            # Weighted vote
            weights = kernel_values / kernel_values.sum()
            prediction = np.dot(weights, y_train)

            predictions.append(1 if prediction > 0.5 else 0)

        return predictions


# Example usage
def example_quantum_drug_screening():
    """Demonstrate quantum-enhanced drug screening."""
    print("=" * 70)
    print("Quantum-Enhanced Drug Screening Example")
    print("=" * 70)

    # Check quantum backend status
    status = check_quantum_backend()
    print(f"\nQuantum backend status:")
    print(f"  Available: {status['pennylane_installed']}")
    print(f"  Backends: {status['available_backends']}")

    # Create sample molecules
    print("\nCreating sample molecules...")
    target_mol = Molecule.from_smiles('CCO')  # Ethanol (target)
    candidate_mols = [
        Molecule.from_smiles('CCCO'),   # Propanol (similar)
        Molecule.from_smiles('CCCCO'),  # Butanol (similar)
        Molecule.from_smiles('CC(C)O'), # Isopropanol (similar)
        Molecule.from_smiles('c1ccccc1'),  # Benzene (different)
        Molecule.from_smiles('CC(=O)O'),   # Acetic acid (different)
    ]

    # Generate fingerprints
    target_fp = MolecularFingerprint.from_molecule(target_mol, 256)
    candidate_fps = [
        MolecularFingerprint.from_molecule(mol, 256)
        for mol in candidate_mols
    ]

    # Create quantum-enhanced screener
    print("\nInitializing quantum-enhanced screening...")
    screener = QuantumEnhancedDrugScreening(num_qubits=8)

    # Screen molecules
    print("\nScreening candidates against target...")
    results = screener.screen_molecules(
        target_fp,
        candidate_fps,
        top_k=5
    )

    # Display results
    print("\nTop candidates:")
    for rank, result in enumerate(results, 1):
        method = result['method']
        sim = result['similarity']
        idx = result['index']
        smiles = candidate_mols[idx].to_smiles() if hasattr(candidate_mols[idx], 'to_smiles') else f"mol_{idx}"

        print(f"  {rank}. {smiles}")
        print(f"     Similarity: {sim:.4f} ({method})")

    # Classification example
    print("\n" + "-" * 70)
    print("Quantum Kernel Classification Example")
    print("-" * 70)

    # Create training set (class 0: alcohols, class 1: aromatic)
    train_mols = [
        Molecule.from_smiles('CCO'),    # Alcohol (0)
        Molecule.from_smiles('CCCO'),   # Alcohol (0)
        Molecule.from_smiles('c1ccccc1'),  # Aromatic (1)
        Molecule.from_smiles('c1ccc(cc1)C'),  # Aromatic (1)
    ]
    train_fps = [MolecularFingerprint.from_molecule(mol, 256) for mol in train_mols]
    train_labels = [0, 0, 1, 1]

    # Test molecules
    test_mols = [
        Molecule.from_smiles('CCCCO'),  # Should be class 0 (alcohol)
        Molecule.from_smiles('c1ccc(cc1)O'),  # Should be class 1 (phenol/aromatic)
    ]
    test_fps = [MolecularFingerprint.from_molecule(mol, 256) for mol in test_mols]

    # Classify
    print("\nClassifying test molecules...")
    predictions = screener.classify_molecules(train_fps, train_labels, test_fps)

    class_names = {0: "Alcohol", 1: "Aromatic"}
    for mol, pred in zip(test_mols, predictions):
        smiles = mol.to_smiles() if hasattr(mol, 'to_smiles') else "unknown"
        print(f"  {smiles}: {class_names[pred]} (class {pred})")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == '__main__':
    example_quantum_drug_screening()
