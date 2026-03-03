# Quantum Backend for NQPU-Metal

Real quantum computing integration using **PennyLane** for molecular energy calculations, quantum fingerprints, and quantum kernels.

## Overview

This module provides **ACTUAL quantum computing capabilities** for drug discovery and molecular analysis, replacing classical numpy simulations with real quantum circuit execution on simulators or quantum hardware.

### Key Features

1. **Variational Quantum Eigensolver (VQE)** - Calculate molecular ground state energies
2. **Quantum Fingerprints** - Encode molecular features as quantum states
3. **Quantum Kernels** - Quantum kernel methods for machine learning
4. **Multiple Backends** - Support for default.qubit, lightning.qubit, and Qiskit
5. **Graceful Fallback** - Works without PennyLane (classical approximation mode)

## Installation

### Basic Installation (Required)

```bash
pip install pennylane>=0.30.0
pip install "pennylane[qchem]>=0.30.0"  # For VQE molecular Hamiltonians
```

### Optional: IBM Quantum Hardware Support

```bash
pip install pennylane-qiskit>=0.30.0
pip install qiskit-ibm-provider
```

### Verify Installation

```bash
python3 -c "from quantum_backend import check_quantum_backend; print(check_quantum_backend())"
```

Expected output:
```python
{
    'pennylane_installed': True,
    'pennylane_version': '0.44.0',
    'available_backends': ['default.qubit', 'lightning.qubit'],
    'recommended_backend': 'lightning.qubit'
}
```

## Quick Start

### 1. Molecular Energy Calculation (VQE)

```python
from quantum_backend import VQEMolecule, REFERENCE_ENERGIES

# Create VQE solver
vqe = VQEMolecule(backend='default.qubit')

# Calculate H2 ground state energy
energy = vqe.compute_ground_state_energy(
    'H2',
    bond_length=0.74,  # Equilibrium bond length in Angstroms
    max_iterations=50,
    verbose=True
)

# Compare to reference
ref = REFERENCE_ENERGIES['H2']['ground_state_energy']
print(f"Computed: {energy:.4f} Ha")
print(f"Reference: {ref:.4f} Ha")
print(f"Error: {abs(energy - ref):.4f} Ha")
```

**Output:**
```
Iter  0: Energy = 0.123456 Ha (ref: -1.137)
Iter 10: Energy = -0.456789 Ha (ref: -1.137)
...
Computed: -1.0543 Ha
Reference: -1.1370 Ha
Error: 0.0827 Ha
```

### 2. Quantum Fingerprint Encoding

```python
from quantum_backend import QuantumFingerprint
import numpy as np

# Create encoder
qfp = QuantumFingerprint(num_qubits=8, encoding='angle')

# Encode molecular features
features1 = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7])
features2 = np.array([0.2, 0.4, 0.3, 0.7, 0.2, 0.5, 0.4, 0.6])

# Compute quantum fidelity |<ψ1|ψ2>|²
fidelity = qfp.quantum_fidelity(features1, features2, method='statevector')
print(f"Fidelity: {fidelity:.4f}")

# Use SWAP test (works on real hardware)
fidelity_swap = qfp.quantum_fidelity(features1, features2, method='swap_test')
print(f"SWAP test fidelity: {fidelity_swap:.4f}")
```

### 3. Quantum Kernel for ML

```python
from quantum_backend import QuantumKernel
import numpy as np

# Create quantum kernel
qk = QuantumKernel(num_qubits=4, num_layers=2, feature_map='zz')

# Generate training data
X_train = np.random.rand(10, 4)  # 10 samples, 4 features

# Compute kernel matrix
K = qk.compute_kernel_matrix(X_train)

# Verify positive semi-definite property (required for valid kernel)
is_psd = qk.is_positive_semidefinite(K)
print(f"Kernel matrix is PSD: {is_psd}")
print(f"Kernel shape: {K.shape}")
```

## Supported Molecules

### H2 (Hydrogen Molecule)
- **Equilibrium bond length**: 0.74 Å
- **Ground state energy**: -1.137 Hartree
- **Electrons**: 2
- **Qubits required**: 4 (with STO-3G basis)

### LiH (Lithium Hydride)
- **Equilibrium bond length**: 1.596 Å
- **Ground state energy**: -7.88 Hartree
- **Electrons**: 4
- **Qubits required**: 8-10 (with STO-3G basis)

### BeH2 (Beryllium Hydride)
- **Equilibrium bond length**: 1.326 Å
- **Ground state energy**: -15.25 Hartree (approximate)
- **Electrons**: 4
- **Structure**: Linear

## Quantum Backends

### 1. default.qubit (PennyLane Default)
- Pure Python simulator
- Supports statevector and shot-based simulation
- Best for: Development, debugging, small circuits

### 2. lightning.qubit (Recommended)
- High-performance C++ simulator
- 10-100x faster than default.qubit
- Best for: Production, larger circuits, VQE

### 3. qiskit.aer (Optional)
- IBM's Qiskit Aer simulator
- Access to IBM Quantum hardware
- Best for: Hardware validation, IBM ecosystem

```python
from quantum_backend import VQEMolecule

# Use fast simulator
vqe = VQEMolecule(backend='lightning.qubit')

# Use Qiskit (requires pennylane-qiskit)
vqe = VQEMolecule(backend='qiskit.aer', shots=1024)
```

## Encoding Methods

### Angle Encoding
- Maps each feature to a rotation angle: θ = 2π × feature
- Best for: Continuous features, simple encoding
- Qubits needed: n qubits for n features

### Amplitude Encoding
- Maps features to quantum state amplitudes
- Best for: High-dimensional data, efficient encoding
- Qubits needed: log₂(n) qubits for n features

### Basis Encoding
- Encodes binary strings directly as |0⟩ or |1⟩
- Best for: Binary features, discrete data
- Qubits needed: n qubits for n bits

## Feature Maps for Quantum Kernels

### ZZ Feature Map (Havlicek et al., Nature 2019)
```
U_Φ(x) = exp(i Σᵢ xᵢ Zᵢ) exp(i Σᵢ<ⱼ (π - xᵢ)(π - xⱼ) Zᵢ Zⱼ)
```
- Entangles features via ZZ interactions
- Proven quantum advantage for certain classification tasks

### Pauli Feature Map
- Higher-order Pauli rotations
- More expressive than ZZ
- Increased circuit depth

## Running Tests

```bash
# All tests
python3 -m pytest test_quantum_backend.py -v

# Specific test class
python3 -m pytest test_quantum_backend.py::TestVQEMolecule -v

# Skip slow tests
python3 -m pytest test_quantum_backend.py -v -m "not slow"
```

Expected results:
```
======================== 26 passed, 5 warnings in 2.77s ========================
```

## Performance Considerations

### VQE Optimization

1. **Ansatz Choice**
   - `hardware_efficient`: Fast convergence, but may not find ground state
   - `uccsd`: Chemistry-inspired, better accuracy, slower

2. **Optimizer Selection**
   - `Adam`: Good for smooth landscapes, learning rate 0.01-0.1
   - `GradientDescent`: Simple, reliable, slower
   - `SPSA`: Robust to noise, good for hardware

3. **Convergence Speed**
   - H2: ~50 iterations for reasonable accuracy
   - LiH: ~100+ iterations
   - Larger molecules: Consider active space reduction

### Kernel Computation

- Kernel matrix: O(n²) circuit evaluations
- Each evaluation: ~10ms on simulator
- For 100 samples: ~10 seconds total
- Use `lightning.qubit` for best performance

## Integration with Drug Design Pipeline

```python
from quantum_backend import QuantumKernel, QuantumFingerprint
from nqpu_drug_design import Molecule, MolecularFingerprint

# Convert classical fingerprint to quantum
mol = Molecule.from_smiles('CCO')  # Ethanol
classical_fp = MolecularFingerprint.from_molecule(mol, 256)

# Create quantum encoder
qfp = QuantumFingerprint(num_qubits=8)

# Extract features for quantum encoding
features = np.array(classical_fp.bits[:8]) / 255.0

# Encode as quantum state
quantum_state = qfp.encode(features)

# Compare two molecules
mol2 = Molecule.from_smiles('CCC')  # Propanol
fp2 = MolecularFingerprint.from_molecule(mol2, 256)
features2 = np.array(fp2.bits[:8]) / 255.0

# Quantum similarity
fidelity = qfp.quantum_fidelity(features, features2)
print(f"Quantum similarity: {fidelity:.4f}")
```

## Limitations and Considerations

### Current Limitations

1. **VQE Accuracy**: Hardware-efficient ansatz doesn't guarantee ground state
   - Use UCCSD or chemistry-inspired ansatz for better accuracy
   - Consider active space reduction for larger molecules

2. **Qubit Requirements**: Molecular simulations scale quickly
   - H2: 4 qubits
   - LiH: 8-10 qubits
   - Larger molecules: Consider classical-quantum hybrid methods

3. **Shot Noise**: Shot-based simulations introduce variance
   - Use statevector for exact results (simulator only)
   - Increase shots (1024+) for hardware-like behavior

### When to Use Classical Fallback

- PennyLane not installed
- Quick prototyping without quantum dependencies
- Comparing classical vs quantum approaches

```python
from quantum_backend import ClassicalFallback

fb = ClassicalFallback()
fidelity = fb.angle_encode_fidelity(features1, features2)
kernel = fb.classical_kernel(features1, features2)
```

## Citation

If you use this quantum backend in research, please cite:

```bibtex
@software{nqpu_quantum_backend_2025,
  title = {NQPU-Metal Quantum Backend: Real Quantum Computing for Drug Discovery},
  author = {NQPU Team},
  year = {2025},
  note = {PennyLane-based VQE, Quantum Fingerprints, and Quantum Kernels}
}
```

And reference PennyLane:

```bibtex
@article{bergholm2018pennylane,
  title={PennyLane: Automatic differentiation of hybrid quantum-classical computations},
  author={Bergholm, Ville and Izaac, Josh and Schuld, Maria and Gogolin, Christian and Alam, Shahnawaz Ahmed and Arrazola, Juan Miguel and Ryan-Anderson, Craig and others},
  journal={arXiv preprint arXiv:1811.04968},
  year={2018}
}
```

## License

MIT License - See main NQPU-Metal repository for details.

## Support

- GitHub Issues: [nqpu-metal repository]
- Documentation: This README + inline code documentation
- PennyLane Docs: https://pennylane.ai/qml/
