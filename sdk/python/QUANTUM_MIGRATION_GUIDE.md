# Migrating from Classical to Real Quantum Computing

## What Changed

The `nqpu_drug_design.py` file contains a `QuantumKernel` class that claims to compute quantum similarity, but it's actually just using numpy operations - NOT real quantum computing.

### Before (Fake Quantum - Just NumPy)
```python
# From nqpu_drug_design.py
class QuantumKernel:
    def compute(self, fp1, fp2):
        # This is NOT quantum computing!
        angles1 = self._encode_fingerprint(fp1)
        angles2 = self._encode_fingerprint(fp2)
        
        fidelity = 1.0
        for a1, a2 in zip(angles1, angles2):
            overlap = (np.cos(a1/2) * np.cos(a2/2) + 
                      np.sin(a1/2) * np.sin(a2/2))
            fidelity *= overlap ** 2
        
        return fidelity ** (1.0 / self.num_qubits)
```

**This is classical math, not quantum circuits!**

### After (Real Quantum Computing)
```python
# From quantum_backend.py
class QuantumFingerprint:
    def quantum_fidelity(self, features1, features2):
        # Create ACTUAL quantum device
        dev = qml.device('default.qubit', wires=self.num_qubits)
        
        @qml.qnode(dev)
        def circuit():
            # Encode on REAL qubits
            for i in range(len(features1)):
                qml.RY(2 * np.pi * features1[i], wires=i)
            return qml.state()
        
        # Execute QUANTUM circuit
        state = circuit()
        # Compute actual quantum fidelity
        return np.abs(np.vdot(state1, state2)) ** 2
```

**This is REAL quantum circuit execution!**

## Migration Path

### Option 1: Drop-In Replacement (Easiest)

Replace `nqpu_drug_design.QuantumKernel` with `quantum_backend.QuantumFingerprint`:

```python
# OLD (fake quantum)
from nqpu_drug_design import QuantumKernel
qk = QuantumKernel(num_qubits=8)
similarity = qk.compute(fp1, fp2)

# NEW (real quantum)
from quantum_backend import QuantumFingerprint
qfp = QuantumFingerprint(num_qubits=8)
features1 = fingerprint_to_features(fp1)
features2 = fingerprint_to_features(fp2)
similarity = qfp.quantum_fidelity(features1, features2)
```

### Option 2: Use Hybrid Wrapper (Recommended)

Use `HybridQuantumSimilarity` from `INTEGRATION_GUIDE.md`:

```python
from integration_guide import HybridQuantumSimilarity

# Automatically uses quantum when available, classical when not
similarity = HybridQuantumSimilarity(num_qubits=8, prefer_quantum=True)
sim = similarity.compute_similarity(fp1, fp2)
```

### Option 3: Full Integration (Best)

Replace all similarity computations in drug design pipeline:

```python
from quantum_backend import QuantumKernel

# Replace entire similarity matrix computation
qk = QuantumKernel(num_qubits=8, num_layers=2, feature_map='zz')
K = qk.compute_kernel_matrix(features)

# Use for classification
predictions = classify_with_quantum_kernel(K, train_labels)
```

## Feature Comparison

| Feature | Old (Fake) | New (Real) |
|---------|-----------|------------|
| Quantum circuit execution | ❌ No | ✅ Yes |
| Hardware compatible | ❌ No | ✅ Yes |
| Multiple backends | ❌ No | ✅ Yes |
| VQE energy calculation | ❌ No | ✅ Yes |
| Quantum fidelity | ❌ Fake | ✅ Real |
| SWAP test | ❌ No | ✅ Yes |
| Quantum kernels | ❌ Fake | ✅ Real |
| Shot noise simulation | ❌ No | ✅ Yes |
| Hardware execution | ❌ No | ✅ Yes |

## Performance Impact

### Speed
- **Fake quantum**: O(n) numpy operations (~microseconds)
- **Real quantum (simulator)**: O(n) circuit execution (~milliseconds)
- **Real quantum (hardware)**: O(n) + queue time (~seconds to minutes)

### Accuracy
- **Fake quantum**: Classical approximation
- **Real quantum**: True quantum behavior (entanglement, superposition, interference)

### Scalability
- **Fake quantum**: Unlimited qubits
- **Real quantum (simulator)**: ~30 qubits (statevector memory)
- **Real quantum (hardware)**: 50-1000+ qubits (depending on provider)

## When to Use Each

### Use Fake Quantum (Old) When:
- ❌ Actually, there's no good reason to use fake quantum
- It misleads users into thinking they're doing quantum computing
- Replace with real quantum or honest classical methods

### Use Real Quantum (New) When:
- ✅ You want actual quantum advantage
- ✅ Research requires real quantum behavior
- ✅ Preparing for quantum hardware
- ✅ Need entanglement and interference effects

### Use Classical Methods When:
- ✅ Speed is critical
- ✅ Large-scale screening (millions of compounds)
- ✅ Quantum effects not needed
- ✅ PennyLane not available

## Code Changes Required

### 1. Update Imports
```python
# Remove
from nqpu_drug_design import QuantumKernel

# Add
from quantum_backend import QuantumFingerprint, QuantumKernel
```

### 2. Update Instantiation
```python
# Remove
qk = QuantumKernel(num_qubits=8)

# Add
qfp = QuantumFingerprint(num_qubits=8, encoding='angle', backend='default.qubit')
qk = QuantumKernel(num_qubits=8, num_layers=2, feature_map='zz')
```

### 3. Update Method Calls
```python
# Remove
similarity = qk.compute(fp1, fp2)

# Add (for fingerprints)
features1 = convert_fingerprint_to_features(fp1)
features2 = convert_fingerprint_to_features(fp2)
similarity = qfp.quantum_fidelity(features1, features2)

# OR (for kernel methods)
kernel_value = qk.kernel_circuit(features1, features2)
```

### 4. Add Fallback
```python
try:
    from quantum_backend import QuantumFingerprint
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("Warning: Using classical similarity (PennyLane not installed)")

def compute_similarity(fp1, fp2):
    if QUANTUM_AVAILABLE:
        qfp = QuantumFingerprint(num_qubits=8)
        features1 = convert_fp(fp1)
        features2 = convert_fp(fp2)
        return qfp.quantum_fidelity(features1, features2)
    else:
        return fp1.tanimoto(fp2)  # Classical fallback
```

## Testing Migration

Run the test suite to verify quantum backend:

```bash
# All quantum tests
python3 -m pytest test_quantum_backend.py -v

# Specific functionality
python3 -m pytest test_quantum_backend.py::TestQuantumFingerprint -v
python3 -m pytest test_quantum_backend.py::TestQuantumKernel -v

# Integration test
python3 -c "from integration_guide import example_quantum_drug_screening; example_quantum_drug_screening()"
```

Expected: All 26 tests pass

## Troubleshooting

### Issue: "PennyLane not installed"
**Solution**: `pip install pennylane pennylane[qchem]`

### Issue: "Hamiltonian construction failed"
**Solution**: Install quantum chemistry: `pip install pennylane[qchem]`

### Issue: "lightning.qubit not available"
**Solution**: Use `default.qubit` or install lightning: `pip install pennylane-lightning`

### Issue: "Out of memory for statevector"
**Solution**: Reduce qubits or use shot-based simulation:
```python
qfp = QuantumFingerprint(num_qubits=4)  # Fewer qubits
dev = qml.device('default.qubit', wires=8, shots=1024)  # Shot-based
```

### Issue: "Different results than expected"
**Note**: Real quantum computing has inherent randomness:
- Use `method='statevector'` for exact results (simulator only)
- Use `method='swap_test'` with more shots for hardware-like behavior
- Increase `shots=1024+` to reduce variance

## Summary

✅ **Do**: Replace fake quantum with real quantum
✅ **Do**: Use `quantum_backend.py` for actual quantum computing
✅ **Do**: Add fallbacks for environments without PennyLane
✅ **Do**: Test with real quantum circuits

❌ **Don't**: Keep using fake quantum methods
❌ **Don't**: Claim quantum computing without actual circuits
❌ **Don't**: Ignore the difference between simulation and reality

## Next Steps

1. ✅ Replace all `QuantumKernel` usage in `nqpu_drug_design.py`
2. ✅ Update documentation to clarify real vs. fake quantum
3. ✅ Add quantum backend checks to CI/CD pipeline
4. ✅ Benchmark real quantum vs. classical performance
5. ✅ Test on IBM Quantum hardware (optional)

## Questions?

- See `README_QUANTUM_BACKEND.md` for full documentation
- See `QUANTUM_BACKEND_SUMMARY.md` for implementation details
- See `INTEGRATION_GUIDE.md` for code examples
- Run `quantum_backend_example.py` for live demo
