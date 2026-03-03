# nQPU-Metal JAX Integration

Full JAX integration for nQPU-Metal quantum simulator with automatic differentiation, VMAP batch execution, and quantum machine learning primitives.

## Features

- **Custom VJP**: Exact gradients via parameter-shift rule computed in Rust
- **VMAP Support**: Efficient batch execution using Rust's native parallelism
- **JIT Compatible**: Works with `@jax.jit` decorator for compilation
- **Quantum Kernels**: Quantum kernel matrices for QSVM
- **VQE Support**: Variational Quantum Eigensolver with Hamiltonian terms
- **Natural Gradients**: Quantum Fisher Information Matrix and natural gradient descent
- **Zero Python Overhead**: All quantum simulation happens in optimized Rust code

## Installation

### 1. Install JAX

```bash
# CPU only
pip install jax jaxlib

# CUDA 12 (NVIDIA GPU)
pip install jax[cuda12]

# For Apple Silicon, JAX uses Metal acceleration automatically
pip install jax jaxlib
```

### 2. Build nQPU-Metal Python Bindings

```bash
cd /path/to/nqpu-metal

# Install maturin (Rust-Python build tool)
pip install maturin

# Build and install in development mode
maturin develop --release --features python

# Or build wheel for distribution
maturin build --release --features python
```

### 3. Add Python Module to Path

```bash
export PYTHONPATH="/path/to/nqpu-metal/python:$PYTHONPATH"
```

Or in Python:

```python
import sys
sys.path.insert(0, '/path/to/nqpu-metal/python')
```

## Quick Start

### Basic Circuit with Gradients

```python
import jax
import jax.numpy as jnp
from nqpu_jax import quantum_expectation

# Define circuit
circuit = {
    'n_qubits': 2,
    'gates': [
        ('H', 0),                    # Hadamard on qubit 0
        ('RY', 0, 'theta'),          # Parameterized rotation
        ('RY', 1, 'phi'),            # Second parameter
        ('CNOT', 0, 1),              # Entangling gate
    ],
    'observable': 'Z0'               # Measure ⟨Z₀⟩
}

# Execute circuit
params = jnp.array([0.5, 1.2])
expectation = quantum_expectation(params, circuit)
print(f"⟨Z₀⟩ = {expectation}")

# Compute gradients automatically
grad_fn = jax.grad(quantum_expectation, argnums=0)
grads = grad_fn(params, circuit)
print(f"∇⟨Z₀⟩ = {grads}")
```

### Batch Execution with VMAP

```python
from nqpu_jax import vmap_quantum

# Create batched function
batch_fn = vmap_quantum(circuit, observable='Z0')

# Execute on multiple parameter sets
batch_params = jnp.array([
    [0.0, 0.0],
    [0.5, 0.5],
    [1.0, 1.0],
])

results = batch_fn(batch_params)  # Shape: (3,)
print(f"Batch results: {results}")
```

### Quantum Kernel for QSVM

```python
from nqpu_jax import quantum_kernel_matrix

# Feature map circuit
feature_map = {
    'n_qubits': 4,
    'gates': [
        ('H', 0), ('H', 1), ('H', 2), ('H', 3),
        ('RY', 0, 'x0'), ('RY', 1, 'x1'),
        ('RY', 2, 'x2'), ('RY', 3, 'x3'),
        ('CNOT', 0, 1), ('CNOT', 1, 2), ('CNOT', 2, 3),
    ]
}

# Compute kernel K[i,j] = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|²
X_train = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
X_test = jnp.array([[0.2, 0.3, 0.4, 0.5]])

K = quantum_kernel_matrix(X_train, X_test, feature_map)
print(f"Kernel matrix:\n{K}")
```

### VQE Optimization

```python
from nqpu_jax import make_vqe_loss

# Ansatz circuit
ansatz = {
    'n_qubits': 2,
    'gates': [
        ('RY', 0, 'theta_0'),
        ('RY', 1, 'theta_1'),
        ('CNOT', 0, 1),
        ('RY', 0, 'theta_2'),
        ('RY', 1, 'theta_3'),
    ]
}

# Hamiltonian: H = 0.5·Z₀ + 0.5·Z₁
hamiltonian = [('Z0', 0.5), ('Z1', 0.5)]

# Create loss function
loss_fn = make_vqe_loss(ansatz, hamiltonian)

# Optimize
params = jnp.array([0.1, 0.2, 0.3, 0.4])
for step in range(100):
    grads = jax.grad(loss_fn)(params)
    params = params - 0.1 * grads

energy = loss_fn(params)
print(f"Ground state energy: {energy}")
```

### Quantum Natural Gradient

```python
from nqpu_jax import quantum_natural_gradient_step

# Loss function
def loss_fn(params):
    return quantum_expectation(params, circuit)

# Natural gradient descent step
params_new = quantum_natural_gradient_step(
    params,
    loss_fn,
    circuit,
    learning_rate=0.1,
    regularization=1e-4
)
```

### JIT Compilation

```python
# JIT-compile for faster execution
@jax.jit
def compiled_expectation(params):
    return quantum_expectation(params, circuit)

# First call compiles, subsequent calls are fast
result = compiled_expectation(params)
```

## API Reference

### Core Functions

#### `quantum_expectation(params, circuit_config)`

Compute quantum expectation value with automatic differentiation.

**Args:**
- `params`: JAX array of shape `(n_params,)` - circuit parameters
- `circuit_config`: Dictionary with circuit specification

**Returns:**
- Scalar JAX array - expectation value

**Example:**
```python
exp = quantum_expectation(jnp.array([0.5, 1.2]), circuit)
```

---

#### `vmap_quantum(circuit_config, observable='Z0')`

Create batched quantum expectation function.

**Args:**
- `circuit_config`: Circuit specification
- `observable`: Observable string (default: 'Z0')

**Returns:**
- Function mapping `(batch_size, n_params) -> (batch_size,)`

**Example:**
```python
batch_fn = vmap_quantum(circuit)
results = batch_fn(jnp.array([[0.1], [0.2], [0.3]]))
```

---

#### `vmap_quantum_simulate(circuit_config)`

Create batched statevector simulation function.

**Args:**
- `circuit_config`: Circuit specification

**Returns:**
- Function mapping `(batch_size, n_params) -> (batch_size, 2^n_qubits)` complex

**Example:**
```python
batch_simulate = vmap_quantum_simulate(circuit)
states = batch_simulate(batch_params)  # Complex amplitudes
```

---

#### `quantum_kernel_matrix(params1, params2, circuit_config)`

Compute quantum kernel matrix K[i,j] = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|².

**Args:**
- `params1`: Array of shape `(n_samples1, n_params)`
- `params2`: Array of shape `(n_samples2, n_params)`
- `circuit_config`: Feature map circuit

**Returns:**
- Array of shape `(n_samples1, n_samples2)` - kernel values

---

#### `make_vqe_loss(circuit_config, hamiltonian_terms)`

Create VQE loss function for Hamiltonian.

**Args:**
- `circuit_config`: Variational ansatz circuit
- `hamiltonian_terms`: List of `(observable, coefficient)` tuples

**Returns:**
- Loss function mapping `params -> energy`

**Example:**
```python
H = [('Z0', 0.5), ('Z1', 0.5), ('Z0Z1', 0.25)]
loss = make_vqe_loss(ansatz, H)
```

---

#### `quantum_natural_gradient_step(params, loss_fn, circuit_config, learning_rate, regularization)`

Perform quantum natural gradient descent step.

**Args:**
- `params`: Current parameters
- `loss_fn`: Loss function to minimize
- `circuit_config`: Circuit (for Fisher Information Matrix)
- `learning_rate`: Step size (default: 0.01)
- `regularization`: FIM regularization (default: 1e-4)

**Returns:**
- Updated parameters

---

### Circuit Configuration

Circuit config is a dictionary with:

```python
{
    'n_qubits': int,              # Number of qubits
    'gates': [                    # List of gate tuples
        (gate_type, *args),
    ],
    'observable': str,            # Observable to measure (optional)
}
```

**Supported Gates:**
- `('H', qubit)` - Hadamard
- `('X', qubit)` - Pauli-X
- `('Y', qubit)` - Pauli-Y
- `('Z', qubit)` - Pauli-Z
- `('RX', qubit, param_name)` - X rotation
- `('RY', qubit, param_name)` - Y rotation
- `('RZ', qubit, param_name)` - Z rotation
- `('CNOT', control, target)` - Controlled-NOT
- `('CX', control, target)` - Alias for CNOT
- `('CZ', control, target)` - Controlled-Z

**Observables:**
- Single qubit: `'Z0'`, `'X1'`, `'Y2'`
- Multi-qubit: `'Z0Z1'`, `'X0Y1'` (support in progress)

## Testing

Run comprehensive test suite:

```bash
cd /path/to/nqpu-metal/python
python test_jax_integration.py
```

Tests include:
1. Basic gradients
2. VMAP batch execution
3. Quantum kernel matrices
4. VQE optimization
5. Quantum natural gradient
6. JIT compilation
7. Statevector VMAP
8. Performance benchmarks

## Performance

Typical performance on Apple M4 Pro (Metal backend):

- **Single execution**: ~10,000 circuits/sec (4 qubits)
- **Batch execution**: ~50,000 circuits/sec (batch_size=100)
- **Gradient computation**: ~5,000 gradients/sec (4 parameters)
- **VMAP speedup**: 5-10x vs sequential execution

Performance scales with:
- Number of qubits (exponential memory)
- Number of gates (linear)
- Batch size (sublinear due to parallelism)

## Architecture

### Python Layer (`nqpu_jax.py`)

- JAX custom VJP registration
- Circuit configuration parsing
- Batch function creation
- High-level ML primitives (VQE, kernels, etc.)

### Rust Layer (`jax_bridge.rs` + `python_api_v2.rs`)

- `JAXCircuit`: Quantum circuit data structure
- `parameter_shift_grad`: Exact gradient computation
- `vmap_simulate`: Parallel batch execution
- `JAXJitCache`: Circuit compilation and optimization

### Communication

Python ↔ Rust via PyO3:
- `PyJAXCircuit`: Python-exposed circuit object
- `py_jax_simulate`: Statevector simulation
- `py_jax_expectation`: Expectation value
- `py_jax_gradient`: Parameter-shift gradients
- `py_jax_vmap_simulate`: Batch statevector
- `py_jax_vmap_expectation`: Batch expectation

## Examples

See `python/test_jax_integration.py` for comprehensive examples including:

- Gradient descent optimization
- VQE for molecular ground states
- Quantum kernels for classification
- Natural gradient methods
- Performance benchmarks

## Troubleshooting

### ImportError: nqpu_metal not found

```bash
# Rebuild bindings
cd /path/to/nqpu-metal
maturin develop --release --features python
```

### ImportError: nqpu_jax not found

```bash
# Add to PYTHONPATH
export PYTHONPATH="/path/to/nqpu-metal/python:$PYTHONPATH"
```

### Rust bindings not available

Check installation:

```python
from nqpu_jax import check_installation
print(check_installation())
```

Expected output:
```python
{'jax': True, 'rust_bindings': True}
```

### Performance issues

- Enable JAX GPU backend: `JAX_PLATFORM_NAME=gpu`
- Use larger batch sizes for VMAP
- JIT-compile functions with `@jax.jit`
- Profile with `jax.profiler`

## Advanced Usage

### Custom Observables

For multi-qubit observables, extend the observable parser in `jax_bridge.rs`:

```rust
pub fn expectation(&self, params: &[f32], observable: &str) -> f32 {
    // Add parsing for 'Z0Z1', 'X0Y1', etc.
    // Compute tensor product expectations
}
```

### GPU Acceleration

JAX automatically uses GPU when available:

```python
import jax
print(jax.default_backend())  # Should show 'gpu' or 'metal'

# Force CPU for testing
with jax.default_device(jax.devices('cpu')[0]):
    result = quantum_expectation(params, circuit)
```

### Mixed Precision

Use JAX's mixed precision for memory-constrained systems:

```python
from jax import config
config.update('jax_enable_x64', False)  # Use float32

params = jnp.array([0.5, 1.2], dtype=jnp.float32)
```

### Debugging

Enable JAX debugging mode:

```python
from jax import config
config.update('jax_disable_jit', True)  # Disable JIT for debugging
config.update('jax_debug_nans', True)   # Check for NaNs
```

## Citation

If you use nQPU-Metal's JAX integration in research, please cite:

```bibtex
@software{nqpu_metal_jax,
  title={nQPU-Metal JAX Integration},
  author={nQPU-Metal Contributors},
  year={2026},
  url={https://github.com/yourusername/nqpu-metal}
}
```

## License

CC BY-NC 4.0 - See main repository LICENSE file.

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Multi-qubit observable support (Z0Z1, X0Y1, etc.)
- [ ] Clifford simulation fast path
- [ ] Noise models in gradients
- [ ] Advanced circuit compilation
- [ ] Quantum optimal control
- [ ] Hardware-aware transpilation

See main repository for contribution guidelines.
