# JAX Integration Quick Start

Get up and running with nQPU-Metal JAX integration in 5 minutes.

## Installation

```bash
cd /path/to/nqpu-metal

# Run setup script (installs JAX + builds Rust bindings)
./python/setup_jax.sh

# Add to PYTHONPATH
export PYTHONPATH="$(pwd)/python:$PYTHONPATH"
```

## Hello World

```python
import jax.numpy as jnp
from nqpu_jax import quantum_expectation

# Define circuit
circuit = {
    'n_qubits': 2,
    'gates': [
        ('H', 0),
        ('RY', 0, 'theta'),
        ('CNOT', 0, 1),
    ],
    'observable': 'Z0'
}

# Execute
params = jnp.array([0.5])
exp_val = quantum_expectation(params, circuit)
print(f"Expectation: {exp_val}")

# Gradients
import jax
grad_fn = jax.grad(quantum_expectation, argnums=0)
grads = grad_fn(params, circuit)
print(f"Gradient: {grads}")
```

## Common Patterns

### 1. VQE Optimization

```python
from nqpu_jax import make_vqe_loss
import jax

ansatz = {
    'n_qubits': 2,
    'gates': [
        ('RY', 0, 'p0'), ('RY', 1, 'p1'),
        ('CNOT', 0, 1),
        ('RY', 0, 'p2'), ('RY', 1, 'p3'),
    ]
}

hamiltonian = [('Z0', 0.5), ('Z1', 0.5)]
loss_fn = make_vqe_loss(ansatz, hamiltonian)

params = jnp.array([0.1, 0.2, 0.3, 0.4])
for _ in range(100):
    grads = jax.grad(loss_fn)(params)
    params -= 0.1 * grads

print(f"Ground state energy: {loss_fn(params)}")
```

### 2. Batch Processing

```python
from nqpu_jax import vmap_quantum

# Create batched function
batch_fn = vmap_quantum(circuit)

# Execute on batch
batch_params = jnp.array([[0.1], [0.2], [0.3]])
results = batch_fn(batch_params)  # Fast parallel execution
```

### 3. Quantum Kernel

```python
from nqpu_jax import quantum_kernel_matrix

feature_map = {
    'n_qubits': 4,
    'gates': [
        ('H', i) for i in range(4)
    ] + [
        ('RY', i, f'x{i}') for i in range(4)
    ] + [
        ('CNOT', i, i+1) for i in range(3)
    ]
}

X_train = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
X_test = jnp.array([[0.2, 0.3, 0.4, 0.5]])

K = quantum_kernel_matrix(X_train, X_test, feature_map)
```

## Testing

```bash
# Run full test suite
python python/test_jax_integration.py

# Run specific test
python -c "
from nqpu_jax import check_installation
print(check_installation())
"
```

## Troubleshooting

**Import error?**
```bash
export PYTHONPATH="$(pwd)/python:$PYTHONPATH"
```

**Rust bindings missing?**
```bash
maturin develop --release --features python
```

**JAX not found?**
```bash
pip install jax jaxlib
```

## Next Steps

- Read full docs: `python/README_JAX.md`
- Try Jupyter demo: `python/jax_integration_demo.ipynb`
- Check examples in `test_jax_integration.py`

## Performance Tips

1. **JIT compile** functions with `@jax.jit`
2. **Use VMAP** for batches (5-10x speedup)
3. **Enable GPU** with `JAX_PLATFORM_NAME=gpu`
4. **Larger circuits** benefit more from Rust acceleration

## API Cheat Sheet

| Function | Purpose | Example |
|----------|---------|---------|
| `quantum_expectation(params, circuit)` | Compute ⟨ψ\|H\|ψ⟩ | `exp = quantum_expectation(p, c)` |
| `vmap_quantum(circuit)` | Batch function | `batch_fn = vmap_quantum(c)` |
| `quantum_kernel_matrix(X1, X2, c)` | Quantum kernel | `K = quantum_kernel_matrix(X, X, c)` |
| `make_vqe_loss(ansatz, H)` | VQE loss | `loss = make_vqe_loss(a, H)` |
| `quantum_natural_gradient_step(...)` | Natural grad | `p_new = quantum_natural_gradient_step(...)` |

## Circuit Config Format

```python
{
    'n_qubits': int,              # Number of qubits
    'gates': [                    # Gate list
        ('H', qubit),
        ('RY', qubit, 'param_name'),
        ('CNOT', control, target),
    ],
    'observable': 'Z0'            # What to measure
}
```

## Support

- Issues: GitHub issue tracker
- Docs: `README_JAX.md`
- Examples: `test_jax_integration.py`
- Chat: Project Discord/Slack
