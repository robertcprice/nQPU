# JAX Integration Architecture

Visual architecture documentation for nQPU-Metal JAX integration.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Code (Python)                      │
│  import jax                                                     │
│  from nqpu_jax import quantum_expectation                      │
│                                                                 │
│  params = jnp.array([0.5, 1.2])                                │
│  exp = quantum_expectation(params, circuit)                    │
│  grad = jax.grad(quantum_expectation)(params, circuit)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   JAX Automatic Differentiation                 │
│  - Traces computation graph                                     │
│  - Calls custom VJP on backward pass                           │
│  - Manages JIT compilation                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              nqpu_jax.py (Python Integration Layer)             │
│                                                                 │
│  @custom_vjp                                                    │
│  def quantum_expectation(params, circuit_config):               │
│      circuit = JAXCircuitCompiler.compile(config)               │
│      return py_jax_expectation(circuit, params, obs)            │
│                                                                 │
│  def quantum_expectation_fwd(params, config):                   │
│      exp_val = quantum_expectation(params, config)              │
│      return exp_val, (params, config)  # residuals              │
│                                                                 │
│  def quantum_expectation_bwd(residuals, g):                     │
│      params, config = residuals                                 │
│      grads = py_jax_gradient(circuit, params, qubit)            │
│      return (grads * g, None)                                   │
│                                                                 │
│  quantum_expectation.defvjp(fwd, bwd)                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PyO3 Bindings (Rust ↔ Python Bridge)               │
│                                                                 │
│  #[pyclass] PyJAXCircuit { circuit: JAXCircuit }                │
│                                                                 │
│  #[pyfunction]                                                  │
│  fn py_jax_simulate(circuit, params) -> Vec<(f32, f32)>         │
│                                                                 │
│  #[pyfunction]                                                  │
│  fn py_jax_expectation(circuit, params, obs) -> f32             │
│                                                                 │
│  #[pyfunction]                                                  │
│  fn py_jax_gradient(circuit, params, qubit) -> Vec<f32>         │
│                                                                 │
│  #[pyfunction]                                                  │
│  fn py_jax_vmap_simulate(circuit, batch_params)                 │
│      -> Vec<Vec<(f32, f32)>>                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                jax_bridge.rs (Rust Quantum Engine)              │
│                                                                 │
│  pub struct JAXCircuit {                                        │
│      n_qubits: usize,                                           │
│      gates: Vec<JAXGate>,                                       │
│      parameter_names: Vec<String>,                              │
│  }                                                              │
│                                                                 │
│  impl JAXCircuit {                                              │
│      pub fn simulate(&self, params: &[f32])                     │
│          -> Vec<JaxComplex> { ... }                             │
│                                                                 │
│      pub fn expectation(&self, params: &[f32], obs: &str)       │
│          -> f32 { ... }                                         │
│  }                                                              │
│                                                                 │
│  pub fn parameter_shift_grad(circuit, params, qubit, shift)     │
│      -> Vec<f32> {                                              │
│      // ∂f/∂θ = (f(θ+π/2) - f(θ-π/2)) / 2                      │
│      let exp_plus = circuit.expect_z(&params_plus, qubit);      │
│      let exp_minus = circuit.expect_z(&params_minus, qubit);    │
│      (exp_plus - exp_minus) / (2.0 * shift)                     │
│  }                                                              │
│                                                                 │
│  pub fn vmap_simulate(circuit, batch_params)                    │
│      -> Vec<Vec<JaxComplex>> {                                  │
│      batch_params.par_iter()                                    │
│          .map(|p| circuit.simulate(p))                          │
│          .collect()                                             │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Forward Pass

```
User Code
  ↓ params = [0.5, 1.2]
  ↓ circuit = {'n_qubits': 2, 'gates': [...]}
quantum_expectation(params, circuit)
  ↓
JAXCircuitCompiler.compile(circuit)
  ↓ PyJAXCircuit object
py_jax_expectation(circuit, params, 'Z0')
  ↓ (via PyO3)
JAXCircuit.expectation(params, 'Z0')
  ↓
JAXCircuit.simulate(params)
  ↓ Apply gates: H, RY(0.5), RY(1.2), CNOT
  ↓ statevector = [c0, c1, c2, c3]
JAXCircuit.expect_z(params, 0)
  ↓ Compute ∑ᵢ |cᵢ|² · (±1)
  ↓
return 0.7853... (f32)
  ↓ (via PyO3)
return jnp.array(0.7853)
  ↓
JAX wraps result
```

### Backward Pass (Gradient)

```
JAX calls quantum_expectation_bwd(residuals, g)
  ↓ residuals = (params, circuit_config)
  ↓ g = upstream gradient (1.0 for loss)
py_jax_gradient(circuit, params, qubit=0)
  ↓ (via PyO3)
parameter_shift_grad(circuit, params, 0, π/2)
  ↓
FOR each parameter θᵢ:
  ↓ params_plus = [θ₀, ..., θᵢ+π/2, ..., θₙ]
  ↓ params_minus = [θ₀, ..., θᵢ-π/2, ..., θₙ]
  ↓ exp_plus = circuit.expect_z(params_plus, 0)
  ↓ exp_minus = circuit.expect_z(params_minus, 0)
  ↓ grad[i] = (exp_plus - exp_minus) / π
  ↓
return grads = [∂f/∂θ₀, ∂f/∂θ₁, ...]
  ↓ (via PyO3)
return jnp.array(grads)
  ↓
grads * g  # Chain rule
  ↓
return to JAX
```

## Memory Layout

### Statevector Representation

```rust
// Rust: Vec<JaxComplex>
struct JaxComplex {
    real: f32,  // 4 bytes
    imag: f32,  // 4 bytes
}  // Total: 8 bytes per amplitude

// For n_qubits = 2:
statevector = [
    JaxComplex { real: 0.707, imag: 0.0 },    // |00⟩ amplitude
    JaxComplex { real: 0.0,   imag: 0.0 },    // |01⟩ amplitude
    JaxComplex { real: 0.0,   imag: 0.0 },    // |10⟩ amplitude
    JaxComplex { real: 0.707, imag: 0.0 },    // |11⟩ amplitude
]
// Size: 4 amplitudes × 8 bytes = 32 bytes
```

### Python-Rust Conversion

```python
# Python → Rust
params_python = jnp.array([0.5, 1.2])  # JAX array
params_numpy = np.asarray(params_python, dtype=np.float32)  # NumPy
params_rust = Vec<f32>  # Rust (via PyO3)

# Rust → Python
result_rust = Vec<(f32, f32)>  # Complex pairs
result_python = jnp.asarray(result_rust, dtype=jnp.complex64)  # JAX
```

## Parallelization Strategy

### Batch Execution (VMAP)

```
py_jax_vmap_simulate(circuit, batch_params)
  ↓ batch_params = [[p0], [p1], [p2], ..., [p99]]
vmap_simulate(circuit, batch_params)
  ↓
Rayon parallel iterator:
  batch_params.par_iter()  // Parallel across CPU cores
    .map(|params| circuit.simulate(params))
    ↓
  Thread 0: simulate([p0])  → state0
  Thread 1: simulate([p1])  → state1
  Thread 2: simulate([p2])  → state2
  ...
  Thread N: simulate([p99]) → state99
    ↓
  .collect() → Vec<Vec<JaxComplex>>
    ↓
return [state0, state1, ..., state99]
```

## Circuit Compilation

### JAXCircuit Representation

```rust
circuit = JAXCircuit {
    n_qubits: 4,
    gates: vec![
        JAXGate::H(0),
        JAXGate::RY { qubit: 0, param_idx: 0 },
        JAXGate::RY { qubit: 1, param_idx: 1 },
        JAXGate::CNOT { control: 0, target: 1 },
    ],
    parameter_names: vec!["theta".to_string(), "phi".to_string()],
}
```

### JIT Cache

```rust
cache = JAXJitCache {
    circuit_hash: 0x1a2b3c4d,
    compiled_gates: vec![
        CompiledGate::H { mask: 0b0001 },
        CompiledGate::RY { mask: 0b0001, param_idx: 0 },
        CompiledGate::RY { mask: 0b0010, param_idx: 1 },
        CompiledGate::CNOT {
            control_mask: 0b0001,
            target_mask: 0b0010
        },
    ],
}

// Faster execution:
cache.execute(n_qubits=4, params=[0.5, 1.2])
```

## Error Handling

```
User Code
  ↓
quantum_expectation(params, circuit)
  ↓
try:
    JAXCircuitCompiler.compile(circuit)
except ValueError as e:
    ↓ "Circuit config must have 'n_qubits' field"
    ↓ "Unknown gate type: CUSTOM"

try:
    py_jax_expectation(circuit, params, obs)
except RuntimeError as e:
    ↓ "Rust bindings not available"
    ↓ "Parameter count mismatch"

try:
    JAXCircuit.simulate(params)
except (in Rust):
    ↓ Return error via PyO3
    ↓ Python raises RuntimeError
```

## Type Flow

```
Python Types        PyO3 Types           Rust Types
────────────        ────────────         ──────────
jnp.ndarray[f32]  → Vec<f32>          → &[f32]
jnp.ndarray[c64]  → Vec<(f32, f32)>  → Vec<JaxComplex>
dict              → (converted)        → JAXCircuit
str               → &str              → &str
int               → usize             → usize
```

## Performance Characteristics

### Time Complexity

| Operation | Time | Notes |
|-----------|------|-------|
| Circuit compilation | O(G) | G = number of gates |
| Statevector simulation | O(G · 2ⁿ) | n = number of qubits |
| Expectation value | O(2ⁿ) | Sum over all amplitudes |
| Parameter-shift grad | O(P · G · 2ⁿ) | P = number of parameters |
| VMAP (batch of B) | O(B · G · 2ⁿ / C) | C = number of CPU cores |

### Space Complexity

| Structure | Size | Scaling |
|-----------|------|---------|
| Statevector | 8 · 2ⁿ bytes | Exponential in qubits |
| Circuit | ~100 bytes + G · 20 bytes | Linear in gates |
| Parameters | 4 · P bytes | Linear in parameters |
| Batch (size B) | B · 8 · 2ⁿ bytes | Linear in batch size |

## Gradient Computation Method

### Parameter-Shift Rule

For a gate R(θ) = exp(-iθG/2) where G² = I:

```
∂⟨H⟩/∂θ = (⟨H⟩(θ + π/2) - ⟨H⟩(θ - π/2)) / 2

Implementation:
1. Compute f(θ) with θᵢ → θᵢ + π/2
2. Compute f(θ) with θᵢ → θᵢ - π/2
3. Return (f₊ - f₋) / 2
```

### Advantages

- ✅ Exact gradients (no approximation)
- ✅ No tape storage (recompute on backward)
- ✅ Works with any differentiable observable
- ✅ Parallelizable across parameters

### Disadvantages

- ❌ 2 forward passes per parameter
- ❌ Not optimal for large parameter counts
- ❌ Requires π/2 shift (hardware-dependent)

### Future: Adjoint Method

For expectation values ⟨ψ|H|ψ⟩:

```
∂⟨ψ|H|ψ⟩/∂θ = 2 Re[⟨∂ψ/∂θ|H|ψ⟩]

Advantages:
- Single backward pass
- Memory for adjoint state
- More efficient for large P

To implement:
- Add adjoint_gradient() in jax_bridge.rs
- Track gate derivatives
- Backpropagate through circuit
```

## Extension Points

### Adding New Gates

```rust
// 1. Define gate in enum
pub enum JAXGate {
    // ... existing gates
    SWAP { a: usize, b: usize },
}

// 2. Implement gate application
impl JAXCircuit {
    fn apply_swap(&self, state: &mut [JaxComplex], a: usize, b: usize) {
        // Gate matrix implementation
    }
}

// 3. Add to simulate()
match gate {
    JAXGate::SWAP { a, b } => self.apply_swap(&mut state, *a, *b),
    // ...
}
```

### Adding New Observables

```rust
// Extend expectation() method
pub fn expectation(&self, params: &[f32], observable: &str) -> f32 {
    match observable {
        s if s.starts_with("Z") => { /* single qubit */ }
        s if s.contains("Z") && s.len() > 2 => {
            // Multi-qubit: "Z0Z1"
            self.expect_multi_qubit(params, s)
        }
        _ => 0.0
    }
}

fn expect_multi_qubit(&self, params: &[f32], obs: &str) -> f32 {
    // Parse observable string
    // Compute tensor product expectation
}
```

### Adding Noise Models

```rust
pub struct NoisyJAXCircuit {
    circuit: JAXCircuit,
    noise_model: NoiseModel,
}

impl NoisyJAXCircuit {
    pub fn simulate_noisy(&self, params: &[f32]) -> DensityMatrix {
        // Apply gates with noise channels
        // Return density matrix instead of state
    }
}
```

## Debugging Tools

### Python Side

```python
# Check installation
from nqpu_jax import check_installation
print(check_installation())

# Enable JAX debugging
from jax import config
config.update('jax_disable_jit', True)
config.update('jax_debug_nans', True)

# Print circuit compilation
from nqpu_jax import JAXCircuitCompiler
circuit = JAXCircuitCompiler.compile(config)
print(circuit)
```

### Rust Side

```rust
// Add debug logging
#[cfg(debug_assertions)]
eprintln!("Simulating circuit with {} qubits", n_qubits);

// Assert invariants
debug_assert!(state.len() == 1 << n_qubits);
debug_assert!(params.len() == self.num_parameters());
```

## Conclusion

The architecture provides:

1. **Clean separation**: Python ↔ PyO3 ↔ Rust layers
2. **Zero overhead**: All computation in Rust
3. **JAX integration**: Custom VJP, JIT, VMAP
4. **Parallelization**: Rayon for batch execution
5. **Type safety**: Strong typing throughout
6. **Extensibility**: Clear extension points
7. **Performance**: Optimized gradient computation

The design enables seamless quantum computing in the JAX ecosystem while maintaining high performance and correctness.
