# nQPU Rust SDK Guide

How to use the `nqpu-metal` crate directly in your own Rust projects for quantum simulation, from adding the dependency through advanced topics like tensor network backends and error correction.

## Table of Contents

- [Adding nqpu-metal to Your Project](#adding-nqpu-metal-to-your-project)
- [Core Types](#core-types)
- [Basic Circuit Construction](#basic-circuit-construction)
- [Using the Circuit DSL](#using-the-circuit-dsl)
- [Choosing Backends Programmatically](#choosing-backends-programmatically)
- [Error Correction Integration](#error-correction-integration)
- [Tensor Network Simulation](#tensor-network-simulation)
- [Feature Flags Reference](#feature-flags-reference)
- [Testing Your Quantum Code](#testing-your-quantum-code)

---

## Adding nqpu-metal to Your Project

### From a local path

If you have the nQPU repository checked out locally:

```toml
[dependencies]
nqpu-metal = { path = "../nQPU/sdk/rust" }
```

### From Git

```toml
[dependencies]
nqpu-metal = { git = "https://github.com/nqpu-metal/nqpu-metal.git" }
```

Pin to a specific revision for reproducibility:

```toml
[dependencies]
nqpu-metal = { git = "https://github.com/nqpu-metal/nqpu-metal.git", rev = "725b877" }
```

### Selecting features

The default features (`parallel`, `serde`) are sufficient for CPU-based simulation. Enable additional features as needed:

```toml
[dependencies]
nqpu-metal = { path = "../nQPU/sdk/rust", features = ["metal", "chemistry", "visualization"] }
```

### Build optimization

nQPU configures `opt-level = 2` for dependencies in dev and test profiles because numerical code (ndarray, nalgebra, num-complex) runs 10-50x slower at opt-level 0. Your code remains debuggable at opt-level 0 while the inner loops stay fast. No additional configuration is required on your side.

## Core Types

The crate's fundamental types are defined in `src/lib.rs` and re-exported at the crate root.

### QuantumState

The statevector representation of a quantum register. Stores `2^n` complex amplitudes for an n-qubit system.

```rust
use nqpu_metal::{QuantumState, C64};

// Create a 4-qubit state initialized to |0000>
let mut state = QuantumState::new(4);

// Inspect properties
assert_eq!(state.num_qubits, 4);
assert_eq!(state.dim, 16);  // 2^4

// Read amplitudes
let amp_0000: C64 = state.get(0);       // amplitude of |0000>
assert!((amp_0000.re - 1.0).abs() < 1e-10);

// Get all probabilities
let probs: Vec<f64> = state.probabilities();
assert!((probs[0] - 1.0).abs() < 1e-10);  // 100% in |0000>

// Measure (collapses probabilistically, returns (index, probability))
let (outcome, prob) = state.measure();

// Access raw amplitudes for advanced use
let amps: &[C64] = state.amplitudes_ref();
let amps_mut: &mut [C64] = state.amplitudes_mut();

// Compute fidelity between two states
let other = QuantumState::new(4);
let f = state.fidelity(&other);  // |<psi|phi>|^2
```

### C64 and C32

Type aliases for complex number types from the `num-complex` crate:

```rust
use nqpu_metal::{C64, C32, c64_zero, c64_one, c64_scale};

// C64 = Complex64 (double precision, 16 bytes)
let z = C64::new(0.5, -0.3);
println!("Real: {}, Imag: {}", z.re, z.im);
println!("Magnitude: {}", z.norm());

// Convenience constructors
let zero = c64_zero();   // 0 + 0i
let one = c64_one();     // 1 + 0i
let scaled = c64_scale(z, 2.0);  // 1.0 - 0.6i

// C32 = Complex32 (single precision, 8 bytes, for GPU/memory-constrained work)
```

### GateOperations

A unit struct with static methods for applying quantum gates to a `QuantumState`. Gates modify the state in-place. Multi-threaded via Rayon when the `parallel` feature is enabled.

```rust
use nqpu_metal::{QuantumState, GateOperations};

let mut state = QuantumState::new(3);

// Single-qubit gates
GateOperations::h(&mut state, 0);      // Hadamard
GateOperations::x(&mut state, 1);      // Pauli-X (NOT)
GateOperations::y(&mut state, 2);      // Pauli-Y
GateOperations::z(&mut state, 0);      // Pauli-Z
GateOperations::s(&mut state, 0);      // Phase gate (pi/2)
GateOperations::t(&mut state, 0);      // T gate (pi/8)

// Parameterized rotations
GateOperations::rx(&mut state, 0, std::f64::consts::FRAC_PI_4);  // Rx(pi/4)
GateOperations::ry(&mut state, 1, 0.5);                           // Ry(0.5)
GateOperations::rz(&mut state, 2, 1.0);                           // Rz(1.0)

// Two-qubit gates
GateOperations::cnot(&mut state, 0, 1);  // CNOT: control=0, target=1
GateOperations::cz(&mut state, 1, 2);    // Controlled-Z
GateOperations::swap(&mut state, 0, 2);  // SWAP

// Expectation values
let exp_z: f64 = state.expectation_z(0);  // <Z> on qubit 0
```

### Gate and GateType

The `Gate` struct and `GateType` enum provide a data-driven representation of quantum gates, useful for building circuits as data structures before execution:

```rust
use nqpu_metal::gates::{Gate, GateType};

// GateType covers the full universal gate set:
// Single-qubit: H, X, Y, Z, S, T, SX, Rx(f64), Ry(f64), Rz(f64),
//               Phase(f64), U { theta, phi, lambda }
// Two-qubit:    CNOT, CZ, SWAP, ISWAP, Rxx(f64), Ryy(f64), Rzz(f64),
//               CRx(f64), CRy(f64), CRz(f64), CR(f64)
// Three-qubit:  Toffoli (CCX), CCZ, CSWAP (Fredkin)
// Custom:       Custom(Vec<Vec<C64>>)

// Each GateType can produce its unitary matrix
let h_matrix = GateType::H.matrix();
// h_matrix[0][0] = 1/sqrt(2), h_matrix[0][1] = 1/sqrt(2), ...
```

### Primitives: Sampler and Estimator

For Qiskit V2-style programming, use the `Sampler` and `Estimator` primitives:

```rust
use nqpu_metal::primitives::*;

// Build a circuit
let circuit = CircuitBuilder::new(2)
    .h(0)
    .cx(0, 1)
    .measure_all()
    .build();

// Sample: execute and get measurement counts
let sampler = Sampler::new(SamplerConfig::default());
let result = sampler.run_single(&circuit, 1024);
println!("Counts: {:?}", result.counts);
// Typical output: {0: ~512, 3: ~512}  (|00> and |11>)

// Estimate: compute expectation values
let obs = Observable::from_string("ZZ");
let estimator = Estimator::new(EstimatorConfig::default());
let (value, std_err) = estimator.run_single(&circuit, &obs);
println!("<ZZ> = {:.4} +/- {:.4}", value, std_err);
// Bell state: <ZZ> = 1.0
```

## Basic Circuit Construction

### Manual gate-by-gate construction

The lowest-level approach applies gates directly to a `QuantumState`:

```rust
use nqpu_metal::{QuantumState, GateOperations};

fn bell_state() -> QuantumState {
    let mut state = QuantumState::new(2);
    GateOperations::h(&mut state, 0);
    GateOperations::cnot(&mut state, 0, 1);
    state
}

fn ghz_state(n: usize) -> QuantumState {
    let mut state = QuantumState::new(n);
    GateOperations::h(&mut state, 0);
    for i in 0..(n - 1) {
        GateOperations::cnot(&mut state, i, i + 1);
    }
    state
}

fn main() {
    let bell = bell_state();
    let probs = bell.probabilities();
    // |00> and |11> each have ~50% probability
    println!("|00>: {:.4}, |11>: {:.4}", probs[0], probs[3]);

    let ghz = ghz_state(5);
    let (outcome, _) = ghz.measure();
    // Outcome is either 0 (|00000>) or 31 (|11111>)
    println!("GHZ measurement: {:05b}", outcome);
}
```

### Using GateType data structures

For circuits that need to be stored, serialized, or analyzed before execution:

```rust
use nqpu_metal::gates::{Gate, GateType};

// Build a list of gates
let gates = vec![
    Gate::new(GateType::H, vec![0]),
    Gate::new(GateType::CNOT, vec![0, 1]),
    Gate::new(GateType::Rz(std::f64::consts::FRAC_PI_4), vec![1]),
    Gate::new(GateType::Toffoli, vec![0, 1, 2]),
];

// Analyze before running
println!("Gate count: {}", gates.len());
println!("Two-qubit gates: {}",
    gates.iter().filter(|g| g.target_qubits.len() == 2).count()
);
```

## Using the Circuit DSL

The `circuit_macro` module provides a fluent builder API for constructing quantum circuits, inspired by Qiskit's `QuantumCircuit`:

```rust
use nqpu_metal::circuit_macro::CircuitBuilder;

// Bell state with measurement
let circuit = CircuitBuilder::new(2)
    .h(0)
    .cx(0, 1)
    .measure_all()
    .build();

assert_eq!(circuit.num_qubits(), 2);
assert_eq!(circuit.gate_count(), 2);
```

### Full gate set

```rust
let circuit = CircuitBuilder::new(4)
    // Single-qubit gates
    .h(0)
    .x(1)
    .y(2)
    .z(3)
    .s(0)
    .t(1)
    .sdg(2)                   // S-dagger
    .tdg(3)                   // T-dagger
    .rx(0, 0.5)               // Rx(theta)
    .ry(1, 1.0)               // Ry(theta)
    .rz(2, 0.25)              // Rz(theta)
    .u3(3, 0.1, 0.2, 0.3)    // U3(theta, phi, lambda)

    // Two-qubit gates
    .cx(0, 1)                  // CNOT
    .cz(2, 3)                  // Controlled-Z
    .swap(0, 3)                // SWAP
    .crx(0, 1, 0.5)           // Controlled-Rx
    .crz(2, 3, 0.25)          // Controlled-Rz

    // Three-qubit gates
    .ccx(0, 1, 2)              // Toffoli (CCX)
    .cswap(0, 1, 2)            // Fredkin (CSWAP)

    // Scheduling
    .barrier_all()             // Barrier on all qubits

    // Measurement
    .measure(0, 0)             // Measure qubit 0 into classical bit 0
    .measure_all()             // Measure all qubits

    .build();
```

### Range operations

Apply gates across ranges of qubits:

```rust
let circuit = CircuitBuilder::new(8)
    .h_range(0..4)          // H on qubits 0, 1, 2, 3
    .x_range(4..8)          // X on qubits 4, 5, 6, 7
    .cx_chain(0..4)         // CNOT chain: cx(0,1), cx(1,2), cx(2,3)
    .barrier(vec![0, 1, 2]) // Barrier on specific qubits
    .build();
```

### Circuit statistics and visualization

```rust
let circuit = CircuitBuilder::new(3)
    .h(0)
    .cx(0, 1)
    .cx(1, 2)
    .rz(2, 0.5)
    .measure_all()
    .build();

// Statistics
let stats = circuit.statistics();
println!("Total gates: {}", stats.total_gates);
println!("Single-qubit gates: {}", stats.single_qubit_gates);
println!("Two-qubit gates: {}", stats.two_qubit_gates);
println!("Depth: {}", stats.depth);

// ASCII visualization
println!("{}", circuit.draw());
// Output:
// q0: --|H|--*--------M--
// q1: ------CX--*-----M--
// q2: ---------CX-Rz--M--
```

### Circuit composition

Combine circuits into larger ones:

```rust
let bell = CircuitBuilder::new(2).h(0).cx(0, 1).build();
let measure = CircuitBuilder::new(2).measure_all().build();

// Append one circuit to another
let full = CircuitBuilder::from_circuit(&bell)
    .append(&measure)
    .build();
```

## Choosing Backends Programmatically

nQPU includes an automatic backend selector that analyzes circuit properties and recommends the best execution strategy:

```rust
use nqpu_metal::auto_backend::{AutoBackend, BackendType};

let auto = AutoBackend::default();
let analysis = auto.analyze(&gates);

println!("Recommended: {} ({})", analysis.recommended_backend.name(), analysis.reasoning);
println!("Qubits: {}", analysis.num_qubits);
println!("Gates: {}", analysis.num_gates);
println!("Depth: {}", analysis.depth);
println!("Entanglement: {:.2}", analysis.entanglement_estimate);
println!("Clifford: {}", analysis.is_clifford);
```

### Available backends

| Backend | Type | When to Use | Feature Flag |
|---------|------|-------------|-------------|
| `CPU` | Sequential statevector | < 6 qubits, simple circuits | default |
| `Fused` | Multi-threaded CPU (Rayon) | 6-20 qubits, general purpose | `parallel` |
| `F32Fused` | Single-precision fused CPU | Memory-constrained, 20-25 qubits | default |
| `MetalGPU` | Apple Silicon GPU | 10-29 qubits on macOS | `metal` |
| `CudaGPU` | NVIDIA GPU | 10-29 qubits on Linux/Windows | `cuda` |
| `MPS` | Matrix Product State | 30+ qubits, low entanglement | default |
| `Distributed` | Multi-node MPI | 30+ qubits, full entanglement | `distributed` |

### Manual backend selection

```rust
// Statevector (default)
use nqpu_metal::{QuantumState, GateOperations};
let mut state = QuantumState::new(10);
GateOperations::h(&mut state, 0);

// MPS simulation
use nqpu_metal::tensor_network::MPSSimulator;
let mut mps = MPSSimulator::new(50, 64);  // 50 qubits, bond dim 64
mps.apply_h(0);
mps.apply_cnot(0, 1);

// Stabilizer simulation (Clifford-only circuits, scales to 1000+ qubits)
use nqpu_metal::StabilizerSimulator;
let mut stab = StabilizerSimulator::new(100);
stab.hadamard(0);
stab.cnot(0, 1);

// F32 for half-memory statevector
use nqpu_metal::quantum_f32::QuantumStateF32;
let state_f32 = QuantumStateF32::new(25);  // 256MB instead of 512MB

// GPU (macOS, requires `metal` feature)
#[cfg(feature = "metal")]
{
    use nqpu_metal::MetalParallelQuantumExecutor;
    // ... GPU execution ...
}
```

## Error Correction Integration

nQPU implements several quantum error correction codes and decoders.

### Supported codes

- **Repetition codes**: Bit-flip and phase-flip correction
- **Shor's 9-qubit code**: Corrects arbitrary single-qubit errors
- **Steane's 7-qubit code**: CSS code using stabilizer formalism
- **Surface codes**: Planar and rotated surface codes
- **Color codes**: Topological codes

### Syndrome measurement and decoding

```rust
use nqpu_metal::quantum_error_correction::{Syndrome, Decoder};

// Measure syndromes from stabilizers
let syndrome = Syndrome::new(vec![false, true, false, true]);

println!("Syndrome weight: {}", syndrome.weight());
println!("Trivial (no error): {}", syndrome.is_trivial());

// Decode to identify error locations
// Multiple decoders are available:
// - MWPM (Minimum Weight Perfect Matching) via crate::decoding::mwpm
// - BP (Belief Propagation) via crate::decoding::bp
```

### Production decoders

For production use, the crate provides full MWPM (Blossom V) and belief propagation decoders in the `decoding` module:

```rust
use nqpu_metal::decoding::mwpm;  // Full Blossom V implementation
use nqpu_metal::decoding::bp;    // Min-sum belief propagation
```

### Surface code with dynamic decoder

```rust
use nqpu_metal::{DynamicSurfaceCode, RlDecoder};

let code = DynamicSurfaceCode::new(5);  // distance-5 surface code
let decoder = RlDecoder::new(5);

// Run error correction cycle
let report = code.run_cycle(&decoder);
println!("Corrections applied: {:?}", report);
```

## Tensor Network Simulation

For systems exceeding statevector memory limits, nQPU provides tensor network backends.

### Matrix Product State (MPS)

MPS represents the quantum state as a chain of tensors, trading accuracy for memory efficiency. Memory scales as `O(n * chi^2)` where `chi` is the bond dimension and `n` is the qubit count, compared to `O(2^n)` for statevector.

```rust
use nqpu_metal::tensor_network::MPSSimulator;

// 50 qubits with bond dimension 128
let mut mps = MPSSimulator::new(50, 128);

// Apply gates (same interface as statevector)
mps.apply_h(0);
for i in 0..49 {
    mps.apply_cnot(i, i + 1);
}

// Measure
let outcome = mps.measure(0);
```

### Adaptive MPS

Automatically expands bond dimension when entanglement exceeds thresholds:

```rust
use nqpu_metal::adaptive_mps::AdaptiveConfig;

let config = AdaptiveConfig::new()
    .with_initial_bond_dim(4)          // Start small
    .with_max_bond_dim(256)            // Cap at 256
    .with_expansion_threshold(0.7);    // Expand at 70% capacity

// The adaptive MPS automatically grows bond dimension when the
// entanglement entropy approaches the current limit, preventing
// silent accuracy loss.
```

### Bond dimension guidelines

The bond dimension `chi` controls the tradeoff between accuracy and resources:

| Circuit Type | Recommended chi | Accuracy |
|-------------|----------------|----------|
| Product states (no entanglement) | 1 | Exact |
| GHZ / W states | 2 | Exact |
| Shallow random circuits | 16-64 | Good |
| Molecular ground states (VQE) | 64-256 | Chemical accuracy |
| Deep random circuits (volume law) | Exponential | MPS not suitable |

If your circuit generates volume-law entanglement (e.g., random circuits beyond log-depth), MPS simulation will either require exponential bond dimension or produce inaccurate results. Use statevector or GPU backends instead.

## Feature Flags Reference

| Feature | Default | Description |
|---------|---------|-------------|
| `parallel` | Yes | Multi-threaded CPU simulation via Rayon |
| `serde` | Yes | Serialization support for states and circuits |
| `metal` | No | Apple Metal GPU acceleration (macOS only) |
| `metal4` | No | Metal 4 tensor ops + inline ML (requires macOS 15+, implies `metal`) |
| `amx` | No | Apple AMX (Accelerate) tensor contractions for MPS inner loops |
| `cuda` | No | NVIDIA GPU support via cudarc |
| `rocm` | No | AMD GPU support (stub) |
| `python` | No | Python bindings via PyO3 (implies `serde`) |
| `chemistry` | No | Quantum chemistry: fermion mappings, molecular Hamiltonians, drug design |
| `networking` | No | Quantum networking: channels, repeaters, purification |
| `visualization` | No | Circuit and state visualization via plotters |
| `distributed` | No | Distributed multi-node simulation via MPI |
| `web` | No | REST API server with Axum (implies `serde`) |
| `wasm` | No | WebAssembly/WebGPU backend (conflicts with `metal`/`cuda`) |
| `lsp` | No | QASM Language Server Protocol support (implies `serde`) |
| `qpu` | No | Real QPU hardware connectivity (IBM, Braket, Azure, IonQ, Google) |
| `qpu-ibm` | No | IBM Quantum provider (implies `qpu`) |
| `qpu-braket` | No | Amazon Braket provider (implies `qpu`) |
| `qpu-azure` | No | Azure Quantum provider (implies `qpu`) |
| `qpu-ionq` | No | IonQ Direct provider (implies `qpu`) |
| `qpu-google` | No | Google Quantum AI provider (implies `qpu`) |
| `qpu-all` | No | All QPU providers |
| `experimental` | No | Experimental/research modules |
| `all-gpus` | No | Metal + CUDA + ROCm |
| `full` | No | Everything except experimental (python, all GPUs, viz, distributed, web, chemistry, networking, AMX, all QPUs) |

### Combining features

Common feature combinations:

```toml
# Local macOS development with chemistry
nqpu-metal = { path = "...", features = ["metal", "chemistry"] }

# Linux HPC with NVIDIA GPUs
nqpu-metal = { path = "...", features = ["cuda", "distributed", "chemistry"] }

# Python extension module
nqpu-metal = { path = "...", features = ["python", "chemistry"] }

# Web-based simulator
nqpu-metal = { path = "...", features = ["web", "visualization"] }

# Everything for local testing
nqpu-metal = { path = "...", features = ["full"] }
```

## Testing Your Quantum Code

### Writing quantum tests

Quantum simulation is deterministic (before measurement), so you can test gate operations with exact amplitude comparisons:

```rust
#[cfg(test)]
mod tests {
    use nqpu_metal::{QuantumState, GateOperations, C64};

    #[test]
    fn test_hadamard_creates_superposition() {
        let mut state = QuantumState::new(1);
        GateOperations::h(&mut state, 0);

        let probs = state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state() {
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);

        let probs = state.probabilities();
        // |00> and |11> each have 50% probability
        assert!((probs[0] - 0.5).abs() < 1e-10);  // |00>
        assert!((probs[1]).abs() < 1e-10);          // |01>
        assert!((probs[2]).abs() < 1e-10);          // |10>
        assert!((probs[3] - 0.5).abs() < 1e-10);   // |11>
    }

    #[test]
    fn test_x_gate_flips_qubit() {
        let mut state = QuantumState::new(1);
        GateOperations::x(&mut state, 0);

        // State should be |1>
        let amp_0 = state.get(0);
        let amp_1 = state.get(1);
        assert!(amp_0.norm_sqr() < 1e-10);
        assert!((amp_1.norm_sqr() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fidelity_identical_states() {
        let state = QuantumState::new(3);
        let f = state.fidelity(&state);
        assert!((f - 1.0).abs() < 1e-10);
    }
}
```

### Testing with expectation values

For algorithms that produce statistical results, test expectation values rather than individual measurements:

```rust
#[test]
fn test_z_expectation_after_hadamard() {
    let mut state = QuantumState::new(1);
    GateOperations::h(&mut state, 0);

    // <Z> should be 0 in the |+> state
    let exp_z = state.expectation_z(0);
    assert!(exp_z.abs() < 1e-10);
}

#[test]
fn test_vqe_hydrogen_energy() {
    use nqpu_metal::vqe::{VQESolver, hamiltonians};

    let h2 = hamiltonians::hydrogen_molecule(
        -0.8105, 0.1721, -0.2257, 0.1709, 0.0453, 0.0453,
    );

    let mut solver = VQESolver::new(2, 3, h2, 0.1);
    solver.max_iterations = 500;

    let result = solver.find_ground_state();

    // H2 ground state energy should be approximately -1.137 Ha
    assert!(result.ground_state_energy < -0.8);
    assert!(!result.energy_history.is_empty());
}
```

### Testing with the Sampler primitive

For statistical tests, use enough shots and reasonable tolerances:

```rust
#[test]
fn test_bell_state_sampling() {
    use nqpu_metal::primitives::*;

    let circuit = CircuitBuilder::new(2)
        .h(0)
        .cx(0, 1)
        .measure_all()
        .build();

    let sampler = Sampler::new(SamplerConfig::default());
    let result = sampler.run_single(&circuit, 10_000);

    let count_00 = *result.counts.get(&0b00).unwrap_or(&0);
    let count_11 = *result.counts.get(&0b11).unwrap_or(&0);
    let count_01 = *result.counts.get(&0b01).unwrap_or(&0);
    let count_10 = *result.counts.get(&0b10).unwrap_or(&0);

    // |00> and |11> should dominate
    assert!(count_00 > 4000);
    assert!(count_11 > 4000);
    assert!(count_01 < 100);  // Should be ~0, allow statistical noise
    assert!(count_10 < 100);
}
```

### Running the test suite

```bash
# Run all tests (default features)
cargo test --manifest-path sdk/rust/Cargo.toml

# Run with chemistry module
cargo test --manifest-path sdk/rust/Cargo.toml --features chemistry

# Run a specific test
cargo test --manifest-path sdk/rust/Cargo.toml test_bell_state

# Run benchmarks
cargo bench --manifest-path sdk/rust/Cargo.toml
```

### Debugging tips

1. **Use `probabilities()`** to inspect the full state distribution after each gate.
2. **Use `amplitudes_ref()`** to check complex amplitudes when phase matters.
3. **Use `fidelity()`** to compare your state against a known reference state.
4. **Use the ASCII circuit drawer** (`circuit.draw()`) to visually verify circuit structure.
5. **Start with small qubit counts** (2-4) where you can verify results by hand.
6. **Test gate identities**: `H*H = I`, `X*X = I`, `CNOT*CNOT = I` are reliable sanity checks.
