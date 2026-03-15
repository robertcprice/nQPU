# Getting Started with nQPU

This guide walks you through building the nQPU quantum computing SDK, running
your first simulation, and choosing the right backend for your workload.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Rust** | 1.70+ (edition 2021) | Install via [rustup](https://rustup.rs/) |
| **macOS** | 13+ (Ventura) | Required for Metal GPU acceleration |
| **Xcode CLT** | Latest | Provides the Metal compiler (`xcrun metal`) |

The CPU-only statevector backend compiles on Linux and Windows as well, but
Metal GPU acceleration is macOS-only. CUDA support requires an NVIDIA GPU and
the CUDA toolkit.

## Building from source

Clone the repository and build the release binary:

```bash
git clone https://github.com/entropy-research/nqpu.git
cd nqpu/sdk/rust
cargo build --release
```

The crate is called `nqpu-metal` and produces both a static library (`rlib`)
and a C-compatible dynamic library (`cdylib`).  Default features include
`parallel` (Rayon multi-threading) and `serde` (serialization).

## Your first quantum circuit

The two primary types you need are `QuantumState` and `GateOperations`, both
defined in the crate root.

```rust
use nqpu_metal::{QuantumState, GateOperations};

fn main() {
    // 1. Create a 3-qubit state initialized to |000>
    let mut state = QuantumState::new(3);

    // 2. Put qubit 0 into superposition
    GateOperations::h(&mut state, 0);

    // 3. Entangle qubit 0 with qubit 1 (Bell pair)
    GateOperations::cnot(&mut state, 0, 1);

    // 4. Inspect probabilities
    let probs = state.probabilities();
    println!("Probabilities: {:?}", probs);
    // Expected: ~0.5 for |000> and |011>, 0 elsewhere

    // 5. Measure (collapses to a computational basis state)
    let (outcome, probability) = state.measure();
    println!("Measured |{:03b}> with p = {:.4}", outcome, probability);

    // 6. Multi-shot sampling for statistics
    let state = {
        let mut s = QuantumState::new(3);
        GateOperations::h(&mut s, 0);
        GateOperations::cnot(&mut s, 0, 1);
        s
    };
    let counts = state.sample_bitstrings(1024);
    println!("Histogram: {:?}", counts);
}
```

### Using QuantumSimulator (convenience wrapper)

`QuantumSimulator` wraps a `QuantumState` with method-style gate calls and an
optional circuit optimizer pre-pass:

```rust
use nqpu_metal::QuantumSimulator;

fn main() {
    let mut sim = QuantumSimulator::new(4);

    // Build a GHZ state
    sim.h(0);
    sim.cnot(0, 1);
    sim.cnot(1, 2);
    sim.cnot(2, 3);

    let outcome = sim.measure();
    println!("GHZ measurement: {:04b}", outcome);
    // Should be either 0000 or 1111
}
```

`QuantumSimulator` exposes all standard gates as methods: `h`, `x`, `y`, `z`,
`s`, `t`, `rx`, `ry`, `rz`, `cnot`, `cz`, `cphase`, `swap`, `toffoli`, and
`measure_qubit` for mid-circuit measurement.

## Choosing a backend

nQPU auto-selects the best backend for your circuit using the `AutoSimulator`
routing system. You can also target a specific backend:

| Backend | When to use | Qubit range |
|---------|-------------|-------------|
| **CPU (Rayon)** | Default statevector; works everywhere | 1--28 qubits |
| **Metal GPU** | macOS with Apple Silicon or discrete GPU | 1--30+ qubits |
| **MPS (tensor network)** | Low-entanglement circuits | 30--100+ qubits |
| **Stabilizer** | Clifford-only circuits | Thousands of qubits |
| **CUDA GPU** | NVIDIA hardware (Linux/Windows) | 1--30+ qubits |
| **F32 Fused CPU** | Memory-bandwidth constrained workloads | 1--28 qubits |

### Auto-backend selection

The `AutoBackend::select()` analyzer inspects your circuit's gate composition,
entanglement structure, and qubit count to recommend the optimal backend
automatically.  The `AutoSimulator` then executes on the chosen backend
(density matrix, stabilizer, MPS, fused statevector, or Metal GPU).

```rust
use nqpu_metal::auto_backend::{AutoBackend, BackendType};

let backend = AutoBackend::select(&gates);
match backend {
    BackendType::MPS       => { /* tensor network path */ }
    BackendType::MetalGPU  => { /* GPU-accelerated path */ }
    BackendType::Fused     => { /* fused CPU path */ }
    BackendType::CPU       => { /* sequential CPU path */ }
    BackendType::CudaGPU   => { /* NVIDIA GPU path */ }
    _ => {}
}
```

### Enabling Metal GPU manually

Metal GPU is available on macOS builds automatically (the `metal` crate
dependency is target-gated). To use the Metal parallel executor directly:

```rust
#[cfg(target_os = "macos")]
{
    use nqpu_metal::MetalParallelQuantumExecutor;
    // GPU-accelerated gate execution
}
```

## Running the TUI

nQPU ships with an interactive terminal interface that renders Bloch spheres,
probability histograms, circuit diagrams, and performance dashboards:

```bash
cd sdk/rust
cargo run --release --bin nqpu_tui
```

## Running benchmarks

Several benchmark binaries are available for different subsystems:

```bash
# GPU vs CPU comparison
cargo run --release --bin gpu_bench

# General performance benchmarks
cargo run --release --bin perf_bench

# Qubit scaling analysis
cargo run --release --bin qubit_scaling_bench

# Tensor network benchmarks (MPS vs PEPS)
cargo run --release --bin peps_vs_mps_benchmark

# Stabilizer simulator benchmarks
cargo run --release --bin stabilizer_bench

# Criterion benchmark suite
cargo bench
```

## Feature flags

Feature flags control which backends, bindings, and optional modules are
compiled. Unused backends are excluded at compile time with zero runtime cost.

| Feature | Default | Description |
|---------|---------|-------------|
| `parallel` | Yes | Rayon-based multi-threaded gate execution |
| `serde` | Yes | Serialization/deserialization with serde |
| `metal` | No | Metal GPU backend (macOS only) |
| `metal4` | No | Metal 4 tensor ops + inline ML (requires macOS 15+) |
| `amx` | No | Apple AMX accelerated tensor contractions for MPS |
| `cuda` | No | NVIDIA CUDA GPU backend via `cudarc` |
| `rocm` | No | AMD ROCm GPU backend (stub) |
| `python` | No | PyO3 Python bindings |
| `visualization` | No | Plotters/image-based visualization |
| `distributed` | No | MPI-based distributed simulation |
| `web` | No | Axum REST API and web GUI |
| `wasm` | No | WebAssembly/WebGPU backend |
| `qpu` | No | Real QPU hardware connectivity |
| `qpu-ibm` | No | IBM Quantum provider |
| `qpu-braket` | No | Amazon Braket provider |
| `qpu-azure` | No | Azure Quantum provider |
| `qpu-ionq` | No | IonQ direct provider |
| `qpu-google` | No | Google Quantum AI provider |
| `qpu-all` | No | All QPU providers |
| `lsp` | No | OpenQASM language server protocol |
| `chemistry` | No | Quantum chemistry modules |
| `networking` | No | Quantum networking modules |
| `experimental` | No | Research modules (orch_or, creative_quantum) |
| `full` | No | Everything except `experimental` |

Enable features at build time:

```bash
# Metal GPU + visualization
cargo build --release --features metal,visualization

# Full build with all features
cargo build --release --features full

# Python bindings
cargo build --release --features python
```

## Running tests

```bash
# All tests (default features)
cargo test

# Tests with specific features
cargo test --features metal,visualization

# Single test module
cargo test tensor_network
```

Note: dev and test profiles compile dependencies at `opt-level = 2` for
acceptable numeric performance while keeping your code debuggable.

## Next steps

- [Architecture Guide](ARCHITECTURE.md) -- deep dive into the 14 domain
  modules, backend system, and design decisions
- [API Reference](https://docs.rs/nqpu-metal) -- auto-generated Rustdoc
- [README](../README.md) -- project overview and Python SDK quick start
