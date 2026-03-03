# nQPU-Metal

Quantum simulation toolkit in Rust with Python bindings and Apple Silicon Metal acceleration.

**nQPU-Metal is a simulation SDK.** It does not connect to quantum hardware. It excels at:

- Algorithm development and prototyping
- QEC research with integrated decoders
- Quantum chemistry exploration
- Education and teaching
- Performance-critical simulation on Apple Silicon

## Current Status (2026-02-21)

- Crate: `nqpu-metal` version `1.0.0`
- Python package: `nqpu-metal` version `1.0.0`
- Cargo publish dry-run: passing
- Python wheel build + `twine check` + smoke test: passing

## Features

### Core (Production-Ready)

| Feature | Description |
|---------|-------------|
| Statevector simulator | Full-state simulation, 25-29 qubits (RAM-limited) |
| Stabilizer simulator | Clifford-only circuits, 1000+ qubits, 6.57M gates/sec |
| Metal GPU acceleration | Apple Silicon GPU backend, up to 18.7x speedup |
| MPS tensor network | Matrix Product States, validated to 8192 qubits (low entanglement) |
| ZX-calculus optimizer | Graph-based circuit optimization and simplification |
| MWPM decoder | Full Blossom V implementation for surface code decoding |
| BP decoder | Min-sum belief propagation on Tanner graphs |
| BP-OSD decoder | Belief Propagation + Ordered Statistics Decoding for qLDPC codes |
| Sliding window decoder | Real-time QEC decoding, boundary-aware, 1-6.4M rounds/sec |
| Magic state factory | T-state distillation (15-to-1, 20-to-4, Reed-Muller, Litinski) |

### Functional

| Feature | Description |
|---------|-------------|
| Quantum finance | Option pricing, portfolio optimization via quantum algorithms |
| QKD protocols | BB84, E91 simulation for quantum key distribution |
| Circuit optimizer | Gate cancellation, fusion, commutation analysis |
| Autodiff circuits | Parameter-shift gradients for variational algorithms (standalone, no PyTorch/JAX dependency) |
| ADAPT-VQE | Adaptive variational eigensolver for quantum chemistry (H2, LiH, BeH2) |
| Quantum walks | Continuous + discrete walks on arbitrary graphs, spatial search, mixing time |
| QASM 2.0 parser | Import/export for common OpenQASM 2.0 circuits |
| QASM 3.0 parser | Partial OpenQASM 3.0 subset (common constructs, not full spec) |

### Basic / Work-in-Progress

| Feature | Description |
|---------|-------------|
| PEPS | Basic 2D tensor network data structure (approximate contraction, small systems) |
| TTN | Basic tree tensor network structure (limited long-range gate support) |
| Neural decoder | Proof-of-concept GNN decoder (research/educational only) |

### Experimental (behind `--features experimental`)

| Feature | Description |
|---------|-------------|
| Orch-OR simulation | Penrose-Hameroff microtubule quantum coherence (research exploration) |
| Creative quantum | Novel QRNG approaches using consumer hardware (proof-of-concept) |

## Qubit Capacity

| Backend | Max Qubits | Metric | Condition |
|---------|-----------|--------|-----------|
| Statevector | 25-29 | Full simulation | Limited by RAM (2^n amplitudes) |
| MPS | 8192 | Validated | Low entanglement, bond dim 64 |
| Stabilizer | 1000+ | 6.57M gates/sec | Clifford-only circuits |
| Metal GPU | 20-25 | 18.7x speedup | Statevector on Apple Silicon |

## Benchmarks

### Methodology

All benchmarks measured on Apple M-series Silicon (single machine). Timings are median of multiple runs.

**Stabilizer throughput (6.57 MHz)**: Measures `apply_batch()` gate-level throughput on random Clifford circuits at 1000 qubits. This is gates-per-second for gate-by-gate simulation, *not* Stim-style compiled syndrome sampling (which pre-compiles to Pauli frame tracking). These are different metrics measuring different things.

**Metal GPU speedup (18.7x)**: Grover-14 circuit, GPU vs best CPU path on same hardware.

### Rust backend

- Metal GPU speedup: **18.7x** (Grover-14)
- Stabilizer at 1000 qubits: **6.57M Clifford gates/sec** (gate-level simulation)
- QFT-15: CPU 9.236 ms, fused 5.175 ms, GPU 0.620 ms

### Python competitor benchmarks (same machine)

Bell circuit (2000 shots, median):

- `nqpu_metal`: **0.057 ms**
- `qiskit` 2.3.0 + Aer 0.17.2: 1.370 ms
- `cirq` 1.6.1: 2.643 ms
- `pennylane` 0.44.0: 1.499 ms
- `qutip` 5.2.3: 0.333 ms

QFT-10 (median):

- `nqpu_metal`: **2.089 ms**
- `qiskit`: 1.425 ms
- `cirq`: 1.770 ms

Note: Python-layer microbenchmarks vary across runs due to framework startup/caching effects; use the latest report artifacts for publish claims.

### Reproducing benchmarks

```bash
. .venv_compare/bin/activate
python scripts/release_benchmark_suite.py       # reuse Rust logs
python scripts/release_benchmark_suite.py --run-rust  # re-run everything
```

Outputs: `reports/release/final_release_benchmarks_latest.json`, `reports/release/FINAL_BENCHMARK_REPORT_latest.md`

## Install

### Rust

```toml
[dependencies]
nqpu-metal = "1.0"
```

With Metal GPU acceleration (macOS only):

```toml
[dependencies]
nqpu-metal = { version = "1.0", features = ["metal"] }
```

### Python

```bash
pip install nqpu-metal
```

## Quick Usage

### Rust

```rust
use nqpu_metal::{QuantumState, GateOperations};

let mut state = QuantumState::new(2);
GateOperations::h(&mut state, 0);
GateOperations::cnot(&mut state, 0, 1);
let probs = state.probabilities();
```

### Python

```python
import nqpu_metal

sim = nqpu_metal.QuantumSimulator(2)
sim.h(0)
sim.cx(0, 1)
print(sim.probabilities())
```

## Release / Publish

```bash
scripts/release.sh
```

Live publish:

```bash
export CARGO_REGISTRY_TOKEN="..."
export MATURIN_PYPI_TOKEN="..."
scripts/release.sh --publish-crates --publish-pypi
```

Full guide: `docs/RELEASE_PUBLISHING.md`
