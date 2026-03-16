# Changelog

All notable changes to the nQPU project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-16

### Added

#### Rust SDK - Core Framework (`sdk/rust/src/`)

- **core**: Foundational quantum types, gate operations (Pauli, Clifford, parametric),
  state models (statevector, density matrix), stabilizer formalism, and computation
  primitives organized across 5 subdirectories (computation, foundations, gates,
  stabilizer, state_models)
- **circuits**: Circuit construction with gate scheduling, depth analysis, synthesis from
  unitary matrices, format conversion (OpenQASM 2.0/3.0, JSON serialization),
  visualization (ASCII and text-based), and noise-aware routing with SABRE heuristics
- **algorithms**: Variational solvers (QAOA with 2328 lines and 61 tests, warm-start
  QAOA, SSVQE with 2544 lines and 47 tests), transform algorithms (Grover search with
  33 tests, HHL linear solver with 22 tests, amplitude estimation with 1404 lines and
  19 tests, quantum walk with 2184 lines and 37 tests), optimization routines, and
  quantum dynamics simulation (2509 lines, 44 tests)
- **backends**: Multi-backend execution layer with Metal GPU acceleration for macOS,
  trapped ion simulator (3900+ lines, 75 tests), superconducting transmon simulator
  (2400+ lines, 40 tests), neutral atom with Rydberg interactions (1473 lines, 46
  tests), pulse-level control, runtime management, and automatic hardware-aware backend
  selection (14 tests)
- **noise**: Noise model framework with Lindblad dynamics simulation, error mitigation
  strategies (ZNE, PEC), and realistic device noise models calibrated to published
  hardware specifications
- **quantum_ml**: ~20K lines with 260+ tests covering QNN layers, QSVM kernels, quantum
  attention mechanisms, parameter-shift gradients, UCCSD/hardware-efficient/tree tensor
  ansatze, neural quantum states (RBM + VMC), quantum transformer architecture, PyTorch
  and JAX bridge layers (standalone, no external deps), and experimental modules for
  Rydberg reservoir computing and microtubule consciousness simulation
- **networking**: Quantum networking protocols (entanglement swapping, purification),
  hardware interfaces, quantum randomness generation, and protocol validation utilities
- **physics**: Quantum biology (photosynthesis energy transfer, avian magnetoreception,
  quantum olfaction, enzyme tunneling), consciousness models (Orch-OR), foundational
  physics, quantum information theory, condensed matter simulation, and quantum transport
- **measurement**: Quantum state characterization (randomized benchmarking, cross-entropy
  benchmarking), tomography protocols, and verification procedures
- **infra**: Automatic differentiation engine, benchmarking harness, Python/C FFI
  bindings, distributed execution over MPI, and research experiment utilities
- **chemistry**: Electronic structure methods (Hartree-Fock, coupled cluster) and quantum
  chemistry applications (molecular ground state energy, reaction pathway optimization)
- **applications**: Creative quantum computing (quantum art generation), quantum-enhanced
  decision making (POMDP solvers), and quantum games (strategic entanglement)
- **error_correction**: Quantum error correcting codes (surface, color, toric), decoders
  (MWPM, union-find), resource estimation tooling (620 lines, 21 tests), and correction
  workflows with syndrome extraction
- **tensor_networks**: MPS architectures, tensor contraction engines with optimal
  ordering, time evolution (TEBD, TDVP), higher-dimensional PEPS networks, and Metal
  GPU-accelerated contraction

#### Rust SDK - Transpiler & Routing

- SABRE routing with 3 heuristic modes (basic, lookahead, decay) in `transpiler.rs`
- Noise-aware routing with decoherence-weighted SWAP scoring in
  `noise_aware_routing.rs`
- Device presets for IBM Eagle/Heron, Google Sycamore, IonQ Aria, Rigetti Aspen

#### Rust SDK - Auto-Backend Selection

- Hardware-aware automatic backend routing in `auto_backend.rs`
- Toffoli-density heuristic: circuits with >10% 3-qubit gates route to neutral atom
- Supports superconducting, trapped ion, and neutral atom backend targets

#### Python SDK - Batch 1: Foundation Packages (`sdk/python/nqpu/`)

- **qkd**: Quantum key distribution with BB84, B92, E91, and decoy-state protocols
  (2076 lines, 69 tests)
- **optimizers**: Classical and quantum optimization algorithms including COBYLA, SPSA,
  Adam, and natural gradient (1420 lines, 71 tests)
- **chem**: Quantum chemistry with VQE driver, molecular Hamiltonians, fermionic
  operators, one/two-electron integrals, and hardware-efficient ansatze (3300 lines,
  84 tests)
- **bio**: Quantum biology simulations covering photosynthesis energy transfer, avian
  magnetoreception, quantum olfaction, DNA mutation via proton tunneling, and enzyme
  tunneling (3174 lines, 129 tests)
- **finance**: Quantum finance with Monte Carlo option pricing, portfolio optimization,
  credit risk analysis, and quantum amplitude estimation (2816 lines, 102 tests)
- **benchmarks**: Cross-backend benchmarking framework with neutral atom support,
  performance profiling, and comparative analysis (1570 lines, 101 tests)

#### Python SDK - Batch 2: Error Mitigation & Characterization

- **mitigation**: Zero-noise extrapolation (ZNE), probabilistic error cancellation
  (PEC), measurement error mitigation, and Clifford data regression (2689 lines,
  105 tests)
- **tomography**: Full quantum state tomography, process tomography, gate set
  tomography, and maximum likelihood estimation (2774 lines, 109 tests)
- **qrng**: Quantum random number generation with entropy extraction, NIST statistical
  testing, and multiple source protocols (3735 lines, 148 tests)
- **error_correction**: Surface codes, Steane codes, repetition codes, syndrome
  decoding, and logical qubit management (3333 lines, 156 tests)

#### Python SDK - Batch 3: Compilation & Simulation

- **qcl**: Quantum circuit learning with parameterized circuits, data encoding
  strategies, gradient computation, and classification/regression tasks (3066 lines,
  141 tests)
- **simulation**: Statevector, density matrix, and stabilizer simulation backends
  with noise injection and measurement sampling (3311 lines, 148 tests)
- **transpiler**: Circuit optimization passes, gate decomposition, qubit mapping,
  routing with coupling-map awareness, and basis gate translation (2762 lines,
  156 tests)

#### Python SDK - Batch 4: Advanced Modules

- **tensor_networks**: Matrix product state (MPS) simulation, tensor contraction,
  DMRG-inspired variational methods, and entanglement analysis (2755 lines, 120 tests)
- **games**: Quantum game theory with Prisoner's Dilemma, coin flipping, Mermin-Peres
  magic square, and quantum strategy optimization (2418 lines, 125 tests)

#### Python SDK - Batch 5: Hardware Integration

- **neutral_atom**: Neutral atom backend with Rydberg gate physics, atom arrangement
  optimization, blockade radius modeling, and multi-qubit entangling operations
  (3160 lines, 94 tests)
- **cross-backend benchmarks**: Comparative performance analysis across trapped ion,
  superconducting, and neutral atom backends with standardized circuit suites

#### Python SDK - Additional Packages

- **metal**: Metal GPU backend bindings for macOS acceleration
- **ion_trap**: Trapped ion backend with Molmer-Sorensen gates and shuttling (2900 lines)
- **superconducting**: Superconducting qubit backend with transmon gate models
- **trading**: Quantum-enhanced trading strategies (4200 lines)
- **physics**: Quantum physics simulations and models

#### Documentation

- `docs/GETTING_STARTED.md`: Quickstart guide for Rust and Python SDKs
- `docs/ARCHITECTURE.md`: System architecture and module dependency overview
- `docs/GPU_ACCELERATION.md`: Metal and CUDA GPU acceleration guide
- `docs/DRUG_DISCOVERY.md`: Quantum chemistry for drug discovery workflows
- `docs/QUANTUM_DOMAINS.md`: Overview of all quantum computing domains covered
- `docs/RUST_SDK.md`: Rust SDK API reference and usage patterns
- `docs/TUI.md`: Terminal user interface documentation
- `docs/NOVEL_OPPORTUNITIES.md`: Research directions and monetization opportunities

#### Project Infrastructure

- `.github/workflows/python-tests.yml`: CI pipeline for Python tests across 3.10-3.12
- `.github/workflows/rust-tests.yml`: CI pipeline for Rust tests with warning-as-error
- `CHANGELOG.md`: Retroactive changelog covering all development batches
- `SECURITY.md`: Vulnerability reporting policy

#### Test Suite

- Rust SDK: 700+ tests across all 14 domain modules, all passing with `-D warnings`
- Python SDK: 1900+ tests across 20 packages with numpy-only dependency
- Cross-backend integration tests validating consistent results across hardware targets
- Noise model regression tests ensuring ideal-mode produces zero error rates

### Fixed

- Circuit optimizer fusion (`test_fusion_rz_rz`) and peephole reduction
  (`test_peephole_reduces`) edge cases in gate combination logic
- Measurement operations on pure quantum states (`test_measure_pure_quantum`)
  now correctly collapse to basis states
- Snake mapping in circuit layout producing incorrect qubit ordering
- All compiler warnings in `bin/` executables cleaned up across 5 binary targets
- Finance module infinite recursion in delta bump-reprice where creating a new pricer
  for sensitivity analysis triggered another delta calculation (added `_compute_delta`
  flag guard)
- SSVQE identity test handling for pure constant Hamiltonians with no qubit operators
  by using explicit `(0, 'I')` identity operator with `num_qubits=0`
- Noise model ideal-mode error rates where all error-rate functions now check the
  enables flag before returning non-zero rates, preventing spurious decoherence
- Integer overflow in large qubit register initialization for `(0..80000)` ranges
  on `Vec<u8>` by casting through `u32` intermediaries
- Type inference breakage after file reorganization resolved with explicit `f64`
  annotations on all ambiguous numeric expressions
- Two-qubit unitary simulation using direct matrix application instead of native gate
  decomposition for higher fidelity

### Changed

- Reorganized Rust SDK from flat file layout to 14 domain directories with categorical
  subdirectories, each domain containing only `mod.rs` plus organized subdirs
- Auto-backend routing updated to include neutral atom target alongside superconducting
  and trapped ion
- `lib.rs` maintains backward compatibility via `pub use domain::*` re-exports

### Removed

- Pre-reorganization `.bak` archive files cleaned up
- Orphaned files from `git stash pop` during reorganization
