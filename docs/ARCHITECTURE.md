# nQPU Architecture Guide

This document describes the internal architecture of the nQPU quantum computing
SDK: how the Rust core and Python SDK fit together, how backends are selected
and dispatched, and the key design decisions that shape the system.

## High-level structure

nQPU is a two-layer system. The performance-critical core is a Rust crate
(`nqpu-metal`) organized into 14 domain modules. On top of that sits a pure-Python
SDK (`sdk/python/nqpu/`) with 22 subpackages covering applied quantum computing
workflows. Both layers share the same design philosophy: self-contained modules,
numpy-compatible data types, and no mandatory heavy dependencies.

```
nqpu-metal (Rust crate -- sdk/rust/src/)
 |
 |-- Core data types: QuantumState, GateOperations, QuantumSimulator
 |
 +-- 14 domain modules ─────────────────────────────────
 |   core/              Quantum primitives (stabilizer, density matrix, CV)
 |   tensor_networks/   MPS, PEPS, MERA, DMRG, TEBD, GPU-accelerated
 |   error_correction/  QEC codes, decoders, magic state distillation
 |   noise/             Noise models, Lindblad dynamics, error mitigation
 |   algorithms/        VQE, QAOA, QPE, Grover, Shor, HHL, AE, QWalk
 |   quantum_ml/        Kernels, transformers, NQS, JAX/PyTorch bridges
 |   chemistry/         Molecular simulation, drug design, materials
 |   backends/          Metal, CUDA, ROCm, auto-select, pulse control
 |   circuits/          Optimizer, transpiler, ZX-calculus, QASM/QIR
 |   networking/        QKD, QRNG, entropy, PQC assessment
 |   physics/           Walks, topology, thermodynamics, quantum biology
 |   applications/      Finance, logistics, games, NLP, generative art
 |   measurement/       Tomography, QCVV, classical shadows
 |   infra/             Traits, SIMD ops, autodiff, benchmarks, FFI
 |
 +-- Standalone modules ─────────────────────────────────
     decoding/          Classical decoding utilities
     qpu/               Real QPU hardware (feature-gated)
     web/               REST API and web GUI (feature-gated)

Python SDK (sdk/python/nqpu/) ──────────────────────────
 |
 +-- Foundation ─────────────────────────────────────────
 |   core/              Quantum primitives and shared utilities
 |   metal/             Metal GPU bindings (macOS)
 |   physics/           Physics research tools
 |
 +-- Hardware backends ──────────────────────────────────
 |   ion_trap/          Trapped-ion backend (digital->analog->atomic)
 |   superconducting/   Transmon backend with pulse-level control
 |   neutral_atom/      Neutral-atom Rydberg blockade backend
 |   benchmarks/        Cross-backend performance benchmarks
 |
 +-- Algorithms & circuit tools ─────────────────────────
 |   optimizers/        Variational optimizers (SPSA, Adam, PSR, VQE)
 |   transpiler/        SABRE routing, gate cancellation, KAK decomposition
 |   simulation/        Hamiltonian dynamics (Trotter, QITE, VarQTE)
 |   tensor_networks/   MPS, MPO, DMRG, TEBD
 |   qcl/               Quantum circuit learning, QSVM, kernel methods
 |
 +-- Error handling ─────────────────────────────────────
 |   error_correction/  Surface/color codes, MWPM, union-find, lattice surgery
 |   mitigation/        ZNE, PEC, CDR, Pauli twirling, readout correction
 |
 +-- Measurement & verification ─────────────────────────
 |   tomography/        State/process/shadow tomography, fidelity, QCVV
 |   qrng/              Quantum random number generation, NIST tests
 |
 +-- Domain applications ────────────────────────────────
 |   chem/              Quantum chemistry (VQE, molecular Hamiltonians)
 |   bio/               Quantum biology (photosynthesis, tunneling, olfaction)
 |   finance/           Option pricing, portfolio optimization, risk analysis
 |   trading/           Quantum-enhanced trading strategies and backtesting
 |   qkd/               Quantum key distribution (BB84, E91, B92, decoy-state)
 |   games/             Quantum game theory, combinatorial optimization
```

## Core data types

Three types in `lib.rs` form the foundation of every simulation.

### QuantumState

A statevector simulator storing `2^n` complex amplitudes in a `Vec<Complex64>`.

```rust
pub struct QuantumState {
    amplitudes: Vec<C64>,    // 2^num_qubits complex amplitudes
    pub num_qubits: usize,
    pub dim: usize,          // 2^num_qubits (cached)
}
```

Key operations:
- `new(n)` -- initialize to |0...0>
- `probabilities()` -- squared magnitudes of all amplitudes
- `measure()` -- single-shot measurement (weighted random sample)
- `sample(n_shots)` -- multi-shot sampling with CDF binary search
- `sample_bitstrings(n_shots)` -- measurement histogram as bitstrings
- `fidelity(other)` -- state overlap |<psi|phi>|^2
- `expectation_z(qubit)`, `expectation_x(qubit)`, `expectation_y(qubit)` --
  single-qubit Pauli expectations
- `expectation_pauli_string(ops)` -- arbitrary tensor-product Pauli observable
- `expectation_hamiltonian(terms)` -- weighted sum of Pauli strings

Amplitudes are stored as `num_complex::Complex64` (double precision).  A
parallel `C32` (single-precision) path exists for GPU and memory-constrained
workloads.

### GateOperations

A zero-sized struct (`pub struct GateOperations;`) whose associated functions
apply unitary gates to a `QuantumState` in-place.  Each gate implementation
uses a two-tier parallelism strategy:

1. **Low-stride qubits** (qubit index < 12): `par_chunks_mut` over contiguous
   amplitude blocks, which preserves cache locality.
2. **High-stride qubits** (qubit index >= 12): indexed parallel iteration
   (`par_bridge`) over `dim/2` independent (i, j) pairs, avoiding the
   degenerate case where chunk parallelism produces too few work items.

SIMD kernels (`simd_ops::SimdMatrix2x2`) accelerate the inner loops on x86_64
and aarch64.  The threshold between the two paths is a compile-time constant
(`HIGH_STRIDE_THRESHOLD = 4096`).

Available gates: H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CZ, CPhase, SWAP,
Toffoli, and arbitrary 2x2 / 4x4 unitaries.

### QuantumSimulator

A convenience wrapper that pairs a `QuantumState` with an optional
`CircuitOptimizer` pre-pass.  Provides method-style gate calls (`sim.h(0)`)
and mid-circuit measurement (`sim.measure_qubit(q)`).

```rust
pub struct QuantumSimulator {
    pub state: QuantumState,
    optimizer: Option<CircuitOptimizer>,
}
```

## The 14 Rust domain modules

### 1. core/ -- Quantum primitives

State representations beyond the dense statevector: density matrices
(`DensityMatrixSimulator`), stabilizer tableaux (multiple implementations
including SIMD and AVX-512 variants), Clifford+T decomposition, Pauli algebra,
quantum channels (Kraus, Choi), entanglement analysis, mid-circuit
measurement, measurement-based quantum computation (MBQC), continuous-variable
(CV) Gaussian states, decision diagrams, matchgate simulation, and
fermionic Gaussian states.

Notable submodules:
- `stabilizer`, `fast_stabilizer`, `optimized_stabilizer`, `simd_stabilizer`,
  `avx512_stabilizer` -- a progression of stabilizer implementations from
  naive to highly optimized
- `stabilizer_router` -- automatic selection of the fastest stabilizer backend
- `cv_quantum` -- continuous-variable Gaussian boson sampling
- `decision_diagram` -- BDD/ADD representations for certain circuit families

### 2. tensor_networks/ -- Tensor network methods

Matrix Product State (MPS), Projected Entangled Pair State (PEPS), MERA,
tree tensor networks (TTN), DMRG, TDVP, TEBD, and advanced contraction
strategies.

Notable submodules:
- `tensor_network` -- core MPS simulator for 100+ qubit circuits with bounded
  entanglement
- `adaptive_mps` -- dynamically adjusts bond dimension during simulation
- `peps`, `peps_gates`, `peps_simulator` -- 2D tensor network for lattice
  systems
- `contraction_optimizer` -- finds optimal contraction orderings
- `dmrg_tdvp` -- ground-state search and real-time evolution
- `gpu_mps`, `metal_mps`, `dmrg_metal` -- Metal GPU-accelerated tensor
  contractions (macOS only)
- `camps` -- compressed approximate MPS
- `mera_happy` -- MERA with holographic error correction connections
- `simulation_3d`, `lattice_mps_4d` -- higher-dimensional tensor networks

### 3. error_correction/ -- QEC codes and decoders

A comprehensive error correction pipeline: code construction, syndrome
extraction, decoding, and magic state distillation.

**Codes:**
- Surface codes (standard, XZZX, dynamic, yoked)
- Floquet codes (standard, hyperbolic)
- Bosonic codes, cat qubit concatenation
- QLDPC (quantum low-density parity check), bivariate bicycle, trivariate
- Holographic codes
- Approximate dynamical QEC

**Decoders:**
- MWPM (minimum weight perfect matching) with GPU acceleration
- BP-OSD (belief propagation + ordered statistics decoding)
- Relay BP, MBBP-LD
- Neural decoders (standard, unified, transformer-based, Mamba-based)
- Metal-accelerated neural decoder
- Sliding window decoder for real-time decoding
- Adaptive real-time decoder
- Differentiable QEC (for decoder training)

**Tooling:**
- `magic_state_factory` -- magic state distillation for fault-tolerant T gates
- `decoder_aware_transpiler` -- transpilation that accounts for decoder
  structure
- `qec_interop` -- Stim-compatible detector model import/export
- `bulk_qec_sampling` -- fast Monte Carlo sampling for threshold estimation

### 4. noise/ -- Noise models and error mitigation

Realistic noise modeling and techniques to mitigate its effects.

**Noise models:**
- `noise_models`, `advanced_noise` -- depolarizing, amplitude damping, phase
  damping, correlated noise, thermal relaxation
- `device_noise` -- device-calibrated noise from hardware data
- `bayesian_noise` -- Bayesian noise characterization
- `non_markovian` -- memory effects in open quantum systems
- `leakage_simulation` -- qubit leakage to non-computational levels
- `lindblad`, `lindblad_shadows` -- Lindbladian master equation dynamics
- `process_tensor` -- process tensor formalism for non-Markovian dynamics

**Mitigation:**
- `error_mitigation`, `advanced_error_mitigation` -- general mitigation
- `enhanced_zne` -- zero-noise extrapolation with multiple folding strategies
- `pec` -- probabilistic error cancellation
- `compilation_informed_pec` -- PEC that leverages transpiler information
- `pna` -- probabilistic noise amplification
- `pauli_twirling` -- symmetrization of noise channels
- `dynamical_decoupling` -- pulse-level error suppression

### 5. algorithms/ -- Quantum algorithms

**Variational:**
- VQE (standard, ADAPT-VQE, GGA-VQE, Meta-VQE)
- QAOA (standard, warm-start)
- QAMOO (multi-objective optimization)

**Phase estimation:**
- QPE (standard, Heisenberg-limited)
- QFT (standard, 2D)

**Simulation:**
- Trotter-Suzuki (improved), QSwift
- Schrodinger-Feynman hybrid
- Low-depth UCC, Tucker state preparation
- Symmetry-adapted simulation

**Other:**
- Shor's algorithm
- QSP/QSVT (quantum signal processing / singular value transformation)
- Quantum annealing
- QRAM, QUBO encoder
- Pauli propagation (CPU and GPU-accelerated)
- SQD (selected quantum dynamics)

### 6. quantum_ml/ -- Quantum machine learning

Quantum kernels, variational quantum neural networks, quantum natural
gradient, neural quantum states, quantum reservoir computing (including
Rydberg atom reservoirs), barren plateau analysis, and quantum transformers
with multi-head attention.  Bridges to JAX and PyTorch for hybrid
classical-quantum training.

### 7. chemistry/ -- Quantum chemistry

Molecular integral computation, double-factorized Hamiltonians, CAMPS+DMRG
chemistry solver, quantum drug design (molecular fingerprinting, ADMET
prediction), and quantum materials simulation.

### 8. backends/ -- Hardware backends

The execution backend layer. Each backend implements quantum gate dispatch for
a specific hardware target.

**GPU backends:**
- `metal_backend`, `metal_gpu_full`, `metal_gpu_fixed`,
  `metal_parallel_quantum` -- Metal GPU pipeline (macOS)
- `metal4_backend` -- Metal 4 tensor operations (macOS 15+)
- `m4_pro_optimization` -- M4 Pro-specific tuning
- `amx_tensor`, `tensor_ops` -- Apple AMX accelerated tensor contractions
- `cuda_backend` -- NVIDIA CUDA via cudarc (feature-gated)
- `rocm_backend` -- AMD ROCm (feature-gated, stub)

**Automatic selection:**
- `auto_backend` -- rule-based backend recommendation from circuit analysis
- `auto_simulator` -- full routing engine (density matrix, stabilizer, MPS,
  fused, GPU) with circuit symmetry detection
- `auto_tuning` -- runtime tuning of backend parameters

**Infrastructure:**
- `uma_dispatch`, `concurrent_uma` -- Unified Memory Architecture dispatch for
  Apple Silicon (zero-copy CPU/GPU sharing)
- `cache_blocking` -- cache-aware amplitude blocking
- `mixed_precision`, `f32_fusion` -- single-precision execution paths
- `gpu_memory_pool` -- buffer pooling and reuse for GPU backends
- `hardware_calibration`, `live_calibration` -- calibration data management
- `thermal_scheduler` -- thermal-aware scheduling on mobile/laptop GPUs
- `pulse_control`, `pulse_level`, `pulse_simulation` -- pulse-level control
- `ibm_quantum`, `google_quantum` -- cloud QPU provider clients
- `neutral_atom_array`, `photonic_advantage` -- emerging hardware models
- `pinnacle_architecture` -- architecture-specific optimization

### 9. circuits/ -- Circuit tools

Circuit representation, optimization, and interchange formats.

- `circuit_optimizer` -- multi-level optimization (gate cancellation, fusion,
  commutation)
- `gate_fusion` -- adjacent gate fusion into single unitaries
- `transpiler` -- gate set transpilation to hardware-native gates
- `ai_transpiler` -- ML-assisted transpilation
- `zx_calculus` -- ZX-calculus graph rewriting
- `treespilation` -- tree-decomposition-based transpilation
- `shaded_lightcones` -- lightcone-based circuit slicing
- `ft_compilation` -- fault-tolerant compilation
- `quantum_synthesis` -- Solovay-Kitaev decomposition
- `parametric_circuits` -- parameterized circuit templates
- `qasm`, `qasm3` -- OpenQASM 2.0 / 3.0 import/export
- `qir` -- Quantum Intermediate Representation support
- `circuit_cutting` -- automatic circuit partitioning for distributed execution
- `circuit_cache` -- JIT-style circuit caching (aliased as `jit_compiler`)
- `ascii_viz` -- terminal-based circuit diagrams and Bloch sphere rendering
- `visualization` -- plotters-based graphical output (feature-gated)

### 10. networking/ -- Quantum networking

QKD protocols, metropolitan QKD network simulation, QRNG (quantum random
number generation with multiple entropy sources and extraction methods),
certified randomness, NIST statistical test suite, PQC (post-quantum
cryptography) assessment, quantum network OS, and hardware quantum entropy
sources (camera, wireless, SSD-based).

### 11. physics/ -- Quantum physics

Quantum walks (discrete and continuous), cellular automata, thermodynamics and
quantum batteries, quantum cloning machines, quantum chaos and echoes, quantum
Darwinism, many-worlds analysis, Hayden-Preskill protocol, contextuality,
closed timelike curve simulation, topological quantum computing (Fibonacci
anyons, Majorana fermions, string-net condensation), quantum biology, and
integrated information theory (IIT).

Experimental modules (behind the `experimental` feature flag): Orch-OR
consciousness model and microtubule quantum effects.

### 12. applications/ -- Domain applications

Quantum finance (portfolio optimization, option pricing), logistics
(vehicle routing, scheduling), climate modeling, quantum games (chess, poker,
game theory), quantum NLP, quantum generative art, quantum cognition, and
selfish routing / network equilibria.

### 13. measurement/ -- Quantum measurement

State tomography and process tomography, classical shadows protocol, quantum
Fisher information, general quantum measurement formalism, quantum source
certification, QCVV (quantum characterization, verification, and validation),
layer fidelity benchmarking, and property testing.

### 14. infra/ -- Infrastructure

Cross-cutting concerns shared by all domain modules.

- `traits` -- `QuantumBackend`, `StateVectorBackend`, `ErrorModel`,
  `TensorContractor` trait definitions
- `simd_ops` -- SIMD-accelerated 2x2 unitary and diagonal kernels
- `autodiff`, `adjoint_diff` -- automatic differentiation for variational
  circuits
- `distributed_mpi`, `distributed_metal_mpi`, `distributed_adjoint` --
  distributed computing primitives
- `parallel_quantum`, `parallel_feedforward` -- massively parallel transformer
  execution
- `adaptive_batching`, `shot_batching` -- dynamic workload batching
- `advanced_cache`, `state_checkpoint` -- circuit result caching and state
  snapshots
- `resource_estimation` -- qubit and gate count resource estimation
- `benchmark_suite`, `comprehensive_benchmarks`, `max_qubit_benchmark`,
  `willow_benchmark` -- benchmarking infrastructure
- `tui` -- terminal user interface implementation
- `python`, `python_api_v2` -- PyO3 Python bindings (feature-gated)
- `c_ffi` -- C foreign function interface
- `wasm_backend`, `wasm_bindings` -- WebAssembly backend (feature-gated)

## The 22 Python SDK subpackages

The Python SDK provides accessible, research-friendly interfaces for applied
quantum computing. Every package is pure Python with numpy as the only required
dependency. Packages use dataclasses for structured results and numpy arrays
for quantum states.

| Package | Purpose | Key classes |
|---------|---------|-------------|
| `core` | Quantum primitives and shared utilities | -- |
| `metal` | Metal GPU bindings (macOS) | -- |
| `physics` | Physics research tools | -- |
| `ion_trap` | Trapped-ion backend (digital, analog, atomic layers) | `IonTrapBackend` |
| `superconducting` | Transmon backend with pulse-level control | `SuperconductingBackend` |
| `neutral_atom` | Neutral-atom Rydberg blockade backend | `NeutralAtomBackend` |
| `benchmarks` | Cross-backend performance comparison | `CrossBackendBenchmark`, `BackendComparison` |
| `optimizers` | Variational optimizers for VQE/QAOA | `SPSA`, `Adam`, `VQEOptimizer`, `NaturalGradient` |
| `transpiler` | Circuit routing and basis decomposition | `SABRERouter`, `QuantumCircuit`, `CouplingMap` |
| `simulation` | Hamiltonian dynamics and time evolution | `TrotterEvolution`, `QITE`, `VarQTE` |
| `tensor_networks` | Tensor network methods for 1-D systems | `MPS`, `MPO`, `DMRG`, `TEBD` |
| `qcl` | Quantum circuit learning and kernel methods | `QCLTrainer`, `QSVM`, `QuantumKernel` |
| `error_correction` | Stabilizer codes and syndrome decoders | `SurfaceCode`, `MWPMDecoder`, `UnionFindDecoder` |
| `mitigation` | Error mitigation for near-term hardware | `ZNEEstimator`, `PECEstimator`, `CDREstimator` |
| `tomography` | State reconstruction and verification | `StateTomographer`, `ClassicalShadow` |
| `qrng` | Quantum random number generation | `MeasurementQRNG`, `RandomnessReport`, `CHSHCertifier` |
| `chem` | Quantum chemistry (VQE, molecular Hamiltonians) | `MolecularVQE`, `UCCSD`, `FermionicHamiltonian` |
| `bio` | Quantum biology simulation | `FMOComplex`, `EnzymeTunneling`, `RadicalPair` |
| `finance` | Option pricing, portfolio optimization, risk | `QuantumOptionPricer`, `PortfolioOptimizer`, `RiskAnalyzer` |
| `trading` | Quantum-enhanced trading strategies | `QuantumVolatilitySurface`, `QuantumRegimeDetector` |
| `qkd` | Quantum key distribution protocols | `BB84Protocol`, `E91Protocol`, `QKDNetwork` |
| `games` | Quantum game theory and combinatorial optimization | `PrisonersDilemma`, `MaxCut`, `QuantumBayesian` |

## Backend selection pipeline

When you execute a circuit through `AutoSimulator`, the system runs a
multi-stage analysis pipeline to choose the optimal execution path:

```
Circuit (list of Gates)
       |
       v
  +-----------------+
  | CircuitAnalysis  |  Counts gate types, estimates entanglement,
  |                  |  detects Clifford fraction, identifies symmetry
  +-----------------+
       |
       v
  +-----------------+
  | Backend Routing  |  Rule-based selection using RoutingConfig:
  |                  |
  |  Clifford-only?  |---> StabilizerSimulator (any qubit count)
  |  Near-Clifford?  |---> NearCliffordSimulator (few T gates)
  |  Noisy + <=13q?  |---> DensityMatrixSimulator
  |  >25q + low ent? |---> MPSSimulator (tensor network)
  |  Metal available? |---> MetalSimulator (GPU)
  |  Otherwise        |---> Fused CPU statevector
  +-----------------+
       |
       v
  +-----------------+
  | CircuitOptimizer |  Gate cancellation, commutation, fusion
  +-----------------+
       |
       v
  +-----------------+
  | Gate Fusion      |  Merge adjacent 1q/2q gates into single unitaries
  +-----------------+
       |
       v
  +-----------------+
  | Execution        |  Run on selected backend
  +-----------------+
       |
       v
  QuantumState (result)
```

## Tensor network backends

For circuits exceeding statevector memory limits, nQPU provides tensor network
simulation:

**MPS (Matrix Product State):** Represents the state as a chain of tensors,
one per qubit, connected by bond indices of bounded dimension `chi`.  Efficient
for 1D systems and circuits with bounded entanglement.  Scales to 100+ qubits
when entanglement entropy is low.  The `adaptive_mps` variant dynamically
grows and shrinks bond dimension based on measured entanglement during
simulation.

**PEPS (Projected Entangled Pair States):** Generalizes MPS to 2D lattices.
Each tensor has up to four bond indices (up, down, left, right).  Contraction
is #P-hard in general, but approximate contraction via boundary MPS or CTM
(corner transfer matrix) methods provides practical accuracy for lattice
problems.

**MERA (Multi-scale Entanglement Renormalization Ansatz):** Hierarchical
tensor network with disentanglers and isometries at each scale.  Natural for
critical systems and CFT ground states.  The `mera_happy` module connects MERA
to holographic quantum error correction.

**TTN (Tree Tensor Networks):** Tree-structured tensor network for systems
with hierarchical entanglement patterns.

All tensor network backends have Metal GPU-accelerated variants on macOS
(`gpu_mps`, `metal_mps`, `dmrg_metal`) for accelerating the tensor
contraction inner loops.

## Error correction pipeline

The QEC subsystem follows a layered architecture:

```
Physical qubits
       |
       v
  +------------------+
  | Code Construction |  Surface, Floquet, QLDPC, bosonic, ...
  | (syndrome map)    |
  +------------------+
       |
       v
  +------------------+
  | Syndrome Extract  |  Measure stabilizers, build detector model
  +------------------+
       |
       v
  +------------------+
  | Decoder           |  MWPM, BP-OSD, neural, sliding-window, ...
  | (error estimate)  |
  +------------------+
       |
       v
  +------------------+
  | Correction        |  Apply Pauli frame updates
  +------------------+
       |
       v
  +------------------+
  | Magic State       |  Distillation factories for non-Clifford gates
  | Distillation      |
  +------------------+
       |
       v
  Logical qubits (fault-tolerant computation)
```

The `qec_interop` module provides Stim-compatible detector model import/export,
enabling interoperability with the broader QEC ecosystem.  The
`decoder_aware_transpiler` feeds decoder structure back into compilation to
reduce logical error rates.

## Circuit optimization pipeline

```
User circuit
       |
       v
  +-------------------+
  | CircuitOptimizer   |  Multi-level optimization:
  |                    |  - Trivial gate elimination (XX = I)
  |                    |  - Gate commutation and reordering
  |                    |  - Template matching
  +-------------------+
       |
       v
  +-------------------+
  | Transpiler         |  Map to hardware-native gate set
  |                    |  (e.g., {Rz, SX, CNOT} for IBM)
  +-------------------+
       |
       v
  +-------------------+
  | Gate Fusion        |  Merge adjacent 1q gates into single U3
  |                    |  Merge adjacent 2q gates into single 4x4
  +-------------------+
       |
       v
  Optimized circuit --> Backend execution
```

The `zx_calculus` module provides an alternative optimization path using
ZX-diagram graph rewriting, which can find simplifications invisible to
gate-level peephole optimization.  The `ai_transpiler` uses learned heuristics
to improve transpilation decisions.

## Module dependency flow

```
                       infra/traits
                      (QuantumBackend)
                           |
              +------------+------------+
              |            |            |
           core/        backends/    circuits/
        (QuantumState,  (Metal,     (optimizer,
         gates,         CUDA,       transpiler,
         stabilizer)    auto-sel)   QASM)
              |            |            |
              +------+-----+-----+-----+
                     |           |
              +------+-----+    |
              |            |    |
        algorithms/   tensor_   error_
        (VQE, QAOA,   networks/ correction/
         QPE, Shor)   (MPS,     (surface,
              |        PEPS,     decoders,
              |        DMRG)     magic)
              |            |         |
        +-----+-----+-----+---------+
        |           |           |
   quantum_ml/  chemistry/  noise/
   (kernels,    (molecular, (noise models,
    NQS,        drug,       mitigation,
    transformers) materials)  Lindblad)
        |           |           |
        +-----+-----+-----+----+
              |           |
         physics/    networking/
         (walks,     (QKD, QRNG,
          topology,   entropy,
          thermo)     PQC)
              |           |
              +-----+-----+
                    |
             applications/    measurement/
             (finance,        (tomography,
              logistics,       shadows,
              games)           QCVV)
```

## Feature flags and conditional compilation

The crate uses Cargo feature flags extensively to keep the default build lean
while allowing full-featured builds.  Here is how conditional compilation
works:

**Target-gated dependencies** (always available on the target platform):
```toml
[target.'cfg(target_os = "macos")'.dependencies]
metal = "0.27"     # Metal GPU crate
objc = "0.2"       # Objective-C FFI
cocoa = "0.25"     # macOS windowing
```

Metal GPU modules use `#[cfg(target_os = "macos")]` guards.  On non-macOS
platforms, these modules are excluded entirely -- no stub code, no runtime
checks.

**Feature-gated dependencies** (opt-in at build time):
```toml
cuda = ["cudarc"]                    # NVIDIA GPU
python = ["pyo3", "pyo3-build-config", ...]  # Python bindings
visualization = ["plotters", "image"]         # Graphical output
distributed = ["mpi"]                         # MPI parallelism
web = ["axum", "tokio", "tower-http", ...]   # REST API
wasm = ["wasm-bindgen", "getrandom"]          # WebAssembly
```

**Feature composition:**
- `all-gpus` = `metal` + `cuda` + `rocm`
- `qpu-all` = all five QPU provider features
- `full` = everything except `experimental`

**Default features:** `parallel` (Rayon) and `serde` (serialization).  A bare
`cargo build` produces a multi-threaded CPU statevector simulator with
serialization support.

## Key design decisions

### SIMD-accelerated gate application

Every single-qubit gate dispatches through `simd_ops::SimdMatrix2x2`, a
struct holding the real and imaginary parts of a 2x2 unitary as separate
scalars.  The SIMD kernels process amplitude pairs using platform-native
vector instructions.  This avoids the overhead of `Complex64` multiply-add
sequences and keeps the hot loop free of branch mispredictions.

### Parallel gate enumeration (two-tier strategy)

A single-qubit gate on qubit `q` pairs amplitudes at indices `(i, i + 2^q)`.
For low `q`, these pairs are close together in memory and
`par_chunks_mut(stride * 2)` produces many chunks with good locality.  For
high `q` (stride > 4096), `par_chunks_mut` produces only a few large chunks,
under-utilizing parallelism.  The system switches to a `par_bridge` indexed
strategy that creates `dim/2` independent tasks, giving Rayon full freedom to
work-steal.

The same two-tier pattern applies to two-qubit gates via the
`insert_zero_bits` helper, which maps compact indices in `[0, dim/4)` to full
state indices where specified bit positions are zero.

### UMA dispatch on Apple Silicon

Apple Silicon's Unified Memory Architecture (UMA) allows CPU and GPU to share
the same physical memory without explicit copies.  The `uma_dispatch` and
`concurrent_uma` modules exploit this to:

1. Decide at runtime whether a gate is faster on CPU or GPU based on qubit
   count and gate type.
2. Execute CPU and GPU work concurrently on different qubits of the same state
   vector, since both processors see the same memory.
3. Avoid the PCIe transfer bottleneck that limits discrete GPU performance on
   small circuits.

### Auto-backend routing

Rather than requiring users to choose a backend, the `auto_simulator` module
analyzes every circuit before execution.  The analysis extracts:

- Gate type distribution (Clifford fraction, T-gate count)
- Entanglement structure (max entanglement width, connected components)
- Circuit symmetry (translational, reflection, permutation)
- Magic level (non-stabilizerness estimate)

These properties feed into a configurable `RoutingConfig` that maps circuits
to the fastest backend.  The routing is deterministic and auditable -- the
`CircuitAnalysis` struct captures the reasoning for every decision.

### Stabilizer fast paths

Clifford circuits (composed entirely of H, S, CNOT, and Pauli gates) can be
simulated in polynomial time using the stabilizer formalism.  nQPU provides
five stabilizer implementations at increasing optimization levels:

1. `stabilizer` -- reference implementation, clear and correct
2. `fast_stabilizer` -- optimized tableau operations
3. `optimized_stabilizer` -- bitwise tableau packing
4. `simd_stabilizer` -- SIMD-accelerated row operations
5. `avx512_stabilizer` -- AVX-512 specialization for server CPUs

The `stabilizer_router` automatically selects the fastest available
implementation based on CPU feature detection.

### Profile optimization for tests

Numeric code (ndarray, nalgebra, num-complex) is 10--50x slower at
`opt-level = 0`.  The Cargo profiles set `opt-level = 2` for all dependencies
in dev and test builds, keeping your own code debuggable while ensuring tests
complete in reasonable time:

```toml
[profile.dev.package."*"]
opt-level = 2

[profile.test.package."*"]
opt-level = 2
```

## Further reading

- [Getting Started](GETTING_STARTED.md) -- build, run, and write your first
  circuit
- [Python SDK Guide](PYTHON_SDK.md) -- installation, quick start, and API
  patterns for the Python SDK
- [README](../README.md) -- project overview
- [Quantum Domains](QUANTUM_DOMAINS.md) -- educational deep-dive into each
  Rust domain module
- [Cargo.toml](../sdk/rust/Cargo.toml) -- full dependency and feature list
- [lib.rs](../sdk/rust/src/lib.rs) -- crate root with all re-exports
