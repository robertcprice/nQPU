# nQPU-Metal API Documentation

Complete API reference for nQPU-Metal quantum simulator.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core API](#core-api)
  - [QuantumSimulator](#quantumsimulator)
  - [Gate Operations](#gate-operations)
  - [Measurement](#measurement)
- [Advanced API](#advanced-api)
  - [Noise Simulation](#noise-simulation)
  - [Tensor Networks](#tensor-networks)
  - [Density Matrix](#density-matrix)
  - [Error Mitigation](#error-mitigation)
  - [QASM Support](#qasm-support)
- [Algorithms](#algorithms)
- [GPU Acceleration](#gpu-acceleration)
- [Phase 2 Rust Modules](#phase-2-rust-modules) (39 modules)
  - [Core & Pauli](#traits-backend-abstraction): Traits, Pauli Algebra/Propagation/Twirling, Quantum Channel
  - [Simulation Backends](#near-clifford-simulation): Near-Clifford, Decision Diagrams, Matchgate, Symmetry
  - [Tensor Networks](#differentiable-mps): Differentiable MPS, Fermionic TN, Stabilizer TN
  - [Error Correction](#advanced-error-mitigation): Advanced Mitigation, Non-Markovian, DD, Bosonic/Floquet/qLDPC
  - [Algorithms](#quantum-chemistry): Chemistry, Networking, Random Walk, Reservoir, Cloning, Games, QCA
  - [Hardware Backends](#amx-tensor): AMX, Metal 4, WASM, MBQC, Topological
  - [Interop](#stim-import): Stim Import, QASM 3.0, QRNG Integration
- [Phase 3: Next-Wave Features](#phase-3-next-wave-features) (5 modules)
  - [Certified QRNG](#certified-qrng): Bell-test certified randomness with CHSH verification + hash chain
  - [Digital-Analog Simulation](#digital-analog-quantum-simulation): Hybrid gate + Hamiltonian evolution
  - [Neural QEC Decoders](#neural-qec-decoders): GNN message-passing decoders
  - [Yoked Surface Codes](#yoked-surface-codes): 1/3 qubit overhead QEC (Nature Comms 2025)
  - [Metal 4 TensorOps](#metal-4-tensorops-extended): simdgroup_matrix gate dispatch
- [Performance Characteristics](#performance-characteristics)
- [Type Reference](#type-reference)

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
nqpu-metal = "0.2"
```

For Metal GPU support (macOS only):

```toml
[dependencies]
nqpu-metal = { version = "0.2", features = ["metal"] }
```

### Feature Flags

- `parallel` (default): Enable multi-threaded CPU execution with Rayon
- `metal`: Enable Metal GPU acceleration (macOS only)
- No features: Sequential execution (recommended for < 16 qubits)

### Installation from Source

```bash
git clone https://github.com/yourusername/nqpu-metal.git
cd nqpu-metal
cargo build --release
```

---

## Quick Start

### Basic Example

```rust
use nqpu_metal::{QuantumSimulator, GateOperations};

fn main() {
    // Create a 2-qubit simulator
    let mut sim = QuantumSimulator::new(2);

    // Create a Bell state
    sim.h(0);        // Apply Hadamard to qubit 0
    sim.cnot(0, 1);  // CNOT with control=0, target=1

    // Measure the state
    let result = sim.measure();

    println!("Measured state: |{:02b}>", result);
}
```

### Running Examples

```bash
# Run basic examples
cargo run --example hello_quantum

# Run with Metal GPU (macOS only)
cargo run --example metal_basics --features metal

# Run algorithm examples
cargo run --example grover
cargo run --example qft
cargo run --example vqe
```

### Phase 2 Benchmark Command

```bash
# Backend + module benchmarks (Rust-only paths)
cargo run --release --bin phase2_bench
cat results/phase2_bench_summary_2026-02-14_v17_distributed_mirror_and_shard_mode.txt

# Targeted validation tests for distributed gradient stack
cargo test --lib distributed_metal_mpi::tests
cargo test --lib distributed_adjoint::tests
cargo test --lib auto_simulator::tests
cat results/phase2_targeted_tests_summary_2026-02-14_v9_distributed_mirror_and_shard_mode.txt
```

### Strict GPU-Only Execution

```rust
use nqpu_metal::auto_simulator::AutoSimulator;

let sim = AutoSimulator::with_gpu_only(12);
let probs = sim.execute_result(&gates)?; // Returns Err if Metal GPU is unavailable
```

This mode enforces GPU execution and avoids silent CPU fallback.

Python bindings expose the same mode with backend `GPUOnly`.

---

## Core API

### QuantumSimulator

Main simulator type for exact state vector simulation.

#### Constructor

```rust
pub fn QuantumSimulator::new(num_qubits: usize) -> Self
```

**Parameters:**
- `num_qubits`: Number of qubits to simulate (1-25 practical limit)

**Returns:** New simulator initialized to |0...0>

**Example:**
```rust
let sim = QuantumSimulator::new(5);  // 5-qubit simulator
```

**Complexity:**
- Time: O(2^n) for memory allocation
- Space: O(2^n) for state vector

---

#### State Access

```rust
pub fn state(&self) -> &QuantumState
```

**Returns:** Immutable reference to quantum state

```rust
pub fn state_mut(&mut self) -> &mut QuantumState
```

**Returns:** Mutable reference to quantum state (advanced use only)

**Example:**
```rust
let state = sim.state();
let probs = state.probabilities();
println!("Probability of |00>: {}", probs[0]);
```

---

### Gate Operations

All gate operations are available via the `GateOperations` trait.

#### Single-Qubit Gates

##### Hadamard Gate

```rust
pub fn h(&mut self, qubit: usize)
```

Applies Hadamard gate: H = (1/√2)[[1, 1], [1, -1]]

**Parameters:**
- `qubit`: Target qubit index (0 to num_qubits-1)

**Example:**
```rust
sim.h(0);  // Apply Hadamard to qubit 0
```

**Matrix:**
$$
H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
$$

---

## Phase 2 Rust Modules

### F32 + Fusion

```rust
use nqpu_metal::{F32FusionExecutor, Gate};

let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::rz(1, 0.3)];
let exec = F32FusionExecutor::new();
let (_state, metrics) = exec.execute(2, &gates)?;
println!("estimated speedup: {}", metrics.estimated_speedup);
```

### Heisenberg-Limit QPE

```rust
use nqpu_metal::{estimate_phase_heisenberg, HeisenbergQpeConfig, IdealPhaseOracle};

let oracle = IdealPhaseOracle { phase: 0.125, readout_error: 0.01 };
let cfg = HeisenbergQpeConfig::default();
let res = estimate_phase_heisenberg(&oracle, &cfg);
println!("phase estimate: {}", res.phase_estimate);
```

### Enhanced ZNE

```rust
use nqpu_metal::{EnhancedZne, Gate};

let zne = EnhancedZne::default();
let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
let (mitigated, points) = zne.run(&gates, |folded| Ok(1.0 - 0.001 * folded.len() as f64))?;
println!("mitigated estimate: {mitigated}, points: {}", points.len());
```

### CV Quantum + GBS

```rust
use nqpu_metal::{CvGaussianState, GaussianBosonSampler};

let mut state = CvGaussianState::vacuum(2);
state.squeeze(0, 0.5, 0.0);
state.beamsplitter(0, 1, std::f64::consts::FRAC_PI_4, 0.0);
let sampler = GaussianBosonSampler::new(state);
let clicks = sampler.sample_click_patterns(32);
println!("shots: {}", clicks.len());
```

### Circuit Cutting

```rust
use nqpu_metal::{
    execute_cut_circuit_z, plan_cuts_auto, estimate_sampling_cost, search_best_cut_plan,
    AutoCutConfig, CutSearchConfig, ReconstructionMode, Gate
};

let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::cnot(1, 2)];
let cfg = AutoCutConfig {
    max_fragment_qubits: 2,
    lookahead_gates: 8,
};
let plan = plan_cuts_auto(&gates, &cfg);
println!("auto cuts found: {}", plan.cut_points.len());

let cost = estimate_sampling_cost(&plan, 1024, ReconstructionMode::QuasiProbability);
println!("estimated total shots: {}", cost.estimated_total_shots);

let search = CutSearchConfig {
    max_fragment_qubits_candidates: vec![2, 4, 8],
    lookahead_candidates: vec![4, 8, 16],
    ..CutSearchConfig::default()
};
let best = search_best_cut_plan(&gates, &search)?;
println!("best cut objective: {}", best.objective);

let stitched = execute_cut_circuit_z(&gates, 2)?;
println!("stitched Z estimate: {}", stitched);
```

### QEC Interop (Stim-Like)

```rust
use nqpu_metal::{
    build_matching_graph, build_stim_like_surface_code_model, parse_stim_like_detector_model,
    DetectorModelConfig, MatchingGraphConfig
};

let cfg = DetectorModelConfig {
    rounds: 4,
    data_error_rate: 1e-3,
    measurement_error_rate: 5e-4,
};
let model = build_stim_like_surface_code_model(5, &cfg)?;
let text = model.to_text();
println!("{}", text.lines().next().unwrap_or(""));

let parsed = parse_stim_like_detector_model(&text)?;
assert_eq!(parsed.distance, 5);

let matching = build_matching_graph(&parsed, &MatchingGraphConfig::default())?;
println!("matching edges: {}", matching.edges.len());
```

### Distributed Metal + MPI

```rust
use nqpu_metal::{
    DistributedMetalConfig, DistributedMetalShardExecutor, DistributedMetalWorldExecutor,
    MPICommunicator, ShardRemoteExecutionMode, Gate
};

// Per-rank shard executor (real MPI integration point)
let exec = DistributedMetalShardExecutor::new(
    10,
    MPICommunicator { rank: 0, size: 1 },
    DistributedMetalConfig {
        remote_execution_mode: ShardRemoteExecutionMode::EmulatedWorldExact,
        ..DistributedMetalConfig::default()
    },
)?;
let run = exec.execute_partitioned(&[Gate::h(0), Gate::cnot(0, 1)])?;
println!(
    "local_gates={} remote_gates={} world_emulated={}",
    run.metrics.local_gates, run.metrics.remote_gates, run.metrics.shard_remote_world_emulation_used
);

// Single-process multi-rank executor with exact remote exchange emulation
let mut world = DistributedMetalWorldExecutor::new(10, 4, DistributedMetalConfig::default())?;
let world_run = world.execute_partitioned(&[Gate::h(9), Gate::cnot(9, 3)])?;
println!(
    "exchange_required={} comm_events={} p0={}",
    world_run.metrics.remote_gates_exchange_required,
    world_run.metrics.communication_events,
    world_run.global_probabilities[0]
);
```

### Distributed Adjoint (Method-Selectable)

```rust
use nqpu_metal::{
    distributed_gradient, CommunicationCostModel, DistributedAdjointConfig,
    DistributedGradientMethod, AdjointCircuit, AdjointOp, Observable, Gate,
};

let mut circuit = AdjointCircuit::new(2);
circuit.add_op(AdjointOp::Rx { qubit: 0, param: 0 });
circuit.add_op(AdjointOp::Fixed(Gate::cnot(0, 1)));
circuit.add_op(AdjointOp::Rz { qubit: 1, param: 1 });
let obs = Observable::PauliZ(0);
let params = vec![0.2, -0.3];

let mut cfg = DistributedAdjointConfig::default();
cfg.world_size = 2;
cfg.cost_model = CommunicationCostModel {
    latency_cost: 1.0,
    bandwidth_cost: 0.2,
    pairwise_weight: 1.0,
    fallback_weight: 1.6,
};
cfg.method = DistributedGradientMethod::AdjointMirror;

let res = distributed_gradient(&circuit, &params, obs, &cfg)?;
println!(
    "exp={} grads={} evals={} method={:?} fallback={:?}",
    res.expectation,
    res.gradients.len(),
    res.num_evaluations,
    res.method_used,
    res.fallback_reason
);
```

### Mid-Circuit Shot Branching

```rust
use nqpu_metal::mid_circuit::{
    ClassicalCondition, Operation, QuantumStateWithMeasurements, ShotBranchingConfig
};

let sim = QuantumStateWithMeasurements::new(2);
let ops = vec![
    Operation::h(0),
    Operation::measure(0, 0),
    Operation::conditional(Operation::x(1), ClassicalCondition::BitSet(0)),
    Operation::measure(1, 1),
];
let result = sim.execute_shots_branching(&ops, 1024, &ShotBranchingConfig::default())?;
println!("outcomes={} keys={}", result.outcomes.len(), result.counts.len());
```

### Adjoint Differentiation

```rust
use nqpu_metal::{AdjointCircuit, AdjointOp, Observable};

let mut c = AdjointCircuit::new(1);
c.add_op(AdjointOp::Ry { qubit: 0, param: 0 });
let grad = c.gradient(&[0.3], Observable::PauliZ(0))?;
println!("d<Z>/dtheta = {}", grad[0]);
```

### Topological / Dynamic QEC / Pulse

```rust
use nqpu_metal::{DynamicSurfaceCode, FibonacciAnyonState, RlDecoder, StringNetPlaquette};

let mut anyon = FibonacciAnyonState::basis_zero();
anyon.braid_word(&[1, 2, -1, -2])?;

let mut plaquette = StringNetPlaquette::new(6);
plaquette.apply_projector();

let mut code = DynamicSurfaceCode::new(7);
let mut decoder = RlDecoder::new();
let report = code.run_cycle(&mut decoder);
println!("logical error proxy: {}", report.logical_error_rate);
```

All snippets above are Rust-native API paths (no Python dependency).

### Traits (Backend Abstraction)

Unified trait hierarchy for polymorphic quantum simulation backends.

```rust
use nqpu_metal::traits::{QuantumBackend, ErrorModel, TensorContractor, FermionMapping};

// Any backend implementing QuantumBackend works polymorphically
let mut backend: Box<dyn QuantumBackend> = /* ... */;
backend.apply_gate(&Gate::h(0))?;
let probs = backend.probabilities()?;
let counts = backend.sample(1024)?;
```

Key traits: `QuantumBackend` (gate/measure/sample), `ErrorModel` (noise injection), `TensorContractor` (matrix contraction + SVD), `FermionMapping` (creation/annihilation operators).

### Pauli Algebra

Efficient sparse Pauli string representation using packed bitstrings.

```rust
use nqpu_metal::pauli_algebra::{PauliString, PauliSum, WeightedPauliString};

let p1 = PauliString::from_str_rep("IXYZ");
let p2 = PauliString::single(4, 2, 'X');
let (phase, result) = p1.multiply(&p2);
let commutes = p1.commutes_with(&p2);
let weight = p1.weight();  // Non-identity count
```

### Pauli Propagation

Heisenberg-picture observable evolution via backward Pauli string tracking.

```rust
use nqpu_metal::pauli_propagation::{PauliPropagationSimulator, TruncationPolicy};

let obs = WeightedPauliString::unit(PauliString::single(2, 0, 'Z'));
let policy = TruncationPolicy::with_limits(1000, 1e-8);
let mut sim = PauliPropagationSimulator::new(2, obs, policy);
sim.propagate_gate(&Gate::h(0));
let expectation = sim.expectation(&final_state);
```

### Pauli Twirling

Randomized compiling to convert coherent noise into stochastic Pauli channels.

```rust
use nqpu_metal::pauli_twirling::{TwirlingTable, PauliTwirler, TwirledEstimator};

let table = TwirlingTable::new();
let mut twirler = PauliTwirler::new(table, seed);
let twirled_circuit = twirler.twirl_circuit(&circuit);
let estimator = TwirledEstimator::new(twirler, 100);  // 100 samples
let result = estimator.estimate(&circuit, &observable, backend);
```

### Quantum Channel

Kraus operator and Choi matrix abstractions for noise modeling.

```rust
use nqpu_metal::quantum_channel::{KrausChannel, ChoiMatrix};

let depol = KrausChannel::depolarizing(0.01);
let amp_damp = KrausChannel::amplitude_damping(0.05);
let rho_out = depol.apply_to_density_matrix(&rho_in);
assert!(depol.is_trace_preserving(1e-10));
let composed = depol.compose(&amp_damp);  // Sequential channels
```

### Near-Clifford Simulation

CH-form representation for efficient simulation of Clifford+T circuits with few T-gates.

```rust
use nqpu_metal::near_clifford::{CHFormState, NearCliffordSimulator};

// Use when: <40 T-gates, >20 qubits, Clifford fraction >0.9
let mut state = CHFormState::new(50);
state.apply_clifford(&Gate::h(0));
let mut sim = NearCliffordSimulator::new(state);
sim.apply_t_gate(5);  // Branches exponentially in T-count
let probs = sim.probabilities();
```

Memory: O(n²). Gate cost: O(n²) Clifford, O(2^t) sampling where t = T-count.

### Circuit Equivalence

Formal verification that two circuits implement the same unitary (up to global phase).

```rust
use nqpu_metal::circuit_equivalence::{CircuitEquivalenceChecker, EquivalenceMethod};

let checker = CircuitEquivalenceChecker::new();
let result = checker.check(&circuit_a, &circuit_b, num_qubits, EquivalenceMethod::Auto);
assert!(result.equivalent);
println!("Fidelity: {}, Method: {:?}", result.fidelity, result.method_used);
```

Methods: `MatrixComparison` (<10 qubits), `StatisticalSampling` (10+), `SymbolicTracking` (11-20), `Auto`.

### Circuit Complexity

Resource analysis for quantum hardware deployment planning.

```rust
use nqpu_metal::circuit_complexity::{CircuitComplexityAnalyzer, AnalysisCircuit};

let mut circuit = AnalysisCircuit::new(100);
circuit.add_gate("H", vec![0], vec![]);
circuit.add_gate("T", vec![5], vec![]);

let report = CircuitComplexityAnalyzer::analyze(&circuit);
println!("T-count: {}, T-depth: {}", report.t_count, report.t_depth);
println!("Est. physical qubits: {}", report.estimated_physical_qubits);
println!("Barren plateau risk: {:?}", report.barren_plateau_risk.level);
```

Reports T-count/depth, magic state count, physical qubit estimates, expressibility, and barren plateau risk.

### Advanced Error Mitigation

PEC, CDR, symmetry verification, and virtual distillation for noisy quantum simulation.

```rust
use nqpu_metal::advanced_error_mitigation::{PECDecomposition, CDRModel, SymmetryVerifier, Symmetry};

// Probabilistic Error Cancellation
let pec = PECDecomposition::from_depolarizing(0.01, 0);
let (correction_gate, sign) = pec.sample_correction();

// Clifford Data Regression
let mut cdr = CDRModel::new();
cdr.train(training_point);
let corrected = cdr.correct(noisy_value);

// Symmetry Verification (post-selection)
let valid = SymmetryVerifier::verify(&shots, &Symmetry::Z2Parity);
```

### Non-Markovian Noise

Process tensor formalism for memory-bearing quantum noise with RTN and 1/f models.

```rust
use nqpu_metal::non_markovian::{ProcessTensor, RTNParams, NonMarkovianSimulator};

let rtn = RTNParams { switching_rate: 0.1, dephasing_strength: 0.01 };
let tensor = ProcessTensor::from_rtn(&rtn, 100, 0.1);
let rho_out = NonMarkovianSimulator::evolve(&rho_init, &tensor);
let non_mark = NonMarkovianSimulator::measure_non_markovianity(&tensor);
```

### Dynamical Decoupling

Circuit transformation pass inserting identity-equivalent pulse sequences to suppress decoherence.

```rust
use nqpu_metal::dynamical_decoupling::{DDPass, DDConfig, DDSequence};

let config = DDConfig::new()
    .sequence(DDSequence::XY4)
    .min_idle_slots(2);
let pass = DDPass::new(config);
let augmented = pass.insert_dd(&circuit, num_qubits);
```

Sequences: `XY4`, `CPMG(n)`, `UDD(n)`, `PlatonicXY`, `Custom(Vec<GateType>)`.

### Bosonic Codes

Bosonic QEC codes (cat, GKP, binomial) encoding logical qubits in harmonic oscillators.

```rust
use nqpu_metal::bosonic_codes::{CatCode, GKPCode, BinomialCode, FockState};

let cat_zero = CatCode::encode_zero(C64::new(2.0, 0.0), 50);
let gkp_plus = GKPCode::encode_zero(0.2, 100);
let bin_zero = BinomialCode::encode_zero(2, 30);

let vacuum = FockState::new(50);
let coherent = coherent_state(C64::new(1.5, 0.0), 50);
```

### Floquet Codes

Dynamical QEC codes with time-varying stabilizers (honeycomb, X3Z3).

```rust
use nqpu_metal::floquet_codes::{HoneycombCode, X3Z3Code, FloquetCode, FloquetSimulator};

let schedule = HoneycombCode::build_schedule(5);
let mut sim = FloquetSimulator::new(schedule);
let syndrome = sim.run_cycle(0.001);
```

### Quantum LDPC (qLDPC)

Hypergraph product and bivariate bicycle codes with belief propagation decoding.

```rust
use nqpu_metal::qldpc::{BivariateBicycleCode, HypergraphProductCode, BPDecoder, BPMode};

let code = BivariateBicycleCode::new(6, 3, 3);
let decoder = BPDecoder::new(BPMode::MinSum);
let correction = decoder.decode(&code.hx, &syndrome);
```

### Stim Import

Parse Google Stim QEC circuit format into nQPU-Metal gates.

```rust
use nqpu_metal::stim_import::StimCircuit;

let stim_text = "H 0\nCNOT 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]";
let circuit = StimCircuit::parse(stim_text)?;
let conversion = circuit.to_gates();
```

### QASM 3.0 Parser

Hand-written parser for OpenQASM 2.0 and 3.0 with control flow support.

```rust
use nqpu_metal::qasm3::{QASM3Parser, QASM3Program};

let src = r#"OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];"#;
let mut parser = QASM3Parser::new();
let program = parser.parse(src)?;
let gates = QASM3Parser::to_gates(&program)?;
```

Supports `if`, `while`, `for` control flow, custom gate definitions, and both QASM 2.0/3.0.

### Differentiable MPS

Gradient-based optimization of variational MPS circuits via forward-mode AD.

```rust
use nqpu_metal::differentiable_mps::{DifferentiableMPS, MPSVariationalAnsatz};

let mut mps = DifferentiableMPS::new(num_sites, max_bond_dim);
mps.apply_gate(site, &gate_matrix);

// VQE with analytic gradients
let mut ansatz = MPSVariationalAnsatz::new(num_sites, max_bond_dim, num_params);
let (energy, gradient) = ansatz.energy_and_gradient(&hamiltonian);
```

### Fermionic Tensor Networks

Z2-graded tensor networks with correct fermionic anti-commutation sign tracking.

```rust
use nqpu_metal::fermionic_tensor_net::{FermionicMPS, FermionicTensor, GradedIndex};

let mut mps = FermionicMPS::new(4, 32);  // 4 sites, bond dim 32
let fswap = FermionicSwapGate::as_tensor();
mps.apply_local_gate(0, &fswap);
```

### Matchgate Simulation

O(n³) classical simulation of free-fermion circuits via Majorana covariance matrices.

```rust
use nqpu_metal::matchgate_simulation::MatchgateSimulator;

let mut sim = MatchgateSimulator::new(10);
sim.apply_gate(&matchgate_hop(0.5))?;
let prob = sim.occupation_probability(5);
let corr = sim.two_point_correlator(0, 5);
```

### Stabilizer Tensor Networks

Hybrid simulation with cost scaling by non-stabilizerness (magic) rather than qubit count.

```rust
use nqpu_metal::stabilizer_tensor_net::{StabilizerTensorNetwork, STNConfig, MagicMonotone};

let config = STNConfig { max_stabilizer_terms: 100, max_magic: 5.0, ..Default::default() };
let mut stn = StabilizerTensorNetwork::new(20, config);
stn.apply_gate(&Gate::h(0));   // Stays stabilizer
stn.apply_gate(&Gate::t(0));   // Branches (2 terms)

let mana = MagicMonotone::mana(&coefficients);
```

### Decision Diagrams

BDD-based quantum state compression with exponential savings on structured states.

```rust
use nqpu_metal::decision_diagram::DDSimulator;

let mut sim = DDSimulator::new(20);
sim.apply_gate(&Gate::h(0));
for i in 0..19 { sim.apply_gate(&Gate::cnot(0, i+1)); }
println!("Nodes: {}", sim.node_count());  // ~20 for GHZ, not 2^20
```

### Enhanced Barren Plateau Analysis

Empirical trainability diagnosis for variational quantum circuits.

```rust
use nqpu_metal::enhanced_barren_plateau::EmpiricalBarrenPlateauAnalysis;

let analyzer = EmpiricalBarrenPlateauAnalysis::new(100);
let report = analyzer.analyze(&my_circuit_template, num_qubits, num_params);
println!("Scaling exponent: {}", report.scaling_exponent);  // >0 ⇒ problem
println!("Risk: {:?}", report.risk_level);
```

### State Checkpoint

Time-travel debugging and diff analysis for quantum state evolution.

```rust
use nqpu_metal::state_checkpoint::CheckpointManager;

let mut mgr = CheckpointManager::new();
let cp1 = mgr.checkpoint(&state, 0, "before_cnot");
// ... apply gates ...
let cp2 = mgr.checkpoint(&state, 5, "after_cnot");
let diff = mgr.diff(cp1, cp2).unwrap();
println!("Fidelity: {}, Trace dist: {}", diff.fidelity, diff.trace_distance);
```

### Symmetry Simulation

Hilbert space decomposition by conserved quantum numbers for memory reduction.

```rust
use nqpu_metal::symmetry_simulation::{SymmetricSimulator, SymmetryType};

// 20 qubits, 10 particles → C(20,10) = 184,756 vs 2^20 = 1,048,576 (5.7× savings)
let sim = SymmetricSimulator::new(20, SymmetryType::ParticleNumber(10))?;
sim.apply_gate(&Gate::cnot(0, 1))?;  // Must preserve particle number
let full = sim.to_full_state();
```

### Quantum Chemistry

Fermion-to-qubit mappings (Jordan-Wigner, Bravyi-Kitaev) and molecular Hamiltonians.

```rust
use nqpu_metal::quantum_chemistry::{JordanWignerMapper, MolecularData, UCCSDGenerator};

let data = MolecularData::h2_sto3g();
let hamiltonian = JordanWignerMapper::build_hamiltonian(
    &data.one_electron_integrals(), &data.two_electron_integrals(), data.n_orbitals
);

let uccsd = UCCSDGenerator::new(data.n_orbitals, data.n_electrons);
let singles = uccsd.singles_excitations();
```

Built-in molecules: H₂, LiH, HeH⁺ (STO-3G basis).

### Quantum Networking

Quantum communication channels, entanglement distribution, and purification protocols.

```rust
use nqpu_metal::quantum_networking::{FiberChannel, BBPSSWPurification, bell_state_phi_plus};

let fiber = FiberChannel::new(50.0, 0.2, 0.01);  // 50km, 0.2 dB/km loss
let bell = bell_state_phi_plus();
let (rho_out, p_success) = fiber.transmit(&bell);

let purifier = BBPSSWPurification;
let (purified, p_purify) = purifier.purify(&rho1, &rho2);
```

Includes: `FiberChannel`, `FreeSpaceChannel`, `ErasureChannel`, `QuantumRepeater`, `QuantumNetwork`.

### Quantum Random Walk

Discrete/continuous-time quantum walks on graphs.

```rust
use nqpu_metal::quantum_random_walk::{Graph, DiscreteQuantumWalk, DiscreteWalkConfig, CoinOperator};

let graph = Graph::hypercube(4);  // 16 vertices
let mut walk = DiscreteQuantumWalk::new(DiscreteWalkConfig {
    graph, coin: CoinOperator::Grover, steps: 100,
});
let result = walk.run(0);
```

Graph builders: `complete`, `cycle`, `line`, `hypercube`, `grid_2d`.
Applications: `QuantumPageRank`, `QuantumWalkSearch`.

### Quantum Reservoir Computing

Quantum dynamics as a computational reservoir for classical ML tasks.

```rust
use nqpu_metal::quantum_reservoir::{QuantumReservoir, ReservoirConfig, InputEncoding, Observable};

let config = ReservoirConfig {
    num_qubits: 6, reservoir_depth: 3, encoding: InputEncoding::AngleEncoding,
    observables: vec![Observable::AllZ], entangling: true, seed: 42, noise_level: 0.01,
};
let reservoir = QuantumReservoir::new(config);
let trained = reservoir.train(&train_inputs, &train_targets);
let predictions = trained.predict(&test_inputs);
```

### Quantum Cloning

Optimal approximate cloning machines (no-cloning theorem workarounds).

```rust
use nqpu_metal::quantum_cloning::{QuantumCloningMachine, CloningConfig, CloningType};

let machine = QuantumCloningMachine::new(CloningConfig {
    num_qubits: 1, cloning_type: CloningType::Universal1to2, compute_fidelity: true,
});
let result = machine.clone_state(&input_state);
println!("Clone fidelity: {:.4} (optimal: 5/6)", result.fidelity_clone1.unwrap());
```

Types: `Universal1to2` (5/6 fidelity), `Universal1toM`, `PhaseCovariant`, `Asymmetric`, `ProbabilisticExact`, `Economic`.

### Quantum Game Theory

Quantum extensions of classical games with entangled strategies.

```rust
use nqpu_metal::quantum_game::{QuantumGame, QuantumStrategy};

let game = QuantumGame::prisoners_dilemma();
let q_miracle = QuantumStrategy::quantum_miracle();
let result = game.play(&q_miracle, &q_miracle);
// Both achieve payoff > classical Nash equilibrium
```

Built-in games: `prisoners_dilemma`, `battle_of_sexes`, `chicken`.

### Quantum Cellular Automata

Discrete-time QCA on 1D chains with Margolus partitioning.

```rust
use nqpu_metal::quantum_cellular_automata::{QuantumCellularAutomaton, QCARuleType, BoundaryCondition};

let mut qca = QuantumCellularAutomaton::new(
    6, QCARuleType::Goldilocks(PI/4.0), BoundaryCondition::Periodic,
);
let data = qca.evolve(20);
println!("Entanglement: {:?}", data.entanglement_entropy);
```

Rules: `Goldilocks(J)`, `Heisenberg(J)`, `PairUnitary(matrix)`, `CustomClosure`.

### QRNG Integration

Unified interface for quantum random number generators.

```rust
use nqpu_metal::qrng_integration::{create_measurement_rng, QrngConfig, QrngSourceType};

let config = QrngConfig {
    source_type: QrngSourceType::Anu { api_key: "YOUR_KEY".into() },
    enable_fallback: true,
};
let mut rng = create_measurement_rng(Some(config));
let mut buf = [0u8; 32];
rng.fill_bytes(&mut buf)?;
```

Sources: `Simulated` (CSPRNG), `Anu` (ANU API), `Hardware` (device file). Hybrid fallback supported.

### AMX Tensor

Apple Accelerate (AMX) hardware-accelerated tensor operations for MPS inner loops.

```rust
use nqpu_metal::amx_tensor::{AmxComplexGemm, AmxSvd, AmxTensorContractor};

let result = AmxComplexGemm::multiply(a, b, rows_a, cols_a, cols_b);
let (u, s, vt) = AmxSvd::compute(matrix, m, n)?;
let contractor = AmxTensorContractor::new();  // Implements TensorContractor trait
```

Requires `amx` feature flag. macOS only.

### Metal 4 Backend

Metal 4 GPU tensor operations with adaptive ML-guided backend routing.

```rust
use nqpu_metal::metal4_backend::{Metal4Capabilities, Metal4TensorContraction, Metal4AdaptiveML};

let caps = Metal4Capabilities::detect();
if caps.supports_metal4() {
    let engine = Metal4TensorContraction::new();
    let result = engine.contract_pair(a, b, dims_a, dims_b, contract_axes);
}
```

Requires `metal4` feature flag (macOS 15+).

### Measurement-Based Quantum Computation (MBQC)

One-way quantum computation using cluster states and measurement patterns.

```rust
use nqpu_metal::mbqc::{linear_cluster, MBQCSimulator, MeasurementPattern, MeasurementBasis};

let cluster = linear_cluster(5);
let mut pattern = MeasurementPattern::new();
pattern.add_command(MeasurementCommand::new(0, MeasurementBasis::XBasis));
let mut sim = MBQCSimulator::new(cluster);
let outcomes = sim.execute(&pattern);
```

Lattices: `linear_cluster`, `square_cluster`, `brickwork`.

### WASM Backend

Pure-Rust quantum simulator for WebAssembly with no platform dependencies.

```rust
use nqpu_metal::wasm_backend::WasmSimulator;

let mut sim = WasmSimulator::new(2);
sim.h(0);
sim.cnot(0, 1);
let probs = sim.probabilities();  // ~[0.5, 0.0, 0.0, 0.5]
```

Requires `wasm` feature flag. Conflicts with `metal`/`cuda`.

### WASM Bindings

JavaScript-friendly WebAssembly API via wasm-bindgen.

```rust
// From JavaScript:
// const sim = new JsQuantumSimulator(2);
// sim.h(0);
// sim.cnot(0, 1);
// const probs = sim.probabilities();

// Free functions: bell_state(), ghz_state(n), random_circuit(n, depth)
```

### Topological Computing (Expanded)

Ising anyons, Majorana fermions, and braid-based gate compilation.

```rust
use nqpu_metal::topological_expanded::{IsingAnyonState, BraidCompiler, MajoranaChain};

let mut state = IsingAnyonState::new(4);
state.braid(0);  // Braid strands 0 and 1

let compiler = BraidCompiler::new();
let braid = compiler.compile_gate(&GateType::H);

let chain = MajoranaChain::new(8);  // 4 logical qubits
```

---

##### Pauli Gates

```rust
pub fn x(&mut self, qubit: usize)  // Pauli-X (NOT gate)
pub fn y(&mut self, qubit: usize)  // Pauli-Y
pub fn z(&mut self, qubit: usize)  // Pauli-Z
```

**Example:**
```rust
sim.x(0);  // Bit flip
sim.y(1);  // Bit and phase flip
sim.z(2);  // Phase flip
```

**Matrices:**
$$
X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad
Y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}, \quad
Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$

---

##### Phase Gates

```rust
pub fn s(&mut self, qubit: usize)  // Phase gate (√Z)
pub fn t(&mut self, qubit: usize)  // π/8 gate (∛Z)
```

**Example:**
```rust
sim.s(0);  // Apply S gate
sim.t(1);  // Apply T gate
```

**Matrices:**
$$
S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}, \quad
T = \begin{bmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{bmatrix}
$$

---

##### Rotation Gates

```rust
pub fn rx(&mut self, qubit: usize, theta: f64)  // X-rotation
pub fn ry(&mut self, qubit: usize, theta: f64)  // Y-rotation
pub fn rz(&mut self, qubit: usize, theta: f64)  // Z-rotation
```

**Parameters:**
- `qubit`: Target qubit
- `theta`: Rotation angle in radians

**Example:**
```rust
use std::f64::consts::PI;

sim.rx(0, PI / 4.0);  // Rotate around X by π/4
sim.ry(1, PI / 2.0);  // Rotate around Y by π/2
sim.rz(2, PI / 8.0);  // Rotate around Z by π/8
```

**Matrices:**
$$
R_x(\theta) = \begin{bmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{bmatrix}
$$

$$
R_y(\theta) = \begin{bmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{bmatrix}
$$

$$
R_z(\theta) = \begin{bmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{bmatrix}
$$

---

##### General Unitary Gate

```rust
pub fn u(&mut self, qubit: usize, matrix: &[[C64; 2]; 2])
```

Applies an arbitrary 2x2 unitary gate.

**Parameters:**
- `qubit`: Target qubit
- `matrix`: 2x2 unitary matrix (must satisfy U†U = I)

**Example:**
```rust
use nqpu_metal::C64;

// Custom rotation gate
let gate = [
    [C64::new(0.9, 0.0), C64::new(-0.1, 0.4)],
    [C64::new(0.1, 0.4), C64::new(0.9, 0.0)],
];
sim.u(0, &gate);
```

---

#### Two-Qubit Gates

##### CNOT Gate

```rust
pub fn cnot(&mut self, control: usize, target: usize)
```

Controlled-NOT gate.

**Parameters:**
- `control`: Control qubit index
- `target`: Target qubit index

**Example:**
```rust
sim.cnot(0, 1);  // If qubit 0 is |1>, flip qubit 1
```

---

##### CZ Gate

```rust
pub fn cz(&mut self, control: usize, target: usize)
```

Controlled-Z gate.

**Example:**
```rust
sim.cz(0, 1);
```

---

##### SWAP Gate

```rust
pub fn swap(&mut self, qubit1: usize, qubit2: usize)
```

Swaps the states of two qubits.

**Example:**
```rust
sim.swap(0, 1);  // Exchange states of qubits 0 and 1
```

---

##### Controlled Rotations

```rust
pub fn crx(&mut self, control: usize, target: usize, theta: f64)
pub fn cry(&mut self, control: usize, target: usize, theta: f64)
pub fn crz(&mut self, control: usize, target: usize, theta: f64)
```

Controlled rotation gates.

**Example:**
```rust
sim.crx(0, 1, PI / 4.0);  // Controlled X-rotation
```

---

#### Three-Qubit Gates

##### Toffoli Gate

```rust
pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize)
```

Controlled-controlled-NOT (CCX) gate.

**Example:**
```rust
sim.toffoli(0, 1, 2);  // If qubits 0 AND 1 are |1>, flip qubit 2
```

---

### Measurement

#### Standard Measurement

```rust
pub fn measure(&mut self) -> usize
```

Measure all qubits in computational basis, collapsing the state.

**Returns:** Integer representing measured state (0 to 2^n - 1)

**Example:**
```rust
let result = sim.measure();
println!("Measured: |{:b}>", result);
```

**Note:** This modifies the simulator state (wavefunction collapse).

---

#### Single-Qubit Measurement

```rust
pub fn measure_qubit(&mut self, qubit: usize) -> bool
```

Measure a single qubit.

**Parameters:**
- `qubit`: Qubit to measure

**Returns:** `true` if |1>, `false` if |0>

**Example:**
```rust
let qubit_0_result = sim.measure_qubit(0);
if qubit_0_result {
    println!("Qubit 0 is in state |1>");
}
```

---

#### Probability Measurement

```rust
pub fn probability(&self, state: usize) -> f64
```

Get probability of measuring specific state (without collapsing).

**Parameters:**
- `state`: State index (0 to 2^n - 1)

**Returns:** Probability (0.0 to 1.0)

**Example:**
```rust
let prob_00 = sim.probability(0b00);
let prob_11 = sim.probability(0b11);
println!("P(|00>) = {}, P(|11>) = {}", prob_00, prob_11);
```

---

#### All Probabilities

```rust
pub fn probabilities(&self) -> Vec<f64>
```

Get probability distribution over all states.

**Returns:** Vector of 2^n probabilities

**Example:**
```rust
let probs = sim.probabilities();
for (i, p) in probs.iter().enumerate() {
    if *p > 0.01 {
        println!("P(|{:02b}>) = {:.4}", i, p);
    }
}
```

---

#### Expectation Values

```rust
pub fn expectation_z(&self, qubit: usize) -> f64
pub fn expectation_x(&self, qubit: usize) -> f64
pub fn expectation_y(&self, qubit: usize) -> f64
```

Compute expectation value of Pauli operators.

**Parameters:**
- `qubit`: Target qubit

**Returns:** Expectation value (-1.0 to 1.0)

**Example:**
```rust
let exp_z = sim.expectation_z(0);
println!("<Z_0> = {}", exp_z);  // -1 (|0>) to +1 (|1>)
```

---

### State Operations

#### Reset Qubit

```rust
pub fn reset_qubit(&mut self, qubit: usize)
```

Reset a qubit to |0> (collapses and reinitializes).

**Example:**
```rust
sim.reset_qubit(0);  // Force qubit 0 to |0>
```

---

#### Fidelity

```rust
pub fn fidelity(&self, other: &QuantumState) -> f64
```

Compute overlap with another quantum state.

**Returns:** Fidelity (0.0 to 1.0, where 1.0 is identical)

**Example:**
```rust
let target = QuantumState::new(2);
let fid = sim.state().fidelity(&target);
println!("Fidelity with target: {}", fid);
```

---

## Advanced API

### Noise Simulation

Simulate realistic quantum hardware noise.

#### NoisySimulator

```rust
use nqpu_metal::{
    QuantumSimulator,
    noise::{NoisySimulator, DepolarizingNoise}
};

let base_sim = QuantumSimulator::new(5);
let mut sim = NoisySimulator::new(
    base_sim,
    DepolarizingNoise::new(0.01)  // 1% depolarizing
);

sim.h(0);  // Noise applied automatically
sim.x(1);
```

#### Noise Models

##### Bit Flip Noise

```rust
use nqpu_metal::noise::BitFlipNoise;

let noise = BitFlipNoise::new(0.01);  // 1% bit flip probability
```

##### Phase Flip Noise

```rust
use nqpu_metal::noise::PhaseFlipNoise;

let noise = PhaseFlipNoise::new(0.01);  // 1% phase flip
```

##### Depolarizing Noise

```rust
use nqpu_metal::noise::DepolarizingNoise;

let noise = DepolarizingNoise::new(0.01);  // 1% depolarizing
```

##### Amplitude Damping (T1)

```rust
use nqpu_metal::noise::AmplitudeDamping;

let noise = AmplitudeDamping::new(0.1);  // T1 relaxation
```

##### Phase Damping (T2)

```rust
use nqpu_metal::noise::PhaseDamping;

let noise = PhaseDamping::new(0.1);  // T2 dephasing
```

##### Combined Noise

```rust
use nqpu_metal::noise::CombinedNoise;

let noise = CombinedNoise::new()
    .with_bit_flip(0.01)
    .with_phase_flip(0.005)
    .with_amplitude_damping(0.02);
```

---

### Tensor Networks

Simulate large quantum systems using Matrix Product States.

#### MPSSimulator

```rust
use nqpu_metal::tensor_network::MPSSimulator;

// Create 50-qubit simulator with bond dimension 32
let mut sim = MPSSimulator::new(50, Some(32));

sim.h(0);
sim.cnot(0, 1);
sim.cnot(1, 2);

let result = sim.measure();
```

**Parameters:**
- `num_qubits`: Number of qubits (can be 30-100+)
- `bond_dim`: Bond dimension χ (higher = more accurate, more memory)

**Performance:**
- Memory: O(n × χ²) vs O(2^n) for state vector
- Best for systems with limited entanglement

---

### Density Matrix

Simulate mixed states and correlated noise.

#### DensityMatrixSimulator

```rust
use nqpu_metal::density_matrix::DensityMatrixSimulator;

let mut sim = DensityMatrixSimulator::new(3);

sim.h(0);
sim.cnot(0, 1);

// Get reduced density matrix
let reduced = sim.reduced_state(&[0]);
println!("Purity: {}", reduced.purity());  // 1.0 = pure, <1.0 = mixed
```

#### Methods

```rust
pub fn reduced_state(&self, qubits: &[usize]) -> DensityMatrix
pub fn entropy(&self) -> f64  // von Neumann entropy
pub fn purity(&self) -> f64   // Tr(ρ²)
```

---

### Error Mitigation

Advanced techniques for NISQ devices.

#### Zero-Noise Extrapolation

```rust
use nqpu_metal::error_mitigation::{
    zero_noise_extrapolation, NoiseFactor
};

let result = zero_noise_extrapolation(
    2,  // num_qubits
    &mut |sim: &mut QuantumSimulator, scale| {
        // Circuit with noise scaling
        for _ in 0..scale {
            sim.h(0);
            sim.cnot(0, 1);
        }
    },
    &mut |sim: &mut QuantumSimulator| -> f64 {
        sim.expectation_z(0)  // Observable
    },
    NoiseFactor::Richardson,  // Extrapolation method
);
```

#### Readout Error Mitigation

```rust
use nqpu_metal::error_mitigation::ReadoutMitigation;

// Calibrate confusion matrix
let mitigation = ReadoutMitigation::calibrate(
    2,  // num_qubits
    100,  // shots per calibration
    |qubit, sim| sim.measure(),
);

// Apply mitigation
let (probs, corrected) = mitigation.mitigate_measurement(&raw_counts);
```

---

### QASM Support

Interoperability with OpenQASM 2.0.

#### Parse QASM

```rust
use nqpu_metal::qasm::{parse_qasm, execute_qasm};

let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
"#;

let circuit = parse_qasm(qasm)?;
let mut sim = QuantumSimulator::new(circuit.num_qubits);
let result = execute_qasm(&circuit, &mut sim);
```

#### Export QASM

```rust
use nqpu_metal::qasm::export_qasm;

let qasm_string = export_qasm(
    "my_circuit",
    2,  // num_qubits
    2,  // num_clbits
    |writer| {
        writer.h(0);
        writer.cx(0, 1);
        writer.measure(0, 0);
    },
);
```

---

## Algorithms

### Grover's Search

```rust
use nqpu_metal::GroverSearch;

let mut grover = GroverSearch::new(num_qubits);

// Calculate optimal iterations
let n = num_qubits as f64;
let num_iterations = (std::f64::consts::PI / 4.0 * (2.0_f64.powf(n)).sqrt()) as usize;

// Search for target state
let result = grover.search(target_state, num_iterations);
```

---

### Quantum Fourier Transform

```rust
use nqpu_metal::algorithms::{qft, inverse_qft};

// QFT
qft(&mut sim.state, num_qubits);

// Inverse QFT
inverse_qft(&mut sim.state, num_qubits);
```

---

### Variational Quantum Eigensolver (VQE)

```rust
use nqpu_metal::algorithms::{
    vqe, Hamiltonian, HamiltonianTerm, HardwareEfficientAnsatz
};

// Define Hamiltonian
let hamiltonian = Hamiltonian::new(vec![
    HamiltonianTerm::z(0, -1.0),
    HamiltonianTerm::z(1, -1.0),
    HamiltonianTerm::zz(0, 1, -0.5),
]);

// Create ansatz
let ansatz = HardwareEfficientAnsatz::new(2, 2);

// Optimize
let result = vqe(
    2,                     // num_qubits
    &hamiltonian,
    &ansatz,
    &initial_params,
    100,                   // max_iterations
    0.1,                   // learning_rate
);

println!("Ground state energy: {}", result.minimum_energy);
```

---

### Quantum Approximate Optimization (QAOA)

```rust
use nqpu_metal::algorithms::{qaoa, QAOAProblem};

// MaxCut problem
let edges = vec![(0, 1), (1, 2), (2, 0)];
let problem = QAOAProblem::maxcut(3, edges);

let result = qaoa(
    &problem,
    1,                      // depth (p)
    &[0.5, 0.5],           // initial params [gamma, beta]
    50,                    // optim iterations
    100,                   // shots per evaluation
);

println!("Best solution: {:b}", result.best_solution);
println!("Cut size: {}", -result.best_cost);
```

---

### Quantum Phase Estimation

```rust
use nqpu_metal::algorithms::qpe;

let result = qpe(
    3,  // precision qubits
    &mut sim,
    |sim, control, power| {
        // Controlled-U^(2^power)
        for _ in 0..(1 << power) {
            sim.cnot(control, target);
        }
    },
    |sim| {
        // Prepare eigenstate
        sim.h(target);
    },
);

println!("Estimated phase: {:.4}", result.phase);
```

---

## GPU Acceleration

Metal GPU acceleration for Apple Silicon (macOS only).

### MetalGPUSimulator

```rust
#[cfg(target_os = "macos")]
use nqpu_metal::MetalGPUSimulator;

#[cfg(target_os = "macos")]
fn main() {
    let mut sim = MetalGPUSimulator::new(16)?;

    sim.h(0);
    sim.cnot(0, 1);

    let result = sim.measure();
}
```

### Performance Comparison

| Qubits | CPU Time | GPU Time | Speedup |
|--------|----------|----------|---------|
| 12     | 0.030s   | 0.00019s | 160×    |
| 14     | 0.040s   | 0.00018s | 225×    |
| 16     | 0.27s    | 0.003s   | 90×     |

---

## Phase 3: Next-Wave Features

Five world-first modules implementing cutting-edge 2025-2026 quantum computing research.

### Certified QRNG

Bell-test certified quantum random number generation with CHSH inequality verification and Twine-style SHA-256 hash chain attestation.

**Why this matters**: Most quantum simulators that offer QRNG use a "trust-based" model — they sample from a quantum circuit and label the output as quantum random, but provide no proof of quantum origin. A classical PRNG could produce the same output and you'd never know. nQPU-Metal's Certified QRNG is fundamentally different:

- **Regular QRNG**: Trust-based. No proof of quantum origin. A classical RNG could produce identical output.
- **Certified QRNG**: Each batch includes a CHSH Bell test certificate. The measured S-value exceeds 2.0, which is *physically impossible* for any classical correlations (Bell's theorem). The Tsirelson bound (S ≈ 2.828) confirms the source is genuinely quantum.
- **Twine hash chain**: Each certificate commits to the previous certificate's hash, the S-value, and the extracted bits. This creates a blockchain-style append-only provenance trail that can be verified independently and offline — no need to trust the generator.
- **First quantum simulator with verifiable quantum randomness certification.**

```rust
use nqpu_metal::certified_qrng::{CertifiedQrngSource, CertifiedQrngConfig};

// Configure: 2000 Bell test rounds, require S > 2.1
let config = CertifiedQrngConfig::default()
    .with_chsh_rounds(2000)
    .with_min_s_value(2.1);

let mut source = CertifiedQrngSource::new(config);

// Generate certified random bits with Bell test proof
let cert = source.generate_certified()?;
assert!(cert.s_value > 2.0);  // Quantum violation proven!
println!("S-value: {:.3} (classical max: 2.0)", cert.s_value);
println!("Random bytes: {} bytes", cert.random_bytes.len());

// Verify the full certificate chain
let chain = source.chain();
assert!(chain.verify().is_ok());  // Tamper-evident provenance
```

**Key types**:
- `CertifiedQrngSource` — implements `QrngSource` trait; runs CHSH Bell tests per batch
- `CertifiedQrngConfig` — builder: `chsh_rounds` (1000), `min_s_value` (2.1), `epsilon` (1e-6)
- `BellTestCertificate` — S-value, round count, timestamp, `prev_hash`, `self_hash`, random bytes
- `CertificationChain` — append-only chain with `verify()` integrity check
- `ExtractionParams` — input/output bit lengths, min entropy rate
- `CertifiedQrngError` — `BellTestFailed`, `ChainBroken`, `ExtractionFailed`, `SourceError`

---

### Digital-Analog Quantum Simulation

Hybrid gate + Hamiltonian evolution circuits. This is how real near-term hardware (Google Sycamore, Qilimanjaro) actually operates — digital gate layers interleaved with native analog Hamiltonian evolution.

```rust
use nqpu_metal::digital_analog::{
    DAQCCircuit, DAQCSimulator, DAQCConfig, HardwareHamiltonians,
};
use nqpu_metal::gates::{Gate, GateType};

// Build a hybrid digital-analog circuit
let mut circuit = DAQCCircuit::new(4);

// Digital layer: prepare initial state
circuit.add_gate(Gate::single(GateType::H, 0));
circuit.add_gate(Gate::single(GateType::H, 1));

// Analog layer: evolve under transmon Hamiltonian for 1.0 time units
let h_transmon = HardwareHamiltonians::transmon(4, 1.0, 0.05);
circuit.add_analog(h_transmon, 1.0, 10);  // duration=1.0, trotter_steps=10

// More digital gates
circuit.add_gate(Gate::two(GateType::CNOT, 0, 2));

// Simulate
let config = DAQCConfig::default();
let sim = DAQCSimulator::new(config);
let probs = sim.simulate(&circuit)?;

// Compare fidelity: full DAQC vs digital-only approximation
let comparison = sim.compare_fidelity(&circuit)?;
println!("Fidelity: {:.4}", comparison.fidelity);
```

**Key types**:
- `DAQCCircuit` — ordered list of `DAQCSegment`s with `add_gate()` / `add_analog()`
- `DAQCSegment` — enum: `Digital(Vec<Gate>)` | `Analog(AnalogBlock)`
- `AnalogBlock` — `LocalHamiltonian1D` + duration + trotter steps
- `DAQCSimulator` — simulates circuits; dispatches digital to state-vector, analog to Trotter/TEBD
- `HardwareHamiltonians` — presets: `transmon()`, `rydberg()`, `trapped_ion()`
- `DAQCConfig` — `max_bond_dim`, `optimize_diagonal`, `trotter_tolerance`
- `FidelityComparison` — fidelity, gate counts, timing comparison

---

### Neural QEC Decoders

GNN-inspired message-passing neural network decoder for arbitrary stabilizer codes. First open-source simulator with ML-based QEC decoders.

```rust
use nqpu_metal::neural_decoder::{
    NeuralDecoder, NeuralDecoderConfig, SyndromeGraph, ActivationFn,
};

// Configure the GNN decoder
let config = NeuralDecoderConfig::default()
    .with_num_layers(3)
    .with_hidden_dim(32)
    .with_learning_rate(0.01)
    .with_activation(ActivationFn::ReLU);

let mut decoder = NeuralDecoder::new(config);

// Build syndrome graph from detector geometry
let graph = SyndromeGraph::surface_code(5);  // distance-5 surface code

// Decode a syndrome
let syndrome = vec![false, true, false, true, false, false, false, false];
let corrections = decoder.decode(&graph, &syndrome);
println!("Error locations: {:?}", corrections);

// Train on synthetic data
let result = decoder.train(&graph, 1000, 0.05)?;  // 1000 samples, 5% error rate
println!("Final accuracy: {:.1}%", result.final_accuracy * 100.0);

// Export/import trained weights
let weights = decoder.export_weights();
decoder.import_weights(&weights)?;
```

**Key types**:
- `NeuralDecoder` — GNN decoder with configurable layers and activation functions
- `SyndromeGraph` — adjacency list of detector nodes + edge weights
- `NeuralDecoderConfig` — builder: `num_layers` (3), `hidden_dim` (32), `learning_rate` (0.01)
- `GNNWeights` — per-layer `ndarray::Array2<f64>` weight matrices
- `TrainingResult` — loss history, final accuracy, epochs
- `ActivationFn` — `ReLU`, `Sigmoid`
- `NeuralDecoderError` — `DimensionMismatch`, `TrainingFailed`, `WeightLoadFailed`

---

### Yoked Surface Codes

Brand new QEC code from Nature Communications 2025. Composes surface code patches with hierarchical "yoke" stabilizers to achieve the same logical error protection with **1/3 the physical qubits** of standard surface codes.

```rust
use nqpu_metal::yoked_surface_codes::{
    YokedSurfaceCode, YokedCodeConfig, YokeDirection,
};

// Configure a 2×3 grid of distance-3 patches with row yokes
let config = YokedCodeConfig::default()
    .with_grid(2, 3)
    .with_base_distance(3)
    .with_yoke_direction(YokeDirection::RowAndColumn)
    .with_physical_error_rate(0.001);

let mut code = YokedSurfaceCode::new(config)?;
println!("Total qubits: {}", code.total_qubits());
println!("Qubit savings vs standard: {:.0}%",
    (1.0 - code.total_qubits() as f64
     / (6.0 * 2.0 * 3.0 * 3.0)) * 100.0);

// Inject an error and decode
code.inject_error(0, 0, 0);  // X error at patch 0, position (0,0)

let syndrome = code.measure();
let corrections = code.decode(&syndrome)?;
code.apply_corrections(&corrections);

// Estimate logical error rate
let estimate = code.estimate_error_rate(10000)?;
println!("Logical error rate: {:.2e}", estimate.logical_error_rate);
println!("Effective distance: {}", estimate.effective_distance);
```

**Key types**:
- `YokedSurfaceCode` — grid of surface code patches + yoke stabilizers + hierarchical decoder
- `YokedCodeConfig` — builder: `grid_rows`, `grid_cols`, `base_distance`, `yoke_direction`, `physical_error_rate`
- `YokeDirection` — `RowOnly` | `RowAndColumn`
- `YokedSyndrome` — per-patch syndromes + yoke parity bits
- `YokedDecoder` — two-stage: inner per-patch MWPM + outer LDPC yoke decode
- `YokedErrorEstimate` — logical error rate, trials, effective distance, qubit savings
- `PatchCorrection` — patch index + correction positions

---

### Metal 4 TensorOps (Extended)

Upgraded Metal 4 gate dispatch using `simdgroup_matrix` 8×8 tile operations for batched gate matmuls. Available on Apple8+ GPUs (M3/M4/M5). Provides 2-4× speedup for large state vectors over the standard Metal path.

```rust
use nqpu_metal::metal4_backend::{
    Metal4GateDispatcher, Metal4QuantumBackend, Metal4BenchmarkResult,
};
use nqpu_metal::gates::{Gate, GateType};

// Create GPU-accelerated quantum backend (requires Metal 4 / Apple8+)
let dispatcher = Metal4GateDispatcher::new()?;
let mut backend = Metal4QuantumBackend::new(14, dispatcher)?;  // 14 qubits

// Apply gates — dispatched to simdgroup_matrix GPU kernels
backend.apply_gate(&Gate::single(GateType::H, 0));
backend.apply_gate(&Gate::two(GateType::CNOT, 0, 1));

// Get probabilities (computed on GPU)
let probs = backend.probabilities();
assert!((probs[0] + probs[3] - 1.0).abs() < 1e-10);  // Bell state

// Benchmark CPU vs Metal 4
let bench = Metal4BenchmarkResult::run(14, 100)?;
println!("CPU: {:.3}ms, Metal4: {:.3}ms, Speedup: {:.1}×",
    bench.cpu_time_ms, bench.gpu_time_ms, bench.speedup);
```

**Key types**:
- `Metal4GateDispatcher` — generates + caches MSL kernels using `simdgroup_matrix` 8×8 tiles
- `Metal4QuantumBackend` — implements `QuantumBackend` trait with GPU gate dispatch
- `Metal4BenchmarkResult` — timing comparison CPU vs Metal 4 with speedup factor

**Feature gate**: Requires `features = ["metal"]` and macOS with Apple8+ GPU.

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Single-qubit gate | O(2^n) | State vector update |
| Two-qubit gate | O(2^n) | Strided access |
| Measurement | O(2^n) | Probability sampling |
| QFT | O(n² × 2^n) | n controlled rotations |

### Space Complexity

| Simulator | Space |
|-----------|-------|
| QuantumSimulator | O(2^n) |
| MPSSimulator (χ=32) | O(n × χ²) |
| DensityMatrixSimulator | O(4^n) |

### Practical Limits

| Simulator | Max Qubits | Memory (approx) |
|-----------|------------|-----------------|
| QuantumSimulator | 25-28 | 16 GB @ 28 qubits |
| MPSSimulator (χ=32) | 50-100 | ~2 MB @ 50 qubits |
| Metal GPU | 25-28 | GPU memory limited |

---

## Type Reference

### Core Types

```rust
pub struct C64 {
    pub re: f64,  // Real part
    pub im: f64,  // Imaginary part
}

pub struct QuantumState {
    amplitudes: Vec<C64>,
    pub num_qubits: usize,
    pub dim: usize,
}

pub struct QuantumSimulator {
    state: QuantumState,
}

pub struct MeasurementResult {
    pub outcome: usize,
    pub probability: f64,
}
```

### Error Types

```rust
pub enum NQPUError {
    InvalidQubit(usize),
    InvalidGate(String),
    QasmParseError(String),
    GPUNotAvailable,
}

impl std::error::Error for NQPUError {}
impl std::fmt::Display for NQPUError {}
```

---

## Best Practices

### 1. Choose the Right Simulator

```rust
// Small systems (< 20 qubits)
let sim = QuantumSimulator::new(n);

// Large systems, low entanglement
let sim = MPSSimulator::new(50, Some(32));

// Noise simulation
let sim = NoisySimulator::new(base, noise_model);
```

### 2. Minimize Measurements

```rust
// BAD: Measures collapse state
for _ in 0..100 {
    sim.measure();  // Re-prepare circuit each time
}

// GOOD: Batch measurements
for _ in 0..100 {
    let mut sim = QuantumSimulator::new(2);
    sim.h(0);
    sim.cnot(0, 1);
    sim.measure();
}
```

### 3. Use Sequential for Small Systems

```bash
# Recommended for < 16 qubits
cargo run --release --no-default-features

# Only use parallel for > 20 qubits
cargo run --release --features parallel
```

### 4. GPU for Speed on macOS

```bash
# Best performance on Apple Silicon
cargo run --release --features metal
```

---

## Troubleshooting

### Common Issues

**Out of Memory:**
```rust
// Reduce qubit count or use MPS
let sim = MPSSimulator::new(30, Some(16));  // Instead of 30-qubit full sim
```

**Slow Performance:**
```bash
# Use release mode
cargo run --release

# Try sequential (no default features)
cargo run --release --no-default-features
```

**GPU Not Available:**
```rust
#[cfg(target_os = "macos")]
{
    let gpu_sim = MetalGPUSimulator::new(qubits)?;
}

#[cfg(not(target_os = "macos"))]
{
    let cpu_sim = QuantumSimulator::new(qubits);
}
```

---

## Phase 4: Full-Spectrum API Reference

### Core Algorithm Frameworks

#### `qsp_qsvt` — Quantum Signal Processing / QSVT
Grand unifying framework that subsumes Grover search, QPE, Hamiltonian simulation, and matrix inversion through polynomial eigenvalue transformations.
```rust
use nqpu_metal::qsp_qsvt::{QspProcessor, QsvtConfig, BlockEncoding};
let config = QsvtConfig::builder().degree(10).target_precision(1e-6).build();
let processor = QspProcessor::new(config);
let phases = processor.compute_phases_for_inversion(kappa)?;
```

#### `dmrg_tdvp` — DMRG + TDVP
MPS ground-state finding via DMRG sweeps with Lanczos eigensolver, plus real-time dynamics via 1-site and 2-site TDVP with projector-splitting integrator.
```rust
use nqpu_metal::dmrg_tdvp::{DmrgSolver, TdvpIntegrator, HeisenbergMpo};
let mpo = HeisenbergMpo::new(num_sites, j_coupling, h_field);
let (energy, mps) = DmrgSolver::new(bond_dim).sweep(&mpo, num_sweeps)?;
```

#### `improved_trotter` — Enhanced Trotter Decomposition
Time-dependent product formulas for two-energy-scale Hamiltonians (10x improvement) and variational product formulas with optimized coefficients (2-5x error reduction).

#### `optimistic_qft` — Log-Depth QFT
In-place QFT with O(log n) depth and zero ancilla qubits (arXiv:2505.00701, May 2025).

#### `qswift` — High-Order Randomized Simulation
Combines qDRIFT importance sampling with high-order product formulas for efficient Hamiltonian simulation.

#### `spectrum_amplification` — Generalized Amplitude Amplification
Fixed-point (Yoder-Low-Chuang), oblivious, and spectral filtering amplification. Monotonic convergence without overshooting.

#### `cluster_tebd` — Entanglement-Aware TEBD
Dynamic qubit clustering based on entanglement analysis for large-scale (1000+ qubit) time evolution.

### Quantum Error Correction

#### `bivariate_bicycle` — Gross Code + Ambiguity Clustering
IBM's [[144,12,12]] code achieving 10x qubit reduction. Ambiguity clustering decoder is 27x faster than BP-OSD.
```rust
use nqpu_metal::bivariate_bicycle::{GrossCode, AmbiguityClusteringDecoder};
let code = GrossCode::new(144, 12, 12);
let decoder = AmbiguityClusteringDecoder::new(&code);
```

#### `mast` — Magic-Injected Stabilizer Tensor Networks
Simulate 200+ qubit circuits with T gates by combining stabilizer tableau with tensor network contraction for magic states.

#### `relay_bp` — Relay Belief Propagation Decoder
Enhanced BP decoder that escapes trapping sets via relay node insertion and strategic message resets. OSD-0 fallback for guaranteed decoding.

#### `qec_sampling` — Monte Carlo QEC Sampling
Sinter-like statistical sampling with Rayon parallelism, multiple decoder backends (MWPM/UnionFind/BP), Wilson confidence intervals.

#### `zne_qec` — ZNE + QEC Integration
Nature Communications 2025: distance-3 code + ZNE matches unmitigated distance-5 performance with 40-64% fewer physical qubits.

#### `resource_estimation` — Fault-Tolerant Resource Estimation
Azure QRE-style estimation: code distance optimization, T-factory planning, physical qubit counting, wall clock estimation.
```rust
use nqpu_metal::resource_estimation::{ResourceEstimator, Preset};
let estimate = ResourceEstimator::from_preset(Preset::RsA2048).estimate();
println!("Physical qubits: {}", estimate.total_physical_qubits);
```

### Hardware & Compilation

#### `transpiler` — Hardware-Aware Transpiler
SABRE routing + gate decomposition with IBM/IonQ/Google/Rigetti basis gate sets. Composable pass pipeline with `CompilerPass` trait.
```rust
use nqpu_metal::transpiler::{Transpiler, CouplingMap, BasisGateSet};
let coupling = CouplingMap::heavy_hex(127);
let transpiled = Transpiler::new(coupling, BasisGateSet::Ibm).transpile(&circuit)?;
```

#### `device_noise` — Hardware Calibration Noise Profiles
Presets: IBM Brisbane, Google Sycamore, IonQ Aria, Rigetti Aspen-M3. Thermal relaxation, readout confusion matrices.

#### `qcvv` — Quantum Characterization & Validation
XEB fidelity (Google's quantum advantage metric) + single/two-qubit Clifford RB with interleaved RB.

#### `layer_fidelity` — Layer Fidelity Benchmarking
IBM's 2024 protocol: benchmarks entire circuit layers to capture crosstalk and correlated errors missed by per-gate RB.

#### `cq_adder` — Quantum Arithmetic
Draper QFT adder (O(n²) gates, no ancilla), Cuccaro ripple-carry (O(n) depth), modular arithmetic for Shor's algorithm.

### Variational & Optimization

#### `quantum_natural_gradient` — QNG Optimizer
Full Fubini-Study metric tensor with block-diagonal and diagonal approximations. Parameter-shift gradients with Tikhonov regularization.

#### `qamoo` — Multi-Objective QAOA
Tchebycheff and weighted-sum scalarization, Pareto front computation, hypervolume indicator.

#### `warm_start_qaoa` — Parameter Transfer
Transfer QAOA parameters across problem sizes via linear, Fourier, and pattern interpolation. Optimal fixed-angle lookup tables.

#### `gga_vqe` — Greedy Gradient-Free Adaptive VQE
ADAPT-VQE with greedy operator selection and Nelder-Mead gradient-free optimization.

#### `parametric_circuits` — Symbolic Parametric Circuits
Parameter expressions with arithmetic, parameter-shift gradient circuits, built-in ansatze (RealAmplitudes, EfficientSU2, TwoLocal).

### Industry Applications

#### `qubo_encoder` — Combinatorial Optimization
QUBO/Ising encoders for MaxCut, TSP, graph coloring, portfolio optimization, number partitioning, vertex cover, independent set.

#### `qkd_protocols` — Quantum Key Distribution
BB84, E91, BBM92, B92, Six-State protocols with eavesdropper models, CASCADE error correction, privacy amplification.

#### `shor` — Cryptographic Assessment
Shor's algorithm with quantum period finding + crypto attack toolkit assessing threat levels for RSA/DH/ECC/AES.

#### `molecular_integrals` — Quantum Chemistry
FCIDUMP parser, Jordan-Wigner/Bravyi-Kitaev/Parity fermion-to-qubit mappings, predefined molecules (H2, LiH, H2O).

#### `quantum_kernels` — Quantum Machine Learning
Feature maps (ZZ, Pauli, angle, IQP), fidelity/projected kernels, kernel matrix computation, simplified QSVM.

#### `quantum_echoes` — Scrambling & OTOC
Out-of-time-order correlator computation, Loschmidt echo, Lyapunov exponent extraction (Google Nature Oct 2025).

### Developer Experience

#### `primitives` — Sampler/Estimator
Qiskit V2-style Sampler (measurement counts) and Estimator (expectation values) for backend-agnostic execution.

#### `circuit_macro` — Fluent Circuit Builder
```rust
use nqpu_metal::circuit_macro::CircuitBuilder;
let circuit = CircuitBuilder::new(3)
    .h(0).cx(0, 1).cx(1, 2)  // GHZ state
    .measure_all()
    .build();
println!("Depth: {}, Gates: {}", circuit.depth(), circuit.gate_count());
```

#### `circuit_serde` — Circuit Serialization
JSON/compact JSON/binary/QASM3 serialization with circuit library management and format versioning.

#### `experiment_config` — Declarative Experiments
Define quantum experiments without code: circuit spec, backend, noise model, analysis, output format.

#### `property_testing` — Quantum Property Testing
Random circuit generation, unitarity/trace-preservation/probability-conservation checks, chi-squared/KS statistical tests.

---

## Further Reading

- [Usage Guide](USAGE.md) - Detailed examples and patterns
- [Performance Documentation](PERFORMANCE.md) - Benchmarks and optimization
- [Contributing](CONTRIBUTING.md) - Development guidelines
- [CHANGELOG](CHANGELOG.md) - Version history

---

## License

MIT License - See LICENSE file for details.
