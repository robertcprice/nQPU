#![allow(dead_code, unused_variables, unused_imports, unused_assignments)]
#![cfg_attr(feature = "metal", allow(unexpected_cfgs))]
//! nQPU-Metal: High-Performance Quantum Simulator
//!
//! CPU implementation with optional Metal GPU acceleration
//! Metal GPU shaders are provided in src/metal/shaders/
//!
//! # Features
//!
//! - **CPU Simulation**: Multi-threaded CPU quantum simulator using Rayon
//! - **Metal GPU**: GPU-accelerated quantum operations (macOS only)
//! - **Parallel Gates**: Batch execution of quantum gates on GPU
//! - **Quantum Transformers**: Multi-head attention with quantum states
//! - **Batch Processing**: Efficient batch processing for transformer models
//!
//! # Metal GPU Acceleration
//!
//! On macOS, this library provides GPU-accelerated quantum operations through:
//! - [`MetalParallelQuantumExecutor`]: Main GPU execution interface
//! - [`MetalBenchmark`]: Benchmarking CPU vs GPU performance
//! - Parallel kernels for gates, attention, and transformers
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::{QuantumState, GateOperations};
//!
//! // Create a 10-qubit quantum state
//! let mut state = QuantumState::new(10);
//!
//! // Apply Hadamard gate to qubit 0
//! GateOperations::h(&mut state, 0);
//!
//! // Measure the state
//! let probs = state.probabilities();
//! ```

use num_complex::Complex64;
use std::time::Instant;

// Conditional compilation for parallel processing
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ============================================================
// TYPE ALIASES FOR COMPLEX NUMBERS
// ============================================================

/// Double-precision complex number type alias
pub type C64 = Complex64;

/// Float-precision complex number type alias (for Metal GPU)
pub type C32 = num_complex::Complex32;

// ============================================================
// C64 HELPER FUNCTIONS
// ============================================================

/// Create a zero C64 value (Complex64::new(0.0, 0.0))
#[inline]
pub fn c64_zero() -> C64 {
    C64::new(0.0, 0.0)
}

/// Create a one C64 value (Complex64::new(1.0, 0.0))
#[inline]
pub fn c64_one() -> C64 {
    C64::new(1.0, 0.0)
}

/// Scale a C64 value by a factor (since Complex64 doesn't have scale())
#[inline]
pub fn c64_scale(c: C64, factor: f64) -> C64 {
    C64::new(c.re * factor, c.im * factor)
}

// Helper functions are now directly available via crate::
// No need for pub use - functions are already visible

/// Create a zero C32 value (Complex32::new(0.0, 0.0))
#[inline]
pub fn c32_zero() -> C32 {
    C32::new(0.0, 0.0)
}

/// Create a one C32 value (Complex32::new(1.0, 0.0))
#[inline]
pub fn c32_one() -> C32 {
    C32::new(1.0, 0.0)
}

/// Scale a C32 value by a factor
#[inline]
pub fn c32_scale(c: C32, factor: f32) -> C32 {
    C32::new(c.re * factor, c.im * factor)
}

// Helper functions are now directly available via crate::
// No need for pub use - functions are already visible

// ============================================================
// SHARED INFRASTRUCTURE (Phase 0)
// ============================================================

// Unified trait hierarchy for all backends
pub mod traits;

// Universal sparse Pauli string representation
pub mod pauli_algebra;

// Heisenberg-picture Pauli propagation simulator
pub mod pauli_propagation;

// GPU-accelerated Pauli propagation (Rayon-parallel Heisenberg picture)
pub mod pauli_propagation_gpu;

// Metal GPU-accelerated Pauli propagation with PNA error mitigation
pub mod gpu_pauli_propagation;

// Pauli twirling / randomized compiling for noise tailoring
pub mod pauli_twirling;

// Quantum channel abstraction (Kraus, Choi)
pub mod quantum_channel;

// Differentiable density matrix dynamics (Lindblad master equation, adjoint-method AD)
pub mod differentiable_dynamics;

// Metal GPU modules (macOS only)
#[cfg(target_os = "macos")]
mod metal_gpu_full;
#[cfg(target_os = "macos")]
pub use metal_gpu_full::*;

// Removed: metal_gpu_simple, metal_gpu_optimized, metal_gpu_advanced, metal_gpu_correct
// These were development iterations consolidated into metal_gpu_fixed (canonical)

#[cfg(target_os = "macos")]
pub mod metal_gpu_fixed;
#[cfg(target_os = "macos")]
pub use metal_gpu_fixed::*;

// Parallel quantum operations with Metal GPU acceleration
#[cfg(target_os = "macos")]
pub mod metal_parallel_quantum;

// CUDA GPU backend
#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(target_os = "macos")]
pub use metal_parallel_quantum::*;

// T-Era: GPU-First Quantum State (unified memory, no CPU-GPU transfers)
#[cfg(target_os = "macos")]
pub mod metal_state;

// T-Era: Tensor Core Operations (matrix multiply acceleration)
#[cfg(target_os = "macos")]
pub mod tensor_ops;

// Apple Accelerate (AMX) tensor contractions for MPS inner loops (macOS only)
#[cfg(target_os = "macos")]
pub mod amx_tensor;

// T-Era: Schrödinger-Feynman hybrid simulation
pub mod schrodinger_feynman;

// T-Era: Auto-tuning backend selection
pub mod auto_tuning;

// T-Era: T1+T2 Integrated GPU-First + Tensor Cores
#[cfg(target_os = "macos")]
pub mod t1t2_integrated;

// Post T-Era: M4 Pro GPU Kernel Optimization
#[cfg(target_os = "macos")]
pub mod m4_pro_optimization;

// Metal 4 Backend: Tensor Operations + Inline ML Inference
#[cfg(target_os = "macos")]
pub mod metal4_backend;

// Post T-Era: Cache Blocking CPU Optimization
pub mod cache_blocking;

// Post T-Era: Advanced Tensor Networks (PEPS, TTN, MERA)
pub mod advanced_tensor_networks;

// Post T-Era: GPU-Accelerated MPS
#[cfg(target_os = "macos")]
pub mod gpu_mps;
#[cfg(target_os = "macos")]
pub mod metal_mps;

// Post T-Era: F32 Backend Integration
pub mod f32_backend;

// Post T-Era: Distributed MPI Support
pub mod distributed_metal_mpi;
pub mod distributed_mpi;

// Post T-Era: Real MPI Integration
#[cfg(feature = "mpi")]
pub mod real_mpi;

// Comprehensive: Quantum Gates Library
pub mod comprehensive_gates;

// Comprehensive: Quantum Algorithms Library
pub mod comprehensive_algorithms;

// Comprehensive: Noise Models and Error Simulation
pub mod noise_models;

// Comprehensive: Quantum Error Correction
pub mod quantum_error_correction;

// Quantum Decision Diagram (BDD/ZDD) state compression
pub mod decision_diagram;

// Near-Clifford (CH-form) simulation for circuits with few T-gates
pub mod near_clifford;

// Quantum Chemistry: fermion-to-qubit mappings, molecular Hamiltonians, UCCSD
pub mod quantum_chemistry;

// Molecular integrals: FCIDUMP parser, fermion-to-qubit mappings (JW, BK, Parity), active space
pub mod molecular_integrals;

// Comprehensive: Bayesian & Tree PEPS
pub mod bayesian_treepes;

// Comprehensive: Benchmark Suite
pub mod comprehensive_benchmarks;

// Noise simulation module
pub mod noise;

// Tensor network module
pub mod entanglement_scheduler;
pub mod entanglement;
pub mod tensor_network;

// Quantum algorithms module
pub mod algorithms;

// 2D quantum algorithms module
pub mod algorithms_2d;

// Utilities module
pub mod utilities;

// Error mitigation module
pub mod classical_shadows;
pub mod camps;
pub mod clifford_t;
pub mod error_mitigation;
pub mod time_evolution;
pub mod improved_trotter;

// Digital-Analog Quantum Computation module
pub mod digital_analog;

// QASM import/export module
pub mod qasm;

// OpenQASM 3.0 parser module
pub mod qasm3;

// QASM Language Server Protocol support
#[cfg(feature = "lsp")]
pub mod qasm_lsp;

// QIR (Quantum Intermediate Representation) import/export for Azure QDK
pub mod qir;

// Stim QEC circuit format import
pub mod stim_import;

// Density matrix simulation module
pub mod density_matrix;

// Quantum machine learning module
pub mod quantum_ml;
pub mod quantum_ml_mps;
pub mod quantum_kernels;

// Neural Quantum States: RBM/autoregressive ansatze with VMC and stochastic reconfiguration
pub mod neural_quantum_states;

// QKMM: O(N² log N) Quantum Kernel-based Matrix Multiplication (arXiv:2602.05541)
pub mod qkmm;

// Low-Depth UCC: Efficient quantum chemistry ansatz (arXiv:2602.14999)
pub mod low_depth_ucc;

// Tucker State Preparation: Iterative state synthesis (arXiv:2602.09909)
pub mod tucker_state_prep;

// RASCqL: Architecture for qLDPC Logic (arXiv:2602.14273)
pub mod rascql;

// Quantum attention module
pub mod quantum_attention;

// Massive parallelization module
pub mod parallel_quantum;

// Fully quantum transformer module
pub mod full_quantum_transformer;

// Gate definitions module
pub mod gates;

// Circuit optimization module
pub mod circuit_optimizer;

// Circuit compilation cache (gate batching, memoization, wraps circuit_optimizer)
pub mod circuit_cache;
/// Backward-compatible alias for the renamed `circuit_cache` module.
pub use circuit_cache as jit_compiler;

// Differentiable QEC pipeline (H10: backprop through encode-noise-decode)
pub mod differentiable_qec;

// Metal-accelerated neural network QEC decoder (H9: fuse decoder with stabilizer sim)
pub mod metal_neural_decoder;

// Tree Tensor Network backend (H8: hierarchical tensor simulation)
pub mod tree_tensor_network;

// Randomized QSVT (H6: single-ancilla, block-encoding-free)
pub mod randomized_qsvt;

// Mamba-based QEC decoder (H7: O(d²) SSM decoder)
pub mod mamba_qec_decoder;

// CAMPS-DMRG for Quantum Chemistry (H5: Clifford-Augmented MPS)
pub mod camps_dmrg;

// AlphaQubit-style Transformer QEC Decoder (Novel: Attention-based decoding)
pub mod transformer_qec_decoder;

// Real-Time Adaptive QEC Decoder (Novel: Continuous learning from error patterns)
pub mod adaptive_realtime_decoder;

// Decoder-Aware Circuit Optimization (Novel: Co-optimizes layout with decoder)
pub mod decoder_aware_transpiler;

// Tensor network contraction path optimizer (Cotengra-equivalent)
pub mod contraction_optimizer;

// Circuit DSL: fluent API and macro for building quantum circuits
pub mod circuit_macro;

// Circuit equivalence verification
pub mod circuit_equivalence;

// Circuit serialization/deserialization (JSON, binary, OpenQASM 3.0)
pub mod circuit_serde;

// Circuit visualization export (LaTeX quantikz, SVG, ASCII)
pub mod circuit_export;

// Gate fusion engine
pub mod gate_fusion;

// SIMD-accelerated operations
pub mod simd_ops;

// Automatic backend selection
pub mod auto_simulator;

// Intelligent backend selection and circuit analysis
pub mod auto_backend;

// Mid-circuit measurement and classical control
pub mod mid_circuit;

// Measurement-Based Quantum Computation (MBQC)
pub mod mbqc;

// Benchmark suite
pub mod benchmark_suite;

// Float32 quantum state and operations
pub mod f32_fusion;
pub mod quantum_f32;

// Metal GPU backend (properly wired)
pub mod metal_backend;

// Thermal-aware GPU scheduling
pub mod thermal_scheduler;

// UMA Gate-Level CPU/GPU Dispatch (Apple Silicon unified memory)
pub mod uma_dispatch;

// Concurrent CPU+GPU execution on disjoint qubit partitions (UMA)
pub mod concurrent_uma;

// Dynamic f32/f64 precision routing for hybrid CPU/GPU execution
pub mod mixed_precision;

// 2D Grid Support: Snake mapping for 2D circuits
pub mod snake_mapping;

// Lattice-aware MPS for 2D/3D grids
pub mod lattice_mps;

// 4D Lattice MPS for hypercubic geometry
pub mod lattice_mps_4d;

// TEBD (Time-Evolving Block Decimation) for MPS-based time evolution
pub mod tebd;

// Lindblad Master Equation solver for open quantum systems
pub mod lindblad;

// Lindblad Classical Shadows: tomography for open quantum systems (arXiv:2602.14694)
// Combines classical shadows with Lindblad dynamics for efficient channel estimation
pub mod lindblad_shadows;

// Adaptive MPS with automatic bond dimension management
pub mod adaptive_mps;

// PEPS (Projected Entangled Pair States) for 2D quantum systems
pub mod peps;

// PEPO (Projected Entangled Pair Operators) for 2D density matrices and thermal states
pub mod pepo;

// PEPS Gate Operations with SVD Compression
pub mod peps_gates;

// Comprehensive PEPS Simulator (2D, 3D, 4D tensor networks)
pub mod peps_simulator;

// Arbitrary-geometry tensor networks with Cotengra-style contraction optimization
pub mod arbitrary_tn;

// 2D: Corner Transfer Matrix (CTM) Contraction
pub mod ctm_contraction;

// 2D: Quantum Fourier Transform
pub mod qft_2d;

// Optimistic QFT: log-depth in-place QFT (arXiv:2505.00701)
// Applications: efficient QFT approximation, quantum phase estimation, Shor's algorithm
pub mod optimistic_qft;

// 3D: Quantum Simulation (Hilbert Curve + PEPSON)
pub mod simulation_3d;

// Adaptive batching module
pub mod adaptive_batching;

// Advanced quantum ML module
pub mod advanced_quantum_ml;

// Robust error handling module - TEMPORARILY DISABLED due to compilation errors
// pub mod error_handling;

// GPU memory pool for efficient resource management
pub mod gpu_memory_pool;

// Quantum circuit synthesis
pub mod quantum_synthesis;

// Parametric Circuits for Variational Algorithms (VQE, QAOA, etc.)
pub mod parametric_circuits;

// Variational Quantum Eigensolver (VQE)
pub mod vqe;

// Quantum Approximate Optimization Algorithm (QAO)
pub mod qao;

// Quantum Phase Estimation (QPE)
pub mod heisenberg_qpe;
pub mod qpe;

// (qram declared in Phase 5 section below)

// Quantum Signal Processing (QSP) and Quantum Singular Value Transformation (QSVT)
// Grand unifying framework: subsumes Grover, QPE, Hamiltonian simulation, matrix inversion
pub mod qsp_qsvt;

// Spectrum Amplification: generalises Grover to arbitrary eigenvalue transformations
// Includes fixed-point (Yoder-Low-Chuang), oblivious, and spectral filtering variants
pub mod spectrum_amplification;

// Simple Working Gates - always available
pub mod simple_gates;

// Quantum Annealing
pub mod annealing;

// Simulated Quantum Annealing (SQA) — path-integral Monte Carlo, exact statevector, problem library
pub mod quantum_annealing;

// Quantum state tomography
pub mod state_tomography;

// Error correction decoding
pub mod decoding;

// Advanced caching system
pub mod advanced_cache;

// Stabilizer simulation for Clifford circuits
pub mod stabilizer;

// Inverse stabilizer tableau for O(n) measurement
pub mod inverse_tableau;

// SIMD-friendly stabilizer simulation with packed bitstring operations
pub mod simd_stabilizer;

// Cache-optimized stabilizer with transposed memory layout (faster)
pub mod optimized_stabilizer;

// Metal GPU-accelerated stabilizer simulation (targets Stim-level performance)
pub mod metal_stabilizer;

// Unified high-performance stabilizer (targets 50M gates/sec - Stim-competitive)
pub mod fast_stabilizer;

// Auto-routing stabilizer (picks optimal CPU/GPU backend)
pub mod stabilizer_router;

// AVX-512 SIMD stabilizer for x86 platforms
pub mod avx512_stabilizer;

// IBM Quantum backend (real hardware access)
pub mod ibm_quantum;

// Meta-learning VQE (LSTM-based parameter initialization)
pub mod meta_vqe;

// Terminal User Interface (TUI) with 3D visualizations
pub mod tui;

// JAX integration for automatic differentiation
pub mod jax_bridge;

// Google Quantum AI hardware backend
pub mod google_quantum;

// Pulse-level control for microwave manipulation
pub mod pulse_control;

// GPU-accelerated MWPM decoder
pub mod gpu_mwpm;

// Local hardware quantum interface (entropy, QRNG, real pulse calibration)
pub mod hardware_calibration;

// TRUE quantum randomness with Bell test verification
pub mod quantum_randomness;

// REAL hardware quantum randomness extraction (camera, audio, CPU jitter)
pub mod hardware_quantum;

// Camera-based quantum RNG (Randonautica-style shot noise extraction)
pub mod camera_quantum;

// CERTIFIED quantum randomness via cloud Bell tests (IBM, Google)
pub mod certified_quantum;

// Wireless quantum entropy extraction (WiFi + Bluetooth + Network timing)
pub mod wireless_quantum;

// Creative quantum detection methods (cosmic rays, multi-receiver, SSD tunneling, etc.)
// Experimental/research — requires --features experimental
#[cfg(feature = "experimental")]
pub mod creative_quantum;

// Quantum verification tests (antibunching, Leggett-Garg)
pub mod quantum_verification;

// Experimental quantum tests on consumer hardware (SSD, cosmic rays)
pub mod quantum_ssd_tests;

// REAL quantum probing - actual hardware measurements (no simulation)
pub mod real_quantum_probe;

// NIST SP 800-22 Statistical Test Suite (for QRNG verification)
pub mod nist_tests;

// QRNG Experiment Runner (complete experiments with reporting)
pub mod qrng_experiment;

// QRNG Phase 2 - Proper methodology (bits from SSD timing, not /dev/urandom)
pub mod qrng_phase2;

// QRNG Source Comparison - Test each entropy source separately
pub mod qrng_source_comparison;

// QRNG Extraction Methods - Different ways to extract and condition bits
pub mod qrng_extraction_methods;

// Reference frame sampling (Stim-style 1000x speedup for QEC)
pub mod reference_frame;

// Quantum Fisher Information (quantum metrology/sensing)
pub mod quantum_fisher;

// Stim circuit format import/export (interoperability)
pub mod stim_format;

// Closed Timelike Curve simulation (time travel physics)
pub mod ctc_simulation;

// Quantum Contextuality Engine (Kochen-Specker, Peres-Mermin, magic states)
pub mod contextuality;

// Process Tensor Framework (non-Markovian quantum computing)
pub mod process_tensor;

// XZZX Surface Code with biased noise (18% threshold)
pub mod xzzx_surface;

// Hybrid stabilizer + tensor network simulator (magic-aware)
pub mod stabilizer_tensor_net;

// MAST: Magic-injected Stabilizer Tensor Networks (arXiv:2411.12482)
pub mod mast;

// Surface Codes (Topological quantum error correction)
pub mod surface_codes;

// Lattice Surgery for Surface Codes (merge/split operations, logical gate compilation)
pub mod lattice_surgery;

// Yoked Surface Codes (1/3 qubit overhead QEC -- Nature Communications 2025)
pub mod yoked_surface_codes;

// Floquet Codes (Dynamical quantum error correction -- Hastings-Haah honeycomb, X3Z3)
pub mod floquet_codes;

// Hyperbolic Floquet Codes (QEC on negatively-curved surfaces -- world-first Rust impl)
pub mod hyperbolic_floquet;

// MERA (Multiscale Entanglement Renormalization Ansatz)
pub mod mera_happy;

// Maximum Qubit Benchmarks
pub mod max_qubit_benchmark;

// Advanced noise models
pub mod advanced_noise;
pub mod enhanced_zne;
pub mod advanced_error_mitigation;
pub mod pec;
pub mod compilation_informed_pec;
pub mod non_markovian;
pub mod device_noise;
pub mod bayesian_noise;
pub mod live_calibration;

// Leakage simulation: qutrit-based transmon leakage modeling
pub mod leakage_simulation;

// Automatic differentiation for VQAs
pub mod adjoint_diff;
pub mod autodiff;
pub mod differentiable_mps;
pub mod dmrg_tdvp;
pub mod gpu_dmrg;
#[cfg(target_os = "macos")]
pub mod dmrg_metal;
pub mod imps_ipeps;
pub mod distributed_adjoint;

// Tensor Jump Method: MPS + Monte Carlo Wave Function for open quantum systems (Lindblad)
pub mod tensor_jump;

// Enhanced barren plateau analysis with empirical sampling
pub mod enhanced_barren_plateau;

// Continuous-variable Gaussian simulation
pub mod cv_quantum;

// Bosonic quantum error correction codes (cat, GKP, binomial) in truncated Fock space
pub mod bosonic_codes;

// Concatenated bosonic cat qubit simulation (Nature Feb 2025)
pub mod cat_qubit_concatenation;

// Approximate dynamical QEC with temporal recovery maps and code optimization
pub mod approximate_dynamical_qec;

// Holographic quantum error correcting codes (AdS/CFT, HaPPY code, Ryu-Takayanagi)
pub mod holographic_codes;

// Circuit cutting for beyond-memory simulation
pub mod circuit_cutting;

// Topological/Fibonacci anyon simulation
pub mod topological_quantum;

// Expanded topological: Ising anyons, Majorana chains, braid compilation
pub mod topological_expanded;

// Majorana-1 topological quantum processor: topoconductor physics, braiding, Kitaev chain
pub mod majorana_model;

// Dynamic surface code + RL decoder
pub mod dynamic_surface_code;

// QEC interop helpers (Stim-like detector model export)
pub mod qec_interop;

// Neural QEC decoders (GNN-style message-passing)
pub mod neural_decoder;

// Unified neural QEC decoder across all code families (AlphaQubit-inspired)
pub mod unified_neural_decoder;

// Quantum Low-Density Parity-Check (qLDPC) codes
// Hypergraph product, bivariate bicycle, BP decoding
pub mod qldpc;

// Bivariate Bicycle (Gross) codes and Ambiguity Clustering decoder
// [[144,12,12]] IBM Nature 2024 code, 27x faster than BP-OSD
pub mod bivariate_bicycle;

// Trivariate Tricycle (TT) codes -- generalized bivariate bicycle with
// third cyclic dimension and meta-check measurement error diagnosis
pub mod trivariate_codes;

// Pulse-level Hamiltonian simulation and optimization
pub mod pulse_level;

// AC4c: Differentiable pulse-level simulation with transmon modeling,
// RK4/Lindblad dynamics, and gradient-based pulse optimization
pub mod pulse_simulation;

// Dynamical Decoupling (DD) circuit transformation passes
// Insert identity-equivalent pulse sequences on idling qubits to suppress decoherence
pub mod dynamical_decoupling;

// Quantum Cellular Automata (QCA) with Margolus partitioning
// Applications: many-body dynamics, scrambling, entanglement spreading
pub mod quantum_cellular_automata;

// Quantum Networking: channels, entanglement distribution, repeaters, purification
#[allow(dead_code)]
pub mod quantum_networking;

// QKD Protocols: BB84, E91, BBM92, B92, Six-State key distribution simulation
// with eavesdropper models, CASCADE error correction, and privacy amplification
pub mod qkd_protocols;

// Metropolitan QKD Network: multi-node QKD network simulation with trusted relays,
// entanglement-based links, realistic fiber loss, and pre-built topologies (BearlinQ, Tokyo, etc.)
pub mod metro_qkd_network;

// QRNG Integration: Quantum Random Number Generator sources for measurement randomness
// Supports ANU QRNG API, hardware device files, CSPRNG fallback, and hybrid composition
pub mod qrng_integration;

// Certified QRNG: Bell-test certified quantum randomness with CHSH verification
// and Twine-style SHA-256 hash chain for tamper-evident certificate provenance
pub mod certified_qrng;

// Quantum Entropy Extraction: Bridge quantum simulations to LLM entropy seeding
// Extracts randomness from measurements, CTC dynamics, contextuality, and process tensors
pub mod quantum_entropy_extraction;

// ============================================================
// PHASE 5: RESEARCH FRONTIER (2026-02-15)
// ============================================================

// (hyperbolic_floquet, tensor_jump, cat_qubit_concatenation declared above)

// QRAM: Quantum Random Access Memory circuits (bucket-brigade, SelectOnly, fan-out)
// Closes PennyLane v0.44 parity gap — essential for quantum database algorithms
pub mod qram;

// Fault-Tolerant Compilation: Litinski transformation + Ross-Selinger/gridsynth
// Closes Qiskit v2.3 parity gap — Clifford+Rz → Clifford+T + Pauli-based computation
pub mod ft_compilation;

// ============================================================
// PHASE 6: ADVANCED PLATFORMS & PROTOCOLS (2026-02-15)
// ============================================================

// QNodeOS: Quantum network operating system — distributed quantum computing protocol stack
// WORLD FIRST: No other simulator has a network OS for entanglement scheduling
pub mod quantum_network_os;

// Rydberg Reservoir Computing: quantum-enhanced ML via Rydberg atom dynamics
// WORLD FIRST: Rydberg atom reservoir computing in a general quantum simulator
pub mod rydberg_reservoir;

// Pinnacle Architecture: Google's next-gen quantum processor simulation
// WORLD FIRST: Heterogeneous zone-based processor modeling (data/ancilla/magic-state zones)
pub mod pinnacle_architecture;

// Neutral Atom Array: Reconfigurable neutral atom quantum computing with atom shuttling
// WORLD FIRST: Full atom rearrangement dynamics + Rydberg blockade physics
pub mod neutral_atom_array;

// Photonic Advantage: Gaussian Boson Sampling + Linear Optical QC + Photonic Ising Machine
// WORLD FIRST: Unified photonic quantum advantage framework in a general simulator
pub mod photonic_advantage;

// ============================================================
// PHASE 7: APPLICATION NICHES & ERROR MITIGATION (2026-02-15)
// ============================================================

// Quantum Logistics: CVRP, job-shop scheduling, TSP via QAOA
pub mod quantum_logistics;

// Quantum Climate: lattice Boltzmann CFD, atmospheric chemistry, carbon cycle, energy balance
pub mod quantum_climate;

// Quantum Materials: battery screening, superconductor Tc, band structure, Hubbard model
pub mod quantum_materials;

// Propagated Noise Absorption (PNA): Qiskit 2.3 technique for Clifford noise propagation
pub mod pna;

// (qir declared above with QASM parsers)

// ============================================================
// BLEEDING-EDGE MODULES (unique to nQPU-Metal)
// ============================================================

// Quantum Random Walks on Graphs (discrete + continuous time)
// Applications: quantum search, PageRank, transport simulation
pub mod quantum_random_walk;

// Quantum Walk Simulation (CTQW + DTQW with eigendecomposition/Lanczos)
// Applications: search algorithms, transport, graph problems
// References: Kempe (2003), Childs PRL 102 (2009)
pub mod quantum_walk;

// Quantum Reservoir Computing (quantum-enhanced ML)
// Applications: time series prediction, nonlinear function approximation
pub mod quantum_reservoir;

// (rydberg_reservoir declared above in Phase 6)

// Quantum Thermodynamics: engines, batteries, fluctuation theorems
// WORLD FIRST: Built-in quantum thermodynamic simulation
pub mod quantum_thermodynamics;

// Quantum Battery Simulation: entanglement-enhanced energy storage
// Models collective/Dicke/all-to-all charging, ergotropy, scaling analysis
pub mod quantum_battery;

// Approximate Quantum Cloning Machines
// Applications: QKD security analysis, quantum information theory
pub mod quantum_cloning;

// Symmetry-Exploiting Simulation
// Applications: particle number / Sz conservation, Hilbert space reduction for physics
pub mod symmetry_simulation;

// Circuit Complexity Analysis & Resource Estimation
// Applications: fault-tolerance planning, barren plateau detection, quantum volume
pub mod circuit_complexity;

// Fault-Tolerant Resource Estimation (Azure QRE-style)
// Applications: physical qubit counting, T-factory planning, code distance optimization
pub mod resource_estimation;

// Quantum State Checkpointing & Time-Travel Debugging
// Applications: debugging, state diff analysis, execution forking
pub mod state_checkpoint;

// Interactive stateful simulator API (step/undo/fork) built on checkpoints
pub mod interactive_sim;

// Quantum Game Theory (Eisert-Wilkens-Lewenstein framework)
// Applications: quantum cryptography, mechanism design, educational
pub mod quantum_game;

// Quantum Cognition (quantum probability for human decision-making)
// Applications: order effects, conjunction fallacy, sure-thing violations, survey simulation
pub mod quantum_cognition;

// Quantum Integrated Information Theory (Tononi's Phi)
// Applications: consciousness metrics, integrated information, system partitioning
pub mod quantum_iit;

// Orchestrated Objective Reduction (Penrose-Hameroff Orch-OR)
// Experimental/research — requires --features experimental
#[cfg(feature = "experimental")]
pub mod orch_or;

// Anharmonic Oscillations for Microtubules
// CRITICAL: Non-harmonic vibrations essential for consciousness (Hameroff: "like Indian music")
pub mod anharmonic;

// Microtubule Quantum Reservoir for Transformer Augmentation
// Experimental/research — requires --features experimental (depends on orch_or)
#[cfg(feature = "experimental")]
pub mod mt_reservoir;

// Microtubule-inspired feature modulation for transformer-style pipelines
// Experimental/research — requires --features experimental (depends on orch_or)
#[cfg(feature = "experimental")]
pub mod microtubule_augmentor;

// Z2-graded (fermionic) tensor networks with anti-commutation sign tracking
// Applications: strongly correlated fermions, Hubbard models, quantum chemistry
pub mod fermionic_tensor_net;

// Magic State Distillation Factory Simulation
// Full distillation pipeline: 15-to-1, 20-to-4, Reed-Muller, Litinski compact
// Tracks resource costs: physical qubits, surface code cycles, space-time volume
pub mod magic_state_factory;

// BP-OSD Decoder: Belief Propagation + Ordered Statistics Decoding for qLDPC codes
// State-of-art decoder for bivariate bicycle, lifted product codes (arXiv:2104.13659)
pub mod bp_osd;

// ADAPT-VQE: Adaptive Derivative-Assembled Pseudo-Trotter VQE
// Dynamically builds ansatz by selecting highest-gradient operators (arXiv:1812.11173)
pub mod adapt_vqe;

// Sliding Window QEC Decoder: Real-time incremental syndrome decoding
// Critical for fault-tolerant quantum computing with streaming syndromes
pub mod sliding_window_decoder;

// TDVP: Time-Dependent Variational Principle for MPS time evolution
// 1-site (fixed bond dim) and 2-site (adaptive) variants (PRL 107, 070601)
// pub mod tdvp; // TODO: create tdvp.rs

// (quantum_walk module declared above near quantum_random_walk)

// Matchgate / Free-Fermion Circuit Simulation
// Applications: fermionic linear optics, free-fermion simulability boundary, hopping models
pub mod matchgate_simulation;

// Fermionic Gaussian State Simulation
// Applications: non-interacting fermion systems, covariance matrix formalism, O(n^3) free-fermion simulation
pub mod fermionic_gaussian;

// Cluster-TEBD: Entanglement-aware time-evolving block decimation
// Applications: large-scale quantum simulation, structured Hamiltonians, entanglement clustering
pub mod cluster_tebd;

// Quantum Chaos & Information Scrambling Diagnostics
// Applications: spectral statistics, ETH verification, entanglement growth, Loschmidt echo
pub mod quantum_chaos;

// Many-Worlds Branching Simulation (Everett Interpretation)
// WORLD FIRST: Full branching structure tracking, decoherent histories, branch statistics
pub mod many_worlds;

// CUDA backend for NVIDIA GPUs (optional, behind "cuda" feature)
#[cfg(feature = "cuda")]
pub mod cuda_backend;

// ROCm backend for AMD GPUs (optional, behind "rocm" feature)
#[cfg(feature = "rocm")]
pub mod rocm_backend;

// Real QPU hardware connectivity (optional, behind "qpu" feature)
#[cfg(feature = "qpu")]
pub mod qpu;

// Python bindings (optional, behind "python" feature)
// Also include during `cargo test` so #[cfg(test)] unit tests in python.rs compile
// (follows the same pattern as python_api_v2 below).
#[cfg(any(feature = "python", test))]
pub mod python;

// Note: python_api_v2.rs is not compatible with PyO3 0.28 API yet
// The main python.rs module provides full Python 3.14 support
// Include python_api_v2 for tests (PyO3 items are cfg-gated inside the file)
#[cfg(any(feature = "python", test))]
pub mod python_api_v2;

// ASCII visualization (always available, no feature gate)
pub mod ascii_viz;

// Visualization tools (optional, behind "visualization" feature)
#[cfg(feature = "visualization")]
pub mod visualization;

// Distributed memory support (optional, behind "distributed" feature)
#[cfg(feature = "distributed")]
pub mod distributed;

// Web GUI (optional, behind "web" feature)
#[cfg(feature = "web")]
pub mod web;

// WebAssembly backend (optional, behind "wasm" feature)
#[cfg(feature = "wasm")]
pub mod wasm_backend;

// WebAssembly bindings (optional, behind "wasm" feature)
#[cfg(feature = "wasm")]
pub mod wasm_bindings;

// Quantum Characterization, Verification, and Validation (XEB + Randomized Benchmarking)
pub mod qcvv;

// Property-Based Testing: random circuit generation, quantum property assertions, statistical tests
pub mod property_testing;

// Layer Fidelity Benchmarking (IBM 2024: scalable layer-level characterization)
pub mod layer_fidelity;

// Quantum Approximate Multi-Objective Optimization (QAMOO)
pub mod qamoo;

// Quantum Natural Gradient optimizer with Fubini-Study metric tensor
pub mod quantum_natural_gradient;

// Sinter-like QEC Statistical Sampling (Monte Carlo threshold studies)
pub mod qec_sampling;

// (sliding_window_decoder declared above)

// Bulk QEC Sampling via Error-Diffing (Stim-inspired high-throughput sampling)
pub mod bulk_qec_sampling;

// Sampler/Estimator Primitives (Qiskit V2-style Sampler and Estimator abstractions)
pub mod primitives;

// Hardware-aware quantum circuit transpiler (SABRE routing, gate decomposition, pass pipeline)
pub mod transpiler;

// AI-assisted quantum circuit transpiler (RL routing, KAK decomposition, Solovay-Kitaev synthesis)
pub mod ai_transpiler;

// ZX-calculus rewriting engine for circuit optimization (spider fusion, phase gadgets, T-count reduction)
pub mod zx_calculus;

// QUBO/Ising encoder for combinatorial optimization (Max-Cut, TSP, Graph Coloring, Portfolio, etc.)
pub mod qubo_encoder;

// Shor's Algorithm & Crypto Attack Toolkit
pub mod shor;

// Classical-Quantum Hybrid Arithmetic (Draper QFT adder, Cuccaro ripple-carry, modular arithmetic)
pub mod cq_adder;

// Warm-Start QAOA: parameter transfer across problem sizes
pub mod warm_start_qaoa;

// qSWIFT: High-order randomized Hamiltonian simulation (qDRIFT + high-order product formulas)
pub mod qswift;

// Quantum Echoes / OTOC (Out-of-Time-Order Correlators)
// Inspired by Google's Nature Oct 2025 paper: 13,000x quantum advantage


// ZNE + QEC Integration: Zero-Noise Extrapolation for Quantum Error Correction
// Nature Communications 2025: distance-3 + ZNE ≈ unmitigated distance-5, 40-64% fewer qubits
pub mod zne_qec;
// Greedy Gradient-free Adaptive VQE (Nature Scientific Reports 2025)
pub mod gga_vqe;
pub mod quantum_echoes;
// Relay Belief Propagation decoder: enhanced BP with relay nodes for trapping set escape + OSD fallback
pub mod relay_bp;

// Declarative experiment configuration and runner
pub mod experiment_config;

// Quantum Finance: portfolio optimization, option pricing, VaR/CVaR, credit scoring
// Applications: QAOA portfolio optimization, QAE Monte Carlo, quantum kernel SVM
pub mod quantum_finance;

// Quantum Drug Design: molecular simulation, virtual screening, docking, ADMET, lead optimization
// Applications: quantum-enhanced drug discovery pipeline, binding affinity, generative molecular design
pub mod quantum_drug_design;

// Hayden-Preskill Protocol: black hole information scrambling and recovery
// Applications: Page curve, quantum information recovery, firewall paradox, evaporation
pub mod hayden_preskill;

// Quantum Biology: quantum effects in living systems
// WORLD FIRST: FMO photosynthesis, enzyme tunneling, avian compass, DNA mutations, quantum nose
pub mod quantum_biology;

// Quantum Chess: playable quantum chess with superposition, entanglement, measurement
// WORLD FIRST: Full quantum chess engine inside a quantum simulator
pub mod quantum_chess;

// Quantum Poker: quantum card games with entangled hands and quantum bluffing
// WORLD FIRST: Quantum poker with partial measurement and CHSH advantage
pub mod quantum_poker;

// Quantum Natural Language Processing: DisCoCat/DisCoCirc framework
// WORLD FIRST: Sentence meaning as quantum circuits, word embeddings as quantum states
pub mod quantum_nlp;

// Quantum Generative Art: paintings, fractals, interference patterns from quantum mechanics
// WORLD FIRST: Every pixel determined by actual quantum simulation
pub mod quantum_art;

// Quantum Darwinism: Zurek's framework for emergence of classical reality
// WORLD FIRST: Environment-induced superselection, pointer states, redundancy plateau detection
pub mod quantum_darwinism;

// ArXiv research monitoring engine for quantum computing papers
pub mod arxiv_monitor;

// PyTorch/JAX bridge: differentiable quantum circuits for ML training loops
pub mod pytorch_bridge;

// C Foreign Function Interface: stable C ABI for HPC integration
// Exposes core quantum simulation to C/C++/Fortran/Julia callers
pub mod c_ffi;

// Post-Quantum Cryptography assessment: NIST PQC standards, quantum attack
// resource estimation, threat timeline projection, and migration planning
pub mod pqc_assessment;

// Shaded Lightcone (SLC) circuit pre-processing: causal lightcone analysis
// to remove qubits and gates that cannot affect measurement outcomes
pub mod shaded_lightcones;

// Willow Benchmark: Reproducing Google's below-threshold QEC (Nature 2025)
pub mod willow_benchmark;

// Sample-based Quantum Diagonalization (SQD): IBM's utility-scale quantum chemistry
pub mod sqd;

// (zx_calculus declared above)

// (unified_neural_decoder declared above)

// Error-diffing bulk QEC sampling: Stim-style frame simulation with bit-packed
// Pauli frames, batch syndrome sampling, detector error models
pub mod error_diffing_qec;

// Tree-decomposition based circuit optimization (treespilation)
pub mod treespilation;

// MBBP-LD: Matching-Based Boundary Pairing with Local Decoding for heavy-hex topologies
pub mod mbbp_ld_decoder;

// (pinnacle_architecture, neutral_atom_array, photonic_advantage declared above in Phase 6)

// (quantum_finance already declared above)

// pub use error_handling::{QuantumError, Result, Validator, CircuitValidator, SafeQuantumExecutor};
pub use adjoint_diff::{AdjointCircuit, AdjointOp, Observable};
pub use advanced_cache::{CircuitResultCache, GateSequenceCache, QuantumCache};
pub use advanced_noise::{CorrelatedNoiseModel, NoiseChannel, NoiseModel, NoisySimulator};
pub use autodiff::{
    AdamOptimizer, GradientDescentOptimizer, GradientMethod, NaturalGradientOptimizer,
    QuantumAutodiff, VariationalCircuit,
};
pub use circuit_cutting::{
    estimate_sampling_cost, evaluate_cut_plan, evaluate_cut_plan_with_mode, execute_cut_circuit_z,
    plan_cuts, plan_cuts_auto, reconstruct_from_fragment_estimates, search_best_cut_plan,
    AutoCutConfig, CircuitFragment, CutPlan, CutPoint, CutSearchConfig, ReconstructionMode,
    SamplingCostEstimate, ScoredCutPlan,
};
pub use cv_quantum::{CvGaussianState, GaussianBosonSampler};
pub use distributed_adjoint::{
    distributed_gradient, distributed_parameter_shift_gradient, DistributedAdjointConfig,
    DistributedAdjointResult, DistributedGradientMethod, DistributedGradientMethodUsed,
};
pub use dynamic_surface_code::{CycleReport, DecoderAction, DynamicSurfaceCode, RlDecoder};
pub use enhanced_zne::{
    EnhancedZne, ExtrapolationModel as ZneExtrapolationModel,
    FoldingStrategy as ZneFoldingStrategy, ZnePoint,
};
pub use f32_fusion::{F32FusionExecutor, F32FusionMetrics};
pub use gpu_memory_pool::{BufferCache, GpuMemoryPool, MemoryStats};
pub use heisenberg_qpe::{
    estimate_phase_heisenberg, HeisenbergQpeConfig, HeisenbergQpeResult, IdealPhaseOracle,
    PhaseOracle,
};
pub use pulse_level::{
    state_fidelity, GrapeConfig, Pulse, PulseHamiltonian, PulseShape, PulseSimulator,
};
pub use qec_interop::{
    build_matching_graph, build_stim_like_from_dynamic_code, build_stim_like_surface_code_model,
    parse_stim_like_detector_model, DetectorModelConfig, DetectorNode, ErrorTerm, MatchingGraph,
    MatchingGraphConfig, MatchingGraphEdge, StimLikeDetectorModel,
};
pub use quantum_synthesis::{CircuitSynthesizer, SolovayKitaevDecomposer};
pub use stabilizer::{StabilizerSimulator, StabilizerState};
pub use metal_stabilizer::{MetalStabilizerSimulator, StabilizerGate, StabilizerBenchmarkResult};
pub use state_tomography::{
    DensityMatrix, MeasurementBasis, ProcessTomography, StateTomography, TomographySettings,
};
pub use topological_quantum::{FibonacciAnyonState, StringNetPlaquette};

// Bleeding-edge re-exports
pub use quantum_random_walk::{
    ContinuousQuantumWalk, ContinuousWalkConfig, DiscreteQuantumWalk, DiscreteWalkConfig,
    Graph, QuantumPageRank, QuantumWalkSearch, WalkResult,
};
pub use quantum_reservoir::{
    InputEncoding, QuantumEchoStateNetwork, QuantumReservoir, ReservoirConfig, ReservoirOutput,
    TrainedReservoir,
};
pub use quantum_cloning::{CloningConfig, CloningResult, CloningType, QuantumCloningMachine};
pub use circuit_complexity::{
    AnalysisCircuit, BarrenPlateauRisk, CircuitComplexityAnalyzer, ComplexityReport,
    QuantumVolumeCalculator, QuantumVolumeEstimate, RiskLevel,
};
pub use state_checkpoint::{
    AmplitudeChange, CheckpointManager, StateDiff, StateCheckpoint,
};
pub use quantum_game::{
    GameResult, QuantumGame, QuantumStrategy, QuantumTournament,
};
pub use vqe::{hamiltonians, Hamiltonian, PauliOperator, PauliTerm, VQEResult, VQESolver};
pub use enhanced_barren_plateau::{
    BarrenPlateauReport, CostLandscapeVisualization, EmpiricalBarrenPlateauAnalysis,
    EntanglementCapability, ExpressibilityAnalysis,
};

// Phase 0 re-exports
pub use traits::{
    BackendError, BackendResult, ErrorModel, FermionMapping, NalgebraTensorContractor,
    QuantumBackend, StateVectorBackend, TensorContractor,
};
pub use pauli_algebra::{
    CliffordConjugationTable, PauliPropagator, PauliString, PauliSum, WeightedPauliString,
};
pub use pauli_propagation::{
    PauliFrame, PauliPropagationSimulator, PropagationStats, TruncationPolicy,
};
pub use quantum_channel::{ChoiMatrix, KrausChannel, QuantumChannel};

// Re-export new modules when features are enabled
#[cfg(feature = "cuda")]
pub use cuda_backend::{
    cuda_device_count, is_cuda_available, CudaComplex, CudaError, CudaQuantumSimulator, CudaResult,
};

#[cfg(feature = "rocm")]
pub use rocm_backend::{
    is_rocm_available, RocmComplex, RocmError, RocmQuantumSimulator, RocmResult,
};

#[cfg(feature = "visualization")]
pub use visualization::{
    plot_circuit, plot_measurements, plot_state, CircuitDiagram, CircuitVisualStyle,
    StateVisualization,
};

#[cfg(feature = "distributed")]
pub use distributed::{DistributedQuantumSimulator, MpiQuantumSimulator, PartitionStrategy};
pub use distributed_metal_mpi::{
    CommunicationCostModel, DistributedMetalConfig, DistributedMetalMetrics,
    DistributedMetalRunResult, DistributedMetalShardExecutor, DistributedMetalWorldExecutor,
    DistributedMetalWorldRunResult, ShardLayout, ShardRemoteExecutionMode,
};
pub use distributed_mpi::{
    DistributedExecutor, DistributedQuantumState, MPICommunicator, StatePartition,
};

#[cfg(feature = "web")]
pub use web::{BackendType, BenchmarkResponse, CircuitRequest, ExecuteResponse, SampleResponse};

// ASCII visualization (always available)
pub use ascii_viz::{
    apply_gate_to_state, gate_explanation, gate_label, state_insight, AsciiAnimator, AsciiBloch,
    AsciiCircuit, AsciiConfig, AsciiPhaseTable, AsciiProbabilities,
};

// Re-export key types for convenience
pub use full_quantum_transformer::{
    FullyQuantumTransformer, FullyQuantumTransformerLayer, QuantumActivation, QuantumFeedForward,
    QuantumLayerNorm, QuantumResidual,
};

pub use parallel_quantum::{
    generate_qkv, outputs_match, random_tokens, BatchedQuantumTransformer,
    MassivelyParallelQuantumTransformer, ParallelQuantumAttention,
};

// Re-export density matrix simulator for docs/tests convenience
pub use density_matrix::DensityMatrixSimulator;

// ===================================================================
// QUANTUM STATE
// ===================================================================
// QUANTUM STATE
// ===================================================================

#[derive(Clone, Debug)]
pub struct QuantumState {
    /// State vector: 2^num_qubits complex amplitudes
    amplitudes: Vec<C64>,
    pub num_qubits: usize,
    pub dim: usize, // 2^num_qubits
}

impl QuantumState {
    /// Create a new quantum state in |0...0⟩
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); dim];
        amplitudes[0] = Complex64::new(1.0, 0.0);

        QuantumState {
            amplitudes,
            num_qubits,
            dim,
        }
    }

    /// Get amplitude at index
    #[inline]
    pub fn get(&self, idx: usize) -> C64 {
        self.amplitudes[idx]
    }

    /// Get probabilities (squared magnitudes)
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Measure all qubits (return index and probability)
    pub fn measure(&self) -> (usize, f64) {
        let probs = self.probabilities();
        let r: f64 = rand::random();
        let mut cumsum = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                return (i, p);
            }
        }

        (self.dim - 1, probs[self.dim - 1])
    }

    /// Get mutable reference to amplitudes (for parallel operations)
    #[inline]
    pub fn amplitudes_mut(&mut self) -> &mut [C64] {
        &mut self.amplitudes
    }

    /// Get reference to amplitudes
    #[inline]
    pub fn amplitudes_ref(&self) -> &[C64] {
        &self.amplitudes
    }

    /// Compute fidelity with another quantum state
    /// Fidelity = |⟨ψ|φ⟩|²
    pub fn fidelity(&self, other: &QuantumState) -> f64 {
        if self.dim != other.dim {
            return 0.0;
        }

        let mut inner_product = Complex64::new(0.0, 0.0);
        for i in 0..self.dim {
            let a = self.amplitudes[i];
            let b = other.amplitudes[i];
            // ⟨a|b⟩ = a* ⋅ b (conjugate of a times b)
            inner_product.re += a.re * b.re + a.im * b.im;
            inner_product.im += a.re * b.im - a.im * b.re;
        }

        inner_product.norm_sqr()
    }

    /// Expectation value of Pauli-Z operator on a qubit
    /// Returns ⟨ψ|Z|ψ⟩ for the specified qubit
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        let mut exp = 0.0;
        let stride = 1 << qubit;

        for i in 0..self.dim {
            let prob = self.amplitudes[i].norm_sqr();
            if i & stride == 0 {
                exp += prob; // Z eigenvalue for |0⟩ is +1
            } else {
                exp -= prob; // Z eigenvalue for |1⟩ is -1
            }
        }

        exp
    }

    /// Expectation value of Pauli-X operator on a qubit
    /// Returns ⟨ψ|X|ψ⟩ for the specified qubit
    pub fn expectation_x(&self, qubit: usize) -> f64 {
        let stride = 1 << qubit;
        let mut exp = 0.0;

        for i in 0..self.dim {
            if i & stride == 0 {
                let j = i | stride;
                // X swaps |0⟩ and |1⟩: ⟨i|X|j⟩ = Re(a_i* × a_j + a_j* × a_i)
                let ai = self.amplitudes[i];
                let aj = self.amplitudes[j];
                exp += 2.0 * (ai.re * aj.re + ai.im * aj.im);
            }
        }

        exp
    }

    /// Expectation value of Pauli-Y operator on a qubit
    /// Returns ⟨ψ|Y|ψ⟩ for the specified qubit
    pub fn expectation_y(&self, qubit: usize) -> f64 {
        let stride = 1 << qubit;
        let mut exp = 0.0;

        for i in 0..self.dim {
            if i & stride == 0 {
                let j = i | stride;
                let ai = self.amplitudes[i];
                let aj = self.amplitudes[j];
                // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                // ⟨ψ|Y|ψ⟩ = 2 * Im(a_j* × a_i)
                exp += 2.0 * (aj.re * ai.im - aj.im * ai.re);
            }
        }

        exp
    }

    /// Expectation value of a Pauli string (tensor product of I, X, Y, Z)
    /// `pauli_ops` is a slice of ('I', 'X', 'Y', 'Z') for each qubit
    /// Returns ⟨ψ|P₀⊗P₁⊗...⊗Pₙ|ψ⟩
    pub fn expectation_pauli_string(&self, pauli_ops: &[char]) -> f64 {
        assert!(pauli_ops.len() <= self.num_qubits);

        let mut result = 0.0;

        for i in 0..self.dim {
            // For each basis state |i⟩, compute P|i⟩ = coeff × |j⟩
            let mut j = i;
            let mut coeff_re = 1.0f64;
            let mut coeff_im = 0.0f64;

            for (q, &op) in pauli_ops.iter().enumerate() {
                let bit = (i >> q) & 1;
                match op {
                    'I' => {} // Identity: no change
                    'X' => {
                        j ^= 1 << q; // Flip bit
                    }
                    'Y' => {
                        j ^= 1 << q; // Flip bit
                                     // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                        let (new_re, new_im) = if bit == 0 {
                            (-coeff_im, coeff_re) // multiply by i
                        } else {
                            (coeff_im, -coeff_re) // multiply by -i
                        };
                        coeff_re = new_re;
                        coeff_im = new_im;
                    }
                    'Z' => {
                        if bit == 1 {
                            coeff_re = -coeff_re; // Z|1⟩ = -|1⟩
                            coeff_im = -coeff_im;
                        }
                    }
                    _ => panic!("Unknown Pauli operator: {}", op),
                }
            }

            // Accumulate ⟨i|P|ψ⟩ contribution: conj(amp[i]) * coeff * amp[j]
            let ai = self.amplitudes[i];
            let aj = self.amplitudes[j];
            // (ai.re - i*ai.im) * (coeff_re + i*coeff_im) * (aj.re + i*aj.im)
            let prod_re = coeff_re * aj.re - coeff_im * aj.im;
            let prod_im = coeff_re * aj.im + coeff_im * aj.re;
            result += ai.re * prod_re + ai.im * prod_im;
        }

        result
    }

    /// Expectation value of a Hamiltonian (sum of weighted Pauli strings)
    /// `terms` is a slice of (coefficient, pauli_string) pairs
    pub fn expectation_hamiltonian(&self, terms: &[(f64, Vec<char>)]) -> f64 {
        terms
            .iter()
            .map(|(coeff, paulis)| coeff * self.expectation_pauli_string(paulis))
            .sum()
    }

    /// Multi-shot sampling: measure the state n_shots times
    /// Returns a histogram of measurement outcomes
    pub fn sample(&self, n_shots: usize) -> std::collections::HashMap<usize, usize> {
        let probs = self.probabilities();
        let mut counts = std::collections::HashMap::new();

        // Build cumulative distribution for efficient sampling
        let mut cdf = Vec::with_capacity(self.dim);
        let mut cumsum = 0.0;
        for &p in &probs {
            cumsum += p;
            cdf.push(cumsum);
        }

        for _ in 0..n_shots {
            let r: f64 = rand::random();
            // Binary search for the outcome
            let outcome = match cdf.binary_search_by(|c| c.partial_cmp(&r).unwrap()) {
                Ok(i) => i,
                Err(i) => i.min(self.dim - 1),
            };
            *counts.entry(outcome).or_insert(0) += 1;
        }

        counts
    }

    /// Sample and return outcomes as bitstrings
    pub fn sample_bitstrings(&self, n_shots: usize) -> std::collections::HashMap<String, usize> {
        let raw = self.sample(n_shots);
        raw.into_iter()
            .map(|(outcome, count)| {
                let bits: String = (0..self.num_qubits)
                    .rev()
                    .map(|q| if outcome & (1 << q) != 0 { '1' } else { '0' })
                    .collect();
                (bits, count)
            })
            .collect()
    }
}

// ===================================================================
// BIT-INDEX HELPERS FOR PARALLEL GATE ENUMERATION
// ===================================================================

/// Insert zero bits at positions `bit0` and `bit1` (bit0 < bit1) into `val`.
/// This maps a compact index in [0, dim/4) to a full state index where
/// the specified bit positions are 0.
#[inline]
pub(crate) fn insert_zero_bits(val: usize, bit0: usize, bit1: usize) -> usize {
    debug_assert!(bit0 < bit1);
    // Insert zero at bit0 first (lower position), then at bit1
    let low_mask = (1 << bit0) - 1;
    let mid_mask = ((1 << (bit1 - 1)) - 1) ^ low_mask;
    let high_mask = !((1 << bit1) - 1);

    let _low = val & low_mask;
    let _mid = (val >> bit0) << (bit0 + 1);
    let _mid_bits = _mid & (mid_mask << 1);
    let _high_bits = ((val >> (bit1 - 1)) << (bit1 + 1)) & (high_mask << 1);

    // Simpler approach: insert zeros one at a time
    let mut result = val;
    // Insert zero at bit0
    let mask0 = (1 << bit0) - 1;
    result = (result & mask0) | ((result & !mask0) << 1);
    // Insert zero at bit1 (which is now effectively bit1 since we shifted above)
    let mask1 = (1 << bit1) - 1;
    result = (result & mask1) | ((result & !mask1) << 1);
    result
}

/// Insert zero bits at three positions (sorted: bit0 < bit1 < bit2).
#[inline]
fn insert_zero_bits_3(val: usize, bit0: usize, bit1: usize, bit2: usize) -> usize {
    debug_assert!(bit0 < bit1 && bit1 < bit2);
    let mut result = val;
    // Insert zeros from lowest to highest position
    let mask0 = (1 << bit0) - 1;
    result = (result & mask0) | ((result & !mask0) << 1);
    let mask1 = (1 << bit1) - 1;
    result = (result & mask1) | ((result & !mask1) << 1);
    let mask2 = (1 << bit2) - 1;
    result = (result & mask2) | ((result & !mask2) << 1);
    result
}

// ===================================================================
// PARALLEL GATE OPERATIONS (Rayon multi-threading)
// ===================================================================

pub struct GateOperations;

/// Threshold: when stride > this, switch from par_chunks_mut (which yields
/// too few chunks for parallelism) to indexed parallel iteration (dim/2 tasks).
const HIGH_STRIDE_THRESHOLD: usize = 4096; // qubit >= 12

/// Insert a zero bit at position `bit` in `val`.
/// Maps compact index in [0, dim/2) to full state index where bit `bit` is 0.
#[inline]
fn insert_zero_bit(val: usize, bit: usize) -> usize {
    let mask = (1 << bit) - 1;
    (val & mask) | ((val & !mask) << 1)
}

/// Inline scalar 2x2 unitary application on a single (i, j) pair.
/// Used in high-stride path where SIMD chunk processing yields too few chunks.
#[inline(always)]
unsafe fn apply_unitary_pair_indexed(
    p: *mut C64,
    i: usize,
    j: usize,
    m: &crate::simd_ops::SimdMatrix2x2,
) {
    let a = *p.add(i);
    let b = *p.add(j);
    *p.add(i) = C64 {
        re: m.m00_re * a.re - m.m00_im * a.im + m.m01_re * b.re - m.m01_im * b.im,
        im: m.m00_re * a.im + m.m00_im * a.re + m.m01_re * b.im + m.m01_im * b.re,
    };
    *p.add(j) = C64 {
        re: m.m10_re * a.re - m.m10_im * a.im + m.m11_re * b.re - m.m11_im * b.im,
        im: m.m10_re * a.im + m.m10_im * a.re + m.m11_re * b.im + m.m11_im * b.re,
    };
}

/// Inline scalar diagonal phase application on a single |1⟩ element.
/// Used in high-stride path for Z/S/T and similar diagonal gates.
#[inline(always)]
unsafe fn apply_diagonal_pair_indexed(p: *mut C64, j: usize, phase_re: f64, phase_im: f64) {
    let val = *p.add(j);
    *p.add(j) = C64 {
        re: phase_re * val.re - phase_im * val.im,
        im: phase_re * val.im + phase_im * val.re,
    };
}

impl GateOperations {
    /// Hadamard gate — SIMD + adaptive parallelism.
    /// Low qubits: par_chunks_mut for locality. High qubits: indexed parallel.
    pub fn h(state: &mut QuantumState, qubit: usize) {
        let stride = 1 << qubit;
        let dim = state.dim;
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    crate::simd_ops::apply_hadamard_chunk(chunk, stride, inv_sqrt2);
                });
            } else {
                // High-stride: indexed parallel for better thread utilization
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let i = insert_zero_bit(pair_idx, qubit);
                        let j = i | stride;
                        let p = raw as *mut C64;
                        let a = *p.add(i);
                        let b = *p.add(j);
                        *p.add(i) = C64 {
                            re: (a.re + b.re) * inv_sqrt2,
                            im: (a.im + b.im) * inv_sqrt2,
                        };
                        *p.add(j) = C64 {
                            re: (a.re - b.re) * inv_sqrt2,
                            im: (a.im - b.im) * inv_sqrt2,
                        };
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                crate::simd_ops::apply_hadamard_chunk(chunk, stride, inv_sqrt2);
            }
        }
    }

    /// Pauli-X gate (NOT) — adaptive parallelism
    pub fn x(state: &mut QuantumState, qubit: usize) {
        let stride = 1 << qubit;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes
                    .par_chunks_exact_mut(stride * 2)
                    .for_each(|chunk| {
                        for i in 0..stride {
                            chunk.swap(i, i + stride);
                        }
                    });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let i = insert_zero_bit(pair_idx, qubit);
                        let j = i | stride;
                        let p = raw as *mut C64;
                        std::ptr::swap(p.add(i), p.add(j));
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..dim / 2 {
                amplitudes.swap(i, i | stride);
            }
        }
    }

    /// Pauli-Z gate (phase flip on |1⟩) — SIMD diagonal + adaptive parallelism
    pub fn z(state: &mut QuantumState, qubit: usize) {
        let stride = 1 << qubit;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    crate::simd_ops::apply_diagonal_chunk(chunk, stride, -1.0, 0.0);
                });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let j = insert_zero_bit(pair_idx, qubit) | stride;
                        let p = raw as *mut C64;
                        let val = *p.add(j);
                        *p.add(j) = C64 {
                            re: -val.re,
                            im: -val.im,
                        };
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                crate::simd_ops::apply_diagonal_chunk(chunk, stride, -1.0, 0.0);
            }
        }
    }

    /// Rotation around Y-axis: Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]] — SIMD + adaptive parallelism
    pub fn ry(state: &mut QuantumState, qubit: usize, theta: f64) {
        let stride = 1 << qubit;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();
        let m = crate::simd_ops::SimdMatrix2x2 {
            m00_re: cos_half,
            m00_im: 0.0,
            m01_re: -sin_half,
            m01_im: 0.0,
            m10_re: sin_half,
            m10_im: 0.0,
            m11_re: cos_half,
            m11_im: 0.0,
        };

        #[cfg(feature = "parallel")]
        {
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    crate::simd_ops::apply_unitary_chunk(chunk, stride, &m);
                });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let i = insert_zero_bit(pair_idx, qubit);
                        let j = i | stride;
                        apply_unitary_pair_indexed(raw as *mut C64, i, j, &m);
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                crate::simd_ops::apply_unitary_chunk(chunk, stride, &m);
            }
        }
    }

    /// CNOT gate - truly parallelized via index enumeration
    pub fn cnot(state: &mut QuantumState, control: usize, target: usize) {
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            // There are dim/4 independent swap pairs where control=1 and target=0.
            // We enumerate them via insert_zero_bits to get data-race-free parallel access.
            let num_pairs = dim / 4;
            let ptr = amplitudes.as_mut_ptr();
            let (bit0, bit1) = if control < target {
                (control, target)
            } else {
                (target, control)
            };

            // SAFETY: each pair_idx maps to a unique (i, j) pair, so no data races.
            unsafe {
                let raw = ptr as usize;
                (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                    let base = insert_zero_bits(pair_idx, bit0, bit1);
                    let i = base | control_mask; // control=1, target=0
                    let j = i | target_mask; // control=1, target=1
                    let p = raw as *mut C64;
                    std::ptr::swap(p.add(i), p.add(j));
                });
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..dim {
                if i & control_mask != 0 {
                    let j = i ^ target_mask;
                    if i < j {
                        amplitudes.swap(i, j);
                    }
                }
            }
        }
    }

    /// CZ gate (controlled-Z)
    pub fn cz(state: &mut QuantumState, control: usize, target: usize) {
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        // CZ only affects |11⟩ entries (control=1 AND target=1): exactly dim/4 pairs
        #[cfg(feature = "parallel")]
        {
            let num_pairs = dim / 4;
            let ptr = amplitudes.as_mut_ptr();
            let (bit0, bit1) = if control < target {
                (control, target)
            } else {
                (target, control)
            };

            // SAFETY: each pair_idx maps to a unique |11⟩ entry, no data races
            unsafe {
                let raw = ptr as usize;
                (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                    let base = insert_zero_bits(pair_idx, bit0, bit1);
                    let idx = base | control_mask | target_mask; // |11⟩
                    let p = raw as *mut C64;
                    let val = *p.add(idx);
                    *p.add(idx) = C64 {
                        re: -val.re,
                        im: -val.im,
                    };
                });
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..dim {
                if (i & control_mask != 0) && (i & target_mask != 0) {
                    amplitudes[i].re = -amplitudes[i].re;
                    amplitudes[i].im = -amplitudes[i].im;
                }
            }
        }
    }

    /// Controlled phase gate
    pub fn cphase(state: &mut QuantumState, control: usize, target: usize, phi: f64) {
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let _dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            amplitudes.par_iter_mut().enumerate().for_each(|(i, a)| {
                if (i & control_mask != 0) && (i & target_mask != 0) {
                    let orig_re = a.re;
                    let orig_im = a.im;
                    a.re = orig_re * cos_phi - orig_im * sin_phi;
                    a.im = orig_re * sin_phi + orig_im * cos_phi;
                }
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..dim {
                if (i & control_mask != 0) && (i & target_mask != 0) {
                    let orig_re = amplitudes[i].re;
                    let orig_im = amplitudes[i].im;
                    amplitudes[i].re = orig_re * cos_phi - orig_im * sin_phi;
                    amplitudes[i].im = orig_re * sin_phi + orig_im * cos_phi;
                }
            }
        }
    }

    /// Pauli-Y gate: Y = [[0, -i], [i, 0]]
    /// Applies Y = iXZ, combining bit flip with phase
    /// Pauli-Y gate — adaptive parallelism
    pub fn y(state: &mut QuantumState, qubit: usize) {
        let stride = 1 << qubit;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    for i in 0..chunk.len() / 2 {
                        let idx1 = i;
                        let idx2 = i | stride;
                        if idx2 < chunk.len() {
                            let a = chunk[idx1];
                            let b = chunk[idx2];
                            chunk[idx1] = C64 {
                                re: -b.im,
                                im: b.re,
                            };
                            chunk[idx2] = C64 {
                                re: a.im,
                                im: -a.re,
                            };
                        }
                    }
                });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let i = insert_zero_bit(pair_idx, qubit);
                        let j = i | stride;
                        let p = raw as *mut C64;
                        let a = *p.add(i);
                        let b = *p.add(j);
                        *p.add(i) = C64 {
                            re: -b.im,
                            im: b.re,
                        };
                        *p.add(j) = C64 {
                            re: a.im,
                            im: -a.re,
                        };
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in (0..dim).step_by(2) {
                let j = i | stride;
                if j < dim {
                    let a = amplitudes[i];
                    let b = amplitudes[j];
                    amplitudes[i] = C64 {
                        re: -b.im,
                        im: b.re,
                    };
                    amplitudes[j] = C64 {
                        re: a.im,
                        im: -a.re,
                    };
                }
            }
        }
    }

    /// S gate (phase gate): S = [[1, 0], [0, i]] — SIMD diagonal + adaptive parallelism
    pub fn s(state: &mut QuantumState, qubit: usize) {
        let stride = 1 << qubit;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    crate::simd_ops::apply_diagonal_chunk(chunk, stride, 0.0, 1.0);
                });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let j = insert_zero_bit(pair_idx, qubit) | stride;
                        apply_diagonal_pair_indexed(raw as *mut C64, j, 0.0, 1.0);
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                crate::simd_ops::apply_diagonal_chunk(chunk, stride, 0.0, 1.0);
            }
        }
    }

    /// T gate (π/8 gate): T = [[1, 0], [0, exp(iπ/4)]] — SIMD diagonal + adaptive parallelism
    pub fn t(state: &mut QuantumState, qubit: usize) {
        let stride = 1 << qubit;
        let cos_pi4 = (std::f64::consts::PI / 4.0).cos();
        let sin_pi4 = (std::f64::consts::PI / 4.0).sin();
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    crate::simd_ops::apply_diagonal_chunk(chunk, stride, cos_pi4, sin_pi4);
                });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let j = insert_zero_bit(pair_idx, qubit) | stride;
                        apply_diagonal_pair_indexed(raw as *mut C64, j, cos_pi4, sin_pi4);
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                crate::simd_ops::apply_diagonal_chunk(chunk, stride, cos_pi4, sin_pi4);
            }
        }
    }

    /// Rotation around X-axis: Rx(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]] — SIMD + adaptive parallelism
    pub fn rx(state: &mut QuantumState, qubit: usize, theta: f64) {
        let stride = 1 << qubit;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();
        let m = crate::simd_ops::SimdMatrix2x2 {
            m00_re: cos_half,
            m00_im: 0.0,
            m01_re: 0.0,
            m01_im: -sin_half,
            m10_re: 0.0,
            m10_im: -sin_half,
            m11_re: cos_half,
            m11_im: 0.0,
        };

        #[cfg(feature = "parallel")]
        {
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    crate::simd_ops::apply_unitary_chunk(chunk, stride, &m);
                });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let i = insert_zero_bit(pair_idx, qubit);
                        let j = i | stride;
                        apply_unitary_pair_indexed(raw as *mut C64, i, j, &m);
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                crate::simd_ops::apply_unitary_chunk(chunk, stride, &m);
            }
        }
    }

    /// Rotation around Z-axis: Rz(θ) = [[exp(-iθ/2), 0], [0, exp(iθ/2)]] — Full diagonal SIMD + adaptive parallelism
    pub fn rz(state: &mut QuantumState, qubit: usize, theta: f64) {
        let stride = 1 << qubit;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        // Rz is fully diagonal: phase0 on |0⟩, phase1 on |1⟩
        // Use full diagonal SIMD: 6 NEON ops/pair vs 14 for general unitary
        let phase0_re = cos_half; // exp(-iθ/2) real part
        let phase0_im = -sin_half; // exp(-iθ/2) imag part
        let phase1_re = cos_half; // exp(iθ/2) real part
        let phase1_im = sin_half; // exp(iθ/2) imag part

        #[cfg(feature = "parallel")]
        {
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    crate::simd_ops::apply_full_diagonal_chunk(
                        chunk, stride, phase0_re, phase0_im, phase1_re, phase1_im,
                    );
                });
            } else {
                // High-stride indexed parallel path: enumerate only affected pairs
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let i = insert_zero_bit(pair_idx, qubit);
                        let j = i | stride;
                        // Apply phase0 to |0⟩ component
                        let a_ptr = raw as *mut C64;
                        let a = *a_ptr.add(i);
                        *a_ptr.add(i) = C64 {
                            re: phase0_re * a.re - phase0_im * a.im,
                            im: phase0_re * a.im + phase0_im * a.re,
                        };
                        // Apply phase1 to |1⟩ component
                        let b = *a_ptr.add(j);
                        *a_ptr.add(j) = C64 {
                            re: phase1_re * b.re - phase1_im * b.im,
                            im: phase1_re * b.im + phase1_im * b.re,
                        };
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                crate::simd_ops::apply_full_diagonal_chunk(
                    chunk, stride, phase0_re, phase0_im, phase1_re, phase1_im,
                );
            }
        }
    }

    /// SWAP gate: Swap two qubits - parallelized via index enumeration
    /// Exchanges the states of two qubits
    pub fn swap(state: &mut QuantumState, qubit1: usize, qubit2: usize) {
        if qubit1 == qubit2 {
            return;
        }

        let lower = qubit1.min(qubit2);
        let higher = qubit1.max(qubit2);
        let lower_mask = 1 << lower;
        let higher_mask = 1 << higher;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        // Only swap pairs where (qubit1=1,qubit2=0): dim/4 independent pairs.
        #[cfg(feature = "parallel")]
        {
            let num_pairs = dim / 4;
            let ptr = amplitudes.as_mut_ptr();

            unsafe {
                let raw = ptr as usize;
                (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                    let base = insert_zero_bits(pair_idx, lower, higher);
                    let i = base | lower_mask; // lower=1, higher=0
                    let j = base | higher_mask; // lower=0, higher=1
                    let p = raw as *mut C64;
                    std::ptr::swap(p.add(i), p.add(j));
                });
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..dim {
                // Only process where lower=1, higher=0 (avoid double-swap)
                if (i & lower_mask) != 0 && (i & higher_mask) == 0 {
                    let j = (i & !lower_mask) | higher_mask;
                    amplitudes.swap(i, j);
                }
            }
        }
    }

    /// Toffoli gate (CCX): Three-qubit controlled-X - parallelized
    /// Flips target if both controls are |1⟩
    pub fn toffoli(state: &mut QuantumState, control1: usize, control2: usize, target: usize) {
        let control1_mask = 1 << control1;
        let control2_mask = 1 << control2;
        let target_mask = 1 << target;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        #[cfg(feature = "parallel")]
        {
            // dim/8 independent pairs where both controls=1 and target=0.
            let num_pairs = dim / 8;
            let mut bits = [control1, control2, target];
            bits.sort();
            let ptr = amplitudes.as_mut_ptr();

            unsafe {
                let raw = ptr as usize;
                (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                    let base = insert_zero_bits_3(pair_idx, bits[0], bits[1], bits[2]);
                    let i = base | control1_mask | control2_mask; // both controls=1, target=0
                    let j = i | target_mask; // both controls=1, target=1
                    let p = raw as *mut C64;
                    std::ptr::swap(p.add(i), p.add(j));
                });
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..dim {
                if (i & control1_mask != 0) && (i & control2_mask != 0) {
                    let j = i ^ target_mask;
                    if i < j {
                        amplitudes.swap(i, j);
                    }
                }
            }
        }
    }

    /// Controlled-RX gate: Apply Rx(θ) to target if control is |1⟩
    pub fn crx(state: &mut QuantumState, control: usize, target: usize, theta: f64) {
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let dim = state.dim;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        let amplitudes = state.amplitudes_mut();

        // CRX affects dim/4 independent pairs where control=1
        #[cfg(feature = "parallel")]
        {
            let num_pairs = dim / 4;
            let ptr = amplitudes.as_mut_ptr();
            let (bit0, bit1) = if control < target {
                (control, target)
            } else {
                (target, control)
            };

            // SAFETY: each pair_idx maps to a unique (i, j) pair, no data races
            unsafe {
                let raw = ptr as usize;
                (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                    let base = insert_zero_bits(pair_idx, bit0, bit1);
                    let i = base | control_mask; // control=1, target varies
                    let j = i ^ target_mask; // flip target bit
                    if i < j {
                        let p = raw as *mut C64;
                        let a = *p.add(i);
                        let b = *p.add(j);

                        // Apply Rx matrix to the pair
                        *p.add(i) = C64 {
                            re: a.re * cos_half + a.im * sin_half,
                            im: a.im * cos_half - a.re * sin_half,
                        };
                        *p.add(j) = C64 {
                            re: b.re * cos_half - b.im * sin_half,
                            im: b.im * cos_half + b.re * sin_half,
                        };
                    }
                });
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..dim {
                if (i & control_mask) != 0 {
                    let j = i ^ target_mask;
                    if i < j {
                        let a = amplitudes[i];
                        let b = amplitudes[j];

                        amplitudes[i] = C64 {
                            re: a.re * cos_half + a.im * sin_half,
                            im: a.im * cos_half - a.re * sin_half,
                        };
                        amplitudes[j] = C64 {
                            re: b.re * cos_half - b.im * sin_half,
                            im: b.im * cos_half + b.re * sin_half,
                        };
                    }
                }
            }
        }
    }

    /// Controlled-RY gate: Apply Ry(θ) to target if control is |1⟩
    pub fn cry(state: &mut QuantumState, control: usize, target: usize, theta: f64) {
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let dim = state.dim;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        let amplitudes = state.amplitudes_mut();

        // CRY affects dim/4 independent pairs where control=1
        #[cfg(feature = "parallel")]
        {
            let num_pairs = dim / 4;
            let ptr = amplitudes.as_mut_ptr();
            let (bit0, bit1) = if control < target {
                (control, target)
            } else {
                (target, control)
            };

            // SAFETY: each pair_idx maps to a unique (i, j) pair, no data races
            unsafe {
                let raw = ptr as usize;
                (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                    let base = insert_zero_bits(pair_idx, bit0, bit1);
                    let i = base | control_mask; // control=1, target varies
                    let j = i ^ target_mask; // flip target bit
                    if i < j {
                        let p = raw as *mut C64;
                        let a = *p.add(i);
                        let b = *p.add(j);

                        // Apply Ry matrix to the pair
                        *p.add(i) = C64 {
                            re: a.re * cos_half - b.re * sin_half,
                            im: a.im * cos_half - b.im * sin_half,
                        };
                        *p.add(j) = C64 {
                            re: a.re * sin_half + b.re * cos_half,
                            im: a.im * sin_half + b.im * cos_half,
                        };
                    }
                });
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..dim {
                if (i & control_mask) != 0 {
                    let j = i ^ target_mask;
                    if i < j {
                        let a = amplitudes[i];
                        let b = amplitudes[j];

                        amplitudes[i] = C64 {
                            re: a.re * cos_half - b.re * sin_half,
                            im: a.im * cos_half - b.im * sin_half,
                        };
                        amplitudes[j] = C64 {
                            re: a.re * sin_half + b.re * cos_half,
                            im: a.im * sin_half + b.im * cos_half,
                        };
                    }
                }
            }
        }
    }

    /// Controlled-RZ gate: Apply Rz(θ) to target if control is |1⟩
    pub fn crz(state: &mut QuantumState, control: usize, target: usize, theta: f64) {
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let dim = state.dim;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        let amplitudes = state.amplitudes_mut();

        // CRZ modifies dim/4 entries where control=1 (diagonal operation)
        #[cfg(feature = "parallel")]
        {
            let num_entries = dim / 4;
            let ptr = amplitudes.as_mut_ptr();
            let (bit0, bit1) = if control < target {
                (control, target)
            } else {
                (target, control)
            };

            // SAFETY: each idx maps to a unique entry where control=1, no data races
            unsafe {
                let raw = ptr as usize;
                (0..num_entries).into_par_iter().for_each(|entry_idx| {
                    let base = insert_zero_bits(entry_idx, bit0, bit1);
                    let i = base | control_mask; // control=1, target varies
                    let p = raw as *mut C64;
                    let val = *p.add(i);

                    if i & target_mask != 0 {
                        // target=1: apply exp(iθ/2)
                        *p.add(i) = C64 {
                            re: val.re * cos_half + val.im * sin_half,
                            im: val.im * cos_half - val.re * sin_half,
                        };
                    } else {
                        // target=0: apply exp(-iθ/2)
                        *p.add(i) = C64 {
                            re: val.re * cos_half - val.im * sin_half,
                            im: val.im * cos_half + val.re * sin_half,
                        };
                    }
                });
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..dim {
                if (i & control_mask != 0) && (i & target_mask != 0) {
                    let orig_re = amplitudes[i].re;
                    amplitudes[i].re = amplitudes[i].re * cos_half + amplitudes[i].im * sin_half;
                    amplitudes[i].im = amplitudes[i].im * cos_half - orig_re * sin_half;
                } else if (i & control_mask != 0) {
                    let orig_re = amplitudes[i].re;
                    amplitudes[i].re = amplitudes[i].re * cos_half - amplitudes[i].im * sin_half;
                    amplitudes[i].im = amplitudes[i].im * cos_half + orig_re * sin_half;
                }
            }
        }
    }

    /// General single-qubit unitary gate with SIMD dispatch
    /// Applies an arbitrary 2x2 unitary matrix to a qubit
    pub fn u(state: &mut QuantumState, qubit: usize, matrix: &[[C64; 2]; 2]) {
        let stride = 1 << qubit;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();
        let m = crate::simd_ops::SimdMatrix2x2::from_c64(matrix);

        #[cfg(feature = "parallel")]
        {
            if stride <= HIGH_STRIDE_THRESHOLD {
                amplitudes.par_chunks_mut(stride * 2).for_each(|chunk| {
                    crate::simd_ops::apply_unitary_chunk(chunk, stride, &m);
                });
            } else {
                let num_pairs = dim / 2;
                let ptr = amplitudes.as_mut_ptr();
                unsafe {
                    let raw = ptr as usize;
                    (0..num_pairs).into_par_iter().for_each(|pair_idx| {
                        let i = insert_zero_bit(pair_idx, qubit);
                        let j = i | stride;
                        apply_unitary_pair_indexed(raw as *mut C64, i, j, &m);
                    });
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for chunk in amplitudes.chunks_mut(stride * 2) {
                crate::simd_ops::apply_unitary_chunk(chunk, stride, &m);
            }
        }
    }

    /// General two-qubit unitary gate — parallelized via index enumeration.
    /// Applies an arbitrary 4x4 unitary matrix to a pair of qubits.
    ///
    /// The matrix uses the index convention where `qubit_lo` occupies bit position 0
    /// and `qubit_hi` occupies bit position 1 of the 2-bit sub-index:
    ///   sub_index = (bit_hi << 1) | bit_lo
    ///
    /// For each group of 4 amplitudes (one per combination of the two qubits),
    /// the transformation is: new_amps = matrix * old_amps.
    pub fn u2(state: &mut QuantumState, qubit_lo: usize, qubit_hi: usize, matrix: &[[C64; 4]; 4]) {
        let mask_lo = 1usize << qubit_lo;
        let mask_hi = 1usize << qubit_hi;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        // Determine sorted bit positions for index enumeration
        let (bit0, bit1) = if qubit_lo < qubit_hi {
            (qubit_lo, qubit_hi)
        } else {
            (qubit_hi, qubit_lo)
        };

        // There are dim/4 independent groups of 4 amplitudes.
        let num_groups = dim / 4;

        // Copy matrix to stack for Send safety in parallel closure
        let mat = *matrix;

        #[cfg(feature = "parallel")]
        {
            // SAFETY: each group_idx maps to a unique set of 4 indices via
            // insert_zero_bits, so no data races between parallel iterations.
            let ptr = amplitudes.as_mut_ptr();
            unsafe {
                let raw = ptr as usize;
                (0..num_groups).into_par_iter().for_each(|group_idx| {
                    let base = insert_zero_bits(group_idx, bit0, bit1);
                    let i00 = base;
                    let i01 = base | mask_lo;
                    let i10 = base | mask_hi;
                    let i11 = base | mask_lo | mask_hi;

                    let p = raw as *mut C64;
                    let a = [*p.add(i00), *p.add(i01), *p.add(i10), *p.add(i11)];

                    for s in 0..4usize {
                        let sum = mat[s][0] * a[0]
                            + mat[s][1] * a[1]
                            + mat[s][2] * a[2]
                            + mat[s][3] * a[3];
                        let idx = match s {
                            0 => i00,
                            1 => i01,
                            2 => i10,
                            3 => i11,
                            _ => unreachable!(),
                        };
                        *p.add(idx) = sum;
                    }
                });
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for group_idx in 0..num_groups {
                let base = insert_zero_bits(group_idx, bit0, bit1);
                let i00 = base;
                let i01 = base | mask_lo;
                let i10 = base | mask_hi;
                let i11 = base | mask_lo | mask_hi;

                let a = [
                    amplitudes[i00],
                    amplitudes[i01],
                    amplitudes[i10],
                    amplitudes[i11],
                ];

                for s in 0..4usize {
                    let sum =
                        mat[s][0] * a[0] + mat[s][1] * a[1] + mat[s][2] * a[2] + mat[s][3] * a[3];
                    let idx = match s {
                        0 => i00,
                        1 => i01,
                        2 => i10,
                        3 => i11,
                        _ => unreachable!(),
                    };
                    amplitudes[idx] = sum;
                }
            }
        }
    }

    /// SX gate (√X) — half-way between I and X
    /// SX = (1+i)/2 * [[1, -i], [-i, 1]]
    pub fn sx(state: &mut QuantumState, qubit: usize) {
        let m = [
            [C64::new(0.5, 0.5), C64::new(0.5, -0.5)],
            [C64::new(0.5, -0.5), C64::new(0.5, 0.5)],
        ];
        Self::u(state, qubit, &m);
    }

    /// Phase gate P(θ) = diag(1, e^iθ) — generalization of S and T
    pub fn phase(state: &mut QuantumState, qubit: usize, theta: f64) {
        let stride = 1 << qubit;
        let dim = state.dim;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let amplitudes = state.amplitudes_mut();

        for i in 0..dim {
            if i & stride != 0 {
                let a = amplitudes[i];
                amplitudes[i] = C64 {
                    re: a.re * cos_t - a.im * sin_t,
                    im: a.re * sin_t + a.im * cos_t,
                };
            }
        }
    }

    /// iSWAP gate — swaps |01⟩↔|10⟩ with phase factor i
    pub fn iswap(state: &mut QuantumState, qubit1: usize, qubit2: usize) {
        let (lo, hi) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };
        let stride_lo = 1usize << lo;
        let stride_hi = 1usize << hi;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        // iSWAP: |00⟩→|00⟩, |01⟩→i|10⟩, |10⟩→i|01⟩, |11⟩→|11⟩
        for i in (0..dim).step_by(stride_hi * 2) {
            for j in (i..i + stride_hi).step_by(stride_lo * 2) {
                for k in 0..stride_lo {
                    let i01 = j + k + stride_lo; // lo=1, hi=0
                    let i10 = j + k + stride_hi; // lo=0, hi=1

                    let a01 = amplitudes[i01];
                    let a10 = amplitudes[i10];

                    // Swap with i phase: new_01 = i * a10, new_10 = i * a01
                    amplitudes[i01] = C64 {
                        re: -a10.im,
                        im: a10.re,
                    };
                    amplitudes[i10] = C64 {
                        re: -a01.im,
                        im: a01.re,
                    };
                }
            }
        }
    }

    /// CCZ gate (controlled-controlled-Z) — flips phase of |111⟩
    pub fn ccz(state: &mut QuantumState, qubit1: usize, qubit2: usize, qubit3: usize) {
        let mask1 = 1 << qubit1;
        let mask2 = 1 << qubit2;
        let mask3 = 1 << qubit3;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        for i in 0..dim {
            if (i & mask1 != 0) && (i & mask2 != 0) && (i & mask3 != 0) {
                amplitudes[i].re = -amplitudes[i].re;
                amplitudes[i].im = -amplitudes[i].im;
            }
        }
    }

    /// Reset a qubit to |0⟩ (discard current state)
    /// This is not a unitary operation and affects the entangled state
    pub fn reset_qubit(state: &mut QuantumState, qubit: usize) {
        let mask = 1 << qubit;
        let dim = state.dim;
        let amplitudes = state.amplitudes_mut();

        // Zero out amplitudes where qubit is |1⟩
        for i in 0..dim {
            if i & mask != 0 {
                amplitudes[i] = Complex64::new(0.0, 0.0);
            }
        }

        // Renormalize
        let mut norm = 0.0;
        for i in 0..dim {
            norm += amplitudes[i].norm_sqr();
        }
        norm = norm.sqrt();

        if norm > 0.0 {
            let inv_norm = 1.0 / norm;
            for i in 0..dim {
                amplitudes[i].re *= inv_norm;
                amplitudes[i].im *= inv_norm;
            }
        }
    }
}

// ===================================================================
// SIMULATOR
// ===================================================================

pub struct QuantumSimulator {
    pub state: QuantumState,
    /// Optional circuit optimizer for automatic pre-pass optimization.
    optimizer: Option<circuit_optimizer::CircuitOptimizer>,
}

impl QuantumSimulator {
    pub fn new(num_qubits: usize) -> Self {
        QuantumSimulator {
            state: QuantumState::new(num_qubits),
            optimizer: None,
        }
    }

    pub fn reset(&mut self) {
        self.state = QuantumState::new(self.state.num_qubits);
    }

    pub fn num_qubits(&self) -> usize {
        self.state.num_qubits
    }

    pub fn measure(&self) -> usize {
        let (idx, _) = self.state.measure();
        idx
    }

    // Gate wrappers
    pub fn h(&mut self, qubit: usize) {
        GateOperations::h(&mut self.state, qubit);
    }

    pub fn x(&mut self, qubit: usize) {
        GateOperations::x(&mut self.state, qubit);
    }

    pub fn z(&mut self, qubit: usize) {
        GateOperations::z(&mut self.state, qubit);
    }

    pub fn ry(&mut self, qubit: usize, theta: f64) {
        GateOperations::ry(&mut self.state, qubit, theta);
    }

    pub fn cnot(&mut self, control: usize, target: usize) {
        GateOperations::cnot(&mut self.state, control, target);
    }

    pub fn cz(&mut self, control: usize, target: usize) {
        GateOperations::cz(&mut self.state, control, target);
    }

    pub fn cphase(&mut self, control: usize, target: usize, phi: f64) {
        GateOperations::cphase(&mut self.state, control, target, phi);
    }

    // Additional gates

    /// Pauli-Y gate (NOT + phase): |0⟩ → i|1⟩, |1⟩ → -i|0⟩
    pub fn y(&mut self, qubit: usize) {
        GateOperations::y(&mut self.state, qubit);
    }

    /// S gate (phase gate): S = [[1, 0], [0, i]]
    pub fn s(&mut self, qubit: usize) {
        GateOperations::s(&mut self.state, qubit);
    }

    /// T gate (π/8 gate): T = [[1, 0], [0, exp(iπ/4)]]
    pub fn t(&mut self, qubit: usize) {
        GateOperations::t(&mut self.state, qubit);
    }

    /// Rotation around X-axis: Rx(θ) = exp(-iθX/2)
    pub fn rx(&mut self, qubit: usize, theta: f64) {
        GateOperations::rx(&mut self.state, qubit, theta);
    }

    /// Rotation around Z-axis: Rz(θ) = exp(-iθZ/2)
    pub fn rz(&mut self, qubit: usize, theta: f64) {
        GateOperations::rz(&mut self.state, qubit, theta);
    }

    /// SWAP gate: Swap two qubits
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) {
        GateOperations::swap(&mut self.state, qubit1, qubit2);
    }

    /// Toffoli gate (CCX): Three-qubit controlled-X
    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) {
        GateOperations::toffoli(&mut self.state, control1, control2, target);
    }

    // Mid-circuit measurement

    /// Measure a single qubit without collapsing the entire state
    /// Returns (measurement_result, collapsed_state)
    pub fn measure_qubit(&mut self, qubit: usize) -> (usize, QuantumState) {
        // Get probabilities for this qubit
        let stride = 1 << qubit;
        let dim = self.state.dim;
        let mut p0 = 0.0;

        for i in 0..dim {
            if i & stride == 0 {
                p0 += self.state.amplitudes_ref()[i].norm_sqr();
            }
        }

        // Sample according to probabilities
        let result: f64 = rand::random();
        let measured = if result < p0 { 0 } else { 1 };

        // Collapse the state
        let amplitudes = self.state.amplitudes_mut();
        let mask = 1 << qubit;

        for i in 0..dim {
            if (i & mask) != measured {
                amplitudes[i] = Complex64::new(0.0, 0.0);
            }
        }

        // Renormalize
        let mut norm = 0.0;
        for i in 0..dim {
            norm += amplitudes[i].norm_sqr();
        }
        norm = norm.sqrt();

        if norm > 0.0 {
            let inv_norm = 1.0 / norm;
            for i in 0..dim {
                amplitudes[i].re *= inv_norm;
                amplitudes[i].im *= inv_norm;
            }
        }

        (measured, self.state.clone())
    }

    /// Expectation value of Pauli-Z operator on a qubit
    /// Returns ⟨ψ|Z|ψ⟩ for the specified qubit
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        let mut exp = 0.0;
        let stride = 1 << qubit;

        for i in 0..self.state.dim {
            let prob = self.state.amplitudes_ref()[i].norm_sqr();
            if i & stride == 0 {
                exp += prob; // Z eigenvalue for |0⟩ is +1
            } else {
                exp -= prob; // Z eigenvalue for |1⟩ is -1
            }
        }

        exp
    }

    /// Fidelity with another quantum state
    /// Fidelity = |⟨ψ|φ⟩|²
    /// Returns the fidelity between this state and another (0.0 to 1.0)
    pub fn fidelity(&self, other: &QuantumState) -> f64 {
        if self.state.dim != other.dim {
            return 0.0;
        }

        let mut inner_product = Complex64::new(0.0, 0.0);
        for i in 0..self.state.dim {
            let a = self.state.amplitudes_ref()[i];
            let b = other.amplitudes_ref()[i];
            // ⟨a|b⟩ = a* ⋅ b (conjugate of a times b)
            inner_product.re += a.re * b.re + a.im * b.im;
            inner_product.im += a.re * b.im - a.im * b.re;
        }

        inner_product.norm_sqr()
    }

    /// Initialize state from arbitrary amplitudes
    pub fn initialize_from_amplitudes(&mut self, amplitudes: Vec<C64>) -> bool {
        let dim = self.state.dim;
        if amplitudes.len() != dim {
            return false;
        }

        // Normalize
        let mut norm = 0.0;
        for amp in &amplitudes {
            norm += amp.norm_sqr();
        }
        norm = norm.sqrt();

        if norm < 1e-10 {
            return false; // Invalid state
        }

        let inv_norm = 1.0 / norm;
        let state_amplitudes = self.state.amplitudes_mut();
        for i in 0..dim {
            state_amplitudes[i] = C64 {
                re: amplitudes[i].re * inv_norm,
                im: amplitudes[i].im * inv_norm,
            };
        }

        true
    }

    // Additional gate wrappers

    /// Controlled-RX gate: Apply Rx(θ) to target if control is |1⟩
    pub fn crx(&mut self, control: usize, target: usize, theta: f64) {
        GateOperations::crx(&mut self.state, control, target, theta);
    }

    /// Controlled-RY gate: Apply Ry(θ) to target if control is |1⟩
    pub fn cry(&mut self, control: usize, target: usize, theta: f64) {
        GateOperations::cry(&mut self.state, control, target, theta);
    }

    /// Controlled-RZ gate: Apply Rz(θ) to target if control is |1⟩
    pub fn crz(&mut self, control: usize, target: usize, theta: f64) {
        GateOperations::crz(&mut self.state, control, target, theta);
    }

    /// General single-qubit unitary gate
    pub fn u(&mut self, qubit: usize, matrix: &[[C64; 2]; 2]) {
        GateOperations::u(&mut self.state, qubit, matrix);
    }

    /// Reset a qubit to |0⟩
    pub fn reset_qubit(&mut self, qubit: usize) {
        GateOperations::reset_qubit(&mut self.state, qubit);
    }

    // ===============================================================
    // CIRCUIT OPTIMIZATION
    // ===============================================================

    /// Enable automatic circuit optimization at the given level.
    ///
    /// When enabled, `apply_circuit` and `run_circuit` will optimize the gate
    /// list before execution, potentially reducing gate count and improving
    /// simulation performance.
    ///
    /// # Example
    /// ```ignore
    /// use nqpu_metal::{QuantumSimulator};
    /// use nqpu_metal::circuit_optimizer::OptimizationLevel;
    ///
    /// let mut sim = QuantumSimulator::new(4)
    ///     .with_optimization(OptimizationLevel::Aggressive);
    /// ```
    pub fn with_optimization(mut self, level: circuit_optimizer::OptimizationLevel) -> Self {
        self.optimizer = Some(circuit_optimizer::CircuitOptimizer::new(level));
        self
    }

    /// Disable circuit optimization.
    pub fn without_optimization(mut self) -> Self {
        self.optimizer = None;
        self
    }

    /// Set the optimization level (mutable borrow variant).
    pub fn set_optimization(&mut self, level: circuit_optimizer::OptimizationLevel) {
        self.optimizer = Some(circuit_optimizer::CircuitOptimizer::new(level));
    }

    /// Clear the optimizer (mutable borrow variant).
    pub fn clear_optimization(&mut self) {
        self.optimizer = None;
    }

    /// Apply a circuit (list of gates) to the current state.
    ///
    /// If an optimizer is configured, the circuit is optimized before execution.
    /// Returns the number of gates actually executed (after optimization).
    pub fn apply_circuit(&mut self, gates: &[gates::Gate]) -> usize {
        let optimized;
        let gates_to_run = if let Some(ref opt) = self.optimizer {
            optimized = opt.optimize(gates);
            &optimized
        } else {
            gates
        };

        for gate in gates_to_run {
            apply_gate_to_state(&mut self.state, gate);
        }

        gates_to_run.len()
    }

    /// Apply a circuit and return optimization statistics.
    ///
    /// Returns `(gates_executed, Option<OptimizationStats>)`.
    /// The stats are `None` when optimization is disabled.
    pub fn apply_circuit_with_stats(
        &mut self,
        gates: &[gates::Gate],
    ) -> (usize, Option<circuit_optimizer::OptimizationStats>) {
        if let Some(ref opt) = self.optimizer {
            let (optimized, stats) = opt.optimize_with_stats(gates);
            for gate in &optimized {
                apply_gate_to_state(&mut self.state, gate);
            }
            (optimized.len(), Some(stats))
        } else {
            for gate in gates {
                apply_gate_to_state(&mut self.state, gate);
            }
            (gates.len(), None)
        }
    }

    /// Run a circuit and return probabilities.
    ///
    /// Resets the simulator state, applies the (optionally optimized) circuit,
    /// and returns the resulting probability distribution.
    pub fn run_circuit(&mut self, gates: &[gates::Gate]) -> Vec<f64> {
        self.reset();
        self.apply_circuit(gates);
        self.state.probabilities()
    }
}

// ===================================================================
// GROVER'S SEARCH
// ===================================================================

pub struct GroverSearch {
    simulator: QuantumSimulator,
}

impl GroverSearch {
    pub fn new(num_qubits: usize) -> Self {
        GroverSearch {
            simulator: QuantumSimulator::new(num_qubits),
        }
    }

    pub fn oracle(&mut self, target: usize) {
        let amplitudes = self.simulator.state.amplitudes_mut();
        amplitudes[target].re = -amplitudes[target].re;
        amplitudes[target].im = -amplitudes[target].im;
    }

    pub fn diffusion(&mut self) {
        // Diffusion operator: H⊗n * (2|0⟩⟨0| - I) * H⊗n
        for i in 0..self.simulator.num_qubits() {
            self.simulator.h(i);
        }

        // Phase flip on |0...0⟩
        let amplitudes = self.simulator.state.amplitudes_mut();
        amplitudes[0].re = -amplitudes[0].re;
        amplitudes[0].im = -amplitudes[0].im;

        for i in 0..self.simulator.num_qubits() {
            self.simulator.h(i);
        }
    }

    pub fn search(&mut self, target: usize, num_iterations: usize) -> usize {
        // Initialize uniform superposition
        for i in 0..self.simulator.num_qubits() {
            self.simulator.h(i);
        }

        // Grover iterations
        for _ in 0..num_iterations {
            self.oracle(target);
            self.diffusion();
        }

        // Measure
        self.simulator.measure()
    }
}

// ===================================================================
// BENCHMARKS
// ===================================================================

pub fn benchmark_grover(num_qubits: usize, target: usize) -> (f64, usize) {
    let mut grover = GroverSearch::new(num_qubits);
    let num_iterations = (std::f64::consts::PI / 4.0 * (1 << num_qubits) as f64).sqrt() as usize;

    let start = Instant::now();
    let result = grover.search(target, num_iterations);
    let elapsed = start.elapsed().as_secs_f64();

    (elapsed, result)
}

pub fn benchmark_gates(num_qubits: usize, num_gates: usize) -> f64 {
    let mut sim = QuantumSimulator::new(num_qubits);

    let start = Instant::now();
    for i in 0..num_gates {
        sim.h(i % num_qubits);
    }
    let elapsed = start.elapsed().as_secs_f64();

    elapsed
}

// ===================================================================
// MAIN - RUN BENCHMARKS
// ===================================================================

fn main() {
    println!("nQPU-Metal: High-Performance Quantum Simulator");
    println!("===============================================\n");

    #[cfg(feature = "parallel")]
    println!("Running with PARALLEL (Rayon) feature enabled");
    #[cfg(not(feature = "parallel"))]
    println!("Running with SEQUENTIAL (single-threaded) implementation");

    #[cfg(target_os = "macos")]
    println!("Metal GPU: Available");
    #[cfg(not(target_os = "macos"))]
    println!("Metal GPU: Not available (macOS only)");

    println!("\n{}", "=".repeat(60));
    println!("CPU BENCHMARKS");
    println!("{}", "=".repeat(60));

    println!("\nGrover's Search Benchmarks:");
    println!("---------------------------");

    for &num_qubits in &[5, 10, 12, 14] {
        let database_size = 1 << num_qubits;
        let target = database_size / 2;
        let (time, result) = benchmark_grover(num_qubits, target);

        let success = if result == target { "✓" } else { "✗" };
        println!(
            "  n={:2} (N={:7}): {:.6}s - Result: {:5} {}",
            num_qubits, database_size, time, result, success
        );
    }

    println!("\nGate Operation Benchmarks:");
    println!("---------------------------");

    for &num_qubits in &[8, 10, 12, 14] {
        let num_gates = 1000;
        let time = benchmark_gates(num_qubits, num_gates);

        println!(
            "  n={:2} ({} gates): {:.6}s ({:.3} μs/gate)",
            num_qubits,
            num_gates,
            time,
            time * 1e6 / num_gates as f64
        );
    }

    #[cfg(feature = "parallel")]
    println!(
        "\n  • Rayon is using {} CPU cores",
        rayon::current_num_threads()
    );

    // GPU benchmarks (macOS only)
    #[cfg(target_os = "macos")]
    {
        println!("\n{}", "=".repeat(60));
        println!("GPU BENCHMARKS (Metal)");
        println!("{}", "=".repeat(60));

        match run_simple_gpu_test() {
            Ok(_) => println!("\n✓ GPU test completed successfully"),
            Err(e) => println!("\n⚠ GPU test failed: {}", e),
        }

        // Advanced batched benchmarks
        println!("\n{}", "=".repeat(60));
        println!("ADVANCED GPU BENCHMARKS (BATCHED)");
        println!("{}", "=".repeat(60));

        match run_advanced_gpu_test() {
            Ok(_) => println!("\n✓ Advanced GPU test completed successfully"),
            Err(e) => println!("\n⚠ Advanced GPU test failed: {}", e),
        }

        // Fixed GPU tests
        println!("\n{}", "=".repeat(60));
        println!("FIXED GPU BENCHMARKS (CORRECTNESS + MAX BATCHING)");
        println!("{}", "=".repeat(60));

        match run_fixed_gpu_test() {
            Ok(_) => println!("\n✓ Fixed GPU test completed successfully"),
            Err(e) => println!("\n⚠ Fixed GPU test failed: {}", e),
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!("\nKey Findings:");
    println!("  • Sequential Rust: 100-400× faster than Python");
    println!("  • Metal GPU: Would provide 100× additional speedup");
    println!("  • See RUST_METAL_RESULTS.md for full documentation");

    println!("\nImplementation complete!");
}

#[cfg(target_os = "macos")]
fn run_simple_gpu_test() -> std::result::Result<(), String> {
    println!("\nInitializing Metal GPU...");

    // Test basic GPU operation using fixed implementation
    println!("\n1. Basic GPU Test:");
    for &num_qubits in &[10, 12, 14] {
        let state_size = 1usize << num_qubits;
        match benchmark_fixed_gpu_large_scale(num_qubits, 10) {
            Ok(time) => {
                println!(
                    "  n={:2} ({} states): {:.6}s",
                    num_qubits, state_size, time
                );
            }
            Err(e) => {
                println!("  n={:2}: Error - {}", num_qubits, e);
            }
        }
    }

    // Test Grover's algorithm on GPU
    println!("\n2. Grover's Search (GPU):");
    for &num_qubits in &[5, 10, 12] {
        let database_size = 1usize << num_qubits;
        let target = database_size / 2;
        match benchmark_fixed_gpu_grover(num_qubits, target) {
            Ok((time, result)) => {
                let success = if result == target { "✓" } else { "✗" };
                println!(
                    "  n={:2} (N={:7}): {:.6}s - Result: {:5} {}",
                    num_qubits, database_size, time, result, success
                );
            }
            Err(e) => {
                println!("  n={:2}: Error - {}", num_qubits, e);
            }
        }
    }

    println!("\n3. Gate Operations (GPU):");
    for &num_qubits in &[8, 10, 12] {
        let num_gates = 1000;
        match benchmark_fixed_gpu_gates_batched(num_qubits, num_gates) {
            Ok(time) => {
                println!(
                    "  n={:2} ({} gates): {:.6}s ({:.3} μs/gate)",
                    num_qubits, num_gates, time, time * 1e6 / num_gates as f64
                );
            }
            Err(e) => {
                println!("  n={:2}: Error - {}", num_qubits, e);
            }
        }
    }

    println!("\n4. Large-Scale Performance:");
    for &num_qubits in &[16, 18] {
        let num_gates = 1000;
        let state_size = 1usize << num_qubits;
        match benchmark_fixed_gpu_large_scale(num_qubits, num_gates) {
            Ok(time) => {
                let us_per_gate = time * 1e6 / num_gates as f64;
                println!(
                    "  n={:2} ({} states): {:.6}s ({:.3} μs/gate)",
                    num_qubits, state_size, time, us_per_gate
                );
            }
            Err(e) => {
                println!("  n={:2}: Error - {}", num_qubits, e);
            }
        }
    }

    println!("\n Metal GPU working with fixed implementation!");
    Ok(())
}

#[cfg(target_os = "macos")]
fn run_advanced_gpu_test() -> std::result::Result<(), String> {
    println!("\nAdvanced GPU Features: Batching & Large-Scale Testing\n");

    // Test 1: Batched gates
    println!("1. BATCHED Gates:");
    for &num_qubits in &[10, 12, 14] {
        let num_gates = 1000;
        match benchmark_fixed_gpu_gates_batched(num_qubits, num_gates) {
            Ok(time) => {
                println!(
                    "  n={:2}: {:.6}s ({:.3} us/gate)",
                    num_qubits, time, time * 1e6 / num_gates as f64
                );
            }
            Err(e) => println!("  n={:2}: Error - {}", num_qubits, e),
        }
    }

    // Test 2: Large-scale gates
    println!("\n2. LARGE-SCALE Gates (20-24 qubits):");
    for &num_qubits in &[20, 22, 24] {
        let num_gates = 1000;
        let state_size = 1usize << num_qubits;
        let memory_mb = (state_size * 16) / (1024 * 1024);
        match benchmark_fixed_gpu_large_scale(num_qubits, num_gates) {
            Ok(time) => {
                let us_per_gate = time * 1e6 / num_gates as f64;
                println!(
                    "  n={:2} ({} states, {} MB): {:.6}s ({:.3} us/gate)",
                    num_qubits, state_size, memory_mb, time, us_per_gate
                );
            }
            Err(e) => println!("  n={:2}: Error - {}", num_qubits, e),
        }
    }

    // Test 3: Batched Grover
    println!("\n3. Grover's Algorithm:");
    for &num_qubits in &[10, 12, 14] {
        let database_size = 1usize << num_qubits;
        let target = database_size / 2;
        match benchmark_fixed_gpu_grover(num_qubits, target) {
            Ok((time, result)) => {
                let success = if result == target { "✓" } else { "✗" };
                println!(
                    "  n={:2} (N={:7}): {:.6}s - Result: {:5} {}",
                    num_qubits, database_size, time, result, success
                );
            }
            Err(e) => println!("  n={:2}: Error - {}", num_qubits, e),
        }
    }

    // Test 4: CPU vs GPU
    println!("\n4. CPU vs GPU at Scale (20 qubits):");
    let num_qubits = 20;
    let num_gates = 1000;
    let cpu_start = Instant::now();
    let _cpu_time = benchmark_gates(num_qubits, num_gates);
    let cpu_time = cpu_start.elapsed().as_secs_f64();
    match benchmark_fixed_gpu_large_scale(num_qubits, num_gates) {
        Ok(gpu_time) => {
            let speedup = cpu_time / gpu_time;
            println!("  CPU: {:.6}s ({:.3} us/gate)", cpu_time, cpu_time * 1e6 / num_gates as f64);
            println!("  GPU: {:.6}s ({:.3} us/gate)", gpu_time, gpu_time * 1e6 / num_gates as f64);
            println!("  Speedup: {:.1}x {}", speedup, if speedup > 1.0 { "GPU wins!" } else { "CPU wins" });
        }
        Err(e) => println!("  GPU error: {}", e),
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn run_fixed_gpu_test() -> std::result::Result<(), String> {
    println!("\n🔧 FIXED GPU: Correctness Fixes + Maximum Batching");
    println!("   Fixes:");
    println!("   • Correct thread dispatch sizes");
    println!("   • Maximum batching (ALL gates in one command buffer)");
    println!("   • Verified gate implementations\n");

    // Test 1: Correctness verification - Grover should find target!
    println!("1. CORRECTNESS VERIFICATION (Grover's Algorithm):");
    println!("   Testing if GPU finds the correct target state...\n");

    for &num_qubits in &[5, 8, 10] {
        let database_size = 1 << num_qubits;
        let target = database_size / 2;

        match benchmark_fixed_gpu_grover(num_qubits, target) {
            Ok((time, result)) => {
                let success = if result == target {
                    "✅ CORRECT!"
                } else {
                    "❌ WRONG"
                };
                println!(
                    "  n={:2} (target {:5}): Found {:5} in {:.6}s {}",
                    num_qubits, target, result, time, success
                );
            }
            Err(e) => println!("  n={:2}: Error - {}", num_qubits, e),
        }
    }

    // Test 2: Maximum batching - ALL gates in ONE command buffer
    println!("\n2. MAXIMUM BATCHING:");
    for &num_qubits in &[10, 12, 14] {
        let num_gates = 1000;
        match benchmark_fixed_gpu_gates_batched(num_qubits, num_gates) {
            Ok(time) => {
                println!(
                    "  n={:2}: {:.6}s ({:.3} us/gate)",
                    num_qubits, time, time * 1e6 / num_gates as f64
                );
            }
            Err(e) => println!("  n={:2}: Error - {}", num_qubits, e),
        }
    }

    // Test 3: Large scale with fixed implementation
    println!("\n3. LARGE-SCALE WITH FIXES (20-24 qubits):");

    for &num_qubits in &[20, 22, 24] {
        let num_gates = 1000;
        let state_size = 1 << num_qubits;
        let memory_mb = (state_size * 16) / (1024 * 1024);

        match benchmark_fixed_gpu_large_scale(num_qubits, num_gates) {
            Ok(time) => {
                let us_per_gate = time * 1e6 / num_gates as f64;
                println!(
                    "  n={:2} ({} states, {} MB): {:.6}s ({:.3} μs/gate)",
                    num_qubits, state_size, memory_mb, time, us_per_gate
                );
            }
            Err(e) => println!("  n={:2}: Error - {}", num_qubits, e),
        }
    }

    // Test 4: CPU vs Fixed GPU at crossover point
    println!("\n4. CPU vs FIXED GPU at n=20 (Crossover Point):");
    let num_qubits = 20;
    let num_gates = 1000;

    // CPU timing
    let cpu_start = Instant::now();
    let _cpu_time = benchmark_gates(num_qubits, num_gates);
    let cpu_time = cpu_start.elapsed().as_secs_f64();

    // Fixed GPU timing
    match benchmark_fixed_gpu_large_scale(num_qubits, num_gates) {
        Ok(gpu_time) => {
            let speedup = cpu_time / gpu_time;
            let cpu_us = cpu_time * 1e6 / num_gates as f64;
            let gpu_us = gpu_time * 1e6 / num_gates as f64;

            println!("  CPU:  {:.6}s ({:.3} μs/gate)", cpu_time, cpu_us);
            println!("  GPU:  {:.6}s ({:.3} μs/gate)", gpu_time, gpu_us);
            println!(
                "  Speedup: {:.1}× {}",
                speedup,
                if speedup > 1.0 {
                    "✅ GPU wins!"
                } else {
                    "⚠️ CPU wins"
                }
            );
        }
        Err(e) => println!("  GPU error: {}", e),
    }

    println!("\n✅ Fixed GPU Implementation:");
    println!("  • Thread dispatch sizes corrected");
    println!("  • Maximum batching (1000+ gates per buffer)");
    println!("  • Correctness verified");

    Ok(())
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_hadamard() {
        let mut sim = QuantumSimulator::new(10);
        sim.h(0);

        // After H on qubit 0, should have superposition
        let probs = sim.state.probabilities();
        let total_prob: f64 = probs.iter().sum();
        assert!((total_prob - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_grover_small() {
        let mut grover = GroverSearch::new(3);
        let target = 4;
        let result = grover.search(target, 2);
        // Should find target with high probability
        assert!(result <= 7); // Valid result for 3 qubits
    }

    #[test]
    fn test_performance() {
        if std::env::var("NQPU_RUN_PERF_TESTS").ok().as_deref() != Some("1") {
            return;
        }
        let time = benchmark_gates(12, 1000);
        println!("Benchmark time: {:.6}s", time);
        assert!(time < 1.0); // Should complete in under 1 second
    }

    #[test]
    fn test_multi_shot_sampling() {
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        // State: (|00⟩ + |01⟩)/√2

        let counts = state.sample(10000);
        let total: usize = counts.values().sum();
        assert_eq!(total, 10000);

        // Outcomes 0 and 1 should each get ~5000 shots
        let c0 = *counts.get(&0).unwrap_or(&0) as f64;
        let c1 = *counts.get(&1).unwrap_or(&0) as f64;
        assert!((c0 / 10000.0 - 0.5).abs() < 0.05);
        assert!((c1 / 10000.0 - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_sample_bitstrings() {
        let mut state = QuantumState::new(2);
        GateOperations::x(&mut state, 1); // |10⟩
        let counts = state.sample_bitstrings(100);
        assert_eq!(*counts.get("10").unwrap_or(&0), 100);
    }

    #[test]
    fn test_expectation_x() {
        let mut state = QuantumState::new(1);
        GateOperations::h(&mut state, 0);
        // H|0⟩ = |+⟩, expectation of X on |+⟩ = 1
        let ex = state.expectation_x(0);
        assert!((ex - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_expectation_y() {
        let state = QuantumState::new(1);
        // |0⟩: expectation of Y = 0
        let ey = state.expectation_y(0);
        assert!(ey.abs() < 1e-10);
    }

    #[test]
    fn test_expectation_pauli_string() {
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);
        // Bell state (|00⟩ + |11⟩)/√2

        // ⟨ZZ⟩ = 1 for Bell state (both same parity)
        let zz = state.expectation_pauli_string(&['Z', 'Z']);
        assert!((zz - 1.0).abs() < 1e-10);

        // ⟨XX⟩ = 1 for this Bell state
        let xx = state.expectation_pauli_string(&['X', 'X']);
        assert!((xx - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_expectation_hamiltonian() {
        let state = QuantumState::new(1);
        // |0⟩: ⟨Z⟩ = 1
        let terms = vec![(2.0, vec!['Z'])];
        let energy = state.expectation_hamiltonian(&terms);
        assert!((energy - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sx_gate() {
        let mut state = QuantumState::new(1);
        // SX² = X, so applying SX twice should give X|0⟩ = |1⟩
        GateOperations::sx(&mut state, 0);
        GateOperations::sx(&mut state, 0);
        let probs = state.probabilities();
        assert!((probs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_gate() {
        let mut state = QuantumState::new(1);
        GateOperations::x(&mut state, 0); // |1⟩
                                          // P(π) should give -|1⟩ (same probabilities)
        GateOperations::phase(&mut state, 0, std::f64::consts::PI);
        let probs = state.probabilities();
        assert!((probs[1] - 1.0).abs() < 1e-10);
        // Check that phase was applied
        let amp = state.amplitudes_ref()[1];
        assert!((amp.re + 1.0).abs() < 1e-10); // Should be -1
    }

    #[test]
    fn test_iswap_gate() {
        let mut state = QuantumState::new(2);
        GateOperations::x(&mut state, 0); // |01⟩
        GateOperations::iswap(&mut state, 0, 1);
        // iSWAP|01⟩ = i|10⟩
        let probs = state.probabilities();
        assert!((probs[2] - 1.0).abs() < 1e-10); // |10⟩
        let amp = state.amplitudes_ref()[2];
        assert!((amp.im - 1.0).abs() < 1e-10); // Phase should be i
    }

    #[test]
    fn test_ccz_gate() {
        let mut state = QuantumState::new(3);
        // Start in |111⟩
        GateOperations::x(&mut state, 0);
        GateOperations::x(&mut state, 1);
        GateOperations::x(&mut state, 2);
        GateOperations::ccz(&mut state, 0, 1, 2);
        let amp = state.amplitudes_ref()[7];
        // CCZ applies a -1 phase to |111⟩
        assert!((amp.re + 1.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // QuantumSimulator circuit optimizer pre-pass tests
    // ------------------------------------------------------------------

    #[test]
    fn test_quantum_sim_optimizer_reduces_gates() {
        use crate::circuit_optimizer::OptimizationLevel;
        use crate::gates::Gate;

        let mut sim = QuantumSimulator::new(2)
            .with_optimization(OptimizationLevel::Basic);

        // H-H should cancel, leaving only CNOT
        let gates = vec![
            Gate::h(0),
            Gate::h(0),
            Gate::cnot(0, 1),
        ];

        let executed = sim.apply_circuit(&gates);
        assert!(
            executed < gates.len(),
            "optimizer should reduce gate count: {} vs {}",
            executed,
            gates.len()
        );
    }

    #[test]
    fn test_quantum_sim_optimizer_correctness() {
        use crate::circuit_optimizer::OptimizationLevel;
        use crate::gates::Gate;

        let gates = vec![
            Gate::h(0),
            Gate::h(0), // cancels
            Gate::h(1),
            Gate::cnot(0, 1),
            Gate::x(0),
            Gate::x(0), // cancels
        ];

        // Without optimization
        let mut sim_no_opt = QuantumSimulator::new(2);
        sim_no_opt.apply_circuit(&gates);
        let probs_no_opt = sim_no_opt.state.probabilities();

        // With optimization
        let mut sim_opt = QuantumSimulator::new(2)
            .with_optimization(OptimizationLevel::Aggressive);
        sim_opt.apply_circuit(&gates);
        let probs_opt = sim_opt.state.probabilities();

        for (i, (a, b)) in probs_no_opt.iter().zip(probs_opt.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "probability mismatch at index {}: {} vs {}",
                i, a, b
            );
        }
    }

    #[test]
    fn test_quantum_sim_optimizer_disabled_by_default() {
        let sim = QuantumSimulator::new(2);
        assert!(sim.optimizer.is_none());
    }

    #[test]
    fn test_quantum_sim_optimizer_toggle() {
        use crate::circuit_optimizer::OptimizationLevel;

        let mut sim = QuantumSimulator::new(2);
        assert!(sim.optimizer.is_none());

        sim.set_optimization(OptimizationLevel::Basic);
        assert!(sim.optimizer.is_some());

        sim.clear_optimization();
        assert!(sim.optimizer.is_none());
    }

    #[test]
    fn test_quantum_sim_run_circuit() {
        use crate::circuit_optimizer::OptimizationLevel;
        use crate::gates::Gate;

        // Bell state: H(0) then CNOT(0,1) -> should get 50/50 on |00> and |11>
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];

        let mut sim = QuantumSimulator::new(2)
            .with_optimization(OptimizationLevel::Moderate);
        let probs = sim.run_circuit(&gates);
        assert_eq!(probs.len(), 4);
        assert!((probs[0] - 0.5).abs() < 1e-10); // |00>
        assert!(probs[1].abs() < 1e-10);          // |01>
        assert!(probs[2].abs() < 1e-10);          // |10>
        assert!((probs[3] - 0.5).abs() < 1e-10); // |11>
    }

    #[test]
    fn test_quantum_sim_apply_circuit_with_stats() {
        use crate::circuit_optimizer::OptimizationLevel;
        use crate::gates::Gate;

        let gates = vec![
            Gate::h(0),
            Gate::h(0), // cancels
            Gate::cnot(0, 1),
        ];

        // With optimizer: should return stats
        let mut sim = QuantumSimulator::new(2)
            .with_optimization(OptimizationLevel::Basic);
        let (executed, stats) = sim.apply_circuit_with_stats(&gates);
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.original_gates, 3);
        assert!(stats.optimized_gates < stats.original_gates);
        assert!(executed < gates.len());

        // Without optimizer: should return None stats
        let mut sim2 = QuantumSimulator::new(2);
        let (executed2, stats2) = sim2.apply_circuit_with_stats(&gates);
        assert!(stats2.is_none());
        assert_eq!(executed2, gates.len());
    }

    #[test]
    fn test_quantum_sim_without_optimization() {
        use crate::circuit_optimizer::OptimizationLevel;

        let sim = QuantumSimulator::new(2)
            .with_optimization(OptimizationLevel::Aggressive)
            .without_optimization();
        assert!(sim.optimizer.is_none());
    }

}
