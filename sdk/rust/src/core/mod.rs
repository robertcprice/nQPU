//! Core quantum computing primitives, gates, backends, and simulation engines.

// Core gate definitions and arithmetic circuit building blocks.
#[path = "gates/gates.rs"]
pub mod gates;
#[path = "gates/simple_gates.rs"]
pub mod simple_gates;
#[path = "gates/comprehensive_gates.rs"]
pub mod comprehensive_gates;
#[path = "gates/cq_adder.rs"]
pub mod cq_adder;

// Stabilizer-family simulators and related near-Clifford methods.
#[path = "stabilizer/avx512_stabilizer.rs"]
pub mod avx512_stabilizer;
#[path = "stabilizer/clifford_t.rs"]
pub mod clifford_t;
#[path = "stabilizer/fast_stabilizer.rs"]
pub mod fast_stabilizer;
#[path = "stabilizer/inverse_tableau.rs"]
pub mod inverse_tableau;
#[path = "stabilizer/near_clifford.rs"]
pub mod near_clifford;
#[path = "stabilizer/optimized_stabilizer.rs"]
pub mod optimized_stabilizer;
#[path = "stabilizer/reference_frame.rs"]
pub mod reference_frame;
#[path = "stabilizer/simd_stabilizer.rs"]
pub mod simd_stabilizer;
#[path = "stabilizer/stabilizer.rs"]
pub mod stabilizer;
#[path = "stabilizer/stabilizer_router.rs"]
pub mod stabilizer_router;

// Alternative quantum state models and representations.
#[path = "state_models/cv_quantum.rs"]
pub mod cv_quantum;
#[path = "state_models/density_matrix.rs"]
pub mod density_matrix;
#[path = "state_models/entanglement.rs"]
pub mod entanglement;
#[path = "state_models/f32_backend.rs"]
pub mod f32_backend;
#[path = "state_models/fermionic_gaussian.rs"]
pub mod fermionic_gaussian;
#[path = "state_models/quantum_channel.rs"]
pub mod quantum_channel;
#[path = "state_models/quantum_f32.rs"]
pub mod quantum_f32;

// Other core computation models and execution semantics.
#[path = "computation/decision_diagram.rs"]
pub mod decision_diagram;
#[path = "computation/digital_analog.rs"]
pub mod digital_analog;
#[path = "computation/matchgate_simulation.rs"]
pub mod matchgate_simulation;
#[path = "computation/mbqc.rs"]
pub mod mbqc;
#[path = "computation/mid_circuit.rs"]
pub mod mid_circuit;

// Foundational algebra and execution interfaces.
#[path = "foundations/pauli_algebra.rs"]
pub mod pauli_algebra;
#[path = "foundations/primitives.rs"]
pub mod primitives;
