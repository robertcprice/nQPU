//! Quantum machine learning: kernels, neural networks, transformers, and classical ML bridges.

#[path = "models/advanced_quantum_ml.rs"]
pub mod advanced_quantum_ml;
#[path = "optimization/enhanced_barren_plateau.rs"]
pub mod enhanced_barren_plateau;
#[path = "models/full_quantum_transformer.rs"]
pub mod full_quantum_transformer;
#[path = "bridges/jax_bridge.rs"]
pub mod jax_bridge;
#[path = "models/neural_quantum_states.rs"]
pub mod neural_quantum_states;
#[path = "bridges/pytorch_bridge.rs"]
pub mod pytorch_bridge;
#[path = "core/quantum_attention.rs"]
pub mod quantum_attention;
#[path = "core/quantum_kernels.rs"]
pub mod quantum_kernels;
#[path = "core/qml_core.rs"]
pub mod qml_core;
pub use qml_core as quantum_ml;
#[path = "models/quantum_ml_mps.rs"]
pub mod quantum_ml_mps;
#[path = "optimization/quantum_natural_gradient.rs"]
pub mod quantum_natural_gradient;
#[path = "models/quantum_neural_net.rs"]
pub mod quantum_neural_net;
#[path = "models/quantum_reservoir.rs"]
pub mod quantum_reservoir;
#[path = "models/rydberg_reservoir.rs"]
pub mod rydberg_reservoir;

// Experimental modules
#[cfg(feature = "experimental")]
#[path = "experimental/mt_reservoir.rs"]
pub mod mt_reservoir;
