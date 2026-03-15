//! Quantum measurement: tomography, QCVV, classical shadows, and verification.

// Tomography and shadow-based reconstruction.
#[path = "tomography/classical_shadows.rs"]
pub mod classical_shadows;
#[path = "tomography/state_tomography.rs"]
pub mod state_tomography;

// Characterization and metrology routines.
#[path = "characterization/layer_fidelity.rs"]
pub mod layer_fidelity;
#[path = "characterization/qcvv.rs"]
pub mod qcvv;
#[path = "characterization/quantum_fisher.rs"]
pub mod quantum_fisher;

// Verification, certification, and test-style measurements.
#[path = "verification/property_testing.rs"]
pub mod property_testing;
#[path = "verification/quantum_measurement.rs"]
pub mod quantum_measurement;
#[path = "verification/quantum_source_certification.rs"]
pub mod quantum_source_certification;
#[path = "verification/quantum_verification.rs"]
pub mod quantum_verification;
