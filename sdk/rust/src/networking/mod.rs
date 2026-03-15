//! Quantum networking: QKD, QRNG, entropy extraction, PQC assessment, and network protocols.

// Hardware-backed randomness sources and device probes.
#[path = "hardware/camera_quantum.rs"]
pub mod camera_quantum;
#[path = "hardware/hardware_quantum.rs"]
pub mod hardware_quantum;
#[path = "hardware/real_quantum_probe.rs"]
pub mod real_quantum_probe;

// QKD and networking protocol stacks.
#[path = "protocols/metro_qkd_network.rs"]
pub mod metro_qkd_network;
#[path = "protocols/pqc_assessment.rs"]
pub mod pqc_assessment;
#[path = "protocols/qkd_protocols.rs"]
pub mod qkd_protocols;
#[path = "protocols/quantum_network_os.rs"]
pub mod quantum_network_os;
#[path = "protocols/quantum_networking.rs"]
pub mod quantum_networking;
#[path = "protocols/wireless_quantum.rs"]
pub mod wireless_quantum;

// QRNG pipelines, extraction, and certified randomness.
#[path = "randomness/certified_qrng.rs"]
pub mod certified_qrng;
#[path = "randomness/certified_quantum.rs"]
pub mod certified_quantum;
#[path = "randomness/qrng_experiment.rs"]
pub mod qrng_experiment;
#[path = "randomness/qrng_extraction_methods.rs"]
pub mod qrng_extraction_methods;
#[path = "randomness/qrng_integration.rs"]
pub mod qrng_integration;
#[path = "randomness/qrng_phase2.rs"]
pub mod qrng_phase2;
#[path = "randomness/qrng_source_comparison.rs"]
pub mod qrng_source_comparison;
#[path = "randomness/quantum_entropy_extraction.rs"]
pub mod quantum_entropy_extraction;
#[path = "randomness/quantum_randomness.rs"]
pub mod quantum_randomness;

// Randomness validation and experimental statistical tests.
#[path = "validation/nist_tests.rs"]
pub mod nist_tests;
#[path = "validation/quantum_ssd_tests.rs"]
pub mod quantum_ssd_tests;
