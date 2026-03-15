//! Quantum error correction codes, decoders, magic state distillation, and QEC tooling.

// Code families and stabilizer constructions.
#[path = "codes/bivariate_bicycle.rs"]
pub mod bivariate_bicycle;
#[path = "codes/bosonic_codes.rs"]
pub mod bosonic_codes;
#[path = "codes/cat_qubit_concatenation.rs"]
pub mod cat_qubit_concatenation;
#[path = "codes/dynamic_surface_code.rs"]
pub mod dynamic_surface_code;
#[path = "codes/floquet_codes.rs"]
pub mod floquet_codes;
#[path = "codes/holographic_codes.rs"]
pub mod holographic_codes;
#[path = "codes/hyperbolic_floquet.rs"]
pub mod hyperbolic_floquet;
#[path = "codes/qldpc.rs"]
pub mod qldpc;
#[path = "codes/quantum_error_correction.rs"]
pub mod quantum_error_correction;
#[path = "codes/rascql.rs"]
pub mod rascql;
#[path = "codes/surface_codes.rs"]
pub mod surface_codes;
#[path = "codes/trivariate_codes.rs"]
pub mod trivariate_codes;
#[path = "codes/xzzx_surface.rs"]
pub mod xzzx_surface;
#[path = "codes/yoked_surface_codes.rs"]
pub mod yoked_surface_codes;

// Decoder implementations.
#[path = "decoders/adaptive_realtime_decoder.rs"]
pub mod adaptive_realtime_decoder;
#[path = "decoders/bp_osd.rs"]
pub mod bp_osd;
#[path = "decoders/gpu_mwpm.rs"]
pub mod gpu_mwpm;
#[path = "decoders/mamba_qec_decoder.rs"]
pub mod mamba_qec_decoder;
#[path = "decoders/mbbp_ld_decoder.rs"]
pub mod mbbp_ld_decoder;
#[path = "decoders/metal_neural_decoder.rs"]
pub mod metal_neural_decoder;
#[path = "decoders/neural_decoder.rs"]
pub mod neural_decoder;
#[path = "decoders/relay_bp.rs"]
pub mod relay_bp;
#[path = "decoders/sliding_window_decoder.rs"]
pub mod sliding_window_decoder;
#[path = "decoders/transformer_qec_decoder.rs"]
pub mod transformer_qec_decoder;
#[path = "decoders/unified_neural_decoder.rs"]
pub mod unified_neural_decoder;

// Interop, import/export, and transpilation helpers.
#[path = "tooling/decoder_aware_transpiler.rs"]
pub mod decoder_aware_transpiler;
#[path = "tooling/qec_interop.rs"]
pub mod qec_interop;
#[path = "tooling/stim_format.rs"]
pub mod stim_format;
#[path = "tooling/stim_import.rs"]
pub mod stim_import;

// End-to-end QEC workflows and analysis pipelines.
#[path = "workflows/approximate_dynamical_qec.rs"]
pub mod approximate_dynamical_qec;
#[path = "workflows/bulk_qec_sampling.rs"]
pub mod bulk_qec_sampling;
#[path = "workflows/differentiable_qec.rs"]
pub mod differentiable_qec;
#[path = "workflows/error_diffing_qec.rs"]
pub mod error_diffing_qec;
#[path = "workflows/lattice_surgery.rs"]
pub mod lattice_surgery;
#[path = "workflows/magic_state_factory.rs"]
pub mod magic_state_factory;
#[path = "workflows/qec_sampling.rs"]
pub mod qec_sampling;
#[path = "workflows/zne_qec.rs"]
pub mod zne_qec;
