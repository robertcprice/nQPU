//! Quantum Error Correction Decoding Module
//!
//! Production-quality decoders for surface code quantum error correction.
//!
//! # Decoder Implementations
//!
//! - **MWPM**: Minimum Weight Perfect Matching (Blossom V) — gold standard for surface codes
//! - **BP**: Min-sum belief propagation on Tanner graphs — fast iterative decoder
//!
//! # Applications
//!
//! - Surface code error correction (rotated planar, XYZ color codes)
//! - Fault-tolerant quantum computing
//! - Topological quantum codes

pub mod mwpm;
pub mod bp;

pub use mwpm::{MWPMDecoder, SurfaceCodeConfig, SurfaceCodeDecoder, Syndrome};
pub use bp::BPDecoder;
