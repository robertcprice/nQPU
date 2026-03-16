//! Circuit optimization, transpilation, QASM/QIR formats, DSL, and visualization.

// Circuit analysis and optimization reports.
#[path = "analysis/circuit_cache.rs"]
pub mod circuit_cache;
#[path = "analysis/circuit_complexity.rs"]
pub mod circuit_complexity;
#[path = "analysis/circuit_cutting.rs"]
pub mod circuit_cutting;
#[path = "analysis/circuit_equivalence.rs"]
pub mod circuit_equivalence;
#[path = "analysis/optimization_report.rs"]
pub mod optimization_report;
#[path = "analysis/shaded_lightcones.rs"]
pub mod shaded_lightcones;

// Serialization, interchange, and language tooling.
#[path = "formats/circuit_export.rs"]
pub mod circuit_export;
#[path = "formats/circuit_serde.rs"]
pub mod circuit_serde;
#[path = "formats/qasm.rs"]
pub mod qasm;
#[path = "formats/qasm3.rs"]
pub mod qasm3;
#[path = "formats/qir.rs"]
pub mod qir;

// Synthesis, transpilation, and circuit construction.
#[path = "synthesis/ai_transpiler.rs"]
pub mod ai_transpiler;
#[path = "synthesis/circuit_macro.rs"]
pub mod circuit_macro;
#[path = "synthesis/circuit_optimizer.rs"]
pub mod circuit_optimizer;
#[path = "synthesis/ft_compilation.rs"]
pub mod ft_compilation;
#[path = "synthesis/gate_fusion.rs"]
pub mod gate_fusion;
#[path = "synthesis/parametric_circuits.rs"]
pub mod parametric_circuits;
#[path = "synthesis/quantum_synthesis.rs"]
pub mod quantum_synthesis;
#[path = "synthesis/transpiler.rs"]
pub mod transpiler;
#[path = "synthesis/treespilation.rs"]
pub mod treespilation;
#[path = "synthesis/noise_aware_routing.rs"]
pub mod noise_aware_routing;
#[path = "synthesis/zx_calculus.rs"]
pub mod zx_calculus;

#[cfg(feature = "lsp")]
#[path = "formats/qasm_lsp.rs"]
pub mod qasm_lsp;

#[path = "visualization/ascii_viz.rs"]
pub mod ascii_viz;
#[cfg(feature = "visualization")]
#[path = "visualization/visualization.rs"]
pub mod visualization;
