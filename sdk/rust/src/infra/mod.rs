//! Infrastructure: traits, utilities, benchmarks, FFI, bindings, and distributed computing.

// Foundational traits and helpers shared across the crate.
#[path = "execution/traits.rs"]
pub mod traits;
#[path = "execution/utilities.rs"]
pub mod utilities;
// pub mod error_handling; // temporarily disabled due to compilation errors

// Differentiation and gradient tooling.
#[path = "autodiff/adjoint_diff.rs"]
pub mod adjoint_diff;
#[path = "autodiff/autodiff.rs"]
pub mod autodiff;

// Benchmarking and verification harnesses.
#[path = "benchmarks/benchmark_suite.rs"]
pub mod benchmark_suite;
#[path = "benchmarks/comprehensive_benchmarks.rs"]
pub mod comprehensive_benchmarks;
#[path = "benchmarks/max_qubit_benchmark.rs"]
pub mod max_qubit_benchmark;
#[path = "benchmarks/test_2d_simple.rs"]
pub mod test_2d_simple;
#[path = "benchmarks/willow_benchmark.rs"]
pub mod willow_benchmark;

// Runtime orchestration, batching, caching, and checkpointing.
#[path = "execution/adaptive_batching.rs"]
pub mod adaptive_batching;
#[path = "execution/advanced_cache.rs"]
pub mod advanced_cache;
#[path = "execution/entanglement_scheduler.rs"]
pub mod entanglement_scheduler;
#[path = "execution/interactive_sim.rs"]
pub mod interactive_sim;
#[path = "execution/parallel_feedforward.rs"]
pub mod parallel_feedforward;
#[path = "execution/parallel_quantum.rs"]
pub mod parallel_quantum;
#[path = "execution/shot_batching.rs"]
pub mod shot_batching;
#[path = "execution/simd_ops.rs"]
pub mod simd_ops;
#[path = "execution/state_checkpoint.rs"]
pub mod state_checkpoint;

// Research configuration and estimator utilities.
#[path = "research/arxiv_monitor.rs"]
pub mod arxiv_monitor;
#[path = "research/experiment_config.rs"]
pub mod experiment_config;
#[path = "research/resource_estimation.rs"]
pub mod resource_estimation;

#[cfg(feature = "mpi")]
#[path = "distributed/real_mpi.rs"]
pub mod real_mpi;

#[cfg(any(feature = "python", test))]
#[path = "bindings/python.rs"]
pub mod python;

#[cfg(any(feature = "python", test))]
#[path = "bindings/python_api_v2.rs"]
pub mod python_api_v2;

#[path = "bindings/c_ffi.rs"]
pub mod c_ffi;

#[path = "bindings/tui.rs"]
pub mod tui;

#[cfg(feature = "wasm")]
#[path = "bindings/wasm_backend.rs"]
pub mod wasm_backend;

#[cfg(feature = "wasm")]
#[path = "bindings/wasm_bindings.rs"]
pub mod wasm_bindings;

#[cfg(feature = "distributed")]
#[path = "distributed/distributed.rs"]
pub mod distributed;

#[path = "distributed/distributed_adjoint.rs"]
pub mod distributed_adjoint;
#[path = "distributed/distributed_metal_mpi.rs"]
pub mod distributed_metal_mpi;
#[path = "distributed/distributed_mpi.rs"]
pub mod distributed_mpi;
