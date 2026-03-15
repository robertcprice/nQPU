//! Quantum algorithms spanning variational, phase estimation, simulation, and optimization.
//!
//! Naming convention:
//! - concrete descriptive modules live on disk
//! - compatibility aliases preserve the older flat names during the reorg

#[path = "variational/adapt_vqe.rs"]
pub mod adapt_vqe;
#[path = "transforms/foundational_algorithms.rs"]
pub mod foundational_algorithms;
pub use foundational_algorithms as algorithms;
#[path = "transforms/grid_algorithms.rs"]
pub mod grid_algorithms;
pub use grid_algorithms as algorithms_2d;
#[path = "optimization/annealing.rs"]
pub mod annealing;
#[path = "simulation/comprehensive_algorithms.rs"]
pub mod comprehensive_algorithms;
#[path = "variational/gga_vqe.rs"]
pub mod gga_vqe;
#[path = "simulation/gpu_pauli_propagation.rs"]
pub mod gpu_pauli_propagation;
#[path = "transforms/heisenberg_qpe.rs"]
pub mod heisenberg_qpe;
#[path = "simulation/improved_trotter.rs"]
pub mod improved_trotter;
#[path = "variational/low_depth_ucc.rs"]
pub mod low_depth_ucc;
#[path = "variational/meta_vqe.rs"]
pub mod meta_vqe;
#[path = "transforms/optimistic_qft.rs"]
pub mod optimistic_qft;
#[path = "simulation/pauli_propagation.rs"]
pub mod pauli_propagation;
#[path = "simulation/pauli_propagation_gpu.rs"]
pub mod pauli_propagation_gpu;
#[path = "optimization/qamoo.rs"]
pub mod qamoo;
#[path = "optimization/qao.rs"]
pub mod qao;
#[path = "transforms/qft_2d.rs"]
pub mod qft_2d;
#[path = "transforms/qkmm.rs"]
pub mod qkmm;
#[path = "transforms/qpe.rs"]
pub mod qpe;
#[path = "simulation/qram.rs"]
pub mod qram;
#[path = "transforms/qsp_qsvt.rs"]
pub mod qsp_qsvt;
#[path = "transforms/qswift.rs"]
pub mod qswift;
#[path = "optimization/quantum_annealing.rs"]
pub mod quantum_annealing;
#[path = "optimization/qubo_encoder.rs"]
pub mod qubo_encoder;
#[path = "transforms/randomized_qsvt.rs"]
pub mod randomized_qsvt;
#[path = "simulation/schrodinger_feynman.rs"]
pub mod schrodinger_feynman;
#[path = "transforms/shor.rs"]
pub mod shor;
#[path = "transforms/spectrum_amplification.rs"]
pub mod spectrum_amplification;
#[path = "optimization/sqd.rs"]
pub mod sqd;
#[path = "simulation/symmetry_simulation.rs"]
pub mod symmetry_simulation;
#[path = "simulation/tucker_state_prep.rs"]
pub mod tucker_state_prep;
#[path = "variational/vqe.rs"]
pub mod vqe;
#[path = "optimization/warm_start_qaoa.rs"]
pub mod warm_start_qaoa;
