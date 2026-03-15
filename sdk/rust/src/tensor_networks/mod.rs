//! Tensor network methods including MPS, PEPS, MERA, TEBD, DMRG, and advanced contraction strategies.

// MPS-focused simulators, mappers, and related helpers.
#[path = "mps/adaptive_mps.rs"]
pub mod adaptive_mps;
#[path = "mps/camps.rs"]
pub mod camps;
#[path = "mps/differentiable_mps.rs"]
pub mod differentiable_mps;
#[path = "mps/lattice_mps.rs"]
pub mod lattice_mps;
#[path = "mps/lattice_mps_4d.rs"]
pub mod lattice_mps_4d;
#[path = "mps/mps_simulator.rs"]
pub mod mps_simulator;
pub use mps_simulator as tensor_network;
#[path = "mps/snake_mapping.rs"]
pub mod snake_mapping;

// Time evolution and sweep-based tensor-network solvers.
#[path = "evolution/cluster_tebd.rs"]
pub mod cluster_tebd;
#[path = "evolution/dmrg_tdvp.rs"]
pub mod dmrg_tdvp;
#[path = "evolution/tebd.rs"]
pub mod tebd;
#[path = "evolution/tensor_jump.rs"]
pub mod tensor_jump;
#[path = "evolution/time_evolution.rs"]
pub mod time_evolution;

// Contraction planning and environment construction.
#[path = "contraction/contraction_optimizer.rs"]
pub mod contraction_optimizer;
#[path = "contraction/ctm_contraction.rs"]
pub mod ctm_contraction;

// Higher-dimensional PEPS/iPEPS-style methods.
#[path = "higher_dimensional/advanced_tensor_networks.rs"]
pub mod advanced_tensor_networks;
#[path = "higher_dimensional/imps_ipeps.rs"]
pub mod imps_ipeps;
#[path = "higher_dimensional/pepo.rs"]
pub mod pepo;
#[path = "higher_dimensional/peps.rs"]
pub mod peps;
#[path = "higher_dimensional/peps_gates.rs"]
pub mod peps_gates;
#[path = "higher_dimensional/peps_simulator.rs"]
pub mod peps_simulator;
#[path = "higher_dimensional/simulation_3d.rs"]
pub mod simulation_3d;

// Alternative tensor-network architectures and hybrid formulations.
#[path = "architectures/arbitrary_tn.rs"]
pub mod arbitrary_tn;
#[path = "architectures/bayesian_treepes.rs"]
pub mod bayesian_treepes;
#[path = "architectures/fermionic_tensor_net.rs"]
pub mod fermionic_tensor_net;
#[path = "architectures/mast.rs"]
pub mod mast;
#[path = "architectures/mera_happy.rs"]
pub mod mera_happy;
#[path = "architectures/stabilizer_tensor_net.rs"]
pub mod stabilizer_tensor_net;
#[path = "architectures/tree_tensor_network.rs"]
pub mod tree_tensor_network;

#[cfg(target_os = "macos")]
#[path = "acceleration/dmrg_metal.rs"]
pub mod dmrg_metal;
#[cfg(target_os = "macos")]
#[path = "acceleration/gpu_mps.rs"]
pub mod gpu_mps;
#[cfg(target_os = "macos")]
#[path = "acceleration/metal_mps.rs"]
pub mod metal_mps;
#[path = "acceleration/gpu_dmrg.rs"]
pub mod gpu_dmrg;
