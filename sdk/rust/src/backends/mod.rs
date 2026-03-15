//! Hardware backends: Metal, CUDA, ROCm, UMA dispatch, pulse control, and hardware providers.

// Runtime orchestration and backend selection.
#[path = "runtime/auto_backend.rs"]
pub mod auto_backend;
#[path = "runtime/auto_simulator.rs"]
pub mod auto_simulator;
#[path = "runtime/auto_tuning.rs"]
pub mod auto_tuning;
#[path = "runtime/concurrent_uma.rs"]
pub mod concurrent_uma;
#[path = "runtime/thermal_scheduler.rs"]
pub mod thermal_scheduler;
#[path = "runtime/uma_dispatch.rs"]
pub mod uma_dispatch;

// Cross-platform accelerator utilities.
#[path = "gpu/cache_blocking.rs"]
pub mod cache_blocking;
#[path = "gpu/f32_fusion.rs"]
pub mod f32_fusion;
#[path = "gpu/gpu_memory_pool.rs"]
pub mod gpu_memory_pool;
#[path = "gpu/mixed_precision.rs"]
pub mod mixed_precision;

// Hardware-provider and lab-facing backends.
#[path = "hardware/google_quantum.rs"]
pub mod google_quantum;
#[path = "hardware/hardware_calibration.rs"]
pub mod hardware_calibration;
#[path = "hardware/ibm_quantum.rs"]
pub mod ibm_quantum;
#[path = "hardware/live_calibration.rs"]
pub mod live_calibration;
#[path = "hardware/neutral_atom_array.rs"]
pub mod neutral_atom_array;
#[path = "hardware/neutral_atom_backend.rs"]
pub mod neutral_atom_backend;
#[path = "hardware/photonic_advantage.rs"]
pub mod photonic_advantage;
#[path = "hardware/pinnacle_architecture.rs"]
pub mod pinnacle_architecture;
#[path = "hardware/superconducting.rs"]
pub mod superconducting;
#[path = "hardware/digital_twin.rs"]
pub mod digital_twin;
#[path = "hardware/trapped_ion.rs"]
pub mod trapped_ion;

// Pulse-level control and simulation.
#[path = "pulse/pulse_control.rs"]
pub mod pulse_control;
#[path = "pulse/pulse_level.rs"]
pub mod pulse_level;
#[path = "pulse/pulse_simulation.rs"]
pub mod pulse_simulation;
#[path = "pulse/dispersive_readout.rs"]
pub mod dispersive_readout;
#[path = "pulse/cr_calibration.rs"]
pub mod cr_calibration;
#[path = "pulse/transmon_drag.rs"]
pub mod transmon_drag;
#[path = "pulse/grape_optimizer.rs"]
pub mod grape_optimizer;
#[cfg(target_os = "macos")]
#[path = "pulse/metal_pulse_backend.rs"]
pub mod metal_pulse_backend;

// macOS-only Metal GPU backends
#[cfg(target_os = "macos")]
#[path = "metal/amx_tensor.rs"]
pub mod amx_tensor;
#[cfg(target_os = "macos")]
#[path = "metal/m4_pro_optimization.rs"]
pub mod m4_pro_optimization;
#[cfg(target_os = "macos")]
#[path = "metal/metal4_backend.rs"]
pub mod metal4_backend;
#[cfg(target_os = "macos")]
#[path = "metal/metal_gpu_fixed.rs"]
pub mod metal_gpu_fixed;
#[cfg(target_os = "macos")]
#[path = "metal/metal_gpu_full.rs"]
pub mod metal_gpu_full;
#[cfg(target_os = "macos")]
#[path = "metal/metal_parallel_quantum.rs"]
pub mod metal_parallel_quantum;
#[path = "metal/metal_backend.rs"]
pub mod metal_backend;
#[path = "metal/metal_stabilizer.rs"]
pub mod metal_stabilizer;
#[cfg(target_os = "macos")]
#[path = "metal/metal_state.rs"]
pub mod metal_state;
#[cfg(target_os = "macos")]
#[path = "metal/t1t2_integrated.rs"]
pub mod t1t2_integrated;
#[cfg(target_os = "macos")]
#[path = "metal/tensor_ops.rs"]
pub mod tensor_ops;

// CUDA backend
#[cfg(feature = "cuda")]
#[path = "gpu/cuda_backend.rs"]
pub mod cuda_backend;

// ROCm backend
#[cfg(feature = "rocm")]
#[path = "gpu/rocm_backend.rs"]
pub mod rocm_backend;
