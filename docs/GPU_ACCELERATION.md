# GPU Acceleration

nQPU-Metal supports GPU-accelerated quantum simulation on Apple Metal, NVIDIA CUDA, and AMD ROCm. The auto-backend system selects the fastest execution path for each circuit, but you can override it when needed.

## Table of Contents

- [Metal GPU (macOS)](#metal-gpu-macos)
- [CUDA (NVIDIA)](#cuda-nvidia)
- [ROCm (AMD)](#rocm-amd)
- [Auto-Backend Selection](#auto-backend-selection)
- [UMA Dispatch](#uma-dispatch)
- [M4 Pro Optimizations](#m4-pro-optimizations)
- [Metal 4 Backend](#metal-4-backend)
- [Thermal-Aware Scheduling](#thermal-aware-scheduling)
- [Performance Tuning](#performance-tuning)
- [Benchmarking](#benchmarking)

---

## Metal GPU (macOS)

Metal is the primary GPU backend. It is available on any Mac with Apple Silicon or a discrete AMD GPU.

### Requirements

- macOS (any version with Metal support)
- The `metal` feature flag (included in `default` builds on macOS)

### How It Works

The Metal backend (`MetalSimulator`) compiles compute shaders once at initialization, caches all pipeline states, and batches an entire circuit into a single command buffer. The state vector is stored in `StorageModeShared` buffers, which means zero-copy access from both CPU and GPU on Apple Silicon's unified memory.

Key design decisions:

- **f32 precision on GPU.** Metal lacks native f64 compute, so the GPU path operates in single precision. Use the mixed-precision system (described below) when you need f64 accuracy for specific gates.
- **One compute encoder per gate.** This gives Metal implicit memory barriers between gates without manual synchronization.
- **Pre-compiled pipelines.** Hadamard, Pauli-X/Y/Z, S, T, rotations, CNOT, CZ, SWAP, Toffoli, and generic 2x2/4x4 unitary kernels are all compiled once from `metal/quantum_gates.metal`.

### Usage

```rust
use nqpu_metal::metal_backend::MetalSimulator;
use nqpu_metal::gates::Gate;

let mut sim = MetalSimulator::new(10).unwrap();
sim.run_circuit(&[Gate::h(0), Gate::cnot(0, 1)]);
let probs = sim.probabilities();
```

### Module Reference

| Module | Purpose |
|--------|---------|
| `backends/metal_backend.rs` | Primary Metal simulator with cached pipelines, gate fusion, and thermal scheduling |
| `backends/metal_gpu_full.rs` | `MetalQuantumSimulator` -- full GPU simulator with direct buffer management |
| `backends/metal_gpu_fixed.rs` | `FixedMetalSimulator` -- maximally batched executor (100k+ gates per command buffer, pipelined async execution) |
| `backends/metal_parallel_quantum.rs` | `MetalParallelQuantumExecutor` -- parallel gate kernels, QFT, Grover oracle/diffusion, and transformer attention pipelines |

### Automatic Detection

On macOS, `Device::system_default()` returns the system GPU. No configuration is needed. On non-macOS platforms, Metal constructors return `Err("Metal GPU is only supported on macOS")`.

---

## CUDA (NVIDIA)

### Requirements

- Linux or Windows with an NVIDIA GPU
- CUDA toolkit installed (for `nvrtc` runtime compilation)
- Build with the `cuda` feature flag

```bash
cargo build --release --features cuda
```

### How It Works

The `CudaParallelQuantumExecutor` uses the `cudarc` crate to:

1. Initialize a `CudaDevice` on GPU 0.
2. Compile quantum gate kernels from `cuda/shaders/parallel_quantum.cu` at runtime using NVRTC.
3. Dispatch gate operations as CUDA kernel launches.

```rust
use nqpu_metal::cuda_backend::CudaParallelQuantumExecutor;

let mut executor = CudaParallelQuantumExecutor::new().unwrap();
executor.load_kernels().unwrap();
```

When the `cuda` feature is not enabled, `CudaParallelQuantumExecutor::new()` returns `Err(CudaError::DeviceNotFound)`.

---

## ROCm (AMD)

### Requirements

- Linux with an AMD GPU and the ROCm toolkit installed
- Build with the `rocm` feature flag

```bash
cargo build --release --features rocm
```

The ROCm backend (`RocmParallelQuantumExecutor`) mirrors the CUDA interface but targets AMD's HIP runtime. It uses a `RocmComplex` struct that matches HIP's complex number layout.

> **Note:** The ROCm backend is currently a stub implementation. The dispatch logic and error types are defined, but full kernel execution requires linking against the HIP runtime.

---

## Auto-Backend Selection

nQPU has two layers of automatic backend selection that analyze your circuit and choose the fastest path.

### Layer 1: `AutoBackend` (Gate-Level)

Located in `backends/auto_backend.rs`. Examines qubit count, gate count, circuit depth, entanglement estimate, and Clifford-only detection.

| Condition | Backend | Rationale |
|-----------|---------|-----------|
| 30+ qubits | `Distributed` | State vector too large for single node |
| 25+ qubits, low entanglement (<0.3) | `MPS` | Matrix Product State is efficient here |
| 50+ gates, macOS | `MetalGPU` | GPU parallelism pays off |
| 50+ gates, CUDA enabled | `CudaGPU` | GPU parallelism pays off |
| 10+ qubits, 10+ gates | `F32Fused` | f32 + fusion reduces memory traffic |
| 5+ gates | `Fused` | Gate fusion reduces overhead |
| Everything else | `CPU` | Sequential execution is fastest for tiny circuits |

```rust
use nqpu_metal::auto_backend::{AutoBackend, ExecutionConfig};

// Automatic selection
let selector = AutoBackend::new();
let analysis = selector.analyze(&gates);
println!("Recommended: {} -- {}", analysis.recommended_backend.name(), analysis.reasoning);

// Override with explicit backend
let config = ExecutionConfig::new()
    .with_backend(BackendType::MetalGPU)
    .with_thermal_aware(true);
```

### Layer 2: `AutoSimulator` (Circuit-Level)

Located in `backends/auto_simulator.rs`. Performs deeper circuit analysis including Clifford fraction, T-gate count, magic level, connectivity, and symmetry detection.

| Condition | Backend | Rationale |
|-----------|---------|-----------|
| Noisy circuit, <=13 qubits | `DensityMatrix` | Full noise model support |
| 100% Clifford gates | `Stabilizer` | Exponentially faster (Gottesman-Knill) |
| 30+ qubits, >=90% Clifford | `PauliPropagation` | Efficient for mostly-Clifford circuits |
| Few T-gates (<40), >=90% Clifford | `NearClifford` | CH-form decomposition |
| Low magic (<0.3), few T-gates | `StabilizerTensorNetwork` | Tensor contraction over stabilizer frames |
| 25+ qubits, low entanglement width | `MPS` / `MetalMPS` | Tensor network with optional GPU |
| High gate count, macOS | `MetalGPU` | Metal acceleration |
| High gate count, CUDA | `CudaGPU` | CUDA acceleration |
| 10+ qubits | `StateVectorF32Fused` | f32 + fusion |
| Default | `StateVectorFused` | CPU with gate fusion |

The `AutoSimulator` also supports runtime performance tracking. After each execution, it records actual vs. estimated runtimes and adjusts future routing decisions with correction factors.

```rust
use nqpu_metal::auto_simulator::AutoSimulator;

let mut sim = AutoSimulator::new(&gates, num_qubits, false);
let result = sim.execute(&gates);

// Or force a specific backend
let mut sim = AutoSimulator::with_backend(SimBackend::MetalGPU, 10);
```

---

## UMA Dispatch

Apple Silicon's Unified Memory Architecture (UMA) allows a unique optimization: the CPU and GPU share the same physical memory, so nQPU can route individual gates to whichever processor is faster -- without copying the state vector.

### Gate-Level Routing (`backends/uma_dispatch.rs`)

The `UmaDispatcher` makes per-gate decisions based on a cost model:

| Gate Class | Dispatch Target | Rationale |
|------------|-----------------|-----------|
| Single-qubit (H, X, Y, Z, S, T) | CPU | ~50 ns on CPU; GPU dispatch overhead ~5 us |
| Diagonal gates (Rz, CZ, Phase) | CPU | Element-wise multiply, cache-friendly |
| Multi-qubit gates above threshold | GPU | Massive parallelism pays off at scale |
| Disjoint gate pairs | Both | CPU handles one gate while GPU handles another |

When `enable_adaptive` is set, the dispatcher measures wall-clock times and refines the cost model so that CPU/GPU breakeven thresholds converge to the actual hardware characteristics.

### Concurrent Execution (`backends/concurrent_uma.rs`)

The `ConcurrentUmaExecutor` extends gate-level dispatch with true parallelism. It:

1. Builds a `QubitDependencyGraph` from the circuit.
2. Identifies layers of gates that touch disjoint qubit subsets.
3. For each layer, partitions gates between CPU (via `rayon`) and GPU (via Metal command buffers) and runs them simultaneously.
4. Synchronizes using `std::sync::atomic::fence(Ordering::SeqCst)` on the CPU side and `waitUntilCompleted` on the GPU side -- no data copies needed.

This is only possible on Apple Silicon. CUDA and ROCm platforms require explicit host-device transfers over PCIe, so concurrent CPU+GPU execution on the same state vector is not supported there.

---

## M4 Pro Optimizations

The `backends/m4_pro_optimization.rs` module contains kernel tuning specifically for the Apple M4 Pro GPU architecture.

### M4 Pro Specs (as Configured)

| Parameter | Value |
|-----------|-------|
| GPU cores | 20 |
| L1 cache per core | 128 KB |
| Threadgroup shared memory | 32 KB |
| SIMD shuffle | Enabled |
| Prefetch distance | 8 |

### Adaptive Threadgroup Sizing

`M4ProConfig::for_qubits(n)` selects optimal threadgroup sizes based on state vector size:

| State Vector Size | Threadgroup Size | Rationale |
|-------------------|-----------------|-----------|
| < 4,096 elements | 256 | Smaller groups for better occupancy on small problems |
| 4,096 -- 65,535 | 512 | Balanced throughput |
| >= 65,536 | 1,024 | Maximize memory coalescing on large problems |

The module also computes 2D grid dimensions to improve load balancing across the M4 Pro's 20 GPU cores.

---

## Metal 4 Backend

The `backends/metal4_backend.rs` module targets Metal 4 features available on M4-class hardware (macOS 15+). Enable it with:

```bash
cargo build --release --features metal4
```

The `metal4` feature implies `metal`.

### Capabilities

`Metal4Capabilities::detect()` probes the system GPU at runtime for:

- **Tensor operations** (Metal 4 tensor API for accelerated tensor contraction)
- **Mesh shaders** (Metal 3+)
- **Hardware ray tracing**
- **GPU family** (e.g., `Apple9` for M4)
- **Max buffer length** and **max threads per threadgroup**

The module also includes:

- An ML-based decision-tree predictor for backend routing
- Dynamic Metal shader generation for gate kernels
- GPU profiling and bottleneck identification

When Metal 4 features are unavailable, detection returns conservative defaults and the system falls back to the standard Metal backend.

---

## Thermal-Aware Scheduling

Sustained GPU workloads on Apple Silicon cause thermal throttling, which can produce 3-5x performance variance. The `ThermalScheduler` (`backends/thermal_scheduler.rs`) adapts at runtime.

### How It Works

1. The scheduler maintains a sliding window of the last 10 execution times.
2. When recent times exceed 125% of the baseline (degradation threshold = 0.75), it reduces the threadgroup size to lower power consumption and heat generation.
3. When performance recovers to within 5% of baseline (improvement threshold = 0.95), it gradually increases the threadgroup size.
4. Changes require 2 consecutive confirming samples to avoid oscillation.

### Usage

```rust
use nqpu_metal::thermal_scheduler::ThermalScheduler;
use std::time::Instant;

let mut scheduler = ThermalScheduler::new(256, 1024, 64);

loop {
    let start = Instant::now();
    execute_on_gpu(&mut sim, &gates);
    let elapsed = start.elapsed();

    if let Some(adjusted_size) = scheduler.adjust(elapsed) {
        sim.set_threadgroup_size(adjusted_size);
    }
}
```

The `MetalSimulator` in `metal_backend.rs` integrates the thermal scheduler automatically.

---

## Performance Tuning

### Mixed Precision (`backends/mixed_precision.rs`)

The `MixedPrecisionSimulator` switches between f32 and f64 on a per-gate basis with zero transfer cost on UMA:

| Policy | f32 Usage | f64 Usage |
|--------|-----------|-----------|
| `AlwaysF64` | Never | All gates |
| `AlwaysF32` | All gates | Never |
| `Adaptive` | Forward pass gates | Gradient and measurement gates |
| `Custom` | User-specified | User-specified |

The simulator tracks accumulated precision error by periodically computing the norm difference between f32 and f64 representations. When error exceeds a configurable threshold, it forces f64 until a checkpoint restores accuracy.

On macOS, f32 gates execute through the Metal backend. On other platforms, f32 execution falls back to CPU-side emulation.

### Cache Blocking (`backends/cache_blocking.rs`)

The `CacheAwareQuantumState` optimizes CPU-side gate application by blocking operations to fit within the cache hierarchy:

| Cache Level | Size (M4 Pro) | Block Size |
|-------------|---------------|------------|
| L1 | 128 KB | 1,024 elements |
| L2 | 12 MB | 32,768 elements |
| L3 | 24 MB | 262,144 elements |

`CacheConfig::optimal_block_size(state_size)` selects the largest block that fits in the smallest sufficient cache level. Combined with NEON SIMD alignment, this minimizes cache misses during state vector operations.

### Feature Combinations

Build with multiple GPU backends and optimizations:

```bash
# All GPU backends
cargo build --release --features all-gpus

# Metal with M4 optimizations
cargo build --release --features metal4

# Everything (excludes experimental)
cargo build --release --features full
```

---

## Benchmarking

### Running Benchmarks

nQPU includes a Criterion benchmark suite:

```bash
# Run all benchmarks
cargo bench

# Run the quantum vs classical benchmark
cargo bench --bench quantum_vs_classical
```

The `quantum_vs_classical` benchmark measures Hadamard gate throughput on 10 qubits, including state creation, gate application, and probability extraction.

### Comparing Backends Manually

Use `AutoSimulator` with explicit backend selection to compare performance:

```rust
use nqpu_metal::auto_simulator::{AutoSimulator, SimBackend};
use nqpu_metal::gates::Gate;
use std::time::Instant;

let gates: Vec<Gate> = (0..10).map(|q| Gate::h(q)).collect();

for backend in &[SimBackend::StateVector, SimBackend::StateVectorFused, SimBackend::MetalGPU] {
    let mut sim = AutoSimulator::with_backend(*backend, 10);
    let start = Instant::now();
    let _ = sim.execute(&gates);
    println!("{}: {:?}", backend, start.elapsed());
}
```

### When GPU Beats CPU

GPU acceleration provides the most benefit when:

- **Gate count is high** (50+ gates). GPU dispatch has fixed overhead (~5 us), so small circuits run faster on CPU.
- **State vector is large** (12+ qubits = 4,096+ amplitudes). The GPU's parallel threads need enough work to saturate.
- **Gates are independent.** The UMA concurrent executor can overlap CPU and GPU work on disjoint qubit subsets.

For circuits with fewer than ~50 gates on fewer than ~10 qubits, the CPU backends (`Fused`, `F32Fused`) are typically faster due to lower dispatch overhead.
