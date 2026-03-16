# nQPU Documentation

nQPU is a high-performance quantum computing SDK built on a Rust core with a
pure-Python SDK on top. It provides GPU-accelerated quantum simulation, 14 Rust
domain modules spanning the full quantum computing stack, and 22 Python
subpackages for applied quantum computing -- from chemistry and biology to
finance, cryptography, and game theory. Whether you are running your first
quantum circuit or building fault-tolerant error correction pipelines, these
guides will help you get started and go deep.

Project logos are available in [docs/assets/](assets/).

---

## Documentation Index

| Guide | Description |
|-------|-------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Installation, environment setup, and running your first quantum circuit |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design overview: 14 Rust domain modules, 22 Python subpackages, backend routing, and key design decisions |
| [PYTHON_SDK.md](PYTHON_SDK.md) | Python SDK guide: installation, quick start examples, package reference, API patterns, and testing |
| [QUANTUM_DOMAINS.md](QUANTUM_DOMAINS.md) | Educational guide to each of the 14 Rust domain directories, covering the underlying math/physics, key files, prerequisites, and example workflows |
| [GPU_ACCELERATION.md](GPU_ACCELERATION.md) | Setting up Metal (macOS), CUDA (NVIDIA), and ROCm (AMD) backends for GPU-accelerated simulation |
| [DRUG_DISCOVERY.md](DRUG_DISCOVERY.md) | End-to-end drug design workflow: molecular Hamiltonians, candidate generation, ADMET prediction, and virtual screening |
| [RUST_SDK.md](RUST_SDK.md) | Using the Rust crate directly: API reference, trait system, custom backends, and performance tuning |
| [TUI.md](TUI.md) | Terminal UI guide: launching the dashboard, monitoring simulations, and interpreting real-time metrics |

---

## Feature Overview

### Rust Core (14 domain modules)

- **Statevector simulation** with SIMD-accelerated gate operations and automatic backend routing
- **Tensor networks**: MPS, PEPS, MERA, DMRG, TEBD with Metal GPU acceleration
- **Error correction**: surface codes, Floquet codes, QLDPC, MWPM/BP-OSD/neural decoders, magic state distillation
- **Noise modeling**: device-calibrated noise, Lindblad dynamics, non-Markovian processes, leakage simulation
- **Algorithms**: VQE, QAOA, QPE, Grover, Shor, HHL, amplitude estimation, quantum walks, QSP/QSVT
- **Quantum ML**: variational circuits, quantum kernels, NQS, reservoir computing, JAX/PyTorch bridges
- **Chemistry**: molecular Hamiltonians, double-factorized methods, drug design, materials simulation
- **GPU backends**: Apple Metal (with UMA dispatch), NVIDIA CUDA, AMD ROCm
- **Circuit tools**: optimizer, ZX-calculus, SABRE routing, OpenQASM 2/3, QIR, circuit cutting
- **Networking**: QKD protocols, QRNG with NIST testing, post-quantum cryptography assessment
- **Physics**: topological QC, quantum walks, thermodynamics, quantum biology, IIT
- **Applications**: finance, logistics, game theory, NLP, generative art
- **Measurement**: state/process tomography, classical shadows, QCVV, quantum Fisher information
- **Infrastructure**: autodiff, distributed MPI, WASM backend, Python/C FFI, benchmarking

### Python SDK (22 subpackages)

All packages are pure Python with numpy as the only required dependency.

| Category | Packages |
|----------|----------|
| **Hardware backends** | `ion_trap`, `superconducting`, `neutral_atom`, `benchmarks` |
| **Algorithms & circuits** | `optimizers`, `transpiler`, `simulation`, `tensor_networks`, `qcl` |
| **Error handling** | `error_correction`, `mitigation` |
| **Measurement** | `tomography`, `qrng` |
| **Chemistry & biology** | `chem`, `bio` |
| **Finance** | `finance`, `trading` |
| **Cryptography** | `qkd` |
| **Game theory** | `games` |
| **Foundation** | `core`, `metal`, `physics` |
