# nQPU Documentation

nQPU is a high-performance quantum computing SDK built on a Rust core with Python bindings. It provides GPU-accelerated quantum simulation, a comprehensive algorithm library spanning 14 domain areas, and applied workflows for drug discovery, quantum machine learning, finance, and more. Whether you are running your first quantum circuit or building fault-tolerant error correction pipelines, these guides will help you get started and go deep.

Project logos are available in [docs/assets/](assets/).

---

## Documentation Index

| Guide | Description |
|-------|-------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Installation, environment setup, and running your first quantum circuit |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design overview: Rust core, Python SDK, backend architecture, and module boundaries |
| [QUANTUM_DOMAINS.md](QUANTUM_DOMAINS.md) | Educational guide to each of the 14 domain directories in the Rust SDK, covering the underlying math/physics, key files, prerequisites, and example workflows |
| [GPU_ACCELERATION.md](GPU_ACCELERATION.md) | Setting up Metal (macOS), CUDA (NVIDIA), and ROCm (AMD) backends for GPU-accelerated simulation |
| [DRUG_DISCOVERY.md](DRUG_DISCOVERY.md) | End-to-end drug design workflow: molecular Hamiltonians, candidate generation, ADMET prediction, and virtual screening |
| [RUST_SDK.md](RUST_SDK.md) | Using the Rust crate directly: API reference, trait system, custom backends, and performance tuning |
| [TUI.md](TUI.md) | Terminal UI guide: launching the dashboard, monitoring simulations, and interpreting real-time metrics |
