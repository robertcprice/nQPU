# Rust Source Layout

`src/` is organized by responsibility instead of keeping hundreds of unrelated
modules at the crate root.

Directories:
- `core/`: state vectors, gates, stabilizers, channels, and primitive simulators
- `tensor_networks/`: MPS, PEPS, TEBD, DMRG, contraction, and tensor-network utilities
- `algorithms/`: QFT, VQE, QAOA, QPE, QSVT, Grover, and other algorithmic workflows
- `error_correction/`: codes, decoders, magic-state tooling, and QEC interop
- `noise/`: noise channels, mitigation, Lindblad/open-system simulation
- `chemistry/`: chemistry, materials, and drug-design modules
- `quantum_ml/`: kernels, neural-network layers, embeddings, and QML bridges
- `backends/`: Metal, CUDA, ROCm, pulse, hardware, and simulator backends
- `circuits/`: transpilation, QASM/QIR, circuit optimization, and visualization
- `networking/`: QKD, QRNG, entropy extraction, and networking protocols
- `measurement/`: tomography, shadows, QCVV, and verification
- `physics/`: many-body, topology, thermodynamics, walks, and related models
- `applications/`: domain-facing demos and application modules
- `infra/`: traits, FFI, bindings, benchmarks, utilities, and distributed plumbing
- `bin/`: standalone binaries and experiments

Naming conventions:
- Prefer descriptive module names such as `mps_simulator.rs` over repeated names
  like `tensor_network.rs`.
- `mod.rs` files define the domain boundary and may expose backward-compatible
  aliases while imports migrate.
- Keep new modules inside the closest domain folder instead of adding new files
  to `src/` directly.
- When a domain grows too large, add subfolders inside that domain, such as
  `algorithms/variational/` or `algorithms/transforms/`, instead of creating a
  second flat pile.
- Use subfolders that describe responsibility, such as `backends/hardware/`,
  `backends/metal/`, `error_correction/decoders/`, or
  `error_correction/workflows/`.
- Apply the same split inside support domains, for example `noise/models/`,
  `noise/mitigation/`, and `noise/dynamics/`.
- Keep circuit code grouped by purpose too, such as `circuits/analysis/`,
  `circuits/formats/`, `circuits/synthesis/`, and `circuits/visualization/`.
- Keep infrastructure split by function, for example `infra/bindings/`,
  `infra/benchmarks/`, `infra/distributed/`, `infra/execution/`, and
  `infra/research/`.
- Group networking code by responsibility too, for example
  `networking/protocols/`, `networking/randomness/`,
  `networking/hardware/`, and `networking/validation/`.
- Apply the same pattern to domain folders like `measurement/` and
  `chemistry/`, e.g. `measurement/tomography/` or
  `chemistry/electronic_structure/`.
- Split tensor-network code by method family, for example `tensor_networks/mps/`,
  `tensor_networks/evolution/`, `tensor_networks/contraction/`,
  `tensor_networks/higher_dimensional/`, and `tensor_networks/acceleration/`.
- Keep `core/` split between gate definitions, stabilizer methods, state models,
  computation styles, and foundational interfaces instead of mixing them in one
  flat folder.
- Keep provider integrations and application demos grouped too, such as
  `qpu/providers/`, `qpu/support/`, `applications/games/`, and
  `applications/creative/`.
- Separate physics modules by actual research area, such as
  `physics/matter/`, `physics/information/`, `physics/transport/`, and
  `physics/consciousness/`, instead of leaving them in one flat folder.
