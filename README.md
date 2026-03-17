# nQPU - Neural Quantum Processing Unit

<div align="center">

![nQPU Logo](docs/assets/logo.png)

**High-Performance Quantum Computing SDK with GPU Acceleration**

[![Rust](https://img.shields.io/badge/Rust-1.85+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Metal](https://img.shields.io/badge/Metal-GPU%20Accelerated-green.svg)](https://developer.apple.com/metal/)

</div>

---

## What is nQPU?

nQPU is a **quantum computing platform** that combines:

| Feature | Description |
|---------|-------------|
| **GPU Acceleration** | Metal-based GPU acceleration for 100x speedup |
| **Multiple Backends** | State vector, tensor network, stabilizer simulation |
| **Model QPU Research API** | Model Hamiltonians, exact diagonalization, Rust-backed DMRG, quenches, and parameter sweeps |
| **Drug Discovery** | Quantum molecular fingerprinting & ADMET prediction |
| **Quantum Biology** | Photosynthesis, enzyme catalysis, quantum coherence |
| **Ever-Growing Library** | Drug design, quantum biology, finance, and more |

## Competitive Advantages

| vs Qiskit | vs Cirq | vs PennyLane |
|-----------|---------|--------------|
| ✅ Native GPU (Metal) | ❌ CPU only | ❌ CPU only |
| ✅ Rust core (fast) | ❌ Python only | ❌ Python only |
| ✅ Drug design tools | ❌ General QC only | ❌ General QC only |
| ✅ Growing tool library | ❌ General QC only | ❌ General QC only |
| ✅ Real-time dashboard | ❌ No dashboard | ❌ No dashboard |

## Quick Start

### Installation

```bash
# Core SDK (quantum simulation)
pip install nqpu

# With chemistry/drug design tools
pip install nqpu[chem]

# With biology tools
pip install nqpu[bio]

# With quantum trading tools
pip install nqpu[trading]

# Everything
pip install nqpu[all]

# Or from source
git clone https://github.com/robertcprice/nQPU.git
cd nqpu/sdk/python
pip install -e ".[all]"
```

### Optional Extras

| Package | Install | Description |
|---------|---------|-------------|
| Core | `pip install nqpu` | Quantum simulation, GPU acceleration |
| Chemistry | `pip install nqpu[chem]` | Drug design, ADMET, molecular fingerprints |
| Biology | `pip install nqpu[bio]` | Quantum biology, genome tools |
| Trading | `pip install nqpu[trading]` | Quantum volatility, regime detection, signal processing |
| Finance | `pip install nqpu[finance]` | QAE option pricing, QAOA portfolios, VaR/CVaR (adds matplotlib) |
| Web | `pip install nqpu[web]` | QKD network planning REST API (adds FastAPI + uvicorn) |
| All | `pip install nqpu[all]` | Everything |

### Basic Usage

```python
from nqpu import QuantumCircuit, NQPUBackend

# Create a circuit
circuit = QuantumCircuit(4)
circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)
circuit.measure_all()

# Run with GPU acceleration
backend = NQPUBackend(gpu=True)
result = backend.run(circuit, shots=1000)
print(result.counts)
```

### Quantum Physics Research

```python
import numpy as np
from nqpu import (
    ModelQPU,
    TransverseFieldIsing1D,
    load_entanglement_spectrum_result,
    load_loschmidt_echo_result,
    load_response_spectrum_result,
    load_ground_state_result,
    load_sweep_result,
    load_tensor_network_state,
    load_two_time_correlator_result,
    save_entanglement_spectrum_result,
    save_loschmidt_echo_result,
    save_response_spectrum_result,
    save_ground_state_result,
    save_sweep_result,
    save_tensor_network_state,
    save_two_time_correlator_result,
)

qpu = ModelQPU()
model = TransverseFieldIsing1D(
    num_sites=8,
    coupling=1.0,
    transverse_field=0.8,
    boundary="open",
)

# Ground state with observables and entanglement entropy
ground_state = qpu.ground_state(
    model,
    observables=["magnetization_z", "Z0Z1"],
    subsystem=[0, 1, 2, 3],
)
print(ground_state.ground_state_energy)
print(ground_state.observables)

# Warm-start a nearby ground-state solve from a prior tensor-network state
retuned_state = qpu.ground_state(
    TransverseFieldIsing1D(num_sites=8, coupling=1.0, transverse_field=0.9),
    initial_state=ground_state,
    observables=["magnetization_z"],
)

# Parameter sweep across a phase transition
sweep = qpu.sweep_parameter(
    model,
    "transverse_field",
    np.linspace(0.2, 1.8, 17),
    observables=["magnetization_z"],
    subsystem=[0, 1, 2, 3],
)
print(sweep.energies)
print(sweep.spectral_gaps)
print(sweep.entanglement_entropy)

# Adaptive phase-transition zoom on top of the coarse grid
adaptive = qpu.adaptive_sweep_parameter(
    model,
    "transverse_field",
    [0.2, 0.6, 1.0, 1.4, 1.8],
    observables=["magnetization_z"],
    subsystem=[0, 1, 2, 3],
    metric="spectral_gap",
    max_refinement_rounds=2,
    refinements_per_round=1,
    checkpoint_path="checkpoints/tfim_adaptive.json",
)
print(adaptive.values)
print(adaptive.refinement_history)

# Reuse a prepared state as the initial condition for a quench
quench = qpu.quench(
    TransverseFieldIsing1D(num_sites=8, coupling=1.0, transverse_field=1.1),
    times=[0.0, 0.05, 0.10],
    initial_state=ground_state,
    observables=["magnetization_z", "Z0Z1"],
    subsystem=[0, 1, 2, 3],
)
print(quench.observables["Z0Z1"])
print(quench.entanglement_entropy)

# Build correlation functions and structure factors from the same solver surface
correlations = qpu.correlation_matrix(model, pauli="Z", connected=True)
static_sf = qpu.structure_factor(model, [0.0, np.pi / 2.0, np.pi], connected=True)
dynamic_sf = qpu.dynamic_structure_factor(
    model,
    times=[0.0, 0.05, 0.10],
    momenta=[0.0, np.pi],
    pauli="Z",
    connected=True,
    initial_state="neel",
)
frequency_sf = qpu.frequency_structure_factor(
    dynamic_sf,
    frequencies=[-20.0, 0.0, 20.0],
    window="hann",
)
two_time = qpu.two_time_correlator(
    model,
    times=[0.0, 0.05, 0.10],
    pauli="Z",
    connected=True,
)
local_response = qpu.linear_response_spectrum(
    model,
    times=[0.0, 0.05, 0.10],
    pauli="Z",
    source_sites=[0, 2, 4],
    frequencies=[-20.0, 0.0, 20.0],
    window="hann",
)
spectrum = qpu.entanglement_spectrum(
    model,
    subsystem=[0, 1, 2, 3],
    num_levels=4,
)
echo = qpu.loschmidt_echo(
    model,
    times=[0.0, 0.05, 0.10],
    initial_state="neel",
)
response = qpu.linear_response_spectrum(
    two_time,
    momenta=[0.0, np.pi],
    frequencies=[-20.0, 0.0, 20.0],
    window="hann",
)
print(correlations.matrix)
print(static_sf.values)
print(dynamic_sf.values.shape)
print(frequency_sf.intensity)
print(two_time.values.shape)
print(local_response.measure_sites)
print(spectrum.eigenvalues, spectrum.entanglement_energies)
print(echo.echo, echo.return_rate)
print(response.intensity)

# Checkpoint and restore a Rust-backed tensor-network state
save_tensor_network_state(ground_state, "checkpoints/tfim_ground.json")
restored_state = load_tensor_network_state("checkpoints/tfim_ground.json")

# Save the full ground-state result manifest plus sidecar backend state
save_ground_state_result(ground_state, "checkpoints/tfim_ground_result.json")
restored_ground = load_ground_state_result("checkpoints/tfim_ground_result.json")
save_entanglement_spectrum_result(spectrum, "checkpoints/tfim_entanglement_spectrum.json")
restored_spectrum = load_entanglement_spectrum_result(
    "checkpoints/tfim_entanglement_spectrum.json"
)
save_two_time_correlator_result(two_time, "checkpoints/tfim_two_time.json")
restored_two_time = load_two_time_correlator_result("checkpoints/tfim_two_time.json")
save_response_spectrum_result(response, "checkpoints/tfim_response.json")
restored_response = load_response_spectrum_result("checkpoints/tfim_response.json")
save_loschmidt_echo_result(echo, "checkpoints/tfim_echo.json")
restored_echo = load_loschmidt_echo_result("checkpoints/tfim_echo.json")

# Save a sweep manifest with phase-diagram data and per-point metadata
save_sweep_result(sweep, "checkpoints/tfim_sweep.json")
restored_sweep = load_sweep_result("checkpoints/tfim_sweep.json")

# Long sweeps can checkpoint progress and resume later
resumable = qpu.sweep_parameter(
    model,
    "transverse_field",
    np.linspace(0.2, 1.8, 17),
    observables=["magnetization_z"],
    subsystem=[0, 1, 2, 3],
    checkpoint_path="checkpoints/tfim_resume.json",
)
resumed = qpu.sweep_parameter(
    model,
    "transverse_field",
    np.linspace(0.2, 1.8, 17),
    observables=["magnetization_z"],
    subsystem=[0, 1, 2, 3],
    checkpoint_path="checkpoints/tfim_resume.json",
    resume=True,
)
print(resumed.completed_points, resumed.is_complete)
```

For larger open-boundary 1D chains, `ModelQPU()` will use the Rust tensor-network
path automatically when the optional `nqpu_metal` bindings are installed with
`maturin develop --release --features python`: DMRG for supported ground-state
workflows and TDVP for supported real-time quenches from product states or
previously prepared tensor-network states. The same Rust path can also compress
dense statevectors and dense exact-result states into MPS handles when the
model is supported. Quenches can also request a single entanglement-entropy
trace with `subsystem=[...]`; the Rust TDVP path currently supports prefix cuts
that map onto one MPS bond. Rust tensor-network states can also be checkpointed
to JSON and loaded back into later `quench(...)` calls, and full result
manifests can be saved with observable data plus a backend-state sidecar for
restartable research workflows. Supported product-state labels now
include `"all_up"`, `"all_down"`, `"neel"`, `"plus_x"`, `"minus_x"`, `"plus_y"`,
`"minus_y"`, `"domain_wall"`, `"anti_domain_wall"`, and explicit per-site
strings like `"+-RL01"`. Rust-backed Loschmidt echo manifests also retain the
evolved tensor-network state as a sidecar checkpoint. Parameter sweeps can also
be saved
as JSON manifests that preserve energy curves, spectral gaps, entanglement
traces, solver provenance, per-point model metadata, and checkpoint progress
for resumable long-running phase-diagram jobs. Rust-backed sweeps now also save
per-point tensor-network sidecars so resume can warm-start the next unfinished
point from the last completed point's backend state. Adaptive sweeps build on
the same manifest format: the checkpoint records the evolving sweep grid,
refinement metric, refinement strategy, target value when applicable, and
inserted midpoints so resumed runs continue from the expanded phase-diagram
scan instead of restarting from the seed grid. In addition to the default
lowest-gap refinement, adaptive sweeps now support `strategy="curvature"` for
high-curvature regions and `strategy="target_crossing"` with
`target_value=...` to zoom in on observable crossings. Crossing scans now also
support `insertion_policy="target_linear"` to place the new point at a linear
interpolation estimate of the crossing, or `insertion_policy="equal_spacing"`
with `points_per_interval > 1` to split a selected interval into multiple
interior samples in one refinement round. For Rust-backed sweeps, pending
points are now evaluated by nearest available backend-state anchor in parameter
space instead of raw index order, which reduces cold starts after adaptive grid
expansion or non-prefix checkpoint restores. The same API can also build
ground-state correlation matrices plus static and dynamical structure factors
for `X`, `Y`, or `Z` spin components. Those derived measurements reuse the
existing solver paths, so the Rust-backed route can accelerate them when the
required one- and two-site Pauli observables are supported. Time-resolved
structure-factor data can also be Fourier transformed into a frequency-domain
response through `ModelQPU.frequency_structure_factor(...)` or the exported
`fourier_transform_structure_factor(...)` helper. This frequency-domain view is
currently the Fourier transform of the time-resolved equal-time structure
factor from a quench workflow, not yet a full two-time correlation-function
implementation of canonical `S(q, \omega)`. There is now also an
`ModelQPU.entanglement_spectrum(...)` helper for reduced-density-matrix
eigenvalues and entanglement energies, plus an exact-only
`ModelQPU.loschmidt_echo(...)` helper for return amplitudes, echo probabilities,
and per-site return rates, plus an exact-only two-time correlator and
`ModelQPU.linear_response_spectrum(...)` for
commutator-based response analysis on small dense systems, with optional
`measure_sites=` and `source_sites=` controls to restrict the work to a
selected set of lattice sites. For larger supported open-boundary 1D chains,
`ModelQPU.two_time_correlator(...)` and
`ModelQPU.linear_response_spectrum(...)` can also use an approximate Rust
TDVP-transition path when the reference state is a Rust-backed DMRG ground
state, and `ModelQPU.entanglement_spectrum(...)` can use the Rust tensor-network
path for prefix subsystems that map to a single MPS bond. `ModelQPU.loschmidt_echo(...)`
can also use a Rust TDVP-overlap path on the same supported 1D models when the
initial/reference states are product states or Rust tensor-network backend
states. The current Rust-backed model family now includes open-boundary
`TransverseFieldIsing1D`, `HeisenbergXXZ1D`, and `HeisenbergXYZ1D` chains.
`ModelQPU.dqpt_diagnostics(...)` can then analyze a Loschmidt trace for
candidate DQPT cusps with configurable prominence and slope-jump thresholds.
When a Rust tensor-network state prepared under one Hamiltonian is reused under
another, `ModelQPU.quench(...)`, `ModelQPU.loschmidt_echo(...)`, and
`ModelQPU.dqpt_diagnostics(...)` now require explicit
`initial_state_model=` / `reference_state_model=` provenance instead of
silently accepting a mismatched backend handle. The same explicit provenance
rule now also applies to Rust-backed `ModelQPU.two_time_correlator(...)` and
`ModelQPU.linear_response_spectrum(...)` workflows. DQPT diagnostics can also
be saved and restored with `save_dqpt_diagnostics_result(...)` and
`load_dqpt_diagnostics_result(...)`, including a Rust backend-state sidecar
when present. `ModelQPU.scan_dqpt_parameter(...)` now builds checkpointable
parameter scans of return-rate traces and detected cusp candidates across a
sweep grid. `ModelQPU.adaptive_scan_dqpt_parameter(...)` extends that with the
same refinement controls used by adaptive phase-diagram sweeps, so scans can
insert new parameter points around strong cusp-strength gradients, target
crossings in candidate times, or other DQPT summary traces while remaining
checkpointable and resumable after grid expansion.
`ModelQPU.adaptive_dqpt_diagnostics(...)` does the same thing in time, refining
the Loschmidt/DQPT time grid around sharp return-rate features before a
parameter scan is even necessary.

### Drug Discovery

```python
from nqpu.chem import DrugDesigner, Molecule

# Design drug candidates
designer = DrugDesigner()
molecules = designer.generate_candidates(
    target="BACE1",  # Alzheimer's target
    constraints={"mw": (200, 500), "logp": (-1, 5)}
)

# Get ADMET predictions
for mol in molecules:
    print(f"{mol.smiles}: QED={mol.qed:.3f}, Lipinski={mol.lipinski}")
```

### Quantum Trading

```python
from nqpu.trading import (
    QuantumVolatilitySurface,
    QuantumRegimeDetector,
    QuantumSignalGenerator,
    KellyCriterion,
)
import numpy as np

# Detect market regime
prices = np.array([...])  # your price data
detector = QuantumRegimeDetector(n_qubits=4)
regime = detector.detect(prices)
print(f"Current regime: {regime.label} (confidence: {regime.confidence:.2f})")

# Generate quantum-enhanced trading signals
signal_gen = QuantumSignalGenerator(n_qubits=4)
signals = signal_gen.generate(prices, volume=volume)
print(f"Signal: {signals.direction}, Confidence: {signals.confidence:.2f}")

# Quantum-aware position sizing
kelly = KellyCriterion(n_qubits=3)
position_size = kelly.optimal_fraction(win_rate=0.65, avg_win=0.02, avg_loss=0.01)
```

### Quantum Finance

```python
from nqpu.finance import (
    QuantumOptionPricer, black_scholes_call,
    PortfolioOptimizer, compute_efficient_frontier,
    RiskAnalyzer, RiskConfig, quantum_var,
)

# QAE-based European call pricing vs Black-Scholes
pricer = QuantumOptionPricer(spot=100, strike=100, rate=0.05,
                             volatility=0.2, maturity=1.0)
result = pricer.price()
bs = black_scholes_call(100, 100, 0.05, 0.2, 1.0)
print(f"QAE: {result.price:.2f}  BS: {bs:.2f}")

# QAOA portfolio optimization
optimizer = PortfolioOptimizer(expected_returns, covariance,
                               risk_aversion=1.0, budget=3)
portfolio = optimizer.optimize()
frontier = compute_efficient_frontier(expected_returns, covariance, n_points=20)
```

See `demos/quantum_finance_demo.py` and `demos/quantum_finance_demo.ipynb` for
a full walkthrough covering option pricing, portfolio optimization, VaR/CVaR
risk analysis, and quantum trading signal backtesting.

### QKD Network Planning API

```bash
# Start the REST API
pip install nqpu[web]
nqpu-qkd-api
# -> Swagger UI at http://localhost:8000/docs
```

```python
# Or use programmatically
import requests

# Create a network and add nodes
net = requests.post("http://localhost:8000/networks").json()
nid = net["network_id"]
requests.post(f"http://localhost:8000/networks/{nid}/nodes",
              json={"node_id": "alice", "x": 0, "y": 0})
requests.post(f"http://localhost:8000/networks/{nid}/nodes",
              json={"node_id": "bob", "x": 50, "y": 0})
requests.post(f"http://localhost:8000/networks/{nid}/links",
              json={"node_a": "alice", "node_b": "bob"})

# Establish a quantum key
key = requests.post(f"http://localhost:8000/networks/{nid}/establish-key",
                    json={"node_a": "alice", "node_b": "bob",
                          "protocol": "BB84", "n_bits": 10000}).json()
print(f"Secure: {key['secure']}, QBER: {key['qber']:.4f}")
```

Supports BB84, E91, and B92 protocols, trusted relay paths, star/line/mesh
topology generation, per-link security reports, and eavesdropper simulation.

### QPU Emulator

```python
from nqpu.emulator import QPU, HardwareProfile
from nqpu.transpiler import QuantumCircuit

# Build a GHZ circuit
qc = QuantumCircuit(3)
qc.h(0).cx(0, 1).cx(1, 2)

# Emulate on real hardware profiles
for profile in [HardwareProfile.IONQ_ARIA, HardwareProfile.IBM_HERON,
                HardwareProfile.QUERA_AQUILA]:
    qpu = QPU(profile, noise=True, seed=42, max_qubits=8)
    job = qpu.run(qc, shots=1000)
    r = job.result
    print(f"{qpu.name}: fidelity={r.fidelity_estimate:.4f}, "
          f"depth={r.circuit_depth}, runtime={r.estimated_runtime_us:.0f}us")

# Cross-backend comparison in one call
results = QPU.compare(qc, shots=1000, seed=42)
```

Wraps trapped-ion, superconducting, and neutral-atom backends behind a single
`QPU` interface with 9 real hardware profiles (IonQ Aria/Forte, Quantinuum H2,
IBM Eagle/Heron, Google Sycamore, Rigetti Ankaa-2, QuEra Aquila, Atom Computing).
Supports noisy/ideal modes, statevector extraction, and automatic Toffoli
decomposition (native on neutral-atom, 6-CNOT on others).

### Physics-Application Bridges

```python
from nqpu.bridges import IsingCorrelationModel, HamiltonianVolatility
import numpy as np

# Map asset correlations to a quantum spin model
cov = np.array([[0.04, 0.02], [0.02, 0.09]])
model = IsingCorrelationModel.from_covariance(cov, names=["AAPL", "GOOG"])
print(f"Critical temperature: {model.critical_temperature:.3f}")
print(f"Systemic risk (entanglement): {model.entanglement_risk():.4f}")

# Evolve an implied volatility surface via Hamiltonian dynamics
hv = HamiltonianVolatility(n_strikes=4, coupling=0.5)
result = hv.evolve_surface(np.array([0.2, 0.3, 0.25, 0.35]), t_final=1.0)
print(f"Final vol profile: {result['final_vols']}")
```

Seven bridge modules connecting the physics and simulation engines to domain
applications: Ising models for financial correlations, quantum walks for option
pricing, Hamiltonian dynamics for volatility surfaces, phase transitions for
regime detection, Lindblad master equations for bio/chem validation, game-theoretic
Nash equilibria via Ising ground states, quantum auction models, QAOA vs exact
MaxCut benchmarking, Lindblad volatility surface decoherence, noisy trading
signals, and noise-aware VQE benchmarking across 9 hardware profiles.

### Hardware Decision Engine

```python
from nqpu.emulator import HardwareAdvisor

advisor = HardwareAdvisor()
rec = advisor.recommend([("h", 0), ("cx", 0, 1), ("ccx", 0, 1, 2)])
print(rec.best_profile.name)   # Recommended QPU
print(rec.reasoning)           # Why this hardware is optimal
```

Scores all 9 QPU profiles across fidelity (40%), speed (20%), capacity (15%),
Toffoli efficiency (15%), and connectivity (10%). Toffoli-heavy circuits are
automatically routed to neutral-atom backends with native 3Q gates. Includes
a `full_report()` that executes the circuit on the top 3 platforms for
empirical validation.

### Hardware Benchmarking

```bash
# Run all 7 benchmark circuits across 3 backends (3-8 qubits)
python scripts/run_benchmarks.py

# Generate publication-quality figures
pip install nqpu[finance]  # adds matplotlib
python scripts/generate_figures.py
```

Compares trapped-ion, superconducting, and neutral-atom backends on Bell, GHZ,
QFT, Random Clifford, Toffoli-heavy, QAOA, and Supremacy circuits. Key finding:
neutral-atom backends achieve 83% entangling gate reduction on Toffoli circuits
via native CCZ. See `papers/hardware_benchmarking.md` for the full analysis.

## Architecture

```
nQPU/
├── sdk/
│   ├── python/nqpu/           # Python SDK — 31 packages, numpy-only
│   │   ├── core/              # Quantum state, circuits, backends
│   │   ├── metal/             # Apple Metal GPU utilities
│   │   ├── physics/           # Model QPU research API
│   │   ├── ion_trap/          # Trapped-ion hardware backend
│   │   ├── superconducting/   # Superconducting hardware backend
│   │   ├── neutral_atom/      # Neutral-atom hardware backend
│   │   ├── emulator/          # Multi-backend QPU emulator + hardware advisor
│   │   ├── transpiler/        # Circuit routing, optimization, decomposition
│   │   ├── simulation/        # Hamiltonians, time evolution, Lindblad
│   │   ├── error_correction/  # Stabilizer codes, decoders, noise models
│   │   ├── mitigation/        # ZNE, PEC, twirling, readout correction
│   │   ├── tomography/        # State/process/shadow tomography
│   │   ├── optimizers/        # Classical optimizers for variational algorithms
│   │   ├── qcl/               # Quantum circuit learning, kernels
│   │   ├── tensor_networks/   # MPS, MPO, DMRG, TEBD
│   │   ├── chem/              # Molecular simulation, drug discovery
│   │   ├── bio/               # Quantum biology (photosynthesis, tunneling)
│   │   ├── finance/           # QAE option pricing, portfolio, risk
│   │   ├── trading/           # Quantum volatility, regime detection, signals
│   │   ├── games/             # Game theory, combinatorial optimization
│   │   ├── qkd/               # QKD protocols (BB84, E91, B92)
│   │   ├── qrng/              # Quantum random number generation
│   │   ├── benchmarks/        # Cross-backend benchmarking
│   │   ├── bridges/           # Physics-application integrations
│   │   ├── web/               # FastAPI QKD network planning API
│   │   ├── calibration/       # Hardware calibration & export
│   │   ├── classical_inspired/# Classical benchmarks & linear algebra
│   │   ├── crypto/            # Blind computation, oblivious transfer
│   │   ├── dashboard/         # Cost estimator, benchmark dashboard
│   │   ├── education/         # Interactive tutorials & exercises
│   │   └── visualization/     # Bloch sphere, circuit drawing
│   └── rust/src/              # Rust core engine — 14 domain modules
│       ├── core/              # State vectors, gates, stabilizers, channels
│       ├── algorithms/        # VQE, QAOA, QPE, Shor, QSP/QSVT
│       ├── backends/          # Metal, CUDA, ROCm, pulse control
│       ├── circuits/          # Optimizer, transpiler, QASM/QIR
│       ├── tensor_networks/   # MPS, PEPS, MERA, DMRG
│       ├── error_correction/  # QEC codes, decoders, magic states
│       ├── noise/             # Noise models, error mitigation
│       ├── quantum_ml/        # Kernels, transformers, NQS
│       ├── chemistry/         # Molecular simulation, drug design
│       ├── networking/        # QKD, QRNG, entropy extraction
│       ├── physics/           # Walks, topology, thermodynamics
│       ├── applications/      # Finance, logistics, games
│       ├── measurement/       # Tomography, QCVV, shadows
│       └── infra/             # Traits, FFI, benchmarks, TUI
├── tests/                     # 1400+ pytest tests (22 test files)
├── demos/                     # Jupyter notebooks & demo scripts
├── papers/                    # Research writeups
├── scripts/                   # Benchmark & figure generation scripts
└── docs/                      # Documentation & guides
```

## Features

### Quantum Backends

| Backend | Qubits | Speed | Best For |
|---------|--------|-------|----------|
| State Vector | 30+ | Fast | General circuits |
| Tensor Network (MPS) | 100+ | Very Fast | Shallow circuits |
| Stabilizer | 1000+ | Ultra Fast | Clifford circuits |
| GPU (Metal) | 30+ | 100x faster | Apple Silicon |

### Chemistry Tools

- **Molecular Fingerprints**: ECFP4/6, MACCS, Atom Pair, Topological Torsion
- **ADMET Prediction**: Absorption, Distribution, Metabolism, Excretion, Toxicity
- **Drug-likeness**: Lipinski Ro5, QED score, Synthetic Accessibility
- **Similarity Search**: Tanimoto, Dice coefficients

### Biology Tools

- **Quantum Coherence**: Photosynthetic complex simulation
- **Enzyme Catalysis**: Tunneling rate calculations
- **Genomic Analysis**: DNA/RNA quantum encoding

## Benchmarks

| Operation | CPU | GPU (Metal) | Speedup |
|-----------|-----|-------------|---------|
| 20-qubit random circuit | 1.2s | 12ms | 100x |
| 30-qubit QFT | 4.5s | 45ms | 100x |
| 1000-shot sampling | 0.8s | 8ms | 100x |
| Molecular fingerprint | 50ms | 0.5ms | 100x |

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) — Installation and first steps
- [Architecture](docs/ARCHITECTURE.md) — System design overview
- [Physics Research](docs/PHYSICS_RESEARCH.md) — Model QPU workflows for quantum-physics studies
- [Quantum Domains](docs/QUANTUM_DOMAINS.md) — Guide to each domain directory
- [GPU Acceleration](docs/GPU_ACCELERATION.md) — Metal, CUDA, and ROCm setup
- [Drug Discovery](docs/DRUG_DISCOVERY.md) — Drug design workflow guide
- [Rust SDK](docs/RUST_SDK.md) — Using the Rust crate directly
- [TUI Guide](docs/TUI.md) — Terminal UI with Bloch sphere visualization

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{nqpu2025,
  title = {nQPU: Neural Quantum Processing Unit},
  author = {Entropy Research},
  year = {2025},
  url = {https://github.com/robertcprice/nQPU}
}
```

---

<div align="center">

**Built with ❤️ by [Entropy Research](https://github.com/entropy-research)**

</div>
