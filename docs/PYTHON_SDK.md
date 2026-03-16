# Python SDK Guide

The nQPU Python SDK provides 22 subpackages for applied quantum computing
research. Every package is pure Python, requires only numpy, and follows
consistent API conventions: dataclasses for structured results, numpy arrays
for quantum states, and factory functions for common setups.

This guide covers installation, quick start examples, the full package
reference, shared API patterns, testing, and contributing.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Package Reference](#package-reference)
4. [API Patterns](#api-patterns)
5. [Testing](#testing)
6. [Contributing](#contributing)

---

## Installation

### Prerequisites

- Python 3.9 or later
- NumPy 1.20 or later

### Setup

The Python SDK lives in `sdk/python/`. No `pip install` step is required for
development -- add the SDK directory to your Python path:

```bash
# Option 1: Set PYTHONPATH
export PYTHONPATH="/path/to/nQPU/sdk/python:$PYTHONPATH"

# Option 2: Install in editable mode (if a setup.py/pyproject.toml is present)
cd sdk/python
pip install -e .
```

All 22 subpackages depend only on numpy. No heavy frameworks (Qiskit,
Cirq, PennyLane) are required. Optional visualization features use matplotlib.

### Verify installation

```python
import nqpu.chem
import nqpu.simulation
import nqpu.qkd
print("nQPU Python SDK ready")
```

---

## Quick Start

### Quantum chemistry: find the ground state of H2

```python
from nqpu.chem import MolecularVQE, h2

vqe = MolecularVQE(h2(), basis="sto-3g")
result = vqe.compute_ground_state()
print(f"Ground state energy: {result.energy:.6f} Ha")
print(f"Converged: {result.converged}")
print(f"Iterations: {result.num_iterations}")
```

The `chem` package provides a self-contained quantum chemistry pipeline:
molecular geometry construction, Gaussian integral evaluation, fermion-to-qubit
mapping (Jordan-Wigner, Bravyi-Kitaev, parity), variational ansatze (UCCSD,
hardware-efficient), and an end-to-end VQE driver.

### Hamiltonian simulation: Ising model dynamics

```python
import numpy as np
from nqpu.simulation import ising_model, TrotterEvolution, Magnetization

# Build a 4-qubit transverse-field Ising Hamiltonian
H = ising_model(n_qubits=4, J=1.0, h=0.5)

# Prepare |0000> and evolve with 2nd-order Trotter
psi0 = np.zeros(16, dtype=complex)
psi0[0] = 1.0
evolver = TrotterEvolution(H, order=2)
result = evolver.evolve(psi0, t_final=2.0, n_steps=100)

# Track magnetization over time
mag = Magnetization(n_qubits=4)
m_values = mag.total_trajectory(result.times, result.states)
print(f"Final magnetization: {m_values[-1]:.4f}")
```

### Quantum key distribution: BB84 protocol

```python
from nqpu.qkd import BB84Protocol, QuantumChannel

channel = QuantumChannel(error_rate=0.02, loss_probability=0.1)
protocol = BB84Protocol(seed=42)
result = protocol.generate_key(n_bits=10000, channel=channel)

print(f"QBER: {result.qber:.4f}")
print(f"Raw key length: {result.raw_key_length}")
print(f"Final key length: {len(result.final_key)}")
print(f"Secure: {result.qber < 0.11}")
```

### Error mitigation: zero-noise extrapolation

```python
from nqpu.mitigation import mitigate

# Define a noisy circuit executor (your hardware or simulator)
def noisy_executor(circuit):
    # Returns a noisy expectation value
    ...

result = mitigate(circuit, noisy_executor, method="zne")
print(f"Raw value: {result.raw_values[0]:.6f}")
print(f"Mitigated value: {result.estimated_value:.6f}")
```

---

## Package Reference

### Foundation

| Package | Description | Key exports |
|---------|-------------|-------------|
| `core` | Quantum primitives and shared utilities | -- |
| `metal` | Metal GPU bindings for macOS | -- |
| `physics` | Physics research tools | -- |

### Hardware Backends

| Package | Description | Key exports |
|---------|-------------|-------------|
| `ion_trap` | Trapped-ion backend with digital, analog, and atomic layers | `IonTrapBackend` |
| `superconducting` | Transmon backend with pulse-level control | `SuperconductingBackend` |
| `neutral_atom` | Neutral-atom Rydberg blockade backend with native multi-qubit gates | `NeutralAtomBackend` |
| `benchmarks` | Cross-backend fidelity, gate count, and scaling comparison | `CrossBackendBenchmark`, `BackendComparison` |

### Algorithms and Circuit Tools

| Package | Description | Key exports |
|---------|-------------|-------------|
| `optimizers` | Variational optimizers for VQE, QAOA, and QML | `SPSA`, `Adam`, `COBYLA`, `NaturalGradient`, `VQEOptimizer`, `minimize` |
| `transpiler` | Circuit routing (SABRE), gate cancellation, KAK decomposition | `QuantumCircuit`, `SABRERouter`, `CouplingMap`, `BasisSet`, `optimize`, `decompose` |
| `simulation` | Hamiltonian dynamics: Trotter, QDrift, adiabatic, QITE, VarQTE | `ising_model`, `TrotterEvolution`, `QITE`, `VarQTE`, `Magnetization`, `EntanglementEntropy` |
| `tensor_networks` | MPS, MPO, DMRG ground-state search, TEBD time evolution | `MPS`, `MPO`, `IsingMPO`, `DMRG`, `TEBD`, `dmrg_ground_state` |
| `qcl` | Quantum circuit learning: data encoding, ansatze, QSVM, kernel methods | `CircuitTemplate`, `QCLTrainer`, `QSVM`, `QuantumKernel`, `ExpressibilityAnalyzer` |

### Error Handling

| Package | Description | Key exports |
|---------|-------------|-------------|
| `error_correction` | Stabilizer codes (surface, Steane, Shor, color), decoders (MWPM, union-find, BP), lattice surgery | `SurfaceCode`, `MWPMDecoder`, `UnionFindDecoder`, `ThresholdEstimator`, `LatticeSurgery` |
| `mitigation` | Near-term error mitigation: ZNE, PEC, CDR, Pauli twirling, readout correction | `mitigate`, `ZNEEstimator`, `PECEstimator`, `CDREstimator`, `ReadoutCorrector` |

### Measurement and Verification

| Package | Description | Key exports |
|---------|-------------|-------------|
| `tomography` | State/process/shadow tomography, fidelity, purity, entanglement witnesses | `StateTomographer`, `ProcessTomographer`, `ClassicalShadow`, `concurrence`, `von_neumann_entropy` |
| `qrng` | Quantum random number generation with NIST SP 800-22 testing and certification | `random_bits`, `random_uniform`, `RandomnessReport`, `CHSHCertifier`, `ToeplitzExtractor` |

### Domain Applications

| Package | Description | Key exports |
|---------|-------------|-------------|
| `chem` | Quantum chemistry: molecular geometry, integrals, fermion mapping, VQE | `MolecularVQE`, `UCCSD`, `FermionicHamiltonian`, `jordan_wigner`, `h2`, `lih` |
| `bio` | Quantum biology: photosynthesis, enzyme tunneling, olfaction, avian navigation, DNA mutation | `FMOComplex`, `EnzymeTunneling`, `QuantumNose`, `RadicalPair`, `TautomerTunneling` |
| `finance` | Quantum finance: amplitude estimation option pricing, QAOA portfolio optimization, VaR/CVaR | `QuantumOptionPricer`, `PortfolioOptimizer`, `RiskAnalyzer`, `quantum_var` |
| `trading` | Quantum-enhanced trading: volatility surfaces, regime detection, signal generation, backtesting | `QuantumVolatilitySurface`, `QuantumRegimeDetector`, `QuantumSignalGenerator`, `QuantumBacktester` |
| `qkd` | Quantum key distribution: BB84, E91, B92 protocols with post-processing and network simulation | `BB84Protocol`, `E91Protocol`, `B92Protocol`, `QKDNetwork`, `QuantumChannel` |
| `games` | Quantum game theory, combinatorial optimization (MaxCut, TSP), Bayesian inference, quantum auctions | `PrisonersDilemma`, `MaxCut`, `GraphColoring`, `QuantumBayesian`, `QuantumAuction` |

---

## API Patterns

All 22 packages follow consistent conventions that make the SDK predictable
and easy to learn.

### Dataclass results

Every computation returns a structured result object (typically a dataclass or
named class) rather than raw tuples or dictionaries. Result objects carry both
the primary output and metadata about the computation:

```python
from nqpu.chem import MolecularVQE, h2

result = MolecularVQE(h2()).compute_ground_state()
# result.energy       -- the ground-state energy (float)
# result.converged    -- whether the optimizer converged (bool)
# result.num_iterations -- iteration count (int)
# result.parameters   -- optimal variational parameters (np.ndarray)
```

### NumPy arrays for quantum states

Quantum states are represented as numpy `complex128` arrays throughout the SDK.
Density matrices are 2D arrays. Hamiltonians are represented as lists of
weighted Pauli strings or sparse matrices, never as opaque objects:

```python
import numpy as np

# A 2-qubit state |00>
psi = np.zeros(4, dtype=np.complex128)
psi[0] = 1.0

# Density matrix
rho = np.outer(psi, psi.conj())
```

### Factory functions and predefined objects

Packages provide factory functions for common setups, reducing boilerplate:

```python
from nqpu.chem import h2, lih, h2o          # Predefined molecules
from nqpu.simulation import ising_model      # Standard Hamiltonians
from nqpu.bio import FMOComplex              # FMOComplex.standard()
from nqpu.tensor_networks import IsingMPO    # Pre-built MPO
```

### Circuit representation

Circuits are represented as lists of `(gate_name, qubits, params)` tuples
in packages that do not provide their own circuit class. The `transpiler`
package provides a full `QuantumCircuit` class with a fluent builder API:

```python
from nqpu.transpiler import QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0).cx(0, 1).cx(1, 2)  # Fluent chaining
```

### Executor pattern for mitigation

Error mitigation packages use an executor callback pattern. You provide a
function that takes a circuit and returns an expectation value. The mitigation
method handles noise amplification, sampling, and extrapolation internally:

```python
from nqpu.mitigation import ZNEEstimator, ExtrapolationMethod

estimator = ZNEEstimator(
    noise_factors=[1, 3, 5],
    method=ExtrapolationMethod.LINEAR,
)
result = estimator.estimate(circuit, executor_fn)
```

### Consistent naming

- Classes use `PascalCase`: `MolecularVQE`, `SurfaceCode`, `BB84Protocol`
- Functions use `snake_case`: `jordan_wigner`, `ising_model`, `random_bits`
- Result classes end in `Result`: `VQEResult`, `QKDResult`, `DMRGResult`
- Enums use `PascalCase` members: `ExtrapolationMethod.LINEAR`, `BarrierShape.ECKART`

---

## Testing

### Running the full test suite

Tests live in `sdk/python/tests/` and use pytest:

```bash
cd sdk/python
python -m pytest tests/ -v
```

### Running tests for a single package

```bash
# Test only the chemistry package
python -m pytest tests/test_chem.py -v

# Test only error correction
python -m pytest tests/test_error_correction.py -v
```

### Test conventions

Each package has a corresponding test file (`test_<package>.py`). Tests follow
these conventions:

- **Numerical validation**: Results are compared against known analytical
  values or reference implementations (e.g., Szabo & Ostlund for chemistry,
  Black-Scholes for finance).
- **Deterministic seeds**: Stochastic tests use fixed random seeds for
  reproducibility.
- **Shared fixtures**: The root `conftest.py` provides a seeded numpy RNG
  (`rng`) and a 2-qubit zero state (`small_state`).
- **No external dependencies**: Tests rely only on numpy and pytest. No
  network access, no GPU requirements.

### Test file mapping

| Test file | Package tested |
|-----------|---------------|
| `test_chem.py` | `nqpu.chem` |
| `test_bio.py` | `nqpu.bio` |
| `test_finance.py` | `nqpu.finance` |
| `test_simulation.py` | `nqpu.simulation` |
| `test_error_correction.py` | `nqpu.error_correction` |
| `test_mitigation.py` | `nqpu.mitigation` |
| `test_tomography.py` | `nqpu.tomography` |
| `test_qrng.py` | `nqpu.qrng` |
| `test_qkd.py` | `nqpu.qkd` |
| `test_optimizers.py` | `nqpu.optimizers` |
| `test_transpiler.py` | `nqpu.transpiler` |
| `test_tensor_networks.py` | `nqpu.tensor_networks` |
| `test_qcl.py` | `nqpu.qcl` |
| `test_games.py` | `nqpu.games` |
| `test_benchmarks.py` | `nqpu.benchmarks` |
| `test_neutral_atom.py` | `nqpu.neutral_atom` |

---

## Contributing

### Adding a new package

1. Create a directory under `sdk/python/nqpu/<package_name>/`.
2. Add an `__init__.py` that exports all public classes, functions, and
   constants via `__all__`.
3. Write a module docstring with a usage example in the `__init__.py`.
4. Add a test file `sdk/python/tests/test_<package_name>.py`.
5. Update `docs/ARCHITECTURE.md` and this file with the new package.

### Code conventions

- **Pure Python + numpy only.** Do not introduce dependencies on Qiskit, Cirq,
  PennyLane, or other quantum frameworks. The SDK must remain self-contained.
- **Dataclasses for results.** Every function that returns structured data
  should return a dataclass or a named result object, not a raw dictionary.
- **NumPy arrays for states.** Quantum states are `np.ndarray` with
  `dtype=np.complex128`. Density matrices are 2D arrays.
- **Type annotations.** All public functions and classes should have complete
  type annotations.
- **Docstrings.** Use NumPy-style docstrings with `Parameters`, `Returns`, and
  `Example` sections.
- **Deterministic tests.** Any test involving randomness must use a fixed seed.

### Module structure

A typical package follows this layout:

```
nqpu/<package>/
    __init__.py       # Public API, __all__, module docstring with example
    core_module.py    # Primary implementation
    helpers.py        # Supporting utilities
    constants.py      # Predefined data (optional)
```

The `__init__.py` re-exports everything the user needs. Internal helpers that
are not part of the public API should be prefixed with an underscore or omitted
from `__all__`.

### Running checks before submitting

```bash
# Run the full Python test suite
cd sdk/python && python -m pytest tests/ -v

# Run the Rust test suite
cd sdk/rust && cargo test

# Check for import errors across all packages
python -c "
import nqpu.chem, nqpu.bio, nqpu.finance, nqpu.trading
import nqpu.qkd, nqpu.optimizers, nqpu.mitigation, nqpu.tomography
import nqpu.qrng, nqpu.error_correction, nqpu.qcl, nqpu.simulation
import nqpu.transpiler, nqpu.tensor_networks, nqpu.games, nqpu.benchmarks
print('All packages import successfully')
"
```
