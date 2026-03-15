# Quantum Drug Discovery with nQPU

End-to-end guide for using the nQPU quantum computing SDK to accelerate drug discovery workflows, from molecular Hamiltonian construction through lead optimization.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quantum Computational Chemistry Foundations](#quantum-computational-chemistry-foundations)
- [Setting Up a Molecular Hamiltonian](#setting-up-a-molecular-hamiltonian)
- [Running VQE for Ground State Energy](#running-vqe-for-ground-state-energy)
- [Drug-Target Interaction Screening](#drug-target-interaction-screening)
- [Active Space Selection Strategies](#active-space-selection-strategies)
- [Error Mitigation for Chemistry Calculations](#error-mitigation-for-chemistry-calculations)
- [Example: Computing Binding Energy of H2O](#example-computing-binding-energy-of-h2o)
- [Performance Considerations](#performance-considerations)
- [Module Reference](#module-reference)

---

## Overview

nQPU provides a complete quantum chemistry stack for drug discovery, built on top of the `nqpu-metal` Rust crate. The stack connects molecular structure definition to quantum simulation through several layers:

1. **Molecular representation** -- Define molecules with atoms, bonds, and 3D coordinates.
2. **Hamiltonian construction** -- Convert molecular integrals into qubit Hamiltonians via fermion-to-qubit mappings (Jordan-Wigner, Bravyi-Kitaev, Parity).
3. **Ground state solvers** -- Find molecular ground state energies using VQE or ADAPT-VQE.
4. **Drug design modules** -- Score protein-ligand docking, predict ADMET properties, evaluate drug-likeness, and run lead optimization.

The drug design module is based on results from Nature Biotechnology (2025) showing 21.5% improvement over classical-only approaches for lead compound identification.

## Prerequisites

Enable the `chemistry` feature flag in your `Cargo.toml`:

```toml
[dependencies]
nqpu-metal = { path = "sdk/rust", features = ["chemistry"] }
```

For GPU-accelerated simulation on macOS, add the `metal` feature:

```toml
nqpu-metal = { path = "sdk/rust", features = ["chemistry", "metal"] }
```

For the full stack including visualization and distributed computing:

```toml
nqpu-metal = { path = "sdk/rust", features = ["full"] }
```

## Quantum Computational Chemistry Foundations

Classical computational chemistry methods like Hartree-Fock and DFT approximate the electronic Schrodinger equation. Quantum computers can, in principle, represent the full many-body wavefunction directly, avoiding the exponential scaling of exact classical methods (Full CI).

The key idea: map the molecular electronic Hamiltonian

```
H = sum_{pq} h_{pq} a+_p a_q + (1/2) sum_{pqrs} g_{pqrs} a+_p a+_q a_s a_r + E_nuc
```

onto qubit operators using fermion-to-qubit transformations, then find the ground state energy using a variational quantum algorithm.

**nQPU implements three fermion-to-qubit mappings:**

| Mapping | Qubit Count | Pauli Weight | Best For |
|---------|-------------|-------------|----------|
| Jordan-Wigner | N (spin orbitals) | O(N) per operator | Small molecules, simplicity |
| Bravyi-Kitaev | N (spin orbitals) | O(log N) per operator | Medium molecules, lower gate depth |
| Parity | N (spin orbitals) | O(1) for number-conserving | Exploiting symmetries |

## Setting Up a Molecular Hamiltonian

There are three approaches to constructing a molecular Hamiltonian, depending on your starting point.

### Approach 1: Use predefined molecules

nQPU ships with hardcoded integrals for common benchmark molecules:

```rust
use nqpu_metal::molecular_integrals::{
    MolecularHamiltonian, QubitHamiltonian,
    jordan_wigner, bravyi_kitaev, parity_mapping,
    water_molecule,
};

// Load predefined H2O integrals (7 spatial orbitals, 10 electrons)
let mol_ham = water_molecule();

// Map to qubit Hamiltonian using Jordan-Wigner
let qubit_ham: QubitHamiltonian = jordan_wigner(&mol_ham);

println!("Qubits required: {}", qubit_ham.num_qubits);
println!("Pauli terms: {}", qubit_ham.terms.len());
println!("Nuclear repulsion: {:.6} Ha", qubit_ham.constant);
```

### Approach 2: Parse an FCIDUMP file

FCIDUMP is the standard format for exchanging molecular integrals between quantum chemistry programs (PySCF, Psi4, Gaussian). nQPU parses these natively:

```rust
use nqpu_metal::molecular_integrals::{
    parse_fcidump, build_molecular_hamiltonian, jordan_wigner,
};

// Read integrals from an FCIDUMP file
let fcidump_content = std::fs::read_to_string("molecule.fcidump")
    .expect("Failed to read FCIDUMP file");

// Parse the file (converts 1-indexed to 0-indexed automatically)
let fcidump_data = parse_fcidump(&fcidump_content)
    .expect("Failed to parse FCIDUMP");

println!("Spatial orbitals: {}", fcidump_data.norb);
println!("Electrons: {}", fcidump_data.nelec);

// Build the second-quantized Hamiltonian in spin-orbital basis
let mol_ham = build_molecular_hamiltonian(&fcidump_data);

// Convert to qubit Hamiltonian
let qubit_ham = jordan_wigner(&mol_ham);
```

FCIDUMP files follow this format:

```text
&FCI NORB=4, NELEC=2, MS2=0,
 ORBSYM=1,1,1,1,
&END
  0.674588     1  1  1  1
  0.181210     1  1  2  2
  ...
  -1.252477    1  1  0  0
  -0.475934    2  2  0  0
  0.706141     0  0  0  0
```

### Approach 3: Build from one- and two-electron integrals

If you have integrals from an external program, construct the Hamiltonian directly:

```rust
use nqpu_metal::quantum_chemistry::JordanWignerMapper;

// One-electron integrals h_{pq} (4 spatial orbitals)
let one_electron = vec![
    vec![-1.2525, 0.0, 0.0, 0.0],
    vec![ 0.0,   -0.4760, 0.0, 0.0],
    vec![ 0.0,    0.0,   0.5, 0.0],
    vec![ 0.0,    0.0,   0.0, 0.6],
];

// Two-electron integrals g_{pqrs} (simplified diagonal terms)
let n = 4;
let two_electron = vec![vec![vec![vec![0.0; n]; n]; n]; n];
// ... populate with your integrals ...

// Build Hamiltonian using JordanWignerMapper
let hamiltonian = JordanWignerMapper::build_hamiltonian(
    &one_electron, &two_electron, n,
);
```

## Running VQE for Ground State Energy

The Variational Quantum Eigensolver (VQE) finds the ground state energy by optimizing a parameterized quantum circuit (ansatz) to minimize the energy expectation value. nQPU provides two VQE variants.

### Standard VQE

Uses a fixed-structure hardware-efficient ansatz with parameter-shift rule gradients:

```rust
use nqpu_metal::vqe::{VQESolver, Hamiltonian, hamiltonians};

// Example: H2 molecule Hamiltonian at equilibrium bond length
// Coefficients from STO-3G basis at R = 0.735 A
let h2_hamiltonian = hamiltonians::hydrogen_molecule(
    -0.8105,  // g0: constant (identity)
     0.1721,  // g1: Z_0 coefficient
    -0.2257,  // g2: Z_1 coefficient
     0.1709,  // g3: Z_0 Z_1 coefficient
     0.0453,  // g4: X_0 X_1 coefficient
     0.0453,  // g5: Y_0 Y_1 coefficient
);

// Create VQE solver
// Parameters: num_qubits, ansatz_depth, hamiltonian, learning_rate
let mut solver = VQESolver::new(2, 3, h2_hamiltonian, 0.1);
solver.max_iterations = 500;
solver.convergence_threshold = 1e-6;

// Run optimization
let result = solver.find_ground_state();

println!("Ground state energy: {:.6} Ha", result.ground_state_energy);
println!("Converged: {}", result.converged);
println!("Iterations: {}", result.iterations);
println!("Final parameters: {:?}", result.parameters);
```

### ADAPT-VQE

ADAPT-VQE dynamically builds the ansatz by iteratively selecting the operator with the largest energy gradient from an operator pool. This produces shorter circuits than fixed UCCSD:

```rust
use nqpu_metal::adapt_vqe::{AdaptVqe, OperatorPool};

// Build operator pool for 4 spin-orbitals, 2 electrons
let pool = OperatorPool::generalized_singles_doubles(4, 2);

// Create ADAPT-VQE engine
let mut engine = AdaptVqe::new(pool, 4, 2);
engine.gradient_threshold = 1e-3;  // Convergence criterion
engine.max_adapt_cycles = 20;

// Build the molecular Hamiltonian (using molecular_integrals module)
let hamiltonian = /* your MolecularHamiltonian */;

// Run ADAPT-VQE
let result = engine.run(&hamiltonian);

println!("Energy: {:.6} Ha", result.energy);
println!("Converged: {}", result.converged);
println!("Operators selected: {}", result.num_operators);
println!("Circuit depth: {}", result.circuit_depth);
```

**Choosing between VQE and ADAPT-VQE:**

| Criterion | Standard VQE | ADAPT-VQE |
|-----------|-------------|-----------|
| Circuit depth | Fixed, can be deep | Adaptive, typically shorter |
| Convergence | Depends on ansatz choice | Systematically improvable |
| Classical overhead | Lower (fixed structure) | Higher (gradient screening each cycle) |
| Best for | Quick prototyping, small molecules | Production accuracy, NISQ devices |

## Drug-Target Interaction Screening

The `quantum_drug_design` module provides a complete pipeline for screening candidate molecules against a protein target.

### Step 1: Define molecules

```rust
use nqpu_metal::quantum_drug_design::{
    Molecule, Element, BondType, Atom,
};

// Build a water molecule
let mut water = Molecule::new("water");
let o = water.add_atom(Element::O, [0.0, 0.0, 0.0], -0.834);
let h1 = water.add_atom(Element::H, [0.757, 0.586, 0.0], 0.417);
let h2 = water.add_atom(Element::H, [-0.757, 0.586, 0.0], 0.417);
water.add_bond(o, h1, BondType::Single);
water.add_bond(o, h2, BondType::Single);

// Validate structure
water.validate().expect("Invalid molecule");

// Inspect computed properties
println!("Molecular weight: {:.2} Da", water.molecular_weight());
println!("H-bond donors: {}", water.h_bond_donors());
println!("H-bond acceptors: {}", water.h_bond_acceptors());
println!("LogP estimate: {:.2}", water.estimated_log_p());
```

### Step 2: Evaluate drug-likeness

Before running expensive quantum simulations, filter candidates with drug-likeness checks:

```rust
use nqpu_metal::quantum_drug_design::evaluate_drug_likeness;

let result = evaluate_drug_likeness(&candidate_molecule);

println!("Lipinski violations: {}/4", result.lipinski_violations);
println!("QED score: {:.3}", result.qed_score);  // 0-1, higher is better
println!("Synthetic accessibility: {:.3}", result.synthetic_accessibility);

// Filter: proceed only if drug-like
if result.lipinski_violations <= 1 && result.qed_score > 0.4 {
    // Continue to quantum scoring...
}
```

The drug-likeness evaluation includes:
- **Lipinski Rule of Five**: MW < 500, LogP < 5, HBD <= 5, HBA <= 10
- **QED score**: Quantitative Estimate of Drug-likeness (Bickerton et al. 2012), weighted geometric mean of desirability functions
- **Synthetic accessibility**: Estimated ease of synthesis based on structural complexity

### Step 3: Compute interaction energy with Quantum Force Field

```rust
use nqpu_metal::quantum_drug_design::QuantumForceField;

let qff = QuantumForceField {
    num_qubits: 6,
    num_layers: 3,
    max_iterations: 200,
    convergence_threshold: 1e-4,
};

// Compute interaction energy between protein pocket and ligand
let energy = qff.interaction_energy(&protein_pocket, &ligand)
    .expect("Energy calculation failed");

println!("Interaction energy: {:.4} kcal/mol", energy);
```

The Quantum Force Field combines classical terms (Coulomb, Lennard-Jones, hydrogen bonding) with a VQE-based quantum correction that captures electronic correlation effects missed by classical force fields.

### Step 4: Score docking poses

```rust
use nqpu_metal::quantum_drug_design::{
    QuantumDockingScorer, DockingConfig, ScoringFunction,
};

let scorer = QuantumDockingScorer::new(DockingConfig {
    num_qubits: 4,
    scoring_function: ScoringFunction::Hybrid,
    num_conformations: 10,
    optimization_steps: 50,
});

let docking_score = scorer.score(&protein, &ligand)
    .expect("Docking failed");
```

### Step 5: Predict ADMET properties

```rust
use nqpu_metal::quantum_drug_design::{
    AdmetPredictor, AdmetProperty,
};

let predictor = AdmetPredictor::new(4, vec![
    AdmetProperty::Absorption,
    AdmetProperty::Toxicity,
    AdmetProperty::BBBPermeability,
    AdmetProperty::Solubility,
]);

let predictions = predictor.predict(&candidate)
    .expect("ADMET prediction failed");

for result in &predictions {
    println!("{:?}: {:.3}", result.property, result.score);
}
```

### Step 6: Molecular similarity with quantum kernels

Compare molecular fingerprints using quantum kernel methods:

```rust
use nqpu_metal::quantum_drug_design::{
    MolecularFingerprint, QuantumKernel,
};

// Generate fingerprints
let fp_a = MolecularFingerprint::from_molecule(&mol_a, 1024);
let fp_b = MolecularFingerprint::from_molecule(&mol_b, 1024);

// Classical Tanimoto similarity
let tanimoto = fp_a.tanimoto(&fp_b);
println!("Tanimoto similarity: {:.3}", tanimoto);

// Quantum kernel similarity (fidelity-based)
let kernel = QuantumKernel::new(6);
let quantum_sim = kernel.similarity(&mol_a, &mol_b)
    .expect("Kernel computation failed");
println!("Quantum kernel similarity: {:.3}", quantum_sim);
```

## Active Space Selection Strategies

For molecules larger than a few atoms, simulating all orbitals is intractable even on a quantum computer. Active space selection restricts the calculation to the chemically relevant orbitals while freezing core electrons.

### Using the active_space function

```rust
use nqpu_metal::molecular_integrals::{
    build_molecular_hamiltonian, parse_fcidump, active_space,
    jordan_wigner,
};

// Parse full molecular Hamiltonian (e.g., 20 spatial orbitals)
let fcidump = parse_fcidump(&content).unwrap();
let full_ham = build_molecular_hamiltonian(&fcidump);

// Select active space:
// - Freeze core orbitals 0, 1 (deep core, chemically inert)
// - Active orbitals 2, 3, 4, 5 (valence, near Fermi level)
let frozen_core = vec![0, 1];
let active_orbitals = vec![2, 3, 4, 5];

let reduced_ham = active_space(&full_ham, &active_orbitals, &frozen_core);

// Now only 8 spin-orbitals (4 spatial * 2 spins) instead of 40
let qubit_ham = jordan_wigner(&reduced_ham);
println!("Active qubits: {}", qubit_ham.num_qubits);  // 8
```

### Strategy guidelines

| Molecule Size | Recommended Active Space | Qubits | Method |
|--------------|--------------------------|--------|--------|
| H2, HeH+ | Full space (2 orbitals) | 4 | VQE |
| LiH | (2e, 5o) valence | 10 | VQE or ADAPT-VQE |
| H2O | (4e, 4o) valence | 8 | ADAPT-VQE |
| Drug fragment | (6e, 6o) frontier | 12 | ADAPT-VQE + error mitigation |
| Large drug | (10e, 10o) frontier | 20 | CAMPS-DMRG or double factorized |

**Rules of thumb for choosing active orbitals:**
1. Always include the HOMO and LUMO.
2. Include orbitals within 1-2 eV of the Fermi level.
3. For transition metals, include the d-shell orbitals.
4. For bond-breaking studies, include the bonding and anti-bonding pair.
5. Verify by checking that the active space energy is within chemical accuracy (1 kcal/mol) of a larger active space.

## Error Mitigation for Chemistry Calculations

Quantum hardware introduces noise that corrupts energy estimates. nQPU provides three error mitigation techniques particularly useful for chemistry:

### Readout error mitigation

Corrects measurement bit-flip errors using a per-qubit confusion matrix:

```rust
use nqpu_metal::error_mitigation::ReadoutMitigator;

// Create mitigator with known symmetric error rate
let mitigator = ReadoutMitigator::from_symmetric_error(
    4,     // num_qubits
    0.02,  // 2% readout error per qubit
);

// Correct raw measurement probabilities
let raw_probs = vec![0.45, 0.05, 0.05, 0.45]; // noisy
let corrected = mitigator.mitigate_probs(&raw_probs);
```

For production use, calibrate the confusion matrix from hardware:

```rust
let mut mitigator = ReadoutMitigator::new(4);
// Set calibrated values: mitigator.matrix[qubit] = [[p(0|0), p(1|0)], [p(0|1), p(1|1)]]
mitigator.matrix[0] = [[0.98, 0.02], [0.03, 0.97]];
// ... calibrate remaining qubits ...
```

### Zero-noise extrapolation (ZNE)

Amplifies noise systematically and extrapolates to the zero-noise limit:

```rust
use nqpu_metal::error_mitigation::{fold_gates_local, fold_gates_global};

// Original circuit gates
let gates: Vec<Gate> = /* your circuit */;

// Run at multiple noise levels
let energies_at_scales: Vec<(f64, f64)> = vec![
    (1.0, run_circuit(&gates)),                          // scale 1x
    (3.0, run_circuit(&fold_gates_local(&gates, 3))),    // scale 3x
    (5.0, run_circuit(&fold_gates_local(&gates, 5))),    // scale 5x
];

// Richardson extrapolation to zero noise
let e_mitigated = richardson_extrapolation(&energies_at_scales);
```

Two folding strategies are available:
- **`fold_gates_global`**: Repeats the entire circuit (simpler, more noise amplification).
- **`fold_gates_local`**: Inserts `g, g_dagger, g` for each gate (finer control, preserves ideal unitary). Uses proper gate inversion: `Rz(theta)` becomes `Rz(-theta)`, `T` becomes `Tdg`, etc.

### Symmetry verification

For molecular Hamiltonians that conserve particle number or spin, post-select on the correct symmetry sector to discard unphysical results. This is especially effective when combined with VQE:

```rust
// After VQE optimization, verify the result respects symmetry
let state = vqe_solver.get_final_state();
let particle_number = measure_particle_number(&state);

if particle_number != expected_electrons {
    println!("Warning: symmetry violation detected, re-running with constrained ansatz");
}
```

## Example: Computing Binding Energy of H2O

This end-to-end example computes the ground state energy of water using the predefined integrals and VQE.

```rust
use nqpu_metal::molecular_integrals::{
    water_molecule, active_space, jordan_wigner,
};
use nqpu_metal::vqe::{VQESolver, Hamiltonian, PauliTerm, PauliOperator};

fn main() {
    // 1. Load the full H2O Hamiltonian (7 spatial orbitals, 10 electrons)
    let full_ham = water_molecule();
    println!(
        "Full H2O: {} spatial orbitals, {} electrons, {} spin-orbitals",
        full_ham.num_orbitals, full_ham.num_electrons, full_ham.spin_orbitals
    );

    // 2. Reduce to active space (4 active orbitals, freeze 3 core)
    let frozen = vec![0, 1, 2];        // O 1s, 2s, 2p_z core
    let active = vec![3, 4, 5, 6];     // Valence orbitals
    let active_ham = active_space(&full_ham, &active, &frozen);
    println!(
        "Active space: {} spin-orbitals ({} qubits needed)",
        active_ham.spin_orbitals, active_ham.spin_orbitals
    );

    // 3. Map to qubit Hamiltonian via Jordan-Wigner
    let qubit_ham = jordan_wigner(&active_ham);
    println!(
        "Qubit Hamiltonian: {} terms, {} qubits, constant = {:.6}",
        qubit_ham.terms.len(), qubit_ham.num_qubits, qubit_ham.constant
    );

    // 4. Convert to VQE-compatible Hamiltonian format
    let vqe_hamiltonian = convert_to_vqe_hamiltonian(&qubit_ham);

    // 5. Run VQE
    let num_qubits = qubit_ham.num_qubits;
    let mut solver = VQESolver::new(num_qubits, 4, vqe_hamiltonian, 0.05);
    solver.max_iterations = 1000;
    solver.convergence_threshold = 1e-6;

    let result = solver.find_ground_state();

    // 6. Total energy = VQE energy + nuclear repulsion
    let total_energy = result.ground_state_energy + qubit_ham.constant;
    println!("Ground state energy: {:.6} Ha", total_energy);
    println!("Converged: {} in {} iterations", result.converged, result.iterations);

    // Reference: FCI/STO-3G for H2O is approximately -75.01 Ha
}
```

## Performance Considerations

### CAMPS-DMRG for large molecules

For molecules exceeding 12-14 qubits, the statevector simulator becomes memory-limited (2^N amplitudes). The CAMPS-DMRG solver uses tensor network (MPS) representation to handle large active spaces efficiently:

```rust
use nqpu_metal::camps_dmrg::{CAMPSDMRG, ChemistryConfig, Molecule};

let config = ChemistryConfig::new()
    .with_active_space(10, 10)   // 10 electrons, 10 orbitals
    .with_bond_dimension(256);   // Controls accuracy vs. memory tradeoff

let mut solver = CAMPSDMRG::new(config);
let result = solver.solve(&molecule);
println!("Energy: {:.6} Ha", result.energy);
```

Scaling reference from the CAMPS-DMRG module:

| System | Orbitals | Bond Dim | Time | Memory |
|--------|----------|----------|------|--------|
| H2 | 2 | 16 | 1ms | 1MB |
| LiH | 6 | 64 | 100ms | 10MB |
| H2O | 12 | 256 | 10s | 500MB |
| N2 | 20 | 512 | 5min | 4GB |

### Double factorized representation

For fault-tolerant resource estimation, the double factorized representation decomposes two-electron integrals into a sum of squared one-body operators, dramatically reducing T-gate count:

```rust
use nqpu_metal::double_factorized::{
    DoubleFactorizedHamiltonian, DFConfig,
};

let config = DFConfig::default();
let df_ham = DoubleFactorizedHamiltonian::from_integrals(
    &mol_ham.one_body,
    &mol_ham.two_body,
    mol_ham.nuclear_repulsion,
    &config,
).expect("Factorization failed");

let estimate = df_ham.estimate_resources(1e-3);
println!("Estimated T-gates: {}", estimate.num_t_gates);
```

### GPU acceleration

Enable the `metal` feature for Apple Silicon GPU acceleration, or `cuda` for NVIDIA GPUs. The auto-backend selector chooses the optimal backend based on circuit analysis:

```rust
use nqpu_metal::auto_backend::AutoBackend;

let analysis = AutoBackend::default().analyze(&gates);
println!("Recommended backend: {:?}", analysis.recommended_backend);
println!("Reasoning: {}", analysis.reasoning);
```

Backend selection follows these heuristics:
- **MPS**: 30+ qubits with low entanglement (molecular ground states)
- **Metal/CUDA GPU**: Medium circuits (10-29 qubits) with high gate count
- **Fused CPU**: Small-to-medium circuits with multi-threading via Rayon
- **Sequential CPU**: Fewer than ~6 qubits

### Memory planning

| Qubits | Statevector Memory (f64) | Statevector Memory (f32) |
|--------|-------------------------|-------------------------|
| 10 | 16 KB | 8 KB |
| 20 | 16 MB | 8 MB |
| 25 | 512 MB | 256 MB |
| 30 | 16 GB | 8 GB |

For large molecules, use the `QuantumStateF32` type to halve memory at the cost of reduced numerical precision, or switch to MPS simulation where memory scales linearly with qubit count (at fixed bond dimension).

## Module Reference

| Module | Path | Purpose |
|--------|------|---------|
| `quantum_drug_design` | `chemistry/quantum_drug_design.rs` | Drug design pipeline: molecules, docking, ADMET, QED, fingerprints, Pareto optimization |
| `quantum_chemistry` | `chemistry/quantum_chemistry.rs` | Jordan-Wigner mapper, UCCSD ansatz, hardcoded integrals (H2, LiH, HeH+) |
| `molecular_integrals` | `chemistry/molecular_integrals.rs` | FCIDUMP parser, MolecularHamiltonian, JW/BK/Parity mappings, active space selection, predefined molecules |
| `double_factorized` | `chemistry/double_factorized.rs` | Double factorized Hamiltonian for T-gate reduction: Cholesky, SVD, THC methods |
| `camps_dmrg` | `chemistry/camps_dmrg.rs` | CAMPS-DMRG tensor network solver for large active spaces |
| `quantum_materials` | `chemistry/quantum_materials.rs` | Battery screening, superconductor Tc prediction, band structure |
| `vqe` | `algorithms/vqe.rs` | Standard VQE solver with parameter-shift gradients |
| `adapt_vqe` | `algorithms/adapt_vqe.rs` | ADAPT-VQE with GSD and qubit-ADAPT operator pools |
| `error_mitigation` | `noise/error_mitigation.rs` | Readout mitigation, ZNE (global and local gate folding) |

### Key types quick reference

```
MolecularHamiltonian   -- Second-quantized Hamiltonian (one_body, two_body arrays)
QubitHamiltonian       -- Sum of PauliTerm on qubits + constant offset
FcidumpData            -- Raw parsed FCIDUMP (norb, nelec, integrals)
FermionMapping         -- Enum: JordanWigner | BravyiKitaev | Parity
Molecule               -- Atoms + bonds + name + SMILES
DrugProperties         -- MW, LogP, HBD, HBA, rotatable bonds, PSA
DrugLikenessResult     -- Lipinski violations, QED score, synthetic accessibility
QuantumForceField      -- VQE-corrected classical force field
QuantumDockingScorer   -- QUBO-encoded protein-ligand scoring
AdmetPredictor         -- QNN-based ADMET property prediction
MolecularFingerprint   -- Binary fingerprint for quantum kernel methods
QuantumKernel          -- Fidelity-based molecular similarity
BindingAffinityEstimator -- Quantum binding free energy
VQESolver              -- Variational Quantum Eigensolver
AdaptVqe               -- Adaptive VQE with operator pool
OperatorPool           -- GSD or qubit-ADAPT operator collections
ReadoutMitigator       -- Per-qubit confusion matrix correction
```
