# Quantum Domains Guide

This guide walks through each of the 14 domain directories in the nQPU Rust SDK (`sdk/rust/src/`). Every section explains the underlying math and physics, the problems each domain solves, key source files, prerequisites for further reading, and a brief usage sketch. No prior quantum computing background is assumed -- each section starts from first principles.

---

## Table of Contents

1. [core/](#1-core--quantum-state-representation-and-primitives)
2. [tensor\_networks/](#2-tensor_networks--large-scale-quantum-simulation)
3. [error\_correction/](#3-error_correction--protecting-quantum-information)
4. [noise/](#4-noise--realistic-noise-and-error-mitigation)
5. [algorithms/](#5-algorithms--quantum-algorithms-and-subroutines)
6. [quantum\_ml/](#6-quantum_ml--quantum-machine-learning)
7. [chemistry/](#7-chemistry--quantum-chemistry-and-drug-design)
8. [backends/](#8-backends--hardware-acceleration-and-device-targets)
9. [circuits/](#9-circuits--circuit-optimization-and-compilation)
10. [networking/](#10-networking--quantum-networking-and-cryptography)
11. [physics/](#11-physics--advanced-quantum-physics-models)
12. [applications/](#12-applications--industry-applications)
13. [measurement/](#13-measurement--tomography-and-characterization)
14. [infra/](#14-infra--infrastructure-traits-and-bindings)

---

## 1. `core/` -- Quantum State Representation and Primitives

### What the math/physics is about

A quantum computer stores information in **qubits**. Unlike classical bits (0 or 1), a qubit can exist in a superposition -- a weighted combination of 0 and 1 described by a complex vector. When you have multiple qubits, the combined state lives in an exponentially large vector space. A 2-qubit system needs a 4-element vector; a 30-qubit system needs over a billion elements. The `core/` module provides the fundamental data structures and operations for manipulating these states.

Beyond pure-state vectors, real quantum systems interact with their environment, causing **mixed states** described by density matrices. The core module also implements **stabilizer simulation** (an efficient shortcut for circuits built only from Clifford gates), **Pauli algebra** (the mathematical language of single-qubit errors and measurements), **quantum channels** (describing how quantum information transforms, including noisy processes), and **entanglement measures** (quantifying correlations that have no classical analogue). Additional capabilities include mid-circuit measurement, measurement-based quantum computing (MBQC), continuous-variable (CV) quantum computing, and decision diagram representations.

### What problems it solves

- Representing and evolving quantum states (statevector and density matrix simulation).
- Defining and applying quantum gates (Hadamard, CNOT, Toffoli, parametric rotations, and more).
- Efficiently simulating Clifford-only circuits via the stabilizer formalism.
- Computing entanglement entropy and other correlation measures.
- Modeling quantum channels (depolarizing, amplitude damping, general Kraus maps).

### Key files and entry points

| File | Purpose |
|------|---------|
| `gates.rs` | Gate definitions (unitary matrices, parametric gates, controlled variants) |
| `density_matrix.rs` | Density matrix construction, partial trace, purity, von Neumann entropy |
| `stabilizer.rs` | Tableau-based Clifford simulation (Gottesman-Knill) |
| `pauli_algebra.rs` | Pauli string operations, commutation, expectation values |
| `quantum_channel.rs` | Kraus operators, Choi matrices, channel composition |

### Prerequisites and background reading

- Linear algebra fundamentals (vectors, matrices, tensor products).
- Nielsen & Chuang, *Quantum Computation and Quantum Information*, Chapters 1-4.
- For stabilizer formalism: Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits" (2004).
- For quantum channels: Wilde, *Quantum Information Theory*, Chapter 4.

### Typical workflow

```
1. Construct an initial state (e.g., |00...0>).
2. Apply a sequence of gates from gates.rs.
3. Optionally convert to a density matrix for mixed-state analysis.
4. Measure entanglement or channel fidelity.
5. For Clifford circuits, switch to stabilizer.rs for exponential speedup.
```

---

## 2. `tensor_networks/` -- Large-Scale Quantum Simulation

### What the math/physics is about

A naive simulation of *n* qubits requires storing 2^n complex numbers -- making 50+ qubits intractable on any classical computer. **Tensor networks** exploit the observation that many physically relevant quantum states have limited entanglement. By decomposing the giant state vector into a network of smaller tensors connected by shared indices, you can represent and manipulate states of 100+ qubits efficiently, as long as entanglement stays bounded.

The most common decomposition is the **Matrix Product State (MPS)**, a one-dimensional chain of tensors. For two-dimensional systems there are **PEPS** (Projected Entangled Pair States). **MERA** (Multi-scale Entanglement Renormalization Ansatz) captures critical systems with scale-invariant entanglement. **Tree Tensor Networks** provide hierarchical decompositions. The module also includes **DMRG** (Density Matrix Renormalization Group) and **TDVP** (Time-Dependent Variational Principle) algorithms for finding ground states and simulating time evolution, as well as contraction optimizers that find the most efficient order to multiply tensors together.

### What problems it solves

- Simulating quantum circuits with 100+ qubits when entanglement is moderate.
- Ground-state search for many-body Hamiltonians (condensed matter physics, chemistry).
- Real-time evolution of quantum states via TDVP.
- Optimizing the computational cost of tensor contractions.

### Key files and entry points

| File | Purpose |
|------|---------|
| `tensor_network.rs` | MPS construction, canonicalization, gate application, measurement |
| `peps.rs` | Two-dimensional tensor network states |
| `tree_tensor_network.rs` | Hierarchical tree decompositions |
| `contraction_optimizer.rs` | Optimal contraction ordering (cost minimization) |
| `dmrg_tdvp.rs` | DMRG ground-state solver and TDVP time evolution |

### Prerequisites and background reading

- Singular Value Decomposition (SVD) and matrix factorization.
- Orus, "A practical introduction to tensor networks" (2014), arXiv:1306.2164.
- Schollwoeck, "The density-matrix renormalization group in the age of matrix product states" (2011).

### Typical workflow

```
1. Convert a quantum circuit or Hamiltonian into tensor network form.
2. Choose bond dimension (controls accuracy vs. cost tradeoff).
3. For ground states: run DMRG sweeps until energy converges.
4. For dynamics: evolve with TDVP, monitoring bond dimension growth.
5. Measure observables by contracting the tensor network.
```

---

## 3. `error_correction/` -- Protecting Quantum Information

### What the math/physics is about

Qubits are fragile. Environmental noise causes **decoherence**, corrupting stored quantum information. Quantum error correction (QEC) encodes a single **logical qubit** across many **physical qubits** so that errors can be detected and corrected without destroying the quantum state. The key mathematical insight is that quantum errors can be discretized into Pauli errors (bit-flip X, phase-flip Z, or both Y) and corrected by measuring **stabilizer operators** that reveal error syndromes without collapsing the encoded information.

This module implements the leading QEC code families: **surface codes** (the current front-runner for near-term fault-tolerant computing), **color codes** (which allow transversal gates), **quantum LDPC codes** (achieving better encoding rates), **Floquet codes** (dynamically generated via periodic measurements), and **holographic codes** (connecting QEC to quantum gravity via AdS/CFT). It also provides a suite of decoders -- algorithms that infer the most likely error from the syndrome -- including neural decoders, transformer-based decoders, minimum-weight perfect matching (MWPM), belief propagation with ordered statistics decoding (BP-OSD), and sliding-window decoders for real-time operation. **Magic state distillation** enables non-Clifford gates (like T gates) needed for universal fault-tolerant computation.

### What problems it solves

- Encoding quantum information to survive noise.
- Decoding error syndromes to identify and correct errors.
- Estimating logical error rates and code thresholds.
- Distilling high-fidelity magic states for fault-tolerant T gates.

### Key files and entry points

| File | Purpose |
|------|---------|
| `surface_codes.rs` | Rotated and unrotated surface code lattices, syndrome extraction |
| `qldpc.rs` | Quantum low-density parity-check code construction |
| `neural_decoder.rs` | ML-based decoding with neural networks and transformers |
| `magic_state_factory.rs` | Magic state distillation protocols and resource estimation |

### Prerequisites and background reading

- Stabilizer formalism (see `core/stabilizer.rs`).
- Terhal, "Quantum error correction for quantum memories" (2015), Rev. Mod. Phys.
- Fowler et al., "Surface codes: Towards practical large-scale quantum computation" (2012).
- For LDPC codes: Breuckmann & Eberhardt, "Quantum Low-Density Parity-Check Codes" (2021).

### Typical workflow

```
1. Select a code family (e.g., surface code with distance d=5).
2. Simulate noisy syndrome extraction rounds.
3. Feed syndromes to a decoder (MWPM for speed, neural for accuracy).
4. Estimate logical error rate across many trials.
5. For fault-tolerant circuits: integrate magic state distillation.
```

---

## 4. `noise/` -- Realistic Noise and Error Mitigation

### What the math/physics is about

Real quantum hardware is noisy. Gates are imperfect, qubits lose energy (amplitude damping), phases randomize (dephasing), and crosstalk between qubits introduces correlated errors. Accurate simulation requires modeling these noise processes. Mathematically, noise is described by **quantum channels** -- completely positive, trace-preserving maps. The **Lindblad master equation** governs continuous-time open quantum system dynamics under Markovian (memoryless) noise. For systems with memory effects, **non-Markovian dynamics** are needed.

Since near-term quantum computers cannot implement full error correction, **error mitigation** techniques extract better answers from noisy results without the overhead of encoding. **Zero-noise extrapolation (ZNE)** runs circuits at multiple noise levels and extrapolates to zero noise. **Probabilistic error cancellation (PEC)** inserts random correction operations. **Pauli twirling** converts general noise into simpler Pauli noise. **Probabilistic noise amplification (PNA)** provides another angle on noise scaling.

### What problems it solves

- Simulating realistic noise to predict how algorithms perform on actual hardware.
- Modeling device-specific noise profiles (IBM, Google, ion trap, etc.).
- Mitigating errors in near-term (NISQ) quantum computations without QEC.
- Solving open quantum system dynamics (Lindblad equation, non-Markovian processes).

### Key files and entry points

| File | Purpose |
|------|---------|
| `noise.rs` | Noise model definitions (depolarizing, amplitude damping, device profiles) |
| `error_mitigation.rs` | ZNE, PEC, PNA, Pauli twirling implementations |
| `lindblad.rs` | Lindblad master equation solver for open quantum systems |
| `enhanced_zne.rs` | Advanced ZNE with polynomial/exponential extrapolation |

### Prerequisites and background reading

- Quantum channels (see `core/quantum_channel.rs`).
- Preskill, "Quantum computing in the NISQ era and beyond" (2018).
- Temme et al., "Error mitigation for short-depth quantum circuits" (2017).
- Breuer & Petruccione, *The Theory of Open Quantum Systems* (for Lindblad equation).

### Typical workflow

```
1. Define a noise model (built-in or device-calibrated).
2. Attach the noise model to a circuit simulation.
3. Run noisy simulation to obtain raw expectation values.
4. Apply error mitigation (e.g., ZNE with 3 noise scale factors).
5. Compare mitigated result to ideal value.
```

---

## 5. `algorithms/` -- Quantum Algorithms and Subroutines

### What the math/physics is about

Quantum algorithms exploit interference and entanglement to solve certain problems faster than any known classical algorithm. The **Variational Quantum Eigensolver (VQE)** finds the lowest energy of a quantum system by optimizing a parameterized circuit -- critical for chemistry and materials science. **QAOA** (Quantum Approximate Optimization Algorithm) tackles combinatorial optimization. **Quantum Phase Estimation (QPE)** extracts eigenvalues with exponential precision. **Shor's algorithm** factors large integers in polynomial time, threatening RSA cryptography.

More recent developments include **Quantum Signal Processing (QSP)** and **Quantum Singular Value Transformation (QSVT)**, which provide a unifying mathematical framework: nearly every known quantum algorithm can be understood as a special case of QSVT. The module also includes **quantum annealing**, **Hamiltonian simulation** methods (Trotter-Suzuki decomposition, qDRIFT randomized compilation), and **Pauli propagation** for efficient classical simulation of certain quantum circuits.

### What problems it solves

- Ground-state energy estimation for molecules and materials (VQE).
- Combinatorial optimization (QAOA for MaxCut, scheduling, etc.).
- Integer factoring and discrete logarithm (Shor's algorithm).
- General Hamiltonian simulation (quantum dynamics, materials science).
- Unified algorithm design via QSP/QSVT.

### Key files and entry points

| File | Purpose |
|------|---------|
| `vqe.rs` | VQE with multiple optimizers and ansatz choices |
| `qsp_qsvt.rs` | QSP/QSVT framework for polynomial transformations of block-encoded matrices |
| `shor.rs` | Shor's factoring algorithm with modular exponentiation |
| `pauli_propagation.rs` | Efficient Pauli-frame tracking for Clifford+few-T circuits |

### Prerequisites and background reading

- Basic quantum circuit model (see `core/`).
- Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor" (2014) for VQE.
- Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014) for QAOA.
- Martyn et al., "Grand Unification of Quantum Algorithms" (2021) for QSP/QSVT.

### Typical workflow

```
1. Define the problem (Hamiltonian, cost function, or integer to factor).
2. Select algorithm (VQE for chemistry, QAOA for optimization, QPE for precision).
3. Build the corresponding circuit.
4. Run on simulator or hardware backend.
5. Post-process results (classical optimization loop for variational methods).
```

---

## 6. `quantum_ml/` -- Quantum Machine Learning

### What the math/physics is about

Quantum machine learning investigates whether quantum computers can learn patterns from data faster or more expressively than classical methods. **Quantum kernel methods** embed classical data into a high-dimensional quantum Hilbert space and compute kernel functions (inner products) that may be intractable classically. **Variational quantum neural networks** use parameterized quantum circuits as trainable models. **Quantum natural gradient** adapts gradient descent to the curved geometry of quantum state space, analogous to natural gradient methods in classical ML.

**Neural quantum states (NQS)** flip the relationship: instead of using quantum hardware for ML, they use classical neural networks to represent quantum states, enabling variational Monte Carlo simulations of many-body physics. The module also includes **quantum transformers** (attention mechanisms implemented in quantum circuits), **reservoir computing** (using the natural dynamics of a quantum system as a computational resource), and **barren plateau analysis** (diagnosing when variational circuits have vanishing gradients that make training impossible).

### What problems it solves

- Classification and regression with quantum-enhanced feature spaces.
- Training variational quantum circuits with efficient gradient methods.
- Representing complex many-body wavefunctions with neural networks.
- Diagnosing trainability issues (barren plateaus) before committing hardware resources.

### Key files and entry points

| File | Purpose |
|------|---------|
| `quantum_ml.rs` | Top-level ML module, model construction, training loops |
| `quantum_kernels.rs` | Quantum kernel estimation, kernel alignment, SVM integration |
| `full_quantum_transformer.rs` | Quantum self-attention and transformer architectures |
| `neural_quantum_states.rs` | NQS ansatze (RBM, CNN, Transformer) with VMC optimization |

### Prerequisites and background reading

- Classical machine learning basics (kernels, gradient descent, neural networks).
- Schuld & Petruccione, *Machine Learning with Quantum Computers* (2021).
- McClean et al., "Barren plateaus in quantum neural network training landscapes" (2018).
- Carleo & Troyer, "Solving the quantum many-body problem with artificial neural networks" (2017) for NQS.

### Typical workflow

```
1. Encode classical data into quantum states (amplitude or angle encoding).
2. Choose model type (kernel method, variational circuit, or NQS).
3. Train using quantum natural gradient or parameter-shift rules.
4. Evaluate on test set; compare against classical baselines.
5. Run barren plateau diagnostics if training stalls.
```

---

## 7. `chemistry/` -- Quantum Chemistry and Drug Design

### What the math/physics is about

The electronic structure of molecules is fundamentally quantum mechanical. The energy of a molecule is determined by the **molecular Hamiltonian**, which describes electrons interacting with nuclei and each other. Solving this Hamiltonian exactly is exponentially hard on classical computers -- but a quantum computer can represent the electronic wavefunction naturally. The first step is mapping fermionic (electron) operators to qubit operators using transformations like **Jordan-Wigner** or **Bravyi-Kitaev**. Then variational ansatze like **UCCSD** (Unitary Coupled Cluster Singles and Doubles) prepare approximate ground states on a quantum circuit.

**Double factorization** compresses the molecular Hamiltonian to reduce the number of quantum gates needed. The module extends beyond basic chemistry into applied **drug design workflows**: generating molecular candidates, predicting ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties, computing quantum molecular fingerprints, and running **materials science** simulations for band structures and phonon spectra.

### What problems it solves

- Computing molecular ground-state energies and potential energy surfaces.
- Predicting chemical reaction pathways and transition states.
- Drug candidate generation, scoring, and ADMET profiling.
- Materials property prediction (band gaps, magnetic ordering).

### Key files and entry points

| File | Purpose |
|------|---------|
| `quantum_chemistry.rs` | Hamiltonian construction, fermion-qubit mappings, VQE integration |
| `molecular_integrals.rs` | One- and two-electron integral computation, basis set handling |
| `quantum_drug_design.rs` | Drug design pipeline: generation, scoring, ADMET, virtual screening |

### Prerequisites and background reading

- Introductory quantum chemistry (molecular orbitals, Hartree-Fock).
- McArdle et al., "Quantum computational chemistry" (2020), Rev. Mod. Phys.
- Cao et al., "Quantum Chemistry in the Age of Quantum Computing" (2019).
- For drug design: see also `docs/DRUG_DISCOVERY.md`.

### Typical workflow

```
1. Specify molecule geometry and basis set.
2. Compute molecular integrals (molecular_integrals.rs).
3. Map to qubit Hamiltonian (Jordan-Wigner or Bravyi-Kitaev).
4. Run VQE with UCCSD ansatz to find ground-state energy.
5. For drug design: feed candidates into the ADMET pipeline.
```

---

## 8. `backends/` -- Hardware Acceleration and Device Targets

### What the math/physics is about

Quantum simulation at its core is dense linear algebra: multiplying large complex matrices and vectors. The computational bottleneck is the exponential size of the state vector. **GPU acceleration** maps these matrix operations onto massively parallel hardware. On Apple Silicon, the **Metal** compute API provides direct access to the GPU with unified memory architecture (UMA), avoiding costly CPU-GPU data transfers. For NVIDIA and AMD hardware, **CUDA** and **ROCm** backends are available.

Beyond simulation backends, this module handles **pulse-level control** -- translating abstract quantum gates into the microwave or laser pulses that physically drive qubits on real hardware. It also provides interfaces to cloud quantum hardware providers (IBM Quantum, Google Quantum AI) and an **auto-backend** selector that picks the best available backend based on the system's hardware.

### What problems it solves

- Accelerating quantum simulation by 10-100x using GPU parallelism.
- Abstracting over hardware differences (Metal, CUDA, ROCm, CPU fallback).
- Translating circuits to pulse-level instructions for real quantum hardware.
- Automatically selecting the best available compute backend.

### Key files and entry points

| File | Purpose |
|------|---------|
| `metal_backend.rs` | Apple Metal GPU backend with UMA dispatch |
| `auto_backend.rs` | Automatic backend detection and selection |
| `pulse_control.rs` | Pulse-level gate decomposition and scheduling |

### Prerequisites and background reading

- Basic GPU programming concepts (threads, workgroups, shared memory).
- Apple Metal documentation (developer.apple.com/metal).
- For pulse control: Krantz et al., "A quantum engineer's guide to superconducting qubits" (2019).
- See also `docs/GPU_ACCELERATION.md` for setup instructions.

### Typical workflow

```
1. auto_backend.rs detects available hardware (Metal GPU, CUDA, CPU).
2. Select backend explicitly or let auto-selection choose.
3. Submit circuit for simulation; backend handles parallelization.
4. For real hardware: transpile circuit, convert to pulses, submit to provider.
5. Collect and post-process results.
```

---

## 9. `circuits/` -- Circuit Optimization and Compilation

### What the math/physics is about

A quantum algorithm is expressed as a **quantum circuit**: a sequence of gates applied to qubits over time. Before execution, circuits must be **compiled** to the target hardware's native gate set and connectivity constraints (not every qubit can interact with every other qubit). **Transpilation** maps abstract gates to hardware-native gates and inserts SWAP operations to route qubits. **Circuit optimization** reduces gate count and depth through algebraic simplifications, **gate fusion** (combining adjacent gates into single operations), and **ZX-calculus** (a graphical language for reasoning about quantum circuits where simplification rules can discover optimizations invisible to other methods).

The module supports parsing circuits from standard formats -- **OpenQASM 2.0/3.0** (the lingua franca of quantum circuits) and **QIR** (Quantum Intermediate Representation from Microsoft). A circuit **DSL** allows programmatic construction. Visualization outputs circuits as ASCII art or SVG images.

### What problems it solves

- Reducing gate count and circuit depth for faster, less noisy execution.
- Mapping circuits onto hardware connectivity constraints.
- Importing/exporting circuits in standard formats (QASM, QIR).
- Visualizing circuits for debugging and presentation.

### Key files and entry points

| File | Purpose |
|------|---------|
| `circuit_optimizer.rs` | Peephole optimization, gate cancellation, commutation rules |
| `transpiler.rs` | Qubit routing, SWAP insertion, native gate decomposition |
| `qasm3.rs` | OpenQASM 2.0/3.0 parser and serializer |
| `ascii_viz.rs` | ASCII circuit diagram rendering |

### Prerequisites and background reading

- Quantum circuit model basics (gates, wires, measurement).
- Amy et al., "A meet-in-the-middle algorithm for fast synthesis of depth-optimal quantum circuits" (2013).
- van de Wetering, "ZX-calculus for the working quantum computer scientist" (2020).
- OpenQASM 3.0 specification: arXiv:2104.14722.

### Typical workflow

```
1. Build circuit programmatically or parse from QASM file.
2. Run optimization passes (gate cancellation, fusion, ZX simplification).
3. Transpile to target hardware's native gate set and qubit topology.
4. Visualize the optimized circuit (ascii_viz.rs or SVG export).
5. Hand off to a backend for execution.
```

---

## 10. `networking/` -- Quantum Networking and Cryptography

### What the math/physics is about

Quantum mechanics enables communication protocols with security guarantees rooted in the laws of physics rather than computational assumptions. **Quantum Key Distribution (QKD)** allows two parties to generate a shared secret key where any eavesdropping attempt is detectable -- because measuring a quantum state disturbs it. Protocols like BB84, E91, and continuous-variable QKD achieve information-theoretic security.

**Quantum Random Number Generation (QRNG)** exploits the fundamental randomness of quantum measurement to produce true random numbers, unlike classical pseudo-random generators. The module includes **entropy extraction** algorithms that distill raw quantum measurements into uniformly distributed bits, and a full suite of **NIST randomness tests** to validate output quality. On the defensive side, **post-quantum cryptography (PQC) assessment** tools evaluate classical cryptographic systems against quantum attacks (Shor's algorithm for RSA/ECC, Grover's for symmetric ciphers).

### What problems it solves

- Generating provably secure encryption keys via QKD.
- Producing certifiably random numbers from quantum measurements.
- Validating randomness quality against NIST SP 800-22 test suite.
- Assessing vulnerability of existing cryptographic infrastructure to quantum attacks.

### Key files and entry points

| File | Purpose |
|------|---------|
| `qkd_protocols.rs` | BB84, E91, and CV-QKD protocol simulation |
| `quantum_randomness.rs` | QRNG sources, entropy extraction, randomness certification |
| `nist_tests.rs` | Full NIST SP 800-22 statistical test suite |

### Prerequisites and background reading

- Basic cryptography (symmetric keys, public-key encryption).
- Bennett & Brassard, "Quantum cryptography" (1984) for BB84.
- NIST SP 800-22: "A Statistical Test Suite for Random and Pseudorandom Number Generators."
- For PQC: NIST Post-Quantum Cryptography standardization project.

### Typical workflow

```
1. Simulate a QKD protocol between Alice and Bob over a noisy channel.
2. Perform sifting, error estimation, and privacy amplification.
3. Estimate secure key rate as a function of channel noise.
4. For QRNG: generate raw random bits, apply entropy extraction.
5. Validate output with NIST statistical tests.
```

---

## 11. `physics/` -- Advanced Quantum Physics Models

### What the math/physics is about

This module explores quantum physics beyond the standard circuit model. **Quantum walks** are the quantum analogue of classical random walks -- a particle evolves in superposition across a graph, enabling search algorithms and transport simulations. **Quantum thermodynamics** studies heat, work, and entropy at the quantum scale, where concepts like quantum heat engines and Landauer's principle (the energy cost of erasing information) apply.

**Topological quantum computing** encodes information in **anyons** -- exotic quasiparticles that exist only in two-dimensional systems. Braiding anyons performs quantum gates that are inherently fault-tolerant because the information is stored in global topological properties, not local states. **Majorana fermions** are a promising physical realization. The module also includes **quantum biology** models (energy transfer in photosynthesis, enzyme tunneling), **quantum chaos** diagnostics (level spacing statistics, out-of-time-order correlators), **many-worlds simulation** (Everettian branching), and **quantum consciousness models** (Integrated Information Theory and Orchestrated Objective Reduction).

### What problems it solves

- Simulating quantum walk dynamics for search and transport problems.
- Modeling quantum heat engines and thermodynamic protocols.
- Simulating topological quantum computing with anyon braiding.
- Exploring quantum effects in biological systems.
- Investigating foundational questions in quantum mechanics.

### Key files and entry points

| File | Purpose |
|------|---------|
| `quantum_walk.rs` | Discrete and continuous-time quantum walks on arbitrary graphs |
| `topological_quantum.rs` | Anyon models, braiding gates, Majorana-based computation |
| `quantum_thermodynamics.rs` | Quantum heat engines, work extraction, entropy production |

### Prerequisites and background reading

- Quantum mechanics fundamentals (superposition, measurement, entanglement).
- Kempe, "Quantum random walks: an introductory overview" (2003).
- Nayak et al., "Non-Abelian anyons and topological quantum computation" (2008).
- Deffner & Campbell, *Quantum Thermodynamics* (2019).

### Typical workflow

```
1. Choose a physics model (walk on graph, anyon braiding, heat engine).
2. Define system parameters (graph topology, anyon type, temperature).
3. Run simulation (time evolution, braiding sequence, thermodynamic cycle).
4. Extract observables (probability distributions, gate fidelity, work/heat).
5. Visualize and analyze results.
```

---

## 12. `applications/` -- Industry Applications

### What the math/physics is about

Quantum computing promises practical speedups for specific industry problems. **Quantum finance** leverages quantum amplitude estimation and QAOA to solve portfolio optimization, option pricing, and risk analysis problems that are computationally expensive classically. The mathematical core is mapping financial optimization problems (e.g., Markowitz mean-variance optimization) onto quantum Hamiltonians whose ground states encode optimal portfolios.

**Quantum logistics** tackles vehicle routing (CVRP), traveling salesman (TSP), and supply chain optimization by encoding constraints as penalty terms in a QAOA or quantum annealing formulation. **Quantum climate modeling** uses quantum simulation to model complex atmospheric and oceanic systems. More exploratory applications include **quantum games** (chess and poker with superposition moves), **quantum NLP** (compositional distributional semantics on quantum circuits), and **quantum art** (generative algorithms producing quantum-inspired visual patterns).

### What problems it solves

- Portfolio optimization with quantum speedup (finance).
- Vehicle routing and logistics scheduling (operations research).
- Climate system modeling with quantum simulation (earth science).
- Experimental applications in games, language processing, and generative art.

### Key files and entry points

| File | Purpose |
|------|---------|
| `quantum_finance.rs` | Portfolio optimization, option pricing, risk analysis |
| `quantum_game.rs` | Quantum chess, poker, and game-theoretic simulations |

### Prerequisites and background reading

- QAOA (see `algorithms/` section).
- Orus et al., "Quantum computing for finance: overview and prospects" (2019).
- For logistics: Lucas, "Ising formulations of many NP problems" (2014).

### Typical workflow

```
1. Define the optimization problem (portfolio weights, vehicle routes, etc.).
2. Encode as a QUBO (Quadratic Unconstrained Binary Optimization) or Ising model.
3. Solve with QAOA, VQE, or quantum annealing backend.
4. Decode the quantum solution back to the application domain.
5. Benchmark against classical solvers.
```

---

## 13. `measurement/` -- Tomography and Characterization

### What the math/physics is about

To understand what a quantum computer is actually doing, you need to **characterize** its behavior. **Quantum state tomography** reconstructs the full density matrix of a quantum state from repeated measurements in different bases -- but the number of measurements scales exponentially with qubit count. **Classical shadows** provide a revolutionary alternative: by performing random measurements and storing compact classical descriptions, you can efficiently predict many properties of the state (expectation values, entanglement, fidelity) using only polynomially many measurements.

**Quantum Fisher information** quantifies the maximum precision achievable in parameter estimation tasks (quantum metrology). **QCVV** (Quantum Characterization, Verification, and Validation) encompasses a family of protocols -- randomized benchmarking, gate set tomography, cross-entropy benchmarking -- that assess the quality of quantum gates, circuits, and devices. **Layer fidelity** measures how well multi-qubit gate layers perform, and **property testing** efficiently checks whether a quantum state has a specific property without full tomography.

### What problems it solves

- Reconstructing quantum states from measurement data.
- Efficiently predicting state properties with classical shadows.
- Benchmarking quantum gate and circuit quality.
- Estimating precision limits for quantum sensing and metrology.
- Validating that quantum hardware meets performance specifications.

### Key files and entry points

| File | Purpose |
|------|---------|
| `state_tomography.rs` | Full and compressed state tomography, maximum likelihood estimation |
| `classical_shadows.rs` | Randomized measurement protocol, property prediction |
| `qcvv.rs` | Randomized benchmarking, gate set tomography, cross-entropy benchmarking |

### Prerequisites and background reading

- Density matrices and measurement theory (see `core/`).
- Huang et al., "Predicting many properties of a quantum system from very few measurements" (2020) for classical shadows.
- Eisert et al., "Quantum certification and benchmarking" (2020).
- Blume-Kohout, "Optimal, reliable estimation of quantum states" (2010) for MLE tomography.

### Typical workflow

```
1. Choose characterization protocol (tomography, shadows, or benchmarking).
2. Generate measurement circuits (random Pauli bases for shadows, Clifford sequences for RB).
3. Execute circuits on simulator or hardware.
4. Feed measurement outcomes to the reconstruction/analysis algorithm.
5. Extract figures of merit (fidelity, error rate, Fisher information).
```

---

## 14. `infra/` -- Infrastructure, Traits, and Bindings

### What the math/physics is about

This module is less about physics and more about **software engineering infrastructure** that makes everything else work. It defines the core **traits** (Rust interfaces) that all backends, gates, and states implement, ensuring interoperability across the SDK. **SIMD operations** use CPU vector instructions (AVX2, NEON) to accelerate complex number arithmetic -- the inner loop of all quantum simulation. **Autodiff** (automatic differentiation) enables gradient computation through quantum circuits, essential for variational algorithms.

The module provides **FFI bindings** for Python (via PyO3), WebAssembly (WASM), and C, allowing the Rust core to be called from other languages. **Distributed computing** support via MPI enables multi-node simulation for large qubit counts. The **TUI** (Terminal User Interface) provides a real-time dashboard for monitoring simulations. **Benchmarking** utilities measure and compare performance across backends.

### What problems it solves

- Defining consistent interfaces across all SDK components.
- Accelerating inner-loop arithmetic with SIMD vectorization.
- Exposing the Rust core to Python, JavaScript (WASM), and C consumers.
- Distributing simulations across multiple compute nodes.
- Providing real-time monitoring via a terminal dashboard.
- Computing gradients through quantum circuits for optimization.

### Key files and entry points

| File | Purpose |
|------|---------|
| `traits.rs` | Core trait definitions (QuantumBackend, QuantumGate, QuantumState) |
| `simd_ops.rs` | SIMD-accelerated complex number operations |
| `tui.rs` | Terminal UI dashboard for simulation monitoring |
| `autodiff.rs` | Automatic differentiation for variational parameter gradients |

### Prerequisites and background reading

- Rust traits and generics (The Rust Programming Language, Chapter 10).
- SIMD programming concepts (Intel Intrinsics Guide or ARM NEON documentation).
- PyO3 documentation for Python-Rust interop.
- For autodiff: Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (2018).
- See also `docs/TUI.md` for terminal UI usage.

### Typical workflow

```
1. Import core traits to build custom backends or gate sets.
2. Use SIMD operations for performance-critical inner loops.
3. For Python users: import the nqpu Python package (built via PyO3 bindings).
4. For distributed simulation: configure MPI and use distributed backends.
5. Launch TUI to monitor running simulations in real time.
```

---

## Cross-Domain Relationships

The 14 domains are designed to compose naturally. Here is how they connect:

```
                        +----------+
                        |  core/   |  <-- Foundation for everything
                        +----+-----+
                             |
           +---------+-------+-------+---------+
           |         |               |         |
     +-----v---+ +---v-----+  +-----v---+ +---v------+
     |circuits/ | |  noise/ |  |backends/| |  infra/  |
     +---------+ +---------+  +---------+ +----------+
           |         |               |
     +-----v---------v---------------v-----+
     |          algorithms/                |
     +--+--------+--------+--------+------+
        |        |        |        |
   +----v--+ +--v----+ +-v------+ +v-----------+
   |chem/  | |qml/   | |physics/| |applications/|
   +-------+ +-------+ +--------+ +-------------+
        |
   +----v-----------+     +-------------+     +------------+
   |error_correction/|     |measurement/ |     |networking/ |
   +-----------------+     +-------------+     +------------+
```

- **core/** is the foundation: every other module depends on its state representations and gate definitions.
- **circuits/** and **noise/** sit between core and algorithms, providing compilation and realistic noise.
- **backends/** and **infra/** provide the execution layer.
- **algorithms/** implements the quantum algorithms that **chemistry/**, **quantum\_ml/**, **physics/**, and **applications/** build upon.
- **error\_correction/** and **measurement/** focus on quality and reliability.
- **networking/** stands somewhat independently, focused on cryptographic applications.

---

## Getting Started

If you are new to nQPU, a recommended reading order:

1. **core/** -- Understand qubits, gates, and state representations.
2. **circuits/** -- Learn how circuits are built, optimized, and visualized.
3. **backends/** -- Run your first simulation on CPU or GPU.
4. **algorithms/** -- Implement your first quantum algorithm (start with VQE or QAOA).
5. **Pick a domain** -- Dive into chemistry, ML, physics, or applications based on your interests.

For installation and setup, see [GETTING_STARTED.md](GETTING_STARTED.md). For system architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).
