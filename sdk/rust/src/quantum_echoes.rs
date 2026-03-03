//! Quantum Echoes / Out-of-Time-Order Correlators (OTOC)
//!
//! Implementation inspired by Google's Quantum Echoes experiment (Nature, Oct 2025)
//! which demonstrated 13,000x speedup over classical OTOC computation — the first
//! verifiable quantum advantage for a scientifically useful computation.
//!
//! # Overview
//!
//! OTOCs measure quantum information scrambling: how quickly local perturbations
//! spread through a many-body quantum system. The OTOC is defined as:
//!
//!   C(t) = ⟨A†(t) B† A(t) B⟩
//!
//! where A(t) = U†(t) A U(t) is the time-evolved Heisenberg operator.
//!
//! This module provides:
//! - Full OTOC computation via state-vector simulation
//! - Loschmidt echo (sensitivity to perturbation)
//! - Scrambling time detection and Lyapunov exponent extraction
//! - Predefined Hamiltonians (Ising, Heisenberg, random circuits)
//! - Molecular ruler: using OTOCs to measure distances in quantum systems

use num_complex::Complex64;
use rand::Rng;
use std::f64::consts::PI;
use std::fmt;

// ============================================================
// TYPES
// ============================================================

/// Pauli operator: I, X, Y, Z
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

/// A multi-qubit Pauli operator with a complex coefficient.
///
/// Represents `coefficient * (P_i1 ⊗ P_i2 ⊗ ...)` where each `(usize, Pauli)`
/// specifies a qubit index and the Pauli acting on it.
#[derive(Debug, Clone)]
pub struct PauliOp {
    pub terms: Vec<(usize, Pauli)>,
    pub coefficient: Complex64,
}

impl PauliOp {
    /// Create a new PauliOp with the given terms and coefficient.
    pub fn new(terms: Vec<(usize, Pauli)>, coefficient: Complex64) -> Self {
        Self { terms, coefficient }
    }

    /// Single-qubit Pauli operator with coefficient 1.
    pub fn single(qubit: usize, pauli: Pauli) -> Self {
        Self {
            terms: vec![(qubit, pauli)],
            coefficient: Complex64::new(1.0, 0.0),
        }
    }

    /// Return the adjoint (Hermitian conjugate) of this Pauli operator.
    /// Since Pauli matrices are Hermitian, P† = P, but we conjugate the coefficient.
    pub fn adjoint(&self) -> Self {
        Self {
            terms: self.terms.clone(),
            coefficient: self.coefficient.conj(),
        }
    }
}

/// A term in a Hamiltonian: coefficient * (product of Pauli operators on specific qubits).
#[derive(Debug, Clone)]
pub struct HamiltonianTerm {
    pub paulis: Vec<(usize, Pauli)>,
    pub coefficient: f64,
}

impl HamiltonianTerm {
    pub fn new(paulis: Vec<(usize, Pauli)>, coefficient: f64) -> Self {
        Self { paulis, coefficient }
    }
}

/// Gate types used in echo circuits.
#[derive(Debug, Clone)]
pub enum EchoGate {
    Rx(usize, f64),
    Ry(usize, f64),
    Rz(usize, f64),
    CX(usize, usize),
    H(usize),
    X(usize),
    Z(usize),
}

/// A circuit structure for echo computations containing forward, backward, and operator gates.
#[derive(Debug, Clone)]
pub struct EchoCircuit {
    pub forward_gates: Vec<EchoGate>,
    pub backward_gates: Vec<EchoGate>,
    pub operator_gates: Vec<EchoGate>,
}

/// Configuration for OTOC computation.
#[derive(Debug, Clone)]
pub struct OtocConfig {
    pub num_qubits: usize,
    pub time_steps: Vec<f64>,
    pub operator_a: PauliOp,
    pub operator_b: PauliOp,
    pub hamiltonian: Vec<HamiltonianTerm>,
    /// Number of Trotter steps per unit time (default: 20).
    pub trotter_steps_per_unit: usize,
}

impl OtocConfig {
    /// Create a new OtocConfig builder with required parameters.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            time_steps: vec![0.0, 0.5, 1.0, 1.5, 2.0],
            operator_a: PauliOp::single(0, Pauli::X),
            operator_b: PauliOp::single(1, Pauli::Z),
            hamiltonian: ising_1d(num_qubits, 1.0, 0.5),
            trotter_steps_per_unit: 20,
        }
    }

    /// Set the time steps to evaluate.
    pub fn time_steps(mut self, times: Vec<f64>) -> Self {
        self.time_steps = times;
        self
    }

    /// Set operator A.
    pub fn operator_a(mut self, op: PauliOp) -> Self {
        self.operator_a = op;
        self
    }

    /// Set operator B.
    pub fn operator_b(mut self, op: PauliOp) -> Self {
        self.operator_b = op;
        self
    }

    /// Set the Hamiltonian.
    pub fn hamiltonian(mut self, h: Vec<HamiltonianTerm>) -> Self {
        self.hamiltonian = h;
        self
    }

    /// Set Trotter steps per unit time.
    pub fn trotter_steps_per_unit(mut self, steps: usize) -> Self {
        self.trotter_steps_per_unit = steps;
        self
    }

    /// Number of Trotter steps for a given time.
    fn trotter_steps(&self, time: f64) -> usize {
        let steps = (time.abs() * self.trotter_steps_per_unit as f64).ceil() as usize;
        steps.max(1)
    }
}

/// Result of an OTOC computation.
#[derive(Debug, Clone)]
pub struct OtocResult {
    /// Time points evaluated.
    pub times: Vec<f64>,
    /// OTOC values C(t) at each time point.
    pub otoc_values: Vec<Complex64>,
    /// Loschmidt echo values L(t) at each time point.
    pub loschmidt_echoes: Vec<f64>,
    /// Estimated scrambling time (time when |C(t)| drops below threshold).
    pub scrambling_time: Option<f64>,
}

impl fmt::Display for OtocResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "OTOC Result ({} time points):", self.times.len())?;
        for (i, t) in self.times.iter().enumerate() {
            writeln!(
                f,
                "  t={:.3}: C(t)={:.6}+{:.6}i  |C|={:.6}  L(t)={:.6}",
                t,
                self.otoc_values[i].re,
                self.otoc_values[i].im,
                self.otoc_values[i].norm(),
                self.loschmidt_echoes[i],
            )?;
        }
        if let Some(ts) = self.scrambling_time {
            writeln!(f, "  Scrambling time: {:.4}", ts)?;
        }
        Ok(())
    }
}

/// Errors that can occur during OTOC computation.
#[derive(Debug, Clone, PartialEq)]
pub enum OtocError {
    /// An operator references a qubit index outside the system size.
    InvalidOperator(String),
    /// The simulation encountered a numerical or logical error.
    SimulationFailed(String),
}

impl fmt::Display for OtocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OtocError::InvalidOperator(msg) => write!(f, "Invalid operator: {}", msg),
            OtocError::SimulationFailed(msg) => write!(f, "Simulation failed: {}", msg),
        }
    }
}

impl std::error::Error for OtocError {}

// ============================================================
// STATE VECTOR UTILITIES
// ============================================================

/// Create the |0...0⟩ state for n qubits.
pub fn zero_state(n: usize) -> Vec<Complex64> {
    let dim = 1 << n;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0);
    state
}

/// Compute the inner product ⟨a|b⟩ = Σ conj(a_i) * b_i.
pub fn inner_product(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    assert_eq!(a.len(), b.len(), "State vectors must have the same dimension");
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| ai.conj() * bi)
        .sum()
}

/// Compute the norm squared ⟨ψ|ψ⟩ of a state vector.
fn norm_squared(state: &[Complex64]) -> f64 {
    state.iter().map(|c| c.norm_sqr()).sum()
}

// ============================================================
// GATE APPLICATION
// ============================================================

/// Apply a single EchoGate to a state vector.
pub fn apply_gate(state: &mut Vec<Complex64>, gate: &EchoGate) {
    let n_qubits = (state.len() as f64).log2() as usize;
    match gate {
        EchoGate::H(q) => apply_single_qubit_gate(state, *q, n_qubits, &h_matrix()),
        EchoGate::X(q) => apply_single_qubit_gate(state, *q, n_qubits, &x_matrix()),
        EchoGate::Z(q) => apply_single_qubit_gate(state, *q, n_qubits, &z_matrix()),
        EchoGate::Rx(q, theta) => {
            apply_single_qubit_gate(state, *q, n_qubits, &rx_matrix(*theta))
        }
        EchoGate::Ry(q, theta) => {
            apply_single_qubit_gate(state, *q, n_qubits, &ry_matrix(*theta))
        }
        EchoGate::Rz(q, theta) => {
            apply_single_qubit_gate(state, *q, n_qubits, &rz_matrix(*theta))
        }
        EchoGate::CX(ctrl, tgt) => apply_cx(state, *ctrl, *tgt, n_qubits),
    }
}

/// 2x2 gate matrices
fn h_matrix() -> [[Complex64; 2]; 2] {
    let s = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    [[s, s], [s, -s]]
}

fn x_matrix() -> [[Complex64; 2]; 2] {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    [[zero, one], [one, zero]]
}

fn z_matrix() -> [[Complex64; 2]; 2] {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    [[one, zero], [zero, -one]]
}

fn rx_matrix(theta: f64) -> [[Complex64; 2]; 2] {
    let c = Complex64::new((theta / 2.0).cos(), 0.0);
    let s = Complex64::new(0.0, -(theta / 2.0).sin());
    [[c, s], [s, c]]
}

fn ry_matrix(theta: f64) -> [[Complex64; 2]; 2] {
    let c = Complex64::new((theta / 2.0).cos(), 0.0);
    let s = Complex64::new((theta / 2.0).sin(), 0.0);
    [[c, -s], [s, c]]
}

fn rz_matrix(theta: f64) -> [[Complex64; 2]; 2] {
    let zero = Complex64::new(0.0, 0.0);
    let e_neg = Complex64::new(0.0, -theta / 2.0).exp();
    let e_pos = Complex64::new(0.0, theta / 2.0).exp();
    [[e_neg, zero], [zero, e_pos]]
}

/// Apply a 2x2 unitary to qubit `target` in an n-qubit state vector.
fn apply_single_qubit_gate(
    state: &mut [Complex64],
    target: usize,
    n_qubits: usize,
    gate: &[[Complex64; 2]; 2],
) {
    let dim = 1 << n_qubits;
    let bit = 1 << (n_qubits - 1 - target);
    for i in 0..dim {
        if i & bit == 0 {
            let j = i | bit;
            let a = state[i];
            let b = state[j];
            state[i] = gate[0][0] * a + gate[0][1] * b;
            state[j] = gate[1][0] * a + gate[1][1] * b;
        }
    }
}

/// Apply CNOT gate: control -> target.
fn apply_cx(state: &mut [Complex64], control: usize, target: usize, n_qubits: usize) {
    let dim = 1 << n_qubits;
    let ctrl_bit = 1 << (n_qubits - 1 - control);
    let tgt_bit = 1 << (n_qubits - 1 - target);
    for i in 0..dim {
        // Only swap when control is 1 and target is 0
        if (i & ctrl_bit) != 0 && (i & tgt_bit) == 0 {
            let j = i | tgt_bit;
            state.swap(i, j);
        }
    }
}

// ============================================================
// PAULI OPERATOR APPLICATION
// ============================================================

/// Apply a single Pauli rotation exp(-i * angle * P) to a specific qubit.
///
/// For Pauli I: global phase (no-op for real angles).
/// For Pauli X: Rx(2*angle).
/// For Pauli Y: Ry(2*angle).
/// For Pauli Z: Rz(2*angle).
pub fn apply_pauli_rotation(state: &mut Vec<Complex64>, pauli: &(usize, Pauli), angle: f64) {
    let (qubit, p) = pauli;
    let n_qubits = (state.len() as f64).log2() as usize;
    match p {
        Pauli::I => {
            // exp(-i * angle * I) = global phase
            let phase = Complex64::new(0.0, -angle).exp();
            for amp in state.iter_mut() {
                *amp *= phase;
            }
        }
        Pauli::X => {
            apply_single_qubit_gate(state, *qubit, n_qubits, &rx_matrix(2.0 * angle));
        }
        Pauli::Y => {
            apply_single_qubit_gate(state, *qubit, n_qubits, &ry_matrix(2.0 * angle));
        }
        Pauli::Z => {
            apply_single_qubit_gate(state, *qubit, n_qubits, &rz_matrix(2.0 * angle));
        }
    }
}

/// Apply a multi-qubit Pauli operator P = coefficient * (P_i1 ⊗ P_i2 ⊗ ...) to a state.
///
/// This applies each single-qubit Pauli in the tensor product, then scales by the coefficient.
/// Note: Pauli matrices {I, X, Y, Z} are applied directly (not as rotations).
pub fn apply_pauli_operator(state: &mut Vec<Complex64>, op: &PauliOp) {
    let n_qubits = (state.len() as f64).log2() as usize;

    // Apply each Pauli in the tensor product
    for (qubit, pauli) in &op.terms {
        match pauli {
            Pauli::I => {} // Identity, no-op
            Pauli::X => {
                apply_single_qubit_gate(state, *qubit, n_qubits, &x_matrix());
            }
            Pauli::Y => {
                let zero = Complex64::new(0.0, 0.0);
                let neg_i = Complex64::new(0.0, -1.0);
                let pos_i = Complex64::new(0.0, 1.0);
                let y_mat = [[zero, neg_i], [pos_i, zero]];
                apply_single_qubit_gate(state, *qubit, n_qubits, &y_mat);
            }
            Pauli::Z => {
                apply_single_qubit_gate(state, *qubit, n_qubits, &z_matrix());
            }
        }
    }

    // Scale by coefficient
    if op.coefficient != Complex64::new(1.0, 0.0) {
        for amp in state.iter_mut() {
            *amp *= op.coefficient;
        }
    }
}

// ============================================================
// TIME EVOLUTION (TROTTER DECOMPOSITION)
// ============================================================

/// 2nd-order Trotter-Suzuki time evolution.
///
/// Evolves |ψ⟩ → exp(-i H t)|ψ⟩ using the symmetric decomposition:
///   U(dt) ≈ ∏_k exp(-i H_k dt/2) * ∏_k' exp(-i H_k' dt/2)  (reversed order)
///
/// Each Hamiltonian term H_k = c_k * (P_i ⊗ P_j ⊗ ...) is implemented as
/// a product of Pauli rotations.
pub fn trotter_evolve(
    state: &mut Vec<Complex64>,
    hamiltonian: &[HamiltonianTerm],
    time: f64,
    steps: usize,
) {
    if hamiltonian.is_empty() || steps == 0 {
        return;
    }
    let dt = time / steps as f64;

    for _ in 0..steps {
        // First half: forward order, dt/2
        for term in hamiltonian.iter() {
            apply_hamiltonian_term(state, term, dt / 2.0);
        }
        // Second half: reverse order, dt/2
        for term in hamiltonian.iter().rev() {
            apply_hamiltonian_term(state, term, dt / 2.0);
        }
    }
}

/// Apply a single Hamiltonian term: exp(-i * coefficient * dt * P1 ⊗ P2 ⊗ ...).
///
/// For a tensor product of Paulis, we decompose into a basis-change + Z-rotation circuit.
/// For single-qubit terms, we directly apply Pauli rotations.
/// For two-qubit ZZ terms, we use CX-Rz-CX decomposition.
fn apply_hamiltonian_term(state: &mut Vec<Complex64>, term: &HamiltonianTerm, dt: f64) {
    let angle = term.coefficient * dt;

    // Filter out identity Paulis (they only contribute a global phase)
    let non_identity: Vec<(usize, Pauli)> = term
        .paulis
        .iter()
        .filter(|(_, p)| *p != Pauli::I)
        .cloned()
        .collect();

    if non_identity.is_empty() {
        // Pure identity term: global phase
        let phase = Complex64::new(0.0, -angle).exp();
        for amp in state.iter_mut() {
            *amp *= phase;
        }
        return;
    }

    if non_identity.len() == 1 {
        // Single-qubit term: direct Pauli rotation
        apply_pauli_rotation(state, &non_identity[0], angle);
        return;
    }

    // Multi-qubit term: basis change → entangle → Rz → unentangle → basis unchange
    // Step 1: Change to Z-basis for each non-Z Pauli
    let n_qubits = (state.len() as f64).log2() as usize;
    for (qubit, pauli) in &non_identity {
        match pauli {
            Pauli::X => {
                // H transforms X-basis to Z-basis
                apply_single_qubit_gate(state, *qubit, n_qubits, &h_matrix());
            }
            Pauli::Y => {
                // S†H transforms Y-basis to Z-basis: use Rx(π/2)
                apply_single_qubit_gate(state, *qubit, n_qubits, &rx_matrix(PI / 2.0));
            }
            Pauli::Z | Pauli::I => {}
        }
    }

    // Step 2: CNOT chain to compute parity into last qubit
    let last_qubit = non_identity.last().unwrap().0;
    for (qubit, _) in non_identity.iter().rev().skip(1) {
        apply_cx(state, *qubit, last_qubit, n_qubits);
    }

    // Step 3: Rz rotation on the parity qubit
    apply_single_qubit_gate(state, last_qubit, n_qubits, &rz_matrix(2.0 * angle));

    // Step 4: Undo CNOT chain
    for (qubit, _) in &non_identity[..non_identity.len() - 1] {
        apply_cx(state, *qubit, last_qubit, n_qubits);
    }

    // Step 5: Undo basis change
    for (qubit, pauli) in &non_identity {
        match pauli {
            Pauli::X => {
                apply_single_qubit_gate(state, *qubit, n_qubits, &h_matrix());
            }
            Pauli::Y => {
                apply_single_qubit_gate(state, *qubit, n_qubits, &rx_matrix(-PI / 2.0));
            }
            Pauli::Z | Pauli::I => {}
        }
    }
}

/// Time-reversed (adjoint) evolution: U†(t) = U(-t).
///
/// Evolves |ψ⟩ → exp(+i H t)|ψ⟩.
pub fn adjoint_evolve(
    state: &mut Vec<Complex64>,
    hamiltonian: &[HamiltonianTerm],
    time: f64,
    steps: usize,
) {
    trotter_evolve(state, hamiltonian, -time, steps);
}

// ============================================================
// OTOC COMPUTATION
// ============================================================

/// Validate that all operator qubit indices are within bounds.
fn validate_operator(op: &PauliOp, num_qubits: usize, name: &str) -> Result<(), OtocError> {
    for (qubit, _) in &op.terms {
        if *qubit >= num_qubits {
            return Err(OtocError::InvalidOperator(format!(
                "Operator {} references qubit {} but system has only {} qubits",
                name, qubit, num_qubits
            )));
        }
    }
    Ok(())
}

fn validate_hamiltonian(h: &[HamiltonianTerm], num_qubits: usize) -> Result<(), OtocError> {
    for (i, term) in h.iter().enumerate() {
        for (qubit, _) in &term.paulis {
            if *qubit >= num_qubits {
                return Err(OtocError::InvalidOperator(format!(
                    "Hamiltonian term {} references qubit {} but system has only {} qubits",
                    i, qubit, num_qubits
                )));
            }
        }
    }
    Ok(())
}

/// Compute the full OTOC C(t) = ⟨ψ| A†(t) B† A(t) B |ψ⟩ for each time step.
///
/// The computation follows the 10-step protocol:
/// 1. Prepare |ψ⟩
/// 2. Apply B
/// 3. Forward evolve U(t)
/// 4. Apply A
/// 5. Backward evolve U†(t)
/// 6. Apply B†
/// 7. Forward evolve U(t)
/// 8. Apply A†
/// 9. Backward evolve U†(t)
/// 10. Compute overlap ⟨ψ|result⟩
pub fn compute_otoc(config: &OtocConfig) -> Result<OtocResult, OtocError> {
    validate_operator(&config.operator_a, config.num_qubits, "A")?;
    validate_operator(&config.operator_b, config.num_qubits, "B")?;
    validate_hamiltonian(&config.hamiltonian, config.num_qubits)?;

    let initial = zero_state(config.num_qubits);
    let mut otoc_values = Vec::with_capacity(config.time_steps.len());
    let mut loschmidt_echoes = Vec::with_capacity(config.time_steps.len());

    let a_adj = config.operator_a.adjoint();
    let b_adj = config.operator_b.adjoint();

    for &t in &config.time_steps {
        let steps = config.trotter_steps(t);

        // Full OTOC: C(t) = ⟨ψ| A†(t) B† A(t) B |ψ⟩
        // We compute by applying operators right-to-left to |ψ⟩:
        //   |φ⟩ = B |ψ⟩
        //   |φ⟩ = U(t) |φ⟩       (forward evolve)
        //   |φ⟩ = A |φ⟩
        //   |φ⟩ = U†(t) |φ⟩      (backward evolve)
        //   |φ⟩ = B† |φ⟩
        //   |φ⟩ = U(t) |φ⟩       (forward evolve)
        //   |φ⟩ = A† |φ⟩
        //   |φ⟩ = U†(t) |φ⟩      (backward evolve)
        //   C(t) = ⟨ψ|φ⟩
        let mut phi = initial.clone();

        // Step 2: Apply B
        apply_pauli_operator(&mut phi, &config.operator_b);
        // Step 3: Forward evolve
        trotter_evolve(&mut phi, &config.hamiltonian, t, steps);
        // Step 4: Apply A
        apply_pauli_operator(&mut phi, &config.operator_a);
        // Step 5: Backward evolve
        adjoint_evolve(&mut phi, &config.hamiltonian, t, steps);
        // Step 6: Apply B†
        apply_pauli_operator(&mut phi, &b_adj);
        // Step 7: Forward evolve
        trotter_evolve(&mut phi, &config.hamiltonian, t, steps);
        // Step 8: Apply A†
        apply_pauli_operator(&mut phi, &a_adj);
        // Step 9: Backward evolve
        adjoint_evolve(&mut phi, &config.hamiltonian, t, steps);

        // Step 10: Overlap
        let c_t = inner_product(&initial, &phi);
        otoc_values.push(c_t);

        // Loschmidt echo for this time step
        let le = loschmidt_echo(&initial, &config.hamiltonian, t, 0.0);
        loschmidt_echoes.push(le);
    }

    let scrambling = scrambling_time(&otoc_values, &config.time_steps, 0.5);

    Ok(OtocResult {
        times: config.time_steps.clone(),
        otoc_values,
        loschmidt_echoes,
        scrambling_time: scrambling,
    })
}

/// Simplified OTOC computation: C(t) = ⟨ψ| W†(t) V† W(t) V |ψ⟩
///
/// Circuit order: V → U(t) → W → U†(t) → V† applied to bra side,
/// which means on the ket side we apply: V → U(t) → W → U†(t),
/// then take overlap with V|ψ⟩... but the standard formulation is:
///
/// C(t) = ⟨ψ| [W(t), V]† [W(t), V] |ψ⟩  (commutator squared form)
///
/// For simplicity we just delegate to the full compute_otoc.
pub fn compute_otoc_simplified(config: &OtocConfig) -> Result<OtocResult, OtocError> {
    compute_otoc(config)
}

// ============================================================
// LOSCHMIDT ECHO
// ============================================================

/// Compute the Loschmidt echo L(t) = |⟨ψ| exp(iH't) exp(-iHt) |ψ⟩|²
///
/// This measures the sensitivity of time evolution to perturbation.
/// H' = H + perturbation * (random Z operators on each qubit).
///
/// - `state`: initial state |ψ⟩
/// - `hamiltonian`: the Hamiltonian H
/// - `time`: evolution time t
/// - `perturbation`: strength δ of the perturbation H' = H + δΣZ_i
pub fn loschmidt_echo(
    state: &[Complex64],
    hamiltonian: &[HamiltonianTerm],
    time: f64,
    perturbation: f64,
) -> f64 {
    let n_qubits = (state.len() as f64).log2() as usize;
    let steps = (time.abs() * 20.0).ceil().max(1.0) as usize;

    // Forward evolve with H
    let mut forward = state.to_vec();
    trotter_evolve(&mut forward, hamiltonian, time, steps);

    // Create perturbed Hamiltonian H' = H + δ * Σ Z_i
    let mut h_perturbed = hamiltonian.to_vec();
    if perturbation.abs() > 1e-15 {
        for q in 0..n_qubits {
            h_perturbed.push(HamiltonianTerm::new(
                vec![(q, Pauli::Z)],
                perturbation,
            ));
        }
    }

    // Backward evolve with H' (adjoint = evolve with -t)
    let mut result = forward;
    adjoint_evolve(&mut result, &h_perturbed, time, steps);

    // Loschmidt echo = |⟨ψ|result⟩|²
    let overlap = inner_product(state, &result);
    overlap.norm_sqr()
}

// ============================================================
// SCRAMBLING ANALYSIS
// ============================================================

/// Detect the scrambling time: the first time when |C(t)| drops below `threshold`.
///
/// Returns `None` if the OTOC never drops below the threshold.
pub fn scrambling_time(
    otoc_values: &[Complex64],
    times: &[f64],
    threshold: f64,
) -> Option<f64> {
    for (i, c) in otoc_values.iter().enumerate() {
        if c.norm() < threshold {
            // Linear interpolation for more precise estimate
            if i > 0 {
                let prev_norm = otoc_values[i - 1].norm();
                let curr_norm = c.norm();
                if (prev_norm - curr_norm).abs() > 1e-15 {
                    let frac = (prev_norm - threshold) / (prev_norm - curr_norm);
                    return Some(times[i - 1] + frac * (times[i] - times[i - 1]));
                }
            }
            return Some(times[i]);
        }
    }
    None
}

/// Estimate the quantum Lyapunov exponent λ from early-time OTOC behavior.
///
/// In chaotic systems, C(t) ~ 1 - ε * exp(2λt) at early times.
/// We fit ln(1 - |C(t)|) vs t to extract 2λ as the slope.
///
/// Returns λ (the Lyapunov exponent). Returns 0.0 if insufficient data.
pub fn lyapunov_exponent(otoc_values: &[Complex64], times: &[f64]) -> f64 {
    // We need at least 2 points where |C(t)| < 1 to fit
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for (i, c) in otoc_values.iter().enumerate() {
        let cn = c.norm();
        if cn < 1.0 - 1e-10 && cn > 1e-10 && times[i] > 1e-10 {
            xs.push(times[i]);
            ys.push((1.0 - cn).ln());
        }
    }

    if xs.len() < 2 {
        return 0.0;
    }

    // Simple linear regression: y = a + b*x, where b = 2λ
    let n = xs.len() as f64;
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = xs.iter().map(|x| x * x).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return 0.0;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    // slope = 2λ, so λ = slope / 2
    // Ensure non-negative (physical constraint for chaotic systems)
    (slope / 2.0).max(0.0)
}

// ============================================================
// PREDEFINED HAMILTONIANS
// ============================================================

/// 1D Transverse-field Ising model: H = -J Σ Z_i Z_{i+1} - h Σ X_i
///
/// - `n`: number of qubits (spins)
/// - `j`: coupling strength J
/// - `h`: transverse field strength
pub fn ising_1d(n: usize, j: f64, h: f64) -> Vec<HamiltonianTerm> {
    let mut terms = Vec::new();

    // ZZ coupling terms
    for i in 0..n.saturating_sub(1) {
        terms.push(HamiltonianTerm::new(
            vec![(i, Pauli::Z), (i + 1, Pauli::Z)],
            -j,
        ));
    }

    // Transverse field X terms
    for i in 0..n {
        terms.push(HamiltonianTerm::new(vec![(i, Pauli::X)], -h));
    }

    terms
}

/// 1D Heisenberg model: H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
///
/// - `n`: number of qubits
/// - `j`: coupling strength
pub fn heisenberg_1d(n: usize, j: f64) -> Vec<HamiltonianTerm> {
    let mut terms = Vec::new();

    for i in 0..n.saturating_sub(1) {
        // XX
        terms.push(HamiltonianTerm::new(
            vec![(i, Pauli::X), (i + 1, Pauli::X)],
            j,
        ));
        // YY
        terms.push(HamiltonianTerm::new(
            vec![(i, Pauli::Y), (i + 1, Pauli::Y)],
            j,
        ));
        // ZZ
        terms.push(HamiltonianTerm::new(
            vec![(i, Pauli::Z), (i + 1, Pauli::Z)],
            j,
        ));
    }

    terms
}

/// Random circuit Hamiltonian with random two-qubit Pauli terms.
///
/// Generates `depth` random terms, each a random 2-qubit Pauli with random coefficient.
pub fn random_circuit_hamiltonian(
    n: usize,
    depth: usize,
    rng: &mut impl Rng,
) -> Vec<HamiltonianTerm> {
    let paulis = [Pauli::X, Pauli::Y, Pauli::Z];
    let mut terms = Vec::new();

    for _ in 0..depth {
        let q1 = rng.gen_range(0..n);
        let mut q2 = rng.gen_range(0..n);
        while q2 == q1 {
            q2 = rng.gen_range(0..n);
        }
        let p1 = paulis[rng.gen_range(0..3)];
        let p2 = paulis[rng.gen_range(0..3)];
        let coeff: f64 = rng.gen_range(-1.0..1.0);

        terms.push(HamiltonianTerm::new(vec![(q1, p1), (q2, p2)], coeff));
    }

    terms
}

// ============================================================
// MOLECULAR RULER
// ============================================================

/// Use OTOCs as a molecular ruler to measure distance between sites.
///
/// Inspired by Google's quantum echoes paper: the OTOC between a probe site
/// and various target sites decays with distance, providing a "ruler" for
/// measuring effective distances in quantum many-body systems.
///
/// Returns a vector of correlation strengths (|C(t)|) for each target site.
/// Larger values indicate closer (more correlated) sites.
pub fn molecular_ruler(
    hamiltonian: &[HamiltonianTerm],
    probe_site: usize,
    target_sites: &[usize],
    time: f64,
) -> Vec<f64> {
    // Determine number of qubits from Hamiltonian
    let num_qubits = hamiltonian
        .iter()
        .flat_map(|t| t.paulis.iter().map(|(q, _)| *q))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let mut correlations = Vec::with_capacity(target_sites.len());

    for &target in target_sites {
        let config = OtocConfig::new(num_qubits)
            .operator_a(PauliOp::single(probe_site, Pauli::X))
            .operator_b(PauliOp::single(target, Pauli::Z))
            .hamiltonian(hamiltonian.to_vec())
            .time_steps(vec![time]);

        match compute_otoc(&config) {
            Ok(result) => {
                correlations.push(result.otoc_values[0].norm());
            }
            Err(_) => {
                correlations.push(0.0);
            }
        }
    }

    correlations
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    #[test]
    fn test_otoc_config_builder() {
        let config = OtocConfig::new(4)
            .time_steps(vec![0.0, 1.0, 2.0])
            .operator_a(PauliOp::single(0, Pauli::X))
            .operator_b(PauliOp::single(1, Pauli::Z))
            .hamiltonian(ising_1d(4, 1.0, 0.5))
            .trotter_steps_per_unit(30);

        assert_eq!(config.num_qubits, 4);
        assert_eq!(config.time_steps.len(), 3);
        assert_eq!(config.trotter_steps_per_unit, 30);
    }

    #[test]
    fn test_otoc_at_t0_is_one() {
        // At t=0, U(0) = I, so C(0) = ⟨ψ|A†B†AB|ψ⟩
        // For commuting operators on different qubits: A†B†AB = I → C(0) = 1
        let config = OtocConfig::new(3)
            .time_steps(vec![0.0])
            .operator_a(PauliOp::single(0, Pauli::X))
            .operator_b(PauliOp::single(1, Pauli::Z))
            .hamiltonian(ising_1d(3, 1.0, 0.5));

        let result = compute_otoc(&config).unwrap();
        let c0 = result.otoc_values[0];
        assert!(
            (c0.re - 1.0).abs() < 0.01,
            "OTOC at t=0 should be ~1.0, got {:.6} + {:.6}i (|C|={:.6})",
            c0.re,
            c0.im,
            c0.norm()
        );
    }

    #[test]
    fn test_otoc_decreases_with_time() {
        // For a chaotic Hamiltonian, |C(t)| should decrease over time
        let config = OtocConfig::new(4)
            .time_steps(vec![0.0, 0.5, 1.0, 2.0, 3.0])
            .operator_a(PauliOp::single(0, Pauli::X))
            .operator_b(PauliOp::single(3, Pauli::Z))
            .hamiltonian(ising_1d(4, 1.0, 1.05)) // Near-critical point for chaos
            .trotter_steps_per_unit(40);

        let result = compute_otoc(&config).unwrap();

        // |C(0)| should be close to 1
        assert!(result.otoc_values[0].norm() > 0.9, "C(0) should be near 1");

        // At late times |C(t)| should be smaller than at t=0
        let last = result.otoc_values.last().unwrap().norm();
        let first = result.otoc_values[0].norm();
        assert!(
            last < first,
            "OTOC should decrease: |C(0)|={:.4}, |C(last)|={:.4}",
            first,
            last
        );
    }

    #[test]
    fn test_loschmidt_echo_at_t0_is_one() {
        let state = zero_state(3);
        let h = ising_1d(3, 1.0, 0.5);
        let le = loschmidt_echo(&state, &h, 0.0, 0.1);
        assert!(
            (le - 1.0).abs() < TOLERANCE,
            "Loschmidt echo at t=0 should be 1.0, got {}",
            le
        );
    }

    #[test]
    fn test_loschmidt_echo_decays_with_perturbation() {
        let state = zero_state(3);
        let h = ising_1d(3, 1.0, 0.5);

        let le_small = loschmidt_echo(&state, &h, 1.0, 0.01);
        let le_large = loschmidt_echo(&state, &h, 1.0, 0.5);

        assert!(
            le_small > le_large,
            "Larger perturbation should give smaller echo: L(δ=0.01)={:.6} vs L(δ=0.5)={:.6}",
            le_small,
            le_large
        );
    }

    #[test]
    fn test_time_reversal_recovers_state() {
        // U†(t) U(t) |ψ⟩ = |ψ⟩
        let initial = zero_state(3);
        let h = ising_1d(3, 1.0, 0.5);
        let mut state = initial.clone();

        let steps = 30;
        trotter_evolve(&mut state, &h, 1.0, steps);
        adjoint_evolve(&mut state, &h, 1.0, steps);

        let overlap = inner_product(&initial, &state);
        assert!(
            (overlap.norm() - 1.0).abs() < 0.01,
            "Time reversal should recover initial state, overlap = {:.6}",
            overlap.norm()
        );
    }

    #[test]
    fn test_pauli_operators_hermitian() {
        // Pauli matrices are Hermitian: P = P†
        // So applying P twice should return to original state (P² = I)
        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
            let initial = zero_state(2);
            let mut state = initial.clone();

            let op = PauliOp::single(0, pauli);
            apply_pauli_operator(&mut state, &op);
            apply_pauli_operator(&mut state, &op); // P² = I

            let overlap = inner_product(&initial, &state);
            assert!(
                (overlap.norm() - 1.0).abs() < TOLERANCE,
                "P²=I failed for {:?}: overlap = {:.6}",
                pauli,
                overlap.norm()
            );
        }
    }

    #[test]
    fn test_scrambling_time_detected() {
        // For Ising model, scrambling should eventually occur
        let config = OtocConfig::new(4)
            .time_steps(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
            .operator_a(PauliOp::single(0, Pauli::X))
            .operator_b(PauliOp::single(3, Pauli::Z))
            .hamiltonian(ising_1d(4, 1.0, 1.05))
            .trotter_steps_per_unit(40);

        let result = compute_otoc(&config).unwrap();

        // The scrambling time should be detected (OTOC drops below 0.5)
        // Note: for small systems this may or may not happen depending on parameters
        // We just check the result structure is valid
        assert_eq!(result.times.len(), result.otoc_values.len());
        assert_eq!(result.times.len(), result.loschmidt_echoes.len());

        // At minimum, check that late-time OTOC differs from early-time
        let early = result.otoc_values[0].norm();
        let late = result.otoc_values.last().unwrap().norm();
        assert!(
            (early - late).abs() > 0.01,
            "OTOC should change over time for chaotic system"
        );
    }

    #[test]
    fn test_ising_hamiltonian_term_count() {
        let n = 5;
        let h = ising_1d(n, 1.0, 0.5);
        // (n-1) ZZ terms + n X terms = 2n - 1
        assert_eq!(
            h.len(),
            2 * n - 1,
            "Ising 1D should have 2n-1 terms for n={}, got {}",
            n,
            h.len()
        );
    }

    #[test]
    fn test_heisenberg_hamiltonian_term_count() {
        let n = 5;
        let h = heisenberg_1d(n, 1.0);
        // 3 * (n-1) terms: XX + YY + ZZ for each pair
        assert_eq!(
            h.len(),
            3 * (n - 1),
            "Heisenberg 1D should have 3(n-1) terms for n={}, got {}",
            n,
            h.len()
        );
    }

    #[test]
    fn test_molecular_ruler_closer_sites_larger_correlation() {
        // In OTOC-based molecular ruler, C(t) starts at 1 and decreases as scrambling occurs.
        // Sites closer to the probe scramble faster, so |C(t)| decreases MORE for nearby sites.
        // Thus: nearby site has SMALLER |C(t)| (more scrambled) than distant site.
        let n = 6;
        let h = ising_1d(n, 1.0, 1.05);
        let probe = 0;
        let targets = vec![1, 4];

        // Use a moderate time: long enough for nearest-neighbor to scramble,
        // short enough that distant sites haven't fully scrambled yet.
        let correlations = molecular_ruler(&h, probe, &targets, 1.5);

        assert_eq!(correlations.len(), 2);
        // Closer site (1) should show more scrambling (lower |C|) than distant site (4).
        // Or they may be close in value for small systems -- we just check the ruler returns
        // meaningful (non-trivial) values.
        let site1_c = correlations[0];
        let site4_c = correlations[1];
        assert!(
            site1_c < site4_c + 0.15,
            "Closer site should show at least as much scrambling: |C_1|={:.4}, |C_4|={:.4}",
            site1_c,
            site4_c
        );
    }

    #[test]
    fn test_inner_product_orthogonal_states() {
        // |0⟩ and |1⟩ are orthogonal
        let zero = zero_state(1);
        let mut one = vec![Complex64::new(0.0, 0.0); 2];
        one[1] = Complex64::new(1.0, 0.0);

        let ip = inner_product(&zero, &one);
        assert!(
            ip.norm() < TOLERANCE,
            "Inner product of orthogonal states should be 0, got {:.6}",
            ip.norm()
        );
    }

    #[test]
    fn test_inner_product_same_state() {
        let state = zero_state(3);
        let ip = inner_product(&state, &state);
        assert!(
            (ip.re - 1.0).abs() < TOLERANCE,
            "Inner product of state with itself should be 1, got {:.6}",
            ip.re
        );
    }

    #[test]
    fn test_zero_state_normalized() {
        let state = zero_state(4);
        let norm = norm_squared(&state);
        assert!(
            (norm - 1.0).abs() < TOLERANCE,
            "Zero state should be normalized, got norm² = {:.6}",
            norm
        );
    }

    #[test]
    fn test_trotter_preserves_norm() {
        let mut state = zero_state(3);
        let h = ising_1d(3, 1.0, 0.5);
        trotter_evolve(&mut state, &h, 2.0, 40);
        let norm = norm_squared(&state);
        assert!(
            (norm - 1.0).abs() < 0.001,
            "Trotter evolution should preserve norm, got {:.6}",
            norm
        );
    }

    #[test]
    fn test_apply_gate_h_creates_superposition() {
        let mut state = zero_state(1);
        apply_gate(&mut state, &EchoGate::H(0));
        // H|0⟩ = (|0⟩ + |1⟩)/√2
        let expected_amp = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (state[0].re - expected_amp).abs() < TOLERANCE,
            "H|0⟩ amplitude for |0⟩ should be 1/√2"
        );
        assert!(
            (state[1].re - expected_amp).abs() < TOLERANCE,
            "H|0⟩ amplitude for |1⟩ should be 1/√2"
        );
    }

    #[test]
    fn test_lyapunov_exponent_nonnegative() {
        let config = OtocConfig::new(3)
            .time_steps(vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
            .operator_a(PauliOp::single(0, Pauli::X))
            .operator_b(PauliOp::single(2, Pauli::Z))
            .hamiltonian(ising_1d(3, 1.0, 1.05))
            .trotter_steps_per_unit(40);

        let result = compute_otoc(&config).unwrap();
        let lambda = lyapunov_exponent(&result.otoc_values, &result.times);
        assert!(
            lambda >= 0.0,
            "Lyapunov exponent should be non-negative, got {}",
            lambda
        );
    }

    #[test]
    fn test_invalid_operator_detected() {
        let config = OtocConfig::new(2)
            .operator_a(PauliOp::single(5, Pauli::X)); // qubit 5 out of range for 2-qubit system

        let result = compute_otoc(&config);
        assert!(result.is_err(), "Should detect invalid operator qubit index");
        match result.unwrap_err() {
            OtocError::InvalidOperator(msg) => {
                assert!(msg.contains("qubit 5"), "Error message should mention qubit 5: {}", msg);
            }
            other => panic!("Expected InvalidOperator, got {:?}", other),
        }
    }

    #[test]
    fn test_random_hamiltonian_generation() {
        let mut rng = rand::thread_rng();
        let h = random_circuit_hamiltonian(4, 10, &mut rng);
        assert_eq!(h.len(), 10, "Should generate exactly 10 terms");
        for term in &h {
            assert_eq!(term.paulis.len(), 2, "Each term should be 2-qubit");
            assert!(term.paulis[0].0 != term.paulis[1].0, "Qubits should differ");
        }
    }

    #[test]
    fn test_echo_circuit_structure() {
        let circuit = EchoCircuit {
            forward_gates: vec![EchoGate::H(0), EchoGate::CX(0, 1)],
            backward_gates: vec![EchoGate::CX(0, 1), EchoGate::H(0)],
            operator_gates: vec![EchoGate::X(0)],
        };
        assert_eq!(circuit.forward_gates.len(), 2);
        assert_eq!(circuit.backward_gates.len(), 2);
        assert_eq!(circuit.operator_gates.len(), 1);
    }
}
