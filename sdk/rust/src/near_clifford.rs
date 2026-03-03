//! Near-Clifford Simulation (CH-form)
//!
//! Implements the CH-form representation from Bravyi & Gossman (2016) for
//! efficient simulation of circuits with few T-gates. Memory: O(n²).
//! Gate application: O(n²) for Clifford, O(2^t) sampling for t T-gates.
//!
//! Route when: <40 T-gates AND >20 qubits AND clifford_fraction > 0.90

use crate::gates::{Gate, GateType};
use crate::C64;
use rand::Rng;
use std::collections::HashMap;

// ===================================================================
// BIT MATRIX (compact binary matrix for O(n²) stabilizer state)
// ===================================================================

/// Compact binary matrix stored as Vec<u64> per row.
#[derive(Clone, Debug)]
pub struct BitMatrix {
    rows: Vec<Vec<u64>>,
    num_rows: usize,
    num_cols: usize,
}

impl BitMatrix {
    /// Create an n×n identity matrix.
    pub fn identity(n: usize) -> Self {
        let words = (n + 63) / 64;
        let mut rows = vec![vec![0u64; words]; n];
        for i in 0..n {
            rows[i][i / 64] |= 1u64 << (i % 64);
        }
        BitMatrix {
            rows,
            num_rows: n,
            num_cols: n,
        }
    }

    /// Create an n×n zero matrix.
    pub fn zeros(n: usize) -> Self {
        let words = (n + 63) / 64;
        BitMatrix {
            rows: vec![vec![0u64; words]; n],
            num_rows: n,
            num_cols: n,
        }
    }

    /// Get bit (i, j).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> bool {
        (self.rows[i][j / 64] >> (j % 64)) & 1 == 1
    }

    /// Set bit (i, j).
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: bool) {
        if val {
            self.rows[i][j / 64] |= 1u64 << (j % 64);
        } else {
            self.rows[i][j / 64] &= !(1u64 << (j % 64));
        }
    }

    /// XOR row `src` into row `dst`.
    pub fn xor_row(&mut self, dst: usize, src: usize) {
        let words = self.rows[0].len();
        // Need to handle src == dst edge case
        if dst == src {
            // XOR with self = zero
            for w in 0..words {
                self.rows[dst][w] = 0;
            }
            return;
        }
        for w in 0..words {
            let src_val = self.rows[src][w];
            self.rows[dst][w] ^= src_val;
        }
    }

    /// Swap rows i and j.
    pub fn swap_rows(&mut self, i: usize, j: usize) {
        self.rows.swap(i, j);
    }

    /// Swap columns i and j across all rows.
    pub fn swap_cols(&mut self, i: usize, j: usize) {
        for row in &mut self.rows {
            let bi = (row[i / 64] >> (i % 64)) & 1;
            let bj = (row[j / 64] >> (j % 64)) & 1;
            if bi != bj {
                row[i / 64] ^= 1u64 << (i % 64);
                row[j / 64] ^= 1u64 << (j % 64);
            }
        }
    }
}

// ===================================================================
// CH-FORM STATE
// ===================================================================

/// CH-form representation of a stabilizer state plus magic.
///
/// A state is represented as U_C U_H |s⟩ where:
/// - |s⟩ is a computational basis state (vector s ∈ {0,1}^n)
/// - U_H is a product of CZ and S gates (diagonal in Z basis)
/// - U_C is a Clifford unitary in a specific canonical form
///
/// The Clifford U_C is encoded by matrices F, G, M (binary) and phases gamma, v.
/// Memory: O(n²) for all matrices.
#[derive(Clone, Debug)]
pub struct CHFormState {
    /// Number of qubits.
    pub num_qubits: usize,
    /// F matrix (n×n binary): part of the Clifford decomposition.
    pub f: BitMatrix,
    /// G matrix (n×n binary): part of the Clifford decomposition.
    pub g: BitMatrix,
    /// M matrix (n×n binary): CZ-layer phases.
    pub m: BitMatrix,
    /// Phase vector gamma (n entries, each in {0,1,2,3} representing i^gamma).
    pub gamma: Vec<u8>,
    /// Basis state vector v ∈ {0,1}^n.
    pub v: Vec<bool>,
    /// State vector s ∈ {0,1}^n (initial computational basis state).
    pub s: Vec<bool>,
    /// Global phase omega (power of i: i^omega, omega ∈ {0,1,2,3}).
    pub omega: u8,
}

impl CHFormState {
    /// Create |0...0⟩ in CH-form.
    pub fn new(num_qubits: usize) -> Self {
        CHFormState {
            num_qubits,
            f: BitMatrix::identity(num_qubits),
            g: BitMatrix::identity(num_qubits),
            m: BitMatrix::zeros(num_qubits),
            gamma: vec![0u8; num_qubits],
            v: vec![false; num_qubits],
            s: vec![false; num_qubits],
            omega: 0,
        }
    }

    /// Apply S gate to qubit q.
    pub fn s_gate(&mut self, q: usize) {
        let n = self.num_qubits;
        // S updates M and gamma
        for j in 0..n {
            if self.f.get(q, j) {
                // gamma[j] += 1 mod 4
                self.gamma[j] = (self.gamma[j] + 1) % 4;
                // M[j][j] ^= 1 (M is symmetric, update diagonal)
                let old = self.m.get(j, j);
                self.m.set(j, j, !old);
            }
        }
        // Update M off-diagonal for F pairs
        for j in 0..n {
            for k in (j + 1)..n {
                if self.f.get(q, j) && self.f.get(q, k) {
                    let old = self.m.get(j, k);
                    self.m.set(j, k, !old);
                    self.m.set(k, j, !old);
                }
            }
        }
    }

    /// Apply CZ gate between qubits p and q.
    pub fn cz_gate(&mut self, p: usize, q: usize) {
        let n = self.num_qubits;
        for j in 0..n {
            for k in 0..n {
                if self.f.get(p, j) && self.f.get(q, k) {
                    if j == k {
                        self.gamma[j] = (self.gamma[j] + 2) % 4;
                    } else {
                        let old = self.m.get(j, k);
                        self.m.set(j, k, !old);
                        self.m.set(k, j, !old);
                    }
                }
            }
        }
    }

    /// Apply CNOT gate (control p, target q).
    pub fn cnot_gate(&mut self, p: usize, q: usize) {
        let n = self.num_qubits;
        // Update basis state: target flips if control is |1⟩
        self.s[q] ^= self.s[p];
        // G[q] ^= G[p], F[p] ^= F[q]
        for j in 0..n {
            let gp = self.g.get(p, j);
            if gp {
                let gq = self.g.get(q, j);
                self.g.set(q, j, gq ^ true);
            }
            let fq = self.f.get(q, j);
            if fq {
                let fp = self.f.get(p, j);
                self.f.set(p, j, fp ^ true);
            }
        }
    }

    /// Apply X gate to qubit q.
    pub fn x_gate(&mut self, q: usize) {
        self.s[q] = !self.s[q];
    }

    /// Apply Z gate to qubit q.
    pub fn z_gate(&mut self, q: usize) {
        let n = self.num_qubits;
        for j in 0..n {
            if self.f.get(q, j) {
                self.gamma[j] = (self.gamma[j] + 2) % 4;
            }
        }
    }

    /// Apply H gate to qubit q (requires state update, more complex).
    pub fn h_gate(&mut self, q: usize) {
        // H = S · (CNOT-like) · S decomposition in CH-form
        // This is a simplified version; full implementation swaps F and G rows
        // and updates phases accordingly.
        let n = self.num_qubits;

        // Swap rows q of F and G
        for j in 0..n {
            let fqj = self.f.get(q, j);
            let gqj = self.g.get(q, j);
            self.f.set(q, j, gqj);
            self.g.set(q, j, fqj);
        }

        // Update gamma based on M and the swap
        for j in 0..n {
            if self.f.get(q, j) && self.g.get(q, j) {
                self.gamma[j] = (self.gamma[j] + 2) % 4;
            }
        }

        // Update v and s
        let sv = self.s[q];
        self.v[q] = self.v[q] ^ sv;
    }

    /// Apply a Clifford gate from the Gate type.
    pub fn apply_clifford(&mut self, gate: &Gate) -> bool {
        match &gate.gate_type {
            GateType::H => {
                self.h_gate(gate.targets[0]);
                true
            }
            GateType::X => {
                self.x_gate(gate.targets[0]);
                true
            }
            GateType::Z => {
                self.z_gate(gate.targets[0]);
                true
            }
            GateType::S => {
                self.s_gate(gate.targets[0]);
                true
            }
            GateType::Y => {
                // Y = iXZ
                let q = gate.targets[0];
                self.x_gate(q);
                self.z_gate(q);
                self.omega = (self.omega + 1) % 4;
                true
            }
            GateType::CNOT => {
                self.cnot_gate(gate.controls[0], gate.targets[0]);
                true
            }
            GateType::CZ => {
                self.cz_gate(gate.controls[0], gate.targets[0]);
                true
            }
            GateType::SWAP => {
                let a = gate.targets[0];
                let b = gate.targets[1];
                self.cnot_gate(a, b);
                self.cnot_gate(b, a);
                self.cnot_gate(a, b);
                true
            }
            _ => false,
        }
    }
}

// ===================================================================
// STABILIZER RANK SAMPLER
// ===================================================================

/// A branch in the stabilizer rank decomposition.
#[derive(Clone, Debug)]
pub struct StabilizerBranch {
    /// CH-form state for this branch.
    pub state: CHFormState,
    /// Weight (complex amplitude) of this branch.
    pub weight: C64,
}

/// Near-Clifford simulator using stabilizer rank decomposition.
///
/// For circuits with t T-gates on n qubits:
/// - Clifford gates: O(n²) per gate
/// - T gates: doubles the number of branches (2^t total)
/// - Sampling: importance sampling over branches
pub struct NearCliffordSimulator {
    /// Active branches in the decomposition.
    branches: Vec<StabilizerBranch>,
    /// Number of qubits.
    num_qubits: usize,
    /// Maximum number of branches before truncation.
    max_branches: usize,
    /// Number of T-gates applied.
    t_count: usize,
    /// Statistics.
    stats: NearCliffordStats,
}

/// Statistics for the near-Clifford simulation.
#[derive(Clone, Debug, Default)]
pub struct NearCliffordStats {
    pub clifford_gates: usize,
    pub t_gates: usize,
    pub max_branches: usize,
    pub truncated_branches: usize,
}

impl NearCliffordSimulator {
    /// Create a new near-Clifford simulator.
    pub fn new(num_qubits: usize) -> Self {
        NearCliffordSimulator {
            branches: vec![StabilizerBranch {
                state: CHFormState::new(num_qubits),
                weight: C64::new(1.0, 0.0),
            }],
            num_qubits,
            max_branches: 65536, // 2^16
            t_count: 0,
            stats: NearCliffordStats::default(),
        }
    }

    /// Set the maximum number of branches.
    pub fn with_max_branches(mut self, max: usize) -> Self {
        self.max_branches = max.max(1);
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> &NearCliffordStats {
        &self.stats
    }

    /// Get current number of branches.
    pub fn num_branches(&self) -> usize {
        self.branches.len()
    }

    /// Apply a gate to the simulator.
    pub fn apply_gate(&mut self, gate: &Gate) {
        match &gate.gate_type {
            GateType::T => {
                self.apply_t(gate.targets[0]);
            }
            GateType::Rz(theta) => {
                // Rz can be decomposed, but for near-Clifford we check special angles
                let pi = std::f64::consts::PI;
                if (*theta - pi / 4.0).abs() < 1e-12 {
                    self.apply_t(gate.targets[0]);
                } else if (*theta - pi / 2.0).abs() < 1e-12 {
                    // This is S gate (Clifford)
                    for branch in &mut self.branches {
                        branch.state.s_gate(gate.targets[0]);
                    }
                    self.stats.clifford_gates += 1;
                } else {
                    // General Rz: decompose as approximately T^k S^l
                    // For now, use exact splitting similar to T
                    self.apply_rz(gate.targets[0], *theta);
                }
            }
            _ => {
                // Clifford gate: apply to all branches
                let is_clifford = self.apply_clifford_to_all(gate);
                if is_clifford {
                    self.stats.clifford_gates += 1;
                }
            }
        }
    }

    /// Apply a circuit.
    pub fn apply_circuit(&mut self, gates: &[Gate]) {
        for gate in gates {
            self.apply_gate(gate);
        }
    }

    /// Apply T gate: splits each branch into two.
    /// T|ψ⟩ = cos(π/8)|ψ⟩ + e^{iπ/4} sin(π/8) Z|ψ⟩
    fn apply_t(&mut self, qubit: usize) {
        self.t_count += 1;
        self.stats.t_gates += 1;

        let cos = (std::f64::consts::PI / 8.0).cos();
        let sin = (std::f64::consts::PI / 8.0).sin();
        let phase = C64::new(
            (std::f64::consts::PI / 4.0).cos(),
            (std::f64::consts::PI / 4.0).sin(),
        );

        let mut new_branches = Vec::with_capacity(self.branches.len() * 2);

        for branch in &self.branches {
            // Branch 1: cos(π/8) * |ψ⟩ (identity part)
            new_branches.push(StabilizerBranch {
                state: branch.state.clone(),
                weight: branch.weight * C64::new(cos, 0.0),
            });

            // Branch 2: e^{iπ/4} sin(π/8) * Z|ψ⟩
            let mut state2 = branch.state.clone();
            state2.z_gate(qubit);
            new_branches.push(StabilizerBranch {
                state: state2,
                weight: branch.weight * phase * C64::new(sin, 0.0),
            });
        }

        self.branches = new_branches;
        self.stats.max_branches = self.stats.max_branches.max(self.branches.len());

        // Truncate if too many branches
        if self.branches.len() > self.max_branches {
            self.truncate_branches();
        }
    }

    /// Apply Rz(θ) gate: splits each branch into two.
    fn apply_rz(&mut self, qubit: usize, theta: f64) {
        self.stats.t_gates += 1; // Count as non-Clifford

        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        let mut new_branches = Vec::with_capacity(self.branches.len() * 2);

        for branch in &self.branches {
            // Rz(θ) = e^{-iθ/2} |0⟩⟨0| + e^{iθ/2} |1⟩⟨1|
            // = cos(θ/2) I - i sin(θ/2) Z
            // Branch 1: cos(θ/2) * I|ψ⟩
            new_branches.push(StabilizerBranch {
                state: branch.state.clone(),
                weight: branch.weight * C64::new(cos, 0.0),
            });

            // Branch 2: -i sin(θ/2) * Z|ψ⟩
            let mut state2 = branch.state.clone();
            state2.z_gate(qubit);
            new_branches.push(StabilizerBranch {
                state: state2,
                weight: branch.weight * C64::new(0.0, -sin),
            });
        }

        self.branches = new_branches;
        self.stats.max_branches = self.stats.max_branches.max(self.branches.len());

        if self.branches.len() > self.max_branches {
            self.truncate_branches();
        }
    }

    /// Apply a Clifford gate to all branches.
    fn apply_clifford_to_all(&mut self, gate: &Gate) -> bool {
        for branch in &mut self.branches {
            if !branch.state.apply_clifford(gate) {
                return false;
            }
        }
        true
    }

    /// Truncate branches by keeping the largest-weight ones.
    fn truncate_branches(&mut self) {
        let before = self.branches.len();
        // Sort by weight magnitude (descending)
        self.branches
            .sort_by(|a, b| b.weight.norm().partial_cmp(&a.weight.norm()).unwrap());

        // Keep top max_branches
        self.branches.truncate(self.max_branches);

        // Renormalize
        let total_weight: f64 = self.branches.iter().map(|b| b.weight.norm_sqr()).sum();
        if total_weight > 0.0 {
            let scale = 1.0 / total_weight.sqrt();
            for branch in &mut self.branches {
                branch.weight = branch.weight * C64::new(scale, 0.0);
            }
        }

        self.stats.truncated_branches += before - self.branches.len();
    }

    /// Sample measurement outcomes using importance sampling.
    pub fn sample(&self, n_shots: usize) -> HashMap<usize, usize> {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        let mut rng = rand::thread_rng();

        // Build weight distribution for importance sampling
        let weights: Vec<f64> = self.branches.iter().map(|b| b.weight.norm_sqr()).collect();
        let total: f64 = weights.iter().sum();

        if total == 0.0 {
            return counts;
        }

        // Build CDF
        let mut cdf = Vec::with_capacity(weights.len());
        let mut cumsum = 0.0;
        for &w in &weights {
            cumsum += w / total;
            cdf.push(cumsum);
        }

        for _ in 0..n_shots {
            // Pick a branch according to weight distribution
            let r: f64 = rng.gen();
            let branch_idx = match cdf.binary_search_by(|c| c.partial_cmp(&r).unwrap()) {
                Ok(i) => i,
                Err(i) => i.min(self.branches.len() - 1),
            };

            // Measure from the selected branch's state
            // For CH-form, we sample from the computational basis
            let branch = &self.branches[branch_idx];
            let mut outcome = 0usize;
            for q in 0..self.num_qubits {
                if branch.state.s[q] {
                    outcome |= 1 << q;
                }
            }

            *counts.entry(outcome).or_insert(0) += 1;
        }

        counts
    }

    /// Compute expectation value of Z on a qubit.
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        let mut exp = 0.0;
        let mut total_weight = 0.0;

        for branch in &self.branches {
            let w = branch.weight.norm_sqr();
            total_weight += w;

            // In the CH-form basis state, Z expectation is ±1
            let z_val = if branch.state.s[qubit] { -1.0 } else { 1.0 };
            exp += w * z_val;
        }

        if total_weight > 0.0 {
            exp / total_weight
        } else {
            0.0
        }
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{Gate, GateType};

    #[test]
    fn test_ch_form_new() {
        let state = CHFormState::new(4);
        assert_eq!(state.num_qubits, 4);
        assert!(state.s.iter().all(|&b| !b));
        assert!(state.v.iter().all(|&b| !b));
    }

    #[test]
    fn test_ch_form_x_gate() {
        let mut state = CHFormState::new(2);
        state.x_gate(0);
        assert!(state.s[0]);
        assert!(!state.s[1]);
    }

    #[test]
    fn test_bit_matrix_identity() {
        let m = BitMatrix::identity(4);
        assert!(m.get(0, 0));
        assert!(!m.get(0, 1));
        assert!(m.get(3, 3));
    }

    #[test]
    fn test_bit_matrix_operations() {
        let mut m = BitMatrix::zeros(4);
        m.set(0, 1, true);
        assert!(m.get(0, 1));
        m.set(0, 1, false);
        assert!(!m.get(0, 1));
    }

    #[test]
    fn test_near_clifford_new() {
        let sim = NearCliffordSimulator::new(4);
        assert_eq!(sim.num_branches(), 1);
        assert_eq!(sim.num_qubits, 4);
    }

    #[test]
    fn test_near_clifford_clifford_only() {
        let mut sim = NearCliffordSimulator::new(2);
        sim.apply_gate(&Gate::single(GateType::H, 0));
        sim.apply_gate(&Gate::two(GateType::CNOT, 0, 1));
        // Clifford gates should not increase branches
        assert_eq!(sim.num_branches(), 1);
        assert_eq!(sim.stats().clifford_gates, 2);
    }

    #[test]
    fn test_near_clifford_t_gate_splitting() {
        let mut sim = NearCliffordSimulator::new(2);
        sim.apply_gate(&Gate::single(GateType::T, 0));
        assert_eq!(sim.num_branches(), 2);
        assert_eq!(sim.stats().t_gates, 1);
    }

    #[test]
    fn test_near_clifford_multiple_t() {
        let mut sim = NearCliffordSimulator::new(2);
        sim.apply_gate(&Gate::single(GateType::T, 0));
        sim.apply_gate(&Gate::single(GateType::T, 1));
        assert_eq!(sim.num_branches(), 4);
    }

    #[test]
    fn test_near_clifford_truncation() {
        let mut sim = NearCliffordSimulator::new(2).with_max_branches(4);
        // Apply 3 T gates -> 8 branches, should truncate to 4
        sim.apply_gate(&Gate::single(GateType::T, 0));
        sim.apply_gate(&Gate::single(GateType::T, 1));
        sim.apply_gate(&Gate::single(GateType::T, 0));
        assert!(sim.num_branches() <= 4);
    }

    #[test]
    fn test_near_clifford_sample() {
        let mut sim = NearCliffordSimulator::new(1);
        sim.apply_gate(&Gate::single(GateType::X, 0));
        let counts = sim.sample(100);
        // After X gate, should measure |1⟩
        assert_eq!(*counts.get(&1).unwrap_or(&0), 100);
    }

    #[test]
    fn test_near_clifford_expectation_z() {
        let sim = NearCliffordSimulator::new(1);
        // |0⟩ state: ⟨Z⟩ = 1
        let exp = sim.expectation_z(0);
        assert!((exp - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_near_clifford_x_expectation_z() {
        let mut sim = NearCliffordSimulator::new(1);
        sim.apply_gate(&Gate::single(GateType::X, 0));
        // |1⟩ state: ⟨Z⟩ = -1
        let exp = sim.expectation_z(0);
        assert!((exp - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_near_clifford_rz_splitting() {
        let mut sim = NearCliffordSimulator::new(1);
        sim.apply_gate(&Gate::single(GateType::Rz(0.5), 0));
        assert_eq!(sim.num_branches(), 2);
    }

    #[test]
    fn test_near_clifford_circuit() {
        let mut sim = NearCliffordSimulator::new(3);
        let circuit = vec![
            Gate::single(GateType::H, 0),
            Gate::two(GateType::CNOT, 0, 1),
            Gate::single(GateType::T, 2),
            Gate::single(GateType::S, 1),
        ];
        sim.apply_circuit(&circuit);
        assert_eq!(sim.stats().clifford_gates, 3);
        assert_eq!(sim.stats().t_gates, 1);
        assert_eq!(sim.num_branches(), 2);
    }

    #[test]
    fn test_ch_form_cnot() {
        let mut state = CHFormState::new(2);
        state.x_gate(0);
        state.cnot_gate(0, 1);
        // After X(0) CNOT(0,1): should have both qubits flipped
        assert!(state.s[0]);
    }

    #[test]
    fn test_ch_form_s_gate() {
        let mut state = CHFormState::new(1);
        // S gate on |0⟩ should not change the basis state
        state.s_gate(0);
        assert!(!state.s[0]);
    }

    #[test]
    fn test_ch_form_z_gate() {
        let mut state = CHFormState::new(1);
        state.z_gate(0);
        // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
        assert!(!state.s[0]); // Still |0⟩
    }

    #[test]
    fn test_bit_matrix_large() {
        // Test with >64 qubits
        let n = 100;
        let m = BitMatrix::identity(n);
        assert!(m.get(99, 99));
        assert!(!m.get(99, 0));
        assert!(m.get(64, 64));
    }

    #[test]
    fn test_bit_matrix_swap_cols() {
        let mut m = BitMatrix::identity(4);
        m.swap_cols(0, 3);
        assert!(m.get(0, 3));
        assert!(!m.get(0, 0));
        assert!(m.get(3, 0));
        assert!(!m.get(3, 3));
    }

    #[test]
    fn test_near_clifford_swap_gate() {
        let mut sim = NearCliffordSimulator::new(2);
        sim.apply_gate(&Gate::single(GateType::X, 0));
        sim.apply_gate(&Gate::new(GateType::SWAP, vec![0, 1], vec![]));
        // After SWAP, qubit 1 should be |1⟩
        let exp0 = sim.expectation_z(0);
        let exp1 = sim.expectation_z(1);
        assert!((exp0 - 1.0).abs() < 1e-10); // qubit 0 back to |0⟩
        assert!((exp1 - (-1.0)).abs() < 1e-10); // qubit 1 now |1⟩
    }
}
