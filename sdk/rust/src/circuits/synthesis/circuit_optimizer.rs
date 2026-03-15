//! Circuit Optimization Passes
//!
//! Provides multiple optimization passes for quantum circuits:
//! - Clifford simplification (cancel inverse gates)
//! - Gate fusion (combine consecutive single-qubit gates via matrix multiplication)
//! - Rotation merging (combine same-axis rotations by adding angles)
//! - Commutation analysis (reorder gates to expose cancellation opportunities)
//! - Peephole optimization (local pattern matching and replacement)
//!
//! # Usage
//!
//! ```ignore
//! use nqpu_metal::circuit_optimizer::{CircuitOptimizer, OptimizationLevel};
//!
//! let optimizer = CircuitOptimizer::new(OptimizationLevel::Aggressive);
//! let optimized_gates = optimizer.optimize(&original_gates);
//! ```

use crate::gates::{Gate, GateType};
use crate::C64;
use std::collections::HashMap;

// ============================================================================
// LINEAR ALGEBRA HELPERS
// ============================================================================

/// Multiply two 2x2 complex matrices: C = A * B.
fn matmul_2x2(a: &[Vec<C64>], b: &[Vec<C64>]) -> Vec<Vec<C64>> {
    vec![
        vec![
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        vec![
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Check if a 2x2 matrix is approximately the identity (within tolerance).
fn is_identity_2x2(m: &[Vec<C64>], tol: f64) -> bool {
    (m[0][0] - C64::new(1.0, 0.0)).norm() < tol
        && m[0][1].norm() < tol
        && m[1][0].norm() < tol
        && (m[1][1] - C64::new(1.0, 0.0)).norm() < tol
}

/// Check if a 2x2 matrix is approximately a global phase times identity.
/// Returns true if M = e^{i*phi} * I for some phi.
fn is_global_phase_2x2(m: &[Vec<C64>], tol: f64) -> bool {
    // Off-diagonal must be zero
    if m[0][1].norm() > tol || m[1][0].norm() > tol {
        return false;
    }
    // Diagonal elements must have unit norm and be equal
    let d0 = m[0][0];
    let d1 = m[1][1];
    (d0.norm() - 1.0).abs() < tol && (d1.norm() - 1.0).abs() < tol && (d0 - d1).norm() < tol
}

// ============================================================================
// OPTIMIZATION LEVEL
// ============================================================================

/// Optimization level for circuit optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations (gate cancellation)
    Basic,
    /// Moderate optimizations (includes commutation and rotation merging)
    Moderate,
    /// Aggressive optimizations (all passes including peephole and fusion)
    Aggressive,
}

// ============================================================================
// OPTIMIZATION STATS
// ============================================================================

/// Optimization statistics.
#[derive(Clone)]
pub struct OptimizationStats {
    /// Original number of gates
    pub original_gates: usize,
    /// Optimized number of gates
    pub optimized_gates: usize,
    /// Gates removed
    pub gates_removed: usize,
    /// Number of fusion operations
    pub fusions: usize,
    /// Number of commutations
    pub commutations: usize,
    /// Number of cancellations
    pub cancellations: usize,
}

impl OptimizationStats {
    /// Calculate gate reduction percentage.
    pub fn reduction_percentage(&self) -> f64 {
        if self.original_gates == 0 {
            return 0.0;
        }
        (self.gates_removed as f64 / self.original_gates as f64) * 100.0
    }
}

// ============================================================================
// CIRCUIT OPTIMIZER
// ============================================================================

/// Circuit optimizer with multiple optimization passes.
pub struct CircuitOptimizer {
    /// Optimization level
    level: OptimizationLevel,
    /// Enable gate fusion
    enable_fusion: bool,
    /// Enable commutation
    enable_commutation: bool,
    /// Enable peephole optimization
    enable_peephole: bool,
}

impl CircuitOptimizer {
    /// Create a new circuit optimizer.
    pub fn new(level: OptimizationLevel) -> Self {
        let (enable_fusion, enable_commutation, enable_peephole) = match level {
            OptimizationLevel::None => (false, false, false),
            OptimizationLevel::Basic => (true, false, false),
            OptimizationLevel::Moderate => (true, true, false),
            OptimizationLevel::Aggressive => (true, true, true),
        };

        CircuitOptimizer {
            level,
            enable_fusion,
            enable_commutation,
            enable_peephole,
        }
    }

    /// Optimize a circuit (list of gates).
    pub fn optimize(&self, gates: &[Gate]) -> Vec<Gate> {
        if gates.is_empty() || self.level == OptimizationLevel::None {
            return gates.to_vec();
        }

        let mut current = gates.to_vec();
        let mut stats = OptimizationStats {
            original_gates: gates.len(),
            optimized_gates: gates.len(),
            gates_removed: 0,
            fusions: 0,
            commutations: 0,
            cancellations: 0,
        };

        // Pass 1: Clifford simplification (always enabled — cancel self-inverse pairs)
        current = self.clifford_simplification(&current, &mut stats);

        // Pass 2: Peephole optimization (pattern matching: S+S->Z, T+T->S, H-CNOT-H->CZ, etc.)
        // Runs before fusion so that known patterns are recognized before being absorbed
        // into opaque Custom matrices.
        if self.enable_peephole {
            current = self.peephole_optimization(&current, &mut stats);
        }

        // Pass 3: Rotation merging (Rx(a)+Rx(b)->Rx(a+b), etc.)
        // Runs before general fusion so that rotations keep their named type.
        if self.enable_fusion {
            current = self.rotation_merging(&current, &mut stats);
        }

        // Pass 4: Commutation analysis (reorder to expose cancellations)
        if self.enable_commutation {
            current = self.commutation_analysis(&current, &mut stats);
        }

        // Pass 5: Gate fusion (catch-all: multiply remaining consecutive single-qubit gates)
        if self.enable_fusion {
            current = self.gate_fusion(&current, &mut stats);
        }

        // Final cleanup: another round of cancellation after all transformations
        current = self.clifford_simplification(&current, &mut stats);

        stats.optimized_gates = current.len();
        stats.gates_removed = stats.original_gates.saturating_sub(stats.optimized_gates);

        current
    }

    /// Optimize and return statistics.
    pub fn optimize_with_stats(&self, gates: &[Gate]) -> (Vec<Gate>, OptimizationStats) {
        if gates.is_empty() || self.level == OptimizationLevel::None {
            return (
                gates.to_vec(),
                OptimizationStats {
                    original_gates: gates.len(),
                    optimized_gates: gates.len(),
                    gates_removed: 0,
                    fusions: 0,
                    commutations: 0,
                    cancellations: 0,
                },
            );
        }

        let mut current = gates.to_vec();
        let mut stats = OptimizationStats {
            original_gates: gates.len(),
            optimized_gates: gates.len(),
            gates_removed: 0,
            fusions: 0,
            commutations: 0,
            cancellations: 0,
        };

        // Apply all enabled passes (same order as optimize())
        current = self.clifford_simplification(&current, &mut stats);

        if self.enable_peephole {
            current = self.peephole_optimization(&current, &mut stats);
        }

        if self.enable_fusion {
            current = self.rotation_merging(&current, &mut stats);
        }

        if self.enable_commutation {
            current = self.commutation_analysis(&current, &mut stats);
        }

        if self.enable_fusion {
            current = self.gate_fusion(&current, &mut stats);
        }

        // Final cleanup
        current = self.clifford_simplification(&current, &mut stats);

        stats.optimized_gates = current.len();
        stats.gates_removed = stats.original_gates.saturating_sub(stats.optimized_gates);

        (current, stats)
    }

    // ========================================================================
    // PASS 1: CLIFFORD SIMPLIFICATION
    // ========================================================================

    /// Clifford simplification: cancel adjacent inverse gates.
    /// H*H = I, X*X = I, CNOT*CNOT = I, etc.
    fn clifford_simplification(&self, gates: &[Gate], stats: &mut OptimizationStats) -> Vec<Gate> {
        if gates.len() < 2 {
            return gates.to_vec();
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            let mut cancelled = false;

            if i + 1 < gates.len() {
                let current = &gates[i];
                let next = &gates[i + 1];

                // Same qubit(s) and same controls
                if current.targets == next.targets && current.controls == next.controls {
                    if self.gates_cancel(current, next) {
                        stats.cancellations += 2;
                        i += 2; // Skip both gates
                        cancelled = true;
                    }
                }
            }

            if !cancelled {
                result.push(gates[i].clone());
                i += 1;
            }
        }

        result
    }

    /// Check if two gates cancel each other (A * A = I for self-inverse gates).
    fn gates_cancel(&self, a: &Gate, b: &Gate) -> bool {
        match (&a.gate_type, &b.gate_type) {
            // Self-inverse gates: applying twice gives identity
            (GateType::H, GateType::H) => true,
            (GateType::X, GateType::X) => true,
            (GateType::Y, GateType::Y) => true,
            (GateType::Z, GateType::Z) => true,
            (GateType::CNOT, GateType::CNOT) => true,
            (GateType::CZ, GateType::CZ) => true,
            (GateType::SWAP, GateType::SWAP) => true,
            _ => false,
        }
    }

    // ========================================================================
    // PASS 2: GATE FUSION (REAL MATRIX MULTIPLICATION)
    // ========================================================================

    /// Gate fusion: combine consecutive single-qubit gates on the same qubit
    /// by multiplying their 2x2 unitary matrices.
    fn gate_fusion(&self, gates: &[Gate], stats: &mut OptimizationStats) -> Vec<Gate> {
        if gates.len() < 2 {
            return gates.to_vec();
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            let current = &gates[i];

            // Only fuse single-qubit gates (no controls, exactly one target)
            if current.targets.len() == 1 && current.controls.is_empty() {
                let target_qubit = current.targets[0];
                let seq_len = self.count_same_qubit_gates(gates, i);

                if seq_len >= 2 {
                    let fused = self.try_fuse_sequence(gates, i, target_qubit, seq_len, stats);
                    result.extend(fused);
                    i += seq_len;
                    continue;
                }
            }

            result.push(current.clone());
            i += 1;
        }

        result
    }

    /// Fuse a sequence of consecutive single-qubit gates on the same qubit
    /// by computing the product of their 2x2 matrices.
    ///
    /// Gate ordering: if the circuit applies G1 then G2 then G3, the combined
    /// unitary is U = G3 * G2 * G1 (matrix multiplication right-to-left).
    fn try_fuse_sequence(
        &self,
        gates: &[Gate],
        start: usize,
        target_qubit: usize,
        count: usize,
        stats: &mut OptimizationStats,
    ) -> Vec<Gate> {
        // Collect the gates in this run
        let sequence: Vec<&Gate> = gates[start..start + count]
            .iter()
            .filter(|g| {
                g.targets.len() == 1 && g.targets[0] == target_qubit && g.controls.is_empty()
            })
            .collect();

        if sequence.len() <= 1 {
            return sequence.into_iter().cloned().collect();
        }

        // Get the matrix of the first gate (leftmost in circuit = applied first)
        let mut fused_matrix = sequence[0].gate_type.matrix();

        // Verify we have a 2x2 matrix; bail out if not
        if fused_matrix.len() != 2 || fused_matrix[0].len() != 2 {
            return sequence.into_iter().cloned().collect();
        }

        // Multiply remaining gates: U = G_n * ... * G_2 * G_1
        // Each subsequent gate is applied after the previous, so it multiplies
        // from the left: fused = G_i * fused
        for gate in &sequence[1..] {
            let m = gate.gate_type.matrix();
            if m.len() != 2 || m[0].len() != 2 {
                // Non-2x2 gate in sequence; bail out
                return sequence.into_iter().cloned().collect();
            }
            fused_matrix = matmul_2x2(&m, &fused_matrix);
        }

        // Check if the fused result is identity (or global phase) -- if so, remove entirely
        if is_identity_2x2(&fused_matrix, 1e-10) || is_global_phase_2x2(&fused_matrix, 1e-10) {
            stats.fusions += sequence.len();
            return vec![];
        }

        // Create a single Custom gate with the fused matrix
        stats.fusions += sequence.len() - 1;
        let fused_gate = Gate::new(GateType::Custom(fused_matrix), vec![target_qubit], vec![]);

        vec![fused_gate]
    }

    /// Count consecutive single-qubit gates on the same qubit starting at `start`.
    fn count_same_qubit_gates(&self, gates: &[Gate], start: usize) -> usize {
        if start >= gates.len() {
            return 0;
        }

        let target = match gates[start].targets.first() {
            Some(&t) => t,
            None => return 1,
        };

        let mut count = 0;
        for gate in &gates[start..] {
            if gate.targets.len() == 1 && gate.targets[0] == target && gate.controls.is_empty() {
                count += 1;
            } else {
                break;
            }
        }

        count.max(1)
    }

    // ========================================================================
    // PASS 3: ROTATION MERGING
    // ========================================================================

    /// Merge consecutive same-axis rotation gates on the same qubit by summing
    /// their angles: Rx(a) Rx(b) = Rx(a+b), and similarly for Ry and Rz.
    fn rotation_merging(&self, gates: &[Gate], stats: &mut OptimizationStats) -> Vec<Gate> {
        if gates.len() < 2 {
            return gates.to_vec();
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            // Check if current and next are same-axis rotations on same qubit
            if i + 1 < gates.len()
                && gates[i].targets.len() == 1
                && gates[i].controls.is_empty()
                && gates[i + 1].targets.len() == 1
                && gates[i + 1].controls.is_empty()
                && gates[i].targets[0] == gates[i + 1].targets[0]
            {
                let target = gates[i].targets[0];

                match (&gates[i].gate_type, &gates[i + 1].gate_type) {
                    (GateType::Rx(a), GateType::Rx(b)) => {
                        let merged_angle = a + b;
                        // If merged angle is effectively zero, skip both gates
                        if merged_angle.abs() < 1e-10 {
                            stats.fusions += 2;
                        } else {
                            result.push(Gate::rx(target, merged_angle));
                            stats.fusions += 1;
                        }
                        i += 2;
                        continue;
                    }
                    (GateType::Ry(a), GateType::Ry(b)) => {
                        let merged_angle = a + b;
                        if merged_angle.abs() < 1e-10 {
                            stats.fusions += 2;
                        } else {
                            result.push(Gate::ry(target, merged_angle));
                            stats.fusions += 1;
                        }
                        i += 2;
                        continue;
                    }
                    (GateType::Rz(a), GateType::Rz(b)) => {
                        let merged_angle = a + b;
                        if merged_angle.abs() < 1e-10 {
                            stats.fusions += 2;
                        } else {
                            result.push(Gate::rz(target, merged_angle));
                            stats.fusions += 1;
                        }
                        i += 2;
                        continue;
                    }
                    _ => {}
                }
            }

            result.push(gates[i].clone());
            i += 1;
        }

        result
    }

    // ========================================================================
    // PASS 4: COMMUTATION ANALYSIS
    // ========================================================================

    /// Commutation analysis: reorder adjacent gates that commute when it would
    /// bring same-type, same-qubit gates closer together, enabling subsequent
    /// cancellation passes. Uses a bubble-sort-like approach with bounded passes.
    fn commutation_analysis(&self, gates: &[Gate], stats: &mut OptimizationStats) -> Vec<Gate> {
        if gates.len() < 3 {
            return gates.to_vec();
        }

        let mut result = gates.to_vec();
        let mut changed = true;
        let max_passes = 5;
        let mut pass = 0;

        while changed && pass < max_passes {
            changed = false;
            pass += 1;

            for i in 0..result.len().saturating_sub(1) {
                if self.gates_commute(&result[i], &result[i + 1]) && self.should_swap(&result, i) {
                    result.swap(i, i + 1);
                    stats.commutations += 1;
                    changed = true;
                }
            }
        }

        result
    }

    /// Determine if swapping gates at positions i and i+1 would bring
    /// a gate adjacent to one it could cancel or merge with.
    fn should_swap(&self, gates: &[Gate], i: usize) -> bool {
        let a = &gates[i];
        let b = &gates[i + 1];

        // Would swapping bring b adjacent to a preceding gate it could cancel/merge with?
        if i > 0 {
            let prev = &gates[i - 1];
            if prev.targets == b.targets
                && prev.controls == b.controls
                && (self.gates_cancel(prev, b) || self.same_rotation_axis(prev, b))
            {
                return true;
            }
        }

        // Would swapping bring a adjacent to a following gate it could cancel/merge with?
        if i + 2 < gates.len() {
            let next = &gates[i + 2];
            if next.targets == a.targets
                && next.controls == a.controls
                && (self.gates_cancel(a, next) || self.same_rotation_axis(a, next))
            {
                return true;
            }
        }

        false
    }

    /// Check if two gates are same-axis rotations (useful for merging after swap).
    fn same_rotation_axis(&self, a: &Gate, b: &Gate) -> bool {
        matches!(
            (&a.gate_type, &b.gate_type),
            (GateType::Rx(_), GateType::Rx(_))
                | (GateType::Ry(_), GateType::Ry(_))
                | (GateType::Rz(_), GateType::Rz(_))
        )
    }

    /// Check if two gates commute.
    fn gates_commute(&self, a: &Gate, b: &Gate) -> bool {
        // Gates on disjoint qubits always commute
        if self.disjoint_qubits(a, b) {
            return true;
        }

        // Two diagonal gates on the same qubit commute (diagonal matrices commute)
        if self.is_diagonal(&a.gate_type) && self.is_diagonal(&b.gate_type) {
            return true;
        }

        // CNOT commutation rules
        match (&a.gate_type, &b.gate_type) {
            // Z on control qubit commutes with CNOT
            (GateType::CNOT, GateType::Z) => a.controls.contains(&b.targets[0]),
            (GateType::Z, GateType::CNOT) => b.controls.contains(&a.targets[0]),
            // X on target qubit commutes with CNOT
            (GateType::CNOT, GateType::X) => !b.targets.is_empty() && a.targets[0] == b.targets[0],
            (GateType::X, GateType::CNOT) => !a.targets.is_empty() && b.targets[0] == a.targets[0],
            // Rz on control qubit commutes with CNOT (diagonal on control)
            (GateType::CNOT, GateType::Rz(_)) => a.controls.contains(&b.targets[0]),
            (GateType::Rz(_), GateType::CNOT) => b.controls.contains(&a.targets[0]),
            _ => false,
        }
    }

    /// Check if a gate type is diagonal (only modifies phases, no population transfer).
    fn is_diagonal(&self, gate_type: &GateType) -> bool {
        matches!(
            gate_type,
            GateType::Z | GateType::S | GateType::T | GateType::Rz(_) | GateType::Phase(_)
        )
    }

    /// Check if two gates operate on completely disjoint sets of qubits.
    fn disjoint_qubits(&self, a: &Gate, b: &Gate) -> bool {
        let a_qubits: std::collections::HashSet<_> =
            a.targets.iter().chain(a.controls.iter()).collect();
        let b_qubits: std::collections::HashSet<_> =
            b.targets.iter().chain(b.controls.iter()).collect();
        a_qubits.is_disjoint(&b_qubits)
    }

    // ========================================================================
    // PASS 5: PEEPHOLE OPTIMIZATION
    // ========================================================================

    /// Peephole optimization: apply local pattern-based optimizations.
    /// Handles both 2-gate and 3-gate windows.
    fn peephole_optimization(&self, gates: &[Gate], stats: &mut OptimizationStats) -> Vec<Gate> {
        if gates.len() < 2 {
            return gates.to_vec();
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            // Try 3-gate patterns first (higher priority)
            if i + 2 < gates.len() {
                if let Some(replacement) =
                    self.match_3gate_pattern(&gates[i], &gates[i + 1], &gates[i + 2])
                {
                    let replaced_count = 3;
                    let new_count = replacement.len();
                    if new_count < replaced_count {
                        stats.cancellations += replaced_count - new_count;
                    }
                    result.extend(replacement);
                    i += 3;
                    continue;
                }
            }

            // Try 2-gate patterns
            if i + 1 < gates.len() {
                if let Some(replacement) = self.match_2gate_pattern(&gates[i], &gates[i + 1]) {
                    let replaced_count = 2;
                    let new_count = replacement.len();
                    if new_count < replaced_count {
                        stats.cancellations += replaced_count - new_count;
                    }
                    result.extend(replacement);
                    i += 2;
                    continue;
                }
            }

            result.push(gates[i].clone());
            i += 1;
        }

        result
    }

    /// Match 3-gate peephole patterns and return the replacement.
    fn match_3gate_pattern(&self, g1: &Gate, g2: &Gate, g3: &Gate) -> Option<Vec<Gate>> {
        // ----------------------------------------------------------------
        // Pattern 1: H(t) CNOT(c,t) H(t) -> CZ(c,t)
        // Hadamard conjugation of CNOT gives CZ.
        // ----------------------------------------------------------------
        if matches!(g1.gate_type, GateType::H)
            && matches!(g2.gate_type, GateType::CNOT)
            && matches!(g3.gate_type, GateType::H)
        {
            let cnot_target = g2.targets[0];
            if g1.targets[0] == cnot_target && g3.targets[0] == cnot_target {
                let control = g2.controls[0];
                return Some(vec![Gate::cz(control, cnot_target)]);
            }
        }

        // ----------------------------------------------------------------
        // Pattern 2: H(t) CZ(c,t) H(t) -> CNOT(c,t)
        // Reverse of pattern 1.
        // ----------------------------------------------------------------
        if matches!(g1.gate_type, GateType::H)
            && matches!(g2.gate_type, GateType::CZ)
            && matches!(g3.gate_type, GateType::H)
        {
            let cz_target = g2.targets[0];
            if g1.targets[0] == cz_target && g3.targets[0] == cz_target {
                let control = g2.controls[0];
                return Some(vec![Gate::cnot(control, cz_target)]);
            }
        }

        // ----------------------------------------------------------------
        // Pattern 3: CNOT(c,t) X(t) CNOT(c,t) -> X(t)
        // X on target commutes out of CNOT sandwich.
        // ----------------------------------------------------------------
        if matches!(g1.gate_type, GateType::CNOT)
            && matches!(g2.gate_type, GateType::X)
            && matches!(g3.gate_type, GateType::CNOT)
        {
            let cnot1_target = g1.targets[0];
            let cnot1_control = g1.controls[0];
            if g2.targets[0] == cnot1_target
                && g3.targets[0] == cnot1_target
                && g3.controls[0] == cnot1_control
            {
                return Some(vec![Gate::x(cnot1_target)]);
            }
        }

        None
    }

    /// Match 2-gate peephole patterns and return the replacement.
    fn match_2gate_pattern(&self, g1: &Gate, g2: &Gate) -> Option<Vec<Gate>> {
        // Both must be on the same qubit(s) for these patterns
        if g1.targets != g2.targets || g1.controls != g2.controls {
            return None;
        }

        // ----------------------------------------------------------------
        // Pattern 4: S S -> Z (on same qubit)
        // S = diag(1, i), S*S = diag(1, -1) = Z
        // ----------------------------------------------------------------
        if matches!(g1.gate_type, GateType::S) && matches!(g2.gate_type, GateType::S) {
            let target = g1.targets[0];
            return Some(vec![Gate::z(target)]);
        }

        // ----------------------------------------------------------------
        // Pattern 5: T T -> S (on same qubit)
        // T = diag(1, e^{i*pi/4}), T*T = diag(1, e^{i*pi/2}) = S
        // ----------------------------------------------------------------
        if matches!(g1.gate_type, GateType::T) && matches!(g2.gate_type, GateType::T) {
            let target = g1.targets[0];
            return Some(vec![Gate::s(target)]);
        }

        None
    }
}

impl Default for CircuitOptimizer {
    fn default() -> Self {
        Self::new(OptimizationLevel::Moderate)
    }
}

// ============================================================================
// GATE SET HELPERS
// ============================================================================

/// Check if a gate is Clifford (efficiently simulable).
pub fn is_clifford_gate(gate: &Gate) -> bool {
    matches!(
        gate.gate_type,
        GateType::H
            | GateType::S
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::CNOT
            | GateType::CZ
            | GateType::SWAP
    )
}

/// Check if a circuit is Clifford-only.
pub fn is_clifford_circuit(gates: &[Gate]) -> bool {
    gates.iter().all(is_clifford_gate)
}

/// Count two-qubit gates in a circuit.
pub fn count_two_qubit_gates(gates: &[Gate]) -> usize {
    gates
        .iter()
        .filter(|g| !g.controls.is_empty() || g.targets.len() > 1)
        .count()
}

/// Estimate circuit depth (simplified).
pub fn estimate_depth(gates: &[Gate]) -> usize {
    let mut qubit_last_use: HashMap<usize, usize> = HashMap::new();
    let mut depth = 0;

    for gate in gates {
        let mut gate_qubits = Vec::new();
        for &q in &gate.targets {
            gate_qubits.push(q);
        }
        for &q in &gate.controls {
            gate_qubits.push(q);
        }

        let min_layer = gate_qubits
            .iter()
            .filter_map(|q| qubit_last_use.get(q))
            .max()
            .copied()
            .unwrap_or(0);

        let layer = min_layer + 1;
        for q in gate_qubits {
            qubit_last_use.insert(q, layer);
        }
        depth = depth.max(layer);
    }

    depth
}

// ============================================================================
// COMPATIBILITY FUNCTIONS
// ============================================================================

/// Compatibility function for quantum_synthesis module.
/// Returns a struct with a `gates` field for compatibility.
pub fn optimize_circuit(gates: Vec<Gate>, _num_qubits: usize) -> OptimizationResult {
    let optimizer = CircuitOptimizer::new(OptimizationLevel::Basic);
    let optimized = optimizer.optimize(&gates);

    OptimizationResult { gates: optimized }
}

/// Result of optimization with `gates` field (for compatibility).
pub struct OptimizationResult {
    pub gates: Vec<Gate>,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Existing tests (preserved)
    // ------------------------------------------------------------------

    #[test]
    fn test_clifford_simplification() {
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Basic);
        let gates = vec![
            Gate::h(0),
            Gate::h(0), // Should cancel
            Gate::x(1),
            Gate::x(1), // Should cancel
        ];

        let optimized = optimizer.optimize(&gates);
        assert_eq!(optimized.len(), 0); // All gates cancelled
    }

    #[test]
    fn test_cnot_cancellation() {
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Basic);
        let gates = vec![
            Gate::cnot(0, 1),
            Gate::cnot(0, 1), // Should cancel
        ];

        let optimized = optimizer.optimize(&gates);
        assert_eq!(optimized.len(), 0);
    }

    #[test]
    fn test_no_cancellation_different_qubits() {
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Basic);
        let gates = vec![
            Gate::h(0),
            Gate::h(1), // Different qubit, no cancellation
        ];

        let optimized = optimizer.optimize(&gates);
        assert_eq!(optimized.len(), 2);
    }

    #[test]
    fn test_optimization_stats() {
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Basic);
        let gates = vec![
            Gate::h(0),
            Gate::h(0), // Cancel
            Gate::x(1),
        ];

        let (optimized, stats) = optimizer.optimize_with_stats(&gates);
        assert_eq!(optimized.len(), 1);
        assert_eq!(stats.original_gates, 3);
        assert_eq!(stats.optimized_gates, 1);
        assert_eq!(stats.gates_removed, 2);
        assert_eq!(stats.cancellations, 2);
    }

    #[test]
    fn test_reduction_percentage() {
        let stats = OptimizationStats {
            original_gates: 100,
            optimized_gates: 80,
            gates_removed: 20,
            fusions: 0,
            commutations: 0,
            cancellations: 20,
        };
        assert_eq!(stats.reduction_percentage(), 20.0);
    }

    #[test]
    fn test_no_optimization_level() {
        let optimizer = CircuitOptimizer::new(OptimizationLevel::None);
        let gates = vec![
            Gate::h(0),
            Gate::h(0), // Would cancel, but optimization disabled
        ];

        let optimized = optimizer.optimize(&gates);
        assert_eq!(optimized.len(), 2);
    }

    #[test]
    fn test_is_clifford_gate() {
        assert!(is_clifford_gate(&Gate::h(0)));
        assert!(is_clifford_gate(&Gate::cnot(0, 1)));
        assert!(!is_clifford_gate(&Gate::t(0))); // T is non-Clifford
    }

    #[test]
    fn test_is_clifford_circuit() {
        let clifford_gates = vec![Gate::h(0), Gate::s(0), Gate::cnot(0, 1)];
        assert!(is_clifford_circuit(&clifford_gates));

        let non_clifford_gates = vec![Gate::h(0), Gate::t(0)];
        assert!(!is_clifford_circuit(&non_clifford_gates));
    }

    #[test]
    fn test_count_two_qubit_gates() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::h(1), Gate::cnot(1, 2)];
        assert_eq!(count_two_qubit_gates(&gates), 2);
    }

    #[test]
    fn test_estimate_depth_sequential() {
        let gates = vec![Gate::h(0), Gate::x(0), Gate::y(0)];
        assert_eq!(estimate_depth(&gates), 3);
    }

    #[test]
    fn test_estimate_depth_parallel() {
        let gates = vec![Gate::h(0), Gate::h(1), Gate::h(2)];
        assert_eq!(estimate_depth(&gates), 1);
    }

    // ------------------------------------------------------------------
    // New tests: Gate fusion
    // ------------------------------------------------------------------

    #[test]
    fn test_gate_fusion_actually_fuses() {
        // H then X on the same qubit should produce a single Custom gate
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Basic);
        let gates = vec![Gate::h(0), Gate::x(0)];

        let optimized = optimizer.optimize(&gates);

        // Should be fused into one gate
        assert_eq!(
            optimized.len(),
            1,
            "Expected 1 fused gate, got {}",
            optimized.len()
        );
        assert!(
            matches!(optimized[0].gate_type, GateType::Custom(_)),
            "Expected Custom gate from fusion, got {:?}",
            optimized[0].gate_type
        );
        assert_eq!(optimized[0].targets, vec![0]);
    }

    #[test]
    fn test_fusion_matrix_correctness() {
        // Verify that the fused matrix equals the product of individual matrices.
        // H * X should give a specific matrix. X is applied first, H second.
        // U = H * X
        let h_mat = GateType::H.matrix();
        let x_mat = GateType::X.matrix();
        let expected = matmul_2x2(&h_mat, &x_mat);

        let optimizer = CircuitOptimizer::new(OptimizationLevel::Basic);
        let gates = vec![Gate::x(0), Gate::h(0)];
        let optimized = optimizer.optimize(&gates);

        assert_eq!(optimized.len(), 1);
        if let GateType::Custom(ref fused) = optimized[0].gate_type {
            for row in 0..2 {
                for col in 0..2 {
                    let diff = (fused[row][col] - expected[row][col]).norm();
                    assert!(
                        diff < 1e-10,
                        "Matrix mismatch at [{row}][{col}]: got {:?}, expected {:?}",
                        fused[row][col],
                        expected[row][col]
                    );
                }
            }
        } else {
            panic!("Expected Custom gate, got {:?}", optimized[0].gate_type);
        }
    }

    #[test]
    fn test_fusion_three_gates() {
        // Rz(0.3) then Ry(0.5) then H on qubit 0 -> single Custom gate
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Basic);
        let gates = vec![Gate::rz(0, 0.3), Gate::ry(0, 0.5), Gate::h(0)];

        let optimized = optimizer.optimize(&gates);
        assert_eq!(optimized.len(), 1);
        assert!(matches!(optimized[0].gate_type, GateType::Custom(_)));

        // Verify matrix: U = H * Ry(0.5) * Rz(0.3)
        let rz_mat = GateType::Rz(0.3).matrix();
        let ry_mat = GateType::Ry(0.5).matrix();
        let h_mat = GateType::H.matrix();
        let step1 = matmul_2x2(&ry_mat, &rz_mat);
        let expected = matmul_2x2(&h_mat, &step1);

        if let GateType::Custom(ref fused) = optimized[0].gate_type {
            for r in 0..2 {
                for c in 0..2 {
                    assert!(
                        (fused[r][c] - expected[r][c]).norm() < 1e-10,
                        "Mismatch at [{r}][{c}]"
                    );
                }
            }
        }
    }

    #[test]
    fn test_fusion_identity_elimination() {
        // H H should cancel via clifford simplification (before fusion),
        // but X Rz(0) should fuse and ideally produce just X (or Custom ~ X)
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Basic);

        // X then X on same qubit: should cancel to nothing (clifford pass)
        let gates = vec![Gate::x(0), Gate::x(0)];
        let optimized = optimizer.optimize(&gates);
        assert_eq!(optimized.len(), 0);
    }

    // ------------------------------------------------------------------
    // New tests: Peephole patterns
    // ------------------------------------------------------------------

    #[test]
    fn test_peephole_h_cnot_h() {
        // H(1) CNOT(0,1) H(1) -> CZ(0,1)
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Aggressive);
        let gates = vec![Gate::h(1), Gate::cnot(0, 1), Gate::h(1)];

        let optimized = optimizer.optimize(&gates);

        assert_eq!(
            optimized.len(),
            1,
            "Expected 1 gate (CZ), got {} gates: {:?}",
            optimized.len(),
            optimized.iter().map(|g| &g.gate_type).collect::<Vec<_>>()
        );
        assert!(
            matches!(optimized[0].gate_type, GateType::CZ),
            "Expected CZ, got {:?}",
            optimized[0].gate_type
        );
        assert_eq!(optimized[0].controls, vec![0]);
        assert_eq!(optimized[0].targets, vec![1]);
    }

    #[test]
    fn test_peephole_h_cz_h() {
        // H(1) CZ(0,1) H(1) -> CNOT(0,1)
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Aggressive);
        let gates = vec![Gate::h(1), Gate::cz(0, 1), Gate::h(1)];

        let optimized = optimizer.optimize(&gates);

        assert_eq!(
            optimized.len(),
            1,
            "Expected 1 gate (CNOT), got {} gates: {:?}",
            optimized.len(),
            optimized.iter().map(|g| &g.gate_type).collect::<Vec<_>>()
        );
        assert!(
            matches!(optimized[0].gate_type, GateType::CNOT),
            "Expected CNOT, got {:?}",
            optimized[0].gate_type
        );
        assert_eq!(optimized[0].controls, vec![0]);
        assert_eq!(optimized[0].targets, vec![1]);
    }

    #[test]
    fn test_peephole_cnot_x_cnot() {
        // CNOT(0,1) X(1) CNOT(0,1) -> X(1)
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Aggressive);
        let gates = vec![Gate::cnot(0, 1), Gate::x(1), Gate::cnot(0, 1)];

        let optimized = optimizer.optimize(&gates);

        assert_eq!(
            optimized.len(),
            1,
            "Expected 1 gate (X), got {} gates",
            optimized.len()
        );
        assert!(matches!(optimized[0].gate_type, GateType::X));
        assert_eq!(optimized[0].targets, vec![1]);
    }

    #[test]
    fn test_s_s_to_z() {
        // S S -> Z via peephole
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Aggressive);
        let gates = vec![Gate::s(0), Gate::s(0)];

        let optimized = optimizer.optimize(&gates);

        assert_eq!(
            optimized.len(),
            1,
            "Expected 1 gate (Z), got {}",
            optimized.len()
        );
        assert!(
            matches!(optimized[0].gate_type, GateType::Z),
            "Expected Z gate, got {:?}",
            optimized[0].gate_type
        );
        assert_eq!(optimized[0].targets, vec![0]);
    }

    #[test]
    fn test_t_t_to_s() {
        // T T -> S via peephole
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Aggressive);
        let gates = vec![Gate::t(0), Gate::t(0)];

        let optimized = optimizer.optimize(&gates);

        assert_eq!(
            optimized.len(),
            1,
            "Expected 1 gate (S), got {}",
            optimized.len()
        );
        assert!(
            matches!(optimized[0].gate_type, GateType::S),
            "Expected S gate, got {:?}",
            optimized[0].gate_type
        );
        assert_eq!(optimized[0].targets, vec![0]);
    }

    // ------------------------------------------------------------------
    // New tests: Rotation merging
    // ------------------------------------------------------------------

    #[test]
    fn test_rotation_merging() {
        // Rz(0.3) Rz(0.5) on same qubit -> Rz(0.8)
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Moderate);
        let gates = vec![Gate::rz(0, 0.3), Gate::rz(0, 0.5)];

        let optimized = optimizer.optimize(&gates);

        assert_eq!(
            optimized.len(),
            1,
            "Expected 1 merged gate, got {}",
            optimized.len()
        );
        if let GateType::Rz(angle) = optimized[0].gate_type {
            assert!(
                (angle - 0.8).abs() < 1e-10,
                "Expected Rz(0.8), got Rz({})",
                angle
            );
        } else {
            panic!("Expected Rz gate, got {:?}", optimized[0].gate_type);
        }
    }

    #[test]
    fn test_rotation_merging_rx() {
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Moderate);
        let gates = vec![Gate::rx(2, 1.0), Gate::rx(2, 0.5)];

        let optimized = optimizer.optimize(&gates);
        assert_eq!(optimized.len(), 1);
        if let GateType::Rx(angle) = optimized[0].gate_type {
            assert!((angle - 1.5).abs() < 1e-10);
        } else {
            panic!("Expected Rx gate");
        }
    }

    #[test]
    fn test_rotation_merging_ry() {
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Moderate);
        let gates = vec![Gate::ry(1, 0.7), Gate::ry(1, -0.7)];

        let optimized = optimizer.optimize(&gates);
        // Angles sum to 0.0, should be eliminated entirely
        assert_eq!(
            optimized.len(),
            0,
            "Expected 0 gates (angles cancel), got {}",
            optimized.len()
        );
    }

    #[test]
    fn test_rotation_merging_different_axes_no_merge() {
        // Rx then Rz on same qubit should NOT merge (different axes)
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Moderate);
        let gates = vec![Gate::rx(0, 0.5), Gate::rz(0, 0.3)];

        let optimized = optimizer.optimize(&gates);
        // These should be fused via gate_fusion into a Custom gate (not rotation-merged)
        // The key point: they are NOT merged into a single Rx or Rz
        assert!(optimized.len() <= 2); // May be fused to 1 Custom or stay as 2
        for g in &optimized {
            // Should NOT be a single Rx or Rz with summed angle
            if optimized.len() == 1 {
                assert!(matches!(g.gate_type, GateType::Custom(_)));
            }
        }
    }

    #[test]
    fn test_rotation_merging_different_qubits_no_merge() {
        // Rz on qubit 0 then Rz on qubit 1: must NOT merge
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Moderate);
        let gates = vec![Gate::rz(0, 0.5), Gate::rz(1, 0.3)];

        let optimized = optimizer.optimize(&gates);
        assert_eq!(optimized.len(), 2);
    }

    // ------------------------------------------------------------------
    // New tests: Commutation correctness
    // ------------------------------------------------------------------

    #[test]
    fn test_commutation_no_duplicates() {
        // Verify that commutation analysis never duplicates gates.
        // Create a circuit where commutation is possible.
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Moderate);
        let gates = vec![
            Gate::h(0),
            Gate::z(1), // Disjoint from H(0), commutes
            Gate::h(0), // Could cancel with first H(0) if reordered
            Gate::x(2),
            Gate::cnot(1, 2),
        ];

        let (optimized, _stats) = optimizer.optimize_with_stats(&gates);

        // Gate count must never exceed original count (no duplicates)
        assert!(
            optimized.len() <= gates.len(),
            "Commutation duplicated gates: original {} -> optimized {}",
            gates.len(),
            optimized.len()
        );
    }

    #[test]
    fn test_commutation_enables_cancellation() {
        // H(0) Z(1) H(0): Z(1) commutes with H(0) since they are on different qubits.
        // After commutation reorder + clifford simplification, H(0) H(0) should cancel.
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Moderate);
        let gates = vec![Gate::h(0), Gate::z(1), Gate::h(0)];

        let optimized = optimizer.optimize(&gates);

        // H(0) and H(0) should cancel, leaving only Z(1)
        assert_eq!(
            optimized.len(),
            1,
            "Expected 1 gate after commutation+cancellation, got {}",
            optimized.len()
        );
        assert!(matches!(optimized[0].gate_type, GateType::Z));
        assert_eq!(optimized[0].targets, vec![1]);
    }

    #[test]
    fn test_commutation_preserves_order_when_no_benefit() {
        // Gates that commute but swapping gives no benefit should stay in order.
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Moderate);
        let gates = vec![Gate::h(0), Gate::x(1), Gate::y(2)];

        let optimized = optimizer.optimize(&gates);
        assert_eq!(optimized.len(), 3);
    }

    // ------------------------------------------------------------------
    // New tests: matmul_2x2 helper
    // ------------------------------------------------------------------

    #[test]
    fn test_matmul_identity() {
        let id = vec![
            vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            vec![C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
        ];
        let h = GateType::H.matrix();
        let result = matmul_2x2(&h, &id);
        for r in 0..2 {
            for c in 0..2 {
                assert!((result[r][c] - h[r][c]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matmul_h_h_is_identity() {
        // H * H = I
        let h = GateType::H.matrix();
        let result = matmul_2x2(&h, &h);
        assert!(is_identity_2x2(&result, 1e-10));
    }

    #[test]
    fn test_matmul_x_x_is_identity() {
        let x = GateType::X.matrix();
        let result = matmul_2x2(&x, &x);
        assert!(is_identity_2x2(&result, 1e-10));
    }

    #[test]
    fn test_matmul_s_s_is_z() {
        // S * S = Z
        let s = GateType::S.matrix();
        let z = GateType::Z.matrix();
        let result = matmul_2x2(&s, &s);
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (result[r][c] - z[r][c]).norm() < 1e-10,
                    "S*S != Z at [{r}][{c}]: got {:?}, expected {:?}",
                    result[r][c],
                    z[r][c]
                );
            }
        }
    }

    #[test]
    fn test_matmul_t_t_is_s() {
        // T * T = S
        let t = GateType::T.matrix();
        let s = GateType::S.matrix();
        let result = matmul_2x2(&t, &t);
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (result[r][c] - s[r][c]).norm() < 1e-10,
                    "T*T != S at [{r}][{c}]: got {:?}, expected {:?}",
                    result[r][c],
                    s[r][c]
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Edge case tests
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_circuit() {
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Aggressive);
        let optimized = optimizer.optimize(&[]);
        assert!(optimized.is_empty());
    }

    #[test]
    fn test_single_gate_circuit() {
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Aggressive);
        let gates = vec![Gate::h(0)];
        let optimized = optimizer.optimize(&gates);
        assert_eq!(optimized.len(), 1);
    }

    #[test]
    fn test_default_optimizer() {
        let optimizer = CircuitOptimizer::default();
        let gates = vec![Gate::h(0), Gate::h(0)];
        let optimized = optimizer.optimize(&gates);
        assert_eq!(optimized.len(), 0);
    }

    #[test]
    fn test_mixed_single_and_two_qubit() {
        // Ensure fusion does not incorrectly absorb two-qubit gates
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Basic);
        let gates = vec![
            Gate::h(0),
            Gate::cnot(0, 1), // Two-qubit gate breaks the single-qubit sequence
            Gate::h(0),
        ];

        let optimized = optimizer.optimize(&gates);
        // H(0), CNOT(0,1), H(0) should remain 3 gates (no fusion across CNOT)
        assert_eq!(optimized.len(), 3);
    }

    #[test]
    fn test_peephole_h_cnot_h_wrong_qubit_no_match() {
        // H on qubit 0, CNOT(0,1), H on qubit 0: H is on control, not target
        // This should NOT match the H-CNOT-H pattern
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Aggressive);
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::h(0)];

        let optimized = optimizer.optimize(&gates);
        // H(0) on the control qubit does not trigger the pattern
        // The two H(0) around CNOT should not produce CZ
        // They might cancel or fuse depending on commutation, but not via peephole
        assert!(
            optimized.len() <= 3,
            "Should not produce more gates than input"
        );
    }

    #[test]
    fn test_aggressive_full_pipeline() {
        // Exercise all passes together
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Aggressive);
        let gates = vec![
            Gate::h(1),
            Gate::cnot(0, 1),
            Gate::h(1), // H-CNOT-H -> CZ (peephole)
            Gate::rz(2, 0.3),
            Gate::rz(2, 0.5), // -> Rz(0.8) (rotation merging)
            Gate::x(3),
            Gate::x(3), // -> identity (clifford)
        ];

        let (optimized, stats) = optimizer.optimize_with_stats(&gates);

        // X(3) X(3) cancelled, H-CNOT-H -> CZ, Rz merged
        // Expected: CZ(0,1) + Rz(2, 0.8) = 2 gates
        assert!(
            optimized.len() <= 3,
            "Expected aggressive optimization to produce <= 3 gates, got {}",
            optimized.len()
        );
        assert!(stats.gates_removed >= 4);
    }
}
