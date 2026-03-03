//! Gate Fusion Engine
//!
//! Fuses consecutive single-qubit gates on the same qubit into a single 2x2 unitary,
//! and absorbs pending single-qubit gates into adjacent 2-qubit gates as fused 4x4 unitaries,
//! reducing state-vector traversals.

use crate::ascii_viz::apply_gate_to_state;
use crate::gates::{Gate, GateType};
use crate::{GateOperations, QuantumState, C64};
use num_complex::Complex64;
use std::collections::HashMap;

// ===================================================================
// MATRIX TYPES
// ===================================================================

/// Compact 2x2 complex matrix for fused single-qubit unitaries.
#[derive(Clone, Debug)]
pub struct Matrix2x2 {
    pub data: [[C64; 2]; 2],
}

impl Matrix2x2 {
    /// Identity matrix.
    pub fn identity() -> Self {
        Matrix2x2 {
            data: [
                [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                [C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
            ],
        }
    }
}

/// Compact 4x4 complex matrix for fused two-qubit unitaries.
#[derive(Clone, Debug)]
pub struct Matrix4x4 {
    pub data: [[C64; 4]; 4],
}

impl Matrix4x4 {
    /// Identity matrix.
    pub fn identity() -> Self {
        let mut data = [[Complex64::new(0.0, 0.0); 4]; 4];
        for i in 0..4 {
            data[i][i] = Complex64::new(1.0, 0.0);
        }
        Matrix4x4 { data }
    }
}

/// A fused single-qubit gate: one 2x2 unitary replacing several gates.
#[derive(Clone, Debug)]
pub struct FusedGate {
    pub matrix: Matrix2x2,
    pub target: usize,
    pub fused_count: usize,
}

/// A fused two-qubit gate: one 4x4 unitary replacing a 2-qubit gate
/// with absorbed single-qubit neighbors.
#[derive(Clone, Debug)]
pub struct FusedTwoQubitGate {
    pub matrix: Matrix4x4,
    pub qubit_lo: usize,
    pub qubit_hi: usize,
    pub fused_count: usize,
}

/// A fused diagonal operation: multiple diagonal gates applied in one state vector pass.
/// Diagonal gates (Z, S, T, Rz, CZ, CR, CRz) commute with each other and can be
/// accumulated into compact phase factors, reducing O(n) state vector traversals to O(1).
#[derive(Clone, Debug)]
pub struct FusedDiagonal {
    /// Single-qubit phases: (qubit, phase_0, phase_1).
    /// For state |i⟩: multiply by phase_0 if bit(i,qubit)==0, or phase_1 if bit(i,qubit)==1.
    pub single_phases: Vec<(usize, C64, C64)>,
    /// Two-qubit controlled phases: (qubit_lo, qubit_hi, phase_11).
    /// For state |i⟩: if bit(i,lo)==1 AND bit(i,hi)==1, multiply by phase_11.
    pub pair_phases: Vec<(usize, usize, C64)>,
    /// Number of original gates fused into this diagonal.
    pub fused_count: usize,
}

/// Either a fused single-qubit gate, a fused two-qubit gate, a diagonal, or an original gate.
#[derive(Clone, Debug)]
pub enum FusedOrOriginal {
    Fused(FusedGate),
    FusedTwo(FusedTwoQubitGate),
    Diagonal(FusedDiagonal),
    Original(Gate),
}

/// Result of the fusion pass.
#[derive(Clone, Debug)]
pub struct FusionResult {
    pub gates: Vec<FusedOrOriginal>,
    pub original_count: usize,
    pub fused_count: usize,
    pub gates_eliminated: usize,
}

// ===================================================================
// MATRIX OPERATIONS (2x2)
// ===================================================================

/// Multiply two 2x2 complex matrices: result = B * A.
/// When gate A is applied first and B second, the combined unitary is B*A.
#[inline]
pub fn matrix_multiply_2x2(a: &Matrix2x2, b: &Matrix2x2) -> Matrix2x2 {
    let mut out = [[Complex64::new(0.0, 0.0); 2]; 2];
    for r in 0..2 {
        for c in 0..2 {
            out[r][c] = b.data[r][0] * a.data[0][c] + b.data[r][1] * a.data[1][c];
        }
    }
    Matrix2x2 { data: out }
}

/// Convert a single-qubit GateType to its 2x2 matrix representation.
///
/// Returns analytical closed-form matrices for standard gates without
/// any heap allocations. For U gate with custom parameters, we fall back
/// to the provided matrix directly.
///
/// Returns `None` for multi-qubit or unsupported gates.
pub fn gate_to_matrix(gate_type: &GateType) -> Option<Matrix2x2> {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

    let data = match gate_type {
        // Standard gates with closed-form matrices
        GateType::H => [
            [C64::new(inv_sqrt2, 0.0), C64::new(inv_sqrt2, 0.0)],
            [C64::new(inv_sqrt2, 0.0), C64::new(-inv_sqrt2, 0.0)],
        ],
        GateType::X => [
            [C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        ],
        GateType::Y => [
            [C64::new(0.0, 0.0), C64::new(0.0, -1.0)],
            [C64::new(0.0, 1.0), C64::new(0.0, 0.0)],
        ],
        GateType::Z => [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(-1.0, 0.0)],
        ],
        GateType::S => [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(0.0, 1.0)],
        ],
        GateType::T => {
            let sqrt2_2 = 2.0_f64.sqrt() / 2.0;
            [
                [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                [C64::new(0.0, 0.0), C64::new(sqrt2_2, sqrt2_2)],
            ]
        }
        GateType::Rx(theta) => {
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            [
                [C64::new(cos, 0.0), C64::new(0.0, -sin)],
                [C64::new(0.0, -sin), C64::new(cos, 0.0)],
            ]
        }
        GateType::Ry(theta) => {
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            [
                [C64::new(cos, 0.0), C64::new(-sin, 0.0)],
                [C64::new(sin, 0.0), C64::new(cos, 0.0)],
            ]
        }
        GateType::Rz(theta) => {
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            [
                [C64::new(cos, -sin), C64::new(0.0, 0.0)],
                [C64::new(0.0, 0.0), C64::new(cos, sin)],
            ]
        }
        // U gate: compute matrix from theta, phi, lambda parameters
        GateType::U { theta, phi, lambda } => {
            let cos_theta_2 = (theta / 2.0).cos();
            let sin_theta_2 = (theta / 2.0).sin();
            [
                [
                    C64::new(cos_theta_2, 0.0),
                    C64::new(-sin_theta_2 * lambda.cos(), -sin_theta_2 * lambda.sin()),
                ],
                [
                    C64::new(sin_theta_2 * phi.cos(), sin_theta_2 * phi.sin()),
                    C64::new(
                        cos_theta_2 * (phi + lambda).cos(),
                        cos_theta_2 * (phi + lambda).sin(),
                    ),
                ],
            ]
        }
        GateType::SX => [
            [C64::new(0.5, 0.5), C64::new(0.5, -0.5)],
            [C64::new(0.5, -0.5), C64::new(0.5, 0.5)],
        ],
        GateType::Phase(theta) => [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(theta.cos(), theta.sin())],
        ],
        // Multi-qubit gates not supported for 1Q fusion
        _ => return None,
    };

    Some(Matrix2x2 { data })
}

/// Check whether a 2x2 matrix is the identity (within epsilon tolerance).
pub fn is_identity_matrix(m: &Matrix2x2, eps: f64) -> bool {
    (m.data[0][0].re - 1.0).abs() < eps
        && m.data[0][0].im.abs() < eps
        && m.data[0][1].re.abs() < eps
        && m.data[0][1].im.abs() < eps
        && m.data[1][0].re.abs() < eps
        && m.data[1][0].im.abs() < eps
        && (m.data[1][1].re - 1.0).abs() < eps
        && m.data[1][1].im.abs() < eps
}

// ===================================================================
// MATRIX OPERATIONS (4x4)
// ===================================================================

/// Multiply two 4x4 complex matrices: result = B * A.
/// When gate A is applied first and B second, the combined unitary is B*A.
#[inline]
pub fn matrix_multiply_4x4(a: &Matrix4x4, b: &Matrix4x4) -> Matrix4x4 {
    let mut out = [[Complex64::new(0.0, 0.0); 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..4 {
                sum = sum + b.data[r][k] * a.data[k][c];
            }
            out[r][c] = sum;
        }
    }
    Matrix4x4 { data: out }
}

/// Kronecker (tensor) product of two 2x2 matrices: A_hi (x) B_lo -> 4x4 matrix.
///
/// A acts on the higher qubit, B acts on the lower qubit.
/// Index convention: row/col = (bit_hi << 1) | bit_lo,
/// so result[r][c] = a_hi[r>>1][c>>1] * b_lo[r&1][c&1].
#[inline]
pub fn kronecker_2x2(a_hi: &Matrix2x2, b_lo: &Matrix2x2) -> Matrix4x4 {
    let mut out = [[Complex64::new(0.0, 0.0); 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = a_hi.data[r >> 1][c >> 1] * b_lo.data[r & 1][c & 1];
        }
    }
    Matrix4x4 { data: out }
}

/// Check whether a 4x4 matrix is the identity (within epsilon tolerance).
pub fn is_identity_4x4(m: &Matrix4x4, eps: f64) -> bool {
    for r in 0..4 {
        for c in 0..4 {
            let expected_re = if r == c { 1.0 } else { 0.0 };
            if (m.data[r][c].re - expected_re).abs() > eps {
                return false;
            }
            if m.data[r][c].im.abs() > eps {
                return false;
            }
        }
    }
    true
}

/// Build a 4x4 controlled-U matrix given remapped control/target (0 or 1)
/// and the 2x2 unitary elements U = [[u00, u01], [u10, u11]].
///
/// When control is active (bit=1), U is applied to the target qubit.
/// Index convention: state = (bit_hi << 1) | bit_lo.
#[inline]
fn controlled_u_matrix(
    control: usize,
    target: usize,
    u00: C64,
    u01: C64,
    u10: C64,
    u11: C64,
) -> [[C64; 4]; 4] {
    let zero = C64::new(0.0, 0.0);
    let one = C64::new(1.0, 0.0);

    if control == 1 && target == 0 {
        // control=hi, target=lo
        [
            [one, zero, zero, zero],
            [zero, one, zero, zero],
            [zero, zero, u00, u01],
            [zero, zero, u10, u11],
        ]
    } else {
        // control=lo, target=hi
        [
            [one, zero, zero, zero],
            [zero, u00, zero, u01],
            [zero, zero, one, zero],
            [zero, u10, zero, u11],
        ]
    }
}

/// Extract a 4x4 matrix from a 2-qubit gate using analytical forms.
///
/// Returns standard matrices for CNOT, CZ, SWAP without any heap allocations.
/// The lower-indexed original qubit maps to qubit 0 (bit position 0).
///
/// Returns `None` if the gate is not a supported 2-qubit gate.
pub fn two_qubit_gate_to_matrix(gate: &Gate) -> Option<Matrix4x4> {
    // Collect all qubits involved
    let mut qubits: Vec<usize> = gate
        .targets
        .iter()
        .chain(gate.controls.iter())
        .copied()
        .collect();
    qubits.sort();
    qubits.dedup();

    if qubits.len() != 2 {
        return None;
    }

    let qa = qubits[0]; // lower-indexed -> maps to qubit 0
    let qb = qubits[1]; // higher-indexed -> maps to qubit 1

    // Build a remapped gate: qa -> 0, qb -> 1
    let remap = |q: usize| -> usize {
        if q == qa {
            0
        } else if q == qb {
            1
        } else {
            q
        }
    };

    let remapped_targets: Vec<usize> = gate.targets.iter().map(|&q| remap(q)).collect();
    let remapped_controls: Vec<usize> = gate.controls.iter().map(|&q| remap(q)).collect();

    // Analytical matrices for standard 2-qubit gates
    let zero = C64::new(0.0, 0.0);
    let one = C64::new(1.0, 0.0);
    let mone = C64::new(-1.0, 0.0);

    let control = if !remapped_controls.is_empty() {
        remapped_controls[0]
    } else {
        0
    };
    let target = if !remapped_targets.is_empty() {
        remapped_targets[0]
    } else {
        0
    };

    let data = match &gate.gate_type {
        GateType::CNOT => {
            if control == 1 && target == 0 {
                // Standard CNOT: control on qubit 1, target on qubit 0
                [
                    [one, zero, zero, zero],
                    [zero, one, zero, zero],
                    [zero, zero, zero, one],
                    [zero, zero, one, zero],
                ]
            } else {
                // CNOT(control=0, target=1)
                [
                    [one, zero, zero, zero],
                    [zero, zero, zero, one],
                    [zero, zero, one, zero],
                    [zero, one, zero, zero],
                ]
            }
        }
        GateType::CZ => {
            // CZ is symmetric: applies -1 to |11> only
            [
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, one, zero],
                [zero, zero, zero, mone],
            ]
        }
        GateType::SWAP => [
            [one, zero, zero, zero],
            [zero, zero, one, zero],
            [zero, one, zero, zero],
            [zero, zero, zero, one],
        ],
        GateType::CR(angle) => {
            // Controlled phase: |11> gets phase e^{i*angle}, rest unchanged
            let phase = C64::new(angle.cos(), angle.sin());
            [
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, one, zero],
                [zero, zero, zero, phase],
            ]
        }
        // CRz: GateOperations::crz has inverted sign convention vs standard.
        // Skip matrix extraction; goes through apply_gate_to_state.
        GateType::CRz(_) => return None,
        // CRx/CRy: GateOperations::crx/cry implementations don't match
        // the standard matrix definitions in GateType, so we skip matrix
        // extraction for these and let them go through apply_gate_to_state.
        GateType::CRx(_) | GateType::CRy(_) => return None,
        GateType::ISWAP => {
            let i_val = C64::new(0.0, 1.0);
            [
                [one, zero, zero, zero],
                [zero, zero, i_val, zero],
                [zero, i_val, zero, zero],
                [zero, zero, zero, one],
            ]
        }
        _ => return None,
    };

    Some(Matrix4x4 { data })
}

// ===================================================================
// FUSION ALGORITHM
// ===================================================================

/// Returns true if this Gate is a single-qubit gate that can be fused.
fn is_fusable_single_qubit(gate: &Gate) -> bool {
    if !gate.controls.is_empty() {
        return false;
    }
    if gate.targets.len() != 1 {
        return false;
    }
    matches!(
        gate.gate_type,
        GateType::H
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::S
            | GateType::T
            | GateType::Rx(_)
            | GateType::Ry(_)
            | GateType::Rz(_)
            | GateType::U { .. }
            | GateType::SX
            | GateType::Phase(_)
    )
}

/// Collect all qubits touched by a gate (targets + controls).
fn gate_qubits(gate: &Gate) -> Vec<usize> {
    gate.targets
        .iter()
        .chain(gate.controls.iter())
        .copied()
        .collect()
}

/// Flush a single qubit's accumulated matrix into the output.
/// When only a single gate was accumulated (count=1), emit it as Original
/// so it can use the specialized SIMD dispatch (H→hadamard, Rz→diagonal, etc.)
/// instead of the generic u() path.
fn flush_qubit(
    qubit: usize,
    accum: &mut HashMap<usize, (Matrix2x2, usize, Option<Gate>)>,
    output: &mut Vec<FusedOrOriginal>,
    eliminated: &mut usize,
) {
    if let Some((matrix, count, orig_gate)) = accum.remove(&qubit) {
        if count == 0 {
            return;
        }
        if is_identity_matrix(&matrix, 1e-12) {
            *eliminated += count;
            return;
        }
        if count == 1 {
            if let Some(gate) = orig_gate {
                // Single gate: emit as Original for specialized dispatch
                output.push(FusedOrOriginal::Original(gate));
                return;
            }
        }
        // Multiple fused gates: emit as Fused
        output.push(FusedOrOriginal::Fused(FusedGate {
            matrix,
            target: qubit,
            fused_count: count,
        }));
        *eliminated += count.saturating_sub(1);
    }
}

/// Extract a pending accumulator entry for a qubit, returning the matrix and count.
/// If the qubit has no pending accumulator, returns None.
fn take_pending(
    qubit: usize,
    accum: &mut HashMap<usize, (Matrix2x2, usize, Option<Gate>)>,
) -> Option<(Matrix2x2, usize)> {
    accum.remove(&qubit).map(|(m, c, _)| (m, c))
}

/// Main fusion algorithm.
///
/// Walks the gate list, accumulating per-qubit 2x2 matrices for consecutive
/// single-qubit gates. When a 2-qubit gate is encountered, pending single-qubit
/// matrices on those qubits are absorbed into a fused 4x4 unitary — but only
/// when the circuit is small enough that the generic u2() path is beneficial.
/// For circuits with >=16 qubits, 2-qubit absorption is skipped because the
/// specialized gate dispatch (parallel CNOT, etc.) outperforms generic u2().
/// When a 3+ qubit gate is encountered, all qubits it touches are flushed first.
///
/// Single-qubit fusion (consecutive 1Q gates → fused unitary) is always enabled
/// regardless of circuit size.
///
/// With the GPU 4x4 unitary kernel, fused 2-qubit gates are efficient at any
/// practical circuit size, so we set this very high.
const TWO_QUBIT_FUSION_THRESHOLD: usize = 256;

pub fn fuse_gates(gates: &[Gate]) -> FusionResult {
    let original_count = gates.len();
    let mut accum: HashMap<usize, (Matrix2x2, usize, Option<Gate>)> = HashMap::new();
    let mut output: Vec<FusedOrOriginal> = Vec::with_capacity(gates.len());
    let mut eliminated: usize = 0;

    // Infer circuit size from max qubit index
    let num_qubits = gates
        .iter()
        .flat_map(|g| gate_qubits(g))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let enable_2q_fusion = num_qubits < TWO_QUBIT_FUSION_THRESHOLD;

    for gate in gates {
        if is_fusable_single_qubit(gate) {
            let qubit = gate.targets[0];
            let mat = gate_to_matrix(&gate.gate_type).expect("fusable gate must have 2x2 matrix");

            let entry = accum
                .entry(qubit)
                .or_insert_with(|| (Matrix2x2::identity(), 0, None));
            entry.0 = matrix_multiply_2x2(&entry.0, &mat);
            entry.1 += 1;
            // Track original gate for count=1 → specialized dispatch
            if entry.1 == 1 {
                entry.2 = Some(gate.clone());
            } else {
                entry.2 = None; // Multiple gates, can't use specialized dispatch
            }
        } else {
            let qubits = gate_qubits(gate);
            let mut unique_qubits = qubits.clone();
            unique_qubits.sort();
            unique_qubits.dedup();
            let num_qubits = unique_qubits.len();

            if num_qubits == 2 {
                if enable_2q_fusion {
                    // 2-qubit gate: attempt absorption of pending single-qubit gates
                    if let Some(gate_4x4) = two_qubit_gate_to_matrix(gate) {
                        let q_lo = unique_qubits[0];
                        let q_hi = unique_qubits[1];

                        let pending_lo = take_pending(q_lo, &mut accum);
                        let pending_hi = take_pending(q_hi, &mut accum);

                        let mat_lo = pending_lo
                            .as_ref()
                            .map(|(m, _)| m.clone())
                            .unwrap_or_else(Matrix2x2::identity);
                        let mat_hi = pending_hi
                            .as_ref()
                            .map(|(m, _)| m.clone())
                            .unwrap_or_else(Matrix2x2::identity);

                        let count_lo = pending_lo.as_ref().map(|(_, c)| *c).unwrap_or(0);
                        let count_hi = pending_hi.as_ref().map(|(_, c)| *c).unwrap_or(0);

                        if count_lo == 0 && count_hi == 0 {
                            // No pending gates — use specialized dispatch
                            output.push(FusedOrOriginal::Original(gate.clone()));
                        } else {
                            let pre = kronecker_2x2(&mat_hi, &mat_lo);
                            let absorbed = matrix_multiply_4x4(&pre, &gate_4x4);

                            let fused_count = 1 + count_lo + count_hi;
                            eliminated += count_lo + count_hi;

                            output.push(FusedOrOriginal::FusedTwo(FusedTwoQubitGate {
                                matrix: absorbed,
                                qubit_lo: q_lo,
                                qubit_hi: q_hi,
                                fused_count,
                            }));
                        }
                    } else {
                        // Can't extract matrix — flush + Original
                        for &q in &unique_qubits {
                            flush_qubit(q, &mut accum, &mut output, &mut eliminated);
                        }
                        output.push(FusedOrOriginal::Original(gate.clone()));
                    }
                } else {
                    // Large circuit: skip 2-qubit fusion, flush pending as separate
                    // Fused1Q gates, emit 2-qubit gate as Original for specialized dispatch
                    for &q in &unique_qubits {
                        flush_qubit(q, &mut accum, &mut output, &mut eliminated);
                    }
                    output.push(FusedOrOriginal::Original(gate.clone()));
                }
            } else {
                // 3+ qubit gate: flush all qubits it touches, emit Original
                for &q in &unique_qubits {
                    flush_qubit(q, &mut accum, &mut output, &mut eliminated);
                }
                output.push(FusedOrOriginal::Original(gate.clone()));
            }
        }
    }

    // Flush all remaining accumulators
    let remaining_qubits: Vec<usize> = accum.keys().copied().collect();
    for q in remaining_qubits {
        flush_qubit(q, &mut accum, &mut output, &mut eliminated);
    }

    // Second pass: merge consecutive diagonal gates into FusedDiagonal blocks
    let output = fuse_diagonal_pass(&output);

    let fused_count = output
        .iter()
        .filter(|g| {
            matches!(
                g,
                FusedOrOriginal::Fused(_)
                    | FusedOrOriginal::FusedTwo(_)
                    | FusedOrOriginal::Diagonal(_)
            )
        })
        .count();

    FusionResult {
        gates: output,
        original_count,
        fused_count,
        gates_eliminated: eliminated,
    }
}

// ===================================================================
// DIAGONAL GATE FUSION
// ===================================================================

/// Returns true if this gate is a diagonal single-qubit gate.
fn is_diagonal_1q(gate_type: &GateType) -> bool {
    matches!(
        gate_type,
        GateType::Z | GateType::S | GateType::T | GateType::Rz(_) | GateType::Phase(_)
    )
}

/// Returns true if this gate is a diagonal two-qubit gate (controlled-phase).
fn is_diagonal_2q(gate_type: &GateType) -> bool {
    matches!(gate_type, GateType::CZ | GateType::CR(_))
}

/// Returns true if this FusedOrOriginal is a diagonal gate that can be merged.
fn is_diagonal_item(item: &FusedOrOriginal) -> bool {
    match item {
        FusedOrOriginal::Original(gate) => {
            if gate.targets.len() == 1 && gate.controls.is_empty() {
                is_diagonal_1q(&gate.gate_type)
            } else {
                is_diagonal_2q(&gate.gate_type)
            }
        }
        // Fused/FusedTwo matrices may or may not be diagonal; skip them
        _ => false,
    }
}

/// Extract phase factors for a single-qubit diagonal gate.
/// Returns (phase_0, phase_1) where the gate applies phase_0 to |0⟩ and phase_1 to |1⟩.
fn diagonal_1q_phases(gate_type: &GateType) -> (C64, C64) {
    match gate_type {
        GateType::Z => (C64::new(1.0, 0.0), C64::new(-1.0, 0.0)),
        GateType::S => (C64::new(1.0, 0.0), C64::new(0.0, 1.0)),
        GateType::T => {
            let sqrt2_2 = 2.0_f64.sqrt() / 2.0;
            (C64::new(1.0, 0.0), C64::new(sqrt2_2, sqrt2_2))
        }
        GateType::Rz(theta) => {
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            (C64::new(cos, -sin), C64::new(cos, sin))
        }
        GateType::Phase(theta) => (C64::new(1.0, 0.0), C64::new(theta.cos(), theta.sin())),
        _ => (C64::new(1.0, 0.0), C64::new(1.0, 0.0)),
    }
}

/// Extract the controlled phase for a 2-qubit diagonal gate.
/// Returns the phase applied to |11⟩ (both qubits set).
/// For CRz, also returns per-qubit phases on the target.
fn diagonal_2q_phases(gate_type: &GateType) -> (C64, Option<(C64, C64)>) {
    match gate_type {
        GateType::CZ => (C64::new(-1.0, 0.0), None),
        GateType::CR(angle) => (C64::new(angle.cos(), angle.sin()), None),
        GateType::CRz(theta) => {
            // CRz applies Rz(theta) to target when control=1
            // This is diag(1, 1, e^{-itheta/2}, e^{itheta/2}) for control=hi
            // We handle it as: pair_phase = e^{itheta/2} for |11⟩,
            // plus single_phase = e^{-itheta/2} on target when control=1.
            // Actually, CRz is: when control=1, apply [e^{-iθ/2}, 0; 0, e^{iθ/2}] to target.
            // In terms of phases:
            //   |00⟩: 1, |01⟩: 1, |10⟩: e^{-iθ/2}, |11⟩: e^{iθ/2}
            // (assuming control=hi, target=lo)
            // This factors as: phase on target_qubit when control=1:
            //   target=0: e^{-iθ/2}, target=1: e^{iθ/2}
            // But we can't represent "conditional single-qubit phase" in our model.
            // Instead, represent as full diagonal:
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            let p0 = C64::new(cos, -sin);
            let p1 = C64::new(cos, sin);
            // pair_phase for |11⟩ = p1, and we need target phase for |10⟩ = p0
            // We'll handle CRz specially: return pair_phase and target_phases
            (p1, Some((p0, p1)))
        }
        _ => (C64::new(1.0, 0.0), None),
    }
}

/// Post-fusion pass: merge consecutive diagonal gates into FusedDiagonal blocks.
///
/// Walks through the fused gate list and accumulates runs of diagonal-compatible
/// gates (Z, S, T, Rz, CZ, CR, CRz) into compact FusedDiagonal representations.
/// This reduces state vector traversals from O(diagonal_gates) to O(diagonal_runs).
pub fn fuse_diagonal_pass(gates: &[FusedOrOriginal]) -> Vec<FusedOrOriginal> {
    if gates.is_empty() {
        return Vec::new();
    }

    let mut output: Vec<FusedOrOriginal> = Vec::with_capacity(gates.len());
    // Accumulators for the current diagonal run
    let mut diag_1q: HashMap<usize, (C64, C64)> = HashMap::new(); // qubit -> (phase_0, phase_1)
    let mut diag_2q: HashMap<(usize, usize), C64> = HashMap::new(); // (lo, hi) -> phase_11
    let mut diag_count: usize = 0;

    let flush_diag = |output: &mut Vec<FusedOrOriginal>,
                      diag_1q: &mut HashMap<usize, (C64, C64)>,
                      diag_2q: &mut HashMap<(usize, usize), C64>,
                      diag_count: &mut usize| {
        if *diag_count < 2 {
            // Not worth fusing a single diagonal gate
            return false;
        }
        let single_phases: Vec<(usize, C64, C64)> =
            diag_1q.drain().map(|(q, (p0, p1))| (q, p0, p1)).collect();
        let pair_phases: Vec<(usize, usize, C64)> =
            diag_2q.drain().map(|((lo, hi), p)| (lo, hi, p)).collect();
        output.push(FusedOrOriginal::Diagonal(FusedDiagonal {
            single_phases,
            pair_phases,
            fused_count: *diag_count,
        }));
        *diag_count = 0;
        true
    };

    for item in gates {
        if is_diagonal_item(item) {
            if let FusedOrOriginal::Original(gate) = item {
                if gate.targets.len() == 1 && gate.controls.is_empty() {
                    // Single-qubit diagonal
                    let q = gate.targets[0];
                    let (p0, p1) = diagonal_1q_phases(&gate.gate_type);
                    let entry = diag_1q
                        .entry(q)
                        .or_insert((C64::new(1.0, 0.0), C64::new(1.0, 0.0)));
                    entry.0 = entry.0 * p0;
                    entry.1 = entry.1 * p1;
                    diag_count += 1;
                } else {
                    // Two-qubit diagonal (CZ, CR, CRz)
                    let mut qubits: Vec<usize> = gate
                        .targets
                        .iter()
                        .chain(gate.controls.iter())
                        .copied()
                        .collect();
                    qubits.sort();
                    qubits.dedup();
                    if qubits.len() == 2 {
                        let lo = qubits[0];
                        let hi = qubits[1];
                        // CZ and CR are pure pair-phase diagonal gates
                        let (phase_11, _) = diagonal_2q_phases(&gate.gate_type);
                        let entry = diag_2q.entry((lo, hi)).or_insert(C64::new(1.0, 0.0));
                        *entry = *entry * phase_11;
                        diag_count += 1;
                    } else {
                        output.push(item.clone());
                    }
                }
            }
        } else {
            // Non-diagonal gate: flush diagonal accumulator
            if diag_count >= 2 {
                flush_diag(&mut output, &mut diag_1q, &mut diag_2q, &mut diag_count);
            } else {
                drain_diag_singles(&mut diag_1q, &mut diag_2q, &mut diag_count, &mut output);
            }
            output.push(item.clone());
        }
    }

    // Flush remaining diagonal accumulator
    if diag_count >= 2 {
        flush_diag(&mut output, &mut diag_1q, &mut diag_2q, &mut diag_count);
    } else {
        drain_diag_singles(&mut diag_1q, &mut diag_2q, &mut diag_count, &mut output);
    }

    output
}

/// Helper: drain single accumulated diagonal gates back as Original gates (when run is too short).
fn drain_diag_singles(
    diag_1q: &mut HashMap<usize, (C64, C64)>,
    diag_2q: &mut HashMap<(usize, usize), C64>,
    count: &mut usize,
    output: &mut Vec<FusedOrOriginal>,
) {
    // Re-emit as individual diagonal gates. Since we've accumulated phases,
    // emit them as Fused single-qubit gates with the diagonal matrix.
    for (q, (p0, p1)) in diag_1q.drain() {
        let matrix = Matrix2x2 {
            data: [
                [p0, Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), p1],
            ],
        };
        if is_identity_matrix(&matrix, 1e-12) {
            continue;
        }
        output.push(FusedOrOriginal::Fused(FusedGate {
            matrix,
            target: q,
            fused_count: 1,
        }));
    }
    // For 2Q diagonal singles, emit as Original. But we've lost the original gate info.
    // Since run length < 2 means at most 1 gate, we'd need to reconstruct.
    // This case is rare (single CZ/CR in isolation). For safety, emit a FusedTwo with diagonal matrix.
    for ((lo, hi), phase) in diag_2q.drain() {
        let one = C64::new(1.0, 0.0);
        let zero = C64::new(0.0, 0.0);
        let matrix = Matrix4x4 {
            data: [
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, one, zero],
                [zero, zero, zero, phase],
            ],
        };
        if is_identity_4x4(&matrix, 1e-12) {
            continue;
        }
        output.push(FusedOrOriginal::FusedTwo(FusedTwoQubitGate {
            matrix,
            qubit_lo: lo,
            qubit_hi: hi,
            fused_count: 1,
        }));
    }
    *count = 0;
}

/// Execute a FusedDiagonal on a quantum state with phase lookup optimization.
///
/// S-Tier fix (2026-02-10): Fixed regression where QFT-20 Fused was 131ms.
/// The issue was scalar sequential loop over all amplitudes with N inner multiplications.
///
/// Optimization: Precompute phase lookup table, then apply with parallel iteration.
/// This turns O(dim * num_phases) → O(2^k + dim) where k = number of diagonal qubits.
fn execute_diagonal(state: &mut QuantumState, diag: &FusedDiagonal) {
    let amps = state.amplitudes_mut();
    let _dim = amps.len();

    // Early exit for empty diagonal
    if diag.single_phases.is_empty() && diag.pair_phases.is_empty() {
        return;
    }

    // Step 1: Collect all diagonal qubits and sort them
    let mut diag_qubits: Vec<usize> = diag.single_phases.iter().map(|(q, _, _)| *q).collect();
    for &(lo, hi, _) in &diag.pair_phases {
        if !diag_qubits.contains(&lo) {
            diag_qubits.push(lo);
        }
        if !diag_qubits.contains(&hi) {
            diag_qubits.push(hi);
        }
    }
    diag_qubits.sort();
    diag_qubits.dedup();

    let num_diag_qubits = diag_qubits.len();

    // Step 2: Build phase lookup table
    // For small diagonal circuits (<= 16 qubits), precompute all 2^k phase combinations
    // For larger circuits, fall back to the original scalar loop (rare case)
    if num_diag_qubits <= 16 {
        let lookup_size = 1usize << num_diag_qubits;
        let mut phase_lookup: Vec<C64> = Vec::with_capacity(lookup_size);

        // Precompute phase for each bit pattern
        for pattern in 0..lookup_size {
            let mut phase = C64::new(1.0, 0.0);

            // Apply single-qubit phases
            for &(q, p0, p1) in &diag.single_phases {
                let bit_pos = diag_qubits.iter().position(|&dq| dq == q).unwrap();
                if (pattern >> bit_pos) & 1 == 1 {
                    phase = phase * p1;
                } else {
                    phase = phase * p0;
                }
            }

            // Apply pair phases
            for &(lo, hi, p11) in &diag.pair_phases {
                let lo_pos = diag_qubits.iter().position(|&dq| dq == lo).unwrap();
                let hi_pos = diag_qubits.iter().position(|&dq| dq == hi).unwrap();
                if ((pattern >> lo_pos) & 1 == 1) && ((pattern >> hi_pos) & 1 == 1) {
                    phase = phase * p11;
                }
            }

            phase_lookup.push(phase);
        }

        // Step 3: Parallel phase application with lookup table
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
                // Extract relevant bits for lookup index
                let mut lookup_idx = 0;
                for (bit_pos, &qubit) in diag_qubits.iter().enumerate() {
                    if (i >> qubit) & 1 == 1 {
                        lookup_idx |= 1 << bit_pos;
                    }
                }
                *amp = *amp * phase_lookup[lookup_idx];
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..dim {
                let mut lookup_idx = 0;
                for (bit_pos, &qubit) in diag_qubits.iter().enumerate() {
                    if (i >> qubit) & 1 == 1 {
                        lookup_idx |= 1 << bit_pos;
                    }
                }
                amps[i] = amps[i] * phase_lookup[lookup_idx];
            }
        }
    } else {
        // Fallback: too many diagonal qubits (>16), use original scalar loop
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            amps.par_iter_mut().enumerate().for_each(|(i, amp)| {
                let mut phase = C64::new(1.0, 0.0);
                for &(q, p0, p1) in &diag.single_phases {
                    if (i >> q) & 1 == 1 {
                        phase = phase * p1;
                    } else {
                        phase = phase * p0;
                    }
                }
                for &(lo, hi, p11) in &diag.pair_phases {
                    if ((i >> lo) & 1 == 1) && ((i >> hi) & 1 == 1) {
                        phase = phase * p11;
                    }
                }
                *amp = *amp * phase;
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..dim {
                let mut phase = C64::new(1.0, 0.0);
                for &(q, p0, p1) in &diag.single_phases {
                    if (i >> q) & 1 == 1 {
                        phase = phase * p1;
                    } else {
                        phase = phase * p0;
                    }
                }
                for &(lo, hi, p11) in &diag.pair_phases {
                    if ((i >> lo) & 1 == 1) && ((i >> hi) & 1 == 1) {
                        phase = phase * p11;
                    }
                }
                amps[i] = amps[i] * phase;
            }
        }
    }
}

// ===================================================================
// LAYER DECOMPOSITION (Phase 5A)
// ===================================================================

/// Decompose a fused circuit result into layers where each layer contains
/// gates operating on disjoint qubits. This reduces dispatch overhead by
/// grouping non-conflicting gates.
///
/// For FusedOrOriginal items, we extract the qubits they touch.
/// Single-qubit fused gates touch only their target.
/// Two-qubit fused gates touch their two qubits.
/// Original gates can have arbitrary targets/controls.
///
/// Returns a Vec of Vec<FusedOrOriginal> where each inner Vec is one layer.
pub fn decompose_fused_layers(fused_gates: &[FusedOrOriginal]) -> Vec<Vec<FusedOrOriginal>> {
    if fused_gates.is_empty() {
        return Vec::new();
    }

    let mut layers = Vec::new();
    let mut current_layer_qubits: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    let mut current_layer: Vec<FusedOrOriginal> = Vec::new();

    for item in fused_gates {
        // Get qubits touched by this item
        let item_qubits = match item {
            FusedOrOriginal::Fused(fg) => {
                let mut set = std::collections::HashSet::new();
                set.insert(fg.target);
                set
            }
            FusedOrOriginal::FusedTwo(fg) => {
                let mut set = std::collections::HashSet::new();
                set.insert(fg.qubit_lo);
                set.insert(fg.qubit_hi);
                set
            }
            FusedOrOriginal::Diagonal(diag) => {
                let mut set = std::collections::HashSet::new();
                for &(q, _, _) in &diag.single_phases {
                    set.insert(q);
                }
                for &(lo, hi, _) in &diag.pair_phases {
                    set.insert(lo);
                    set.insert(hi);
                }
                set
            }
            FusedOrOriginal::Original(gate) => gate
                .targets
                .iter()
                .chain(gate.controls.iter())
                .copied()
                .collect(),
        };

        // Check if this item conflicts with any item in current layer
        let conflicts = current_layer_qubits.intersection(&item_qubits).count() > 0;

        if conflicts {
            // Start a new layer
            layers.push(std::mem::take(&mut current_layer));
            current_layer_qubits = item_qubits;
            current_layer.push(item.clone());
        } else {
            // Add to current layer
            current_layer_qubits = current_layer_qubits.union(&item_qubits).copied().collect();
            current_layer.push(item.clone());
        }
    }

    // Don't forget the last layer
    if !current_layer.is_empty() {
        layers.push(current_layer);
    }

    layers
}

/// Execute a layer of fused gates on a quantum state.
/// All gates in a layer operate on disjoint qubits and can be applied
/// in any order (we apply sequentially but they could be parallelized).
fn execute_fused_layer(state: &mut QuantumState, layer: &[FusedOrOriginal]) {
    for item in layer {
        match item {
            FusedOrOriginal::Fused(fg) => {
                GateOperations::u(state, fg.target, &fg.matrix.data);
            }
            FusedOrOriginal::FusedTwo(fg) => {
                GateOperations::u2(state, fg.qubit_lo, fg.qubit_hi, &fg.matrix.data);
            }
            FusedOrOriginal::Diagonal(diag) => {
                execute_diagonal(state, diag);
            }
            FusedOrOriginal::Original(gate) => {
                apply_gate_to_state(state, gate);
            }
        }
    }
}

/// Execute a fused circuit using layer decomposition.
/// This reduces dispatch overhead by grouping non-conflicting gates into layers.
pub fn execute_fused_circuit_with_layers(state: &mut QuantumState, fusion: &FusionResult) {
    let layers = decompose_fused_layers(&fusion.gates);

    for layer in layers {
        execute_fused_layer(state, &layer);
    }
}

// ===================================================================
// EXECUTION
// ===================================================================

/// Execute a fused circuit on a quantum state.
///
/// Uses layer decomposition to group non-conflicting gates and reduce
/// dispatch overhead. Fused single-qubit gates are dispatched via
/// `GateOperations::u()`. Fused two-qubit gates are dispatched via
/// `GateOperations::u2()`. Original (unfused) gates are dispatched via
/// `apply_gate_to_state()`.
pub fn execute_fused_circuit(state: &mut QuantumState, fusion: &FusionResult) {
    execute_fused_circuit_with_layers(state, fusion);
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{Gate, GateType};

    /// Helper: check two 2x2 matrices are approximately equal.
    fn matrices_approx_eq(a: &Matrix2x2, b: &Matrix2x2, eps: f64) -> bool {
        for r in 0..2 {
            for c in 0..2 {
                if (a.data[r][c].re - b.data[r][c].re).abs() > eps {
                    return false;
                }
                if (a.data[r][c].im - b.data[r][c].im).abs() > eps {
                    return false;
                }
            }
        }
        true
    }

    /// Helper: check two 4x4 matrices are approximately equal.
    fn matrices_4x4_approx_eq(a: &Matrix4x4, b: &Matrix4x4, eps: f64) -> bool {
        for r in 0..4 {
            for c in 0..4 {
                if (a.data[r][c].re - b.data[r][c].re).abs() > eps {
                    return false;
                }
                if (a.data[r][c].im - b.data[r][c].im).abs() > eps {
                    return false;
                }
            }
        }
        true
    }

    // ---------------------------------------------------------------
    // Existing single-qubit fusion tests
    // ---------------------------------------------------------------

    #[test]
    fn test_h_h_cancels_to_identity() {
        let gates = vec![Gate::h(0), Gate::h(0)];
        let result = fuse_gates(&gates);
        // H*H = I, so should be eliminated entirely
        assert_eq!(result.gates_eliminated, 2);
        assert_eq!(result.fused_count, 0);
    }

    #[test]
    fn test_rx_merge() {
        let gates = vec![Gate::rx(0, 0.3), Gate::rx(0, 0.5)];
        let result = fuse_gates(&gates);
        assert_eq!(result.fused_count, 1);

        // The fused matrix should equal Rx(0.8)
        if let FusedOrOriginal::Fused(fg) = &result.gates[0] {
            let expected = gate_to_matrix(&GateType::Rx(0.8)).unwrap();
            assert!(
                matrices_approx_eq(&fg.matrix, &expected, 1e-10),
                "Rx(0.3)*Rx(0.5) should equal Rx(0.8)"
            );
            assert_eq!(fg.fused_count, 2);
        } else {
            panic!("Expected a fused gate");
        }
    }

    #[test]
    fn test_multi_qubit_flushes_correctly() {
        // H(0), H(1), CNOT(0,1), H(0)
        // With 2-qubit absorption: H(0) and H(1) get absorbed into CNOT
        // producing FusedTwo, then H(0) remains as a trailing Fused.
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(0, 1), Gate::h(0)];
        let result = fuse_gates(&gates);

        let mut fused_one = 0;
        let mut fused_two = 0;
        let mut originals = 0;
        for g in &result.gates {
            match g {
                FusedOrOriginal::Fused(_) => fused_one += 1,
                FusedOrOriginal::FusedTwo(_) => fused_two += 1,
                FusedOrOriginal::Diagonal(_) => fused_one += 1,
                FusedOrOriginal::Original(_) => originals += 1,
            }
        }
        assert_eq!(
            originals, 1,
            "trailing H(0) should be emitted as Original for specialized dispatch"
        );
        assert_eq!(
            fused_two, 1,
            "Should have 1 FusedTwo (CNOT with absorbed H gates)"
        );
        assert_eq!(
            fused_one, 0,
            "No single-gate Fused (count=1 emits as Original)"
        );
    }

    #[test]
    fn test_fidelity_qft_like() {
        // Build a small QFT-like circuit on 4 qubits
        let mut gates = Vec::new();
        let n = 4;
        for i in 0..n {
            gates.push(Gate::h(i));
            for j in (i + 1)..n {
                let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
                gates.push(Gate::new(GateType::CR(angle), vec![j], vec![i]));
            }
        }

        // Execute unfused
        let mut state_unfused = QuantumState::new(n);
        for gate in &gates {
            apply_gate_to_state(&mut state_unfused, gate);
        }

        // Execute fused
        let fusion = fuse_gates(&gates);
        let mut state_fused = QuantumState::new(n);
        execute_fused_circuit(&mut state_fused, &fusion);

        // Fidelity check
        let fidelity = state_fused.fidelity(&state_unfused);
        assert!(
            fidelity > 1.0 - 1e-10,
            "Fused vs unfused fidelity too low: {}",
            fidelity
        );
    }

    #[test]
    fn test_x_x_cancels() {
        let gates = vec![Gate::x(0), Gate::x(0)];
        let result = fuse_gates(&gates);
        assert_eq!(result.gates_eliminated, 2);
    }

    #[test]
    fn test_mixed_qubits_fuse_independently() {
        // H(0), Rx(1, 0.5), H(0), Rx(1, 0.3)
        let gates = vec![Gate::h(0), Gate::rx(1, 0.5), Gate::h(0), Gate::rx(1, 0.3)];
        let result = fuse_gates(&gates);
        // H*H on qubit 0 -> identity (eliminated)
        // Rx(0.5)*Rx(0.3) on qubit 1 -> Rx(0.8) (1 fused gate)
        assert!(result.gates_eliminated >= 2, "H*H should cancel");
        assert!(result.fused_count >= 1, "Rx should fuse");
    }

    #[test]
    fn test_single_gate_preserved() {
        let gates = vec![Gate::h(0)];
        let result = fuse_gates(&gates);
        assert_eq!(result.gates.len(), 1);
        // Single gate emitted as Original for specialized SIMD dispatch
        assert!(matches!(&result.gates[0], FusedOrOriginal::Original(_)));
    }

    #[test]
    fn test_empty_circuit() {
        let gates: Vec<Gate> = vec![];
        let result = fuse_gates(&gates);
        assert_eq!(result.gates.len(), 0);
        assert_eq!(result.original_count, 0);
        assert_eq!(result.gates_eliminated, 0);
    }

    #[test]
    fn test_matrix_multiply_identity() {
        let id = Matrix2x2::identity();
        let h_mat = gate_to_matrix(&GateType::H).unwrap();
        let result = matrix_multiply_2x2(&h_mat, &id);
        assert!(matrices_approx_eq(&result, &h_mat, 1e-15));
    }

    #[test]
    fn test_fused_execution_correctness() {
        // Verify fused execution matches gate-by-gate on a 3-qubit circuit
        let gates = vec![
            Gate::h(0),
            Gate::rx(0, 1.2),
            Gate::rz(0, 0.7),
            Gate::h(1),
            Gate::ry(1, 0.5),
            Gate::cnot(0, 1),
            Gate::t(2),
            Gate::s(2),
            Gate::h(2),
        ];

        let mut state_ref = QuantumState::new(3);
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }

        let fusion = fuse_gates(&gates);
        let mut state_fused = QuantumState::new(3);
        execute_fused_circuit(&mut state_fused, &fusion);

        let fidelity = state_fused.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-12,
            "Fused execution fidelity: {}",
            fidelity
        );
    }

    // ---------------------------------------------------------------
    // 2-qubit fusion tests
    // ---------------------------------------------------------------

    #[test]
    fn test_h_absorbed_into_cnot() {
        // H(0), CNOT(0,1) -> should produce 1 FusedTwo
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let result = fuse_gates(&gates);

        let fused_two_count = result
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::FusedTwo(_)))
            .count();
        assert_eq!(fused_two_count, 1, "Should produce 1 FusedTwo");

        let fused_one_count = result
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::Fused(_)))
            .count();
        let original_count = result
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::Original(_)))
            .count();
        assert_eq!(fused_one_count, 0, "No standalone Fused gates");
        assert_eq!(original_count, 0, "No Original gates");

        assert_eq!(result.gates.len(), 1, "Total output should be 1 gate");

        // Verify fidelity
        let mut state_ref = QuantumState::new(2);
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }
        let mut state_fused = QuantumState::new(2);
        execute_fused_circuit(&mut state_fused, &result);
        let fidelity = state_fused.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-12,
            "H absorbed into CNOT fidelity: {}",
            fidelity
        );
    }

    #[test]
    fn test_both_qubits_absorbed() {
        // H(0), H(1), CNOT(0,1) -> 1 FusedTwo with both H gates absorbed
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(0, 1)];
        let result = fuse_gates(&gates);

        let fused_two_count = result
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::FusedTwo(_)))
            .count();
        assert_eq!(fused_two_count, 1, "Should produce 1 FusedTwo");
        assert_eq!(result.gates.len(), 1, "Total output should be 1 gate");

        // Check fused_count includes both H gates plus the CNOT
        if let FusedOrOriginal::FusedTwo(fg) = &result.gates[0] {
            assert_eq!(
                fg.fused_count, 3,
                "Should fuse H(0) + H(1) + CNOT = 3 gates"
            );
        } else {
            panic!("Expected FusedTwo");
        }

        // Verify fidelity
        let mut state_ref = QuantumState::new(2);
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }
        let mut state_fused = QuantumState::new(2);
        execute_fused_circuit(&mut state_fused, &result);
        let fidelity = state_fused.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-12,
            "Both qubits absorbed fidelity: {}",
            fidelity
        );
    }

    #[test]
    fn test_absorption_fidelity() {
        // Complex circuit with H+CNOT+T patterns on 5 qubits
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::cnot(0, 1),
            Gate::t(0),
            Gate::t(1),
            Gate::h(2),
            Gate::cnot(1, 2),
            Gate::t(2),
            Gate::h(3),
            Gate::h(4),
            Gate::cnot(3, 4),
            Gate::t(3),
            Gate::s(4),
            Gate::cnot(2, 3),
            Gate::h(0),
            Gate::rx(1, 0.7),
            Gate::cnot(0, 1),
        ];

        // Execute unfused
        let mut state_ref = QuantumState::new(5);
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }

        // Execute fused
        let fusion = fuse_gates(&gates);
        let mut state_fused = QuantumState::new(5);
        execute_fused_circuit(&mut state_fused, &fusion);

        let fidelity = state_fused.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-10,
            "Absorption fidelity too low: {} (should be > 1 - 1e-10)",
            fidelity
        );

        // Verify we actually fused some 2-qubit gates
        let fused_two_count = fusion
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::FusedTwo(_)))
            .count();
        assert!(
            fused_two_count > 0,
            "Should have at least one FusedTwo gate"
        );
    }

    #[test]
    fn test_kronecker_identity() {
        // I (x) I should equal I4
        let id2 = Matrix2x2::identity();
        let result = kronecker_2x2(&id2, &id2);
        assert!(is_identity_4x4(&result, 1e-15), "I (x) I should be I4");
    }

    #[test]
    fn test_kronecker_h() {
        // H (x) I should produce known matrix
        let h_mat = gate_to_matrix(&GateType::H).unwrap();
        let id2 = Matrix2x2::identity();
        let result = kronecker_2x2(&h_mat, &id2);

        // H (x) I: the higher qubit gets H, lower qubit gets I.
        // Index convention: row = (bit_hi << 1) | bit_lo
        //
        // Expected (H on qubit 1, I on qubit 0):
        // |00> -> 1/sqrt(2)(|00> + |10>)
        // |01> -> 1/sqrt(2)(|01> + |11>)
        // |10> -> 1/sqrt(2)(|00> - |10>)
        // |11> -> 1/sqrt(2)(|01> - |11>)
        //
        // Matrix:
        // [h00*1,  0, h01*1,  0]
        // [ 0, h00*1,  0, h01*1]
        // [h10*1,  0, h11*1,  0]
        // [ 0, h10*1,  0, h11*1]
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

        // Check specific entries
        assert!((result.data[0][0].re - inv_sqrt2).abs() < 1e-12);
        assert!(result.data[0][1].re.abs() < 1e-12);
        assert!((result.data[0][2].re - inv_sqrt2).abs() < 1e-12);
        assert!(result.data[0][3].re.abs() < 1e-12);

        assert!(result.data[1][0].re.abs() < 1e-12);
        assert!((result.data[1][1].re - inv_sqrt2).abs() < 1e-12);
        assert!(result.data[1][2].re.abs() < 1e-12);
        assert!((result.data[1][3].re - inv_sqrt2).abs() < 1e-12);

        assert!((result.data[2][0].re - inv_sqrt2).abs() < 1e-12);
        assert!(result.data[2][1].re.abs() < 1e-12);
        assert!((result.data[2][2].re + inv_sqrt2).abs() < 1e-12);
        assert!(result.data[2][3].re.abs() < 1e-12);

        assert!(result.data[3][0].re.abs() < 1e-12);
        assert!((result.data[3][1].re - inv_sqrt2).abs() < 1e-12);
        assert!(result.data[3][2].re.abs() < 1e-12);
        assert!((result.data[3][3].re + inv_sqrt2).abs() < 1e-12);
    }

    #[test]
    fn test_3qubit_gate_not_absorbed() {
        // Toffoli gate should still emit Original (not FusedTwo)
        let gates = vec![Gate::h(0), Gate::toffoli(0, 1, 2)];
        let result = fuse_gates(&gates);

        let original_count = result
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::Original(_)))
            .count();
        // Both H(0) (single gate → Original) and Toffoli should be Original
        assert_eq!(
            original_count, 2,
            "Both H(0) and Toffoli should be Original"
        );

        let fused_two_count = result
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::FusedTwo(_)))
            .count();
        assert_eq!(fused_two_count, 0, "Toffoli should NOT produce FusedTwo");

        // Verify fidelity
        let mut state_ref = QuantumState::new(3);
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }
        let mut state_fused = QuantumState::new(3);
        execute_fused_circuit(&mut state_fused, &result);
        let fidelity = state_fused.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-12,
            "3-qubit gate fidelity: {}",
            fidelity
        );
    }

    #[test]
    fn test_two_qubit_gate_to_matrix_cnot() {
        // Verify CNOT matrix extraction matches expected
        let cnot = Gate::cnot(0, 1);
        let mat = two_qubit_gate_to_matrix(&cnot).unwrap();

        // CNOT with control=0, target=1:
        // In our convention with qubit 0 (control) as the low bit:
        // |00> -> |00>  (control=0, no flip)
        // |01> -> |01>  (control=0 in bit 0 means qubit 0=1, but control is qubit 0...)
        //
        // Actually: control=0 means qubit index 0 is the control.
        // Remapped: qa=0 -> qubit 0, qb=1 -> qubit 1.
        // State index = (qubit1_bit << 1) | qubit0_bit
        //
        // CNOT(control=0, target=1):
        // |00> = q0=0,q1=0 -> q0=0,q1=0 = |00>  -> col0: [1,0,0,0]
        // |01> = q0=1,q1=0 -> q0=1,q1=1 = |11>  -> col1: [0,0,0,1]
        // |10> = q0=0,q1=1 -> q0=0,q1=1 = |10>  -> col2: [0,0,1,0]
        // |11> = q0=1,q1=1 -> q0=1,q1=0 = |01>  -> col3: [0,1,0,0]
        assert!((mat.data[0][0].re - 1.0).abs() < 1e-12);
        assert!((mat.data[3][1].re - 1.0).abs() < 1e-12);
        assert!((mat.data[2][2].re - 1.0).abs() < 1e-12);
        assert!((mat.data[1][3].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_two_qubit_gate_to_matrix_swap() {
        // SWAP gate matrix extraction
        let swap = Gate::swap(0, 1);
        let mat = two_qubit_gate_to_matrix(&swap).unwrap();

        // SWAP:
        // |00> -> |00>
        // |01> -> |10>
        // |10> -> |01>
        // |11> -> |11>
        assert!((mat.data[0][0].re - 1.0).abs() < 1e-12);
        assert!((mat.data[2][1].re - 1.0).abs() < 1e-12);
        assert!((mat.data[1][2].re - 1.0).abs() < 1e-12);
        assert!((mat.data[3][3].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_multiply_4x4_identity() {
        let id = Matrix4x4::identity();
        let cnot_gate = Gate::cnot(0, 1);
        let cnot_mat = two_qubit_gate_to_matrix(&cnot_gate).unwrap();
        let result = matrix_multiply_4x4(&cnot_mat, &id);
        assert!(
            matrices_4x4_approx_eq(&result, &cnot_mat, 1e-12),
            "CNOT * I4 should equal CNOT"
        );
    }

    #[test]
    fn test_is_identity_4x4() {
        let id = Matrix4x4::identity();
        assert!(is_identity_4x4(&id, 1e-15));

        let cnot_gate = Gate::cnot(0, 1);
        let cnot_mat = two_qubit_gate_to_matrix(&cnot_gate).unwrap();
        assert!(!is_identity_4x4(&cnot_mat, 1e-12));
    }

    #[test]
    fn test_swap_absorption() {
        // H(0), SWAP(0,1) should produce 1 FusedTwo
        let gates = vec![Gate::h(0), Gate::swap(0, 1)];
        let result = fuse_gates(&gates);

        let fused_two_count = result
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::FusedTwo(_)))
            .count();
        assert_eq!(fused_two_count, 1);

        // Verify fidelity
        let mut state_ref = QuantumState::new(2);
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }
        let mut state_fused = QuantumState::new(2);
        execute_fused_circuit(&mut state_fused, &result);
        let fidelity = state_fused.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-12,
            "SWAP absorption fidelity: {}",
            fidelity
        );
    }

    #[test]
    fn test_noncontiguous_qubit_absorption() {
        // H(0), H(3), CNOT(0,3) on a 4-qubit system - non-adjacent qubits
        let gates = vec![Gate::h(0), Gate::h(3), Gate::cnot(0, 3)];
        let result = fuse_gates(&gates);

        let fused_two_count = result
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::FusedTwo(_)))
            .count();
        assert_eq!(
            fused_two_count, 1,
            "Non-adjacent qubits should still produce FusedTwo"
        );

        // Verify fidelity
        let mut state_ref = QuantumState::new(4);
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }
        let mut state_fused = QuantumState::new(4);
        execute_fused_circuit(&mut state_fused, &result);
        let fidelity = state_fused.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-12,
            "Non-contiguous qubit absorption fidelity: {}",
            fidelity
        );
    }

    // ---------------------------------------------------------------
    // Layer decomposition tests
    // ---------------------------------------------------------------

    #[test]
    fn test_layer_decomposition_single_qubit() {
        // H(0), H(1), H(2) - all can be in same layer (disjoint qubits)
        let gates = vec![Gate::h(0), Gate::h(1), Gate::h(2)];

        let fusion = fuse_gates(&gates);
        let layers = decompose_fused_layers(&fusion.gates);

        assert_eq!(
            layers.len(),
            1,
            "All single-qubit gates should be in one layer"
        );
        assert_eq!(layers[0].len(), 3, "Layer should contain all 3 gates");
    }

    #[test]
    fn test_layer_decomposition_conflicting() {
        // H(0), X(0) - get fused into single Fused gate, so one layer
        // (gate fusion happens before layer decomposition)
        let gates = vec![Gate::h(0), Gate::x(0)];

        let fusion = fuse_gates(&gates);
        let layers = decompose_fused_layers(&fusion.gates);

        // After fusion, these become one Fused gate, so one layer
        assert_eq!(
            layers.len(),
            1,
            "Fused gates on same qubit produce one layer"
        );
    }

    #[test]
    fn test_layer_decomposition_mixed() {
        // H(0), H(1), CNOT(0,1), H(2) ->
        // After fusion: H(0)+CNOT -> FusedTwo, H(1) absorbed, H(2) separate
        // Layer decomposition: H(2) is disjoint from FusedTwo
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(0, 1), Gate::h(2)];

        let fusion = fuse_gates(&gates);
        let layers = decompose_fused_layers(&fusion.gates);

        // H(0) and H(1) get absorbed into CNOT(0,1) -> FusedTwo
        // H(2) is on disjoint qubit
        // Should have 1 layer (all can execute in parallel)
        assert!(layers.len() <= 2, "Should have at most 2 layers");
    }

    #[test]
    fn test_layer_decomposition_two_qubit() {
        // CNOT(0,1), CNOT(2,3) - disjoint, can be in same layer
        let gates = vec![Gate::cnot(0, 1), Gate::cnot(2, 3)];

        let fusion = fuse_gates(&gates);
        let layers = decompose_fused_layers(&fusion.gates);

        assert_eq!(
            layers.len(),
            1,
            "Disjoint two-qubit gates should be in same layer"
        );
        assert_eq!(layers[0].len(), 2, "Layer should contain both CNOTs");
    }

    #[test]
    fn test_layer_decomposition_with_fusion() {
        // Verify layered execution produces same results as sequential
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::rx(0, 0.5),
            Gate::ry(1, 0.3),
            Gate::cnot(0, 1),
            Gate::rz(0, 0.7),
            Gate::h(2),
        ];

        let mut state_ref = QuantumState::new(3);
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }

        let fusion = fuse_gates(&gates);
        let mut state_layered = QuantumState::new(3);
        execute_fused_circuit_with_layers(&mut state_layered, &fusion);

        let fidelity = state_layered.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-12,
            "Layered execution fidelity: {}",
            fidelity
        );
    }

    #[test]
    fn test_layer_decomposition_empty() {
        let layers = decompose_fused_layers(&[]);
        assert_eq!(layers.len(), 0, "Empty input should produce empty layers");
    }

    #[test]
    fn test_layer_decomposition_all_conflicting() {
        // All gates on qubit 0 - get fused into single Fused gate
        // (gate fusion happens before layer decomposition)
        let gates = vec![Gate::h(0), Gate::x(0), Gate::y(0), Gate::z(0)];

        let fusion = fuse_gates(&gates);
        let layers = decompose_fused_layers(&fusion.gates);

        // After fusion, all gates on same qubit become one Fused gate
        assert_eq!(
            layers.len(),
            1,
            "All gates on same qubit fuse into one layer"
        );
    }

    // ---------------------------------------------------------------
    // Phase 4A: Diagonal fusion tests
    // ---------------------------------------------------------------

    #[test]
    fn test_diagonal_fusion_rz_sequence() {
        // Rz(0.3, q0), Rz(0.5, q0) → should be fused by 1Q fusion (same qubit)
        // Rz(0.3, q0), Rz(0.5, q1) → two gates on different qubits, both diagonal
        // → should produce a FusedDiagonal with 2 single-qubit entries
        let gates = vec![
            Gate::new(GateType::Rz(0.3), vec![0], vec![]),
            Gate::new(GateType::Rz(0.5), vec![1], vec![]),
        ];
        let result = fuse_gates(&gates);

        let diag_count = result
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::Diagonal(_)))
            .count();
        assert_eq!(
            diag_count, 1,
            "Two different-qubit Rz gates should produce 1 FusedDiagonal"
        );

        // Verify fidelity
        let mut state_ref = QuantumState::new(2);
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }
        let mut state_fused = QuantumState::new(2);
        execute_fused_circuit(&mut state_fused, &result);
        let fidelity = state_fused.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-12,
            "Diagonal Rz fusion fidelity: {}",
            fidelity
        );
    }

    #[test]
    fn test_diagonal_fusion_cr_sequence() {
        // Multiple CR gates on different qubit pairs (like QFT pattern)
        let pi = std::f64::consts::PI;
        let gates = vec![
            Gate::new(GateType::CR(pi / 2.0), vec![1], vec![0]),
            Gate::new(GateType::CR(pi / 4.0), vec![2], vec![0]),
            Gate::new(GateType::CR(pi / 8.0), vec![3], vec![0]),
        ];
        let result = fuse_gates(&gates);

        let diag_count = result
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::Diagonal(_)))
            .count();
        assert!(
            diag_count >= 1,
            "Consecutive CR gates should produce FusedDiagonal"
        );

        // Verify fidelity
        let mut state_ref = QuantumState::new(4);
        // Need some non-zero amplitudes to test phases
        apply_gate_to_state(&mut state_ref, &Gate::h(0));
        apply_gate_to_state(&mut state_ref, &Gate::h(1));
        apply_gate_to_state(&mut state_ref, &Gate::h(2));
        apply_gate_to_state(&mut state_ref, &Gate::h(3));
        let state_before = state_ref.clone();
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }

        let mut state_fused = state_before;
        execute_fused_circuit(&mut state_fused, &result);
        let fidelity = state_fused.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-10,
            "Diagonal CR fusion fidelity: {}",
            fidelity
        );
    }

    #[test]
    fn test_diagonal_fusion_mixed_1q_2q() {
        // Mix of single-qubit diagonal (Z, T) and two-qubit diagonal (CZ, CR)
        let pi = std::f64::consts::PI;
        let gates = vec![
            Gate::new(GateType::Z, vec![0], vec![]),
            Gate::new(GateType::T, vec![1], vec![]),
            Gate::cz(0, 1),
            Gate::new(GateType::CR(pi / 4.0), vec![2], vec![0]),
            Gate::new(GateType::S, vec![2], vec![]),
        ];
        let result = fuse_gates(&gates);

        // Verify fidelity
        let mut state_ref = QuantumState::new(3);
        apply_gate_to_state(&mut state_ref, &Gate::h(0));
        apply_gate_to_state(&mut state_ref, &Gate::h(1));
        apply_gate_to_state(&mut state_ref, &Gate::h(2));
        let state_before = state_ref.clone();
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }

        let mut state_fused = state_before;
        execute_fused_circuit(&mut state_fused, &result);
        let fidelity = state_fused.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-10,
            "Mixed diagonal fusion fidelity: {}",
            fidelity
        );
    }

    #[test]
    fn test_diagonal_fusion_qft_pattern() {
        // Full QFT-4 pattern: H, then CR chain, repeat.
        // The CR chains should be diagonally fused.
        let mut gates = Vec::new();
        let n = 4;
        for i in 0..n {
            gates.push(Gate::h(i));
            for j in (i + 1)..n {
                let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
                gates.push(Gate::new(GateType::CR(angle), vec![j], vec![i]));
            }
        }

        // Execute unfused
        let mut state_unfused = QuantumState::new(n);
        for gate in &gates {
            apply_gate_to_state(&mut state_unfused, gate);
        }

        // Execute fused (with diagonal fusion)
        let fusion = fuse_gates(&gates);
        let mut state_fused = QuantumState::new(n);
        execute_fused_circuit(&mut state_fused, &fusion);

        let fidelity = state_fused.fidelity(&state_unfused);
        assert!(
            fidelity > 1.0 - 1e-10,
            "QFT-4 with diagonal fusion fidelity: {}",
            fidelity
        );

        // Verify we actually created some diagonal fusions
        let diag_count = fusion
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::Diagonal(_)))
            .count();
        assert!(
            diag_count > 0,
            "QFT should produce at least one FusedDiagonal from CR chains"
        );
    }

    // ---------------------------------------------------------------
    // Phase 4B: Threshold removal test
    // ---------------------------------------------------------------

    #[test]
    fn test_cr_absorption_large_circuit() {
        // With threshold raised to 256, CR gates should be absorbed
        // even in circuits with >16 qubits
        let pi = std::f64::consts::PI;
        let n = 20; // > old threshold of 16

        let gates = vec![
            Gate::h(0),
            Gate::h(5),
            Gate::new(GateType::CR(pi / 4.0), vec![5], vec![0]),
        ];

        let result = fuse_gates(&gates);

        // With the new threshold (256), this should produce a FusedTwo
        let fused_two_count = result
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::FusedTwo(_)))
            .count();
        assert_eq!(
            fused_two_count, 1,
            "CR with pending H gates should produce FusedTwo at 20 qubits"
        );

        // Verify fidelity
        let mut state_ref = QuantumState::new(n);
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }
        let mut state_fused = QuantumState::new(n);
        execute_fused_circuit(&mut state_fused, &result);
        let fidelity = state_fused.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-10,
            "Large circuit 2Q absorption fidelity: {}",
            fidelity
        );
    }

    #[test]
    fn test_cr_matrix_extraction() {
        // Verify CR(pi/4) matrix extraction produces correct diagonal
        let pi = std::f64::consts::PI;
        let gate = Gate::new(GateType::CR(pi / 4.0), vec![1], vec![0]);
        let mat = two_qubit_gate_to_matrix(&gate).unwrap();

        // CR(θ) is diag(1, 1, 1, e^{iθ})
        let phase = C64::new((pi / 4.0).cos(), (pi / 4.0).sin());
        assert!((mat.data[0][0].re - 1.0).abs() < 1e-12);
        assert!((mat.data[1][1].re - 1.0).abs() < 1e-12);
        assert!((mat.data[2][2].re - 1.0).abs() < 1e-12);
        assert!((mat.data[3][3].re - phase.re).abs() < 1e-12);
        assert!((mat.data[3][3].im - phase.im).abs() < 1e-12);
        // Off-diagonal should be zero
        assert!(mat.data[0][1].re.abs() < 1e-12);
        assert!(mat.data[1][0].re.abs() < 1e-12);
    }

    #[test]
    fn test_cr_absorption_fidelity() {
        // Verify CR(pi/4) absorption into fused 2-qubit gate
        let pi = std::f64::consts::PI;
        let gate = Gate::new(GateType::CR(pi / 4.0), vec![1], vec![0]);

        // Verify fidelity of CR execution through fusion
        let gates = vec![Gate::h(0), gate];
        let mut state_ref = QuantumState::new(2);
        for g in &gates {
            apply_gate_to_state(&mut state_ref, g);
        }
        let fusion = fuse_gates(&gates);
        let mut state_fused = QuantumState::new(2);
        execute_fused_circuit(&mut state_fused, &fusion);
        let fidelity = state_fused.fidelity(&state_ref);
        assert!(
            fidelity > 1.0 - 1e-10,
            "CR absorption fidelity: {}",
            fidelity
        );

        // Should have absorbed H into CR as FusedTwo
        let fused_two = fusion
            .gates
            .iter()
            .filter(|g| matches!(g, FusedOrOriginal::FusedTwo(_)))
            .count();
        assert_eq!(fused_two, 1, "H should be absorbed into CR");
    }
}
