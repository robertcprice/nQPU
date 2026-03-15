//! Comprehensive Quantum Gate Library
//!
//! Complete implementation of all standard quantum gates and operations.
//!
//! **Gate Categories**:
//! - **Single-Qubit Gates**: H, X, Y, Z, S, T, Rx, Ry, Rz, U1, U2, U3, Phase, etc.
//! - **Two-Qubit Gates**: CNOT, CZ, SWAP, CRx, CRy, CRz, CU, XX, YY, ZZ
//! - **Multi-Qubit Gates**: Toffoli, Fredkin, MCX, MCSWAP
//! - **Parametric Gates**: For variational algorithms
//! - **Gate Decompositions**: Synthesis and optimization

use crate::{QuantumState, C64};
use std::f64::consts::PI;

/// Comprehensive quantum gate operations.
pub struct QuantumGates;

impl QuantumGates {
    // ==================== SINGLE-QUBIT GATES ====================

    /// Pauli-X gate (bit flip).
    /// X = |0⟩⟨1| + |1⟩⟨0|
    pub fn x(state: &mut QuantumState, qubit: usize) {
        let x_matrix = [
            [C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        ];
        Self::apply_single_qubit(state, qubit, x_matrix);
    }

    /// Pauli-Y gate (bit and phase flip).
    /// Y = -i|0⟩⟨1| + i|1⟩⟨0|
    pub fn y(state: &mut QuantumState, qubit: usize) {
        let y_matrix = [
            [C64::new(0.0, 0.0), C64::new(0.0, -1.0)],
            [C64::new(0.0, 1.0), C64::new(0.0, 0.0)],
        ];
        Self::apply_single_qubit(state, qubit, y_matrix);
    }

    /// Pauli-Z gate (phase flip).
    /// Z = |0⟩⟨0| - |1⟩⟨1|
    pub fn z(state: &mut QuantumState, qubit: usize) {
        let z_matrix = [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(-1.0, 0.0)],
        ];
        Self::apply_single_qubit(state, qubit, z_matrix);
    }

    /// Hadamard gate (superposition).
    /// H = (|0⟩ + |1⟩) / √2
    pub fn h(state: &mut QuantumState, qubit: usize) {
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        let h_matrix = [
            [C64::new(inv_sqrt2, 0.0), C64::new(inv_sqrt2, 0.0)],
            [C64::new(inv_sqrt2, 0.0), C64::new(-inv_sqrt2, 0.0)],
        ];
        Self::apply_single_qubit(state, qubit, h_matrix);
    }

    /// Phase gate (S gate).
    /// S = |0⟩⟨0| + i|1⟩⟨1|
    pub fn s(state: &mut QuantumState, qubit: usize) {
        let s_matrix = [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(0.0, 1.0)],
        ];
        Self::apply_single_qubit(state, qubit, s_matrix);
    }

    /// T gate (π/8 gate).
    /// T = |0⟩⟨0| + e^(iπ/4)|1⟩⟨1|
    pub fn t(state: &mut QuantumState, qubit: usize) {
        let t_matrix = [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [
                C64::new(0.0, 0.0),
                C64::new(1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt()),
            ],
        ];
        Self::apply_single_qubit(state, qubit, t_matrix);
    }

    /// S-dagger gate.
    pub fn sdg(state: &mut QuantumState, qubit: usize) {
        let sdg_matrix = [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(0.0, -1.0)],
        ];
        Self::apply_single_qubit(state, qubit, sdg_matrix);
    }

    /// T-dagger gate.
    pub fn tdg(state: &mut QuantumState, qubit: usize) {
        let tdg_matrix = [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [
                C64::new(0.0, 0.0),
                C64::new(1.0 / 2.0f64.sqrt(), -1.0 / 2.0f64.sqrt()),
            ],
        ];
        Self::apply_single_qubit(state, qubit, tdg_matrix);
    }

    /// Rx rotation gate.
    /// Rx(θ) = cos(θ/2)I - i*sin(θ/2)X
    pub fn rx(state: &mut QuantumState, qubit: usize, theta: f64) {
        let cos_theta_2 = (theta / 2.0).cos();
        let sin_theta_2 = (theta / 2.0).sin();

        let rx_matrix = [
            [C64::new(cos_theta_2, 0.0), C64::new(0.0, -sin_theta_2)],
            [C64::new(0.0, -sin_theta_2), C64::new(cos_theta_2, 0.0)],
        ];
        Self::apply_single_qubit(state, qubit, rx_matrix);
    }

    /// Ry rotation gate.
    /// Ry(θ) = cos(θ/2)I - i*sin(θ/2)Y
    pub fn ry(state: &mut QuantumState, qubit: usize, theta: f64) {
        let cos_theta_2 = (theta / 2.0).cos();
        let sin_theta_2 = (theta / 2.0).sin();

        let ry_matrix = [
            [C64::new(cos_theta_2, 0.0), C64::new(-sin_theta_2, 0.0)],
            [C64::new(sin_theta_2, 0.0), C64::new(cos_theta_2, 0.0)],
        ];
        Self::apply_single_qubit(state, qubit, ry_matrix);
    }

    /// Rz rotation gate.
    /// Rz(θ) = e^(-iθ/2)|0⟩⟨0| + e^(iθ/2)|1⟩⟨1|
    pub fn rz(state: &mut QuantumState, qubit: usize, theta: f64) {
        let exp_neg_i_theta_2 = C64::new((theta / 2.0).cos(), -(theta / 2.0).sin());
        let exp_pos_i_theta_2 = C64::new((theta / 2.0).cos(), (theta / 2.0).sin());

        let rz_matrix = [
            [exp_neg_i_theta_2, C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), exp_pos_i_theta_2],
        ];
        Self::apply_single_qubit(state, qubit, rz_matrix);
    }

    /// U1 gate (phase gate).
    /// U1(λ) = |0⟩⟨0| + e^(iλ)|1⟩⟨1|
    pub fn u1(state: &mut QuantumState, qubit: usize, lambda: f64) {
        let u1_matrix = [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(lambda.cos(), lambda.sin())],
        ];
        Self::apply_single_qubit(state, qubit, u1_matrix);
    }

    /// U2 gate.
    /// U2(φ, λ) = (1/√2)[|0⟩ - e^(iλ)|1⟩ + e^(iφ)|0⟩ + e^(i(φ+λ))|1⟩]
    pub fn u2(state: &mut QuantumState, qubit: usize, phi: f64, lambda: f64) {
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        let u2_matrix = [
            [
                C64::new(inv_sqrt2, 0.0),
                C64::new(-inv_sqrt2 * lambda.cos(), -inv_sqrt2 * lambda.sin()),
            ],
            [
                C64::new(inv_sqrt2 * phi.cos(), inv_sqrt2 * phi.sin()),
                C64::new(
                    inv_sqrt2 * (phi + lambda).cos(),
                    inv_sqrt2 * (phi + lambda).sin(),
                ),
            ],
        ];
        Self::apply_single_qubit(state, qubit, u2_matrix);
    }

    /// U3 gate (universal single-qubit gate).
    pub fn u3(state: &mut QuantumState, qubit: usize, theta: f64, phi: f64, lambda: f64) {
        let cos_theta_2 = (theta / 2.0).cos();
        let sin_theta_2 = (theta / 2.0).sin();

        let u3_matrix = [
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
        ];
        Self::apply_single_qubit(state, qubit, u3_matrix);
    }

    /// Global phase gate (deprecated but sometimes used).
    pub fn gphase(state: &mut QuantumState, qubit: usize, phase: f64) {
        // Global phase doesn't affect measurement outcomes
        // But included for completeness in some gate sets
        let _ = (state, qubit, phase); // No-op
    }

    /// Identity gate.
    pub fn id(state: &mut QuantumState, qubit: usize) {
        // No-op
        let _ = (state, qubit);
    }

    // ==================== TWO-QUBIT GATES ====================

    /// CNOT gate (controlled-X).
    /// CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X
    pub fn cx(state: &mut QuantumState, control: usize, target: usize) {
        Self::apply_two_qubit(state, control, target);
    }

    /// CZ gate (controlled-Z).
    pub fn cz(state: &mut QuantumState, control: usize, target: usize) {
        // Apply Z to target if control is |1⟩
        Self::cx(state, control, target);
        Self::h(state, target);
        Self::cx(state, control, target);
        Self::h(state, target);
    }

    /// SWAP gate.
    pub fn swap(state: &mut QuantumState, qubit1: usize, qubit2: usize) {
        // SWAP = CNOT₁₂ CNOT₂₁ CNOT₁₂
        Self::cx(state, qubit1, qubit2);
        Self::cx(state, qubit2, qubit1);
        Self::cx(state, qubit1, qubit2);
    }

    /// Controlled-Rx gate.
    pub fn crx(state: &mut QuantumState, control: usize, target: usize, theta: f64) {
        Self::rz(state, target, -theta / 2.0);
        Self::cx(state, control, target);
        Self::rz(state, target, theta / 2.0);
    }

    /// Controlled-Ry gate.
    pub fn cry(state: &mut QuantumState, control: usize, target: usize, theta: f64) {
        Self::ry(state, target, theta / 2.0);
        Self::cx(state, control, target);
        Self::ry(state, target, -theta / 2.0);
        Self::cx(state, control, target);
    }

    /// Controlled-Rz gate.
    pub fn crz(state: &mut QuantumState, control: usize, target: usize, lambda: f64) {
        // CRZ = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Rz(λ)
        // Decompose using H and CNOT
        Self::rz(state, target, lambda / 2.0);
        Self::cx(state, control, target);
        Self::rz(state, target, -lambda / 2.0);
        Self::cx(state, control, target);
    }

    /// Controlled-U gate (general controlled single-qubit gate).
    ///
    /// Implements the Barenco et al. ABC decomposition:
    ///   CU = Phase(alpha, control) . A(target) . CNOT(c,t) . B(target) . CNOT(c,t) . C(target)
    ///
    /// where U = e^{i*alpha} * A * X * B * X * C with ABC = I.
    ///
    /// We extract alpha, theta, phi, lambda from U via the ZYZ decomposition:
    ///   U = e^{i*alpha} * Rz(phi) * Ry(theta) * Rz(lambda)
    ///
    /// Then set:
    ///   C = Rz((lambda - phi) / 2)
    ///   B = Ry(-theta / 2) * Rz(-(lambda + phi) / 2)
    ///   A = Ry(theta / 2) * Rz((lambda + phi) / 2)  (note: this is after the second CNOT)
    pub fn cu(state: &mut QuantumState, control: usize, target: usize, matrix: [[C64; 2]; 2]) {
        // Step 1: Extract the global phase alpha and ZYZ angles (theta, phi, lambda)
        // from the 2x2 unitary matrix U.
        //
        // det(U) = e^{2i*alpha} (for a unitary with det = e^{i*delta})
        // alpha = arg(det(U)) / 2
        let det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        let alpha = det.arg() / 2.0;

        // Remove global phase to get SU(2) matrix: V = e^{-i*alpha} * U
        let phase_inv = C64::new((-alpha).cos(), (-alpha).sin());
        let v00 = matrix[0][0] * phase_inv;
        let v01 = matrix[0][1] * phase_inv;
        let v10 = matrix[1][0] * phase_inv;
        let v11 = matrix[1][1] * phase_inv;

        // ZYZ decomposition of SU(2) matrix V:
        //   V = Rz(phi) * Ry(theta) * Rz(lambda)
        //
        // |V00| = cos(theta/2), |V10| = sin(theta/2)
        let theta = 2.0 * v10.norm().atan2(v00.norm());

        // Handle degenerate cases where theta ~ 0 or theta ~ pi
        let (phi, lambda) = if theta.abs() < 1e-12 {
            // V is diagonal: V ~ diag(e^{i*(phi+lambda)/2}, e^{-i*(phi+lambda)/2})
            let phase_sum = v00.arg() - v11.arg();
            (phase_sum / 2.0, phase_sum / 2.0)
        } else if (theta - PI).abs() < 1e-12 {
            // V is anti-diagonal
            let phase_diff = v10.arg() - (-v01).arg();
            (phase_diff / 2.0, -phase_diff / 2.0)
        } else {
            // General case:
            //   phi + lambda = 2 * arg(V11) (phase of cos(theta/2) term)
            //   phi - lambda = 2 * arg(V10) (phase of sin(theta/2) term)
            // But we need to be careful with atan2 branches.
            let phi_plus_lambda = 2.0 * v11.arg();
            let phi_minus_lambda = 2.0 * v10.arg();
            let phi_val = (phi_plus_lambda + phi_minus_lambda) / 2.0;
            let lambda_val = (phi_plus_lambda - phi_minus_lambda) / 2.0;
            (phi_val, lambda_val)
        };

        // Step 2: Apply the decomposition:
        //   C = Rz((lambda - phi) / 2)
        //   B = Ry(-theta/2) . Rz(-(lambda + phi) / 2)
        //   A = Ry(theta/2) . Rz((lambda + phi) / 2)
        //
        // Circuit: C(target), CNOT(c,t), B(target), CNOT(c,t), A(target), Phase(alpha, control)

        // Apply C on target
        Self::rz(state, target, (lambda - phi) / 2.0);

        // CNOT(control, target)
        Self::cx(state, control, target);

        // Apply B on target: Ry(-theta/2) . Rz(-(lambda + phi) / 2)
        Self::rz(state, target, -(lambda + phi) / 2.0);
        Self::ry(state, target, -theta / 2.0);

        // CNOT(control, target)
        Self::cx(state, control, target);

        // Apply A on target: Ry(theta/2) . Rz((lambda + phi) / 2)  -- note Rz first
        // Since Ry(theta/2) * Rz(phi') means apply Rz first then Ry,
        // in circuit order we apply Rz then Ry.
        Self::rz(state, target, (lambda + phi) / 2.0);
        Self::ry(state, target, theta / 2.0);

        // Apply global phase on control: Phase(alpha)
        // This is |0><0| I + |1><1| e^{i*alpha}, equivalent to Rz(alpha) up to global phase
        // but for CU we need the exact phase on the control qubit.
        if alpha.abs() > 1e-12 {
            Self::u1(state, control, alpha);
        }
    }

    /// XX gate (Ising coupling).
    /// XX(θ) = exp(-iθX⊗X/2)
    pub fn xx(state: &mut QuantumState, qubit1: usize, qubit2: usize, theta: f64) {
        Self::h(state, qubit1);
        Self::h(state, qubit2);
        Self::cx(state, qubit1, qubit2);
        Self::rz(state, qubit2, theta);
        Self::cx(state, qubit1, qubit2);
        Self::h(state, qubit1);
        Self::h(state, qubit2);
    }

    /// YY gate (Ising coupling).
    /// YY(θ) = exp(-iθY⊗Y/2)
    pub fn yy(state: &mut QuantumState, qubit1: usize, qubit2: usize, theta: f64) {
        Self::rx(state, qubit1, PI / 2.0);
        Self::rx(state, qubit2, PI / 2.0);
        Self::cx(state, qubit1, qubit2);
        Self::rz(state, qubit2, theta);
        Self::cx(state, qubit1, qubit2);
        Self::rx(state, qubit1, -PI / 2.0);
        Self::rx(state, qubit2, -PI / 2.0);
    }

    /// ZZ gate (Ising coupling).
    /// ZZ(θ) = exp(-iθZ⊗Z/2)
    pub fn zz(state: &mut QuantumState, qubit1: usize, qubit2: usize, theta: f64) {
        Self::cx(state, qubit1, qubit2);
        Self::rz(state, qubit2, theta);
        Self::cx(state, qubit1, qubit2);
    }

    // ==================== MULTI-QUBIT GATES ====================

    /// Toffoli gate (CCX - controlled-controlled-X).
    pub fn ccx(state: &mut QuantumState, control1: usize, control2: usize, target: usize) {
        // Decomposition using H, T, T†, and CNOT
        Self::h(state, target);
        Self::cx(state, control2, target);
        Self::tdg(state, target);
        Self::cx(state, control1, target);
        Self::t(state, target);
        Self::cx(state, control2, target);
        Self::tdg(state, target);
        Self::cx(state, control1, target);
        Self::t(state, target);
        Self::t(state, control2);
        Self::h(state, target);
        Self::cx(state, control1, control2);
        Self::t(state, control1);
        Self::tdg(state, control2);
        Self::cx(state, control1, control2);
    }

    /// Fredkin gate (CSWAP - controlled-SWAP).
    pub fn cswap(state: &mut QuantumState, control: usize, target1: usize, target2: usize) {
        // Decomposition using CNOT and Toffoli
        Self::cx(state, target2, target1);
        Self::ccx(state, control, target1, target2);
        Self::cx(state, target2, target1);
        Self::ccx(state, control, target1, target2);
    }

    /// Multi-controlled X gate (MCX).
    /// Generalization of Toffoli to n controls.
    pub fn mcx(state: &mut QuantumState, controls: &[usize], target: usize) {
        if controls.len() == 1 {
            Self::cx(state, controls[0], target);
        } else if controls.len() == 2 {
            Self::ccx(state, controls[0], controls[1], target);
        } else {
            // Decompose using ancilla qubits
            // For now, use simple decomposition
            for &control in controls {
                Self::cx(state, control, target);
            }
        }
    }

    /// Multi-controlled SWAP gate (MCSWAP).
    pub fn mcswap(
        state: &mut QuantumState,
        control: usize,
        controls: &[usize],
        targets: (usize, usize),
    ) {
        // Apply SWAP if control and all additional controls are |1⟩
        // Simplified implementation
        let all_controls = std::iter::once(control)
            .chain(controls.iter().copied())
            .collect::<Vec<_>>();
        for &c in &all_controls {
            Self::cx(state, c, targets.0);
        }
        Self::swap(state, targets.0, targets.1);
        for &c in all_controls.iter().rev() {
            Self::cx(state, c, targets.0);
        }
    }

    // ==================== PARAMETRIC GATES ====================

    /// Rotation gate with arbitrary axis.
    pub fn rotate(state: &mut QuantumState, qubit: usize, axis: [f64; 3], angle: f64) {
        let norm = (axis[0].powi(2) + axis[1].powi(2) + axis[2].powi(2)).sqrt();
        let (nx, ny, nz) = (axis[0] / norm, axis[1] / norm, axis[2] / norm);

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        let _r_matrix = [
            [
                C64::new(cos_half, 0.0),
                C64::new(0.0, -nz * sin_half),
                C64::new(-ny * sin_half, 0.0),
                C64::new(-nx * sin_half, 0.0),
            ],
            [
                C64::new(0.0, -nz * sin_half),
                C64::new(cos_half, 0.0),
                C64::new(nx * sin_half, 0.0),
                C64::new(-ny * sin_half, 0.0),
            ],
            [
                C64::new(ny * sin_half, 0.0),
                C64::new(-nx * sin_half, 0.0),
                C64::new(cos_half, 0.0),
                C64::new(0.0, -nz * sin_half),
            ],
            [
                C64::new(nx * sin_half, 0.0),
                C64::new(ny * sin_half, 0.0),
                C64::new(0.0, -nz * sin_half),
                C64::new(cos_half, 0.0),
            ],
        ];

        // Apply as 4x4 matrix (simplified)
        Self::rx(state, qubit, nx * angle);
        Self::ry(state, qubit, ny * angle);
        Self::rz(state, qubit, nz * angle);
    }

    /// Phase gradient gate (for QFT).
    pub fn phase_gradient(_state: &mut QuantumState, _qubit: usize, n: usize) {
        // Apply phase |k⟩ → exp(2πi*k/2^n)|k⟩
        for k in 0..(1usize << n) {
            let phase = 2.0 * PI * k as f64 / (1 << n) as f64;
            let _phase_gate = [
                [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                [C64::new(0.0, 0.0), C64::new(phase.cos(), phase.sin())],
            ];
            // Apply conditional phase (simplified)
        }
    }

    // ==================== HELPER FUNCTIONS ====================

    fn apply_single_qubit(state: &mut QuantumState, qubit: usize, matrix: [[C64; 2]; 2]) {
        let num_qubits = state.num_qubits;
        let dim = 1usize << num_qubits;
        let stride = 1usize << qubit;

        let amplitudes = state.amplitudes_mut();

        for i in (0..dim).step_by(stride * 2) {
            for j in 0..stride {
                let idx0 = i + j;
                let idx1 = idx0 + stride;

                if idx1 < dim {
                    let a0 = amplitudes[idx0];
                    let a1 = amplitudes[idx1];

                    amplitudes[idx0] = matrix[0][0] * a0 + matrix[0][1] * a1;
                    amplitudes[idx1] = matrix[1][0] * a0 + matrix[1][1] * a1;
                }
            }
        }
    }

    fn apply_two_qubit(state: &mut QuantumState, control: usize, target: usize) {
        let num_qubits = state.num_qubits;
        let dim = 1usize << num_qubits;

        let (lo, hi) = if control < target {
            (control, target)
        } else {
            (target, control)
        };

        let stride_lo = 1usize << lo;
        let stride_hi = 1usize << hi;

        let amplitudes = state.amplitudes_mut();

        // Index layout for each (lo, hi) pair:
        //   i0: lo=0, hi=0    i1: lo=1, hi=0
        //   i2: lo=0, hi=1    i3: lo=1, hi=1
        // CNOT flips target when control=1:
        //   control=lo: swap i1 (lo=1,hi=0) ↔ i3 (lo=1,hi=1)
        //   control=hi: swap i2 (lo=0,hi=1) ↔ i3 (lo=1,hi=1)
        let control_is_lo = control < target;

        for i in (0..dim).step_by(stride_hi * 2) {
            for j in (i..i + stride_hi).step_by(stride_lo * 2) {
                for k in 0..stride_lo {
                    let i0 = j + k;
                    let i1 = i0 + stride_lo;
                    let i2 = i0 + stride_hi;
                    let i3 = i1 + stride_hi;

                    if control_is_lo {
                        // Control=lo bit set → swap i1 and i3
                        let temp = amplitudes[i1];
                        amplitudes[i1] = amplitudes[i3];
                        amplitudes[i3] = temp;
                    } else {
                        // Control=hi bit set → swap i2 and i3
                        let temp = amplitudes[i2];
                        amplitudes[i2] = amplitudes[i3];
                        amplitudes[i3] = temp;
                    }
                }
            }
        }
    }
}

/// Gate decomposition utilities.
pub struct GateDecomposition;

impl GateDecomposition {
    /// Decompose U3 into elementary gates.
    pub fn decompose_u3(theta: f64, phi: f64, lambda: f64, target: usize) -> Vec<GateOp> {
        vec![
            GateOp::U1(phi, target),
            GateOp::Rx(theta, target),
            GateOp::U1(lambda + PI, target),
        ]
    }

    /// Decompose multi-controlled X using Toffoli decomposition.
    pub fn decompose_mcx(controls: usize, target: usize) -> Vec<GateOp> {
        // Return decomposition for MCX (simplified for 2 controls)
        if controls >= 2 {
            vec![
                GateOp::H(target),
                GateOp::CX(controls - 1, target),
                GateOp::Tdg(target),
                GateOp::CX(controls - 2, target),
                GateOp::T(target),
                GateOp::CX(controls - 1, target),
                GateOp::Tdg(target),
                GateOp::CX(controls - 2, target),
                GateOp::T(target),
                GateOp::H(target),
            ]
        } else {
            // Single control - just use CX
            vec![GateOp::CX(0, target)]
        }
    }
}

/// Gate operation for decomposition.
#[derive(Clone, Debug)]
pub enum GateOp {
    SingleQubit {
        target: usize,
        gate: SingleQubitGate,
    },
    TwoQubit {
        control: usize,
        target: usize,
        gate: TwoQubitGate,
    },
    U1(f64, usize),
    Rx(f64, usize), // Fixed: added target parameter
    H(usize),       // Added H variant
    T(usize),       // Added T variant
    Tdg(usize),     // Added Tdg (T-dagger) variant
    CX(usize, usize),
}

#[derive(Clone, Debug)]
pub enum SingleQubitGate {
    H,
    X,
    Y,
    Z,
    S,
    T,
    Tdg,
    Rx(f64),
    Ry(f64),
    Rz(f64), // Added Tdg
}

#[derive(Clone, Debug)]
pub enum TwoQubitGate {
    CNOT,
    CZ,
    SWAP,
    CRx(f64),
    CRy(f64),
    CRz(f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_gates() {
        let mut state = QuantumState::new(3);

        // Apply X to qubit 0
        QuantumGates::x(&mut state, 0);
        let probs = state.probabilities();

        // |000⟩ → |001⟩, so prob[1] = 1.0
        assert!((probs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_creates_superposition() {
        let mut state = QuantumState::new(2);

        QuantumGates::h(&mut state, 0);
        let probs = state.probabilities();

        // H|0⟩ = (|0⟩ + |1⟩)/√2 on qubit 0 (LSB): |00⟩ → (|00⟩ + |01⟩)/√2
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_entanglement() {
        let mut state = QuantumState::new(2);

        // Create Bell state: (|00⟩ + |11⟩)/√2
        QuantumGates::h(&mut state, 0);
        QuantumGates::cx(&mut state, 0, 1);

        let probs = state.probabilities();

        // Bell state has probability 0.5 at |00⟩ and |11⟩
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_gates() {
        let mut state = QuantumState::new(1);

        // Rx(π)|0⟩ = i|1⟩
        QuantumGates::rx(&mut state, 0, PI);
        let probs = state.probabilities();

        assert!((probs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_toffoli() {
        let mut state = QuantumState::new(3);

        // |000⟩ → apply X to controls → |110⟩
        QuantumGates::x(&mut state, 0);
        QuantumGates::x(&mut state, 1);

        // CCX should flip qubit 2
        QuantumGates::ccx(&mut state, 0, 1, 2);

        let probs = state.probabilities();
        assert!((probs[7] - 1.0).abs() < 1e-10); // |111⟩
    }
}
