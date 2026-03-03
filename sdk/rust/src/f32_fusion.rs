//! Integrated f32 + gate-fusion execution.
//!
//! This module combines the existing fusion planner with the f32 backend to
//! reduce memory bandwidth and state-vector traversals in one pass.

use crate::gate_fusion::{fuse_gates, FusedOrOriginal, FusionResult};
use crate::gates::{Gate, GateType};
use crate::quantum_f32::{GateOpsF32, QuantumStateF32};
use crate::{GateOperations, QuantumState, C32, C64};

/// Performance and quality telemetry for a fused-f32 execution.
#[derive(Clone, Debug)]
pub struct F32FusionMetrics {
    pub original_gates: usize,
    pub fused_ops: usize,
    pub gates_eliminated: usize,
    pub estimated_speedup: f64,
    /// Optional quality signal. `NaN` means not computed.
    pub fidelity_vs_f64: f64,
}

/// F32 + Fusion execution engine.
#[derive(Clone, Debug)]
pub struct F32FusionExecutor {
    /// If true, unsupported gates are applied by temporary f64 fallback.
    pub fallback_to_f64: bool,
    /// If true, run an additional fused f64 reference pass to compute fidelity.
    /// Disabled by default because it roughly doubles execution cost.
    pub compute_reference_fidelity: bool,
}

impl Default for F32FusionExecutor {
    fn default() -> Self {
        Self {
            fallback_to_f64: true,
            compute_reference_fidelity: false,
        }
    }
}

impl F32FusionExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_reference_fidelity(mut self, enabled: bool) -> Self {
        self.compute_reference_fidelity = enabled;
        self
    }

    /// Execute a gate list through fusion on an f32 state vector.
    pub fn execute(
        &self,
        num_qubits: usize,
        gates: &[Gate],
    ) -> Result<(QuantumStateF32, F32FusionMetrics), String> {
        let fusion = fuse_gates(gates);

        let mut state32 = QuantumStateF32::new(num_qubits);
        self.execute_fusion_into(&mut state32, &fusion)?;

        let fidelity = if self.compute_reference_fidelity && num_qubits <= 24 {
            let mut reference = QuantumState::new(num_qubits);
            crate::gate_fusion::execute_fused_circuit(&mut reference, &fusion);
            state32.fidelity_vs_f64(&reference)
        } else {
            f64::NAN
        };

        let reduction = fusion.gates_eliminated as f64 / fusion.original_count.max(1) as f64;
        let estimated_speedup = (1.0 + reduction) * 1.8;

        let metrics = F32FusionMetrics {
            original_gates: fusion.original_count,
            fused_ops: fusion.gates.len(),
            gates_eliminated: fusion.gates_eliminated,
            estimated_speedup,
            fidelity_vs_f64: fidelity,
        };

        Ok((state32, metrics))
    }

    pub fn execute_fusion_into(
        &self,
        state: &mut QuantumStateF32,
        fusion: &FusionResult,
    ) -> Result<(), String> {
        for op in &fusion.gates {
            match op {
                FusedOrOriginal::Fused(fused) => {
                    let m = [
                        [
                            fused.matrix.data[0][0].re as f32,
                            fused.matrix.data[0][0].im as f32,
                        ],
                        [
                            fused.matrix.data[0][1].re as f32,
                            fused.matrix.data[0][1].im as f32,
                        ],
                        [
                            fused.matrix.data[1][0].re as f32,
                            fused.matrix.data[1][0].im as f32,
                        ],
                        [
                            fused.matrix.data[1][1].re as f32,
                            fused.matrix.data[1][1].im as f32,
                        ],
                    ];
                    GateOpsF32::u(state, fused.target, &m);
                }
                FusedOrOriginal::FusedTwo(fused2) => {
                    let mut m = [[C32::new(0.0, 0.0); 4]; 4];
                    for (r, row) in m.iter_mut().enumerate() {
                        for (c, entry) in row.iter_mut().enumerate() {
                            let v = fused2.matrix.data[r][c];
                            *entry = C32::new(v.re as f32, v.im as f32);
                        }
                    }
                    GateOpsF32::u2(state, fused2.qubit_lo, fused2.qubit_hi, &m);
                }
                FusedOrOriginal::Diagonal(diag) => {
                    let amps = state.amplitudes_mut();
                    for (idx, amp) in amps.iter_mut().enumerate() {
                        let mut phase = C32::new(1.0, 0.0);

                        for &(q, p0, p1) in &diag.single_phases {
                            let p = if ((idx >> q) & 1) == 0 { p0 } else { p1 };
                            phase = phase * C32::new(p.re as f32, p.im as f32);
                        }

                        for &(q0, q1, p11) in &diag.pair_phases {
                            if ((idx >> q0) & 1) == 1 && ((idx >> q1) & 1) == 1 {
                                phase = phase * C32::new(p11.re as f32, p11.im as f32);
                            }
                        }

                        *amp = *amp * phase;
                    }
                }
                FusedOrOriginal::Original(g) => {
                    self.apply_original_gate(state, g)?;
                }
            }
        }
        Ok(())
    }

    fn apply_original_gate(&self, state: &mut QuantumStateF32, gate: &Gate) -> Result<(), String> {
        match gate.gate_type {
            GateType::H => GateOpsF32::h(state, gate.targets[0]),
            GateType::X => GateOpsF32::x(state, gate.targets[0]),
            GateType::Y => {
                let m = [[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, 0.0]];
                GateOpsF32::u(state, gate.targets[0], &m);
            }
            GateType::Z => GateOpsF32::z(state, gate.targets[0]),
            GateType::S => {
                let m = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]];
                GateOpsF32::u(state, gate.targets[0], &m);
            }
            GateType::T => {
                let a = std::f32::consts::FRAC_1_SQRT_2;
                let m = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [a, a]];
                GateOpsF32::u(state, gate.targets[0], &m);
            }
            GateType::Rx(theta) => {
                let t = theta as f32;
                let c = (t * 0.5).cos();
                let s = (t * 0.5).sin();
                let m = [[c, 0.0], [0.0, -s], [0.0, -s], [c, 0.0]];
                GateOpsF32::u(state, gate.targets[0], &m);
            }
            GateType::Ry(theta) => {
                let t = theta as f32;
                let c = (t * 0.5).cos();
                let s = (t * 0.5).sin();
                let m = [[c, 0.0], [-s, 0.0], [s, 0.0], [c, 0.0]];
                GateOpsF32::u(state, gate.targets[0], &m);
            }
            GateType::Rz(theta) => {
                let t = theta as f32;
                let c = (t * 0.5).cos();
                let s = (t * 0.5).sin();
                let m = [[c, -s], [0.0, 0.0], [0.0, 0.0], [c, s]];
                GateOpsF32::u(state, gate.targets[0], &m);
            }
            GateType::CNOT => GateOpsF32::cnot(state, gate.controls[0], gate.targets[0]),
            GateType::CZ => {
                let m = [
                    [
                        C32::new(1.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                    ],
                    [
                        C32::new(0.0, 0.0),
                        C32::new(1.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                    ],
                    [
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(1.0, 0.0),
                        C32::new(0.0, 0.0),
                    ],
                    [
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(-1.0, 0.0),
                    ],
                ];
                GateOpsF32::u2(state, gate.controls[0], gate.targets[0], &m);
            }
            GateType::SWAP => {
                let m = [
                    [
                        C32::new(1.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                    ],
                    [
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(1.0, 0.0),
                        C32::new(0.0, 0.0),
                    ],
                    [
                        C32::new(0.0, 0.0),
                        C32::new(1.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                    ],
                    [
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(1.0, 0.0),
                    ],
                ];
                GateOpsF32::u2(state, gate.targets[0], gate.targets[1], &m);
            }
            GateType::Phase(theta) => {
                let t = theta as f32;
                let m = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [t.cos(), t.sin()]];
                GateOpsF32::u(state, gate.targets[0], &m);
            }
            GateType::SX => {
                let m = [[0.5, 0.5], [0.5, -0.5], [0.5, -0.5], [0.5, 0.5]];
                GateOpsF32::u(state, gate.targets[0], &m);
            }
            GateType::ISWAP => {
                let m = [
                    [
                        C32::new(1.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                    ],
                    [
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 1.0),
                        C32::new(0.0, 0.0),
                    ],
                    [
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 1.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                    ],
                    [
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(0.0, 0.0),
                        C32::new(1.0, 0.0),
                    ],
                ];
                GateOpsF32::u2(state, gate.targets[0], gate.targets[1], &m);
            }
            GateType::CCZ => {
                let c0 = gate.controls[0];
                let c1 = gate.controls[1];
                let t = gate.targets[0];
                for (idx, amp) in state.amplitudes_mut().iter_mut().enumerate() {
                    if ((idx >> c0) & 1) == 1 && ((idx >> c1) & 1) == 1 && ((idx >> t) & 1) == 1 {
                        *amp = -*amp;
                    }
                }
            }
            _ => {
                if self.fallback_to_f64 {
                    let mut f64_state = state.to_f64();
                    apply_gate_f64(&mut f64_state, gate)?;
                    *state = QuantumStateF32::from_f64(&f64_state);
                } else {
                    return Err(format!(
                        "unsupported gate in f32 fusion path: {:?}",
                        gate.gate_type
                    ));
                }
            }
        }

        Ok(())
    }
}

fn apply_gate_f64(state: &mut QuantumState, gate: &Gate) -> Result<(), String> {
    match &gate.gate_type {
        GateType::H => GateOperations::h(state, gate.targets[0]),
        GateType::X => GateOperations::x(state, gate.targets[0]),
        GateType::Y => GateOperations::y(state, gate.targets[0]),
        GateType::Z => GateOperations::z(state, gate.targets[0]),
        GateType::S => GateOperations::s(state, gate.targets[0]),
        GateType::T => GateOperations::t(state, gate.targets[0]),
        GateType::Rx(theta) => GateOperations::rx(state, gate.targets[0], *theta),
        GateType::Ry(theta) => GateOperations::ry(state, gate.targets[0], *theta),
        GateType::Rz(theta) => GateOperations::rz(state, gate.targets[0], *theta),
        GateType::CNOT => GateOperations::cnot(state, gate.controls[0], gate.targets[0]),
        GateType::CZ => GateOperations::cz(state, gate.controls[0], gate.targets[0]),
        GateType::SWAP => GateOperations::swap(state, gate.targets[0], gate.targets[1]),
        GateType::Toffoli => {
            GateOperations::toffoli(state, gate.controls[0], gate.controls[1], gate.targets[0])
        }
        GateType::CRx(theta) => {
            GateOperations::crx(state, gate.controls[0], gate.targets[0], *theta)
        }
        GateType::CRy(theta) => {
            GateOperations::cry(state, gate.controls[0], gate.targets[0], *theta)
        }
        GateType::CRz(theta) => {
            GateOperations::crz(state, gate.controls[0], gate.targets[0], *theta)
        }
        GateType::CR(theta) => GateOperations::phase(state, gate.targets[0], *theta),
        GateType::SX => {
            let m = [
                [C64::new(0.5, 0.5), C64::new(0.5, -0.5)],
                [C64::new(0.5, -0.5), C64::new(0.5, 0.5)],
            ];
            GateOperations::u(state, gate.targets[0], &m);
        }
        GateType::Phase(theta) => GateOperations::phase(state, gate.targets[0], *theta),
        GateType::ISWAP => {
            GateOperations::swap(state, gate.targets[0], gate.targets[1]);
            let phase_i = C64::new(0.0, 1.0);
            let masks = [
                (1usize << gate.targets[0]) | (0usize << gate.targets[1]),
                (0usize << gate.targets[0]) | (1usize << gate.targets[1]),
            ];
            for (idx, amp) in state.amplitudes_mut().iter_mut().enumerate() {
                let b0 = (idx >> gate.targets[0]) & 1;
                let b1 = (idx >> gate.targets[1]) & 1;
                if (b0 ^ b1) == 1 {
                    *amp = *amp * phase_i;
                }
            }
            let _ = masks;
        }
        GateType::CCZ => {
            let c0 = gate.controls[0];
            let c1 = gate.controls[1];
            let t = gate.targets[0];
            for (idx, amp) in state.amplitudes_mut().iter_mut().enumerate() {
                if ((idx >> c0) & 1) == 1 && ((idx >> c1) & 1) == 1 && ((idx >> t) & 1) == 1 {
                    *amp = -*amp;
                }
            }
        }
        GateType::U { theta, phi, lambda } => {
            let c = (theta * 0.5).cos();
            let s = (theta * 0.5).sin();
            let m = [
                [
                    C64::new(c, 0.0),
                    C64::new(-s * lambda.cos(), -s * lambda.sin()),
                ],
                [
                    C64::new(s * phi.cos(), s * phi.sin()),
                    C64::new(c * (phi + lambda).cos(), c * (phi + lambda).sin()),
                ],
            ];
            GateOperations::u(state, gate.targets[0], &m);
        }
        _ => {
            return Err(format!(
                "unsupported gate for fallback application: {:?}",
                gate.gate_type
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_fusion::execute_fused_circuit;

    #[test]
    fn test_f32_fusion_matches_f64_reference() {
        let gates = vec![
            Gate::h(0),
            Gate::rx(0, 0.37),
            Gate::cnot(0, 1),
            Gate::rz(1, -0.21),
            Gate::h(1),
            Gate::cz(1, 2),
            Gate::ry(2, 0.48),
        ];

        let exec = F32FusionExecutor::new();
        let (state32, metrics) = exec.execute(3, &gates).expect("f32 fusion execute");

        let fusion = fuse_gates(&gates);
        let mut state64 = QuantumState::new(3);
        execute_fused_circuit(&mut state64, &fusion);

        assert!(state32.fidelity_vs_f64(&state64) > 0.999_99);
        assert_eq!(metrics.original_gates, gates.len());
        assert!(metrics.fused_ops <= metrics.original_gates);
    }

    #[test]
    fn test_f32_fusion_diagonal_support() {
        let gates = vec![
            Gate::rz(0, 0.5),
            Gate::rz(1, -0.2),
            Gate::cz(0, 1),
            Gate::phase(1, 0.8),
        ];

        let exec = F32FusionExecutor::new();
        let (state32, _) = exec.execute(2, &gates).expect("f32 fusion execute");

        let probs: f64 = state32.probabilities().iter().sum();
        assert!((probs - 1.0).abs() < 1e-5);
    }
}
