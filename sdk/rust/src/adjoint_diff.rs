//! Adjoint differentiation for variational quantum circuits.

use crate::gates::{Gate, GateType};
use crate::{GateOperations, QuantumState, C64};

#[derive(Clone, Debug)]
pub enum AdjointOp {
    Fixed(Gate),
    Rx { qubit: usize, param: usize },
    Ry { qubit: usize, param: usize },
    Rz { qubit: usize, param: usize },
}

#[derive(Clone, Copy, Debug)]
pub enum Observable {
    PauliZ(usize),
}

#[derive(Clone, Debug)]
pub struct AdjointCircuit {
    pub num_qubits: usize,
    pub ops: Vec<AdjointOp>,
}

impl AdjointCircuit {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            ops: Vec::new(),
        }
    }

    pub fn add_op(&mut self, op: AdjointOp) {
        self.ops.push(op);
    }

    pub fn forward(&self, params: &[f64]) -> Result<QuantumState, String> {
        let mut state = QuantumState::new(self.num_qubits);
        for op in &self.ops {
            apply_op(&mut state, op, params)?;
        }
        Ok(state)
    }

    pub fn expectation(&self, params: &[f64], obs: Observable) -> Result<f64, String> {
        let state = self.forward(params)?;
        Ok(match obs {
            Observable::PauliZ(q) => state.expectation_z(q),
        })
    }

    /// Reverse-mode adjoint gradient.
    pub fn gradient(&self, params: &[f64], obs: Observable) -> Result<Vec<f64>, String> {
        let num_params = self
            .ops
            .iter()
            .filter_map(|op| match op {
                AdjointOp::Rx { param, .. } => Some(*param),
                AdjointOp::Ry { param, .. } => Some(*param),
                AdjointOp::Rz { param, .. } => Some(*param),
                AdjointOp::Fixed(_) => None,
            })
            .max()
            .map(|i| i + 1)
            .unwrap_or(0);

        let mut forward_states = Vec::with_capacity(self.ops.len() + 1);
        let mut state = QuantumState::new(self.num_qubits);
        forward_states.push(state.clone());

        for op in &self.ops {
            apply_op(&mut state, op, params)?;
            forward_states.push(state.clone());
        }

        let mut lambda = apply_observable(&state, obs)?;
        let mut grad = vec![0.0; num_params];

        for (i, op) in self.ops.iter().enumerate().rev() {
            let psi_before = &forward_states[i];

            match op {
                AdjointOp::Rx { qubit, param } => {
                    let mut dpsi = psi_before.clone();
                    apply_derivative_rx(&mut dpsi, *qubit, params[*param]);
                    grad[*param] += 2.0 * inner_product_real(&lambda, &dpsi);
                }
                AdjointOp::Ry { qubit, param } => {
                    let mut dpsi = psi_before.clone();
                    apply_derivative_ry(&mut dpsi, *qubit, params[*param]);
                    grad[*param] += 2.0 * inner_product_real(&lambda, &dpsi);
                }
                AdjointOp::Rz { qubit, param } => {
                    let mut dpsi = psi_before.clone();
                    apply_derivative_rz(&mut dpsi, *qubit, params[*param]);
                    grad[*param] += 2.0 * inner_product_real(&lambda, &dpsi);
                }
                AdjointOp::Fixed(_) => {}
            }

            apply_inverse_op(&mut lambda, op, params)?;
        }

        Ok(grad)
    }
}

fn apply_observable(state: &QuantumState, obs: Observable) -> Result<QuantumState, String> {
    let mut out = state.clone();
    match obs {
        Observable::PauliZ(q) => {
            let mask = 1usize << q;
            for (i, amp) in out.amplitudes_mut().iter_mut().enumerate() {
                if (i & mask) != 0 {
                    *amp = -*amp;
                }
            }
        }
    }
    Ok(out)
}

fn apply_op(state: &mut QuantumState, op: &AdjointOp, params: &[f64]) -> Result<(), String> {
    match op {
        AdjointOp::Fixed(g) => apply_gate(state, g),
        AdjointOp::Rx { qubit, param } => {
            GateOperations::rx(state, *qubit, params[*param]);
            Ok(())
        }
        AdjointOp::Ry { qubit, param } => {
            GateOperations::ry(state, *qubit, params[*param]);
            Ok(())
        }
        AdjointOp::Rz { qubit, param } => {
            GateOperations::rz(state, *qubit, params[*param]);
            Ok(())
        }
    }
}

fn apply_inverse_op(
    state: &mut QuantumState,
    op: &AdjointOp,
    params: &[f64],
) -> Result<(), String> {
    match op {
        AdjointOp::Fixed(g) => {
            let inv = Gate {
                gate_type: g.gate_type.inverse(),
                targets: g.targets.clone(),
                controls: g.controls.clone(),
                params: g.params.clone(),
            };
            apply_gate(state, &inv)
        }
        AdjointOp::Rx { qubit, param } => {
            GateOperations::rx(state, *qubit, -params[*param]);
            Ok(())
        }
        AdjointOp::Ry { qubit, param } => {
            GateOperations::ry(state, *qubit, -params[*param]);
            Ok(())
        }
        AdjointOp::Rz { qubit, param } => {
            GateOperations::rz(state, *qubit, -params[*param]);
            Ok(())
        }
    }
}

fn apply_derivative_rx(state: &mut QuantumState, qubit: usize, theta: f64) {
    let c = (theta * 0.5).cos();
    let s = (theta * 0.5).sin();
    let m = [
        [C64::new(-0.5 * s, 0.0), C64::new(0.0, -0.5 * c)],
        [C64::new(0.0, -0.5 * c), C64::new(-0.5 * s, 0.0)],
    ];
    GateOperations::u(state, qubit, &m);
}

fn apply_derivative_ry(state: &mut QuantumState, qubit: usize, theta: f64) {
    let c = (theta * 0.5).cos();
    let s = (theta * 0.5).sin();
    let m = [
        [C64::new(-0.5 * s, 0.0), C64::new(-0.5 * c, 0.0)],
        [C64::new(0.5 * c, 0.0), C64::new(-0.5 * s, 0.0)],
    ];
    GateOperations::u(state, qubit, &m);
}

fn apply_derivative_rz(state: &mut QuantumState, qubit: usize, theta: f64) {
    let a = theta * 0.5;
    let e_m = C64::new(a.cos(), -a.sin());
    let e_p = C64::new(a.cos(), a.sin());
    let m = [
        [C64::new(0.0, -0.5) * e_m, C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(0.0, 0.5) * e_p],
    ];
    GateOperations::u(state, qubit, &m);
}

fn inner_product_real(a: &QuantumState, b: &QuantumState) -> f64 {
    let mut re = 0.0;
    for (aa, bb) in a.amplitudes_ref().iter().zip(b.amplitudes_ref().iter()) {
        re += aa.re * bb.re + aa.im * bb.im;
    }
    re
}

fn apply_gate(state: &mut QuantumState, gate: &Gate) -> Result<(), String> {
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
        GateType::Phase(theta) => GateOperations::phase(state, gate.targets[0], *theta),
        _ => {
            return Err(format!(
                "unsupported fixed gate in adjoint engine: {:?}",
                gate.gate_type
            ))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_diff(circ: &AdjointCircuit, params: &[f64], idx: usize, eps: f64) -> f64 {
        let mut p1 = params.to_vec();
        let mut p2 = params.to_vec();
        p1[idx] += eps;
        p2[idx] -= eps;
        let f1 = circ.expectation(&p1, Observable::PauliZ(0)).unwrap();
        let f2 = circ.expectation(&p2, Observable::PauliZ(0)).unwrap();
        (f1 - f2) / (2.0 * eps)
    }

    #[test]
    fn test_single_param_rx_gradient() {
        let mut c = AdjointCircuit::new(1);
        c.add_op(AdjointOp::Rx { qubit: 0, param: 0 });

        let theta = 0.37;
        let g = c.gradient(&[theta], Observable::PauliZ(0)).unwrap();

        assert!((g[0] + theta.sin()).abs() < 1e-6);
    }

    #[test]
    fn test_matches_finite_difference() {
        let mut c = AdjointCircuit::new(1);
        c.add_op(AdjointOp::Ry { qubit: 0, param: 0 });
        c.add_op(AdjointOp::Rz { qubit: 0, param: 1 });
        c.add_op(AdjointOp::Rx { qubit: 0, param: 2 });

        let params = [0.3, -0.2, 0.5];
        let g = c.gradient(&params, Observable::PauliZ(0)).unwrap();

        for i in 0..params.len() {
            let fd = finite_diff(&c, &params, i, 1e-6);
            assert!((g[i] - fd).abs() < 1e-4);
        }
    }
}
