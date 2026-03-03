//! MPS / Metal-accelerated Quantum Neural Networks.

use crate::adaptive_mps::AdaptiveMPS;
use crate::tensor_network::MPSSimulator;
use rand::Rng;

pub trait QNNBackend {
    fn num_qubits(&self) -> usize;
    fn h(&mut self, qubit: usize);
    fn rx(&mut self, qubit: usize, theta: f64);
    fn ry(&mut self, qubit: usize, theta: f64);
    fn rz(&mut self, qubit: usize, theta: f64);
    fn cnot(&mut self, control: usize, target: usize);
}

impl QNNBackend for MPSSimulator {
    fn num_qubits(&self) -> usize {
        self.num_qubits()
    }
    fn h(&mut self, qubit: usize) {
        self.h(qubit);
    }
    fn rx(&mut self, qubit: usize, theta: f64) {
        self.rx(qubit, theta);
    }
    fn ry(&mut self, qubit: usize, theta: f64) {
        self.ry(qubit, theta);
    }
    fn rz(&mut self, qubit: usize, theta: f64) {
        self.rz(qubit, theta);
    }
    fn cnot(&mut self, control: usize, target: usize) {
        self.cnot(control, target);
    }
}

impl QNNBackend for AdaptiveMPS {
    fn num_qubits(&self) -> usize {
        self.num_qubits()
    }
    fn h(&mut self, qubit: usize) {
        self.h(qubit);
    }
    fn rx(&mut self, qubit: usize, theta: f64) {
        self.rx(qubit, theta);
    }
    fn ry(&mut self, qubit: usize, theta: f64) {
        self.ry(qubit, theta);
    }
    fn rz(&mut self, qubit: usize, theta: f64) {
        self.rz(qubit, theta);
    }
    fn cnot(&mut self, control: usize, target: usize) {
        self.cnot(control, target);
    }
}

#[cfg(target_os = "macos")]
use crate::metal_mps::MetalMPSimulator;

#[cfg(target_os = "macos")]
impl QNNBackend for MetalMPSimulator {
    fn num_qubits(&self) -> usize {
        self.num_qubits()
    }
    fn h(&mut self, qubit: usize) {
        let _ = self.h(qubit);
    }
    fn rx(&mut self, qubit: usize, theta: f64) {
        let _ = self.rx(qubit, theta);
    }
    fn ry(&mut self, qubit: usize, theta: f64) {
        let _ = self.ry(qubit, theta);
    }
    fn rz(&mut self, qubit: usize, theta: f64) {
        let _ = self.rz(qubit, theta);
    }
    fn cnot(&mut self, control: usize, target: usize) {
        let _ = self.cnot(control, target);
    }
}

#[derive(Clone, Debug)]
pub struct MPSQNNLayer {
    pub rotations: Vec<(usize, String)>,
    pub entanglement: Vec<(usize, usize)>,
    pub parameters: Vec<f64>,
}

impl MPSQNNLayer {
    pub fn new(num_qubits: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut rotations = Vec::new();
        let mut parameters = Vec::new();

        for q in 0..num_qubits {
            let gate = match q % 3 {
                0 => "rx",
                1 => "ry",
                _ => "rz",
            };
            rotations.push((q, gate.to_string()));
            parameters.push(rng.gen_range(0.0..std::f64::consts::PI));
        }

        let mut entanglement = Vec::new();
        for q in 0..num_qubits.saturating_sub(1) {
            entanglement.push((q, q + 1));
        }

        Self {
            rotations,
            entanglement,
            parameters,
        }
    }

    pub fn forward<B: QNNBackend>(&self, sim: &mut B) {
        for (i, (q, gate)) in self.rotations.iter().enumerate() {
            let theta = self.parameters[i];
            match gate.as_str() {
                "rx" => sim.rx(*q, theta),
                "ry" => sim.ry(*q, theta),
                "rz" => sim.rz(*q, theta),
                _ => {}
            }
        }
        for (c, t) in &self.entanglement {
            sim.cnot(*c, *t);
        }
    }

    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }
}

#[derive(Clone, Debug)]
pub struct MPSQuantumNN {
    pub layers: Vec<MPSQNNLayer>,
    pub num_qubits: usize,
}

impl MPSQuantumNN {
    pub fn new(num_qubits: usize, num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(MPSQNNLayer::new(num_qubits));
        }
        Self { layers, num_qubits }
    }

    pub fn forward<B: QNNBackend>(&self, sim: &mut B) {
        for layer in &self.layers {
            layer.forward(sim);
        }
    }

    pub fn total_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.num_parameters()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mps_qnn_forward() {
        let mut sim = MPSSimulator::new(6, Some(16));
        let qnn = MPSQuantumNN::new(6, 3);
        qnn.forward(&mut sim);
        let _ = sim.measure();
    }
}
