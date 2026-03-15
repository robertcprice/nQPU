//! Metal-accelerated MPS simulator (GPU + CPU fallback for SVD).
//!
//! This integrates GPU kernels for single-qubit ops and tensor contraction
//! with CPU fallback for SVD/truncation to keep correctness while Metal SVD
//! is still under development.

use crate::gpu_mps::GPUMPSState;
use num_complex::Complex64;

/// Metal-backed MPS simulator.
pub struct MetalMPSimulator {
    gpu: GPUMPSState,
    num_qubits: usize,
}

impl MetalMPSimulator {
    /// Create a new Metal MPS simulator.
    pub fn new(num_qubits: usize, bond_dim: usize) -> Result<Self, String> {
        let gpu = GPUMPSState::new(num_qubits, bond_dim)?;
        Ok(Self { gpu, num_qubits })
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Apply Hadamard.
    pub fn h(&mut self, qubit: usize) -> Result<(), String> {
        let inv = 1.0 / 2.0_f64.sqrt();
        let h = [
            [Complex64::new(inv, 0.0), Complex64::new(inv, 0.0)],
            [Complex64::new(inv, 0.0), Complex64::new(-inv, 0.0)],
        ];
        self.gpu.apply_single_qubit_gate(qubit, h)
    }

    /// Apply X.
    pub fn x(&mut self, qubit: usize) -> Result<(), String> {
        let x = [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ];
        self.gpu.apply_single_qubit_gate(qubit, x)
    }

    /// Apply Y.
    pub fn y(&mut self, qubit: usize) -> Result<(), String> {
        let y = [
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
            [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
        ];
        self.gpu.apply_single_qubit_gate(qubit, y)
    }

    /// Apply Z.
    pub fn z(&mut self, qubit: usize) -> Result<(), String> {
        let z = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
        ];
        self.gpu.apply_single_qubit_gate(qubit, z)
    }

    /// Apply S.
    pub fn s(&mut self, qubit: usize) -> Result<(), String> {
        let s = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
        ];
        self.gpu.apply_single_qubit_gate(qubit, s)
    }

    /// Apply T.
    pub fn t(&mut self, qubit: usize) -> Result<(), String> {
        let phase = std::f64::consts::FRAC_PI_4;
        let t = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(phase.cos(), phase.sin()),
            ],
        ];
        self.gpu.apply_single_qubit_gate(qubit, t)
    }

    /// Apply Rx.
    pub fn rx(&mut self, qubit: usize, theta: f64) -> Result<(), String> {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let rx = [
            [Complex64::new(c, 0.0), Complex64::new(0.0, -s)],
            [Complex64::new(0.0, -s), Complex64::new(c, 0.0)],
        ];
        self.gpu.apply_single_qubit_gate(qubit, rx)
    }

    /// Apply Ry.
    pub fn ry(&mut self, qubit: usize, theta: f64) -> Result<(), String> {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let ry = [
            [Complex64::new(c, 0.0), Complex64::new(-s, 0.0)],
            [Complex64::new(s, 0.0), Complex64::new(c, 0.0)],
        ];
        self.gpu.apply_single_qubit_gate(qubit, ry)
    }

    /// Apply Rz.
    pub fn rz(&mut self, qubit: usize, theta: f64) -> Result<(), String> {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let rz = [
            [Complex64::new(c, -s), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(c, s)],
        ];
        self.gpu.apply_single_qubit_gate(qubit, rz)
    }

    /// Apply CNOT (adjacent qubits recommended).
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<(), String> {
        let cnot = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        ];
        self.gpu.apply_two_qubit_gate(control, target, cnot)
    }

    /// Measure all qubits (CPU fallback via MPS measurement).
    pub fn measure(&mut self) -> Result<usize, String> {
        // Avoid full state-vector expansion; use MPS measurement.
        let mut cpu_mps = self.gpu.to_cpu_mps()?;
        let mut result: usize = 0;
        let n_qubits = cpu_mps.num_qubits();
        let usable_bits = std::mem::size_of::<usize>() * 8;

        for i in 0..n_qubits.min(usable_bits) {
            let (bit, collapsed) = cpu_mps.measure_qubit(i);
            cpu_mps = collapsed;
            let shift = if n_qubits > usable_bits {
                usable_bits - 1 - i
            } else {
                n_qubits - 1 - i
            };
            result |= bit << shift;
        }
        Ok(result)
    }

    /// Return maximum current bond dimension (CPU readback).
    pub fn max_bond_dim(&self) -> Result<usize, String> {
        let cpu_mps = self.gpu.to_cpu_mps()?;
        Ok(cpu_mps.max_current_bond_dim())
    }

    /// Convert to full state vector (CPU fallback).
    pub fn to_state_vector(&self) -> Result<Vec<Complex64>, String> {
        let cpu_mps = self.gpu.to_cpu_mps()?;
        Ok(cpu_mps.to_state_vector())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_mps_basic() {
        if let Ok(mut sim) = MetalMPSimulator::new(6, 8) {
            let _ = sim.h(0);
            let _ = sim.cnot(0, 1);
            let _ = sim.rz(2, 0.3);
            let _ = sim.measure();
        }
    }
}
