//! Time evolution utilities (TEBD / Trotter) with MPS focus.

use crate::tensor_network::MPSSimulator;
use num_complex::Complex64;

#[derive(Clone, Debug, Default)]
pub struct LocalHamiltonian1D {
    /// ZZ couplings: (i, j, coeff)
    pub zz: Vec<(usize, usize, f64)>,
    /// X fields: (i, coeff)
    pub x: Vec<(usize, f64)>,
    /// Z fields: (i, coeff)
    pub z: Vec<(usize, f64)>,
}

fn zz_gate(theta: f64) -> [[Complex64; 4]; 4] {
    // exp(-i theta Z⊗Z / 2)
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    let p0 = Complex64::new(c, -s); // e^{-iθ/2}
    let p1 = Complex64::new(c, s); // e^{iθ/2}
    [
        [
            p0,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            p1,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            p1,
            Complex64::new(0.0, 0.0),
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            p0,
        ],
    ]
}

/// TEBD evolution for a local 1D Hamiltonian.
///
/// Uses first-order Trotter:
/// - Apply single-qubit X/Z fields
/// - Apply ZZ couplings on even bonds
/// - Apply ZZ couplings on odd bonds
pub fn tebd_evolve_mps(sim: &mut MPSSimulator, h: &LocalHamiltonian1D, dt: f64, steps: usize) {
    for _ in 0..steps {
        // Single-qubit fields
        for (q, coeff) in &h.x {
            sim.rx(*q, 2.0 * coeff * dt);
        }
        for (q, coeff) in &h.z {
            sim.rz(*q, 2.0 * coeff * dt);
        }

        // Even bonds
        for (i, j, coeff) in &h.zz {
            if (i % 2 == 0) && (*j == *i + 1) {
                let gate = zz_gate(2.0 * coeff * dt);
                sim.apply_two_qubit_gate_matrix(*i, *j, &gate);
            }
        }
        // Odd bonds
        for (i, j, coeff) in &h.zz {
            if (i % 2 == 1) && (*j == *i + 1) {
                let gate = zz_gate(2.0 * coeff * dt);
                sim.apply_two_qubit_gate_matrix(*i, *j, &gate);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tebd_small() {
        let mut sim = MPSSimulator::new(6, Some(8));
        let mut h = LocalHamiltonian1D::default();
        h.zz.push((0, 1, 1.0));
        h.zz.push((1, 2, 1.0));
        h.x.push((0, 0.5));
        tebd_evolve_mps(&mut sim, &h, 0.01, 5);
        let _ = sim.measure();
    }
}
