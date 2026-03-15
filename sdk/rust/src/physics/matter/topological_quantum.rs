//! Topological quantum simulation primitives.
//!
//! Includes a Fibonacci-anyon braid-space simulator and minimal string-net
//! plaquette projection utilities.

use crate::C64;

/// Two-dimensional fusion space for 3 Fibonacci anyons with total charge τ.
/// Basis: {|(ττ)_1 τ>, |(ττ)_τ τ>}.
#[derive(Clone, Debug)]
pub struct FibonacciAnyonState {
    pub amplitudes: [C64; 2],
}

impl FibonacciAnyonState {
    pub fn basis_zero() -> Self {
        Self {
            amplitudes: [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        }
    }

    pub fn basis_one() -> Self {
        Self {
            amplitudes: [C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
        }
    }

    pub fn normalize(&mut self) {
        let n = (self.amplitudes[0].norm_sqr() + self.amplitudes[1].norm_sqr()).sqrt();
        if n > 0.0 {
            self.amplitudes[0] /= n;
            self.amplitudes[1] /= n;
        }
    }

    pub fn probabilities(&self) -> [f64; 2] {
        [self.amplitudes[0].norm_sqr(), self.amplitudes[1].norm_sqr()]
    }

    /// Apply braid generator σ1 (exchange anyons 1 and 2).
    pub fn braid_sigma1(&mut self, inverse: bool) {
        let r1 = phase(-4.0 * std::f64::consts::PI / 5.0);
        let rt = phase(3.0 * std::f64::consts::PI / 5.0);
        let (a, b) = (self.amplitudes[0], self.amplitudes[1]);

        if inverse {
            self.amplitudes[0] = a * r1.conj();
            self.amplitudes[1] = b * rt.conj();
        } else {
            self.amplitudes[0] = a * r1;
            self.amplitudes[1] = b * rt;
        }
    }

    /// Apply braid generator σ2 (exchange anyons 2 and 3).
    pub fn braid_sigma2(&mut self, inverse: bool) {
        let f = fibonacci_f_matrix();
        let r = if inverse {
            [
                [phase(4.0 * std::f64::consts::PI / 5.0), C64::new(0.0, 0.0)],
                [C64::new(0.0, 0.0), phase(-3.0 * std::f64::consts::PI / 5.0)],
            ]
        } else {
            [
                [phase(-4.0 * std::f64::consts::PI / 5.0), C64::new(0.0, 0.0)],
                [C64::new(0.0, 0.0), phase(3.0 * std::f64::consts::PI / 5.0)],
            ]
        };

        // σ2 = F^{-1} R F (F is real orthogonal here => F^{-1} = F^T = F)
        let m = mat2_mul(&mat2_mul(&f, &r), &f);
        self.amplitudes = mat2_vec_mul(&m, &self.amplitudes);
    }

    /// Apply a braid word where ±1 means σ1^(±1), ±2 means σ2^(±1).
    pub fn braid_word(&mut self, word: &[i32]) -> Result<(), String> {
        for &w in word {
            match w {
                1 => self.braid_sigma1(false),
                -1 => self.braid_sigma1(true),
                2 => self.braid_sigma2(false),
                -2 => self.braid_sigma2(true),
                _ => return Err(format!("unsupported braid generator {}", w)),
            }
        }
        Ok(())
    }
}

/// Minimal Levin-Wen style string-net plaquette state.
#[derive(Clone, Debug)]
pub struct StringNetPlaquette {
    /// Edge labels around the plaquette (0 = vacuum, 1 = string).
    pub edges: Vec<u8>,
}

impl StringNetPlaquette {
    pub fn new(num_edges: usize) -> Self {
        Self {
            edges: vec![0; num_edges],
        }
    }

    /// Apply a simplified plaquette projector that mixes allowed string loops.
    pub fn apply_projector(&mut self) {
        for e in &mut self.edges {
            *e ^= 1;
        }
    }

    /// Count active strings.
    pub fn string_count(&self) -> usize {
        self.edges.iter().map(|&x| x as usize).sum()
    }
}

fn phase(theta: f64) -> C64 {
    C64::new(theta.cos(), theta.sin())
}

fn fibonacci_f_matrix() -> [[C64; 2]; 2] {
    let phi = (1.0 + 5.0_f64.sqrt()) * 0.5;
    let a = 1.0 / phi;
    let b = 1.0 / phi.sqrt();
    [
        [C64::new(a, 0.0), C64::new(b, 0.0)],
        [C64::new(b, 0.0), C64::new(-a, 0.0)],
    ]
}

fn mat2_mul(a: &[[C64; 2]; 2], b: &[[C64; 2]; 2]) -> [[C64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn mat2_vec_mul(m: &[[C64; 2]; 2], v: &[C64; 2]) -> [C64; 2] {
    [
        m[0][0] * v[0] + m[0][1] * v[1],
        m[1][0] * v[0] + m[1][1] * v[1],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_braid_generators_preserve_norm() {
        let mut s = FibonacciAnyonState {
            amplitudes: [C64::new(0.7, 0.1), C64::new(0.2, -0.3)],
        };
        s.normalize();

        let norm0 = s.probabilities()[0] + s.probabilities()[1];
        s.braid_sigma1(false);
        s.braid_sigma2(false);
        let norm1 = s.probabilities()[0] + s.probabilities()[1];

        assert!((norm0 - norm1).abs() < 1e-10);
    }

    #[test]
    fn test_sigma1_order_ten_identity() {
        let mut s = FibonacciAnyonState::basis_zero();
        for _ in 0..10 {
            s.braid_sigma1(false);
        }
        assert!((s.amplitudes[0] - C64::new(1.0, 0.0)).norm() < 1e-9);
        assert!(s.amplitudes[1].norm() < 1e-9);
    }

    #[test]
    fn test_braid_word_inverse() {
        let mut s = FibonacciAnyonState {
            amplitudes: [C64::new(0.3, -0.2), C64::new(0.6, 0.1)],
        };
        s.normalize();
        let original = s.clone();

        s.braid_word(&[1, 2, -2, -1]).unwrap();

        assert!((s.amplitudes[0] - original.amplitudes[0]).norm() < 1e-9);
        assert!((s.amplitudes[1] - original.amplitudes[1]).norm() < 1e-9);
    }

    #[test]
    fn test_string_net_projector_changes_configuration() {
        let mut p = StringNetPlaquette::new(6);
        let c0 = p.string_count();
        p.apply_projector();
        let c1 = p.string_count();
        assert_ne!(c0, c1);
    }
}
