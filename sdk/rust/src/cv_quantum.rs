//! Continuous-variable (CV) quantum simulation with Gaussian states.

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Gaussian CV state in quadrature form.
/// Ordering: [x0, p0, x1, p1, ...].
#[derive(Clone, Debug)]
pub struct CvGaussianState {
    pub num_modes: usize,
    pub mean: Array1<f64>,
    pub covariance: Array2<f64>,
}

impl CvGaussianState {
    /// Create an n-mode vacuum state.
    pub fn vacuum(num_modes: usize) -> Self {
        let dim = 2 * num_modes;
        let mean = Array1::zeros(dim);
        let mut covariance = Array2::zeros((dim, dim));
        for i in 0..dim {
            covariance[[i, i]] = 0.5;
        }
        Self {
            num_modes,
            mean,
            covariance,
        }
    }

    /// Apply phase-space displacement D(alpha).
    pub fn displace(&mut self, mode: usize, alpha_re: f64, alpha_im: f64) {
        let idx = 2 * mode;
        self.mean[idx] += (2.0f64).sqrt() * alpha_re;
        self.mean[idx + 1] += (2.0f64).sqrt() * alpha_im;
    }

    /// Apply single-mode squeezing S(r, phi).
    pub fn squeeze(&mut self, mode: usize, r: f64, phi: f64) {
        let c = phi.cos();
        let s = phi.sin();
        let e_m = (-r).exp();
        let e_p = r.exp();

        // R(phi) diag(e^-r, e^r) R(-phi)
        let s00 = c * c * e_m + s * s * e_p;
        let s01 = c * s * (e_m - e_p);
        let s10 = s01;
        let s11 = s * s * e_m + c * c * e_p;

        let mut s_mat = Array2::eye(2 * self.num_modes);
        let i = 2 * mode;
        s_mat[[i, i]] = s00;
        s_mat[[i, i + 1]] = s01;
        s_mat[[i + 1, i]] = s10;
        s_mat[[i + 1, i + 1]] = s11;

        self.apply_symplectic(&s_mat);
    }

    /// Apply two-mode beamsplitter B(theta, phi).
    pub fn beamsplitter(&mut self, mode_a: usize, mode_b: usize, theta: f64, phi: f64) {
        let c = theta.cos();
        let s = theta.sin();
        let cp = phi.cos();
        let sp = phi.sin();

        let mut s_mat = Array2::eye(2 * self.num_modes);
        let a = 2 * mode_a;
        let b = 2 * mode_b;

        // Block acting on [xa, pa, xb, pb]
        // Derived from a unitary with transmissivity cos(theta) and phase phi.
        s_mat[[a, a]] = c;
        s_mat[[a, b]] = s * cp;
        s_mat[[a, b + 1]] = -s * sp;

        s_mat[[a + 1, a + 1]] = c;
        s_mat[[a + 1, b]] = s * sp;
        s_mat[[a + 1, b + 1]] = s * cp;

        s_mat[[b, a]] = -s * cp;
        s_mat[[b, a + 1]] = -s * sp;
        s_mat[[b, b]] = c;

        s_mat[[b + 1, a]] = s * sp;
        s_mat[[b + 1, a + 1]] = -s * cp;
        s_mat[[b + 1, b + 1]] = c;

        self.apply_symplectic(&s_mat);
    }

    /// Apply a general symplectic transform S.
    pub fn apply_symplectic(&mut self, s: &Array2<f64>) {
        self.mean = s.dot(&self.mean);
        self.covariance = s.dot(&self.covariance).dot(&s.t());
    }

    /// Mean photon number in a mode.
    pub fn mean_photon(&self, mode: usize) -> f64 {
        let i = 2 * mode;
        let x = self.mean[i];
        let p = self.mean[i + 1];
        let vxx = self.covariance[[i, i]];
        let vpp = self.covariance[[i + 1, i + 1]];
        0.5 * (vxx + vpp + x * x + p * p - 1.0)
    }

    /// Total mean photon number across all modes.
    pub fn total_mean_photon(&self) -> f64 {
        (0..self.num_modes).map(|m| self.mean_photon(m)).sum()
    }

    /// Simple homodyne sample of x quadrature.
    pub fn sample_x(&self, mode: usize) -> f64 {
        let i = 2 * mode;
        let mean = self.mean[i];
        let var = self.covariance[[i, i]].max(1e-12);
        mean + var.sqrt() * standard_normal()
    }
}

/// Gaussian Boson Sampling helper (threshold detector model).
#[derive(Clone, Debug)]
pub struct GaussianBosonSampler {
    pub state: CvGaussianState,
}

impl GaussianBosonSampler {
    pub fn new(state: CvGaussianState) -> Self {
        Self { state }
    }

    /// Sample threshold click patterns from per-mode mean-photon proxy.
    pub fn sample_click_patterns(&self, shots: usize) -> Vec<Vec<bool>> {
        let means: Vec<f64> = (0..self.state.num_modes)
            .map(|m| self.state.mean_photon(m).max(0.0))
            .collect();

        let mut out = Vec::with_capacity(shots);
        for _ in 0..shots {
            let mut click = Vec::with_capacity(self.state.num_modes);
            for &mu in &means {
                // Poisson(n=0) probability exp(-mu), threshold click if n>0.
                let p_click = 1.0 - (-mu).exp();
                click.push(rand::random::<f64>() < p_click);
            }
            out.push(click);
        }
        out
    }
}

#[inline]
fn standard_normal() -> f64 {
    // Box-Muller transform.
    let u1 = (1.0 - rand::random::<f64>()).clamp(1e-12, 1.0);
    let u2 = rand::random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vacuum_photon_number_zero() {
        let s = CvGaussianState::vacuum(3);
        for m in 0..3 {
            assert!(s.mean_photon(m).abs() < 1e-9);
        }
    }

    #[test]
    fn test_displacement_increases_photon_number() {
        let mut s = CvGaussianState::vacuum(1);
        s.displace(0, 0.8, -0.3);
        assert!(s.mean_photon(0) > 0.0);
    }

    #[test]
    fn test_squeezing_changes_covariance() {
        let mut s = CvGaussianState::vacuum(1);
        let vxx0 = s.covariance[[0, 0]];
        let vpp0 = s.covariance[[1, 1]];

        s.squeeze(0, 0.7, 0.0);

        assert!(s.covariance[[0, 0]] < vxx0);
        assert!(s.covariance[[1, 1]] > vpp0);
    }

    #[test]
    fn test_beamsplitter_conserves_total_mean_photon() {
        let mut s = CvGaussianState::vacuum(2);
        s.displace(0, 1.0, 0.0);
        let before = s.total_mean_photon();

        s.beamsplitter(0, 1, std::f64::consts::FRAC_PI_4, 0.0);
        let after = s.total_mean_photon();

        assert!((before - after).abs() < 1e-8);
    }

    #[test]
    fn test_gbs_sampler_output_shape() {
        let s = CvGaussianState::vacuum(4);
        let gbs = GaussianBosonSampler::new(s);
        let samples = gbs.sample_click_patterns(16);
        assert_eq!(samples.len(), 16);
        assert!(samples.iter().all(|v| v.len() == 4));
    }
}
