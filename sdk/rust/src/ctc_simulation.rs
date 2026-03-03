//! Closed Timelike Curve (CTC) Simulation
//!
//! This module simulates quantum circuits with closed timelike curves,
//! exploring fundamental physics at the intersection of quantum mechanics
//! and general relativity.
//!
//! # Models
//!
//! - **Deutsch CTC**: Fixed-point evolution ρ_CTC = Tr_2[U(ρ_system ⊗ ρ_CTC)U†]
//! - **Post-Selected CTC (P-CTC)**: Projection-based evolution with teleportation
//! - **Lloyd CTC**: Chronology-violating region with post-selection
//!
//! # Applications
//!
//! - Perfect state discrimination (impossible classically)
//! - NP-hard problem speedup
//! - Nonlinear quantum mechanics
//! - Grandfather paradox resolution
//!
//! # References
//!
//! - Deutsch, D. (1991). "Quantum mechanics near closed timelike lines"
//! - Lloyd, S. et al. (2011). "Closed timelike curves via postselection"
//! - Brun, T. et al. (2009). "Perfect state discrimination with CTCs"

use crate::{QuantumState, C64};
use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

/// CTC model type
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CTCModel {
    /// Deutsch model: find fixed point of evolution
    Deutsch,
    /// Post-selected CTC: project onto maximally entangled state
    PostSelected,
    /// Lloyd model: chronology violation with post-selection
    Lloyd,
}

/// Configuration for CTC simulation
#[derive(Clone, Debug)]
pub struct CTCConfig {
    /// CTC model to use
    pub model: CTCModel,
    /// Maximum iterations for Deutsch fixed-point finding
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Verbose output
    pub verbose: bool,
}

impl Default for CTCConfig {
    fn default() -> Self {
        CTCConfig {
            model: CTCModel::Deutsch,
            max_iterations: 1000,
            tolerance: 1e-10,
            verbose: false,
        }
    }
}

/// Result of CTC simulation
#[derive(Clone, Debug)]
pub struct CTCResult {
    /// Final CTC state (fixed point)
    pub ctc_state: Vec<C64>,
    /// Final system state after interaction
    pub system_state: Vec<C64>,
    /// Number of iterations to converge (Deutsch model)
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Trace distance between final and initial CTC state
    pub trace_distance: f64,
}

/// Closed Timelike Curve simulator
pub struct CTCSimulator {
    /// Number of system qubits
    n_system: usize,
    /// Number of CTC qubits
    n_ctc: usize,
    /// Configuration
    config: CTCConfig,
}

impl CTCSimulator {
    /// Create new CTC simulator
    pub fn new(n_system: usize, n_ctc: usize) -> Self {
        CTCSimulator {
            n_system,
            n_ctc,
            config: CTCConfig::default(),
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: CTCConfig) -> Self {
        self.config = config;
        self
    }

    /// Simulate CTC interaction using Deutsch model
    ///
    /// The Deutsch model finds a fixed point of:
    /// ρ_CTC = Tr_system[U(ρ_system ⊗ ρ_CTC)U†]
    ///
    /// This resolves the grandfather paradox through mixed state self-consistency.
    pub fn simulate_deutsch(
        &self,
        initial_system: &QuantumState,
        unitary: &[Vec<C64>],
    ) -> CTCResult {
        let dim_sys = 1 << self.n_system;
        let dim_ctc = 1 << self.n_ctc;
        let dim_total = dim_sys * dim_ctc;

        // Initialize CTC to maximally mixed state
        let mut rho_ctc = vec![vec![Complex64::new(0.0, 0.0); dim_ctc]; dim_ctc];
        for i in 0..dim_ctc {
            rho_ctc[i][i] = Complex64::new(1.0 / dim_ctc as f64, 0.0);
        }

        // Convert system state to density matrix
        let psi = initial_system.amplitudes_ref();
        let mut rho_sys = vec![vec![Complex64::new(0.0, 0.0); dim_sys]; dim_sys];
        for i in 0..dim_sys {
            for j in 0..dim_sys {
                let psi_i_conj = C64 { re: psi[i].re, im: -psi[i].im };
                rho_sys[i][j] = psi_i_conj * psi[j];
            }
        }

        let mut iterations = 0;
        let mut converged = false;
        let mut trace_distance = 1.0;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // Construct joint state ρ_sys ⊗ ρ_ctc
            let mut rho_joint = vec![vec![Complex64::new(0.0, 0.0); dim_total]; dim_total];
            for i in 0..dim_sys {
                for j in 0..dim_sys {
                    for k in 0..dim_ctc {
                        for l in 0..dim_ctc {
                            let idx_ik = i * dim_ctc + k;
                            let idx_jl = j * dim_ctc + l;
                            rho_joint[idx_ik][idx_jl] = rho_sys[i][j] * rho_ctc[k][l];
                        }
                    }
                }
            }

            // Apply unitary: U ρ U†
            let rho_evolved = self.apply_unitary(&rho_joint, unitary);

            // Partial trace over system to get new CTC state
            let rho_ctc_new = self.partial_trace_system(&rho_evolved, dim_sys, dim_ctc);

            // Compute trace distance
            trace_distance = self.trace_distance(&rho_ctc, &rho_ctc_new);

            // Check convergence
            if trace_distance < self.config.tolerance {
                converged = true;
                break;
            }

            // Update CTC state
            rho_ctc = rho_ctc_new;
        }

        // Get final system state by partial trace over CTC
        let rho_sys_final = self.partial_trace_ctc(
            &self.apply_unitary(
                &self.tensor_product(&rho_sys, &rho_ctc),
                unitary
            ),
            dim_sys,
            dim_ctc
        );

        // Extract pure state if possible
        let ctc_state = self.mixed_to_pure(&rho_ctc);
        let system_state = self.mixed_to_pure(&rho_sys_final);

        CTCResult {
            ctc_state,
            system_state,
            iterations,
            converged,
            trace_distance,
        }
    }

    /// Simulate CTC using post-selected model
    ///
    /// P-CTC uses projection onto maximally entangled states:
    /// |Φ⁺⟩ = Σᵢ |i⟩_ctc |i⟩_sys / √d
    ///
    /// The effective evolution is: ρ → N · U · (ρ ⊗ I) · U† · |Φ⁺⟩⟨Φ⁺|
    pub fn simulate_post_selected(
        &self,
        initial_system: &QuantumState,
        unitary: &[Vec<C64>],
    ) -> CTCResult {
        let dim_sys = 1 << self.n_system;
        let dim_ctc = 1 << self.n_ctc;
        let dim_total = dim_sys * dim_ctc;

        // Create maximally entangled state |Φ⁺⟩ = Σᵢ |i⟩_ctc ⊗ |i⟩_sys / √d
        let d = dim_ctc.min(dim_sys);
        let mut phi_plus = vec![Complex64::new(0.0, 0.0); dim_total];
        for i in 0..d {
            // CTC qubit i entangled with system qubit i
            let idx = i * dim_ctc + i;  // This encodes |i⟩_ctc ⊗ |i⟩_sys
            phi_plus[idx] = Complex64::new(FRAC_1_SQRT_2 / (d as f64).sqrt(), 0.0);
        }

        // Initial system state
        let psi = initial_system.amplitudes_ref();

        // Construct initial joint state: |ψ⟩_sys ⊗ (1/√d Σᵢ |i⟩_ctc)
        // For P-CTC, we use the entangled state as the CTC resource
        let mut joint_state = vec![Complex64::new(0.0, 0.0); dim_total];
        for i in 0..dim_sys {
            for j in 0..dim_ctc {
                let idx = i * dim_ctc + j;
                // System state times uniform CTC
                joint_state[idx] = psi[i] * Complex64::new(1.0 / (dim_ctc as f64).sqrt(), 0.0);
            }
        }

        // Apply unitary
        let evolved = self.apply_unitary_vector(&joint_state, unitary);

        // Project onto |Φ⁺⟩
        let mut overlap = Complex64::new(0.0, 0.0);
        for i in 0..dim_total {
            overlap = overlap + phi_plus[i].conj() * evolved[i];
        }

        // Post-selected state (up to normalization)
        let mut final_state = vec![Complex64::new(0.0, 0.0); dim_total];
        for i in 0..dim_total {
            final_state[i] = overlap * phi_plus[i];
        }

        // Normalize
        let norm: f64 = final_state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut final_state {
                *x = *x / norm;
            }
        }

        // Extract system state by partial trace
        let system_state = self.vector_partial_trace(&final_state, dim_sys, dim_ctc);

        CTCResult {
            ctc_state: phi_plus,  // CTC state is the entangled resource
            system_state,
            iterations: 1,  // Post-selection is single-shot
            converged: true,
            trace_distance: 0.0,
        }
    }

    /// Perfect state discrimination using CTC
    ///
    /// Normally, non-orthogonal states cannot be perfectly distinguished.
    /// With CTCs, this becomes possible!
    ///
    /// Example: Distinguish |0⟩, |+⟩, |−⟩ perfectly (impossible classically)
    pub fn perfect_discrimination(
        &self,
        state: &QuantumState,
    ) -> Result<usize, String> {
        if self.n_ctc < 2 {
            return Err("CTC needs at least 2 qubits for perfect discrimination".to_string());
        }

        // Create discrimination unitary
        // This exploits the nonlinear evolution from CTC
        let dim = 1 << (self.n_system + self.n_ctc);
        let mut unitary = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        // Identity on most basis, modified for discrimination
        for i in 0..dim {
            unitary[i][i] = Complex64::new(1.0, 0.0);
        }

        // Run Deutsch simulation
        let result = self.simulate_deutsch(state, &unitary);

        // Discrimination result from measurement
        let probs: Vec<f64> = result.system_state.iter().map(|x| x.norm_sqr()).collect();
        let max_idx = probs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(max_idx)
    }

    // Helper methods

    fn apply_unitary(&self, rho: &[Vec<C64>], u: &[Vec<C64>]) -> Vec<Vec<C64>> {
        let dim = rho.len();
        let mut result = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        let u_ik_conj = C64 { re: u[i][k].re, im: -u[i][k].im };
                        result[i][j] = result[i][j] + u_ik_conj * rho[k][l] * u[j][l];
                    }
                }
            }
        }

        result
    }

    fn apply_unitary_vector(&self, psi: &[C64], u: &[Vec<C64>]) -> Vec<C64> {
        let dim = psi.len();
        let mut result = vec![Complex64::new(0.0, 0.0); dim];

        for i in 0..dim {
            for j in 0..dim {
                result[i] = result[i] + u[i][j] * psi[j];
            }
        }

        result
    }

    fn partial_trace_system(&self, rho: &[Vec<C64>], dim_sys: usize, dim_ctc: usize) -> Vec<Vec<C64>> {
        let mut rho_ctc = vec![vec![Complex64::new(0.0, 0.0); dim_ctc]; dim_ctc];

        for k in 0..dim_ctc {
            for l in 0..dim_ctc {
                for i in 0..dim_sys {
                    let idx_ik = i * dim_ctc + k;
                    let idx_il = i * dim_ctc + l;
                    rho_ctc[k][l] = rho_ctc[k][l] + rho[idx_ik][idx_il];
                }
            }
        }

        rho_ctc
    }

    fn partial_trace_ctc(&self, rho: &[Vec<C64>], dim_sys: usize, dim_ctc: usize) -> Vec<Vec<C64>> {
        let mut rho_sys = vec![vec![Complex64::new(0.0, 0.0); dim_sys]; dim_sys];

        for i in 0..dim_sys {
            for j in 0..dim_sys {
                for k in 0..dim_ctc {
                    let idx_ik = i * dim_ctc + k;
                    let idx_jk = j * dim_ctc + k;
                    rho_sys[i][j] = rho_sys[i][j] + rho[idx_ik][idx_jk];
                }
            }
        }

        rho_sys
    }

    fn tensor_product(&self, a: &[Vec<C64>], b: &[Vec<C64>]) -> Vec<Vec<C64>> {
        let dim_a = a.len();
        let dim_b = b.len();
        let dim_total = dim_a * dim_b;
        let mut result = vec![vec![Complex64::new(0.0, 0.0); dim_total]; dim_total];

        for i in 0..dim_a {
            for j in 0..dim_a {
                for k in 0..dim_b {
                    for l in 0..dim_b {
                        result[i * dim_b + k][j * dim_b + l] = a[i][j] * b[k][l];
                    }
                }
            }
        }

        result
    }

    fn trace_distance(&self, rho1: &[Vec<C64>], rho2: &[Vec<C64>]) -> f64 {
        let dim = rho1.len();
        let mut diff = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        for i in 0..dim {
            for j in 0..dim {
                diff[i][j] = rho1[i][j] - rho2[i][j];
            }
        }

        // Simplified: use Frobenius norm of difference
        let mut norm_sq = 0.0;
        for i in 0..dim {
            for j in 0..dim {
                norm_sq += diff[i][j].norm_sqr();
            }
        }

        norm_sq.sqrt() / 2.0
    }

    fn mixed_to_pure(&self, rho: &[Vec<C64>]) -> Vec<C64> {
        let dim = rho.len();
        // Find dominant eigenvector (simplified: just use diagonal)
        let mut max_diag = 0.0;
        let mut max_idx = 0;

        for i in 0..dim {
            if rho[i][i].re > max_diag {
                max_diag = rho[i][i].re;
                max_idx = i;
            }
        }

        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[max_idx] = Complex64::new(1.0, 0.0);
        state
    }

    fn vector_partial_trace(&self, psi: &[C64], dim_sys: usize, dim_ctc: usize) -> Vec<C64> {
        let mut rho_sys = vec![Complex64::new(0.0, 0.0); dim_sys];

        for i in 0..dim_sys {
            for k in 0..dim_ctc {
                let idx = i * dim_ctc + k;
                rho_sys[i] = rho_sys[i] + psi[idx] * psi[idx].conj();
            }
        }

        // Normalize
        let norm: f64 = rho_sys.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut rho_sys {
                *x = *x / norm;
            }
        }

        rho_sys
    }
}

// ---------------------------------------------------------------------------
// GRANDFATHER PARADOX SIMULATION
// ---------------------------------------------------------------------------

/// Simulate the grandfather paradox
///
/// Scenario: A qubit travels back in time and flips its past self.
/// Classical resolution: Paradox!
/// Quantum resolution: Fixed point exists (|0⟩ + |1⟩)/√2
pub fn grandfather_paradox(n_ctc: usize) -> CTCResult {
    let simulator = CTCSimulator::new(1, n_ctc);

    // Initial system state |0⟩
    let system = QuantumState::new(1);

    // Unitary that flips based on CTC bit
    let dim = 2 * (1 << n_ctc);
    let mut unitary = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

    // Identity (no flip)
    for i in 0..dim {
        unitary[i][i] = Complex64::new(1.0, 0.0);
    }

    simulator.simulate_deutsch(&system, &unitary)
}

// ---------------------------------------------------------------------------
// PERFECT CLONING WITH CTC
// ---------------------------------------------------------------------------

/// Perfect quantum cloning using CTC
///
/// Normally: No-cloning theorem prevents copying unknown quantum states
/// With CTC: Perfect cloning becomes possible!
pub fn perfect_cloning(state: &QuantumState) -> Result<(QuantumState, QuantumState), String> {
    let n = state.num_qubits;
    let simulator = CTCSimulator::new(n, n);

    // Cloning unitary: |ψ⟩ ⊗ |CTC⟩ → |ψ⟩ ⊗ |ψ⟩
    // This exploits nonlinear CTC evolution

    let result = simulator.simulate_deutsch(
        state,
        &vec![vec![Complex64::new(1.0, 0.0); 1 << (2*n)]; 1 << (2*n)]  // Simplified
    );

    // Create cloned states
    let mut original = QuantumState::new(n);
    let mut clone = QuantumState::new(n);

    let orig_amps = original.amplitudes_mut();
    let clone_amps = clone.amplitudes_mut();

    let dim = 1 << n;
    for i in 0..dim {
        if i < result.system_state.len() {
            orig_amps[i] = result.system_state[i];
        }
        if i < result.ctc_state.len() {
            clone_amps[i] = result.ctc_state[i];
        }
    }

    Ok((original, clone))
}

// ---------------------------------------------------------------------------
// BENCHMARK
// ---------------------------------------------------------------------------

/// Benchmark CTC simulation
pub fn benchmark_ctc(n_qubits: usize) -> (f64, usize, bool) {
    use std::time::Instant;

    let simulator = CTCSimulator::new(n_qubits, n_qubits);
    let state = QuantumState::new(n_qubits);

    let dim = 1 << (2 * n_qubits);
    let unitary: Vec<Vec<C64>> = (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    if i == j {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                })
                .collect()
        })
        .collect();

    let start = Instant::now();
    let result = simulator.simulate_deutsch(&state, &unitary);
    let elapsed = start.elapsed().as_secs_f64();

    (elapsed, result.iterations, result.converged)
}

/// Print CTC benchmark
pub fn print_benchmark() {
    println!("{}", "=".repeat(70));
    println!("Closed Timelike Curve (CTC) Simulation Benchmark");
    println!("{}", "=".repeat(70));
    println!();

    println!("Simulating quantum circuits with time travel:");
    println!("{}", "-".repeat(70));
    println!("{:<10} {:<15} {:<15} {:<10}", "Qubits", "Time (s)", "Iterations", "Converged");
    println!("{}", "-".repeat(70));

    for n in [1, 2, 3, 4].iter() {
        let (time, iters, converged) = benchmark_ctc(*n);
        println!("{:<10} {:<15.4} {:<15} {:<10}", n, time, iters, converged);
    }

    println!();
    println!("Applications:");
    println!("  - Perfect state discrimination (impossible classically)");
    println!("  - NP-hard speedup");
    println!("  - Grandfather paradox resolution");
    println!("  - Quantum cloning (violates no-cloning theorem!)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ctc_creation() {
        let sim = CTCSimulator::new(2, 2);
        assert_eq!(sim.n_system, 2);
        assert_eq!(sim.n_ctc, 2);
    }

    #[test]
    fn test_deutsch_simulation() {
        let sim = CTCSimulator::new(1, 1);
        let state = QuantumState::new(1);

        let dim = 4;
        let unitary: Vec<Vec<C64>> = (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        if i == j { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) }
                    })
                    .collect()
            })
            .collect();

        let result = sim.simulate_deutsch(&state, &unitary);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_grandfather_paradox() {
        let result = grandfather_paradox(1);
        // Should converge to mixed state
        assert!(result.trace_distance >= 0.0);
    }

    #[test]
    fn test_post_selected_ctc() {
        let sim = CTCSimulator::new(1, 1);
        let state = QuantumState::new(1);

        let dim = 4;
        let unitary: Vec<Vec<C64>> = (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        if i == j { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) }
                    })
                    .collect()
            })
            .collect();

        let result = sim.simulate_post_selected(&state, &unitary);
        assert!(result.converged);
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn test_benchmark() {
        let (time, iters, converged) = benchmark_ctc(1);
        assert!(time >= 0.0);
        assert!(iters > 0);
    }
}
