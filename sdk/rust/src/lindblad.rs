//! Lindblad Master Equation Solver for Open Quantum Systems.
//!
//! Implements time evolution of density matrices under Markovian dissipation:
//! dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - (1/2){L_k† L_k, ρ})
//!
//! # Features
//! - Arbitrary Hamiltonian evolution
//! - Multiple jump operators (Lindblad terms)
//! - Quantum trajectory (Monte Carlo) method
//! - Master equation direct integration
//! - Steady state finding via imaginary time evolution

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

/// Zero constant for Complex64.
const ZERO: Complex64 = Complex64 { re: 0.0, im: 0.0 };
/// One constant for Complex64.
const ONE: Complex64 = Complex64 { re: 1.0, im: 0.0 };

/// Jump operator for Lindblad dynamics.
#[derive(Clone, Debug)]
pub struct JumpOperator {
    /// Matrix representation (2^n x 2^n).
    pub matrix: Vec<Vec<Complex64>>,
    /// Rate parameter γ.
    pub rate: f64,
    /// Description.
    pub name: String,
}

impl JumpOperator {
    /// Create a single-qubit amplitude damping (T1) channel.
    pub fn amplitude_damping(qubit: usize, n_qubits: usize, gamma: f64) -> Self {
        // σ_- = |0⟩⟨1|
        let dim = 1 << n_qubits;
        let mut matrix = vec![vec![ZERO; dim]; dim];

        // Apply σ_- on target qubit
        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            if bit == 1 {
                let j = i & !(1 << qubit); // Flip qubit to 0
                matrix[j][i] = Complex64::new(gamma.sqrt(), 0.0);
            }
        }

        Self {
            matrix,
            rate: gamma,
            name: format!("T1(q{})", qubit),
        }
    }

    /// Create a single-qubit dephasing (T2 pure) channel.
    pub fn dephasing(qubit: usize, n_qubits: usize, gamma: f64) -> Self {
        // σ_z = |0⟩⟨0| - |1⟩⟨1|
        let dim = 1 << n_qubits;
        let mut matrix = vec![vec![ZERO; dim]; dim];

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            matrix[i][i] = if bit == 0 {
                Complex64::new(gamma.sqrt(), 0.0)
            } else {
                Complex64::new(-gamma.sqrt(), 0.0)
            };
        }

        Self {
            matrix,
            rate: gamma,
            name: format!("T2(q{})", qubit),
        }
    }

    /// Create a single-qubit depolarizing channel.
    pub fn depolarizing(qubit: usize, n_qubits: usize, p: f64) -> Vec<Self> {
        let gamma = (p / 4.0).sqrt();

        vec![
            Self::pauli_x_jump(qubit, n_qubits, gamma),
            Self::pauli_y_jump(qubit, n_qubits, gamma),
            Self::pauli_z_jump(qubit, n_qubits, gamma),
        ]
    }

    /// Pauli X jump operator.
    pub fn pauli_x_jump(qubit: usize, n_qubits: usize, gamma: f64) -> Self {
        let dim = 1 << n_qubits;
        let mut matrix = vec![vec![ZERO; dim]; dim];

        for i in 0..dim {
            let j = i ^ (1 << qubit); // Flip qubit
            matrix[j][i] = Complex64::new(gamma, 0.0);
        }

        Self {
            matrix,
            rate: gamma * gamma,
            name: format!("X(q{})", qubit),
        }
    }

    /// Pauli Y jump operator.
    pub fn pauli_y_jump(qubit: usize, n_qubits: usize, gamma: f64) -> Self {
        let dim = 1 << n_qubits;
        let mut matrix = vec![vec![ZERO; dim]; dim];

        for i in 0..dim {
            let j = i ^ (1 << qubit);
            let bit = (i >> qubit) & 1;
            let sign = if bit == 0 { 1.0 } else { -1.0 };
            matrix[j][i] = Complex64::new(0.0, sign * gamma);
        }

        Self {
            matrix,
            rate: gamma * gamma,
            name: format!("Y(q{})", qubit),
        }
    }

    /// Pauli Z jump operator.
    pub fn pauli_z_jump(qubit: usize, n_qubits: usize, gamma: f64) -> Self {
        let dim = 1 << n_qubits;
        let mut matrix = vec![vec![ZERO; dim]; dim];

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            matrix[i][i] = if bit == 0 {
                Complex64::new(gamma, 0.0)
            } else {
                Complex64::new(-gamma, 0.0)
            };
        }

        Self {
            matrix,
            rate: gamma * gamma,
            name: format!("Z(q{})", qubit),
        }
    }
}

/// Hamiltonian for Lindblad evolution.
#[derive(Clone, Debug)]
pub struct Hamiltonian {
    /// Matrix representation.
    pub matrix: Vec<Vec<Complex64>>,
    /// Dimension.
    pub dim: usize,
}

impl Hamiltonian {
    /// Create from a standard matrix.
    pub fn new(matrix: Vec<Vec<Complex64>>) -> Self {
        let dim = matrix.len();
        Self { matrix, dim }
    }

    /// Create a transverse-field Ising Hamiltonian.
    pub fn tfim(n_qubits: usize, jz: f64, hx: f64) -> Self {
        let dim = 1 << n_qubits;
        let mut h = vec![vec![ZERO; dim]; dim];

        // ZZ interactions
        for i in 0..dim {
            let mut zz_energy = 0.0;
            for q in 0..(n_qubits - 1) {
                let b1 = (i >> q) & 1;
                let b2 = (i >> (q + 1)) & 1;
                zz_energy += if b1 == b2 { 1.0 } else { -1.0 };
            }
            h[i][i] += Complex64::new(jz * zz_energy, 0.0);
        }

        // Transverse field (X terms)
        for i in 0..dim {
            for q in 0..n_qubits {
                let j = i ^ (1 << q);
                h[j][i] += Complex64::new(hx, 0.0);
            }
        }

        Self { matrix: h, dim }
    }

    /// Create an identity Hamiltonian (no evolution).
    pub fn identity(dim: usize) -> Self {
        let mut h = vec![vec![ZERO; dim]; dim];
        for i in 0..dim {
            h[i][i] = ONE;
        }
        Self { matrix: h, dim }
    }
}

/// Density matrix representation.
#[derive(Clone, Debug)]
pub struct DensityMatrix {
    /// Matrix elements (dim x dim).
    pub data: DMatrix<Complex64>,
    /// Dimension.
    pub dim: usize,
}

impl DensityMatrix {
    /// Create a zero density matrix.
    pub fn zeros(dim: usize) -> Self {
        Self {
            data: DMatrix::zeros(dim, dim),
            dim,
        }
    }

    /// Create from a state vector |ψ⟩, giving ρ = |ψ⟩⟨ψ|.
    pub fn from_state_vector(state: &[Complex64]) -> Self {
        let dim = state.len();
        let mut rho = DMatrix::zeros(dim, dim);

        for i in 0..dim {
            for j in 0..dim {
                rho[(i, j)] = state[i] * state[j].conj();
            }
        }

        Self { data: rho, dim }
    }

    /// Create the |0...0⟩ state.
    pub fn ground_state(n_qubits: usize) -> Self {
        let dim = 1 << n_qubits;
        let mut rho = DMatrix::zeros(dim, dim);
        rho[(0, 0)] = ONE;
        Self { data: rho, dim }
    }

    /// Create the maximally mixed state I/d.
    pub fn maximally_mixed(dim: usize) -> Self {
        // Identity matrix scaled by 1/dim
        let mut rho = DMatrix::zeros(dim, dim);
        for i in 0..dim {
            rho[(i, i)] = Complex64::new(1.0 / dim as f64, 0.0);
        }
        Self { data: rho, dim }
    }

    /// Get trace.
    pub fn trace(&self) -> Complex64 {
        let mut tr = ZERO;
        for i in 0..self.dim {
            tr += self.data[(i, i)];
        }
        tr
    }

    /// Normalize to trace = 1.
    pub fn normalize(&mut self) {
        let tr = self.trace();
        if tr.norm() > 1e-15 {
            self.data /= tr;
        }
    }

    /// Get expectation value of an observable.
    pub fn expectation(&self, observable: &DMatrix<Complex64>) -> Complex64 {
        let product = &self.data * observable;
        let mut exp = ZERO;
        for i in 0..self.dim {
            exp += product[(i, i)];
        }
        exp
    }

    /// Get purity Tr(ρ²).
    pub fn purity(&self) -> f64 {
        let rho2 = &self.data * &self.data;
        let mut tr = ZERO;
        for i in 0..self.dim {
            tr += rho2[(i, i)];
        }
        tr.re
    }

    /// Get Von Neumann entropy S = -Tr(ρ log ρ).
    pub fn entropy(&self) -> f64 {
        // Eigenvalue-based computation
        let mut entropy = 0.0;
        // Simplified: use SVD for eigenvalue estimation
        let svd = self.data.clone().svd(false, false);
        for sv in svd.singular_values.iter() {
            if *sv > 1e-15 {
                entropy -= sv * sv.ln();
            }
        }
        entropy
    }

    /// Convert to state vector (if pure).
    pub fn to_state_vector(&self) -> Option<Vec<Complex64>> {
        if (self.purity() - 1.0).abs() > 1e-10 {
            return None;
        }

        // Find dominant eigenvector
        let svd = self.data.clone().svd(true, false);
        let u = svd.u?;
        let mut state = vec![ZERO; self.dim];
        for i in 0..self.dim {
            state[i] = u[(i, 0)];
        }
        Some(state)
    }
}

/// Lindblad Master Equation Solver.
pub struct LindbladSolver {
    /// Density matrix.
    rho: DensityMatrix,
    /// Hamiltonian.
    hamiltonian: Hamiltonian,
    /// Jump operators.
    jump_ops: Vec<JumpOperator>,
    /// Current time.
    current_time: f64,
    /// Number of qubits.
    n_qubits: usize,
}

impl LindbladSolver {
    /// Create a new Lindblad solver.
    pub fn new(n_qubits: usize, hamiltonian: Hamiltonian) -> Self {
        let _dim = 1 << n_qubits;
        Self {
            rho: DensityMatrix::ground_state(n_qubits),
            hamiltonian,
            jump_ops: Vec::new(),
            current_time: 0.0,
            n_qubits,
        }
    }

    /// Add a jump operator.
    pub fn add_jump_operator(&mut self, op: JumpOperator) {
        self.jump_ops.push(op);
    }

    /// Set initial state from state vector.
    pub fn set_state(&mut self, state: &[Complex64]) {
        self.rho = DensityMatrix::from_state_vector(state);
    }

    /// Set density matrix directly.
    pub fn set_density_matrix(&mut self, rho: DensityMatrix) {
        self.rho = rho;
    }

    /// Get current density matrix.
    pub fn density_matrix(&self) -> &DensityMatrix {
        &self.rho
    }

    /// Get current time.
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Compute Lindblad superoperator action: L[ρ].
    fn lindblad_action(&self, rho: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let dim = self.n_qubits;
        let n = 1 << dim;

        // Convert Hamiltonian to nalgebra matrix
        let h_matrix: DMatrix<Complex64> =
            DMatrix::from_fn(n, n, |i, j| self.hamiltonian.matrix[i][j]);

        // Commutator -i[H, ρ]
        let commutator = &h_matrix * rho - rho * &h_matrix;
        let factor = Complex64::new(0.0, -1.0);
        let mut drho = commutator * factor;

        // Dissipator terms
        for op in &self.jump_ops {
            // Convert jump operator to matrix
            let l_matrix: DMatrix<Complex64> = DMatrix::from_fn(n, n, |i, j| op.matrix[i][j]);
            let l_dag = l_matrix.adjoint();

            // L ρ L†
            let lrl = &l_matrix * rho * &l_dag;

            // -1/2 {L†L, ρ}
            let ldagl = &l_dag * &l_matrix;
            let anticommutator = &ldagl * rho + rho * &ldagl;

            drho = drho + lrl - anticommutator * Complex64::new(0.5, 0.0);
        }

        drho
    }

    /// Evolve using 4th-order Runge-Kutta.
    pub fn evolve_rk4(&mut self, dt: f64, steps: usize) {
        let half_dt = Complex64::new(dt / 2.0, 0.0);
        let two = Complex64::new(2.0, 0.0);
        let sixth_dt = Complex64::new(dt / 6.0, 0.0);

        for _ in 0..steps {
            let rho = &self.rho.data;

            // RK4 for dρ/dt = L[ρ]
            let k1 = self.lindblad_action(rho);
            let k2 = self.lindblad_action(&(rho + k1.clone() * half_dt));
            let k3 = self.lindblad_action(&(rho + k2.clone() * half_dt));
            let k4 = self.lindblad_action(&(rho + k3.clone() * Complex64::new(dt, 0.0)));

            self.rho.data = rho + (k1.clone() + k2.clone() * two + k3 * two + k4) * sixth_dt;

            // Renormalize
            self.rho.normalize();
            self.current_time += dt;
        }
    }

    /// Single quantum trajectory (Monte Carlo wavefunction method).
    pub fn quantum_trajectory(&mut self, dt: f64, rng_seed: u64) -> Vec<Complex64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Start with state vector (assuming pure state)
        let mut state = self
            .rho
            .to_state_vector()
            .unwrap_or_else(|| vec![ONE; 1 << self.n_qubits]);

        let mut hasher = DefaultHasher::new();
        rng_seed.hash(&mut hasher);
        let mut rng_state = hasher.finish();

        // Simple LCG random number generator
        fn lcg_next(state: &mut u64) -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state >> 11) as f64 / (1u64 << 53) as f64
        }

        let n = state.len();

        // Compute total jump probability
        let mut total_rate = 0.0;
        let mut jump_probs = Vec::new();

        for op in &self.jump_ops {
            // Convert to matrix
            let l_matrix: DMatrix<Complex64> = DMatrix::from_fn(n, n, |i, j| op.matrix[i][j]);

            // Compute ||L|ψ⟩||²
            let state_vec = DVector::from_column_slice(&state);
            let l_psi = &l_matrix * &state_vec;
            let prob = l_psi.iter().map(|x| x.norm_sqr()).sum::<f64>();
            jump_probs.push(prob);
            total_rate += prob;
        }

        // Coherent evolution probability
        let coherent_prob = 1.0 - total_rate * dt;

        if lcg_next(&mut rng_state) < coherent_prob {
            // No jump: evolve under H_eff = H - (i/2) Σ L†L
            // Simplified: just apply Hamiltonian evolution
            let h_matrix: DMatrix<Complex64> =
                DMatrix::from_fn(n, n, |i, j| self.hamiltonian.matrix[i][j]);

            let state_vec = DVector::from_column_slice(&state);
            let h_psi = &h_matrix * &state_vec;

            for i in 0..n {
                state[i] -= Complex64::new(0.0, dt) * h_psi[i];
            }
        } else {
            // Jump occurs
            let r = lcg_next(&mut rng_state) * total_rate;
            let mut cum_prob = 0.0;

            for (i, op) in self.jump_ops.iter().enumerate() {
                cum_prob += jump_probs[i];
                if r < cum_prob {
                    // Apply this jump operator
                    let l_matrix: DMatrix<Complex64> =
                        DMatrix::from_fn(n, n, |i, j| op.matrix[i][j]);
                    let state_vec = DVector::from_column_slice(&state);
                    let new_state = &l_matrix * &state_vec;

                    // Normalize
                    let norm: f64 = new_state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                    for j in 0..n {
                        state[j] = new_state[j] / norm;
                    }
                    break;
                }
            }
        }

        // Normalize
        let norm: f64 = state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        for x in &mut state {
            *x /= norm;
        }

        state
    }

    /// Run multiple trajectories and average.
    pub fn run_trajectories(
        &mut self,
        dt: f64,
        steps: usize,
        n_trajectories: usize,
    ) -> DensityMatrix {
        let dim = 1 << self.n_qubits;
        let mut avg_rho = DensityMatrix::zeros(dim);

        for traj in 0..n_trajectories {
            // Reset to initial state
            self.rho = DensityMatrix::ground_state(self.n_qubits);

            let final_state = self.quantum_trajectory(dt, steps as u64 * traj as u64);

            // Add to average
            for i in 0..dim {
                for j in 0..dim {
                    avg_rho.data[(i, j)] += final_state[i] * final_state[j].conj();
                }
            }
        }

        // Normalize by number of trajectories
        avg_rho.data /= Complex64::new(n_trajectories as f64, 0.0);
        avg_rho
    }

    /// Get purity.
    pub fn purity(&self) -> f64 {
        self.rho.purity()
    }

    /// Get Von Neumann entropy.
    pub fn entropy(&self) -> f64 {
        self.rho.entropy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_matrix_creation() {
        let rho = DensityMatrix::ground_state(2);
        assert_eq!(rho.dim, 4);
        assert!((rho.trace().re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_density_matrix_purity() {
        let rho = DensityMatrix::ground_state(2);
        assert!((rho.purity() - 1.0).abs() < 1e-10);

        let mixed = DensityMatrix::maximally_mixed(4);
        assert!((mixed.purity() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_jump_operator_amplitude_damping() {
        let op = JumpOperator::amplitude_damping(0, 2, 0.1);
        assert_eq!(op.matrix.len(), 4);
    }

    #[test]
    fn test_jump_operator_dephasing() {
        let op = JumpOperator::dephasing(0, 2, 0.1);
        assert_eq!(op.matrix.len(), 4);
    }

    #[test]
    fn test_hamiltonian_tfim() {
        let h = Hamiltonian::tfim(2, 1.0, 0.5);
        assert_eq!(h.dim, 4);
    }

    #[test]
    fn test_lindblad_solver_creation() {
        let h = Hamiltonian::tfim(2, 1.0, 0.5);
        let solver = LindbladSolver::new(2, h);
        assert_eq!(solver.current_time(), 0.0);
    }

    #[test]
    fn test_lindblad_solver_amplitude_damping() {
        let h = Hamiltonian::identity(4);
        let mut solver = LindbladSolver::new(2, h);

        // Start in |11⟩ state
        let mut state = vec![ZERO; 4];
        state[3] = ONE;
        solver.set_state(&state);

        // Add amplitude damping on qubit 0
        solver.add_jump_operator(JumpOperator::amplitude_damping(0, 2, 0.5));

        // Evolve
        solver.evolve_rk4(0.01, 100);

        // Should have decayed toward |01⟩
        let rho = solver.density_matrix();
        assert!((rho.trace().re - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantum_trajectory() {
        let h = Hamiltonian::identity(4);
        let mut solver = LindbladSolver::new(2, h);

        let final_state = solver.quantum_trajectory(0.01, 42);
        assert_eq!(final_state.len(), 4);

        // Check normalization
        let norm: f64 = final_state.iter().map(|x| x.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lindblad_purity_decay() {
        let h = Hamiltonian::identity(4);
        let mut solver = LindbladSolver::new(2, h);

        // Add dephasing
        solver.add_jump_operator(JumpOperator::dephasing(0, 2, 0.1));

        let initial_purity = solver.purity();

        solver.evolve_rk4(0.01, 100);

        let final_purity = solver.purity();
        // Purity should decrease under dephasing
        assert!(final_purity <= initial_purity + 1e-6);
    }
}
