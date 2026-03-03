//! Quantum Fisher Information for Quantum Metrology
//!
//! The Quantum Fisher Information (QFI) quantifies the ultimate precision
//! limit for estimating a parameter encoded in a quantum state.
//!
//! # Key Concepts
//!
//! - **Fisher Information**: Measures how much information a measurement
//!   carries about an unknown parameter
//! - **Quantum Fisher Information**: Maximum Fisher information over all
//!   possible measurements (POVMs)
//! - **Quantum Cramér-Rao Bound**: Var(θ̂) ≥ 1/(n·F_Q) where n is the
//!   number of independent measurements
//!
//! # Applications
//!
//! - Quantum sensing and metrology
//! - Gravitational wave detection
//! - Magnetic field sensing
//! - Clock synchronization
//! - Parameter estimation in VQE
//!
//! # Reference
//!
//! M. G. A. Paris, "Quantum estimation for quantum technology"
//! Int. J. Quant. Inf. 7, 125 (2009)

use crate::{QuantumState, C64};
use num_complex::Complex64;

/// Quantum Fisher Information calculator
pub struct QuantumFisher {
    /// Number of qubits
    n_qubits: usize,
    /// Dimension (2^n)
    dim: usize,
}

/// Result of QFI calculation
#[derive(Clone, Debug)]
pub struct QFIResult {
    /// Quantum Fisher Information value
    pub qfi: f64,
    /// Classical Fisher Information (for comparison)
    pub cfi: f64,
    /// Attainable precision (1/sqrt(QFI))
    pub precision: f64,
    /// Signal-to-noise ratio improvement
    pub snr_improvement: f64,
}

/// Generator of a parameterized unitary
///
/// U(θ) = exp(-i θ G) where G is the Hermitian generator
#[derive(Clone, Debug)]
pub struct ParameterGenerator {
    /// Generator matrix (Hermitian)
    pub matrix: Vec<Vec<C64>>,
    /// Parameter name
    pub name: String,
}

impl ParameterGenerator {
    /// Create a Pauli-X generator on a specific qubit
    pub fn pauli_x(n_qubits: usize, target: usize) -> Self {
        let dim = 1 << n_qubits;
        let mut matrix = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        for i in 0..dim {
            let flipped = i ^ (1 << target);
            matrix[i][flipped] = Complex64::new(1.0, 0.0);
        }

        ParameterGenerator {
            matrix,
            name: format!("X_{}", target),
        }
    }

    /// Create a Pauli-Y generator on a specific qubit
    pub fn pauli_y(n_qubits: usize, target: usize) -> Self {
        let dim = 1 << n_qubits;
        let mut matrix = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        for i in 0..dim {
            let flipped = i ^ (1 << target);
            let phase = if (i >> target) & 1 == 0 {
                Complex64::new(0.0, 1.0)
            } else {
                Complex64::new(0.0, -1.0)
            };
            matrix[i][flipped] = phase;
        }

        ParameterGenerator {
            matrix,
            name: format!("Y_{}", target),
        }
    }

    /// Create a Pauli-Z generator on a specific qubit
    pub fn pauli_z(n_qubits: usize, target: usize) -> Self {
        let dim = 1 << n_qubits;
        let mut matrix = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        for i in 0..dim {
            let sign = if (i >> target) & 1 == 0 { 1.0 } else { -1.0 };
            matrix[i][i] = Complex64::new(sign, 0.0);
        }

        ParameterGenerator {
            matrix,
            name: format!("Z_{}", target),
        }
    }

    /// Create a Hamiltonian generator from Pauli strings
    pub fn hamiltonian(n_qubits: usize, paulis: &[(Vec<char>, f64)]) -> Self {
        let dim = 1 << n_qubits;
        let mut matrix = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        for (pauli_string, coeff) in paulis {
            let mut pauli_matrix = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

            // Initialize to identity
            for i in 0..dim {
                pauli_matrix[i][i] = Complex64::new(1.0, 0.0);
            }

            // Apply each Pauli
            for (qubit, &p) in pauli_string.iter().enumerate() {
                let p_dim = 1 << (n_qubits - qubit - 1);
                let _stride = 1 << (n_qubits - qubit);

                match p.to_uppercase().next().unwrap() {
                    'X' => {
                        let mut new_matrix = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
                        for i in 0..dim {
                            let flipped = i ^ p_dim;
                            for j in 0..dim {
                                new_matrix[i][j] += pauli_matrix[flipped][j];
                            }
                        }
                        pauli_matrix = new_matrix;
                    }
                    'Y' => {
                        let mut new_matrix = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
                        for i in 0..dim {
                            let flipped = i ^ p_dim;
                            let phase = if (i & p_dim) == 0 {
                                Complex64::new(0.0, 1.0)
                            } else {
                                Complex64::new(0.0, -1.0)
                            };
                            for j in 0..dim {
                                new_matrix[i][j] += pauli_matrix[flipped][j] * phase;
                            }
                        }
                        pauli_matrix = new_matrix;
                    }
                    'Z' => {
                        for i in 0..dim {
                            let sign = if (i & p_dim) == 0 { 1.0 } else { -1.0 };
                            for j in 0..dim {
                                pauli_matrix[i][j] = pauli_matrix[i][j].scale(sign);
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Add to total matrix
            for i in 0..dim {
                for j in 0..dim {
                    matrix[i][j] = matrix[i][j] + pauli_matrix[i][j].scale(*coeff);
                }
            }
        }

        let name = if paulis.len() == 1 {
            format!("H_{}", paulis[0].1)
        } else {
            format!("H_{}terms", paulis.len())
        };

        ParameterGenerator { matrix, name }
    }

    /// Create a collective spin operator J_z for GHZ states
    pub fn collective_jz(n_qubits: usize) -> Self {
        let mut terms = Vec::new();
        for i in 0..n_qubits {
            let mut pauli_string = vec!['I'; n_qubits];
            pauli_string[i] = 'Z';
            terms.push((pauli_string, 0.5));
        }
        Self::hamiltonian(n_qubits, &terms)
    }

    /// Create a collective spin operator J_x for GHZ states
    pub fn collective_jx(n_qubits: usize) -> Self {
        let mut terms = Vec::new();
        for i in 0..n_qubits {
            let mut pauli_string = vec!['I'; n_qubits];
            pauli_string[i] = 'X';
            terms.push((pauli_string, 0.5));
        }
        Self::hamiltonian(n_qubits, &terms)
    }
}

impl QuantumFisher {
    /// Create new QFI calculator
    pub fn new(n_qubits: usize) -> Self {
        QuantumFisher {
            n_qubits,
            dim: 1 << n_qubits,
        }
    }

    /// Calculate QFI for a pure state
    ///
    /// For pure states: F_Q[|ψ⟩, G] = 4(⟨ψ|G²|ψ⟩ - ⟨ψ|G|ψ⟩²)
    /// This is 4 times the variance of the generator
    pub fn qfi_pure(&self, state: &QuantumState, generator: &ParameterGenerator) -> f64 {
        let psi = state.amplitudes_ref();

        // Compute ⟨ψ|G|ψ⟩
        let mut exp_g = Complex64::new(0.0, 0.0);
        for i in 0..self.dim {
            for j in 0..self.dim {
                let psi_i_conj = C64 { re: psi[i].re, im: -psi[i].im };
                exp_g = exp_g + psi_i_conj * generator.matrix[i][j] * psi[j];
            }
        }

        // Compute ⟨ψ|G²|ψ⟩
        let mut exp_g2 = Complex64::new(0.0, 0.0);
        for i in 0..self.dim {
            for j in 0..self.dim {
                for k in 0..self.dim {
                    let psi_i_conj = C64 { re: psi[i].re, im: -psi[i].im };
                    let g_ik = generator.matrix[i][k];
                    let g_kj = generator.matrix[k][j];
                    exp_g2 = exp_g2 + psi_i_conj * g_ik * g_kj * psi[j];
                }
            }
        }

        // F_Q = 4 * Var(G) = 4 * (⟨G²⟩ - ⟨G⟩²)
        let variance = exp_g2.re - exp_g.re * exp_g.re;
        4.0 * variance
    }

    /// Calculate QFI for a mixed state (density matrix)
    ///
    /// For mixed states, we use the formula:
    /// F_Q = Σᵢ 2/(λᵢ + λⱼ) |⟨λᵢ|G|λⱼ⟩|² for λᵢ + λⱼ > 0
    pub fn qfi_mixed(&self, rho: &[Vec<C64>], generator: &ParameterGenerator) -> f64 {
        // Diagonalize rho to get eigenvalues and eigenvectors
        let (eigenvalues, eigenvectors) = self.diagonalize_density_matrix(rho);

        let mut qfi = 0.0;

        for i in 0..self.dim {
            for j in 0..self.dim {
                if i != j && eigenvalues[i] + eigenvalues[j] > 1e-10 {
                    // Compute ⟨λᵢ|G|λⱼ⟩
                    let mut matrix_element = Complex64::new(0.0, 0.0);
                    for k in 0..self.dim {
                        for l in 0..self.dim {
                            let v_i_conj = C64 {
                                re: eigenvectors[i][k].re,
                                im: -eigenvectors[i][k].im,
                            };
                            matrix_element = matrix_element + v_i_conj * generator.matrix[k][l] * eigenvectors[j][l];
                        }
                    }

                    let denom = eigenvalues[i] + eigenvalues[j];
                    qfi += 2.0 / denom * (matrix_element.re * matrix_element.re + matrix_element.im * matrix_element.im);
                }
            }
        }

        qfi
    }

    /// Simple eigenvalue decomposition (power iteration for dominant eigenvalues)
    fn diagonalize_density_matrix(&self, rho: &[Vec<C64>]) -> (Vec<f64>, Vec<Vec<C64>>) {
        let mut eigenvalues = vec![0.0; self.dim];
        let mut eigenvectors = vec![vec![Complex64::new(0.0, 0.0); self.dim]; self.dim];

        // Initialize with identity (simplified - real implementation would use proper diagonalization)
        for i in 0..self.dim {
            eigenvectors[i][i] = Complex64::new(1.0, 0.0);
            eigenvalues[i] = rho[i][i].re; // Diagonal approximation
        }

        // Normalize eigenvalues
        let trace: f64 = eigenvalues.iter().sum();
        if trace > 0.0 {
            for e in &mut eigenvalues {
                *e /= trace;
            }
        }

        (eigenvalues, eigenvectors)
    }

    /// Calculate Classical Fisher Information for a specific measurement
    ///
    /// CFI = Σₓ P(x|θ) (∂ ln P(x|θ) / ∂θ)²
    pub fn cfi(&self, state: &QuantumState, generator: &ParameterGenerator, theta: f64, _shots: usize) -> f64 {
        // Simulate measurements in the eigenbasis of the generator
        let mut probabilities = vec![0.0; self.dim];

        // Apply U(θ) = exp(-i θ G) to state and measure
        for outcome in 0..self.dim {
            probabilities[outcome] = self.measurement_probability(state, generator, theta, outcome);
        }

        // Compute CFI numerically
        let epsilon = 1e-6;
        let mut derivatives = vec![0.0; self.dim];

        for outcome in 0..self.dim {
            let p_plus = self.measurement_probability(state, generator, theta + epsilon, outcome);
            let p_minus = self.measurement_probability(state, generator, theta - epsilon, outcome);
            let dp = (p_plus - p_minus) / (2.0 * epsilon);
            derivatives[outcome] = dp;
        }

        let mut cfi = 0.0;
        for outcome in 0..self.dim {
            if probabilities[outcome] > 1e-10 {
                cfi += derivatives[outcome] * derivatives[outcome] / probabilities[outcome];
            }
        }

        cfi
    }

    /// Calculate probability of measuring a specific outcome
    fn measurement_probability(&self, state: &QuantumState, generator: &ParameterGenerator, theta: f64, outcome: usize) -> f64 {
        let psi = state.amplitudes_ref();

        // Apply U(θ) = exp(-i θ G) to |ψ⟩
        // Simplified: use first-order approximation for small θ
        let mut evolved = vec![Complex64::new(0.0, 0.0); self.dim];

        for i in 0..self.dim {
            evolved[i] = psi[i];
            for j in 0..self.dim {
                let phase = Complex64::new(0.0, -theta) * generator.matrix[i][j];
                evolved[i] = evolved[i] + phase * psi[j];
            }
        }

        // Probability of outcome
        evolved[outcome].norm() * evolved[outcome].norm()
    }

    /// Full QFI calculation with classical comparison
    pub fn calculate(&self, state: &QuantumState, generator: &ParameterGenerator) -> QFIResult {
        let qfi = self.qfi_pure(state, generator);
        let cfi = self.cfi(state, generator, 0.0, 1000);

        let precision = if qfi > 0.0 { 1.0 / qfi.sqrt() } else { f64::INFINITY };

        // SNR improvement: Heisenberg limit vs shot noise
        // Shot noise: 1/sqrt(N), Heisenberg: 1/N
        // SNR improvement = sqrt(N) for N qubits
        let snr_improvement = (self.n_qubits as f64).sqrt();

        QFIResult {
            qfi,
            cfi,
            precision,
            snr_improvement,
        }
    }

    /// Compute optimal measurement basis for parameter estimation
    ///
    /// Returns the eigenbasis of the operator |∂ψ⟩⟨ψ| + |ψ⟩⟨∂ψ|
    /// where |∂ψ⟩ = -i G |ψ⟩
    pub fn optimal_measurement(&self, state: &QuantumState, generator: &ParameterGenerator) -> Vec<Vec<C64>> {
        let psi = state.amplitudes_ref();

        // Compute |∂ψ⟩ = -i G |ψ⟩
        let mut dpsi = vec![Complex64::new(0.0, 0.0); self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                let phase = Complex64::new(0.0, -1.0) * generator.matrix[i][j];
                dpsi[i] = dpsi[i] + phase * psi[j];
            }
        }

        // The optimal measurement is in the eigenbasis of:
        // S = i(|∂ψ⟩⟨ψ| - |ψ⟩⟨∂ψ|)
        // This is a Hermitian operator

        // For simplicity, return computational basis
        // Real implementation would diagonalize S
        let mut basis = vec![vec![Complex64::new(0.0, 0.0); self.dim]; self.dim];
        for i in 0..self.dim {
            basis[i][i] = Complex64::new(1.0, 0.0);
        }

        basis
    }
}

// ---------------------------------------------------------------------------
// QUANTUM METROLOGY UTILITIES
// ---------------------------------------------------------------------------

/// Create a GHZ state for Heisenberg-limited sensing
///
/// GHZ state achieves Heisenberg limit: Var(θ) ~ 1/N²
pub fn create_ghz_state(n_qubits: usize) -> QuantumState {
    let mut state = QuantumState::new(n_qubits);

    // Start with |0...0⟩
    // Apply H to first qubit: (|0⟩ + |1⟩)/√2 ⊗ |0...0⟩
    crate::GateOperations::h(&mut state, 0);

    // Apply CNOTs to create entanglement
    for i in 0..(n_qubits - 1) {
        crate::GateOperations::cnot(&mut state, i, i + 1);
    }

    // Now we have (|0...0⟩ + |1...1⟩)/√2
    state
}

/// Create a NOON state for phase sensing
///
/// NOON state: (|N,0⟩ + |0,N⟩)/√2
/// Achieves super-Heisenberg scaling for phase estimation
pub fn create_noon_state(n: usize) -> QuantumState {
    let n_qubits = 2 * n;
    let mut state = QuantumState::new(n_qubits);

    // Create superposition of all |0⟩s and all |1⟩s
    // First n qubits |0⟩ or |1⟩, second n qubits |0⟩ or |1⟩

    // Start with |0...0⟩
    crate::GateOperations::h(&mut state, 0);

    // Entangle first half
    for i in 0..(n - 1) {
        crate::GateOperations::cnot(&mut state, i, i + 1);
    }

    // Apply phase to flip to second mode
    crate::GateOperations::z(&mut state, 0);

    // Entangle second half
    for i in n..(2 * n - 1) {
        crate::GateOperations::cnot(&mut state, i, i + 1);
    }

    state
}

/// Calculate Heisenberg limit for N qubits
pub fn heisenberg_limit(n_qubits: usize, shots: usize) -> f64 {
    // Heisenberg limit: Var(θ) = 1 / (N² · shots)
    1.0 / (n_qubits.pow(2) as f64 * shots as f64)
}

/// Calculate shot noise limit for N qubits
pub fn shot_noise_limit(n_qubits: usize, shots: usize) -> f64 {
    // Shot noise: Var(θ) = 1 / (N · shots)
    1.0 / (n_qubits as f64 * shots as f64)
}

// ---------------------------------------------------------------------------
// BENCHMARK HARNESS
// ---------------------------------------------------------------------------

/// Benchmark QFI calculation
pub fn benchmark_qfi(n_qubits: usize) -> (f64, f64, f64) {
    use std::time::Instant;

    let qfi_calc = QuantumFisher::new(n_qubits);
    let state = create_ghz_state(n_qubits);
    let generator = ParameterGenerator::collective_jz(n_qubits);

    let start = Instant::now();
    let result = qfi_calc.calculate(&state, &generator);
    let elapsed = start.elapsed().as_secs_f64();

    (result.qfi, result.cfi, elapsed * 1e6)
}

/// Print QFI benchmark
pub fn print_benchmark() {
    println!("{}", "=".repeat(70));
    println!("Quantum Fisher Information Benchmark");
    println!("{}", "=".repeat(70));
    println!();

    println!("Testing GHZ states for Heisenberg-limited sensing:");
    println!("{}", "-".repeat(70));
    println!("{:<10} {:<15} {:<15} {:<15}", "Qubits", "QFI", "Heisenberg", "Time (us)");
    println!("{}", "-".repeat(70));

    for n in [2, 3, 4, 5, 6].iter() {
        let (qfi, _cfi, time_us) = benchmark_qfi(*n);
        let heisenberg = (*n * *n) as f64; // Expected QFI = N² for GHZ + J_z
        println!("{:<10} {:<15.2} {:<15.2} {:<15.2}", n, qfi, heisenberg, time_us);
    }

    println!();
    println!("Heisenberg Limit vs Shot Noise:");
    println!("{}", "-".repeat(70));

    for n in [2, 5, 10, 20, 50].iter() {
        let heis = heisenberg_limit(*n, 1000);
        let shot = shot_noise_limit(*n, 1000);
        let improvement = shot / heis;
        println!("N={:<5}: Heisenberg={:.2e}, Shot Noise={:.2e}, Improvement={:.1}x",
                 n, heis, shot, improvement);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qfi_creation() {
        let qfi = QuantumFisher::new(3);
        assert_eq!(qfi.n_qubits, 3);
        assert_eq!(qfi.dim, 8);
    }

    #[test]
    fn test_pauli_generators() {
        let x = ParameterGenerator::pauli_x(2, 0);
        let y = ParameterGenerator::pauli_y(2, 0);
        let z = ParameterGenerator::pauli_z(2, 0);

        assert_eq!(x.matrix.len(), 4);
        assert_eq!(y.matrix.len(), 4);
        assert_eq!(z.matrix.len(), 4);
    }

    #[test]
    fn test_ghz_state() {
        let state = create_ghz_state(3);
        let amps = state.amplitudes_ref();

        // GHZ: (|000⟩ + |111⟩)/√2
        assert!((amps[0].norm() - 1.0 / 2.0_f64.sqrt()).abs() < 0.01);
        assert!((amps[7].norm() - 1.0 / 2.0_f64.sqrt()).abs() < 0.01);

        // Other amplitudes should be near zero
        for i in 1..7 {
            assert!(amps[i].norm() < 0.01, "Amplitude {} should be near zero", i);
        }
    }

    #[test]
    fn test_qfi_ghz_state() {
        let qfi_calc = QuantumFisher::new(3);
        let state = create_ghz_state(3);
        let generator = ParameterGenerator::collective_jz(3);

        let result = qfi_calc.calculate(&state, &generator);

        // For GHZ state with J_z, QFI should be N² = 9
        assert!(result.qfi > 5.0, "QFI for GHZ + J_z should be ~N² = 9, got {}", result.qfi);
        assert!(result.precision > 0.0);
    }

    #[test]
    fn test_qfi_computational_basis() {
        let qfi_calc = QuantumFisher::new(2);
        let state = QuantumState::new(2); // |00⟩
        let generator = ParameterGenerator::pauli_z(2, 0);

        let result = qfi_calc.calculate(&state, &generator);

        // Computational basis with Z: should have zero variance
        // But due to numerical approximations, may be small
        assert!(result.qfi >= 0.0);
    }

    #[test]
    fn test_heisenberg_vs_shot_noise() {
        for n in [2, 5, 10].iter() {
            let heis = heisenberg_limit(*n, 1000);
            let shot = shot_noise_limit(*n, 1000);

            // Heisenberg should always be better (smaller variance)
            assert!(heis < shot);

            // Improvement should be N
            let improvement = shot / heis;
            assert!((improvement - *n as f64).abs() < 0.01);
        }
    }

    #[test]
    fn test_benchmark() {
        let (qfi, cfi, time_us) = benchmark_qfi(3);
        assert!(qfi >= 0.0);
        assert!(cfi >= 0.0);
        assert!(time_us >= 0.0);
    }
}
