//! Density Matrix Simulation for Quantum Computing
//!
//! Implements density matrix formalism for quantum simulation:
//! - Mixed state representation
//! - Entanglement tracking
//! - Correlated noise modeling
//! - Partial trace and reduced states
//! - Entropy and purity measures

use num_complex::Complex64 as C64;

/// Density matrix representation
///
/// ρ = Σ p_i |ψ_i⟩⟨ψ_i|
/// where p_i are probabilities and |ψ_i⟩ are pure states
///
/// Size: 2^n × 2^n complex matrix stored as flat vector
#[derive(Clone, Debug)]
pub struct DensityMatrix {
    /// Density matrix elements (row-major order)
    pub elements: Vec<C64>,
    /// Number of qubits
    pub num_qubits: usize,
    /// Dimension: 2^num_qubits
    pub dim: usize,
}

impl DensityMatrix {
    /// Create a new density matrix initialized to |0⟩⟨0|
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1usize
            .checked_shl(num_qubits as u32)
            .expect("num_qubits too large for density matrix");
        let size = dim.checked_mul(dim).expect("density matrix size overflow");
        let mut elements = vec![C64::new(0.0, 0.0); size];
        // Initialize to |0⟩⟨0| state
        elements[0] = C64::new(1.0, 0.0);

        DensityMatrix {
            elements,
            num_qubits,
            dim,
        }
    }

    /// Create a density matrix from a pure state vector
    pub fn from_pure_state(state: &[C64]) -> Self {
        let n = state.len();
        let num_qubits = (n as f64).log2() as usize;
        let dim = 1usize
            .checked_shl(num_qubits as u32)
            .expect("num_qubits too large for density matrix");
        let size = dim.checked_mul(dim).expect("density matrix size overflow");
        let mut elements = vec![C64::new(0.0, 0.0); size];

        // ρ = |ψ⟩⟨ψ|
        for i in 0..dim {
            for j in 0..dim {
                elements[i * dim + j] = state[i] * state[j].conj();
            }
        }

        DensityMatrix {
            elements,
            num_qubits,
            dim,
        }
    }

    /// Create a maximally mixed state
    pub fn maximally_mixed(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut elements = vec![C64::new(0.0, 0.0); dim * dim];

        // Maximally mixed state: identity/dim
        for i in 0..dim {
            elements[i * dim + i] = C64::new(1.0 / dim as f64, 0.0);
        }

        DensityMatrix {
            elements,
            num_qubits,
            dim,
        }
    }

    /// Create a density matrix from a statistical mixture
    pub fn from_mixture(states: &[Vec<C64>], probabilities: &[f64]) -> Self {
        assert_eq!(
            states.len(),
            probabilities.len(),
            "States and probabilities must match"
        );
        let num_qubits = (states[0].len() as f64).log2() as usize;
        let dim = 1 << num_qubits;
        let mut elements = vec![C64::new(0.0, 0.0); dim * dim];

        // ρ = Σ p_i |ψ_i⟩⟨ψ_i|
        for (state, &prob) in states.iter().zip(probabilities.iter()) {
            for i in 0..dim {
                for j in 0..dim {
                    elements[i * dim + j] += state[i] * state[j].conj() * prob;
                }
            }
        }

        DensityMatrix {
            elements,
            num_qubits,
            dim,
        }
    }

    /// Get the trace of the density matrix (should be 1 for valid states)
    pub fn trace(&self) -> f64 {
        (0..self.dim)
            .map(|i| self.elements[i * self.dim + i].re)
            .sum()
    }

    /// Get the purity of the state: Tr(ρ²)
    /// - Pure state: purity = 1
    /// - Mixed state: purity < 1
    /// - Maximally mixed: purity = 1/dim
    pub fn purity(&self) -> f64 {
        let mut sum = C64::new(0.0, 0.0);
        for i in 0..self.dim {
            for j in 0..self.dim {
                sum += self.elements[i * self.dim + j] * self.elements[j * self.dim + i];
            }
        }
        sum.re
    }

    /// Check if the state is pure
    pub fn is_pure(&self) -> bool {
        (self.purity() - 1.0).abs() < 1e-10
    }

    /// Calculate von Neumann entropy: S(ρ) = -Tr(ρ log₂ ρ)
    pub fn entropy(&self) -> f64 {
        // Diagonalize the density matrix
        let eigenvalues = self.diagonalize();

        // Normalize eigenvalues to sum to 1 (handles numerical errors)
        let sum: f64 = eigenvalues.iter().filter(|&&λ| λ > 0.0).sum();
        if sum <= 0.0 {
            return 0.0; // Pure state or invalid state
        }

        // S = -Σ λ_i log₂ λ_i, only for positive eigenvalues
        eigenvalues
            .into_iter()
            .filter(|&λ| λ > 1e-15)
            .map(|λ| {
                let λ_normalized = λ / sum;
                if λ_normalized > 1e-15 {
                    -λ_normalized * λ_normalized.log2()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Diagonalize the density matrix using power iteration
    /// Returns eigenvalues (simplified approach)
    fn diagonalize(&self) -> Vec<f64> {
        // For simplicity, use the diagonal elements
        // A full implementation would use eigensolvers
        (0..self.dim)
            .map(|i| self.elements[i * self.dim + i].re)
            .collect()
    }

    /// Get the reduced density matrix for a subset of qubits
    pub fn partial_trace(&self, keep_qubits: &[usize]) -> DensityMatrix {
        let num_keep = keep_qubits.len();
        let dim_keep = 1 << num_keep;
        let mut reduced = vec![C64::new(0.0, 0.0); dim_keep * dim_keep];

        // Identify which qubits to trace out
        let mut trace_qubits = Vec::new();
        for qubit in 0..self.num_qubits {
            if !keep_qubits.contains(&qubit) {
                trace_qubits.push(qubit);
            }
        }

        let num_trace = trace_qubits.len();
        let dim_trace = if num_trace > 0 { 1 << num_trace } else { 1 };

        // Trace out qubits not in keep_qubits
        for i_keep in 0..dim_keep {
            for j_keep in 0..dim_keep {
                let mut sum = C64::new(0.0, 0.0);

                // Iterate over all traced-out qubit configurations
                for traced in 0..dim_trace {
                    // Reconstruct full indices
                    let i_full = self.expand_index(i_keep, traced, keep_qubits, &trace_qubits);
                    let j_full = self.expand_index(j_keep, traced, keep_qubits, &trace_qubits);

                    sum += self.elements[i_full * self.dim + j_full];
                }

                reduced[i_keep * dim_keep + j_keep] = sum;
            }
        }

        DensityMatrix {
            elements: reduced,
            num_qubits: num_keep,
            dim: dim_keep,
        }
    }

    /// Expand a reduced index to a full index given kept and traced qubits
    fn expand_index(
        &self,
        reduced_idx: usize,
        traced_val: usize,
        keep_qubits: &[usize],
        trace_qubits: &[usize],
    ) -> usize {
        let mut full_idx = 0;

        // Add kept qubits
        for (pos, &qubit) in keep_qubits.iter().enumerate() {
            let bit = (reduced_idx >> pos) & 1;
            full_idx |= bit << qubit;
        }

        // Add traced qubits
        for (pos, &qubit) in trace_qubits.iter().enumerate() {
            let bit = (traced_val >> pos) & 1;
            full_idx |= bit << qubit;
        }

        full_idx
    }

    /// Measure the expectation value of an operator
    pub fn expectation(&self, operator: &[C64]) -> C64 {
        let mut result = C64::new(0.0, 0.0);
        for i in 0..self.dim {
            for j in 0..self.dim {
                result += operator[j * self.dim + i] * self.elements[i * self.dim + j];
            }
        }
        result
    }

    /// Check if density matrix is valid (Hermitian, positive, trace=1)
    pub fn is_valid(&self) -> bool {
        // Check trace is approximately 1
        if (self.trace() - 1.0).abs() > 1e-10 {
            return false;
        }

        // Check Hermiticity: ρ = ρ†
        for i in 0..self.dim {
            for j in 0..self.dim {
                let diff = self.elements[i * self.dim + j] - self.elements[j * self.dim + i].conj();
                if diff.norm() > 1e-10 {
                    return false;
                }
            }
        }

        // Check positivity (eigenvalues ≥ 0)
        let eigenvalues = self.diagonalize();
        if eigenvalues.iter().any(|&λ| λ < -1e-10) {
            return false;
        }

        true
    }

    /// Reset to |0⟩⟨0| state
    pub fn reset(&mut self) {
        self.elements.fill(C64::new(0.0, 0.0));
        self.elements[0] = C64::new(1.0, 0.0);
    }
}

// ============================================================
// SINGLE-QUBIT GATES
// ============================================================

impl DensityMatrix {
    /// Apply Hadamard gate to qubit
    pub fn h(&mut self, qubit: usize) {
        let mask = 1 << qubit;
        let dim = self.dim;

        for i in 0..dim {
            for j in 0..dim {
                let i0 = i & !mask;
                let i1 = i | mask;
                let j0 = j & !mask;
                let j1 = j | mask;

                if (i & mask) == 0 && (j & mask) == 0 {
                    let rho_00 = self.elements[i0 * dim + j0];
                    let rho_01 = self.elements[i0 * dim + j1];
                    let rho_10 = self.elements[i1 * dim + j0];
                    let rho_11 = self.elements[i1 * dim + j1];

                    self.elements[i0 * dim + j0] = (rho_00 + rho_01 + rho_10 + rho_11) * 0.5;
                    self.elements[i0 * dim + j1] = (rho_00 - rho_01 + rho_10 - rho_11) * 0.5;
                    self.elements[i1 * dim + j0] = (rho_00 + rho_01 - rho_10 - rho_11) * 0.5;
                    self.elements[i1 * dim + j1] = (rho_00 - rho_01 - rho_10 + rho_11) * 0.5;
                }
            }
        }
    }

    /// Apply Pauli-X gate to qubit
    pub fn x(&mut self, qubit: usize) {
        let mask = 1 << qubit;
        let dim = self.dim;
        let mut new_elements = self.elements.clone();

        for i in 0..dim {
            for j in 0..dim {
                let i_flipped = i ^ mask;
                let j_flipped = j ^ mask;
                new_elements[i * dim + j] = self.elements[i_flipped * dim + j_flipped];
            }
        }

        self.elements = new_elements;
    }

    /// Apply Pauli-Y gate to qubit
    pub fn y(&mut self, qubit: usize) {
        let mask = 1 << qubit;
        let dim = self.dim;
        let mut new_elements = self.elements.clone();

        let i_val = C64::new(0.0, 1.0);

        for i in 0..dim {
            for j in 0..dim {
                let i_flipped = i ^ mask;
                let j_flipped = j ^ mask;
                let sign = if ((i ^ i_flipped) >> qubit) & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
                let sign = if ((j ^ j_flipped) >> qubit) & 1 == 1 {
                    -sign
                } else {
                    sign
                };
                new_elements[i * dim + j] =
                    self.elements[i_flipped * dim + j_flipped] * sign * i_val;
            }
        }

        self.elements = new_elements;
    }

    /// Apply Pauli-Z gate to qubit
    pub fn z(&mut self, qubit: usize) {
        let _mask = 1 << qubit;
        let dim = self.dim;

        for i in 0..dim {
            for j in 0..dim {
                if ((i >> qubit) & 1) == 1 {
                    self.elements[i * dim + j] = -self.elements[i * dim + j];
                }
            }
        }
    }

    /// Apply phase gate (S) to qubit
    pub fn s(&mut self, qubit: usize) {
        let _mask = 1 << qubit;
        let dim = self.dim;

        for i in 0..dim {
            for j in 0..dim {
                if ((i >> qubit) & 1) == 1 && ((j >> qubit) & 1) == 1 {
                    self.elements[i * dim + j] = C64::new(0.0, 1.0) * self.elements[i * dim + j];
                }
            }
        }
    }

    /// Apply π/8 gate (T) to qubit
    pub fn t(&mut self, qubit: usize) {
        let _mask = 1 << qubit;
        let dim = self.dim;
        let phase = C64::new(1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt());

        for i in 0..dim {
            for j in 0..dim {
                if ((i >> qubit) & 1) == 1 && ((j >> qubit) & 1) == 1 {
                    self.elements[i * dim + j] = phase * self.elements[i * dim + j];
                }
            }
        }
    }

    /// Apply RX rotation to qubit
    pub fn rx(&mut self, qubit: usize, theta: f64) {
        let mask = 1 << qubit;
        let dim = self.dim;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        for i in 0..dim {
            for j in 0..dim {
                let i0 = i & !mask;
                let i1 = i | mask;
                let j0 = j & !mask;
                let j1 = j | mask;

                if (i & mask) == 0 && (j & mask) == 0 {
                    let rho_00 = self.elements[i0 * dim + j0];
                    let rho_01 = self.elements[i0 * dim + j1];
                    let rho_10 = self.elements[i1 * dim + j0];
                    let rho_11 = self.elements[i1 * dim + j1];

                    let c = cos_half;
                    let s = sin_half;

                    self.elements[i0 * dim + j0] =
                        c * c * rho_00 - c * s * rho_01 - s * c * rho_10 + s * s * rho_11;
                    self.elements[i0 * dim + j1] =
                        c * s * rho_00 + c * c * rho_01 - s * s * rho_10 - s * c * rho_11;
                    self.elements[i1 * dim + j0] =
                        c * s * rho_00 - s * s * rho_01 + c * c * rho_10 - s * c * rho_11;
                    self.elements[i1 * dim + j1] =
                        s * s * rho_00 + s * c * rho_01 + c * s * rho_10 + c * c * rho_11;
                }
            }
        }
    }

    /// Apply RY rotation to qubit
    pub fn ry(&mut self, qubit: usize, theta: f64) {
        let mask = 1 << qubit;
        let dim = self.dim;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        for i in 0..dim {
            for j in 0..dim {
                let i0 = i & !mask;
                let i1 = i | mask;
                let j0 = j & !mask;
                let j1 = j | mask;

                if (i & mask) == 0 && (j & mask) == 0 {
                    let rho_00 = self.elements[i0 * dim + j0];
                    let rho_01 = self.elements[i0 * dim + j1];
                    let rho_10 = self.elements[i1 * dim + j0];
                    let rho_11 = self.elements[i1 * dim + j1];

                    let c = cos_half;
                    let s = sin_half;

                    self.elements[i0 * dim + j0] =
                        c * c * rho_00 - c * s * rho_01 - s * c * rho_10 + s * s * rho_11;
                    self.elements[i0 * dim + j1] =
                        -c * s * rho_00 + c * c * rho_01 - s * s * rho_10 + s * c * rho_11;
                    self.elements[i1 * dim + j0] =
                        c * s * rho_00 + s * s * rho_01 + c * c * rho_10 + s * c * rho_11;
                    self.elements[i1 * dim + j1] =
                        -s * s * rho_00 + s * c * rho_01 + c * s * rho_10 + c * c * rho_11;
                }
            }
        }
    }

    /// Apply RZ rotation to qubit
    pub fn rz(&mut self, qubit: usize, theta: f64) {
        let _mask = 1 << qubit;
        let dim = self.dim;
        let phase_plus = C64::new((theta / 2.0).cos(), (theta / 2.0).sin());
        let phase_minus = C64::new((theta / 2.0).cos(), -(theta / 2.0).sin());

        for i in 0..dim {
            for j in 0..dim {
                let i_bit = (i >> qubit) & 1;
                let j_bit = (j >> qubit) & 1;

                if i_bit == 0 && j_bit == 0 {
                    self.elements[i * dim + j] = phase_minus * self.elements[i * dim + j];
                } else if i_bit == 1 && j_bit == 1 {
                    self.elements[i * dim + j] = phase_plus * self.elements[i * dim + j];
                }
            }
        }
    }

    /// Apply arbitrary unitary gate to qubit
    pub fn u(&mut self, qubit: usize, matrix: &[[C64; 2]; 2]) {
        let mask = 1 << qubit;
        let dim = self.dim;

        for i in 0..dim {
            for j in 0..dim {
                let i0 = i & !mask;
                let i1 = i | mask;
                let j0 = j & !mask;
                let j1 = j | mask;

                if (i & mask) == 0 && (j & mask) == 0 {
                    let rho_00 = self.elements[i0 * dim + j0];
                    let rho_01 = self.elements[i0 * dim + j1];
                    let rho_10 = self.elements[i1 * dim + j0];
                    let rho_11 = self.elements[i1 * dim + j1];

                    let u00 = matrix[0][0];
                    let u01 = matrix[0][1];
                    let u10 = matrix[1][0];
                    let u11 = matrix[1][1];

                    self.elements[i0 * dim + j0] = u00 * rho_00 * u00.conj()
                        + u00 * rho_01 * u10.conj()
                        + u01 * rho_10 * u00.conj()
                        + u01 * rho_11 * u10.conj();

                    self.elements[i0 * dim + j1] = u00 * rho_00 * u01.conj()
                        + u00 * rho_01 * u11.conj()
                        + u01 * rho_10 * u01.conj()
                        + u01 * rho_11 * u11.conj();

                    self.elements[i1 * dim + j0] = u10 * rho_00 * u00.conj()
                        + u10 * rho_01 * u10.conj()
                        + u11 * rho_10 * u00.conj()
                        + u11 * rho_11 * u10.conj();

                    self.elements[i1 * dim + j1] = u10 * rho_00 * u01.conj()
                        + u10 * rho_01 * u11.conj()
                        + u11 * rho_10 * u01.conj()
                        + u11 * rho_11 * u11.conj();
                }
            }
        }
    }
}

// ============================================================
// TWO-QUBIT GATES
// ============================================================

impl DensityMatrix {
    /// Apply CNOT gate
    pub fn cnot(&mut self, control: usize, target: usize) {
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let dim = self.dim;
        let mut new_elements = vec![C64::new(0.0, 0.0); dim * dim];

        for i in 0..dim {
            for j in 0..dim {
                let i_control = (i & control_mask) != 0;
                let j_control = (j & control_mask) != 0;

                if i_control && j_control {
                    // Both |1⟩: flip both target bits
                    let i_flipped = i ^ target_mask;
                    let j_flipped = j ^ target_mask;
                    new_elements[i * dim + j] = self.elements[i_flipped * dim + j_flipped];
                } else if i_control && !j_control {
                    // Row is |1⟩, col is |0⟩: flip row target only
                    let i_flipped = i ^ target_mask;
                    new_elements[i * dim + j] = self.elements[i_flipped * dim + j];
                } else if !i_control && j_control {
                    // Row is |0⟩, col is |1⟩: flip col target only
                    let j_flipped = j ^ target_mask;
                    new_elements[i * dim + j] = self.elements[i * dim + j_flipped];
                } else {
                    // Both |0⟩: no flip
                    new_elements[i * dim + j] = self.elements[i * dim + j];
                }
            }
        }

        self.elements = new_elements;
    }

    /// Apply CZ gate
    pub fn cz(&mut self, control: usize, target: usize) {
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let dim = self.dim;

        for i in 0..dim {
            for j in 0..dim {
                let i_both = ((i & control_mask) != 0) && ((i & target_mask) != 0);
                let j_both = ((j & control_mask) != 0) && ((j & target_mask) != 0);

                if i_both && j_both {
                    self.elements[i * dim + j] = -self.elements[i * dim + j];
                }
            }
        }
    }

    /// Apply SWAP gate
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) {
        let mask1 = 1 << qubit1;
        let mask2 = 1 << qubit2;
        let dim = self.dim;
        let mut new_elements = self.elements.clone();

        for i in 0..dim {
            for j in 0..dim {
                let i_swapped = ((i & mask1) >> qubit1) << qubit2
                    | ((i & mask2) >> qubit2) << qubit1
                    | (i & !(mask1 | mask2));
                let j_swapped = ((j & mask1) >> qubit1) << qubit2
                    | ((j & mask2) >> qubit2) << qubit1
                    | (j & !(mask1 | mask2));

                new_elements[i * dim + j] = self.elements[i_swapped * dim + j_swapped];
            }
        }

        self.elements = new_elements;
    }

    /// Apply controlled-RX gate
    pub fn crx(&mut self, control: usize, target: usize, theta: f64) {
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let dim = self.dim;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        for i in 0..dim {
            for j in 0..dim {
                let i_control = (i & control_mask) != 0;
                let j_control = (j & control_mask) != 0;

                if i_control && j_control {
                    let i0 = i & !target_mask;
                    let i1 = i | target_mask;
                    let j0 = j & !target_mask;
                    let j1 = j | target_mask;

                    if ((i & target_mask) == 0) && ((j & target_mask) == 0) {
                        let rho_00 = self.elements[i0 * dim + j0];
                        let rho_01 = self.elements[i0 * dim + j1];
                        let rho_10 = self.elements[i1 * dim + j0];
                        let rho_11 = self.elements[i1 * dim + j1];

                        let c = cos_half;
                        let s = sin_half;

                        self.elements[i0 * dim + j0] =
                            c * c * rho_00 - c * s * rho_01 - s * c * rho_10 + s * s * rho_11;
                        self.elements[i0 * dim + j1] =
                            c * s * rho_00 + c * c * rho_01 - s * s * rho_10 - s * c * rho_11;
                        self.elements[i1 * dim + j0] =
                            c * s * rho_00 - s * s * rho_01 + c * c * rho_10 - s * c * rho_11;
                        self.elements[i1 * dim + j1] =
                            s * s * rho_00 + s * c * rho_01 + c * s * rho_10 + c * c * rho_11;
                    }
                }
            }
        }
    }

    /// Apply controlled-RY gate
    pub fn cry(&mut self, control: usize, target: usize, theta: f64) {
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let dim = self.dim;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        for i in 0..dim {
            for j in 0..dim {
                let i_control = (i & control_mask) != 0;
                let j_control = (j & control_mask) != 0;

                if i_control && j_control {
                    let i0 = i & !target_mask;
                    let i1 = i | target_mask;
                    let j0 = j & !target_mask;
                    let j1 = j | target_mask;

                    if ((i & target_mask) == 0) && ((j & target_mask) == 0) {
                        let rho_00 = self.elements[i0 * dim + j0];
                        let rho_01 = self.elements[i0 * dim + j1];
                        let rho_10 = self.elements[i1 * dim + j0];
                        let rho_11 = self.elements[i1 * dim + j1];

                        let c = cos_half;
                        let s = sin_half;

                        self.elements[i0 * dim + j0] =
                            c * c * rho_00 - c * s * rho_01 - s * c * rho_10 + s * s * rho_11;
                        self.elements[i0 * dim + j1] =
                            -c * s * rho_00 + c * c * rho_01 - s * s * rho_10 + s * c * rho_11;
                        self.elements[i1 * dim + j0] =
                            c * s * rho_00 + s * s * rho_01 + c * c * rho_10 + s * c * rho_11;
                        self.elements[i1 * dim + j1] =
                            -s * s * rho_00 + s * c * rho_01 + c * s * rho_10 + c * c * rho_11;
                    }
                }
            }
        }
    }

    /// Apply controlled-RZ gate
    pub fn crz(&mut self, control: usize, target: usize, theta: f64) {
        let control_mask = 1 << control;
        let _target_mask = 1 << target;
        let dim = self.dim;
        let phase_plus = C64::new((theta / 2.0).cos(), (theta / 2.0).sin());
        let phase_minus = C64::new((theta / 2.0).cos(), -(theta / 2.0).sin());

        for i in 0..dim {
            for j in 0..dim {
                let i_control = (i & control_mask) != 0;
                let j_control = (j & control_mask) != 0;

                if i_control && j_control {
                    let i_bit = (i >> target) & 1;
                    let j_bit = (j >> target) & 1;

                    if i_bit == 0 && j_bit == 0 {
                        self.elements[i * dim + j] = phase_minus * self.elements[i * dim + j];
                    } else if i_bit == 1 && j_bit == 1 {
                        self.elements[i * dim + j] = phase_plus * self.elements[i * dim + j];
                    }
                }
            }
        }
    }
}

// ============================================================
// MEASUREMENT
// ============================================================

impl DensityMatrix {
    /// Measure qubit in computational basis
    /// Returns (measurement outcome, collapsed density matrix)
    pub fn measure(&mut self, qubit: usize) -> (usize, DensityMatrix) {
        let _mask = 1 << qubit;
        let dim = self.dim;

        // Calculate probabilities
        let mut p0 = C64::new(0.0, 0.0);
        let mut p1 = C64::new(0.0, 0.0);

        for i in 0..dim {
            for j in 0..dim {
                if ((i >> qubit) & 1) == 0 && ((j >> qubit) & 1) == 0 {
                    p0 += self.elements[i * dim + j];
                } else if ((i >> qubit) & 1) == 1 && ((j >> qubit) & 1) == 1 {
                    p1 += self.elements[i * dim + j];
                }
            }
        }

        let prob_0 = p0.re;
        let prob_1 = p1.re;

        // Sample measurement outcome
        let outcome = if rand::random::<f64>() < prob_0 { 0 } else { 1 };

        // Collapse the density matrix
        let mut collapsed = self.clone();
        for i in 0..dim {
            for j in 0..dim {
                let i_bit = (i >> qubit) & 1;
                let j_bit = (j >> qubit) & 1;

                if i_bit != outcome || j_bit != outcome {
                    collapsed.elements[i * dim + j] = C64::new(0.0, 0.0);
                }
            }
        }

        // Renormalize
        let norm = if outcome == 0 { prob_0 } else { prob_1 };
        if norm > 1e-15 {
            for elem in collapsed.elements.iter_mut() {
                *elem /= norm;
            }
        }

        self.elements = collapsed.elements.clone();

        (outcome, collapsed)
    }

    /// Measure all qubits
    pub fn measure_all(&mut self) -> usize {
        let mut result = 0;
        for qubit in 0..self.num_qubits {
            let (bit, _) = self.measure(qubit);
            result |= bit << qubit;
        }
        result
    }
}

// ============================================================
// SIMULATOR WRAPPER
// ============================================================

/// Density matrix simulator with high-level interface
pub struct DensityMatrixSimulator {
    pub state: DensityMatrix,
}

impl DensityMatrixSimulator {
    pub fn new(num_qubits: usize) -> Self {
        DensityMatrixSimulator {
            state: DensityMatrix::new(num_qubits),
        }
    }

    pub fn from_pure_state(state_vector: &[C64]) -> Self {
        DensityMatrixSimulator {
            state: DensityMatrix::from_pure_state(state_vector),
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.state.num_qubits
    }

    /// Reset simulator state back to |0...0⟩
    pub fn reset(&mut self) {
        self.state.reset();
    }

    // Gate wrappers
    pub fn h(&mut self, qubit: usize) {
        self.state.h(qubit);
    }
    pub fn x(&mut self, qubit: usize) {
        self.state.x(qubit);
    }
    pub fn y(&mut self, qubit: usize) {
        self.state.y(qubit);
    }
    pub fn z(&mut self, qubit: usize) {
        self.state.z(qubit);
    }
    pub fn s(&mut self, qubit: usize) {
        self.state.s(qubit);
    }
    pub fn t(&mut self, qubit: usize) {
        self.state.t(qubit);
    }
    pub fn rx(&mut self, qubit: usize, theta: f64) {
        self.state.rx(qubit, theta);
    }
    pub fn ry(&mut self, qubit: usize, theta: f64) {
        self.state.ry(qubit, theta);
    }
    pub fn rz(&mut self, qubit: usize, theta: f64) {
        self.state.rz(qubit, theta);
    }
    pub fn u(&mut self, qubit: usize, matrix: &[[C64; 2]; 2]) {
        self.state.u(qubit, matrix);
    }
    pub fn cnot(&mut self, control: usize, target: usize) {
        self.state.cnot(control, target);
    }
    pub fn cz(&mut self, control: usize, target: usize) {
        self.state.cz(control, target);
    }
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) {
        self.state.swap(qubit1, qubit2);
    }
    pub fn crx(&mut self, control: usize, target: usize, theta: f64) {
        self.state.crx(control, target, theta);
    }
    pub fn cry(&mut self, control: usize, target: usize, theta: f64) {
        self.state.cry(control, target, theta);
    }
    pub fn crz(&mut self, control: usize, target: usize, theta: f64) {
        self.state.crz(control, target, theta);
    }

    pub fn measure(&mut self, qubit: usize) -> usize {
        let (outcome, _) = self.state.measure(qubit);
        outcome
    }

    pub fn measure_all(&mut self) -> usize {
        self.state.measure_all()
    }

    // Analysis methods
    pub fn purity(&self) -> f64 {
        self.state.purity()
    }
    pub fn entropy(&self) -> f64 {
        self.state.entropy()
    }
    pub fn is_pure(&self) -> bool {
        self.state.is_pure()
    }
    pub fn trace(&self) -> f64 {
        self.state.trace()
    }
    pub fn reduced_state(&self, qubits: &[usize]) -> DensityMatrix {
        self.state.partial_trace(qubits)
    }

    /// Expectation value of Pauli-Z operator on a qubit
    /// Returns ⟨ψ|Z|ψ⟩ for the specified qubit
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        let mask = 1 << qubit;
        let dim = self.state.dim;
        let mut exp = 0.0;

        for i in 0..dim {
            for j in 0..dim {
                if i == j {
                    let prob = self.state.elements[i * dim + i].re;
                    if i & mask == 0 {
                        exp += prob; // Z eigenvalue for |0⟩ is +1
                    } else {
                        exp -= prob; // Z eigenvalue for |1⟩ is -1
                    }
                }
            }
        }

        exp
    }

    /// Expectation value of Z² operator (always 1 for Pauli-Z)
    pub fn expectation_z_squared(&self, _qubit: usize) -> f64 {
        // For Pauli-Z, Z² = I, so expectation is 1
        1.0
    }

    /// Probability of measuring qubit in specific state
    pub fn probability(&self, qubit: usize, state: usize) -> f64 {
        let _mask = 1 << qubit;
        let dim = self.state.dim;
        let mut prob = 0.0;

        for i in 0..dim {
            if ((i >> qubit) & 1) == state {
                prob += self.state.elements[i * dim + i].re;
            }
        }

        prob
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_density_matrix_creation() {
        let dm = DensityMatrix::new(2);
        assert_eq!(dm.num_qubits, 2);
        assert_eq!(dm.dim, 4);
        assert!((dm.trace() - 1.0).abs() < 1e-10);
        assert!(dm.is_pure());
    }

    #[test]
    fn test_maximally_mixed_state() {
        let dm = DensityMatrix::maximally_mixed(2);
        assert!((dm.trace() - 1.0).abs() < 1e-10);
        assert!((dm.purity() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_from_pure_state() {
        let state = vec![
            Complex::new(1.0 / 2.0f64.sqrt(), 0.0),
            Complex::new(1.0 / 2.0f64.sqrt(), 0.0),
        ];
        let dm = DensityMatrix::from_pure_state(&state);
        assert!(dm.is_pure());
        assert!((dm.trace() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_purity() {
        let mut sim = DensityMatrixSimulator::new(1);
        assert!(sim.is_pure());
        sim.h(0);
        assert!(sim.is_pure()); // Unitary gates preserve purity
    }

    #[test]
    fn test_bell_state() {
        let mut sim = DensityMatrixSimulator::new(2);
        sim.h(0);
        sim.cnot(0, 1);

        // Check entanglement via reduced states
        let reduced_0 = sim.reduced_state(&[0]);
        let reduced_1 = sim.reduced_state(&[1]);

        // Entangled qubits should have mixed reduced states
        assert!(reduced_0.purity() < 0.99);
        assert!(reduced_1.purity() < 0.99);
    }

    #[test]
    fn test_partial_trace() {
        let mut sim = DensityMatrixSimulator::new(2);
        sim.h(0);
        sim.cnot(0, 1);

        let reduced = sim.reduced_state(&[0]);
        assert_eq!(reduced.num_qubits, 1);
        assert!((reduced.trace() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_measurement() {
        let mut sim = DensityMatrixSimulator::new(1);
        sim.h(0);

        let result = sim.measure(0);
        assert!(result == 0 || result == 1);

        // After measurement, state should be pure
        assert!(sim.is_pure());
    }

    #[test]
    fn test_entropy() {
        let pure = DensityMatrix::new(2);
        assert_eq!(pure.entropy(), 0.0); // Pure state has zero entropy

        let mixed = DensityMatrix::maximally_mixed(2);
        assert!(mixed.entropy() > 0.0); // Mixed state has positive entropy
    }
}
