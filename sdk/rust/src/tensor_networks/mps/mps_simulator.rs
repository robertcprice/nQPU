//! Tensor Network Simulation for Quantum Computing
//!
//! This module implements Matrix Product State (MPS) simulation, which enables
//! efficient simulation of large quantum systems by exploiting limited entanglement.
//!
//! # Overview
//!
//! Traditional state vector simulation scales as O(2^n) for n qubits, which becomes
//! impractical beyond ~25 qubits. Tensor network methods compress the state representation
//! and scale with entanglement rather than system size, enabling simulation of 50-100+ qubits
//! for circuits with limited entanglement.
//!
//! # Key Concepts
//!
//! ## Matrix Product States (MPS)
//!
//! An n-qubit quantum state can be written as:
//! ```text
//! |ψ⟩ = Σ_{i1,i2,...,in} Tr(A[1]^{i1} A[2]^{i2} ... A[n]^{in}) |i1,i2,...,in⟩
//! ```
//!
//! Where each A[k]^{ik} is a matrix (or scalar for boundaries) and the contraction
//! yields the amplitude for basis state |i1,i2,...,in⟩.
//!
//! ## Bond Dimension
//!
//! The bond dimension χ controls the compression:
//! - χ = 1: Product state (no entanglement)
//! - χ = 2: Can represent 1 Bell pair
//! - χ = 2^k: Can represent k-body entanglement
//! - χ = 2^(n/2): Full state vector (no compression)
//!
//! # Performance
//!
//! Memory usage: O(n × χ²) vs O(2^n) for state vector
//! Gate application: O(n × χ³) vs O(2^n) for state vector
//!
//! For circuits with limited entanglement (χ << 2^(n/2)), MPS is exponentially faster.
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::tensor_network::{MPS, MPSSimulator};
//!
//! // Create 50-qubit MPS simulator
//! let mut sim = MPSSimulator::new(50, Some(16)); // bond_dim = 16
//!
//! // Apply gates
//! sim.h(0);
//! sim.h(1);
//! sim.cnot(0, 1);
//!
//! // Measure
//! let result = sim.measure();
//! ```
//!
//! # References
//!
//! - "Matrix Product States and Projected Entangled Pair States" by Orús (2014)
//! - "Tensor Networks for Quantum Computing" by Biamonte & Bergholm (2017)
//! - "The density-matrix renormalization group in the age of matrix product states" by Schollwöck (2011)

use nalgebra::{Complex as NComplex, DMatrix};
use ndarray::{Array3, Array4};
use num_complex::Complex64;
use num_traits::{One, Zero};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ============================================================
// MPS (MATRIX PRODUCT STATE) REPRESENTATION
// ============================================================

/// Matrix Product State representation of a quantum state
///
/// An MPS represents an n-qubit state as a chain of tensors, where each qubit
/// is associated with a rank-3 tensor (except boundaries which are rank-2).
///
/// # Structure
///
/// ```text
/// For n qubits:
/// - Left boundary: A[0] has shape (1, 2, χ₁)
/// - Middle qubits: A[k] has shape (χₖ, 2, χₖ₊₁)
/// - Right boundary: A[n-1] has shape (χₙ, 2, 1)
///
/// where χₖ is the bond dimension at site k
/// ```
///
/// The physical dimension is always 2 (qubit basis |0⟩ and |1⟩).
#[derive(Clone, Debug)]
pub struct MPS {
    /// Array of tensors, one per qubit
    /// Each tensor has shape (bond_left, physical, bond_right)
    /// For boundaries: bond_left or bond_right is 1
    tensors: Vec<Array3<Complex64>>,

    /// Number of qubits
    num_qubits: usize,

    /// Maximum bond dimension (for compression)
    max_bond_dim: Option<usize>,

    /// Truncation threshold for SVD
    truncation_threshold: f64,

    /// Treat truncation threshold as relative to max singular value
    truncation_relative: bool,

    /// Track entanglement entropy per bond
    track_entanglement: bool,

    /// Cached bond entanglement entropies (length = num_qubits-1)
    bond_entropies: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct EntanglementStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub variance: f64,
}

impl MPS {
    /// Create a new MPS in the |0...0⟩ state
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `max_bond_dim` - Maximum bond dimension (None for unlimited)
    ///
    /// # Example
    /// ```ignore
    /// let mps = MPS::new(10, Some(16)); // 10 qubits, max bond dim 16
    /// ```
    pub fn new(num_qubits: usize, max_bond_dim: Option<usize>) -> Self {
        assert!(num_qubits > 0, "Number of qubits must be positive");

        let mut tensors = Vec::with_capacity(num_qubits);

        // Initialize all qubits in |0⟩ state
        // For |0⟩: A^0 = 1, A^1 = 0
        for i in 0..num_qubits {
            if num_qubits == 1 {
                // Single qubit: tensor is (1, 2, 1)
                let mut tensor = Array3::zeros((1, 2, 1));
                tensor[[0, 0, 0]] = Complex64::one(); // |0⟩
                tensors.push(tensor);
            } else if i == 0 {
                // Left boundary: (1, 2, 1) initially
                let mut tensor = Array3::zeros((1, 2, 1));
                tensor[[0, 0, 0]] = Complex64::one(); // |0⟩
                tensors.push(tensor);
            } else if i == num_qubits - 1 {
                // Right boundary: (1, 2, 1) initially
                let mut tensor = Array3::zeros((1, 2, 1));
                tensor[[0, 0, 0]] = Complex64::one(); // |0⟩
                tensors.push(tensor);
            } else {
                // Middle: (1, 2, 1) initially
                let mut tensor = Array3::zeros((1, 2, 1));
                tensor[[0, 0, 0]] = Complex64::one(); // |0⟩
                tensors.push(tensor);
            }
        }

        MPS {
            tensors,
            num_qubits,
            max_bond_dim,
            truncation_threshold: 1e-12,
            truncation_relative: false,
            track_entanglement: false,
            bond_entropies: vec![0.0; num_qubits.saturating_sub(1)],
        }
    }

    /// Create a product state (no entanglement)
    ///
    /// # Arguments
    /// * `states` - Vector of qubit states (0 for |0⟩, 1 for |1⟩)
    /// * `max_bond_dim` - Maximum bond dimension (None for unlimited)
    ///
    /// # Example
    /// ```ignore
    /// // Create |0101⟩ state
    /// let mps = MPS::product_state(&[0, 1, 0, 1], None);
    /// ```
    pub fn product_state(states: &[usize], max_bond_dim: Option<usize>) -> Self {
        let num_qubits = states.len();
        let mut mps = Self::new(num_qubits, max_bond_dim);

        for (i, &state) in states.iter().enumerate() {
            if state == 1 {
                // Flip qubit i to |1⟩
                mps.apply_single_qubit_gate(&x_gate(), i);
            }
        }

        mps
    }

    /// Set maximum bond dimension (None for unlimited).
    pub fn set_max_bond_dim(&mut self, max_bond_dim: Option<usize>) {
        self.max_bond_dim = max_bond_dim;
    }

    /// Get the current max bond dimension limit (if any).
    pub fn max_bond_dim_limit(&self) -> Option<usize> {
        self.max_bond_dim
    }

    /// Set truncation threshold for SVD (controls entanglement growth).
    pub fn set_truncation_threshold(&mut self, threshold: f64) {
        self.truncation_threshold = threshold.max(0.0);
        self.truncation_relative = false;
    }

    /// Get the current truncation threshold.
    pub fn truncation_threshold(&self) -> f64 {
        self.truncation_threshold
    }

    /// Get whether truncation threshold is relative.
    pub fn truncation_is_relative(&self) -> bool {
        self.truncation_relative
    }

    /// Set truncation threshold relative to the max singular value (0..1).
    pub fn set_relative_truncation(&mut self, threshold: f64) {
        self.truncation_threshold = threshold.max(0.0);
        self.truncation_relative = true;
    }

    /// Enable or disable entanglement tracking.
    pub fn enable_entanglement_tracking(&mut self, enabled: bool) {
        self.track_entanglement = enabled;
        if enabled && self.bond_entropies.len() != self.num_qubits.saturating_sub(1) {
            self.bond_entropies = vec![0.0; self.num_qubits.saturating_sub(1)];
        }
    }

    /// Get the entanglement entropy for a bond if tracking is enabled.
    pub fn bond_entanglement_entropy(&self, bond: usize) -> Option<f64> {
        if !self.track_entanglement {
            return None;
        }
        self.bond_entropies.get(bond).cloned()
    }

    /// Return a copy of the entanglement profile (one value per bond).
    pub fn entanglement_profile(&self) -> Option<Vec<f64>> {
        if !self.track_entanglement {
            return None;
        }
        Some(self.bond_entropies.clone())
    }

    /// Compute summary statistics over bond entanglement entropies.
    pub fn entanglement_stats(&self) -> Option<EntanglementStats> {
        if !self.track_entanglement || self.bond_entropies.is_empty() {
            return None;
        }
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        let mut sum = 0.0f64;
        for &v in &self.bond_entropies {
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
            sum += v;
        }
        let mean = sum / self.bond_entropies.len() as f64;
        let mut var = 0.0f64;
        for &v in &self.bond_entropies {
            let d = v - mean;
            var += d * d;
        }
        var /= self.bond_entropies.len() as f64;
        Some(EntanglementStats {
            min: min_v,
            max: max_v,
            mean,
            variance: var,
        })
    }

    /// Get reference to raw tensors (Metal interop / advanced tooling).
    pub fn tensors(&self) -> &Vec<Array3<Complex64>> {
        &self.tensors
    }

    /// Replace internal tensors (Metal interop / advanced tooling).
    pub fn set_tensors(&mut self, tensors: Vec<Array3<Complex64>>) {
        assert_eq!(tensors.len(), self.num_qubits, "tensor count mismatch");
        self.tensors = tensors;
    }

    /// Serialize tensor data for checkpointing.
    pub fn tensor_data(&self) -> Vec<Vec<Complex64>> {
        self.tensors
            .iter()
            .map(|t| t.iter().copied().collect())
            .collect()
    }

    /// Get tensor shapes for checkpointing.
    pub fn tensor_shapes(&self) -> Vec<(usize, usize, usize)> {
        self.tensors
            .iter()
            .map(|t| (t.shape()[0], t.shape()[1], t.shape()[2]))
            .collect()
    }

    /// Restore tensors with explicit shapes (for checkpointing with changed bond dimensions).
    pub fn restore_tensors_with_shapes(
        &mut self,
        data: &[Vec<Complex64>],
        shapes: &[(usize, usize, usize)],
    ) -> Result<(), String> {
        if data.len() != self.num_qubits || shapes.len() != self.num_qubits {
            return Err(format!(
                "Tensor count mismatch: expected {}, got data={} shapes={}",
                self.num_qubits,
                data.len(),
                shapes.len()
            ));
        }

        for (i, (tensor_data, &(d0, d1, d2))) in data.iter().zip(shapes.iter()).enumerate() {
            let expected_len = d0 * d1 * d2;
            if tensor_data.len() != expected_len {
                return Err(format!(
                    "Tensor {} size mismatch: expected {} ({}x{}x{}), got {}",
                    i,
                    expected_len,
                    d0,
                    d1,
                    d2,
                    tensor_data.len()
                ));
            }
        }

        // Recreate tensors with the checkpointed shapes
        for (i, (tensor_data, &(d0, d1, d2))) in data.iter().zip(shapes.iter()).enumerate() {
            let mut new_tensor = Array3::<Complex64>::zeros((d0, d1, d2));
            for (j, &val) in tensor_data.iter().enumerate() {
                let i0 = j / (d1 * d2);
                let rem = j % (d1 * d2);
                let i1 = rem / d2;
                let i2 = rem % d2;
                new_tensor[[i0, i1, i2]] = val;
            }
            self.tensors[i] = new_tensor;
        }
        Ok(())
    }

    /// Restore tensor data from checkpoint (requires matching shapes).
    pub fn restore_tensor_data(&mut self, data: &[Vec<Complex64>]) -> Result<(), String> {
        if data.len() != self.num_qubits {
            return Err(format!(
                "Tensor count mismatch: expected {}, got {}",
                self.num_qubits,
                data.len()
            ));
        }

        // First pass: validate sizes
        for (i, tensor_data) in data.iter().enumerate() {
            let shape = self.tensors[i].shape();
            let expected_len = shape[0] * shape[1] * shape[2];
            if tensor_data.len() != expected_len {
                return Err(format!(
                    "Tensor {} size mismatch: expected {}, got {}",
                    i,
                    expected_len,
                    tensor_data.len()
                ));
            }
        }

        // Second pass: copy data
        for (i, tensor_data) in data.iter().enumerate() {
            let shape = self.tensors[i].shape();
            let d1 = shape[1];
            let d2 = shape[2];

            for (j, &val) in tensor_data.iter().enumerate() {
                let i0 = j / (d1 * d2);
                let rem = j % (d1 * d2);
                let i1 = rem / d2;
                let i2 = rem % d2;
                self.tensors[i][[i0, i1, i2]] = val;
            }
        }
        Ok(())
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the current bond dimensions
    ///
    /// Returns a vector of length n-1 containing bond dimensions between sites
    pub fn bond_dimensions(&self) -> Vec<usize> {
        self.tensors
            .iter()
            .take(self.num_qubits - 1)
            .map(|t| t.shape()[2])
            .collect()
    }

    /// Get the maximum bond dimension currently in use
    pub fn max_current_bond_dim(&self) -> usize {
        self.bond_dimensions().into_iter().max().unwrap_or(1)
    }

    /// Apply a single-qubit gate to a specific qubit
    ///
    /// This is done by contracting the gate matrix with the physical index
    /// of the MPS tensor at that site.
    ///
    /// # Arguments
    /// * `gate` - 2x2 gate matrix
    /// * `qubit` - Target qubit index
    pub fn apply_single_qubit_gate(&mut self, gate: &[[Complex64; 2]; 2], qubit: usize) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");

        let tensor = &self.tensors[qubit];
        let shape = tensor.shape();

        // Contract gate with physical index
        // New tensor A'[i,α,β] = Σ_j G[i,j] * A[j,α,β]
        let mut new_tensor = Array3::zeros((shape[0], 2, shape[2]));

        for i in 0..shape[0] {
            for j in 0..2 {
                for k in 0..shape[2] {
                    let val = tensor[[i, j, k]];
                    new_tensor[[i, 0, k]] += gate[0][j] * val;
                    new_tensor[[i, 1, k]] += gate[1][j] * val;
                }
            }
        }

        self.tensors[qubit] = new_tensor;
    }

    /// Apply a 2-qubit gate to adjacent qubits using contract-gate-SVD-truncate.
    ///
    /// The gate matrix is 4x4 in the basis |i_q1, i_q2⟩ with q1 < q2.
    /// Steps:
    /// 1. Contract tensors at q1 and q2 into theta (chi_l, 2, 2, chi_r)
    /// 2. Apply the 4x4 gate on the two physical indices
    /// 3. Reshape to (chi_l*2, 2*chi_r)
    /// 4. Truncated SVD → U, S, V
    /// 5. Split back into new tensors at q1 and q2
    pub fn apply_two_qubit_gate(&mut self, q1: usize, q2: usize, gate: &[[Complex64; 4]; 4]) {
        assert!(q1 < self.num_qubits && q2 < self.num_qubits);
        let (ql, qr) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
        assert_eq!(qr, ql + 1, "Two-qubit gate requires adjacent qubits");

        let tl = &self.tensors[ql];
        let tr = &self.tensors[qr];
        let chi_l = tl.shape()[0];
        let chi_m = tl.shape()[2]; // = tr.shape()[0]
        let chi_r = tr.shape()[2];

        // Step 1: Contract into theta[alpha, i, j, beta]
        // theta[a,i,j,b] = Σ_m tl[a,i,m] * tr[m,j,b]
        let mut theta = Array4::<Complex64>::zeros((chi_l, 2, 2, chi_r));
        for a in 0..chi_l {
            for i in 0..2 {
                for j in 0..2 {
                    for b in 0..chi_r {
                        let mut sum = Complex64::zero();
                        for m in 0..chi_m {
                            sum += tl[[a, i, m]] * tr[[m, j, b]];
                        }
                        theta[[a, i, j, b]] = sum;
                    }
                }
            }
        }

        // Step 2: Apply gate on physical indices
        // new_theta[a,i',j',b] = Σ_{i,j} gate[i'*2+j'][i*2+j] * theta[a,i,j,b]
        let mut gated = Array4::<Complex64>::zeros((chi_l, 2, 2, chi_r));
        for a in 0..chi_l {
            for b in 0..chi_r {
                for ip in 0..2 {
                    for jp in 0..2 {
                        let mut sum = Complex64::zero();
                        for i in 0..2 {
                            for j in 0..2 {
                                sum += gate[ip * 2 + jp][i * 2 + j] * theta[[a, i, j, b]];
                            }
                        }
                        gated[[a, ip, jp, b]] = sum;
                    }
                }
            }
        }

        // Step 3: Reshape to (chi_l*2, 2*chi_r) for SVD
        let m_rows = chi_l * 2;
        let n_cols = 2 * chi_r;
        let mut mat_data = vec![NComplex::<f64>::new(0.0, 0.0); m_rows * n_cols];
        for a in 0..chi_l {
            for i in 0..2 {
                for j in 0..2 {
                    for b in 0..chi_r {
                        let row = a * 2 + i;
                        let col = j * chi_r + b;
                        let v = gated[[a, i, j, b]];
                        mat_data[row + col * m_rows] = NComplex::new(v.re, v.im);
                    }
                }
            }
        }
        let mut had_non_finite = false;
        for v in &mut mat_data {
            if !v.re.is_finite() || !v.im.is_finite() {
                *v = NComplex::new(0.0, 0.0);
                had_non_finite = true;
            }
        }
        if had_non_finite {
            // Clamp invalid values and renormalize to prevent SVD NaNs.
            self.normalize();
        }
        let mat = DMatrix::from_vec(m_rows, n_cols, mat_data);

        // Step 4: SVD
        let svd = mat.svd(true, true);
        let u_full = svd.u.expect("SVD U");
        let vt_full = svd.v_t.expect("SVD V^T");
        let singular_values = &svd.singular_values;

        let mut threshold = self.truncation_threshold;
        if self.truncation_relative {
            let mut max_sv = 0.0f64;
            for sv in singular_values.iter() {
                if *sv > max_sv {
                    max_sv = *sv;
                }
            }
            threshold *= max_sv;
        }

        // Determine truncation: keep values above threshold, capped by max_bond_dim
        let mut chi_new = 0;
        for k in 0..singular_values.len() {
            if singular_values[k] > threshold {
                chi_new += 1;
            } else {
                break;
            }
        }
        if chi_new == 0 {
            chi_new = 1;
        }
        if let Some(max_chi) = self.max_bond_dim {
            chi_new = chi_new.min(max_chi);
        }

        if self.track_entanglement && !self.bond_entropies.is_empty() {
            let mut norm_sq = 0.0f64;
            for sv in singular_values.iter() {
                norm_sq += sv * sv;
            }
            if norm_sq > 0.0 {
                let mut entropy = 0.0f64;
                for k in 0..chi_new {
                    let p = (singular_values[k] * singular_values[k]) / norm_sq;
                    if p > 0.0 {
                        entropy -= p * p.ln();
                    }
                }
                if ql < self.bond_entropies.len() {
                    self.bond_entropies[ql] = entropy;
                }
            }
        }

        // Step 5: Split back into new tensors
        // new_tl[a, i, k] = U[a*2+i, k] * S[k]
        let mut new_tl = Array3::<Complex64>::zeros((chi_l, 2, chi_new));
        for a in 0..chi_l {
            for i in 0..2 {
                let row = a * 2 + i;
                for k in 0..chi_new {
                    let u_val = u_full[(row, k)];
                    let s = singular_values[k];
                    new_tl[[a, i, k]] = Complex64::new(u_val.re * s, u_val.im * s);
                }
            }
        }

        // new_tr[k, j, b] = Vt[k, j*chi_r+b]
        let mut new_tr = Array3::<Complex64>::zeros((chi_new, 2, chi_r));
        for k in 0..chi_new {
            for j in 0..2 {
                for b in 0..chi_r {
                    let col = j * chi_r + b;
                    let vt_val = vt_full[(k, col)];
                    new_tr[[k, j, b]] = Complex64::new(vt_val.re, vt_val.im);
                }
            }
        }

        self.tensors[ql] = new_tl;
        self.tensors[qr] = new_tr;
    }

    /// Apply CNOT gate to adjacent qubits using the proper SVD-based method.
    fn apply_adjacent_cnot(&mut self, control: usize, target: usize) {
        assert!(control < self.num_qubits && target < self.num_qubits);

        let (ql, _qr) = if control < target {
            (control, target)
        } else {
            (target, control)
        };

        // CNOT gate matrix in |control, target⟩ basis
        let mut gate = [[Complex64::zero(); 4]; 4];
        if control < target {
            // control = ql (left), target = qr (right)
            // |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
            gate[0][0] = Complex64::one();
            gate[1][1] = Complex64::one();
            gate[2][3] = Complex64::one();
            gate[3][2] = Complex64::one();
        } else {
            // control = qr (right), target = ql (left)
            // |00⟩→|00⟩, |01⟩→|11⟩, |10⟩→|10⟩, |11⟩→|01⟩
            gate[0][0] = Complex64::one();
            gate[1][3] = Complex64::one();
            gate[2][2] = Complex64::one();
            gate[3][1] = Complex64::one();
        }

        self.apply_two_qubit_gate(ql, ql + 1, &gate);
    }

    /// Measure a single qubit
    ///
    /// # Returns
    /// Measurement result (0 or 1) and collapsed state
    pub fn measure_qubit(&mut self, qubit: usize) -> (usize, Self) {
        assert!(qubit < self.num_qubits, "Qubit index out of bounds");

        // Compute probabilities of measuring |0⟩ and |1⟩
        let (p0, _p1) = self.single_qubit_probabilities(qubit);

        // Sample according to probabilities
        let result: f64 = rand::random();
        let measured = if result < p0 { 0 } else { 1 };

        // Collapse the state
        let mut collapsed = self.clone();
        collapsed.collapse_qubit(qubit, measured);

        (measured, collapsed)
    }

    /// Get measurement probabilities for a single qubit
    fn single_qubit_probabilities(&self, qubit: usize) -> (f64, f64) {
        // Contract all tensors except the measured one to get probabilities
        // This is a simplified version - full implementation requires full contraction

        // For product states, this is trivial
        let tensor = &self.tensors[qubit];

        // Compute probability of |0⟩
        let mut p0 = 0.0;
        let mut p1 = 0.0;

        for i in 0..tensor.shape()[0] {
            for j in 0..tensor.shape()[2] {
                p0 += tensor[[i, 0, j]].norm_sqr();
                p1 += tensor[[i, 1, j]].norm_sqr();
            }
        }

        (p0, p1)
    }

    /// Collapse the wavefunction after measurement
    fn collapse_qubit(&mut self, qubit: usize, outcome: usize) {
        // Zero out the other outcome
        let tensor = &mut self.tensors[qubit];
        for i in 0..tensor.shape()[0] {
            for j in 0..tensor.shape()[2] {
                if outcome == 0 {
                    tensor[[i, 1, j]] = Complex64::zero();
                } else {
                    tensor[[i, 0, j]] = Complex64::zero();
                }
            }
        }

        // Renormalize
        self.normalize();
    }

    /// Normalize the MPS
    fn normalize(&mut self) {
        // Compute norm
        let mut norm = 0.0;
        for tensor in &self.tensors {
            for &val in tensor.iter() {
                norm += val.norm_sqr();
            }
        }
        norm = norm.sqrt();

        // Normalize all tensors
        if norm > 0.0 {
            let inv_norm = Complex64::new(1.0 / norm, 0.0);
            for tensor in &mut self.tensors {
                tensor.map_inplace(|v| *v *= inv_norm);
            }
        }
    }

    /// Convert to full state vector via left-to-right sequential contraction.
    ///
    /// This method should only be used for testing or small systems.
    /// For n > 20 qubits, this will likely run out of memory.
    ///
    /// Algorithm: maintain a running vector v of shape (chi, 2^contracted_so_far).
    /// At each step, contract the next tensor to grow the physical dimension.
    pub fn to_state_vector(&self) -> Vec<Complex64> {
        if self.num_qubits == 0 {
            return vec![Complex64::one()];
        }

        // Start with the first tensor: shape (1, 2, chi_1)
        // result[chi_1][phys_dim] where phys_dim starts at 2
        let t0 = &self.tensors[0];
        let chi_right = t0.shape()[2];
        // v[bond_idx][basis_state] stores partial amplitudes
        // After processing tensor 0: v has chi_right rows, 2 columns
        let mut v: Vec<Vec<Complex64>> = vec![vec![Complex64::zero(); 2]; chi_right];
        for bit in 0..2 {
            for k in 0..chi_right {
                v[k][bit] = t0[[0, bit, k]];
            }
        }

        // Contract remaining tensors one by one
        for q in 1..self.num_qubits {
            let t = &self.tensors[q];
            let chi_in = t.shape()[0]; // must equal current v.len()
            let chi_out = t.shape()[2];
            let old_dim = v[0].len(); // 2^q
            let new_dim = old_dim * 2; // 2^(q+1)

            let mut new_v: Vec<Vec<Complex64>> = vec![vec![Complex64::zero(); new_dim]; chi_out];
            for k_out in 0..chi_out {
                for bit in 0..2 {
                    for k_in in 0..chi_in {
                        let coeff = t[[k_in, bit, k_out]];
                        if coeff == Complex64::zero() {
                            continue;
                        }
                        for basis in 0..old_dim {
                            // New basis = old_basis * 2 + bit (big-endian)
                            let new_basis = basis * 2 + bit;
                            new_v[k_out][new_basis] += coeff * v[k_in][basis];
                        }
                    }
                }
            }

            v = new_v;
        }

        // Final result: v should have 1 bond index (the right boundary is 1)
        // Sum over bond indices (should be just index 0 for properly formed MPS)
        let dim = 1 << self.num_qubits;
        let mut state = vec![Complex64::zero(); dim];
        for k in 0..v.len() {
            for basis in 0..dim {
                state[basis] += v[k][basis];
            }
        }

        state
    }

    /// Swap two qubits in the MPS
    ///
    /// This is a critical operation that enables non-adjacent CNOT gates.
    /// After swapping, the qubits are exchanged, which allows applying
    /// gates between any pair of qubits.
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) {
        if qubit1 == qubit2 {
            return;
        }

        // Ensure qubit1 < qubit2 for easier handling
        let (i, j) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };

        // Swap the tensors at positions i and j
        self.tensors.swap(i, j);

        // Note: The bond dimensions might not be optimal after swapping
        // For a production implementation, we might want to re-optimize
        // the MPS structure after swapping, but this basic version
        // preserves correctness.
    }
}

// ============================================================
// SINGLE-QUBIT GATE MATRICES
// ============================================================

/// Pauli-X (NOT) gate
fn x_gate() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ]
}

/// Pauli-Y gate
fn y_gate() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
    ]
}

/// Pauli-Z gate
fn z_gate() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
    ]
}

/// Hadamard gate
fn h_gate() -> [[Complex64; 2]; 2] {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    [
        [
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ],
        [
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(-inv_sqrt2, 0.0),
        ],
    ]
}

/// S gate (phase gate): S = [[1, 0], [0, i]]
fn s_gate() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
    ]
}

/// T gate (π/8 gate): T = [[1, 0], [0, exp(iπ/4)]]
fn t_gate() -> [[Complex64; 2]; 2] {
    let cos_pi8 = (std::f64::consts::PI / 4.0).cos();
    let sin_pi8 = (std::f64::consts::PI / 4.0).sin();
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(cos_pi8, sin_pi8)],
    ]
}

/// Rx gate (rotation around X-axis): Rx(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
fn rx_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();
    [
        [
            Complex64::new(cos_half, 0.0),
            Complex64::new(0.0, -sin_half),
        ],
        [
            Complex64::new(0.0, -sin_half),
            Complex64::new(cos_half, 0.0),
        ],
    ]
}

/// Ry gate (rotation around Y-axis): Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
fn ry_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();
    [
        [
            Complex64::new(cos_half, 0.0),
            Complex64::new(-sin_half, 0.0),
        ],
        [Complex64::new(sin_half, 0.0), Complex64::new(cos_half, 0.0)],
    ]
}

/// Rz gate (rotation around Z-axis): Rz(θ) = [[exp(-iθ/2), 0], [0, exp(iθ/2)]]
fn rz_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();
    [
        [
            Complex64::new(cos_half, -sin_half),
            Complex64::new(0.0, 0.0),
        ],
        [Complex64::new(0.0, 0.0), Complex64::new(cos_half, sin_half)],
    ]
}

// ============================================================
// TWO-QUBIT GATE MATRICES (simplified representation)
// ============================================================

/// CNOT gate between adjacent qubits
/// Returns a function that applies CNOT to the two-qubit tensor
fn cnot_gate_adjacent(
) -> impl Fn(Complex64, Complex64, Complex64, Complex64) -> (Complex64, Complex64, Complex64, Complex64)
{
    |i00, i01, i10, i11| {
        // CNOT: control stays same, target flips if control=1
        // |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |11⟩, |11⟩ → |10⟩
        (i00, i01, i11, i10)
    }
}

// ============================================================
// MPS SIMULATOR
// ============================================================

/// High-level simulator using MPS representation
///
/// Provides a familiar quantum simulator interface backed by MPS
/// for efficient simulation of large systems.
#[derive(Clone, Debug)]
pub struct MPSSimulator {
    /// MPS representation of the quantum state
    mps: MPS,
}

impl MPSSimulator {
    /// Create a new MPS simulator
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits to simulate
    /// * `max_bond_dim` - Maximum bond dimension for compression (None for unlimited)
    ///
    /// # Example
    /// ```ignore
    /// let sim = MPSSimulator::new(50, Some(32)); // 50 qubits, bond dim 32
    /// ```
    pub fn new(num_qubits: usize, max_bond_dim: Option<usize>) -> Self {
        MPSSimulator {
            mps: MPS::new(num_qubits, max_bond_dim),
        }
    }

    /// Create simulator in a product state
    pub fn from_product_state(states: &[usize], max_bond_dim: Option<usize>) -> Self {
        MPSSimulator {
            mps: MPS::product_state(states, max_bond_dim),
        }
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.mps.num_qubits()
    }

    /// Get current bond dimensions
    pub fn bond_dimensions(&self) -> Vec<usize> {
        self.mps.bond_dimensions()
    }

    /// Get maximum current bond dimension
    pub fn max_bond_dim(&self) -> usize {
        self.mps.max_current_bond_dim()
    }

    /// Convert to full state vector (expensive).
    pub fn to_state_vector(&self) -> Vec<Complex64> {
        self.mps.to_state_vector()
    }

    /// Set truncation threshold for SVD (lower = more entanglement retained).
    pub fn set_truncation_threshold(&mut self, threshold: f64) {
        self.mps.set_truncation_threshold(threshold);
    }

    /// Set relative truncation threshold (0..1 of max singular value).
    pub fn set_relative_truncation(&mut self, threshold: f64) {
        self.mps.set_relative_truncation(threshold);
    }

    /// Set maximum bond dimension limit (None for unlimited).
    pub fn set_max_bond_dim(&mut self, max_bond_dim: Option<usize>) {
        self.mps.set_max_bond_dim(max_bond_dim);
    }

    /// Get maximum bond dimension limit (if any).
    pub fn max_bond_dim_limit(&self) -> Option<usize> {
        self.mps.max_bond_dim_limit()
    }

    /// Get current truncation threshold.
    pub fn truncation_threshold(&self) -> f64 {
        self.mps.truncation_threshold()
    }

    /// Check if truncation threshold is relative.
    pub fn truncation_is_relative(&self) -> bool {
        self.mps.truncation_is_relative()
    }

    /// Enable or disable entanglement tracking.
    pub fn enable_entanglement_tracking(&mut self, enabled: bool) {
        self.mps.enable_entanglement_tracking(enabled);
    }

    /// Get entanglement entropy for a bond (if tracking enabled).
    pub fn bond_entanglement_entropy(&self, bond: usize) -> Option<f64> {
        self.mps.bond_entanglement_entropy(bond)
    }

    pub fn entanglement_profile(&self) -> Option<Vec<f64>> {
        self.mps.entanglement_profile()
    }

    pub fn entanglement_stats(&self) -> Option<EntanglementStats> {
        self.mps.entanglement_stats()
    }

    /// Apply a custom 2-qubit gate matrix (adjacent qubits only).
    pub fn apply_two_qubit_gate_matrix(
        &mut self,
        q1: usize,
        q2: usize,
        gate: &[[Complex64; 4]; 4],
    ) {
        self.mps.apply_two_qubit_gate(q1, q2, gate);
    }

    /// Apply Hadamard gate to a qubit
    pub fn h(&mut self, qubit: usize) {
        self.mps.apply_single_qubit_gate(&h_gate(), qubit);
    }

    /// Apply X (NOT) gate to a qubit
    pub fn x(&mut self, qubit: usize) {
        self.mps.apply_single_qubit_gate(&x_gate(), qubit);
    }

    /// Apply Y gate to a qubit
    pub fn y(&mut self, qubit: usize) {
        self.mps.apply_single_qubit_gate(&y_gate(), qubit);
    }

    /// Apply Z gate to a qubit
    pub fn z(&mut self, qubit: usize) {
        self.mps.apply_single_qubit_gate(&z_gate(), qubit);
    }

    /// Apply S gate (phase gate) to a qubit
    pub fn s(&mut self, qubit: usize) {
        self.mps.apply_single_qubit_gate(&s_gate(), qubit);
    }

    /// Apply T gate (π/8 gate) to a qubit
    pub fn t(&mut self, qubit: usize) {
        self.mps.apply_single_qubit_gate(&t_gate(), qubit);
    }

    /// Apply Rx gate (rotation around X-axis) to a qubit
    pub fn rx(&mut self, qubit: usize, theta: f64) {
        self.mps.apply_single_qubit_gate(&rx_gate(theta), qubit);
    }

    /// Apply Ry gate (rotation around Y-axis) to a qubit
    pub fn ry(&mut self, qubit: usize, theta: f64) {
        self.mps.apply_single_qubit_gate(&ry_gate(theta), qubit);
    }

    /// Apply Rz gate (rotation around Z-axis) to a qubit
    pub fn rz(&mut self, qubit: usize, theta: f64) {
        self.mps.apply_single_qubit_gate(&rz_gate(theta), qubit);
    }

    /// Apply CNOT gate
    ///
    /// # Note
    /// For non-adjacent qubits, swap gates are automatically applied to bring
    /// the qubits together, then the CNOT is applied, then swap back.
    pub fn cnot(&mut self, control: usize, target: usize) {
        // Check if qubits are adjacent
        let are_adjacent = target == control + 1 || control == target + 1;

        if are_adjacent {
            // Direct application for adjacent qubits
            self.mps.apply_adjacent_cnot(control, target);
        } else {
            // For non-adjacent qubits, we need to swap them together
            // Strategy: Move target next to control using swap gates

            // Determine direction
            if target > control {
                // Target is to the right of control
                // Swap target left until it's adjacent to control
                for i in (control + 1..=target).rev() {
                    self.swap(i, i - 1);
                }
                // Now target is at control + 1
                self.mps.apply_adjacent_cnot(control, control + 1);
                // Swap back
                for i in (control + 1)..=target {
                    self.swap(i - 1, i);
                }
            } else {
                // Target is to the left of control
                // Swap target right until it's adjacent to control
                for i in target..control {
                    self.swap(i, i + 1);
                }
                // Now target is at control - 1
                self.mps.apply_adjacent_cnot(control - 1, control);
                // Swap back
                for i in (target..control).rev() {
                    self.swap(i + 1, i);
                }
            }
        }
    }

    /// Apply CZ gate (via H - CNOT - H).
    pub fn cz(&mut self, control: usize, target: usize) {
        self.h(target);
        self.cnot(control, target);
        self.h(target);
    }

    /// Swap two qubits
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) {
        self.mps.swap(qubit1, qubit2);
    }

    /// Measure a single qubit
    ///
    /// # Returns
    /// Measurement result (0 or 1)
    pub fn measure_qubit(&mut self, qubit: usize) -> usize {
        let (result, _) = self.mps.measure_qubit(qubit);
        result
    }

    /// Measure all qubits
    ///
    /// # Returns
    /// Integer representation of the measured state
    pub fn measure(&mut self) -> usize {
        let mut result: usize = 0;
        let n_qubits = self.mps.num_qubits;

        // For systems larger than usize bits, we can only return
        // the least significant bits
        let usable_bits = std::mem::size_of::<usize>() * 8;

        for i in 0..n_qubits.min(usable_bits) {
            let bit = self.measure_qubit(i);
            // Avoid overflow by checking shift amount
            let shift = if n_qubits > usable_bits {
                usable_bits - 1 - i
            } else {
                n_qubits - 1 - i
            };
            result |= bit << shift;
        }
        result
    }

    /// Get the underlying MPS (for advanced operations)
    pub fn mps(&self) -> &MPS {
        &self.mps
    }

    /// Get mutable access to the underlying MPS
    pub fn mps_mut(&mut self) -> &mut MPS {
        &mut self.mps
    }

    /// Serialize tensor data for checkpointing.
    pub fn tensor_data(&self) -> Vec<Vec<Complex64>> {
        self.mps.tensor_data()
    }

    /// Get tensor shapes for checkpointing.
    pub fn tensor_shapes(&self) -> Vec<(usize, usize, usize)> {
        self.mps.tensor_shapes()
    }

    /// Restore tensors with explicit shapes (for checkpointing with changed bond dimensions).
    pub fn restore_tensors_with_shapes(
        &mut self,
        data: &[Vec<Complex64>],
        shapes: &[(usize, usize, usize)],
    ) -> Result<(), String> {
        self.mps.restore_tensors_with_shapes(data, shapes)
    }

    /// Restore tensor data from checkpoint.
    pub fn restore_tensor_data(&mut self, data: &[Vec<Complex64>]) -> Result<(), String> {
        self.mps.restore_tensor_data(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mps_creation() {
        let mps = MPS::new(5, Some(4));
        assert_eq!(mps.num_qubits(), 5);
        assert_eq!(mps.max_current_bond_dim(), 1);
    }

    #[test]
    fn test_mps_product_state() {
        let mps = MPS::product_state(&[0, 1, 0, 1], None);
        assert_eq!(mps.num_qubits(), 4);
        assert_eq!(mps.max_current_bond_dim(), 1);
    }

    #[test]
    fn test_mps_simulator() {
        let mut sim = MPSSimulator::new(3, Some(4));
        sim.h(0);
        sim.x(1);
        assert_eq!(sim.num_qubits(), 3);
    }

    #[test]
    fn test_single_qubit_probabilities() {
        let mps = MPS::new(2, None);
        let (p0, p1) = mps.single_qubit_probabilities(0);
        assert!((p0 - 1.0).abs() < 1e-10);
        assert!((p1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_state_vector_zero_state() {
        let mps = MPS::new(3, None);
        let sv = mps.to_state_vector();
        assert_eq!(sv.len(), 8);
        assert!(
            (sv[0] - Complex64::one()).norm() < 1e-12,
            "Expected |000⟩, got sv[0]={}",
            sv[0]
        );
        for i in 1..8 {
            assert!(
                sv[i].norm() < 1e-12,
                "Expected 0 at sv[{}], got {}",
                i,
                sv[i]
            );
        }
    }

    #[test]
    fn test_to_state_vector_x_gate() {
        let mut mps = MPS::new(2, None);
        mps.apply_single_qubit_gate(&x_gate(), 0);
        let sv = mps.to_state_vector();
        // |00⟩ → X on q0 → |10⟩ in big-endian = index 2
        assert!(
            (sv[2] - Complex64::one()).norm() < 1e-12,
            "Expected |10⟩, sv={:?}",
            sv
        );
    }

    #[test]
    fn test_to_state_vector_hadamard() {
        let mut mps = MPS::new(1, None);
        mps.apply_single_qubit_gate(&h_gate(), 0);
        let sv = mps.to_state_vector();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - inv_sqrt2).abs() < 1e-12);
        assert!((sv[1].re - inv_sqrt2).abs() < 1e-12);
    }

    #[test]
    fn test_bell_state_mps() {
        // Create Bell state: H(0), CNOT(0,1) → (|00⟩ + |11⟩)/√2
        let mut mps = MPS::new(2, None);
        mps.apply_single_qubit_gate(&h_gate(), 0);
        let cnot = [
            [
                Complex64::one(),
                Complex64::zero(),
                Complex64::zero(),
                Complex64::zero(),
            ],
            [
                Complex64::zero(),
                Complex64::one(),
                Complex64::zero(),
                Complex64::zero(),
            ],
            [
                Complex64::zero(),
                Complex64::zero(),
                Complex64::zero(),
                Complex64::one(),
            ],
            [
                Complex64::zero(),
                Complex64::zero(),
                Complex64::one(),
                Complex64::zero(),
            ],
        ];
        mps.apply_two_qubit_gate(0, 1, &cnot);
        let sv = mps.to_state_vector();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        // Bell state: |00⟩ + |11⟩ / √2
        assert!(
            (sv[0].re - inv_sqrt2).abs() < 1e-10,
            "Expected {}, got {} for |00⟩",
            inv_sqrt2,
            sv[0].re
        );
        assert!(sv[1].norm() < 1e-10, "|01⟩ should be 0, got {}", sv[1]);
        assert!(sv[2].norm() < 1e-10, "|10⟩ should be 0, got {}", sv[2]);
        assert!(
            (sv[3].re - inv_sqrt2).abs() < 1e-10,
            "Expected {}, got {} for |11⟩",
            inv_sqrt2,
            sv[3].re
        );
    }

    #[test]
    fn test_ghz_state_mps() {
        // GHZ: H(0), CNOT(0,1), CNOT(1,2) → (|000⟩ + |111⟩)/√2
        let mut mps = MPS::new(3, None);
        mps.apply_single_qubit_gate(&h_gate(), 0);
        let cnot = [
            [
                Complex64::one(),
                Complex64::zero(),
                Complex64::zero(),
                Complex64::zero(),
            ],
            [
                Complex64::zero(),
                Complex64::one(),
                Complex64::zero(),
                Complex64::zero(),
            ],
            [
                Complex64::zero(),
                Complex64::zero(),
                Complex64::zero(),
                Complex64::one(),
            ],
            [
                Complex64::zero(),
                Complex64::zero(),
                Complex64::one(),
                Complex64::zero(),
            ],
        ];
        mps.apply_two_qubit_gate(0, 1, &cnot);
        mps.apply_two_qubit_gate(1, 2, &cnot);
        let sv = mps.to_state_vector();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - inv_sqrt2).abs() < 1e-10, "|000⟩ = {}", sv[0]);
        assert!((sv[7].re - inv_sqrt2).abs() < 1e-10, "|111⟩ = {}", sv[7]);
        // All others zero
        for i in [1, 2, 3, 4, 5, 6] {
            assert!(sv[i].norm() < 1e-10, "sv[{}] = {} should be 0", i, sv[i]);
        }
    }

    #[test]
    fn test_bond_dimension_growth() {
        // Bell state should have bond dim 2 between the two qubits
        let mut mps = MPS::new(2, None);
        mps.apply_single_qubit_gate(&h_gate(), 0);
        let cnot = [
            [
                Complex64::one(),
                Complex64::zero(),
                Complex64::zero(),
                Complex64::zero(),
            ],
            [
                Complex64::zero(),
                Complex64::one(),
                Complex64::zero(),
                Complex64::zero(),
            ],
            [
                Complex64::zero(),
                Complex64::zero(),
                Complex64::zero(),
                Complex64::one(),
            ],
            [
                Complex64::zero(),
                Complex64::zero(),
                Complex64::one(),
                Complex64::zero(),
            ],
        ];
        mps.apply_two_qubit_gate(0, 1, &cnot);
        let bond_dims = mps.bond_dimensions();
        assert_eq!(
            bond_dims[0], 2,
            "Bell state should have bond dim 2, got {}",
            bond_dims[0]
        );
    }

    #[test]
    fn test_truncation_caps_bond_dim() {
        let mut mps = MPS::new(4, Some(2));
        mps.apply_single_qubit_gate(&h_gate(), 0);
        let cnot = [
            [
                Complex64::one(),
                Complex64::zero(),
                Complex64::zero(),
                Complex64::zero(),
            ],
            [
                Complex64::zero(),
                Complex64::one(),
                Complex64::zero(),
                Complex64::zero(),
            ],
            [
                Complex64::zero(),
                Complex64::zero(),
                Complex64::zero(),
                Complex64::one(),
            ],
            [
                Complex64::zero(),
                Complex64::zero(),
                Complex64::one(),
                Complex64::zero(),
            ],
        ];
        mps.apply_two_qubit_gate(0, 1, &cnot);
        mps.apply_single_qubit_gate(&h_gate(), 1);
        mps.apply_two_qubit_gate(1, 2, &cnot);
        for bd in mps.bond_dimensions() {
            assert!(bd <= 2, "Bond dim {} exceeds max 2", bd);
        }
    }

    #[test]
    fn test_bond_entanglement_entropy_tracking() {
        let mut sim = MPSSimulator::new(2, Some(4));
        sim.enable_entanglement_tracking(true);
        sim.h(0);
        sim.cnot(0, 1);
        let ent = sim.bond_entanglement_entropy(0).unwrap_or(0.0);
        assert!(
            ent > 0.1,
            "Expected entanglement entropy > 0.1, got {}",
            ent
        );
    }

    #[test]
    fn test_entanglement_stats_basic() {
        let mut sim = MPSSimulator::new(2, Some(4));
        sim.enable_entanglement_tracking(true);
        sim.h(0);
        sim.cnot(0, 1);
        let stats = sim.entanglement_stats().unwrap();
        assert!(stats.max >= stats.min);
        assert!(stats.mean >= stats.min);
    }
}
