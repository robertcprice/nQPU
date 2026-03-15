//! Entanglement Protocols and Measures
//!
//! Comprehensive implementation of:
//! 1. Entanglement measures (entropy, concurrence, negativity)
//! 2. Entanglement distillation (DEJMPS, BBPSSW)
//! 3. Entanglement swapping
//! 4. Multi-partite entanglement
//!
//! # Example
//!
//! ```
//! use nqpu_metal::entanglement::{EntanglementDistiller, DEJMPSProtocol};
//!
//! // Create distiller for Bell pairs
//! let mut distiller = EntanglementDistiller::new(2);
//!
//! // Distill noisy pairs into high-fidelity pairs
//! let fidelity_before = 0.75;
//! let fidelity_after = distiller.distill(fidelity_before);
//! println!("Fidelity: {} -> {}", fidelity_before, fidelity_after);
//! ```

// ============================================================
// ENTANGLEMENT MEASURES
// ============================================================

/// Entanglement entropy (von Neumann entropy of reduced state).
///
/// For a bipartite pure state |ψ⟩_AB, the entanglement entropy is:
///   S(A) = -Tr(ρ_A log₂ ρ_A)
///
/// where ρ_A = Tr_B(|ψ⟩⟨ψ|) is the reduced density matrix.
///
/// Returns:
/// - 0 for product states
/// - 1 for maximally entangled 2-qubit states (Bell pairs)
/// - log₂(min(d_A, d_B)) for maximally entangled higher dimensions
pub fn entanglement_entropy(reduced_eigenvalues: &[f64]) -> f64 {
    let mut entropy = 0.0;
    for &lambda in reduced_eigenvalues {
        if lambda > 1e-10 {
            entropy -= lambda * lambda.log2();
        }
    }
    entropy
}

/// Concurrence for two-qubit states.
///
/// For a two-qubit density matrix ρ, the concurrence is:
///   C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
///
/// where λᵢ are the square roots of eigenvalues of ρ(σ_y ⊗ σ_y)ρ*(σ_y ⊗ σ_y)
/// in decreasing order.
///
/// Returns:
/// - 0 for separable states
/// - 1 for maximally entangled states
pub fn concurrence(rho: &[[num_complex::Complex64; 4]; 4]) -> f64 {
    use num_complex::Complex64 as C;

    // σ_y ⊗ σ_y matrix (4x4)
    // σ_y = [[0, -i], [i, 0]]
    // σ_y ⊗ σ_y has nonzero entries at positions determined by tensor product
    let mut sigma_yy = [[C::new(0.0, 0.0); 4]; 4];
    // (σ_y ⊗ σ_y)_{00,00} = σ_y[0,0]*σ_y[0,0] = 0
    // (σ_y ⊗ σ_y)_{00,11} = σ_y[0,1]*σ_y[0,1] = (-i)(-i) = -1
    sigma_yy[0][3] = C::new(-1.0, 0.0);
    // (σ_y ⊗ σ_y)_{01,10} = σ_y[0,1]*σ_y[1,0] = (-i)(i) = 1
    sigma_yy[1][2] = C::new(1.0, 0.0);
    // (σ_y ⊗ σ_y)_{10,01} = σ_y[1,0]*σ_y[0,1] = (i)(-i) = 1
    sigma_yy[2][1] = C::new(1.0, 0.0);
    // (σ_y ⊗ σ_y)_{11,00} = σ_y[1,1]*σ_y[1,1] = 0... wait
    // Actually: σ_y[1,0]*σ_y[1,0] = (i)(i) = -1
    sigma_yy[3][0] = C::new(-1.0, 0.0);

    // Compute rho_tilde = (σ_y ⊗ σ_y) · rho* · (σ_y ⊗ σ_y)
    // First: rho* (complex conjugate)
    let mut rho_conj = [[C::new(0.0, 0.0); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            rho_conj[i][j] = rho[i][j].conj();
        }
    }

    // temp = rho_conj * sigma_yy
    let mut temp = [[C::new(0.0, 0.0); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                temp[i][j] += rho_conj[i][k] * sigma_yy[k][j];
            }
        }
    }

    // rho_tilde = sigma_yy * temp
    let mut rho_tilde = [[C::new(0.0, 0.0); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                rho_tilde[i][j] += sigma_yy[i][k] * temp[k][j];
            }
        }
    }

    // R = rho * rho_tilde
    let mut r_matrix = [[C::new(0.0, 0.0); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                r_matrix[i][j] += rho[i][k] * rho_tilde[k][j];
            }
        }
    }

    // Get eigenvalues of R using characteristic polynomial approach for 4x4
    // For numerical stability, use iterative QR-like approach
    let eigenvalues = eigenvalues_4x4_real_parts(&r_matrix);

    // Take square roots of eigenvalues (they should be non-negative real)
    let mut lambdas: Vec<f64> = eigenvalues
        .iter()
        .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
        .collect();
    lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    let c = lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3];
    c.max(0.0)
}

/// Negativity for quantifying entanglement.
///
/// N = (||ρ^(T_A)||₁ - 1) / 2
///
/// where ρ^(T_A) is the partial transpose with respect to subsystem A,
/// and ||·||₁ is the trace norm.
pub fn negativity(rho: &[[num_complex::Complex64; 4]; 4]) -> f64 {
    use num_complex::Complex64 as C;

    // Compute partial transpose with respect to subsystem A (first qubit)
    // For a 2x2 bipartite system with basis |00⟩, |01⟩, |10⟩, |11⟩:
    // ρ^(T_A)_{ij,kl} = ρ_{kj,il}  (swap first qubit indices)
    let mut pt = [[C::new(0.0, 0.0); 4]; 4];
    for i in 0..2usize {
        for j in 0..2usize {
            for k in 0..2usize {
                for l in 0..2usize {
                    // Original index: row = 2*i+j, col = 2*k+l
                    // PT_A index: row = 2*k+j, col = 2*i+l
                    pt[2 * k + j][2 * i + l] = rho[2 * i + j][2 * k + l];
                }
            }
        }
    }

    // Get eigenvalues of the partial transpose (it's Hermitian, so real eigenvalues)
    let eigenvalues = eigenvalues_4x4_hermitian(&pt);

    // Negativity = sum of |λ_i| for negative eigenvalues = (||ρ^(T_A)||₁ - 1) / 2
    let mut neg = 0.0;
    for &ev in &eigenvalues {
        if ev < -1e-12 {
            neg += ev.abs();
        }
    }
    neg
}

/// Compute eigenvalues of a 4x4 complex matrix, returning the real parts.
/// Uses power iteration with deflation for the dominant eigenvalues.
fn eigenvalues_4x4_real_parts(m: &[[num_complex::Complex64; 4]; 4]) -> [f64; 4] {
    use num_complex::Complex64 as C;

    // Use QR iteration on the 4x4 matrix
    let mut a = *m;
    for _ in 0..200 {
        // QR decomposition via modified Gram-Schmidt
        let (q, r) = qr_4x4(&a);
        // A' = R * Q
        a = mat_mul_4x4(&r, &q);
    }
    // Eigenvalues are on the diagonal
    [a[0][0].re, a[1][1].re, a[2][2].re, a[3][3].re]
}

/// Compute eigenvalues of a 4x4 Hermitian matrix (returns real eigenvalues).
/// Uses Jacobi eigenvalue algorithm for Hermitian matrices.
fn eigenvalues_4x4_hermitian(m: &[[num_complex::Complex64; 4]; 4]) -> [f64; 4] {
    use num_complex::Complex64 as C;

    // Convert Hermitian matrix to real symmetric form via real/imag splitting
    // For a Hermitian matrix, eigenvalues are real. Use QR iteration.
    let mut a = *m;
    for _ in 0..200 {
        let (q, r) = qr_4x4(&a);
        a = mat_mul_4x4(&r, &q);
    }
    [a[0][0].re, a[1][1].re, a[2][2].re, a[3][3].re]
}

/// QR decomposition of a 4x4 complex matrix using modified Gram-Schmidt.
fn qr_4x4(
    a: &[[num_complex::Complex64; 4]; 4],
) -> (
    [[num_complex::Complex64; 4]; 4],
    [[num_complex::Complex64; 4]; 4],
) {
    use num_complex::Complex64 as C;

    let zero = C::new(0.0, 0.0);
    let mut q = [[zero; 4]; 4];
    let mut r = [[zero; 4]; 4];

    // Extract columns of A
    let mut cols: [[C; 4]; 4] = [[zero; 4]; 4];
    for j in 0..4 {
        for i in 0..4 {
            cols[j][i] = a[i][j];
        }
    }

    let mut u: [[C; 4]; 4] = [[zero; 4]; 4];

    for j in 0..4 {
        // Start with column j
        u[j] = cols[j];

        // Subtract projections onto previous orthogonal vectors
        for k in 0..j {
            let mut dot = zero;
            for i in 0..4 {
                dot += q[k][i].conj() * cols[j][i];
            }
            r[k][j] = dot;
            for i in 0..4 {
                u[j][i] -= dot * q[k][i];
            }
        }

        // Normalize
        let mut norm_sq = 0.0f64;
        for i in 0..4 {
            norm_sq += u[j][i].norm_sqr();
        }
        let norm = norm_sq.sqrt();
        r[j][j] = C::new(norm, 0.0);

        if norm > 1e-15 {
            for i in 0..4 {
                q[j][i] = u[j][i] / C::new(norm, 0.0);
            }
        }
    }

    // Convert q from column-major storage to row-major matrix
    let mut q_mat = [[zero; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            q_mat[i][j] = q[j][i];
        }
    }

    (q_mat, r)
}

/// Multiply two 4x4 complex matrices.
fn mat_mul_4x4(
    a: &[[num_complex::Complex64; 4]; 4],
    b: &[[num_complex::Complex64; 4]; 4],
) -> [[num_complex::Complex64; 4]; 4] {
    use num_complex::Complex64 as C;
    let zero = C::new(0.0, 0.0);
    let mut result = [[zero; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// Entanglement of formation.
///
/// E_F = h((1 + √(1 - C²)) / 2)
///
/// where h(x) = -x log₂(x) - (1-x) log₂(1-x) is the binary entropy.
pub fn entanglement_of_formation(concurrence: f64) -> f64 {
    let c = concurrence.clamp(0.0, 1.0);
    if c < 1e-10 {
        return 0.0;
    }

    let x = (1.0 + (1.0 - c * c).sqrt()) / 2.0;
    if x < 1e-10 || x > 1.0 - 1e-10 {
        return 0.0;
    }

    -x * x.log2() - (1.0 - x) * (1.0 - x).log2()
}

// ============================================================
// ENTANGLEMENT DISTILLATION
// ============================================================

/// Protocol for entanglement distillation.
#[derive(Debug, Clone, Copy)]
pub enum DistillationProtocol {
    /// Deutsch-Ekert-Jozsa-Macchiavello-Popescu-Sanpera
    /// Works on Werner states, uses XOR operations
    DEJMPS,
    /// Bennett-Brassard-Popescu-Schumacher-Smolin-Wootters
    /// Uses bilateral rotations and measurements
    BBPSSW,
    /// Recurrence protocol - iterative improvement
    Recurrence { iterations: usize },
    /// Hashing protocol - asymptotically optimal
    Hashing { rate: f64 },
}

/// Entanglement distiller for improving Bell pair fidelity.
pub struct EntanglementDistiller {
    /// Number of qubits per party
    num_qubits: usize,
    /// Distillation protocol
    protocol: DistillationProtocol,
    /// Current fidelity
    fidelity: f64,
}

impl EntanglementDistiller {
    /// Create a new entanglement distiller.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            protocol: DistillationProtocol::DEJMPS,
            fidelity: 1.0,
        }
    }

    /// Set the distillation protocol.
    pub fn with_protocol(mut self, protocol: DistillationProtocol) -> Self {
        self.protocol = protocol;
        self
    }

    /// Distill noisy Bell pairs to improve fidelity.
    ///
    /// Uses multiple noisy pairs to produce fewer higher-fidelity pairs.
    /// The yield depends on initial fidelity and protocol.
    ///
    /// Returns the new fidelity after one round of distillation.
    pub fn distill(&mut self, input_fidelity: f64) -> f64 {
        self.fidelity = input_fidelity;
        match self.protocol {
            DistillationProtocol::DEJMPS => self.dejmps_round(),
            DistillationProtocol::BBPSSW => self.bbpssw_round(),
            DistillationProtocol::Recurrence { iterations } => {
                let mut f = input_fidelity;
                for _ in 0..iterations {
                    f = self.bbpssw_round_with(f);
                }
                f
            }
            DistillationProtocol::Hashing { rate } => self.hashing_round(rate),
        }
    }

    /// DEJMPS protocol round.
    ///
    /// For Werner state with fidelity F, one round produces:
    ///   F' = (F² + (1-F)²/9) / (F² + 2F(1-F)/3 + 5(1-F)²/9)
    ///
    /// Yield: ~1/2 pairs consumed
    fn dejmps_round(&self) -> f64 {
        let f = self.fidelity;
        let f_sq = f * f;
        let one_minus_f = 1.0 - f;
        let one_minus_f_sq = one_minus_f * one_minus_f;

        let numerator = f_sq + one_minus_f_sq / 9.0;
        let denominator = f_sq + 2.0 * f * one_minus_f / 3.0 + 5.0 * one_minus_f_sq / 9.0;

        numerator / denominator
    }

    /// BBPSSW protocol round.
    ///
    /// For Werner state with fidelity F, one round produces:
    ///   F' = (F² + (1-F)²/9) / P(success)
    ///   P(success) = F² + 2F(1-F)/3 + 5(1-F)²/9
    fn bbpssw_round(&self) -> f64 {
        self.bbpssw_round_with(self.fidelity)
    }

    fn bbpssw_round_with(&self, f: f64) -> f64 {
        let f_sq = f * f;
        let one_minus_f = 1.0 - f;
        let one_minus_f_sq = one_minus_f * one_minus_f;

        let numerator = f_sq + one_minus_f_sq / 9.0;
        let denominator = f_sq + 2.0 * f * one_minus_f / 3.0 + 5.0 * one_minus_f_sq / 9.0;

        numerator / denominator
    }

    /// Hashing protocol (asymptotically optimal).
    ///
    /// Rate: 1 - H(F) where H is entropy
    fn hashing_round(&self, rate: f64) -> f64 {
        // Hashing achieves arbitrary fidelity with rate approaching
        // 1 - H(F) for large number of pairs
        let f = self.fidelity;

        // Entropy of Werner state
        let h = if f > 0.5 {
            -f * f.log2() - (1.0 - f) * (1.0 - f).log2()
        } else {
            1.0
        };

        // Achievable rate
        let achievable_rate = (1.0 - h).max(0.0);

        // Higher fidelity with hashing
        f + rate * achievable_rate * (1.0 - f)
    }

    /// Get the yield (fraction of pairs retained) after distillation.
    pub fn yield_fraction(&self) -> f64 {
        let f = self.fidelity;
        let f_sq = f * f;
        let one_minus_f = 1.0 - f;
        let one_minus_f_sq = one_minus_f * one_minus_f;

        match self.protocol {
            DistillationProtocol::DEJMPS | DistillationProtocol::BBPSSW => {
                // Success probability = F² + 2F(1-F)/3 + 5(1-F)²/9
                f_sq + 2.0 * f * one_minus_f / 3.0 + 5.0 * one_minus_f_sq / 9.0
            }
            DistillationProtocol::Recurrence { iterations } => {
                // Yield decreases exponentially with iterations
                let single_yield = f_sq + 2.0 * f * one_minus_f / 3.0 + 5.0 * one_minus_f_sq / 9.0;
                single_yield.powi(iterations as i32)
            }
            DistillationProtocol::Hashing { rate } => rate,
        }
    }
}

// ============================================================
// ENTANGLEMENT SWAPPING
// ============================================================

/// Entanglement swapping protocol.
///
/// Creates entanglement between two particles that have never interacted,
/// by performing a Bell measurement on intermediate particles.
pub struct EntanglementSwapping {
    /// Fidelity of first Bell pair (A-B)
    fidelity_ab: f64,
    /// Fidelity of second Bell pair (C-D)
    fidelity_cd: f64,
}

impl EntanglementSwapping {
    /// Create a new entanglement swapping protocol.
    pub fn new(fidelity_ab: f64, fidelity_cd: f64) -> Self {
        Self {
            fidelity_ab,
            fidelity_cd,
        }
    }

    /// Perform entanglement swapping.
    ///
    /// Bell measurement on B and C creates entanglement between A and D.
    ///
    /// Fidelity of resulting pair: F_AD = F_AB * F_CD + (1-F_AB)(1-F_CD)/9
    pub fn swap(&self) -> EntanglementSwapResult {
        let f_ab = self.fidelity_ab;
        let f_cd = self.fidelity_cd;

        // Resulting fidelity
        let f_ad = f_ab * f_cd + (1.0 - f_ab) * (1.0 - f_cd) / 9.0;

        // Success probability (always succeeds for ideal case)
        let success_prob = 1.0;

        // Which Bell state results (random for Werner states)
        let bell_state = if rand_prob() < f_ad {
            BellState::PhiPlus
        } else {
            BellState::random()
        };

        EntanglementSwapResult {
            fidelity: f_ad,
            success_probability: success_prob,
            bell_state,
        }
    }

    /// Perform nested swapping (quantum repeater).
    ///
    /// Connects two distant nodes through multiple swapping steps.
    pub fn nested_swap(levels: usize, initial_fidelity: f64) -> f64 {
        let mut f = initial_fidelity;
        for _ in 0..levels {
            // Each level halves the distance but reduces fidelity
            f = f * f + (1.0 - f) * (1.0 - f) / 9.0;
        }
        f
    }
}

/// Result of entanglement swapping.
#[derive(Debug, Clone)]
pub struct EntanglementSwapResult {
    /// Fidelity of resulting Bell pair
    pub fidelity: f64,
    /// Probability of successful swap
    pub success_probability: f64,
    /// Resulting Bell state
    pub bell_state: BellState,
}

/// Bell states (EPR pairs).
#[derive(Debug, Clone, Copy)]
pub enum BellState {
    /// |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
    PhiPlus,
    /// |Φ⁻⟩ = (|00⟩ - |11⟩) / √2
    PhiMinus,
    /// |Ψ⁺⟩ = (|01⟩ + |10⟩) / √2
    PsiPlus,
    /// |Ψ⁻⟩ = (|01⟩ - |10⟩) / √2
    PsiMinus,
}

impl BellState {
    /// Random Bell state (for simulation).
    fn random() -> Self {
        match (rand_prob() * 4.0) as usize {
            0 => BellState::PhiPlus,
            1 => BellState::PhiMinus,
            2 => BellState::PsiPlus,
            _ => BellState::PsiMinus,
        }
    }
}

// Simple random number for simulation
fn rand_prob() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as f64) / 1_000_000_000.0
}

// ============================================================
// MULTI-PARTITE ENTANGLEMENT
// ============================================================

/// Three-tangle (residual tangle) for three-qubit states.
///
/// τ₃ = τ(A:BC) - τ(A:B) - τ(A:C)
///
/// Non-zero for GHZ-type entanglement, zero for W-type.
pub fn three_tangle(concurrence_ab: f64, concurrence_ac: f64, concurrence_abc: f64) -> f64 {
    let tau_ab = concurrence_ab * concurrence_ab;
    let tau_ac = concurrence_ac * concurrence_ac;
    let tau_abc = concurrence_abc * concurrence_abc;

    (tau_abc - tau_ab - tau_ac).max(0.0)
}

/// Global entanglement measure (Meyer-Wallach).
///
/// Q = 2/n Σᵢ (1 - Tr(ρᵢ²))
///
/// - 0 for product states
/// - 1 for GHZ states
/// - 2/3 for W states
pub fn global_entanglement(purities: &[f64]) -> f64 {
    let n = purities.len();
    if n == 0 {
        return 0.0;
    }

    let sum: f64 = purities.iter().map(|&p| 1.0 - p).sum();
    2.0 * sum / n as f64
}

/// Geometric measure of entanglement.
///
/// E_G = 1 - max|⟨ψ|φ⟩|²
///
/// where the max is over all product states |φ⟩.
pub fn geometric_entanglement(max_overlap: f64) -> f64 {
    1.0 - max_overlap * max_overlap
}

// ============================================================
// ENTANGLEMENT WITNESSES
// ============================================================

/// Entanglement witness operator.
///
/// A Hermitian operator W such that:
/// - Tr(Wρ) ≥ 0 for all separable states ρ
/// - Tr(Wσ) < 0 for at least one entangled state σ
pub struct EntanglementWitness {
    /// Matrix representation
    matrix: Vec<Vec<num_complex::Complex64>>,
    /// Description
    description: String,
}

impl EntanglementWitness {
    /// Create the canonical witness for Bell states.
    ///
    /// W = I/2 - |Φ⁺⟩⟨Φ⁺| where |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
    ///
    /// This gives Tr(W|Φ⁺⟩⟨Φ⁺|) = -0.5 < 0 (detects entanglement)
    /// and Tr(W|00⟩⟨00|) = 0 ≥ 0 (separable states pass)
    pub fn bell_witness() -> Self {
        let i = num_complex::Complex64::new(1.0, 0.0);
        let zero = num_complex::Complex64::new(0.0, 0.0);

        // W = I/2 - |Φ⁺⟩⟨Φ⁺|
        // |Φ⁺⟩⟨Φ⁺| = 0.5 * |00⟩⟨00| + 0.5 * |00⟩⟨11| + 0.5 * |11⟩⟨00| + 0.5 * |11⟩⟨11|
        // W = diag(0.5, 0.5, 0.5, 0.5) - [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
        // W = [[0, 0, 0, -0.5], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [-0.5, 0, 0, 0]]
        let matrix = vec![
            vec![zero, zero, zero, -i * 0.5],
            vec![zero, i * 0.5, zero, zero],
            vec![zero, zero, i * 0.5, zero],
            vec![-i * 0.5, zero, zero, zero],
        ];

        Self {
            matrix,
            description: "Bell state witness".to_string(),
        }
    }

    /// Detect entanglement in a state.
    ///
    /// Returns true if entanglement is detected (negative expectation).
    pub fn detect(&self, rho: &[[num_complex::Complex64; 4]; 4]) -> bool {
        let expectation = self.expectation(rho);
        expectation < 0.0
    }

    /// Compute expectation value ⟨W⟩ = Tr(Wρ).
    pub fn expectation(&self, rho: &[[num_complex::Complex64; 4]; 4]) -> f64 {
        let mut sum = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                sum += (self.matrix[i][j] * rho[j][i]).re;
            }
        }
        sum
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entanglement_entropy_bell() {
        // Bell state has entanglement entropy = 1
        let eigenvalues = vec![0.5, 0.5];
        let entropy = entanglement_entropy(&eigenvalues);
        assert!((entropy - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entanglement_entropy_product() {
        // Product state has entanglement entropy = 0
        let eigenvalues = vec![1.0];
        let entropy = entanglement_entropy(&eigenvalues);
        assert!((entropy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dejmps_distillation() {
        let mut distiller =
            EntanglementDistiller::new(2).with_protocol(DistillationProtocol::DEJMPS);

        // Starting fidelity 0.75 should improve
        let f_after = distiller.distill(0.75);
        assert!(
            f_after > 0.75,
            "Fidelity should improve: {} > 0.75",
            f_after
        );
        assert!(f_after <= 1.0, "Fidelity should be <= 1.0: {}", f_after);
    }

    #[test]
    fn test_bbpssw_distillation() {
        let mut distiller =
            EntanglementDistiller::new(2).with_protocol(DistillationProtocol::BBPSSW);

        let f_after = distiller.distill(0.75);
        assert!(f_after > 0.75);
    }

    #[test]
    fn test_entanglement_swapping() {
        let swapping = EntanglementSwapping::new(0.9, 0.9);
        let result = swapping.swap();

        // Fidelity should be product plus correction
        let expected = 0.9 * 0.9 + 0.1 * 0.1 / 9.0;
        assert!((result.fidelity - expected).abs() < 1e-10);
    }

    #[test]
    fn test_nested_swapping() {
        // More levels = lower fidelity
        let f1 = EntanglementSwapping::nested_swap(1, 0.9);
        let f2 = EntanglementSwapping::nested_swap(2, 0.9);

        assert!(
            f2 < f1,
            "More levels should reduce fidelity: {} < {}",
            f2,
            f1
        );
    }

    #[test]
    fn test_three_tangle_ghz() {
        // GHZ state has three-tangle = 1
        let tangle = three_tangle(0.0, 0.0, 1.0);
        assert!((tangle - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_three_tangle_w() {
        // W state has three-tangle = 0
        let tangle = three_tangle(0.5, 0.5, 0.5);
        // For W: τ(ABC) = τ(A:B) + τ(A:C), so residual = 0
        assert!(tangle.abs() < 0.5);
    }

    #[test]
    fn test_global_entanglement() {
        // For Bell state (n=2), each purity = 0.5
        let purities = vec![0.5, 0.5];
        let q = global_entanglement(&purities);
        assert!((q - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entanglement_of_formation() {
        // Concurrence = 1 should give E_F = 1
        let eof = entanglement_of_formation(1.0);
        assert!((eof - 1.0).abs() < 1e-10);

        // Concurrence = 0 should give E_F = 0
        let eof = entanglement_of_formation(0.0);
        assert!((eof - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bell_witness() {
        let witness = EntanglementWitness::bell_witness();

        // Bell state density matrix
        let i = num_complex::Complex64::new(1.0, 0.0);
        let zero = num_complex::Complex64::new(0.0, 0.0);
        let half = num_complex::Complex64::new(0.5, 0.0);

        let rho_bell = [
            [half, zero, zero, half],
            [zero, zero, zero, zero],
            [zero, zero, zero, zero],
            [half, zero, zero, half],
        ];

        // Should detect entanglement (negative expectation)
        assert!(witness.detect(&rho_bell));
    }
}
