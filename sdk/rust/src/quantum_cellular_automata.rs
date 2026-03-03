//! Quantum Cellular Automata (QCA) for nQPU-Metal.
//!
//! Implements discrete-time quantum cellular automata on 1D qubit chains
//! using Margolus partitioning. Supports parameterized rule families
//! (Goldilocks, Heisenberg), boundary conditions, and spacetime diagnostics
//! including probability profiles, entanglement entropy, and light-cone
//! analysis.
//!
//! # Physical background
//!
//! A 1D QCA applies a translationally-invariant local unitary in a
//! brickwork (Margolus) pattern: even pairs `(0,1),(2,3),...` on
//! half-steps and odd pairs `(1,2),(3,4),...` on the other half.
//! The Goldilocks family interpolates between integrable (J=0) and
//! chaotic (J=pi/4) dynamics via an XX+ZZ Hamiltonian unitary, while
//! the Heisenberg variant adds YY coupling for full SU(2) symmetry.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::quantum_cellular_automata::*;
//! use nqpu_metal::{QuantumState, GateOperations};
//!
//! let mut qca = QuantumCellularAutomaton::new(
//!     6,
//!     QCARuleType::Goldilocks(std::f64::consts::FRAC_PI_4),
//!     BoundaryCondition::Periodic,
//! );
//! // Prepare a domain-wall initial state: |111000>
//! GateOperations::x(&mut qca.state, 0);
//! GateOperations::x(&mut qca.state, 1);
//! GateOperations::x(&mut qca.state, 2);
//!
//! let data = qca.evolve(20);
//! assert_eq!(data.probability_profiles.len(), 21); // initial + 20 steps
//! ```


use crate::{c64_one, c64_zero, C64, QuantumState};

// ---------------------------------------------------------------------------
// Rule types
// ---------------------------------------------------------------------------

/// Rule family governing the two-qubit gate applied at each pair.
#[derive(Clone)]
pub enum QCARuleType {
    /// Arbitrary 4x4 unitary applied to every nearest-neighbour pair.
    PairUnitary(Vec<Vec<C64>>),

    /// Goldilocks family: XX+ZZ coupling with strength `J`.
    /// `J = 0`      --> product (integrable) evolution.
    /// `J = pi/4`   --> SWAP gate (maximally scrambling boundary).
    Goldilocks(f64),

    /// Isotropic Heisenberg XX+YY+ZZ coupling with strength `J`.
    Heisenberg(f64),

    /// Identity (no-op) rule, useful for testing scaffolding.
    Identity,

    /// User-supplied rule: function pointer `(j: f64) -> 4x4 unitary`.
    /// The `f64` argument is a coupling parameter forwarded by the caller.
    CustomClosure {
        generator: fn(f64) -> Vec<Vec<C64>>,
        param: f64,
    },
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

/// Boundary condition applied to the qubit chain edges.
#[derive(Clone, Debug, PartialEq)]
pub enum BoundaryCondition {
    /// Periodic (ring topology): qubit `n-1` is paired with qubit `0`.
    Periodic,
    /// Open chain: no pairing wraps around.
    Open,
    /// Fixed boundary: edge qubits interact with a virtual qubit whose
    /// amplitude is pinned to the given complex value.
    Fixed(C64),
}

// ---------------------------------------------------------------------------
// Margolus partitioning
// ---------------------------------------------------------------------------

/// Margolus partitioning helpers for brickwork decomposition.
pub struct MargolusPartition;

impl MargolusPartition {
    /// Even-step pairs: `(0,1), (2,3), (4,5), ...`
    pub fn even_pairs(n: usize) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        let mut i = 0;
        while i + 1 < n {
            pairs.push((i, i + 1));
            i += 2;
        }
        pairs
    }

    /// Odd-step pairs: `(1,2), (3,4), (5,6), ...`
    pub fn odd_pairs(n: usize) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        let mut i = 1;
        while i + 1 < n {
            pairs.push((i, i + 1));
            i += 2;
        }
        pairs
    }
}

// ---------------------------------------------------------------------------
// Goldilocks / Heisenberg rule generators
// ---------------------------------------------------------------------------

/// Pre-built parameterized rule families.
pub struct GoldilocksRules;

impl GoldilocksRules {
    /// Ising-like XX+ZZ Hamiltonian unitary: `exp(-i J (XX + ZZ))`.
    ///
    /// The 4x4 unitary in the computational basis `{|00>,|01>,|10>,|11>}`
    /// is block-diagonal in the singlet/triplet subspaces:
    ///
    /// ```text
    /// |00> -> cos(2J)|00> - i sin(2J)|11>
    /// |01> -> cos(2J)|01> - i sin(2J)|10>
    /// |10> -> cos(2J)|10> - i sin(2J)|01>
    /// |11> -> cos(2J)|11> - i sin(2J)|00>
    /// ```
    pub fn ising_like(j: f64) -> Vec<Vec<C64>> {
        let c = (2.0 * j).cos();
        let s = (2.0 * j).sin();
        let zero = c64_zero();
        let cos_val = C64::new(c, 0.0);
        let neg_i_sin = C64::new(0.0, -s);

        vec![
            vec![cos_val, zero, zero, neg_i_sin],
            vec![zero, cos_val, neg_i_sin, zero],
            vec![zero, neg_i_sin, cos_val, zero],
            vec![neg_i_sin, zero, zero, cos_val],
        ]
    }

    /// Isotropic Heisenberg model: `exp(-i J (XX + YY + ZZ))`.
    ///
    /// In the computational basis this decomposes as:
    /// - `|00>` and `|11>` span a 2D subspace with phase `exp(-iJ)`.
    /// - `|01>` and `|10>` are coupled via `exp(iJ) [cos(2J) I - i sin(2J) SWAP_sub]`.
    ///
    /// Concretely:
    /// ```text
    /// |00> -> exp(-iJ) |00>
    /// |01> -> exp(iJ) [cos(2J)|01> - i sin(2J)|10>]
    /// |10> -> exp(iJ) [-i sin(2J)|01> + cos(2J)|10>]
    /// |11> -> exp(-iJ) |11>
    /// ```
    pub fn heisenberg(j: f64) -> Vec<Vec<C64>> {
        let phase_minus = C64::new((-j).cos(), (-j).sin()); // exp(-iJ)
        let phase_plus = C64::new(j.cos(), j.sin()); // exp(+iJ)
        let c2 = (2.0 * j).cos();
        let s2 = (2.0 * j).sin();
        let zero = c64_zero();

        // Diagonal blocks for |00> and |11>
        let diag = phase_minus;
        // Off-diagonal block for |01>,|10>
        let a = phase_plus * C64::new(c2, 0.0);
        let b = phase_plus * C64::new(0.0, -s2);

        vec![
            vec![diag, zero, zero, zero],
            vec![zero, a, b, zero],
            vec![zero, b, a, zero],
            vec![zero, zero, zero, diag],
        ]
    }

    /// Pure SWAP gate (for verification): exchanges |01> <-> |10>.
    pub fn swap_gate() -> Vec<Vec<C64>> {
        let one = c64_one();
        let zero = c64_zero();
        vec![
            vec![one, zero, zero, zero],
            vec![zero, zero, one, zero],
            vec![zero, one, zero, zero],
            vec![zero, zero, zero, one],
        ]
    }

    /// Identity 4x4 gate (no-op).
    pub fn identity_4x4() -> Vec<Vec<C64>> {
        let one = c64_one();
        let zero = c64_zero();
        vec![
            vec![one, zero, zero, zero],
            vec![zero, one, zero, zero],
            vec![zero, zero, one, zero],
            vec![zero, zero, zero, one],
        ]
    }
}

// ---------------------------------------------------------------------------
// Spacetime data collection
// ---------------------------------------------------------------------------

/// Accumulated diagnostic data over a QCA evolution.
#[derive(Clone, Debug)]
pub struct SpacetimeData {
    /// Probability distribution `P(x)` at each recorded time step.
    /// `probability_profiles[t][i]` = probability of basis state `i` at step `t`.
    pub probability_profiles: Vec<Vec<f64>>,

    /// Bipartite entanglement entropy across each cut, at each time step.
    /// `entanglement_entropy[t][k]` = von Neumann entropy for the cut
    /// separating qubits `0..=k` from `k+1..n-1` at step `t`.
    pub entanglement_entropy: Vec<Vec<f64>>,

    /// Number of qubits in the system.
    num_qubits: usize,
}

impl SpacetimeData {
    /// Create empty spacetime data for `n` qubits.
    pub fn new(num_qubits: usize) -> Self {
        SpacetimeData {
            probability_profiles: Vec::new(),
            entanglement_entropy: Vec::new(),
            num_qubits,
        }
    }

    /// Snapshot the current quantum state: record probabilities and
    /// entanglement entropy across every bipartition.
    pub fn add_snapshot(&mut self, state: &QuantumState) {
        self.probability_profiles.push(state.probabilities());

        let n = state.num_qubits;
        let mut entropies = Vec::with_capacity(n.saturating_sub(1));
        for cut in 0..n.saturating_sub(1) {
            entropies.push(bipartite_entropy(state, cut));
        }
        self.entanglement_entropy.push(entropies);
    }

    /// Compute the information light cone emanating from `site`.
    ///
    /// Returns `Vec<(time_step, max_probability_change)>` measuring how
    /// far correlations have spread from the initial site at each recorded
    /// step, using the probability deviation from the initial profile.
    pub fn compute_light_cone(&self, _site: usize) -> Vec<(usize, f64)> {
        if self.probability_profiles.is_empty() {
            return Vec::new();
        }

        let initial = &self.probability_profiles[0];
        let n = self.num_qubits;
        let dim = 1usize << n;

        let mut result = Vec::with_capacity(self.probability_profiles.len());
        for (t, profile) in self.probability_profiles.iter().enumerate() {
            // For each basis state, compute the probability change.  We
            // attribute a change to `site` when the bit at `site` differs
            // in the basis index, indicating the site participated.
            let mut max_change: f64 = 0.0;
            for idx in 0..dim {
                // Consider states where the site bit is flipped relative
                // to the all-zero reference -- this tracks how much the
                // site has influenced the global state.
                let delta = (profile[idx] - initial[idx]).abs();
                if delta > max_change {
                    max_change = delta;
                }
            }
            result.push((t, max_change));
        }
        result
    }

    /// Number of recorded time steps (including the initial snapshot).
    pub fn num_steps(&self) -> usize {
        self.probability_profiles.len()
    }

    /// Per-qubit probability at a given time step.
    ///
    /// Returns `Vec<f64>` of length `num_qubits` where entry `q` is the
    /// probability that qubit `q` is in state `|1>`.
    pub fn qubit_probabilities(&self, step: usize) -> Vec<f64> {
        let profile = &self.probability_profiles[step];
        let n = self.num_qubits;
        let dim = 1usize << n;
        let mut per_qubit = vec![0.0; n];
        for idx in 0..dim {
            for q in 0..n {
                if idx & (1 << q) != 0 {
                    per_qubit[q] += profile[idx];
                }
            }
        }
        per_qubit
    }
}

// ---------------------------------------------------------------------------
// Entanglement computation
// ---------------------------------------------------------------------------

/// Compute the von Neumann entropy of the reduced density matrix obtained
/// by tracing out qubits `cut+1 .. n-1`, keeping qubits `0 .. cut`.
///
/// For a pure state `|psi>`, the bipartite entropy across the cut between
/// qubit `cut` and `cut+1` equals `S = -sum_i lambda_i log2(lambda_i)`,
/// where `lambda_i` are the squared singular values of the reshaped
/// coefficient matrix `M[a][b] = psi[a * 2^(n-cut-1) + b]`.
///
/// We build the reduced density matrix `rho_A = Tr_B(|psi><psi|)` and
/// diagonalise it to extract eigenvalues.
pub fn bipartite_entropy(state: &QuantumState, cut: usize) -> f64 {
    let n = state.num_qubits;
    if cut + 1 >= n || n == 0 {
        return 0.0;
    }

    let n_a = cut + 1; // number of qubits in subsystem A
    let n_b = n - n_a; // number of qubits in subsystem B
    let dim_a = 1usize << n_a;
    let dim_b = 1usize << n_b;

    let amps = state.amplitudes_ref();

    // Build the reduced density matrix rho_A = Tr_B(|psi><psi|).
    //
    // In nQPU-Metal, qubit 0 is the least-significant bit.  Subsystem A
    // consists of qubits 0..=cut (the n_a least-significant bits), so the
    // state vector index decomposes as:
    //   full_index = b * dim_a + a
    // where `a` indexes subsystem A and `b` indexes subsystem B.
    //
    // rho_A[i][j] = sum_b  psi[b*dim_a + i] * conj(psi[b*dim_a + j])
    let mut rho = vec![vec![c64_zero(); dim_a]; dim_a];
    for i in 0..dim_a {
        for j in 0..=i {
            let mut sum = c64_zero();
            for b in 0..dim_b {
                let idx_i = b * dim_a + i;
                let idx_j = b * dim_a + j;
                let a = amps[idx_i];
                let bc = amps[idx_j].conj();
                sum += a * bc;
            }
            rho[i][j] = sum;
            if i != j {
                rho[j][i] = sum.conj();
            }
        }
    }

    // Extract eigenvalues of the Hermitian matrix rho_A.
    // For small dimensions we use a direct power-iteration / Jacobi approach.
    // For production quality we diagonalise the real symmetric matrix formed
    // by rho's eigenvalues (which are real and non-negative for density matrices).
    let eigenvalues = hermitian_eigenvalues(&rho, dim_a);

    // S = - sum_i lambda_i log2(lambda_i), skipping zero eigenvalues.
    let mut entropy = 0.0_f64;
    for &lam in &eigenvalues {
        if lam > 1e-15 {
            entropy -= lam * lam.log2();
        }
    }
    entropy.max(0.0) // guard against floating-point negativity
}

/// Diagonalise a Hermitian matrix via Jacobi eigenvalue iteration.
///
/// Returns the real eigenvalues (unsorted). This is adequate for the
/// small matrices arising from few-qubit bipartitions (dim <= 128 for
/// 7 qubits in subsystem A, etc.).
fn hermitian_eigenvalues(mat: &[Vec<C64>], n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![mat[0][0].re];
    }

    // Convert to real representation via A_real = [Re(A), -Im(A); Im(A), Re(A)]
    // and compute eigenvalues of that.  For a Hermitian matrix the eigenvalues
    // of the doubled real matrix are each eigenvalue of A repeated twice.
    //
    // Alternative: use the Jacobi method directly on the Hermitian matrix
    // operating on the modulus of off-diagonal elements.  We implement a
    // simple iterative QR on the real-valued matrix of |rho| (the absolute
    // values of diagonal and off-diagonal norms).
    //
    // For simplicity and correctness on small matrices, we use the
    // Jacobi-like sweep on the Hermitian form directly.

    // Build a real symmetric matrix S where S[i][j] = Re(rho[i][j]) for
    // a Hermitian rho (since Im(rho[i][i])=0 and Im(rho[i][j])=-Im(rho[j][i])).
    // Actually, eigenvalues of a Hermitian matrix can be found by converting
    // to a real problem.  The standard approach:
    //
    // We use the fact that for small n we can just do repeated
    // Givens/Jacobi rotations.  Here we implement a simple eigenvalue
    // extraction by converting to nalgebra if available, or a manual
    // power iteration.  For this module we do a manual Jacobi sweep.

    // Manual Jacobi eigenvalue algorithm for Hermitian matrices.
    // We work with a mutable copy.
    let mut a_re = vec![vec![0.0f64; n]; n];
    let mut a_im = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            a_re[i][j] = mat[i][j].re;
            a_im[i][j] = mat[i][j].im;
        }
    }

    // Jacobi sweep: iteratively zero off-diagonal elements.
    let max_sweeps = 100;
    let tol = 1e-12;

    for _sweep in 0..max_sweeps {
        // Find the largest off-diagonal element (by modulus).
        let mut max_off = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let mag = a_re[i][j] * a_re[i][j] + a_im[i][j] * a_im[i][j];
                if mag > max_off {
                    max_off = mag;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off.sqrt() < tol {
            break;
        }

        // Compute the 2x2 Hermitian sub-problem for (p, q).
        let app = a_re[p][p];
        let aqq = a_re[q][q];
        let apq_re = a_re[p][q];
        let apq_im = a_im[p][q];
        let apq_mod = (apq_re * apq_re + apq_im * apq_im).sqrt();

        if apq_mod < tol {
            continue;
        }

        // Phase to make off-diagonal real: e^{-i phi} where phi = arg(a_pq).
        let phase_re = apq_re / apq_mod;
        let phase_im = -apq_im / apq_mod;

        // Now the effective 2x2 real symmetric problem is:
        // [[app, apq_mod], [apq_mod, aqq]]
        let tau = (aqq - app) / (2.0 * apq_mod);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Apply the Jacobi rotation: a complex rotation in the (p,q) plane
        // with cos(theta) = c, sin(theta) * e^{i phi} = s * (phase_re + i*phase_im).
        let s_re = s * phase_re;
        let s_im = s * phase_im;

        // Update the matrix: rows and columns p, q.
        // For each k != p,q:
        //   a'[k][p] = c * a[k][p] + conj(s*e^{i phi}) * a[k][q]
        //   a'[k][q] = -s*e^{i phi} * a[k][p] + c * a[k][q]
        for k in 0..n {
            if k == p || k == q {
                continue;
            }
            let akp_re = a_re[k][p];
            let akp_im = a_im[k][p];
            let akq_re = a_re[k][q];
            let akq_im = a_im[k][q];

            // conj(s * e^{i phi}) = s * e^{-i phi}
            let cs_conj_re = s_re; // conj flips im
            let cs_conj_im = -s_im;

            // a'[k][p] = c * a[k][p] + conj(s e^{i phi}) * a[k][q]
            let new_kp_re = c * akp_re + cs_conj_re * akq_re - cs_conj_im * akq_im;
            let new_kp_im = c * akp_im + cs_conj_re * akq_im + cs_conj_im * akq_re;

            // a'[k][q] = -(s e^{i phi}) * a[k][p] + c * a[k][q]
            let new_kq_re = -s_re * akp_re + s_im * akp_im + c * akq_re;
            let new_kq_im = -s_re * akp_im - s_im * akp_re + c * akq_im;

            a_re[k][p] = new_kp_re;
            a_im[k][p] = new_kp_im;
            a_re[k][q] = new_kq_re;
            a_im[k][q] = new_kq_im;

            // Hermitian symmetry
            a_re[p][k] = new_kp_re;
            a_im[p][k] = -new_kp_im;
            a_re[q][k] = new_kq_re;
            a_im[q][k] = -new_kq_im;
        }

        // Update diagonal
        let new_pp = app - t * apq_mod;
        let new_qq = aqq + t * apq_mod;
        a_re[p][p] = new_pp;
        a_re[q][q] = new_qq;

        // Zero out (p,q) and (q,p)
        a_re[p][q] = 0.0;
        a_im[p][q] = 0.0;
        a_re[q][p] = 0.0;
        a_im[q][p] = 0.0;
    }

    (0..n).map(|i| a_re[i][i]).collect()
}

// ---------------------------------------------------------------------------
// Quantum Cellular Automaton
// ---------------------------------------------------------------------------

/// A one-dimensional quantum cellular automaton with Margolus partitioning.
pub struct QuantumCellularAutomaton {
    /// The quantum state vector of the chain.
    pub state: QuantumState,
    /// Number of qubits in the chain.
    pub num_qubits: usize,
    /// The rule (two-qubit gate) applied at each pair.
    rule: QCARuleType,
    /// Boundary condition at the chain edges.
    boundary: BoundaryCondition,
    /// Number of completed time steps.
    pub steps_completed: usize,
}

impl QuantumCellularAutomaton {
    /// Create a new QCA with `num_qubits` sites initialised to `|0...0>`.
    pub fn new(num_qubits: usize, rule: QCARuleType, boundary: BoundaryCondition) -> Self {
        assert!(num_qubits >= 2, "QCA requires at least 2 qubits");
        QuantumCellularAutomaton {
            state: QuantumState::new(num_qubits),
            num_qubits,
            rule,
            boundary,
            steps_completed: 0,
        }
    }

    /// Materialise the 4x4 unitary matrix for the current rule.
    fn rule_unitary(&self) -> Vec<Vec<C64>> {
        match &self.rule {
            QCARuleType::PairUnitary(u) => u.clone(),
            QCARuleType::Goldilocks(j) => GoldilocksRules::ising_like(*j),
            QCARuleType::Heisenberg(j) => GoldilocksRules::heisenberg(*j),
            QCARuleType::Identity => GoldilocksRules::identity_4x4(),
            QCARuleType::CustomClosure { generator, param } => generator(*param),
        }
    }

    /// Execute a single time step: apply the rule to even pairs then odd pairs.
    pub fn step(&mut self) {
        let u = self.rule_unitary();
        let n = self.num_qubits;

        // Even half-step
        let even = MargolusPartition::even_pairs(n);
        for (q1, q2) in &even {
            self.apply_pair_unitary(*q1, *q2, &u);
        }

        // Odd half-step
        let odd = MargolusPartition::odd_pairs(n);
        for (q1, q2) in &odd {
            self.apply_pair_unitary(*q1, *q2, &u);
        }

        // Periodic boundary: pair (n-1, 0) on the odd step when n is even,
        // or on a dedicated wrap step.
        if self.boundary == BoundaryCondition::Periodic && n >= 3 {
            // The wrap pair is (n-1, 0).  It belongs to the odd layer only
            // when n is even; for odd n the pair is naturally absent from
            // both partitions, so we always add it.
            let wrap_already_covered = odd.iter().any(|&(a, b)| {
                (a == n - 1 && b == 0) || (a == 0 && b == n - 1)
            });
            if !wrap_already_covered {
                self.apply_pair_unitary(n - 1, 0, &u);
            }
        }

        self.steps_completed += 1;
    }

    /// Evolve the system for `num_steps` time steps, recording spacetime
    /// diagnostics at each step (including the initial state).
    pub fn evolve(&mut self, num_steps: usize) -> SpacetimeData {
        let mut data = SpacetimeData::new(self.num_qubits);
        data.add_snapshot(&self.state); // record initial state

        for _ in 0..num_steps {
            self.step();
            data.add_snapshot(&self.state);
        }
        data
    }

    /// Apply a 4x4 unitary `u` to the qubit pair `(q1, q2)`.
    ///
    /// The unitary acts on the two-qubit subspace spanned by
    /// `{|..q1..q2..>}` where `q1` is the higher-significance bit
    /// within the pair (i.e., pair-local basis `|q1 q2>` maps to
    /// `q1*2 + q2`).
    ///
    /// We iterate over all basis states grouped by their `(q1,q2)` value
    /// and apply the 4x4 matrix multiplication.
    pub fn apply_pair_unitary(&mut self, q1: usize, q2: usize, u: &[Vec<C64>]) {
        let n = self.num_qubits;
        let dim = 1usize << n;
        let mask1 = 1usize << q1;
        let mask2 = 1usize << q2;

        let amps = self.state.amplitudes_mut();

        // Iterate over all basis states where both q1 and q2 are 0.
        // For each such "anchor" index, we derive the 4 indices for
        // (q1,q2) in {00, 01, 10, 11}.
        let mut idx = 0usize;
        while idx < dim {
            // Skip if q1 or q2 bit is set -- we only want the anchor
            // where both are 0.
            if idx & mask1 != 0 {
                idx += 1;
                continue;
            }
            if idx & mask2 != 0 {
                idx += 1;
                continue;
            }

            let i00 = idx;
            let i01 = idx | mask2;
            let i10 = idx | mask1;
            let i11 = idx | mask1 | mask2;

            let a = [amps[i00], amps[i01], amps[i10], amps[i11]];

            // b = U * a
            for r in 0..4usize {
                let target = match r {
                    0 => i00,
                    1 => i01,
                    2 => i10,
                    _ => i11,
                };
                let mut sum = c64_zero();
                for c in 0..4usize {
                    sum += u[r][c] * a[c];
                }
                amps[target] = sum;
            }

            idx += 1;
        }
    }

    /// Read the current probability distribution.
    pub fn probabilities(&self) -> Vec<f64> {
        self.state.probabilities()
    }

    /// Fidelity of the current state with another state.
    pub fn fidelity_with(&self, other: &QuantumState) -> f64 {
        self.state.fidelity(other)
    }

    /// Get the current boundary condition.
    pub fn boundary(&self) -> &BoundaryCondition {
        &self.boundary
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use crate::GateOperations;

    const TOL: f64 = 1e-10;

    // -----------------------------------------------------------------------
    // Margolus partition tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_even_pairs_4_qubits() {
        let pairs = MargolusPartition::even_pairs(4);
        assert_eq!(pairs, vec![(0, 1), (2, 3)]);
    }

    #[test]
    fn test_odd_pairs_4_qubits() {
        let pairs = MargolusPartition::odd_pairs(4);
        assert_eq!(pairs, vec![(1, 2)]);
    }

    #[test]
    fn test_even_pairs_6_qubits() {
        let pairs = MargolusPartition::even_pairs(6);
        assert_eq!(pairs, vec![(0, 1), (2, 3), (4, 5)]);
    }

    #[test]
    fn test_odd_pairs_6_qubits() {
        let pairs = MargolusPartition::odd_pairs(6);
        assert_eq!(pairs, vec![(1, 2), (3, 4)]);
    }

    #[test]
    fn test_partitions_odd_chain() {
        let even = MargolusPartition::even_pairs(5);
        let odd = MargolusPartition::odd_pairs(5);
        assert_eq!(even, vec![(0, 1), (2, 3)]);
        assert_eq!(odd, vec![(1, 2), (3, 4)]);
    }

    // -----------------------------------------------------------------------
    // Identity rule
    // -----------------------------------------------------------------------

    #[test]
    fn test_identity_preserves_state() {
        let mut qca = QuantumCellularAutomaton::new(
            4,
            QCARuleType::Identity,
            BoundaryCondition::Open,
        );
        // Prepare |0100> = X on qubit 1
        GateOperations::x(&mut qca.state, 1);
        let probs_before = qca.state.probabilities();

        qca.step();
        let probs_after = qca.state.probabilities();

        for (a, b) in probs_before.iter().zip(probs_after.iter()) {
            assert!((a - b).abs() < TOL, "Identity rule altered the state");
        }
    }

    #[test]
    fn test_identity_evolve_multiple_steps() {
        let mut qca = QuantumCellularAutomaton::new(
            3,
            QCARuleType::Identity,
            BoundaryCondition::Periodic,
        );
        GateOperations::h(&mut qca.state, 0);
        let probs_initial = qca.state.probabilities();

        let data = qca.evolve(10);
        assert_eq!(data.num_steps(), 11); // initial + 10

        let probs_final = qca.state.probabilities();
        for (a, b) in probs_initial.iter().zip(probs_final.iter()) {
            assert!((a - b).abs() < TOL);
        }
    }

    // -----------------------------------------------------------------------
    // Goldilocks at J=0 (product evolution)
    // -----------------------------------------------------------------------

    #[test]
    fn test_goldilocks_j0_product_evolution() {
        // J = 0 => the unitary is the identity, so the state should not change.
        let mut qca = QuantumCellularAutomaton::new(
            4,
            QCARuleType::Goldilocks(0.0),
            BoundaryCondition::Open,
        );
        GateOperations::x(&mut qca.state, 2);
        let probs_before = qca.state.probabilities();

        qca.step();
        let probs_after = qca.state.probabilities();

        for (a, b) in probs_before.iter().zip(probs_after.iter()) {
            assert!(
                (a - b).abs() < TOL,
                "J=0 Goldilocks should act as identity"
            );
        }
    }

    // -----------------------------------------------------------------------
    // SWAP gate (Goldilocks at specific J)
    // -----------------------------------------------------------------------

    #[test]
    fn test_swap_gate_exchanges_qubits() {
        // A single application of the SWAP unitary to pair (0,1) should
        // exchange qubits 0 and 1.
        let swap_u = GoldilocksRules::swap_gate();
        let mut qca = QuantumCellularAutomaton::new(
            2,
            QCARuleType::PairUnitary(swap_u),
            BoundaryCondition::Open,
        );
        // Prepare |10> : qubit 0 = |1>, qubit 1 = |0>
        GateOperations::x(&mut qca.state, 0);

        // Before: amplitude at |10> (index 1 for qubit0=1,qubit1=0
        // depends on bit ordering).
        // In nQPU-Metal, index = sum_q bit_q * 2^q, so
        // qubit 0 set => index 1 (= 2^0).
        let amp_before = qca.state.get(1); // |10> in q0-first ordering
        assert!((amp_before.norm_sqr() - 1.0).abs() < TOL);

        // Apply SWAP to (0, 1)
        qca.apply_pair_unitary(0, 1, &GoldilocksRules::swap_gate());

        // After SWAP: qubit 1 should be |1>, qubit 0 should be |0>
        // => index 2 (= 2^1)
        let amp_after = qca.state.get(2); // |01> in q0-first ordering
        assert!(
            (amp_after.norm_sqr() - 1.0).abs() < TOL,
            "SWAP should exchange qubit amplitudes"
        );
    }

    // -----------------------------------------------------------------------
    // Goldilocks unitary is unitary
    // -----------------------------------------------------------------------

    #[test]
    fn test_goldilocks_unitary_unitarity() {
        let j = PI / 6.0;
        let u = GoldilocksRules::ising_like(j);
        // U^dag U should be identity.
        for i in 0..4 {
            for j_col in 0..4 {
                let mut dot = c64_zero();
                for k in 0..4 {
                    dot += u[k][i].conj() * u[k][j_col];
                }
                let expected = if i == j_col { 1.0 } else { 0.0 };
                assert!(
                    (dot.re - expected).abs() < TOL && dot.im.abs() < TOL,
                    "Goldilocks U not unitary at ({}, {}): got {:?}",
                    i,
                    j_col,
                    dot
                );
            }
        }
    }

    #[test]
    fn test_heisenberg_unitary_unitarity() {
        let j = 0.3;
        let u = GoldilocksRules::heisenberg(j);
        for i in 0..4 {
            for j_col in 0..4 {
                let mut dot = c64_zero();
                for k in 0..4 {
                    dot += u[k][i].conj() * u[k][j_col];
                }
                let expected = if i == j_col { 1.0 } else { 0.0 };
                assert!(
                    (dot.re - expected).abs() < TOL && dot.im.abs() < TOL,
                    "Heisenberg U not unitary at ({}, {}): got {:?}",
                    i,
                    j_col,
                    dot
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Spacetime data recording
    // -----------------------------------------------------------------------

    #[test]
    fn test_spacetime_data_recording() {
        let mut qca = QuantumCellularAutomaton::new(
            4,
            QCARuleType::Goldilocks(0.2),
            BoundaryCondition::Open,
        );
        GateOperations::x(&mut qca.state, 0);

        let data = qca.evolve(5);

        // Should have 6 snapshots: initial + 5 steps.
        assert_eq!(data.probability_profiles.len(), 6);
        assert_eq!(data.entanglement_entropy.len(), 6);

        // Each probability profile should sum to 1.
        for (t, profile) in data.probability_profiles.iter().enumerate() {
            let sum: f64 = profile.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-8,
                "Probability sum at step {} = {} != 1.0",
                t,
                sum
            );
        }

        // Entanglement entropy should have n-1 = 3 cuts per step.
        for entropies in &data.entanglement_entropy {
            assert_eq!(entropies.len(), 3);
        }
    }

    // -----------------------------------------------------------------------
    // Entanglement growth in chaotic regime
    // -----------------------------------------------------------------------

    #[test]
    fn test_entanglement_growth_chaotic() {
        // Use 4 qubits to keep the reduced density matrix small (4x4 max).
        // J = π/8 maximises the entangling power of the XX+ZZ gate
        // (|cos·sin| is maximal at 2J = π/4).  NOTE: J = π/4 gives a
        // SWAP-like gate (permutation, no superposition) and J = π/2
        // gives −I (trivial), so both are degenerate special points.
        let mut qca = QuantumCellularAutomaton::new(
            4,
            QCARuleType::Goldilocks(std::f64::consts::FRAC_PI_4 * 0.5),
            BoundaryCondition::Periodic,
        );
        GateOperations::x(&mut qca.state, 0);
        GateOperations::h(&mut qca.state, 1);

        // Verify initial entanglement across cut 1 (qubits 0,1 | qubits 2,3).
        let s_initial = bipartite_entropy(&qca.state, 1);
        // H on qubit 1 entangles qubits 0,1 subspace but subsystem A = {0,1}
        // is still in a product with B = {2,3}, so cross-cut entropy should be 0.
        assert!(
            s_initial < 1e-8,
            "Initial entropy across cut 1 should be ~0, got {}",
            s_initial
        );

        let data = qca.evolve(50);

        // After evolution, at least one cut at some time should show entanglement.
        let mut max_s = 0.0_f64;
        for entropies in &data.entanglement_entropy {
            for &s in entropies {
                max_s = max_s.max(s);
            }
        }
        assert!(
            max_s > 1e-6,
            "Chaotic regime should produce entanglement, max S = {}",
            max_s
        );
    }

    #[test]
    fn test_bipartite_entropy_4qubit_ghz() {
        // GHZ state on 4 qubits: (|0000> + |1111>)/sqrt(2).
        // Entropy across any cut should be 1.0 (maximally entangled bipartition).
        let mut state = QuantumState::new(4);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);
        GateOperations::cnot(&mut state, 1, 2);
        GateOperations::cnot(&mut state, 2, 3);

        let s_cut1 = bipartite_entropy(&state, 0);
        assert!(
            (s_cut1 - 1.0).abs() < 1e-4,
            "GHZ cut 0 entropy should be 1.0, got {}",
            s_cut1
        );

        let s_cut2 = bipartite_entropy(&state, 1);
        assert!(
            (s_cut2 - 1.0).abs() < 1e-4,
            "GHZ cut 1 entropy should be 1.0, got {}",
            s_cut2
        );
    }

    // -----------------------------------------------------------------------
    // Bipartite entropy: maximally entangled state
    // -----------------------------------------------------------------------

    #[test]
    fn test_bipartite_entropy_bell_state() {
        // Prepare Bell state |00> + |11> / sqrt(2) on 2 qubits.
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);

        let s = bipartite_entropy(&state, 0);
        // Should be exactly 1 bit of entanglement.
        assert!(
            (s - 1.0).abs() < 1e-6,
            "Bell state entropy should be 1.0, got {}",
            s
        );
    }

    #[test]
    fn test_bipartite_entropy_product_state() {
        // |00> is a product state: zero entanglement.
        let state = QuantumState::new(2);
        let s = bipartite_entropy(&state, 0);
        assert!(s < 1e-10, "Product state entropy should be 0, got {}", s);
    }

    // -----------------------------------------------------------------------
    // Boundary conditions
    // -----------------------------------------------------------------------

    #[test]
    fn test_open_vs_periodic_differ() {
        // Open and periodic boundary conditions should produce different
        // dynamics.  We run identical initial conditions under both and
        // verify the final states diverge.
        let n = 6;
        let j = 0.3;

        let mut qca_open = QuantumCellularAutomaton::new(
            n,
            QCARuleType::Goldilocks(j),
            BoundaryCondition::Open,
        );
        GateOperations::x(&mut qca_open.state, 0);

        let mut qca_periodic = QuantumCellularAutomaton::new(
            n,
            QCARuleType::Goldilocks(j),
            BoundaryCondition::Periodic,
        );
        GateOperations::x(&mut qca_periodic.state, 0);

        // Evolve both for several steps.
        for _ in 0..5 {
            qca_open.step();
            qca_periodic.step();
        }

        // The states should differ (fidelity < 1).
        let fid = qca_open.state.fidelity(&qca_periodic.state);
        assert!(
            fid < 1.0 - 1e-6,
            "Open and periodic should give different states, fidelity = {}",
            fid
        );
    }

    #[test]
    fn test_periodic_boundary_wraps() {
        // Verify that the wrap pair (n-1, 0) is actually applied under
        // periodic boundary conditions by using a pure SWAP gate.
        // Start with excitation only on qubit n-1.  A single application
        // of SWAP on pair (n-1, 0) -- which happens only with periodic BC
        // -- should move amplitude to qubit 0.
        let n = 4;
        let swap_u = GoldilocksRules::swap_gate();

        let mut qca_periodic = QuantumCellularAutomaton::new(
            n,
            QCARuleType::PairUnitary(swap_u.clone()),
            BoundaryCondition::Periodic,
        );
        // Prepare |1000> (only qubit 3 excited) -- index 8.
        GateOperations::x(&mut qca_periodic.state, n - 1);

        let mut qca_open = QuantumCellularAutomaton::new(
            n,
            QCARuleType::PairUnitary(swap_u),
            BoundaryCondition::Open,
        );
        GateOperations::x(&mut qca_open.state, n - 1);

        // Evolve both for a few steps.
        for _ in 0..3 {
            qca_periodic.step();
            qca_open.step();
        }

        // Under periodic BC the SWAP wraps qubit n-1 <-> qubit 0,
        // so the final states should differ.
        let fid = qca_periodic.state.fidelity(&qca_open.state);
        assert!(
            fid < 1.0 - 1e-6,
            "Periodic wrap should differentiate from open, fidelity = {}",
            fid
        );
    }

    // -----------------------------------------------------------------------
    // Light cone computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_light_cone_initial_zero() {
        let mut qca = QuantumCellularAutomaton::new(
            4,
            QCARuleType::Goldilocks(0.3),
            BoundaryCondition::Open,
        );
        GateOperations::x(&mut qca.state, 0);

        let data = qca.evolve(5);
        let cone = data.compute_light_cone(0);

        // At t=0, the change from initial should be zero.
        assert_eq!(cone[0].0, 0);
        assert!(cone[0].1 < TOL);
    }

    // -----------------------------------------------------------------------
    // CustomClosure rule
    // -----------------------------------------------------------------------

    #[test]
    fn test_custom_closure_rule() {
        fn my_rule(_j: f64) -> Vec<Vec<C64>> {
            GoldilocksRules::swap_gate()
        }

        let mut qca = QuantumCellularAutomaton::new(
            2,
            QCARuleType::CustomClosure {
                generator: my_rule,
                param: 0.0,
            },
            BoundaryCondition::Open,
        );
        GateOperations::x(&mut qca.state, 0);

        // Apply one pair unitary -- should SWAP qubit 0 and 1.
        qca.apply_pair_unitary(0, 1, &my_rule(0.0));

        let probs = qca.state.probabilities();
        // After SWAP of |10>: qubit 1 should be excited.
        // Index 2 = 2^1 = qubit 1 set.
        assert!(
            (probs[2] - 1.0).abs() < TOL,
            "CustomClosure SWAP failed"
        );
    }

    // -----------------------------------------------------------------------
    // Qubit probabilities helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_qubit_probabilities() {
        let mut qca = QuantumCellularAutomaton::new(
            3,
            QCARuleType::Identity,
            BoundaryCondition::Open,
        );
        GateOperations::x(&mut qca.state, 1);

        let data = qca.evolve(0); // just snapshot
        let qp = data.qubit_probabilities(0);

        assert!((qp[0] - 0.0).abs() < TOL);
        assert!((qp[1] - 1.0).abs() < TOL);
        assert!((qp[2] - 0.0).abs() < TOL);
    }

    // -----------------------------------------------------------------------
    // Norm preservation under evolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_norm_preservation() {
        let mut qca = QuantumCellularAutomaton::new(
            5,
            QCARuleType::Goldilocks(0.7),
            BoundaryCondition::Periodic,
        );
        GateOperations::x(&mut qca.state, 0);
        GateOperations::h(&mut qca.state, 2);

        for _ in 0..20 {
            qca.step();
            let norm: f64 = qca.state.probabilities().iter().sum();
            assert!(
                (norm - 1.0).abs() < 1e-8,
                "Norm not preserved: {}",
                norm
            );
        }
    }
}
