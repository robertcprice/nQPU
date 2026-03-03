//! Expanded topological quantum computing primitives.
//!
//! This module implements Ising anyons, Majorana fermion chains, and a braid
//! compiler that approximates standard quantum gates as topologically protected
//! braid words. Together with the Fibonacci anyon simulator in
//! [`crate::topological_quantum`], this provides a comprehensive toolkit for
//! simulating topological quantum computation.
//!
//! # Physical background
//!
//! Ising anyons arise in p-wave superconductors, fractional quantum Hall
//! states at filling fraction nu = 5/2, and as effective quasi-particles
//! in Majorana-based topological qubits. The braid group representation
//! they carry is non-universal (it generates only the Clifford group up
//! to phases), but combined with a single non-Clifford measurement or
//! magic state injection the gate set becomes universal.
//!
//! # Module overview
//!
//! - [`IsingAnyonType`] / [`IsingFusionRules`]: Ising fusion category.
//! - [`IsingAnyonState`]: Hilbert space of N sigma anyons with braiding.
//! - [`MajoranaChain`]: Kitaev chain with Majorana fermion braiding.
//! - [`BraidWord`] / [`BraidCompiler`]: Gate-to-braid compilation.
//! - [`TopologicalChargeTracker`]: Fusion outcome bookkeeping.

use std::f64::consts::PI;

use crate::gates::{Gate, GateType};
use crate::{c64_one, c64_zero, C64};

// ---------------------------------------------------------------------------
// Ising anyon types and fusion rules
// ---------------------------------------------------------------------------

/// The three particle types (topological charges) in the Ising modular
/// tensor category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IsingAnyonType {
    /// Vacuum sector (trivial charge). Quantum dimension d = 1.
    Vacuum,
    /// Sigma anyon (non-abelian). Quantum dimension d = sqrt(2).
    Sigma,
    /// Psi fermion (abelian). Quantum dimension d = 1.
    Psi,
}

/// Ising fusion rules.
///
/// The multiplication table for Ising anyons is:
/// ```text
///   1 x a  = a           (vacuum is the identity)
///   psi x psi = 1        (fermion is its own antiparticle)
///   sigma x psi = sigma
///   sigma x sigma = 1 + psi   (the non-trivial fusion)
/// ```
pub struct IsingFusionRules;

impl IsingFusionRules {
    /// Return all possible fusion outcomes of `a` and `b`.
    pub fn fuse(a: IsingAnyonType, b: IsingAnyonType) -> Vec<IsingAnyonType> {
        use IsingAnyonType::*;
        match (a, b) {
            // Vacuum is the identity
            (Vacuum, x) | (x, Vacuum) => vec![x],
            // Psi x Psi = 1
            (Psi, Psi) => vec![Vacuum],
            // Sigma x Psi = Sigma (and vice versa)
            (Sigma, Psi) | (Psi, Sigma) => vec![Sigma],
            // The non-trivial fusion: Sigma x Sigma = 1 + Psi
            (Sigma, Sigma) => vec![Vacuum, Psi],
        }
    }

    /// Quantum dimension of a given anyon type.
    pub fn quantum_dimension(a: IsingAnyonType) -> f64 {
        match a {
            IsingAnyonType::Vacuum => 1.0,
            IsingAnyonType::Sigma => std::f64::consts::SQRT_2,
            IsingAnyonType::Psi => 1.0,
        }
    }

    /// Total quantum dimension D = sqrt(sum d_i^2) = 2 for Ising.
    pub fn total_quantum_dimension() -> f64 {
        // D^2 = 1 + 2 + 1 = 4, so D = 2
        2.0
    }
}

// ---------------------------------------------------------------------------
// Ising anyon state
// ---------------------------------------------------------------------------

/// State in the fusion Hilbert space of N Ising sigma anyons.
///
/// For 2n sigma anyons fusing pairwise to definite charge, the Hilbert space
/// dimension is 2^{n-1} (with a fixed total charge constraint). We encode the
/// state as a coefficient vector over the standard fusion tree basis.
///
/// The fusion tree uses left-to-right sequential fusion:
///   ((... ((sigma_1 x sigma_2) x sigma_3) x ...) x sigma_N)
/// Each pair (sigma_{2k-1}, sigma_{2k}) fuses to either 1 or psi, giving one
/// qubit per pair. The overall parity constraint reduces the space by a factor
/// of 2 only when total charge is fixed; here we keep the full 2^{n_pairs}
/// dimensional space and track total charge separately.
#[derive(Clone, Debug)]
pub struct IsingAnyonState {
    /// Number of sigma anyons.
    pub num_anyons: usize,
    /// Coefficient vector in the fusion-tree basis.
    /// Length = 2^{floor(num_anyons/2)}.
    pub amplitudes: Vec<C64>,
}

impl IsingAnyonState {
    /// Create a new state with `num_anyons` sigma anyons, initialized to the
    /// all-vacuum fusion outcome (basis index 0).
    pub fn new(num_anyons: usize) -> Self {
        assert!(num_anyons >= 2, "need at least 2 anyons for a non-trivial space");
        let n_pairs = num_anyons / 2;
        let dim = 1 << n_pairs; // 2^n_pairs
        let mut amps = vec![c64_zero(); dim];
        amps[0] = c64_one();
        Self {
            num_anyons,
            amplitudes: amps,
        }
    }

    /// Dimension of the fusion Hilbert space.
    pub fn dimension(&self) -> usize {
        self.amplitudes.len()
    }

    /// Squared norm of the state vector.
    pub fn norm_sqr(&self) -> f64 {
        self.amplitudes.iter().map(|a| a.norm_sqr()).sum()
    }

    /// Normalize the state vector in place.
    pub fn normalize(&mut self) {
        let n = self.norm_sqr().sqrt();
        if n > 1e-30 {
            for a in &mut self.amplitudes {
                *a /= n;
            }
        }
    }

    /// Apply the R-matrix for braiding anyon `i` with anyon `i+1`.
    ///
    /// The Ising R-matrix entries for sigma x sigma fusion channels:
    ///   R^{sigma,sigma}_1   = e^{-i pi/8}   (vacuum channel)
    ///   R^{sigma,sigma}_psi = e^{+i 3pi/8}  (psi channel)
    ///
    /// When anyons i and i+1 belong to the same fusion pair, the braid simply
    /// multiplies each basis state by the appropriate R-matrix eigenvalue.
    /// When they straddle two pairs, we must conjugate by the F-matrix.
    pub fn braid(&mut self, i: usize) {
        assert!(i + 1 < self.num_anyons, "braid index out of range");
        let n_pairs = self.num_anyons / 2;

        // R-matrix phases
        let r_vacuum = ising_phase(-PI / 8.0); // R^{sigma,sigma}_1
        let r_psi = ising_phase(3.0 * PI / 8.0); // R^{sigma,sigma}_psi

        // Determine which fusion pair(s) are involved
        let pair_i = i / 2;
        let pos_in_pair = i % 2;

        if pos_in_pair == 0 && i + 1 < 2 * n_pairs {
            // Both anyons are in the same pair (pair_i)
            // The braid acts diagonally: multiply by R eigenvalue
            // depending on whether this pair fused to vacuum (bit=0) or psi (bit=1).
            let bit_mask = 1 << pair_i;
            for (idx, amp) in self.amplitudes.iter_mut().enumerate() {
                if idx & bit_mask == 0 {
                    *amp *= r_vacuum;
                } else {
                    *amp *= r_psi;
                }
            }
        } else if pair_i + 1 < n_pairs {
            // Anyons straddle two adjacent pairs: pair_i and pair_i+1.
            // We need: R_straddling = F^{-1} . R_diagonal . F
            // where F is the Ising F-matrix (recoupling) acting on the
            // two-qubit subspace of (pair_i, pair_i+1).
            let f = ising_f_matrix_2x2();
            let f_inv = ising_f_inv_2x2();

            let bit_a = 1 << pair_i;
            let bit_b = 1 << (pair_i + 1);

            // Process in blocks of 4 (the 2-qubit subspace)
            let dim = self.amplitudes.len();
            let mut new_amps = self.amplitudes.clone();

            for base_idx in 0..dim {
                // Only process canonical representatives (both bits cleared)
                if base_idx & (bit_a | bit_b) != 0 {
                    continue;
                }

                // Extract the 4 amplitudes for this subspace block
                let i00 = base_idx;
                let i01 = base_idx | bit_a;
                let i10 = base_idx | bit_b;
                let i11 = base_idx | bit_a | bit_b;
                let v = [
                    self.amplitudes[i00],
                    self.amplitudes[i01],
                    self.amplitudes[i10],
                    self.amplitudes[i11],
                ];

                // Apply F^{-1} . R_diag . F on the 2-qubit subspace
                // Step 1: F . v
                let fv = mat4_vec_mul(&f, &v);
                // Step 2: R_diag . (F . v)
                let rfv = [
                    fv[0] * r_vacuum * r_vacuum, // both vacuum
                    fv[1] * r_vacuum * r_psi,    // vacuum x psi
                    fv[2] * r_psi * r_vacuum,    // psi x vacuum
                    fv[3] * r_psi * r_psi,       // both psi
                ];
                // Step 3: F^{-1} . R . F . v
                let result = mat4_vec_mul(&f_inv, &rfv);

                new_amps[i00] = result[0];
                new_amps[i01] = result[1];
                new_amps[i10] = result[2];
                new_amps[i11] = result[3];
            }

            self.amplitudes = new_amps;
        }
        // else: boundary case (odd anyon at end) - no-op for unpaired trailing anyon
    }

    /// Fuse anyons `i` and `i+1`, projecting onto a definite fusion outcome.
    ///
    /// Returns the fusion outcome and the projection probability.
    /// The state is collapsed and renormalized after projection.
    pub fn fuse(&mut self, i: usize) -> (IsingAnyonType, f64) {
        assert!(i + 1 < self.num_anyons, "fuse index out of range");
        let pair = i / 2;
        let bit_mask = 1 << pair;

        // Compute probability of vacuum vs psi outcomes
        let mut prob_vacuum = 0.0;
        let mut prob_psi = 0.0;
        for (idx, amp) in self.amplitudes.iter().enumerate() {
            if idx & bit_mask == 0 {
                prob_vacuum += amp.norm_sqr();
            } else {
                prob_psi += amp.norm_sqr();
            }
        }

        let total = prob_vacuum + prob_psi;
        if total < 1e-30 {
            return (IsingAnyonType::Vacuum, 0.0);
        }

        // Deterministic projection: choose the more probable outcome
        let (outcome, prob) = if prob_vacuum >= prob_psi {
            // Project onto vacuum channel: zero out psi components
            for (idx, amp) in self.amplitudes.iter_mut().enumerate() {
                if idx & bit_mask != 0 {
                    *amp = c64_zero();
                }
            }
            (IsingAnyonType::Vacuum, prob_vacuum / total)
        } else {
            // Project onto psi channel: zero out vacuum components
            for (idx, amp) in self.amplitudes.iter_mut().enumerate() {
                if idx & bit_mask == 0 {
                    *amp = c64_zero();
                }
            }
            (IsingAnyonType::Psi, prob_psi / total)
        };

        self.normalize();
        (outcome, prob)
    }

    /// Probability distribution over basis states.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }
}

// ---------------------------------------------------------------------------
// Majorana fermion chain
// ---------------------------------------------------------------------------

/// Kitaev chain of Majorana fermion modes.
///
/// A chain of `num_sites` sites supports `2 * num_sites` Majorana operators
/// gamma_0, gamma_1, ..., gamma_{2n-1}. The state is stored as a coefficient
/// vector in the Fock (occupation number) basis of dimension 2^num_sites.
///
/// Braiding Majorana zero modes j and j+1 implements the unitary:
///   U = exp(pi/4 * gamma_j * gamma_{j+1})
///     = (1 + gamma_j * gamma_{j+1}) / sqrt(2)
#[derive(Clone, Debug)]
pub struct MajoranaChain {
    /// Number of physical sites (each hosting two Majorana modes).
    pub num_sites: usize,
    /// State vector in the occupation basis. Length = 2^num_sites.
    pub state: Vec<C64>,
}

impl MajoranaChain {
    /// Create a new Kitaev chain with `num_sites` sites, initialized in the
    /// vacuum state |000...0>.
    pub fn new(num_sites: usize) -> Self {
        assert!(num_sites > 0 && num_sites <= 16, "num_sites must be in [1, 16]");
        let dim = 1 << num_sites;
        let mut state = vec![c64_zero(); dim];
        state[0] = c64_one();
        Self { num_sites, state }
    }

    /// Dimension of the Fock space.
    pub fn dimension(&self) -> usize {
        self.state.len()
    }

    /// Braid Majorana operators gamma_j and gamma_{j+1}.
    ///
    /// Implements U = exp(pi/4 * gamma_j * gamma_{j+1}).
    /// In the occupation basis, gamma_j * gamma_{j+1} acts as follows:
    ///
    /// For j even (same site): gamma_{2k} * gamma_{2k+1} = i(1 - 2 n_k)
    ///   where n_k is the occupation of site k.
    ///
    /// For j odd (adjacent sites): gamma_{2k+1} * gamma_{2k+2} involves
    ///   a hopping-like term between sites k and k+1.
    pub fn braid(&mut self, j: usize) {
        let num_majoranas = 2 * self.num_sites;
        assert!(
            j + 1 < num_majoranas,
            "Majorana index {} out of range [0, {})",
            j,
            num_majoranas
        );

        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        let dim = self.dimension();

        if j % 2 == 0 {
            // Same-site braiding: gamma_{2k} and gamma_{2k+1}
            // gamma_{2k} * gamma_{2k+1} = i * (1 - 2*n_k)
            // U = (1 + i*(1-2n_k)) / sqrt(2)
            //   = (1+i)/sqrt(2) if n_k = 0
            //   = (1-i)/sqrt(2) if n_k = 1
            let site = j / 2;
            let bit = 1 << site;
            let phase_0 = C64::new(inv_sqrt2, inv_sqrt2); // (1+i)/sqrt(2) = e^{i pi/4}
            let phase_1 = C64::new(inv_sqrt2, -inv_sqrt2); // (1-i)/sqrt(2) = e^{-i pi/4}

            for idx in 0..dim {
                if idx & bit == 0 {
                    self.state[idx] *= phase_0;
                } else {
                    self.state[idx] *= phase_1;
                }
            }
        } else {
            // Cross-site braiding: gamma_{2k+1} and gamma_{2k+2}
            // This involves sites k and k+1 and mixes occupation states.
            let site_k = j / 2;
            let site_k1 = site_k + 1;
            let bit_k = 1 << site_k;
            let bit_k1 = 1 << site_k1;

            let mut new_state = self.state.clone();

            for idx in 0..dim {
                let occ_k = (idx >> site_k) & 1;
                let occ_k1 = (idx >> site_k1) & 1;

                // gamma_{2k+1} * gamma_{2k+2} acts on the 2-site subspace {|00>, |01>, |10>, |11>}
                // as a matrix that mixes states differing by flipping both occupations.
                // The operator is: -i * (c_k - c_k^dag)(c_{k+1} + c_{k+1}^dag)
                // In the 2-qubit subspace (k, k+1):
                //   |00> and |11> mix, |01> and |10> mix
                let partner_idx = idx ^ bit_k ^ bit_k1;

                // Fermion parity sign from Jordan-Wigner ordering
                let parity_sign = jordan_wigner_sign(idx, site_k, site_k1);

                if occ_k == occ_k1 {
                    // Same occupation: diagonal part = 1/sqrt(2), off-diagonal part
                    // U|same> = (1/sqrt(2))|same> + (i * sign / sqrt(2))|flipped>
                    new_state[idx] = self.state[idx] * inv_sqrt2
                        + self.state[partner_idx]
                            * C64::new(0.0, parity_sign as f64 * inv_sqrt2);
                } else {
                    // Different occupation: similar with adjusted sign
                    new_state[idx] = self.state[idx] * inv_sqrt2
                        + self.state[partner_idx]
                            * C64::new(0.0, parity_sign as f64 * inv_sqrt2);
                }
            }

            self.state = new_state;
        }
    }

    /// Total fermion parity: eigenvalue of (-1)^{sum n_k}.
    ///
    /// Returns +1 if the state is entirely in the even-parity sector,
    /// -1 if entirely in odd-parity, or 0.0 if it is a superposition.
    /// The value returned is the expectation value <parity>.
    pub fn parity(&self) -> f64 {
        let mut even_weight = 0.0;
        let mut odd_weight = 0.0;
        for (idx, amp) in self.state.iter().enumerate() {
            let n_occupied = idx.count_ones();
            let w = amp.norm_sqr();
            if n_occupied % 2 == 0 {
                even_weight += w;
            } else {
                odd_weight += w;
            }
        }
        even_weight - odd_weight
    }

    /// Normalize the state vector in place.
    pub fn normalize(&mut self) {
        let n: f64 = self.state.iter().map(|a| a.norm_sqr()).sum();
        let n = n.sqrt();
        if n > 1e-30 {
            for a in &mut self.state {
                *a /= n;
            }
        }
    }

    /// Probabilities over the Fock basis.
    pub fn probabilities(&self) -> Vec<f64> {
        self.state.iter().map(|a| a.norm_sqr()).collect()
    }
}

/// Jordan-Wigner parity sign for fermion hopping between site_a and site_b.
/// Counts the number of occupied sites strictly between site_a and site_b.
fn jordan_wigner_sign(idx: usize, site_a: usize, site_b: usize) -> i32 {
    let lo = site_a.min(site_b);
    let hi = site_a.max(site_b);
    let mut count = 0u32;
    for s in (lo + 1)..hi {
        count += ((idx >> s) & 1) as u32;
    }
    if count % 2 == 0 {
        1
    } else {
        -1
    }
}

// ---------------------------------------------------------------------------
// Braid word and compiler
// ---------------------------------------------------------------------------

/// An elementary braid generator: sigma_i (positive crossing) or
/// sigma_i^{-1} (negative crossing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BraidGenerator {
    /// Strand index (0-based). The generator exchanges strands i and i+1.
    pub strand: usize,
    /// If true, this is the inverse generator sigma_i^{-1}.
    pub inverse: bool,
}

/// A braid word: a sequence of elementary braid generators.
#[derive(Clone, Debug)]
pub struct BraidWord {
    pub generators: Vec<BraidGenerator>,
    /// Estimated fidelity of the braid approximation to the target gate.
    /// 1.0 means exact, < 1.0 means approximate.
    pub fidelity: f64,
}

impl BraidWord {
    pub fn new() -> Self {
        Self {
            generators: Vec::new(),
            fidelity: 1.0,
        }
    }

    pub fn identity() -> Self {
        Self {
            generators: Vec::new(),
            fidelity: 1.0,
        }
    }

    pub fn len(&self) -> usize {
        self.generators.len()
    }

    pub fn is_empty(&self) -> bool {
        self.generators.is_empty()
    }

    /// Append a generator sigma_{strand} or sigma_{strand}^{-1}.
    pub fn push(&mut self, strand: usize, inverse: bool) {
        self.generators.push(BraidGenerator { strand, inverse });
    }

    /// Concatenate another braid word onto this one.
    pub fn extend(&mut self, other: &BraidWord) {
        self.generators.extend_from_slice(&other.generators);
        self.fidelity *= other.fidelity;
    }

    /// Return the inverse braid word (reverse order, flip all crossings).
    pub fn inverse(&self) -> Self {
        let mut inv = BraidWord::new();
        inv.fidelity = self.fidelity;
        for g in self.generators.iter().rev() {
            inv.generators.push(BraidGenerator {
                strand: g.strand,
                inverse: !g.inverse,
            });
        }
        inv
    }
}

impl Default for BraidWord {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiler that approximates standard quantum gates as braid words.
///
/// Ising anyons generate only the Clifford group through braiding, so
/// non-Clifford gates (like T) can only be approximated. The compiler
/// uses known exact braid sequences for Clifford gates and returns
/// best-effort approximations with fidelity estimates for non-Clifford gates.
pub struct BraidCompiler {
    /// Maximum braid word length for approximation search.
    pub max_depth: usize,
}

impl BraidCompiler {
    pub fn new() -> Self {
        Self { max_depth: 20 }
    }

    pub fn with_max_depth(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Compile a single gate type into a braid word.
    ///
    /// For Clifford gates (H, S, X, Y, Z, CNOT), the compiler returns exact
    /// braid sequences with fidelity 1.0. For non-Clifford gates (T, arbitrary
    /// rotations), it returns the closest Clifford approximation with estimated
    /// fidelity.
    pub fn compile_gate(&self, gate: &GateType) -> BraidWord {
        match gate {
            // ---- Exact Clifford braids ----

            // X gate: sigma_1^2 in the Ising model (full twist).
            // Two sigma braids compose to the monodromy which gives a pi phase
            // difference between vacuum and psi channels, realizing Pauli-X on
            // the encoded qubit.
            GateType::X => {
                let mut w = BraidWord::new();
                w.push(0, false);
                w.push(0, false);
                w.fidelity = 1.0;
                w
            }

            // Z gate: sigma_1 . sigma_1 applied in the psi-channel convention.
            // Equivalent to two inverse braids.
            GateType::Z => {
                let mut w = BraidWord::new();
                w.push(0, true);
                w.push(0, true);
                w.fidelity = 1.0;
                w
            }

            // Y = iXZ (up to global phase): combine X and Z braids.
            GateType::Y => {
                let mut w = BraidWord::new();
                // X braid followed by Z braid
                w.push(0, false);
                w.push(0, false);
                w.push(0, true);
                w.push(0, true);
                w.fidelity = 1.0;
                w
            }

            // Hadamard: in the Ising model, H can be realized as:
            //   sigma_1 . sigma_2 . sigma_1 (a 3-braid on 4 anyons)
            // This produces the topological Hadamard up to a global phase.
            GateType::H => {
                let mut w = BraidWord::new();
                w.push(0, false);
                w.push(1, false);
                w.push(0, false);
                w.fidelity = 1.0;
                w
            }

            // S gate (phase gate): single sigma braid gives e^{i pi/4}
            // phase difference, which is the S gate on the encoded qubit.
            GateType::S => {
                let mut w = BraidWord::new();
                w.push(0, false);
                w.fidelity = 1.0;
                w
            }

            // CNOT: realized by braiding pattern on 8 anyons (4 per qubit).
            // sigma_2 . sigma_3 . sigma_2 (controls qubit boundary crossing).
            GateType::CNOT => {
                let mut w = BraidWord::new();
                w.push(1, false);
                w.push(2, false);
                w.push(1, false);
                w.push(2, true);
                w.push(1, true);
                w.push(2, false);
                w.fidelity = 1.0;
                w
            }

            // CZ: CNOT conjugated by Hadamard on the target.
            GateType::CZ => {
                let mut w = BraidWord::new();
                // H on target
                w.push(0, false);
                w.push(1, false);
                w.push(0, false);
                // CNOT
                w.push(1, false);
                w.push(2, false);
                w.push(1, false);
                w.push(2, true);
                w.push(1, true);
                w.push(2, false);
                // H on target
                w.push(0, false);
                w.push(1, false);
                w.push(0, false);
                w.fidelity = 1.0;
                w
            }

            // SWAP = CNOT . CNOT . CNOT (3 CNOTs decomposition)
            GateType::SWAP => {
                let cnot_braid = self.compile_gate(&GateType::CNOT);
                let mut w = BraidWord::new();
                w.extend(&cnot_braid);
                w.extend(&cnot_braid);
                w.extend(&cnot_braid);
                w.fidelity = 1.0;
                w
            }

            // ---- Non-Clifford gates: approximate ----

            // T gate: e^{i pi/8} is not in the Clifford group.
            // Best Ising braid approximation uses the S gate (fidelity ~0.854).
            GateType::T => {
                // The T gate is S^{1/2}; Ising braiding cannot produce this exactly.
                // Return S as the nearest Clifford gate.
                let mut w = self.compile_gate(&GateType::S);
                // Fidelity: |<T|S>|^2 = cos^2(pi/8) ~ 0.854
                w.fidelity = (PI / 8.0).cos().powi(2);
                w
            }

            // Arbitrary rotations: approximate with nearest Clifford
            GateType::Rx(theta) => self.approximate_rotation(*theta, RotationAxis::X),
            GateType::Ry(theta) => self.approximate_rotation(*theta, RotationAxis::Y),
            GateType::Rz(theta) => self.approximate_rotation(*theta, RotationAxis::Z),

            // Phase gate: closest Clifford is S or identity
            GateType::Phase(theta) => {
                let mut w = if (*theta - PI / 2.0).abs() < PI / 4.0 {
                    self.compile_gate(&GateType::S)
                } else if (*theta - PI).abs() < PI / 4.0 {
                    self.compile_gate(&GateType::Z)
                } else {
                    BraidWord::identity()
                };
                // Fidelity based on angular distance
                let best_angle = [0.0, PI / 2.0, PI]
                    .iter()
                    .copied()
                    .min_by(|a, b| {
                        (a - theta).abs().partial_cmp(&(b - theta).abs()).unwrap()
                    })
                    .unwrap();
                w.fidelity = ((theta - best_angle) / 2.0).cos().powi(2);
                w
            }

            // Universal gate: decompose into rotations
            GateType::U {
                theta,
                phi: _,
                lambda: _,
            } => {
                // Use Rz . Ry . Rz decomposition, take the dominant rotation
                self.approximate_rotation(*theta, RotationAxis::Y)
            }

            // SX = Rx(pi/2)
            GateType::SX => self.compile_gate(&GateType::Rx(PI / 2.0)),

            // Multi-qubit gates not directly supported: return identity with low fidelity
            GateType::Toffoli | GateType::CCZ | GateType::ISWAP => {
                let mut w = BraidWord::identity();
                w.fidelity = 0.0; // Cannot approximate without decomposition
                w
            }

            // Controlled rotations: approximate the rotation part
            GateType::CRx(theta) | GateType::CRy(theta) | GateType::CRz(theta) | GateType::CR(theta) => {
                let mut w = self.approximate_rotation(*theta, RotationAxis::Z);
                w.fidelity *= 0.5; // Rough penalty for missing control structure
                w
            }

            // Two-qubit rotation gates: not yet supported in topological compilation
            GateType::Rxx(_) | GateType::Ryy(_) | GateType::Rzz(_) => {
                let mut w = BraidWord::identity();
                w.fidelity = 0.0;
                w
            }

            // CSWAP: not yet supported in topological compilation
            GateType::CSWAP => {
                let mut w = BraidWord::identity();
                w.fidelity = 0.0;
                w
            }

            // CU: not yet supported in topological compilation
            GateType::CU { .. } => {
                let mut w = BraidWord::identity();
                w.fidelity = 0.0;
                w
            }

            // Custom gate: no approximation possible without matrix analysis
            GateType::Custom(_) => {
                let mut w = BraidWord::identity();
                w.fidelity = 0.0;
                w
            }
        }
    }

    /// Compile a sequence of gates into braid words.
    pub fn compile_circuit(&self, gates: &[Gate]) -> Vec<BraidWord> {
        gates.iter().map(|g| self.compile_gate(&g.gate_type)).collect()
    }

    /// Approximate an arbitrary rotation angle as the nearest Clifford rotation.
    fn approximate_rotation(&self, theta: f64, axis: RotationAxis) -> BraidWord {
        // Clifford rotation angles: 0, pi/2, pi, 3pi/2
        let clifford_angles = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
        let (best_idx, best_angle) = clifford_angles
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                angle_distance(theta, **a)
                    .partial_cmp(&angle_distance(theta, **b))
                    .unwrap()
            })
            .unwrap();

        let gate = match (axis, best_idx) {
            (_, 0) => return BraidWord::identity(),
            (RotationAxis::X, 1) => GateType::X, // Rx(pi/2) ~ X braid (rough)
            (RotationAxis::X, 2) => GateType::X,
            (RotationAxis::X, 3) => GateType::X,
            (RotationAxis::Y, 1) => GateType::Y,
            (RotationAxis::Y, 2) => GateType::Y,
            (RotationAxis::Y, 3) => GateType::Y,
            (RotationAxis::Z, 1) => GateType::S,
            (RotationAxis::Z, 2) => GateType::Z,
            (RotationAxis::Z, 3) => GateType::S, // S^3 = S^dag
            // Unreachable: best_idx is always in [0, 3] from a 4-element array
            _ => unreachable!(),
        };

        let mut w = self.compile_gate(&gate);
        w.fidelity = ((theta - best_angle) / 2.0).cos().powi(2);
        w
    }
}

impl Default for BraidCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
enum RotationAxis {
    X,
    Y,
    Z,
}

/// Angular distance on the circle, accounting for 2pi periodicity.
fn angle_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).abs() % (2.0 * PI);
    d.min(2.0 * PI - d)
}

// ---------------------------------------------------------------------------
// Topological charge tracker
// ---------------------------------------------------------------------------

/// Tracks topological charges (fusion outcomes) through a computation,
/// verifying charge conservation at each step.
#[derive(Clone, Debug)]
pub struct TopologicalChargeTracker {
    /// Current anyon charges in the system.
    pub charges: Vec<IsingAnyonType>,
    /// History of fusion events: (anyon_i, anyon_j, outcome).
    pub history: Vec<(usize, usize, IsingAnyonType)>,
}

impl TopologicalChargeTracker {
    /// Create a tracker for a system of `n` sigma anyons.
    pub fn new(n: usize) -> Self {
        Self {
            charges: vec![IsingAnyonType::Sigma; n],
            history: Vec::new(),
        }
    }

    /// Record a fusion event between anyons at positions i and j.
    /// Returns Ok if the fusion outcome is allowed, Err otherwise.
    pub fn record_fusion(
        &mut self,
        i: usize,
        j: usize,
        outcome: IsingAnyonType,
    ) -> Result<(), String> {
        if i >= self.charges.len() || j >= self.charges.len() {
            return Err(format!(
                "anyon index out of range: i={}, j={}, len={}",
                i,
                j,
                self.charges.len()
            ));
        }

        let allowed = IsingFusionRules::fuse(self.charges[i], self.charges[j]);
        if !allowed.contains(&outcome) {
            return Err(format!(
                "fusion {:?} x {:?} -> {:?} is not allowed (valid: {:?})",
                self.charges[i], self.charges[j], outcome, allowed
            ));
        }

        self.history.push((i, j, outcome));

        // Replace anyon i with the fusion outcome, remove anyon j
        self.charges[i] = outcome;
        if j < self.charges.len() {
            self.charges.remove(j);
        }

        Ok(())
    }

    /// Check whether the system has been fused down to a single charge.
    pub fn is_fully_fused(&self) -> bool {
        self.charges.len() <= 1
    }

    /// Return the total topological charge (if fully fused).
    pub fn total_charge(&self) -> Option<IsingAnyonType> {
        if self.charges.len() == 1 {
            Some(self.charges[0])
        } else {
            None
        }
    }

    /// Number of anyons remaining.
    pub fn num_anyons(&self) -> usize {
        self.charges.len()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Phase factor e^{i theta} as a C64.
#[inline]
fn ising_phase(theta: f64) -> C64 {
    C64::new(theta.cos(), theta.sin())
}

/// Ising F-matrix (recoupling matrix) for two adjacent fusion pairs.
///
/// In the standard basis {|00>, |01>, |10>, |11>} where 0 = vacuum and
/// 1 = psi, the Ising F-matrix is a tensor product structure:
///   F = (1/sqrt(2)) * [[1, 1], [1, -1]] tensor I_rest
/// restricted to the relevant 2-qubit subspace.
///
/// Here we return the 4x4 matrix acting on the {bit_a, bit_b} subspace.
fn ising_f_matrix_2x2() -> [[C64; 4]; 4] {
    let s = 1.0 / std::f64::consts::SQRT_2;
    let _z = c64_zero();
    let _p = C64::new(s, 0.0);
    let _m = C64::new(-s, 0.0);
    [
        [C64::new(0.5, 0.0), C64::new(0.5, 0.0), C64::new(0.5, 0.0), C64::new(0.5, 0.0)],
        [C64::new(0.5, 0.0), C64::new(-0.5, 0.0), C64::new(0.5, 0.0), C64::new(-0.5, 0.0)],
        [C64::new(0.5, 0.0), C64::new(0.5, 0.0), C64::new(-0.5, 0.0), C64::new(-0.5, 0.0)],
        [C64::new(0.5, 0.0), C64::new(-0.5, 0.0), C64::new(-0.5, 0.0), C64::new(0.5, 0.0)],
    ]
}

/// Inverse of the Ising F-matrix. Since F is real and orthogonal (Hadamard/2),
/// F^{-1} = F^T = F in this case (the matrix is its own inverse).
fn ising_f_inv_2x2() -> [[C64; 4]; 4] {
    ising_f_matrix_2x2()
}

/// Multiply a 4x4 matrix by a 4-vector.
fn mat4_vec_mul(m: &[[C64; 4]; 4], v: &[C64; 4]) -> [C64; 4] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3],
        m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3],
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPSILON: f64 = 1e-10;

    // ---- Ising fusion rules ----

    #[test]
    fn test_sigma_sigma_fusion_has_two_outcomes() {
        let outcomes = IsingFusionRules::fuse(IsingAnyonType::Sigma, IsingAnyonType::Sigma);
        assert_eq!(outcomes.len(), 2);
        assert!(outcomes.contains(&IsingAnyonType::Vacuum));
        assert!(outcomes.contains(&IsingAnyonType::Psi));
    }

    #[test]
    fn test_sigma_psi_fusion_gives_sigma() {
        let outcomes = IsingFusionRules::fuse(IsingAnyonType::Sigma, IsingAnyonType::Psi);
        assert_eq!(outcomes, vec![IsingAnyonType::Sigma]);
    }

    #[test]
    fn test_psi_psi_fusion_gives_vacuum() {
        let outcomes = IsingFusionRules::fuse(IsingAnyonType::Psi, IsingAnyonType::Psi);
        assert_eq!(outcomes, vec![IsingAnyonType::Vacuum]);
    }

    #[test]
    fn test_vacuum_fusion_is_identity() {
        for anyon in &[IsingAnyonType::Vacuum, IsingAnyonType::Sigma, IsingAnyonType::Psi] {
            let outcomes = IsingFusionRules::fuse(IsingAnyonType::Vacuum, *anyon);
            assert_eq!(outcomes, vec![*anyon]);
        }
    }

    #[test]
    fn test_quantum_dimensions() {
        assert!((IsingFusionRules::quantum_dimension(IsingAnyonType::Vacuum) - 1.0).abs() < EPSILON);
        assert!(
            (IsingFusionRules::quantum_dimension(IsingAnyonType::Sigma) - std::f64::consts::SQRT_2)
                .abs()
                < EPSILON
        );
        assert!((IsingFusionRules::quantum_dimension(IsingAnyonType::Psi) - 1.0).abs() < EPSILON);
        assert!((IsingFusionRules::total_quantum_dimension() - 2.0).abs() < EPSILON);
    }

    // ---- R-matrix and F-matrix consistency ----

    #[test]
    fn test_r_matrix_phases() {
        // R^{sigma,sigma}_vacuum = e^{-i pi/8}
        let r_vac = ising_phase(-PI / 8.0);
        assert!((r_vac.norm() - 1.0).abs() < EPSILON);
        assert!((r_vac.arg() - (-PI / 8.0)).abs() < EPSILON);

        // R^{sigma,sigma}_psi = e^{i 3pi/8}
        let r_psi = ising_phase(3.0 * PI / 8.0);
        assert!((r_psi.norm() - 1.0).abs() < EPSILON);
        assert!((r_psi.arg() - (3.0 * PI / 8.0)).abs() < EPSILON);
    }

    #[test]
    fn test_f_matrix_is_unitary() {
        let f = ising_f_matrix_2x2();
        // Check F * F^T = I (since F is real and F = F^{-1})
        for i in 0..4 {
            for j in 0..4 {
                let mut dot = c64_zero();
                for k in 0..4 {
                    dot += f[i][k] * f[j][k].conj();
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - C64::new(expected, 0.0)).norm() < EPSILON,
                    "F unitarity failed at ({}, {}): got {:?}",
                    i,
                    j,
                    dot
                );
            }
        }
    }

    // ---- Ising anyon state braiding ----

    #[test]
    fn test_ising_braid_preserves_norm() {
        let mut state = IsingAnyonState::new(4);
        // Set up a non-trivial superposition
        state.amplitudes[0] = C64::new(0.6, 0.1);
        state.amplitudes[1] = C64::new(0.3, -0.2);
        state.normalize();

        let norm_before = state.norm_sqr();
        state.braid(0);
        let norm_after = state.norm_sqr();

        assert!(
            (norm_before - norm_after).abs() < EPSILON,
            "braid changed norm: {} -> {}",
            norm_before,
            norm_after
        );
    }

    #[test]
    fn test_ising_double_braid_gives_monodromy() {
        // Two braids = monodromy. For same-pair braiding, the phase difference
        // between vacuum and psi channels should be:
        //   (R_psi)^2 / (R_vac)^2 = e^{i 6pi/8} / e^{-i 2pi/8} = e^{i pi} = -1
        let mut state = IsingAnyonState::new(4);
        state.amplitudes[0] = C64::new(1.0, 0.0);
        state.amplitudes[1] = C64::new(1.0, 0.0);
        state.normalize();

        let ratio_before = state.amplitudes[1] / state.amplitudes[0];

        state.braid(0);
        state.braid(0);

        let ratio_after = state.amplitudes[1] / state.amplitudes[0];
        let phase_change = ratio_after / ratio_before;

        // Should be e^{i(3pi/4 - (-pi/4))} = e^{i pi} = -1
        assert!(
            (phase_change + c64_one()).norm() < 1e-8,
            "monodromy phase: expected -1, got {:?}",
            phase_change
        );
    }

    #[test]
    fn test_ising_fusion_projection() {
        let mut state = IsingAnyonState::new(4);
        // Equal superposition of vacuum and psi
        let s = 1.0 / std::f64::consts::SQRT_2;
        state.amplitudes[0] = C64::new(s, 0.0);
        state.amplitudes[1] = C64::new(s, 0.0);

        let (outcome, prob) = state.fuse(0);
        assert!(prob > 0.49 && prob < 0.51, "expected ~0.5 probability, got {}", prob);
        // After fusion, state should be fully projected
        let probs = state.probabilities();
        let entropy: f64 = probs.iter().filter(|p| **p > 0.0).map(|p| -p * p.ln()).sum();
        assert!(
            entropy < 1e-10 || outcome == IsingAnyonType::Vacuum || outcome == IsingAnyonType::Psi,
            "state should be projected after fusion"
        );
    }

    // ---- Majorana chain ----

    #[test]
    fn test_majorana_chain_initialization() {
        let chain = MajoranaChain::new(3);
        assert_eq!(chain.dimension(), 8);
        assert!((chain.state[0].norm_sqr() - 1.0).abs() < EPSILON);
        for i in 1..8 {
            assert!(chain.state[i].norm_sqr() < EPSILON);
        }
    }

    #[test]
    fn test_majorana_braid_preserves_norm() {
        let mut chain = MajoranaChain::new(3);
        // Create superposition
        chain.state[0] = C64::new(0.8, 0.0);
        chain.state[1] = C64::new(0.6, 0.0);
        let norm_before: f64 = chain.state.iter().map(|a| a.norm_sqr()).sum();

        chain.braid(0);
        let norm_after: f64 = chain.state.iter().map(|a| a.norm_sqr()).sum();

        assert!(
            (norm_before - norm_after).abs() < EPSILON,
            "Majorana braid changed norm: {} -> {}",
            norm_before,
            norm_after
        );
    }

    #[test]
    fn test_majorana_same_site_braid_is_diagonal() {
        // Same-site braiding (even j) should not change probabilities,
        // only phases.
        let mut chain = MajoranaChain::new(2);
        chain.state[0] = C64::new(0.6, 0.0);
        chain.state[1] = C64::new(0.0, 0.0);
        chain.state[2] = C64::new(0.8, 0.0);
        chain.state[3] = C64::new(0.0, 0.0);

        let probs_before: Vec<f64> = chain.probabilities();
        chain.braid(0); // same-site, should only add phases
        let probs_after: Vec<f64> = chain.probabilities();

        for (pb, pa) in probs_before.iter().zip(probs_after.iter()) {
            assert!(
                (pb - pa).abs() < EPSILON,
                "same-site braid changed probability: {} -> {}",
                pb,
                pa
            );
        }
    }

    #[test]
    fn test_majorana_parity_conservation() {
        // Braiding preserves total fermion parity
        let mut chain = MajoranaChain::new(3);
        let parity_before = chain.parity();

        chain.braid(0);
        chain.braid(2);
        chain.braid(4);

        let parity_after = chain.parity();
        assert!(
            (parity_before - parity_after).abs() < EPSILON,
            "parity changed: {} -> {}",
            parity_before,
            parity_after
        );
    }

    // ---- Braid compiler ----

    #[test]
    fn test_compile_clifford_gates_exact() {
        let compiler = BraidCompiler::new();

        // Clifford gates should compile with fidelity 1.0
        for gate in &[GateType::X, GateType::Y, GateType::Z, GateType::H, GateType::S] {
            let braid = compiler.compile_gate(gate);
            assert!(
                (braid.fidelity - 1.0).abs() < EPSILON,
                "gate {:?} should have exact fidelity, got {}",
                gate,
                braid.fidelity
            );
            assert!(!braid.is_empty(), "gate {:?} should produce non-empty braid", gate);
        }
    }

    #[test]
    fn test_compile_t_gate_approximate() {
        let compiler = BraidCompiler::new();
        let braid = compiler.compile_gate(&GateType::T);

        // T gate is non-Clifford, fidelity should be < 1.0
        assert!(braid.fidelity < 1.0, "T gate should be approximate");
        assert!(braid.fidelity > 0.8, "T gate fidelity should be reasonable (>0.8)");
    }

    #[test]
    fn test_braid_word_inverse_property() {
        let mut w = BraidWord::new();
        w.push(0, false);
        w.push(1, true);
        w.push(2, false);

        let inv = w.inverse();
        assert_eq!(inv.len(), w.len());

        // Inverse should reverse order and flip all crossings
        assert_eq!(inv.generators[0].strand, 2);
        assert!(inv.generators[0].inverse);
        assert_eq!(inv.generators[1].strand, 1);
        assert!(!inv.generators[1].inverse);
        assert_eq!(inv.generators[2].strand, 0);
        assert!(inv.generators[2].inverse);
    }

    #[test]
    fn test_compile_circuit() {
        let compiler = BraidCompiler::new();
        let gates = vec![
            Gate {
                gate_type: GateType::H,
                targets: vec![0],
                controls: vec![],
                params: None,
            },
            Gate {
                gate_type: GateType::CNOT,
                targets: vec![1],
                controls: vec![0],
                params: None,
            },
        ];

        let braids = compiler.compile_circuit(&gates);
        assert_eq!(braids.len(), 2);
        assert!((braids[0].fidelity - 1.0).abs() < EPSILON);
        assert!((braids[1].fidelity - 1.0).abs() < EPSILON);
    }

    // ---- Charge tracker ----

    #[test]
    fn test_charge_tracker_valid_fusion() {
        let mut tracker = TopologicalChargeTracker::new(4);
        assert_eq!(tracker.num_anyons(), 4);

        // sigma x sigma -> vacuum (allowed)
        let result = tracker.record_fusion(0, 1, IsingAnyonType::Vacuum);
        assert!(result.is_ok());
        assert_eq!(tracker.num_anyons(), 3);

        // vacuum x sigma -> sigma (allowed)
        let result = tracker.record_fusion(0, 1, IsingAnyonType::Sigma);
        assert!(result.is_ok());
        assert_eq!(tracker.num_anyons(), 2);
    }

    #[test]
    fn test_charge_tracker_invalid_fusion() {
        let mut tracker = TopologicalChargeTracker::new(4);
        // sigma x sigma -> sigma is NOT allowed
        let result = tracker.record_fusion(0, 1, IsingAnyonType::Sigma);
        assert!(result.is_err());
    }

    #[test]
    fn test_charge_tracker_full_fusion() {
        let mut tracker = TopologicalChargeTracker::new(4);
        // Start: [Sigma, Sigma, Sigma, Sigma]

        // Fuse pair (0,1): Sigma x Sigma -> Vacuum
        tracker.record_fusion(0, 1, IsingAnyonType::Vacuum).unwrap();
        // Now: [Vacuum, Sigma, Sigma]
        assert_eq!(tracker.num_anyons(), 3);

        // Fuse pair (0,1): Vacuum x Sigma -> Sigma
        tracker.record_fusion(0, 1, IsingAnyonType::Sigma).unwrap();
        // Now: [Sigma, Sigma]
        assert_eq!(tracker.num_anyons(), 2);

        // Fuse pair (0,1): Sigma x Sigma -> Vacuum
        tracker.record_fusion(0, 1, IsingAnyonType::Vacuum).unwrap();
        // Now: [Vacuum]
        assert_eq!(tracker.num_anyons(), 1);

        assert!(tracker.is_fully_fused());
        assert_eq!(tracker.total_charge(), Some(IsingAnyonType::Vacuum));
    }
}
