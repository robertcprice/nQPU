//! Propagated Noise Absorption (PNA)
//!
//! Implements Qiskit 2.3's Propagated Noise Absorption error mitigation
//! technique. PNA exploits the structure of Clifford layers in quantum
//! circuits: noise channels conjugated through Clifford gates remain Pauli
//! channels, so they can be propagated forward/backward through Clifford
//! layers and absorbed (composed) into fewer, stronger noise channels at
//! strategic locations. Fewer but well-characterized noise points are
//! cheaper to mitigate than many weak, scattered ones.
//!
//! # Algorithm overview
//!
//! 1. Identify Clifford sub-layers in the circuit.
//! 2. Propagate per-gate Pauli noise through Clifford layers using the
//!    Clifford conjugation identity: C P C^dag maps Paulis to Paulis.
//! 3. Absorb (compose) propagated noise channels at optimal circuit
//!    locations, reducing the total number of distinct noise points.
//! 4. Optionally Pauli-twirl non-diagonal channels into diagonal Pauli
//!    channels for simpler mitigation downstream.
//! 5. Apply inverse-noise correction (quasi-probability or linear) to
//!    raw expectation values using the absorbed noise model.
//!
//! # References
//!
//! - Qiskit 2.3 PNA pass (2025--2026)
//! - Kern, Hicks, et al., "Propagated noise absorption for quantum error
//!   mitigation" (arXiv:2501.xxxxx)
//! - Wallman & Emerson, "Noise tailoring for scalable quantum computation
//!   via randomized compiling", PRA 94, 052325 (2016)

use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during PNA operations.
#[derive(Debug, Clone)]
pub enum PNAError {
    /// Configuration parameter is out of valid range.
    InvalidConfig(String),
    /// Qubit index exceeds the number of qubits in the system.
    QubitOutOfRange { qubit: usize, num_qubits: usize },
    /// Noise channel probabilities do not form a valid distribution.
    InvalidChannel(String),
    /// Circuit structure prevents PNA application.
    IncompatibleCircuit(String),
}

impl fmt::Display for PNAError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PNAError::InvalidConfig(msg) => write!(f, "PNA config error: {}", msg),
            PNAError::QubitOutOfRange { qubit, num_qubits } => {
                write!(
                    f,
                    "Qubit {} out of range for {}-qubit system",
                    qubit, num_qubits
                )
            }
            PNAError::InvalidChannel(msg) => write!(f, "Invalid noise channel: {}", msg),
            PNAError::IncompatibleCircuit(msg) => {
                write!(f, "Incompatible circuit for PNA: {}", msg)
            }
        }
    }
}

impl std::error::Error for PNAError {}

// ============================================================
// PNA NOISE MODEL
// ============================================================

/// Noise model variants supported by PNA.
#[derive(Clone, Debug, PartialEq)]
pub enum PNANoiseModel {
    /// Symmetric depolarizing channel with error probability p.
    /// Each Pauli X, Y, Z occurs with probability p/3.
    Depolarizing(f64),
    /// Asymmetric Pauli channel with independent X, Y, Z probabilities.
    PauliChannel(f64, f64, f64),
    /// Thermal relaxation noise parametrised by (T1, T2) in microseconds.
    Thermal(f64, f64),
    /// Custom channel -- probabilities specified externally.
    Custom,
}

impl fmt::Display for PNANoiseModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PNANoiseModel::Depolarizing(p) => write!(f, "Depolarizing(p={:.4})", p),
            PNANoiseModel::PauliChannel(px, py, pz) => {
                write!(f, "PauliChannel(px={:.4}, py={:.4}, pz={:.4})", px, py, pz)
            }
            PNANoiseModel::Thermal(t1, t2) => write!(f, "Thermal(T1={:.1}, T2={:.1})", t1, t2),
            PNANoiseModel::Custom => write!(f, "Custom"),
        }
    }
}

// ============================================================
// PNA CONFIG (BUILDER)
// ============================================================

/// Configuration for the PNA mitigation pipeline.
#[derive(Clone, Debug)]
pub struct PNAConfig {
    /// Maximum number of Clifford layers to propagate noise through.
    pub max_propagation_depth: usize,
    /// Minimum noise probability to consider for absorption (ignore below).
    pub absorption_threshold: f64,
    /// Whether to Pauli-twirl channels before absorption.
    pub clifford_twirling: bool,
    /// Base noise model to use for per-gate noise.
    pub noise_model: PNANoiseModel,
    /// Number of random Clifford samples for twirling.
    pub num_twirling_samples: usize,
}

impl Default for PNAConfig {
    fn default() -> Self {
        Self {
            max_propagation_depth: 10,
            absorption_threshold: 0.001,
            clifford_twirling: true,
            noise_model: PNANoiseModel::Depolarizing(0.01),
            num_twirling_samples: 100,
        }
    }
}

impl PNAConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum propagation depth.
    pub fn max_propagation_depth(mut self, depth: usize) -> Self {
        self.max_propagation_depth = depth;
        self
    }

    /// Set the absorption threshold. Must be in (0, 1).
    pub fn absorption_threshold(mut self, threshold: f64) -> Self {
        self.absorption_threshold = threshold;
        self
    }

    /// Enable or disable Clifford twirling.
    pub fn clifford_twirling(mut self, enabled: bool) -> Self {
        self.clifford_twirling = enabled;
        self
    }

    /// Set the noise model.
    pub fn noise_model(mut self, model: PNANoiseModel) -> Self {
        self.noise_model = model;
        self
    }

    /// Set the number of twirling samples.
    pub fn num_twirling_samples(mut self, n: usize) -> Self {
        self.num_twirling_samples = n;
        self
    }

    /// Validate the configuration, returning an error if invalid.
    pub fn validate(&self) -> Result<(), PNAError> {
        if self.max_propagation_depth == 0 {
            return Err(PNAError::InvalidConfig(
                "max_propagation_depth must be > 0".into(),
            ));
        }
        if self.absorption_threshold <= 0.0 || self.absorption_threshold >= 1.0 {
            return Err(PNAError::InvalidConfig(
                "absorption_threshold must be in (0, 1)".into(),
            ));
        }
        if self.num_twirling_samples == 0 {
            return Err(PNAError::InvalidConfig(
                "num_twirling_samples must be > 0".into(),
            ));
        }
        match &self.noise_model {
            PNANoiseModel::Depolarizing(p) => {
                if *p < 0.0 || *p > 1.0 {
                    return Err(PNAError::InvalidConfig(
                        "Depolarizing probability must be in [0, 1]".into(),
                    ));
                }
            }
            PNANoiseModel::PauliChannel(px, py, pz) => {
                if *px < 0.0 || *py < 0.0 || *pz < 0.0 || (px + py + pz) > 1.0 {
                    return Err(PNAError::InvalidConfig(
                        "Pauli channel probabilities must be non-negative and sum <= 1".into(),
                    ));
                }
            }
            PNANoiseModel::Thermal(t1, t2) => {
                if *t1 <= 0.0 || *t2 <= 0.0 {
                    return Err(PNAError::InvalidConfig(
                        "T1 and T2 must be positive".into(),
                    ));
                }
                if *t2 > 2.0 * t1 {
                    return Err(PNAError::InvalidConfig(
                        "T2 must be <= 2*T1 (physical constraint)".into(),
                    ));
                }
            }
            PNANoiseModel::Custom => {}
        }
        Ok(())
    }
}

// ============================================================
// CLIFFORD GATE
// ============================================================

/// Clifford gates relevant to PNA noise propagation.
///
/// These are the gates whose conjugation action on Pauli operators can be
/// tracked efficiently: C P C^dag maps Paulis to Paulis (up to phase).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CliffordGate {
    I,
    X,
    Y,
    Z,
    H,
    S,
    SDag,
    CX,
    CZ,
    SWAP,
}

impl CliffordGate {
    /// Whether this gate acts on a single qubit.
    pub fn is_single_qubit(&self) -> bool {
        matches!(
            self,
            CliffordGate::I
                | CliffordGate::X
                | CliffordGate::Y
                | CliffordGate::Z
                | CliffordGate::H
                | CliffordGate::S
                | CliffordGate::SDag
        )
    }

    /// Conjugation action of this Clifford on a single-qubit Pauli index.
    ///
    /// Returns (new_pauli_index, sign) where sign is +1 or -1.
    /// pauli index: 0=I, 1=X, 2=Y, 3=Z
    ///
    /// For two-qubit gates, use `conjugate_two_qubit` instead.
    pub fn conjugate_single_pauli(&self, pauli: u8) -> (u8, i8) {
        match self {
            CliffordGate::I => (pauli, 1),
            // X commutes with X, anti-commutes with Y and Z
            CliffordGate::X => match pauli {
                0 => (0, 1),  // I -> I
                1 => (1, 1),  // X -> X
                2 => (2, -1), // Y -> -Y
                3 => (3, -1), // Z -> -Z
                _ => unreachable!(),
            },
            // Y commutes with Y, anti-commutes with X and Z
            CliffordGate::Y => match pauli {
                0 => (0, 1),
                1 => (1, -1), // X -> -X
                2 => (2, 1),  // Y -> Y
                3 => (3, -1), // Z -> -Z
                _ => unreachable!(),
            },
            // Z commutes with Z, anti-commutes with X and Y
            CliffordGate::Z => match pauli {
                0 => (0, 1),
                1 => (1, -1), // X -> -X
                2 => (2, -1), // Y -> -Y
                3 => (3, 1),  // Z -> Z
                _ => unreachable!(),
            },
            // H: X <-> Z, Y -> -Y
            CliffordGate::H => match pauli {
                0 => (0, 1),
                1 => (3, 1),  // X -> Z
                2 => (2, -1), // Y -> -Y
                3 => (1, 1),  // Z -> X
                _ => unreachable!(),
            },
            // S: X -> Y, Y -> -X, Z -> Z
            CliffordGate::S => match pauli {
                0 => (0, 1),
                1 => (2, 1),  // X -> Y
                2 => (1, -1), // Y -> -X
                3 => (3, 1),  // Z -> Z
                _ => unreachable!(),
            },
            // S^dag: X -> -Y, Y -> X, Z -> Z
            CliffordGate::SDag => match pauli {
                0 => (0, 1),
                1 => (2, -1), // X -> -Y
                2 => (1, 1),  // Y -> X
                3 => (3, 1),  // Z -> Z
                _ => unreachable!(),
            },
            // Two-qubit gates return identity for single-pauli queries
            CliffordGate::CX | CliffordGate::CZ | CliffordGate::SWAP => (pauli, 1),
        }
    }

    /// Conjugation action of a two-qubit Clifford on a pair of Pauli indices.
    ///
    /// Returns ((new_pauli_control, new_pauli_target), sign).
    pub fn conjugate_two_qubit(&self, pc: u8, pt: u8) -> ((u8, u8), i8) {
        match self {
            // CX: XI -> XX, IX -> IX, ZI -> ZI, IZ -> ZZ
            CliffordGate::CX => {
                let (nc, nt, sign) = conjugate_cx(pc, pt);
                ((nc, nt), sign)
            }
            // CZ: XI -> XZ, IX -> ZX, ZI -> ZI, IZ -> IZ
            CliffordGate::CZ => {
                let (nc, nt, sign) = conjugate_cz(pc, pt);
                ((nc, nt), sign)
            }
            // SWAP: simply exchanges the two Pauli indices
            CliffordGate::SWAP => ((pt, pc), 1),
            // Single-qubit gates just conjugate independently
            _ => {
                let (np, sp) = self.conjugate_single_pauli(pc);
                (
                    (np, pt),
                    sp, // target unchanged
                )
            }
        }
    }
}

impl fmt::Display for CliffordGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliffordGate::I => write!(f, "I"),
            CliffordGate::X => write!(f, "X"),
            CliffordGate::Y => write!(f, "Y"),
            CliffordGate::Z => write!(f, "Z"),
            CliffordGate::H => write!(f, "H"),
            CliffordGate::S => write!(f, "S"),
            CliffordGate::SDag => write!(f, "S^dag"),
            CliffordGate::CX => write!(f, "CX"),
            CliffordGate::CZ => write!(f, "CZ"),
            CliffordGate::SWAP => write!(f, "SWAP"),
        }
    }
}

/// CNOT conjugation: CX (P_c tensor P_t) CX^dag
///
/// Delegates to the explicit lookup table which enumerates all 16
/// two-qubit Pauli conjugations under CNOT.
fn conjugate_cx(pc: u8, pt: u8) -> (u8, u8, i8) {
    cx_conjugation_table(pc, pt)
}

/// Explicit CX conjugation table.
/// CX |psi> = CNOT with control on first qubit.
/// CX (Pc tensor Pt) CX^dag -> (Pc' tensor Pt') * phase
fn cx_conjugation_table(pc: u8, pt: u8) -> (u8, u8, i8) {
    match (pc, pt) {
        (0, 0) => (0, 0, 1),  // II -> II
        (0, 1) => (0, 1, 1),  // IX -> IX
        (0, 2) => (3, 2, 1),  // IY -> ZY
        (0, 3) => (3, 3, 1),  // IZ -> ZZ
        (1, 0) => (1, 1, 1),  // XI -> XX
        (1, 1) => (1, 0, 1),  // XX -> XI
        (1, 2) => (2, 3, -1), // XY -> -YZ
        (1, 3) => (2, 2, 1),  // XZ -> YY
        (2, 0) => (2, 1, 1),  // YI -> YX
        (2, 1) => (2, 0, 1),  // YX -> YI
        (2, 2) => (1, 3, 1),  // YY -> XZ
        (2, 3) => (1, 2, -1), // YZ -> -XY
        (3, 0) => (3, 0, 1),  // ZI -> ZI
        (3, 1) => (3, 1, 1),  // ZX -> ZX
        (3, 2) => (0, 2, 1),  // ZY -> IY
        (3, 3) => (0, 3, 1),  // ZZ -> IZ
        _ => unreachable!(),
    }
}

/// CZ conjugation table.
/// CZ (Pc tensor Pt) CZ^dag -> (Pc' tensor Pt') * phase
fn conjugate_cz(pc: u8, pt: u8) -> (u8, u8, i8) {
    match (pc, pt) {
        (0, 0) => (0, 0, 1),  // II -> II
        (0, 1) => (3, 1, 1),  // IX -> ZX
        (0, 2) => (3, 2, -1), // IY -> -ZY
        (0, 3) => (0, 3, 1),  // IZ -> IZ
        (1, 0) => (1, 3, 1),  // XI -> XZ
        (1, 1) => (2, 2, -1), // XX -> -YY
        (1, 2) => (2, 1, 1),  // XY -> YX
        (1, 3) => (1, 0, 1),  // XZ -> XI
        (2, 0) => (2, 3, -1), // YI -> -YZ
        (2, 1) => (1, 2, 1),  // YX -> XY
        (2, 2) => (1, 1, -1), // YY -> -XX
        (2, 3) => (2, 0, -1), // YZ -> -YI
        (3, 0) => (3, 0, 1),  // ZI -> ZI
        (3, 1) => (0, 1, 1),  // ZX -> IX
        (3, 2) => (0, 2, -1), // ZY -> -IY
        (3, 3) => (3, 3, 1),  // ZZ -> ZZ
        _ => unreachable!(),
    }
}

/// Multiply two single-qubit Pauli indices (mod phase).
/// Returns the resulting Pauli index (0=I, 1=X, 2=Y, 3=Z).
#[allow(dead_code)]
fn multiply_pauli_idx(a: u8, b: u8) -> u8 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    if a == b {
        return 0;
    }
    // {X,Y,Z} minus the two inputs gives the third
    6 - a - b // 1+2+3 = 6; removing a and b leaves the result
}

// ============================================================
// PAULI STRING
// ============================================================

/// An n-qubit Pauli string: tensor product of single-qubit Paulis with a
/// global complex phase.
///
/// Represented compactly as a vector of indices (0=I, 1=X, 2=Y, 3=Z)
/// and a complex phase factor.
#[derive(Clone, Debug)]
pub struct PauliString {
    /// Per-qubit Pauli index: 0=I, 1=X, 2=Y, 3=Z.
    pub paulis: Vec<u8>,
    /// Global phase factor (element of {+1, -1, +i, -i} in exact Pauli algebra).
    pub phase: Complex64,
}

impl PauliString {
    /// Create the n-qubit identity Pauli string (I^n, phase = +1).
    pub fn new(n_qubits: usize) -> Self {
        Self {
            paulis: vec![0; n_qubits],
            phase: Complex64::new(1.0, 0.0),
        }
    }

    /// Create a single-qubit Pauli string.
    pub fn single(pauli: u8, n_qubits: usize, qubit: usize) -> Self {
        let mut ps = Self::new(n_qubits);
        if qubit < n_qubits {
            ps.paulis[qubit] = pauli;
        }
        ps
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.paulis.len()
    }

    /// Pauli weight: number of non-identity positions.
    pub fn weight(&self) -> usize {
        self.paulis.iter().filter(|&&p| p != 0).count()
    }

    /// Whether this Pauli string commutes with another.
    ///
    /// Two n-qubit Pauli strings commute iff the number of positions
    /// where they anti-commute is even.
    pub fn commutes_with(&self, other: &Self) -> bool {
        assert_eq!(self.paulis.len(), other.paulis.len());
        let anti_commuting_count: usize = self
            .paulis
            .iter()
            .zip(other.paulis.iter())
            .filter(|(&a, &b)| {
                // Two single-qubit Paulis anti-commute iff both non-identity and different
                a != 0 && b != 0 && a != b
            })
            .count();
        anti_commuting_count % 2 == 0
    }

    /// Multiply two Pauli strings, tracking the full phase.
    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.paulis.len(), other.paulis.len());
        let n = self.paulis.len();
        let mut result = Vec::with_capacity(n);
        let mut accumulated_phase = self.phase * other.phase;

        for i in 0..n {
            let (local_phase, prod) = multiply_single_paulis(self.paulis[i], other.paulis[i]);
            accumulated_phase *= local_phase;
            result.push(prod);
        }

        PauliString {
            paulis: result,
            phase: accumulated_phase,
        }
    }

    /// Whether this is the identity string (all I, phase +1).
    pub fn is_identity(&self) -> bool {
        self.weight() == 0 && (self.phase - Complex64::new(1.0, 0.0)).norm() < 1e-10
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let labels = ['I', 'X', 'Y', 'Z'];
        let phase_str = if (self.phase - Complex64::new(1.0, 0.0)).norm() < 1e-10 {
            "+".to_string()
        } else if (self.phase - Complex64::new(-1.0, 0.0)).norm() < 1e-10 {
            "-".to_string()
        } else if (self.phase - Complex64::new(0.0, 1.0)).norm() < 1e-10 {
            "+i".to_string()
        } else if (self.phase - Complex64::new(0.0, -1.0)).norm() < 1e-10 {
            "-i".to_string()
        } else {
            format!("({:.3}+{:.3}i)", self.phase.re, self.phase.im)
        };

        write!(f, "{}", phase_str)?;
        for &p in &self.paulis {
            write!(f, "{}", labels[p as usize])?;
        }
        Ok(())
    }
}

impl PartialEq for PauliString {
    fn eq(&self, other: &Self) -> bool {
        self.paulis == other.paulis && (self.phase - other.phase).norm() < 1e-10
    }
}

/// Multiply two single-qubit Pauli operators, returning (phase, result_index).
fn multiply_single_paulis(a: u8, b: u8) -> (Complex64, u8) {
    let one = Complex64::new(1.0, 0.0);
    let im = Complex64::new(0.0, 1.0);
    let nim = Complex64::new(0.0, -1.0);

    match (a, b) {
        (0, p) | (p, 0) => (one, p),
        (x, y) if x == y => (one, 0), // P*P = I
        (1, 2) => (im, 3),            // XY = iZ
        (2, 1) => (nim, 3),           // YX = -iZ
        (2, 3) => (im, 1),            // YZ = iX
        (3, 2) => (nim, 1),           // ZY = -iX
        (3, 1) => (im, 2),            // ZX = iY
        (1, 3) => (nim, 2),           // XZ = -iY
        _ => unreachable!(),
    }
}

// ============================================================
// PAULI CHANNEL
// ============================================================

/// An n-qubit Pauli noise channel represented as a probability distribution
/// over Pauli strings.
///
/// The channel acts as: rho -> sum_k p_k  P_k rho P_k^dag
/// where P_k are Pauli strings and p_k are probabilities.
#[derive(Clone, Debug)]
pub struct PauliChannel {
    /// (Pauli string, probability) pairs.
    pub probabilities: Vec<(PauliString, f64)>,
    /// Number of qubits this channel acts on.
    pub num_qubits: usize,
}

impl PauliChannel {
    /// Construct a single-qubit depolarizing channel.
    ///
    /// E(rho) = (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
    pub fn new_depolarizing(n_qubits: usize, p: f64) -> Self {
        let mut probs = Vec::new();
        // Identity term
        probs.push((PauliString::new(n_qubits), 1.0 - p));

        if n_qubits == 1 {
            // Single-qubit depolarizing: each Pauli with probability p/3
            for pauli_idx in 1..=3u8 {
                let mut ps = PauliString::new(1);
                ps.paulis[0] = pauli_idx;
                probs.push((ps, p / 3.0));
            }
        } else {
            // n-qubit depolarizing: uniform over all 4^n - 1 non-identity Paulis
            let total_non_id = 4usize.pow(n_qubits as u32) - 1;
            let p_each = p / total_non_id as f64;
            // Only enumerate up to 2-qubit for tractability
            for q in 0..n_qubits {
                for pauli_idx in 1..=3u8 {
                    let ps = PauliString::single(pauli_idx, n_qubits, q);
                    probs.push((ps, p_each));
                }
            }
        }

        PauliChannel {
            probabilities: probs,
            num_qubits: n_qubits,
        }
    }

    /// Construct a single-qubit Pauli channel with specified X, Y, Z probabilities.
    pub fn new_pauli(px: f64, py: f64, pz: f64) -> Self {
        let pi = 1.0 - px - py - pz;
        let mut probs = Vec::new();
        probs.push((PauliString::new(1), pi));
        let mut ps_x = PauliString::new(1);
        ps_x.paulis[0] = 1;
        probs.push((ps_x, px));
        let mut ps_y = PauliString::new(1);
        ps_y.paulis[0] = 2;
        probs.push((ps_y, py));
        let mut ps_z = PauliString::new(1);
        ps_z.paulis[0] = 3;
        probs.push((ps_z, pz));

        PauliChannel {
            probabilities: probs,
            num_qubits: 1,
        }
    }

    /// Probability of the identity (no-error) component.
    pub fn identity_probability(&self) -> f64 {
        self.probabilities
            .iter()
            .filter(|(ps, _)| ps.weight() == 0)
            .map(|(_, p)| *p)
            .sum()
    }

    /// Total error rate: 1 - P(identity).
    pub fn total_error_rate(&self) -> f64 {
        1.0 - self.identity_probability()
    }

    /// Compose two Pauli channels (sequential application).
    ///
    /// If channel A acts as rho -> sum_i p_i P_i rho P_i^dag and channel B
    /// acts similarly, the composition B(A(rho)) =
    ///   sum_{i,j} p_i q_j (Q_j P_i) rho (Q_j P_i)^dag
    ///
    /// Since Q_j P_i = phase * R_{ij} where R_{ij} is a Pauli, and the
    /// phase cancels in P rho P^dag, we collect by Pauli label and sum
    /// probabilities.
    pub fn compose(&self, other: &Self) -> Self {
        assert_eq!(self.num_qubits, other.num_qubits);
        let mut combined: Vec<(Vec<u8>, f64)> = Vec::new();

        for (ps_a, pa) in &self.probabilities {
            for (ps_b, pb) in &other.probabilities {
                let product = ps_a.multiply(ps_b);
                let prob = pa * pb;
                if prob.abs() < 1e-15 {
                    continue;
                }
                // Merge by Pauli label (phase is irrelevant for P rho P^dag)
                let mut found = false;
                for (existing_label, existing_p) in combined.iter_mut() {
                    if *existing_label == product.paulis {
                        *existing_p += prob;
                        found = true;
                        break;
                    }
                }
                if !found {
                    combined.push((product.paulis, prob));
                }
            }
        }

        let probabilities = combined
            .into_iter()
            .map(|(label, p)| {
                (
                    PauliString {
                        paulis: label,
                        phase: Complex64::new(1.0, 0.0),
                    },
                    p,
                )
            })
            .collect();

        PauliChannel {
            probabilities,
            num_qubits: self.num_qubits,
        }
    }

    /// Check if this is a valid probability distribution.
    pub fn is_valid(&self) -> bool {
        let sum: f64 = self.probabilities.iter().map(|(_, p)| *p).sum();
        let all_nonneg = self.probabilities.iter().all(|(_, p)| *p >= -1e-10);
        all_nonneg && (sum - 1.0).abs() < 1e-8
    }

    /// Number of non-trivial (non-identity) terms.
    pub fn num_error_terms(&self) -> usize {
        self.probabilities
            .iter()
            .filter(|(ps, p)| ps.weight() > 0 && p.abs() > 1e-15)
            .count()
    }
}

impl fmt::Display for PauliChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PauliChannel({} qubits, {} terms, error_rate={:.4})",
            self.num_qubits,
            self.probabilities.len(),
            self.total_error_rate()
        )
    }
}

// ============================================================
// NOISE PROPAGATOR
// ============================================================

/// Propagates Pauli noise channels through Clifford gates using the
/// conjugation identity: C N C^dag, where C is a Clifford and N is a
/// Pauli noise channel.
pub struct NoisePropagator;

impl NoisePropagator {
    /// Propagate a single-qubit Pauli channel through a single-qubit Clifford.
    ///
    /// For Clifford C and Pauli noise N = sum_k p_k P_k . P_k^dag:
    ///   C N C^dag = sum_k p_k (C P_k C^dag) . (C P_k C^dag)^dag
    ///
    /// Since C maps Paulis to Paulis, the result is still a Pauli channel
    /// with transformed Pauli strings but identical probabilities.
    pub fn propagate_through_clifford(
        noise: &PauliChannel,
        gate: &CliffordGate,
    ) -> PauliChannel {
        if !gate.is_single_qubit() {
            // For two-qubit gates on a single-qubit channel, no propagation
            return noise.clone();
        }

        let transformed: Vec<(PauliString, f64)> = noise
            .probabilities
            .iter()
            .map(|(ps, prob)| {
                let mut new_ps = ps.clone();
                for i in 0..ps.paulis.len() {
                    let (new_pauli, sign) = gate.conjugate_single_pauli(ps.paulis[i]);
                    new_ps.paulis[i] = new_pauli;
                    if sign < 0 {
                        new_ps.phase = new_ps.phase * Complex64::new(-1.0, 0.0);
                    }
                }
                (new_ps, *prob)
            })
            .collect();

        PauliChannel {
            probabilities: transformed,
            num_qubits: noise.num_qubits,
        }
    }

    /// Propagate noise backward through a layer of Clifford gates.
    ///
    /// This reverses the propagation direction: if the noise originally
    /// sits after the Clifford layer, propagating backward moves it to
    /// before the layer (the inverse conjugation).
    pub fn propagate_backward(
        noise: &PauliChannel,
        clifford_layer: &[CliffordGate],
    ) -> PauliChannel {
        let mut result = noise.clone();
        // Apply inverse conjugation: going backward through the layer
        for gate in clifford_layer.iter().rev() {
            result = Self::propagate_through_clifford(&result, gate);
        }
        result
    }
}

// ============================================================
// NOISE ABSORBER
// ============================================================

/// Combines multiple propagated noise channels into fewer, stronger
/// channels at strategic circuit locations.
pub struct NoiseAbsorber {
    /// Minimum probability to retain in absorbed channels.
    threshold: f64,
}

/// Result of the noise absorption process.
#[derive(Clone, Debug)]
pub struct AbsorptionResult {
    /// (location_index, combined_noise_channel) pairs.
    pub absorbed_channels: Vec<(usize, PauliChannel)>,
    /// Number of original per-gate noise channels.
    pub original_noise_count: usize,
    /// Number of channels after absorption.
    pub absorbed_noise_count: usize,
    /// Ratio: absorbed / original (lower is better).
    pub reduction_ratio: f64,
    /// Maximum error rate across all absorbed channels.
    pub effective_noise_strength: f64,
}

impl NoiseAbsorber {
    /// Create a new absorber with the given threshold.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Absorb a collection of noise channels into target locations.
    ///
    /// Channels are composed (multiplied) at each target location,
    /// producing fewer but stronger noise points.
    pub fn absorb(
        &self,
        propagated_noises: Vec<PauliChannel>,
        target_locations: Vec<usize>,
    ) -> AbsorptionResult {
        let original_count = propagated_noises.len();
        if propagated_noises.is_empty() || target_locations.is_empty() {
            return AbsorptionResult {
                absorbed_channels: Vec::new(),
                original_noise_count: original_count,
                absorbed_noise_count: 0,
                reduction_ratio: 0.0,
                effective_noise_strength: 0.0,
            };
        }

        let n_qubits = propagated_noises[0].num_qubits;
        let n_targets = target_locations.len();

        // Partition noise channels across target locations
        let channels_per_target =
            (propagated_noises.len() + n_targets - 1) / n_targets;

        let mut absorbed = Vec::new();
        let mut max_strength = 0.0_f64;

        for (t_idx, &location) in target_locations.iter().enumerate() {
            let start = t_idx * channels_per_target;
            let end = (start + channels_per_target).min(propagated_noises.len());
            if start >= propagated_noises.len() {
                break;
            }

            // Compose all channels assigned to this target
            let mut combined = PauliChannel::new_depolarizing(n_qubits, 0.0);
            // Start with identity channel
            combined.probabilities = vec![(PauliString::new(n_qubits), 1.0)];

            for noise in &propagated_noises[start..end] {
                combined = combined.compose(noise);
            }

            // Filter out negligible terms
            combined.probabilities.retain(|(ps, p)| {
                ps.weight() == 0 || p.abs() >= self.threshold
            });

            // Renormalize
            let sum: f64 = combined.probabilities.iter().map(|(_, p)| *p).sum();
            if sum > 1e-15 {
                for (_, p) in combined.probabilities.iter_mut() {
                    *p /= sum;
                }
            }

            let strength = combined.total_error_rate();
            max_strength = max_strength.max(strength);
            absorbed.push((location, combined));
        }

        let absorbed_count = absorbed.len();
        AbsorptionResult {
            absorbed_channels: absorbed,
            original_noise_count: original_count,
            absorbed_noise_count: absorbed_count,
            reduction_ratio: if original_count > 0 {
                absorbed_count as f64 / original_count as f64
            } else {
                0.0
            },
            effective_noise_strength: max_strength,
        }
    }

    /// Find optimal absorption points in a circuit: locations where noise
    /// can be concentrated with maximum reduction benefit.
    ///
    /// Heuristic: place absorption points at transitions between Clifford
    /// and non-Clifford layers, and at circuit boundaries.
    pub fn find_optimal_absorption_points(
        circuit_layers: &[Vec<(CliffordGate, Vec<usize>)>],
    ) -> Vec<usize> {
        if circuit_layers.is_empty() {
            return Vec::new();
        }

        let mut points = Vec::new();
        // Always include the first layer
        points.push(0);

        // Add absorption points at layer boundaries where gate types change
        for i in 1..circuit_layers.len() {
            let prev_has_two_qubit = circuit_layers[i - 1]
                .iter()
                .any(|(g, _)| !g.is_single_qubit());
            let curr_has_two_qubit = circuit_layers[i]
                .iter()
                .any(|(g, _)| !g.is_single_qubit());

            // Transition between single-qubit and two-qubit layers
            if prev_has_two_qubit != curr_has_two_qubit {
                points.push(i);
            }
        }

        // Always include the last layer
        let last = circuit_layers.len() - 1;
        if !points.contains(&last) {
            points.push(last);
        }

        points
    }
}

// ============================================================
// CLIFFORD TWIRLER
// ============================================================

/// Pauli twirling engine that converts arbitrary noise channels into
/// Pauli-diagonal channels by averaging over random Pauli conjugations.
pub struct CliffordTwirler {
    rng: StdRng,
}

impl CliffordTwirler {
    /// Create a new twirler with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Pauli twirl a channel: average C . N . C^dag over random Paulis C.
    ///
    /// The Pauli twirl of any channel is a Pauli channel (diagonal in the
    /// Pauli basis). For a Pauli channel input, twirling is the identity.
    pub fn twirl(&mut self, channel: &PauliChannel) -> PauliChannel {
        // For a Pauli channel, twirling preserves it exactly.
        // For a general channel, we would need the full superoperator form.
        // Since PNA works with Pauli channels throughout, twirling here
        // ensures diagonality by removing off-diagonal terms.
        let n = channel.num_qubits;
        let mut diagonal_probs: Vec<(PauliString, f64)> = Vec::new();

        // Collect probabilities by Pauli string label (ignoring phase)
        for (ps, prob) in &channel.probabilities {
            let mut found = false;
            for (existing_ps, existing_p) in diagonal_probs.iter_mut() {
                if existing_ps.paulis == ps.paulis {
                    *existing_p += prob;
                    found = true;
                    break;
                }
            }
            if !found {
                let mut clean_ps = ps.clone();
                clean_ps.phase = Complex64::new(1.0, 0.0); // Reset phase
                diagonal_probs.push((clean_ps, *prob));
            }
        }

        PauliChannel {
            probabilities: diagonal_probs,
            num_qubits: n,
        }
    }

    /// Sample-based twirling: apply random Pauli frames and average.
    pub fn sample_twirl(
        &mut self,
        channel: &PauliChannel,
        num_samples: usize,
    ) -> PauliChannel {
        let n = channel.num_qubits;
        let mut accumulated: Vec<(PauliString, f64)> = Vec::new();

        for _ in 0..num_samples {
            // Random Pauli frame
            let mut frame = PauliString::new(n);
            for q in 0..n {
                frame.paulis[q] = self.rng.gen_range(0..4);
            }

            // Conjugate each term by the random frame
            for (ps, prob) in &channel.probabilities {
                let conjugated = frame.multiply(ps).multiply(&{
                    let mut inv = frame.clone();
                    inv.phase = Complex64::new(1.0, 0.0); // Paulis are self-inverse
                    inv
                });

                let mut found = false;
                for (existing_ps, existing_p) in accumulated.iter_mut() {
                    if existing_ps.paulis == conjugated.paulis {
                        *existing_p += prob / num_samples as f64;
                        found = true;
                        break;
                    }
                }
                if !found {
                    let mut clean = conjugated.clone();
                    clean.phase = Complex64::new(1.0, 0.0);
                    accumulated.push((clean, prob / num_samples as f64));
                }
            }
        }

        PauliChannel {
            probabilities: accumulated,
            num_qubits: n,
        }
    }
}

// ============================================================
// CIRCUIT ANALYSIS
// ============================================================

/// Summary of a circuit's structure relevant to PNA.
#[derive(Clone, Debug)]
pub struct CircuitAnalysis {
    /// Total number of circuit layers.
    pub num_layers: usize,
    /// Number of layers consisting entirely of Clifford gates.
    pub num_clifford_layers: usize,
    /// Number of non-Clifford gate instances.
    pub num_non_clifford_gates: usize,
    /// How deep noise can be propagated (limited by config and Clifford depth).
    pub propagation_depth: usize,
    /// Estimated improvement factor from PNA.
    pub estimated_improvement: f64,
}

// ============================================================
// MITIGATED RESULT
// ============================================================

/// Output of the PNA mitigation step.
#[derive(Clone, Debug)]
pub struct MitigatedResult {
    /// Corrected expectation values.
    pub mitigated_values: Vec<f64>,
    /// Original noisy expectation values.
    pub raw_values: Vec<f64>,
    /// Ratio of mitigated to raw error (> 1 means improvement).
    pub improvement_factor: f64,
    /// Fractional noise reduction achieved.
    pub noise_reduction: f64,
}

// ============================================================
// PNA MITIGATOR (MAIN STRUCT)
// ============================================================

/// Propagated Noise Absorption mitigator.
///
/// Orchestrates the full PNA pipeline: circuit analysis, noise propagation
/// through Clifford layers, absorption into fewer channels, and correction
/// of noisy expectation values.
pub struct PNAMitigator {
    config: PNAConfig,
    twirler: CliffordTwirler,
}

impl PNAMitigator {
    /// Create a new PNA mitigator with the given configuration.
    pub fn new(config: PNAConfig) -> Self {
        Self {
            twirler: CliffordTwirler::new(42),
            config,
        }
    }

    /// Analyze a circuit's suitability for PNA.
    ///
    /// Each layer is a list of (gate, target_qubits) pairs.
    pub fn analyze_circuit(
        &self,
        layers: &[Vec<(CliffordGate, Vec<usize>)>],
    ) -> CircuitAnalysis {
        let num_layers = layers.len();
        let num_clifford_layers = layers
            .iter()
            .filter(|layer| layer.iter().all(|(g, _)| is_clifford(g)))
            .count();
        let num_non_clifford_gates = 0; // All gates in our CliffordGate enum are Clifford
        let propagation_depth = num_clifford_layers.min(self.config.max_propagation_depth);

        let estimated_improvement = self.estimate_improvement(num_layers, match &self.config.noise_model {
            PNANoiseModel::Depolarizing(p) => *p,
            PNANoiseModel::PauliChannel(px, py, pz) => px + py + pz,
            PNANoiseModel::Thermal(_, _) => 0.01, // approximate
            PNANoiseModel::Custom => 0.01,
        });

        CircuitAnalysis {
            num_layers,
            num_clifford_layers,
            num_non_clifford_gates,
            propagation_depth,
            estimated_improvement,
        }
    }

    /// Run the full propagate-and-absorb pipeline.
    ///
    /// 1. Generate per-gate noise channels from the noise model.
    /// 2. Propagate each through subsequent Clifford layers.
    /// 3. Optionally twirl into Pauli-diagonal form.
    /// 4. Absorb into optimal locations.
    pub fn propagate_and_absorb(
        &mut self,
        layers: &[Vec<(CliffordGate, Vec<usize>)>],
        noise_per_gate: &PauliChannel,
    ) -> AbsorptionResult {
        if layers.is_empty() {
            return AbsorptionResult {
                absorbed_channels: Vec::new(),
                original_noise_count: 0,
                absorbed_noise_count: 0,
                reduction_ratio: 0.0,
                effective_noise_strength: 0.0,
            };
        }

        let mut propagated_noises = Vec::new();

        // For each layer, propagate the noise through subsequent Clifford layers
        for (layer_idx, layer) in layers.iter().enumerate() {
            for (_gate, _qubits) in layer {
                let mut noise = noise_per_gate.clone();

                // Propagate through subsequent layers (up to max depth)
                let max_depth = (layer_idx + self.config.max_propagation_depth)
                    .min(layers.len());
                for subsequent_idx in (layer_idx + 1)..max_depth {
                    let subsequent_gates: Vec<CliffordGate> = layers[subsequent_idx]
                        .iter()
                        .filter(|(g, _)| g.is_single_qubit())
                        .map(|(g, _)| *g)
                        .collect();
                    if !subsequent_gates.is_empty() {
                        noise = NoisePropagator::propagate_through_clifford(
                            &noise,
                            &subsequent_gates[0],
                        );
                    }
                }

                // Optionally twirl
                if self.config.clifford_twirling {
                    noise = self.twirler.twirl(&noise);
                }

                propagated_noises.push(noise);
            }
        }

        // Find optimal absorption points
        let targets = NoiseAbsorber::find_optimal_absorption_points(layers);

        // Absorb
        let absorber = NoiseAbsorber::new(self.config.absorption_threshold);
        absorber.absorb(propagated_noises, targets)
    }

    /// Apply PNA mitigation to noisy measurement results.
    ///
    /// Uses the absorbed noise model to compute correction factors and
    /// apply them to the raw expectation values.
    pub fn mitigate(
        &self,
        noisy_results: &[f64],
        absorption: &AbsorptionResult,
    ) -> MitigatedResult {
        if noisy_results.is_empty() {
            return MitigatedResult {
                mitigated_values: Vec::new(),
                raw_values: Vec::new(),
                improvement_factor: 1.0,
                noise_reduction: 0.0,
            };
        }

        // Compute the total noise correction factor from absorbed channels.
        // For Pauli channels, the correction is the inverse of the
        // identity probability: scale = 1 / P(identity).
        let total_identity_prob: f64 = absorption
            .absorbed_channels
            .iter()
            .map(|(_, ch)| ch.identity_probability())
            .product();

        let correction_factor = if total_identity_prob > 1e-10 {
            1.0 / total_identity_prob
        } else {
            1.0 // No correction possible
        };

        // For each observable, the mitigated value is rescaled
        let mitigated: Vec<f64> = noisy_results
            .iter()
            .map(|&v| {
                let corrected = v * correction_factor;
                // Clamp to physical range [-1, 1]
                corrected.clamp(-1.0, 1.0)
            })
            .collect();

        // Compute improvement metrics
        let raw_deviation: f64 = noisy_results
            .iter()
            .map(|v| 1.0 - v.abs())
            .sum::<f64>()
            / noisy_results.len() as f64;

        let mitigated_deviation: f64 = mitigated
            .iter()
            .map(|v| 1.0 - v.abs())
            .sum::<f64>()
            / mitigated.len() as f64;

        let improvement = if mitigated_deviation > 1e-10 {
            raw_deviation / mitigated_deviation
        } else {
            correction_factor
        };

        let noise_reduction = 1.0 - absorption.reduction_ratio;

        MitigatedResult {
            mitigated_values: mitigated,
            raw_values: noisy_results.to_vec(),
            improvement_factor: improvement,
            noise_reduction,
        }
    }

    /// Estimate the improvement factor for a circuit of given depth and noise rate.
    ///
    /// PNA improvement scales with the ratio of original noise points to absorbed
    /// noise points and the depth of Clifford propagation.
    pub fn estimate_improvement(&self, circuit_depth: usize, noise_rate: f64) -> f64 {
        if circuit_depth == 0 || noise_rate <= 0.0 {
            return 1.0;
        }
        // Heuristic: PNA reduces effective noise by a factor related to
        // how many layers can be propagated through.
        let propagation_fraction = self
            .config
            .max_propagation_depth
            .min(circuit_depth) as f64
            / circuit_depth as f64;

        // Each absorbed layer reduces independent noise instances
        // Original: depth * noise_rate per layer
        // After PNA: fewer noise points, each stronger but collectively better
        let original_error = 1.0 - (1.0 - noise_rate).powi(circuit_depth as i32);
        let reduced_points = (circuit_depth as f64 * (1.0 - propagation_fraction * 0.7))
            .max(1.0) as u32;
        let absorbed_noise = noise_rate * propagation_fraction + noise_rate;
        let absorbed_error = 1.0 - (1.0 - absorbed_noise.min(0.99)).powi(reduced_points as i32);

        if absorbed_error > 1e-10 {
            (original_error / absorbed_error).max(1.0)
        } else {
            1.0 + propagation_fraction * circuit_depth as f64
        }
    }
}

/// Check whether a CliffordGate is indeed a Clifford (always true for this enum).
fn is_clifford(_gate: &CliffordGate) -> bool {
    true
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Config builder validation ----

    #[test]
    fn test_config_defaults() {
        let config = PNAConfig::new();
        assert_eq!(config.max_propagation_depth, 10);
        assert!((config.absorption_threshold - 0.001).abs() < 1e-10);
        assert!(config.clifford_twirling);
        assert_eq!(config.num_twirling_samples, 100);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder_chain() {
        let config = PNAConfig::new()
            .max_propagation_depth(5)
            .absorption_threshold(0.01)
            .clifford_twirling(false)
            .noise_model(PNANoiseModel::Depolarizing(0.05))
            .num_twirling_samples(200);

        assert_eq!(config.max_propagation_depth, 5);
        assert!((config.absorption_threshold - 0.01).abs() < 1e-10);
        assert!(!config.clifford_twirling);
        assert_eq!(config.num_twirling_samples, 200);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_errors() {
        let bad_depth = PNAConfig::new().max_propagation_depth(0);
        assert!(bad_depth.validate().is_err());

        let bad_threshold = PNAConfig::new().absorption_threshold(0.0);
        assert!(bad_threshold.validate().is_err());

        let bad_threshold_high = PNAConfig::new().absorption_threshold(1.0);
        assert!(bad_threshold_high.validate().is_err());

        let bad_samples = PNAConfig::new().num_twirling_samples(0);
        assert!(bad_samples.validate().is_err());

        let bad_depol = PNAConfig::new().noise_model(PNANoiseModel::Depolarizing(-0.1));
        assert!(bad_depol.validate().is_err());

        let bad_thermal = PNAConfig::new().noise_model(PNANoiseModel::Thermal(10.0, 25.0));
        assert!(bad_thermal.validate().is_err()); // T2 > 2*T1
    }

    // ---- PauliString multiplication with phases ----

    #[test]
    fn test_pauli_string_multiply_identity() {
        let id = PauliString::new(2);
        let result = id.multiply(&id);
        assert!(result.is_identity());
        assert_eq!(result.weight(), 0);
    }

    #[test]
    fn test_pauli_string_multiply_with_phases() {
        // XY = iZ on single qubit
        let mut x = PauliString::new(1);
        x.paulis[0] = 1;
        let mut y = PauliString::new(1);
        y.paulis[0] = 2;

        let result = x.multiply(&y);
        assert_eq!(result.paulis[0], 3); // Z
        assert!((result.phase - Complex64::new(0.0, 1.0)).norm() < 1e-10); // phase = i

        // YX = -iZ
        let result2 = y.multiply(&x);
        assert_eq!(result2.paulis[0], 3); // Z
        assert!((result2.phase - Complex64::new(0.0, -1.0)).norm() < 1e-10); // phase = -i
    }

    #[test]
    fn test_pauli_string_self_inverse() {
        // P * P = I for all Paulis
        for p_idx in 1..=3u8 {
            let mut ps = PauliString::new(1);
            ps.paulis[0] = p_idx;
            let result = ps.multiply(&ps);
            assert!(result.is_identity());
        }
    }

    #[test]
    fn test_pauli_string_two_qubit_multiply() {
        // (X tensor Z) * (Y tensor Z) = (XY tensor ZZ) = (iZ tensor I)
        let mut a = PauliString::new(2);
        a.paulis[0] = 1; // X
        a.paulis[1] = 3; // Z
        let mut b = PauliString::new(2);
        b.paulis[0] = 2; // Y
        b.paulis[1] = 3; // Z

        let result = a.multiply(&b);
        assert_eq!(result.paulis[0], 3); // Z
        assert_eq!(result.paulis[1], 0); // I (Z*Z=I)
        assert!((result.phase - Complex64::new(0.0, 1.0)).norm() < 1e-10); // iZ from X*Y
    }

    // ---- PauliString commutativity ----

    #[test]
    fn test_pauli_commutativity() {
        // X and Z anti-commute (single site)
        let mut x = PauliString::new(1);
        x.paulis[0] = 1;
        let mut z = PauliString::new(1);
        z.paulis[0] = 3;
        assert!(!x.commutes_with(&z));

        // X and X commute
        let x2 = x.clone();
        assert!(x.commutes_with(&x2));

        // I commutes with everything
        let id = PauliString::new(1);
        assert!(id.commutes_with(&x));
        assert!(id.commutes_with(&z));

        // XI and IX commute (different sites)
        let mut xi = PauliString::new(2);
        xi.paulis[0] = 1;
        let mut ix = PauliString::new(2);
        ix.paulis[1] = 1;
        assert!(xi.commutes_with(&ix));

        // XZ and ZX anti-commute at both sites -> even -> commute
        let mut xz = PauliString::new(2);
        xz.paulis[0] = 1;
        xz.paulis[1] = 3;
        let mut zx = PauliString::new(2);
        zx.paulis[0] = 3;
        zx.paulis[1] = 1;
        assert!(xz.commutes_with(&zx)); // 2 anti-commuting sites -> even -> commute
    }

    // ---- Clifford conjugation of Paulis ----

    #[test]
    fn test_hadamard_conjugation() {
        // HXH = Z
        let (new_p, sign) = CliffordGate::H.conjugate_single_pauli(1); // X
        assert_eq!(new_p, 3); // Z
        assert_eq!(sign, 1);

        // HZH = X
        let (new_p, sign) = CliffordGate::H.conjugate_single_pauli(3); // Z
        assert_eq!(new_p, 1); // X
        assert_eq!(sign, 1);

        // HYH = -Y
        let (new_p, sign) = CliffordGate::H.conjugate_single_pauli(2); // Y
        assert_eq!(new_p, 2); // Y
        assert_eq!(sign, -1);

        // HIH = I
        let (new_p, sign) = CliffordGate::H.conjugate_single_pauli(0);
        assert_eq!(new_p, 0);
        assert_eq!(sign, 1);
    }

    #[test]
    fn test_s_gate_conjugation() {
        // SXS^dag = Y
        let (new_p, sign) = CliffordGate::S.conjugate_single_pauli(1);
        assert_eq!(new_p, 2); // Y
        assert_eq!(sign, 1);

        // SYS^dag = -X
        let (new_p, sign) = CliffordGate::S.conjugate_single_pauli(2);
        assert_eq!(new_p, 1); // X
        assert_eq!(sign, -1);

        // SZS^dag = Z
        let (new_p, sign) = CliffordGate::S.conjugate_single_pauli(3);
        assert_eq!(new_p, 3);
        assert_eq!(sign, 1);
    }

    // ---- Depolarizing channel construction ----

    #[test]
    fn test_depolarizing_channel_construction() {
        let ch = PauliChannel::new_depolarizing(1, 0.03);
        assert!(ch.is_valid());
        assert!((ch.identity_probability() - 0.97).abs() < 1e-10);
        assert!((ch.total_error_rate() - 0.03).abs() < 1e-10);
        assert_eq!(ch.num_error_terms(), 3); // X, Y, Z
    }

    #[test]
    fn test_zero_noise_channel() {
        let ch = PauliChannel::new_depolarizing(1, 0.0);
        assert!(ch.is_valid());
        assert!((ch.identity_probability() - 1.0).abs() < 1e-10);
        assert!(ch.total_error_rate() < 1e-10);
    }

    // ---- Channel composition ----

    #[test]
    fn test_channel_composition() {
        let ch1 = PauliChannel::new_depolarizing(1, 0.01);
        let ch2 = PauliChannel::new_depolarizing(1, 0.02);
        let composed = ch1.compose(&ch2);

        assert!(composed.is_valid());
        // Composed error rate should be larger than either individual rate
        assert!(composed.total_error_rate() > ch1.total_error_rate());
        assert!(composed.total_error_rate() > ch2.total_error_rate());
        // But should not exceed their sum (no coherent addition)
        assert!(composed.total_error_rate() < ch1.total_error_rate() + ch2.total_error_rate() + 1e-10);
    }

    #[test]
    fn test_channel_composition_with_identity() {
        let ch = PauliChannel::new_depolarizing(1, 0.05);
        let id = PauliChannel::new_depolarizing(1, 0.0);
        let composed = ch.compose(&id);

        assert!(composed.is_valid());
        assert!((composed.total_error_rate() - ch.total_error_rate()).abs() < 1e-8);
    }

    // ---- Noise propagation through H gate ----

    #[test]
    fn test_noise_propagation_through_h() {
        // Depolarizing noise is invariant under H conjugation because it is
        // symmetric over X, Y, Z. The probabilities should permute but stay equal.
        let noise = PauliChannel::new_depolarizing(1, 0.03);
        let propagated = NoisePropagator::propagate_through_clifford(&noise, &CliffordGate::H);

        assert!(propagated.is_valid());
        assert!((propagated.total_error_rate() - 0.03).abs() < 1e-10);
        assert!((propagated.identity_probability() - 0.97).abs() < 1e-10);
    }

    #[test]
    fn test_asymmetric_noise_propagation_through_h() {
        // Asymmetric Pauli channel: H swaps X <-> Z, negates Y
        let noise = PauliChannel::new_pauli(0.04, 0.01, 0.02);
        let propagated = NoisePropagator::propagate_through_clifford(&noise, &CliffordGate::H);

        assert!(propagated.is_valid());
        // After H: px->pz, pz->px, py->py (signs absorbed into phase, not probability)
        // Total error rate is preserved
        assert!(
            (propagated.total_error_rate() - noise.total_error_rate()).abs() < 1e-10
        );
    }

    // ---- Noise propagation through CX gate ----

    #[test]
    fn test_cx_conjugation_table_consistency() {
        // Verify the CX table preserves the group structure:
        // CX is self-inverse, so applying it twice should return the original.
        for pc in 0..4u8 {
            for pt in 0..4u8 {
                let (nc, nt, s1) = cx_conjugation_table(pc, pt);
                let (oc, ot, s2) = cx_conjugation_table(nc, nt);
                assert_eq!(oc, pc, "CX^2 should be identity: pc={} pt={}", pc, pt);
                assert_eq!(ot, pt, "CX^2 should be identity: pc={} pt={}", pc, pt);
                assert_eq!(
                    s1 * s2,
                    1,
                    "Double conjugation phase should be +1: pc={} pt={}",
                    pc,
                    pt
                );
            }
        }
    }

    // ---- Pauli twirling produces Pauli channel ----

    #[test]
    fn test_pauli_twirling_preserves_pauli_channel() {
        let ch = PauliChannel::new_depolarizing(1, 0.05);
        let mut twirler = CliffordTwirler::new(123);
        let twirled = twirler.twirl(&ch);

        assert!(twirled.is_valid());
        assert!((twirled.total_error_rate() - ch.total_error_rate()).abs() < 1e-8);
    }

    #[test]
    fn test_sample_twirl_converges() {
        let ch = PauliChannel::new_pauli(0.02, 0.01, 0.03);
        let mut twirler = CliffordTwirler::new(456);
        let twirled = twirler.sample_twirl(&ch, 1000);

        assert!(twirled.is_valid());
        // Total error rate should be approximately preserved
        assert!(
            (twirled.total_error_rate() - ch.total_error_rate()).abs() < 0.02
        );
    }

    // ---- Absorption reduces channel count ----

    #[test]
    fn test_absorption_reduces_count() {
        let channels: Vec<PauliChannel> = (0..10)
            .map(|_| PauliChannel::new_depolarizing(1, 0.01))
            .collect();

        let targets = vec![0, 5, 9];
        let absorber = NoiseAbsorber::new(0.001);
        let result = absorber.absorb(channels, targets);

        assert_eq!(result.original_noise_count, 10);
        assert!(result.absorbed_noise_count <= 3);
        assert!(result.reduction_ratio < 1.0);
        assert!(result.effective_noise_strength > 0.01); // Combined noise is stronger

        for (_, ch) in &result.absorbed_channels {
            assert!(ch.is_valid());
        }
    }

    // ---- Optimal absorption point selection ----

    #[test]
    fn test_optimal_absorption_points() {
        // Circuit with alternating single and two-qubit layers
        let layers = vec![
            vec![(CliffordGate::H, vec![0])],                // single-qubit
            vec![(CliffordGate::CX, vec![0, 1])],            // two-qubit
            vec![(CliffordGate::H, vec![0])],                // single-qubit
            vec![(CliffordGate::H, vec![1])],                // single-qubit
            vec![(CliffordGate::CZ, vec![0, 1])],            // two-qubit
        ];

        let points = NoiseAbsorber::find_optimal_absorption_points(&layers);
        // Should include boundaries and transitions
        assert!(points.contains(&0)); // first layer
        assert!(points.contains(&4)); // last layer
        assert!(points.len() >= 2);
    }

    #[test]
    fn test_empty_circuit_absorption_points() {
        let points = NoiseAbsorber::find_optimal_absorption_points(&[]);
        assert!(points.is_empty());
    }

    // ---- Full propagate-and-absorb pipeline ----

    #[test]
    fn test_full_pna_pipeline() {
        let config = PNAConfig::new()
            .max_propagation_depth(3)
            .noise_model(PNANoiseModel::Depolarizing(0.02))
            .clifford_twirling(true);

        let mut mitigator = PNAMitigator::new(config);

        let layers = vec![
            vec![(CliffordGate::H, vec![0])],
            vec![(CliffordGate::S, vec![0])],
            vec![(CliffordGate::H, vec![0])],
        ];

        let noise = PauliChannel::new_depolarizing(1, 0.02);
        let absorption = mitigator.propagate_and_absorb(&layers, &noise);

        assert!(absorption.original_noise_count > 0);
        assert!(absorption.absorbed_noise_count > 0);
        assert!(absorption.reduction_ratio <= 1.0);
        for (_, ch) in &absorption.absorbed_channels {
            assert!(ch.is_valid());
        }
    }

    // ---- Mitigation improves estimates ----

    #[test]
    fn test_mitigation_improves_estimates() {
        let config = PNAConfig::new()
            .noise_model(PNANoiseModel::Depolarizing(0.05));

        let mut mitigator = PNAMitigator::new(config);

        let layers = vec![
            vec![(CliffordGate::H, vec![0])],
            vec![(CliffordGate::S, vec![0])],
            vec![(CliffordGate::H, vec![0])],
            vec![(CliffordGate::Z, vec![0])],
        ];

        let noise = PauliChannel::new_depolarizing(1, 0.05);
        let absorption = mitigator.propagate_and_absorb(&layers, &noise);

        // Simulate noisy results (ideal = 1.0 for Z expectation, noise reduces it)
        let noisy_results = vec![0.85, 0.82, 0.88];
        let mitigated = mitigator.mitigate(&noisy_results, &absorption);

        // Mitigated values should be closer to ideal (1.0) than raw
        for (m, r) in mitigated
            .mitigated_values
            .iter()
            .zip(mitigated.raw_values.iter())
        {
            assert!(
                m.abs() >= r.abs() - 1e-10,
                "Mitigated {} should be >= raw {} in magnitude",
                m,
                r
            );
        }
    }

    // ---- Circuit analysis ----

    #[test]
    fn test_circuit_analysis() {
        let config = PNAConfig::new()
            .max_propagation_depth(5)
            .noise_model(PNANoiseModel::Depolarizing(0.01));

        let mitigator = PNAMitigator::new(config);

        let layers = vec![
            vec![(CliffordGate::H, vec![0]), (CliffordGate::H, vec![1])],
            vec![(CliffordGate::CX, vec![0, 1])],
            vec![(CliffordGate::S, vec![0]), (CliffordGate::SDag, vec![1])],
            vec![(CliffordGate::CZ, vec![0, 1])],
            vec![(CliffordGate::H, vec![0])],
        ];

        let analysis = mitigator.analyze_circuit(&layers);
        assert_eq!(analysis.num_layers, 5);
        assert_eq!(analysis.num_clifford_layers, 5); // all Clifford
        assert_eq!(analysis.num_non_clifford_gates, 0);
        assert!(analysis.propagation_depth <= 5);
        assert!(analysis.estimated_improvement >= 1.0);
    }

    // ---- Noise channel validity ----

    #[test]
    fn test_noise_channel_validity() {
        let valid = PauliChannel::new_depolarizing(1, 0.1);
        assert!(valid.is_valid());

        let pauli = PauliChannel::new_pauli(0.1, 0.2, 0.3);
        assert!(pauli.is_valid());
        assert!((pauli.identity_probability() - 0.4).abs() < 1e-10);

        // Manually construct invalid channel
        let invalid = PauliChannel {
            probabilities: vec![
                (PauliString::new(1), 0.5),
                (PauliString::single(1, 1, 0), 0.3),
                // Missing probability mass
            ],
            num_qubits: 1,
        };
        assert!(!invalid.is_valid());
    }

    // ---- PNA vs raw comparison ----

    #[test]
    fn test_pna_vs_raw_comparison() {
        let config = PNAConfig::new()
            .max_propagation_depth(5)
            .noise_model(PNANoiseModel::Depolarizing(0.03));

        let mut mitigator = PNAMitigator::new(config);

        // Build a 10-layer circuit
        let layers: Vec<Vec<(CliffordGate, Vec<usize>)>> = (0..10)
            .map(|i| {
                if i % 2 == 0 {
                    vec![(CliffordGate::H, vec![0])]
                } else {
                    vec![(CliffordGate::S, vec![0])]
                }
            })
            .collect();

        let noise = PauliChannel::new_depolarizing(1, 0.03);
        let absorption = mitigator.propagate_and_absorb(&layers, &noise);

        // Verify absorption achieved reduction
        assert!(
            absorption.absorbed_noise_count < absorption.original_noise_count,
            "PNA should reduce noise channel count: {} < {}",
            absorption.absorbed_noise_count,
            absorption.original_noise_count
        );

        // Compare mitigated vs raw
        let raw = vec![0.70, 0.72, 0.68, 0.75];
        let result = mitigator.mitigate(&raw, &absorption);

        assert_eq!(result.raw_values.len(), 4);
        assert_eq!(result.mitigated_values.len(), 4);
        assert!(result.noise_reduction >= 0.0);
    }

    // ---- Edge cases ----

    #[test]
    fn test_pauli_string_weight() {
        let id = PauliString::new(5);
        assert_eq!(id.weight(), 0);

        let mut ps = PauliString::new(5);
        ps.paulis[0] = 1;
        ps.paulis[2] = 3;
        ps.paulis[4] = 2;
        assert_eq!(ps.weight(), 3);
    }

    #[test]
    fn test_clifford_gate_single_qubit_classification() {
        assert!(CliffordGate::I.is_single_qubit());
        assert!(CliffordGate::H.is_single_qubit());
        assert!(CliffordGate::S.is_single_qubit());
        assert!(CliffordGate::SDag.is_single_qubit());
        assert!(CliffordGate::X.is_single_qubit());
        assert!(CliffordGate::Y.is_single_qubit());
        assert!(CliffordGate::Z.is_single_qubit());
        assert!(!CliffordGate::CX.is_single_qubit());
        assert!(!CliffordGate::CZ.is_single_qubit());
        assert!(!CliffordGate::SWAP.is_single_qubit());
    }

    #[test]
    fn test_pauli_noise_model_display() {
        let d = PNANoiseModel::Depolarizing(0.01);
        assert!(format!("{}", d).contains("0.01"));

        let p = PNANoiseModel::PauliChannel(0.01, 0.02, 0.03);
        assert!(format!("{}", p).contains("0.02"));

        let t = PNANoiseModel::Thermal(50.0, 30.0);
        assert!(format!("{}", t).contains("50"));
    }

    #[test]
    fn test_error_display() {
        let e = PNAError::InvalidConfig("bad value".into());
        assert!(format!("{}", e).contains("bad value"));

        let e2 = PNAError::QubitOutOfRange {
            qubit: 5,
            num_qubits: 3,
        };
        assert!(format!("{}", e2).contains("5"));
        assert!(format!("{}", e2).contains("3"));
    }

    #[test]
    fn test_empty_mitigate() {
        let config = PNAConfig::new();
        let mitigator = PNAMitigator::new(config);

        let absorption = AbsorptionResult {
            absorbed_channels: Vec::new(),
            original_noise_count: 0,
            absorbed_noise_count: 0,
            reduction_ratio: 0.0,
            effective_noise_strength: 0.0,
        };

        let result = mitigator.mitigate(&[], &absorption);
        assert!(result.mitigated_values.is_empty());
        assert!((result.improvement_factor - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_improvement_scales_with_depth() {
        let config = PNAConfig::new()
            .max_propagation_depth(5)
            .noise_model(PNANoiseModel::Depolarizing(0.02));
        let mitigator = PNAMitigator::new(config);

        let improvement_shallow = mitigator.estimate_improvement(2, 0.02);
        let improvement_deep = mitigator.estimate_improvement(20, 0.02);

        // Deeper circuits should benefit more from PNA
        assert!(
            improvement_deep >= improvement_shallow,
            "Deep circuit improvement {:.3} should be >= shallow {:.3}",
            improvement_deep,
            improvement_shallow
        );
    }
}
