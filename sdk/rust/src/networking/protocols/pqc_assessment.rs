//! Post-Quantum Cryptography (PQC) Assessment Module
//!
//! Maps NIST PQC standards to quantum attack feasibility, estimates the
//! physical and logical resources required to break various cryptographic
//! schemes using quantum algorithms, and projects a quantum threat timeline.
//!
//! # Covered Standards
//!
//! - **FIPS 203 (ML-KEM)**: Module-Lattice Key Encapsulation Mechanism
//! - **FIPS 204 (ML-DSA)**: Module-Lattice Digital Signature Algorithm
//! - **FIPS 205 (SLH-DSA)**: Stateless Hash-Based Digital Signature Algorithm
//! - **HQC**: Hamming Quasi-Cyclic (selected March 2025)
//! - **NTRU, BIKE, Classic McEliece**: Additional PQC candidates
//!
//! # Quantum Attacks Modelled
//!
//! - **Shor's algorithm**: Factoring (RSA) and discrete-log (ECC, DH, DSA)
//! - **Grover's algorithm**: Brute-force key search for symmetric ciphers
//! - **BHT (Brassard-Hoyer-Tapper)**: Element distinctness
//! - **Quantum BKZ**: Lattice sieving (for lattice-based PQC)
//! - **Quantum ISD**: Information Set Decoding (for code-based PQC)
//! - **Quantum Walk**: Generalised search on graphs
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::pqc_assessment::*;
//!
//! let config = PqcConfig::default();
//! let assessment = assess_algorithm(&CryptoAlgorithm::RSA { key_bits: 2048 }, &config);
//! assert_eq!(assessment.threat_level, ThreatLevel::Critical);
//! assert_eq!(assessment.migration_urgency, MigrationUrgency::Immediate);
//! ```

use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors produced by the PQC assessment module.
#[derive(Clone, Debug, PartialEq)]
pub enum PqcError {
    /// The supplied key size is invalid for the algorithm.
    InvalidKeySize(usize),
    /// The algorithm is not recognised or not yet supported.
    UnsupportedAlgorithm(String),
    /// The resource estimation procedure failed.
    EstimationFailed(String),
}

impl fmt::Display for PqcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PqcError::InvalidKeySize(bits) => {
                write!(f, "Invalid key size: {} bits", bits)
            }
            PqcError::UnsupportedAlgorithm(name) => {
                write!(f, "Unsupported algorithm: {}", name)
            }
            PqcError::EstimationFailed(reason) => {
                write!(f, "Estimation failed: {}", reason)
            }
        }
    }
}

impl std::error::Error for PqcError {}

// ============================================================
// NIST PQC PARAMETER SETS
// ============================================================

/// ML-KEM parameter sets (FIPS 203).
///
/// Security categories 1, 3, and 5 corresponding to AES-128, AES-192, and
/// AES-256 equivalent security.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MLKEMParams {
    /// ML-KEM-512: Security Level 1 (k = 2, n = 256, q = 3329)
    ML512,
    /// ML-KEM-768: Security Level 3 (k = 3)
    ML768,
    /// ML-KEM-1024: Security Level 5 (k = 4)
    ML1024,
}

impl MLKEMParams {
    /// NIST security level (1, 3, or 5).
    pub fn security_level(&self) -> usize {
        match self {
            MLKEMParams::ML512 => 1,
            MLKEMParams::ML768 => 3,
            MLKEMParams::ML1024 => 5,
        }
    }

    /// Module rank k.
    pub fn module_rank(&self) -> usize {
        match self {
            MLKEMParams::ML512 => 2,
            MLKEMParams::ML768 => 3,
            MLKEMParams::ML1024 => 4,
        }
    }

    /// Lattice dimension (k * n).
    pub fn lattice_dimension(&self) -> usize {
        self.module_rank() * 256
    }

    /// Classical security (bits) from the core-SVP hardness estimate.
    pub fn classical_security_bits(&self) -> usize {
        match self {
            MLKEMParams::ML512 => 118,
            MLKEMParams::ML768 => 182,
            MLKEMParams::ML1024 => 256,
        }
    }

    /// Quantum security (bits) under the core-SVP model with quantum BKZ.
    pub fn quantum_security_bits(&self) -> usize {
        match self {
            MLKEMParams::ML512 => 107,
            MLKEMParams::ML768 => 164,
            MLKEMParams::ML1024 => 232,
        }
    }
}

impl fmt::Display for MLKEMParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MLKEMParams::ML512 => write!(f, "ML-KEM-512"),
            MLKEMParams::ML768 => write!(f, "ML-KEM-768"),
            MLKEMParams::ML1024 => write!(f, "ML-KEM-1024"),
        }
    }
}

/// ML-DSA parameter sets (FIPS 204).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MLDSAParams {
    /// ML-DSA-44: Security Level 2
    ML44,
    /// ML-DSA-65: Security Level 3
    ML65,
    /// ML-DSA-87: Security Level 5
    ML87,
}

impl MLDSAParams {
    pub fn security_level(&self) -> usize {
        match self {
            MLDSAParams::ML44 => 2,
            MLDSAParams::ML65 => 3,
            MLDSAParams::ML87 => 5,
        }
    }

    pub fn lattice_dimension(&self) -> usize {
        match self {
            MLDSAParams::ML44 => 4 * 256,
            MLDSAParams::ML65 => 6 * 256,
            MLDSAParams::ML87 => 8 * 256,
        }
    }

    pub fn classical_security_bits(&self) -> usize {
        match self {
            MLDSAParams::ML44 => 128,
            MLDSAParams::ML65 => 192,
            MLDSAParams::ML87 => 256,
        }
    }

    pub fn quantum_security_bits(&self) -> usize {
        match self {
            MLDSAParams::ML44 => 115,
            MLDSAParams::ML65 => 170,
            MLDSAParams::ML87 => 234,
        }
    }
}

impl fmt::Display for MLDSAParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MLDSAParams::ML44 => write!(f, "ML-DSA-44"),
            MLDSAParams::ML65 => write!(f, "ML-DSA-65"),
            MLDSAParams::ML87 => write!(f, "ML-DSA-87"),
        }
    }
}

/// SLH-DSA parameter sets (FIPS 205).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SLHDSAParams {
    SHA2_128f,
    SHA2_128s,
    SHA2_192f,
    SHA2_192s,
    SHA2_256f,
    SHA2_256s,
}

impl SLHDSAParams {
    pub fn security_level(&self) -> usize {
        match self {
            SLHDSAParams::SHA2_128f | SLHDSAParams::SHA2_128s => 1,
            SLHDSAParams::SHA2_192f | SLHDSAParams::SHA2_192s => 3,
            SLHDSAParams::SHA2_256f | SLHDSAParams::SHA2_256s => 5,
        }
    }

    /// Hash-based security is purely Grover-limited.
    pub fn classical_security_bits(&self) -> usize {
        match self {
            SLHDSAParams::SHA2_128f | SLHDSAParams::SHA2_128s => 128,
            SLHDSAParams::SHA2_192f | SLHDSAParams::SHA2_192s => 192,
            SLHDSAParams::SHA2_256f | SLHDSAParams::SHA2_256s => 256,
        }
    }

    pub fn quantum_security_bits(&self) -> usize {
        // Grover halves the key space exponent.
        self.classical_security_bits() / 2
    }

    /// Is this the "fast" (larger signatures) or "small" variant?
    pub fn is_fast(&self) -> bool {
        matches!(
            self,
            SLHDSAParams::SHA2_128f | SLHDSAParams::SHA2_192f | SLHDSAParams::SHA2_256f
        )
    }
}

impl fmt::Display for SLHDSAParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            SLHDSAParams::SHA2_128f => "SLH-DSA-SHA2-128f",
            SLHDSAParams::SHA2_128s => "SLH-DSA-SHA2-128s",
            SLHDSAParams::SHA2_192f => "SLH-DSA-SHA2-192f",
            SLHDSAParams::SHA2_192s => "SLH-DSA-SHA2-192s",
            SLHDSAParams::SHA2_256f => "SLH-DSA-SHA2-256f",
            SLHDSAParams::SHA2_256s => "SLH-DSA-SHA2-256s",
        };
        write!(f, "{}", name)
    }
}

/// HQC parameter sets (selected NIST Round 4, March 2025).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HQCParams {
    HQC128,
    HQC192,
    HQC256,
}

impl HQCParams {
    pub fn security_level(&self) -> usize {
        match self {
            HQCParams::HQC128 => 1,
            HQCParams::HQC192 => 3,
            HQCParams::HQC256 => 5,
        }
    }

    /// Code length n.
    pub fn code_length(&self) -> usize {
        match self {
            HQCParams::HQC128 => 17_669,
            HQCParams::HQC192 => 35_851,
            HQCParams::HQC256 => 57_637,
        }
    }

    pub fn classical_security_bits(&self) -> usize {
        match self {
            HQCParams::HQC128 => 128,
            HQCParams::HQC192 => 192,
            HQCParams::HQC256 => 256,
        }
    }

    pub fn quantum_security_bits(&self) -> usize {
        match self {
            HQCParams::HQC128 => 115,
            HQCParams::HQC192 => 172,
            HQCParams::HQC256 => 233,
        }
    }
}

impl fmt::Display for HQCParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HQCParams::HQC128 => write!(f, "HQC-128"),
            HQCParams::HQC192 => write!(f, "HQC-192"),
            HQCParams::HQC256 => write!(f, "HQC-256"),
        }
    }
}

/// NTRU parameter sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NTRUParams {
    NTRU509,
    NTRU677,
    NTRU821,
}

impl NTRUParams {
    pub fn security_level(&self) -> usize {
        match self {
            NTRUParams::NTRU509 => 1,
            NTRUParams::NTRU677 => 3,
            NTRUParams::NTRU821 => 5,
        }
    }

    pub fn lattice_dimension(&self) -> usize {
        match self {
            NTRUParams::NTRU509 => 509,
            NTRUParams::NTRU677 => 677,
            NTRUParams::NTRU821 => 821,
        }
    }

    pub fn classical_security_bits(&self) -> usize {
        match self {
            NTRUParams::NTRU509 => 143,
            NTRUParams::NTRU677 => 199,
            NTRUParams::NTRU821 => 248,
        }
    }

    pub fn quantum_security_bits(&self) -> usize {
        match self {
            NTRUParams::NTRU509 => 128,
            NTRUParams::NTRU677 => 178,
            NTRUParams::NTRU821 => 222,
        }
    }
}

impl fmt::Display for NTRUParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NTRUParams::NTRU509 => write!(f, "NTRU-509"),
            NTRUParams::NTRU677 => write!(f, "NTRU-677"),
            NTRUParams::NTRU821 => write!(f, "NTRU-821"),
        }
    }
}

/// BIKE parameter sets (code-based KEM).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BIKEParams {
    BIKE1,
    BIKE3,
    BIKE5,
}

impl BIKEParams {
    pub fn security_level(&self) -> usize {
        match self {
            BIKEParams::BIKE1 => 1,
            BIKEParams::BIKE3 => 3,
            BIKEParams::BIKE5 => 5,
        }
    }

    /// Block length r.
    pub fn block_length(&self) -> usize {
        match self {
            BIKEParams::BIKE1 => 12_323,
            BIKEParams::BIKE3 => 24_659,
            BIKEParams::BIKE5 => 40_973,
        }
    }

    pub fn classical_security_bits(&self) -> usize {
        match self {
            BIKEParams::BIKE1 => 128,
            BIKEParams::BIKE3 => 192,
            BIKEParams::BIKE5 => 256,
        }
    }

    pub fn quantum_security_bits(&self) -> usize {
        match self {
            BIKEParams::BIKE1 => 112,
            BIKEParams::BIKE3 => 170,
            BIKEParams::BIKE5 => 231,
        }
    }
}

impl fmt::Display for BIKEParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BIKEParams::BIKE1 => write!(f, "BIKE-L1"),
            BIKEParams::BIKE3 => write!(f, "BIKE-L3"),
            BIKEParams::BIKE5 => write!(f, "BIKE-L5"),
        }
    }
}

/// Classic McEliece parameter sets (code-based KEM).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McElieceParams {
    MC348864,
    MC460896,
    MC6688128,
    MC6960119,
    MC8192128,
}

impl McElieceParams {
    pub fn security_level(&self) -> usize {
        match self {
            McElieceParams::MC348864 => 1,
            McElieceParams::MC460896 => 3,
            McElieceParams::MC6688128 => 5,
            McElieceParams::MC6960119 => 5,
            McElieceParams::MC8192128 => 5,
        }
    }

    /// Code length n.
    pub fn code_length(&self) -> usize {
        match self {
            McElieceParams::MC348864 => 3488,
            McElieceParams::MC460896 => 4608,
            McElieceParams::MC6688128 => 6688,
            McElieceParams::MC6960119 => 6960,
            McElieceParams::MC8192128 => 8192,
        }
    }

    /// Error-correction capability t.
    pub fn error_capability(&self) -> usize {
        match self {
            McElieceParams::MC348864 => 64,
            McElieceParams::MC460896 => 96,
            McElieceParams::MC6688128 => 128,
            McElieceParams::MC6960119 => 119,
            McElieceParams::MC8192128 => 128,
        }
    }

    pub fn classical_security_bits(&self) -> usize {
        match self {
            McElieceParams::MC348864 => 263,
            McElieceParams::MC460896 => 309,
            McElieceParams::MC6688128 => 370,
            McElieceParams::MC6960119 => 370,
            McElieceParams::MC8192128 => 396,
        }
    }

    pub fn quantum_security_bits(&self) -> usize {
        match self {
            McElieceParams::MC348864 => 194,
            McElieceParams::MC460896 => 230,
            McElieceParams::MC6688128 => 275,
            McElieceParams::MC6960119 => 275,
            McElieceParams::MC8192128 => 295,
        }
    }
}

impl fmt::Display for McElieceParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            McElieceParams::MC348864 => "Classic-McEliece-348864",
            McElieceParams::MC460896 => "Classic-McEliece-460896",
            McElieceParams::MC6688128 => "Classic-McEliece-6688128",
            McElieceParams::MC6960119 => "Classic-McEliece-6960119",
            McElieceParams::MC8192128 => "Classic-McEliece-8192128",
        };
        write!(f, "{}", name)
    }
}

// ============================================================
// CRYPTOGRAPHIC ALGORITHM ENUM
// ============================================================

/// A cryptographic algorithm to assess for quantum vulnerability.
#[derive(Clone, Debug, PartialEq)]
pub enum CryptoAlgorithm {
    // -- Pre-quantum (vulnerable to Shor) --
    /// RSA public-key encryption / signing.
    RSA { key_bits: usize },
    /// Elliptic Curve Cryptography.
    ECC { curve_bits: usize },
    /// Diffie-Hellman key exchange.
    DH { key_bits: usize },
    /// Digital Signature Algorithm.
    DSA { key_bits: usize },

    // -- Symmetric (weakened by Grover) --
    /// AES symmetric cipher.
    AES { key_bits: usize },

    // -- NIST PQC standards (resistant) --
    /// Module-Lattice KEM (FIPS 203).
    MLKEM { parameter_set: MLKEMParams },
    /// Module-Lattice DSA (FIPS 204).
    MLDSA { parameter_set: MLDSAParams },
    /// Stateless Hash-Based DSA (FIPS 205).
    SLHDSA { parameter_set: SLHDSAParams },
    /// Hamming Quasi-Cyclic KEM (selected March 2025).
    HQC { parameter_set: HQCParams },

    // -- Other PQC candidates --
    /// NTRU lattice-based KEM.
    NTRU { parameter_set: NTRUParams },
    /// BIKE code-based KEM.
    BIKE { parameter_set: BIKEParams },
    /// Classic McEliece code-based KEM.
    ClassicMcEliece { parameter_set: McElieceParams },
}

impl fmt::Display for CryptoAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoAlgorithm::RSA { key_bits } => write!(f, "RSA-{}", key_bits),
            CryptoAlgorithm::ECC { curve_bits } => write!(f, "ECC-{}", curve_bits),
            CryptoAlgorithm::DH { key_bits } => write!(f, "DH-{}", key_bits),
            CryptoAlgorithm::DSA { key_bits } => write!(f, "DSA-{}", key_bits),
            CryptoAlgorithm::AES { key_bits } => write!(f, "AES-{}", key_bits),
            CryptoAlgorithm::MLKEM { parameter_set } => write!(f, "{}", parameter_set),
            CryptoAlgorithm::MLDSA { parameter_set } => write!(f, "{}", parameter_set),
            CryptoAlgorithm::SLHDSA { parameter_set } => write!(f, "{}", parameter_set),
            CryptoAlgorithm::HQC { parameter_set } => write!(f, "{}", parameter_set),
            CryptoAlgorithm::NTRU { parameter_set } => write!(f, "{}", parameter_set),
            CryptoAlgorithm::BIKE { parameter_set } => write!(f, "{}", parameter_set),
            CryptoAlgorithm::ClassicMcEliece { parameter_set } => {
                write!(f, "{}", parameter_set)
            }
        }
    }
}

// ============================================================
// QUANTUM ATTACK METHOD
// ============================================================

/// Quantum attack strategy.
#[derive(Clone, Debug, PartialEq)]
pub enum QuantumAttack {
    /// Shor's algorithm for factoring / discrete-log.
    Shor,
    /// Grover's algorithm for brute-force key search.
    Grover,
    /// Brassard-Hoyer-Tapper algorithm (element distinctness).
    BHT { memory: usize },
    /// Quantum-enhanced lattice sieving (BKZ with quantum speed-up).
    QuantumLattice,
    /// Information Set Decoding with optional quantum speed-up.
    ISD { quantum: bool },
    /// Quantum walk on Johnson graph (search problems).
    QuantumWalk,
}

impl fmt::Display for QuantumAttack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantumAttack::Shor => write!(f, "Shor"),
            QuantumAttack::Grover => write!(f, "Grover"),
            QuantumAttack::BHT { memory } => write!(f, "BHT(mem={})", memory),
            QuantumAttack::QuantumLattice => write!(f, "Quantum-BKZ"),
            QuantumAttack::ISD { quantum } => {
                if *quantum {
                    write!(f, "Quantum-ISD")
                } else {
                    write!(f, "Classical-ISD")
                }
            }
            QuantumAttack::QuantumWalk => write!(f, "Quantum-Walk"),
        }
    }
}

// ============================================================
// RESOURCE ESTIMATES
// ============================================================

/// Resource estimate for executing a quantum attack.
#[derive(Clone, Debug)]
pub struct QuantumAttackEstimate {
    /// Name of the target algorithm.
    pub algorithm: String,
    /// Name of the attack strategy.
    pub attack: String,
    /// Logical qubits required.
    pub logical_qubits: usize,
    /// Number of T gates.
    pub t_gates: u64,
    /// Number of Toffoli gates.
    pub toffoli_gates: u64,
    /// Circuit depth (sequential gate layers).
    pub circuit_depth: u64,
    /// Physical qubits with surface-code overhead at distance d.
    pub surface_code_qubits: usize,
    /// Estimated wall-clock time in hours.
    pub wall_clock_hours: f64,
    /// Success probability per run.
    pub success_probability: f64,
    /// Effective security in bits after accounting for the quantum attack.
    pub security_level_bits: usize,
}

impl fmt::Display for QuantumAttackEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Quantum Attack Estimate for {}", self.algorithm)?;
        writeln!(f, "  Attack: {}", self.attack)?;
        writeln!(f, "  Logical qubits: {}", self.logical_qubits)?;
        writeln!(f, "  T gates: {:.2e}", self.t_gates as f64)?;
        writeln!(f, "  Toffoli gates: {:.2e}", self.toffoli_gates as f64)?;
        writeln!(f, "  Circuit depth: {:.2e}", self.circuit_depth as f64)?;
        writeln!(
            f,
            "  Physical qubits (surface code): {}",
            self.surface_code_qubits
        )?;
        writeln!(f, "  Wall-clock time: {:.1} hours", self.wall_clock_hours)?;
        writeln!(f, "  Success probability: {:.4}", self.success_probability)?;
        writeln!(f, "  Effective security: {} bits", self.security_level_bits)?;
        Ok(())
    }
}

/// Shor's algorithm resource estimate.
#[derive(Clone, Debug)]
pub struct ShorEstimate {
    /// Bit-length of the number to factorise.
    pub n_bits: usize,
    /// Logical qubits: 2n + 3.
    pub logical_qubits: usize,
    /// Toffoli gate count for modular exponentiation.
    pub toffoli_gates: u64,
    /// Number of modular exponentiations.
    pub modular_exponentiations: u64,
    /// Sequential circuit depth.
    pub circuit_depth: u64,
}

/// Grover's algorithm resource estimate.
#[derive(Clone, Debug)]
pub struct GroverEstimate {
    /// Key length in bits.
    pub key_bits: usize,
    /// Number of Grover oracle queries: O(2^(n/2)).
    pub quantum_queries: u64,
    /// Effective security after the attack (n/2 bits).
    pub effective_security: usize,
}

// ============================================================
// THREAT ASSESSMENT
// ============================================================

/// Overall threat level from quantum computing.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThreatLevel {
    /// Quantum-safe by design (hash-based, code-based with large parameters).
    Negligible,
    /// Quantum-resistant (lattice-based PQC).
    Low,
    /// Weakened but not broken (e.g. AES-128 with Grover).
    Medium,
    /// Breakable with fault-tolerant quantum computers (5-15 year horizon).
    High,
    /// Breakable with near-term quantum computers (< 5 years).
    Critical,
}

impl fmt::Display for ThreatLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThreatLevel::Critical => write!(f, "CRITICAL"),
            ThreatLevel::High => write!(f, "HIGH"),
            ThreatLevel::Medium => write!(f, "MEDIUM"),
            ThreatLevel::Low => write!(f, "LOW"),
            ThreatLevel::Negligible => write!(f, "NEGLIGIBLE"),
        }
    }
}

/// Migration urgency recommendation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MigrationUrgency {
    /// Already quantum-safe; no action required.
    None,
    /// Monitor developments but no immediate action.
    Monitor,
    /// Include in next planned system upgrade cycle.
    Planned,
    /// Migrate within 2-3 years.
    Soon,
    /// Migrate immediately (data currently at risk from harvest-now-decrypt-later).
    Immediate,
}

impl fmt::Display for MigrationUrgency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MigrationUrgency::Immediate => write!(f, "IMMEDIATE"),
            MigrationUrgency::Soon => write!(f, "SOON (2-3 years)"),
            MigrationUrgency::Planned => write!(f, "PLANNED"),
            MigrationUrgency::Monitor => write!(f, "MONITOR"),
            MigrationUrgency::None => write!(f, "NONE"),
        }
    }
}

/// Full threat assessment for a single cryptographic algorithm.
#[derive(Clone, Debug)]
pub struct ThreatAssessment {
    /// The algorithm under assessment.
    pub algorithm: CryptoAlgorithm,
    /// Classical security in bits.
    pub classical_security_bits: usize,
    /// Quantum security in bits (after best known quantum attack).
    pub quantum_security_bits: usize,
    /// The most effective quantum attack strategy.
    pub best_quantum_attack: QuantumAttack,
    /// Resource estimate for that attack.
    pub attack_estimate: QuantumAttackEstimate,
    /// Qualitative threat level.
    pub threat_level: ThreatLevel,
    /// Migration urgency recommendation.
    pub migration_urgency: MigrationUrgency,
    /// Recommended quantum-safe replacement, if applicable.
    pub recommended_replacement: Option<CryptoAlgorithm>,
}

impl fmt::Display for ThreatAssessment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Threat Assessment: {} ===", self.algorithm)?;
        writeln!(
            f,
            "Classical security: {} bits",
            self.classical_security_bits
        )?;
        writeln!(f, "Quantum security:   {} bits", self.quantum_security_bits)?;
        writeln!(f, "Best quantum attack: {}", self.best_quantum_attack)?;
        writeln!(f, "Threat level: {}", self.threat_level)?;
        writeln!(f, "Migration urgency: {}", self.migration_urgency)?;
        if let Some(ref replacement) = self.recommended_replacement {
            writeln!(f, "Recommended replacement: {}", replacement)?;
        }
        writeln!(f)?;
        write!(f, "{}", self.attack_estimate)?;
        Ok(())
    }
}

// ============================================================
// TIMELINE & COMPARISON STRUCTURES
// ============================================================

/// A single point on the quantum threat timeline.
#[derive(Clone, Debug)]
pub struct QuantumThreatTimeline {
    /// Projected year.
    pub year: usize,
    /// Estimated logical qubits available in that year.
    pub logical_qubits_available: usize,
    /// Algorithms that become breakable at this capability level.
    pub algorithms_at_risk: Vec<String>,
    /// Algorithms that remain safe.
    pub algorithms_safe: Vec<String>,
}

/// Comparison matrix for multiple algorithms.
#[derive(Clone, Debug)]
pub struct PqcComparisonMatrix {
    /// Algorithms included in the comparison.
    pub algorithms: Vec<CryptoAlgorithm>,
    /// Per-algorithm threat assessments.
    pub assessments: Vec<ThreatAssessment>,
}

impl PqcComparisonMatrix {
    /// Return algorithms sorted by quantum security (ascending -- weakest first).
    pub fn sorted_by_quantum_security(&self) -> Vec<&ThreatAssessment> {
        let mut sorted: Vec<&ThreatAssessment> = self.assessments.iter().collect();
        sorted.sort_by_key(|a| a.quantum_security_bits);
        sorted
    }

    /// Return only the algorithms whose threat level is Critical or High.
    pub fn at_risk(&self) -> Vec<&ThreatAssessment> {
        self.assessments
            .iter()
            .filter(|a| {
                a.threat_level == ThreatLevel::Critical || a.threat_level == ThreatLevel::High
            })
            .collect()
    }

    /// Return only quantum-safe algorithms (Low or Negligible threat).
    pub fn quantum_safe(&self) -> Vec<&ThreatAssessment> {
        self.assessments
            .iter()
            .filter(|a| {
                a.threat_level == ThreatLevel::Low || a.threat_level == ThreatLevel::Negligible
            })
            .collect()
    }
}

impl fmt::Display for PqcComparisonMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<30} {:>10} {:>10} {:>10} {:>12}",
            "Algorithm", "Classical", "Quantum", "Threat", "Migrate"
        )?;
        writeln!(f, "{:-<75}", "")?;
        for a in &self.assessments {
            writeln!(
                f,
                "{:<30} {:>10} {:>10} {:>10} {:>12}",
                format!("{}", a.algorithm),
                format!("{} bits", a.classical_security_bits),
                format!("{} bits", a.quantum_security_bits),
                format!("{}", a.threat_level),
                format!("{}", a.migration_urgency),
            )?;
        }
        Ok(())
    }
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for resource estimation.
#[derive(Clone, Debug)]
pub struct PqcConfig {
    /// Surface code distance for physical qubit estimation.
    pub surface_code_distance: usize,
    /// Physical gate error rate (p_phys).
    pub physical_error_rate: f64,
    /// Logical gate clock speed in MHz.
    pub clock_speed_mhz: f64,
    /// Number of magic-state distillation factories.
    pub magic_state_factory_count: usize,
}

impl Default for PqcConfig {
    fn default() -> Self {
        Self {
            surface_code_distance: 23,
            physical_error_rate: 1e-3,
            clock_speed_mhz: 10.0,
            magic_state_factory_count: 4,
        }
    }
}

impl PqcConfig {
    /// Builder: set the surface code distance.
    pub fn surface_code_distance(mut self, d: usize) -> Self {
        self.surface_code_distance = d;
        self
    }

    /// Builder: set the physical error rate.
    pub fn physical_error_rate(mut self, p: f64) -> Self {
        self.physical_error_rate = p;
        self
    }

    /// Builder: set the logical clock speed in MHz.
    pub fn clock_speed_mhz(mut self, mhz: f64) -> Self {
        self.clock_speed_mhz = mhz;
        self
    }

    /// Builder: set the magic-state factory count.
    pub fn magic_state_factory_count(mut self, n: usize) -> Self {
        self.magic_state_factory_count = n;
        self
    }

    /// Physical qubits per logical qubit: 2 * d^2 for surface code.
    pub fn physical_per_logical(&self) -> usize {
        2 * self.surface_code_distance * self.surface_code_distance
    }

    /// Total physical qubits for a given logical qubit count, including
    /// factory overhead.
    pub fn total_physical_qubits(&self, logical_qubits: usize) -> usize {
        let data_physical = logical_qubits * self.physical_per_logical();
        // Each factory uses approximately 16 * d^2 physical qubits
        // (two levels of 15-to-1 distillation).
        let factory_physical = self.magic_state_factory_count
            * 16
            * self.surface_code_distance
            * self.surface_code_distance;
        data_physical + factory_physical
    }

    /// Convert circuit depth to wall-clock hours given the clock speed.
    pub fn depth_to_hours(&self, circuit_depth: u64) -> f64 {
        let gates_per_second = self.clock_speed_mhz * 1e6;
        let seconds = circuit_depth as f64 / gates_per_second;
        seconds / 3600.0
    }
}

// ============================================================
// SHOR'S ALGORITHM RESOURCE ESTIMATION
// ============================================================

/// Estimate resources for Shor's algorithm to factor an n-bit RSA modulus.
///
/// Uses the model from Gidney & Ekera (2021):
/// - Logical qubits: 2n + 3
/// - Toffoli gates: O(n^3) for modular exponentiation (Schoolbook)
/// - With windowed arithmetic: ~0.3 * n^3
/// - Circuit depth: O(n^2 * log(n)) with parallelism
pub fn shor_rsa_estimate(key_bits: usize) -> ShorEstimate {
    let n = key_bits;
    // Gidney-Ekera (2021) optimised construction: 2n + 1 logical qubits
    // with additional ancilla for modular arithmetic we use 2n + 3.
    let logical_qubits = 2 * n + 3;

    // Toffoli count: windowed modular exponentiation.
    // From Gidney & Ekera: ~0.3 * n^3 Toffoli gates.
    let n3 = (n as u64) * (n as u64) * (n as u64);
    let toffoli_gates = (0.3 * n3 as f64) as u64;

    // One modular exponentiation per run; typically need ~2 runs.
    let modular_exponentiations = 2;

    // Circuit depth: O(n^2 * log(n)) with parallel addition chains.
    let log_n = (n as f64).log2().ceil() as u64;
    let circuit_depth = (n as u64) * (n as u64) * log_n;

    ShorEstimate {
        n_bits: n,
        logical_qubits,
        toffoli_gates,
        modular_exponentiations,
        circuit_depth,
    }
}

/// Estimate resources for Shor's algorithm to solve the discrete-log problem
/// on an elliptic curve of the given bit-size.
///
/// For ECC on a curve of order ~2^b, Shor's ECDLP variant requires:
/// - Logical qubits: ~6b + additional ancilla
/// - The circuit is more expensive per bit than RSA but the key sizes are
///   much smaller (256-bit ECC is equivalent to 3072-bit RSA).
pub fn shor_ecc_estimate(curve_bits: usize) -> ShorEstimate {
    let b = curve_bits;
    // Roetteler et al. (2017): ~6b logical qubits for ECDLP.
    // We add ancilla overhead.
    let logical_qubits = 6 * b + 10;

    // Toffoli count: elliptic curve point addition is ~448 * b^2 per
    // scalar multiplication, with ~b iterations.
    let toffoli_gates = 448_u64 * (b as u64) * (b as u64) * (b as u64);

    let modular_exponentiations = 1;
    let log_b = (b as f64).log2().ceil() as u64;
    let circuit_depth = (b as u64) * (b as u64) * log_b;

    ShorEstimate {
        n_bits: b,
        logical_qubits,
        toffoli_gates,
        modular_exponentiations,
        circuit_depth,
    }
}

/// Estimate resources for Shor's algorithm on a Diffie-Hellman or DSA
/// modular discrete-log problem with the given modulus bit-size.
pub fn shor_dh_dsa_estimate(key_bits: usize) -> ShorEstimate {
    // The DLP in Z*_p is structurally similar to factoring.
    // Slightly more overhead than RSA factoring.
    let n = key_bits;
    let logical_qubits = 2 * n + 5;
    let n3 = (n as u64) * (n as u64) * (n as u64);
    let toffoli_gates = (0.35 * n3 as f64) as u64;
    let modular_exponentiations = 2;
    let log_n = (n as f64).log2().ceil() as u64;
    let circuit_depth = (n as u64) * (n as u64) * log_n;

    ShorEstimate {
        n_bits: n,
        logical_qubits,
        toffoli_gates,
        modular_exponentiations,
        circuit_depth,
    }
}

// ============================================================
// GROVER'S ALGORITHM RESOURCE ESTIMATION
// ============================================================

/// Estimate resources for Grover's search on a symmetric key space.
///
/// Grover's algorithm provides a quadratic speed-up: searching a space
/// of size 2^n requires O(2^(n/2)) oracle queries, each of which
/// implements the cipher in a reversible quantum circuit.
pub fn grover_estimate(key_bits: usize) -> GroverEstimate {
    let effective_security = key_bits / 2;
    // The number of Grover queries is pi/4 * 2^(n/2).
    // For key_bits <= 128, we can represent this exactly.
    // For larger keys we cap at u64::MAX.
    let quantum_queries = if effective_security >= 64 {
        u64::MAX
    } else {
        let base = 1u64 << effective_security;
        // pi/4 * 2^(n/2) -- truncated to integer
        ((std::f64::consts::FRAC_PI_4) * base as f64) as u64
    };

    GroverEstimate {
        key_bits,
        quantum_queries,
        effective_security,
    }
}

/// Convert a Grover estimate into a full attack estimate with physical
/// resource costs.
///
/// The oracle circuit for AES requires ~n + 128 logical qubits (key register
/// plus working space for the cipher evaluation).
pub fn grover_attack_estimate(key_bits: usize, config: &PqcConfig) -> QuantumAttackEstimate {
    let grover = grover_estimate(key_bits);

    // AES oracle circuit: key register + plaintext + ancilla.
    // Grassl et al. (2016): AES-128 requires ~2953 qubits, ~2^86 T-gates
    // per Grover oracle call. We approximate.
    let logical_qubits = match key_bits {
        128 => 2953,
        192 => 4449,
        256 => 6681,
        _ => key_bits * 23 + 9, // rough approximation
    };

    // T-gate count per oracle call * number of queries.
    // AES-128: ~2^40 T-gates per oracle evaluation.
    let t_per_oracle: u64 = 1u64 << (key_bits.min(60) / 3);
    let t_gates = t_per_oracle.saturating_mul(grover.quantum_queries);
    let toffoli_gates = t_gates / 4; // approximate

    // Depth: each oracle is O(key_bits * rounds) deep; AES-128 has 10 rounds.
    let oracle_depth = (key_bits as u64) * 10;
    let circuit_depth = oracle_depth.saturating_mul(grover.quantum_queries);

    let surface_code_qubits = config.total_physical_qubits(logical_qubits);
    let wall_clock_hours = config.depth_to_hours(circuit_depth);

    QuantumAttackEstimate {
        algorithm: format!("AES-{}", key_bits),
        attack: "Grover".to_string(),
        logical_qubits,
        t_gates,
        toffoli_gates,
        circuit_depth,
        surface_code_qubits,
        wall_clock_hours,
        success_probability: 0.5, // single Grover iteration success
        security_level_bits: grover.effective_security,
    }
}

// ============================================================
// LATTICE ATTACK ESTIMATION
// ============================================================

/// Estimate the cost of quantum lattice sieving (quantum BKZ) against a
/// lattice of the given dimension.
///
/// The core-SVP model estimates:
///   classical: T = 2^(0.292 * beta)
///   quantum:   T = 2^(0.265 * beta)
///
/// where beta is the BKZ block size needed to solve the lattice problem.
/// For MLWE/MLWR instances, beta ~ dimension * noise-ratio (simplified).
pub fn lattice_security_estimate(
    lattice_dimension: usize,
    security_level: usize,
) -> (usize, usize) {
    // Estimate the BKZ block size beta from the lattice dimension.
    // For ML-KEM parameters, the dimension-to-beta mapping is:
    // beta ~ 0.48 * dimension for level 1, ~0.45 for level 3, ~0.43 for level 5.
    let beta_ratio = match security_level {
        1 => 0.48,
        2 => 0.46,
        3 => 0.45,
        5 => 0.43,
        _ => 0.46,
    };
    let beta = (beta_ratio * lattice_dimension as f64) as usize;

    let classical_bits = (0.292 * beta as f64) as usize;
    let quantum_bits = (0.265 * beta as f64) as usize;

    (classical_bits, quantum_bits)
}

/// Estimate quantum resources for a lattice attack at the given security
/// level (in quantum bits).
pub fn lattice_attack_estimate(
    algorithm_name: &str,
    lattice_dimension: usize,
    quantum_security_bits: usize,
    config: &PqcConfig,
) -> QuantumAttackEstimate {
    // Quantum lattice sieving requires approximately 2^(0.2075 * beta)
    // qubits for the sieving subroutine (Laarhoven 2015).
    // We approximate the logical qubit count.
    let beta = (quantum_security_bits as f64 / 0.265) as usize;
    let logical_qubits = (2.0f64.powf(0.2075 * beta as f64).min(1e12)) as usize;
    let logical_qubits = logical_qubits.max(1000); // floor for realistic estimates

    // Gate count scales exponentially with security.
    let t_gates = if quantum_security_bits < 64 {
        1u64 << quantum_security_bits
    } else {
        u64::MAX
    };
    let toffoli_gates = t_gates / 2;

    let circuit_depth = t_gates / lattice_dimension.max(1) as u64;
    let surface_code_qubits = config.total_physical_qubits(logical_qubits.min(1_000_000));
    let wall_clock_hours = config.depth_to_hours(circuit_depth);

    QuantumAttackEstimate {
        algorithm: algorithm_name.to_string(),
        attack: "Quantum-BKZ".to_string(),
        logical_qubits,
        t_gates,
        toffoli_gates,
        circuit_depth,
        surface_code_qubits,
        wall_clock_hours,
        success_probability: 0.99, // Lattice attacks are deterministic once resources suffice
        security_level_bits: quantum_security_bits,
    }
}

// ============================================================
// CODE-BASED ATTACK ESTIMATION (ISD)
// ============================================================

/// Estimate quantum Information Set Decoding cost for a code-based scheme.
///
/// Classical ISD (Prange): O(2^(t * H(k/n)))
/// Quantum ISD (Bernstein): quadratic speed-up on inner loops, ~sqrt factor.
///
/// For Classic McEliece and HQC, the quantum ISD attack is the dominant
/// threat but still far too expensive to execute.
pub fn isd_attack_estimate(
    algorithm_name: &str,
    code_length: usize,
    error_capability: usize,
    classical_security_bits: usize,
    config: &PqcConfig,
) -> QuantumAttackEstimate {
    // Quantum ISD gives roughly a sqrt speed-up on the work factor
    // of the best classical ISD algorithm.
    // Classical cost: ~2^(classical_security_bits)
    // Quantum cost: ~2^(classical_security_bits * 0.72) -- from detailed analyses
    // (The reduction is not exactly sqrt because ISD has a search + linear
    //  algebra structure that limits quantum speed-up.)
    let quantum_security_bits = (classical_security_bits as f64 * 0.72) as usize;

    // Logical qubits: need to store the parity check matrix and syndrome
    // in superposition. Approximately code_length qubits.
    let logical_qubits = code_length + error_capability * 2;

    let t_gates = if quantum_security_bits < 64 {
        1u64 << quantum_security_bits
    } else {
        u64::MAX
    };
    let toffoli_gates = t_gates / 3;
    let circuit_depth = t_gates / code_length.max(1) as u64;

    let surface_code_qubits = config.total_physical_qubits(logical_qubits.min(1_000_000));
    let wall_clock_hours = config.depth_to_hours(circuit_depth);

    QuantumAttackEstimate {
        algorithm: algorithm_name.to_string(),
        attack: "Quantum-ISD".to_string(),
        logical_qubits,
        t_gates,
        toffoli_gates,
        circuit_depth,
        surface_code_qubits,
        wall_clock_hours,
        success_probability: 0.5,
        security_level_bits: quantum_security_bits,
    }
}

// ============================================================
// SHOR ATTACK ESTIMATE (FULL)
// ============================================================

/// Build a full QuantumAttackEstimate from a ShorEstimate.
fn shor_to_attack_estimate(
    algorithm_name: &str,
    shor: &ShorEstimate,
    config: &PqcConfig,
) -> QuantumAttackEstimate {
    let t_gates = shor.toffoli_gates * 4; // Each Toffoli decomposes to ~4 T gates
    let surface_code_qubits = config.total_physical_qubits(shor.logical_qubits);
    let wall_clock_hours = config.depth_to_hours(shor.circuit_depth);

    QuantumAttackEstimate {
        algorithm: algorithm_name.to_string(),
        attack: "Shor".to_string(),
        logical_qubits: shor.logical_qubits,
        t_gates,
        toffoli_gates: shor.toffoli_gates,
        circuit_depth: shor.circuit_depth,
        surface_code_qubits,
        wall_clock_hours,
        success_probability: 0.67, // Single-shot success ~2/3
        security_level_bits: 0,    // Algorithm is fully broken
    }
}

// ============================================================
// NIST SECURITY LEVEL MAPPING
// ============================================================

/// Map a NIST security level (1-5) to equivalent AES key bits.
pub fn nist_level_to_aes_bits(level: usize) -> usize {
    match level {
        1 => 128,
        2 => 128, // Level 2 is between 1 and 3, maps to AES-128 collision
        3 => 192,
        4 => 192,
        5 => 256,
        _ => 128,
    }
}

/// Map a NIST security level (1-5) to equivalent classical security bits.
pub fn nist_level_to_classical_bits(level: usize) -> usize {
    nist_level_to_aes_bits(level)
}

/// Map a NIST security level to quantum security bits (after Grover).
pub fn nist_level_to_quantum_bits(level: usize) -> usize {
    nist_level_to_aes_bits(level) / 2
}

// ============================================================
// MAIN ASSESSMENT FUNCTION
// ============================================================

/// Assess a single cryptographic algorithm for quantum vulnerability.
///
/// This is the primary entry point. It selects the best quantum attack,
/// estimates the resources required, and produces a full threat assessment
/// with migration recommendations.
pub fn assess_algorithm(algorithm: &CryptoAlgorithm, config: &PqcConfig) -> ThreatAssessment {
    match algorithm {
        // ---- Pre-quantum public-key (Shor's algorithm) ----
        CryptoAlgorithm::RSA { key_bits } => {
            let shor = shor_rsa_estimate(*key_bits);
            let estimate = shor_to_attack_estimate(&format!("RSA-{}", key_bits), &shor, config);
            let replacement_level = if *key_bits <= 3072 {
                MLKEMParams::ML768
            } else {
                MLKEMParams::ML1024
            };
            ThreatAssessment {
                algorithm: algorithm.clone(),
                classical_security_bits: rsa_classical_security(*key_bits),
                quantum_security_bits: 0,
                best_quantum_attack: QuantumAttack::Shor,
                attack_estimate: estimate,
                threat_level: ThreatLevel::Critical,
                migration_urgency: MigrationUrgency::Immediate,
                recommended_replacement: Some(CryptoAlgorithm::MLKEM {
                    parameter_set: replacement_level,
                }),
            }
        }
        CryptoAlgorithm::ECC { curve_bits } => {
            let shor = shor_ecc_estimate(*curve_bits);
            let estimate = shor_to_attack_estimate(&format!("ECC-{}", curve_bits), &shor, config);
            let replacement_level = if *curve_bits <= 256 {
                MLDSAParams::ML65
            } else {
                MLDSAParams::ML87
            };
            ThreatAssessment {
                algorithm: algorithm.clone(),
                classical_security_bits: ecc_classical_security(*curve_bits),
                quantum_security_bits: 0,
                best_quantum_attack: QuantumAttack::Shor,
                attack_estimate: estimate,
                threat_level: ThreatLevel::Critical,
                migration_urgency: MigrationUrgency::Immediate,
                recommended_replacement: Some(CryptoAlgorithm::MLDSA {
                    parameter_set: replacement_level,
                }),
            }
        }
        CryptoAlgorithm::DH { key_bits } => {
            let shor = shor_dh_dsa_estimate(*key_bits);
            let estimate = shor_to_attack_estimate(&format!("DH-{}", key_bits), &shor, config);
            ThreatAssessment {
                algorithm: algorithm.clone(),
                classical_security_bits: dh_classical_security(*key_bits),
                quantum_security_bits: 0,
                best_quantum_attack: QuantumAttack::Shor,
                attack_estimate: estimate,
                threat_level: ThreatLevel::Critical,
                migration_urgency: MigrationUrgency::Immediate,
                recommended_replacement: Some(CryptoAlgorithm::MLKEM {
                    parameter_set: MLKEMParams::ML768,
                }),
            }
        }
        CryptoAlgorithm::DSA { key_bits } => {
            let shor = shor_dh_dsa_estimate(*key_bits);
            let estimate = shor_to_attack_estimate(&format!("DSA-{}", key_bits), &shor, config);
            ThreatAssessment {
                algorithm: algorithm.clone(),
                classical_security_bits: dh_classical_security(*key_bits),
                quantum_security_bits: 0,
                best_quantum_attack: QuantumAttack::Shor,
                attack_estimate: estimate,
                threat_level: ThreatLevel::Critical,
                migration_urgency: MigrationUrgency::Immediate,
                recommended_replacement: Some(CryptoAlgorithm::MLDSA {
                    parameter_set: MLDSAParams::ML65,
                }),
            }
        }

        // ---- Symmetric ciphers (Grover) ----
        CryptoAlgorithm::AES { key_bits } => {
            let estimate = grover_attack_estimate(*key_bits, config);
            let grover = grover_estimate(*key_bits);
            let threat_level = if grover.effective_security < 80 {
                ThreatLevel::Medium
            } else {
                ThreatLevel::Low
            };
            let urgency = if *key_bits <= 128 {
                MigrationUrgency::Soon
            } else {
                MigrationUrgency::Monitor
            };
            let replacement = if *key_bits < 256 {
                Some(CryptoAlgorithm::AES { key_bits: 256 })
            } else {
                None
            };
            ThreatAssessment {
                algorithm: algorithm.clone(),
                classical_security_bits: *key_bits,
                quantum_security_bits: grover.effective_security,
                best_quantum_attack: QuantumAttack::Grover,
                attack_estimate: estimate,
                threat_level,
                migration_urgency: urgency,
                recommended_replacement: replacement,
            }
        }

        // ---- NIST PQC: Lattice-based ----
        CryptoAlgorithm::MLKEM { parameter_set } => {
            let q_sec = parameter_set.quantum_security_bits();
            let estimate = lattice_attack_estimate(
                &format!("{}", parameter_set),
                parameter_set.lattice_dimension(),
                q_sec,
                config,
            );
            ThreatAssessment {
                algorithm: algorithm.clone(),
                classical_security_bits: parameter_set.classical_security_bits(),
                quantum_security_bits: q_sec,
                best_quantum_attack: QuantumAttack::QuantumLattice,
                attack_estimate: estimate,
                threat_level: ThreatLevel::Low,
                migration_urgency: MigrationUrgency::None,
                recommended_replacement: None,
            }
        }
        CryptoAlgorithm::MLDSA { parameter_set } => {
            let q_sec = parameter_set.quantum_security_bits();
            let estimate = lattice_attack_estimate(
                &format!("{}", parameter_set),
                parameter_set.lattice_dimension(),
                q_sec,
                config,
            );
            ThreatAssessment {
                algorithm: algorithm.clone(),
                classical_security_bits: parameter_set.classical_security_bits(),
                quantum_security_bits: q_sec,
                best_quantum_attack: QuantumAttack::QuantumLattice,
                attack_estimate: estimate,
                threat_level: ThreatLevel::Low,
                migration_urgency: MigrationUrgency::None,
                recommended_replacement: None,
            }
        }
        CryptoAlgorithm::NTRU { parameter_set } => {
            let q_sec = parameter_set.quantum_security_bits();
            let estimate = lattice_attack_estimate(
                &format!("{}", parameter_set),
                parameter_set.lattice_dimension(),
                q_sec,
                config,
            );
            ThreatAssessment {
                algorithm: algorithm.clone(),
                classical_security_bits: parameter_set.classical_security_bits(),
                quantum_security_bits: q_sec,
                best_quantum_attack: QuantumAttack::QuantumLattice,
                attack_estimate: estimate,
                threat_level: ThreatLevel::Low,
                migration_urgency: MigrationUrgency::None,
                recommended_replacement: None,
            }
        }

        // ---- NIST PQC: Hash-based ----
        CryptoAlgorithm::SLHDSA { parameter_set } => {
            // Hash-based signatures rely only on the security of the hash
            // function. The only quantum threat is Grover's algorithm applied
            // to preimage/collision finding, but the parameters are chosen to
            // maintain adequate security post-Grover.
            let classical_bits = parameter_set.classical_security_bits();
            let quantum_bits = parameter_set.quantum_security_bits();
            let grover = grover_attack_estimate(classical_bits, config);
            ThreatAssessment {
                algorithm: algorithm.clone(),
                classical_security_bits: classical_bits,
                quantum_security_bits: quantum_bits,
                best_quantum_attack: QuantumAttack::Grover,
                attack_estimate: grover,
                threat_level: ThreatLevel::Negligible,
                migration_urgency: MigrationUrgency::None,
                recommended_replacement: None,
            }
        }

        // ---- NIST PQC: Code-based ----
        CryptoAlgorithm::HQC { parameter_set } => {
            let classical_bits = parameter_set.classical_security_bits();
            let quantum_bits = parameter_set.quantum_security_bits();
            let estimate = isd_attack_estimate(
                &format!("{}", parameter_set),
                parameter_set.code_length(),
                parameter_set.code_length() / 50, // approximate error capability
                classical_bits,
                config,
            );
            ThreatAssessment {
                algorithm: algorithm.clone(),
                classical_security_bits: classical_bits,
                quantum_security_bits: quantum_bits,
                best_quantum_attack: QuantumAttack::ISD { quantum: true },
                attack_estimate: estimate,
                threat_level: ThreatLevel::Low,
                migration_urgency: MigrationUrgency::None,
                recommended_replacement: None,
            }
        }
        CryptoAlgorithm::BIKE { parameter_set } => {
            let classical_bits = parameter_set.classical_security_bits();
            let quantum_bits = parameter_set.quantum_security_bits();
            let estimate = isd_attack_estimate(
                &format!("{}", parameter_set),
                parameter_set.block_length() * 2,
                parameter_set.block_length() / 100,
                classical_bits,
                config,
            );
            ThreatAssessment {
                algorithm: algorithm.clone(),
                classical_security_bits: classical_bits,
                quantum_security_bits: quantum_bits,
                best_quantum_attack: QuantumAttack::ISD { quantum: true },
                attack_estimate: estimate,
                threat_level: ThreatLevel::Low,
                migration_urgency: MigrationUrgency::None,
                recommended_replacement: None,
            }
        }
        CryptoAlgorithm::ClassicMcEliece { parameter_set } => {
            let classical_bits = parameter_set.classical_security_bits();
            let quantum_bits = parameter_set.quantum_security_bits();
            let estimate = isd_attack_estimate(
                &format!("{}", parameter_set),
                parameter_set.code_length(),
                parameter_set.error_capability(),
                classical_bits,
                config,
            );
            ThreatAssessment {
                algorithm: algorithm.clone(),
                classical_security_bits: classical_bits,
                quantum_security_bits: quantum_bits,
                best_quantum_attack: QuantumAttack::ISD { quantum: true },
                attack_estimate: estimate,
                threat_level: ThreatLevel::Negligible,
                migration_urgency: MigrationUrgency::None,
                recommended_replacement: None,
            }
        }
    }
}

// ============================================================
// CLASSICAL SECURITY HELPERS
// ============================================================

/// Approximate classical security of RSA in bits (NIST SP 800-57).
fn rsa_classical_security(key_bits: usize) -> usize {
    match key_bits {
        0..=1023 => 40,
        1024..=2047 => 80,
        2048..=3071 => 112,
        3072..=4095 => 128,
        4096..=7679 => 152,
        7680..=15359 => 192,
        _ => 256,
    }
}

/// Approximate classical security of ECC in bits.
fn ecc_classical_security(curve_bits: usize) -> usize {
    // ECC security is approximately curve_bits / 2.
    curve_bits / 2
}

/// Approximate classical security of DH/DSA in bits (same as RSA).
fn dh_classical_security(key_bits: usize) -> usize {
    rsa_classical_security(key_bits)
}

// ============================================================
// COMPARISON MATRIX BUILDER
// ============================================================

/// Build a comparison matrix for a set of cryptographic algorithms.
pub fn build_comparison_matrix(
    algorithms: &[CryptoAlgorithm],
    config: &PqcConfig,
) -> PqcComparisonMatrix {
    let assessments: Vec<ThreatAssessment> = algorithms
        .iter()
        .map(|a| assess_algorithm(a, config))
        .collect();

    PqcComparisonMatrix {
        algorithms: algorithms.to_vec(),
        assessments,
    }
}

/// Build a comprehensive comparison matrix covering all major algorithm
/// families at typical parameter choices.
pub fn build_comprehensive_matrix(config: &PqcConfig) -> PqcComparisonMatrix {
    let algorithms = vec![
        // Pre-quantum
        CryptoAlgorithm::RSA { key_bits: 2048 },
        CryptoAlgorithm::RSA { key_bits: 4096 },
        CryptoAlgorithm::ECC { curve_bits: 256 },
        CryptoAlgorithm::ECC { curve_bits: 384 },
        CryptoAlgorithm::DH { key_bits: 2048 },
        CryptoAlgorithm::DSA { key_bits: 3072 },
        // Symmetric
        CryptoAlgorithm::AES { key_bits: 128 },
        CryptoAlgorithm::AES { key_bits: 256 },
        // NIST PQC
        CryptoAlgorithm::MLKEM {
            parameter_set: MLKEMParams::ML512,
        },
        CryptoAlgorithm::MLKEM {
            parameter_set: MLKEMParams::ML768,
        },
        CryptoAlgorithm::MLKEM {
            parameter_set: MLKEMParams::ML1024,
        },
        CryptoAlgorithm::MLDSA {
            parameter_set: MLDSAParams::ML44,
        },
        CryptoAlgorithm::MLDSA {
            parameter_set: MLDSAParams::ML65,
        },
        CryptoAlgorithm::MLDSA {
            parameter_set: MLDSAParams::ML87,
        },
        CryptoAlgorithm::SLHDSA {
            parameter_set: SLHDSAParams::SHA2_128s,
        },
        CryptoAlgorithm::SLHDSA {
            parameter_set: SLHDSAParams::SHA2_256f,
        },
        CryptoAlgorithm::HQC {
            parameter_set: HQCParams::HQC128,
        },
        CryptoAlgorithm::HQC {
            parameter_set: HQCParams::HQC256,
        },
        CryptoAlgorithm::NTRU {
            parameter_set: NTRUParams::NTRU677,
        },
        CryptoAlgorithm::BIKE {
            parameter_set: BIKEParams::BIKE3,
        },
        CryptoAlgorithm::ClassicMcEliece {
            parameter_set: McElieceParams::MC6688128,
        },
    ];

    build_comparison_matrix(&algorithms, config)
}

// ============================================================
// THREAT TIMELINE PROJECTION
// ============================================================

/// Hardware milestone: projected year and available logical qubits.
struct HardwareMilestone {
    year: usize,
    logical_qubits: usize,
}

/// Generate hardware milestones from public roadmaps.
///
/// These are consensus estimates from IBM, Google, Microsoft, and IonQ
/// roadmaps as of early 2025.
fn hardware_roadmap() -> Vec<HardwareMilestone> {
    vec![
        HardwareMilestone {
            year: 2025,
            logical_qubits: 10,
        },
        HardwareMilestone {
            year: 2027,
            logical_qubits: 100,
        },
        HardwareMilestone {
            year: 2029,
            logical_qubits: 1_000,
        },
        HardwareMilestone {
            year: 2031,
            logical_qubits: 4_000,
        },
        HardwareMilestone {
            year: 2033,
            logical_qubits: 10_000,
        },
        HardwareMilestone {
            year: 2035,
            logical_qubits: 20_000,
        },
        HardwareMilestone {
            year: 2040,
            logical_qubits: 100_000,
        },
        HardwareMilestone {
            year: 2045,
            logical_qubits: 1_000_000,
        },
    ]
}

/// Key algorithms and the logical qubit count needed to break them.
fn algorithm_qubit_requirements() -> Vec<(String, usize)> {
    vec![
        ("RSA-2048".to_string(), 2 * 2048 + 3), // ~4099
        ("RSA-4096".to_string(), 2 * 4096 + 3), // ~8195
        ("ECC-256".to_string(), 6 * 256 + 10),  // ~1546
        ("ECC-384".to_string(), 6 * 384 + 10),  // ~2314
        ("DH-2048".to_string(), 2 * 2048 + 5),  // ~4101
        ("AES-128".to_string(), 2953),          // Grover oracle
        ("AES-256".to_string(), 6681),          // Grover oracle (infeasible queries though)
    ]
}

/// Project the quantum threat timeline.
///
/// For each hardware milestone year, determines which algorithms become
/// breakable (have enough logical qubits) and which remain safe. Note that
/// having enough qubits is necessary but not sufficient -- circuit depth
/// and error rates also matter. This function uses qubits as a proxy.
pub fn project_threat_timeline() -> Vec<QuantumThreatTimeline> {
    let milestones = hardware_roadmap();
    let requirements = algorithm_qubit_requirements();

    // PQC algorithms are always safe in this model.
    let pqc_safe = vec![
        "ML-KEM-768".to_string(),
        "ML-DSA-65".to_string(),
        "SLH-DSA-SHA2-256f".to_string(),
        "HQC-256".to_string(),
        "Classic-McEliece-6688128".to_string(),
    ];

    milestones
        .iter()
        .map(|m| {
            let at_risk: Vec<String> = requirements
                .iter()
                .filter(|(_, qubits)| *qubits <= m.logical_qubits)
                .map(|(name, _)| name.clone())
                .collect();

            let mut safe: Vec<String> = requirements
                .iter()
                .filter(|(_, qubits)| *qubits > m.logical_qubits)
                .map(|(name, _)| name.clone())
                .collect();
            safe.extend(pqc_safe.clone());

            QuantumThreatTimeline {
                year: m.year,
                logical_qubits_available: m.logical_qubits,
                algorithms_at_risk: at_risk,
                algorithms_safe: safe,
            }
        })
        .collect()
}

/// Estimate the year when a specific algorithm becomes breakable, based
/// on the projected hardware roadmap.
///
/// Returns `None` if the algorithm is projected to remain safe beyond 2045.
pub fn estimate_break_year(algorithm: &CryptoAlgorithm, config: &PqcConfig) -> Option<usize> {
    let assessment = assess_algorithm(algorithm, config);

    // If it is already quantum-safe, it will not be broken.
    if assessment.threat_level == ThreatLevel::Low
        || assessment.threat_level == ThreatLevel::Negligible
    {
        return None;
    }

    // For Grover-based attacks (AES), even though the qubit count is
    // achievable, the number of sequential queries (2^64 or 2^128) makes
    // the attack infeasible in practice for AES-256.
    if let CryptoAlgorithm::AES { key_bits } = algorithm {
        if *key_bits >= 256 {
            return None;
        }
    }

    let logical_needed = assessment.attack_estimate.logical_qubits;
    let milestones = hardware_roadmap();

    for m in &milestones {
        if m.logical_qubits >= logical_needed {
            return Some(m.year);
        }
    }

    // Beyond the roadmap.
    None
}

// ============================================================
// MIGRATION RECOMMENDATION ENGINE
// ============================================================

/// Migration recommendation for a specific algorithm.
#[derive(Clone, Debug)]
pub struct MigrationRecommendation {
    /// The algorithm being migrated from.
    pub from: CryptoAlgorithm,
    /// The recommended replacement.
    pub to: CryptoAlgorithm,
    /// Urgency level.
    pub urgency: MigrationUrgency,
    /// Rationale.
    pub rationale: String,
    /// Estimated break year for the original algorithm.
    pub estimated_break_year: Option<usize>,
}

/// Generate migration recommendations for a set of algorithms.
pub fn generate_migration_plan(
    algorithms: &[CryptoAlgorithm],
    config: &PqcConfig,
) -> Vec<MigrationRecommendation> {
    algorithms
        .iter()
        .filter_map(|algo| {
            let assessment = assess_algorithm(algo, config);
            assessment.recommended_replacement.map(|replacement| {
                let break_year = estimate_break_year(algo, config);
                MigrationRecommendation {
                    from: algo.clone(),
                    to: replacement,
                    urgency: assessment.migration_urgency,
                    rationale: format!(
                        "{} has {} quantum security (threat: {}). \
                         Harvest-now-decrypt-later risk applies to all data \
                         with long-term confidentiality requirements.",
                        algo,
                        if assessment.quantum_security_bits == 0 {
                            "ZERO".to_string()
                        } else {
                            format!("{}-bit", assessment.quantum_security_bits)
                        },
                        assessment.threat_level,
                    ),
                    estimated_break_year: break_year,
                }
            })
        })
        .collect()
}

// ============================================================
// UTILITY: KEY SIZE VALIDATION
// ============================================================

/// Validate that a key size is sensible for the given algorithm.
pub fn validate_key_size(algorithm: &CryptoAlgorithm) -> Result<(), PqcError> {
    match algorithm {
        CryptoAlgorithm::RSA { key_bits } => {
            if *key_bits < 512 || *key_bits > 65536 {
                Err(PqcError::InvalidKeySize(*key_bits))
            } else {
                Ok(())
            }
        }
        CryptoAlgorithm::ECC { curve_bits } => {
            if *curve_bits < 128 || *curve_bits > 1024 {
                Err(PqcError::InvalidKeySize(*curve_bits))
            } else {
                Ok(())
            }
        }
        CryptoAlgorithm::DH { key_bits } | CryptoAlgorithm::DSA { key_bits } => {
            if *key_bits < 512 || *key_bits > 65536 {
                Err(PqcError::InvalidKeySize(*key_bits))
            } else {
                Ok(())
            }
        }
        CryptoAlgorithm::AES { key_bits } => {
            if *key_bits != 128 && *key_bits != 192 && *key_bits != 256 {
                Err(PqcError::InvalidKeySize(*key_bits))
            } else {
                Ok(())
            }
        }
        _ => Ok(()), // PQC algorithms have fixed parameter sets
    }
}

// ============================================================
// SUMMARY REPORT GENERATION
// ============================================================

/// Generate a human-readable summary report for a set of assessments.
pub fn generate_report(matrix: &PqcComparisonMatrix) -> String {
    let mut report = String::new();

    report.push_str("========================================\n");
    report.push_str(" Post-Quantum Cryptography Assessment\n");
    report.push_str("========================================\n\n");

    // Critical threats
    let critical: Vec<&ThreatAssessment> = matrix
        .assessments
        .iter()
        .filter(|a| a.threat_level == ThreatLevel::Critical)
        .collect();

    if !critical.is_empty() {
        report.push_str("CRITICAL THREATS (migrate immediately):\n");
        for a in &critical {
            report.push_str(&format!(
                "  - {} (classical: {} bits, quantum: {} bits)\n",
                a.algorithm, a.classical_security_bits, a.quantum_security_bits
            ));
            if let Some(ref r) = a.recommended_replacement {
                report.push_str(&format!("    Replacement: {}\n", r));
            }
        }
        report.push('\n');
    }

    // Quantum-safe
    let safe: Vec<&ThreatAssessment> = matrix
        .assessments
        .iter()
        .filter(|a| a.threat_level == ThreatLevel::Low || a.threat_level == ThreatLevel::Negligible)
        .collect();

    if !safe.is_empty() {
        report.push_str("QUANTUM-SAFE ALGORITHMS:\n");
        for a in &safe {
            report.push_str(&format!(
                "  - {} (quantum security: {} bits, threat: {})\n",
                a.algorithm, a.quantum_security_bits, a.threat_level
            ));
        }
        report.push('\n');
    }

    // Timeline
    let timeline = project_threat_timeline();
    report.push_str("THREAT TIMELINE:\n");
    for t in &timeline {
        if !t.algorithms_at_risk.is_empty() {
            report.push_str(&format!(
                "  {} ({} logical qubits): {} breakable\n",
                t.year,
                t.logical_qubits_available,
                t.algorithms_at_risk.join(", ")
            ));
        }
    }

    report
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PqcConfig {
        PqcConfig::default()
    }

    // ---- Shor's algorithm resource tests ----

    #[test]
    fn test_rsa_2048_shor_logical_qubits() {
        let shor = shor_rsa_estimate(2048);
        // 2n + 3 = 4099
        assert_eq!(shor.logical_qubits, 4099);
        assert_eq!(shor.n_bits, 2048);
    }

    #[test]
    fn test_rsa_2048_t_gate_count_order_of_magnitude() {
        let shor = shor_rsa_estimate(2048);
        // Toffoli gates should be in the billions for 2048-bit RSA.
        // 0.3 * 2048^3 = ~2.58e9
        assert!(shor.toffoli_gates > 1_000_000_000);
        assert!(shor.toffoli_gates < 10_000_000_000);
        // T gates = 4 * Toffoli
        let t_gates = shor.toffoli_gates * 4;
        assert!(t_gates > 4_000_000_000);
    }

    #[test]
    fn test_rsa_4096_more_resources_than_2048() {
        let shor_2048 = shor_rsa_estimate(2048);
        let shor_4096 = shor_rsa_estimate(4096);
        assert!(shor_4096.logical_qubits > shor_2048.logical_qubits);
        assert!(shor_4096.toffoli_gates > shor_2048.toffoli_gates);
        assert!(shor_4096.circuit_depth > shor_2048.circuit_depth);
    }

    #[test]
    fn test_ecc_256_fewer_qubits_than_rsa() {
        let shor_rsa = shor_rsa_estimate(2048);
        let shor_ecc = shor_ecc_estimate(256);
        // ECC-256 needs ~1546 logical qubits vs RSA-2048 ~4099
        assert!(shor_ecc.logical_qubits < shor_rsa.logical_qubits);
    }

    // ---- Grover tests ----

    #[test]
    fn test_aes_128_grover_64_bit_security() {
        let grover = grover_estimate(128);
        assert_eq!(grover.effective_security, 64);
        assert_eq!(grover.key_bits, 128);
    }

    #[test]
    fn test_aes_256_grover_128_bit_security() {
        let grover = grover_estimate(256);
        assert_eq!(grover.effective_security, 128);
        assert_eq!(grover.key_bits, 256);
    }

    // ---- PQC algorithm assessments ----

    #[test]
    fn test_mlkem_768_quantum_security_low() {
        let config = default_config();
        let assessment = assess_algorithm(
            &CryptoAlgorithm::MLKEM {
                parameter_set: MLKEMParams::ML768,
            },
            &config,
        );
        assert_eq!(assessment.threat_level, ThreatLevel::Low);
        assert!(assessment.quantum_security_bits >= 128);
    }

    #[test]
    fn test_mldsa_65_quantum_security_low() {
        let config = default_config();
        let assessment = assess_algorithm(
            &CryptoAlgorithm::MLDSA {
                parameter_set: MLDSAParams::ML65,
            },
            &config,
        );
        assert_eq!(assessment.threat_level, ThreatLevel::Low);
        assert!(assessment.quantum_security_bits >= 128);
    }

    #[test]
    fn test_slhdsa_quantum_security_negligible() {
        let config = default_config();
        let assessment = assess_algorithm(
            &CryptoAlgorithm::SLHDSA {
                parameter_set: SLHDSAParams::SHA2_256f,
            },
            &config,
        );
        assert_eq!(assessment.threat_level, ThreatLevel::Negligible);
        assert!(assessment.quantum_security_bits >= 64);
    }

    #[test]
    fn test_hqc_128_quantum_security() {
        let config = default_config();
        let assessment = assess_algorithm(
            &CryptoAlgorithm::HQC {
                parameter_set: HQCParams::HQC128,
            },
            &config,
        );
        assert_eq!(assessment.threat_level, ThreatLevel::Low);
        assert!(assessment.quantum_security_bits >= 100);
    }

    #[test]
    fn test_classic_mceliece_quantum_security_negligible() {
        let config = default_config();
        let assessment = assess_algorithm(
            &CryptoAlgorithm::ClassicMcEliece {
                parameter_set: McElieceParams::MC6688128,
            },
            &config,
        );
        assert_eq!(assessment.threat_level, ThreatLevel::Negligible);
        assert!(assessment.quantum_security_bits >= 200);
    }

    // ---- Threat levels ----

    #[test]
    fn test_threat_level_rsa_critical() {
        let config = default_config();
        let assessment = assess_algorithm(&CryptoAlgorithm::RSA { key_bits: 2048 }, &config);
        assert_eq!(assessment.threat_level, ThreatLevel::Critical);
    }

    #[test]
    fn test_threat_level_mlkem_low() {
        let config = default_config();
        let assessment = assess_algorithm(
            &CryptoAlgorithm::MLKEM {
                parameter_set: MLKEMParams::ML768,
            },
            &config,
        );
        assert_eq!(assessment.threat_level, ThreatLevel::Low);
    }

    // ---- Migration urgency ----

    #[test]
    fn test_migration_urgency_rsa_immediate() {
        let config = default_config();
        let assessment = assess_algorithm(&CryptoAlgorithm::RSA { key_bits: 2048 }, &config);
        assert_eq!(assessment.migration_urgency, MigrationUrgency::Immediate);
    }

    #[test]
    fn test_migration_urgency_mlkem_none() {
        let config = default_config();
        let assessment = assess_algorithm(
            &CryptoAlgorithm::MLKEM {
                parameter_set: MLKEMParams::ML768,
            },
            &config,
        );
        assert_eq!(assessment.migration_urgency, MigrationUrgency::None);
    }

    // ---- Recommended replacements ----

    #[test]
    fn test_recommended_replacement_rsa_to_mlkem() {
        let config = default_config();
        let assessment = assess_algorithm(&CryptoAlgorithm::RSA { key_bits: 2048 }, &config);
        assert!(assessment.recommended_replacement.is_some());
        match assessment.recommended_replacement {
            Some(CryptoAlgorithm::MLKEM { parameter_set }) => {
                assert_eq!(parameter_set, MLKEMParams::ML768);
            }
            _ => panic!("Expected ML-KEM recommendation for RSA-2048"),
        }
    }

    // ---- Physical qubit estimation ----

    #[test]
    fn test_physical_qubit_surface_code_overhead() {
        let config = PqcConfig::default();
        // Physical per logical = 2 * 23^2 = 1058
        assert_eq!(config.physical_per_logical(), 2 * 23 * 23);

        let total = config.total_physical_qubits(4099); // RSA-2048 logical qubits
                                                        // Data: 4099 * 1058 = ~4,336,742
                                                        // Factory: 4 * 16 * 529 = 33,856
                                                        // Total: ~4,370,598
        assert!(total > 4_000_000);
        assert!(total < 5_000_000);
    }

    // ---- Timeline projection ----

    #[test]
    fn test_timeline_rsa_breakable_estimate_year() {
        let config = default_config();
        let break_year = estimate_break_year(&CryptoAlgorithm::RSA { key_bits: 2048 }, &config);
        // RSA-2048 needs ~4099 logical qubits. Projected: 2033 (10K available).
        assert!(break_year.is_some());
        let year = break_year.unwrap();
        assert!(year >= 2029);
        assert!(year <= 2040);
    }

    #[test]
    fn test_timeline_aes_256_safe_indefinitely() {
        let config = default_config();
        let break_year = estimate_break_year(&CryptoAlgorithm::AES { key_bits: 256 }, &config);
        // AES-256 requires 2^128 Grover queries -- infeasible.
        assert!(break_year.is_none());
    }

    // ---- Comparison matrix ----

    #[test]
    fn test_comparison_matrix_all_algorithms_ranked() {
        let config = default_config();
        let matrix = build_comprehensive_matrix(&config);
        // Should have 21 algorithms in the comprehensive matrix.
        assert!(matrix.assessments.len() >= 20);

        // At-risk should include RSA and ECC entries.
        let at_risk = matrix.at_risk();
        assert!(at_risk.len() >= 4);

        // Quantum-safe should include PQC entries.
        let safe = matrix.quantum_safe();
        assert!(safe.len() >= 8);

        // Sorted by quantum security should have RSA/ECC/DH/DSA first (0 bits).
        let sorted = matrix.sorted_by_quantum_security();
        assert_eq!(sorted[0].quantum_security_bits, 0);
    }

    // ---- Circuit depth calculation ----

    #[test]
    fn test_shor_circuit_depth_calculation() {
        let shor = shor_rsa_estimate(2048);
        // Circuit depth = n^2 * log2(n)
        let n = 2048u64;
        let log_n = (n as f64).log2().ceil() as u64;
        let expected = n * n * log_n;
        assert_eq!(shor.circuit_depth, expected);
    }

    // ---- Modular exponentiation cost ----

    #[test]
    fn test_modular_exponentiation_cost() {
        let shor = shor_rsa_estimate(1024);
        // 0.3 * 1024^3 = ~322,122,547
        let expected = (0.3 * (1024.0f64).powi(3)) as u64;
        assert_eq!(shor.toffoli_gates, expected);
        assert_eq!(shor.modular_exponentiations, 2);
    }

    // ---- Lattice security ----

    #[test]
    fn test_lattice_security_estimation() {
        let (classical, quantum) = lattice_security_estimate(768, 3);
        // For ML-KEM-768 (dimension 768, level 3):
        // beta ~ 0.45 * 768 = 345
        // classical ~ 0.292 * 345 = ~100
        // quantum ~ 0.265 * 345 = ~91
        assert!(classical > 80);
        assert!(classical < 150);
        assert!(quantum > 70);
        assert!(quantum < 130);
        assert!(classical > quantum);
    }

    // ---- Code-based ISD cost ----

    #[test]
    fn test_code_based_isd_cost() {
        let config = default_config();
        let estimate = isd_attack_estimate("McEliece-6688128", 6688, 128, 370, &config);
        // Quantum ISD: 0.72 * 370 = 266 bits security
        assert!(estimate.security_level_bits > 200);
        assert!(estimate.security_level_bits < 300);
        assert!(estimate.logical_qubits > 6000);
    }

    // ---- Config effects ----

    #[test]
    fn test_config_surface_code_distance_effect() {
        let config_d17 = PqcConfig::default().surface_code_distance(17);
        let config_d31 = PqcConfig::default().surface_code_distance(31);

        // Higher distance = more physical qubits per logical qubit.
        let p17 = config_d17.physical_per_logical(); // 2 * 289 = 578
        let p31 = config_d31.physical_per_logical(); // 2 * 961 = 1922

        assert_eq!(p17, 2 * 17 * 17);
        assert_eq!(p31, 2 * 31 * 31);
        assert!(p31 > p17);
    }

    #[test]
    fn test_config_clock_speed_effect_on_wall_time() {
        let slow = PqcConfig::default().clock_speed_mhz(1.0);
        let fast = PqcConfig::default().clock_speed_mhz(100.0);

        let depth = 1_000_000u64;
        let hours_slow = slow.depth_to_hours(depth);
        let hours_fast = fast.depth_to_hours(depth);

        // 100x faster clock should yield ~1/100 the wall time.
        assert!((hours_slow / hours_fast - 100.0).abs() < 1.0);
    }

    // ---- NIST security levels ----

    #[test]
    fn test_nist_security_levels_correct() {
        assert_eq!(nist_level_to_aes_bits(1), 128);
        assert_eq!(nist_level_to_aes_bits(3), 192);
        assert_eq!(nist_level_to_aes_bits(5), 256);

        assert_eq!(nist_level_to_quantum_bits(1), 64);
        assert_eq!(nist_level_to_quantum_bits(3), 96);
        assert_eq!(nist_level_to_quantum_bits(5), 128);
    }

    // ---- Config builder defaults ----

    #[test]
    fn test_config_builder_defaults() {
        let config = PqcConfig::default();
        assert_eq!(config.surface_code_distance, 23);
        assert!((config.physical_error_rate - 1e-3).abs() < 1e-10);
        assert!((config.clock_speed_mhz - 10.0).abs() < 1e-10);
        assert_eq!(config.magic_state_factory_count, 4);
    }

    // ---- All PQC algorithms run without error ----

    #[test]
    fn test_all_pqc_algorithms_assessment_runs() {
        let config = default_config();

        let algorithms = vec![
            CryptoAlgorithm::RSA { key_bits: 2048 },
            CryptoAlgorithm::RSA { key_bits: 4096 },
            CryptoAlgorithm::ECC { curve_bits: 256 },
            CryptoAlgorithm::ECC { curve_bits: 384 },
            CryptoAlgorithm::DH { key_bits: 2048 },
            CryptoAlgorithm::DSA { key_bits: 3072 },
            CryptoAlgorithm::AES { key_bits: 128 },
            CryptoAlgorithm::AES { key_bits: 256 },
            CryptoAlgorithm::MLKEM {
                parameter_set: MLKEMParams::ML512,
            },
            CryptoAlgorithm::MLKEM {
                parameter_set: MLKEMParams::ML768,
            },
            CryptoAlgorithm::MLKEM {
                parameter_set: MLKEMParams::ML1024,
            },
            CryptoAlgorithm::MLDSA {
                parameter_set: MLDSAParams::ML44,
            },
            CryptoAlgorithm::MLDSA {
                parameter_set: MLDSAParams::ML65,
            },
            CryptoAlgorithm::MLDSA {
                parameter_set: MLDSAParams::ML87,
            },
            CryptoAlgorithm::SLHDSA {
                parameter_set: SLHDSAParams::SHA2_128f,
            },
            CryptoAlgorithm::SLHDSA {
                parameter_set: SLHDSAParams::SHA2_128s,
            },
            CryptoAlgorithm::SLHDSA {
                parameter_set: SLHDSAParams::SHA2_192f,
            },
            CryptoAlgorithm::SLHDSA {
                parameter_set: SLHDSAParams::SHA2_192s,
            },
            CryptoAlgorithm::SLHDSA {
                parameter_set: SLHDSAParams::SHA2_256f,
            },
            CryptoAlgorithm::SLHDSA {
                parameter_set: SLHDSAParams::SHA2_256s,
            },
            CryptoAlgorithm::HQC {
                parameter_set: HQCParams::HQC128,
            },
            CryptoAlgorithm::HQC {
                parameter_set: HQCParams::HQC192,
            },
            CryptoAlgorithm::HQC {
                parameter_set: HQCParams::HQC256,
            },
            CryptoAlgorithm::NTRU {
                parameter_set: NTRUParams::NTRU509,
            },
            CryptoAlgorithm::NTRU {
                parameter_set: NTRUParams::NTRU677,
            },
            CryptoAlgorithm::NTRU {
                parameter_set: NTRUParams::NTRU821,
            },
            CryptoAlgorithm::BIKE {
                parameter_set: BIKEParams::BIKE1,
            },
            CryptoAlgorithm::BIKE {
                parameter_set: BIKEParams::BIKE3,
            },
            CryptoAlgorithm::BIKE {
                parameter_set: BIKEParams::BIKE5,
            },
            CryptoAlgorithm::ClassicMcEliece {
                parameter_set: McElieceParams::MC348864,
            },
            CryptoAlgorithm::ClassicMcEliece {
                parameter_set: McElieceParams::MC460896,
            },
            CryptoAlgorithm::ClassicMcEliece {
                parameter_set: McElieceParams::MC6688128,
            },
            CryptoAlgorithm::ClassicMcEliece {
                parameter_set: McElieceParams::MC6960119,
            },
            CryptoAlgorithm::ClassicMcEliece {
                parameter_set: McElieceParams::MC8192128,
            },
        ];

        for algo in &algorithms {
            let assessment = assess_algorithm(algo, &config);
            // Every assessment should produce a valid threat level.
            assert!(
                assessment.classical_security_bits > 0 || assessment.quantum_security_bits == 0,
                "Assessment for {} should be consistent",
                algo
            );
            // Display implementations should not panic.
            let _ = format!("{}", assessment);
            let _ = format!("{}", assessment.attack_estimate);
        }
    }

    // ---- Additional coverage tests ----

    #[test]
    fn test_ecc_shor_estimate_qubits() {
        let shor = shor_ecc_estimate(256);
        // 6 * 256 + 10 = 1546
        assert_eq!(shor.logical_qubits, 1546);
    }

    #[test]
    fn test_dh_shor_estimate() {
        let shor = shor_dh_dsa_estimate(2048);
        assert_eq!(shor.logical_qubits, 2 * 2048 + 5);
        assert!(shor.toffoli_gates > 0);
    }

    #[test]
    fn test_threat_timeline_projection() {
        let timeline = project_threat_timeline();
        assert!(!timeline.is_empty());

        // In 2025 (10 logical qubits) nothing should be at risk.
        let y2025 = timeline.iter().find(|t| t.year == 2025).unwrap();
        assert!(y2025.algorithms_at_risk.is_empty());

        // By 2040 (100K logical qubits) RSA-2048 should be at risk.
        let y2040 = timeline.iter().find(|t| t.year == 2040).unwrap();
        assert!(
            y2040
                .algorithms_at_risk
                .iter()
                .any(|a| a.contains("RSA-2048")),
            "RSA-2048 should be at risk by 2040"
        );

        // PQC should always be safe.
        for t in &timeline {
            assert!(
                t.algorithms_safe.iter().any(|a| a.contains("ML-KEM")),
                "ML-KEM should always be safe"
            );
        }
    }

    #[test]
    fn test_migration_plan_generation() {
        let config = default_config();
        let algorithms = vec![
            CryptoAlgorithm::RSA { key_bits: 2048 },
            CryptoAlgorithm::ECC { curve_bits: 256 },
            CryptoAlgorithm::AES { key_bits: 128 },
            CryptoAlgorithm::MLKEM {
                parameter_set: MLKEMParams::ML768,
            },
        ];
        let plan = generate_migration_plan(&algorithms, &config);
        // RSA, ECC, and AES should have recommendations. ML-KEM should not.
        assert_eq!(plan.len(), 3);
        assert!(plan.iter().any(|r| format!("{}", r.from).contains("RSA")));
        assert!(plan.iter().any(|r| format!("{}", r.from).contains("ECC")));
    }

    #[test]
    fn test_validate_key_size() {
        assert!(validate_key_size(&CryptoAlgorithm::RSA { key_bits: 2048 }).is_ok());
        assert!(validate_key_size(&CryptoAlgorithm::RSA { key_bits: 64 }).is_err());
        assert!(validate_key_size(&CryptoAlgorithm::AES { key_bits: 128 }).is_ok());
        assert!(validate_key_size(&CryptoAlgorithm::AES { key_bits: 512 }).is_err());
        assert!(validate_key_size(&CryptoAlgorithm::ECC { curve_bits: 256 }).is_ok());
        assert!(validate_key_size(&CryptoAlgorithm::ECC { curve_bits: 32 }).is_err());
    }

    #[test]
    fn test_report_generation() {
        let config = default_config();
        let matrix = build_comprehensive_matrix(&config);
        let report = generate_report(&matrix);
        assert!(report.contains("CRITICAL THREATS"));
        assert!(report.contains("QUANTUM-SAFE"));
        assert!(report.contains("THREAT TIMELINE"));
        assert!(report.contains("RSA-2048"));
    }

    #[test]
    fn test_display_implementations() {
        // Verify all Display implementations produce non-empty strings.
        assert!(!format!("{}", MLKEMParams::ML768).is_empty());
        assert!(!format!("{}", MLDSAParams::ML65).is_empty());
        assert!(!format!("{}", SLHDSAParams::SHA2_256f).is_empty());
        assert!(!format!("{}", HQCParams::HQC128).is_empty());
        assert!(!format!("{}", NTRUParams::NTRU677).is_empty());
        assert!(!format!("{}", BIKEParams::BIKE3).is_empty());
        assert!(!format!("{}", McElieceParams::MC6688128).is_empty());
        assert!(!format!("{}", ThreatLevel::Critical).is_empty());
        assert!(!format!("{}", MigrationUrgency::Immediate).is_empty());
        assert!(!format!("{}", QuantumAttack::Shor).is_empty());
        assert!(!format!("{}", QuantumAttack::BHT { memory: 100 }).is_empty());
        assert!(!format!("{}", QuantumAttack::ISD { quantum: true }).is_empty());
        assert!(!format!("{}", QuantumAttack::ISD { quantum: false }).is_empty());
    }

    #[test]
    fn test_pqc_error_display() {
        let e1 = PqcError::InvalidKeySize(42);
        assert!(format!("{}", e1).contains("42"));

        let e2 = PqcError::UnsupportedAlgorithm("FooBar".to_string());
        assert!(format!("{}", e2).contains("FooBar"));

        let e3 = PqcError::EstimationFailed("overflow".to_string());
        assert!(format!("{}", e3).contains("overflow"));
    }

    #[test]
    fn test_comparison_matrix_display() {
        let config = default_config();
        let matrix = build_comparison_matrix(
            &[
                CryptoAlgorithm::RSA { key_bits: 2048 },
                CryptoAlgorithm::MLKEM {
                    parameter_set: MLKEMParams::ML768,
                },
            ],
            &config,
        );
        let display = format!("{}", matrix);
        assert!(display.contains("RSA-2048"));
        assert!(display.contains("ML-KEM-768"));
    }

    #[test]
    fn test_slhdsa_fast_variants() {
        assert!(SLHDSAParams::SHA2_128f.is_fast());
        assert!(!SLHDSAParams::SHA2_128s.is_fast());
        assert!(SLHDSAParams::SHA2_256f.is_fast());
        assert!(!SLHDSAParams::SHA2_256s.is_fast());
    }

    #[test]
    fn test_mceliece_code_parameters() {
        assert_eq!(McElieceParams::MC6688128.code_length(), 6688);
        assert_eq!(McElieceParams::MC6688128.error_capability(), 128);
        assert_eq!(McElieceParams::MC8192128.code_length(), 8192);
        assert_eq!(McElieceParams::MC8192128.error_capability(), 128);
    }

    #[test]
    fn test_hqc_code_length() {
        assert_eq!(HQCParams::HQC128.code_length(), 17_669);
        assert_eq!(HQCParams::HQC192.code_length(), 35_851);
        assert_eq!(HQCParams::HQC256.code_length(), 57_637);
    }

    #[test]
    fn test_bike_block_length() {
        assert_eq!(BIKEParams::BIKE1.block_length(), 12_323);
        assert_eq!(BIKEParams::BIKE3.block_length(), 24_659);
        assert_eq!(BIKEParams::BIKE5.block_length(), 40_973);
    }

    #[test]
    fn test_ntru_lattice_dimension() {
        assert_eq!(NTRUParams::NTRU509.lattice_dimension(), 509);
        assert_eq!(NTRUParams::NTRU677.lattice_dimension(), 677);
        assert_eq!(NTRUParams::NTRU821.lattice_dimension(), 821);
    }

    #[test]
    fn test_mlkem_lattice_dimension() {
        assert_eq!(MLKEMParams::ML512.lattice_dimension(), 512);
        assert_eq!(MLKEMParams::ML768.lattice_dimension(), 768);
        assert_eq!(MLKEMParams::ML1024.lattice_dimension(), 1024);
    }
}
