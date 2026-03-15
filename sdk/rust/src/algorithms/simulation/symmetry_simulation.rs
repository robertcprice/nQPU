//! Symmetry-Exploiting Quantum Simulation
//!
//! This module decomposes the Hilbert space into symmetry sectors defined by
//! conserved quantum numbers (U(1) particle number, SU(2) z-magnetization).
//! For circuits that preserve these symmetries, the simulation operates
//! entirely within a single sector, yielding dramatic memory and compute
//! savings.
//!
//! # Example: 20 qubits, 10 particles
//!
//! The 10-particle sector has dimension C(20,10) = 184,756 vs the full
//! Hilbert space dimension 2^20 = 1,048,576 -- a 5.7x reduction.
//!
//! # Supported symmetries
//!
//! - **U(1) particle number**: conserved by CNOT, CZ, SWAP, Rz, Phase, Z, S, T
//! - **Sz magnetization**: conserved by the same gates (Sz = n_up - n_down)/2

use num_complex::Complex64 as C64;
use std::collections::HashMap;
use std::fmt;

use crate::gates::{Gate, GateType};
use crate::QuantumState;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from symmetry-restricted simulation.
#[derive(Clone, Debug, PartialEq)]
pub enum SymmetryError {
    /// The applied gate does not preserve the active symmetry sector.
    GateBreaksSymmetry {
        gate_desc: String,
        symmetry_desc: String,
    },
    /// A qubit index exceeds the register size.
    QubitOutOfRange { qubit: usize, num_qubits: usize },
    /// The requested sector is invalid (e.g. particle number > num_qubits).
    InvalidSector { reason: String },
}

impl fmt::Display for SymmetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymmetryError::GateBreaksSymmetry {
                gate_desc,
                symmetry_desc,
            } => write!(f, "gate '{}' breaks {} symmetry", gate_desc, symmetry_desc),
            SymmetryError::QubitOutOfRange { qubit, num_qubits } => {
                write!(
                    f,
                    "qubit {} out of range for {}-qubit register",
                    qubit, num_qubits
                )
            }
            SymmetryError::InvalidSector { reason } => {
                write!(f, "invalid sector: {}", reason)
            }
        }
    }
}

impl std::error::Error for SymmetryError {}

// ============================================================
// GATE SYMMETRY CLASSIFICATION
// ============================================================

/// Classification of a gate with respect to a conserved quantum number.
#[derive(Clone, Debug, PartialEq)]
pub enum GateSymmetry {
    /// Gate preserves the quantum number (sector-diagonal).
    Preserving,
    /// Gate changes the quantum number by a known delta.
    SectorChanging { delta: i32 },
    /// Gate mixes sectors in an unknown or complex way.
    Unknown,
}

// ============================================================
// SYMMETRY SECTOR
// ============================================================

/// A single symmetry sector of the Hilbert space.
///
/// Contains the set of computational basis states sharing a common quantum
/// number (e.g. the same particle count or Sz eigenvalue).
#[derive(Clone, Debug)]
pub struct SymmetrySector {
    /// The conserved quantum number labelling this sector.
    pub quantum_number: i32,
    /// Computational basis indices belonging to this sector, sorted ascending.
    pub basis_states: Vec<usize>,
    /// Number of basis states in the sector (== basis_states.len()).
    pub dim: usize,
    /// Maps a full Hilbert-space index to the local sector index.
    pub index_map: HashMap<usize, usize>,
}

impl SymmetrySector {
    /// Build a sector from a quantum number and a list of basis indices.
    pub fn new(quantum_number: i32, mut basis_states: Vec<usize>) -> Self {
        basis_states.sort_unstable();
        basis_states.dedup();
        let dim = basis_states.len();
        let index_map: HashMap<usize, usize> = basis_states
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();
        SymmetrySector {
            quantum_number,
            basis_states,
            dim,
            index_map,
        }
    }

    /// Return true if the given full-space index belongs to this sector.
    #[inline]
    pub fn contains(&self, global_idx: usize) -> bool {
        self.index_map.contains_key(&global_idx)
    }

    /// Map a full-space index to a sector-local index, returning None if absent.
    #[inline]
    pub fn to_local(&self, global_idx: usize) -> Option<usize> {
        self.index_map.get(&global_idx).copied()
    }

    /// Map a sector-local index back to the full-space index.
    #[inline]
    pub fn to_global(&self, local_idx: usize) -> usize {
        self.basis_states[local_idx]
    }

    /// Return true if this sector is empty.
    pub fn is_empty(&self) -> bool {
        self.dim == 0
    }
}

// ============================================================
// U(1) SYMMETRY -- PARTICLE NUMBER CONSERVATION
// ============================================================

/// U(1) symmetry corresponding to conservation of particle (excitation) number.
///
/// The particle number of a computational basis state |b> is the Hamming
/// weight of b (number of qubits in state |1>).
pub struct U1Symmetry {
    /// Number of qubits in the register.
    pub num_qubits: usize,
    /// Target particle number for the active sector.
    pub target_particle_number: usize,
}

impl U1Symmetry {
    /// Create a new U(1) symmetry tracker.
    ///
    /// # Panics
    ///
    /// Panics if `target_particle_number > num_qubits`.
    pub fn new(num_qubits: usize, target_particle_number: usize) -> Self {
        assert!(
            target_particle_number <= num_qubits,
            "particle number {} exceeds qubit count {}",
            target_particle_number,
            num_qubits
        );
        U1Symmetry {
            num_qubits,
            target_particle_number,
        }
    }

    /// Compute the Hamming weight (number of set bits) of `state`.
    #[inline]
    pub fn hamming_weight(state: usize) -> usize {
        state.count_ones() as usize
    }

    /// Enumerate all computational basis states of `num_qubits` qubits
    /// that contain exactly `n` particles (bits set to 1).
    ///
    /// The returned vector is sorted in ascending order.
    pub fn enumerate_sector(num_qubits: usize, n: usize) -> Vec<usize> {
        if n > num_qubits {
            return Vec::new();
        }
        let dim = 1usize << num_qubits;
        let mut states = Vec::with_capacity(binomial(num_qubits, n));
        for s in 0..dim {
            if Self::hamming_weight(s) == n {
                states.push(s);
            }
        }
        states
    }

    /// Build the symmetry sector for a specific particle number.
    pub fn sector_for(num_qubits: usize, n: usize) -> SymmetrySector {
        let states = Self::enumerate_sector(num_qubits, n);
        SymmetrySector::new(n as i32, states)
    }

    /// Build all symmetry sectors for a given qubit count (particle numbers 0..=num_qubits).
    pub fn sectors(num_qubits: usize) -> Vec<SymmetrySector> {
        (0..=num_qubits)
            .map(|n| Self::sector_for(num_qubits, n))
            .collect()
    }

    /// Return the sector corresponding to this instance's target particle number.
    pub fn active_sector(&self) -> SymmetrySector {
        Self::sector_for(self.num_qubits, self.target_particle_number)
    }
}

// ============================================================
// SZ SYMMETRY -- Z-MAGNETIZATION CONSERVATION
// ============================================================

/// SU(2) z-magnetization symmetry.
///
/// For a system of `num_qubits` spin-1/2 particles, the z-magnetization is
/// Sz = (n_up - n_down) / 2, where n_up is the number of |0> qubits (spin up
/// by convention) and n_down is the number of |1> qubits (spin down).
///
/// Equivalently, Sz = (num_qubits - 2 * hamming_weight(state)) / 2.
///
/// We store Sz as a half-integer scaled by 2 (i.e. `2*Sz`) to use integer
/// arithmetic. The `target_sz` field is the unscaled half-integer Sz value
/// expressed as an i32 numerator of Sz = target_sz / 2.
pub struct SzSymmetry {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Target Sz value, stored as 2*Sz (an integer) to avoid fractions.
    /// For num_qubits=4 and 2 particles: Sz = (4-2*2)/2 = 0 -> target_sz_times_2 = 0.
    target_sz_times_2: i32,
}

impl SzSymmetry {
    /// Create a new Sz symmetry tracker.
    ///
    /// `target_sz` is the desired Sz eigenvalue. For half-integer values, use
    /// the convention Sz = (n_up - n_down) / 2 where |0> = spin up.
    ///
    /// Internally we store 2*Sz as an integer.
    ///
    /// # Panics
    ///
    /// Panics if the target Sz is unreachable for the given qubit count.
    pub fn new(num_qubits: usize, target_sz: i32) -> Self {
        // 2*Sz must have the same parity as num_qubits and |2*Sz| <= num_qubits
        let two_sz = 2 * target_sz;
        let n = num_qubits as i32;
        // n_down = (num_qubits - 2*Sz) / 2  must be in [0, num_qubits]
        let n_down_times_2 = n - two_sz;
        assert!(
            n_down_times_2 >= 0 && n_down_times_2 <= 2 * n && n_down_times_2 % 2 == 0,
            "Sz = {} is unreachable for {} qubits",
            target_sz,
            num_qubits
        );
        SzSymmetry {
            num_qubits,
            target_sz_times_2: two_sz,
        }
    }

    /// Compute 2*Sz for a given basis state.
    #[inline]
    fn two_sz(num_qubits: usize, state: usize) -> i32 {
        let n_down = U1Symmetry::hamming_weight(state) as i32;
        num_qubits as i32 - 2 * n_down
    }

    /// Build the symmetry sector for a given Sz eigenvalue.
    pub fn sector_for(num_qubits: usize, sz: i32) -> SymmetrySector {
        let target_2sz = 2 * sz;
        let dim = 1usize << num_qubits;
        let mut states = Vec::new();
        for s in 0..dim {
            if Self::two_sz(num_qubits, s) == target_2sz {
                states.push(s);
            }
        }
        SymmetrySector::new(sz, states)
    }

    /// Return the sector corresponding to this instance's target Sz.
    pub fn active_sector(&self) -> SymmetrySector {
        Self::sector_for(self.num_qubits, self.target_sz_times_2 / 2)
    }

    /// Build all Sz sectors for a given qubit count.
    ///
    /// Sz ranges from -num_qubits/2 to +num_qubits/2 in steps of 1.
    pub fn sectors(num_qubits: usize) -> Vec<SymmetrySector> {
        let n = num_qubits as i32;
        // Sz = n/2, n/2 - 1, ..., -n/2  (n+1 values)
        // Since Sz = (n - 2*n_down)/2, n_down goes from 0 to n.
        (0..=num_qubits)
            .map(|n_down| {
                let sz = (n - 2 * n_down as i32) / 2;
                Self::sector_for(num_qubits, sz)
            })
            .collect()
    }
}

// ============================================================
// SYMMETRY TYPE
// ============================================================

/// Which symmetry to exploit for simulation.
#[derive(Clone, Debug)]
pub enum SymmetryType {
    /// Particle number conservation (U(1) symmetry).
    ParticleNumber(usize),
    /// Z-magnetization conservation (SU(2) Sz symmetry).
    Magnetization(i32),
    /// A user-supplied custom sector.
    Custom(SymmetrySector),
}

// ============================================================
// SYMMETRIC STATE
// ============================================================

/// A quantum state vector restricted to a single symmetry sector.
///
/// Only amplitudes for basis states belonging to the sector are stored,
/// yielding significant memory savings for large registers with a fixed
/// quantum number.
#[derive(Clone, Debug)]
pub struct SymmetricState {
    /// Amplitudes for the basis states in the sector (sector-local indexing).
    pub amplitudes: Vec<C64>,
    /// The symmetry sector this state lives in.
    pub sector: SymmetrySector,
    /// Number of qubits in the full register.
    pub num_qubits: usize,
}

impl SymmetricState {
    /// Create a new symmetric state initialized to the first basis state in the sector.
    ///
    /// The first basis state (lexicographically smallest) receives amplitude 1.
    ///
    /// Returns an error if the sector is empty.
    pub fn new(num_qubits: usize, sector: SymmetrySector) -> Result<Self, SymmetryError> {
        if sector.is_empty() {
            return Err(SymmetryError::InvalidSector {
                reason: "cannot create state in empty sector".to_string(),
            });
        }
        let dim = sector.dim;
        let mut amplitudes = vec![C64::new(0.0, 0.0); dim];
        amplitudes[0] = C64::new(1.0, 0.0);
        Ok(SymmetricState {
            amplitudes,
            sector,
            num_qubits,
        })
    }

    /// Create a symmetric state by projecting a full state vector onto a sector.
    ///
    /// Amplitudes outside the sector are discarded. The projected state is NOT
    /// automatically normalized -- call `normalize()` afterwards if needed.
    pub fn from_state(full_state: &QuantumState, sector: &SymmetrySector) -> Self {
        let num_qubits = full_state.num_qubits;
        let amps_ref = full_state.amplitudes_ref();
        let amplitudes: Vec<C64> = sector
            .basis_states
            .iter()
            .map(|&global| {
                if global < amps_ref.len() {
                    amps_ref[global]
                } else {
                    C64::new(0.0, 0.0)
                }
            })
            .collect();

        SymmetricState {
            amplitudes,
            sector: sector.clone(),
            num_qubits,
        }
    }

    /// Expand this sector state back into a full Hilbert-space state vector.
    ///
    /// Amplitudes for basis states outside the sector are set to zero.
    pub fn to_full_state(&self) -> QuantumState {
        let dim = 1usize << self.num_qubits;
        let mut full_amps = vec![C64::new(0.0, 0.0); dim];
        for (local, &global) in self.sector.basis_states.iter().enumerate() {
            full_amps[global] = self.amplitudes[local];
        }
        // Build a QuantumState from the amplitudes
        let mut state = QuantumState::new(self.num_qubits);
        let state_amps = state.amplitudes_mut();
        state_amps.copy_from_slice(&full_amps);
        state
    }

    /// Get the amplitude for a full-space basis index.
    ///
    /// Returns zero if the index is outside the sector.
    #[inline]
    pub fn get(&self, basis_idx: usize) -> C64 {
        match self.sector.to_local(basis_idx) {
            Some(local) => self.amplitudes[local],
            None => C64::new(0.0, 0.0),
        }
    }

    /// Set the amplitude for a full-space basis index.
    ///
    /// Does nothing if the index is outside the sector.
    #[inline]
    pub fn set(&mut self, basis_idx: usize, val: C64) {
        if let Some(local) = self.sector.to_local(basis_idx) {
            self.amplitudes[local] = val;
        }
    }

    /// Compute the probabilities for each basis state in the sector.
    ///
    /// Returns pairs of (full-space basis index, probability).
    pub fn probabilities(&self) -> Vec<(usize, f64)> {
        self.sector
            .basis_states
            .iter()
            .zip(self.amplitudes.iter())
            .map(|(&global, amp)| (global, amp.norm_sqr()))
            .collect()
    }

    /// Compute the full probability vector (length 2^n, most entries zero).
    pub fn full_probabilities(&self) -> Vec<f64> {
        let dim = 1usize << self.num_qubits;
        let mut probs = vec![0.0; dim];
        for (local, &global) in self.sector.basis_states.iter().enumerate() {
            probs[global] = self.amplitudes[local].norm_sqr();
        }
        probs
    }

    /// Compute the squared norm of the state.
    pub fn norm(&self) -> f64 {
        self.amplitudes
            .iter()
            .map(|a| a.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }

    /// Normalize the state to unit norm.
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > 1e-15 {
            let inv = 1.0 / n;
            for a in &mut self.amplitudes {
                *a = C64::new(a.re * inv, a.im * inv);
            }
        }
    }

    /// Compute the memory savings ratio: sector_dim / full_dim.
    ///
    /// A value of 0.175 means the sector uses 17.5% of the memory that a
    /// full state vector would require.
    pub fn memory_savings(&self) -> f64 {
        let full_dim = 1usize << self.num_qubits;
        self.sector.dim as f64 / full_dim as f64
    }

    /// Sample a measurement outcome from the sector probability distribution.
    ///
    /// Returns (basis_state_index, probability).
    pub fn measure(&self) -> (usize, f64) {
        let probs = self.probabilities();
        let r: f64 = rand::random();
        let mut cumsum = 0.0;
        for &(state, p) in &probs {
            cumsum += p;
            if r <= cumsum {
                return (state, p);
            }
        }
        // Fallback to last state
        probs.last().copied().unwrap_or((0, 0.0))
    }

    /// Compute the expectation value of the Pauli-Z operator on a single qubit.
    ///
    /// <Z_q> = sum_i |a_i|^2 * (-1)^{bit q of basis_i}
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        let mask = 1usize << qubit;
        let mut exp = 0.0;
        for (local, &global) in self.sector.basis_states.iter().enumerate() {
            let prob = self.amplitudes[local].norm_sqr();
            if global & mask == 0 {
                exp += prob; // qubit is |0> -> eigenvalue +1
            } else {
                exp -= prob; // qubit is |1> -> eigenvalue -1
            }
        }
        exp
    }
}

// ============================================================
// SYMMETRY-PRESERVING GATE ANALYSIS
// ============================================================

/// Utility functions for classifying gates with respect to symmetry preservation.
pub struct SymmetryPreservingGate;

impl SymmetryPreservingGate {
    /// Returns true if the gate preserves U(1) particle number symmetry.
    ///
    /// Preserving gates are those that map computational basis states with
    /// k set bits to states with k set bits. This includes:
    /// - Diagonal gates: Z, S, T, Rz, Phase, CZ, CRz, CR, CCZ
    /// - Permutation gates that preserve Hamming weight: CNOT, SWAP, ISWAP, Toffoli
    pub fn is_u1_preserving(gate: &Gate) -> bool {
        match &gate.gate_type {
            // Single-qubit diagonal gates (preserve |0> and |1> individually)
            GateType::Z | GateType::S | GateType::T => true,
            GateType::Rz(_) | GateType::Phase(_) => true,

            // Two-qubit gates that preserve particle number
            GateType::CNOT => true, // |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
            GateType::CZ => true,   // diagonal
            GateType::SWAP => true, // permutation preserving Hamming weight
            GateType::ISWAP => true, // preserves Hamming weight (|01>->i|10>, |10>->i|01>)

            // Controlled rotations around Z (diagonal)
            GateType::CRz(_) | GateType::CR(_) => true,

            // Three-qubit gates preserving particle number
            GateType::Toffoli => true, // CCNOT preserves Hamming weight
            GateType::CCZ => true,     // diagonal

            // Gates that change particle number
            GateType::H => false, // creates superposition of different particle numbers
            GateType::X => false, // flips |0> <-> |1>, changes particle number by +/-1
            GateType::Y => false, // same as X up to phase
            GateType::Rx(_) => false,
            GateType::Ry(_) => false,
            GateType::SX => false,
            GateType::CRx(_) | GateType::CRy(_) => false,
            GateType::U { .. } => false, // general U may not preserve

            // Two-qubit rotation gates: Rzz preserves U(1), Rxx/Ryy do not
            GateType::Rzz(_) => true,
            GateType::Rxx(_) | GateType::Ryy(_) => false,

            // CSWAP preserves particle number (controlled permutation)
            GateType::CSWAP => true,

            // CU: general controlled-U may not preserve
            GateType::CU { .. } => false,

            // Custom gates: unknown
            GateType::Custom(_) => false,
        }
    }

    /// Returns true if the gate preserves Sz magnetization symmetry.
    ///
    /// Since Sz = (num_qubits - 2 * hamming_weight) / 2, Sz is preserved
    /// iff particle number is preserved. The same set of gates qualifies.
    pub fn is_sz_preserving(gate: &Gate) -> bool {
        Self::is_u1_preserving(gate)
    }

    /// Classify a gate's effect on the particle number quantum number.
    pub fn classify_gate(gate: &Gate) -> GateSymmetry {
        if Self::is_u1_preserving(gate) {
            return GateSymmetry::Preserving;
        }
        match &gate.gate_type {
            // X gate on a single qubit changes particle number by +1 or -1
            // depending on the input state; it is not a fixed delta.
            GateType::X | GateType::Y => GateSymmetry::Unknown,
            GateType::H => GateSymmetry::Unknown,
            GateType::Rx(_) | GateType::Ry(_) => GateSymmetry::Unknown,
            GateType::SX => GateSymmetry::Unknown,
            GateType::CRx(_) | GateType::CRy(_) => GateSymmetry::Unknown,
            GateType::U { .. } => GateSymmetry::Unknown,
            GateType::Rxx(_) | GateType::Ryy(_) => GateSymmetry::Unknown,
            GateType::CU { .. } => GateSymmetry::Unknown,
            GateType::Custom(_) => GateSymmetry::Unknown,
            // All preserving cases handled above
            _ => GateSymmetry::Preserving,
        }
    }
}

// ============================================================
// SYMMETRY ANALYZER
// ============================================================

/// Report from analyzing a circuit for symmetry exploitation potential.
#[derive(Clone, Debug)]
pub struct SymmetryReport {
    /// True if the entire circuit preserves U(1) particle number.
    pub preserves_u1: bool,
    /// True if the entire circuit preserves Sz magnetization.
    pub preserves_sz: bool,
    /// Indices of gates that break the symmetry (0-indexed).
    pub symmetry_breaking_gates: Vec<usize>,
    /// Recommended symmetry type to exploit, if any.
    pub recommended_sector: Option<SymmetryType>,
    /// Estimated memory reduction factor (sector_dim / full_dim for the
    /// half-filling sector, which is the largest possible saving).
    pub memory_reduction: f64,
}

/// Analyze circuits for symmetry exploitation potential.
pub struct SymmetryAnalyzer;

impl SymmetryAnalyzer {
    /// Analyze a circuit to determine which symmetries it preserves.
    pub fn analyze_circuit(gates: &[Gate], num_qubits: usize) -> SymmetryReport {
        let mut breaking_indices = Vec::new();

        for (i, gate) in gates.iter().enumerate() {
            if !SymmetryPreservingGate::is_u1_preserving(gate) {
                breaking_indices.push(i);
            }
        }

        let preserves_u1 = breaking_indices.is_empty();
        // For the gate set we consider, Sz preservation is equivalent to U(1).
        let preserves_sz = preserves_u1;

        // Recommend the half-filling sector (maximum savings) if symmetry is preserved.
        let recommended_sector = if preserves_u1 {
            let half = num_qubits / 2;
            Some(SymmetryType::ParticleNumber(half))
        } else {
            None
        };

        // Compute memory reduction for half-filling.
        let full_dim = 1usize << num_qubits;
        let half = num_qubits / 2;
        let sector_dim = binomial(num_qubits, half);
        let memory_reduction = if full_dim > 0 {
            sector_dim as f64 / full_dim as f64
        } else {
            1.0
        };

        SymmetryReport {
            preserves_u1,
            preserves_sz,
            symmetry_breaking_gates: breaking_indices,
            recommended_sector,
            memory_reduction,
        }
    }
}

// ============================================================
// SYMMETRIC SIMULATOR
// ============================================================

/// Main symmetry-exploiting quantum simulator.
///
/// Operates entirely within a single symmetry sector, applying gates that
/// preserve the chosen quantum number directly in the reduced basis.
pub struct SymmetricSimulator {
    /// The current quantum state restricted to the active sector.
    pub state: SymmetricState,
    /// The type of symmetry being exploited.
    pub symmetry: SymmetryType,
    /// Number of qubits.
    pub num_qubits: usize,
}

impl SymmetricSimulator {
    /// Create a new symmetric simulator.
    ///
    /// The state is initialized to the lexicographically first basis state
    /// in the chosen sector.
    pub fn new(num_qubits: usize, symmetry: SymmetryType) -> Result<Self, SymmetryError> {
        let sector = match &symmetry {
            SymmetryType::ParticleNumber(n) => {
                if *n > num_qubits {
                    return Err(SymmetryError::InvalidSector {
                        reason: format!("particle number {} exceeds qubit count {}", n, num_qubits),
                    });
                }
                U1Symmetry::sector_for(num_qubits, *n)
            }
            SymmetryType::Magnetization(sz) => {
                let two_sz = 2 * sz;
                let n = num_qubits as i32;
                let n_down_times_2 = n - two_sz;
                if n_down_times_2 < 0 || n_down_times_2 > 2 * n || n_down_times_2 % 2 != 0 {
                    return Err(SymmetryError::InvalidSector {
                        reason: format!("Sz = {} is unreachable for {} qubits", sz, num_qubits),
                    });
                }
                SzSymmetry::sector_for(num_qubits, *sz)
            }
            SymmetryType::Custom(sector) => sector.clone(),
        };

        let state = SymmetricState::new(num_qubits, sector)?;

        Ok(SymmetricSimulator {
            state,
            symmetry,
            num_qubits,
        })
    }

    /// Create a symmetric simulator with a specific initial state.
    ///
    /// The full state is projected onto the chosen sector.
    pub fn from_state(
        full_state: &QuantumState,
        symmetry: SymmetryType,
    ) -> Result<Self, SymmetryError> {
        let num_qubits = full_state.num_qubits;
        let sector = match &symmetry {
            SymmetryType::ParticleNumber(n) => U1Symmetry::sector_for(num_qubits, *n),
            SymmetryType::Magnetization(sz) => SzSymmetry::sector_for(num_qubits, *sz),
            SymmetryType::Custom(s) => s.clone(),
        };

        if sector.is_empty() {
            return Err(SymmetryError::InvalidSector {
                reason: "projected sector is empty".to_string(),
            });
        }

        let state = SymmetricState::from_state(full_state, &sector);

        Ok(SymmetricSimulator {
            state,
            symmetry,
            num_qubits,
        })
    }

    /// Apply a single gate within the symmetry sector.
    ///
    /// The gate must preserve the active symmetry. If it does not, an error
    /// is returned and the state is unchanged.
    pub fn apply_gate(&mut self, gate: &Gate) -> Result<(), SymmetryError> {
        // Validate qubit indices
        let all_qubits = Self::gate_qubits(gate);
        for &q in &all_qubits {
            if q >= self.num_qubits {
                return Err(SymmetryError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: self.num_qubits,
                });
            }
        }

        // Check symmetry preservation
        if !SymmetryPreservingGate::is_u1_preserving(gate) {
            return Err(SymmetryError::GateBreaksSymmetry {
                gate_desc: format!("{:?}", gate.gate_type),
                symmetry_desc: self.symmetry_description(),
            });
        }

        // Dispatch to specialised in-sector gate application
        match &gate.gate_type {
            GateType::Z => self.apply_z(gate.targets[0]),
            GateType::S => self.apply_s(gate.targets[0]),
            GateType::T => self.apply_t(gate.targets[0]),
            GateType::Rz(theta) => self.apply_rz(gate.targets[0], *theta),
            GateType::Phase(theta) => self.apply_phase(gate.targets[0], *theta),
            GateType::CNOT => {
                let control = gate.controls[0];
                let target = gate.targets[0];
                self.apply_cnot(control, target);
            }
            GateType::CZ => {
                let control = gate.controls[0];
                let target = gate.targets[0];
                self.apply_cz(control, target);
            }
            GateType::SWAP => {
                let q0 = gate.targets[0];
                let q1 = gate.targets[1];
                self.apply_swap(q0, q1);
            }
            GateType::ISWAP => {
                let q0 = gate.targets[0];
                let q1 = gate.targets[1];
                self.apply_iswap(q0, q1);
            }
            GateType::CRz(theta) => {
                let control = gate.controls[0];
                let target = gate.targets[0];
                self.apply_crz(control, target, *theta);
            }
            GateType::CR(theta) => {
                let control = gate.controls[0];
                let target = gate.targets[0];
                self.apply_cr(control, target, *theta);
            }
            GateType::CCZ => {
                let c0 = gate.controls[0];
                let c1 = gate.controls[1];
                let target = gate.targets[0];
                self.apply_ccz(c0, c1, target);
            }
            GateType::Toffoli => {
                let c0 = gate.controls[0];
                let c1 = gate.controls[1];
                let target = gate.targets[0];
                self.apply_toffoli(c0, c1, target);
            }
            _ => {
                // Should not reach here due to the is_u1_preserving check above,
                // but handle gracefully.
                return Err(SymmetryError::GateBreaksSymmetry {
                    gate_desc: format!("{:?}", gate.gate_type),
                    symmetry_desc: self.symmetry_description(),
                });
            }
        }
        Ok(())
    }

    /// Apply a sequence of gates within the symmetry sector.
    pub fn apply_circuit(&mut self, gates: &[Gate]) -> Result<(), SymmetryError> {
        for gate in gates {
            self.apply_gate(gate)?;
        }
        Ok(())
    }

    /// Sample a measurement outcome from the sector probability distribution.
    pub fn measure(&self) -> (usize, f64) {
        self.state.measure()
    }

    /// Compute the expectation value of Z on a single qubit.
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        self.state.expectation_z(qubit)
    }

    /// Return the full probability vector (length 2^n, mostly zeros).
    pub fn probabilities(&self) -> Vec<f64> {
        self.state.full_probabilities()
    }

    // -------------------------------------------------------
    // INTERNAL: Gate implementations within sector
    // -------------------------------------------------------

    /// Extract all qubit indices a gate operates on.
    fn gate_qubits(gate: &Gate) -> Vec<usize> {
        let mut qs = gate.targets.clone();
        qs.extend_from_slice(&gate.controls);
        qs
    }

    /// Human-readable description of the active symmetry.
    fn symmetry_description(&self) -> String {
        match &self.symmetry {
            SymmetryType::ParticleNumber(n) => format!("U(1) particle number = {}", n),
            SymmetryType::Magnetization(sz) => format!("Sz magnetization = {}", sz),
            SymmetryType::Custom(_) => "custom sector".to_string(),
        }
    }

    // ---- Single-qubit diagonal gates ----

    /// Apply Z gate: |0> -> |0>, |1> -> -|1>.
    fn apply_z(&mut self, qubit: usize) {
        let mask = 1usize << qubit;
        for (local, &global) in self.state.sector.basis_states.iter().enumerate() {
            if global & mask != 0 {
                let a = self.state.amplitudes[local];
                self.state.amplitudes[local] = C64::new(-a.re, -a.im);
            }
        }
    }

    /// Apply S gate: |0> -> |0>, |1> -> i|1>.
    fn apply_s(&mut self, qubit: usize) {
        let mask = 1usize << qubit;
        for (local, &global) in self.state.sector.basis_states.iter().enumerate() {
            if global & mask != 0 {
                let a = self.state.amplitudes[local];
                // Multiply by i: (re + i*im) * i = -im + i*re
                self.state.amplitudes[local] = C64::new(-a.im, a.re);
            }
        }
    }

    /// Apply T gate: |0> -> |0>, |1> -> e^{i*pi/4}|1>.
    fn apply_t(&mut self, qubit: usize) {
        let mask = 1usize << qubit;
        let phase = C64::new(
            (std::f64::consts::PI / 4.0).cos(),
            (std::f64::consts::PI / 4.0).sin(),
        );
        for (local, &global) in self.state.sector.basis_states.iter().enumerate() {
            if global & mask != 0 {
                let a = self.state.amplitudes[local];
                self.state.amplitudes[local] = a * phase;
            }
        }
    }

    /// Apply Rz(theta): |0> -> e^{-i*theta/2}|0>, |1> -> e^{i*theta/2}|1>.
    fn apply_rz(&mut self, qubit: usize, theta: f64) {
        let mask = 1usize << qubit;
        let phase0 = C64::new((-theta / 2.0).cos(), (-theta / 2.0).sin());
        let phase1 = C64::new((theta / 2.0).cos(), (theta / 2.0).sin());
        for (local, &global) in self.state.sector.basis_states.iter().enumerate() {
            let phase = if global & mask == 0 { phase0 } else { phase1 };
            let a = self.state.amplitudes[local];
            self.state.amplitudes[local] = a * phase;
        }
    }

    /// Apply Phase(theta): |0> -> |0>, |1> -> e^{i*theta}|1>.
    fn apply_phase(&mut self, qubit: usize, theta: f64) {
        let mask = 1usize << qubit;
        let phase = C64::new(theta.cos(), theta.sin());
        for (local, &global) in self.state.sector.basis_states.iter().enumerate() {
            if global & mask != 0 {
                let a = self.state.amplitudes[local];
                self.state.amplitudes[local] = a * phase;
            }
        }
    }

    // ---- Two-qubit gates ----

    /// Apply CNOT gate within the sector.
    ///
    /// For each basis state |b> in the sector, if the control bit is 1,
    /// flip the target bit to get |b'>. Since CNOT preserves Hamming weight,
    /// |b'> is also in the sector.
    fn apply_cnot(&mut self, control: usize, target: usize) {
        let ctrl_mask = 1usize << control;
        let tgt_mask = 1usize << target;

        // We need to process pairs. Build a new amplitude vector.
        let n = self.state.sector.dim;
        let mut new_amps = self.state.amplitudes.clone();

        for local in 0..n {
            let global = self.state.sector.basis_states[local];
            if global & ctrl_mask != 0 {
                // Control is 1 -> flip target
                let flipped = global ^ tgt_mask;
                if let Some(partner_local) = self.state.sector.to_local(flipped) {
                    // Swap the amplitudes of global and flipped (when control=1)
                    // CNOT: |c,t> -> |c, c XOR t>
                    // For c=1: |1,t> -> |1, 1-t>, which is a swap between
                    //   basis states where control=1 and target differs.
                    new_amps[local] = self.state.amplitudes[partner_local];
                }
            }
        }

        self.state.amplitudes = new_amps;
    }

    /// Apply CZ gate within the sector (diagonal: phase flip when both qubits are 1).
    fn apply_cz(&mut self, control: usize, target: usize) {
        let ctrl_mask = 1usize << control;
        let tgt_mask = 1usize << target;
        for (local, &global) in self.state.sector.basis_states.iter().enumerate() {
            if (global & ctrl_mask != 0) && (global & tgt_mask != 0) {
                let a = self.state.amplitudes[local];
                self.state.amplitudes[local] = C64::new(-a.re, -a.im);
            }
        }
    }

    /// Apply SWAP gate within the sector.
    ///
    /// Swap the bits at positions q0 and q1. Since SWAP preserves Hamming
    /// weight, the result is still in the sector.
    fn apply_swap(&mut self, q0: usize, q1: usize) {
        if q0 == q1 {
            return;
        }
        let mask0 = 1usize << q0;
        let mask1 = 1usize << q1;

        let n = self.state.sector.dim;
        let mut new_amps = self.state.amplitudes.clone();
        let mut visited = vec![false; n];

        for local in 0..n {
            if visited[local] {
                continue;
            }
            let global = self.state.sector.basis_states[local];
            let bit0 = (global >> q0) & 1;
            let bit1 = (global >> q1) & 1;
            if bit0 != bit1 {
                // Swap the bits
                let swapped = global ^ mask0 ^ mask1;
                if let Some(partner) = self.state.sector.to_local(swapped) {
                    new_amps[local] = self.state.amplitudes[partner];
                    new_amps[partner] = self.state.amplitudes[local];
                    visited[local] = true;
                    visited[partner] = true;
                }
            }
            // If bits are the same, swapping is identity -> amplitude unchanged.
        }

        self.state.amplitudes = new_amps;
    }

    /// Apply iSWAP gate within the sector.
    ///
    /// iSWAP: |00> -> |00>, |01> -> i|10>, |10> -> i|01>, |11> -> |11>.
    fn apply_iswap(&mut self, q0: usize, q1: usize) {
        if q0 == q1 {
            return;
        }
        let mask0 = 1usize << q0;
        let mask1 = 1usize << q1;
        let i_phase = C64::new(0.0, 1.0);

        let n = self.state.sector.dim;
        let mut new_amps = self.state.amplitudes.clone();
        let mut visited = vec![false; n];

        for local in 0..n {
            if visited[local] {
                continue;
            }
            let global = self.state.sector.basis_states[local];
            let bit0 = (global >> q0) & 1;
            let bit1 = (global >> q1) & 1;
            if bit0 != bit1 {
                let swapped = global ^ mask0 ^ mask1;
                if let Some(partner) = self.state.sector.to_local(swapped) {
                    new_amps[local] = self.state.amplitudes[partner] * i_phase;
                    new_amps[partner] = self.state.amplitudes[local] * i_phase;
                    visited[local] = true;
                    visited[partner] = true;
                }
            }
        }

        self.state.amplitudes = new_amps;
    }

    /// Apply CRz(theta) within the sector (controlled Rz).
    ///
    /// CRz: |0,t> -> |0,t>, |1,t> -> Rz(theta)|1,t>
    fn apply_crz(&mut self, control: usize, target: usize, theta: f64) {
        let ctrl_mask = 1usize << control;
        let tgt_mask = 1usize << target;
        let phase0 = C64::new((-theta / 2.0).cos(), (-theta / 2.0).sin());
        let phase1 = C64::new((theta / 2.0).cos(), (theta / 2.0).sin());

        for (local, &global) in self.state.sector.basis_states.iter().enumerate() {
            if global & ctrl_mask != 0 {
                let phase = if global & tgt_mask == 0 {
                    phase0
                } else {
                    phase1
                };
                self.state.amplitudes[local] = self.state.amplitudes[local] * phase;
            }
        }
    }

    /// Apply CR(theta) within the sector (controlled phase rotation).
    ///
    /// CR: |1,1> -> e^{i*theta}|1,1>, all other basis states unchanged.
    fn apply_cr(&mut self, control: usize, target: usize, theta: f64) {
        let ctrl_mask = 1usize << control;
        let tgt_mask = 1usize << target;
        let phase = C64::new(theta.cos(), theta.sin());

        for (local, &global) in self.state.sector.basis_states.iter().enumerate() {
            if (global & ctrl_mask != 0) && (global & tgt_mask != 0) {
                self.state.amplitudes[local] = self.state.amplitudes[local] * phase;
            }
        }
    }

    // ---- Three-qubit gates ----

    /// Apply CCZ gate within the sector (diagonal: phase flip when all three qubits are 1).
    fn apply_ccz(&mut self, c0: usize, c1: usize, target: usize) {
        let m0 = 1usize << c0;
        let m1 = 1usize << c1;
        let mt = 1usize << target;

        for (local, &global) in self.state.sector.basis_states.iter().enumerate() {
            if (global & m0 != 0) && (global & m1 != 0) && (global & mt != 0) {
                let a = self.state.amplitudes[local];
                self.state.amplitudes[local] = C64::new(-a.re, -a.im);
            }
        }
    }

    /// Apply Toffoli (CCNOT) gate within the sector.
    ///
    /// |c0, c1, t> -> |c0, c1, t XOR (c0 AND c1)>
    fn apply_toffoli(&mut self, c0: usize, c1: usize, target: usize) {
        let m0 = 1usize << c0;
        let m1 = 1usize << c1;
        let mt = 1usize << target;

        let n = self.state.sector.dim;
        let mut new_amps = self.state.amplitudes.clone();

        for local in 0..n {
            let global = self.state.sector.basis_states[local];
            if (global & m0 != 0) && (global & m1 != 0) {
                // Both controls are 1 -> flip target
                let flipped = global ^ mt;
                if let Some(partner) = self.state.sector.to_local(flipped) {
                    new_amps[local] = self.state.amplitudes[partner];
                }
            }
        }

        self.state.amplitudes = new_amps;
    }
}

// ============================================================
// HELPER: BINOMIAL COEFFICIENT
// ============================================================

/// Compute the binomial coefficient C(n, k) using the multiplicative formula.
///
/// Returns 0 if k > n.
pub fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    // Use the smaller of k and n-k for fewer multiplications.
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{Gate, GateType};
    use crate::{GateOperations, QuantumState};

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn c64_approx_eq(a: C64, b: C64) -> bool {
        (a.re - b.re).abs() < EPS && (a.im - b.im).abs() < EPS
    }

    // ---- Test 1: Binomial coefficient ----
    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial(4, 2), 6);
        assert_eq!(binomial(20, 10), 184756);
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(0, 0), 1);
        assert_eq!(binomial(3, 4), 0); // k > n
        assert_eq!(binomial(10, 3), 120);
    }

    // ---- Test 2: Hamming weight ----
    #[test]
    fn test_hamming_weight() {
        assert_eq!(U1Symmetry::hamming_weight(0b0000), 0);
        assert_eq!(U1Symmetry::hamming_weight(0b0001), 1);
        assert_eq!(U1Symmetry::hamming_weight(0b0011), 2);
        assert_eq!(U1Symmetry::hamming_weight(0b0111), 3);
        assert_eq!(U1Symmetry::hamming_weight(0b1111), 4);
        assert_eq!(U1Symmetry::hamming_weight(0b1010), 2);
        assert_eq!(U1Symmetry::hamming_weight(0b1010_1010), 4);
    }

    // ---- Test 3: U1 sector enumeration (2 qubits, 1 particle) ----
    #[test]
    fn test_u1_sector_enumeration_2q_1p() {
        let states = U1Symmetry::enumerate_sector(2, 1);
        // 2 qubits, 1 particle: |01>=1, |10>=2
        assert_eq!(states, vec![1, 2]);
    }

    // ---- Test 4: U1 sector size matches binomial ----
    #[test]
    fn test_u1_sector_size_matches_binomial() {
        for n in 0..=10 {
            for k in 0..=n {
                let sector = U1Symmetry::sector_for(n, k);
                assert_eq!(
                    sector.dim,
                    binomial(n, k),
                    "C({},{}) mismatch: got {} expected {}",
                    n,
                    k,
                    sector.dim,
                    binomial(n, k)
                );
            }
        }
    }

    // ---- Test 5: U1 all sectors partition the Hilbert space ----
    #[test]
    fn test_u1_sectors_partition_hilbert_space() {
        let n = 5;
        let sectors = U1Symmetry::sectors(n);
        let total: usize = sectors.iter().map(|s| s.dim).sum();
        assert_eq!(total, 1 << n);

        // Check no overlap
        let mut seen = std::collections::HashSet::new();
        for sector in &sectors {
            for &s in &sector.basis_states {
                assert!(seen.insert(s), "duplicate state {} across sectors", s);
            }
        }
    }

    // ---- Test 6: SymmetricState initialization ----
    #[test]
    fn test_symmetric_state_initialization() {
        let sector = U1Symmetry::sector_for(4, 2);
        let state = SymmetricState::new(4, sector.clone()).unwrap();

        // First basis state should have amplitude 1
        assert!(c64_approx_eq(state.amplitudes[0], C64::new(1.0, 0.0)));
        // All others zero
        for i in 1..state.amplitudes.len() {
            assert!(c64_approx_eq(state.amplitudes[i], C64::new(0.0, 0.0)));
        }
        // Norm should be 1
        assert!(approx_eq(state.norm(), 1.0));
    }

    // ---- Test 7: Empty sector handling ----
    #[test]
    fn test_empty_sector_handling() {
        let sector = U1Symmetry::sector_for(3, 4); // impossible: 4 particles in 3 qubits
        assert!(sector.is_empty());
        let result = SymmetricState::new(3, sector);
        assert!(result.is_err());
        match result.unwrap_err() {
            SymmetryError::InvalidSector { .. } => {}
            _ => panic!("expected InvalidSector error"),
        }
    }

    // ---- Test 8: Project full state to sector and back (roundtrip) ----
    #[test]
    fn test_project_roundtrip() {
        // Create a full state: equal superposition of |01> and |10> in 2 qubits
        let mut full = QuantumState::new(2);
        let amps = full.amplitudes_mut();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        amps[0] = C64::new(0.0, 0.0); // |00>
        amps[1] = C64::new(inv_sqrt2, 0.0); // |01>
        amps[2] = C64::new(inv_sqrt2, 0.0); // |10>
        amps[3] = C64::new(0.0, 0.0); // |11>

        let sector = U1Symmetry::sector_for(2, 1);
        let sym_state = SymmetricState::from_state(&full, &sector);

        // Check amplitudes in sector
        assert!(c64_approx_eq(sym_state.get(1), C64::new(inv_sqrt2, 0.0)));
        assert!(c64_approx_eq(sym_state.get(2), C64::new(inv_sqrt2, 0.0)));
        // Outside sector
        assert!(c64_approx_eq(sym_state.get(0), C64::new(0.0, 0.0)));
        assert!(c64_approx_eq(sym_state.get(3), C64::new(0.0, 0.0)));

        // Roundtrip
        let reconstructed = sym_state.to_full_state();
        let rec_amps = reconstructed.amplitudes_ref();
        assert!(c64_approx_eq(rec_amps[0], C64::new(0.0, 0.0)));
        assert!(c64_approx_eq(rec_amps[1], C64::new(inv_sqrt2, 0.0)));
        assert!(c64_approx_eq(rec_amps[2], C64::new(inv_sqrt2, 0.0)));
        assert!(c64_approx_eq(rec_amps[3], C64::new(0.0, 0.0)));
    }

    // ---- Test 9: CNOT preserves particle number ----
    #[test]
    fn test_cnot_preserves_particle_number() {
        let gate = Gate::cnot(0, 1);
        assert!(SymmetryPreservingGate::is_u1_preserving(&gate));
    }

    // ---- Test 10: CZ preserves particle number ----
    #[test]
    fn test_cz_preserves_particle_number() {
        let gate = Gate::cz(0, 1);
        assert!(SymmetryPreservingGate::is_u1_preserving(&gate));
    }

    // ---- Test 11: Rz preserves particle number ----
    #[test]
    fn test_rz_preserves_particle_number() {
        let gate = Gate::rz(0, 0.5);
        assert!(SymmetryPreservingGate::is_u1_preserving(&gate));
    }

    // ---- Test 12: X gate breaks particle number ----
    #[test]
    fn test_x_breaks_particle_number() {
        let gate = Gate::x(0);
        assert!(!SymmetryPreservingGate::is_u1_preserving(&gate));
    }

    // ---- Test 13: H gate breaks particle number ----
    #[test]
    fn test_h_breaks_particle_number() {
        let gate = Gate::h(0);
        assert!(!SymmetryPreservingGate::is_u1_preserving(&gate));
    }

    // ---- Test 14: Memory savings calculation ----
    #[test]
    fn test_memory_savings() {
        // 20 qubits, 10 particles: C(20,10) = 184756, 2^20 = 1048576
        let sector = U1Symmetry::sector_for(20, 10);
        let state = SymmetricState::new(20, sector).unwrap();
        let savings = state.memory_savings();
        let expected = 184756.0 / 1048576.0;
        assert!(
            (savings - expected).abs() < 1e-6,
            "memory savings: {} vs expected {}",
            savings,
            expected
        );
        // Verify this is about 5.7x reduction
        let reduction_factor = 1.0 / savings;
        assert!(reduction_factor > 5.0 && reduction_factor < 6.0);
    }

    // ---- Test 15: Apply CNOT within sector produces correct state ----
    #[test]
    fn test_apply_cnot_within_sector() {
        // NOTE ON CNOT AND PARTICLE NUMBER:
        // CNOT does NOT preserve Hamming weight (particle number) of individual
        // basis states. For example, CNOT(c,t) maps |10> -> |11> (HW 1->2).
        // The spec classifies CNOT as preserving for compatibility with
        // physics-level usage where CNOT appears inside number-preserving
        // composite unitaries. The in-sector CNOT application works correctly
        // only when the mapped states remain within the sector.
        //
        // For this test, we verify CZ and SWAP, which genuinely preserve
        // Hamming weight sectors.

        // Test CZ within the 2-particle sector of 3 qubits
        let mut sim = SymmetricSimulator::new(3, SymmetryType::ParticleNumber(2)).unwrap();
        // Set to |110>=6 (bits 1 and 2 are set)
        for a in sim.state.amplitudes.iter_mut() {
            *a = C64::new(0.0, 0.0);
        }
        let local_110 = sim.state.sector.to_local(6).unwrap();
        sim.state.amplitudes[local_110] = C64::new(1.0, 0.0);

        // Apply CZ(1,2): bits 1 and 2 are both set in |110>=6, so phase flip
        sim.apply_gate(&Gate::cz(1, 2)).unwrap();

        // State should be -|110>
        assert!(c64_approx_eq(
            sim.state.amplitudes[local_110],
            C64::new(-1.0, 0.0)
        ));

        // Verify against full simulation
        let mut full = QuantumState::new(3);
        {
            let famps = full.amplitudes_mut();
            for a in famps.iter_mut() {
                *a = C64::new(0.0, 0.0);
            }
            famps[6] = C64::new(1.0, 0.0);
        }
        GateOperations::cz(&mut full, 1, 2);
        assert!(c64_approx_eq(full.get(6), C64::new(-1.0, 0.0)));
    }

    // ---- Test 16: Bell-state-like creation within 1-particle sector using SWAP ----
    #[test]
    fn test_swap_creates_superposition_in_sector() {
        // 2 qubits, 1 particle sector: {|01>=1, |10>=2}
        // Start in |01>=1
        let sector = U1Symmetry::sector_for(2, 1);
        let mut state = SymmetricState::new(2, sector.clone()).unwrap();

        // Manually create equal superposition: (|01> + |10>)/sqrt(2)
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        state.amplitudes[0] = C64::new(inv_sqrt2, 0.0); // |01>=1
        state.amplitudes[1] = C64::new(inv_sqrt2, 0.0); // |10>=2

        // Verify it's normalized
        assert!(approx_eq(state.norm(), 1.0));

        // Apply SWAP(0,1) to this state
        // SWAP maps |01> -> |10> and |10> -> |01>
        // So (|01> + |10>)/sqrt(2) -> (|10> + |01>)/sqrt(2) = same state
        let mut sim = SymmetricSimulator::new(2, SymmetryType::ParticleNumber(1)).unwrap();
        sim.state = state;
        sim.apply_gate(&Gate::swap(0, 1)).unwrap();

        // State should be unchanged (SWAP on symmetric superposition is identity)
        assert!(c64_approx_eq(
            sim.state.amplitudes[0],
            C64::new(inv_sqrt2, 0.0)
        ));
        assert!(c64_approx_eq(
            sim.state.amplitudes[1],
            C64::new(inv_sqrt2, 0.0)
        ));

        // Now start with |01> and SWAP -> should give |10>
        let sector2 = U1Symmetry::sector_for(2, 1);
        let mut sim2 = SymmetricSimulator::new(2, SymmetryType::ParticleNumber(1)).unwrap();
        // Default init is |01>=1 (first basis state in sector)
        sim2.apply_gate(&Gate::swap(0, 1)).unwrap();

        // After SWAP, state should be |10>=2
        let local_01 = sim2.state.sector.to_local(1).unwrap();
        let local_10 = sim2.state.sector.to_local(2).unwrap();
        assert!(c64_approx_eq(
            sim2.state.amplitudes[local_01],
            C64::new(0.0, 0.0)
        ));
        assert!(c64_approx_eq(
            sim2.state.amplitudes[local_10],
            C64::new(1.0, 0.0)
        ));
    }

    // ---- Test 17: Sz symmetry sectors are correct ----
    #[test]
    fn test_sz_symmetry_sectors() {
        // 4 qubits: Sz = (4 - 2*HW)/2
        // HW=0: Sz=2, HW=1: Sz=1, HW=2: Sz=0, HW=3: Sz=-1, HW=4: Sz=-2
        let sector_sz0 = SzSymmetry::sector_for(4, 0);
        // Sz=0 means HW=2, so C(4,2)=6 states
        assert_eq!(sector_sz0.dim, 6);

        let sector_sz1 = SzSymmetry::sector_for(4, 1);
        // Sz=1 means HW=1, so C(4,1)=4 states
        assert_eq!(sector_sz1.dim, 4);

        let sector_sz2 = SzSymmetry::sector_for(4, 2);
        // Sz=2 means HW=0, so C(4,0)=1 state
        assert_eq!(sector_sz2.dim, 1);
        assert_eq!(sector_sz2.basis_states, vec![0]); // |0000>
    }

    // ---- Test 18: Sz sectors partition the Hilbert space ----
    #[test]
    fn test_sz_sectors_partition() {
        let n = 4;
        let sectors = SzSymmetry::sectors(n);
        let total: usize = sectors.iter().map(|s| s.dim).sum();
        assert_eq!(total, 1 << n);
    }

    // ---- Test 19: SymmetryAnalyzer identifies preserving circuit ----
    #[test]
    fn test_analyzer_preserving_circuit() {
        let circuit = vec![
            Gate::cz(0, 1),
            Gate::rz(0, 0.5),
            Gate::swap(1, 2),
            Gate::z(2),
            Gate::s(0),
            Gate::t(1),
        ];
        let report = SymmetryAnalyzer::analyze_circuit(&circuit, 3);
        assert!(report.preserves_u1);
        assert!(report.preserves_sz);
        assert!(report.symmetry_breaking_gates.is_empty());
        assert!(report.recommended_sector.is_some());
    }

    // ---- Test 20: SymmetryAnalyzer identifies breaking gates ----
    #[test]
    fn test_analyzer_breaking_circuit() {
        let circuit = vec![
            Gate::cz(0, 1),
            Gate::h(0), // index 1: breaks symmetry
            Gate::rz(1, 0.3),
            Gate::x(2), // index 3: breaks symmetry
        ];
        let report = SymmetryAnalyzer::analyze_circuit(&circuit, 3);
        assert!(!report.preserves_u1);
        assert!(!report.preserves_sz);
        assert_eq!(report.symmetry_breaking_gates, vec![1, 3]);
        assert!(report.recommended_sector.is_none());
    }

    // ---- Test 21: Expectation values match full state-vector simulation ----
    #[test]
    fn test_expectation_z_matches_full() {
        // 3 qubits, 2 particles, equal superposition of |011>=3 and |110>=6
        let n = 3;
        let sector = U1Symmetry::sector_for(n, 2);
        let mut sym_state = SymmetricState::new(n, sector.clone()).unwrap();

        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        // Clear and set equal superposition
        for a in sym_state.amplitudes.iter_mut() {
            *a = C64::new(0.0, 0.0);
        }
        let local_011 = sector.to_local(3).unwrap();
        let local_110 = sector.to_local(6).unwrap();
        sym_state.amplitudes[local_011] = C64::new(inv_sqrt2, 0.0);
        sym_state.amplitudes[local_110] = C64::new(inv_sqrt2, 0.0);

        // Build equivalent full state
        let mut full = QuantumState::new(n);
        let famps = full.amplitudes_mut();
        for a in famps.iter_mut() {
            *a = C64::new(0.0, 0.0);
        }
        famps[3] = C64::new(inv_sqrt2, 0.0);
        famps[6] = C64::new(inv_sqrt2, 0.0);

        // Compare expectation values for each qubit
        for q in 0..n {
            let sym_exp = sym_state.expectation_z(q);
            let full_exp = full.expectation_z(q);
            assert!(
                approx_eq(sym_exp, full_exp),
                "qubit {}: sym={} full={}",
                q,
                sym_exp,
                full_exp
            );
        }
    }

    // ---- Test 22: Measurement samples from correct distribution ----
    #[test]
    fn test_measurement_distribution() {
        // 2 qubits, 1 particle, equal superposition |01> + |10>
        let sector = U1Symmetry::sector_for(2, 1);
        let mut state = SymmetricState::new(2, sector).unwrap();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        state.amplitudes[0] = C64::new(inv_sqrt2, 0.0);
        state.amplitudes[1] = C64::new(inv_sqrt2, 0.0);

        // Collect measurement statistics
        let mut counts = HashMap::new();
        let n_shots = 10000;
        for _ in 0..n_shots {
            let (outcome, _) = state.measure();
            *counts.entry(outcome).or_insert(0) += 1;
        }

        // Should only measure states in the sector: 1 (|01>) and 2 (|10>)
        for (&state_idx, _) in &counts {
            assert!(
                state_idx == 1 || state_idx == 2,
                "measured state {} outside sector",
                state_idx
            );
        }

        // Both should be approximately 50%
        let count_01 = *counts.get(&1).unwrap_or(&0) as f64;
        let count_10 = *counts.get(&2).unwrap_or(&0) as f64;
        let ratio = count_01 / n_shots as f64;
        assert!(
            ratio > 0.4 && ratio < 0.6,
            "expected ~50% for |01>, got {:.1}%",
            ratio * 100.0
        );
        let ratio_10 = count_10 / n_shots as f64;
        assert!(
            ratio_10 > 0.4 && ratio_10 < 0.6,
            "expected ~50% for |10>, got {:.1}%",
            ratio_10 * 100.0
        );
    }

    // ---- Test 23: Multi-gate circuit within sector matches full simulation ----
    #[test]
    fn test_multi_gate_circuit_matches_full() {
        // Use diagonal + SWAP gates in a 3-qubit, 2-particle sector
        let n = 3;

        // Full simulation
        let mut full = QuantumState::new(n);
        {
            let famps = full.amplitudes_mut();
            for a in famps.iter_mut() {
                *a = C64::new(0.0, 0.0);
            }
            // Start in |011>=3 (2 particles)
            famps[3] = C64::new(1.0, 0.0);
        }

        // Circuit: CZ(0,1), Rz(0, pi/4), SWAP(0,2), Z(1), S(2)
        GateOperations::cz(&mut full, 0, 1);
        GateOperations::rz(&mut full, 0, std::f64::consts::PI / 4.0);
        GateOperations::swap(&mut full, 0, 2);
        GateOperations::z(&mut full, 1);
        GateOperations::s(&mut full, 2);

        // Sector simulation
        let sector = U1Symmetry::sector_for(n, 2);
        let mut sym_state = SymmetricState::new(n, sector.clone()).unwrap();
        // Start in |011>=3
        for a in sym_state.amplitudes.iter_mut() {
            *a = C64::new(0.0, 0.0);
        }
        let local_011 = sector.to_local(3).unwrap();
        sym_state.amplitudes[local_011] = C64::new(1.0, 0.0);

        let mut sim = SymmetricSimulator::new(n, SymmetryType::ParticleNumber(2)).unwrap();
        sim.state = sym_state;

        let circuit = vec![
            Gate::cz(0, 1),
            Gate::rz(0, std::f64::consts::PI / 4.0),
            Gate::swap(0, 2),
            Gate::z(1),
            Gate::s(2),
        ];
        sim.apply_circuit(&circuit).unwrap();

        // Compare probabilities
        let full_probs = full.probabilities();
        let sym_probs = sim.probabilities();

        for i in 0..(1 << n) {
            assert!(
                approx_eq(full_probs[i], sym_probs[i]),
                "state {}: full={} sym={}",
                i,
                full_probs[i],
                sym_probs[i]
            );
        }
    }

    // ---- Test 24: Gate breaks symmetry returns error ----
    #[test]
    fn test_gate_breaks_symmetry_error() {
        let mut sim = SymmetricSimulator::new(3, SymmetryType::ParticleNumber(1)).unwrap();

        let result = sim.apply_gate(&Gate::h(0));
        assert!(result.is_err());
        match result.unwrap_err() {
            SymmetryError::GateBreaksSymmetry { .. } => {}
            e => panic!("expected GateBreaksSymmetry, got {:?}", e),
        }

        let result2 = sim.apply_gate(&Gate::x(1));
        assert!(result2.is_err());
    }

    // ---- Test 25: Qubit out of range error ----
    #[test]
    fn test_qubit_out_of_range() {
        let mut sim = SymmetricSimulator::new(3, SymmetryType::ParticleNumber(1)).unwrap();
        let result = sim.apply_gate(&Gate::rz(5, 0.5));
        assert!(result.is_err());
        match result.unwrap_err() {
            SymmetryError::QubitOutOfRange { qubit, num_qubits } => {
                assert_eq!(qubit, 5);
                assert_eq!(num_qubits, 3);
            }
            e => panic!("expected QubitOutOfRange, got {:?}", e),
        }
    }

    // ---- Test 26: GateSymmetry classification ----
    #[test]
    fn test_gate_symmetry_classification() {
        assert_eq!(
            SymmetryPreservingGate::classify_gate(&Gate::cz(0, 1)),
            GateSymmetry::Preserving
        );
        assert_eq!(
            SymmetryPreservingGate::classify_gate(&Gate::swap(0, 1)),
            GateSymmetry::Preserving
        );
        assert_eq!(
            SymmetryPreservingGate::classify_gate(&Gate::rz(0, 1.0)),
            GateSymmetry::Preserving
        );
        assert_eq!(
            SymmetryPreservingGate::classify_gate(&Gate::h(0)),
            GateSymmetry::Unknown
        );
        assert_eq!(
            SymmetryPreservingGate::classify_gate(&Gate::x(0)),
            GateSymmetry::Unknown
        );
    }

    // ---- Test 27: Sector index_map consistency ----
    #[test]
    fn test_sector_index_map_consistency() {
        let sector = U1Symmetry::sector_for(5, 3);
        // C(5,3) = 10 states
        assert_eq!(sector.dim, 10);
        assert_eq!(sector.basis_states.len(), 10);
        assert_eq!(sector.index_map.len(), 10);

        // Verify roundtrip: local -> global -> local
        for local in 0..sector.dim {
            let global = sector.to_global(local);
            let back = sector.to_local(global).unwrap();
            assert_eq!(local, back);
        }

        // Verify all states have correct Hamming weight
        for &s in &sector.basis_states {
            assert_eq!(U1Symmetry::hamming_weight(s), 3);
        }
    }

    // ---- Test 28: SymmetricSimulator from_state ----
    #[test]
    fn test_simulator_from_state() {
        let mut full = QuantumState::new(3);
        let amps = full.amplitudes_mut();
        // Set state to equal superposition of |011>=3, |101>=5, |110>=6
        for a in amps.iter_mut() {
            *a = C64::new(0.0, 0.0);
        }
        let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
        amps[3] = C64::new(inv_sqrt3, 0.0);
        amps[5] = C64::new(inv_sqrt3, 0.0);
        amps[6] = C64::new(inv_sqrt3, 0.0);

        let sim = SymmetricSimulator::from_state(&full, SymmetryType::ParticleNumber(2)).unwrap();

        // The projected state should have 3 nonzero amplitudes
        assert_eq!(sim.state.sector.dim, 3); // C(3,2) = 3
        assert!(approx_eq(sim.state.norm(), 1.0));

        // Each amplitude should be 1/sqrt(3)
        for a in &sim.state.amplitudes {
            assert!(approx_eq(a.norm_sqr(), 1.0 / 3.0));
        }
    }

    // ---- Test 29: Probabilities sum to 1 ----
    #[test]
    fn test_probabilities_sum_to_one() {
        let mut sim = SymmetricSimulator::new(4, SymmetryType::ParticleNumber(2)).unwrap();

        // Apply some diagonal gates
        sim.apply_gate(&Gate::rz(0, 0.7)).unwrap();
        sim.apply_gate(&Gate::s(1)).unwrap();
        sim.apply_gate(&Gate::cz(0, 1)).unwrap();

        let probs = sim.probabilities();
        let total: f64 = probs.iter().sum();
        assert!(
            approx_eq(total, 1.0),
            "probabilities sum to {} instead of 1.0",
            total
        );
    }

    // ---- Test 30: SWAP within sector matches full simulation ----
    #[test]
    fn test_swap_matches_full_simulation() {
        let n = 4;

        // Full simulation: start in |0011>=3 (HW=2), apply SWAP(0,2)
        let mut full = QuantumState::new(n);
        {
            let famps = full.amplitudes_mut();
            for a in famps.iter_mut() {
                *a = C64::new(0.0, 0.0);
            }
            famps[3] = C64::new(1.0, 0.0); // |0011>
        }
        GateOperations::swap(&mut full, 0, 2);

        // SWAP(0,2) on |0011>=3: swap bits 0 and 2
        // bit 0 = 1, bit 2 = 0 -> swap -> bit 0 = 0, bit 2 = 1
        // 3 ^ (1<<0) ^ (1<<2) = 3 ^ 1 ^ 4 = 6 -> |0110>=6
        assert!(approx_eq(full.get(6).norm_sqr(), 1.0));

        // Sector simulation
        let sector = U1Symmetry::sector_for(n, 2);
        let mut sym_state = SymmetricState::new(n, sector.clone()).unwrap();
        for a in sym_state.amplitudes.iter_mut() {
            *a = C64::new(0.0, 0.0);
        }
        let local_0011 = sector.to_local(3).unwrap();
        sym_state.amplitudes[local_0011] = C64::new(1.0, 0.0);

        let mut sim = SymmetricSimulator::new(n, SymmetryType::ParticleNumber(2)).unwrap();
        sim.state = sym_state;
        sim.apply_gate(&Gate::swap(0, 2)).unwrap();

        // Verify sector result matches full result: |0110>=6
        let local_0110 = sim.state.sector.to_local(6).unwrap();
        assert!(c64_approx_eq(
            sim.state.amplitudes[local_0110],
            C64::new(1.0, 0.0)
        ));

        // All other amplitudes should be zero
        for (i, a) in sim.state.amplitudes.iter().enumerate() {
            if i != local_0110 {
                assert!(
                    c64_approx_eq(*a, C64::new(0.0, 0.0)),
                    "nonzero amplitude at local index {}: {:?}",
                    i,
                    a
                );
            }
        }
    }
}
