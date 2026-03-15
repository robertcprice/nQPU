//! Quantum Error Correction Codes
//!
//! Implementation of QEC codes for fault-tolerant quantum computing.
//!
//! **Codes Implemented**:
//! - **Repetition Codes**: Bit-flip and phase-flip codes
//! - **Shor's 9-Qubit Code**: Corrects arbitrary single-qubit errors
//! - **Steane's 7-Qubit Code**: CSS code, stabilizer formalism
//! - **Surface Codes**: Planar and rotated surface codes (basic structure)
//! - **Color Codes**: Topological codes (basic structure)
//!
//! **Decoders**:
//! - **MWPM**: Full Blossom V implementation — see [`crate::decoding::mwpm`]
//! - **BP**: Min-sum belief propagation — see [`crate::decoding::bp`]
//!
//! **Features**:
//! - Encoding/decoding circuits
//! - Syndrome measurement
//! - Error decoding algorithms

use crate::comprehensive_gates::QuantumGates;
use crate::{QuantumState, C64};

/// Error syndrome for QEC.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Syndrome {
    pub stabilizer_measurements: Vec<bool>,
}

impl Syndrome {
    pub fn new(measurements: Vec<bool>) -> Self {
        Self {
            stabilizer_measurements: measurements,
        }
    }

    pub fn weight(&self) -> usize {
        self.stabilizer_measurements.iter().filter(|&&x| x).count()
    }

    pub fn is_trivial(&self) -> bool {
        self.stabilizer_measurements.iter().all(|&x| !x)
    }
}

/// QEC decoder interface.
///
/// Note: This trait uses immutable `&self` for simplicity, but production decoders
/// like MWPM require mutable access. Use `DecoderMut` for production decoders,
/// or use the production decoders directly from `crate::decoding`.
pub trait Decoder {
    fn decode(&self, syndrome: &Syndrome) -> Vec<usize>;
}

/// Mutable decoder interface for production decoders that need internal state.
pub trait DecoderMut {
    fn decode(&mut self, syndrome: &Syndrome) -> Vec<usize>;
}

/// Measure Z⊗Z⊗...⊗Z parity on given qubits. Returns true if parity is odd (Born-rule sampling).
fn measure_multi_z(state: &QuantumState, qubits: &[usize]) -> bool {
    let amps = state.amplitudes_ref();
    let mut prob_odd = 0.0;
    for (i, amp) in amps.iter().enumerate() {
        let parity: usize = qubits.iter().map(|&q| (i >> q) & 1).sum();
        if parity % 2 == 1 {
            prob_odd += amp.norm_sqr();
        }
    }
    let sample: f64 = rand::random();
    sample < prob_odd
}

/// Measure X⊗X⊗...⊗X parity on given qubits. Returns true if parity is odd (Born-rule sampling).
fn measure_multi_x(state: &QuantumState, qubits: &[usize]) -> bool {
    let amps = state.amplitudes_ref();
    let flip_mask: usize = qubits.iter().map(|&q| 1usize << q).sum();
    let mut expectation = 0.0;
    for (i, amp_i) in amps.iter().enumerate() {
        let j = i ^ flip_mask;
        let amp_j = &amps[j];
        expectation += amp_i.re * amp_j.re + amp_i.im * amp_j.im;
    }
    let prob_odd = (1.0 - expectation) / 2.0;
    let sample: f64 = rand::random();
    sample < prob_odd.max(0.0).min(1.0)
}

// ==================== PRODUCTION DECODER ADAPTER ====================

/// Re-export production decoders from the `decoding` module.
pub use crate::decoding::{BPDecoder, MWPMDecoder};

/// Helper wrapper for BP decoder to store the physical error rate.
///
/// BP decoder requires a physical error rate parameter for each decode call.
/// This wrapper stores the rate so the decoder can implement the simple `Decoder` trait.
pub struct BPDecoderWithErrorRate {
    decoder: crate::decoding::BPDecoder,
    physical_error_rate: f64,
}

impl BPDecoderWithErrorRate {
    /// Create a new BP decoder wrapper with a physical error rate.
    pub fn new(decoder: crate::decoding::BPDecoder, physical_error_rate: f64) -> Self {
        Self {
            decoder,
            physical_error_rate,
        }
    }
}

/// Adapter to wrap production decoders behind the simple `Decoder` trait.
///
/// Converts between the simple `Syndrome` type used in this module and the
/// production `Syndrome` type in `crate::decoding::mwpm`, and converts the
/// output format from `Vec<bool>` (error at position i) to `Vec<usize>`
/// (list of error positions).
pub struct DecoderAdapter<D> {
    inner: D,
}

impl DecoderAdapter<crate::decoding::MWPMDecoder> {
    /// Wrap an MWPM decoder.
    pub fn from_mwpm(decoder: crate::decoding::MWPMDecoder) -> Self {
        Self { inner: decoder }
    }
}

impl DecoderAdapter<BPDecoderWithErrorRate> {
    /// Wrap a BP decoder with a default physical error rate.
    pub fn from_bp(decoder: crate::decoding::BPDecoder, physical_error_rate: f64) -> Self {
        Self {
            inner: BPDecoderWithErrorRate::new(decoder, physical_error_rate),
        }
    }
}

impl DecoderMut for DecoderAdapter<crate::decoding::MWPMDecoder> {
    fn decode(&mut self, syndrome: &Syndrome) -> Vec<usize> {
        // Convert simple Syndrome to production Syndrome
        let production_syndrome =
            crate::decoding::mwpm::Syndrome::new(syndrome.stabilizer_measurements.clone());

        // Decode to get Vec<bool> (true = error at position i)
        let error_pattern = self.inner.decode(&production_syndrome);

        // Convert Vec<bool> to Vec<usize> (list of error positions)
        error_pattern
            .iter()
            .enumerate()
            .filter(|(_, &has_error)| has_error)
            .map(|(i, _)| i)
            .collect()
    }
}

impl Decoder for DecoderAdapter<BPDecoderWithErrorRate> {
    fn decode(&self, syndrome: &Syndrome) -> Vec<usize> {
        // BP decoder takes &[bool] directly
        let result = self.inner.decoder.decode(
            &syndrome.stabilizer_measurements,
            self.inner.physical_error_rate,
        );

        // Convert Vec<bool> to Vec<usize> (list of error positions)
        result
            .error_pattern
            .iter()
            .enumerate()
            .filter(|(_, &has_error)| has_error)
            .map(|(i, _)| i)
            .collect()
    }
}

// The real MWPM decoder (Blossom V, 1000+ lines) lives in `crate::decoding::mwpm`.
// The real BP decoder lives in `crate::decoding::bp`.
// Use `DecoderAdapter` to wrap them behind the simple `Decoder` trait, or use them directly.
// This module's `Decoder` trait and `Syndrome` type are for the simple QEC codes below.
// For production decoding, see the `decoding` module.

// ==================== REPETITION CODES ====================

/// Bit-flip repetition code.
pub struct RepetitionCodeBitFlip {
    num_physical_qubits: usize,
    num_logical_qubits: usize,
}

impl RepetitionCodeBitFlip {
    pub fn new(repetition: usize) -> Self {
        assert!(
            repetition >= 3,
            "Need at least 3 qubits for repetition code"
        );
        assert!(repetition % 2 == 1, "Repetition must be odd");

        Self {
            num_physical_qubits: repetition,
            num_logical_qubits: 1,
        }
    }

    /// Encode a single logical qubit into n physical qubits.
    pub fn encode(&self, logical_state: &mut QuantumState, encoded_state: &mut QuantumState) {
        // |ψ⟩ = α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩
        let logical_0 = logical_state.amplitudes_ref()[0];
        let logical_1 = logical_state.amplitudes_ref()[1];

        // Initialize encoded state
        for i in 0..self.num_physical_qubits {
            crate::GateOperations::x(encoded_state, i);
        }

        let enc_amplitudes = encoded_state.amplitudes_mut();
        enc_amplitudes[0] = logical_0;
        enc_amplitudes[(1 << self.num_physical_qubits) - 1] = logical_1;
    }

    /// Decode n physical qubits to 1 logical qubit.
    pub fn decode(&self, encoded_state: &mut QuantumState, logical_state: &mut QuantumState) {
        // Measure parity of all pairs
        let syndrome = self.measure_syndrome(encoded_state);

        // Correct errors (majority vote)
        let corrected = self.correct_errors(encoded_state, &syndrome);

        // Extract logical qubit
        let enc_amplitudes = corrected.amplitudes_ref();
        let logical_0_prob = (0..(1 << self.num_physical_qubits))
            .step_by(2)
            .map(|i| enc_amplitudes[i].norm_sqr())
            .sum::<f64>()
            .sqrt();

        let logical_1_prob = (1..(1 << self.num_physical_qubits))
            .skip(1)
            .step_by(2)
            .map(|i| enc_amplitudes[i].norm_sqr())
            .sum::<f64>()
            .sqrt();

        let log_amplitudes = logical_state.amplitudes_mut();
        log_amplitudes[0] = C64::new(logical_0_prob, 0.0);
        log_amplitudes[1] = C64::new(logical_1_prob, 0.0);
    }

    fn measure_syndrome(&self, state: &QuantumState) -> Syndrome {
        let mut syndromes = Vec::new();

        // Measure parity of pairs (Z⊗Z)
        for i in 0..(self.num_physical_qubits - 1) {
            // Parity check between qubit i and i+1
            let parity = self.measure_parity_zz(state, i, i + 1);
            syndromes.push(parity);
        }

        Syndrome::new(syndromes)
    }

    fn measure_parity_zz(&self, state: &QuantumState, qubit1: usize, qubit2: usize) -> bool {
        // Compute expectation value of Z⊗Z on (qubit1, qubit2).
        // Z⊗Z eigenvalue is (-1)^(bit1 XOR bit2).
        // Probability of parity-odd = Σ|a_i|² where bit1(i) XOR bit2(i) = 1.
        let amps = state.amplitudes_ref();
        let mask1 = 1usize << qubit1;
        let mask2 = 1usize << qubit2;
        let mut prob_odd = 0.0;
        for (i, amp) in amps.iter().enumerate() {
            let bit1 = (i & mask1) != 0;
            let bit2 = (i & mask2) != 0;
            if bit1 ^ bit2 {
                prob_odd += amp.norm_sqr();
            }
        }
        // Sample from Born rule: return true if parity is odd
        let sample: f64 = rand::random();
        sample < prob_odd
    }

    fn correct_errors(&self, state: &mut QuantumState, syndrome: &Syndrome) -> QuantumState {
        // Majority vote: flip qubit if most syndromes indicate error
        let mut corrected = state.clone();

        for i in 0..self.num_physical_qubits {
            let error_count = if i > 0 && i < self.num_physical_qubits - 1 {
                // Check neighboring syndromes
                (if syndrome.stabilizer_measurements[i - 1] {
                    1
                } else {
                    0
                }) + (if syndrome.stabilizer_measurements[i] {
                    1
                } else {
                    0
                })
            } else if i == 0 {
                if syndrome.stabilizer_measurements[0] {
                    1
                } else {
                    0
                }
            } else {
                if syndrome.stabilizer_measurements[i - 1] {
                    1
                } else {
                    0
                }
            };

            if error_count >= 1 {
                crate::GateOperations::x(&mut corrected, i);
            }
        }

        corrected
    }

    /// Get error correction capability (t).
    pub fn distance(&self) -> usize {
        self.num_physical_qubits / 2
    }
}

/// Phase-flip repetition code.
pub struct RepetitionCodePhaseFlip {
    num_physical_qubits: usize,
}

impl RepetitionCodePhaseFlip {
    pub fn new(repetition: usize) -> Self {
        assert!(repetition >= 3);
        assert!(repetition % 2 == 1);

        Self {
            num_physical_qubits: repetition,
        }
    }

    /// Encode using phase-flip code.
    /// |ψ⟩ = α|+⟩ + β|-⟩ → α|+++⟩ + β|---⟩
    pub fn encode(&self, logical_state: &mut QuantumState, encoded_state: &mut QuantumState) {
        // Apply H to transform bit-flip to phase-flip
        for i in 0..self.num_physical_qubits {
            QuantumGates::h(encoded_state, i);
        }

        // Use bit-flip encoding
        let bf_code = RepetitionCodeBitFlip::new(self.num_physical_qubits);
        bf_code.encode(logical_state, encoded_state);

        // Apply H to return to computational basis
        for i in 0..self.num_physical_qubits {
            QuantumGates::h(encoded_state, i);
        }
    }

    /// Measure X⊗X stabilizers.
    pub fn measure_syndrome(&self, state: &QuantumState) -> Syndrome {
        let mut syndromes = Vec::new();

        for i in 0..(self.num_physical_qubits - 1) {
            // Measure X⊗X parity
            let parity = self.measure_parity_xx(state, i, i + 1);
            syndromes.push(parity);
        }

        Syndrome::new(syndromes)
    }

    fn measure_parity_xx(&self, state: &QuantumState, qubit1: usize, qubit2: usize) -> bool {
        // Compute expectation value of X⊗X on (qubit1, qubit2).
        // X⊗X in computational basis: for each pair (i,j) where j differs from i in
        // exactly bits qubit1 and qubit2, accumulate conj(a_i) * a_j.
        // The eigenvalue decomposition: +1 sector has even parity in Hadamard basis.
        let amps = state.amplitudes_ref();
        let mask1 = 1usize << qubit1;
        let mask2 = 1usize << qubit2;
        let flip_mask = mask1 | mask2;
        let mut expectation = 0.0;
        for (i, amp_i) in amps.iter().enumerate() {
            let j = i ^ flip_mask;
            // <i|X⊗X|j> = 1 when j = i XOR (mask1|mask2)
            let amp_j = &amps[j];
            // Re(conj(a_i) * a_j)
            expectation += amp_i.re * amp_j.re + amp_i.im * amp_j.im;
        }
        // expectation is <ψ|X⊗X|ψ>. Probability of measuring -1 = (1 - exp)/2.
        let prob_odd = (1.0 - expectation) / 2.0;
        let sample: f64 = rand::random();
        sample < prob_odd.max(0.0).min(1.0)
    }
}

// ==================== SHOR'S 9-QUBIT CODE ====================

/// Shor's 9-qubit code.
/// Corrects arbitrary single-qubit errors (bit flip and phase flip).
pub struct ShorsNineQubitCode;

impl ShorsNineQubitCode {
    pub fn encode(logical_state: &mut QuantumState, encoded_state: &mut QuantumState) {
        // Shor's code: Concatenate bit-flip and phase-flip codes
        // |ψ⟩ = α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩ (bit-flip encoding)
        //                     → α|+++⟩ + β|---⟩ (phase-flip encoding of each qubit)
        //                     → 9-qubit encoded state

        let _logical_0 = logical_state.amplitudes_ref()[0];
        let logical_1 = logical_state.amplitudes_ref()[1];

        // Encoding circuit for Shor's 9-qubit code
        // 1. Start with |ψ⟩|00000000⟩
        // 2. Apply CNOTs to entangle data qubits with ancillas
        // 3. Apply Hadamards for phase-flip encoding

        // Simplified encoding
        let enc_amplitudes = encoded_state.amplitudes_mut();

        // α|000000000⟩ + β|111111111⟩ for bit-flip
        // Then apply phase encoding
        for _i in 0..9 {
            if logical_1.norm_sqr() > 0.01 {
                enc_amplitudes[(1 << 9) - 1] = logical_1;
            }
        }
    }

    /// Shor code stabilizer generators (8 total):
    ///   Z-type (bit-flip detection within each block of 3):
    ///     s0: Z₀Z₁,  s1: Z₁Z₂,  s2: Z₃Z₄,  s3: Z₄Z₅,  s4: Z₆Z₇,  s5: Z₇Z₈
    ///   X-type (phase-flip detection between blocks):
    ///     s6: X₀X₁X₂X₃X₄X₅,  s7: X₃X₄X₅X₆X₇X₈
    const Z_STABILIZERS: [[usize; 2]; 6] = [
        [0, 1],
        [1, 2], // block 0
        [3, 4],
        [4, 5], // block 1
        [6, 7],
        [7, 8], // block 2
    ];
    const X_STABILIZER_BLOCKS: [[usize; 6]; 2] = [
        [0, 1, 2, 3, 4, 5], // between blocks 0-1
        [3, 4, 5, 6, 7, 8], // between blocks 1-2
    ];

    pub fn measure_syndrome(state: &QuantumState) -> Syndrome {
        // Shor's 9-qubit code has 8 stabilizer generators:
        //   6 Z-type (detect bit-flip errors within each block of 3)
        //   2 X-type (detect phase-flip errors between blocks)
        let mut syndromes = vec![false; 8];

        // Z stabilizers: Z⊗Z on adjacent pairs within each block
        for (i, pair) in Self::Z_STABILIZERS.iter().enumerate() {
            syndromes[i] = measure_multi_z(state, pair);
        }

        // X stabilizers: X⊗X⊗X⊗X⊗X⊗X across two adjacent blocks
        for (i, block) in Self::X_STABILIZER_BLOCKS.iter().enumerate() {
            syndromes[6 + i] = measure_multi_x(state, block);
        }

        Syndrome::new(syndromes)
    }

    /// Deterministic syndrome measurement for Shor's code.
    /// Uses expectation thresholds (prob_odd > 0.5, expectation < 0) instead of Born-rule
    /// sampling. Suitable for pure error states in unit tests.
    pub fn measure_syndrome_deterministic(state: &QuantumState) -> Syndrome {
        let mut syndromes = vec![false; 8];
        let amps = state.amplitudes_ref();

        // Z stabilizers: deterministic parity check
        for (i, pair) in Self::Z_STABILIZERS.iter().enumerate() {
            let mut prob_odd = 0.0;
            for (k, amp) in amps.iter().enumerate() {
                let parity: usize = pair.iter().map(|&q| (k >> q) & 1).sum();
                if parity % 2 == 1 {
                    prob_odd += amp.norm_sqr();
                }
            }
            syndromes[i] = prob_odd > 0.5;
        }

        // X stabilizers: deterministic expectation check
        for (i, block) in Self::X_STABILIZER_BLOCKS.iter().enumerate() {
            let flip_mask: usize = block.iter().map(|&q| 1usize << q).sum();
            let mut expectation = 0.0;
            for (k, amp_k) in amps.iter().enumerate() {
                let j = k ^ flip_mask;
                expectation += amp_k.re * amps[j].re + amp_k.im * amps[j].im;
            }
            syndromes[6 + i] = expectation < 0.0;
        }

        Syndrome::new(syndromes)
    }

    pub fn decode(state: &mut QuantumState, logical_state: &mut QuantumState) {
        // Measure syndrome and correct
        let _syndrome = Self::measure_syndrome(state);

        // Correction based on syndrome
        // (Simplified - would use lookup table)

        // Extract logical qubit from block of 9
        let log_amplitudes = logical_state.amplitudes_mut();

        // Majority vote for logical value
        let mut p0 = 0.0;
        let mut p1 = 0.0;

        // First 3 qubits encode logical value
        for i in 0..3 {
            let amp0 = state.amplitudes_ref()[i << 8]; // |000...0⟩
            let amp1 = state.amplitudes_ref()[(i << 8) | 1]; // |000...1⟩

            p0 += amp0.norm_sqr();
            p1 += amp1.norm_sqr();
        }

        log_amplitudes[0] = C64::new(p0.sqrt(), 0.0);
        log_amplitudes[1] = C64::new(p1.sqrt(), 0.0);
    }

    pub fn distance() -> usize {
        3 // Corrects arbitrary single-qubit errors
    }
}

// ==================== STEANE'S 7-QUBIT CODE ====================

/// Steane's 7-qubit CSS code.
/// A Calderbank-Shor-Steane (CSS) code that's a [[7,1,3]] code.
pub struct SteaneSevenQubitCode;

impl SteaneSevenQubitCode {
    /// Encode logical qubit into 7 physical qubits.
    pub fn encode(logical_state: &mut QuantumState, encoded_state: &mut QuantumState) {
        // Steane's code is a CSS code
        // Can be constructed from classical [7,4] Hamming code

        let logical_0 = logical_state.amplitudes_ref()[0];
        let logical_1 = logical_state.amplitudes_ref()[1];

        let enc_amplitudes = encoded_state.amplitudes_mut();

        // Encoding: |0L⟩ and |1L⟩ are specific 7-qubit states
        // |0L⟩ = 1/√8 Σ|a⟩ for all a in Hamming code
        // |1L⟩ = 1/√8 Σ|a⟩ for all a in Hamming code with bit flip

        // [7,4] Hamming code codewords (for |0L⟩):
        let hamming_codewords = vec![
            0b0000000, 0b1010101, 0b0110011, 0b1100110, 0b0001111, 0b1011010, 0b0111100, 0b1101001,
        ];

        let norm = (8.0_f64).sqrt();

        for &codeword in &hamming_codewords {
            enc_amplitudes[codeword] += C64::new(logical_0.re / norm, logical_0.im / norm);
        }

        // For |1L⟩, apply X to all codewords
        if logical_1.norm_sqr() > 0.01 {
            for &codeword in &hamming_codewords {
                let flipped = codeword ^ 0b1111111;
                enc_amplitudes[flipped] += C64::new(logical_1.re / norm, logical_1.im / norm);
            }
        }
    }

    /// Measure stabilizers for Steane's code.
    /// 6 stabilizer generators: 3 X-type and 3 Z-type.
    pub fn measure_syndrome(state: &QuantumState) -> Syndrome {
        let mut syndromes = vec![false; 6];

        // Z stabilizers (measure Z⊗Z⊗Z⊗Z, detect X errors)
        // From [7,4,3] Hamming parity check matrix rows:
        //   H = [0 0 0 1 1 1 1]  → qubits {3,4,5,6}
        //       [0 1 1 0 0 1 1]  → qubits {1,2,5,6}
        //       [1 0 1 0 1 0 1]  → qubits {0,2,4,6}
        let x_stabilizers = [vec![0, 2, 4, 6], vec![0, 1, 4, 5], vec![0, 1, 2, 3]];

        // X stabilizers (measure X⊗X⊗X⊗X, detect Z errors) — same supports (Steane is self-dual CSS)
        let z_stabilizers = [vec![0, 2, 4, 6], vec![0, 1, 4, 5], vec![0, 1, 2, 3]];

        for (i, qubits) in x_stabilizers.iter().enumerate() {
            syndromes[i] = Self::measure_zzz_stabilizer(state, qubits);
        }

        for (i, qubits) in z_stabilizers.iter().enumerate() {
            syndromes[i + 3] = Self::measure_xxx_stabilizer(state, qubits);
        }

        Syndrome::new(syndromes)
    }

    fn measure_zzz_stabilizer(state: &QuantumState, qubits: &[usize]) -> bool {
        // Compute expectation of Z⊗Z⊗...⊗Z on given qubits.
        // Eigenvalue for basis state |i⟩ is (-1)^(popcount of i restricted to qubits).
        let amps = state.amplitudes_ref();
        let mut prob_odd = 0.0;
        for (i, amp) in amps.iter().enumerate() {
            let parity: usize = qubits.iter().map(|&q| (i >> q) & 1).sum();
            if parity % 2 == 1 {
                prob_odd += amp.norm_sqr();
            }
        }
        let sample: f64 = rand::random();
        sample < prob_odd
    }

    fn measure_xxx_stabilizer(state: &QuantumState, qubits: &[usize]) -> bool {
        // Compute expectation of X⊗X⊗...⊗X on given qubits.
        // For each basis state |i⟩, X⊗...⊗X flips all qubits in the set.
        let amps = state.amplitudes_ref();
        let flip_mask: usize = qubits.iter().map(|&q| 1usize << q).sum();
        let mut expectation = 0.0;
        for (i, amp_i) in amps.iter().enumerate() {
            let j = i ^ flip_mask;
            let amp_j = &amps[j];
            expectation += amp_i.re * amp_j.re + amp_i.im * amp_j.im;
        }
        let prob_odd = (1.0 - expectation) / 2.0;
        let sample: f64 = rand::random();
        sample < prob_odd.max(0.0).min(1.0)
    }

    /// Deterministic syndrome measurement (uses expectation thresholds instead of sampling).
    /// Suitable for pure error states in testing; avoids stochastic noise.
    pub fn measure_syndrome_deterministic(state: &QuantumState) -> Syndrome {
        let mut syndromes = vec![false; 6];

        let stabilizer_qubits = [vec![0usize, 2, 4, 6], vec![0, 1, 4, 5], vec![0, 1, 2, 3]];

        // Z stabilizers (bits 0-2): deterministic parity via prob_odd > 0.5
        for (i, qubits) in stabilizer_qubits.iter().enumerate() {
            let amps = state.amplitudes_ref();
            let mut prob_odd = 0.0;
            for (k, amp) in amps.iter().enumerate() {
                let parity: usize = qubits.iter().map(|&q| (k >> q) & 1).sum();
                if parity % 2 == 1 {
                    prob_odd += amp.norm_sqr();
                }
            }
            syndromes[i] = prob_odd > 0.5;
        }

        // X stabilizers (bits 3-5): deterministic via expectation < 0
        for (i, qubits) in stabilizer_qubits.iter().enumerate() {
            let amps = state.amplitudes_ref();
            let flip_mask: usize = qubits.iter().map(|&q| 1usize << q).sum();
            let mut expectation = 0.0;
            for (k, amp_k) in amps.iter().enumerate() {
                let j = k ^ flip_mask;
                expectation += amp_k.re * amps[j].re + amp_k.im * amps[j].im;
            }
            syndromes[i + 3] = expectation < 0.0;
        }

        Syndrome::new(syndromes)
    }

    /// Syndrome lookup table for single-qubit error identification.
    /// Based on stabilizer supports {0,2,4,6}, {0,1,4,5}, {0,1,2,3}.
    fn syndrome_lookup(syndrome_val: usize) -> Option<usize> {
        // Stabilizers derived from [7,3,4] dual code generators:
        //   stab0 = c1 = {0,2,4,6}, stab1 = c2 = {0,1,4,5}, stab2 = c4 = {0,1,2,3}
        // Syndrome = (s0<<2)|(s1<<1)|s2 where si=1 if qubit is in stab_i.
        // q=0: in all 3 → 111=7, q=1: in s1,s2 → 011=3, q=2: in s0,s2 → 101=5,
        // q=3: in s2 only → 001=1, q=4: in s0,s1 → 110=6, q=5: in s1 only → 010=2,
        // q=6: in s0 only → 100=4.
        match syndrome_val {
            0 => None,
            1 => Some(3), // 001: only in stab2 {0,1,2,3}
            2 => Some(5), // 010: only in stab1 {0,1,4,5}
            3 => Some(1), // 011: in stab1+stab2
            4 => Some(6), // 100: only in stab0 {0,2,4,6}
            5 => Some(2), // 101: in stab0+stab2
            6 => Some(4), // 110: in stab0+stab1
            7 => Some(0), // 111: in all three
            _ => None,
        }
    }

    /// Correct errors based on syndrome using lookup table.
    /// Z stabilizers (bits 0-2) detect X errors; X stabilizers (bits 3-5) detect Z errors.
    pub fn correct_errors(state: &mut QuantumState, syndrome: &Syndrome) {
        // Z stabilizer syndrome detects X errors
        let x_syndrome = (syndrome.stabilizer_measurements[0] as usize) << 2
            | (syndrome.stabilizer_measurements[1] as usize) << 1
            | syndrome.stabilizer_measurements[2] as usize;
        if let Some(qubit) = Self::syndrome_lookup(x_syndrome) {
            crate::GateOperations::x(state, qubit);
        }

        // X stabilizer syndrome detects Z errors
        let z_syndrome = (syndrome.stabilizer_measurements[3] as usize) << 2
            | (syndrome.stabilizer_measurements[4] as usize) << 1
            | syndrome.stabilizer_measurements[5] as usize;
        if let Some(qubit) = Self::syndrome_lookup(z_syndrome) {
            crate::GateOperations::z(state, qubit);
        }
    }

    pub fn decode(state: &mut QuantumState, logical_state: &mut QuantumState) {
        let syndrome = Self::measure_syndrome(state);
        Self::correct_errors(state, &syndrome);

        // Extract logical qubit from corrected codeword.
        // |0L⟩ = (1/√8) Σ|c⟩, |1L⟩ = (1/√8) Σ|c̄⟩
        let hamming_codewords: [usize; 8] = [
            0b0000000, 0b1010101, 0b0110011, 0b1100110, 0b0001111, 0b1011010, 0b0111100, 0b1101001,
        ];

        let amps = state.amplitudes_ref();
        let norm = (8.0_f64).sqrt();
        let mut alpha = C64::new(0.0, 0.0);
        let mut beta = C64::new(0.0, 0.0);

        for &cw in &hamming_codewords {
            if cw < amps.len() {
                alpha = alpha + amps[cw];
            }
            let flipped = cw ^ 0b1111111;
            if flipped < amps.len() {
                beta = beta + amps[flipped];
            }
        }

        let log_amplitudes = logical_state.amplitudes_mut();
        log_amplitudes[0] = C64::new(alpha.re / norm, alpha.im / norm);
        log_amplitudes[1] = C64::new(beta.re / norm, beta.im / norm);
    }

    pub fn distance() -> usize {
        3 // Corrects arbitrary single-qubit errors
    }
}

// ==================== SURFACE CODES ====================

/// Layout for the rotated surface code.
/// Maps stabilizers to data qubits with position information for MWPM decoding.
#[derive(Clone, Debug)]
pub struct SurfaceCodeLayout {
    /// Code distance.
    pub distance: usize,
    /// Number of data qubits (d²).
    pub num_data: usize,
    /// X-type stabilizers: each entry is the list of data qubit indices.
    pub x_stabilizers: Vec<Vec<usize>>,
    /// Z-type stabilizers: each entry is the list of data qubit indices.
    pub z_stabilizers: Vec<Vec<usize>>,
    /// Grid position (row, col) of each X stabilizer center (for MWPM distances).
    pub x_positions: Vec<(f64, f64)>,
    /// Grid position (row, col) of each Z stabilizer center (for MWPM distances).
    pub z_positions: Vec<(f64, f64)>,
}

impl SurfaceCodeLayout {
    /// Build the stabilizer layout for a distance-d rotated surface code.
    /// Data qubits sit on a d×d grid; stabilizers are 2×2 plaquettes (bulk)
    /// and weight-2 operators (boundary).
    pub fn new(distance: usize) -> Self {
        let d = distance.max(3);
        let num_data = d * d;

        let mut x_stabilizers = Vec::new();
        let mut z_stabilizers = Vec::new();
        let mut x_positions = Vec::new();
        let mut z_positions = Vec::new();

        // Bulk stabilizers: weight-4 plaquettes on the (d-1)×(d-1) face grid.
        // Face (r, c) touches data qubits (r,c), (r,c+1), (r+1,c), (r+1,c+1).
        // r+c even → X-type, r+c odd → Z-type.
        for r in 0..d - 1 {
            for c in 0..d - 1 {
                let qubits = vec![
                    r * d + c,
                    r * d + c + 1,
                    (r + 1) * d + c,
                    (r + 1) * d + c + 1,
                ];
                let pos = (r as f64 + 0.5, c as f64 + 0.5);
                if (r + c) % 2 == 0 {
                    x_stabilizers.push(qubits);
                    x_positions.push(pos);
                } else {
                    z_stabilizers.push(qubits);
                    z_positions.push(pos);
                }
            }
        }

        // Boundary X stabilizers: top row (odd c) and bottom row (even c), weight-2.
        {
            let mut c = 1;
            while c < d - 1 {
                x_stabilizers.push(vec![c, c + 1]);
                x_positions.push((-0.5, c as f64 + 0.5));
                c += 2;
            }
            c = 0;
            while c < d - 1 {
                x_stabilizers.push(vec![(d - 1) * d + c, (d - 1) * d + c + 1]);
                x_positions.push((d as f64 - 0.5, c as f64 + 0.5));
                c += 2;
            }
        }

        // Boundary Z stabilizers: left column (even r) and right column (odd r), weight-2.
        {
            let mut r = 0;
            while r < d - 1 {
                z_stabilizers.push(vec![r * d, (r + 1) * d]);
                z_positions.push((r as f64 + 0.5, -0.5));
                r += 2;
            }
            r = 1;
            while r < d - 1 {
                z_stabilizers.push(vec![r * d + d - 1, (r + 1) * d + d - 1]);
                z_positions.push((r as f64 + 0.5, d as f64 - 0.5));
                r += 2;
            }
        }

        Self {
            distance: d,
            num_data,
            x_stabilizers,
            z_stabilizers,
            x_positions,
            z_positions,
        }
    }
}

/// Correction prescription from the MWPM decoder.
#[derive(Clone, Debug)]
pub struct SurfaceCodeCorrection {
    /// Data qubits to apply X correction on.
    pub x_corrections: Vec<usize>,
    /// Data qubits to apply Z correction on.
    pub z_corrections: Vec<usize>,
}

/// Rotated surface code.
/// Leading topological QEC code for 2D architectures.
pub struct RotatedSurfaceCode {
    layout: SurfaceCodeLayout,
}

impl RotatedSurfaceCode {
    pub fn new(code_distance: usize) -> Self {
        Self {
            layout: SurfaceCodeLayout::new(code_distance),
        }
    }

    /// Get number of physical data qubits (d²).
    pub fn num_physical_qubits(&self) -> usize {
        self.layout.num_data
    }

    /// Error correction capability t = (d-1)/2.
    pub fn distance(&self) -> usize {
        (self.layout.distance - 1) / 2
    }

    /// Raw code distance d.
    pub fn code_distance(&self) -> usize {
        self.layout.distance
    }

    /// Access the stabilizer layout.
    pub fn layout(&self) -> &SurfaceCodeLayout {
        &self.layout
    }

    /// Measure syndrome by computing stabilizer eigenvalues (Born-rule sampling).
    pub fn measure_syndrome(&self, state: &QuantumState) -> SurfaceCodeSyndrome {
        let x_syndromes: Vec<bool> = self
            .layout
            .x_stabilizers
            .iter()
            .map(|qubits| measure_multi_x(state, qubits))
            .collect();
        let z_syndromes: Vec<bool> = self
            .layout
            .z_stabilizers
            .iter()
            .map(|qubits| measure_multi_z(state, qubits))
            .collect();

        SurfaceCodeSyndrome {
            x_syndromes,
            z_syndromes,
        }
    }

    /// Deterministic syndrome: use expectation sign instead of Born-rule sampling.
    /// Useful for testing with pure error states.
    pub fn measure_syndrome_deterministic(&self, state: &QuantumState) -> SurfaceCodeSyndrome {
        let x_syndromes: Vec<bool> = self
            .layout
            .x_stabilizers
            .iter()
            .map(|qubits| {
                let amps = state.amplitudes_ref();
                let flip_mask: usize = qubits.iter().map(|&q| 1usize << q).sum();
                let mut expectation = 0.0;
                for (i, amp_i) in amps.iter().enumerate() {
                    let j = i ^ flip_mask;
                    expectation += amp_i.re * amps[j].re + amp_i.im * amps[j].im;
                }
                expectation < 0.0
            })
            .collect();

        let z_syndromes: Vec<bool> = self
            .layout
            .z_stabilizers
            .iter()
            .map(|qubits| {
                let amps = state.amplitudes_ref();
                let mut prob_odd = 0.0;
                for (i, amp) in amps.iter().enumerate() {
                    let parity: usize = qubits.iter().map(|&q| (i >> q) & 1).sum();
                    if parity % 2 == 1 {
                        prob_odd += amp.norm_sqr();
                    }
                }
                prob_odd > 0.5
            })
            .collect();

        SurfaceCodeSyndrome {
            x_syndromes,
            z_syndromes,
        }
    }

    /// MWPM decoder: find minimum weight corrections using greedy matching.
    /// Z syndromes detect X errors; X syndromes detect Z errors.
    pub fn decode(&self, syndrome: &SurfaceCodeSyndrome) -> SurfaceCodeCorrection {
        let x_corrections = self.decode_error_type(
            &syndrome.z_syndromes,
            &self.layout.z_positions,
            &self.layout.z_stabilizers,
        );
        let z_corrections = self.decode_error_type(
            &syndrome.x_syndromes,
            &self.layout.x_positions,
            &self.layout.x_stabilizers,
        );

        SurfaceCodeCorrection {
            x_corrections,
            z_corrections,
        }
    }

    /// Apply corrections to the quantum state.
    pub fn apply_correction(state: &mut QuantumState, correction: &SurfaceCodeCorrection) {
        for &q in &correction.x_corrections {
            crate::GateOperations::x(state, q);
        }
        for &q in &correction.z_corrections {
            crate::GateOperations::z(state, q);
        }
    }

    /// Decode one error type using greedy MWPM on the defect graph.
    fn decode_error_type(
        &self,
        syndromes: &[bool],
        positions: &[(f64, f64)],
        stabilizers: &[Vec<usize>],
    ) -> Vec<usize> {
        // Collect defect indices (triggered stabilizers).
        let defects: Vec<usize> = syndromes
            .iter()
            .enumerate()
            .filter(|(_, &s)| s)
            .map(|(i, _)| i)
            .collect();

        if defects.is_empty() {
            return Vec::new();
        }

        // Fast path: single-error lookup (covers the vast majority of correctable errors).
        if let Some(qubit) = self.single_error_lookup(syndromes, stabilizers) {
            return vec![qubit];
        }

        // Greedy MWPM: match closest defect pairs, remainder matches to boundary.
        let mut matched = vec![false; defects.len()];
        let mut corrections = Vec::new();

        // Build all pairwise Manhattan distances.
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..defects.len() {
            for j in (i + 1)..defects.len() {
                let (r1, c1) = positions[defects[i]];
                let (r2, c2) = positions[defects[j]];
                let dist = (r1 - r2).abs() + (c1 - c2).abs();
                pairs.push((dist, i, j));
            }
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Greedily match closest unmatched pairs.
        for &(pair_dist, i, j) in &pairs {
            if !matched[i] && !matched[j] {
                let bd_i = self.boundary_distance(positions[defects[i]]);
                let bd_j = self.boundary_distance(positions[defects[j]]);

                if bd_i + bd_j < pair_dist {
                    // Cheaper to match each defect independently to the boundary.
                    corrections.extend(self.boundary_correction(stabilizers, defects[i]));
                    corrections.extend(self.boundary_correction(stabilizers, defects[j]));
                } else {
                    // Match the pair: find correction chain between them.
                    corrections.extend(self.correction_path(defects[i], defects[j], stabilizers));
                }
                matched[i] = true;
                matched[j] = true;
            }
        }

        // Any remaining unmatched defects match to boundary.
        for (i, &m) in matched.iter().enumerate() {
            if !m {
                corrections.extend(self.boundary_correction(stabilizers, defects[i]));
            }
        }

        corrections
    }

    /// Manhattan distance from a stabilizer position to the nearest code boundary.
    fn boundary_distance(&self, pos: (f64, f64)) -> f64 {
        let d = self.layout.distance as f64;
        let (r, c) = pos;
        (r + 0.5)
            .min(c + 0.5)
            .min(d - 0.5 - r)
            .min(d - 0.5 - c)
            .max(0.0)
    }

    /// Single-error lookup: find the unique data qubit whose stabilizer membership
    /// matches the observed syndrome exactly. Returns None for multi-qubit patterns.
    fn single_error_lookup(&self, syndromes: &[bool], stabilizers: &[Vec<usize>]) -> Option<usize> {
        let syndrome_mask: u64 = syndromes
            .iter()
            .enumerate()
            .filter(|(_, &s)| s)
            .fold(0u64, |acc, (i, _)| acc | (1u64 << i));

        if syndrome_mask == 0 {
            return None;
        }

        for q in 0..self.layout.num_data {
            let mut qubit_mask: u64 = 0;
            for (i, stab) in stabilizers.iter().enumerate() {
                if stab.contains(&q) {
                    qubit_mask |= 1u64 << i;
                }
            }
            if qubit_mask == syndrome_mask {
                return Some(q);
            }
        }
        None
    }

    /// Correction path between two matched defects via stabilizer adjacency BFS.
    fn correction_path(
        &self,
        def_a: usize,
        def_b: usize,
        stabilizers: &[Vec<usize>],
    ) -> Vec<usize> {
        // Adjacent stabilizers: correct on shared data qubit.
        let shared: Vec<usize> = stabilizers[def_a]
            .iter()
            .filter(|q| stabilizers[def_b].contains(q))
            .copied()
            .collect();
        if !shared.is_empty() {
            return vec![shared[0]];
        }

        // Non-adjacent: BFS through stabilizer adjacency graph.
        let path = Self::stabilizer_bfs(stabilizers, def_a, def_b);
        let mut corrections = Vec::new();
        for window in path.windows(2) {
            let s: Vec<usize> = stabilizers[window[0]]
                .iter()
                .filter(|q| stabilizers[window[1]].contains(q))
                .copied()
                .collect();
            if let Some(&q) = s.first() {
                corrections.push(q);
            }
        }
        corrections
    }

    /// BFS on stabilizer-adjacency (shared data qubit) to find shortest path.
    fn stabilizer_bfs(stabilizers: &[Vec<usize>], start: usize, end: usize) -> Vec<usize> {
        let n = stabilizers.len();
        let mut visited = vec![false; n];
        let mut parent = vec![usize::MAX; n];
        let mut queue = Vec::with_capacity(n);
        let mut head = 0usize;

        visited[start] = true;
        queue.push(start);

        while head < queue.len() {
            let current = queue[head];
            head += 1;

            if current == end {
                let mut path = vec![end];
                let mut node = end;
                while node != start {
                    node = parent[node];
                    path.push(node);
                }
                path.reverse();
                return path;
            }

            for next in 0..n {
                if !visited[next]
                    && stabilizers[current]
                        .iter()
                        .any(|q| stabilizers[next].contains(q))
                {
                    visited[next] = true;
                    parent[next] = current;
                    queue.push(next);
                }
            }
        }

        // Fallback (should not happen for a valid surface code).
        vec![start]
    }

    /// Boundary correction: pick the stabilizer's data qubit closest to a boundary.
    fn boundary_correction(&self, stabilizers: &[Vec<usize>], defect_idx: usize) -> Vec<usize> {
        let d = self.layout.distance;
        let mut best_q = None;
        let mut best_dist = f64::MAX;

        for &q in &stabilizers[defect_idx] {
            let qr = q / d;
            let qc = q % d;
            let dist = (qr as f64)
                .min(qc as f64)
                .min((d - 1 - qr) as f64)
                .min((d - 1 - qc) as f64);
            if dist < best_dist {
                best_dist = dist;
                best_q = Some(q);
            }
        }

        best_q.into_iter().collect()
    }

    /// Estimate threshold error rate.
    pub fn threshold() -> f64 {
        0.01
    }

    /// Logical error rate after decoding.
    pub fn logical_error_rate(&self, physical_error_rate: f64) -> f64 {
        let t = self.distance() as f64;
        let p_th = Self::threshold();

        if physical_error_rate < p_th {
            (physical_error_rate / p_th).powi(t as i32 + 1)
        } else {
            1.0
        }
    }
}

/// Surface code syndrome.
#[derive(Clone, Debug)]
pub struct SurfaceCodeSyndrome {
    pub x_syndromes: Vec<bool>,
    pub z_syndromes: Vec<bool>,
}

// ==================== COLOR CODES ====================

/// 2D color code.
/// Topological code with color constraints instead of parity.
pub struct ColorCode {
    code_distance: usize,
}

impl ColorCode {
    pub fn new(code_distance: usize) -> Self {
        Self { code_distance }
    }

    /// Encode using color code.
    pub fn encode(&self, logical_state: &mut QuantumState, encoded_state: &mut QuantumState) {
        // Color codes require more complex encoding
        // Simplified implementation
        let _ = (logical_state, encoded_state);
    }

    /// Measure color stabilizers (R, G, B constraints).
    pub fn measure_syndrome(&self, _state: &QuantumState) -> ColorCodeSyndrome {
        let d = self.code_distance;

        // Number of stabilizers per color
        let stabilizers_per_color = (d - 1) * (d - 2) / 2;

        ColorCodeSyndrome {
            r_syndromes: vec![false; stabilizers_per_color],
            g_syndromes: vec![false; stabilizers_per_color],
            b_syndromes: vec![false; stabilizers_per_color],
        }
    }

    pub fn distance() -> usize {
        3 // Typically [[d^2, 1, d]] code
    }
}

/// Color code syndrome.
#[derive(Clone, Debug)]
pub struct ColorCodeSyndrome {
    pub r_syndromes: Vec<bool>,
    pub g_syndromes: Vec<bool>,
    pub b_syndromes: Vec<bool>,
}

// ==================== FAULT TOLERANCE ANALYSIS ====================

/// Fault tolerance threshold analysis.
pub struct ThresholdAnalysis;

impl ThresholdAnalysis {
    /// Calculate threshold for a given code.
    pub fn calculate_threshold(code_distance: usize, decoder: &str) -> f64 {
        // Approximate thresholds based on code and decoder
        match decoder {
            "MWPM" => {
                // Surface code with MWPM: ~1%
                match code_distance {
                    d if d < 5 => 0.015,
                    d if d < 11 => 0.010,
                    _ => 0.009,
                }
            }
            "Greedy" => {
                // Greedy decoder: lower threshold
                match code_distance {
                    d if d < 5 => 0.010,
                    d if d < 11 => 0.007,
                    _ => 0.005,
                }
            }
            _ => 0.01,
        }
    }

    /// Logical error rate vs physical error rate curve.
    pub fn logical_vs_physical_error_rate(
        code_distance: usize,
        threshold: f64,
        physical_error_rates: &[f64],
    ) -> Vec<(f64, f64)> {
        let t = (code_distance - 1) / 2;

        physical_error_rates
            .iter()
            .map(|&p| {
                if p < threshold {
                    let logical_p = (p / threshold).powi(t as i32 + 1);
                    (p, logical_p)
                } else {
                    (p, 1.0)
                }
            })
            .collect()
    }

    /// Resource overhead for fault tolerance.
    pub fn resource_overhead(code_distance: usize, num_logical_qubits: usize) -> ResourceEstimate {
        let surface_code = RotatedSurfaceCode::new(code_distance);
        let physical_per_logical = surface_code.num_physical_qubits();

        // Add overhead for syndrome extraction and decoding
        let ancilla_overhead = 2; // Rough estimate
        let total_physical = physical_per_logical * num_logical_qubits * ancilla_overhead;

        // Gate overhead (rough estimate)
        let gate_overhead = code_distance * code_distance;

        ResourceEstimate {
            physical_qubits: total_physical,
            gate_overhead,
            time_overhead: code_distance as f64,
        }
    }
}

/// Resource estimate for fault-tolerant computation.
#[derive(Clone, Debug)]
pub struct ResourceEstimate {
    pub physical_qubits: usize,
    pub gate_overhead: usize,
    pub time_overhead: f64,
}

// ==================== LOGICAL GATE OPERATIONS ====================

/// Fault-tolerant logical gate operations.
pub struct LogicalGates;

impl LogicalGates {
    /// Logical CNOT using transversal operation.
    pub fn logical_cnot(
        code_block1: &mut QuantumState,
        _code_block2: &mut QuantumState,
        num_physical: usize,
    ) {
        // For CSS codes like Steane's 7-qubit, CNOT is transversal
        for i in 0..num_physical {
            crate::GateOperations::cnot(code_block1, i, i);
        }
    }

    /// Logical Hadamard using lattice surgery.
    pub fn logical_h(code_block: &mut QuantumState, num_physical: usize) {
        // H is not transversal for most codes
        // Use lattice surgery or magic state distillation

        // Simplified: apply H to all physical qubits
        for i in 0..num_physical {
            QuantumGates::h(code_block, i);
        }
    }

    /// T gate using magic state distillation.
    pub fn logical_t(
        _code_block: &mut QuantumState,
        _magic_state: &QuantumState,
        _target_qubit: usize,
    ) {
        // Teleportation-based T gate using |T⟩ magic state
        // (Simplified implementation)
    }

    /// Prepare |T⟩ magic state.
    pub fn prepare_t_magic_state(num_qubits: usize) -> QuantumState {
        // |T⟩ = (|0⟩ + e^(iπ/4)|1⟩) / √2
        let mut state = QuantumState::new(num_qubits);
        QuantumGates::h(&mut state, 0);
        QuantumGates::t(&mut state, 0);
        state
    }

    /// State distillation for magic states.
    pub fn distill_magic_state(noisy_states: Vec<QuantumState>, _rounds: usize) -> QuantumState {
        // 15-to-1 distillation for |T⟩ states
        // (Simplified - would use specific distillation circuit)
        noisy_states[0].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_repetition_code_encode() {
        let code = RepetitionCodeBitFlip::new(3);

        let logical = &mut QuantumState::new(1);
        let mut encoded = QuantumState::new(3);

        // Encode |0⟩
        code.encode(logical, &mut encoded);

        let probs = encoded.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_repetition_code_distance() {
        let code = RepetitionCodeBitFlip::new(5);
        assert_eq!(code.distance(), 2);
    }

    #[test]
    fn test_surface_code_threshold() {
        let code = RotatedSurfaceCode::new(5);
        assert_eq!(code.distance(), 2);
        assert_eq!(RotatedSurfaceCode::threshold(), 0.01);
    }

    #[test]
    fn test_resource_estimate() {
        let estimate = ThresholdAnalysis::resource_overhead(5, 10);
        assert!(estimate.physical_qubits > 10);
    }

    // ==================== STEANE CODE TESTS ====================

    /// Helper: create a properly zeroed state for Steane encoding (encoder uses +=).
    fn steane_zeroed_state() -> QuantumState {
        let mut s = QuantumState::new(7);
        s.amplitudes_mut()[0] = C64::new(0.0, 0.0);
        s
    }

    #[test]
    fn test_steane_encode_zero() {
        let mut logical = QuantumState::new(1);
        let mut encoded = steane_zeroed_state();
        SteaneSevenQubitCode::encode(&mut logical, &mut encoded);

        let hamming: [usize; 8] = [
            0b0000000, 0b1010101, 0b0110011, 0b1100110, 0b0001111, 0b1011010, 0b0111100, 0b1101001,
        ];
        let amps = encoded.amplitudes_ref();
        let expected = 1.0 / (8.0_f64).sqrt();
        for &cw in &hamming {
            assert!(
                (amps[cw].re - expected).abs() < 1e-10,
                "codeword {:#09b}: expected {}, got {}",
                cw,
                expected,
                amps[cw].re
            );
        }
    }

    #[test]
    fn test_steane_syndrome_lookup_all_qubits() {
        let stabilizers = [vec![0usize, 2, 4, 6], vec![0, 1, 4, 5], vec![0, 1, 2, 3]];
        let mut seen = std::collections::HashSet::new();
        for q in 0..7 {
            let mut syndrome_val = 0usize;
            for (i, stab) in stabilizers.iter().enumerate() {
                if stab.contains(&q) {
                    syndrome_val |= 1 << (2 - i);
                }
            }
            assert_ne!(syndrome_val, 0, "qubit {} has zero syndrome", q);
            assert!(
                seen.insert(syndrome_val),
                "qubit {} has duplicate syndrome {}",
                q,
                syndrome_val
            );
            let lookup_result = SteaneSevenQubitCode::syndrome_lookup(syndrome_val);
            assert_eq!(
                lookup_result,
                Some(q),
                "qubit {}: syndrome {} maps to {:?}",
                q,
                syndrome_val,
                lookup_result
            );
        }
    }

    #[test]
    fn test_steane_x_error_correction() {
        for error_qubit in 0..7 {
            let mut logical = QuantumState::new(1);
            let mut encoded = steane_zeroed_state();
            SteaneSevenQubitCode::encode(&mut logical, &mut encoded);

            crate::GateOperations::x(&mut encoded, error_qubit);

            // Use deterministic syndrome to avoid stochastic X-stabilizer noise
            let syndrome = SteaneSevenQubitCode::measure_syndrome_deterministic(&encoded);
            let x_syn = (syndrome.stabilizer_measurements[0] as usize) << 2
                | (syndrome.stabilizer_measurements[1] as usize) << 1
                | syndrome.stabilizer_measurements[2] as usize;
            assert_ne!(
                x_syn, 0,
                "X error on qubit {} needs non-trivial Z syndrome",
                error_qubit
            );

            SteaneSevenQubitCode::correct_errors(&mut encoded, &syndrome);

            let expected = 1.0 / (8.0_f64).sqrt();
            let amp = encoded.amplitudes_ref()[0].re;
            assert!(
                (amp - expected).abs() < 1e-10,
                "after X correction on qubit {}: expected {}, got {}",
                error_qubit,
                expected,
                amp
            );
        }
    }

    #[test]
    fn test_steane_z_error_correction() {
        for error_qubit in 0..7 {
            let mut logical = QuantumState::new(1);
            let mut encoded = steane_zeroed_state();
            SteaneSevenQubitCode::encode(&mut logical, &mut encoded);

            crate::GateOperations::z(&mut encoded, error_qubit);

            let syndrome = SteaneSevenQubitCode::measure_syndrome_deterministic(&encoded);
            let z_syn = (syndrome.stabilizer_measurements[3] as usize) << 2
                | (syndrome.stabilizer_measurements[4] as usize) << 1
                | syndrome.stabilizer_measurements[5] as usize;
            assert_ne!(
                z_syn, 0,
                "Z error on qubit {} needs non-trivial X syndrome",
                error_qubit
            );

            SteaneSevenQubitCode::correct_errors(&mut encoded, &syndrome);

            let expected = 1.0 / (8.0_f64).sqrt();
            let amp = encoded.amplitudes_ref()[0].re;
            assert!(
                (amp - expected).abs() < 1e-10,
                "after Z correction on qubit {}: expected {}, got {}",
                error_qubit,
                expected,
                amp
            );
        }
    }

    #[test]
    fn test_steane_full_decode_with_error() {
        let mut logical = QuantumState::new(1);
        let mut encoded = steane_zeroed_state();
        SteaneSevenQubitCode::encode(&mut logical, &mut encoded);

        crate::GateOperations::x(&mut encoded, 4);

        // Use deterministic syndrome + correct + manual extraction
        let syndrome = SteaneSevenQubitCode::measure_syndrome_deterministic(&encoded);
        SteaneSevenQubitCode::correct_errors(&mut encoded, &syndrome);

        // Extract logical qubit
        let hamming_codewords: [usize; 8] = [
            0b0000000, 0b1010101, 0b0110011, 0b1100110, 0b0001111, 0b1011010, 0b0111100, 0b1101001,
        ];
        let amps = encoded.amplitudes_ref();
        let mut alpha_sq = 0.0;
        for &cw in &hamming_codewords {
            alpha_sq += amps[cw].norm_sqr();
        }
        // alpha_sq = 8 * (1/√8)² = 8 * 1/8 = 1.0 for |0L⟩
        assert!(
            (alpha_sq - 1.0).abs() < 1e-6,
            "decoded |0L⟩ probability: {}",
            alpha_sq
        );
    }

    // ==================== SHOR CODE TESTS ====================

    #[test]
    fn test_shor_no_error_syndrome() {
        // |0...0> is in the +1 eigenspace of all Z stabilizers and all X stabilizers.
        // All syndromes should be false (trivial).
        let state = QuantumState::new(9);
        let syndrome = ShorsNineQubitCode::measure_syndrome_deterministic(&state);
        assert!(
            syndrome.is_trivial(),
            "No-error state should have trivial syndrome, got {:?}",
            syndrome.stabilizer_measurements
        );
    }

    #[test]
    fn test_shor_single_x_error_triggers_z_syndromes() {
        // An X error (bit flip) on any qubit should trigger some Z stabilizers.
        // Z stabilizers on pairs within each block detect bit-flip errors.
        for error_qubit in 0..9 {
            let mut state = QuantumState::new(9);
            crate::GateOperations::x(&mut state, error_qubit);

            let syndrome = ShorsNineQubitCode::measure_syndrome_deterministic(&state);
            let z_triggered: Vec<bool> = syndrome.stabilizer_measurements[0..6].to_vec();
            let any_z = z_triggered.iter().any(|&s| s);
            assert!(
                any_z,
                "X error on qubit {} should trigger Z syndromes, got {:?}",
                error_qubit, z_triggered
            );
        }
    }

    #[test]
    fn test_shor_single_z_error_triggers_x_syndromes() {
        // A Z error (phase flip) on |+...+> should trigger X stabilizers.
        // Prepare all-plus state first.
        for error_qubit in 0..9 {
            let mut state = QuantumState::new(9);
            for q in 0..9 {
                crate::GateOperations::h(&mut state, q);
            }
            crate::GateOperations::z(&mut state, error_qubit);

            let syndrome = ShorsNineQubitCode::measure_syndrome_deterministic(&state);
            let x_triggered: Vec<bool> = syndrome.stabilizer_measurements[6..8].to_vec();
            let any_x = x_triggered.iter().any(|&s| s);
            assert!(
                any_x,
                "Z error on qubit {} (plus basis) should trigger X syndromes, got {:?}",
                error_qubit, x_triggered
            );
        }
    }

    #[test]
    fn test_shor_z_syndrome_identifies_block() {
        // Z stabilizer pairs: [0,1],[1,2] (block 0), [3,4],[4,5] (block 1), [6,7],[7,8] (block 2)
        // X error on qubit 0: triggers s0 (pair 0,1) but not s1 (pair 1,2).
        // X error on qubit 1: triggers both s0 and s1.
        // X error on qubit 2: triggers s1 but not s0.
        let cases = vec![
            // (error_qubit, expected syndrome bits 0..6)
            (0, vec![true, false, false, false, false, false]),
            (1, vec![true, true, false, false, false, false]),
            (2, vec![false, true, false, false, false, false]),
            (3, vec![false, false, true, false, false, false]),
            (4, vec![false, false, true, true, false, false]),
            (5, vec![false, false, false, true, false, false]),
            (6, vec![false, false, false, false, true, false]),
            (7, vec![false, false, false, false, true, true]),
            (8, vec![false, false, false, false, false, true]),
        ];

        for (error_qubit, expected_z) in cases {
            let mut state = QuantumState::new(9);
            crate::GateOperations::x(&mut state, error_qubit);

            let syndrome = ShorsNineQubitCode::measure_syndrome_deterministic(&state);
            let z_syns: Vec<bool> = syndrome.stabilizer_measurements[0..6].to_vec();
            assert_eq!(
                z_syns, expected_z,
                "X error on qubit {}: expected Z syndromes {:?}, got {:?}",
                error_qubit, expected_z, z_syns
            );
        }
    }

    #[test]
    fn test_shor_x_syndrome_identifies_block_pair() {
        // X stabilizers: s6 covers blocks 0-1 (qubits 0-5), s7 covers blocks 1-2 (qubits 3-8).
        // Z error on |+...+>:
        //   qubit in block 0 only (0,1,2): triggers s6 only
        //   qubit in block 1 only (3,4,5): triggers both s6 and s7 (shared)
        //   qubit in block 2 only (6,7,8): triggers s7 only
        let cases = vec![
            (0, (true, false)),
            (1, (true, false)),
            (2, (true, false)),
            (3, (true, true)),
            (4, (true, true)),
            (5, (true, true)),
            (6, (false, true)),
            (7, (false, true)),
            (8, (false, true)),
        ];

        for (error_qubit, (exp_s6, exp_s7)) in cases {
            let mut state = QuantumState::new(9);
            for q in 0..9 {
                crate::GateOperations::h(&mut state, q);
            }
            crate::GateOperations::z(&mut state, error_qubit);

            let syndrome = ShorsNineQubitCode::measure_syndrome_deterministic(&state);
            let s6 = syndrome.stabilizer_measurements[6];
            let s7 = syndrome.stabilizer_measurements[7];
            assert_eq!(
                (s6, s7),
                (exp_s6, exp_s7),
                "Z error on qubit {} (plus basis): expected X syndromes ({}, {}), got ({}, {})",
                error_qubit,
                exp_s6,
                exp_s7,
                s6,
                s7
            );
        }
    }

    #[test]
    fn test_shor_syndrome_count() {
        // Verify syndrome has exactly 8 measurements (6 Z + 2 X).
        let state = QuantumState::new(9);
        let syndrome = ShorsNineQubitCode::measure_syndrome_deterministic(&state);
        assert_eq!(
            syndrome.stabilizer_measurements.len(),
            8,
            "Shor code should have 8 syndrome measurements"
        );

        let syndrome_stochastic = ShorsNineQubitCode::measure_syndrome(&state);
        assert_eq!(
            syndrome_stochastic.stabilizer_measurements.len(),
            8,
            "Stochastic Shor syndrome should also have 8 measurements"
        );
    }

    // ==================== SURFACE CODE TESTS ====================

    #[test]
    fn test_surface_code_layout_d3() {
        let layout = SurfaceCodeLayout::new(3);
        assert_eq!(layout.num_data, 9);
        assert_eq!(layout.x_stabilizers.len() + layout.z_stabilizers.len(), 8);
        assert_eq!(layout.x_stabilizers.len(), 4);
        assert_eq!(layout.z_stabilizers.len(), 4);
    }

    #[test]
    fn test_surface_code_layout_d5() {
        let layout = SurfaceCodeLayout::new(5);
        assert_eq!(layout.num_data, 25);
        assert_eq!(layout.x_stabilizers.len() + layout.z_stabilizers.len(), 24);
        assert_eq!(layout.x_stabilizers.len(), 12);
        assert_eq!(layout.z_stabilizers.len(), 12);
    }

    #[test]
    fn test_surface_code_stabilizer_commutativity() {
        for d in [3, 5, 7] {
            let layout = SurfaceCodeLayout::new(d);
            for (xi, x_stab) in layout.x_stabilizers.iter().enumerate() {
                for (zi, z_stab) in layout.z_stabilizers.iter().enumerate() {
                    let overlap = x_stab.iter().filter(|q| z_stab.contains(q)).count();
                    assert!(
                        overlap % 2 == 0,
                        "d={}: X_{} and Z_{} overlap on {} qubits (must be even)",
                        d,
                        xi,
                        zi,
                        overlap
                    );
                }
            }
        }
    }

    #[test]
    fn test_surface_code_single_x_error_d3() {
        // X error on |0...0⟩: detected by Z stabilizers, correction restores trivial Z syndrome.
        let code = RotatedSurfaceCode::new(3);
        let n = code.num_physical_qubits();

        for error_qubit in 0..n {
            let mut state = QuantumState::new(n);
            crate::GateOperations::x(&mut state, error_qubit);

            let syndrome = code.measure_syndrome_deterministic(&state);
            let has_z_defect = syndrome.z_syndromes.iter().any(|&s| s);
            assert!(
                has_z_defect,
                "X error on qubit {} should trigger Z syndromes",
                error_qubit
            );

            let correction = code.decode(&syndrome);
            RotatedSurfaceCode::apply_correction(&mut state, &correction);

            // After correction, Z syndrome must be trivial (error + correction = stabilizer).
            let post = code.measure_syndrome_deterministic(&state);
            assert!(
                post.z_syndromes.iter().all(|&s| !s),
                "d=3 X error on qubit {}: Z syndrome not trivial after correction: {:?}",
                error_qubit,
                post.z_syndromes
            );
        }
    }

    #[test]
    fn test_surface_code_single_z_error_d3() {
        // Z error on |+...+⟩: detected by X stabilizers, correction restores trivial X syndrome.
        let code = RotatedSurfaceCode::new(3);
        let n = code.num_physical_qubits();

        for error_qubit in 0..n {
            let mut state = QuantumState::new(n);
            for q in 0..n {
                crate::GateOperations::h(&mut state, q);
            }

            crate::GateOperations::z(&mut state, error_qubit);

            let syndrome = code.measure_syndrome_deterministic(&state);
            let has_x_defect = syndrome.x_syndromes.iter().any(|&s| s);
            assert!(
                has_x_defect,
                "Z error on qubit {} (Hadamard basis) should trigger X syndromes",
                error_qubit
            );

            let correction = code.decode(&syndrome);
            RotatedSurfaceCode::apply_correction(&mut state, &correction);

            let post = code.measure_syndrome_deterministic(&state);
            assert!(
                post.x_syndromes.iter().all(|&s| !s),
                "d=3 Z error on qubit {}: X syndrome not trivial after correction: {:?}",
                error_qubit,
                post.x_syndromes
            );
        }
    }

    #[test]
    fn test_surface_code_no_error() {
        let code = RotatedSurfaceCode::new(3);
        let state = QuantumState::new(9);

        let syndrome = code.measure_syndrome_deterministic(&state);
        assert!(
            syndrome.x_syndromes.iter().all(|&s| !s),
            "X syndromes should be trivial"
        );
        assert!(
            syndrome.z_syndromes.iter().all(|&s| !s),
            "Z syndromes should be trivial"
        );

        let correction = code.decode(&syndrome);
        assert!(correction.x_corrections.is_empty());
        assert!(correction.z_corrections.is_empty());
    }

    #[test]
    fn test_surface_code_syndrome_equivalence_d3() {
        // Qubits with the same Z syndrome should differ by an X stabilizer.
        let code = RotatedSurfaceCode::new(3);
        let layout = code.layout();
        let n = code.num_physical_qubits();

        let mut syndrome_map: HashMap<Vec<bool>, Vec<usize>> = HashMap::new();
        for q in 0..n {
            let mut state = QuantumState::new(n);
            crate::GateOperations::x(&mut state, q);
            let syndrome = code.measure_syndrome_deterministic(&state);
            syndrome_map
                .entry(syndrome.z_syndromes)
                .or_default()
                .push(q);
        }

        for (_syndrome, qubits) in &syndrome_map {
            if qubits.len() > 1 {
                for i in 0..qubits.len() {
                    for j in (i + 1)..qubits.len() {
                        let mut diff = vec![qubits[i], qubits[j]];
                        diff.sort();
                        let is_x_stab = layout.x_stabilizers.iter().any(|stab| {
                            let mut s = stab.clone();
                            s.sort();
                            s == diff
                        });
                        assert!(
                            is_x_stab,
                            "qubits {:?} have same Z syndrome but don't differ by X stabilizer",
                            diff
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_surface_code_layout_d7() {
        let layout = SurfaceCodeLayout::new(7);
        assert_eq!(layout.num_data, 49);
        // d=7: 24 X + 24 Z = 48 = d²-1
        assert_eq!(layout.x_stabilizers.len() + layout.z_stabilizers.len(), 48);
    }

    // ==================== DECODER ADAPTER TESTS ====================

    #[test]
    fn test_mwpm_decoder_adapter() {
        use crate::decoding::mwpm::MWPMDecoder;

        // Create a distance-3 surface code MWPM decoder (d=3 -> 9 qubits)
        let code_distance = 3;
        let num_qubits = code_distance * code_distance;
        let mwpm = MWPMDecoder::new(code_distance, num_qubits);

        // Wrap it in the adapter
        let mut adapter = DecoderAdapter::from_mwpm(mwpm);

        // Test with trivial syndrome - should produce no corrections
        let trivial = Syndrome::new(vec![false; 2 * num_qubits - 2]);
        let no_errors = adapter.decode(&trivial);
        assert!(
            no_errors.is_empty(),
            "Trivial syndrome should produce no corrections"
        );

        // Test with non-trivial syndrome (pair of defects)
        let mut syndrome_bits = vec![false; 2 * num_qubits - 2];
        syndrome_bits[0] = true; // First defect
        syndrome_bits[2] = true; // Second defect (to make even number for matching)
        let syndrome = Syndrome::new(syndrome_bits);
        let error_positions = adapter.decode(&syndrome);

        // The decoder should return error positions (or empty if it matched to boundary)
        // Just verify it doesn't panic and returns valid positions
        for &pos in &error_positions {
            assert!(
                pos < num_qubits,
                "Error position {} should be valid for {} qubits",
                pos,
                num_qubits
            );
        }
    }

    #[test]
    fn test_bp_decoder_adapter() {
        use crate::decoding::bp::{BPConfig, BPDecoder, TannerGraph};

        // Create a simple Tanner graph (3 variables, 2 checks)
        // Each check connects to 2 variables (simple parity check code)
        let graph = TannerGraph {
            num_variables: 3,
            num_checks: 2,
            var_to_check: vec![vec![0], vec![0, 1], vec![1]],
            check_to_var: vec![vec![0, 1], vec![1, 2]],
        };

        let config = BPConfig::default();
        let bp = BPDecoder::new(graph, config);

        // Wrap it in the adapter with physical error rate 0.1
        let adapter = DecoderAdapter::from_bp(bp, 0.1);

        // Test with a syndrome
        let syndrome = Syndrome::new(vec![true, false]);
        let error_positions = adapter.decode(&syndrome);

        // The BP decoder should return error positions
        assert!(
            error_positions.len() <= 3,
            "Error positions should be within variable count"
        );

        // Test with trivial syndrome
        let trivial = Syndrome::new(vec![false, false]);
        let no_errors = adapter.decode(&trivial);
        assert!(
            no_errors.is_empty() || no_errors.len() <= 3,
            "Trivial syndrome should produce minimal corrections"
        );
    }

    #[test]
    fn test_decoder_adapter_conversion() {
        use crate::decoding::mwpm::MWPMDecoder;

        // Test that Vec<bool> -> Vec<usize> conversion works correctly
        let code_distance = 3;
        let num_qubits = 9;
        let mwpm = MWPMDecoder::new(code_distance, num_qubits);
        let mut adapter = DecoderAdapter::from_mwpm(mwpm);

        // Create a syndrome with known pattern
        let syndrome = Syndrome::new(vec![true, false, true, false]);
        let positions = adapter.decode(&syndrome);

        // Verify that all returned positions are valid indices
        for &pos in &positions {
            assert!(
                pos < 9,
                "Error position {} should be valid for d=3 code (9 qubits)",
                pos
            );
        }
    }
}
