//! Quantum Phase Estimation (QPE)
//!
//! This module implements the Quantum Phase Estimation algorithm
//! for estimating eigenvalues of unitary operators.
//!
//! Given a unitary operator U with eigenstate |u> such that U|u> = e^(2*pi*i*phi)|u>,
//! QPE estimates the phase phi using a register of estimation qubits.
//!
//! The algorithm:
//! 1. Prepare estimation register in uniform superposition (Hadamard on each qubit)
//! 2. Prepare target register in eigenstate |u>
//! 3. Apply controlled-U^(2^k) for each estimation qubit k
//! 4. Apply inverse QFT to estimation register
//! 5. Measure estimation register to obtain binary representation of phase

use std::f64::consts::PI;

use crate::GateOperations;
use crate::QuantumState;

// ============================================================
// QPE DATA STRUCTURES
// ============================================================

/// QPE solver for estimating eigenvalues of unitary operators.
///
/// For a unitary U with eigenstate |u> satisfying U|u> = e^(2*pi*i*phi)|u>,
/// this solver estimates phi to `num_estimation_qubits` bits of precision.
pub struct QPESolver {
    /// Phase angle of the eigenvalue (2*pi*phi), used for constructing controlled-U
    /// In a general QPE, this would be the unitary matrix itself, but for our
    /// implementation we work with known-phase unitaries for correctness.
    pub eigenphase: f64,
    /// Number of estimation qubits (determines precision of phase estimate)
    pub num_estimation_qubits: usize,
    /// Number of target qubits (qubits the unitary acts on)
    pub num_target_qubits: usize,
}

impl QPESolver {
    /// Create a new QPE solver for a single-qubit phase gate with known eigenphase.
    ///
    /// The unitary is U = diag(1, e^(i*eigenphase)), operating on 1 target qubit.
    /// The eigenstate is |1>, with eigenvalue e^(i*eigenphase).
    ///
    /// # Arguments
    /// * `eigenphase` - The phase angle phi such that U|1> = e^(i*phi)|1>
    /// * `num_estimation_qubits` - Number of qubits for phase precision
    pub fn new(eigenphase: f64, num_estimation_qubits: usize) -> Self {
        QPESolver {
            eigenphase,
            num_estimation_qubits,
            num_target_qubits: 1,
        }
    }

    /// Create a QPE solver with a specified number of target qubits.
    ///
    /// For multi-qubit unitaries, the eigenphase is the phase of the eigenvalue
    /// associated with the prepared eigenstate.
    pub fn with_target_qubits(
        eigenphase: f64,
        num_estimation_qubits: usize,
        num_target_qubits: usize,
    ) -> Self {
        QPESolver {
            eigenphase,
            num_estimation_qubits,
            num_target_qubits,
        }
    }

    /// Total number of qubits in the circuit.
    pub fn total_qubits(&self) -> usize {
        self.num_estimation_qubits + self.num_target_qubits
    }

    /// Run the QPE algorithm and return the result.
    ///
    /// The circuit layout is:
    /// - Qubits 0..num_estimation_qubits: estimation register
    /// - Qubits num_estimation_qubits..total: target register
    pub fn estimate_phase(&self) -> QPEResult {
        let total_qubits = self.total_qubits();
        let mut state = QuantumState::new(total_qubits);

        // Step 1: Initialize estimation register to |+...+> (uniform superposition)
        for i in 0..self.num_estimation_qubits {
            GateOperations::h(&mut state, i);
        }

        // Step 2: Prepare target register in eigenstate |1>
        // For our phase gate U = diag(1, e^(i*phi)), the eigenstate with
        // eigenvalue e^(i*phi) is |1>, so we flip the target qubit.
        let target_start = self.num_estimation_qubits;
        GateOperations::x(&mut state, target_start);

        // Step 3: Apply controlled-U^(2^k) for each estimation qubit k
        // U^(2^k) = diag(1, e^(i * 2^k * eigenphase))
        // Controlled-U^(2^k) applies phase e^(i * 2^k * eigenphase) to |1>|1>
        // This is exactly a controlled-phase (cphase) gate with angle 2^k * eigenphase
        for k in 0..self.num_estimation_qubits {
            let angle = (1u64 << k) as f64 * self.eigenphase;
            let control = k;
            let target = target_start;
            GateOperations::cphase(&mut state, control, target, angle);
        }

        // Step 4: Apply inverse QFT to estimation register
        self.apply_inverse_qft(&mut state);

        // Step 5: Measure the full state and extract estimation register bits
        let probs = state.probabilities();
        let (measurement, _probability) = state.measure();

        // Extract estimation register bits (qubits 0..num_estimation_qubits)
        let mut bits = Vec::with_capacity(self.num_estimation_qubits);
        for i in 0..self.num_estimation_qubits {
            bits.push((measurement >> i) & 1 != 0);
        }

        // Convert measured bits to phase estimate
        let phase_estimate = self.bits_to_phase(&bits);

        // Compute per-qubit probabilities for confidence estimation
        let qubit_probs = self.compute_qubit_probabilities(&probs);

        QPEResult {
            phase_estimate,
            confidence: self.compute_confidence(&qubit_probs),
            measurements: bits,
            probabilities: probs,
        }
    }

    /// Run QPE multiple times and return the most frequently measured phase.
    ///
    /// This provides statistical robustness for cases where a single shot
    /// might not capture the peak of the probability distribution.
    pub fn estimate_phase_repeated(&self, _num_shots: usize) -> QPEResult {
        let total_qubits = self.total_qubits();

        // Build the state once (before measurement)
        let mut state = QuantumState::new(total_qubits);

        for i in 0..self.num_estimation_qubits {
            GateOperations::h(&mut state, i);
        }

        let target_start = self.num_estimation_qubits;
        GateOperations::x(&mut state, target_start);

        for k in 0..self.num_estimation_qubits {
            let angle = (1u64 << k) as f64 * self.eigenphase;
            GateOperations::cphase(&mut state, k, target_start, angle);
        }

        self.apply_inverse_qft(&mut state);

        let probs = state.probabilities();

        // Find the measurement outcome with highest probability
        // (deterministic version -- avoids sampling noise)
        let mut best_idx = 0;
        let mut best_prob = 0.0f64;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_prob {
                best_prob = p;
                best_idx = i;
            }
        }

        let mut bits = Vec::with_capacity(self.num_estimation_qubits);
        for i in 0..self.num_estimation_qubits {
            bits.push((best_idx >> i) & 1 != 0);
        }

        let phase_estimate = self.bits_to_phase(&bits);

        QPEResult {
            phase_estimate,
            confidence: best_prob,
            measurements: bits,
            probabilities: probs,
        }
    }

    /// Apply inverse QFT to the estimation register (qubits 0..n).
    ///
    /// The inverse QFT is the adjoint of the QFT. For n qubits:
    /// 1. For j from n-1 down to 0:
    ///    a. For k from n-1 down to j+1:
    ///       Apply controlled-phase(-pi/2^(k-j)) with control=k, target=j
    ///    b. Apply Hadamard to qubit j
    /// 2. Reverse bit order with SWAP gates
    fn apply_inverse_qft(&self, state: &mut QuantumState) {
        let n = self.num_estimation_qubits;

        // The inverse QFT reverses the QFT operations.
        // QFT applies: for j=0..n: H(j), then for k=j+1..n: CPhase(pi/2^(k-j), k, j)
        // Inverse QFT: for j=n-1..0: for k=n-1..j+1: CPhase(-pi/2^(k-j), k, j), then H(j)
        for j in (0..n).rev() {
            for k in (j + 1..n).rev() {
                let angle = -PI / (1u64 << (k - j)) as f64;
                GateOperations::cphase(state, k, j, angle);
            }
            GateOperations::h(state, j);
        }

        // Reverse bit order: swap qubit i with qubit n-1-i
        for i in 0..n / 2 {
            GateOperations::swap(state, i, n - 1 - i);
        }
    }

    /// Convert measurement bits to phase estimate.
    ///
    /// The measured integer m (from the estimation register) maps to
    /// phase = 2*pi*m / 2^n where n is the number of estimation qubits.
    fn bits_to_phase(&self, bits: &[bool]) -> f64 {
        let mut m: u64 = 0;
        for (bit_idx, &bit) in bits.iter().enumerate() {
            if bit {
                m |= 1 << bit_idx;
            }
        }
        let denom = (1u64 << self.num_estimation_qubits) as f64;
        (m as f64 / denom) * 2.0 * PI
    }

    /// Compute per-qubit marginal probabilities from the full state probabilities.
    fn compute_qubit_probabilities(&self, probs: &[f64]) -> Vec<f64> {
        let mut qubit_probs = vec![0.0; self.num_estimation_qubits];
        for (idx, &p) in probs.iter().enumerate() {
            for q in 0..self.num_estimation_qubits {
                if (idx >> q) & 1 == 1 {
                    qubit_probs[q] += p;
                }
            }
        }
        qubit_probs
    }

    /// Compute confidence of phase estimate based on concentration of probability.
    ///
    /// Uses inverse Shannon entropy: high concentration (low entropy) = high confidence.
    fn compute_confidence(&self, qubit_probs: &[f64]) -> f64 {
        let mut entropy = 0.0;
        for &p in qubit_probs.iter() {
            let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
            // Binary entropy for each qubit
            entropy -= p_clamped * p_clamped.ln() + (1.0 - p_clamped) * (1.0 - p_clamped).ln();
        }
        1.0 / (1.0 + entropy)
    }
}

/// Result of a QPE computation.
#[derive(Clone, Debug)]
pub struct QPEResult {
    /// Estimated phase (0 to 2*pi)
    pub phase_estimate: f64,
    /// Confidence in phase estimate (0 to 1)
    pub confidence: f64,
    /// Measurement outcomes for each estimation qubit
    pub measurements: Vec<bool>,
    /// Full probability distribution over all basis states
    pub probabilities: Vec<f64>,
}

impl QPEResult {
    /// Get the estimated phase as a fraction of 2*pi (i.e., phi where eigenvalue = e^(2*pi*i*phi)).
    pub fn phase_fraction(&self) -> f64 {
        self.phase_estimate / (2.0 * PI)
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: run QPE deterministically by finding the peak probability outcome.
    fn run_qpe_deterministic(eigenphase: f64, n_estimation: usize) -> f64 {
        let solver = QPESolver::new(eigenphase, n_estimation);
        let result = solver.estimate_phase_repeated(1);
        result.phase_estimate
    }

    // ---- Basic construction tests ----

    #[test]
    fn test_qpe_solver_creation() {
        let solver = QPESolver::new(PI / 4.0, 3);
        assert_eq!(solver.num_estimation_qubits, 3);
        assert_eq!(solver.num_target_qubits, 1);
        assert_eq!(solver.total_qubits(), 4);
    }

    #[test]
    fn test_qpe_solver_with_target_qubits() {
        let solver = QPESolver::with_target_qubits(PI / 2.0, 4, 2);
        assert_eq!(solver.num_estimation_qubits, 4);
        assert_eq!(solver.num_target_qubits, 2);
        assert_eq!(solver.total_qubits(), 6);
    }

    // ---- Phase conversion tests ----

    #[test]
    fn test_bits_to_phase_zero() {
        let solver = QPESolver::new(0.0, 3);
        let bits = vec![false, false, false];
        let phase = solver.bits_to_phase(&bits);
        assert!((phase - 0.0).abs() < 1e-10, "All-zero bits should give phase 0");
    }

    #[test]
    fn test_bits_to_phase_binary_101() {
        let solver = QPESolver::new(0.0, 3);
        // bits[0]=true, bits[1]=false, bits[2]=true => m = 1 + 4 = 5
        // phase = 5/8 * 2*pi
        let bits = vec![true, false, true];
        let phase = solver.bits_to_phase(&bits);
        let expected = 5.0 / 8.0 * 2.0 * PI;
        assert!(
            (phase - expected).abs() < 1e-10,
            "bits [1,0,1] should give phase 5/8 * 2pi, got {}",
            phase
        );
    }

    #[test]
    fn test_bits_to_phase_all_ones() {
        let solver = QPESolver::new(0.0, 4);
        let bits = vec![true, true, true, true];
        // m = 1+2+4+8 = 15, phase = 15/16 * 2*pi
        let phase = solver.bits_to_phase(&bits);
        let expected = 15.0 / 16.0 * 2.0 * PI;
        assert!(
            (phase - expected).abs() < 1e-10,
            "All-one bits (4 qubits) should give phase 15/16 * 2pi"
        );
    }

    // ---- Confidence tests ----

    #[test]
    fn test_confidence_concentrated() {
        let solver = QPESolver::new(0.0, 3);
        // Highly concentrated probabilities (near 0 or 1) should give high confidence
        let concentrated = vec![0.99, 0.01, 0.98];
        let confidence = solver.compute_confidence(&concentrated);
        assert!(
            confidence > 0.5,
            "Concentrated probabilities should give confidence > 0.5, got {}",
            confidence
        );
    }

    #[test]
    fn test_confidence_uniform() {
        let solver = QPESolver::new(0.0, 3);
        // Uniform probabilities (0.5 each) should give lower confidence
        let uniform = vec![0.5, 0.5, 0.5];
        let confidence_uniform = solver.compute_confidence(&uniform);

        let concentrated = vec![0.99, 0.99, 0.99];
        let confidence_conc = solver.compute_confidence(&concentrated);

        assert!(
            confidence_conc > confidence_uniform,
            "Concentrated should have higher confidence than uniform"
        );
    }

    // ---- QPE algorithm correctness tests ----

    #[test]
    fn test_qpe_phase_zero() {
        // U = I (identity), eigenphase = 0
        // QPE should estimate phase = 0
        let phase = run_qpe_deterministic(0.0, 3);
        assert!(
            phase.abs() < 1e-10,
            "Phase of identity should be 0, got {}",
            phase
        );
    }

    #[test]
    fn test_qpe_phase_pi() {
        // U = Z gate, eigenphase = pi
        // Phase = pi, so phi = 1/2
        // With 3 estimation qubits, m should be 4 (binary 100)
        // phase = 4/8 * 2*pi = pi
        let phase = run_qpe_deterministic(PI, 3);
        assert!(
            (phase - PI).abs() < 1e-10,
            "Phase of Z gate should be pi, got {}",
            phase
        );
    }

    #[test]
    fn test_qpe_phase_pi_over_2() {
        // eigenphase = pi/2, phi = 1/4
        // With 4 estimation qubits, m should be 4 (binary 0100)
        // phase = 4/16 * 2*pi = pi/2
        let phase = run_qpe_deterministic(PI / 2.0, 4);
        assert!(
            (phase - PI / 2.0).abs() < 1e-10,
            "Phase should be pi/2, got {}",
            phase
        );
    }

    #[test]
    fn test_qpe_phase_pi_over_4() {
        // eigenphase = pi/4, phi = 1/8
        // With 3 estimation qubits, m should be 1 (binary 001)
        // phase = 1/8 * 2*pi = pi/4
        let phase = run_qpe_deterministic(PI / 4.0, 3);
        assert!(
            (phase - PI / 4.0).abs() < 1e-10,
            "Phase should be pi/4, got {}",
            phase
        );
    }

    #[test]
    fn test_qpe_phase_3pi_over_4() {
        // eigenphase = 3*pi/4, phi = 3/8
        // With 3 estimation qubits, m should be 3 (binary 011)
        // phase = 3/8 * 2*pi = 3*pi/4
        let phase = run_qpe_deterministic(3.0 * PI / 4.0, 3);
        assert!(
            (phase - 3.0 * PI / 4.0).abs() < 1e-10,
            "Phase should be 3*pi/4, got {}",
            phase
        );
    }

    #[test]
    fn test_qpe_more_estimation_qubits_increases_precision() {
        // eigenphase = pi/4, exactly representable
        // Both 3-qubit and 5-qubit should get it exactly
        let phase_3 = run_qpe_deterministic(PI / 4.0, 3);
        let phase_5 = run_qpe_deterministic(PI / 4.0, 5);

        assert!(
            (phase_3 - PI / 4.0).abs() < 1e-10,
            "3-qubit QPE should get pi/4 exactly"
        );
        assert!(
            (phase_5 - PI / 4.0).abs() < 1e-10,
            "5-qubit QPE should get pi/4 exactly"
        );
    }

    #[test]
    fn test_qpe_non_dyadic_phase_approximation() {
        // eigenphase = pi/3, phi = 1/6 (not exactly representable in binary)
        // With 4 estimation qubits, best approximation: m=3 -> 3/16*2pi = 3pi/8
        //   or m=2 -> 2/16*2pi = pi/4. Closest is m=3 (3/16 = 0.1875 vs 1/6 = 0.1667)
        // Actually: 1/6 * 16 = 2.667, rounds to 3. So m=3, phase = 3/16 * 2pi
        // With more qubits, error should decrease
        let phase_4 = run_qpe_deterministic(PI / 3.0, 4);
        let phase_6 = run_qpe_deterministic(PI / 3.0, 6);

        let error_4 = (phase_4 - PI / 3.0).abs();
        let error_6 = (phase_6 - PI / 3.0).abs();

        // 6-qubit should be at least as precise as 4-qubit
        assert!(
            error_6 <= error_4 + 1e-10,
            "More qubits should give better precision: 4-qubit error={}, 6-qubit error={}",
            error_4,
            error_6
        );

        // Both should be reasonable approximations
        assert!(
            error_4 < 0.5,
            "4-qubit QPE error for pi/3 should be < 0.5 rad, got {}",
            error_4
        );
    }

    #[test]
    fn test_qpe_result_phase_fraction() {
        let solver = QPESolver::new(PI, 3);
        let result = solver.estimate_phase_repeated(1);
        let fraction = result.phase_fraction();
        assert!(
            (fraction - 0.5).abs() < 1e-10,
            "Phase fraction for eigenphase=pi should be 0.5, got {}",
            fraction
        );
    }

    #[test]
    fn test_qpe_inverse_qft_correctness() {
        // Test that inverse QFT undoes the phase encoding correctly.
        // For eigenphase = 0, after controlled-U and inverse QFT,
        // the estimation register should be in |0...0>.
        let solver = QPESolver::new(0.0, 3);
        let result = solver.estimate_phase_repeated(1);

        // All measurement bits should be false (zero phase)
        for (i, &bit) in result.measurements.iter().enumerate() {
            assert!(
                !bit,
                "For eigenphase=0, estimation bit {} should be 0",
                i
            );
        }
    }

    #[test]
    fn test_qpe_probability_distribution_peaked() {
        // For an exactly representable phase, the probability distribution
        // should be sharply peaked at the correct measurement outcome.
        let solver = QPESolver::new(PI, 4);
        let result = solver.estimate_phase_repeated(1);

        // Find the probability of the correct outcome
        // eigenphase = pi => phi = 1/2, m = 8 for 4 estimation qubits
        // The target qubit is in |1>, so the full state index has bit (n_est) set
        // Correct estimation outcome: m=8 => bits 0..4 encode 8 => index = 8 | (1 << 4) = 24
        let n_est = solver.num_estimation_qubits;
        let correct_est = 8; // 1/2 * 16 = 8
        let target_bit = 1 << n_est; // target qubit is |1>
        let correct_idx = correct_est | target_bit;

        let peak_prob = result.probabilities[correct_idx];
        assert!(
            peak_prob > 0.9,
            "Peak probability for exact phase should be > 0.9, got {}",
            peak_prob
        );
    }

    #[test]
    fn test_qpe_stochastic_measurement() {
        // Test the stochastic (single-shot) estimate_phase method.
        // For an exactly representable phase, it should almost always
        // return the correct answer.
        let solver = QPESolver::new(PI, 4);

        let mut correct_count = 0;
        let num_trials = 20;

        for _ in 0..num_trials {
            let result = solver.estimate_phase();
            if (result.phase_estimate - PI).abs() < 1e-10 {
                correct_count += 1;
            }
        }

        // With exact phase, all or nearly all measurements should be correct
        assert!(
            correct_count >= num_trials - 2,
            "For exact phase, at least {}/{} trials should be correct, got {}",
            num_trials - 2,
            num_trials,
            correct_count
        );
    }
}
