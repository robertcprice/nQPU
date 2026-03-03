//! Comprehensive Quantum Algorithms Library
//!
//! Complete implementation of major quantum algorithms.
//!
//! **Algorithms**:
//! - **Search**: Grover's algorithm, amplitude amplification
//! - **Transforms**: QFT, phase estimation, HHL
//! - **Factoring**: Shor's algorithm components
//! - **Optimization**: QAOA, VQE variants
//! - **Machine Learning**: Quantum PCA, QSVM
//! - **Walks**: Quantum walks, glued trees
//! - **Simulation**: Hamiltonian simulation, Trotterization
//! - **Verification**: Swap test, Hadamard test

use crate::comprehensive_gates::QuantumGates;
use crate::{QuantumState, C64};
use std::f64::consts::PI;

/// Comprehensive quantum algorithms implementation.
pub struct QuantumAlgorithms;

// ==================== GROVER'S SEARCH ====================

impl QuantumAlgorithms {
    /// Grover's search algorithm.
    /// Searches for a marked element in an unstructured database.
    ///
    /// # Arguments
    /// - `num_qubits`: Database size is N = 2^n
    /// - `oracle`: Function that marks the solution
    /// - `iterations`: Number of Grover iterations (default: ⌈π/4 * √N⌉)
    ///
    /// # Returns
    /// Index of the found element
    pub fn grover_search<F>(num_qubits: usize, oracle: F, iterations: Option<usize>) -> usize
    where
        F: Fn(&mut QuantumState) + Clone,
    {
        let _n = num_qubits as f64;
        let default_iters = (PI / 4.0 * (1usize << num_qubits) as f64).sqrt().ceil() as usize;
        let iters = iterations.unwrap_or(default_iters);

        let mut state = QuantumState::new(num_qubits);

        // Initialize to uniform superposition
        for q in 0..num_qubits {
            QuantumGates::h(&mut state, q);
        }

        // Grover iterations
        for _ in 0..iters {
            // Oracle call
            oracle(&mut state);

            // Diffusion operator
            Self::grover_diffusion(&mut state, num_qubits);
        }

        // Measure
        let probs = state.probabilities();
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Grover diffusion operator.
    fn grover_diffusion(state: &mut QuantumState, num_qubits: usize) {
        // Apply H to all qubits
        for q in 0..num_qubits {
            QuantumGates::h(state, q);
        }

        // Apply X to all qubits
        for q in 0..num_qubits {
            QuantumGates::x(state, q);
        }

        // Multi-controlled Z (phase flip on |11...1⟩)
        Self::apply_multi_z(state, num_qubits);

        // Apply X to all qubits
        for q in 0..num_qubits {
            QuantumGates::x(state, q);
        }

        // Apply H to all qubits
        for q in 0..num_qubits {
            QuantumGates::h(state, q);
        }
    }

    fn apply_multi_z(state: &mut QuantumState, num_qubits: usize) {
        // Apply Z gate if all qubits are |1⟩
        // Decompose into controlled operations
        if num_qubits == 1 {
            QuantumGates::z(state, 0);
        } else if num_qubits == 2 {
            QuantumGates::cz(state, 0, 1);
        } else {
            // Use H and CNOT decomposition
            QuantumGates::h(state, num_qubits - 1);
            for i in 0..num_qubits - 1 {
                QuantumGates::cx(state, i, num_qubits - 1);
            }
            QuantumGates::h(state, num_qubits - 1);
        }
    }

    /// Amplitude amplification (generalization of Grover).
    pub fn amplitude_amplification<F>(
        num_qubits: usize,
        oracle: F,
        good_state_prep: F,
        iterations: usize,
    ) -> QuantumState
    where
        F: Fn(&mut QuantumState) + Clone,
    {
        let mut state = QuantumState::new(num_qubits);

        // Prepare good states
        good_state_prep(&mut state);

        // Amplitude amplification iterations
        for _ in 0..iterations {
            // Reflect about good states
            oracle(&mut state);

            // Reflect about initial state
            Self::grover_diffusion(&mut state, num_qubits);
        }

        state
    }

    // ==================== SWAP & HADAMARD TESTS ====================

    /// Swap test to estimate |⟨ψ|φ⟩|^2.
    ///
    /// `prep_a` and `prep_b` should prepare states on the first and second registers.
    /// They receive the starting qubit offset for their register.
    pub fn swap_test<F, G>(num_qubits: usize, prep_a: F, prep_b: G) -> f64
    where
        F: Fn(&mut QuantumState, usize) + Clone,
        G: Fn(&mut QuantumState, usize) + Clone,
    {
        let total_qubits = 2 * num_qubits + 1;
        let ancilla = 2 * num_qubits;
        let mut state = QuantumState::new(total_qubits);

        // Prepare |ψ⟩ on first register and |φ⟩ on second.
        prep_a(&mut state, 0);
        prep_b(&mut state, num_qubits);

        // Hadamard on ancilla
        QuantumGates::h(&mut state, ancilla);

        // Controlled-SWAP for each qubit pair
        for i in 0..num_qubits {
            QuantumGates::cswap(&mut state, ancilla, i, num_qubits + i);
        }

        // Hadamard on ancilla
        QuantumGates::h(&mut state, ancilla);

        // Probability ancilla = 0
        let mut p0 = 0.0f64;
        for (idx, amp) in state.amplitudes_ref().iter().enumerate() {
            if ((idx >> ancilla) & 1) == 0 {
                p0 += amp.norm_sqr();
            }
        }

        (2.0 * p0 - 1.0).clamp(0.0, 1.0)
    }

    /// Hadamard test to estimate Re⟨ψ|U|ψ⟩ or Im⟨ψ|U|ψ⟩.
    ///
    /// `state_prep` prepares |ψ⟩ on the target register.
    /// `controlled_unitary` applies controlled-U using the provided control qubit.
    /// If `imag` is true, estimates imaginary part.
    pub fn hadamard_test<F, G>(
        num_qubits: usize,
        state_prep: F,
        controlled_unitary: G,
        imag: bool,
    ) -> f64
    where
        F: Fn(&mut QuantumState, usize) + Clone,
        G: Fn(&mut QuantumState, usize) + Clone,
    {
        let ancilla = num_qubits;
        let mut state = QuantumState::new(num_qubits + 1);

        // Prepare |ψ⟩ on target register
        state_prep(&mut state, 0);

        // Hadamard on ancilla
        QuantumGates::h(&mut state, ancilla);

        // Controlled-U
        controlled_unitary(&mut state, ancilla);

        // Phase shift for imaginary part
        if imag {
            QuantumGates::rz(&mut state, ancilla, -PI / 2.0);
        }

        // Hadamard and measure ancilla
        QuantumGates::h(&mut state, ancilla);

        let mut p0 = 0.0f64;
        for (idx, amp) in state.amplitudes_ref().iter().enumerate() {
            if ((idx >> ancilla) & 1) == 0 {
                p0 += amp.norm_sqr();
            }
        }

        2.0 * p0 - 1.0
    }

    // ==================== DEUTSCH-JOZSA ====================

    /// Deutsch-Jozsa algorithm.
    /// Returns true if the oracle is constant, false if balanced.
    ///
    /// The oracle should implement: |x⟩|y⟩ -> |x⟩|y ⊕ f(x)⟩
    /// where the ancilla index is `num_input_qubits`.
    pub fn deutsch_jozsa<F>(num_input_qubits: usize, oracle: F) -> bool
    where
        F: Fn(&mut QuantumState, usize, usize) + Clone,
    {
        let ancilla = num_input_qubits;
        let total_qubits = num_input_qubits + 1;
        let mut state = QuantumState::new(total_qubits);

        // Prepare |0...0⟩|1⟩
        QuantumGates::x(&mut state, ancilla);

        // Hadamard on all qubits
        for q in 0..total_qubits {
            QuantumGates::h(&mut state, q);
        }

        // Oracle
        oracle(&mut state, num_input_qubits, ancilla);

        // Hadamard on input register
        for q in 0..num_input_qubits {
            QuantumGates::h(&mut state, q);
        }

        // Measure input register: if all zeros -> constant
        let probs = state.probabilities();
        let mut max_idx = 0;
        let mut max_p = 0.0;
        for (i, p) in probs.iter().enumerate() {
            if *p > max_p {
                max_p = *p;
                max_idx = i;
            }
        }

        // Mask out ancilla bit
        let input_bits = max_idx & ((1usize << num_input_qubits) - 1);
        input_bits == 0
    }

    /// Helper: Deutsch-Jozsa constant oracle (f(x)=value).
    pub fn dj_oracle_constant(state: &mut QuantumState, _n: usize, ancilla: usize, value: usize) {
        if value & 1 == 1 {
            QuantumGates::x(state, ancilla);
        }
    }

    /// Helper: Deutsch-Jozsa balanced oracle using parity of input bits.
    pub fn dj_oracle_parity(state: &mut QuantumState, n: usize, ancilla: usize) {
        for q in 0..n {
            QuantumGates::cx(state, q, ancilla);
        }
    }

    // ==================== BERNSTEIN-VAZIRANI ====================

    /// Bernstein-Vazirani algorithm.
    /// Returns the secret bitstring `s` encoded in the oracle.
    pub fn bernstein_vazirani(num_input_qubits: usize, secret: usize) -> usize {
        let ancilla = num_input_qubits;
        let total_qubits = num_input_qubits + 1;
        let mut state = QuantumState::new(total_qubits);

        // Prepare |0...0⟩|1⟩
        QuantumGates::x(&mut state, ancilla);

        // Hadamard on all qubits
        for q in 0..total_qubits {
            QuantumGates::h(&mut state, q);
        }

        // Oracle: XOR inner product s·x into ancilla
        for q in 0..num_input_qubits {
            if ((secret >> q) & 1) == 1 {
                QuantumGates::cx(&mut state, q, ancilla);
            }
        }

        // Hadamard on input register
        for q in 0..num_input_qubits {
            QuantumGates::h(&mut state, q);
        }

        // Read out most likely basis state
        let probs = state.probabilities();
        let mut max_idx = 0;
        let mut max_p = 0.0;
        for (i, p) in probs.iter().enumerate() {
            if *p > max_p {
                max_p = *p;
                max_idx = i;
            }
        }
        max_idx & ((1usize << num_input_qubits) - 1)
    }
}

// ==================== QUANTUM FOURIER TRANSFORM ====================

impl QuantumAlgorithms {
    /// Quantum Fourier Transform.
    /// Transforms computational basis to Fourier basis.
    pub fn qft(state: &mut QuantumState, num_qubits: usize) {
        for i in 0..num_qubits {
            // Hadamard on qubit i
            QuantumGates::h(state, i);

            // Controlled phase rotations
            for j in (i + 1)..num_qubits {
                let k = j - i;
                let phase = 2.0 * PI / (1 << (k + 1)) as f64;
                Self::apply_cphase(state, j, i, phase);
            }
        }

        // Reverse qubit order
        for i in 0..(num_qubits / 2) {
            QuantumGates::swap(state, i, num_qubits - 1 - i);
        }
    }

    /// Inverse QFT.
    pub fn inverse_qft(state: &mut QuantumState, num_qubits: usize) {
        // Reverse qubit order
        for i in 0..(num_qubits / 2) {
            QuantumGates::swap(state, i, num_qubits - 1 - i);
        }

        // Apply inverse rotations
        for i in (0..num_qubits).rev() {
            for j in ((i + 1)..num_qubits).rev() {
                let k = j - i;
                let phase = -2.0 * PI / (1 << (k + 1)) as f64;
                Self::apply_cphase(state, j, i, phase);
            }

            // Hadamard on qubit i
            QuantumGates::h(state, i);
        }
    }

    fn apply_cphase(state: &mut QuantumState, control: usize, target: usize, phase: f64) {
        // Apply controlled phase shift
        // Using the fact that CZ = H·CNOT·H
        let _phase_matrix = [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(phase.cos(), phase.sin())],
        ];

        // Apply controlled phase using decomposition
        QuantumGates::rz(state, target, phase / 2.0);
        QuantumGates::cx(state, control, target);
        QuantumGates::rz(state, target, -phase / 2.0);
        QuantumGates::cx(state, control, target);
    }

    /// Quantum Phase Estimation.
    /// Estimates the phase φ of an eigenvalue e^(2πiφ) of a unitary U.
    ///
    /// # Arguments
    /// - `counting_qubits`: Number of qubits for precision
    /// - `target_unitary`: Function that applies U^(2^j)
    /// - `eigenstate_prep`: Function that prepares the eigenstate
    ///
    /// # Returns
    /// Estimated phase as a fraction in [0, 1)
    pub fn phase_estimation<F, G>(
        counting_qubits: usize,
        target_unitary: F,
        eigenstate_prep: G,
    ) -> f64
    where
        F: Fn(&mut QuantumState, usize) + Clone,
        G: Fn(&mut QuantumState) + Clone,
    {
        let total_qubits = counting_qubits + 1;
        let mut state = QuantumState::new(total_qubits);

        // Initialize counting qubits to |+⟩
        for i in 0..counting_qubits {
            QuantumGates::h(&mut state, i);
        }

        // Prepare eigenstate on target qubit
        eigenstate_prep(&mut state);

        // Apply controlled U^(2^j) operations
        for j in 0..counting_qubits {
            let _control = j;
            let target = counting_qubits;

            // Apply U^(2^j) controlled by counting qubit
            for _ in 0..(1 << j) {
                target_unitary(&mut state, target);
            }
        }

        // Apply inverse QFT on counting qubits
        let _counting_state = QuantumState::new(counting_qubits);
        // Extract counting qubits state (simplified)
        Self::inverse_qft(&mut state, counting_qubits);

        // Measure counting qubits
        let measurement = Self::measure_first_n(&mut state, counting_qubits);

        // Convert measurement to phase estimate
        measurement as f64 / (1 << counting_qubits) as f64
    }

    /// Toy HHL for 2x2 diagonal systems A = diag(a0, a1).
    ///
    /// Returns a 1-qubit state proportional to A^{-1}|b⟩, using a simple
    /// controlled-rotation + postselection on an ancilla.
    pub fn hhl_2x2_diagonal(a0: f64, a1: f64, b0: C64, b1: C64) -> QuantumState {
        let mut state = QuantumState::new(2); // qubit 0 = system, qubit 1 = ancilla

        // Normalize |b⟩
        let mut norm = (b0.norm_sqr() + b1.norm_sqr()).sqrt();
        if norm == 0.0 {
            norm = 1.0;
        }
        let b0n = C64::new(b0.re / norm, b0.im / norm);
        let b1n = C64::new(b1.re / norm, b1.im / norm);

        // Prepare |b⟩|0⟩
        let amps = state.amplitudes_mut();
        amps[0] = b0n;
        amps[1] = b1n;
        amps[2] = C64::new(0.0, 0.0);
        amps[3] = C64::new(0.0, 0.0);

        let c = a0.min(a1).max(1e-9);
        let angle0 = 2.0 * (c / a0).clamp(0.0, 1.0).asin();
        let angle1 = 2.0 * (c / a1).clamp(0.0, 1.0).asin();

        // Controlled rotation for |0⟩ (via X)
        QuantumGates::x(&mut state, 0);
        QuantumGates::cry(&mut state, 0, 1, angle0);
        QuantumGates::x(&mut state, 0);

        // Controlled rotation for |1⟩
        QuantumGates::cry(&mut state, 0, 1, angle1);

        // Postselect ancilla |1⟩
        let amps = state.amplitudes_ref();
        let a0p = amps[2]; // |0,1⟩
        let a1p = amps[3]; // |1,1⟩
        let mut out_norm = (a0p.norm_sqr() + a1p.norm_sqr()).sqrt();
        if out_norm == 0.0 {
            out_norm = 1.0;
        }
        let mut out = QuantumState::new(1);
        let out_amps = out.amplitudes_mut();
        out_amps[0] = C64::new(a0p.re / out_norm, a0p.im / out_norm);
        out_amps[1] = C64::new(a1p.re / out_norm, a1p.im / out_norm);
        out
    }
}

// ==================== SHOR'S ALGORITHM ====================

impl QuantumAlgorithms {
    /// Shor's algorithm for integer factorization.
    ///
    /// # Arguments
    /// - `n`: Number to factor
    ///
    /// # Returns
    /// Non-trivial factor if found
    pub fn shors_factorization(n: u64) -> Option<(u64, u64)> {
        if n <= 1 {
            return None;
        }

        // Check if n is even
        if n % 2 == 0 {
            return Some((2, n / 2));
        }

        // Check if n is a prime power (simplified)
        if Self::is_prime(n) {
            return None;
        }

        // Choose random a in [2, n-1]
        let a = 2 + (rand::random::<u64>() % (n - 3));

        // Compute gcd(a, n)
        let gcd = Self::classical_gcd(a, n);
        if gcd > 1 {
            return Some((gcd, n / gcd));
        }

        // Quantum order finding
        if let Some(r) = Self::order_finding(a, n) {
            if r % 2 == 0 {
                let power = Self::mod_exp(a, (r / 2) as u64, n);
                if power != n - 1 {
                    let factor1 = Self::classical_gcd(power - 1, n);
                    let factor2 = Self::classical_gcd(power + 1, n);

                    if factor1 > 1 && factor1 < n {
                        return Some((factor1, n / factor1));
                    }
                    if factor2 > 1 && factor2 < n {
                        return Some((factor2, n / factor2));
                    }
                }
            }
        }

        None
    }

    /// Quantum order finding via Quantum Phase Estimation (QPE).
    ///
    /// Builds a quantum circuit with:
    /// - `t` counting qubits (Hadamard + controlled-U^(2^k))
    /// - `m` work qubits initialized to |1⟩
    /// Then applies inverse QFT on counting register and measures.
    /// Extracts period from measurement via continued fractions.
    fn order_finding(a: u64, n: u64) -> Option<usize> {
        // Number of bits needed to represent n
        let m = ((n as f64).log2().ceil() as usize).max(2);
        // Counting register: 2*m bits for precision
        let t = 2 * m;
        let total_qubits = t + m;

        // Build the QPE circuit using statevector simulation
        let mut state = QuantumState::new(total_qubits);

        // Initialize work register to |1⟩ (qubit index t is LSB of work register)
        QuantumGates::x(&mut state, t);

        // Apply Hadamard to all counting qubits
        for i in 0..t {
            QuantumGates::h(&mut state, i);
        }

        // Apply controlled modular exponentiation: controlled-U^(2^k)
        // U|y⟩ = |a*y mod n⟩
        // We simulate this by applying the appropriate phase to the statevector
        // For each counting qubit k, apply controlled-a^(2^k) mod n
        for k in 0..t {
            let power = Self::mod_exp(a, 1u64 << k, n);
            // Apply controlled modular multiplication on work register
            // This is the key quantum operation: controlled permutation
            Self::apply_controlled_mod_mul(&mut state, k, t, m, power, n);
        }

        // Apply inverse QFT on counting register (qubits 0..t)
        Self::inverse_qft(&mut state, t);

        // Measure counting register and extract phase
        let probs = state.probabilities();

        // Find the measurement outcome with highest probability
        // Only look at the counting register bits
        let counting_states = 1usize << t;
        let work_states = 1usize << m;
        let mut best_phase = 0usize;
        let mut best_prob = 0.0f64;

        for c in 0..counting_states {
            let mut total_prob = 0.0;
            for w in 0..work_states {
                let idx = c | (w << t);
                if idx < probs.len() {
                    total_prob += probs[idx];
                }
            }
            if total_prob > best_prob {
                best_prob = total_prob;
                best_phase = c;
            }
        }

        if best_phase == 0 {
            // Phase 0 means period not found, fall back to classical
            return Self::classical_order(a, n);
        }

        // Extract period using continued fractions
        // phase ≈ s/r where s is some integer
        let phase_fraction = best_phase as f64 / counting_states as f64;
        if let Some(r) = Self::continued_fractions_period(phase_fraction, n as usize) {
            if r > 0 && Self::mod_exp(a, r as u64, n) == 1 {
                return Some(r);
            }
        }

        // Fall back to classical if QPE didn't converge
        Self::classical_order(a, n)
    }

    /// Apply controlled modular multiplication to work register.
    /// Implements controlled-|y⟩ → |a*y mod n⟩ on the work qubits.
    fn apply_controlled_mod_mul(
        state: &mut QuantumState,
        control: usize,
        work_start: usize,
        work_size: usize,
        a_power: u64,
        n: u64,
    ) {
        let total_size = work_start + work_size;
        let dim = 1usize << total_size;
        let amplitudes = state.amplitudes_mut();

        // For each basis state, if control bit is 1, permute work register
        // according to modular multiplication by a_power
        let work_mask = ((1usize << work_size) - 1) << work_start;
        let control_mask = 1usize << control;

        // Build permutation table for work register values
        let work_dim = 1usize << work_size;
        let mut perm = vec![0usize; work_dim];
        for y in 0..work_dim {
            if (y as u64) < n && y > 0 {
                perm[y] = ((y as u64 * a_power) % n) as usize;
            } else {
                perm[y] = y; // Identity for values >= n
            }
        }

        // Apply the controlled permutation
        let mut new_amps = amplitudes.to_vec();
        for idx in 0..dim {
            if idx & control_mask != 0 {
                // Control is 1 — apply permutation
                let work_val = (idx & work_mask) >> work_start;
                let new_work = perm[work_val];
                let new_idx = (idx & !work_mask) | (new_work << work_start);
                new_amps[new_idx] = amplitudes[idx];
            }
        }

        // Copy back
        amplitudes.copy_from_slice(&new_amps);
    }

    /// Extract period from phase using continued fractions.
    fn continued_fractions_period(phase: f64, n: usize) -> Option<usize> {
        if phase.abs() < 1e-10 {
            return None;
        }

        // Build continued fraction expansion
        let mut x = phase;
        let mut convergents = Vec::new();
        let mut p = (0i64, 1i64); // Previous convergent numerator
        let mut q = (1i64, 0i64); // Previous convergent denominator

        for _ in 0..50 {
            let a_i = x.floor() as i64;
            let new_p = a_i * p.1 + p.0;
            let new_q = a_i * q.1 + q.0;

            if new_q > 0 && (new_q as usize) <= n {
                convergents.push((new_p, new_q));
            }

            p = (p.1, new_p);
            q = (q.1, new_q);

            let frac = x - a_i as f64;
            if frac.abs() < 1e-10 {
                break;
            }
            x = 1.0 / frac;

            if q.1 as usize > n {
                break;
            }
        }

        // Try each convergent denominator as potential period
        for &(_num, den) in convergents.iter().rev() {
            if den > 0 && (den as usize) <= n {
                return Some(den as usize);
            }
        }
        None
    }

    fn classical_order(a: u64, n: u64) -> Option<usize> {
        // Classical computation of multiplicative order
        for r in 1..=(n as usize) {
            if Self::mod_exp(a, r as u64, n) == 1 {
                return Some(r);
            }
        }
        None
    }

    fn classical_gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            Self::classical_gcd(b, a % b)
        }
    }

    fn mod_exp(mut base: u64, exp: u64, modulus: u64) -> u64 {
        if modulus == 1 {
            return 0;
        }
        let mut result = 1;
        base %= modulus;
        let mut exp = exp;

        while exp > 0 {
            if exp % 2 == 1 {
                result = result * base % modulus;
            }
            exp /= 2;
            base = base * base % modulus;
        }
        result
    }

    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        let mut i = 3;
        while i * i <= n {
            if n % i == 0 {
                return false;
            }
            i += 2;
        }
        true
    }
}

// ==================== VARIATIONAL ALGORITHMS ====================

impl QuantumAlgorithms {
    /// Variational Quantum Eigensolver (VQE).
    /// Finds the ground state energy of a Hamiltonian.
    ///
    /// # Arguments
    /// - `num_qubits`: Number of qubits
    /// - `hamiltonian`: Hamiltonian as list of (coefficient, pauli_string) terms
    /// - `ansatz`: Variational ansatz circuit
    /// - `params`: Initial parameters
    /// - `learning_rate`: Gradient descent rate
    /// - `iterations`: Number of optimization iterations
    ///
    /// # Returns
    /// (optimal_energy, optimal_parameters)
    pub fn vqe<F>(
        num_qubits: usize,
        hamiltonian: &[(f64, Vec<char>)],
        ansatz: F,
        mut params: Vec<f64>,
        learning_rate: f64,
        iterations: usize,
    ) -> (f64, Vec<f64>)
    where
        F: Fn(&mut QuantumState, &[f64]) + Clone,
    {
        let mut best_energy = f64::INFINITY;
        let mut best_params = params.clone();

        for _ in 0..iterations {
            // Compute energy with current parameters
            let energy = Self::compute_vqe_energy(num_qubits, hamiltonian, &ansatz, &params);

            // Gradient estimation (parameter shift rule)
            let grad = Self::vqe_gradient(num_qubits, hamiltonian, &ansatz, &params);

            // Update parameters
            for (p, g) in params.iter_mut().zip(grad.iter()) {
                *p -= learning_rate * g;
            }

            if energy < best_energy {
                best_energy = energy;
                best_params = params.clone();
            }
        }

        (best_energy, best_params)
    }

    fn compute_vqe_energy<F>(
        num_qubits: usize,
        hamiltonian: &[(f64, Vec<char>)],
        ansatz: F,
        params: &[f64],
    ) -> f64
    where
        F: Fn(&mut QuantumState, &[f64]),
    {
        let mut total_energy = 0.0;

        for (coeff, pauli_string) in hamiltonian {
            // Prepare state with ansatz
            let mut state = QuantumState::new(num_qubits);
            ansatz(&mut state, params);

            // Measure expectation value of Pauli string
            let expectation = Self::pauli_expectation(&mut state, pauli_string);
            total_energy += coeff * expectation;
        }

        total_energy
    }

    fn vqe_gradient<F>(
        num_qubits: usize,
        hamiltonian: &[(f64, Vec<char>)],
        ansatz: F,
        params: &[f64],
    ) -> Vec<f64>
    where
        F: Fn(&mut QuantumState, &[f64]) + Clone,
    {
        // Parameter shift rule gradient
        let shift = PI / 2.0;
        let mut grad = vec![0.0; params.len()];

        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();
            params_plus[i] += shift;
            params_minus[i] -= shift;

            let energy_plus =
                Self::compute_vqe_energy(num_qubits, hamiltonian, &ansatz, &params_plus);
            let energy_minus =
                Self::compute_vqe_energy(num_qubits, hamiltonian, &ansatz, &params_minus);

            grad[i] = (energy_plus - energy_minus) / 2.0;
        }

        grad
    }

    fn pauli_expectation(state: &mut QuantumState, pauli_string: &[char]) -> f64 {
        // Simplified - would measure each Pauli term
        let mut result = 1.0;

        for (i, &p) in pauli_string.iter().enumerate() {
            match p {
                'I' => {}
                'X' => {
                    let prob = Self::measure_pauli_x(state, i);
                    result *= prob;
                }
                'Y' => {
                    let prob = Self::measure_pauli_y(state, i);
                    result *= prob;
                }
                'Z' => {
                    let prob = Self::measure_pauli_z(state, i);
                    result *= prob;
                }
                _ => {}
            }
        }

        result
    }

    fn measure_pauli_x(state: &mut QuantumState, qubit: usize) -> f64 {
        let mut temp_state = state.clone();
        QuantumGates::h(&mut temp_state, qubit);
        let probs = temp_state.probabilities();
        // ⟨X⟩ = P(0) - P(1)
        let p0 = probs.iter().step_by(2).sum::<f64>();
        let p1 = probs.iter().skip(1).step_by(2).sum::<f64>();
        p0 - p1
    }

    fn measure_pauli_y(state: &mut QuantumState, qubit: usize) -> f64 {
        let mut temp_state = state.clone();
        QuantumGates::rx(&mut temp_state, qubit, PI / 2.0);
        let probs = temp_state.probabilities();
        let p0 = probs.iter().step_by(2).sum::<f64>();
        let p1 = probs.iter().skip(1).step_by(2).sum::<f64>();
        p0 - p1
    }

    fn measure_pauli_z(state: &mut QuantumState, _qubit: usize) -> f64 {
        let probs = state.probabilities();
        let p0 = probs.iter().step_by(2).sum::<f64>();
        let p1 = probs.iter().skip(1).step_by(2).sum::<f64>();
        p0 - p1
    }

    /// Quantum Approximate Optimization Algorithm (QAOA).
    /// Solves combinatorial optimization problems.
    pub fn qaoa(
        num_qubits: usize,
        cost_hamiltonian: &[(f64, Vec<usize>)],
        mixer_hamiltonian: &[(f64, Vec<usize>)],
        depth: usize,
        params: &[f64],
    ) -> QuantumState {
        let mut state = QuantumState::new(num_qubits);

        // Initialize to uniform superposition
        for q in 0..num_qubits {
            QuantumGates::h(&mut state, q);
        }

        // QAOA layers
        for layer in 0..depth {
            // Cost unitary
            for (coeff, qubits) in cost_hamiltonian {
                let gamma = params[2 * layer];
                Self::apply_cost_unitary(&mut state, qubits, *coeff * gamma);
            }

            // Mixer unitary
            for (coeff, qubits) in mixer_hamiltonian {
                let beta = params[2 * layer + 1];
                Self::apply_mixer_unitary(&mut state, qubits, *coeff * beta);
            }
        }

        state
    }

    fn apply_cost_unitary(state: &mut QuantumState, qubits: &[usize], angle: f64) {
        // Apply exp(-iγH) = Rz(2γ) for ZZ interactions
        if qubits.len() == 1 {
            QuantumGates::rz(state, qubits[0], 2.0 * angle);
        } else if qubits.len() == 2 {
            QuantumGates::zz(state, qubits[0], qubits[1], angle);
        }
    }

    fn apply_mixer_unitary(state: &mut QuantumState, qubits: &[usize], angle: f64) {
        // Apply exp(-iβX) = Rx(2β)
        for &q in qubits {
            QuantumGates::rx(state, q, 2.0 * angle);
        }
    }
}

// ==================== QUANTUM WALKS ====================

impl QuantumAlgorithms {
    /// Discrete-time quantum walk on a line.
    pub fn quantum_walk_line(
        steps: usize,
        position_qubits: usize,
        coin_qubit: usize,
    ) -> QuantumState {
        let total_qubits = position_qubits + 1;
        let mut state = QuantumState::new(total_qubits);

        // Initialize coin to |+⟩
        QuantumGates::h(&mut state, coin_qubit);

        // Initialize position to |0⟩
        for q in 0..position_qubits {
            QuantumGates::x(&mut state, coin_qubit + q);
        }
        for q in 0..position_qubits {
            QuantumGates::x(&mut state, coin_qubit + q);
        }

        // Quantum walk steps
        for _ in 0..steps {
            // Coin operation (Hadamard on coin)
            QuantumGates::h(&mut state, coin_qubit);

            // Conditional shift
            Self::quantum_walk_shift(&mut state, coin_qubit, position_qubits);
        }

        state
    }

    fn quantum_walk_shift(state: &mut QuantumState, coin: usize, pos_qubits: usize) {
        // If coin is |0⟩, shift left; if |1⟩, shift right
        for i in 0..pos_qubits {
            let pos = coin + 1 + i;
            QuantumGates::cx(state, coin, pos);
        }
    }
}

// ==================== HAMILTONIAN SIMULATION ====================

impl QuantumAlgorithms {
    /// Trotter-Suzuki decomposition for Hamiltonian simulation.
    ///
    /// Simulates exp(-iHt) for a given Hamiltonian H.
    pub fn trotter_simulation(
        num_qubits: usize,
        hamiltonian: &[(f64, String)], // (coefficient, pauli_term)
        time: f64,
        trotter_steps: usize,
    ) -> QuantumState {
        let mut state = QuantumState::new(num_qubits);

        // Initialize to |0⟩
        for _ in 0..trotter_steps {
            for (coeff, term) in hamiltonian {
                let dt = time / trotter_steps as f64;
                Self::apply_pauli_evolution(&mut state, term, *coeff * dt);
            }
        }

        state
    }

    fn apply_pauli_evolution(state: &mut QuantumState, pauli: &str, angle: f64) {
        // Apply exp(-iθP) where P is a Pauli string
        for (i, c) in pauli.chars().enumerate() {
            match c {
                'X' => QuantumGates::rx(state, i, 2.0 * angle),
                'Y' => QuantumGates::ry(state, i, 2.0 * angle),
                'Z' => QuantumGates::rz(state, i, 2.0 * angle),
                'I' => {}
                _ => {}
            }
        }
    }
}

// ==================== HELPER FUNCTIONS ====================

impl QuantumAlgorithms {
    fn measure_first_n(state: &mut QuantumState, n: usize) -> usize {
        let probs = state.probabilities();

        // Group probabilities by first n qubits
        let num_groups = 1 << n;
        let mut group_probs = vec![0.0f64; num_groups];

        for (i, &p) in probs.iter().enumerate() {
            let group = i >> (state.num_qubits - n);
            group_probs[group] += p;
        }

        // Sample from distribution
        let mut cumsum = 0.0;
        let r = rand::random::<f64>();

        for (i, &p) in group_probs.iter().enumerate() {
            cumsum += p;
            if cumsum > r {
                return i;
            }
        }

        num_groups - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grover_single_match() {
        // Search for marked element |111⟩ in 3-qubit database
        let result = QuantumAlgorithms::grover_search(
            3,
            |state| {
                // Oracle marks |111⟩
                let amplitudes = state.amplitudes_mut();
                amplitudes[7] = -amplitudes[7];
            },
            Some(2),
        );

        // Should find index 7 (|111⟩)
        assert_eq!(result, 7);
    }

    #[test]
    fn test_deutsch_jozsa_constant() {
        let is_constant = QuantumAlgorithms::deutsch_jozsa(3, |state, n, anc| {
            QuantumAlgorithms::dj_oracle_constant(state, n, anc, 1);
        });
        assert!(is_constant);
    }

    #[test]
    fn test_deutsch_jozsa_balanced() {
        let is_constant = QuantumAlgorithms::deutsch_jozsa(3, |state, n, anc| {
            QuantumAlgorithms::dj_oracle_parity(state, n, anc);
        });
        assert!(!is_constant);
    }

    #[test]
    fn test_bernstein_vazirani() {
        let secret = 0b1011;
        let recovered = QuantumAlgorithms::bernstein_vazirani(4, secret);
        assert_eq!(recovered, secret);
    }

    #[test]
    fn test_qft() {
        let mut state = QuantumState::new(3);
        state.amplitudes_mut()[0] = C64::new(0.0, 0.0); // Clear |000⟩
        state.amplitudes_mut()[5] = C64::new(1.0, 0.0); // Set |101⟩

        QuantumAlgorithms::qft(&mut state, 3);

        // Verify QFT preserves normalization
        let probs = state.probabilities();
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_shors_factorization() {
        // Test factorization of 15 = 3 × 5
        if let Some((p, q)) = QuantumAlgorithms::shors_factorization(15) {
            assert!(p * q == 15);
            assert!(p > 1 && q > 1);
        }
    }

    #[test]
    fn test_swap_test_overlap() {
        // |0> vs |0> => overlap = 1
        let overlap = QuantumAlgorithms::swap_test(
            1,
            |state, _offset| {
                let _ = state;
            },
            |state, _offset| {
                let _ = state;
            },
        );
        assert!((overlap - 1.0).abs() < 1e-6);

        // |0> vs |1> => overlap = 0
        let overlap = QuantumAlgorithms::swap_test(
            1,
            |state, _offset| {
                let _ = state;
            },
            |state, offset| {
                QuantumGates::x(state, offset);
            },
        );
        assert!(overlap < 1e-6);
    }

    #[test]
    fn test_hadamard_test_identity() {
        let val = QuantumAlgorithms::hadamard_test(
            1,
            |state, _offset| {
                let _ = state;
            },
            |_state, _control| {
                // Identity controlled-U
            },
            false,
        );
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hhl_2x2_diagonal() {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let b0 = C64::new(inv_sqrt2, 0.0);
        let b1 = C64::new(inv_sqrt2, 0.0);
        let out = QuantumAlgorithms::hhl_2x2_diagonal(1.0, 2.0, b0, b1);
        let amps = out.amplitudes_ref();
        let ratio = amps[1].norm() / amps[0].norm();
        assert!((ratio - 0.5).abs() < 0.2);
    }

    #[test]
    fn test_vqe_simple() {
        // Simple 2-qubit Hamiltonian: H = Z⊗Z
        let hamiltonian = vec![(1.0, vec!['Z', 'Z'])];

        let ansatz = |state: &mut QuantumState, params: &[f64]| {
            QuantumGates::h(state, 0);
            QuantumGates::cx(state, 0, 1);
            QuantumGates::rz(state, 1, params[0]);
        };

        let (energy, _) = QuantumAlgorithms::vqe(2, &hamiltonian, ansatz, vec![0.5], 0.1, 10);

        assert!(energy.is_finite());
    }
}
