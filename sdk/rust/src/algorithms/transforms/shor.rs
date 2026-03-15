//! Shor's Algorithm & Crypto Attack Toolkit
//!
//! Implements Shor's algorithm for integer factorization via quantum period finding,
//! along with a cryptographic threat assessment toolkit for RSA, Diffie-Hellman,
//! ECC, and AES. Also includes Simon's algorithm for hidden period finding.
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::shor::{factor, ShorConfig};
//!
//! let config = ShorConfig::default();
//! let result = factor(15, &config).unwrap();
//! assert!(result.factors.contains(&3));
//! assert!(result.factors.contains(&5));
//! ```

use std::f64::consts::PI;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during Shor's algorithm execution.
#[derive(Debug, Clone, PartialEq)]
pub enum ShorError {
    /// The input number is not composite (prime or < 4).
    NotComposite,
    /// The input number is too large for simulation.
    TooLarge,
    /// Could not find the period after all trials.
    PeriodNotFound,
    /// Only trivial factors (1 or n) were found.
    TrivialFactor,
}

impl std::fmt::Display for ShorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShorError::NotComposite => write!(f, "Input is not composite (prime or too small)"),
            ShorError::TooLarge => write!(f, "Input is too large for simulation"),
            ShorError::PeriodNotFound => write!(f, "Could not find period after all trials"),
            ShorError::TrivialFactor => write!(f, "Only trivial factors found"),
        }
    }
}

impl std::error::Error for ShorError {}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for Shor's algorithm.
#[derive(Debug, Clone)]
pub struct ShorConfig {
    /// Maximum number of random bases to try before giving up.
    pub num_trials: usize,
    /// Number of precision qubits for quantum period finding.
    /// If `None`, defaults to `2 * ceil(log2(n)) + 1`.
    pub precision_qubits: Option<usize>,
    /// Whether to use the semiclassical QFT variant (sequential measurement).
    pub use_semiclassical: bool,
}

impl Default for ShorConfig {
    fn default() -> Self {
        Self {
            num_trials: 10,
            precision_qubits: None,
            use_semiclassical: false,
        }
    }
}

impl ShorConfig {
    /// Create a new ShorConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of trials.
    pub fn num_trials(mut self, n: usize) -> Self {
        self.num_trials = n;
        self
    }

    /// Set the number of precision qubits.
    pub fn precision_qubits(mut self, q: usize) -> Self {
        self.precision_qubits = Some(q);
        self
    }

    /// Enable or disable semiclassical QFT.
    pub fn use_semiclassical(mut self, flag: bool) -> Self {
        self.use_semiclassical = flag;
        self
    }
}

// ============================================================
// RESULT TYPES
// ============================================================

/// Result of Shor's factoring algorithm.
#[derive(Debug, Clone)]
pub struct ShorResult {
    /// The number that was factored.
    pub n: u64,
    /// The non-trivial factors found.
    pub factors: Vec<u64>,
    /// How many random-base trials were used.
    pub num_trials_used: usize,
    /// The period r that was found (0 if classical shortcut was used).
    pub period_found: u64,
    /// Whether factoring succeeded.
    pub success: bool,
}

/// Resource estimate for a factoring circuit.
#[derive(Debug, Clone)]
pub struct FactoringCircuit {
    /// Total number of qubits required.
    pub num_qubits: usize,
    /// Number of gates for modular exponentiation.
    pub modular_exp_gates: usize,
    /// Number of gates for inverse QFT.
    pub qft_gates: usize,
}

// ============================================================
// CLASSICAL NUMBER THEORY HELPERS
// ============================================================

/// Greatest common divisor via the Euclidean algorithm.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Modular exponentiation: base^exp mod modulus.
/// Uses binary exponentiation with u128 intermediaries to avoid overflow.
pub fn mod_pow(base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u128 = 1;
    let mut base_wide: u128 = (base as u128) % (modulus as u128);
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base_wide) % (modulus as u128);
        }
        exp >>= 1;
        base_wide = (base_wide * base_wide) % (modulus as u128);
    }
    result as u64
}

/// Miller-Rabin primality test. Deterministic for n < 3,317,044,064,679,887,385,961,981.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    // Write n-1 as 2^s * d
    let mut d = n - 1;
    let mut s = 0u32;
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }

    // Witnesses sufficient for u64 range
    let witnesses: &[u64] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    'outer: for &a in witnesses {
        if a >= n {
            continue;
        }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..s - 1 {
            x = mod_pow(x, 2, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Check if n is a perfect power: n = a^b for some a >= 2, b >= 2.
/// Returns Some((base, exponent)) if so, None otherwise.
pub fn is_perfect_power(n: u64) -> Option<(u64, u64)> {
    if n < 4 {
        return None;
    }

    // Check exponents from largest down to 2 so we find the smallest base
    // (e.g., 16 = 2^4, not 4^2)
    let max_exp = (n as f64).log2().ceil() as u64;
    for b in (2..=max_exp).rev() {
        // Binary search for a such that a^b == n
        let mut lo: u64 = 2;
        let mut hi: u64 = (n as f64).powf(1.0 / b as f64).ceil() as u64 + 1;
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            // Compute mid^b carefully to avoid overflow
            let val = checked_pow(mid, b);
            match val {
                Some(v) if v == n => return Some((mid, b)),
                Some(v) if v < n => lo = mid + 1,
                _ => {
                    if mid == 0 {
                        break;
                    }
                    hi = mid - 1;
                }
            }
        }
    }
    None
}

/// Checked integer power that returns None on overflow.
fn checked_pow(base: u64, exp: u64) -> Option<u64> {
    let mut result: u128 = 1;
    let base_wide = base as u128;
    for _ in 0..exp {
        result = result.checked_mul(base_wide)?;
        if result > u64::MAX as u128 {
            return None;
        }
    }
    Some(result as u64)
}

/// Trial division up to a given limit.
/// Returns the smallest factor > 1 if found, None otherwise.
pub fn trial_division(n: u64, limit: u64) -> Option<u64> {
    if n < 2 {
        return None;
    }
    if n % 2 == 0 {
        return Some(2);
    }
    if n % 3 == 0 {
        return Some(3);
    }
    let mut i = 5u64;
    let mut w = 2u64;
    while i <= limit && i * i <= n {
        if n % i == 0 {
            return Some(i);
        }
        i += w;
        w = 6 - w; // alternates between 2 and 4 to skip multiples of 2 and 3
    }
    None
}

/// Pollard's rho factoring algorithm (classical, for comparison).
/// Returns a non-trivial factor if found, None otherwise.
pub fn pollard_rho(n: u64) -> Option<u64> {
    if n % 2 == 0 {
        return Some(2);
    }
    if n < 4 {
        return None;
    }

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..20 {
        let c: u64 = rng.gen_range(1..n);
        let mut x: u64 = rng.gen_range(2..n);
        let mut y = x;
        let mut d = 1u64;

        let f = |val: u64| -> u64 { ((val as u128 * val as u128 + c as u128) % n as u128) as u64 };

        while d == 1 {
            x = f(x);
            y = f(f(y));
            let diff = if x > y { x - y } else { y - x };
            d = gcd(diff, n);
        }

        if d != n {
            return Some(d);
        }
    }
    None
}

// ============================================================
// CONTINUED FRACTIONS
// ============================================================

/// Compute convergents of the continued fraction expansion of numerator/denominator.
/// Returns a vector of (p_k, q_k) convergents where q_k <= max_denominator.
pub fn continued_fraction(
    numerator: u64,
    denominator: u64,
    max_denominator: u64,
) -> Vec<(u64, u64)> {
    if denominator == 0 {
        return vec![];
    }

    let mut convergents = Vec::new();
    let mut a = numerator;
    let mut b = denominator;

    // p_{-1} = 1, p_{-2} = 0; q_{-1} = 0, q_{-2} = 1
    let mut p_prev: u64 = 0;
    let mut p_curr: u64 = 1;
    let mut q_prev: u64 = 1;
    let mut q_curr: u64 = 0;

    loop {
        if b == 0 {
            break;
        }
        let quotient = a / b;
        let remainder = a % b;

        let p_next = quotient
            .checked_mul(p_curr)
            .and_then(|v| v.checked_add(p_prev));
        let q_next = quotient
            .checked_mul(q_curr)
            .and_then(|v| v.checked_add(q_prev));

        match (p_next, q_next) {
            (Some(p), Some(q)) if q <= max_denominator => {
                convergents.push((p, q));
                p_prev = p_curr;
                p_curr = p;
                q_prev = q_curr;
                q_curr = q;
            }
            _ => break,
        }

        a = b;
        b = remainder;
    }

    convergents
}

/// Extract period r from a quantum measurement result.
///
/// Given measurement value `s` from a register of size `2^precision`,
/// the phase is approximately s / 2^precision ≈ j/r for some integer j.
/// We use continued fractions to find the best rational approximation
/// with denominator <= n, then return the denominator as the candidate period.
pub fn extract_period(measurement: u64, precision: u64, n: u64) -> Option<u64> {
    if measurement == 0 {
        return None; // measurement of 0 gives no information
    }

    let convergents = continued_fraction(measurement, precision, n);

    // Try each convergent's denominator as a candidate period
    for &(_p, q) in convergents.iter().rev() {
        if q >= 2 && q < n {
            return Some(q);
        }
    }

    None
}

// ============================================================
// QUANTUM PERIOD FINDING (SIMULATED)
// ============================================================

/// Simulate the quantum modular exponentiation circuit and return
/// the probability distribution over the precision register.
///
/// For each measurement outcome s in [0, 2^m), the amplitude is:
///   A(s) = (1/2^m) * sum_{x=0}^{2^m - 1} exp(2*pi*i*s*x / 2^m) * delta(a^x mod n)
///
/// This computes the probability distribution directly without
/// building the exponentially large state vector.
pub fn modular_exp_simulate(a: u64, n: u64, num_qubits: usize) -> Vec<f64> {
    let precision = 1u64 << num_qubits;
    let precision_f = precision as f64;

    // Find the period classically to determine the structure
    let period = classical_find_period(a, n);

    // The probability of measuring s is:
    // P(s) = |A(s)|^2 where A(s) = (1/sqrt(Q)) * sum_{j=0}^{Q-1} exp(2*pi*i*s*j*r / 2^m)
    // where Q = floor(2^m / r) and r is the period.
    // Peaks occur at s ≈ k * 2^m / r for integer k.

    let mut probabilities = vec![0.0f64; precision as usize];

    if period == 0 {
        // Uniform distribution if period cannot be found
        let uniform = 1.0 / precision_f;
        for p in probabilities.iter_mut() {
            *p = uniform;
        }
        return probabilities;
    }

    let r = period;
    let q = precision / r; // number of complete periods

    for s in 0..precision {
        let mut re = 0.0f64;
        let mut im = 0.0f64;

        for j in 0..q {
            let x = j * r;
            let angle = 2.0 * PI * (s as f64) * (x as f64) / precision_f;
            re += angle.cos();
            im += angle.sin();
        }

        let norm_sq = (re * re + im * im) / (precision_f * precision_f);
        probabilities[s as usize] = norm_sq;
    }

    // Normalize to account for incomplete periods
    let total: f64 = probabilities.iter().sum();
    if total > 0.0 {
        for p in probabilities.iter_mut() {
            *p /= total;
        }
    }

    probabilities
}

/// Classically find the period of a^x mod n (for simulation verification).
fn classical_find_period(a: u64, n: u64) -> u64 {
    let mut val = a % n;
    for r in 1..n {
        if val == 1 {
            return r;
        }
        val = ((val as u128 * a as u128) % n as u128) as u64;
    }
    0 // no period found
}

/// Quantum period finding simulation.
///
/// Simulates the quantum circuit for order finding:
/// 1. Prepare superposition over precision register via Hadamard
/// 2. Apply controlled modular exponentiation
/// 3. Apply inverse QFT to precision register
/// 4. Measure and extract period via continued fractions
///
/// Returns Some(period) if a valid period is found, None otherwise.
pub fn quantum_period_finding(a: u64, n: u64, num_precision_qubits: usize) -> Option<u64> {
    let precision = 1u64 << num_precision_qubits;

    // Simulate the quantum circuit to get measurement probabilities
    let probabilities = modular_exp_simulate(a, n, num_precision_qubits);

    // Sample from the probability distribution
    // In a real quantum computer, we'd measure once; here we can try
    // the most probable outcomes.
    let mut indexed: Vec<(usize, f64)> = probabilities
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Try the top measurement outcomes
    let max_attempts = indexed.len().min(20);
    for &(measurement, prob) in indexed.iter().take(max_attempts) {
        if prob < 1e-10 {
            break;
        }

        if let Some(candidate_r) = extract_period(measurement as u64, precision, n) {
            // Verify the candidate period
            if mod_pow(a, candidate_r, n) == 1 {
                return Some(candidate_r);
            }
            // Try multiples of the candidate
            for mult in 2..=5u64 {
                let r = candidate_r * mult;
                if r < n && mod_pow(a, r, n) == 1 {
                    return Some(r);
                }
            }
        }
    }

    // Fallback: try the classically computed period
    let classical_r = classical_find_period(a, n);
    if classical_r > 0 && mod_pow(a, classical_r, n) == 1 {
        return Some(classical_r);
    }

    None
}

// ============================================================
// SHOR'S ALGORITHM
// ============================================================

/// Factor a composite integer n using Shor's algorithm.
///
/// Steps:
/// 1. Handle trivial cases (even, prime, perfect power)
/// 2. Choose random a coprime to n
/// 3. Use quantum period finding to get the order r of a mod n
/// 4. Use r to extract factors via gcd(a^(r/2) +/- 1, n)
///
/// Returns `ShorResult` on success, `ShorError` on failure.
pub fn factor(n: u64, config: &ShorConfig) -> Result<ShorResult, ShorError> {
    // Input validation
    if n < 4 {
        return Err(ShorError::NotComposite);
    }

    // Simulation limit: avoid astronomically large state vectors.
    // We can classically simulate up to ~40-bit numbers reasonably.
    if n > (1u64 << 40) {
        return Err(ShorError::TooLarge);
    }

    // Step 1: Check if n is even
    if n % 2 == 0 {
        let other = n / 2;
        return Ok(ShorResult {
            n,
            factors: vec![2, other],
            num_trials_used: 0,
            period_found: 0,
            success: true,
        });
    }

    // Check if n is prime
    if is_prime(n) {
        return Err(ShorError::NotComposite);
    }

    // Step 2: Check if n is a perfect power
    if let Some((base, _exp)) = is_perfect_power(n) {
        return Ok(ShorResult {
            n,
            factors: vec![base, n / base],
            num_trials_used: 0,
            period_found: 0,
            success: true,
        });
    }

    // Try small trial division first
    if let Some(small_factor) = trial_division(n, 1000) {
        return Ok(ShorResult {
            n,
            factors: vec![small_factor, n / small_factor],
            num_trials_used: 0,
            period_found: 0,
            success: true,
        });
    }

    // Determine precision qubits
    let n_bits = 64 - n.leading_zeros() as usize;
    let num_precision_qubits = config.precision_qubits.unwrap_or(2 * n_bits + 1);

    // Step 3-6: Main Shor loop
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for trial in 0..config.num_trials {
        // Choose random a in [2, n-1]
        let a: u64 = rng.gen_range(2..n);

        // Check gcd(a, n) — might get lucky
        let g = gcd(a, n);
        if g > 1 && g < n {
            return Ok(ShorResult {
                n,
                factors: vec![g, n / g],
                num_trials_used: trial + 1,
                period_found: 0,
                success: true,
            });
        }

        // Step 4: Quantum period finding
        let r = match quantum_period_finding(a, n, num_precision_qubits) {
            Some(r) => r,
            None => continue,
        };

        // Step 5: r must be even
        if r % 2 != 0 {
            continue;
        }

        // Step 6: Check a^(r/2) != -1 (mod n)
        let half_r = r / 2;
        let a_half = mod_pow(a, half_r, n);
        if a_half == n - 1 {
            continue; // a^(r/2) ≡ -1 (mod n), retry
        }

        // Compute factors
        let factor1 = gcd(a_half + 1, n);
        let factor2 = gcd(if a_half > 0 { a_half - 1 } else { 0 }, n);

        // Check for non-trivial factors
        if factor1 > 1 && factor1 < n {
            let other = n / factor1;
            return Ok(ShorResult {
                n,
                factors: vec![factor1, other],
                num_trials_used: trial + 1,
                period_found: r,
                success: true,
            });
        }
        if factor2 > 1 && factor2 < n {
            let other = n / factor2;
            return Ok(ShorResult {
                n,
                factors: vec![factor2, other],
                num_trials_used: trial + 1,
                period_found: r,
                success: true,
            });
        }
    }

    Err(ShorError::PeriodNotFound)
}

// ============================================================
// FACTORING CIRCUIT RESOURCE ESTIMATION
// ============================================================

/// Estimate the quantum resources needed to factor an n-bit number.
///
/// Uses standard estimates:
/// - Precision qubits: 2n + 1 (for phase estimation)
/// - Work qubits: n (for modular arithmetic)
/// - Total: 3n + 1 qubits
/// - Modular exponentiation gates: O(n^3)
/// - QFT gates: O(n^2)
pub fn estimate_factoring_resources(n_bits: usize) -> FactoringCircuit {
    let precision_qubits = 2 * n_bits + 1;
    let work_qubits = n_bits;
    let total_qubits = precision_qubits + work_qubits; // 3n + 1

    // Modular exponentiation: each of 2n+1 controlled multiplications
    // Each multiplication requires O(n^2) gates
    let modular_exp_gates = precision_qubits * n_bits * n_bits;

    // Inverse QFT: O(m^2) gates where m = precision qubits
    let qft_gates = precision_qubits * (precision_qubits - 1) / 2 + precision_qubits;

    FactoringCircuit {
        num_qubits: total_qubits,
        modular_exp_gates,
        qft_gates,
    }
}

// ============================================================
// CRYPTO ATTACK TOOLKIT
// ============================================================

/// Cryptographic system to assess for quantum vulnerability.
#[derive(Debug, Clone)]
pub enum CryptoTarget {
    /// RSA with the given key size in bits.
    RSA { key_bits: usize },
    /// Diffie-Hellman with the given prime size in bits.
    DiffieHellman { prime_bits: usize },
    /// Elliptic Curve Cryptography with the given curve size in bits.
    ECC { curve_bits: usize },
    /// AES symmetric encryption with the given key size in bits.
    AES { key_bits: usize },
}

/// Threat level classification for quantum attacks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreatLevel {
    /// No practical quantum threat.
    None,
    /// Theoretically possible but far from practical.
    Low,
    /// Feasible with near-term fault-tolerant quantum computers.
    Medium,
    /// Feasible with expected quantum computers within 10-15 years.
    High,
    /// Already broken or breakable with current/near-term quantum hardware.
    Critical,
}

impl std::fmt::Display for ThreatLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThreatLevel::None => write!(f, "None"),
            ThreatLevel::Low => write!(f, "Low"),
            ThreatLevel::Medium => write!(f, "Medium"),
            ThreatLevel::High => write!(f, "High"),
            ThreatLevel::Critical => write!(f, "Critical"),
        }
    }
}

/// Assessment of quantum threat to a specific cryptographic target.
#[derive(Debug, Clone)]
pub struct AttackAssessment {
    /// Description of the target system.
    pub target: String,
    /// Number of logical qubits needed for the attack.
    pub quantum_qubits_needed: usize,
    /// Total number of quantum gates required.
    pub quantum_gates_needed: u64,
    /// Circuit depth (longest path of sequential gates).
    pub quantum_depth: u64,
    /// Estimated wall-clock time in seconds (assuming 1 MHz gate rate).
    pub estimated_time_seconds: f64,
    /// Classical security level in bits (e.g., 128 for AES-128).
    pub classical_difficulty_bits: usize,
    /// Description of the quantum speedup achieved.
    pub quantum_speedup: String,
    /// Overall threat level.
    pub threat_level: ThreatLevel,
}

/// Assess the quantum threat to a cryptographic target.
///
/// Provides resource estimates for mounting a quantum attack using
/// the best known quantum algorithm for each cryptographic primitive:
/// - RSA / DH: Shor's algorithm (exponential speedup)
/// - ECC: Shor's for discrete logarithm on elliptic curves
/// - AES: Grover's algorithm (quadratic speedup)
pub fn assess_quantum_threat(target: &CryptoTarget) -> AttackAssessment {
    match target {
        CryptoTarget::RSA { key_bits } => {
            let n = *key_bits;
            // Shor's: 3n + 1 logical qubits (with optimizations ~2n + O(1))
            let qubits = 3 * n + 1;
            // Gate count: O(n^3) for modular exponentiation
            let gates = (n as u64).pow(3) * 64; // constant factor from circuit synthesis
            let depth = (n as u64).pow(2) * 32; // depth is O(n^2) with parallelism
            let time_s = depth as f64 / 1_000_000.0; // at 1 MHz gate rate
            let classical_bits = n / 2; // approximate classical security level

            let threat = match n {
                0..=512 => ThreatLevel::Critical,
                513..=1024 => ThreatLevel::High,
                1025..=2048 => ThreatLevel::Medium,
                2049..=4096 => ThreatLevel::Low,
                _ => ThreatLevel::None,
            };

            AttackAssessment {
                target: format!("RSA-{}", n),
                quantum_qubits_needed: qubits,
                quantum_gates_needed: gates,
                quantum_depth: depth,
                estimated_time_seconds: time_s,
                classical_difficulty_bits: classical_bits,
                quantum_speedup: "Exponential (Shor's algorithm)".to_string(),
                threat_level: threat,
            }
        }
        CryptoTarget::DiffieHellman { prime_bits } => {
            let n = *prime_bits;
            // Same as RSA — Shor's for discrete log
            let qubits = 3 * n + 1;
            let gates = (n as u64).pow(3) * 64;
            let depth = (n as u64).pow(2) * 32;
            let time_s = depth as f64 / 1_000_000.0;
            let classical_bits = n / 2;

            let threat = match n {
                0..=512 => ThreatLevel::Critical,
                513..=1024 => ThreatLevel::High,
                1025..=2048 => ThreatLevel::Medium,
                2049..=4096 => ThreatLevel::Low,
                _ => ThreatLevel::None,
            };

            AttackAssessment {
                target: format!("DH-{}", n),
                quantum_qubits_needed: qubits,
                quantum_gates_needed: gates,
                quantum_depth: depth,
                estimated_time_seconds: time_s,
                classical_difficulty_bits: classical_bits,
                quantum_speedup: "Exponential (Shor's algorithm for discrete log)".to_string(),
                threat_level: threat,
            }
        }
        CryptoTarget::ECC { curve_bits } => {
            let n = *curve_bits;
            // ECC requires fewer qubits: ~6n for Shor's on elliptic curves
            let qubits = 6 * n + 1;
            // Gate count is O(n^3) but with larger constant
            let gates = (n as u64).pow(3) * 256;
            let depth = (n as u64).pow(2) * 128;
            let time_s = depth as f64 / 1_000_000.0;
            let classical_bits = n; // ECC security ≈ curve_bits

            let threat = match n {
                0..=128 => ThreatLevel::High,
                129..=256 => ThreatLevel::Medium,
                257..=384 => ThreatLevel::Low,
                _ => ThreatLevel::None,
            };

            AttackAssessment {
                target: format!("ECC-{}", n),
                quantum_qubits_needed: qubits,
                quantum_gates_needed: gates,
                quantum_depth: depth,
                estimated_time_seconds: time_s,
                classical_difficulty_bits: classical_bits,
                quantum_speedup: "Exponential (Shor's for elliptic curve discrete log)".to_string(),
                threat_level: threat,
            }
        }
        CryptoTarget::AES { key_bits } => {
            let n = *key_bits;
            // Grover's: quadratic speedup → effective security is n/2 bits
            // Qubits: key_bits for Grover search register + ancilla
            let qubits = n + 1; // search register + flag qubit (simplified)
                                // Grover iterations: O(2^{n/2})
            let effective_bits = n / 2;
            let gates = if effective_bits <= 63 {
                (1u64 << effective_bits) * (n as u64) // each iteration is O(n) gates
            } else {
                u64::MAX // overflow
            };
            let depth = gates / 2; // rough estimate
            let time_s = if depth < u64::MAX / 2 {
                depth as f64 / 1_000_000.0
            } else {
                f64::INFINITY
            };

            let threat = match n {
                0..=64 => ThreatLevel::Critical,
                65..=128 => ThreatLevel::Low, // 128 → 64-bit effective, still hard
                129..=256 => ThreatLevel::None, // 256 → 128-bit effective, secure
                _ => ThreatLevel::None,
            };

            AttackAssessment {
                target: format!("AES-{}", n),
                quantum_qubits_needed: qubits,
                quantum_gates_needed: gates,
                quantum_depth: depth,
                estimated_time_seconds: time_s,
                classical_difficulty_bits: n,
                quantum_speedup: format!(
                    "Quadratic (Grover's algorithm): {}-bit → {}-bit effective",
                    n, effective_bits
                ),
                threat_level: threat,
            }
        }
    }
}

// ============================================================
// SIMON'S ALGORITHM
// ============================================================

/// Simon's algorithm: find hidden period s such that f(x) = f(x XOR s).
///
/// Given a black-box oracle f: {0,1}^n -> {0,1}^n with the promise that
/// there exists s such that f(x) = f(y) iff x = y or x = y XOR s,
/// find s using O(n) quantum queries.
///
/// This is a simulation: we evaluate the oracle classically to find s,
/// but the quantum resource count is O(n) queries vs O(2^{n/2}) classical.
///
/// Returns Some(s) if a non-zero period is found, None if s = 0 (1-to-1 function).
pub fn simons_algorithm(oracle: &dyn Fn(u64) -> u64, n_bits: usize) -> Option<u64> {
    if n_bits == 0 || n_bits > 63 {
        return None;
    }

    let domain_size = 1u64 << n_bits;

    // Simulate the quantum procedure:
    // In real quantum Simon's, we'd:
    //   1. Prepare |0>|0>, apply H^n to first register
    //   2. Apply oracle: |x>|0> -> |x>|f(x)>
    //   3. Measure second register (collapses first to {y, y XOR s})
    //   4. Apply H^n to first register and measure → get z with z·s = 0
    //   5. Repeat n-1 times, solve linear system for s
    //
    // For simulation, we find s by detecting collisions in the oracle output.

    // Build a map from oracle output to first input that produced it
    let mut seen = std::collections::HashMap::new();

    for x in 0..domain_size {
        let fx = oracle(x);
        if let Some(&prev_x) = seen.get(&fx) {
            let candidate_s = prev_x ^ x;
            if candidate_s != 0 {
                // Verify: check that f(0) = f(s) (basic sanity)
                if oracle(0) == oracle(candidate_s) {
                    return Some(candidate_s);
                }
            }
        } else {
            seen.insert(fx, x);
        }
    }

    // No collision found → f is 1-to-1 (s = 0)
    None
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shor_config_defaults() {
        let config = ShorConfig::default();
        assert_eq!(config.num_trials, 10);
        assert!(config.precision_qubits.is_none());
        assert!(!config.use_semiclassical);
    }

    #[test]
    fn test_shor_config_builder() {
        let config = ShorConfig::new()
            .num_trials(20)
            .precision_qubits(12)
            .use_semiclassical(true);
        assert_eq!(config.num_trials, 20);
        assert_eq!(config.precision_qubits, Some(12));
        assert!(config.use_semiclassical);
    }

    #[test]
    fn test_factor_15() {
        let config = ShorConfig::default();
        let result = factor(15, &config).unwrap();
        assert!(result.success);
        assert_eq!(result.n, 15);
        let mut factors = result.factors.clone();
        factors.sort();
        assert_eq!(factors, vec![3, 5]);
    }

    #[test]
    fn test_factor_21() {
        let config = ShorConfig::default();
        let result = factor(21, &config).unwrap();
        assert!(result.success);
        assert_eq!(result.n, 21);
        let product: u64 = result.factors.iter().product();
        assert_eq!(product, 21);
        for &f in &result.factors {
            assert!(f > 1 && f < 21);
        }
    }

    #[test]
    fn test_factor_35() {
        let config = ShorConfig::default();
        let result = factor(35, &config).unwrap();
        assert!(result.success);
        let product: u64 = result.factors.iter().product();
        assert_eq!(product, 35);
        let mut factors = result.factors.clone();
        factors.sort();
        assert_eq!(factors, vec![5, 7]);
    }

    #[test]
    fn test_factor_even() {
        let config = ShorConfig::default();
        let result = factor(22, &config).unwrap();
        assert!(result.success);
        assert!(result.factors.contains(&2));
        assert!(result.factors.contains(&11));
        assert_eq!(result.num_trials_used, 0);
    }

    #[test]
    fn test_factor_prime() {
        let config = ShorConfig::default();
        let err = factor(17, &config).unwrap_err();
        assert_eq!(err, ShorError::NotComposite);
    }

    #[test]
    fn test_factor_perfect_power() {
        let config = ShorConfig::default();
        let result = factor(8, &config).unwrap();
        assert!(result.success);
        let product: u64 = result.factors.iter().product();
        assert_eq!(product, 8);
        assert!(result.factors.contains(&2));
    }

    #[test]
    fn test_mod_pow() {
        // 7^4 mod 15 = 2401 mod 15 = 1
        assert_eq!(mod_pow(7, 4, 15), 1);
        // 2^10 mod 1000 = 1024 mod 1000 = 24
        assert_eq!(mod_pow(2, 10, 1000), 24);
        // a^0 mod n = 1
        assert_eq!(mod_pow(123, 0, 17), 1);
        // Edge: modulus = 1
        assert_eq!(mod_pow(5, 3, 1), 0);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(8, 12), 4);
        assert_eq!(gcd(17, 5), 1);
        assert_eq!(gcd(100, 75), 25);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
    }

    #[test]
    fn test_continued_fraction() {
        // 11/16: CF = [0; 1, 2, 5] → convergents: 0/1, 1/1, 2/3, 11/16
        let convergents = continued_fraction(11, 16, 16);
        assert!(!convergents.is_empty());

        // The final convergent should be 11/16
        let last = convergents.last().unwrap();
        assert_eq!(*last, (11, 16));

        // First convergent: 11/16 = 0 + 1/(1 + 1/(2 + 1/5))
        // Convergent 0: 0/1
        assert_eq!(convergents[0], (0, 1));
    }

    #[test]
    fn test_rsa_2048_assessment() {
        let target = CryptoTarget::RSA { key_bits: 2048 };
        let assessment = assess_quantum_threat(&target);
        // 3 * 2048 + 1 = 6145 qubits
        assert_eq!(assessment.quantum_qubits_needed, 6145);
        assert_eq!(assessment.threat_level, ThreatLevel::Medium);
        assert!(assessment.quantum_gates_needed > 0);
        assert_eq!(assessment.classical_difficulty_bits, 1024);
    }

    #[test]
    fn test_aes_128_assessment() {
        let target = CryptoTarget::AES { key_bits: 128 };
        let assessment = assess_quantum_threat(&target);
        // 128 + 1 = 129 qubits for Grover
        assert_eq!(assessment.quantum_qubits_needed, 129);
        assert_eq!(assessment.classical_difficulty_bits, 128);
        assert_eq!(assessment.threat_level, ThreatLevel::Low);
        assert!(assessment.quantum_speedup.contains("Quadratic"));
        assert!(assessment.quantum_speedup.contains("64-bit effective"));
    }

    #[test]
    fn test_is_prime() {
        // Known primes
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(is_prime(13));
        assert!(is_prime(97));
        assert!(is_prime(7919));
        assert!(is_prime(104729));

        // Known composites
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(!is_prime(4));
        assert!(!is_prime(15));
        assert!(!is_prime(100));
        assert!(!is_prime(561)); // Carmichael number
    }

    #[test]
    fn test_pollard_rho() {
        // Factor a small semiprime
        let n = 91u64; // 7 * 13
        let f = pollard_rho(n);
        assert!(f.is_some());
        let factor = f.unwrap();
        assert!(factor > 1 && factor < n);
        assert_eq!(n % factor, 0);
    }

    #[test]
    fn test_is_perfect_power() {
        assert_eq!(is_perfect_power(8), Some((2, 3))); // 2^3
        assert_eq!(is_perfect_power(27), Some((3, 3))); // 3^3
        assert_eq!(is_perfect_power(16), Some((2, 4))); // 2^4 (or 4^2)
        assert!(is_perfect_power(15).is_none());
        assert!(is_perfect_power(2).is_none());
    }

    #[test]
    fn test_extract_period() {
        // For a = 7, n = 15, period r = 4
        // Measurement near k * 2^m / r should give back r
        let precision: u64 = 1 << 8; // 256
                                     // Peak at s = 64 (= 1 * 256 / 4)
        let period = extract_period(64, precision, 15);
        assert_eq!(period, Some(4));
    }

    #[test]
    fn test_quantum_period_finding() {
        // a = 7, n = 15 → period r = 4 (since 7^4 mod 15 = 1)
        let r = quantum_period_finding(7, 15, 8);
        assert!(r.is_some());
        let period = r.unwrap();
        assert_eq!(mod_pow(7, period, 15), 1);
    }

    #[test]
    fn test_factoring_circuit_resources() {
        let circuit = estimate_factoring_resources(10);
        assert_eq!(circuit.num_qubits, 31); // 3*10 + 1
        assert!(circuit.modular_exp_gates > 0);
        assert!(circuit.qft_gates > 0);
    }

    #[test]
    fn test_simons_algorithm() {
        // f(x) = f(x XOR 3) for 3-bit inputs
        let secret = 3u64;
        let oracle = |x: u64| -> u64 {
            // Ensure f(x) = f(x ^ secret) by using min(x, x^s) as canonical
            let y = x ^ secret;
            std::cmp::min(x, y)
        };
        let result = simons_algorithm(&oracle, 3);
        assert_eq!(result, Some(secret));
    }

    #[test]
    fn test_simons_one_to_one() {
        // Identity function — s = 0, should return None
        let oracle = |x: u64| -> u64 { x };
        let result = simons_algorithm(&oracle, 4);
        assert!(result.is_none());
    }

    #[test]
    fn test_modular_exp_simulate() {
        // a = 7, n = 15, period = 4
        // Peaks should appear at multiples of 2^m / 4
        let probs = modular_exp_simulate(7, 15, 8);
        assert_eq!(probs.len(), 256);

        // Sum should be ~1.0
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);

        // Peaks at 0, 64, 128, 192 (= k * 256/4)
        let peak_indices = vec![0, 64, 128, 192];
        for &idx in &peak_indices {
            assert!(probs[idx] > 0.01, "Expected peak at index {}", idx);
        }
    }

    #[test]
    fn test_ecc_assessment() {
        let target = CryptoTarget::ECC { curve_bits: 256 };
        let assessment = assess_quantum_threat(&target);
        assert_eq!(assessment.quantum_qubits_needed, 6 * 256 + 1);
        assert_eq!(assessment.threat_level, ThreatLevel::Medium);
    }

    #[test]
    fn test_dh_assessment() {
        let target = CryptoTarget::DiffieHellman { prime_bits: 2048 };
        let assessment = assess_quantum_threat(&target);
        assert_eq!(assessment.quantum_qubits_needed, 6145);
        assert_eq!(assessment.threat_level, ThreatLevel::Medium);
    }

    #[test]
    fn test_trial_division() {
        assert_eq!(trial_division(15, 100), Some(3));
        assert_eq!(trial_division(77, 100), Some(7));
        assert_eq!(trial_division(97, 100), None); // prime
        assert_eq!(trial_division(4, 100), Some(2));
    }

    #[test]
    fn test_threat_level_display() {
        assert_eq!(format!("{}", ThreatLevel::Critical), "Critical");
        assert_eq!(format!("{}", ThreatLevel::None), "None");
    }

    #[test]
    fn test_shor_error_display() {
        assert!(format!("{}", ShorError::NotComposite).contains("not composite"));
        assert!(format!("{}", ShorError::TooLarge).contains("too large"));
    }

    #[test]
    fn test_factor_small_composites() {
        let config = ShorConfig::default();
        // Test several small semiprimes
        for &n in &[6u64, 10, 14, 15, 21, 33, 35, 39, 51, 55, 77, 91] {
            let result = factor(n, &config).unwrap();
            assert!(result.success, "Failed to factor {}", n);
            let product: u64 = result.factors.iter().product();
            assert_eq!(product, n, "Factors of {} don't multiply back", n);
        }
    }
}
