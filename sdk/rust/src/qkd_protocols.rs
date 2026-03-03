//! Quantum Key Distribution (QKD) Protocol Simulation
//!
//! Implements the major QKD protocols used in quantum-secure communication:
//! - **BB84**: Bennett-Brassard 1984, the foundational prepare-and-measure protocol
//! - **E91**: Ekert 1991, entanglement-based with Bell/CHSH eavesdropper detection
//! - **BBM92**: Bennett-Brassard-Mermin 1992, simplified entanglement-based
//! - **B92**: Bennett 1992, two-state protocol with lower key rate
//! - **Six-State**: Three-basis extension of BB84 with higher QBER tolerance
//!
//! Each protocol includes channel simulation, eavesdropper models, error correction
//! (simplified CASCADE), and privacy amplification.

use rand::Rng;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during QKD protocol execution.
#[derive(Debug, Clone, PartialEq)]
pub enum QkdError {
    /// The sifted key is too short for meaningful security analysis.
    KeyTooShort,
    /// QBER exceeds the protocol's security threshold.
    QberTooHigh { qber: f64, threshold: f64 },
    /// Classical authentication between Alice and Bob failed.
    AuthenticationFailed,
    /// Channel loss is too high to establish a key.
    ChannelLoss { transmission: f64 },
}

impl std::fmt::Display for QkdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QkdError::KeyTooShort => write!(f, "Sifted key too short for security analysis"),
            QkdError::QberTooHigh { qber, threshold } => {
                write!(f, "QBER {:.4} exceeds threshold {:.4}", qber, threshold)
            }
            QkdError::AuthenticationFailed => write!(f, "Classical authentication failed"),
            QkdError::ChannelLoss { transmission } => {
                write!(f, "Channel transmission too low: {:.6}", transmission)
            }
        }
    }
}

impl std::error::Error for QkdError {}

// ============================================================
// EAVESDROPPER MODELS
// ============================================================

/// Model of eavesdropper (Eve) behavior on the quantum channel.
#[derive(Debug, Clone)]
pub enum EavesdropperModel {
    /// No eavesdropper present.
    None,
    /// Intercept-resend attack: Eve intercepts a fraction of qubits,
    /// measures in a random basis, and resends.
    InterceptResend { fraction: f64 },
    /// Collective attack: Eve entangles a probe with each qubit,
    /// contributing a fixed fraction of information leakage.
    Collective { info_fraction: f64 },
    /// Custom eavesdropper with a specified QBER contribution.
    Custom { qber_contribution: f64 },
}

// ============================================================
// PROTOCOL ENUM
// ============================================================

/// Supported QKD protocols.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkdProtocol {
    /// BB84: 4-state, 2-basis prepare-and-measure.
    BB84,
    /// E91: Entanglement-based with CHSH test.
    E91,
    /// BBM92: Simplified entanglement-based (QBER security).
    BBM92,
    /// B92: Two-state protocol.
    B92,
    /// Six-State: 6-state, 3-basis extension of BB84.
    SixState,
}

impl QkdProtocol {
    /// Security threshold for QBER above which the protocol aborts.
    pub fn qber_threshold(&self) -> f64 {
        match self {
            QkdProtocol::BB84 => 0.11,
            QkdProtocol::E91 => 0.11,
            QkdProtocol::BBM92 => 0.11,
            QkdProtocol::B92 => 0.085,
            QkdProtocol::SixState => 0.126,
        }
    }

    /// Expected sifting rate for the protocol.
    pub fn sifting_rate(&self) -> f64 {
        match self {
            QkdProtocol::BB84 => 0.5,
            QkdProtocol::E91 => 0.25,  // 1/4 of basis combos match for key
            QkdProtocol::BBM92 => 0.5,
            QkdProtocol::B92 => 0.25,  // conclusive measurement rate
            QkdProtocol::SixState => 1.0 / 3.0,
        }
    }
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for a QKD simulation run.
#[derive(Debug, Clone)]
pub struct QkdConfig {
    /// Number of raw bits (qubits) Alice sends.
    pub num_bits: usize,
    /// Base channel error rate (independent of eavesdropping).
    pub error_rate: f64,
    /// Eavesdropper model.
    pub eavesdropper: EavesdropperModel,
    /// Channel loss in dB/km.
    pub channel_loss_db: f64,
    /// Distance between Alice and Bob in km.
    pub distance_km: f64,
    /// Detector efficiency (0.0 to 1.0).
    pub detector_efficiency: f64,
    /// Dark count rate per detector per gate (typically ~1e-6).
    pub dark_count_rate: f64,
    /// Fraction of sifted key sacrificed for QBER estimation.
    pub qber_estimation_fraction: f64,
}

impl Default for QkdConfig {
    fn default() -> Self {
        Self {
            num_bits: 1000,
            error_rate: 0.05,
            eavesdropper: EavesdropperModel::None,
            channel_loss_db: 0.2,
            distance_km: 100.0,
            detector_efficiency: 0.9,
            dark_count_rate: 1e-6,
            qber_estimation_fraction: 0.1,
        }
    }
}

impl QkdConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_bits(mut self, n: usize) -> Self {
        self.num_bits = n;
        self
    }

    pub fn error_rate(mut self, e: f64) -> Self {
        self.error_rate = e;
        self
    }

    pub fn eavesdropper(mut self, model: EavesdropperModel) -> Self {
        self.eavesdropper = model;
        self
    }

    pub fn channel_loss_db(mut self, loss: f64) -> Self {
        self.channel_loss_db = loss;
        self
    }

    pub fn distance_km(mut self, d: f64) -> Self {
        self.distance_km = d;
        self
    }

    pub fn detector_efficiency(mut self, eff: f64) -> Self {
        self.detector_efficiency = eff;
        self
    }

    pub fn dark_count_rate(mut self, rate: f64) -> Self {
        self.dark_count_rate = rate;
        self
    }

    pub fn qber_estimation_fraction(mut self, frac: f64) -> Self {
        self.qber_estimation_fraction = frac;
        self
    }
}

// ============================================================
// RESULT
// ============================================================

/// Result of a QKD protocol run.
#[derive(Debug, Clone)]
pub struct QkdResult {
    /// Alice's raw key (before sifting).
    pub raw_key_alice: Vec<u8>,
    /// Bob's raw key (before sifting).
    pub raw_key_bob: Vec<u8>,
    /// Sifted key (matching bases, before error correction).
    pub sifted_key: Vec<u8>,
    /// Final key after error correction and privacy amplification.
    pub final_key: Vec<u8>,
    /// Quantum Bit Error Rate estimated from sacrificed bits.
    pub qber: f64,
    /// Asymptotic secure key rate (bits per channel use).
    pub key_rate: f64,
    /// Whether the key exchange is considered secure.
    pub secure: bool,
    /// Number of raw bits sent by Alice.
    pub num_bits_sent: usize,
    /// Number of bits after sifting.
    pub num_bits_sifted: usize,
    /// Number of bits in the final key.
    pub num_bits_final: usize,
    /// Whether an eavesdropper was detected (QBER above threshold or CHSH violation).
    pub eavesdropper_detected: bool,
    /// CHSH S-value (only for E91; 0.0 otherwise).
    pub chsh_s_value: f64,
}

// ============================================================
// INFORMATION-THEORETIC UTILITIES
// ============================================================

/// Binary entropy function h(p) = -p*log2(p) - (1-p)*log2(1-p).
///
/// Returns 0.0 for p=0 or p=1 (by convention / limit).
pub fn binary_entropy(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

/// Asymptotic secure key rate for a given protocol and observed QBER.
///
/// - BB84/BBM92/SixState: r = 1 - 2*h(QBER)
/// - E91: same formula (entanglement-based equivalent to BB84)
/// - B92: r = 1 - h(QBER) (two-state, lower information leakage)
///
/// Returns 0 if QBER is above the protocol's security threshold.
pub fn asymptotic_key_rate(protocol: &QkdProtocol, qber: f64) -> f64 {
    if qber >= protocol.qber_threshold() || qber >= 0.5 {
        return 0.0;
    }
    let h = binary_entropy(qber);
    let rate = match protocol {
        QkdProtocol::B92 => 1.0 - h,
        // Six-state: tighter Eve bound from 3rd basis → r = 1 - (5/3)*h(QBER)
        QkdProtocol::SixState => 1.0 - (5.0 / 3.0) * h,
        _ => 1.0 - 2.0 * h,
    };
    rate.max(0.0)
}

/// Finite-key secure key rate with statistical corrections.
///
/// Accounts for finite sample size and desired security parameter epsilon.
pub fn finite_key_rate(
    protocol: &QkdProtocol,
    qber: f64,
    num_bits: usize,
    epsilon_sec: f64,
) -> f64 {
    let r_inf = asymptotic_key_rate(protocol, qber);
    if r_inf <= 0.0 || num_bits == 0 {
        return 0.0;
    }
    // Finite-size penalty: O(sqrt(log(1/eps)/n))
    let n = num_bits as f64;
    let penalty = ((epsilon_sec.recip().ln()) / n).sqrt() * 6.0;
    (r_inf - penalty).max(0.0)
}

// ============================================================
// CHANNEL SIMULATION
// ============================================================

/// Compute channel transmission probability from fiber parameters.
///
/// η = 10^(-loss_dB_per_km * distance_km / 10)
pub fn channel_transmission(distance_km: f64, loss_db_per_km: f64) -> f64 {
    let total_loss_db = loss_db_per_km * distance_km;
    10.0_f64.powf(-total_loss_db / 10.0)
}

/// Probability of a detector click given channel transmission.
///
/// P_click = η_channel * η_detector + p_dark (approximately, for small dark counts)
pub fn detector_click_probability(
    transmission: f64,
    detector_efficiency: f64,
    dark_count_rate: f64,
) -> f64 {
    let signal = transmission * detector_efficiency;
    // P(click) = 1 - (1-signal)(1-dark)  ≈ signal + dark for small rates
    1.0 - (1.0 - signal) * (1.0 - dark_count_rate)
}

/// Simulate transmission of a single qubit through a noisy channel.
///
/// Returns (received_bit, was_transmitted_successfully).
/// `basis`: 0 = Z, 1 = X, 2 = Y (for six-state)
/// `bit`: 0 or 1
/// `channel_error`: probability of a bit flip
pub fn simulate_qubit_transmission(
    _basis: u8,
    bit: u8,
    channel_error: f64,
    rng: &mut impl Rng,
) -> (u8, bool) {
    // Determine if the qubit survives transmission
    let transmitted = true; // simplified: always transmitted (loss handled elsewhere)
    // Apply channel noise
    let received = if rng.gen::<f64>() < channel_error {
        1 - bit
    } else {
        bit
    };
    (received, transmitted)
}

// ============================================================
// ERROR CORRECTION (Simplified CASCADE)
// ============================================================

/// Simplified CASCADE error correction.
///
/// Divides keys into blocks, computes parity, and bisects to find errors.
/// Returns (corrected_bob_key, num_bits_leaked_to_eve).
pub fn cascade_error_correct(
    alice_key: &[u8],
    bob_key: &[u8],
    _rng: &mut impl Rng,
) -> (Vec<u8>, usize) {
    assert_eq!(alice_key.len(), bob_key.len());
    let n = alice_key.len();
    if n == 0 {
        return (vec![], 0);
    }

    let mut corrected = bob_key.to_vec();
    let mut bits_leaked: usize = 0;

    // CASCADE pass with progressively larger blocks
    for pass in 0..4 {
        let block_size = (4 << pass).min(n);
        let mut offset = 0;
        while offset < n {
            let end = (offset + block_size).min(n);
            // Compute parities
            let alice_parity: u8 = alice_key[offset..end].iter().fold(0u8, |a, &b| a ^ b);
            let bob_parity: u8 = corrected[offset..end].iter().fold(0u8, |a, &b| a ^ b);
            bits_leaked += 1; // parity bit revealed

            if alice_parity != bob_parity {
                // Binary search for the error within this block
                bits_leaked += bisect_correct(alice_key, &mut corrected, offset, end, &mut bits_leaked);
            }
            offset = end;
        }
    }

    (corrected, bits_leaked)
}

/// Binary-search within a block to locate and correct a single error.
/// Returns additional bits leaked (beyond the outer parity already counted).
fn bisect_correct(
    alice: &[u8],
    bob: &mut [u8],
    lo: usize,
    hi: usize,
    _leaked: &mut usize,
) -> usize {
    if hi - lo <= 1 {
        // Found the error position — flip Bob's bit
        bob[lo] = alice[lo];
        return 0;
    }
    let mid = (lo + hi) / 2;

    let alice_left_parity: u8 = alice[lo..mid].iter().fold(0u8, |a, &b| a ^ b);
    let bob_left_parity: u8 = bob[lo..mid].iter().fold(0u8, |a, &b| a ^ b);
    let extra_leaked = 1; // one more parity bit revealed

    if alice_left_parity != bob_left_parity {
        extra_leaked + bisect_correct(alice, bob, lo, mid, _leaked)
    } else {
        extra_leaked + bisect_correct(alice, bob, mid, hi, _leaked)
    }
}

// ============================================================
// PRIVACY AMPLIFICATION
// ============================================================

/// Privacy amplification using a simple XOR-based universal hash.
///
/// Reduces the key length by `num_bits_leaked` to eliminate Eve's information.
/// Uses a Toeplitz-style XOR hash for simulation purposes.
pub fn privacy_amplify(key: &[u8], num_bits_leaked: usize) -> Vec<u8> {
    if key.is_empty() || num_bits_leaked >= key.len() {
        return vec![];
    }
    let final_len = key.len() - num_bits_leaked;
    if final_len == 0 {
        return vec![];
    }
    // Simple XOR-fold hash: partition key into chunks and XOR-accumulate
    // This is a simulation-grade privacy amplification, not cryptographic
    let mut result = Vec::with_capacity(final_len);
    for i in 0..final_len {
        let mut bit = key[i];
        // XOR with bits at shifted positions (Toeplitz-like)
        if i + final_len < key.len() {
            bit ^= key[i + final_len];
        }
        result.push(bit);
    }
    result
}

// ============================================================
// EAVESDROPPER SIMULATION
// ============================================================

/// Apply eavesdropper to a qubit in transit.
/// Returns the (possibly disturbed) bit and whether Eve acted on this qubit.
fn apply_eavesdropper(
    bit: u8,
    alice_basis: u8,
    eavesdropper: &EavesdropperModel,
    rng: &mut impl Rng,
) -> (u8, bool) {
    match eavesdropper {
        EavesdropperModel::None => (bit, false),
        EavesdropperModel::InterceptResend { fraction } => {
            if rng.gen::<f64>() < *fraction {
                // Eve measures in a random basis
                let eve_basis: u8 = rng.gen_range(0..2);
                if eve_basis != alice_basis {
                    // Wrong basis: 50% chance of flipping the bit
                    let disturbed = if rng.gen::<f64>() < 0.5 { 1 - bit } else { bit };
                    (disturbed, true)
                } else {
                    // Right basis: Eve learns the bit, resends correctly
                    (bit, true)
                }
            } else {
                (bit, false)
            }
        }
        EavesdropperModel::Collective { info_fraction } => {
            // Collective attack: small disturbance proportional to info extracted
            let disturbance = info_fraction * 0.5; // theoretical coupling
            if rng.gen::<f64>() < disturbance {
                (1 - bit, true)
            } else {
                (bit, false)
            }
        }
        EavesdropperModel::Custom { qber_contribution } => {
            if rng.gen::<f64>() < *qber_contribution {
                (1 - bit, true)
            } else {
                (bit, false)
            }
        }
    }
}

// ============================================================
// BB84 PROTOCOL
// ============================================================

/// Run the BB84 (Bennett-Brassard 1984) protocol.
///
/// The foundational prepare-and-measure QKD protocol using 4 states in 2 bases:
/// - Z basis: |0>, |1>
/// - X basis: |+>, |->
pub fn run_bb84(config: &QkdConfig) -> Result<QkdResult, QkdError> {
    let mut rng = rand::thread_rng();
    let n = config.num_bits;

    // Step 1: Alice prepares random bits and bases
    let alice_bits: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
    let alice_bases: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect(); // 0=Z, 1=X

    // Step 2-3: Transmission with optional eavesdropping + channel noise
    let mut transmitted_bits = Vec::with_capacity(n);
    for i in 0..n {
        let (bit_after_eve, _) =
            apply_eavesdropper(alice_bits[i], alice_bases[i], &config.eavesdropper, &mut rng);
        let (bit_after_channel, _) =
            simulate_qubit_transmission(alice_bases[i], bit_after_eve, config.error_rate, &mut rng);
        transmitted_bits.push(bit_after_channel);
    }

    // Step 4: Bob measures in random bases
    let bob_bases: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
    let bob_bits: Vec<u8> = (0..n)
        .map(|i| {
            if bob_bases[i] == alice_bases[i] {
                // Matching basis: Bob gets the (possibly noisy) bit
                transmitted_bits[i]
            } else {
                // Wrong basis: random outcome
                rng.gen_range(0..2)
            }
        })
        .collect();

    // Step 5: Sifting — keep only positions where bases match
    let mut sifted_alice = Vec::new();
    let mut sifted_bob = Vec::new();
    for i in 0..n {
        if alice_bases[i] == bob_bases[i] {
            sifted_alice.push(alice_bits[i]);
            sifted_bob.push(bob_bits[i]);
        }
    }

    if sifted_alice.len() < 20 {
        return Err(QkdError::KeyTooShort);
    }

    // Step 6: QBER estimation (sacrifice a fraction)
    let num_test = (sifted_alice.len() as f64 * config.qber_estimation_fraction).ceil() as usize;
    let num_test = num_test.max(1).min(sifted_alice.len() / 2);
    let mut errors = 0usize;
    for i in 0..num_test {
        if sifted_alice[i] != sifted_bob[i] {
            errors += 1;
        }
    }
    let qber = errors as f64 / num_test as f64;

    // Remove test bits from sifted key
    let key_alice: Vec<u8> = sifted_alice[num_test..].to_vec();
    let key_bob: Vec<u8> = sifted_bob[num_test..].to_vec();

    // Step 7: Security check
    let threshold = QkdProtocol::BB84.qber_threshold();
    let eavesdropper_detected = qber > threshold;
    let secure;

    if eavesdropper_detected {
        // Protocol aborts but we still return the result for analysis
        secure = false;
    } else {
        secure = true;
    }

    // Step 8: Error correction
    let (corrected_bob, bits_leaked) = cascade_error_correct(&key_alice, &key_bob, &mut rng);

    // Step 9: Privacy amplification
    let final_key = if secure {
        privacy_amplify(&corrected_bob, bits_leaked)
    } else {
        vec![] // No key produced if insecure
    };

    let key_rate = asymptotic_key_rate(&QkdProtocol::BB84, qber);
    let num_bits_sifted = key_alice.len();
    let num_bits_final = final_key.len();

    Ok(QkdResult {
        raw_key_alice: alice_bits,
        raw_key_bob: bob_bits,
        sifted_key: sifted_alice,
        final_key,
        qber,
        key_rate,
        secure,
        num_bits_sent: n,
        num_bits_sifted,
        num_bits_final,
        eavesdropper_detected,
        chsh_s_value: 0.0,
    })
}

// ============================================================
// ============================================================
// E91 PROTOCOL
// ============================================================

/// Run the E91 (Ekert 1991) entanglement-based protocol.
///
/// Uses entangled Bell pairs |Phi+> = (|00> + |11>)/sqrt(2) and the CHSH inequality
/// to detect eavesdropping.
///
/// Alice's measurement angles: {0, pi/8, pi/4}  (basis indices 0, 1, 2)
/// Bob's measurement angles:   {pi/8, pi/4, 3*pi/8} (basis indices 0, 1, 2)
///
/// Key bits from matching-angle pairs: (Alice 1, Bob 0) and (Alice 2, Bob 1).
/// CHSH test from pairs: (A0,B0), (A0,B2), (A2,B0), (A2,B2).
pub fn run_e91(config: &QkdConfig) -> Result<QkdResult, QkdError> {
    let mut rng = rand::thread_rng();
    let n = config.num_bits;

    // E91 measurement angles (radians)
    let alice_angles: [f64; 3] = [
        0.0,                                      // a1 = 0
        std::f64::consts::FRAC_PI_8,              // a2 = pi/8
        std::f64::consts::FRAC_PI_4,              // a3 = pi/4
    ];
    let bob_angles: [f64; 3] = [
        std::f64::consts::FRAC_PI_8,              // b1 = pi/8
        std::f64::consts::FRAC_PI_4,              // b2 = pi/4
        3.0 * std::f64::consts::FRAC_PI_8,        // b3 = 3*pi/8
    ];

    // Alice and Bob each choose one of 3 bases
    let alice_bases: Vec<u8> = (0..n).map(|_| rng.gen_range(0..3u8)).collect();
    let bob_bases: Vec<u8> = (0..n).map(|_| rng.gen_range(0..3u8)).collect();

    let mut alice_bits = Vec::with_capacity(n);
    let mut bob_bits = Vec::with_capacity(n);

    // CHSH data: (alice_basis_idx, bob_basis_idx, alice_bit, bob_bit)
    let mut chsh_data: Vec<(u8, u8, u8, u8)> = Vec::new();

    for i in 0..n {
        let a_idx = alice_bases[i] as usize;
        let b_idx = bob_bases[i] as usize;
        let theta_a = alice_angles[a_idx];
        let theta_b = bob_angles[b_idx];
        let angle_diff = theta_a - theta_b;

        // Entangled pair |Phi+>: Alice's outcome is random
        let a_bit: u8 = rng.gen_range(0..2);

        // Quantum correlation: P(same) = cos^2(theta_a - theta_b)
        let p_same_base = angle_diff.cos().powi(2);

        // Apply eavesdropper disturbance (reduces quantum correlation)
        let p_same_eve = match &config.eavesdropper {
            EavesdropperModel::None => p_same_base,
            EavesdropperModel::InterceptResend { fraction } => {
                // Intercept-resend destroys entanglement for intercepted pairs
                (1.0 - fraction) * p_same_base + fraction * 0.5
            }
            EavesdropperModel::Collective { info_fraction } => {
                let noise = info_fraction * 0.5;
                (1.0 - noise) * p_same_base + noise * 0.5
            }
            EavesdropperModel::Custom { qber_contribution } => {
                (1.0 - qber_contribution) * p_same_base + qber_contribution * 0.5
            }
        };

        // Apply channel noise
        let p_same = (1.0 - config.error_rate) * p_same_eve
            + config.error_rate * (1.0 - p_same_eve);

        let b_bit = if rng.gen::<f64>() < p_same {
            a_bit       // same outcome
        } else {
            1 - a_bit   // opposite outcome
        };

        alice_bits.push(a_bit);
        bob_bits.push(b_bit);
        chsh_data.push((alice_bases[i], bob_bases[i], a_bit, b_bit));
    }

    // CHSH test: S = |E(A0,B0) - E(A0,B2) + E(A2,B0) + E(A2,B2)|
    let chsh_s = compute_chsh_value_e91(&chsh_data);

    // Sifting: keep pairs where measurement angles match
    // Matching pairs: (Alice 1, Bob 0) both at pi/8, and (Alice 2, Bob 1) both at pi/4
    let mut sifted_alice = Vec::new();
    let mut sifted_bob = Vec::new();
    for i in 0..n {
        let a = alice_bases[i];
        let b = bob_bases[i];
        if (a == 1 && b == 0) || (a == 2 && b == 1) {
            sifted_alice.push(alice_bits[i]);
            sifted_bob.push(bob_bits[i]);
        }
    }

    if sifted_alice.len() < 20 {
        return Err(QkdError::KeyTooShort);
    }

    // QBER estimation
    let num_test = (sifted_alice.len() as f64 * config.qber_estimation_fraction).ceil() as usize;
    let num_test = num_test.max(1).min(sifted_alice.len() / 2);
    let mut errors = 0usize;
    for i in 0..num_test {
        if sifted_alice[i] != sifted_bob[i] {
            errors += 1;
        }
    }
    let qber = errors as f64 / num_test as f64;

    let key_alice: Vec<u8> = sifted_alice[num_test..].to_vec();
    let key_bob: Vec<u8> = sifted_bob[num_test..].to_vec();

    // Security: CHSH S > 2 means genuine quantum correlations (no eavesdropper)
    // AND QBER below threshold
    let threshold = QkdProtocol::E91.qber_threshold();
    let chsh_secure = chsh_s > 2.0;
    let qber_secure = qber <= threshold;
    let secure = chsh_secure && qber_secure;
    let eavesdropper_detected = !chsh_secure || !qber_secure;

    // Error correction and privacy amplification
    let (corrected_bob, bits_leaked) = cascade_error_correct(&key_alice, &key_bob, &mut rng);
    let final_key = if secure {
        privacy_amplify(&corrected_bob, bits_leaked)
    } else {
        vec![]
    };

    let key_rate = asymptotic_key_rate(&QkdProtocol::E91, qber);
    let num_bits_sifted = key_alice.len();
    let num_bits_final = final_key.len();

    Ok(QkdResult {
        raw_key_alice: alice_bits,
        raw_key_bob: bob_bits,
        sifted_key: sifted_alice,
        final_key,
        qber,
        key_rate,
        secure,
        num_bits_sent: n,
        num_bits_sifted,
        num_bits_final,
        eavesdropper_detected,
        chsh_s_value: chsh_s,
    })
}

/// Compute the CHSH S-value from E91 measurement data.
///
/// Uses basis indices: Alice {0, 2}, Bob {0, 2} for CHSH test.
/// S = |E(A0,B0) - E(A0,B2) + E(A2,B0) + E(A2,B2)|
/// For genuine entanglement: S ~ 2*sqrt(2) ~ 2.828
fn compute_chsh_value_e91(data: &[(u8, u8, u8, u8)]) -> f64 {
    // Compute E(ai, bj) = (N_same - N_diff) / (N_same + N_diff)
    fn correlator(data: &[(u8, u8, u8, u8)], a_basis: u8, b_basis: u8) -> f64 {
        let mut n_same = 0i64;
        let mut n_diff = 0i64;
        for &(ab, bb, a_bit, b_bit) in data {
            if ab == a_basis && bb == b_basis {
                if a_bit == b_bit {
                    n_same += 1;
                } else {
                    n_diff += 1;
                }
            }
        }
        let total = n_same + n_diff;
        if total == 0 {
            return 0.0;
        }
        (n_same - n_diff) as f64 / total as f64
    }

    // CHSH test pairs: (A0,B0), (A0,B2), (A2,B0), (A2,B2)
    let e00 = correlator(data, 0, 0);
    let e02 = correlator(data, 0, 2);
    let e20 = correlator(data, 2, 0);
    let e22 = correlator(data, 2, 2);

    // S = |E(A0,B0) - E(A0,B2) + E(A2,B0) + E(A2,B2)|
    (e00 - e02 + e20 + e22).abs()
}

// BBM92 PROTOCOL
// ============================================================

/// Run the BBM92 (Bennett-Brassard-Mermin 1992) protocol.
///
/// Simplified entanglement-based protocol. Both parties measure in Z or X basis.
/// Uses QBER (not CHSH) for security.
pub fn run_bbm92(config: &QkdConfig) -> Result<QkdResult, QkdError> {
    let mut rng = rand::thread_rng();
    let n = config.num_bits;

    // Both choose Z (0) or X (1) basis
    let alice_bases: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
    let bob_bases: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();

    let mut alice_bits = Vec::with_capacity(n);
    let mut bob_bits = Vec::with_capacity(n);

    for i in 0..n {
        // Entangled pair |Φ+>: same basis → perfect correlation
        let a_bit: u8 = rng.gen_range(0..2);

        let b_bit = if alice_bases[i] == bob_bases[i] {
            let mut b = a_bit;
            let (disturbed, _) =
                apply_eavesdropper(b, alice_bases[i], &config.eavesdropper, &mut rng);
            b = disturbed;
            if rng.gen::<f64>() < config.error_rate {
                1 - b
            } else {
                b
            }
        } else {
            rng.gen_range(0..2) // uncorrelated
        };

        alice_bits.push(a_bit);
        bob_bits.push(b_bit);
    }

    // Sifting
    let mut sifted_alice = Vec::new();
    let mut sifted_bob = Vec::new();
    for i in 0..n {
        if alice_bases[i] == bob_bases[i] {
            sifted_alice.push(alice_bits[i]);
            sifted_bob.push(bob_bits[i]);
        }
    }

    if sifted_alice.len() < 20 {
        return Err(QkdError::KeyTooShort);
    }

    // QBER estimation
    let num_test = (sifted_alice.len() as f64 * config.qber_estimation_fraction).ceil() as usize;
    let num_test = num_test.max(1).min(sifted_alice.len() / 2);
    let mut errors = 0usize;
    for i in 0..num_test {
        if sifted_alice[i] != sifted_bob[i] {
            errors += 1;
        }
    }
    let qber = errors as f64 / num_test as f64;

    let key_alice: Vec<u8> = sifted_alice[num_test..].to_vec();
    let key_bob: Vec<u8> = sifted_bob[num_test..].to_vec();

    let threshold = QkdProtocol::BBM92.qber_threshold();
    let eavesdropper_detected = qber > threshold;
    let secure = !eavesdropper_detected;

    let (corrected_bob, bits_leaked) = cascade_error_correct(&key_alice, &key_bob, &mut rng);
    let final_key = if secure {
        privacy_amplify(&corrected_bob, bits_leaked)
    } else {
        vec![]
    };

    let key_rate = asymptotic_key_rate(&QkdProtocol::BBM92, qber);
    let num_bits_sifted = key_alice.len();
    let num_bits_final = final_key.len();

    Ok(QkdResult {
        raw_key_alice: alice_bits,
        raw_key_bob: bob_bits,
        sifted_key: sifted_alice,
        final_key,
        qber,
        key_rate,
        secure,
        num_bits_sent: n,
        num_bits_sifted,
        num_bits_final,
        eavesdropper_detected,
        chsh_s_value: 0.0,
    })
}

// ============================================================
// B92 PROTOCOL
// ============================================================

/// Run the B92 (Bennett 1992) two-state protocol.
///
/// Alice sends |0> for bit 0 and |+> for bit 1.
/// Bob measures in X (to detect |0>) or Z (to detect |+>).
/// A conclusive result occurs when Bob's measurement unambiguously identifies Alice's state.
pub fn run_b92(config: &QkdConfig) -> Result<QkdResult, QkdError> {
    let mut rng = rand::thread_rng();
    let n = config.num_bits;

    // Alice's random bits
    let alice_bits: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
    // Alice sends |0> for 0 (Z basis), |+> for 1 (X basis)
    let alice_bases: Vec<u8> = alice_bits.iter().map(|&b| b).collect(); // basis = bit value

    // Bob measures in X (1) or Z (0) randomly
    let bob_bases: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();

    let mut bob_bits = Vec::with_capacity(n);
    let mut conclusive = Vec::with_capacity(n);

    for i in 0..n {
        // Apply eavesdropper
        let (bit_after_eve, _) =
            apply_eavesdropper(alice_bits[i], alice_bases[i], &config.eavesdropper, &mut rng);

        // Apply channel noise
        let (bit_after_channel, _) =
            simulate_qubit_transmission(alice_bases[i], bit_after_eve, config.error_rate, &mut rng);

        // B92 conclusiveness:
        // If Alice sent |0> (bit=0, Z basis) and Bob measures in X basis:
        //   Bob gets |+> or |-> with equal prob. Conclusive only if he gets |->
        //   (which would be impossible if Alice truly sent |0> in absence of noise)
        // If Alice sent |+> (bit=1, X basis) and Bob measures in Z basis:
        //   Bob gets |0> or |1> with equal prob. Conclusive only if he gets |1>
        //
        // For simulation: conclusive when Bob uses the "other" basis and gets the
        // distinguishing outcome
        let is_conclusive;
        let b_bit;

        if alice_bases[i] == 0 && bob_bases[i] == 1 {
            // Alice sent Z, Bob measures X
            // Without noise: 50% chance conclusive (gets |->)
            // With the transmitted bit (possibly noisy):
            if rng.gen::<f64>() < 0.5 {
                is_conclusive = true;
                b_bit = bit_after_channel;
            } else {
                is_conclusive = false;
                b_bit = rng.gen_range(0..2);
            }
        } else if alice_bases[i] == 1 && bob_bases[i] == 0 {
            // Alice sent X, Bob measures Z
            if rng.gen::<f64>() < 0.5 {
                is_conclusive = true;
                b_bit = bit_after_channel;
            } else {
                is_conclusive = false;
                b_bit = rng.gen_range(0..2);
            }
        } else {
            // Same basis or other combination: inconclusive
            is_conclusive = false;
            b_bit = rng.gen_range(0..2);
        }

        bob_bits.push(b_bit);
        conclusive.push(is_conclusive);
    }

    // Sifting: keep only conclusive measurements
    let mut sifted_alice = Vec::new();
    let mut sifted_bob = Vec::new();
    for i in 0..n {
        if conclusive[i] {
            sifted_alice.push(alice_bits[i]);
            sifted_bob.push(bob_bits[i]);
        }
    }

    if sifted_alice.len() < 20 {
        return Err(QkdError::KeyTooShort);
    }

    // QBER estimation
    let num_test = (sifted_alice.len() as f64 * config.qber_estimation_fraction).ceil() as usize;
    let num_test = num_test.max(1).min(sifted_alice.len() / 2);
    let mut errors = 0usize;
    for i in 0..num_test {
        if sifted_alice[i] != sifted_bob[i] {
            errors += 1;
        }
    }
    let qber = errors as f64 / num_test as f64;

    let key_alice: Vec<u8> = sifted_alice[num_test..].to_vec();
    let key_bob: Vec<u8> = sifted_bob[num_test..].to_vec();

    let threshold = QkdProtocol::B92.qber_threshold();
    let eavesdropper_detected = qber > threshold;
    let secure = !eavesdropper_detected;

    let (corrected_bob, bits_leaked) = cascade_error_correct(&key_alice, &key_bob, &mut rng);
    let final_key = if secure {
        privacy_amplify(&corrected_bob, bits_leaked)
    } else {
        vec![]
    };

    let key_rate = asymptotic_key_rate(&QkdProtocol::B92, qber);
    let num_bits_sifted = key_alice.len();
    let num_bits_final = final_key.len();

    Ok(QkdResult {
        raw_key_alice: alice_bits,
        raw_key_bob: bob_bits,
        sifted_key: sifted_alice,
        final_key,
        qber,
        key_rate,
        secure,
        num_bits_sent: n,
        num_bits_sifted,
        num_bits_final,
        eavesdropper_detected,
        chsh_s_value: 0.0,
    })
}

// ============================================================
// SIX-STATE PROTOCOL
// ============================================================

/// Run the Six-State protocol.
///
/// Extension of BB84 with 3 bases (X, Y, Z) and 6 states.
/// Higher QBER tolerance (~12.6%) but lower sifting rate (1/3).
pub fn run_six_state(config: &QkdConfig) -> Result<QkdResult, QkdError> {
    let mut rng = rand::thread_rng();
    let n = config.num_bits;

    // Alice chooses random bits and random bases (0=Z, 1=X, 2=Y)
    let alice_bits: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
    let alice_bases: Vec<u8> = (0..n).map(|_| rng.gen_range(0..3)).collect();

    // Transmission with eavesdropping and channel noise
    let mut transmitted_bits = Vec::with_capacity(n);
    for i in 0..n {
        let (bit_after_eve, _) =
            apply_eavesdropper(alice_bits[i], alice_bases[i], &config.eavesdropper, &mut rng);
        let (bit_after_channel, _) =
            simulate_qubit_transmission(alice_bases[i], bit_after_eve, config.error_rate, &mut rng);
        transmitted_bits.push(bit_after_channel);
    }

    // Bob measures in random bases (0=Z, 1=X, 2=Y)
    let bob_bases: Vec<u8> = (0..n).map(|_| rng.gen_range(0..3)).collect();
    let bob_bits: Vec<u8> = (0..n)
        .map(|i| {
            if bob_bases[i] == alice_bases[i] {
                transmitted_bits[i]
            } else {
                rng.gen_range(0..2)
            }
        })
        .collect();

    // Sifting: keep matching bases (probability 1/3)
    let mut sifted_alice = Vec::new();
    let mut sifted_bob = Vec::new();
    for i in 0..n {
        if alice_bases[i] == bob_bases[i] {
            sifted_alice.push(alice_bits[i]);
            sifted_bob.push(bob_bits[i]);
        }
    }

    if sifted_alice.len() < 20 {
        return Err(QkdError::KeyTooShort);
    }

    // QBER estimation
    let num_test = (sifted_alice.len() as f64 * config.qber_estimation_fraction).ceil() as usize;
    let num_test = num_test.max(1).min(sifted_alice.len() / 2);
    let mut errors = 0usize;
    for i in 0..num_test {
        if sifted_alice[i] != sifted_bob[i] {
            errors += 1;
        }
    }
    let qber = errors as f64 / num_test as f64;

    let key_alice: Vec<u8> = sifted_alice[num_test..].to_vec();
    let key_bob: Vec<u8> = sifted_bob[num_test..].to_vec();

    let threshold = QkdProtocol::SixState.qber_threshold();
    let eavesdropper_detected = qber > threshold;
    let secure = !eavesdropper_detected;

    let (corrected_bob, bits_leaked) = cascade_error_correct(&key_alice, &key_bob, &mut rng);
    let final_key = if secure {
        privacy_amplify(&corrected_bob, bits_leaked)
    } else {
        vec![]
    };

    let key_rate = asymptotic_key_rate(&QkdProtocol::SixState, qber);
    let num_bits_sifted = key_alice.len();
    let num_bits_final = final_key.len();

    Ok(QkdResult {
        raw_key_alice: alice_bits,
        raw_key_bob: bob_bits,
        sifted_key: sifted_alice,
        final_key,
        qber,
        key_rate,
        secure,
        num_bits_sent: n,
        num_bits_sifted,
        num_bits_final,
        eavesdropper_detected,
        chsh_s_value: 0.0,
    })
}

// ============================================================
// DISPATCH
// ============================================================

/// Run any supported QKD protocol with the given configuration.
pub fn run_protocol(protocol: &QkdProtocol, config: &QkdConfig) -> Result<QkdResult, QkdError> {
    match protocol {
        QkdProtocol::BB84 => run_bb84(config),
        QkdProtocol::E91 => run_e91(config),
        QkdProtocol::BBM92 => run_bbm92(config),
        QkdProtocol::B92 => run_b92(config),
        QkdProtocol::SixState => run_six_state(config),
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // 1. QkdConfig builder defaults
    // ----------------------------------------------------------
    #[test]
    fn test_config_defaults() {
        let cfg = QkdConfig::new();
        assert_eq!(cfg.num_bits, 1000);
        assert!((cfg.error_rate - 0.05).abs() < 1e-10);
        assert!((cfg.channel_loss_db - 0.2).abs() < 1e-10);
        assert!((cfg.distance_km - 100.0).abs() < 1e-10);
        assert!((cfg.detector_efficiency - 0.9).abs() < 1e-10);
        match &cfg.eavesdropper {
            EavesdropperModel::None => {}
            _ => panic!("Default eavesdropper should be None"),
        }
    }

    #[test]
    fn test_config_builder() {
        let cfg = QkdConfig::new()
            .num_bits(5000)
            .error_rate(0.02)
            .distance_km(50.0)
            .detector_efficiency(0.95)
            .eavesdropper(EavesdropperModel::InterceptResend { fraction: 0.5 });
        assert_eq!(cfg.num_bits, 5000);
        assert!((cfg.error_rate - 0.02).abs() < 1e-10);
        assert!((cfg.distance_km - 50.0).abs() < 1e-10);
    }

    // ----------------------------------------------------------
    // 2. BB84 without eavesdropper: QBER < threshold, secure
    // ----------------------------------------------------------
    #[test]
    fn test_bb84_no_eavesdropper() {
        let cfg = QkdConfig::new()
            .num_bits(5000)
            .error_rate(0.02)
            .eavesdropper(EavesdropperModel::None);
        let result = run_bb84(&cfg).unwrap();
        assert!(result.secure, "BB84 should be secure without eavesdropper");
        assert!(
            result.qber < QkdProtocol::BB84.qber_threshold(),
            "QBER {} should be below threshold {}",
            result.qber,
            QkdProtocol::BB84.qber_threshold()
        );
        assert!(!result.eavesdropper_detected);
        assert!(result.final_key.len() > 0, "Should produce a final key");
        assert!(result.key_rate > 0.0);
    }

    // ----------------------------------------------------------
    // 3. BB84 with intercept-resend: QBER ~ 25%, secure = false
    // ----------------------------------------------------------
    #[test]
    fn test_bb84_intercept_resend() {
        let cfg = QkdConfig::new()
            .num_bits(10000)
            .error_rate(0.01)
            .eavesdropper(EavesdropperModel::InterceptResend { fraction: 1.0 });
        let result = run_bb84(&cfg).unwrap();
        // Full intercept-resend introduces ~25% QBER
        assert!(
            !result.secure,
            "BB84 should be insecure with full intercept-resend"
        );
        assert!(
            result.eavesdropper_detected,
            "Eavesdropper should be detected"
        );
        // QBER should be around 0.25 (Eve uses wrong basis 50% of time, flips 50% of those)
        assert!(
            result.qber > 0.11,
            "QBER {} should exceed BB84 threshold with full intercept-resend",
            result.qber
        );
    }

    // ----------------------------------------------------------
    // 4. BB84 sifting rate ~ 50%
    // ----------------------------------------------------------
    #[test]
    fn test_bb84_sifting_rate() {
        let cfg = QkdConfig::new()
            .num_bits(10000)
            .error_rate(0.0)
            .eavesdropper(EavesdropperModel::None);
        let result = run_bb84(&cfg).unwrap();
        // Sifted key + test bits should be roughly 50% of sent bits
        let total_sifted = result.sifted_key.len();
        let sifting_rate = total_sifted as f64 / result.num_bits_sent as f64;
        assert!(
            (sifting_rate - 0.5).abs() < 0.1,
            "BB84 sifting rate {} should be ~0.5",
            sifting_rate
        );
    }

    // ----------------------------------------------------------
    // 5. E91 CHSH violation for genuine entanglement
    // ----------------------------------------------------------
    #[test]
    fn test_e91_chsh_violation() {
        let cfg = QkdConfig::new()
            .num_bits(20000)
            .error_rate(0.01)
            .eavesdropper(EavesdropperModel::None);
        let result = run_e91(&cfg).unwrap();
        // For genuine entanglement, S should be > 2 (ideally ~2√2 ≈ 2.828)
        assert!(
            result.chsh_s_value > 2.0,
            "CHSH S-value {} should exceed 2 for genuine entanglement",
            result.chsh_s_value
        );
        assert!(result.secure, "E91 should be secure without eavesdropper");
    }

    // ----------------------------------------------------------
    // 6. E91 with eavesdropper: S <= 2 or high QBER
    // ----------------------------------------------------------
    #[test]
    fn test_e91_with_eavesdropper() {
        // Full intercept-resend destroys entanglement correlations
        let cfg = QkdConfig::new()
            .num_bits(20000)
            .error_rate(0.01)
            .eavesdropper(EavesdropperModel::InterceptResend { fraction: 1.0 });
        let result = run_e91(&cfg).unwrap();
        // Either CHSH drops below 2 or QBER is too high (or both)
        assert!(
            result.eavesdropper_detected,
            "E91 should detect eavesdropper (S={}, QBER={})",
            result.chsh_s_value,
            result.qber
        );
        assert!(
            !result.secure,
            "E91 should not be secure with full intercept-resend"
        );
    }

    // ----------------------------------------------------------
    // 7. BBM92 produces key from entangled pairs
    // ----------------------------------------------------------
    #[test]
    fn test_bbm92_entangled_key() {
        let cfg = QkdConfig::new()
            .num_bits(5000)
            .error_rate(0.02)
            .eavesdropper(EavesdropperModel::None);
        let result = run_bbm92(&cfg).unwrap();
        assert!(result.secure, "BBM92 should be secure without eavesdropper");
        assert!(
            result.final_key.len() > 0,
            "BBM92 should produce a final key"
        );
        assert!(
            result.qber < QkdProtocol::BBM92.qber_threshold(),
            "BBM92 QBER {} should be below threshold",
            result.qber
        );
    }

    // ----------------------------------------------------------
    // 8. B92 has lower key rate than BB84
    // ----------------------------------------------------------
    #[test]
    fn test_b92_lower_key_rate() {
        let cfg = QkdConfig::new()
            .num_bits(10000)
            .error_rate(0.02)
            .eavesdropper(EavesdropperModel::None);

        let bb84_result = run_bb84(&cfg).unwrap();
        let b92_result = run_b92(&cfg).unwrap();

        // B92 should sift fewer bits (only conclusive measurements ~25%)
        let bb84_sift_rate =
            bb84_result.sifted_key.len() as f64 / bb84_result.num_bits_sent as f64;
        let b92_sift_rate = b92_result.sifted_key.len() as f64 / b92_result.num_bits_sent as f64;

        assert!(
            b92_sift_rate < bb84_sift_rate,
            "B92 sifting rate {} should be lower than BB84 {}",
            b92_sift_rate,
            bb84_sift_rate
        );
    }

    // ----------------------------------------------------------
    // 9. Six-state has ~1/3 sifting rate
    // ----------------------------------------------------------
    #[test]
    fn test_six_state_sifting_rate() {
        let cfg = QkdConfig::new()
            .num_bits(10000)
            .error_rate(0.0)
            .eavesdropper(EavesdropperModel::None);
        let result = run_six_state(&cfg).unwrap();
        let sifting_rate = result.sifted_key.len() as f64 / result.num_bits_sent as f64;
        assert!(
            (sifting_rate - 1.0 / 3.0).abs() < 0.1,
            "Six-state sifting rate {} should be ~0.333",
            sifting_rate
        );
    }

    // ----------------------------------------------------------
    // 10. Binary entropy: h(0) = 0, h(0.5) = 1
    // ----------------------------------------------------------
    #[test]
    fn test_binary_entropy() {
        assert!((binary_entropy(0.0) - 0.0).abs() < 1e-10);
        assert!((binary_entropy(1.0) - 0.0).abs() < 1e-10);
        assert!((binary_entropy(0.5) - 1.0).abs() < 1e-10);

        // h(p) is symmetric: h(p) == h(1-p)
        let p = 0.3;
        assert!((binary_entropy(p) - binary_entropy(1.0 - p)).abs() < 1e-10);

        // h(p) is in [0, 1]
        for i in 1..100 {
            let p = i as f64 / 100.0;
            let h = binary_entropy(p);
            assert!(h >= 0.0 && h <= 1.0, "h({}) = {} out of range", p, h);
        }
    }

    // ----------------------------------------------------------
    // 11. Key rate = 0 when QBER = 0.5
    // ----------------------------------------------------------
    #[test]
    fn test_key_rate_zero_at_half() {
        assert!(
            asymptotic_key_rate(&QkdProtocol::BB84, 0.5).abs() < 1e-10,
            "Key rate should be 0 at QBER=0.5"
        );
        assert!(
            asymptotic_key_rate(&QkdProtocol::E91, 0.5).abs() < 1e-10,
            "Key rate should be 0 at QBER=0.5"
        );
        assert!(
            asymptotic_key_rate(&QkdProtocol::B92, 0.5).abs() < 1e-10,
            "Key rate should be 0 at QBER=0.5"
        );
    }

    // ----------------------------------------------------------
    // 12. Key rate > 0 when QBER < threshold
    // ----------------------------------------------------------
    #[test]
    fn test_key_rate_positive_below_threshold() {
        let qber = 0.03;
        assert!(
            asymptotic_key_rate(&QkdProtocol::BB84, qber) > 0.0,
            "BB84 key rate should be positive at QBER={}",
            qber
        );
        assert!(
            asymptotic_key_rate(&QkdProtocol::E91, qber) > 0.0,
            "E91 key rate should be positive at QBER={}",
            qber
        );
        assert!(
            asymptotic_key_rate(&QkdProtocol::B92, qber) > 0.0,
            "B92 key rate should be positive at QBER={}",
            qber
        );
        assert!(
            asymptotic_key_rate(&QkdProtocol::SixState, qber) > 0.0,
            "SixState key rate should be positive at QBER={}",
            qber
        );

        // B92 should have higher rate than BB84 at same QBER (1 - h vs 1 - 2h)
        let b92_rate = asymptotic_key_rate(&QkdProtocol::B92, qber);
        let bb84_rate = asymptotic_key_rate(&QkdProtocol::BB84, qber);
        assert!(
            b92_rate > bb84_rate,
            "B92 rate {} should exceed BB84 rate {} at same QBER",
            b92_rate,
            bb84_rate
        );
    }

    // ----------------------------------------------------------
    // 13. Channel loss increases with distance
    // ----------------------------------------------------------
    #[test]
    fn test_channel_loss_distance() {
        let loss_db = 0.2; // typical fiber
        let t_10 = channel_transmission(10.0, loss_db);
        let t_50 = channel_transmission(50.0, loss_db);
        let t_100 = channel_transmission(100.0, loss_db);
        let t_200 = channel_transmission(200.0, loss_db);

        assert!(t_10 > t_50, "Closer distance should have higher transmission");
        assert!(t_50 > t_100);
        assert!(t_100 > t_200);

        // At 0 distance, transmission should be 1.0
        let t_0 = channel_transmission(0.0, loss_db);
        assert!((t_0 - 1.0).abs() < 1e-10);

        // Known value: 100 km at 0.2 dB/km = 20 dB loss → η = 10^(-2) = 0.01
        assert!(
            (t_100 - 0.01).abs() < 1e-10,
            "100km at 0.2 dB/km should give η=0.01, got {}",
            t_100
        );
    }

    // ----------------------------------------------------------
    // 14. Privacy amplification shortens key
    // ----------------------------------------------------------
    #[test]
    fn test_privacy_amplification() {
        let key: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1];
        let leaked = 3;
        let amplified = privacy_amplify(&key, leaked);
        assert_eq!(
            amplified.len(),
            key.len() - leaked,
            "Privacy amplification should shorten key by leaked bits"
        );

        // Amplifying with more leaked than key length gives empty
        let over_amplified = privacy_amplify(&key, key.len() + 1);
        assert_eq!(over_amplified.len(), 0);

        // Empty key stays empty
        let empty = privacy_amplify(&[], 0);
        assert_eq!(empty.len(), 0);
    }

    // ----------------------------------------------------------
    // 15. Dispatch routes correctly
    // ----------------------------------------------------------
    #[test]
    fn test_dispatch() {
        let cfg = QkdConfig::new()
            .num_bits(2000)
            .error_rate(0.02)
            .eavesdropper(EavesdropperModel::None);

        // All protocols should succeed without eavesdropper
        for proto in &[
            QkdProtocol::BB84,
            QkdProtocol::E91,
            QkdProtocol::BBM92,
            QkdProtocol::B92,
            QkdProtocol::SixState,
        ] {
            let result = run_protocol(proto, &cfg);
            assert!(result.is_ok(), "{:?} protocol failed: {:?}", proto, result);
        }
    }

    // ----------------------------------------------------------
    // 16. Detector click probability
    // ----------------------------------------------------------
    #[test]
    fn test_detector_click_probability() {
        let p = detector_click_probability(0.5, 0.9, 1e-6);
        // p ≈ 0.5 * 0.9 + 1e-6 = 0.450001
        assert!((p - 0.450001).abs() < 0.01);

        // Zero transmission → only dark counts
        let p_dark = detector_click_probability(0.0, 0.9, 1e-4);
        assert!(
            (p_dark - 1e-4).abs() < 1e-5,
            "With no signal, click prob should equal dark count rate"
        );
    }

    // ----------------------------------------------------------
    // 17. CASCADE error correction fixes errors
    // ----------------------------------------------------------
    #[test]
    fn test_cascade_correction() {
        let mut rng = rand::thread_rng();
        let alice: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0];
        let mut bob = alice.clone();
        // Introduce a few errors
        bob[2] = 1 - bob[2];
        bob[7] = 1 - bob[7];
        bob[13] = 1 - bob[13];

        let (corrected, leaked) = cascade_error_correct(&alice, &bob, &mut rng);
        assert_eq!(corrected.len(), alice.len());
        assert_eq!(corrected, alice, "CASCADE should correct all errors");
        assert!(leaked > 0, "CASCADE should leak some parity bits");
    }

    // ----------------------------------------------------------
    // 18. Finite key rate is less than asymptotic
    // ----------------------------------------------------------
    #[test]
    fn test_finite_key_rate() {
        let qber = 0.03;
        let r_inf = asymptotic_key_rate(&QkdProtocol::BB84, qber);
        let r_fin = finite_key_rate(&QkdProtocol::BB84, qber, 10000, 1e-10);
        assert!(
            r_fin <= r_inf,
            "Finite key rate {} should not exceed asymptotic {}",
            r_fin,
            r_inf
        );
        assert!(r_fin > 0.0, "Finite rate should be positive for 10k bits");

        // Very small key: finite rate should be zero or very small
        let r_tiny = finite_key_rate(&QkdProtocol::BB84, qber, 10, 1e-10);
        assert!(
            r_tiny < r_fin,
            "Tiny key rate {} should be less than large key rate {}",
            r_tiny,
            r_fin
        );
    }

    // ----------------------------------------------------------
    // 19. Six-state has higher QBER tolerance than BB84
    // ----------------------------------------------------------
    #[test]
    fn test_six_state_higher_tolerance() {
        assert!(
            QkdProtocol::SixState.qber_threshold() > QkdProtocol::BB84.qber_threshold(),
            "Six-state threshold should be higher than BB84"
        );
        // At QBER = 0.115 (above BB84 threshold, below six-state)
        let qber = 0.115;
        let bb84_rate = asymptotic_key_rate(&QkdProtocol::BB84, qber);
        let six_rate = asymptotic_key_rate(&QkdProtocol::SixState, qber);
        assert!(
            bb84_rate == 0.0,
            "BB84 should have zero rate at QBER={}",
            qber
        );
        assert!(
            six_rate > 0.0,
            "Six-state should have positive rate at QBER={}",
            qber
        );
    }
}
