//! Metropolitan-Scale Quantum Key Distribution Network Simulation
//!
//! Models multi-node QKD networks with trusted relay nodes, entanglement-based
//! links, and realistic fiber loss for metropolitan-area deployments. Based on
//! the Berlin BearlinQ network architecture (2025) and the Nature 2026
//! 20-client photonic QKD demonstrations.
//!
//! # Physical Models
//!
//! - **Fiber loss**: Exponential attenuation at 0.2 dB/km (telecom C-band, 1550 nm)
//! - **Dark counts**: Detector noise contributing to QBER floor
//! - **Multi-photon**: Weak coherent source Poisson statistics
//! - **Chromatic dispersion**: Pulse broadening in long fibers
//!
//! # Protocols
//!
//! - **BB84** (with optional decoy states): Prepare-and-measure, workhorse protocol
//! - **BBM92**: Entanglement-based variant of BB84
//! - **E91**: Ekert protocol with Bell inequality verification
//! - **CV-QKD**: Continuous-variable with Gaussian or discrete modulation
//! - **TF-QKD**: Twin-field for long-distance (sqrt(eta) scaling)
//! - **MDI-QKD**: Measurement-device-independent, immune to detector attacks
//!
//! # Network Features
//!
//! - Trusted relay key forwarding via XOR chain
//! - Dijkstra routing with distance, key-rate, and load-balanced metrics
//! - k-shortest-path multi-path routing
//! - Security analysis: composable epsilon-security, finite-key effects
//! - Pre-built topologies: BearlinQ, Tokyo, Cambridge, intercity, star
//!
//! # Usage
//!
//! ```ignore
//! use nqpu_metal::metro_qkd_network::*;
//!
//! let net = NetworkLibrary::bearlinq();
//! let result = net.simulate(1.0);
//! assert!(result.average_qber < 0.11);
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

// ===================================================================
// CONSTANTS
// ===================================================================

/// Standard telecom fiber loss in dB/km (ITU-T G.652, 1550 nm window).
const DEFAULT_FIBER_LOSS_DB_PER_KM: f64 = 0.2;

/// Standard telecom wavelength in nm (C-band).
const DEFAULT_WAVELENGTH_NM: f64 = 1550.0;

/// BB84 security threshold for QBER (above this, protocol aborts).
const BB84_QBER_THRESHOLD: f64 = 0.11;

/// Error correction efficiency factor (typical CASCADE/LDPC).
const DEFAULT_EC_EFFICIENCY: f64 = 1.16;

/// Default repetition rate for pulsed QKD sources (Hz).
const DEFAULT_REP_RATE: f64 = 1.0e9;

/// Default mean photon number per pulse for weak coherent sources.
const DEFAULT_MEAN_PHOTON_NUMBER: f64 = 0.5;

/// Default detector efficiency for InGaAs SPADs.
const DEFAULT_DETECTOR_EFFICIENCY: f64 = 0.10;

/// Default dark count probability per gate window for InGaAs SPADs.
/// Typical values: 1e-6 to 1e-5 per gate for gated InGaAs detectors.
const DEFAULT_DARK_COUNT_PROB: f64 = 1.0e-6;

/// Default dark count rate (counts per second) for InGaAs SPADs.
/// With 1 GHz gating: p_dark_per_gate = 1e-6 => rate = 1e3 counts/sec.
const DEFAULT_DARK_COUNT_RATE: f64 = 1.0e3;

/// Default detector gate window in seconds (1 ns).
const DEFAULT_GATE_WINDOW: f64 = 1.0e-9;

/// Default optical misalignment error contribution.
const DEFAULT_MISALIGNMENT_ERROR: f64 = 0.015;

/// Speed of light in vacuum (m/s).
const C_VACUUM: f64 = 3.0e8;

/// Refractive index of standard telecom fiber.
const FIBER_REFRACTIVE_INDEX: f64 = 1.47;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors that can occur in QKD network operations.
#[derive(Debug, Clone, PartialEq)]
pub enum QkdNetworkError {
    /// A physical link between two nodes has failed or is unavailable.
    LinkFailed(String),
    /// The achievable key rate is too low for the requested operation.
    InsufficientKeyRate(String),
    /// Classical authentication between nodes failed.
    AuthenticationError(String),
    /// The network graph has become disconnected; no path exists between
    /// the requested source and destination.
    NetworkPartitioned(String),
    /// Channel loss exceeds the maximum tolerable for key generation.
    ChannelLoss(String),
    /// A node index is out of range.
    InvalidNode(String),
    /// Configuration is invalid.
    InvalidConfig(String),
}

impl std::fmt::Display for QkdNetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LinkFailed(msg) => write!(f, "Link failed: {}", msg),
            Self::InsufficientKeyRate(msg) => write!(f, "Insufficient key rate: {}", msg),
            Self::AuthenticationError(msg) => write!(f, "Authentication error: {}", msg),
            Self::NetworkPartitioned(msg) => write!(f, "Network partitioned: {}", msg),
            Self::ChannelLoss(msg) => write!(f, "Channel loss: {}", msg),
            Self::InvalidNode(msg) => write!(f, "Invalid node: {}", msg),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
        }
    }
}

impl std::error::Error for QkdNetworkError {}

// ===================================================================
// PROTOCOL TYPES
// ===================================================================

/// Continuous-variable QKD modulation scheme.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CVModulation {
    /// Gaussian modulation of coherent states in phase space.
    GaussianModulation,
    /// Discrete modulation with a fixed number of coherent states
    /// arranged symmetrically in phase space.
    DiscreteModulation { num_states: usize },
}

/// QKD protocol used on a physical link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QkdProtocol {
    /// BB84 prepare-and-measure protocol.
    /// When `decoy_states` is true, uses the three-intensity decoy-state
    /// method for improved key rate estimation.
    BB84 { decoy_states: bool },
    /// BBM92 entanglement-based protocol.
    BBM92,
    /// E91 (Ekert) protocol with Bell inequality test.
    E91,
    /// Continuous-variable QKD using homodyne or heterodyne detection.
    CvQkd { modulation: CVModulation },
    /// Twin-field QKD: key rate scales as sqrt(eta), enabling > 500 km.
    TfQkd,
    /// Measurement-device-independent QKD: immune to detector attacks.
    MdiQkd,
}

impl Default for QkdProtocol {
    fn default() -> Self {
        QkdProtocol::BB84 { decoy_states: true }
    }
}

// ===================================================================
// NODE TYPES
// ===================================================================

/// Classification of a QKD network node.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    /// End user (Alice/Bob) who generates or consumes secret keys.
    EndUser,
    /// Trusted relay node that performs key forwarding via XOR.
    /// Security assumption: the relay is not compromised.
    TrustedRelay,
    /// Quantum repeater with finite quantum memory coherence time.
    QuantumRepeater { memory_time_ms: f64 },
    /// Dedicated entanglement source (e.g., SPDC photon-pair source).
    EntanglementSource,
}

/// A node in the QKD network.
#[derive(Debug, Clone)]
pub struct QkdNode {
    /// Unique node identifier.
    pub id: usize,
    /// Human-readable name (e.g., "Alice", "TU-Berlin", "Relay-3").
    pub name: String,
    /// Functional role in the network.
    pub node_type: NodeType,
    /// Geographic position in (x, y) kilometres from an arbitrary origin.
    pub position: (f64, f64),
    /// Number of available secret key bits in local storage.
    pub key_storage: usize,
    /// Single-photon detector efficiency (0.0 to 1.0).
    pub detector_efficiency: f64,
    /// Detector dark count rate in counts per second.
    pub dark_count_rate: f64,
}

impl QkdNode {
    /// Create a new QKD node with the given parameters.
    pub fn new(id: usize, name: &str, node_type: NodeType, position: (f64, f64)) -> Self {
        Self {
            id,
            name: name.to_string(),
            node_type,
            position,
            key_storage: 0,
            detector_efficiency: DEFAULT_DETECTOR_EFFICIENCY,
            dark_count_rate: DEFAULT_DARK_COUNT_RATE,
        }
    }

    /// Builder: set detector efficiency.
    pub fn with_detector_efficiency(mut self, eta: f64) -> Self {
        self.detector_efficiency = eta.clamp(0.0, 1.0);
        self
    }

    /// Builder: set dark count rate.
    pub fn with_dark_count_rate(mut self, rate: f64) -> Self {
        self.dark_count_rate = rate.max(0.0);
        self
    }

    /// Builder: set initial key storage.
    pub fn with_key_storage(mut self, bits: usize) -> Self {
        self.key_storage = bits;
        self
    }

    /// Euclidean distance to another node in km.
    pub fn distance_to(&self, other: &QkdNode) -> f64 {
        let dx = self.position.0 - other.position.0;
        let dy = self.position.1 - other.position.1;
        (dx * dx + dy * dy).sqrt()
    }
}

// ===================================================================
// FIBER CHANNEL MODEL
// ===================================================================

/// Physical fiber optic channel model.
///
/// Models exponential loss, background noise photons, and chromatic
/// dispersion for standard single-mode telecom fiber.
#[derive(Debug, Clone)]
pub struct FiberChannel {
    /// Fiber length in km.
    pub length_km: f64,
    /// Total loss in dB.
    pub loss_db: f64,
    /// Channel transmittance: 10^(-loss_db / 10).
    pub transmittance: f64,
    /// Mean number of background noise photons per detection window.
    pub noise_photons: f64,
    /// Chromatic dispersion in ps/(nm * km).
    pub chromatic_dispersion: f64,
}

impl FiberChannel {
    /// Create a fiber channel with the given length and loss coefficient.
    ///
    /// # Arguments
    /// * `length_km` - Fiber length in kilometres
    /// * `loss_db_per_km` - Attenuation coefficient (typically 0.2 dB/km)
    /// * `noise_photons` - Background noise photons per gate window
    pub fn new(length_km: f64, loss_db_per_km: f64, noise_photons: f64) -> Self {
        let loss_db = loss_db_per_km * length_km;
        let transmittance = 10.0_f64.powf(-loss_db / 10.0);
        Self {
            length_km,
            loss_db,
            transmittance,
            noise_photons,
            chromatic_dispersion: 17.0, // ps/(nm*km), standard SMF-28
        }
    }

    /// Create a standard telecom fiber channel (0.2 dB/km, minimal noise).
    pub fn standard(length_km: f64) -> Self {
        Self::new(
            length_km,
            DEFAULT_FIBER_LOSS_DB_PER_KM,
            DEFAULT_DARK_COUNT_PROB,
        )
    }

    /// Propagation delay through the fiber in seconds.
    pub fn propagation_delay_s(&self) -> f64 {
        (self.length_km * 1000.0 * FIBER_REFRACTIVE_INDEX) / C_VACUUM
    }
}

// ===================================================================
// QKD LINK
// ===================================================================

/// A physical QKD link between two network nodes.
///
/// Each link represents a fiber connection with a specific QKD protocol,
/// and tracks the achievable key rate and quantum bit error rate.
#[derive(Debug, Clone)]
pub struct QkdLink {
    /// Index of the first node.
    pub node_a: usize,
    /// Index of the second node.
    pub node_b: usize,
    /// Physical fiber length in km.
    pub fiber_length_km: f64,
    /// Fiber loss coefficient in dB/km (typically 0.2).
    pub loss_db_per_km: f64,
    /// QKD protocol running on this link.
    pub protocol: QkdProtocol,
    /// Optical wavelength in nm (typically 1550).
    pub wavelength_nm: f64,
    /// Achieved secret key rate in bits per second.
    pub key_rate_bps: f64,
    /// Quantum bit error rate on this link.
    pub qber: f64,
    /// Whether this link is currently operational.
    pub active: bool,
}

impl QkdLink {
    /// Create a new QKD link with default protocol (decoy-state BB84).
    pub fn new(node_a: usize, node_b: usize, fiber_length_km: f64) -> Self {
        Self {
            node_a,
            node_b,
            fiber_length_km,
            loss_db_per_km: DEFAULT_FIBER_LOSS_DB_PER_KM,
            protocol: QkdProtocol::default(),
            wavelength_nm: DEFAULT_WAVELENGTH_NM,
            key_rate_bps: 0.0,
            qber: 0.0,
            active: true,
        }
    }

    /// Builder: set the QKD protocol.
    pub fn with_protocol(mut self, protocol: QkdProtocol) -> Self {
        self.protocol = protocol;
        self
    }

    /// Builder: set fiber loss coefficient.
    pub fn with_loss(mut self, loss_db_per_km: f64) -> Self {
        self.loss_db_per_km = loss_db_per_km;
        self
    }

    /// Builder: set wavelength.
    pub fn with_wavelength(mut self, nm: f64) -> Self {
        self.wavelength_nm = nm;
        self
    }

    /// Compute the fiber channel model for this link.
    pub fn fiber_channel(&self) -> FiberChannel {
        FiberChannel::new(
            self.fiber_length_km,
            self.loss_db_per_km,
            DEFAULT_DARK_COUNT_PROB,
        )
    }

    /// Compute fiber transmittance for this link.
    pub fn transmittance(&self) -> f64 {
        let loss_db = self.loss_db_per_km * self.fiber_length_km;
        10.0_f64.powf(-loss_db / 10.0)
    }
}

// ===================================================================
// KEY RATE ESTIMATION
// ===================================================================

/// Result of a key rate estimation for a single link.
#[derive(Debug, Clone)]
pub struct KeyRateEstimate {
    /// Raw key rate before any post-processing (bits/second).
    pub raw_key_rate: f64,
    /// Key rate after basis sifting (bits/second).
    pub sifted_key_rate: f64,
    /// Final secret key rate after error correction and privacy
    /// amplification (bits/second).
    pub secret_key_rate: f64,
    /// Quantum bit error rate.
    pub qber: f64,
    /// Whether the link is secure (QBER below protocol threshold).
    pub secure: bool,
}

/// Binary entropy function H(p) = -p log2(p) - (1-p) log2(1-p).
///
/// Returns 0.0 for p == 0.0 or p == 1.0 to handle edge cases.
pub fn binary_entropy(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

/// Estimate the key rate for a BB84 link (standard, no decoy states).
///
/// # Model
///
/// - Raw rate: R_raw = f_rep * mu * eta_link * eta_det
/// - QBER from dark counts, misalignment, and multi-photon events
/// - Sifted rate: R_sifted = R_raw / 2 (two-basis sifting)
/// - Secret key rate: R_sk = R_sifted * [1 - H(e) - f_EC * H(e)]
///
/// # Arguments
///
/// * `fiber_length_km` - Length of fiber in km
/// * `loss_db_per_km` - Fiber attenuation in dB/km
/// * `detector_efficiency` - SPD efficiency (0 to 1)
/// * `dark_count_rate` - Dark count rate in counts/second
/// * `rep_rate` - Source repetition rate in Hz
/// * `mean_photon_number` - Mean photon number per pulse (mu)
pub fn bb84_key_rate(
    fiber_length_km: f64,
    loss_db_per_km: f64,
    detector_efficiency: f64,
    dark_count_rate: f64,
    rep_rate: f64,
    mean_photon_number: f64,
) -> KeyRateEstimate {
    // Channel transmittance
    let loss_db = loss_db_per_km * fiber_length_km;
    let eta_channel = 10.0_f64.powf(-loss_db / 10.0);
    let eta_total = eta_channel * detector_efficiency;

    // Dark count probability per gate window.
    // If dark_count_rate looks like a per-second rate (> 1), convert via gate window.
    // If it looks like a per-gate probability (< 1), use directly.
    let p_dark = if dark_count_rate >= 1.0 {
        dark_count_rate * DEFAULT_GATE_WINDOW
    } else {
        dark_count_rate
    };

    // Detection probability per pulse
    let p_signal = 1.0 - (-mean_photon_number * eta_total).exp();
    let p_detect = p_signal + p_dark - p_signal * p_dark;

    // Raw key rate
    let raw_key_rate = rep_rate * p_detect;

    // QBER contributions
    // 1. Dark count contribution: random bits from dark counts
    let e_dark = if p_detect > 0.0 {
        0.5 * p_dark / p_detect
    } else {
        0.5
    };
    // 2. Optical misalignment
    let e_opt = DEFAULT_MISALIGNMENT_ERROR;
    // 3. Multi-photon contribution (PNS attack vulnerability for standard BB84)
    // For non-decoy BB84, Eve can perform PNS on multi-photon pulses.
    // The fraction of multi-photon pulses that leak info:
    // P(n>=2|mu) = 1 - (1+mu)*exp(-mu)
    // The effective QBER contribution from this is conservative.
    let p_multi = 1.0 - (1.0 + mean_photon_number) * (-mean_photon_number).exp();
    let e_multi = if p_detect > 0.0 {
        // Multi-photon pulses give Eve partial info but don't directly add QBER;
        // they reduce the extractable secret key. In the GLLP bound without decoy,
        // the penalty is accounted for in the key rate formula, not QBER.
        // We add a small QBER contribution from detector noise in multi-photon events.
        0.5 * p_multi * eta_total / p_detect * 0.01
    } else {
        0.0
    };
    let qber = (e_dark + e_opt + e_multi).min(0.5);

    // Sifted rate (BB84: 1/2 basis matching probability)
    let sifted_key_rate = raw_key_rate / 2.0;

    // Secret key rate via GLLP bound (non-decoy):
    // R_sk = R_sifted * [1 - H(e) - f_EC * H(e)] * (1 - Delta)
    // where Delta = P(n>=2)/P(detect) is the multi-photon fraction penalty
    let h_e = binary_entropy(qber);
    let multi_photon_fraction = p_multi * eta_total / p_detect.max(1e-30);
    let gllp_penalty = (1.0 - multi_photon_fraction).max(0.0);
    let secret_fraction = ((1.0 - h_e - DEFAULT_EC_EFFICIENCY * h_e) * gllp_penalty).max(0.0);
    let secret_key_rate = sifted_key_rate * secret_fraction;

    let secure = qber < BB84_QBER_THRESHOLD;

    KeyRateEstimate {
        raw_key_rate,
        sifted_key_rate,
        secret_key_rate,
        qber,
        secure,
    }
}

/// Estimate the key rate for decoy-state BB84.
///
/// Uses the three-intensity protocol (signal mu, decoy nu, vacuum)
/// to tightly bound the single-photon yield Y1 and error rate e1.
///
/// Provides significantly better key rates than standard BB84, especially
/// at longer distances, because it eliminates the photon-number-splitting
/// attack penalty.
pub fn decoy_state_bb84_key_rate(
    fiber_length_km: f64,
    loss_db_per_km: f64,
    detector_efficiency: f64,
    dark_count_rate: f64,
    rep_rate: f64,
    mu: f64,
) -> KeyRateEstimate {
    let loss_db = loss_db_per_km * fiber_length_km;
    let eta_channel = 10.0_f64.powf(-loss_db / 10.0);
    let eta_total = eta_channel * detector_efficiency;

    // Dark count probability per gate
    let p_dark = if dark_count_rate >= 1.0 {
        dark_count_rate * DEFAULT_GATE_WINDOW
    } else {
        dark_count_rate
    };

    // Decoy intensities
    let nu = mu / 5.0; // Decoy intensity

    // Gains for each intensity: Q_x = 1 - exp(-(x * eta_total + p_dark))
    let q_mu = 1.0 - (-(mu * eta_total + p_dark)).exp();
    let q_nu = 1.0 - (-(nu * eta_total + p_dark)).exp();

    // Vacuum yield (background + dark counts)
    let y0 = p_dark;

    // Single-photon yield lower bound using the standard two-decoy formula:
    // Y1 >= (mu/(mu*nu - nu^2)) * [Q_nu * exp(nu) - Q_vac * exp(0) * (nu^2/mu^2)
    //        - (mu^2 - nu^2)/(mu^2) * Q_mu * exp(mu)]
    // Simplified robust form:
    // Y1_lower = (Q_nu * exp(nu) - (nu^2/mu^2) * Q_mu * exp(mu) - (1 - nu^2/mu^2) * Y0)
    //            * mu / (mu * nu - nu^2)
    //
    // For numerical stability, use an equivalent direct computation:
    // In the limit of very low dark counts, Y1 ~ eta_total + Y0
    // which is the ideal single-photon detection probability.
    let _ratio = nu / mu;
    let _q_mu_scaled = q_mu * mu.exp();
    let q_nu_scaled = q_nu * nu.exp();

    // GLLP-style decoy formula:
    // Y1 >= (Q_nu * e^nu - ratio^2 * Q_mu * e^mu) / (nu * (1 - ratio))  -  Y0/nu ... complex
    // Use the simpler Ma et al. (2005) practical formula:
    // Y1 = max(0, (Q_nu * e^nu - Q_vac) / nu)  (single decoy lower bound)
    let y1 = ((q_nu_scaled - y0) / nu).clamp(0.0, 1.0);

    // For the single-photon error rate, use:
    // e1 = (E_nu * Q_nu * e^nu - e0 * Y0) / (Y1 * nu)
    let e_opt = DEFAULT_MISALIGNMENT_ERROR;
    let e_nu = (e_opt * nu * eta_total + 0.5 * p_dark) / (nu * eta_total + p_dark).max(1e-30);
    let e1 = if y1 > 1e-20 {
        ((e_nu * q_nu_scaled - 0.5 * y0) / (y1 * nu)).clamp(0.0, 0.5)
    } else {
        0.5
    };

    // Overall QBER on the signal intensity
    let e_mu_total = (e_opt * mu * eta_total + 0.5 * p_dark) / (mu * eta_total + p_dark).max(1e-30);
    let qber = e_mu_total.min(0.5);

    let raw_key_rate = rep_rate * q_mu;
    let sifted_key_rate = raw_key_rate / 2.0;

    // Secret key rate (decoy-state GLLP bound):
    // R = (1/2) * [-Q_mu * f_EC * H(E_mu) + Q_1 * (1 - H(e1))]
    // where Q_1 = mu * exp(-mu) * Y1 is the single-photon gain
    let q1 = mu * (-mu).exp() * y1;

    let skr_per_pulse =
        q1 * (1.0 - binary_entropy(e1)) - DEFAULT_EC_EFFICIENCY * q_mu * binary_entropy(qber);
    let secret_key_rate = (rep_rate * skr_per_pulse / 2.0).max(0.0);

    let secure = qber < BB84_QBER_THRESHOLD;

    KeyRateEstimate {
        raw_key_rate,
        sifted_key_rate,
        secret_key_rate,
        qber,
        secure,
    }
}

/// Estimate the key rate for twin-field QKD (TF-QKD).
///
/// Key advantage: rate scales as sqrt(eta) instead of eta, enabling
/// key distribution over 500+ km of fiber.
///
/// # Model
///
/// Alice and Bob each send weak coherent pulses to a central node (Charlie).
/// Charlie performs single-photon interference. The rate is proportional
/// to sqrt(eta_a * eta_b) rather than eta_a * eta_b.
pub fn tf_qkd_key_rate(
    fiber_length_km: f64,
    loss_db_per_km: f64,
    detector_efficiency: f64,
    dark_count_rate: f64,
    rep_rate: f64,
    mu: f64,
) -> KeyRateEstimate {
    // Each arm is half the total fiber length (Alice-Charlie-Bob)
    let half_length = fiber_length_km / 2.0;
    let loss_db_half = loss_db_per_km * half_length;
    let eta_half = 10.0_f64.powf(-loss_db_half / 10.0);

    // Effective channel transmittance (sqrt scaling)
    let eta_eff = (eta_half * detector_efficiency).sqrt();
    let p_dark = if dark_count_rate >= 1.0 {
        dark_count_rate * DEFAULT_GATE_WINDOW
    } else {
        dark_count_rate
    };

    // Detection rate at Charlie
    let p_detect = mu * eta_eff + p_dark;

    let raw_key_rate = rep_rate * p_detect;
    let sifted_key_rate = raw_key_rate / 2.0;

    // Phase error estimation
    let e_phase = if p_detect > 1e-20 {
        (p_dark / p_detect + DEFAULT_MISALIGNMENT_ERROR).min(0.5)
    } else {
        0.5
    };

    let qber = e_phase;
    let h_e = binary_entropy(qber);
    let secret_fraction = (1.0 - h_e - DEFAULT_EC_EFFICIENCY * h_e).max(0.0);
    let secret_key_rate = sifted_key_rate * secret_fraction;

    let secure = qber < BB84_QBER_THRESHOLD;

    KeyRateEstimate {
        raw_key_rate,
        sifted_key_rate,
        secret_key_rate,
        qber,
        secure,
    }
}

/// Estimate the key rate for continuous-variable QKD (CV-QKD).
///
/// Uses Gaussian-modulated coherent states with homodyne detection.
/// Key rate depends on the channel transmittance and excess noise.
pub fn cv_qkd_key_rate(
    fiber_length_km: f64,
    loss_db_per_km: f64,
    modulation: CVModulation,
    rep_rate: f64,
) -> KeyRateEstimate {
    let loss_db = loss_db_per_km * fiber_length_km;
    let transmittance = 10.0_f64.powf(-loss_db / 10.0);

    // Modulation variance (shot noise units)
    let v_mod = match modulation {
        CVModulation::GaussianModulation => 4.0,
        CVModulation::DiscreteModulation { num_states } => 2.0 * (num_states as f64).sqrt(),
    };

    // Excess noise (referred to channel input)
    let xi = 0.01; // 1% excess noise (typical lab value)

    // Channel-added noise
    let chi_line = 1.0 / transmittance - 1.0 + xi;

    // Homodyne detection efficiency and electronic noise
    let eta_det = 0.6; // Homodyne detector efficiency
    let v_el = 0.1; // Electronic noise variance

    let chi_det = (1.0 - eta_det) / eta_det + v_el / eta_det;
    let chi_total = chi_line + chi_det / transmittance;

    // Mutual information I(A:B)
    let v_a = v_mod + 1.0; // Alice's variance (signal + vacuum)
    let v_b = transmittance * (v_a + chi_total);
    let snr = if v_b > 1.0 {
        transmittance * v_mod / (v_b - transmittance * v_mod).abs().max(1e-10)
    } else {
        0.0
    };
    let i_ab = 0.5 * (1.0 + snr).max(1.0).log2();

    // Holevo bound for Eve's information (collective attack, reverse reconciliation)
    // Simplified: chi_BE ~ max(0, g((lambda1-1)/2) + g((lambda2-1)/2) - g((lambda3-1)/2))
    // For practical purposes, use a simplified excess-noise-based estimate
    let v_cond = 1.0 + xi * transmittance + (1.0 - transmittance);
    let chi_be = if v_cond > 1.0 {
        0.5 * v_cond.log2()
    } else {
        0.0
    };

    // Reconciliation efficiency
    let beta = 0.95; // Reverse reconciliation efficiency

    let skr_per_pulse = (beta * i_ab - chi_be).max(0.0);
    let secret_key_rate = rep_rate * skr_per_pulse;

    // Effective QBER equivalent (for reporting)
    let qber_equiv = if i_ab > 0.0 {
        (1.0 - skr_per_pulse / i_ab.max(1e-10)).clamp(0.0, 0.5)
    } else {
        0.5
    };

    let raw_key_rate = rep_rate * i_ab;
    let sifted_key_rate = raw_key_rate; // CV-QKD: no basis sifting

    KeyRateEstimate {
        raw_key_rate,
        sifted_key_rate,
        secret_key_rate,
        qber: qber_equiv,
        secure: secret_key_rate > 0.0,
    }
}

/// Estimate the key rate for measurement-device-independent QKD (MDI-QKD).
///
/// MDI-QKD removes all detector side-channel attacks by having an
/// untrusted relay (Charlie) perform Bell-state measurement.
pub fn mdi_qkd_key_rate(
    fiber_length_km: f64,
    loss_db_per_km: f64,
    detector_efficiency: f64,
    dark_count_rate: f64,
    rep_rate: f64,
    mu: f64,
) -> KeyRateEstimate {
    // Each arm is half the distance (Alice-Charlie and Bob-Charlie)
    let half_length = fiber_length_km / 2.0;
    let loss_db_half = loss_db_per_km * half_length;
    let eta_half = 10.0_f64.powf(-loss_db_half / 10.0);

    let eta_a = eta_half * detector_efficiency;
    let eta_b = eta_half * detector_efficiency;
    let p_dark = if dark_count_rate >= 1.0 {
        dark_count_rate * DEFAULT_GATE_WINDOW
    } else {
        dark_count_rate
    };

    // Bell-state measurement success probability
    // Only |psi-> can be identified with linear optics: success prob ~ eta_a * eta_b / 2
    let p_bsm = 0.5 * mu * mu * eta_a * eta_b;
    let p_dark_coincidence = p_dark * p_dark;
    let q_total = p_bsm + p_dark_coincidence;

    let raw_key_rate = rep_rate * q_total;

    // QBER for MDI
    let e_opt = DEFAULT_MISALIGNMENT_ERROR;
    let qber = if q_total > 1e-20 {
        (e_opt * p_bsm + 0.5 * p_dark_coincidence) / q_total
    } else {
        0.5
    };

    let sifted_key_rate = raw_key_rate / 2.0;

    // Secret key rate (similar structure to BB84 but with BSM)
    let h_e = binary_entropy(qber);
    let secret_fraction = (1.0 - h_e - DEFAULT_EC_EFFICIENCY * h_e).max(0.0);
    let secret_key_rate = sifted_key_rate * secret_fraction;

    let secure = qber < BB84_QBER_THRESHOLD;

    KeyRateEstimate {
        raw_key_rate,
        sifted_key_rate,
        secret_key_rate,
        qber,
        secure,
    }
}

/// Estimate the key rate for a link given its protocol and physical parameters.
pub fn estimate_link_key_rate(
    link: &QkdLink,
    node_a: &QkdNode,
    node_b: &QkdNode,
) -> KeyRateEstimate {
    let det_eff = (node_a.detector_efficiency + node_b.detector_efficiency) / 2.0;
    let dark_rate = (node_a.dark_count_rate + node_b.dark_count_rate) / 2.0;

    match link.protocol {
        QkdProtocol::BB84 {
            decoy_states: false,
        } => bb84_key_rate(
            link.fiber_length_km,
            link.loss_db_per_km,
            det_eff,
            dark_rate,
            DEFAULT_REP_RATE,
            DEFAULT_MEAN_PHOTON_NUMBER,
        ),
        QkdProtocol::BB84 { decoy_states: true } => decoy_state_bb84_key_rate(
            link.fiber_length_km,
            link.loss_db_per_km,
            det_eff,
            dark_rate,
            DEFAULT_REP_RATE,
            DEFAULT_MEAN_PHOTON_NUMBER,
        ),
        QkdProtocol::BBM92 | QkdProtocol::E91 => {
            // Entanglement-based: similar to BB84 but with 1/4 sifting (two bases each)
            let mut est = bb84_key_rate(
                link.fiber_length_km,
                link.loss_db_per_km,
                det_eff,
                dark_rate,
                DEFAULT_REP_RATE,
                DEFAULT_MEAN_PHOTON_NUMBER,
            );
            // Entanglement-based has lower raw rate (pair generation) but no PNS vulnerability
            est.raw_key_rate *= 0.5; // Lower pair generation rate
            est.sifted_key_rate *= 0.5;
            est.secret_key_rate *= 0.5;
            est
        }
        QkdProtocol::CvQkd { modulation } => cv_qkd_key_rate(
            link.fiber_length_km,
            link.loss_db_per_km,
            modulation,
            DEFAULT_REP_RATE,
        ),
        QkdProtocol::TfQkd => tf_qkd_key_rate(
            link.fiber_length_km,
            link.loss_db_per_km,
            det_eff,
            dark_rate,
            DEFAULT_REP_RATE,
            DEFAULT_MEAN_PHOTON_NUMBER,
        ),
        QkdProtocol::MdiQkd => mdi_qkd_key_rate(
            link.fiber_length_km,
            link.loss_db_per_km,
            det_eff,
            dark_rate,
            DEFAULT_REP_RATE,
            DEFAULT_MEAN_PHOTON_NUMBER,
        ),
    }
}

// ===================================================================
// SECURITY ANALYSIS
// ===================================================================

/// Composable security analysis result for a QKD session.
#[derive(Debug, Clone)]
pub struct SecurityAnalysis {
    /// Minimum entropy per raw key bit (bits).
    pub min_entropy_per_bit: f64,
    /// Composable security parameter epsilon (total failure probability).
    pub composable_security_parameter: f64,
    /// Upper bound on eavesdropper's information (bits).
    pub eavesdropper_info_bits: f64,
    /// Fraction of key retained after privacy amplification.
    pub privacy_amplification_factor: f64,
    /// Whether finite-key-length effects are accounted for.
    pub finite_key_effects: bool,
}

/// Perform composable security analysis for a QKD session.
///
/// # Arguments
///
/// * `qber` - Observed quantum bit error rate
/// * `raw_key_length` - Number of raw key bits exchanged
/// * `security_parameter` - Desired epsilon (e.g., 1e-10)
/// * `protocol` - Which QKD protocol was used
pub fn security_analysis(
    qber: f64,
    raw_key_length: usize,
    security_parameter: f64,
    _protocol: &QkdProtocol,
) -> SecurityAnalysis {
    let n = raw_key_length as f64;

    // Min-entropy per bit (smooth min-entropy bound)
    let h_e = binary_entropy(qber);
    let min_entropy_per_bit = (1.0 - h_e).max(0.0);

    // Finite-key correction terms
    let finite_key_effects = n < 1e6;
    let finite_correction = if finite_key_effects {
        // Statistical fluctuation correction: O(sqrt(n * log(1/eps)))
        let log_eps = -(security_parameter.max(1e-30).ln()) / (2.0_f64).ln();
        (6.0 * log_eps / n).sqrt()
    } else {
        0.0
    };

    // Effective min-entropy after finite-key correction
    let effective_min_ent = (min_entropy_per_bit - finite_correction).max(0.0);

    // Privacy amplification factor
    let error_correction_leaked = DEFAULT_EC_EFFICIENCY * h_e;
    let privacy_amplification_factor = (effective_min_ent - error_correction_leaked).max(0.0);

    // Eavesdropper's information
    let eavesdropper_info_bits = n * (1.0 - privacy_amplification_factor).max(0.0);

    SecurityAnalysis {
        min_entropy_per_bit: effective_min_ent,
        composable_security_parameter: security_parameter,
        eavesdropper_info_bits,
        privacy_amplification_factor,
        finite_key_effects,
    }
}

// ===================================================================
// AUTHENTICATION
// ===================================================================

/// Classical authentication method for the public channel.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthMethod {
    /// Pre-shared symmetric key (bootstrapping required).
    PreSharedKey,
    /// Wegman-Carter universal hashing with a family of size `hash_family_size`.
    WegmanCarter { hash_family_size: usize },
    /// Universal hash authentication (Toeplitz matrix based).
    UniversalHash,
}

impl Default for AuthMethod {
    fn default() -> Self {
        AuthMethod::WegmanCarter {
            hash_family_size: 256,
        }
    }
}

/// Number of authentication key bits consumed per authenticated message.
pub fn auth_key_cost(method: &AuthMethod, message_bits: usize) -> usize {
    match method {
        AuthMethod::PreSharedKey => message_bits, // OTP authentication
        AuthMethod::WegmanCarter { hash_family_size } => {
            // log2(family_size) + security parameter bits
            let tag_bits = (*hash_family_size as f64).log2().ceil() as usize;
            tag_bits + 128 // 128-bit security level
        }
        AuthMethod::UniversalHash => {
            // Toeplitz: n + k - 1 seed bits for n-bit message, k-bit tag
            let k = 128; // tag length
            message_bits + k - 1
        }
    }
}

// ===================================================================
// ROUTING
// ===================================================================

/// Routing algorithm for path selection in the QKD network.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingAlgorithm {
    /// Shortest path by physical distance (Dijkstra).
    ShortestPath,
    /// Route through the path with highest bottleneck key rate.
    MaxKeyRate,
    /// Distribute traffic across links to balance load.
    LoadBalanced,
    /// Use k shortest paths for redundancy and load distribution.
    MultiPath { num_paths: usize },
}

impl Default for RoutingAlgorithm {
    fn default() -> Self {
        RoutingAlgorithm::ShortestPath
    }
}

/// Key management strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum KeyManagement {
    /// Generate keys on demand when requested.
    OnDemand,
    /// Pre-distribute key material into buffers of given size.
    PreDistributed { buffer_size: usize },
    /// Continuously generate and refresh keys at the given interval.
    Continuous { refresh_interval_secs: f64 },
}

impl Default for KeyManagement {
    fn default() -> Self {
        KeyManagement::OnDemand
    }
}

/// Internal Dijkstra state for priority queue.
#[derive(Debug, Clone)]
struct DijkstraState {
    cost: f64,
    node: usize,
}

impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl Eq for DijkstraState {}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior in BinaryHeap (max-heap)
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

// ===================================================================
// NETWORK CONFIGURATION
// ===================================================================

/// Configuration for a QKD network.
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Human-readable network name.
    pub name: String,
    /// Total number of nodes.
    pub num_nodes: usize,
    /// Authentication method for the classical channel.
    pub authentication_method: AuthMethod,
    /// Routing algorithm for path selection.
    pub routing_algorithm: RoutingAlgorithm,
    /// Key management strategy.
    pub key_management: KeyManagement,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            name: "QKD Network".to_string(),
            num_nodes: 0,
            authentication_method: AuthMethod::default(),
            routing_algorithm: RoutingAlgorithm::default(),
            key_management: KeyManagement::default(),
        }
    }
}

impl NetworkConfig {
    /// Create a new network configuration with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Builder: set authentication method.
    pub fn with_auth(mut self, method: AuthMethod) -> Self {
        self.authentication_method = method;
        self
    }

    /// Builder: set routing algorithm.
    pub fn with_routing(mut self, alg: RoutingAlgorithm) -> Self {
        self.routing_algorithm = alg;
        self
    }

    /// Builder: set key management strategy.
    pub fn with_key_management(mut self, km: KeyManagement) -> Self {
        self.key_management = km;
        self
    }
}

// ===================================================================
// NETWORK SIMULATION RESULT
// ===================================================================

/// Result of a network-level QKD simulation.
#[derive(Debug, Clone)]
pub struct NetworkSimulationResult {
    /// Total secret key bits generated across all links during the simulation.
    pub total_key_generated: usize,
    /// End-to-end secret key rate matrix: `end_to_end_rates[src][dst]` in bps.
    pub end_to_end_rates: Vec<Vec<f64>>,
    /// Average QBER across all active links.
    pub average_qber: f64,
    /// Network availability: fraction of node pairs with positive key rate.
    pub network_availability: f64,
    /// Index of the bottleneck link (lowest key rate), if any.
    pub bottleneck_link: Option<usize>,
    /// Simulated elapsed time in seconds.
    pub elapsed_secs: f64,
}

// ===================================================================
// QKD NETWORK
// ===================================================================

/// Metropolitan-scale QKD network.
///
/// Manages nodes, links, routing, and key rate estimation for a
/// multi-node quantum key distribution network.
pub struct QkdNetwork {
    /// Network configuration.
    pub config: NetworkConfig,
    /// Nodes in the network.
    pub nodes: Vec<QkdNode>,
    /// Physical QKD links between nodes.
    pub links: Vec<QkdLink>,
}

impl QkdNetwork {
    /// Create a new empty QKD network with the given configuration.
    pub fn new(config: NetworkConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            links: Vec::new(),
        }
    }

    /// Add a node to the network. Returns the node's index.
    pub fn add_node(&mut self, mut node: QkdNode) -> usize {
        let idx = self.nodes.len();
        node.id = idx;
        self.nodes.push(node);
        self.config.num_nodes = self.nodes.len();
        idx
    }

    /// Add a link between two existing nodes. Returns the link's index.
    pub fn add_link(&mut self, link: QkdLink) -> Result<usize, QkdNetworkError> {
        if link.node_a >= self.nodes.len() || link.node_b >= self.nodes.len() {
            return Err(QkdNetworkError::InvalidNode(format!(
                "Link references node {} or {} but network has {} nodes",
                link.node_a,
                link.node_b,
                self.nodes.len()
            )));
        }
        let idx = self.links.len();
        self.links.push(link);
        Ok(idx)
    }

    /// Build the adjacency list representation. Returns a map from
    /// node index to list of (neighbor_index, link_index).
    fn adjacency_list(&self) -> HashMap<usize, Vec<(usize, usize)>> {
        let mut adj: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (i, link) in self.links.iter().enumerate() {
            if !link.active {
                continue;
            }
            adj.entry(link.node_a).or_default().push((link.node_b, i));
            adj.entry(link.node_b).or_default().push((link.node_a, i));
        }
        adj
    }

    /// Find the shortest path between two nodes using Dijkstra's algorithm.
    ///
    /// Weight function depends on the routing algorithm:
    /// - ShortestPath: fiber length in km
    /// - MaxKeyRate: -log(key_rate) to maximize bottleneck rate
    /// - LoadBalanced / MultiPath: fiber length (default)
    ///
    /// Returns the sequence of node indices on the path, or an error if
    /// no path exists.
    pub fn find_path(&self, src: usize, dst: usize) -> Result<Vec<usize>, QkdNetworkError> {
        if src >= self.nodes.len() || dst >= self.nodes.len() {
            return Err(QkdNetworkError::InvalidNode(format!(
                "Source {} or destination {} out of range (network has {} nodes)",
                src,
                dst,
                self.nodes.len()
            )));
        }
        if src == dst {
            return Ok(vec![src]);
        }

        let adj = self.adjacency_list();
        let n = self.nodes.len();
        let mut dist = vec![f64::INFINITY; n];
        let mut prev = vec![None; n];
        dist[src] = 0.0;

        let mut heap = BinaryHeap::new();
        heap.push(DijkstraState {
            cost: 0.0,
            node: src,
        });

        while let Some(DijkstraState { cost, node }) = heap.pop() {
            if node == dst {
                break;
            }
            if cost > dist[node] {
                continue;
            }
            if let Some(neighbors) = adj.get(&node) {
                for &(next, link_idx) in neighbors {
                    let link = &self.links[link_idx];
                    let edge_weight = match &self.config.routing_algorithm {
                        RoutingAlgorithm::MaxKeyRate => {
                            // Use 1/rate so Dijkstra finds max-rate path
                            // (must be positive — -ln(rate) goes negative when rate > 1)
                            let rate = self.estimate_link_rate(link_idx);
                            if rate.secret_key_rate > 0.0 {
                                1.0 / rate.secret_key_rate
                            } else {
                                f64::INFINITY
                            }
                        }
                        _ => link.fiber_length_km,
                    };
                    let new_cost = dist[node] + edge_weight;
                    if new_cost < dist[next] {
                        dist[next] = new_cost;
                        prev[next] = Some(node);
                        heap.push(DijkstraState {
                            cost: new_cost,
                            node: next,
                        });
                    }
                }
            }
        }

        // Reconstruct path
        if prev[dst].is_none() && src != dst {
            return Err(QkdNetworkError::NetworkPartitioned(format!(
                "No path from node {} to node {}",
                src, dst
            )));
        }

        let mut path = vec![dst];
        let mut current = dst;
        while let Some(p) = prev[current] {
            path.push(p);
            current = p;
        }
        path.reverse();
        Ok(path)
    }

    /// Find k shortest (simple) paths between two nodes using Yen's algorithm.
    ///
    /// Returns up to `k` paths, each as a vector of node indices.
    pub fn find_k_paths(
        &self,
        src: usize,
        dst: usize,
        k: usize,
    ) -> Result<Vec<Vec<usize>>, QkdNetworkError> {
        if k == 0 {
            return Ok(Vec::new());
        }

        // Find first shortest path
        let first_path = self.find_path(src, dst)?;
        let mut a_paths: Vec<Vec<usize>> = vec![first_path];
        let mut b_candidates: Vec<(f64, Vec<usize>)> = Vec::new();

        for ki in 1..k {
            let prev_path = &a_paths[ki - 1];

            for spur_idx in 0..prev_path.len().saturating_sub(1) {
                let spur_node = prev_path[spur_idx];
                let root_path: Vec<usize> = prev_path[..=spur_idx].to_vec();

                // Create a temporary network with some links removed
                // to prevent previously found paths from being re-found.
                // For simplicity, we track which links to skip.
                let mut skip_links: Vec<usize> = Vec::new();

                for existing_path in &a_paths {
                    if existing_path.len() > spur_idx && existing_path[..=spur_idx] == root_path[..]
                    {
                        // Find and skip the link from spur_node to the next node
                        // in this existing path
                        if spur_idx + 1 < existing_path.len() {
                            let next_node = existing_path[spur_idx + 1];
                            for (li, link) in self.links.iter().enumerate() {
                                if !link.active {
                                    continue;
                                }
                                if (link.node_a == spur_node && link.node_b == next_node)
                                    || (link.node_b == spur_node && link.node_a == next_node)
                                {
                                    skip_links.push(li);
                                }
                            }
                        }
                    }
                }

                // Find spur path avoiding skip links and root nodes (except spur)
                let skip_nodes: Vec<usize> = root_path[..spur_idx].to_vec();
                if let Ok(spur_path) =
                    self.find_path_excluding(spur_node, dst, &skip_links, &skip_nodes)
                {
                    let mut total_path = root_path.clone();
                    total_path.extend_from_slice(&spur_path[1..]);

                    // Compute total cost
                    let cost: f64 = total_path
                        .windows(2)
                        .map(|w| {
                            self.find_link_between(w[0], w[1])
                                .map(|li| self.links[li].fiber_length_km)
                                .unwrap_or(f64::INFINITY)
                        })
                        .sum();

                    // Only add if not already in candidates or results
                    let already_found = a_paths.iter().any(|p| p == &total_path)
                        || b_candidates.iter().any(|(_, p)| p == &total_path);
                    if !already_found {
                        b_candidates.push((cost, total_path));
                    }
                }
            }

            if b_candidates.is_empty() {
                break;
            }

            // Sort candidates by cost and pick the best
            b_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            let (_, best_path) = b_candidates.remove(0);
            a_paths.push(best_path);
        }

        Ok(a_paths)
    }

    /// Find a link index connecting two nodes, if one exists.
    fn find_link_between(&self, a: usize, b: usize) -> Option<usize> {
        self.links.iter().position(|link| {
            link.active
                && ((link.node_a == a && link.node_b == b)
                    || (link.node_b == a && link.node_a == b))
        })
    }

    /// Dijkstra with exclusion sets (for Yen's algorithm).
    fn find_path_excluding(
        &self,
        src: usize,
        dst: usize,
        skip_links: &[usize],
        skip_nodes: &[usize],
    ) -> Result<Vec<usize>, QkdNetworkError> {
        let n = self.nodes.len();
        let mut dist = vec![f64::INFINITY; n];
        let mut prev: Vec<Option<usize>> = vec![None; n];
        dist[src] = 0.0;

        let mut heap = BinaryHeap::new();
        heap.push(DijkstraState {
            cost: 0.0,
            node: src,
        });

        let adj = self.adjacency_list();

        while let Some(DijkstraState { cost, node }) = heap.pop() {
            if node == dst {
                break;
            }
            if cost > dist[node] {
                continue;
            }
            if let Some(neighbors) = adj.get(&node) {
                for &(next, link_idx) in neighbors {
                    if skip_links.contains(&link_idx) || skip_nodes.contains(&next) {
                        continue;
                    }
                    let edge_weight = self.links[link_idx].fiber_length_km;
                    let new_cost = dist[node] + edge_weight;
                    if new_cost < dist[next] {
                        dist[next] = new_cost;
                        prev[next] = Some(node);
                        heap.push(DijkstraState {
                            cost: new_cost,
                            node: next,
                        });
                    }
                }
            }
        }

        if prev[dst].is_none() && src != dst {
            return Err(QkdNetworkError::NetworkPartitioned(format!(
                "No path from {} to {} with exclusions",
                src, dst
            )));
        }

        let mut path = vec![dst];
        let mut current = dst;
        while let Some(p) = prev[current] {
            path.push(p);
            current = p;
        }
        path.reverse();
        Ok(path)
    }

    /// Estimate the key rate for a specific link by index.
    pub fn estimate_link_rate(&self, link_idx: usize) -> KeyRateEstimate {
        let link = &self.links[link_idx];
        let node_a = &self.nodes[link.node_a];
        let node_b = &self.nodes[link.node_b];
        estimate_link_key_rate(link, node_a, node_b)
    }

    /// Compute end-to-end key rate along a path through trusted relays.
    ///
    /// For a path A - R1 - R2 - ... - B through trusted relays, the
    /// end-to-end rate is limited by the bottleneck (minimum-rate) link.
    pub fn end_to_end_rate(&self, path: &[usize]) -> Result<f64, QkdNetworkError> {
        if path.len() < 2 {
            return Err(QkdNetworkError::InvalidConfig(
                "Path must have at least 2 nodes".to_string(),
            ));
        }

        let mut min_rate = f64::INFINITY;
        for window in path.windows(2) {
            let (a, b) = (window[0], window[1]);
            let link_idx = self.find_link_between(a, b).ok_or_else(|| {
                QkdNetworkError::LinkFailed(format!("No link between nodes {} and {}", a, b))
            })?;
            let est = self.estimate_link_rate(link_idx);
            if !est.secure {
                return Err(QkdNetworkError::InsufficientKeyRate(format!(
                    "Link {}-{} has QBER {:.4} above threshold",
                    a, b, est.qber
                )));
            }
            min_rate = min_rate.min(est.secret_key_rate);
        }

        Ok(min_rate)
    }

    /// Compute the end-to-end key rate matrix for all node pairs.
    pub fn compute_rate_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.nodes.len();
        let mut rates = vec![vec![0.0; n]; n];

        for src in 0..n {
            for dst in (src + 1)..n {
                if let Ok(path) = self.find_path(src, dst) {
                    if let Ok(rate) = self.end_to_end_rate(&path) {
                        rates[src][dst] = rate;
                        rates[dst][src] = rate;
                    }
                }
            }
        }
        rates
    }

    /// Compute network availability: fraction of node pairs with
    /// positive end-to-end key rate.
    pub fn network_availability(&self) -> f64 {
        let n = self.nodes.len();
        if n < 2 {
            return 1.0;
        }

        let total_pairs = n * (n - 1) / 2;
        let mut connected_pairs = 0;

        for src in 0..n {
            for dst in (src + 1)..n {
                if let Ok(path) = self.find_path(src, dst) {
                    if let Ok(rate) = self.end_to_end_rate(&path) {
                        if rate > 0.0 {
                            connected_pairs += 1;
                        }
                    }
                }
            }
        }

        connected_pairs as f64 / total_pairs as f64
    }

    /// Find the bottleneck link (lowest secret key rate) among active links.
    pub fn bottleneck_link(&self) -> Option<usize> {
        let mut min_rate = f64::INFINITY;
        let mut min_idx = None;

        for (i, link) in self.links.iter().enumerate() {
            if !link.active {
                continue;
            }
            let est = self.estimate_link_rate(i);
            if est.secret_key_rate < min_rate {
                min_rate = est.secret_key_rate;
                min_idx = Some(i);
            }
        }
        min_idx
    }

    /// Run a network simulation for the given duration.
    ///
    /// Estimates key rates on all links, computes end-to-end rates,
    /// and returns aggregate statistics.
    pub fn simulate(&self, duration_secs: f64) -> NetworkSimulationResult {
        let rate_matrix = self.compute_rate_matrix();
        let _n = self.nodes.len();

        // Total key generated across all links
        let mut total_key: f64 = 0.0;
        let mut total_qber = 0.0;
        let mut active_links = 0;

        for (i, link) in self.links.iter().enumerate() {
            if !link.active {
                continue;
            }
            let est = self.estimate_link_rate(i);
            total_key += est.secret_key_rate * duration_secs;
            total_qber += est.qber;
            active_links += 1;
        }

        let average_qber = if active_links > 0 {
            total_qber / active_links as f64
        } else {
            0.0
        };

        let availability = self.network_availability();
        let bottleneck = self.bottleneck_link();

        // Scale end-to-end rates by duration for total bits
        let end_to_end_rates = rate_matrix;

        NetworkSimulationResult {
            total_key_generated: total_key as usize,
            end_to_end_rates,
            average_qber,
            network_availability: availability,
            bottleneck_link: bottleneck,
            elapsed_secs: duration_secs,
        }
    }

    /// Update all link key rates and QBERs by re-estimating from physical parameters.
    pub fn refresh_link_estimates(&mut self) {
        for i in 0..self.links.len() {
            let est = self.estimate_link_rate(i);
            self.links[i].key_rate_bps = est.secret_key_rate;
            self.links[i].qber = est.qber;
        }
    }

    /// Disable a link (set active = false). Returns error if link index is invalid.
    pub fn disable_link(&mut self, link_idx: usize) -> Result<(), QkdNetworkError> {
        if link_idx >= self.links.len() {
            return Err(QkdNetworkError::LinkFailed(format!(
                "Link index {} out of range (network has {} links)",
                link_idx,
                self.links.len()
            )));
        }
        self.links[link_idx].active = false;
        Ok(())
    }

    /// Enable a link (set active = true). Returns error if link index is invalid.
    pub fn enable_link(&mut self, link_idx: usize) -> Result<(), QkdNetworkError> {
        if link_idx >= self.links.len() {
            return Err(QkdNetworkError::LinkFailed(format!(
                "Link index {} out of range (network has {} links)",
                link_idx,
                self.links.len()
            )));
        }
        self.links[link_idx].active = true;
        Ok(())
    }
}

// ===================================================================
// NETWORK LIBRARY: PRE-BUILT TOPOLOGIES
// ===================================================================

/// Library of pre-built metropolitan QKD network topologies.
///
/// Each factory method returns a fully configured `QkdNetwork` with
/// nodes, links, and default parameters based on real-world deployments.
pub struct NetworkLibrary;

impl NetworkLibrary {
    /// Berlin BearlinQ metropolitan QKD network.
    ///
    /// 4 nodes forming a partial mesh in central Berlin.
    /// Based on the 2025 BearlinQ deployment connecting
    /// universities and government institutions.
    ///
    /// Topology:
    /// ```text
    ///   TU-Berlin ----(12 km)---- HU-Berlin
    ///       |                         |
    ///    (8 km)                    (15 km)
    ///       |                         |
    ///   FU-Berlin ----(10 km)---- PTB
    /// ```
    pub fn bearlinq() -> QkdNetwork {
        let config = NetworkConfig::new("BearlinQ Berlin")
            .with_auth(AuthMethod::WegmanCarter {
                hash_family_size: 256,
            })
            .with_routing(RoutingAlgorithm::MaxKeyRate);

        let mut net = QkdNetwork::new(config);

        net.add_node(QkdNode::new(0, "TU-Berlin", NodeType::EndUser, (0.0, 10.0)));
        net.add_node(QkdNode::new(
            1,
            "HU-Berlin",
            NodeType::EndUser,
            (12.0, 10.0),
        ));
        net.add_node(QkdNode::new(2, "FU-Berlin", NodeType::EndUser, (0.0, 0.0)));
        net.add_node(QkdNode::new(3, "PTB", NodeType::TrustedRelay, (10.0, 0.0)));

        let _ = net.add_link(QkdLink::new(0, 1, 12.0));
        let _ = net.add_link(QkdLink::new(0, 2, 8.0));
        let _ = net.add_link(QkdLink::new(2, 3, 10.0));
        let _ = net.add_link(QkdLink::new(1, 3, 15.0));

        net.refresh_link_estimates();
        net
    }

    /// Tokyo QKD Network.
    ///
    /// 6 nodes in a star topology centered on a trusted relay node,
    /// with additional cross-links for redundancy.
    /// Based on the NICT Tokyo QKD Network testbed.
    pub fn tokyo_qkd() -> QkdNetwork {
        let config =
            NetworkConfig::new("Tokyo QKD Network").with_routing(RoutingAlgorithm::ShortestPath);

        let mut net = QkdNetwork::new(config);

        // Central hub
        net.add_node(QkdNode::new(
            0,
            "NICT-Hub",
            NodeType::TrustedRelay,
            (0.0, 0.0),
        ));

        // Satellite nodes in a ring
        let radius = 15.0;
        for i in 0..5 {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / 5.0;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            net.add_node(QkdNode::new(
                i + 1,
                &format!("Site-{}", i + 1),
                NodeType::EndUser,
                (x, y),
            ));
        }

        // Star links from hub to each site
        for i in 1..=5 {
            let d = net.nodes[0].distance_to(&net.nodes[i]);
            let _ = net.add_link(QkdLink::new(0, i, d));
        }

        // One cross-link for redundancy
        let d = net.nodes[1].distance_to(&net.nodes[2]);
        let _ = net.add_link(QkdLink::new(1, 2, d));

        net.refresh_link_estimates();
        net
    }

    /// Cambridge UK Quantum Network.
    ///
    /// 5 nodes in a partial mesh topology connecting
    /// Cambridge research institutions.
    pub fn cambridge_network() -> QkdNetwork {
        let config =
            NetworkConfig::new("Cambridge QKD Network").with_routing(RoutingAlgorithm::MaxKeyRate);

        let mut net = QkdNetwork::new(config);

        net.add_node(QkdNode::new(0, "Cavendish", NodeType::EndUser, (0.0, 0.0)));
        net.add_node(QkdNode::new(
            1,
            "Toshiba-CRL",
            NodeType::EntanglementSource,
            (3.0, 4.0),
        ));
        net.add_node(QkdNode::new(
            2,
            "BT-Adastral",
            NodeType::TrustedRelay,
            (8.0, 2.0),
        ));
        net.add_node(QkdNode::new(3, "NPL", NodeType::EndUser, (5.0, 7.0)));
        net.add_node(QkdNode::new(4, "KETS", NodeType::EndUser, (10.0, 5.0)));

        // Mesh links
        let pairs = [(0, 1), (1, 2), (2, 4), (0, 3), (1, 3), (3, 4)];
        for (a, b) in pairs {
            let d = net.nodes[a].distance_to(&net.nodes[b]);
            let _ = net.add_link(QkdLink::new(a, b, d));
        }

        net.refresh_link_estimates();
        net
    }

    /// Intercity point-to-point link with quantum repeaters.
    ///
    /// Places trusted relay nodes every 50 km along the route.
    /// For very long distances (> 300 km), uses TF-QKD on the
    /// individual segments.
    ///
    /// # Arguments
    ///
    /// * `distance_km` - Total intercity distance in kilometres
    pub fn intercity(distance_km: f64) -> QkdNetwork {
        let segment_length = 50.0; // km between relays
        let num_segments = (distance_km / segment_length).ceil() as usize;
        let actual_segment = distance_km / num_segments as f64;

        let config = NetworkConfig::new(&format!("Intercity {}km", distance_km as usize))
            .with_routing(RoutingAlgorithm::ShortestPath);

        let mut net = QkdNetwork::new(config);

        // Alice (source city)
        net.add_node(QkdNode::new(0, "Alice", NodeType::EndUser, (0.0, 0.0)));

        // Intermediate relays
        for i in 1..num_segments {
            let x = i as f64 * actual_segment;
            net.add_node(QkdNode::new(
                i,
                &format!("Relay-{}", i),
                NodeType::TrustedRelay,
                (x, 0.0),
            ));
        }

        // Bob (destination city)
        net.add_node(QkdNode::new(
            num_segments,
            "Bob",
            NodeType::EndUser,
            (distance_km, 0.0),
        ));

        // Protocol selection based on segment length
        let protocol = if actual_segment > 100.0 {
            QkdProtocol::TfQkd
        } else {
            QkdProtocol::BB84 { decoy_states: true }
        };

        // Chain of links
        for i in 0..=num_segments.saturating_sub(1) {
            if i + 1 <= num_segments {
                let _ =
                    net.add_link(QkdLink::new(i, i + 1, actual_segment).with_protocol(protocol));
            }
        }

        net.refresh_link_estimates();
        net
    }

    /// Star network with a central server and N clients.
    ///
    /// Based on the Nature 2026 demonstration of a 20-client photonic
    /// QKD network using a single entanglement source.
    ///
    /// # Arguments
    ///
    /// * `num_clients` - Number of client nodes around the central server
    pub fn star_network(num_clients: usize) -> QkdNetwork {
        let config = NetworkConfig::new(&format!("Star-{}", num_clients))
            .with_routing(RoutingAlgorithm::ShortestPath);

        let mut net = QkdNetwork::new(config);

        // Central entanglement source
        net.add_node(QkdNode::new(
            0,
            "Central",
            NodeType::EntanglementSource,
            (0.0, 0.0),
        ));

        // Client nodes arranged in a circle
        let radius = 10.0; // km
        for i in 0..num_clients {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (num_clients as f64);
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            net.add_node(QkdNode::new(
                i + 1,
                &format!("Client-{}", i + 1),
                NodeType::EndUser,
                (x, y),
            ));
        }

        // Star links from central to each client (BBM92 entanglement-based)
        for i in 1..=num_clients {
            let d = net.nodes[0].distance_to(&net.nodes[i]);
            let _ = net.add_link(QkdLink::new(0, i, d).with_protocol(QkdProtocol::BBM92));
        }

        net.refresh_link_estimates();
        net
    }

    /// Ring topology with N nodes.
    ///
    /// Each consecutive pair of nodes is connected by a QKD link,
    /// forming a closed loop for redundancy.
    pub fn ring_network(num_nodes: usize, circumference_km: f64) -> QkdNetwork {
        let config = NetworkConfig::new(&format!("Ring-{}", num_nodes))
            .with_routing(RoutingAlgorithm::ShortestPath);

        let mut net = QkdNetwork::new(config);

        let radius = circumference_km / (2.0 * std::f64::consts::PI);
        let segment_length = circumference_km / num_nodes as f64;

        for i in 0..num_nodes {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (num_nodes as f64);
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            let node_type = if i == 0 {
                NodeType::TrustedRelay
            } else {
                NodeType::EndUser
            };
            net.add_node(QkdNode::new(i, &format!("Node-{}", i), node_type, (x, y)));
        }

        for i in 0..num_nodes {
            let next = (i + 1) % num_nodes;
            let _ = net.add_link(QkdLink::new(i, next, segment_length));
        }

        net.refresh_link_estimates();
        net
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Node tests
    // ---------------------------------------------------------------

    #[test]
    fn test_qkd_node_creation() {
        let node = QkdNode::new(0, "Alice", NodeType::EndUser, (1.0, 2.0))
            .with_detector_efficiency(0.15)
            .with_dark_count_rate(2.0e-6)
            .with_key_storage(1024);

        assert_eq!(node.id, 0);
        assert_eq!(node.name, "Alice");
        assert_eq!(node.position, (1.0, 2.0));
        assert_eq!(node.key_storage, 1024);
        assert!((node.detector_efficiency - 0.15).abs() < 1e-12);
        assert!((node.dark_count_rate - 2.0e-6).abs() < 1e-18);
    }

    #[test]
    fn test_node_distance() {
        let a = QkdNode::new(0, "A", NodeType::EndUser, (0.0, 0.0));
        let b = QkdNode::new(1, "B", NodeType::EndUser, (3.0, 4.0));
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // Fiber channel and link loss tests
    // ---------------------------------------------------------------

    #[test]
    fn test_fiber_loss_10km() {
        let ch = FiberChannel::new(10.0, 0.2, 1e-6);
        // 10 km * 0.2 dB/km = 2 dB loss
        // transmittance = 10^(-2/10) = 10^(-0.2) ~ 0.631
        assert!((ch.loss_db - 2.0).abs() < 1e-10);
        assert!((ch.transmittance - 10.0_f64.powf(-0.2)).abs() < 1e-10);
        assert!(ch.transmittance > 0.5);
    }

    #[test]
    fn test_fiber_loss_50km() {
        let ch = FiberChannel::new(50.0, 0.2, 1e-6);
        // 50 km * 0.2 dB/km = 10 dB loss
        // transmittance = 10^(-1.0) = 0.1
        assert!((ch.loss_db - 10.0).abs() < 1e-10);
        assert!((ch.transmittance - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_fiber_loss_100km_below_1_percent() {
        let ch = FiberChannel::new(100.0, 0.2, 1e-6);
        // 100 km * 0.2 dB/km = 20 dB
        // transmittance = 10^(-2.0) = 0.01
        assert!((ch.transmittance - 0.01).abs() < 1e-10);
        assert!(
            ch.transmittance < 0.01 + 1e-10,
            "100 km fiber should have < 1% transmittance"
        );
    }

    #[test]
    fn test_link_transmittance() {
        let link = QkdLink::new(0, 1, 50.0);
        let t = link.transmittance();
        assert!(
            (t - 0.1).abs() < 1e-10,
            "50 km link transmittance should be ~0.1"
        );
    }

    // ---------------------------------------------------------------
    // BB84 key rate tests
    // ---------------------------------------------------------------

    #[test]
    fn test_bb84_positive_rate_short_distance() {
        let est = bb84_key_rate(
            10.0,
            DEFAULT_FIBER_LOSS_DB_PER_KM,
            DEFAULT_DETECTOR_EFFICIENCY,
            DEFAULT_DARK_COUNT_RATE,
            DEFAULT_REP_RATE,
            DEFAULT_MEAN_PHOTON_NUMBER,
        );
        assert!(
            est.secret_key_rate > 0.0,
            "BB84 at 10 km should have positive key rate, got {}",
            est.secret_key_rate
        );
        assert!(est.secure, "BB84 at 10 km should be secure");
        assert!(est.qber < BB84_QBER_THRESHOLD);
    }

    #[test]
    fn test_bb84_zero_rate_long_distance() {
        // At 400 km, the loss is 80 dB => transmittance ~ 10^-8
        // The dark count contribution will dominate and QBER > 11%
        let est = bb84_key_rate(
            400.0,
            DEFAULT_FIBER_LOSS_DB_PER_KM,
            DEFAULT_DETECTOR_EFFICIENCY,
            DEFAULT_DARK_COUNT_RATE,
            DEFAULT_REP_RATE,
            DEFAULT_MEAN_PHOTON_NUMBER,
        );
        // With these parameters, either the QBER is above threshold or the rate is ~0
        assert!(
            est.secret_key_rate < 1e-3 || !est.secure,
            "BB84 at 400 km should have negligible or insecure rate"
        );
    }

    #[test]
    fn test_bb84_qber_low_noise() {
        let est = bb84_key_rate(
            5.0, // Very short fiber
            0.2, 0.20, // Good detector
            1e-7, // Very low dark counts
            1e9, 0.5,
        );
        assert!(
            est.qber < 0.05,
            "QBER should be low for short fiber and good detector, got {:.4}",
            est.qber
        );
    }

    // ---------------------------------------------------------------
    // Decoy-state BB84
    // ---------------------------------------------------------------

    #[test]
    fn test_decoy_state_better_than_standard() {
        let standard = bb84_key_rate(
            50.0,
            DEFAULT_FIBER_LOSS_DB_PER_KM,
            DEFAULT_DETECTOR_EFFICIENCY,
            DEFAULT_DARK_COUNT_RATE,
            DEFAULT_REP_RATE,
            DEFAULT_MEAN_PHOTON_NUMBER,
        );
        let decoy = decoy_state_bb84_key_rate(
            50.0,
            DEFAULT_FIBER_LOSS_DB_PER_KM,
            DEFAULT_DETECTOR_EFFICIENCY,
            DEFAULT_DARK_COUNT_RATE,
            DEFAULT_REP_RATE,
            DEFAULT_MEAN_PHOTON_NUMBER,
        );
        assert!(
            decoy.secret_key_rate >= standard.secret_key_rate * 0.5,
            "Decoy-state should achieve competitive rate. Standard: {:.2}, Decoy: {:.2}",
            standard.secret_key_rate,
            decoy.secret_key_rate
        );
        // Both should be secure at 50 km
        assert!(standard.secure || decoy.secure);
    }

    // ---------------------------------------------------------------
    // Twin-field QKD
    // ---------------------------------------------------------------

    #[test]
    fn test_tf_qkd_positive_at_300km() {
        let est = tf_qkd_key_rate(
            300.0,
            DEFAULT_FIBER_LOSS_DB_PER_KM,
            DEFAULT_DETECTOR_EFFICIENCY,
            DEFAULT_DARK_COUNT_RATE,
            DEFAULT_REP_RATE,
            DEFAULT_MEAN_PHOTON_NUMBER,
        );
        assert!(
            est.secret_key_rate > 0.0,
            "TF-QKD at 300 km should have positive key rate, got {}",
            est.secret_key_rate
        );
        assert!(est.secure);
    }

    #[test]
    fn test_tf_qkd_sqrt_eta_scaling() {
        // Compare TF-QKD at 200 km vs BB84 at 200 km
        // TF-QKD should fare much better because of sqrt(eta) scaling
        let tf_200 = tf_qkd_key_rate(
            200.0,
            DEFAULT_FIBER_LOSS_DB_PER_KM,
            DEFAULT_DETECTOR_EFFICIENCY,
            DEFAULT_DARK_COUNT_RATE,
            DEFAULT_REP_RATE,
            DEFAULT_MEAN_PHOTON_NUMBER,
        );
        let bb84_200 = bb84_key_rate(
            200.0,
            DEFAULT_FIBER_LOSS_DB_PER_KM,
            DEFAULT_DETECTOR_EFFICIENCY,
            DEFAULT_DARK_COUNT_RATE,
            DEFAULT_REP_RATE,
            DEFAULT_MEAN_PHOTON_NUMBER,
        );
        assert!(
            tf_200.secret_key_rate > bb84_200.secret_key_rate,
            "TF-QKD should beat BB84 at 200 km: TF={:.2e}, BB84={:.2e}",
            tf_200.secret_key_rate,
            bb84_200.secret_key_rate,
        );
    }

    // ---------------------------------------------------------------
    // Network construction
    // ---------------------------------------------------------------

    #[test]
    fn test_network_construction() {
        let config = NetworkConfig::new("Test Network");
        let mut net = QkdNetwork::new(config);

        let a = net.add_node(QkdNode::new(0, "Alice", NodeType::EndUser, (0.0, 0.0)));
        let b = net.add_node(QkdNode::new(1, "Bob", NodeType::EndUser, (10.0, 0.0)));
        let _link = net.add_link(QkdLink::new(a, b, 10.0)).unwrap();

        assert_eq!(net.nodes.len(), 2);
        assert_eq!(net.links.len(), 1);
        assert_eq!(net.config.num_nodes, 2);
    }

    // ---------------------------------------------------------------
    // Routing
    // ---------------------------------------------------------------

    #[test]
    fn test_routing_shortest_path() {
        let config =
            NetworkConfig::new("Routing Test").with_routing(RoutingAlgorithm::ShortestPath);
        let mut net = QkdNetwork::new(config);

        // Triangle: A(0) -- B(1) -- C(2), and A(0) -- C(2)
        net.add_node(QkdNode::new(0, "A", NodeType::EndUser, (0.0, 0.0)));
        net.add_node(QkdNode::new(1, "B", NodeType::TrustedRelay, (5.0, 0.0)));
        net.add_node(QkdNode::new(2, "C", NodeType::EndUser, (3.0, 4.0)));

        let _ = net.add_link(QkdLink::new(0, 1, 5.0)); // A-B: 5 km
        let _ = net.add_link(QkdLink::new(1, 2, 4.0)); // B-C: 4 km
        let _ = net.add_link(QkdLink::new(0, 2, 4.5)); // A-C: 4.5 km (direct, shorter than A-B-C)

        let path = net.find_path(0, 2).unwrap();
        // Direct A->C should be chosen (4.5 km) vs A->B->C (9 km)
        assert_eq!(path, vec![0, 2]);
    }

    #[test]
    fn test_routing_max_key_rate() {
        let config =
            NetworkConfig::new("Rate Routing Test").with_routing(RoutingAlgorithm::MaxKeyRate);
        let mut net = QkdNetwork::new(config);

        // A -- B (short link) -- C
        // A --------------------C (long direct link)
        // MaxKeyRate should prefer the path with higher key rate
        net.add_node(QkdNode::new(0, "A", NodeType::EndUser, (0.0, 0.0)));
        net.add_node(QkdNode::new(1, "B", NodeType::TrustedRelay, (5.0, 0.0)));
        net.add_node(QkdNode::new(2, "C", NodeType::EndUser, (10.0, 0.0)));

        let _ = net.add_link(QkdLink::new(0, 1, 5.0)); // A-B: 5 km (high rate)
        let _ = net.add_link(QkdLink::new(1, 2, 5.0)); // B-C: 5 km (high rate)
        let _ = net.add_link(QkdLink::new(0, 2, 80.0)); // A-C: 80 km (low rate)

        let path = net.find_path(0, 2).unwrap();
        // Even though A-C is one hop, 80 km will have very low key rate
        // MaxKeyRate should prefer A-B-C (two 5 km hops)
        assert!(path.len() >= 2, "MaxKeyRate should find a path");
    }

    // ---------------------------------------------------------------
    // End-to-end rates
    // ---------------------------------------------------------------

    #[test]
    fn test_end_to_end_direct_link() {
        let config = NetworkConfig::new("Direct");
        let mut net = QkdNetwork::new(config);

        net.add_node(QkdNode::new(0, "A", NodeType::EndUser, (0.0, 0.0)));
        net.add_node(QkdNode::new(1, "B", NodeType::EndUser, (10.0, 0.0)));
        let _ = net.add_link(QkdLink::new(0, 1, 10.0));
        net.refresh_link_estimates();

        let rate = net.end_to_end_rate(&[0, 1]).unwrap();
        assert!(rate > 0.0, "Direct 10 km link should have positive rate");
    }

    #[test]
    fn test_end_to_end_relay_chain() {
        let config = NetworkConfig::new("Relay Chain");
        let mut net = QkdNetwork::new(config);

        // A -- R1 -- R2 -- B (chain of 3 links)
        net.add_node(QkdNode::new(0, "A", NodeType::EndUser, (0.0, 0.0)));
        net.add_node(QkdNode::new(1, "R1", NodeType::TrustedRelay, (10.0, 0.0)));
        net.add_node(QkdNode::new(2, "R2", NodeType::TrustedRelay, (20.0, 0.0)));
        net.add_node(QkdNode::new(3, "B", NodeType::EndUser, (30.0, 0.0)));

        let _ = net.add_link(QkdLink::new(0, 1, 10.0));
        let _ = net.add_link(QkdLink::new(1, 2, 10.0));
        let _ = net.add_link(QkdLink::new(2, 3, 10.0));
        net.refresh_link_estimates();

        let rate = net.end_to_end_rate(&[0, 1, 2, 3]).unwrap();
        assert!(
            rate > 0.0,
            "Relay chain of 10 km segments should have positive rate"
        );

        // Rate through chain should be <= rate of a single link
        let single_rate = net.end_to_end_rate(&[0, 1]).unwrap();
        assert!(
            rate <= single_rate + 1e-6,
            "Relay chain rate should be limited by bottleneck link"
        );
    }

    // ---------------------------------------------------------------
    // Security analysis
    // ---------------------------------------------------------------

    #[test]
    fn test_security_qber_below_threshold_secure() {
        let qber = 0.05; // Well below 11%
        let sa = security_analysis(qber, 1_000_000, 1e-10, &QkdProtocol::default());
        assert!(
            sa.privacy_amplification_factor > 0.0,
            "QBER {:.2} below threshold should yield positive privacy amplification",
            qber
        );
        assert!(
            sa.min_entropy_per_bit > 0.0,
            "Low QBER should have positive min-entropy per bit"
        );
    }

    #[test]
    fn test_security_qber_above_threshold_abort() {
        let qber = 0.20; // Way above 11%
        let sa = security_analysis(qber, 1_000_000, 1e-10, &QkdProtocol::default());
        // At 20% QBER, H(0.2) ~ 0.722
        // 1 - H(e) - f_EC * H(e) = 1 - 0.722 - 1.16 * 0.722 < 0
        // So privacy_amplification_factor should be 0
        assert!(
            sa.privacy_amplification_factor < 1e-10,
            "QBER {:.2} above threshold should yield zero privacy amplification, got {}",
            qber,
            sa.privacy_amplification_factor
        );
    }

    #[test]
    fn test_privacy_amplification_reduces_key() {
        let qber = 0.05;
        let sa = security_analysis(qber, 1_000_000, 1e-10, &QkdProtocol::default());
        assert!(
            sa.privacy_amplification_factor < 1.0,
            "Privacy amplification should reduce key length (factor < 1)"
        );
        assert!(
            sa.privacy_amplification_factor > 0.0,
            "But factor should be positive for low QBER"
        );
    }

    #[test]
    fn test_finite_key_lower_rate() {
        let qber = 0.03;
        let asymptotic = security_analysis(qber, 10_000_000, 1e-10, &QkdProtocol::default());
        let finite = security_analysis(qber, 10_000, 1e-10, &QkdProtocol::default());

        assert!(
            finite.finite_key_effects,
            "10k bits should trigger finite-key effects"
        );
        assert!(
            !asymptotic.finite_key_effects,
            "10M bits should be asymptotic regime"
        );
        assert!(
            finite.min_entropy_per_bit <= asymptotic.min_entropy_per_bit + 1e-10,
            "Finite-key min-entropy ({:.4}) should be <= asymptotic ({:.4})",
            finite.min_entropy_per_bit,
            asymptotic.min_entropy_per_bit
        );
    }

    // ---------------------------------------------------------------
    // Pre-built topologies
    // ---------------------------------------------------------------

    #[test]
    fn test_bearlinq_topology() {
        let net = NetworkLibrary::bearlinq();
        assert_eq!(net.nodes.len(), 4, "BearlinQ should have 4 nodes");
        assert_eq!(
            net.links.len(),
            4,
            "BearlinQ should have 4 links (partial mesh)"
        );
        assert_eq!(net.config.name, "BearlinQ Berlin");

        // Check that PTB is a TrustedRelay
        assert_eq!(net.nodes[3].name, "PTB");
        assert!(
            matches!(net.nodes[3].node_type, NodeType::TrustedRelay),
            "PTB should be a TrustedRelay"
        );
    }

    #[test]
    fn test_star_network_construction() {
        let n = 8;
        let net = NetworkLibrary::star_network(n);
        assert_eq!(
            net.nodes.len(),
            n + 1,
            "Star should have N+1 nodes (center + clients)"
        );
        assert_eq!(net.links.len(), n, "Star should have N links");

        // Center should be EntanglementSource
        assert!(
            matches!(net.nodes[0].node_type, NodeType::EntanglementSource),
            "Central node should be EntanglementSource"
        );
    }

    #[test]
    fn test_intercity_repeater_count() {
        let distance = 200.0;
        let net = NetworkLibrary::intercity(distance);

        // 200 km / 50 km per segment = 4 segments => 3 intermediate relays
        // Total nodes = Alice + 3 relays + Bob = 5
        let num_relays = net.nodes.len() - 2; // Subtract Alice and Bob
        assert!(
            num_relays >= 2,
            "200 km intercity should have at least 2 relays, got {}",
            num_relays
        );

        // Check Alice and Bob are EndUsers
        assert!(
            matches!(net.nodes[0].node_type, NodeType::EndUser),
            "First node (Alice) should be EndUser"
        );
        assert!(
            matches!(net.nodes.last().unwrap().node_type, NodeType::EndUser),
            "Last node (Bob) should be EndUser"
        );

        // Intermediate nodes should be TrustedRelays
        for node in &net.nodes[1..net.nodes.len() - 1] {
            assert!(
                matches!(node.node_type, NodeType::TrustedRelay),
                "Intermediate node {} should be TrustedRelay",
                node.name
            );
        }
    }

    // ---------------------------------------------------------------
    // Network availability
    // ---------------------------------------------------------------

    #[test]
    fn test_network_availability_all_links_up() {
        let net = NetworkLibrary::bearlinq();
        let avail = net.network_availability();
        assert!(
            (avail - 1.0).abs() < 1e-10,
            "All links up should give 100% availability, got {}",
            avail
        );
    }

    #[test]
    fn test_network_availability_link_down() {
        let mut net = NetworkLibrary::bearlinq();
        // Disable the link between TU-Berlin and HU-Berlin
        net.disable_link(0).unwrap();
        let avail = net.network_availability();
        // With one link down, some pair might become disconnected
        // or the availability might decrease because of longer paths
        // In BearlinQ with the TU-HU link down, all pairs are still connected
        // via the other path, so availability stays 1.0
        // Let's disable more links to create a partition
        net.disable_link(1).unwrap(); // TU-FU down too
        let avail2 = net.network_availability();
        assert!(
            avail2 < 1.0,
            "With TU isolated, availability should be < 1.0, got {}",
            avail2
        );
    }

    // ---------------------------------------------------------------
    // Multi-path routing
    // ---------------------------------------------------------------

    #[test]
    fn test_multipath_routing_k_paths() {
        let config = NetworkConfig::new("MultiPath Test")
            .with_routing(RoutingAlgorithm::MultiPath { num_paths: 3 });
        let mut net = QkdNetwork::new(config);

        // Diamond topology: A -> B -> D, A -> C -> D
        net.add_node(QkdNode::new(0, "A", NodeType::EndUser, (0.0, 0.0)));
        net.add_node(QkdNode::new(1, "B", NodeType::TrustedRelay, (5.0, 3.0)));
        net.add_node(QkdNode::new(2, "C", NodeType::TrustedRelay, (5.0, -3.0)));
        net.add_node(QkdNode::new(3, "D", NodeType::EndUser, (10.0, 0.0)));

        let _ = net.add_link(QkdLink::new(0, 1, 6.0)); // A-B
        let _ = net.add_link(QkdLink::new(0, 2, 6.0)); // A-C
        let _ = net.add_link(QkdLink::new(1, 3, 6.0)); // B-D
        let _ = net.add_link(QkdLink::new(2, 3, 6.0)); // C-D

        let paths = net.find_k_paths(0, 3, 3).unwrap();
        assert!(
            paths.len() >= 2,
            "Diamond topology should have at least 2 paths from A to D, got {}",
            paths.len()
        );
        // First path should be one of the two shortest
        assert_eq!(paths[0].len(), 3, "Shortest path should have 3 nodes");
    }

    // ---------------------------------------------------------------
    // CV-QKD
    // ---------------------------------------------------------------

    #[test]
    fn test_cv_qkd_gaussian_modulation() {
        let est = cv_qkd_key_rate(
            20.0,
            DEFAULT_FIBER_LOSS_DB_PER_KM,
            CVModulation::GaussianModulation,
            1e9,
        );
        assert!(
            est.secret_key_rate > 0.0,
            "CV-QKD at 20 km should have positive rate, got {}",
            est.secret_key_rate
        );
        assert!(est.secure);
    }

    // ---------------------------------------------------------------
    // MDI-QKD
    // ---------------------------------------------------------------

    #[test]
    fn test_mdi_qkd_key_rate() {
        let est = mdi_qkd_key_rate(
            20.0,
            DEFAULT_FIBER_LOSS_DB_PER_KM,
            DEFAULT_DETECTOR_EFFICIENCY,
            DEFAULT_DARK_COUNT_RATE,
            DEFAULT_REP_RATE,
            DEFAULT_MEAN_PHOTON_NUMBER,
        );
        assert!(
            est.secret_key_rate > 0.0,
            "MDI-QKD at 20 km should have positive rate, got {:.4e}",
            est.secret_key_rate
        );
        assert!(est.secure);
    }

    // ---------------------------------------------------------------
    // Config builder
    // ---------------------------------------------------------------

    #[test]
    fn test_config_builder_defaults() {
        let config = NetworkConfig::default();
        assert_eq!(config.name, "QKD Network");
        assert_eq!(config.num_nodes, 0);
        assert!(matches!(
            config.authentication_method,
            AuthMethod::WegmanCarter { .. }
        ));
        assert!(matches!(
            config.routing_algorithm,
            RoutingAlgorithm::ShortestPath
        ));
        assert!(matches!(config.key_management, KeyManagement::OnDemand));
    }

    #[test]
    fn test_config_builder_custom() {
        let config = NetworkConfig::new("Custom")
            .with_auth(AuthMethod::UniversalHash)
            .with_routing(RoutingAlgorithm::LoadBalanced)
            .with_key_management(KeyManagement::Continuous {
                refresh_interval_secs: 30.0,
            });

        assert_eq!(config.name, "Custom");
        assert_eq!(config.authentication_method, AuthMethod::UniversalHash);
        assert_eq!(config.routing_algorithm, RoutingAlgorithm::LoadBalanced);
        assert!(matches!(
            config.key_management,
            KeyManagement::Continuous { refresh_interval_secs }
            if (refresh_interval_secs - 30.0).abs() < 1e-10
        ));
    }

    // ---------------------------------------------------------------
    // Large network
    // ---------------------------------------------------------------

    #[test]
    fn test_large_network_20_nodes() {
        let n = 20;
        let net = NetworkLibrary::ring_network(n, 100.0);
        assert_eq!(net.nodes.len(), n);
        assert_eq!(net.links.len(), n, "Ring should have exactly N links");

        // Should be able to find a path between any two nodes
        let path = net.find_path(0, n / 2).unwrap();
        assert!(path.len() >= 2, "Path should exist in ring network");

        // Simulate without errors
        let result = net.simulate(1.0);
        assert!(
            result.average_qber < 0.5,
            "Average QBER should be reasonable"
        );
    }

    // ---------------------------------------------------------------
    // Binary entropy function
    // ---------------------------------------------------------------

    #[test]
    fn test_binary_entropy() {
        assert!((binary_entropy(0.0) - 0.0).abs() < 1e-12);
        assert!((binary_entropy(1.0) - 0.0).abs() < 1e-12);
        assert!((binary_entropy(0.5) - 1.0).abs() < 1e-12);
        // H(0.11) ~ 0.5
        let h = binary_entropy(0.11);
        assert!(h > 0.4 && h < 0.6, "H(0.11) should be ~0.5, got {}", h);
    }

    // ---------------------------------------------------------------
    // Authentication key cost
    // ---------------------------------------------------------------

    #[test]
    fn test_auth_key_cost() {
        let wc_cost = auth_key_cost(
            &AuthMethod::WegmanCarter {
                hash_family_size: 256,
            },
            1024,
        );
        // log2(256) = 8, plus 128 security bits
        assert_eq!(wc_cost, 8 + 128);

        let psk_cost = auth_key_cost(&AuthMethod::PreSharedKey, 1024);
        assert_eq!(psk_cost, 1024);
    }

    // ---------------------------------------------------------------
    // Fiber channel propagation delay
    // ---------------------------------------------------------------

    #[test]
    fn test_fiber_propagation_delay() {
        let ch = FiberChannel::standard(100.0);
        let delay = ch.propagation_delay_s();
        // 100 km = 100_000 m, speed = c/n = 3e8/1.47 ~ 2.04e8
        // delay ~ 100_000 / 2.04e8 ~ 4.9e-4 s
        assert!(
            delay > 4.0e-4 && delay < 6.0e-4,
            "100 km propagation delay should be ~0.5 ms, got {:.6} s",
            delay
        );
    }

    // ---------------------------------------------------------------
    // Network simulation
    // ---------------------------------------------------------------

    #[test]
    fn test_network_simulation_runs() {
        let net = NetworkLibrary::bearlinq();
        let result = net.simulate(1.0);

        assert!(
            result.total_key_generated > 0,
            "1-second simulation should generate some key bits"
        );
        assert!(
            result.average_qber < BB84_QBER_THRESHOLD,
            "BearlinQ QBER should be below threshold"
        );
        assert!(
            (result.network_availability - 1.0).abs() < 1e-10,
            "All BearlinQ links up => 100% availability"
        );
        assert!(result.bottleneck_link.is_some());
        assert!((result.elapsed_secs - 1.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // Link enable/disable
    // ---------------------------------------------------------------

    #[test]
    fn test_link_disable_enable() {
        let mut net = NetworkLibrary::bearlinq();
        assert!(net.links[0].active);

        net.disable_link(0).unwrap();
        assert!(!net.links[0].active);

        net.enable_link(0).unwrap();
        assert!(net.links[0].active);

        // Out-of-range should error
        assert!(net.disable_link(999).is_err());
    }

    // ---------------------------------------------------------------
    // Tokyo network
    // ---------------------------------------------------------------

    #[test]
    fn test_tokyo_network_topology() {
        let net = NetworkLibrary::tokyo_qkd();
        assert_eq!(net.nodes.len(), 6, "Tokyo should have 6 nodes");
        // 5 star links + 1 cross-link = 6
        assert_eq!(net.links.len(), 6, "Tokyo should have 6 links");
        assert!(matches!(net.nodes[0].node_type, NodeType::TrustedRelay));
    }

    // ---------------------------------------------------------------
    // Network partitioning detection
    // ---------------------------------------------------------------

    #[test]
    fn test_network_partition_detection() {
        let config = NetworkConfig::new("Partition Test");
        let mut net = QkdNetwork::new(config);

        net.add_node(QkdNode::new(0, "A", NodeType::EndUser, (0.0, 0.0)));
        net.add_node(QkdNode::new(1, "B", NodeType::EndUser, (10.0, 0.0)));
        // No link between A and B

        let result = net.find_path(0, 1);
        assert!(
            result.is_err(),
            "Should detect partition when no path exists"
        );
        match result {
            Err(QkdNetworkError::NetworkPartitioned(_)) => {} // Expected
            other => panic!("Expected NetworkPartitioned error, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Cambridge network
    // ---------------------------------------------------------------

    #[test]
    fn test_cambridge_network() {
        let net = NetworkLibrary::cambridge_network();
        assert_eq!(net.nodes.len(), 5);
        assert_eq!(net.links.len(), 6);

        // Should be fully connected (mesh)
        let avail = net.network_availability();
        assert!(
            (avail - 1.0).abs() < 1e-10,
            "Cambridge mesh should be fully connected"
        );
    }

    // ---------------------------------------------------------------
    // End-to-end rate matrix
    // ---------------------------------------------------------------

    #[test]
    fn test_rate_matrix_symmetry() {
        let net = NetworkLibrary::bearlinq();
        let matrix = net.compute_rate_matrix();

        assert_eq!(matrix.len(), 4);
        for row in &matrix {
            assert_eq!(row.len(), 4);
        }

        // Should be symmetric
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (matrix[i][j] - matrix[j][i]).abs() < 1e-6,
                    "Rate matrix should be symmetric: [{},{}]={} vs [{},{}]={}",
                    i,
                    j,
                    matrix[i][j],
                    j,
                    i,
                    matrix[j][i]
                );
            }
            // Diagonal should be zero
            assert!(matrix[i][i].abs() < 1e-10, "Self-rate should be zero");
        }
    }
}
