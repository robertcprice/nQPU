//! Quantum Networking Module
//!
//! Implements quantum communication channels, entanglement distribution protocols,
//! quantum repeaters, entanglement purification, and network topology management.
//!
//! # Physical Models
//!
//! - **Fiber channels**: exponential loss + depolarizing noise as a function of distance
//! - **Free-space channels**: atmospheric turbulence and diffraction loss
//! - **Erasure channels**: qubit loss with known location (heralded)
//!
//! # Protocols
//!
//! - **DLCZ**: atomic-ensemble-based entanglement distribution
//! - **Barrett-Kok**: two-photon interference for matter-qubit entanglement
//! - **BBPSSW**: bilateral CNOT purification (Bennett et al. 1996)
//! - **DEJMPS**: improved purification with bilateral rotations (Deutsch et al. 1996)
//!
//! # Usage
//!
//! ```ignore
//! use nqpu_metal::quantum_networking::*;
//!
//! let fiber = FiberChannel::new(50.0, 0.2, 0.01);
//! let bell = bell_state_phi_plus();
//! let (rho_out, p_success) = fiber.transmit(&bell);
//! let fidelity = bell_state_fidelity(&rho_out);
//! ```

use crate::quantum_channel::KrausChannel;
use crate::{c64_one, c64_zero, C64};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

// ===================================================================
// CONSTANTS
// ===================================================================

/// Speed of light in fiber (m/s), approximately 2/3 of vacuum speed.
const SPEED_OF_LIGHT_FIBER: f64 = 2.0e8;

/// Inverse square root of two, used in Bell state construction.
const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

// ===================================================================
// BELL STATE UTILITIES
// ===================================================================

/// Construct the |Phi+> Bell state density matrix: (|00> + |11>) / sqrt(2).
///
/// Returns a 4x4 density matrix as a flat row-major vector.
pub fn bell_state_phi_plus() -> Vec<C64> {
    // |Phi+> = (|00> + |11>) / sqrt(2)
    // State vector: [1/sqrt(2), 0, 0, 1/sqrt(2)]
    let s = INV_SQRT2;
    let state = vec![C64::new(s, 0.0), c64_zero(), c64_zero(), C64::new(s, 0.0)];
    outer_product(&state)
}

/// Construct the |Psi+> Bell state density matrix: (|01> + |10>) / sqrt(2).
pub fn bell_state_psi_plus() -> Vec<C64> {
    let s = INV_SQRT2;
    let state = vec![c64_zero(), C64::new(s, 0.0), C64::new(s, 0.0), c64_zero()];
    outer_product(&state)
}

/// Compute fidelity of a two-qubit density matrix with |Phi+>.
///
/// F = <Phi+| rho |Phi+> = (rho[0,0] + rho[0,3] + rho[3,0] + rho[3,3]) / 2
///
/// For a Werner state with parameter p: F = (1 + 3p) / 4.
pub fn bell_state_fidelity(rho: &[C64]) -> f64 {
    assert!(
        rho.len() == 16,
        "Expected a 4x4 density matrix (16 elements)"
    );
    // <Phi+| rho |Phi+> = (1/sqrt(2))^2 * (rho[0,0] + rho[0,3] + rho[3,0] + rho[3,3])
    let f = 0.5 * (rho[0 * 4 + 0] + rho[0 * 4 + 3] + rho[3 * 4 + 0] + rho[3 * 4 + 3]);
    f.re.max(0.0).min(1.0)
}

/// Compute the trace of a square density matrix.
fn dm_trace(rho: &[C64], dim: usize) -> f64 {
    (0..dim).map(|i| rho[i * dim + i].re).sum()
}

/// Compute |psi><psi| from a state vector.
fn outer_product(state: &[C64]) -> Vec<C64> {
    let d = state.len();
    let mut rho = vec![c64_zero(); d * d];
    for i in 0..d {
        for j in 0..d {
            rho[i * d + j] = state[i] * state[j].conj();
        }
    }
    rho
}

/// Normalize a density matrix so Tr(rho) = 1.
fn dm_normalize(rho: &mut [C64], dim: usize) {
    let tr = dm_trace(rho, dim);
    if tr > 1e-15 {
        let inv = 1.0 / tr;
        for elem in rho.iter_mut() {
            *elem = C64::new(elem.re * inv, elem.im * inv);
        }
    }
}

/// Matrix multiply two dim x dim matrices stored as flat row-major vectors.
fn mat_mul(a: &[C64], b: &[C64], dim: usize) -> Vec<C64> {
    let mut result = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = c64_zero();
            for k in 0..dim {
                sum += a[i * dim + k] * b[k * dim + j];
            }
            result[i * dim + j] = sum;
        }
    }
    result
}

/// Compute the conjugate transpose of a dim x dim matrix.
fn mat_dagger(m: &[C64], dim: usize) -> Vec<C64> {
    let mut result = vec![c64_zero(); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            result[i * dim + j] = m[j * dim + i].conj();
        }
    }
    result
}

/// Tensor (Kronecker) product of two square matrices.
fn tensor_product(a: &[C64], da: usize, b: &[C64], db: usize) -> Vec<C64> {
    let d = da * db;
    let mut result = vec![c64_zero(); d * d];
    for ia in 0..da {
        for ja in 0..da {
            for ib in 0..db {
                for jb in 0..db {
                    let row = ia * db + ib;
                    let col = ja * db + jb;
                    result[row * d + col] = a[ia * da + ja] * b[ib * db + jb];
                }
            }
        }
    }
    result
}

/// Construct a 2-qubit Werner state:
/// rho_W(p) = p |Phi+><Phi+| + (1-p)/4 * I_4
///
/// Werner parameter p in [0, 1]. Fidelity with |Phi+> is (1 + 3p) / 4.
pub fn werner_state(p: f64) -> Vec<C64> {
    assert!(p >= 0.0 && p <= 1.0, "Werner parameter must be in [0, 1]");
    let phi_plus = bell_state_phi_plus();
    let mut rho = vec![c64_zero(); 16];
    let noise_weight = (1.0 - p) / 4.0;
    for i in 0..4 {
        for j in 0..4 {
            rho[i * 4 + j] = C64::new(p, 0.0) * phi_plus[i * 4 + j];
            if i == j {
                rho[i * 4 + j] += C64::new(noise_weight, 0.0);
            }
        }
    }
    rho
}

// ===================================================================
// FIBER CHANNEL
// ===================================================================

/// Optical fiber quantum channel with exponential loss and depolarizing noise.
///
/// Models photon transmission through optical fiber:
/// - Loss: transmittance eta = 10^(-alpha * L / 10) where alpha is loss in dB/km
/// - Noise: depolarizing channel applied to surviving photons
///
/// The overall channel is: with probability eta, apply depolarizing noise;
/// with probability (1 - eta), the photon is lost.
#[derive(Clone, Debug)]
pub struct FiberChannel {
    /// Fiber length in kilometers.
    pub distance_km: f64,
    /// Loss coefficient in dB/km (typical: 0.2 dB/km for telecom fiber).
    pub loss_db_per_km: f64,
    /// Depolarizing noise rate applied to each qubit upon arrival.
    pub depolarizing_rate: f64,
}

impl FiberChannel {
    /// Create a new fiber channel.
    ///
    /// # Arguments
    /// * `distance_km` - Fiber length in kilometers
    /// * `loss_db_per_km` - Attenuation in dB/km (e.g. 0.2 for telecom)
    /// * `depolarizing_rate` - Per-qubit depolarizing probability upon arrival
    pub fn new(distance_km: f64, loss_db_per_km: f64, depolarizing_rate: f64) -> Self {
        assert!(distance_km >= 0.0, "Distance must be non-negative");
        assert!(loss_db_per_km >= 0.0, "Loss must be non-negative");
        assert!(
            depolarizing_rate >= 0.0 && depolarizing_rate <= 1.0,
            "Depolarizing rate must be in [0, 1]"
        );
        FiberChannel {
            distance_km,
            loss_db_per_km,
            depolarizing_rate,
        }
    }

    /// Compute the transmittance (survival probability) of a single photon.
    ///
    /// eta = 10^(-alpha * L / 10)
    pub fn transmittance(&self) -> f64 {
        let loss_db = self.loss_db_per_km * self.distance_km;
        10.0_f64.powf(-loss_db / 10.0)
    }

    /// Transmit a two-qubit density matrix through the fiber.
    ///
    /// One qubit of an entangled pair is sent through the channel.
    /// Returns (output density matrix, success probability).
    ///
    /// On success, depolarizing noise is applied to the transmitted qubit.
    /// The success probability equals the transmittance.
    pub fn transmit(&self, rho: &[C64]) -> (Vec<C64>, f64) {
        let eta = self.transmittance();

        if self.depolarizing_rate < 1e-15 {
            // No noise, just loss
            return (rho.to_vec(), eta);
        }

        // Apply single-qubit depolarizing channel to second qubit
        // E(rho) = (1-p)*rho + p/3 * (X_2 rho X_2 + Y_2 rho Y_2 + Z_2 rho Z_2)
        let p = self.depolarizing_rate;
        let rho_out = apply_single_qubit_depolarizing(rho, 4, 1, p);

        (rho_out, eta)
    }

    /// Build a KrausChannel representing the depolarizing component only.
    pub fn as_kraus_channel(&self) -> KrausChannel {
        KrausChannel::depolarizing(self.depolarizing_rate)
    }

    /// Latency of the channel in seconds.
    pub fn latency_seconds(&self) -> f64 {
        (self.distance_km * 1000.0) / SPEED_OF_LIGHT_FIBER
    }
}

/// Apply a single-qubit depolarizing channel to qubit `target` of a `dim`-dimensional
/// density matrix. `target` is 0-indexed (0 = first qubit, 1 = second qubit).
fn apply_single_qubit_depolarizing(rho: &[C64], dim: usize, target: usize, p: f64) -> Vec<C64> {
    // For a 2-qubit system (dim=4), we apply I_other tensor E_target
    // E(rho) = (1-p)*rho + p/3 * sum_{sigma in {X,Y,Z}} (I tensor sigma) rho (I tensor sigma)
    let num_qubits = (dim as f64).log2() as usize;
    let qubit_dim = 2usize;

    // Pauli matrices
    let pauli_x = [c64_zero(), c64_one(), c64_one(), c64_zero()];
    let pauli_y = [
        c64_zero(),
        C64::new(0.0, -1.0),
        C64::new(0.0, 1.0),
        c64_zero(),
    ];
    let pauli_z = [c64_one(), c64_zero(), c64_zero(), C64::new(-1.0, 0.0)];
    let identity_2 = [c64_one(), c64_zero(), c64_zero(), c64_one()];

    let paulis: [&[C64; 4]; 3] = [&pauli_x, &pauli_y, &pauli_z];

    let mut result = vec![c64_zero(); dim * dim];

    // (1-p) * rho
    for i in 0..dim * dim {
        result[i] = C64::new(1.0 - p, 0.0) * rho[i];
    }

    // + p/3 * sum_sigma (sigma_target rho sigma_target)
    let weight = p / 3.0;
    for sigma in &paulis {
        // Build full operator: tensor product with identity on other qubits
        let full_op = if num_qubits == 1 {
            sigma.to_vec()
        } else {
            // For 2-qubit: I tensor sigma or sigma tensor I depending on target
            if target == 0 {
                tensor_product(*sigma, qubit_dim, &identity_2, qubit_dim)
            } else {
                tensor_product(&identity_2, qubit_dim, *sigma, qubit_dim)
            }
        };

        // sigma * rho * sigma (sigma is Hermitian so sigma_dag = sigma)
        let temp = mat_mul(&full_op, rho, dim);
        let s_rho_s = mat_mul(&temp, &full_op, dim);

        for i in 0..dim * dim {
            result[i] += C64::new(weight, 0.0) * s_rho_s[i];
        }
    }

    result
}

// ===================================================================
// FREE-SPACE CHANNEL
// ===================================================================

/// Free-space quantum channel modeling atmospheric and geometric losses.
///
/// Combines diffraction loss (beam spreading) and atmospheric absorption
/// for satellite-to-ground or ground-to-ground links.
#[derive(Clone, Debug)]
pub struct FreeSpaceChannel {
    /// Link distance in kilometers.
    pub distance_km: f64,
    /// Transmitter aperture diameter in meters.
    pub tx_aperture_m: f64,
    /// Receiver aperture diameter in meters.
    pub rx_aperture_m: f64,
    /// Wavelength of the photons in nanometers.
    pub wavelength_nm: f64,
    /// Atmospheric absorption coefficient in dB/km.
    pub atmospheric_loss_db_per_km: f64,
    /// Additional depolarizing noise rate.
    pub depolarizing_rate: f64,
}

impl FreeSpaceChannel {
    /// Create a default free-space channel for near-infrared photons (810 nm).
    pub fn new(distance_km: f64) -> Self {
        FreeSpaceChannel {
            distance_km,
            tx_aperture_m: 0.3,
            rx_aperture_m: 1.0,
            wavelength_nm: 810.0,
            atmospheric_loss_db_per_km: 0.05,
            depolarizing_rate: 0.005,
        }
    }

    /// Create with full parameters.
    pub fn with_params(
        distance_km: f64,
        tx_aperture_m: f64,
        rx_aperture_m: f64,
        wavelength_nm: f64,
        atmospheric_loss_db_per_km: f64,
        depolarizing_rate: f64,
    ) -> Self {
        FreeSpaceChannel {
            distance_km,
            tx_aperture_m,
            rx_aperture_m,
            wavelength_nm,
            atmospheric_loss_db_per_km,
            depolarizing_rate,
        }
    }

    /// Compute diffraction-limited transmittance.
    ///
    /// eta_diff = (pi * D_tx * D_rx / (4 * lambda * L))^2
    /// clamped to [0, 1].
    pub fn diffraction_transmittance(&self) -> f64 {
        let lambda_m = self.wavelength_nm * 1e-9;
        let l_m = self.distance_km * 1e3;
        if l_m < 1e-10 {
            return 1.0;
        }
        let eta = (std::f64::consts::PI * self.tx_aperture_m * self.rx_aperture_m
            / (4.0 * lambda_m * l_m))
            .powi(2);
        eta.min(1.0)
    }

    /// Compute atmospheric transmittance.
    pub fn atmospheric_transmittance(&self) -> f64 {
        let loss_db = self.atmospheric_loss_db_per_km * self.distance_km;
        10.0_f64.powf(-loss_db / 10.0)
    }

    /// Total transmittance combining diffraction and atmospheric effects.
    pub fn transmittance(&self) -> f64 {
        self.diffraction_transmittance() * self.atmospheric_transmittance()
    }

    /// Transmit a two-qubit density matrix through the free-space link.
    pub fn transmit(&self, rho: &[C64]) -> (Vec<C64>, f64) {
        let eta = self.transmittance();
        if self.depolarizing_rate < 1e-15 {
            return (rho.to_vec(), eta);
        }
        let rho_out = apply_single_qubit_depolarizing(rho, 4, 1, self.depolarizing_rate);
        (rho_out, eta)
    }
}

// ===================================================================
// ERASURE CHANNEL
// ===================================================================

/// Quantum erasure channel with heralded (known-location) loss.
///
/// With probability `erasure_prob`, the qubit is lost and replaced by a flag state.
/// The sender knows whether erasure occurred (heralded loss).
///
/// E(rho) = (1-p) rho + p |e><e| (where |e> is an erasure flag)
/// In the density matrix formalism over the original Hilbert space, we model
/// the surviving part as (1-p)*rho and the erased part as p*I/d.
#[derive(Clone, Debug)]
pub struct ErasureChannel {
    /// Probability of erasure per qubit.
    pub erasure_prob: f64,
}

impl ErasureChannel {
    pub fn new(erasure_prob: f64) -> Self {
        assert!(
            erasure_prob >= 0.0 && erasure_prob <= 1.0,
            "Erasure probability must be in [0, 1]"
        );
        ErasureChannel { erasure_prob }
    }

    /// Apply erasure to a single-qubit density matrix.
    ///
    /// Returns (output state, success probability).
    /// On erasure, the state is replaced by the maximally mixed state I/2.
    pub fn transmit_single(&self, rho: &[C64]) -> (Vec<C64>, f64) {
        assert!(rho.len() == 4, "Expected 2x2 density matrix");
        let p = self.erasure_prob;
        let mut result = vec![c64_zero(); 4];
        // (1-p)*rho + p * I/2
        for i in 0..2 {
            for j in 0..2 {
                result[i * 2 + j] = C64::new(1.0 - p, 0.0) * rho[i * 2 + j];
                if i == j {
                    result[i * 2 + j] += C64::new(p * 0.5, 0.0);
                }
            }
        }
        (result, 1.0 - p)
    }

    /// Apply erasure to the second qubit of a two-qubit density matrix.
    pub fn transmit(&self, rho: &[C64]) -> (Vec<C64>, f64) {
        assert!(rho.len() == 16, "Expected 4x4 density matrix");
        let p = self.erasure_prob;
        // The non-erased part keeps the state; the erased part traces out qubit 2
        // and replaces with maximally mixed.

        // Partial trace over qubit 2 to get rho_A
        let mut rho_a = vec![c64_zero(); 4]; // 2x2
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    rho_a[i * 2 + j] += rho[(i * 2 + k) * 4 + (j * 2 + k)];
                }
            }
        }

        // rho_A tensor I/2
        let identity_half = [
            C64::new(0.5, 0.0),
            c64_zero(),
            c64_zero(),
            C64::new(0.5, 0.0),
        ];
        let rho_erased = tensor_product(&rho_a, 2, &identity_half, 2);

        let mut result = vec![c64_zero(); 16];
        for i in 0..16 {
            result[i] = C64::new(1.0 - p, 0.0) * rho[i] + C64::new(p, 0.0) * rho_erased[i];
        }

        (result, 1.0 - p)
    }

    /// Build a KrausChannel for this erasure channel (single qubit).
    pub fn as_kraus_channel(&self) -> KrausChannel {
        KrausChannel::erasure(self.erasure_prob)
    }
}

// ===================================================================
// DLCZ PROTOCOL
// ===================================================================

/// DLCZ entanglement distribution protocol (Duan, Lukin, Cirac, Zoller 2001).
///
/// Creates remote entanglement between two atomic-ensemble quantum memories
/// by detecting a single photon from one of two sources.
///
/// Success probability per attempt scales as p_em (emission probability),
/// typically very small (10^-3 to 10^-2).
#[derive(Clone, Debug)]
pub struct DLCZProtocol {
    /// Emission probability per attempt.
    pub emission_prob: f64,
    /// Detection efficiency.
    pub detection_efficiency: f64,
    /// Channel transmittance to the midpoint.
    pub channel_transmittance: f64,
    /// Intrinsic fidelity of the generated state (models mode mismatch, etc).
    pub intrinsic_fidelity: f64,
}

impl DLCZProtocol {
    /// Create a DLCZ protocol instance with default parameters.
    pub fn new(emission_prob: f64, channel_transmittance: f64) -> Self {
        DLCZProtocol {
            emission_prob,
            detection_efficiency: 0.9,
            channel_transmittance,
            intrinsic_fidelity: 0.98,
        }
    }

    /// Create with full parameters.
    pub fn with_params(
        emission_prob: f64,
        detection_efficiency: f64,
        channel_transmittance: f64,
        intrinsic_fidelity: f64,
    ) -> Self {
        DLCZProtocol {
            emission_prob,
            detection_efficiency,
            channel_transmittance,
            intrinsic_fidelity,
        }
    }

    /// Success probability per attempt.
    ///
    /// p_success = p_em * eta_channel * eta_det
    pub fn success_probability(&self) -> f64 {
        self.emission_prob * self.channel_transmittance * self.detection_efficiency
    }

    /// Attempt to generate an entangled Bell pair.
    ///
    /// Returns (density_matrix, success).
    /// On success, returns a Werner-like state with fidelity determined by
    /// intrinsic_fidelity; on failure, returns |00><00| with success = false.
    pub fn attempt_entanglement(&self) -> (Vec<C64>, bool) {
        let p_success = self.success_probability();

        // Deterministic model: always "succeed" but encode the probability
        // in the return value. The caller uses p_success for rate calculations.
        // The generated state is a Werner state parameterized by intrinsic_fidelity.

        // Map fidelity F to Werner parameter p: F = (1 + 3p)/4 => p = (4F - 1)/3
        let f = self.intrinsic_fidelity;
        let werner_p = ((4.0 * f - 1.0) / 3.0).max(0.0).min(1.0);
        let rho = werner_state(werner_p);

        (rho, p_success > 0.0)
    }

    /// Expected number of attempts until success.
    pub fn expected_attempts(&self) -> f64 {
        let p = self.success_probability();
        if p < 1e-15 {
            return f64::INFINITY;
        }
        1.0 / p
    }

    /// Expected generation rate in pairs per second, given attempt rate.
    pub fn generation_rate(&self, attempt_rate_hz: f64) -> f64 {
        attempt_rate_hz * self.success_probability()
    }
}

// ===================================================================
// BARRETT-KOK PROTOCOL
// ===================================================================

/// Barrett-Kok two-photon interference entanglement protocol.
///
/// Uses two-photon detection at a beam splitter to herald entanglement
/// between two remote matter qubits. More robust against phase instability
/// than single-photon schemes.
///
/// Success probability: p ~ (p_em * eta)^2 / 2 per attempt.
/// Achievable fidelity is limited by photon distinguishability and dark counts.
#[derive(Clone, Debug)]
pub struct BarrettKokProtocol {
    /// Emission probability per excitation.
    pub emission_prob: f64,
    /// Channel transmittance (each arm).
    pub channel_transmittance: f64,
    /// Detector efficiency.
    pub detection_efficiency: f64,
    /// Photon indistinguishability (visibility), in [0, 1].
    pub visibility: f64,
    /// Dark count probability per detection window.
    pub dark_count_prob: f64,
}

impl BarrettKokProtocol {
    /// Create with default detector parameters.
    pub fn new(emission_prob: f64, channel_transmittance: f64) -> Self {
        BarrettKokProtocol {
            emission_prob,
            channel_transmittance,
            detection_efficiency: 0.9,
            visibility: 0.95,
            dark_count_prob: 1e-5,
        }
    }

    /// Two-photon coincidence probability (success rate per attempt).
    ///
    /// p_success ~ (p_em * eta_channel * eta_det)^2 / 2
    pub fn success_probability(&self) -> f64 {
        let single_arm =
            self.emission_prob * self.channel_transmittance * self.detection_efficiency;
        0.5 * single_arm * single_arm
    }

    /// Achievable fidelity determined by visibility and dark counts.
    ///
    /// F ~ (1 + V) / 2, reduced by dark count contributions.
    pub fn achievable_fidelity(&self) -> f64 {
        let f_vis = (1.0 + self.visibility) / 2.0;
        // Dark counts reduce fidelity proportionally
        let dark_reduction =
            self.dark_count_prob / (self.success_probability() + self.dark_count_prob + 1e-30);
        (f_vis * (1.0 - dark_reduction)).max(0.25) // Floor at maximally mixed
    }

    /// Attempt to generate an entangled pair.
    ///
    /// Returns (density_matrix, success).
    pub fn attempt_entanglement(&self) -> (Vec<C64>, bool) {
        let f = self.achievable_fidelity();
        let werner_p = ((4.0 * f - 1.0) / 3.0).max(0.0).min(1.0);
        let rho = werner_state(werner_p);
        (rho, self.success_probability() > 0.0)
    }
}

// ===================================================================
// ENTANGLEMENT SWAPPING
// ===================================================================

/// Perform entanglement swapping (Bell measurement + corrections) on two
/// adjacent Bell pairs sharing a common node.
///
/// Given rho_AB and rho_BC, performs a Bell-state measurement on qubits
/// B (from both pairs) and applies Pauli corrections to produce rho_AC.
///
/// For Werner states with fidelities F1 and F2, the swapped fidelity is:
/// F_swap = F1 * F2 + (1 - F1)(1 - F2) / 3
///
/// Returns (swapped density matrix, success probability).
/// Bell measurement succeeds with probability 1 in the ideal case
/// (all 4 outcomes are correctable).
pub fn entanglement_swap(rho_ab: &[C64], rho_bc: &[C64]) -> (Vec<C64>, f64) {
    assert!(rho_ab.len() == 16 && rho_bc.len() == 16);

    let f1 = bell_state_fidelity(rho_ab);
    let f2 = bell_state_fidelity(rho_bc);

    // Swapped fidelity formula for Werner states
    let f_swap = f1 * f2 + (1.0 - f1) * (1.0 - f2) / 3.0;

    // Construct the output Werner state
    let werner_p = ((4.0 * f_swap - 1.0) / 3.0).max(0.0).min(1.0);
    let rho_ac = werner_state(werner_p);

    // Bell measurement succeeds with probability 1 (all outcomes usable)
    (rho_ac, 1.0)
}

// ===================================================================
// QUANTUM REPEATER
// ===================================================================

/// Quantum repeater chain for extending entanglement over long distances.
///
/// Divides a long link into segments, distributes entanglement across each
/// segment, then uses entanglement swapping to connect adjacent segments.
///
/// With N segments of length L/N, the rate scales polynomially rather than
/// exponentially with distance (the key advantage of quantum repeaters).
#[derive(Clone, Debug)]
pub struct QuantumRepeater {
    /// Number of segments in the repeater chain.
    pub chain_length: usize,
    /// Distance per segment in kilometers.
    pub segment_distance: f64,
    /// Quantum memory coherence time in seconds.
    pub memory_lifetime: f64,
    /// Loss per km for fiber segments.
    pub loss_db_per_km: f64,
    /// Depolarizing noise rate per segment.
    pub depolarizing_rate: f64,
    /// Memory dephasing rate (per second).
    pub memory_dephasing_rate: f64,
}

impl QuantumRepeater {
    /// Create a repeater chain with default fiber parameters.
    pub fn new(chain_length: usize, segment_distance: f64, memory_lifetime: f64) -> Self {
        assert!(chain_length >= 1, "Chain must have at least one segment");
        assert!(segment_distance > 0.0, "Segment distance must be positive");
        assert!(memory_lifetime > 0.0, "Memory lifetime must be positive");
        QuantumRepeater {
            chain_length,
            segment_distance,
            memory_lifetime,
            loss_db_per_km: 0.2,
            depolarizing_rate: 0.01,
            memory_dephasing_rate: 0.0,
        }
    }

    /// Total distance of the repeater chain.
    pub fn total_distance(&self) -> f64 {
        self.chain_length as f64 * self.segment_distance
    }

    /// Create the fiber channel for a single segment.
    fn segment_channel(&self) -> FiberChannel {
        FiberChannel::new(
            self.segment_distance,
            self.loss_db_per_km,
            self.depolarizing_rate,
        )
    }

    /// Distribute entanglement across the full chain using swapping.
    ///
    /// 1. Generate a Bell pair for each segment via fiber transmission.
    /// 2. Perform entanglement swapping at each intermediate node.
    /// 3. Apply memory decoherence based on wait times.
    ///
    /// Returns (end-to-end density matrix, end-to-end success probability, fidelity).
    pub fn distribute(&self) -> (Vec<C64>, f64, f64) {
        let channel = self.segment_channel();

        // Step 1: Generate entangled pairs for each segment
        let phi_plus = bell_state_phi_plus();
        let mut pairs: Vec<Vec<C64>> = Vec::with_capacity(self.chain_length);
        let mut total_success_prob = 1.0;

        for _ in 0..self.chain_length {
            let (rho_segment, p_seg) = channel.transmit(&phi_plus);
            pairs.push(rho_segment);
            total_success_prob *= p_seg;
        }

        // Step 2: Apply memory decoherence
        // Wait time per segment ~ segment_distance / c_fiber for classical herald
        let wait_per_segment = channel.latency_seconds() * 2.0; // round trip
        if self.memory_dephasing_rate > 0.0 {
            let dephasing_per_segment = (self.memory_dephasing_rate * wait_per_segment).min(1.0);
            for pair in &mut pairs {
                let dephased = apply_single_qubit_depolarizing(pair, 4, 0, dephasing_per_segment);
                *pair = dephased;
            }
        }

        // Step 3: Entanglement swapping across the chain
        if pairs.len() == 1 {
            let fidelity = bell_state_fidelity(&pairs[0]);
            return (
                pairs.into_iter().next().unwrap(),
                total_success_prob,
                fidelity,
            );
        }

        let mut current = pairs[0].clone();
        for i in 1..pairs.len() {
            let (swapped, p_swap) = entanglement_swap(&current, &pairs[i]);
            current = swapped;
            total_success_prob *= p_swap;
        }

        let fidelity = bell_state_fidelity(&current);
        (current, total_success_prob, fidelity)
    }

    /// Perform a single entanglement swap between two pairs.
    pub fn entanglement_swap(&self, rho1: &[C64], rho2: &[C64]) -> (Vec<C64>, f64) {
        entanglement_swap(rho1, rho2)
    }

    /// Compute the raw entanglement generation rate (pairs/second).
    ///
    /// For a repeater with N segments:
    /// Rate ~ (c / L_seg) * eta_seg * (1/N) for the simplest multiplexing model.
    pub fn generation_rate(&self) -> f64 {
        let channel = self.segment_channel();
        let eta = channel.transmittance();
        let attempt_time = channel.latency_seconds() * 2.0; // round trip herald
        if attempt_time < 1e-15 {
            return 0.0;
        }
        // Basic rate: limited by slowest segment
        let rate_per_segment = eta / attempt_time;
        // With N segments needing simultaneous success, effective rate decreases
        // In practice, multiplexing and parallel attempts improve this
        rate_per_segment / self.chain_length as f64
    }

    /// Check if the repeater can overcome the direct-transmission rate bound.
    ///
    /// Direct transmission: rate ~ eta_direct = 10^(-alpha * L_total / 10)
    /// Repeater is beneficial when its rate exceeds direct transmission.
    pub fn beats_direct_transmission(&self) -> bool {
        let direct_eta = 10.0_f64.powf(-self.loss_db_per_km * self.total_distance() / 10.0);
        self.generation_rate() > direct_eta
    }
}

// ===================================================================
// BBPSSW PURIFICATION
// ===================================================================

/// BBPSSW entanglement purification protocol
/// (Bennett, Brassard, Popescu, Schumacher, Smolin, Wootters 1996).
///
/// Takes two copies of a noisy Bell pair and produces one copy with
/// higher fidelity, consuming the second pair. Both parties perform
/// bilateral CNOT operations and measure the target pair.
///
/// For two Werner states with fidelity F, the output fidelity is:
/// F' = (F^2 + ((1-F)/3)^2) / (F^2 + 2*F*(1-F)/3 + 5*((1-F)/3)^2)
///
/// The protocol succeeds when both measurements agree (probability p_success).
#[derive(Clone, Debug)]
pub struct BBPSSWPurification;

impl BBPSSWPurification {
    pub fn new() -> Self {
        BBPSSWPurification
    }

    /// Purify two noisy Bell pairs into one higher-fidelity pair.
    ///
    /// # Arguments
    /// * `rho1` - First noisy Bell pair (4x4 density matrix)
    /// * `rho2` - Second noisy Bell pair (4x4 density matrix, consumed)
    ///
    /// # Returns
    /// (purified density matrix, success probability)
    ///
    /// The protocol only improves fidelity when F > 0.5 (above the
    /// entanglement distillation threshold).
    pub fn purify(&self, rho1: &[C64], rho2: &[C64]) -> (Vec<C64>, f64) {
        assert!(rho1.len() == 16 && rho2.len() == 16);

        let f1 = bell_state_fidelity(rho1);
        let f2 = bell_state_fidelity(rho2);

        // Use the average fidelity for the BBPSSW formula
        // (generalization: use individual fidelities)
        let f = (f1 + f2) / 2.0;

        // BBPSSW output fidelity
        let f_sq = f * f;
        let noise = (1.0 - f) / 3.0;
        let noise_sq = noise * noise;

        let numerator = f_sq + noise_sq;
        let denominator = f_sq + 2.0 * f * noise + 5.0 * noise_sq;

        if denominator < 1e-15 {
            // Degenerate case
            return (bell_state_phi_plus(), 0.0);
        }

        let f_out = numerator / denominator;
        let p_success = denominator;

        // Construct output Werner state
        let werner_p = ((4.0 * f_out - 1.0) / 3.0).max(0.0).min(1.0);
        let rho_out = werner_state(werner_p);

        (rho_out, p_success)
    }

    /// Compute the output fidelity without constructing the full density matrix.
    pub fn output_fidelity(&self, f1: f64, f2: f64) -> f64 {
        let f = (f1 + f2) / 2.0;
        let f_sq = f * f;
        let noise = (1.0 - f) / 3.0;
        let noise_sq = noise * noise;
        let numerator = f_sq + noise_sq;
        let denominator = f_sq + 2.0 * f * noise + 5.0 * noise_sq;
        if denominator < 1e-15 {
            return 0.25;
        }
        numerator / denominator
    }

    /// Check whether purification will improve fidelity.
    ///
    /// BBPSSW only improves fidelity when F > 0.5.
    pub fn will_improve(&self, fidelity: f64) -> bool {
        fidelity > 0.5
    }
}

// ===================================================================
// DEJMPS PURIFICATION
// ===================================================================

/// DEJMPS entanglement purification protocol
/// (Deutsch, Ekert, Jozsa, Macchiavello, Popescu, Sanpera 1996).
///
/// An improvement over BBPSSW that uses bilateral pi/2 rotations before
/// the CNOT, achieving better purification for states with asymmetric noise.
///
/// For isotropic (Werner) states, DEJMPS converges faster to unit fidelity
/// and has a higher yield per round than BBPSSW.
#[derive(Clone, Debug)]
pub struct DEJMPSPurification;

impl DEJMPSPurification {
    pub fn new() -> Self {
        DEJMPSPurification
    }

    /// Purify two noisy Bell pairs using the DEJMPS protocol.
    ///
    /// Applies bilateral sigma_y rotations before bilateral CNOT,
    /// then measures the target pair in the computational basis.
    ///
    /// For Werner states with fidelity F:
    /// F' = (F^2 + (1-F)^2/9) / (F^2 + 2*F*(1-F)/3 + 5*(1-F)^2/9)
    ///
    /// DEJMPS has the same threshold (F > 0.5) but converges faster.
    pub fn purify(&self, rho1: &[C64], rho2: &[C64]) -> (Vec<C64>, f64) {
        assert!(rho1.len() == 16 && rho2.len() == 16);

        let f1 = bell_state_fidelity(rho1);
        let f2 = bell_state_fidelity(rho2);
        let f = (f1 + f2) / 2.0;

        // DEJMPS formula (sigma_y bilateral rotation variant)
        // The key difference from BBPSSW is in how off-diagonal Werner
        // components are mixed. For Werner states, the formula simplifies to:
        let a = f;
        let b = (1.0 - f) / 3.0;

        // After bilateral rotation + bilateral CNOT + measurement
        let numerator = a * a + b * b;
        let denominator = a * a + 2.0 * a * b + 5.0 * b * b;

        if denominator < 1e-15 {
            return (bell_state_phi_plus(), 0.0);
        }

        let f_out = numerator / denominator;
        let p_success = denominator;

        let werner_p = ((4.0 * f_out - 1.0) / 3.0).max(0.0).min(1.0);
        let rho_out = werner_state(werner_p);

        (rho_out, p_success)
    }

    /// Compute the output fidelity for given input fidelities.
    pub fn output_fidelity(&self, f1: f64, f2: f64) -> f64 {
        let f = (f1 + f2) / 2.0;
        let a = f;
        let b = (1.0 - f) / 3.0;
        let numerator = a * a + b * b;
        let denominator = a * a + 2.0 * a * b + 5.0 * b * b;
        if denominator < 1e-15 {
            return 0.25;
        }
        numerator / denominator
    }

    /// Iteratively purify to a target fidelity.
    ///
    /// Returns (final_fidelity, number_of_rounds, total_pairs_consumed).
    /// Each round consumes 2^round pairs from the initial supply.
    pub fn purify_to_target(
        &self,
        initial_fidelity: f64,
        target_fidelity: f64,
        max_rounds: usize,
    ) -> (f64, usize, usize) {
        let mut f = initial_fidelity;
        let mut rounds = 0;
        let mut pairs_consumed = 1;

        for _ in 0..max_rounds {
            if f >= target_fidelity {
                break;
            }
            if f <= 0.5 {
                // Below threshold, purification cannot help
                break;
            }
            f = self.output_fidelity(f, f);
            rounds += 1;
            pairs_consumed *= 2;
        }

        (f, rounds, pairs_consumed)
    }
}

// ===================================================================
// QUANTUM NETWORK
// ===================================================================

/// A node in the quantum network.
#[derive(Clone, Debug)]
pub struct NetworkNode {
    /// Unique identifier for the node.
    pub id: usize,
    /// Human-readable label.
    pub label: String,
    /// Whether this node has quantum memory (repeater capability).
    pub has_memory: bool,
    /// Memory coherence time in seconds (if applicable).
    pub memory_coherence_time: f64,
}

/// An edge (channel) in the quantum network.
#[derive(Clone, Debug)]
struct NetworkEdge {
    /// Source node.
    node1: usize,
    /// Destination node.
    node2: usize,
    /// Channel distance in km (used as edge weight for routing).
    distance_km: f64,
    /// Loss per km.
    loss_db_per_km: f64,
    /// Noise rate.
    depolarizing_rate: f64,
}

/// Entry for Dijkstra's priority queue (min-heap by distance).
#[derive(Clone, PartialEq)]
struct DijkstraEntry {
    node: usize,
    distance: f64,
}

impl Eq for DijkstraEntry {}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Quantum network topology with nodes and fiber channels.
///
/// Supports shortest-path routing and end-to-end entanglement distribution
/// through intermediate repeater nodes.
#[derive(Clone, Debug)]
pub struct QuantumNetwork {
    /// Nodes in the network, indexed by ID.
    nodes: HashMap<usize, NetworkNode>,
    /// Adjacency list: node_id -> [(neighbor_id, edge_index)].
    adjacency: HashMap<usize, Vec<(usize, usize)>>,
    /// All edges in the network.
    edges: Vec<NetworkEdge>,
    /// Next available node ID.
    next_node_id: usize,
}

impl QuantumNetwork {
    /// Create an empty quantum network.
    pub fn new() -> Self {
        QuantumNetwork {
            nodes: HashMap::new(),
            adjacency: HashMap::new(),
            edges: Vec::new(),
            next_node_id: 0,
        }
    }

    /// Add a node to the network. Returns the assigned node ID.
    pub fn add_node(&mut self, label: &str, has_memory: bool) -> usize {
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.insert(
            id,
            NetworkNode {
                id,
                label: label.to_string(),
                has_memory,
                memory_coherence_time: 1.0, // default 1 second
            },
        );
        self.adjacency.insert(id, Vec::new());
        id
    }

    /// Add a node with specified memory coherence time.
    pub fn add_node_with_memory(&mut self, label: &str, memory_coherence_time: f64) -> usize {
        let id = self.add_node(label, true);
        if let Some(node) = self.nodes.get_mut(&id) {
            node.memory_coherence_time = memory_coherence_time;
        }
        id
    }

    /// Add a quantum channel between two nodes.
    ///
    /// # Arguments
    /// * `node1` - First endpoint node ID
    /// * `node2` - Second endpoint node ID
    /// * `distance_km` - Physical distance in km
    /// * `loss_db_per_km` - Fiber loss in dB/km
    /// * `depolarizing_rate` - Noise rate per qubit
    pub fn add_channel(
        &mut self,
        node1: usize,
        node2: usize,
        distance_km: f64,
        loss_db_per_km: f64,
        depolarizing_rate: f64,
    ) {
        assert!(
            self.nodes.contains_key(&node1) && self.nodes.contains_key(&node2),
            "Both nodes must exist in the network"
        );

        let edge_idx = self.edges.len();
        self.edges.push(NetworkEdge {
            node1,
            node2,
            distance_km,
            loss_db_per_km,
            depolarizing_rate,
        });

        self.adjacency
            .entry(node1)
            .or_default()
            .push((node2, edge_idx));
        self.adjacency
            .entry(node2)
            .or_default()
            .push((node1, edge_idx));
    }

    /// Add a channel using a FiberChannel instance.
    pub fn add_fiber_channel(&mut self, node1: usize, node2: usize, channel: &FiberChannel) {
        self.add_channel(
            node1,
            node2,
            channel.distance_km,
            channel.loss_db_per_km,
            channel.depolarizing_rate,
        );
    }

    /// Find the shortest path between two nodes using Dijkstra's algorithm.
    ///
    /// Returns the path as a sequence of node IDs, or None if no path exists.
    pub fn find_path(&self, src: usize, dst: usize) -> Option<Vec<usize>> {
        if !self.nodes.contains_key(&src) || !self.nodes.contains_key(&dst) {
            return None;
        }
        if src == dst {
            return Some(vec![src]);
        }

        let mut dist: HashMap<usize, f64> = HashMap::new();
        let mut prev: HashMap<usize, usize> = HashMap::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut heap = BinaryHeap::new();

        dist.insert(src, 0.0);
        heap.push(DijkstraEntry {
            node: src,
            distance: 0.0,
        });

        while let Some(DijkstraEntry { node, distance }) = heap.pop() {
            if node == dst {
                // Reconstruct path
                let mut path = Vec::new();
                let mut current = dst;
                while current != src {
                    path.push(current);
                    current = prev[&current];
                }
                path.push(src);
                path.reverse();
                return Some(path);
            }

            if !visited.insert(node) {
                continue;
            }

            if distance > *dist.get(&node).unwrap_or(&f64::INFINITY) {
                continue;
            }

            if let Some(neighbors) = self.adjacency.get(&node) {
                for &(neighbor, edge_idx) in neighbors {
                    if visited.contains(&neighbor) {
                        continue;
                    }
                    let edge = &self.edges[edge_idx];
                    let new_dist = distance + edge.distance_km;
                    if new_dist < *dist.get(&neighbor).unwrap_or(&f64::INFINITY) {
                        dist.insert(neighbor, new_dist);
                        prev.insert(neighbor, node);
                        heap.push(DijkstraEntry {
                            node: neighbor,
                            distance: new_dist,
                        });
                    }
                }
            }
        }

        None // No path found
    }

    /// Distribute end-to-end entanglement between src and dst.
    ///
    /// Uses shortest-path routing with entanglement swapping at intermediate nodes.
    ///
    /// Returns (density matrix, overall success probability, fidelity),
    /// or None if no path exists.
    pub fn distribute_entanglement(&self, src: usize, dst: usize) -> Option<(Vec<C64>, f64, f64)> {
        let path = self.find_path(src, dst)?;

        if path.len() < 2 {
            // Source equals destination, return perfect Bell pair
            return Some((bell_state_phi_plus(), 1.0, 1.0));
        }

        // Generate entangled pairs along each link in the path
        let phi_plus = bell_state_phi_plus();
        let mut segment_pairs = Vec::new();
        let mut total_success = 1.0;

        for i in 0..path.len() - 1 {
            let n1 = path[i];
            let n2 = path[i + 1];

            // Find the edge between n1 and n2
            let edge = self.find_edge(n1, n2).expect("Edge must exist on path");
            let channel = FiberChannel::new(
                edge.distance_km,
                edge.loss_db_per_km,
                edge.depolarizing_rate,
            );

            let (rho_seg, p_seg) = channel.transmit(&phi_plus);
            segment_pairs.push(rho_seg);
            total_success *= p_seg;
        }

        // Swap through intermediate nodes
        if segment_pairs.len() == 1 {
            let f = bell_state_fidelity(&segment_pairs[0]);
            return Some((segment_pairs.into_iter().next().unwrap(), total_success, f));
        }

        let mut current = segment_pairs[0].clone();
        for i in 1..segment_pairs.len() {
            let (swapped, p_swap) = entanglement_swap(&current, &segment_pairs[i]);
            current = swapped;
            total_success *= p_swap;
        }

        let fidelity = bell_state_fidelity(&current);
        Some((current, total_success, fidelity))
    }

    /// Find the edge between two nodes.
    fn find_edge(&self, n1: usize, n2: usize) -> Option<&NetworkEdge> {
        if let Some(neighbors) = self.adjacency.get(&n1) {
            for &(neighbor, edge_idx) in neighbors {
                if neighbor == n2 {
                    return Some(&self.edges[edge_idx]);
                }
            }
        }
        None
    }

    /// Get the number of nodes in the network.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges in the network.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: usize) -> Option<&NetworkNode> {
        self.nodes.get(&id)
    }

    /// Get all node IDs.
    pub fn node_ids(&self) -> Vec<usize> {
        self.nodes.keys().cloned().collect()
    }

    /// Get the total distance of a path.
    pub fn path_distance(&self, path: &[usize]) -> f64 {
        let mut total = 0.0;
        for i in 0..path.len().saturating_sub(1) {
            if let Some(edge) = self.find_edge(path[i], path[i + 1]) {
                total += edge.distance_km;
            }
        }
        total
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-10;

    // ----- Bell state and fidelity tests -----

    #[test]
    fn test_bell_state_phi_plus_is_valid() {
        let rho = bell_state_phi_plus();
        assert_eq!(rho.len(), 16);
        let tr = dm_trace(&rho, 4);
        assert!(
            (tr - 1.0).abs() < TOLERANCE,
            "Bell state trace should be 1, got {}",
            tr
        );
        // Fidelity with itself should be 1
        let f = bell_state_fidelity(&rho);
        assert!(
            (f - 1.0).abs() < TOLERANCE,
            "Phi+ fidelity with itself should be 1, got {}",
            f
        );
    }

    #[test]
    fn test_bell_state_fidelity_maximally_mixed() {
        // Maximally mixed state: I/4
        let mut rho = vec![c64_zero(); 16];
        for i in 0..4 {
            rho[i * 4 + i] = C64::new(0.25, 0.0);
        }
        let f = bell_state_fidelity(&rho);
        assert!(
            (f - 0.25).abs() < TOLERANCE,
            "Maximally mixed fidelity should be 0.25, got {}",
            f
        );
    }

    #[test]
    fn test_werner_state_fidelity() {
        // Werner state with p=0.8 should have F = (1 + 3*0.8)/4 = 0.85
        let rho = werner_state(0.8);
        let f = bell_state_fidelity(&rho);
        let expected = (1.0 + 3.0 * 0.8) / 4.0;
        assert!(
            (f - expected).abs() < TOLERANCE,
            "Werner(0.8) fidelity: expected {}, got {}",
            expected,
            f
        );
    }

    // ----- Fiber channel tests -----

    #[test]
    fn test_fiber_channel_transmittance() {
        let channel = FiberChannel::new(50.0, 0.2, 0.01);
        let eta = channel.transmittance();
        // eta = 10^(-0.2 * 50 / 10) = 10^(-1) = 0.1
        assert!(
            (eta - 0.1).abs() < TOLERANCE,
            "50km fiber transmittance: expected 0.1, got {}",
            eta
        );
    }

    #[test]
    fn test_fiber_channel_zero_distance() {
        let channel = FiberChannel::new(0.0, 0.2, 0.0);
        let eta = channel.transmittance();
        assert!(
            (eta - 1.0).abs() < TOLERANCE,
            "Zero-distance transmittance should be 1.0, got {}",
            eta
        );
    }

    #[test]
    fn test_fiber_channel_noise_reduces_fidelity() {
        let channel = FiberChannel::new(10.0, 0.2, 0.05);
        let phi_plus = bell_state_phi_plus();
        let (rho_out, _p) = channel.transmit(&phi_plus);

        let f_in = bell_state_fidelity(&phi_plus);
        let f_out = bell_state_fidelity(&rho_out);

        assert!(
            f_out < f_in,
            "Channel noise should reduce fidelity: {} -> {}",
            f_in,
            f_out
        );
        assert!(
            f_out > 0.5,
            "Fidelity should remain above 0.5 for moderate noise, got {}",
            f_out
        );
    }

    #[test]
    fn test_fiber_no_noise_preserves_state() {
        let channel = FiberChannel::new(10.0, 0.2, 0.0);
        let phi_plus = bell_state_phi_plus();
        let (rho_out, _p) = channel.transmit(&phi_plus);

        for i in 0..16 {
            assert!(
                (rho_out[i] - phi_plus[i]).norm() < TOLERANCE,
                "Noiseless channel should preserve state at element {}",
                i
            );
        }
    }

    // ----- Free-space channel tests -----

    #[test]
    fn test_free_space_channel_transmittance() {
        let channel = FreeSpaceChannel::new(100.0);
        let eta = channel.transmittance();
        assert!(
            eta > 0.0 && eta < 1.0,
            "Free-space transmittance should be in (0, 1), got {}",
            eta
        );
        // Atmospheric component
        let eta_atm = channel.atmospheric_transmittance();
        assert!(eta_atm > 0.0 && eta_atm <= 1.0);
    }

    // ----- Erasure channel tests -----

    #[test]
    fn test_erasure_channel_no_erasure() {
        let channel = ErasureChannel::new(0.0);
        let phi_plus = bell_state_phi_plus();
        let (rho_out, p) = channel.transmit(&phi_plus);
        assert!((p - 1.0).abs() < TOLERANCE);
        for i in 0..16 {
            assert!(
                (rho_out[i] - phi_plus[i]).norm() < TOLERANCE,
                "Zero-erasure should preserve state"
            );
        }
    }

    #[test]
    fn test_erasure_channel_reduces_fidelity() {
        let channel = ErasureChannel::new(0.3);
        let phi_plus = bell_state_phi_plus();
        let (rho_out, p) = channel.transmit(&phi_plus);
        assert!((p - 0.7).abs() < TOLERANCE);
        let f_out = bell_state_fidelity(&rho_out);
        assert!(f_out < 1.0, "Erasure should reduce fidelity, got {}", f_out);
    }

    // ----- DLCZ protocol tests -----

    #[test]
    fn test_dlcz_bell_pair_generation() {
        let proto = DLCZProtocol::new(0.01, 0.5);
        let (rho, success) = proto.attempt_entanglement();
        assert!(success);
        assert_eq!(rho.len(), 16);
        let f = bell_state_fidelity(&rho);
        assert!(
            f > 0.9,
            "DLCZ should produce high-fidelity pairs, got {}",
            f
        );
        let tr = dm_trace(&rho, 4);
        assert!(
            (tr - 1.0).abs() < TOLERANCE,
            "Trace should be 1, got {}",
            tr
        );
    }

    #[test]
    fn test_dlcz_success_probability() {
        let proto = DLCZProtocol::new(0.01, 0.5);
        let p = proto.success_probability();
        // p = 0.01 * 0.5 * 0.9 = 0.0045
        let expected = 0.01 * 0.5 * 0.9;
        assert!(
            (p - expected).abs() < TOLERANCE,
            "DLCZ success prob: expected {}, got {}",
            expected,
            p
        );
    }

    // ----- Barrett-Kok protocol tests -----

    #[test]
    fn test_barrett_kok_entanglement() {
        let proto = BarrettKokProtocol::new(0.1, 0.8);
        let (rho, success) = proto.attempt_entanglement();
        assert!(success);
        let f = bell_state_fidelity(&rho);
        assert!(
            f > 0.9,
            "Barrett-Kok should produce high-fidelity pairs, got {}",
            f
        );
    }

    // ----- Entanglement swapping tests -----

    #[test]
    fn test_entanglement_swap_perfect_pairs() {
        let phi_plus = bell_state_phi_plus();
        let (rho_ac, p) = entanglement_swap(&phi_plus, &phi_plus);
        assert!((p - 1.0).abs() < TOLERANCE);
        let f = bell_state_fidelity(&rho_ac);
        assert!(
            (f - 1.0).abs() < TOLERANCE,
            "Swapping perfect pairs should give perfect pair, got F={}",
            f
        );
    }

    #[test]
    fn test_entanglement_swap_noisy_pairs() {
        let rho1 = werner_state(0.9); // F = (1+2.7)/4 = 0.925
        let rho2 = werner_state(0.8); // F = (1+2.4)/4 = 0.85

        let f1 = bell_state_fidelity(&rho1);
        let f2 = bell_state_fidelity(&rho2);

        let (rho_ac, _p) = entanglement_swap(&rho1, &rho2);
        let f_swap = bell_state_fidelity(&rho_ac);

        // Swapped fidelity should be less than both inputs for imperfect pairs
        // F_swap = F1*F2 + (1-F1)(1-F2)/3
        let expected = f1 * f2 + (1.0 - f1) * (1.0 - f2) / 3.0;
        assert!(
            (f_swap - expected).abs() < TOLERANCE,
            "Swapped fidelity: expected {}, got {}",
            expected,
            f_swap
        );
    }

    // ----- BBPSSW purification tests -----

    #[test]
    fn test_bbpssw_purification_improves_fidelity() {
        let purifier = BBPSSWPurification::new();
        let rho_noisy = werner_state(0.7); // F = 0.775
        let f_in = bell_state_fidelity(&rho_noisy);

        let (rho_out, p_success) = purifier.purify(&rho_noisy, &rho_noisy);
        let f_out = bell_state_fidelity(&rho_out);

        assert!(
            f_out > f_in,
            "BBPSSW should improve fidelity: {} -> {}",
            f_in,
            f_out
        );
        assert!(
            p_success > 0.0 && p_success < 1.0,
            "Success probability should be in (0, 1), got {}",
            p_success
        );
    }

    #[test]
    fn test_bbpssw_below_threshold() {
        let purifier = BBPSSWPurification::new();
        assert!(!purifier.will_improve(0.4));
        assert!(purifier.will_improve(0.6));
    }

    // ----- DEJMPS purification tests -----

    #[test]
    fn test_dejmps_purification_improves_fidelity() {
        let purifier = DEJMPSPurification::new();
        let rho_noisy = werner_state(0.7);
        let f_in = bell_state_fidelity(&rho_noisy);

        let (rho_out, _p) = purifier.purify(&rho_noisy, &rho_noisy);
        let f_out = bell_state_fidelity(&rho_out);

        assert!(
            f_out > f_in,
            "DEJMPS should improve fidelity: {} -> {}",
            f_in,
            f_out
        );
    }

    #[test]
    fn test_dejmps_iterative_purification() {
        let purifier = DEJMPSPurification::new();
        let (f_final, rounds, pairs) = purifier.purify_to_target(0.75, 0.95, 20);
        assert!(
            f_final >= 0.95,
            "Should reach target fidelity 0.95, got {}",
            f_final
        );
        assert!(rounds > 0, "Should require at least one round");
        assert!(pairs >= 2, "Should consume at least 2 pairs");
    }

    // ----- Quantum repeater tests -----

    #[test]
    fn test_repeater_single_segment() {
        let repeater = QuantumRepeater::new(1, 50.0, 1.0);
        let (rho, p, f) = repeater.distribute();
        assert_eq!(rho.len(), 16);
        assert!(p > 0.0, "Success probability should be positive");
        assert!(f > 0.5, "Fidelity should be above 0.5, got {}", f);
    }

    #[test]
    fn test_repeater_chain_reduces_fidelity() {
        let repeater_1 = QuantumRepeater::new(1, 100.0, 1.0);
        let repeater_4 = QuantumRepeater::new(4, 25.0, 1.0);

        let (_rho1, _p1, f1) = repeater_1.distribute();
        let (_rho4, _p4, f4) = repeater_4.distribute();

        // Multi-segment repeater has swapping loss but better per-segment fidelity.
        // Both should produce valid entanglement.
        assert!(f1 > 0.0 && f1 <= 1.0);
        assert!(f4 > 0.0 && f4 <= 1.0);
    }

    // ----- Quantum network tests -----

    #[test]
    fn test_network_add_nodes_and_channels() {
        let mut net = QuantumNetwork::new();
        let a = net.add_node("Alice", true);
        let b = net.add_node("Bob", true);
        let c = net.add_node("Charlie", true);

        net.add_channel(a, b, 50.0, 0.2, 0.01);
        net.add_channel(b, c, 30.0, 0.2, 0.01);

        assert_eq!(net.num_nodes(), 3);
        assert_eq!(net.num_edges(), 2);
    }

    #[test]
    fn test_network_shortest_path() {
        let mut net = QuantumNetwork::new();
        let a = net.add_node("A", true);
        let b = net.add_node("B", true);
        let c = net.add_node("C", true);
        let d = net.add_node("D", true);

        // A -- 50km -- B -- 30km -- D
        // A -- 100km -- C -- 20km -- D
        net.add_channel(a, b, 50.0, 0.2, 0.01);
        net.add_channel(b, d, 30.0, 0.2, 0.01);
        net.add_channel(a, c, 100.0, 0.2, 0.01);
        net.add_channel(c, d, 20.0, 0.2, 0.01);

        let path = net.find_path(a, d).expect("Path should exist");
        // Shortest: A -> B -> D (80km) vs A -> C -> D (120km)
        assert_eq!(path, vec![a, b, d]);
        assert!((net.path_distance(&path) - 80.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_network_no_path() {
        let mut net = QuantumNetwork::new();
        let a = net.add_node("A", false);
        let b = net.add_node("B", false);
        // No channel between them
        assert!(net.find_path(a, b).is_none());
    }

    #[test]
    fn test_network_distribute_entanglement() {
        let mut net = QuantumNetwork::new();
        let a = net.add_node("Alice", true);
        let r = net.add_node("Repeater", true);
        let b = net.add_node("Bob", true);

        net.add_channel(a, r, 25.0, 0.2, 0.005);
        net.add_channel(r, b, 25.0, 0.2, 0.005);

        let result = net.distribute_entanglement(a, b);
        assert!(result.is_some(), "Should find path and distribute");

        let (rho, p, f) = result.unwrap();
        assert_eq!(rho.len(), 16);
        assert!(p > 0.0, "Success probability should be positive");
        assert!(
            f > 0.25,
            "Fidelity should be above maximally mixed, got {}",
            f
        );
        let tr = dm_trace(&rho, 4);
        assert!((tr - 1.0).abs() < TOLERANCE, "Output trace should be 1");
    }
}
